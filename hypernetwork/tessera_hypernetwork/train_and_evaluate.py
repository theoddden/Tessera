"""
End-to-end training and evaluation pipeline for Tessera hypernetwork.

This script orchestrates:
1. Hypernetwork training with asymmetric features
2. Checkpoint integration into server
3. Benchmark evaluation using lm_eval
4. Result comparison against baseline
"""

import os
import subprocess
import json
import argparse
from pathlib import Path
import shutil


def train_hypernetwork(
    metadata_dir: str,
    output_dir: str,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    rank: int = 16,
    epochs: int = 50,
    use_curriculum: bool = True,
    device: str = "cuda",
):
    """Run hypernetwork training."""
    print("=" * 60)
    print("STEP 1: Training Hypernetwork")
    print("=" * 60)

    cmd = [
        "python", "-m", "tessera_hypernetwork.train_hypernetwork",
        "--metadata-dir", metadata_dir,
        "--base-model", base_model,
        "--rank", str(rank),
        "--epochs", str(epochs),
        "--output-dir", output_dir,
        "--device", device,
    ]

    if use_curriculum:
        cmd.extend(["--use-curriculum", "--num-curriculum-stages", "5"])
        cmd.extend(["--check-similarity", "--check-contamination"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    print("Training complete!")
    return result.returncode == 0


def update_server_with_checkpoint(
    checkpoint_path: str,
    server_file: str = "tessera_hypernetwork/server.py",
):
    """Update server to load trained hypernetwork checkpoint."""
    print("\n" + "=" * 60)
    print("STEP 2: Integrating Checkpoint into Server")
    print("=" * 60)

    # Read current server file
    with open(server_file) as f:
        server_content = f.read()

    # Check if checkpoint loading is already implemented
    if "load_hypernetwork_checkpoint" in server_content:
        print("Checkpoint loading already implemented in server")
        return True

    # Add checkpoint loading function
    checkpoint_loading_code = '''
def load_hypernetwork_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load trained hypernetwork checkpoint."""
    from tessera_hypernetwork.train_hypernetwork import DomainConditionedHypernetwork, StructuredMetadataEncoder
    from sentence_transformers import SentenceTransformer
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct models
    base_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    encoder = StructuredMetadataEncoder(base_encoder)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    # Get dimensions from checkpoint or use defaults
    num_domains = checkpoint.get('num_domains', 10)
    hypernetwork = DomainConditionedHypernetwork(
        embed_dim=768,
        rank=16,
        d_in=4096,
        d_out=4096,
        hidden_dim=2048,
        num_domains=num_domains,
    )
    hypernetwork.load_state_dict(checkpoint['hypernetwork_state_dict'])

    return encoder, hypernetwork


# Global trained hypernetwork (loaded if checkpoint exists)
trained_encoder = None
trained_hypernetwork = None

CHECKPOINT_PATH = os.environ.get("TESSERA_CHECKPOINT_PATH")
if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    try:
        trained_encoder, trained_hypernetwork = load_hypernetwork_checkpoint(CHECKPOINT_PATH)
        print(f"Loaded trained hypernetwork from {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
'''

    # Insert after imports
    import_end = server_content.find("app = FastAPI")
    if import_end == -1:
        print("Could not find insertion point in server file")
        return False

    server_content = server_content[:import_end] + checkpoint_loading_code + "\n" + server_content[import_end:]

    # Write updated server
    with open(server_file, "w") as f:
        f.write(server_content)

    print(f"Updated {server_file} with checkpoint loading")
    return True


def generate_adapters_with_trained_hypernetwork(
    metadata_dir: str,
    output_dir: str,
    checkpoint_path: str,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    rank: int = 16,
):
    """Generate adapters using trained hypernetwork."""
    print("\n" + "=" * 60)
    print("STEP 3: Generating Adapters with Trained Hypernetwork")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Set environment variable for checkpoint
    os.environ["TESSERA_CHECKPOINT_PATH"] = checkpoint_path

    # Generate adapters for each metadata file
    metadata_files = sorted(Path(metadata_dir).glob("*.json"))
    generated = 0

    for meta_file in metadata_files:
        with open(meta_file) as f:
            metadata = json.load(f)

        adapter_id = metadata.get("id", meta_file.stem)
        output_path = os.path.join(output_dir, f"{adapter_id}.safetensors")

        # Use CLI to generate adapter
        cmd = [
            "tessera", "generate",
            "--from-metadata", json.dumps(metadata),
            "--base-model", base_model,
            "--rank", str(rank),
            "--save", output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            generated += 1
            print(f"Generated {adapter_id}.safetensors")
        except subprocess.CalledProcessError as e:
            print(f"Failed to generate {adapter_id}: {e}")

    print(f"Generated {generated}/{len(metadata_files)} adapters")
    return generated > 0


def run_benchmark_evaluation(
    adapters_dir: str,
    results_dir: str,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    vllm_url: str = "http://localhost:8000/v1/completions",
):
    """Run benchmark evaluation using lm_eval."""
    print("\n" + "=" * 60)
    print("STEP 4: Running Benchmark Evaluation")
    print("=" * 60)

    os.makedirs(results_dir, exist_ok=True)

    # Import adapters into Tessera server
    print("Importing adapters into Tessera server...")
    for adapter_file in sorted(Path(adapters_dir).glob("*.safetensors")):
        adapter_name = adapter_file.stem
        cmd = [
            "tessera", "lorax", "import-adapter",
            "--file", str(adapter_file),
            "--adapter-name", adapter_name,
            "--base-model", base_model,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Imported {adapter_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to import {adapter_name}: {e}")

    # Run lm_eval benchmark
    print("\nRunning lm_eval benchmark...")
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", f"base_url={vllm_url}",
        "--tasks", "mmlu_abstract_algebra,mmlu_anatomy,mmlu_business_ethics",
        "--output_path", results_dir,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Benchmark results saved to {results_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed: {e}")
        return False


def compare_results(
    baseline_results: str,
    trained_results: str,
    output_file: str = "comparison.json",
):
    """Compare baseline vs trained hypernetwork results."""
    print("\n" + "=" * 60)
    print("STEP 5: Comparing Results")
    print("=" * 60)

    # Load baseline results
    with open(baseline_results) as f:
        baseline = json.load(f)

    # Load trained results
    with open(trained_results) as f:
        trained = json.load(f)

    # Compute deltas
    comparison = {
        "baseline": baseline,
        "trained": trained,
        "deltas": {},
    }

    for task in baseline.get("results", {}):
        if task in trained.get("results", {}):
            baseline_acc = baseline["results"][task].get("acc", 0)
            trained_acc = trained["results"][task].get("acc", 0)
            delta = trained_acc - baseline_acc
            comparison["deltas"][task] = {
                "baseline": baseline_acc,
                "trained": trained_acc,
                "delta": delta,
                "delta_percent": delta * 100,
            }

    # Save comparison
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Comparison saved to {output_file}")

    # Print summary
    print("\nSummary:")
    improved = 0
    degraded = 0
    for task, metrics in comparison["deltas"].items():
        delta = metrics["delta_percent"]
        if delta > 0:
            improved += 1
            print(f"  {task}: +{delta:.2f}%")
        elif delta < 0:
            degraded += 1
            print(f"  {task}: {delta:.2f}%")

    print(f"\nImproved: {improved}, Degraded: {degraded}")

    return comparison


def main():
    """Run full training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Train and evaluate Tessera hypernetwork")
    parser.add_argument("--metadata-dir", type=str, required=True,
                        help="Directory containing metadata JSON files")
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints",
                        help="Directory for training checkpoints")
    parser.add_argument("--adapters-dir", type=str, default="./trained_adapters",
                        help="Directory for generated adapters")
    parser.add_argument("--results-dir", type=str, default="./trained_results",
                        help="Directory for benchmark results")
    parser.add_argument("--baseline-results", type=str, default=None,
                        help="Path to baseline results for comparison")
    parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model identifier")
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use existing checkpoint")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip benchmark evaluation")

    args = parser.parse_args()

    # Step 1: Train hypernetwork
    if not args.skip_training:
        success = train_hypernetwork(
            metadata_dir=args.metadata_dir,
            output_dir=args.checkpoints_dir,
            base_model=args.base_model,
            rank=args.rank,
            epochs=args.epochs,
            device=args.device,
        )
        if not success:
            print("Training failed!")
            return 1

    # Find best checkpoint
    checkpoint_path = os.path.join(args.checkpoints_dir, "best_hypernetwork.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return 1

    # Step 2: Update server with checkpoint
    update_server_with_checkpoint(checkpoint_path)

    # Step 3: Generate adapters
    success = generate_adapters_with_trained_hypernetwork(
        metadata_dir=args.metadata_dir,
        output_dir=args.adapters_dir,
        checkpoint_path=checkpoint_path,
        base_model=args.base_model,
        rank=args.rank,
    )
    if not success:
        print("Adapter generation failed!")
        return 1

    # Step 4: Run benchmark
    if not args.skip_evaluation:
        success = run_benchmark_evaluation(
            adapters_dir=args.adapters_dir,
            results_dir=args.results_dir,
            base_model=args.base_model,
        )
        if not success:
            print("Benchmark evaluation failed!")
            return 1

    # Step 5: Compare with baseline
    if args.baseline_results:
        compare_results(
            baseline_results=args.baseline_results,
            trained_results=os.path.join(args.results_dir, "results.json"),
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    main()
