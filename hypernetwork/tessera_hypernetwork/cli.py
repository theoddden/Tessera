"""
Tessera Hypernetwork CLI
Command-line interface for LoRA adapter generation and serving
"""

import click
import json
import subprocess
import sys
import shutil
from pathlib import Path


@click.group()
def cli():
    """Tessera Hypernetwork CLI - Generate and serve LoRA adapters"""
    pass


@cli.command()
@click.option("--from-metadata", type=str, help="JSON metadata string or file path")
@click.option("--from-text", type=str, help="Natural language description")
@click.option("--from-doc", type=str, help="Document content or file path")
@click.option(
    "--base-model",
    type=str,
    default="mistralai/Mistral-7B-Instruct-v0.2",
    help="Base model identifier (default: mistralai/Mistral-7B-Instruct-v0.2)",
)
@click.option("--rank", type=int, default=16, help="LoRA rank (default: 16)")
@click.option(
    "--save", type=str, required=True, help="Output path for safetensors file"
)
@click.option(
    "--mode",
    type=str,
    default=None,
    help="Generation mode: doc, metadata, or text (auto-inferred if not specified)",
)
def generate(from_metadata, from_text, from_doc, base_model, rank, save, mode):
    """Generate LoRA adapter from metadata, text, or document"""

    # Validate input
    inputs_provided = sum([bool(from_metadata), bool(from_text), bool(from_doc)])
    if inputs_provided == 0:
        click.echo(
            "Error: Must provide one of --from-metadata, --from-text, or --from-doc",
            err=True,
        )
        raise click.Abort()
    if inputs_provided > 1:
        click.echo(
            "Error: Must provide exactly one of --from-metadata, --from-text, or --from-doc",
            err=True,
        )
        raise click.Abort()

    # Determine mode
    if mode is None:
        if from_metadata:
            mode = "metadata"
        elif from_doc:
            mode = "doc"
        else:
            mode = "text"

    # Load input content
    content = None
    if from_metadata:
        # Try parsing as JSON first, then check if it's a file path
        try:
            content = json.loads(from_metadata)
        except json.JSONDecodeError:
            # If JSON parsing fails, try as file path
            try:
                with open(from_metadata, "r") as f:
                    content = json.load(f)
            except (FileNotFoundError, IOError, json.JSONDecodeError):
                click.echo(
                    "Error: --from-metadata must be valid JSON or a file path containing JSON",
                    err=True,
                )
                raise click.Abort()
    elif from_doc:
        # Try as file path first, then use as direct content
        try:
            with open(from_doc, "r") as f:
                content = f.read()
        except (FileNotFoundError, IOError):
            content = from_doc
    else:  # from_text
        content = from_text

    # Generate LoRA weights
    click.echo(
        f"Generating LoRA adapter (mode={mode}, rank={rank}, base_model={base_model})..."
    )

    # Lazy-load heavy dependencies and only import what's needed
    from safetensors.torch import save_file

    try:
        if mode == "metadata":
            from tessera_hypernetwork.metadata_to_lora import MetadataToLoRA

            generator = MetadataToLoRA(base_model, default_rank=rank)
            lora_weights = generator.generate(content, rank)
        elif mode == "doc":
            from tessera_hypernetwork.doc_to_lora import DocToLoRA

            generator = DocToLoRA(base_model, use_shine=True, default_rank=rank)
            lora_weights = generator.generate(content, rank)
        else:  # text
            from tessera_hypernetwork.text_to_lora import TextToLoRA

            generator = TextToLoRA(base_model, default_rank=rank)
            lora_weights = generator.generate(content, rank)

        # Save to safetensors
        output_path = Path(save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(lora_weights, str(output_path))

        click.echo(f"✓ LoRA adapter saved to {output_path}")
        click.echo(f"  - lora_A shape: {lora_weights['lora_A'].shape}")
        click.echo(f"  - lora_B shape: {lora_weights['lora_B'].shape}")

    except Exception as e:
        click.echo(f"Error generating LoRA adapter: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
@click.option(
    "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
)
@click.option(
    "--qdrant-url", type=str, default=None, help="Qdrant vector database URL (optional)"
)
@click.option(
    "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
)
@click.option(
    "--base-model",
    type=str,
    default=None,
    help="Base model to auto-start vLLM with (e.g., mistralai/Mistral-7B-Instruct-v0.2)",
)
@click.option(
    "--vllm-port",
    type=int,
    default=8000,
    help="Port for vLLM server (default: 8000)",
)
def serve(port, host, qdrant_url, workers, base_model, vllm_port):
    """Start the Tessera hypernetwork server"""

    # Auto-start vLLM if base-model specified
    if base_model:
        click.echo(f"Auto-starting vLLM with {base_model} on port {vllm_port}...")
        cache_dir = Path.home() / ".tessera" / "models"
        model_path = cache_dir / (base_model.replace("/", "--"))
        model_arg = str(model_path) if model_path.exists() else base_model

        vllm_cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_arg,
            "--port",
            str(vllm_port),
            "--enable-lora",
        ]

        # Start vLLM in background
        try:
            import subprocess

            vllm_process = subprocess.Popen(
                vllm_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            click.echo(f"✓ vLLM started on port {vllm_port} (PID: {vllm_process.pid})")
        except Exception as e:
            click.echo(f"✗ Failed to start vLLM: {e}", err=True)
            raise click.Abort()

    # Lazy-load uvicorn and server
    import uvicorn
    from tessera_hypernetwork.server import app

    if qdrant_url:
        click.echo(
            f"Starting Tessera server on {host}:{port} with Qdrant at {qdrant_url}"
        )
    else:
        click.echo(f"Starting Tessera server on {host}:{port}")

    uvicorn.run(app, host=host, port=port, workers=workers)


@cli.command()
@click.option(
    "--url",
    type=str,
    default="http://localhost:8000",
    help="Server URL (default: http://localhost:8000)",
)
def health(url):
    """Check server health status"""

    import requests

    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            click.echo(f"✓ Server healthy: {data}")
        else:
            click.echo(f"✗ Server returned status {response.status_code}", err=True)
            raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.echo(f"✗ Failed to connect to server: {e}", err=True)
        raise click.Abort()


@cli.command()
def list():
    """List available base models and their dimensions"""

    # Check for cached models
    cache_dir = Path.home() / ".tessera" / "models"
    cached_models = []
    if cache_dir.exists():
        for model_dir in sorted(cache_dir.iterdir()):
            if model_dir.is_dir():
                size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                model_id = model_dir.name.replace("--", "/")
                cached_models.append((model_id, size))

    # Show cached models
    if cached_models:
        click.echo("Base Models (cached):")
        for model_id, size in cached_models:
            click.echo(f"  {model_id}    {size / 1e9:.1f}GB  ○ not running")
        click.echo("")
    else:
        click.echo("Base Models (cached):")
        click.echo("  No models cached. Run: tessera model pull <model_id>")
        click.echo("")

    # Show known model dimensions
    model_dims = {
        "meta-llama/Llama-3-8B": (4096, 4096),
        "meta-llama/Llama-3-70B": (8192, 8192),
        "Qwen/Qwen2-7B": (3584, 3584),
        "deepseek-ai/DeepSeek-V3": (7168, 7168),
    }

    click.echo("Known model dimensions:")
    click.echo("")
    for model, (d_in, d_out) in model_dims.items():
        click.echo(f"  {model}")
        click.echo(f"    Dimensions: {d_in} x {d_out}")
    click.echo("")
    click.echo("Default dimensions for unknown models: 4096 x 4096")


# Model management commands
@cli.group()
def model():
    """Base model management commands"""
    pass


@model.command()
@click.argument("model_id")
def pull(model_id):
    """Download a base model from HuggingFace Hub and cache locally"""

    from huggingface_hub import snapshot_download

    cache_dir = Path.home() / ".tessera" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Downloading {model_id}...")
    try:
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
        )
        click.echo(f"✓ Model cached at {path}")
        click.echo(f"  Use: tessera model serve {model_id}")
    except Exception as e:
        click.echo(f"✗ Failed to download model: {e}", err=True)
        raise click.Abort()


@model.command()
@click.argument("model_id")
@click.option("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=None,
    help="GPU memory utilization fraction (e.g., 0.9)",
)
@click.option(
    "--tensor-parallel-size", type=int, default=1, help="Tensor parallel size (default: 1)"
)
@click.option(
    "--quantization",
    type=str,
    default=None,
    help="Quantization method (e.g., awq, gptq, bitsandbytes)",
)
@click.option(
    "--max-model-len", type=int, default=8192, help="Maximum model length (default: 8192)"
)
def serve_model(model_id, port, gpu_memory_utilization, tensor_parallel_size, quantization, max_model_len):
    """Start vLLM with a specified base model"""

    cache_dir = Path.home() / ".tessera" / "models"
    model_path = cache_dir / (model_id.replace("/", "--"))

    # Use cached path if available, otherwise let vLLM download
    model_arg = str(model_path) if model_path.exists() else model_id

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_arg,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--enable-lora",  # always enable LoRA for Tessera
    ]
    if gpu_memory_utilization:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if quantization:
        cmd += ["--quantization", quantization]

    click.echo(f"Starting vLLM with {model_id} on port {port}...")
    click.echo(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        click.echo("\n✓ vLLM server stopped")
    except Exception as e:
        click.echo(f"✗ Failed to start vLLM: {e}", err=True)
        raise click.Abort()


@model.command()
def list_models():
    """List all locally cached base models"""

    cache_dir = Path.home() / ".tessera" / "models"
    if not cache_dir.exists():
        click.echo("No models cached. Run: tessera model pull <model_id>")
        return

    click.echo("Cached base models:")
    for model_dir in sorted(cache_dir.iterdir()):
        if model_dir.is_dir():
            size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            model_id = model_dir.name.replace("--", "/")
            click.echo(f"  {model_id}: {size / 1e9:.1f}GB")


@model.command()
@click.argument("model_id")
def remove(model_id):
    """Remove a cached base model to free disk space"""

    cache_dir = Path.home() / ".tessera" / "models"
    model_path = cache_dir / (model_id.replace("/", "--"))
    if model_path.exists():
        shutil.rmtree(model_path)
        click.echo(f"✓ Removed {model_id}")
    else:
        click.echo(f"Model not found: {model_id}", err=True)
        raise click.Abort()


# LoRAX commands
@cli.group()
def lorax():
    """LoRAX adapter management commands"""
    pass


@lorax.command()
@click.option(
    "--path", type=str, required=True, help="Path to the adapter safetensors file"
)
@click.option("--name", type=str, required=True, help="Name for the adapter")
@click.option("--base-model", type=str, required=True, help="Base model identifier")
@click.option(
    "--server-url",
    type=str,
    default="http://localhost:8000",
    help="Tessera hypernetwork server URL (default: http://localhost:8000)",
)
def import_adapter(path, name, base_model, server_url):
    """Import an adapter into the Tessera hypernetwork service"""

    import requests

    click.echo(f"Importing adapter '{name}' from {path} to Tessera at {server_url}")

    # Read the adapter file
    try:
        with open(path, "rb") as f:
            adapter_data = f.read()
    except FileNotFoundError:
        click.echo(f"Error: Adapter file not found at {path}", err=True)
        raise click.Abort()

    # Send to Tessera server
    try:
        response = requests.post(
            f"{server_url}/v1/adapters",
            files={"file": (Path(path).name, adapter_data, "application/octet-stream")},
            data={
                "adapter_name": name,
                "base_model": base_model,
            },
            timeout=30,
        )

        if response.status_code == 200:
            click.echo(f"✓ Adapter '{name}' imported successfully")
        else:
            click.echo(
                f"✗ Failed to import adapter: {response.status_code} - {response.text}",
                err=True,
            )
            raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.echo(f"✗ Failed to connect to Tessera server: {e}", err=True)
        raise click.Abort()


@lorax.command()
@click.option(
    "--server-url",
    type=str,
    default="http://localhost:8000",
    help="Tessera hypernetwork server URL (default: http://localhost:8000)",
)
def list_adapters(server_url):
    """List adapters loaded in the Tessera hypernetwork service"""

    import requests

    try:
        response = requests.get(f"{server_url}/v1/adapters", timeout=10)

        if response.status_code == 200:
            adapters = response.json()
            if adapters:
                click.echo("Loaded adapters:")
                for adapter in adapters:
                    click.echo(
                        f"  - {adapter.get('name', 'unknown')}: {adapter.get('base_model', 'unknown')} ({adapter.get('size', 0)} bytes)"
                    )
            else:
                click.echo("No adapters loaded")
        else:
            click.echo(
                f"✗ Failed to list adapters: {response.status_code} - {response.text}",
                err=True,
            )
            raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.echo(f"✗ Failed to connect to Tessera server: {e}", err=True)
        raise click.Abort()


@lorax.command()
@click.option("--name", type=str, required=True, help="Name of the adapter to unload")
@click.option(
    "--server-url",
    type=str,
    default="http://localhost:8000",
    help="Tessera hypernetwork server URL (default: http://localhost:8000)",
)
def unload(name, server_url):
    """Unload an adapter from the Tessera hypernetwork service"""

    import requests

    click.echo(f"Unloading adapter '{name}' from Tessera at {server_url}")

    try:
        response = requests.delete(f"{server_url}/v1/adapters/{name}", timeout=10)

        if response.status_code == 200:
            click.echo(f"✓ Adapter '{name}' unloaded successfully")
        else:
            click.echo(
                f"✗ Failed to unload adapter: {response.status_code} - {response.text}",
                err=True,
            )
            raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.echo(f"✗ Failed to connect to Tessera server: {e}", err=True)
        raise click.Abort()


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
