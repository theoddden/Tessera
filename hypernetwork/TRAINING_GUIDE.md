# Tessera Hypernetwork Training Guide

This guide explains how to train the hypernetwork to generate asymmetric LoRA adapters that improve domain-specific performance.

## Overview

The training pipeline uses **offline distillation**:
1. Train hypernetwork to predict LoRA weight deltas from metadata
2. Target deltas come from domain-specific synthetic targets (or real fine-tuned models)
3. Evaluate trained hypernetwork using the benchmark pipeline
4. Compare results against baseline (zero-weight adapters)

## Prerequisites

- Python 3.8+
- CUDA GPU (recommended for training)
- Metadata packets in JSON format
- vLLM server running for evaluation

## Quick Start

### 1. Train the Hypernetwork

```bash
python -m tessera_hypernetwork.train_hypernetwork \
  --metadata-dir /path/to/metadata_packets \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --rank 16 \
  --epochs 50 \
  --output-dir ./checkpoints \
  --use-curriculum \
  --check-similarity \
  --check-contamination
```

**Key flags:**
- `--use-curriculum`: Enable 5-stage curriculum (law → medical → CS → statistics → econometrics)
- `--check-similarity`: Monitor encoder collapse via embedding similarity
- `--check-contamination`: Monitor cross-domain adapter contamination

### 2. Start Server with Trained Checkpoint

```bash
export TESSERA_CHECKPOINT_PATH=./checkpoints/best_hypernetwork.pt
tessera serve --port 8080
```

The server will automatically load the trained hypernetwork and use it for metadata-mode generation.

### 3. Generate Adapters with Trained Hypernetwork

```bash
for meta in /path/to/metadata_packets/*.json; do
  tessera generate \
    --from-metadata "$(cat $meta)" \
    --base-model mistralai/Mistral-7B-Instruct-v0.2 \
    --rank 16 \
    --save ./trained_adapters/$(basename $meta .json).safetensors
done
```

### 4. Import Adapters into Tessera

```bash
for adapter in ./trained_adapters/*.safetensors; do
  tessera lorax import-adapter \
    --file "$adapter" \
    --adapter-name "$(basename $adapter .safetensors)" \
    --base-model mistralai/Mistral-7B-Instruct-v0.2
done
```

### 5. Run Benchmark Evaluation

```bash
lm_eval \
  --model local-completions \
  --model_args base_url=http://localhost:8080/v1/completions \
  --tasks mmlu_abstract_algebra,mmlu_anatomy,mmlu_business_ethics \
  --output_path ./trained_results
```

### 6. Compare Results

```bash
python -m tessera_hypernetwork.train_and_evaluate \
  --metadata-dir /path/to/metadata_packets \
  --baseline-results ./baseline_results/results.json \
  --skip-training \
  --skip-evaluation
```

## What to Expect

### Baseline (v0.2.17)
- All adapters have zero weights (PlaceholderHypernetwork)
- Delta: ~0% across all domains
- This is the correct baseline for untrained hypernetwork

### After Training (v0.2.18+)
With synthetic targets and domain-conditioned training:
- **Early training (epochs 1-10)**: May see negative deltas (degradation) as model learns
- **Mid training (epochs 10-30)**: Delta approaches 0 as model learns not to hurt performance
- **Late training (epochs 30-50)**: Positive deltas emerge for high-distinctiveness domains (law, medical)

**Expected asymmetric results:**
- International law: +2-5% (high vocabulary distinctiveness)
- Jurisprudence: +1-3% (high vocabulary distinctiveness)
- Medical: +1-2% (moderate distinctiveness)
- Computer science: 0-1% (lower distinctiveness)
- Statistics: -1-0% (hard domain, low distinctiveness)
- Econometrics: -1-0% (hardest domain, lowest distinctiveness)

## Interpreting Training Metrics

### Embedding Similarity
- **< 0.8**: Good separation between domains
- **0.8-0.95**: Some collapse, monitor closely
- **> 0.95**: Encoder collapse detected, domains are too similar

### Cross-Domain Contamination
- **< 0.3**: Good separation, adapters are domain-specific
- **0.3-0.6**: Moderate contamination, acceptable
- **> 0.6**: High contamination, adapters are leaking across domains

### Validation Loss
- Should decrease monotonically
- Sudden increases may indicate overfitting or curriculum stage transition issues

## Troubleshooting

### Training Fails with CUDA Error
```bash
# Use CPU instead (slower)
python -m tessera_hypernetwork.train_hypernetwork \
  --metadata-dir /path/to/metadata_packets \
  --device cpu
```

### Encoder Collapse Detected
- Reduce learning rate: `--lr 5e-4`
- Increase dropout in hypernetwork (edit `train_hypernetwork.py`)
- Use richer metadata encoding (add more fields)

### High Cross-Domain Contamination
- Increase domain embedding dimension
- Add more domain-specific scaling factors
- Train longer on early curriculum stages

### No Improvement Over Baseline
- Increase training epochs: `--epochs 100`
- Use real fine-tuned targets instead of synthetic
- Reduce LoRA rank for cleaner signal: `--rank 8`
- Add more domain-specific training data

## Advanced: Real Fine-Tuned Targets

For production results, replace synthetic targets with real LoRA deltas from domain-specific fine-tuning:

1. Fine-tune Mistral-7B on domain-specific data (e.g., legal documents)
2. Extract LoRA weights from fine-tuned model
3. Compute delta: `fine_tuned_weights - base_weights`
4. Save deltas as safetensors in `--targets-dir`
5. Train hypernetwork with `--targets-dir /path/to/deltas`

This provides the strongest training signal and should yield the best asymmetric results.

## Next Steps

1. **Run the full pipeline** using the orchestration script
2. **Monitor metrics** during training (similarity, contamination, loss)
3. **Iterate on hyperparameters** based on results
4. **Scale to full MMLU** (57 tasks) once positive deltas emerge
5. **Production deployment** with real fine-tuned targets
