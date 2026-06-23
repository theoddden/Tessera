# Tessera Hypernetwork

**Generate per-session LoRA adapters for inference tasks using hypernetwork synthesis.**

**Version:** 1.3.10

## Features

- **Metadata-to-LoRA**: Generate adapters from structured user metadata (JSON)
- **Text-to-LoRA**: Generate adapters from natural language descriptions
- **Doc-to-LoRA with SHINE**: Generate adapters from document content using SHINE (ICML 2026) for long-context internalization
- **Base Model Management**: Download, cache, and serve base models with vLLM integration
- **FastAPI**: Modern async Python web framework
- **OpenAI-compatible API**: Easy integration with existing tooling

## Installation

```bash
pip install tessera-hypernetwork
```

## Quick Start

### Generate LoRA Adapters

**From metadata (JSON string or file):**

```bash
tessera generate \
  --from-metadata '{"task": "classification", "domain": "medical"}' \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --rank 16 \
  --save ./adapter.safetensors
```

**From text description:**

```bash
tessera generate \
  --from-text "Medical diagnosis assistant" \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --rank 16 \
  --save ./adapter.safetensors
```

**From document:**

```bash
tessera generate \
  --from-doc ./document.txt \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --rank 16 \
  --save ./adapter.safetensors
```

### Base Model Management

**Download a base model from HuggingFace Hub:**

```bash
tessera model pull mistralai/Mistral-7B-Instruct-v0.2
tessera model pull meta-llama/Llama-3.1-8B-Instruct
tessera model pull deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

**Start vLLM with a base model:**

```bash
tessera model serve-model mistralai/Mistral-7B-Instruct-v0.2 --port 8000
tessera model serve-model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.9
tessera model serve-model mistralai/Mistral-7B-Instruct-v0.2 --quantization awq
```

**List cached base models:**

```bash
tessera model list-models
```

**Remove a cached model:**

```bash
tessera model remove mistralai/Mistral-7B-Instruct-v0.2
```

### Start Tessera Server

**Start the hypernetwork server (with auto vLLM):**

```bash
tessera serve --port 8080 --base-model mistralai/Mistral-7B-Instruct-v0.2
```

**Start the hypernetwork server (standalone):**

```bash
tessera serve --port 8080 --host 0.0.0.0
```

### Check Server Health

```bash
tessera health --url http://localhost:8080
```

### List Available Models

```bash
tessera list
```

## Commands

### Generate

Generate LoRA adapters from metadata, text, or documents.

**Options:**
- `--from-metadata`: JSON metadata string or file path
- `--from-text`: Natural language description
- `--from-doc`: Document content or file path
- `--base-model`: Base model identifier (default: mistralai/Mistral-7B-Instruct-v0.2)
- `--rank`: LoRA rank (default: 16)
- `--save`: Output path for safetensors file (required)
- `--mode`: Generation mode: doc, metadata, or text (auto-inferred if not specified)

### Model Management

Manage base models for vLLM serving.

**tessera model pull `<model_id>`**
Download a base model from HuggingFace Hub and cache locally.

**tessera model serve-model `<model_id>`**
Start vLLM with a specified base model.

**Options:**
- `--port`: Port to serve on (default: 8000)
- `--gpu-memory-utilization`: GPU memory utilization fraction (e.g., 0.9)
- `--tensor-parallel-size`: Tensor parallel size (default: 1)
- `--quantization`: Quantization method (e.g., awq, gptq, bitsandbytes)
- `--max-model-len`: Maximum model length (default: 8192)

**tessera model list-models**
List all locally cached base models.

**tessera model remove `<model_id>`**
Remove a cached base model to free disk space.

### Serve

Start the Tessera hypernetwork server.

**Options:**
- `--port`: Port to serve on (default: 8080)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--qdrant-url`: Qdrant vector database URL (optional)
- `--workers`: Number of worker processes (default: 1)
- `--base-model`: Base model to auto-start vLLM with (e.g., mistralai/Mistral-7B-Instruct-v0.2)
- `--vllm-port`: Port for vLLM server (default: 8000)

### Health

Check server health status.

**Options:**
- `--url`: Server URL (default: http://localhost:8080)

### List

List available base models and their dimensions, plus cached models.

## LoRAX Adapter Management

Import, list, and unload adapters:

**Import an adapter:**

```bash
tessera lorax import-adapter \
  --path ./adapter.safetensors \
  --name my-adapter \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --server-url http://localhost:8080
```

**List loaded adapters:**

```bash
tessera lorax list-adapters --server-url http://localhost:8080
```

**Unload an adapter:**

```bash
tessera lorax unload --name my-adapter --server-url http://localhost:8080
```

## API Endpoints

The hypernetwork service provides a FastAPI server with the following endpoints:

- `POST /v1/generate` - Generate a LoRA adapter for a given prompt
- `GET /health` - Health check endpoint
- `POST /v1/adapters` - Import adapter safetensors
- `GET /v1/adapters` - List loaded adapters
- `DELETE /v1/adapters/{name}` - Unload adapter

## License

Apache-2.0
