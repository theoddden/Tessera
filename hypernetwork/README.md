# Tessera Hypernetwork

Generate per-session LoRA adapters for inference tasks. This is the Python hypernetwork service component of Tessera, which works alongside the Rust core to provide LoRA adapter generation via hypernetwork synthesis.

## Features

- **Doc-to-LoRA with SHINE**: Generate adapters from document content using SHINE (ICML 2026) for long-context internalization
- **Text-to-LoRA**: Generate adapters from natural language descriptions
- **Metadata-to-LoRA**: Generate adapters from structured user metadata
- **LoRAX-style Adapter Management**: Import, list, and unload adapters via CLI and API
- **OpenAI-compatible Completions**: `/v1/completions` endpoint for lm_eval integration
- **OpenAI-compatible API**: Easy integration with existing tooling
- **FastAPI**: Modern async Python web framework

## Installation

```bash
pip install tessera-hypernetwork
```

## Usage

### LoRAX Adapter Management

The hypernetwork service provides LoRAX-style adapter management for loading and serving LoRA adapters:

```bash
# Import an adapter into the service
tessera lorax import --path ./adapter.safetensors --name my-adapter --base-model meta-llama/Llama-3-8B --server-url http://localhost:8000

# List loaded adapters
tessera lorax list --server-url http://localhost:8000

# Unload an adapter
tessera lorax unload --name my-adapter --server-url http://localhost:8000
```

**API Endpoints**:
- `POST /v1/adapters` - Import adapter safetensors
- `GET /v1/adapters` - List loaded adapters
- `DELETE /v1/adapters/{name}` - Unload adapter

### OpenAI-Compatible Completions

The `/v1/completions` endpoint provides OpenAI-compatible completions for lm_eval integration:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-adapter",
    "prompt": "Hello, world!",
    "max_tokens": 10
  }'
```

The endpoint looks up the adapter by name from the loaded adapters registry and forwards the request to vLLM.

### CLI Commands

The `tessera` CLI provides commands for generating LoRA adapters and running the hypernetwork server:

```bash
# Generate LoRA adapter from metadata
tessera generate --from-metadata '{"task": "classification", "domain": "medical"}' \
  --base-model meta-llama/Llama-3-8B \
  --rank 16 \
  --save ./adapter.safetensors

# Generate LoRA adapter from text description
tessera generate --from-text "Senior litigation associate specializing in IP law" \
  --base-model meta-llama/Llama-3-8B \
  --rank 16 \
  --save ./adapter.safetensors

# Generate LoRA adapter from document
tessera generate --from-doc ./document.txt \
  --base-model meta-llama/Llama-3-8B \
  --rank 16 \
  --save ./adapter.safetensors

# Start the hypernetwork server
tessera serve --port 8080 --host 0.0.0.0

# Start server with Qdrant vector database
tessera serve --port 8080 --qdrant-url http://localhost:6333

# Check server health
tessera health --url http://localhost:8000

# List available base models
tessera list
```

### Server Mode

You can also run the server directly:

```bash
python -m tessera_hypernetwork.server
```

## API

The hypernetwork service provides a FastAPI server with the following endpoints:

- `POST /v1/generate` - Generate a LoRA adapter for a given prompt
- `GET /health` - Health check endpoint

## Development

Install development dependencies:

```bash
pip install tessera-hypernetwork[dev]
```

Run tests:

```bash
pytest
```

## Integration with Tessera

This hypernetwork service is designed to work with the Tessera Rust core. The Rust core handles semantic caching, vector similarity search, and adapter composition, while this Python service handles the actual LoRA adapter generation via hypernetwork synthesis.

### Full Tessera CLI Lifecycle

The Tessera hypernetwork service provides a comprehensive CLI for LoRA adapter generation and serving:

```bash
# Generate LoRA adapter from metadata (JSON string or file)
tessera generate --from-metadata '{"task": "classification", "domain": "medical"}' \
  --base-model meta-llama/Llama-3-8B \
  --rank 16 \
  --save ./adapter.safetensors

# Generate LoRA adapter from natural language description
tessera generate --from-text "Senior litigation associate specializing in IP law" \
  --base-model meta-llama/Llama-3-8B \
  --rank 16 \
  --save ./adapter.safetensors

# Generate LoRA adapter from document content
tessera generate --from-doc ./document.txt \
  --base-model meta-llama/Llama-3-8B \
  --rank 16 \
  --save ./adapter.safetensors

# Start the hypernetwork server
tessera serve --port 8080 --host 0.0.0.0

# Start server with Qdrant vector database integration
tessera serve --port 8080 --qdrant-url http://localhost:6333

# Check server health status
tessera health --url http://localhost:8000

# List available base models and their dimensions
tessera list

# LoRAX adapter management
tessera lorax import --path ./adapter.safetensors --name my-adapter --base-model meta-llama/Llama-3-8B --server-url http://localhost:8000
tessera lorax list --server-url http://localhost:8000
tessera lorax unload --name my-adapter --server-url http://localhost:8000
```

For the complete Tessera system, see: https://github.com/theoddden/Tessera

## License

Apache-2.0
