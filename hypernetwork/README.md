# Tessera Hypernetwork

Generate per-session LoRA adapters for inference tasks. This is the Python hypernetwork service component of Tessera, which works alongside the Rust core to provide LoRA adapter generation via hypernetwork synthesis.

## Features

- **Doc-to-LoRA with SHINE**: Generate adapters from document content using SHINE (ICML 2026) for long-context internalization
- **Text-to-LoRA**: Generate adapters from natural language descriptions
- **Metadata-to-LoRA**: Generate adapters from structured user metadata
- **OpenAI-compatible API**: Easy integration with existing tooling
- **FastAPI**: Modern async Python web framework

## Installation

```bash
pip install tessera-hypernetwork
```

## Usage

Start the hypernetwork server:

```bash
python -m tessera_hypernetwork.server
```

Or use the CLI:

```bash
tessera-hypernetwork serve
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

The complete Tessera system provides a comprehensive CLI for all operations:

```bash
# Show version
tessera --version

# Generate a LoRA adapter
tessera generate "Senior litigation associate specializing in IP law" \
  --base-model meta-llama/Llama-3-8B \
  --rank 16 \
  --output ./adapter.safetensors

# Start the API server
tessera serve --port 8080

# Check if Tessera is running
tessera health --url http://localhost:8080

# List cached adapters
tessera list
tessera list --base-model meta-llama/Llama-3-8B

# Cache management
tessera cache clear
tessera cache stats
tessera cache prune --max-age-days 7

# LoRAx operations
tessera lorax import --path ./adapter.safetensors --name my-adapter
tessera lorax list
tessera lorax unload --name my-adapter

# PEFT operations
tessera peft import --path ./adapter.safetensors --name my-adapter
tessera peft unload --name my-adapter
```

For the complete Tessera system, see: https://github.com/theoddden/Tessera

## License

Apache-2.0
