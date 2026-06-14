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

For the complete Tessera system, see: https://github.com/theoddden/Tessera

## License

Apache-2.0
