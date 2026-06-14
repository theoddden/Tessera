# Tessera Hypernetwork

Generate per-session LoRA adapters for inference tasks.

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

## License

Apache-2.0
