# Tessera Hypernetwork

**Generate per-session LoRA adapters for inference tasks using hypernetwork synthesis.**

**Features**

Metadata-to-LoRA: Generate adapters from structured user metadata (JSON)
Text-to-LoRA: Generate adapters from natural language descriptions
Doc-to-LoRA with SHINE: Generate adapters from document content using SHINE (ICML 2026) for long-context internalization
FastAPI: Modern async Python web framework
OpenAI-compatible API: Easy integration with existing tooling
Installation
pip install tessera-hypernetwork
Quick Start

# Commands

**Generate:**

**From metadata (JSON string or file)**

tessera generate \
  --from-metadata '{"task": "classification", "domain": "medical"}' \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --rank 16 \
  --save ./adapter.safetensors

**From text description:**

tessera generate \
  --from-text "Medical diagnosis assistant" \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --rank 16 \
  --save ./adapter.safetensors

**From document:**

tessera generate \
  --from-doc ./document.txt \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --rank 16 \
  --save ./adapter.safetensors

**Options:**

--from-metadata: JSON metadata string or file path
--from-text: Natural language description
--from-doc: Document content or file path
--base-model: Base model identifier (default: mistralai/Mistral-7B-Instruct-v0.2)
--rank: LoRA rank (default: 16)
--save: Output path for safetensors file (required)
--mode: Generation mode: doc, metadata, or text (auto-inferred if not specified)
Serve

**Start the hypernetwork server:**

tessera serve --port 8000 --host 0.0.0.0
Options:

--port: Port to serve on (default: 8000)
--host: Host to bind to (default: 0.0.0.0)
--qdrant-url: Qdrant vector database URL (optional)
--workers: Number of worker processes (default: 1)
Health

**Check server health status:**

tessera health --url http://localhost:8000
Options:

--url: Server URL (default: http://localhost:8000)
List
List available base models and their dimensions:

tessera list

# LoRAX Adapter Management

Import, list, and unload adapters:

**Import an adapter:**

tessera lorax import-adapter \
  --path ./adapter.safetensors \
  --name my-adapter \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --server-url http://localhost:8000

**List loaded adapters:**

tessera lorax list-adapters --server-url http://localhost:8000

**Unload an adapter:**

tessera lorax unload --name my-adapter --server-url http://localhost:8000

**API Endpoints**

The hypernetwork service provides a FastAPI server with the following endpoints:

POST /v1/generate - Generate a LoRA adapter for a given prompt
GET /health - Health check endpoint
POST /v1/adapters - Import adapter safetensors
GET /v1/adapters - List loaded adapters
DELETE /v1/adapters/{name} - Unload adapter

License
Apache-2.0
