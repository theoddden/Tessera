# Tessera v0.2.8

**Free, open-source LoRA adapter generation API. Metadata in → LoRA adapter out → fast.**

Tessera is a high-performance LoRA adapter generation service that takes user metadata, documents, or task descriptions and returns personalized LoRA adapters via hypernetwork synthesis. It does not serve inference — it only generates adapters that you can load into your own inference stack (Terradev, vLLM, LoRAX, etc.).

**GitHub**: https://github.com/theoddden/Tessera

## Core Value Proposition

- **Generation latency**: Get personalized adapters into the caller's hands as fast as physics allows
- **Atomic skill composition**: Maintain a library of atomic skill adapters, compose them at query time with hypernetwork-predicted mixing weights
- **Cross-architecture support**: Generate adapters for any supported base model from a single universal latent space
- **Cache hit rate approaching 100%**: Cache atomic components, not compositions. Compositions are computed in microseconds from cached components

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              TESSERA RUST CORE (axum HTTP server)           │
│                                                             │
│  POST /generate                                             │
│      │                                                      │
│      ▼                                                      │
│  [RUST] Embed metadata (candle, SIMD)        ~5ms          │
│      │                                                      │
│      ▼                                                      │
│  [RUST] Qdrant similarity search             ~10ms         │
│      │                                                      │
│      ├── HIT → deserialize weights (safetensors) ~20ms     │
│      │         return adapter bytes                         │
│      │                                                      │
│      └── MISS → [RUST] call hypernetwork HTTP  ~300-800ms  │
│                 [RUST] receive + validate weights           │
│                 [RUST] store in cache (async)               │
│                 return adapter bytes                        │
│                                                             │
│  [RUST] LSTM prefetch runs background thread               │
│  [RUST] Cache eviction runs background thread              │
└─────────────────────────────────────────────────────────────┘
         │
         │  (only on cache miss, async)
         ▼
┌─────────────────────────────────────────────────────────────┐
│         HYPERNETWORK SERVICE (Python/PyTorch)               │
│                                                             │
│  Doc-to-LoRA / Text-to-LoRA / Metadata-to-LoRA            │
│  OpenAI-compatible API                                      │
│  Returns: safetensors adapter weights                       │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Rust Core (Hot Path)
- **HTTP API server**: Axum framework, zero-copy request handling
- **Semantic embedding**: Candle-based MiniLM encoder, SIMD-accelerated
- **Vector similarity search**: Native Qdrant client, no Python overhead
- **LSTM prefetch predictor**: Background demand prediction for proactive cache warming
- **Adapter weight serialization**: safetensors-rs, zero-copy tensor I/O
- **Cache state management**: SQLite via rusqlite, lock-free reads
- **Hypernetwork HTTP client**: reqwest with async connection pooling
- **Adapter file I/O**: tokio fs, async and zero-copy where possible

### Python Hypernetwork Service
- **Doc-to-LoRA with SHINE**: Generate adapters from document content using SHINE (ICML 2026) for long-context internalization. Handles contexts 5x longer than standard context windows through hierarchical information extraction and attention-based compression.
- **Text-to-LoRA**: Generate adapters from natural language descriptions
- **Metadata-to-LoRA**: Generate adapters from structured user metadata
- **OpenAI-compatible API**: Easy integration with existing tooling
- **FastAPI**: Modern async Python web framework

### Advanced Features
- **Atomic skill composition**: Compose multiple skill adapters into user-specific blends
- **Cross-architecture decoding**: Generate adapters for any supported base model from universal latent space
- **Semantic caching**: Qdrant-powered vector similarity search for cache hits
- **Predictive prefetching**: LSTM-based demand prediction for proactive cache warming

## Quick Start

### Prerequisites
- Rust 1.88+
- Python 3.10+
- Docker (for containerized deployment)
- NVIDIA GPU (for hypernetwork service)

### Installation

#### Option 1: Build from Source

```bash
git clone https://github.com/theoddden/Tessera.git
cd tessera
cargo build --release
```

#### Option 2: Install Hypernetwork Service via PyPI

```bash
pip install tessera-hypernetwork
```

### CLI Commands

The hypernetwork service provides CLI commands for LoRA adapter generation and serving:

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
tessera lorax import --path ./adapter.safetensors --name my-adapter --base-model meta-llama/Llama-3-8B
tessera lorax list
tessera lorax unload --name my-adapter
```

**Note**: The full Tessera Rust core (cache management, advanced composition features) is under development. The hypernetwork service currently provides the core generation, serving, and LoRAX integration functionality.

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/theoddden/Tessera.git
cd tessera
```

2. **Start the stack with Docker Compose**
```bash
docker-compose up -d
```

This starts:
- Tessera Rust core on port 8080
- Hypernetwork service on port 8000
- Qdrant vector database on port 6333

3. **Generate an adapter via CLI**
```bash
tessera generate --from-text "Senior litigation associate specializing in IP law" \
  --base-model meta-llama/Llama-3-8B \
  --rank 16 \
  --save ./adapter.safetensors
```

4. **Or generate via HTTP API**
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "context": {
      "description": "Senior litigation associate specializing in IP law",
      "domain": "legal"
    },
    "base_model": "meta-llama/Llama-3-8B",
    "target_rank": 16
  }' \
  --output adapter.safetensors
```

5. **Load adapter into Terradev**
```bash
terradev lora add \
  --endpoint http://localhost:8000 \
  --name user-123-session \
  --path ./adapter.safetensors
```

### Complete Lifecycle Example

```bash
# 1. Start the hypernetwork server
tessera serve --port 8000

# 2. In another terminal, check health
tessera health --url http://localhost:8000

# 3. Generate an adapter from text
tessera generate --from-text "Expert in quantum computing algorithms" \
  --base-model meta-llama/Llama-3-8B \
  --rank 16 \
  --save ./quantum-expert.safetensors

# 4. Generate an adapter from metadata
tessera generate --from-metadata '{"task": "classification", "domain": "science"}' \
  --base-model meta-llama/Llama-3-8B \
  --rank 16 \
  --save ./science-classifier.safetensors

# 5. List available base models
tessera list
```

**Note**: Cache management, LoRAx/PEFT operations, and advanced composition features are part of the Tessera Rust core, which is under development. The hypernetwork service currently provides the core generation and serving functionality.

## API Reference

### POST /generate

Generate a LoRA adapter from user context.

**Request**
```json
{
  "user_id": "string",
  "context": {
    "documents": ["string"],  // Optional: document content
    "description": "string", // Optional: task description
    "metadata": {},          // Optional: structured user data
    "domain": "string"       // Optional: domain hint
  },
  "base_model": "meta-llama/Llama-3-8B",
  "target_rank": 16,
  "response_format": "file"  // "file" | "base64" | "url"
}
```

**Response**
```json
{
  "adapter_id": "uuid",
  "adapter": "bytes",
  "base_model": "meta-llama/Llama-3-8B",
  "rank": 16,
  "cache_hit": false,
  "cache_similarity": null,
  "generation_latency_ms": 450,
  "total_latency_ms": 475,
  "archetype_id": "uuid",
  "archetype_label": "text_adapter",
  "metadata": {
    "created_at": "2026-01-01T00:00:00Z",
    "expires_at": null,
    "source_type": "text",
    "estimated_quality": 0.85,
    "recommended_vllm_args": ["--enable-lora", "--max-lora-rank 16"]
  }
}
```

### GET /adapter/{adapter_id}

Retrieve a cached adapter.

### POST /embed

Embed context and check cache without generating.

### GET /health

Health check endpoint.

### GET /metrics

Prometheus metrics endpoint.

## Atomic Skill Composition

Instead of generating monolithic adapters per user, Tessera maintains a library of atomic skill adapters and composes them at query time.

### Architecture

```
User metadata arrives
        ↓
Semantic lookup: which skill adapters are relevant?
["legal_contracts", "finance_valuation", "technical_writing"]
        ↓
Hypernetwork predicts mixing weights:
[0.6, 0.3, 0.1]
        ↓
Compose: adapter = 0.6×legal + 0.3×finance + 0.1×writing
        ↓
Return composed adapter
```

### Benefits

- **Exponential composition space**: 50 atomic skills → millions of unique user-specific compositions
- **Near 100% cache hit rate**: Cache atomic components, not compositions
- **Sub-millisecond composition**: Weighted sum of pre-loaded adapter matrices
- **Compounding moat**: Every new skill added makes every existing user's adapter better

## Cross-Architecture Support

Tessera uses a shared encoder with architecture-specific decoder heads to generate adapters for any supported base model.

### Architecture

```
User metadata / document
        ↓
ENCODER (architecture-agnostic)
Learns: "what does this user need?"
Output: latent skill vector z ∈ ℝᵈ
        ↓
        z
        ↓
DECODER (architecture-specific head)
Learns: "how to express skill z in this model's weight space?"
Output: LoRA weights shaped for target architecture
```

### Supported Models

- meta-llama/Llama-3-8B
- meta-llama/Llama-3-70B
- Qwen/Qwen2-7B
- deepseek-ai/DeepSeek-V3

## Latency Budget

| Operation | Implementation | Target | Realistic |
|---|---|---|---|
| HTTP request parse | Rust (axum) | <1ms | <1ms |
| Context serialization | Rust | <1ms | <1ms |
| Embedding (MiniLM-L6) | Rust (candle, CPU SIMD) | <5ms | 3-8ms |
| Qdrant vector search | Rust (native client) | <10ms | 5-15ms |
| Adapter file load (cache hit) | Rust (tokio fs) | <20ms | 10-30ms |
| HTTP response serialization | Rust | <1ms | <1ms |
| **Total cache hit** | **All Rust** | **<40ms** | **20-55ms** |
| | | | |
| Hypernetwork HTTP call | Rust (reqwest) | <5ms overhead | <5ms |
| Hypernetwork forward pass | Python (PyTorch) | 300-800ms | 300-800ms |
| Adapter validation | Rust (safetensors) | <5ms | <5ms |
| Adapter file save (async) | Rust (tokio fs, background) | 0ms blocking | 0ms blocking |
| **Total cache miss** | **Rust + Python** | **<820ms** | **320-820ms** |

## Configuration

Environment variables for Tessera Rust core:

| Variable | Default | Description |
|---|---|---|
| PORT | 8080 | HTTP server port |
| QDRANT_URL | http://localhost:6333 | Qdrant vector database URL |
| QDRANT_COLLECTION | tessera_adapters | Qdrant collection name |
| SIMILARITY_THRESHOLD | 0.92 | Semantic cache similarity threshold |
| EMBEDDING_MODEL | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| HYPERNETWORK_URL | http://localhost:8000 | Hypernetwork service URL |
| GENERATION_TIMEOUT_MS | 30000 | Hypernetwork generation timeout |
| ADAPTER_STORE_PATH | ./adapters | Adapter storage directory |
| PREFETCH_HORIZON_MINUTES | 60 | Prefetch prediction horizon |
| PREFETCH_TOP_K | 10 | Number of archetypes to prefetch |
| LOG_LEVEL | info | Logging level |
| CACHE_DB_PATH | ./cache.db | SQLite cache database path |

## Development

### Building Rust core
```bash
cargo build --release
```

### Running Rust core
```bash
cargo run
```

### Running hypernetwork service
```bash
cd hypernetwork
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Running tests
```bash
cargo test
```

## License

Apache 2.0

## Relationship to Terradev

Tessera generates adapters. Terradev loads and serves them.

```bash
# Get adapter from Tessera
curl -X POST https://tessera.local/generate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "context": {
      "description": "Senior litigation associate specializing in IP law",
      "domain": "legal"
    },
    "base_model": "meta-llama/Llama-3-8B",
    "target_rank": 16
  }' \
  --output adapter.safetensors

# Load adapter into Terradev
terradev lora add \
  --endpoint http://localhost:8000 \
  --name user-123-session \
  --path ./adapter.safetensors
```

Two tools. One clean interface between them. Tessera is free and open. Terradev is free and open. Together they are the complete personalized inference stack for the 6-18 month market.
