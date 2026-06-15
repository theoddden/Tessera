from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import torch
from safetensors.torch import save as save_safetensors
from fastapi.responses import Response
import io
from typing import Optional, List, Dict, Union
from tessera_hypernetwork.doc_to_lora import DocToLoRA
from functools import lru_cache
import requests
from transformers import AutoTokenizer
import os
import time

app = FastAPI(title="Tessera Hypernetwork Service")

# In-memory adapter storage (for LoRAX-style management)
loaded_adapters: Dict[str, Dict] = {}


# Cache tokenizers for base models
@lru_cache(maxsize=4)
def get_tokenizer_cached(base_model: str):
    """Cached tokenizer getter with LRU eviction"""
    return AutoTokenizer.from_pretrained(base_model)


# Cache hypernetwork models with capacity limit (max 4 models)
@lru_cache(maxsize=4)
def get_hypernetwork_cached(base_model: str, mode: str):
    """Cached hypernetwork model getter with LRU eviction"""
    # Use SHINE-enabled DocToLoRA for document mode
    if mode == "doc":
        return DocToLoRA(base_model, use_shine=True)
    # Placeholder for other modes - in production, load actual models
    else:
        return PlaceholderHypernetwork(base_model, mode)


def load_trained_hypernetwork(checkpoint_path: str, device: str = "cuda"):
    """Load trained hypernetwork checkpoint for use in generation."""
    try:
        from tessera_hypernetwork.train_hypernetwork import (
            DomainConditionedHypernetwork,
            StructuredMetadataEncoder,
        )
        from sentence_transformers import SentenceTransformer

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Reconstruct models
        base_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        encoder = StructuredMetadataEncoder(base_encoder)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])

        # Get dimensions from checkpoint or use defaults
        num_domains = checkpoint.get("num_domains", 10)
        hypernetwork = DomainConditionedHypernetwork(
            embed_dim=768,
            rank=16,
            d_in=4096,
            d_out=4096,
            hidden_dim=2048,
            num_domains=num_domains,
        )
        hypernetwork.load_state_dict(checkpoint["hypernetwork_state_dict"])

        return encoder, hypernetwork
    except Exception as e:
        print(f"Failed to load trained hypernetwork: {e}")
        return None, None


# Load trained hypernetwork if checkpoint path is set
CHECKPOINT_PATH = os.environ.get("TESSERA_CHECKPOINT_PATH")
trained_encoder = None
trained_hypernetwork = None

if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_encoder, trained_hypernetwork = load_trained_hypernetwork(
        CHECKPOINT_PATH, device
    )
    if trained_encoder and trained_hypernetwork:
        print(f"Loaded trained hypernetwork from {CHECKPOINT_PATH}")
    else:
        print(f"Could not load checkpoint from {CHECKPOINT_PATH}, using placeholder")


# Latency monitoring
generation_latencies = []
MAX_LATENCY_WINDOW = 100

# TTFT/TPOT monitoring
try:
    from tessera_hypernetwork.ttft_tpot import TTFTMonitor, TPOTMonitor, AdapterCache

    ttft_monitor = TTFTMonitor()
    tpot_monitor = TPOTMonitor()
    adapter_cache = AdapterCache(max_size=1000)
except ImportError:
    ttft_monitor = None
    tpot_monitor = None
    adapter_cache = None

# Efficiency monitoring
try:
    from tessera_hypernetwork.efficiency import EfficiencyDashboard

    efficiency_dashboard = EfficiencyDashboard()
except ImportError:
    efficiency_dashboard = None


def record_generation_latency(latency_ms: float):
    """Record generation latency for monitoring."""
    generation_latencies.append(latency_ms)
    if len(generation_latencies) > MAX_LATENCY_WINDOW:
        generation_latencies.pop(0)


def get_latency_stats() -> Dict[str, float]:
    """Get latency statistics."""
    if not generation_latencies:
        return {}

    import numpy as np

    stats = {
        "p50_ms": float(np.percentile(generation_latencies, 50)),
        "p95_ms": float(np.percentile(generation_latencies, 95)),
        "p99_ms": float(np.percentile(generation_latencies, 99)),
        "mean_ms": float(np.mean(generation_latencies)),
        "count": len(generation_latencies),
    }

    # Add TTFT/TPOT stats if available
    if ttft_monitor:
        stats["ttft"] = ttft_monitor.get_stats()
    if tpot_monitor:
        stats["tpot"] = tpot_monitor.get_stats()
    if adapter_cache:
        stats["adapter_cache"] = adapter_cache.get_stats()

    # Add efficiency stats if available
    if efficiency_dashboard:
        stats["efficiency"] = efficiency_dashboard.get_dashboard()
        stats["efficiency_score"] = efficiency_dashboard.get_efficiency_score()

    return stats


class GenerateRequest(BaseModel):
    model: str = "hypernetwork"
    messages: List[Dict[str, str]]
    base_model: str
    target_rank: int = 16
    response_format: Dict[str, str]
    mode: Optional[str] = None


class CompletionsRequest(BaseModel):
    model: str
    prompt: Union[str, List[int], List[List[int]]]
    max_tokens: int = 10
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = None


@app.post("/v1/generate")
async def generate(req: GenerateRequest):
    """
    Receive generation request from Tessera Rust core.
    Run hypernetwork forward pass.
    Return safetensors bytes directly.
    """
    start_time = time.perf_counter()
    adapter_gen_start = time.perf_counter()

    content = req.messages[0]["content"]
    # Use mode from request if provided, otherwise infer from content
    mode = req.mode if req.mode else infer_mode(content)

    # Check adapter cache if available
    if adapter_cache and mode == "metadata":
        try:
            import json

            metadata = json.loads(content) if isinstance(content, str) else content
            domain = metadata.get("domain", "general")
            domain_id = hash(domain) % 10
            cached_adapter = adapter_cache.get(metadata, domain_id)
        except Exception:
            cached_adapter = None
    else:
        cached_adapter = None

    # Use trained hypernetwork if available, otherwise fall back to cached models
    if trained_hypernetwork and trained_encoder and mode == "metadata":
        # Parse content as JSON metadata
        try:
            import json

            metadata = json.loads(content) if isinstance(content, str) else content

            # Encode metadata
            metadata_emb = trained_encoder(metadata)

            # Get domain ID
            domain = metadata.get("domain", "general")
            domain_id = hash(domain) % 10  # Simple hash-based domain ID

            # Generate with trained hypernetwork (or use cache)
            if cached_adapter is None:
                with torch.no_grad():
                    lora_weights = trained_hypernetwork(metadata_emb, domain_id)
                # Cache the result
                if adapter_cache:
                    adapter_cache.set(metadata, domain_id, lora_weights)
            else:
                lora_weights = cached_adapter

            # Convert to format expected by serialization
            lora_weights_dict = {
                "lora_A": lora_weights["lora_A"],
                "lora_B": lora_weights["lora_B"],
            }
        except Exception as e:
            print(
                f"Trained hypernetwork generation failed: {e}, falling back to placeholder"
            )
            hypernetwork = get_hypernetwork_cached(req.base_model, mode)
            with torch.no_grad():
                lora_weights = hypernetwork.generate(content, req.target_rank)
            lora_weights_dict = lora_weights
    else:
        # Load appropriate hypernetwork model (cached with LRU eviction)
        hypernetwork = get_hypernetwork_cached(req.base_model, mode)

        # Single forward pass
        with torch.no_grad():
            lora_weights = hypernetwork.generate(content, req.target_rank)
        lora_weights_dict = lora_weights

    # Record adapter generation time
    adapter_gen_time = (time.perf_counter() - adapter_gen_start) * 1000
    if ttft_monitor:
        ttft_monitor.record_adapter_generation(adapter_gen_time)

    # Serialize to safetensors bytes
    adapter_bytes = serialize_lora(lora_weights_dict)

    # Record latency
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    record_generation_latency(latency_ms)

    # Record TTFT
    if ttft_monitor:
        ttft_monitor.record_ttft(latency_ms)

    # Record efficiency metrics
    if efficiency_dashboard:
        # Estimate token counts (in practice, these would come from the tokenizer)
        input_tokens = len(content.split()) if isinstance(content, str) else 100
        output_tokens = 0  # Adapter generation doesn't produce output tokens
        efficiency_dashboard.record_request(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time_ms=latency_ms,
            input_length=input_tokens,
        )

    # Return raw bytes — no JSON wrapping
    return Response(content=adapter_bytes, media_type="application/octet-stream")


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "hypernetwork"}


@app.get("/metrics")
async def metrics():
    """Return latency metrics for monitoring."""
    return get_latency_stats()


# LoRAX-style adapter management endpoints
@app.post("/v1/adapters")
async def import_adapter(
    file: UploadFile = File(...),
    adapter_name: str = Form(...),
    base_model: str = Form(...),
):
    """Import an adapter into the hypernetwork service"""
    try:
        # Read adapter data
        adapter_data = await file.read()

        # Store in memory
        loaded_adapters[adapter_name] = {
            "name": adapter_name,
            "base_model": base_model,
            "data": adapter_data,
            "size": len(adapter_data),
        }

        return {
            "status": "success",
            "message": f"Adapter '{adapter_name}' imported successfully",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to import adapter: {str(e)}"
        )


@app.get("/v1/adapters")
async def list_adapters():
    """List all loaded adapters"""
    adapters = []
    for name, adapter in loaded_adapters.items():
        adapters.append(
            {
                "name": adapter["name"],
                "base_model": adapter["base_model"],
                "size": adapter["size"],
            }
        )
    return adapters


@app.delete("/v1/adapters/{adapter_name}")
async def unload_adapter(adapter_name: str):
    """Unload an adapter from the hypernetwork service"""
    if adapter_name not in loaded_adapters:
        raise HTTPException(
            status_code=404, detail=f"Adapter '{adapter_name}' not found"
        )

    del loaded_adapters[adapter_name]
    return {
        "status": "success",
        "message": f"Adapter '{adapter_name}' unloaded successfully",
    }


@app.post("/v1/completions")
async def completions(req: CompletionsRequest):
    """
    OpenAI-compatible completions endpoint for lm_eval integration.
    Looks up adapter by name and forwards to vLLM.
    """
    # Check if adapter is loaded
    if req.model not in loaded_adapters:
        raise HTTPException(
            status_code=404,
            detail=f"Adapter '{req.model}' not found. Load it first using tessera lorax import",
        )

    adapter = loaded_adapters[req.model]

    # Convert token IDs to string if needed
    if isinstance(req.prompt, list):
        # Check if it's a batch (List[List[int]]) or single sequence (List[int])
        if req.prompt and isinstance(req.prompt[0], list):
            # Batch of sequences - pass directly to vLLM (handles natively)
            prompt_text = req.prompt
        else:
            # Single sequence - decode to string
            tokenizer = get_tokenizer_cached(adapter["base_model"])
            prompt_text = tokenizer.decode(req.prompt)
    else:
        prompt_text = req.prompt

    # Forward request to vLLM
    vllm_url = "http://localhost:8000/v1/completions"

    try:
        # Prepare request for vLLM
        vllm_request = {
            "model": adapter["base_model"],
            "prompt": prompt_text,
            "max_tokens": req.max_tokens,
        }

        if req.temperature is not None:
            vllm_request["temperature"] = req.temperature
        if req.top_p is not None:
            vllm_request["top_p"] = req.top_p
        if req.logprobs is not None:
            vllm_request["logprobs"] = req.logprobs
        else:
            # Default to logprobs=1 for lm_eval compatibility
            vllm_request["logprobs"] = 1
        if req.echo is not None:
            vllm_request["echo"] = req.echo
        else:
            # Default to echo=true for lm_eval loglikelihood
            vllm_request["echo"] = True

        # Send to vLLM
        response = requests.post(vllm_url, json=vllm_request, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"vLLM request failed: {response.text}",
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503, detail=f"Failed to connect to vLLM at {vllm_url}: {str(e)}"
        )


def infer_mode(content: str) -> str:
    """Infer generation mode from content"""
    if len(content) > 1000:
        return "doc"
    elif "{" in content or '"' in content:
        return "metadata"
    else:
        return "text"


def serialize_lora(weights: dict) -> bytes:
    """Convert LoRA weight dict to safetensors bytes"""
    buffer = io.BytesIO()
    save_safetensors(weights, buffer, metadata={})
    return buffer.getvalue()


class PlaceholderHypernetwork:
    """Placeholder hypernetwork — outputs zero weights. Load trained weights for production."""

    def __init__(self, base_model: str, mode: str):
        self.base_model = base_model
        self.mode = mode
        import logging

        logging.warning(
            f"PlaceholderHypernetwork active: mode='{mode}', base_model='{base_model}'. "
            "LoRA weights will be all-zeros. This is not suitable for production inference."
        )

    def generate(self, content: str, rank: int) -> dict:
        """Return zero LoRA weights (untrained placeholder)."""
        d_in, d_out = get_model_dimensions(self.base_model)
        return {
            "lora_A": torch.zeros(rank, d_in),
            "lora_B": torch.zeros(d_out, rank),
        }


def get_model_dimensions(base_model: str) -> tuple:
    """Get input/output dimensions for base model"""
    model_dims = {
        "meta-llama/Llama-3-8B": (4096, 4096),
        "meta-llama/Llama-3-70B": (8192, 8192),
        "Qwen/Qwen2-7B": (3584, 3584),
        "deepseek-ai/DeepSeek-V3": (7168, 7168),
    }
    return model_dims.get(base_model, (4096, 4096))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
