from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import torch
from safetensors.torch import save as safetensors_save
from fastapi.responses import Response
from typing import Optional, List, Dict, Union
from tessera_hypernetwork.doc_to_lora import DocToLoRA
from functools import lru_cache
import httpx
from transformers import AutoTokenizer
import os
import time
import json
import logging
import traceback
from pathlib import Path


def get_hypernetwork_weights():
    """Get hypernetwork weights path, downloading from HuggingFace if needed."""
    # 1. User-specified checkpoint takes priority
    checkpoint = os.environ.get("TESSERA_CHECKPOINT_PATH")
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint from {checkpoint}")
        return checkpoint

    # 2. Check local cache
    cache_path = Path.home() / ".tessera" / "hypernetwork_v1.2.0.pt"
    if cache_path.exists():
        return str(cache_path)

    # 3. Download from HuggingFace on first use
    print(
        "Downloading Tessera v1.2.0 weights (first use, ~1.2GB, cached at ~/.tessera/)..."
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id="southfacing/tessera-weights",
        filename="hypernetwork_v1.2.0.pt",
        local_dir=str(cache_path.parent),
    )
    print("✓ Weights cached at ~/.tessera/")
    return downloaded


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
    # For metadata mode, use the trained hypernetwork if dimensions match
    elif mode == "metadata" and trained_hypernetwork and trained_encoder:
        model_d_in, _ = get_model_dimensions(base_model)
        if model_d_in != trained_d_in:
            logging.warning(
                "Trained hypernetwork was built for d_in=%d but '%s' requires d_in=%d "
                "\u2014 falling back to placeholder.",
                trained_d_in,
                base_model,
                model_d_in,
            )
            return PlaceholderHypernetwork(base_model, mode)
        return TrainedHypernetworkWrapper(
            trained_encoder, trained_hypernetwork, num_domains_loaded
        )
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

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Detect encoder_dim from checkpoint
        first_weight = next(iter(checkpoint["encoder_state_dict"].values()))
        encoder_dim = first_weight.shape[-1]
        print(f"Detected encoder dimension from checkpoint: {encoder_dim}")

        # Detect num_domains from checkpoint (from domain_embedding weight shape)
        domain_embedding_weight = checkpoint["hypernetwork_state_dict"].get(
            "domain_embedding.weight"
        )
        if domain_embedding_weight is not None:
            num_domains = domain_embedding_weight.shape[0]
            print(f"Detected num_domains from checkpoint: {num_domains}")
        else:
            num_domains = checkpoint.get("num_domains", 10)
            print(f"Using default num_domains: {num_domains}")

        # Detect hidden_dim, rank, d_in, d_out from checkpoint weight shapes
        hnet_sd = checkpoint["hypernetwork_state_dict"]
        hidden_dim_w = hnet_sd.get("mlp_lora_A.0.weight")
        lora_A_out_w = hnet_sd.get("mlp_lora_A.8.weight")
        lora_B_out_w = hnet_sd.get("mlp_lora_B.8.weight")
        hidden_dim = hidden_dim_w.shape[0] if hidden_dim_w is not None else 2048
        if lora_A_out_w is not None and lora_B_out_w is not None:
            rank_d_in = lora_A_out_w.shape[0]
            d_out_rank = lora_B_out_w.shape[0]
            for candidate_rank in [16, 8, 32, 4, 64]:
                if rank_d_in % candidate_rank == 0 and d_out_rank % candidate_rank == 0:
                    rank = candidate_rank
                    d_in = rank_d_in // candidate_rank
                    d_out = d_out_rank // candidate_rank
                    break
            else:
                rank, d_in, d_out = 16, 4096, 4096
        else:
            rank, d_in, d_out = 16, 4096, 4096
        print(
            f"Detected hypernetwork dims: rank={rank}, d_in={d_in}, d_out={d_out}, hidden_dim={hidden_dim}"
        )

        # Reconstruct models
        base_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        encoder = StructuredMetadataEncoder(base_encoder, embed_dim=encoder_dim)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        encoder = encoder.to(device)

        hypernetwork = DomainConditionedHypernetwork(
            embed_dim=encoder_dim,
            rank=rank,
            d_in=d_in,
            d_out=d_out,
            hidden_dim=hidden_dim,
            num_domains=num_domains,
        )
        hypernetwork.load_state_dict(checkpoint["hypernetwork_state_dict"])
        hypernetwork = hypernetwork.to(device)

        print(f"✓ Successfully loaded trained hypernetwork on {device}")
        return encoder, hypernetwork, num_domains, d_in
    except Exception as e:
        print(f"✗ Failed to load trained hypernetwork: {e}")
        print(
            "✗ This is a CRITICAL ERROR - server will not function without trained weights"
        )
        traceback.print_exc()
        return None, None, 10, 4096


# Load trained hypernetwork with auto-download from HuggingFace
trained_encoder = None
trained_hypernetwork = None
num_domains_loaded = 10
trained_d_in = 4096

checkpoint_path = get_hypernetwork_weights()
if checkpoint_path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_encoder, trained_hypernetwork, num_domains_loaded, trained_d_in = (
        load_trained_hypernetwork(checkpoint_path, device)
    )
    if trained_encoder and trained_hypernetwork:
        print(f"✓ Loaded trained hypernetwork from {checkpoint_path}")
    else:
        print(f"✗ CRITICAL: Could not load checkpoint from {checkpoint_path}")
        print(
            "✗ Server will use placeholder hypernetwork (zero weights) - THIS IS NOT PRODUCTION READY"
        )
        print("✗ Check the error messages above for details")
else:
    print("✗ CRITICAL: No checkpoint path available")
    print(
        "✗ Server will use placeholder hypernetwork (zero weights) - THIS IS NOT PRODUCTION READY"
    )


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

    if not req.messages:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="messages array must not be empty")

    content = req.messages[0]["content"]
    # Use mode from request if provided, otherwise infer from content
    mode = req.mode if req.mode else infer_mode(content)

    # Check adapter cache first (keyed on content + mode + base_model + rank)
    cache_key = {"c": content, "m": mode, "bm": req.base_model, "r": req.target_rank}
    cached_bytes = adapter_cache.get(cache_key, 0) if adapter_cache else None

    if cached_bytes is not None:
        adapter_bytes = cached_bytes
        adapter_gen_time = (time.perf_counter() - adapter_gen_start) * 1000
    else:
        # Unified path — no inline special-casing; each wrapper handles its own logic
        hypernetwork = get_hypernetwork_cached(req.base_model, mode)
        with torch.no_grad():
            lora_weights_dict = hypernetwork.generate(content, req.target_rank)
        adapter_gen_time = (time.perf_counter() - adapter_gen_start) * 1000
        adapter_bytes = serialize_lora(lora_weights_dict)
        if adapter_cache:
            adapter_cache.set(cache_key, 0, adapter_bytes)

    if ttft_monitor:
        ttft_monitor.record_adapter_generation(adapter_gen_time)

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
    vllm_url = (
        os.environ.get("TESSERA_VLLM_URL", "http://localhost:8000") + "/v1/completions"
    )

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

        # Send to vLLM asynchronously (non-blocking)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(vllm_url, json=vllm_request)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"vLLM request failed: {response.text}",
            )
    except httpx.RequestError as e:
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
    cpu_weights = {k: v.contiguous().cpu() for k, v in weights.items()}
    return safetensors_save(cpu_weights, metadata={})


class TrainedHypernetworkWrapper:
    """Wrapper for trained hypernetwork to provide consistent interface."""

    def __init__(self, encoder, hypernetwork, num_domains: int = 10):
        self.encoder = encoder
        self.hypernetwork = hypernetwork
        self.num_domains = num_domains
        self.device = next(self.encoder.parameters()).device
        # Disable dropout and use running stats — must be set at init, not per-request
        self.encoder.eval()
        self.hypernetwork.eval()

    def generate(self, content: str, rank: int) -> dict:
        """Generate LoRA weights using trained hypernetwork."""
        try:
            metadata = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            metadata = {"task": str(content)}

        # Encode metadata and ensure it's on the correct device
        metadata_emb = self.encoder(metadata)
        metadata_emb = metadata_emb.to(self.device)

        # Fixed: use full domain range from checkpoint (was always % 10, ignoring other domains)
        domain = metadata.get("domain", "general")
        domain_id = hash(domain) % self.num_domains

        lora_weights = self.hypernetwork(metadata_emb, domain_id)

        lora_A = lora_weights["lora_A"].to(self.device)
        lora_B = lora_weights["lora_B"].to(self.device)

        logging.info(
            "Generated LoRA weights - lora_A: mean=%.6f, std=%.6f, "
            "lora_B: mean=%.6f, std=%.6f",
            lora_A.mean().item(),
            lora_A.std().item(),
            lora_B.mean().item(),
            lora_B.std().item(),
        )

        return {"lora_A": lora_A, "lora_B": lora_B}


class PlaceholderHypernetwork:
    """Placeholder hypernetwork — outputs Xavier-initialized (untrained) weights."""

    def __init__(self, base_model: str, mode: str):
        self.base_model = base_model
        self.mode = mode
        logging.warning(
            "PlaceholderHypernetwork active: mode='%s', base_model='%s'. "
            "LoRA weights are Xavier-initialized (NOT trained) — semantically meaningless. "
            "Only 'metadata' mode with a loaded checkpoint produces real adapters.",
            mode,
            base_model,
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
        "meta-llama/Llama-3.1-8B": (4096, 4096),
        "meta-llama/Llama-3.1-70B": (8192, 8192),
        "meta-llama/Llama-3.2-3B": (3072, 3072),
        "Qwen/Qwen2-7B": (3584, 3584),
        "deepseek-ai/DeepSeek-V3": (7168, 7168),
        "mistralai/Mistral-7B-v0.1": (4096, 4096),
        "mistralai/Mistral-7B-Instruct-v0.2": (4096, 4096),
        "google/gemma-2-9b": (3584, 3584),
        "microsoft/Phi-3-mini-4k-instruct": (3072, 3072),
    }
    return model_dims.get(base_model, (4096, 4096))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
