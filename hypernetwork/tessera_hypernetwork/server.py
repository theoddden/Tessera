from fastapi import FastAPI
from pydantic import BaseModel
import torch
from safetensors.torch import save as save_safetensors
from fastapi.responses import Response
import io
from typing import Optional, List, Dict
from tessera_hypernetwork.doc_to_lora import DocToLoRA
from functools import lru_cache

app = FastAPI(title="Tessera Hypernetwork Service")


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


class GenerateRequest(BaseModel):
    model: str = "hypernetwork"
    messages: List[Dict[str, str]]
    base_model: str
    target_rank: int = 16
    response_format: Dict[str, str]
    mode: Optional[str] = None


@app.post("/v1/generate")
async def generate(req: GenerateRequest):
    """
    Receive generation request from Tessera Rust core.
    Run hypernetwork forward pass.
    Return safetensors bytes directly.
    """
    content = req.messages[0]["content"]
    # Use mode from request if provided, otherwise infer from content
    mode = req.mode if req.mode else infer_mode(content)

    # Load appropriate hypernetwork model (cached with LRU eviction)
    hypernetwork = get_hypernetwork_cached(req.base_model, mode)

    # Single forward pass
    with torch.no_grad():
        lora_weights = hypernetwork.generate(content, req.target_rank)

    # Serialize to safetensors bytes
    adapter_bytes = serialize_lora(lora_weights)

    # Return raw bytes — no JSON wrapping
    return Response(content=adapter_bytes, media_type="application/octet-stream")


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "hypernetwork"}


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
    save_safetensors(weights, buffer)
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
