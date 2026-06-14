"""
Text-to-LoRA pipeline for generating LoRA adapters from natural language descriptions.
"""

import torch
from typing import Dict
from sentence_transformers import SentenceTransformer


class TextToLoRA:
    """Generate LoRA adapters from natural language task descriptions"""

    def __init__(
        self,
        base_model: str,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        default_rank: int = 16,
    ):
        self.base_model = base_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(encoder_model)
        self.encoder = self.encoder.to(self.device)
        self.default_rank = default_rank
        self._init_projection_layers(default_rank)

    def _init_projection_layers(self, rank: int):
        """Initialize projection layers with deterministic Xavier weights"""
        embed_dim = self.encoder.get_sentence_embedding_dimension()
        d_in, d_out = self._get_model_dimensions()

        self.proj_lora_A = torch.nn.Linear(embed_dim, rank * d_in)
        self.proj_lora_B = torch.nn.Linear(embed_dim, d_out * rank)

        torch.nn.init.xavier_uniform_(self.proj_lora_A.weight)
        torch.nn.init.zeros_(self.proj_lora_A.bias)
        torch.nn.init.xavier_uniform_(self.proj_lora_B.weight)
        torch.nn.init.zeros_(self.proj_lora_B.bias)

        # Move projection layers to the same device as encoder
        self.proj_lora_A = self.proj_lora_A.to(self.device)
        self.proj_lora_B = self.proj_lora_B.to(self.device)

    def generate(self, description: str, rank: int) -> Dict[str, torch.Tensor]:
        """
        Generate LoRA weights from task description.

        Args:
            description: Natural language task description
            rank: LoRA rank

        Returns:
            Dictionary of LoRA weight tensors
        """
        if rank != self.default_rank:
            self._init_projection_layers(rank)
            self.default_rank = rank

        with torch.no_grad():
            text_embedding = self.encoder.encode(
                description,
                convert_to_tensor=True,
                show_progress_bar=False
            )

        d_in, d_out = self._get_model_dimensions()

        lora_A_weights = self.proj_lora_A(text_embedding)
        lora_B_weights = self.proj_lora_B(text_embedding)

        lora_A = lora_A_weights.view(-1, rank, d_in)
        lora_B = lora_B_weights.view(-1, d_out, rank)

        return {
            "lora_A": lora_A.squeeze(0),
            "lora_B": lora_B.squeeze(0),
        }

    def _get_model_dimensions(self) -> tuple:
        """Get model dimensions"""
        model_dims = {
            "meta-llama/Llama-3-8B": (4096, 4096),
            "meta-llama/Llama-3-70B": (8192, 8192),
            "Qwen/Qwen2-7B": (3584, 3584),
            "deepseek-ai/DeepSeek-V3": (7168, 7168),
        }
        return model_dims.get(self.base_model, (4096, 4096))
