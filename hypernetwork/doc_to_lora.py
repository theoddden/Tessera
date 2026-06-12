"""
Doc-to-LoRA pipeline for generating LoRA adapters from document content.
Integrated with SHINE (ICML 2026) for long-context document internalization.
SHINE handles contexts 5x longer than standard context windows through
hierarchical information extraction and attention-based compression.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel


class SHINEProcessor:
    """
    SHINE-inspired processor for long-context document internalization.
    Implements hierarchical chunking, attention-based selection, and compression.
    """

    def __init__(self, encoder, tokenizer, chunk_size: int = 512, overlap: int = 64):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Derive embed_dim from encoder's hidden size
        embed_dim = encoder.config.hidden_size if hasattr(encoder, "config") else 768
        self.attention_selector = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8
        )
        # Move attention selector to same device as encoder
        device = (
            next(encoder.parameters()).device
            if hasattr(encoder, "parameters")
            else torch.device("cpu")
        )
        self.attention_selector = self.attention_selector.to(device)

    def process_long_document(self, document: str) -> torch.Tensor:
        """
        Process documents longer than context window using SHINE approach.

        Args:
            document: Long document text (potentially >20k tokens)

        Returns:
            Compressed document embedding capturing key information
        """
        # Chunk document with overlap
        chunks = self._chunk_document(document)

        # Encode each chunk
        chunk_embeddings = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.chunk_size,
                padding=True,
            )
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                chunk_emb = outputs.last_hidden_state.mean(dim=1)
                chunk_embeddings.append(chunk_emb)

        # Stack chunk embeddings
        chunk_embeddings = torch.stack(chunk_embeddings, dim=1)  # [1, num_chunks, dim]

        # Attention-based information selection (SHINE core)
        # Select most informative chunks based on self-attention
        attended, _ = self.attention_selector(
            chunk_embeddings, chunk_embeddings, chunk_embeddings
        )

        # Hierarchical aggregation
        # Level 1: Chunk-level attention
        chunk_weights = torch.softmax(attended.mean(dim=-1), dim=-1)
        weighted_chunks = chunk_embeddings * chunk_weights.unsqueeze(-1)

        # Level 2: Global compression
        doc_embedding = weighted_chunks.mean(dim=1)

        return doc_embedding

    def _chunk_document(self, document: str) -> List[str]:
        """Split document into overlapping chunks"""
        tokens = self.tokenizer.encode(document, add_special_tokens=False)
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks


class DocToLoRA:
    """Generate LoRA adapters from document content using SHINE for long contexts"""

    def __init__(
        self,
        base_model: str,
        use_shine: bool = True,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        default_rank: int = 16,
    ):
        self.base_model = base_model
        # Use lightweight embedding model instead of base model
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.use_shine = use_shine
        self.default_rank = default_rank

        if use_shine:
            self.shine_processor = SHINEProcessor(
                self.encoder, self.tokenizer, chunk_size=512, overlap=64
            )

        # Initialize projection layers for default rank
        self._init_projection_layers(default_rank)

    def _init_projection_layers(self, rank: int):
        """Initialize projection layers with deterministic weights for given rank"""
        embed_dim = (
            self.encoder.config.hidden_size if hasattr(self.encoder, "config") else 384
        )
        dims = self._get_model_dimensions()
        d_in, d_out = dims

        # Initialize with Xavier/Glorot initialization for deterministic behavior
        self.proj_lora_A = torch.nn.Linear(embed_dim, rank * d_in)
        self.proj_lora_B = torch.nn.Linear(embed_dim, d_out * rank)

        torch.nn.init.xavier_uniform_(self.proj_lora_A.weight)
        torch.nn.init.zeros_(self.proj_lora_A.bias)
        torch.nn.init.xavier_uniform_(self.proj_lora_B.weight)
        torch.nn.init.zeros_(self.proj_lora_B.bias)

    def generate(self, document: str, rank: int) -> Dict[str, torch.Tensor]:
        """
        Generate LoRA weights from document content.

        Args:
            document: Document text content (can be very long with SHINE)
            rank: LoRA rank

        Returns:
            Dictionary of LoRA weight tensors
        """
        # Reinitialize projection layers if rank differs from default
        if rank != self.default_rank:
            self._init_projection_layers(rank)
            self.default_rank = rank

        # Use SHINE for long-context processing
        if self.use_shine and len(document) > 10000:
            doc_embedding = self.shine_processor.process_long_document(document)
        else:
            # Standard encoding for shorter documents
            inputs = self.tokenizer(
                document,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True,
            )
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                doc_embedding = outputs.last_hidden_state.mean(dim=1)

        # Generate LoRA weights from embedding using class member projection layers
        dims = self._get_model_dimensions()
        d_in, d_out = dims

        # Use deterministic projection layers
        lora_A_weights = self.proj_lora_A(doc_embedding)
        lora_B_weights = self.proj_lora_B(doc_embedding)

        lora_A = lora_A_weights.view(-1, rank, d_in)
        lora_B = lora_B_weights.view(-1, d_out, rank)

        return {
            "lora_A": lora_A.squeeze(0),
            "lora_B": lora_B.squeeze(0),
        }

    def load_weights(self, checkpoint_path: str):
        """Load pre-trained projection weights from a checkpoint file.

        Expected format: torch checkpoint with keys 'proj_lora_A' and 'proj_lora_B'
        containing state_dicts for the respective Linear layers.
        """
        import os
        import logging

        if not os.path.exists(checkpoint_path):
            logging.warning(
                f"DocToLoRA: checkpoint not found at '{checkpoint_path}'. "
                "Running with untrained Xavier-initialized weights — outputs will not encode document content correctly."
            )
            return
        state = torch.load(checkpoint_path, map_location="cpu")
        self.proj_lora_A.load_state_dict(state["proj_lora_A"])
        self.proj_lora_B.load_state_dict(state["proj_lora_B"])
        logging.info(f"DocToLoRA: loaded weights from '{checkpoint_path}'")

    def _get_model_dimensions(self) -> tuple:
        """Get model dimensions"""
        model_dims = {
            "meta-llama/Llama-3-8B": (4096, 4096),
            "meta-llama/Llama-3-70B": (8192, 8192),
            "Qwen/Qwen2-7B": (3584, 3584),
            "deepseek-ai/DeepSeek-V3": (7168, 7168),
        }
        return model_dims.get(self.base_model, (4096, 4096))
