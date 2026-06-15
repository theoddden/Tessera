"""
Layer-wise hypernetwork generation for structural coupling across layers.

Based on: HyperLoader: Integrating Hypernetwork-Based LoRA and Adapter Layers
into Multi-Task Transformers for Sequence Labelling [arXiv:2407.01411]

Key features:
- Layer-specific hypernetworks conditioned on layer position
- Layer embeddings for position encoding
- Reduced task interference
- Better structural coupling across layers
- Improved multi-task performance
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List


class LayerWiseHypernetwork(nn.Module):
    """
    Layer-wise hypernetwork with position-aware generation.

    Each layer has its own hypernetwork conditioned on:
    - Metadata embedding
    - Layer position embedding
    - Layer type embedding (attention vs MLP)
    """

    def __init__(
        self,
        num_layers: int = 32,
        embed_dim: int = 768,
        rank: int = 16,
        d_in: int = 4096,
        d_out: int = 4096,
        hidden_dim: int = 2048,
        num_domains: int = 10,
        layer_types: List[str] = ["attention", "mlp"],
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.rank = rank
        self.d_in = d_in
        self.d_out = d_out
        self.layer_types = layer_types

        # Layer position embeddings
        self.layer_embeddings = nn.Embedding(num_layers, embed_dim)

        # Layer type embeddings
        self.layer_type_embeddings = nn.Embedding(len(layer_types), embed_dim)

        # Layer-specific hypernetworks
        from tessera_hypernetwork.train_hypernetwork import DomainConditionedHypernetwork
        self.layer_hypernetworks = nn.ModuleList([
            DomainConditionedHypernetwork(
                embed_dim=embed_dim,
                rank=rank,
                d_in=d_in,
                d_out=d_out,
                hidden_dim=hidden_dim,
                num_domains=num_domains,
            )
            for _ in range(num_layers)
        ])

        # Fusion layer for combining metadata + layer info
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
        layer_idx: int,
        layer_type: str = "attention",
    ) -> Dict[str, torch.Tensor]:
        """
        Generate layer-specific LoRA weights.

        Args:
            metadata_emb: Metadata embedding [batch_size, embed_dim]
            domain_id: Domain ID for conditioning
            layer_idx: Layer index (0 to num_layers-1)
            layer_type: Layer type ("attention" or "mlp")

        Returns:
            Dictionary with lora_A, lora_B tensors
        """
        # Get layer position embedding
        layer_idx_tensor = torch.tensor([layer_idx], device=metadata_emb.device)
        layer_emb = self.layer_embeddings(layer_idx_tensor)

        # Get layer type embedding
        layer_type_idx = self.layer_types.index(layer_type)
        layer_type_tensor = torch.tensor([layer_type_idx], device=metadata_emb.device)
        type_emb = self.layer_type_embeddings(layer_type_tensor)

        # Fuse metadata + layer position + layer type
        combined_emb = torch.cat([metadata_emb, layer_emb, type_emb], dim=-1)
        fused_emb = self.fusion(combined_emb)

        # Generate layer-specific LoRA
        lora = self.layer_hypernetworks[layer_idx](fused_emb, domain_id)

        return lora

    def generate_all_layers(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate LoRA weights for all layers.

        Args:
            metadata_emb: Metadata embedding
            domain_id: Domain ID

        Returns:
            List of LoRA weights for each layer
        """
        all_loras = []

        for layer_idx in range(self.num_layers):
            # Generate for both attention and MLP
            for layer_type in self.layer_types:
                lora = self.forward(metadata_emb, domain_id, layer_idx, layer_type)
                all_loras.append({
                    "layer_idx": layer_idx,
                    "layer_type": layer_type,
                    "lora_A": lora["lora_A"],
                    "lora_B": lora["lora_B"],
                })

        return all_loras


class SharedLayerHypernetwork(nn.Module):
    """
    Shared hypernetwork with layer conditioning.

    More parameter-efficient than layer-wise hypernetworks.
    Uses a single hypernetwork with layer embeddings as input.
    """

    def __init__(
        self,
        num_layers: int = 32,
        embed_dim: int = 768,
        rank: int = 16,
        d_in: int = 4096,
        d_out: int = 4096,
        hidden_dim: int = 2048,
        num_domains: int = 10,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.rank = rank
        self.d_in = d_in
        self.d_out = d_out

        # Layer position embeddings
        self.layer_embeddings = nn.Embedding(num_layers, embed_dim)

        # Single shared hypernetwork
        from tessera_hypernetwork.train_hypernetwork import DomainConditionedHypernetwork
        self.hypernetwork = DomainConditionedHypernetwork(
            embed_dim=embed_dim,
            rank=rank,
            d_in=d_in,
            d_out=d_out,
            hidden_dim=hidden_dim,
            num_domains=num_domains,
        )

    def forward(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate layer-specific LoRA using shared hypernetwork.

        Args:
            metadata_emb: Metadata embedding
            domain_id: Domain ID
            layer_idx: Layer index

        Returns:
            Dictionary with lora_A, lora_B tensors
        """
        # Add layer embedding to metadata
        layer_idx_tensor = torch.tensor([layer_idx], device=metadata_emb.device)
        layer_emb = self.layer_embeddings(layer_idx_tensor)

        combined_emb = metadata_emb + layer_emb

        # Generate LoRA
        return self.hypernetwork(combined_emb, domain_id)


class ProgressiveLayerHypernetwork(nn.Module):
    """
    Progressive layer-wise hypernetwork with parameter sharing.

    Early layers share more parameters, later layers are more specialized.
    """

    def __init__(
        self,
        num_layers: int = 32,
        embed_dim: int = 768,
        rank: int = 16,
        d_in: int = 4096,
        d_out: int = 4096,
        hidden_dim: int = 2048,
        num_domains: int = 10,
        num_stages: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_stages = num_stages
        self.embed_dim = embed_dim

        # Divide layers into stages
        self.stage_size = num_layers // num_stages

        # Create hypernetwork for each stage
        from tessera_hypernetwork.train_hypernetwork import DomainConditionedHypernetwork
        self.stage_hypernetworks = nn.ModuleList([
            DomainConditionedHypernetwork(
                embed_dim=embed_dim,
                rank=rank,
                d_in=d_in,
                d_out=d_out,
                hidden_dim=hidden_dim,
                num_domains=num_domains,
            )
            for _ in range(num_stages)
        ])

        # Layer embeddings for fine-grained adjustment
        self.layer_embeddings = nn.Embedding(num_layers, embed_dim)

    def forward(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate LoRA using progressive stage hypernetwork.

        Args:
            metadata_emb: Metadata embedding
            domain_id: Domain ID
            layer_idx: Layer index

        Returns:
            Dictionary with lora_A, lora_B tensors
        """
        # Determine stage
        stage_idx = min(layer_idx // self.stage_size, self.num_stages - 1)

        # Get stage hypernetwork
        stage_hypernetwork = self.stage_hypernetworks[stage_idx]

        # Add layer embedding for fine-tuning
        layer_idx_tensor = torch.tensor([layer_idx], device=metadata_emb.device)
        layer_emb = self.layer_embeddings(layer_idx_tensor)

        combined_emb = metadata_emb + layer_emb

        # Generate LoRA
        return stage_hypernetwork(combined_emb, domain_id)


def create_layer_wise_hypernetwork(
    num_layers: int = 32,
    embed_dim: int = 768,
    rank: int = 16,
    d_in: int = 4096,
    d_out: int = 4096,
    mode: str = "shared",  # "full", "shared", or "progressive"
) -> nn.Module:
    """
    Factory function to create layer-wise hypernetwork.

    Args:
        num_layers: Number of transformer layers
        embed_dim: Embedding dimension
        rank: LoRA rank
        d_in: Input dimension
        d_out: Output dimension
        mode: Mode ("full", "shared", or "progressive")

    Returns:
        Layer-wise hypernetwork module
    """
    if mode == "full":
        return LayerWiseHypernetwork(
            num_layers=num_layers,
            embed_dim=embed_dim,
            rank=rank,
            d_in=d_in,
            d_out=d_out,
        )
    elif mode == "shared":
        return SharedLayerHypernetwork(
            num_layers=num_layers,
            embed_dim=embed_dim,
            rank=rank,
            d_in=d_in,
            d_out=d_out,
        )
    elif mode == "progressive":
        return ProgressiveLayerHypernetwork(
            num_layers=num_layers,
            embed_dim=embed_dim,
            rank=rank,
            d_in=d_in,
            d_out=d_out,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # Test layer-wise hypernetwork
    print("Testing Layer-wise Hypernetwork...")

    # Test full layer-wise
    full_hn = LayerWiseHypernetwork(
        num_layers=8,
        embed_dim=768,
        rank=16,
        d_in=4096,
        d_out=4096,
    )

    metadata_emb = torch.randn(1, 768)
    domain_id = 0

    lora = full_hn(metadata_emb, domain_id, layer_idx=0, layer_type="attention")
    print(f"Full layer-wise - LoRA A shape: {lora['lora_A'].shape}")
    print(f"Full layer-wise - LoRA B shape: {lora['lora_B'].shape}")

    # Test shared layer-wise
    shared_hn = SharedLayerHypernetwork(
        num_layers=8,
        embed_dim=768,
        rank=16,
        d_in=4096,
        d_out=4096,
    )

    lora = shared_hn(metadata_emb, domain_id, layer_idx=0)
    print(f"Shared layer-wise - LoRA A shape: {lora['lora_A'].shape}")
    print(f"Shared layer-wise - LoRA B shape: {lora['lora_B'].shape}")

    # Test progressive layer-wise
    prog_hn = ProgressiveLayerHypernetwork(
        num_layers=8,
        embed_dim=768,
        rank=16,
        d_in=4096,
        d_out=4096,
    )

    lora = prog_hn(metadata_emb, domain_id, layer_idx=0)
    print(f"Progressive layer-wise - LoRA A shape: {lora['lora_A'].shape}")
    print(f"Progressive layer-wise - LoRA B shape: {lora['lora_B'].shape}")

    print("\nLayer-wise hypernetwork test passed!")
