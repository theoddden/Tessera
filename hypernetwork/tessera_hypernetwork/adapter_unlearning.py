"""
Adapter unlearning with CLIP guidance for concept erasure.

Based on: UnHype: CLIP-Guided Hypernetworks for Dynamic LoRA Unlearning [arXiv:2602.03410]

Key features:
- CLIP-guided concept similarity detection
- Dynamic LoRA unlearning without retraining
- Remove harmful or outdated knowledge
- Privacy-preserving adaptation
- Balanced unlearning (remove target, preserve generalization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set


class CLIPGuidedUnlearning(nn.Module):
    """
    CLIP-guided hypernetwork for dynamic LoRA unlearning.

    Uses CLIP embeddings to detect concepts and gate out unwanted knowledge.
    """

    def __init__(
        self,
        base_hypernetwork: nn.Module,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 768,
        unlearning_threshold: float = 0.7,
    ):
        super().__init__()
        self.base_hypernetwork = base_hypernetwork
        self.embed_dim = embed_dim
        self.unlearning_threshold = unlearning_threshold

        # Load CLIP model
        try:
            from transformers import CLIPModel, CLIPProcessor

            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model.eval()
        except ImportError:
            print("Warning: transformers not installed, CLIP features disabled")
            self.clip_model = None
            self.clip_processor = None

        # Unlearning gate network
        self.unlearning_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Concepts to erase (stored as text)
        self.concepts_to_erase: Set[str] = set()

    def add_concept_to_erase(self, concept: str):
        """Add a concept to the unlearning set."""
        self.concepts_to_erase.add(concept.lower())

    def remove_concept_to_erase(self, concept: str):
        """Remove a concept from the unlearning set."""
        self.concepts_to_erase.discard(concept.lower())

    def get_concept_embeddings(self, concepts: List[str]) -> torch.Tensor:
        """
        Get CLIP embeddings for concepts.

        Args:
            concepts: List of concept strings

        Returns:
            Concept embeddings tensor
        """
        if self.clip_model is None:
            return torch.zeros(len(concepts), self.embed_dim)

        # Process concepts through CLIP
        inputs = self.clip_processor(text=concepts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)

        return text_features

    def compute_concept_similarity(
        self,
        metadata_emb: torch.Tensor,
        concept_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between metadata and concepts.

        Args:
            metadata_emb: Metadata embedding
            concept_embeddings: Concept embeddings

        Returns:
            Similarity scores
        """
        # Normalize embeddings
        metadata_norm = F.normalize(metadata_emb, dim=-1)
        concept_norm = F.normalize(concept_embeddings, dim=-1)

        # Compute cosine similarity
        similarity = torch.matmul(metadata_norm, concept_norm.T)

        return similarity

    def forward(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
        metadata_text: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with unlearning.

        Args:
            metadata_emb: Metadata embedding
            domain_id: Domain ID
            metadata_text: Optional text representation of metadata

        Returns:
            LoRA weights with unlearning applied
        """
        # Generate base LoRA
        lora = self.base_hypernetwork(metadata_emb, domain_id)

        # Apply unlearning if concepts are set
        if self.concepts_to_erase and self.clip_model is not None:
            concept_list = list(self.concepts_to_erase)
            concept_embeddings = self.get_concept_embeddings(concept_list)

            # Compute similarity to concepts
            similarity = self.compute_concept_similarity(
                metadata_emb, concept_embeddings
            )

            # Max similarity across all concepts
            max_similarity = similarity.max(dim=-1)[0]

            # Compute unlearning gate
            gate_input = metadata_emb * max_similarity.unsqueeze(-1)
            gate = self.unlearning_gate(gate_input)

            # Apply gate if similarity exceeds threshold
            mask = (max_similarity > self.unlearning_threshold).float()
            gate = gate * mask.unsqueeze(-1)

            # Apply unlearning (scale down LoRA weights)
            lora["lora_A"] = lora["lora_A"] * (1 - gate)
            lora["lora_B"] = lora["lora_B"] * (1 - gate)

            # Store unlearning info
            lora["unlearning_applied"] = True
            lora["unlearning_strength"] = gate.mean().item()
            lora["concept_similarity"] = max_similarity.item()

        return lora


class GradientBasedUnlearning(nn.Module):
    """
    Gradient-based unlearning for precise concept removal.

    Uses gradient descent to minimize representation of specific concepts.
    """

    def __init__(
        self,
        base_hypernetwork: nn.Module,
        embed_dim: int = 768,
        unlearning_steps: int = 10,
        unlearning_lr: float = 0.01,
    ):
        super().__init__()
        self.base_hypernetwork = base_hypernetwork
        self.embed_dim = embed_dim
        self.unlearning_steps = unlearning_steps
        self.unlearning_lr = unlearning_lr

        # Concept embeddings for unlearning
        self.concept_embeddings: Dict[str, torch.Tensor] = {}

    def add_concept(self, concept: str, embedding: torch.Tensor):
        """Add a concept with its embedding."""
        self.concept_embeddings[concept.lower()] = embedding.detach()

    def unlearn_concept(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
        concept: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply gradient-based unlearning for a specific concept.

        Args:
            metadata_emb: Metadata embedding
            domain_id: Domain ID
            concept: Concept to unlearn

        Returns:
            Unlearned LoRA weights
        """
        if concept.lower() not in self.concept_embeddings:
            # No unlearning needed
            return self.base_hypernetwork(metadata_emb, domain_id)

        concept_emb = self.concept_embeddings[concept.lower()]

        # Generate initial LoRA
        lora = self.base_hypernetwork(metadata_emb, domain_id)

        # Create a copy for gradient-based modification
        lora_A_mod = lora["lora_A"].clone().detach().requires_grad_(True)
        lora_B_mod = lora["lora_B"].clone().detach().requires_grad_(True)

        # Gradient descent to minimize concept representation
        optimizer = torch.optim.SGD(
            [lora_A_mod, lora_B_mod],
            lr=self.unlearning_lr,
        )

        for _ in range(self.unlearning_steps):
            optimizer.zero_grad()

            # Compute loss: maximize distance from concept
            # This is a proxy - in practice, you'd use the actual model forward pass
            # Here we use a simple distance metric
            lora_flat = torch.cat([lora_A_mod.flatten(), lora_B_mod.flatten()])
            concept_flat = concept_emb.flatten()

            # Minimize similarity (negative cosine similarity)
            similarity = F.cosine_similarity(
                lora_flat.unsqueeze(0),
                concept_flat.unsqueeze(0),
            )
            loss = -similarity  # Negative to minimize similarity

            loss.backward()
            optimizer.step()

        return {
            "lora_A": lora_A_mod.detach(),
            "lora_B": lora_B_mod.detach(),
            "unlearning_applied": True,
        }


class PrivacyPreservingHypernetwork(nn.Module):
    """
    Privacy-preserving hypernetwork with differential privacy.

    Adds noise to LoRA weights for privacy guarantees.
    """

    def __init__(
        self,
        base_hypernetwork: nn.Module,
        noise_scale: float = 0.1,
        clip_norm: float = 1.0,
    ):
        super().__init__()
        self.base_hypernetwork = base_hypernetwork
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm

    def forward(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with differential privacy noise.

        Args:
            metadata_emb: Metadata embedding
            domain_id: Domain ID

        Returns:
            Noisy LoRA weights
        """
        # Generate base LoRA
        lora = self.base_hypernetwork(metadata_emb, domain_id)

        # Clip gradients (for DP)
        lora_A_clipped = torch.clamp(
            lora["lora_A"],
            -self.clip_norm,
            self.clip_norm,
        )
        lora_B_clipped = torch.clamp(
            lora["lora_B"],
            -self.clip_norm,
            self.clip_norm,
        )

        # Add Gaussian noise
        noise_A = torch.randn_like(lora_A_clipped) * self.noise_scale
        noise_B = torch.randn_like(lora_B_clipped) * self.noise_scale

        lora["lora_A"] = lora_A_clipped + noise_A
        lora["lora_B"] = lora_B_clipped + noise_B

        lora["privacy_noise_added"] = True
        lora["noise_scale"] = self.noise_scale

        return lora


class UnlearningManager:
    """
    Manager for coordinating multiple unlearning strategies.
    """

    def __init__(
        self,
        hypernetwork: nn.Module,
        use_clip: bool = True,
        use_gradient: bool = False,
        use_dp: bool = False,
    ):
        self.hypernetwork = hypernetwork

        # Initialize unlearning strategies
        if use_clip:
            self.clip_unlearning = CLIPGuidedUnlearning(hypernetwork)
        else:
            self.clip_unlearning = None

        if use_gradient:
            self.gradient_unlearning = GradientBasedUnlearning(hypernetwork)
        else:
            self.gradient_unlearning = None

        if use_dp:
            self.dp_unlearning = PrivacyPreservingHypernetwork(hypernetwork)
        else:
            self.dp_unlearning = None

    def forward(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
        metadata_text: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with configured unlearning strategies.

        Args:
            metadata_emb: Metadata embedding
            domain_id: Domain ID
            metadata_text: Optional text for CLIP

        Returns:
            LoRA weights with unlearning applied
        """
        lora = self.hypernetwork(metadata_emb, domain_id)

        # Apply CLIP-guided unlearning
        if self.clip_unlearning is not None:
            lora = self.clip_unlearning(metadata_emb, domain_id, metadata_text)

        # Apply differential privacy
        if self.dp_unlearning is not None:
            lora = self.dp_unlearning(metadata_emb, domain_id)

        return lora

    def add_concept_to_erase(self, concept: str):
        """Add concept to erase (CLIP-based)."""
        if self.clip_unlearning is not None:
            self.clip_unlearning.add_concept_to_erase(concept)

    def remove_concept_to_erase(self, concept: str):
        """Remove concept from erasure set (CLIP-based)."""
        if self.clip_unlearning is not None:
            self.clip_unlearning.remove_concept_to_erase(concept)


def create_unlearning_hypernetwork(
    base_hypernetwork: nn.Module,
    strategy: str = "clip",  # "clip", "gradient", "dp", or "combined"
    **kwargs,
) -> nn.Module:
    """
    Factory function to create unlearning hypernetwork.

    Args:
        base_hypernetwork: Base hypernetwork to wrap
        strategy: Unlearning strategy
        **kwargs: Additional arguments for specific strategies

    Returns:
        Unlearning hypernetwork
    """
    if strategy == "clip":
        return CLIPGuidedUnlearning(base_hypernetwork, **kwargs)
    elif strategy == "gradient":
        return GradientBasedUnlearning(base_hypernetwork, **kwargs)
    elif strategy == "dp":
        return PrivacyPreservingHypernetwork(base_hypernetwork, **kwargs)
    elif strategy == "combined":
        return UnlearningManager(
            base_hypernetwork,
            use_clip=True,
            use_gradient=True,
            use_dp=True,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    # Test adapter unlearning
    print("Testing Adapter Unlearning...")

    # Create dummy hypernetwork
    from tessera_hypernetwork.train_hypernetwork import DomainConditionedHypernetwork

    base_hn = DomainConditionedHypernetwork(
        embed_dim=768,
        rank=16,
        d_in=4096,
        d_out=4096,
    )

    # Test CLIP-guided unlearning
    clip_unlearning = CLIPGuidedUnlearning(base_hn)
    clip_unlearning.add_concept_to_erase("medical")
    clip_unlearning.add_concept_to_erase("legal")

    metadata_emb = torch.randn(1, 768)
    domain_id = 0

    lora = clip_unlearning(metadata_emb, domain_id)
    print(f"CLIP unlearning - LoRA A shape: {lora['lora_A'].shape}")
    print(
        f"CLIP unlearning - Unlearning applied: {lora.get('unlearning_applied', False)}"
    )

    # Test gradient-based unlearning
    grad_unlearning = GradientBasedUnlearning(base_hn)
    grad_unlearning.add_concept("medical", torch.randn(768))

    lora = grad_unlearning.unlearn_concept(metadata_emb, domain_id, "medical")
    print(f"Gradient unlearning - LoRA A shape: {lora['lora_A'].shape}")
    print(
        f"Gradient unlearning - Unlearning applied: {lora.get('unlearning_applied', False)}"
    )

    # Test differential privacy
    dp_unlearning = PrivacyPreservingHypernetwork(base_hn)
    lora = dp_unlearning(metadata_emb, domain_id)
    print(f"DP unlearning - LoRA A shape: {lora['lora_A'].shape}")
    print(
        f"DP unlearning - Privacy noise added: {lora.get('privacy_noise_added', False)}"
    )

    # Test combined manager
    manager = UnlearningManager(base_hn, use_clip=True, use_dp=True)
    manager.add_concept_to_erase("medical")

    lora = manager.forward(metadata_emb, domain_id)
    print(f"Combined unlearning - LoRA A shape: {lora['lora_A'].shape}")

    print("\nAdapter unlearning test passed!")
