"""
LoRA Ensemble for uncertainty estimation.

Based on: LoRA-Ensemble: Efficient Uncertainty Modelling for Self-Attention Networks [arXiv:2405.14438]

Key features:
- Train ensemble of hypernetworks for uncertainty estimation
- Efficient uncertainty without full model ensembles
- Mean prediction with variance-based uncertainty
- Detects out-of-distribution inputs
- Minimal parameter overhead
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


class LoRAEnsemble(nn.Module):
    """
    Ensemble of hypernetworks for uncertainty estimation.

    Trains multiple hypernetworks with different initializations
    to provide uncertainty estimates through variance.
    """

    def __init__(
        self,
        num_models: int = 5,
        embed_dim: int = 768,
        rank: int = 16,
        d_in: int = 4096,
        d_out: int = 4096,
        hidden_dim: int = 2048,
        num_domains: int = 10,
    ):
        super().__init__()
        self.num_models = num_models
        self.embed_dim = embed_dim
        self.rank = rank

        # Create ensemble of hypernetworks
        from tessera_hypernetwork.train_hypernetwork import DomainConditionedHypernetwork
        self.models = nn.ModuleList([
            DomainConditionedHypernetwork(
                embed_dim=embed_dim,
                rank=rank,
                d_in=d_in,
                d_out=d_out,
                hidden_dim=hidden_dim,
                num_domains=num_domains,
            )
            for _ in range(num_models)
        ])

    def forward(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
        return_uncertainty: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.

        Args:
            metadata_emb: Metadata embedding [batch_size, embed_dim]
            domain_id: Domain ID
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary with mean prediction and uncertainty
        """
        predictions = []

        # Get predictions from all models
        for model in self.models:
            pred = model(metadata_emb, domain_id)
            predictions.append(pred)

        # Stack predictions
        lora_A_stack = torch.stack([p["lora_A"] for p in predictions])
        lora_B_stack = torch.stack([p["lora_B"] for p in predictions])

        # Compute mean prediction
        mean_lora_A = lora_A_stack.mean(dim=0)
        mean_lora_B = lora_B_stack.mean(dim=0)

        result = {
            "lora_A": mean_lora_A,
            "lora_B": mean_lora_B,
        }

        if return_uncertainty:
            # Compute variance as uncertainty
            uncertainty_lora_A = lora_A_stack.var(dim=0)
            uncertainty_lora_B = lora_B_stack.var(dim=0)

            result["uncertainty_lora_A"] = uncertainty_lora_A
            result["uncertainty_lora_B"] = uncertainty_lora_B

            # Aggregate uncertainty metrics
            result["uncertainty_score"] = (
                uncertainty_lora_A.mean() + uncertainty_lora_B.mean()
            ).item()

        return result

    def get_individual_predictions(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Get individual predictions from each ensemble member.

        Args:
            metadata_emb: Metadata embedding
            domain_id: Domain ID

        Returns:
            List of individual predictions
        """
        predictions = []
        for model in self.models:
            pred = model(metadata_emb, domain_id)
            predictions.append(pred)
        return predictions

    def get_disagreement(self, metadata_emb: torch.Tensor, domain_id: int) -> float:
        """
        Compute ensemble disagreement as uncertainty metric.

        Args:
            metadata_emb: Metadata embedding
            domain_id: Domain ID

        Returns:
            Disagreement score
        """
        predictions = self.get_individual_predictions(metadata_emb, domain_id)

        # Compute pairwise disagreement
        disagreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                diff_A = (predictions[i]["lora_A"] - predictions[j]["lora_A"]).abs().mean()
                diff_B = (predictions[i]["lora_B"] - predictions[j]["lora_B"]).abs().mean()
                disagreements.append((diff_A + diff_B).item())

        return np.mean(disagreements)


class WeightedLoRAEnsemble(nn.Module):
    """
    Weighted ensemble with learnable combination weights.

    Learns optimal combination of ensemble members.
    """

    def __init__(
        self,
        num_models: int = 5,
        embed_dim: int = 768,
        rank: int = 16,
        d_in: int = 4096,
        d_out: int = 4096,
        hidden_dim: int = 2048,
        num_domains: int = 10,
    ):
        super().__init__()
        self.num_models = num_models

        # Create ensemble
        from tessera_hypernetwork.train_hypernetwork import DomainConditionedHypernetwork
        self.models = nn.ModuleList([
            DomainConditionedHypernetwork(
                embed_dim=embed_dim,
                rank=rank,
                d_in=d_in,
                d_out=d_out,
                hidden_dim=hidden_dim,
                num_domains=num_domains,
            )
            for _ in range(num_models)
        ])

        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.ones(num_models) / num_models)

    def forward(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with weighted combination.

        Args:
            metadata_emb: Metadata embedding
            domain_id: Domain ID

        Returns:
            Weighted prediction
        """
        # Normalize weights
        weights = F.softmax(self.combination_weights, dim=0)

        # Get weighted predictions
        weighted_lora_A = torch.zeros_like(self.models[0](metadata_emb, domain_id)["lora_A"])
        weighted_lora_B = torch.zeros_like(self.models[0](metadata_emb, domain_id)["lora_B"])

        for i, model in enumerate(self.models):
            pred = model(metadata_emb, domain_id)
            weighted_lora_A += weights[i] * pred["lora_A"]
            weighted_lora_B += weights[i] * pred["lora_B"]

        return {
            "lora_A": weighted_lora_A,
            "lora_B": weighted_lora_B,
            "weights": weights,
        }


class DeepEnsembleTrainer:
    """
    Trainer for deep ensembles.

    Trains each ensemble member independently with different random seeds.
    """

    def __init__(
        self,
        ensemble: LoRAEnsemble,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        device: str = "cuda",
    ):
        self.ensemble = ensemble
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device

    def train_member(
        self,
        member_idx: int,
        seed: int,
    ) -> Dict[str, List[float]]:
        """
        Train a single ensemble member.

        Args:
            member_idx: Index of ensemble member
            seed: Random seed for this member

        Returns:
            Training history
        """
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Get member
        model = self.ensemble.models[member_idx]
        model = model.to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        # Loss
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.num_epochs):
            # Training
            model.train()
            train_loss = 0.0

            for batch in self.train_loader:
                metadata, target_lora, domain_id = batch
                optimizer.zero_grad()

                pred_lora = model(metadata, domain_id)

                loss_a = criterion(pred_lora["lora_A"], target_lora["lora_A"])
                loss_b = criterion(pred_lora["lora_B"], target_lora["lora_B"])
                loss = loss_a + 2.0 * loss_b

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in self.val_loader:
                    metadata, target_lora, domain_id = batch
                    pred_lora = model(metadata, domain_id)

                    loss_a = criterion(pred_lora["lora_A"], target_lora["lora_A"])
                    loss_b = criterion(pred_lora["lora_B"], target_lora["lora_B"])
                    val_loss += (loss_a + 2.0 * loss_b).item()

            val_loss /= len(self.val_loader)

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Member {member_idx}, Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return history

    def train_ensemble(self) -> Dict[int, Dict[str, List[float]]]:
        """
        Train all ensemble members.

        Returns:
            Dictionary of training histories for each member
        """
        histories = {}

        for i in range(self.ensemble.num_models):
            print(f"\nTraining ensemble member {i+1}/{self.ensemble.num_models}")
            seed = 42 + i  # Different seed for each member
            histories[i] = self.train_member(i, seed)

        return histories


def detect_out_of_distribution(
    ensemble: LoRAEnsemble,
    metadata_emb: torch.Tensor,
    domain_id: int,
    threshold: float = 0.1,
) -> bool:
    """
    Detect if input is out-of-distribution using ensemble uncertainty.

    Args:
        ensemble: LoRA ensemble
        metadata_emb: Metadata embedding
        domain_id: Domain ID
        threshold: Uncertainty threshold for OOD detection

    Returns:
        True if OOD, False otherwise
    """
    with torch.no_grad():
        result = ensemble(metadata_emb, domain_id, return_uncertainty=True)
        uncertainty = result["uncertainty_score"]

    return uncertainty > threshold


if __name__ == "__main__":
    # Test LoRA ensemble
    print("Testing LoRA Ensemble...")

    ensemble = LoRAEnsemble(
        num_models=3,
        embed_dim=768,
        rank=16,
        d_in=4096,
        d_out=4096,
    )

    metadata_emb = torch.randn(1, 768)
    domain_id = 0

    # Forward pass
    result = ensemble(metadata_emb, domain_id, return_uncertainty=True)

    print(f"LoRA A shape: {result['lora_A'].shape}")
    print(f"LoRA B shape: {result['lora_B'].shape}")
    print(f"Uncertainty score: {result['uncertainty_score']:.4f}")
    print(f"Uncertainty A shape: {result['uncertainty_lora_A'].shape}")
    print(f"Uncertainty B shape: {result['uncertainty_lora_B'].shape}")

    # Get disagreement
    disagreement = ensemble.get_disagreement(metadata_emb, domain_id)
    print(f"Ensemble disagreement: {disagreement:.4f}")

    # OOD detection
    is_ood = detect_out_of_distribution(ensemble, metadata_emb, domain_id, threshold=1.0)
    print(f"Is out-of-distribution: {is_ood}")

    print("\nLoRA ensemble test passed!")
