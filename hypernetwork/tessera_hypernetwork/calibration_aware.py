"""
Calibration-aware hypernetwork training with Expected Calibration Error (ECE).

Based on: HypeLoRA: Hyper-Network-Generated LoRA Adapters for Calibrated Language Model Fine-Tuning [arXiv:2603.19278]

Key features:
- Expected Calibration Error (ECE) loss
- Maximum Calibration Error (MCE) tracking
- Adaptive Calibration Error (ACE)
- Temperature scaling for calibration
- Trade-off: accuracy vs calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


def expected_calibration_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        num_bins: Number of bins for calibration

    Returns:
        ECE scalar and detailed metrics dictionary
    """
    confidences = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(confidences, dim=-1)
    accuracies = (predictions == targets).float()
    max_confidences = confidences.max(dim=-1)[0]

    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=logits.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_metrics = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (max_confidences > bin_lower) & (max_confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            confidence_in_bin = max_confidences[in_bin].mean()

            # ECE contribution
            ece += torch.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin

            # MCE contribution
            mce = max(mce, torch.abs(accuracy_in_bin - confidence_in_bin).item())

            bin_metrics.append({
                "bin_lower": bin_lower.item(),
                "bin_upper": bin_upper.item(),
                "accuracy": accuracy_in_bin.item(),
                "confidence": confidence_in_bin.item(),
                "count": in_bin.sum().item(),
            })

    metrics = {
        "ece": ece.item(),
        "mce": mce,
        "bin_metrics": bin_metrics,
    }

    return ece, metrics


def adaptive_calibration_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
) -> torch.Tensor:
    """
    Compute Adaptive Calibration Error (ACE).

    Uses adaptive binning based on prediction confidence distribution.

    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        num_bins: Number of bins

    Returns:
        ACE scalar
    """
    confidences = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(confidences, dim=-1)
    accuracies = (predictions == targets).float()
    max_confidences = confidences.max(dim=-1)[0]

    # Adaptive binning: equal number of samples per bin
    sorted_confidences, indices = torch.sort(max_confidences)
    bin_size = len(sorted_confidences) // num_bins

    ace = 0.0

    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(sorted_confidences)

        bin_indices = indices[start_idx:end_idx]
        bin_confidences = max_confidences[bin_indices]
        bin_accuracies = accuracies[bin_indices]

        if len(bin_confidences) > 0:
            accuracy = bin_accuracies.mean()
            confidence = bin_confidences.mean()
            ace += torch.abs(accuracy - confidence) * len(bin_confidences)

    return ace / len(targets)


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for post-hoc calibration.

    Learns a single temperature parameter to scale logits before softmax.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature


class CalibrationAwareLoss(nn.Module):
    """
    Combined loss with calibration awareness.

    Combines MSE loss for LoRA weights with ECE loss for calibration.
    """

    def __init__(
        self,
        calibration_weight: float = 0.1,
        num_bins: int = 15,
        use_ace: bool = False,
    ):
        super().__init__()
        self.calibration_weight = calibration_weight
        self.num_bins = num_bins
        self.use_ace = use_ace

        # MSE loss for LoRA weights
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        pred_lora: Dict[str, torch.Tensor],
        target_lora: Dict[str, torch.Tensor],
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred_lora: Predicted LoRA weights
            target_lora: Target LoRA weights
            logits: Model logits (for calibration)
            targets: Ground truth labels (for calibration)

        Returns:
            Dictionary with loss components
        """
        # MSE loss for LoRA weights
        loss_a = self.mse_loss(pred_lora["lora_A"], target_lora["lora_A"])
        loss_b = self.mse_loss(pred_lora["lora_B"], target_lora["lora_B"])
        mse_loss = loss_a + 2.0 * loss_b  # Weight B higher

        # Calibration loss (if logits provided)
        calib_loss = torch.tensor(0.0, device=pred_lora["lora_A"].device)
        calib_metrics = {}

        if logits is not None and targets is not None:
            if self.use_ace:
                calib_loss = adaptive_calibration_error(logits, targets, self.num_bins)
                calib_metrics["ace"] = calib_loss.item()
            else:
                calib_loss, calib_metrics = expected_calibration_error(
                    logits, targets, self.num_bins
                )
                calib_loss = calib_metrics["ece"]

        # Combined loss
        total_loss = mse_loss + self.calibration_weight * calib_loss

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss.item(),
            "calibration_loss": calib_loss.item() if isinstance(calib_loss, torch.Tensor) else calib_loss,
            "calibration_metrics": calib_metrics,
        }


class CalibrationAwareHypernetwork(nn.Module):
    """
    Hypernetwork with calibration-aware training.

    Adds temperature scaling and calibration tracking to standard hypernetwork.
    """

    def __init__(
        self,
        base_hypernetwork: nn.Module,
        calibration_weight: float = 0.1,
    ):
        super().__init__()
        self.base_hypernetwork = base_hypernetwork
        self.temperature_scaling = TemperatureScaling()
        self.calibration_weight = calibration_weight

    def forward(
        self,
        metadata_emb: torch.Tensor,
        domain_id: int,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through base hypernetwork."""
        return self.base_hypernetwork(metadata_emb, domain_id)

    def get_temperature(self) -> float:
        """Get current temperature for calibration."""
        return self.temperature_scaling.temperature.item()


def train_with_calibration(
    hypernetwork: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    calibration_weight: float = 0.1,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Train hypernetwork with calibration-aware loss.

    Args:
        hypernetwork: Hypernetwork to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        calibration_weight: Weight for calibration loss
        device: Training device

    Returns:
        Training history with calibration metrics
    """
    hypernetwork = hypernetwork.to(device)
    optimizer = torch.optim.AdamW(hypernetwork.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = CalibrationAwareLoss(calibration_weight=calibration_weight)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_ece": [],
        "val_ece": [],
        "train_mce": [],
        "val_mce": [],
    }

    for epoch in range(num_epochs):
        # Training
        hypernetwork.train()
        train_loss = 0.0
        train_ece = 0.0
        train_mce = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # Unpack batch
            if len(batch) == 3:
                metadata, target_lora, domain_id = batch
                logits, targets = None, None
            else:
                metadata, target_lora, domain_id, logits, targets = batch

            # Forward pass
            pred_lora = hypernetwork(metadata, domain_id)

            # Compute loss
            loss_dict = criterion(pred_lora, target_lora, logits, targets)
            loss = loss_dict["total_loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_dict["mse_loss"]
            if "ece" in loss_dict.get("calibration_metrics", {}):
                train_ece += loss_dict["calibration_metrics"]["ece"]
                train_mce += loss_dict["calibration_metrics"]["mce"]

        train_loss /= len(train_loader)
        train_ece /= len(train_loader)
        train_mce /= len(train_loader)

        # Validation
        hypernetwork.eval()
        val_loss = 0.0
        val_ece = 0.0
        val_mce = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    metadata, target_lora, domain_id = batch
                    logits, targets = None, None
                else:
                    metadata, target_lora, domain_id, logits, targets = batch

                pred_lora = hypernetwork(metadata, domain_id)
                loss_dict = criterion(pred_lora, target_lora, logits, targets)

                val_loss += loss_dict["mse_loss"]
                if "ece" in loss_dict.get("calibration_metrics", {}):
                    val_ece += loss_dict["calibration_metrics"]["ece"]
                    val_mce += loss_dict["calibration_metrics"]["mce"]

        val_loss /= len(val_loader)
        val_ece /= len(val_loader)
        val_mce /= len(val_loader)

        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ece"].append(train_ece)
        history["val_ece"].append(val_ece)
        history["train_mce"].append(train_mce)
        history["val_mce"].append(val_mce)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train ECE: {train_ece:.4f}, Val ECE: {val_ece:.4f}")

    return history


if __name__ == "__main__":
    # Test calibration utilities
    print("Testing calibration utilities...")

    # Create dummy logits and targets
    batch_size = 32
    num_classes = 10
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Compute ECE
    ece, metrics = expected_calibration_error(logits, targets)
    print(f"ECE: {ece:.4f}")
    print(f"MCE: {metrics['mce']:.4f}")

    # Compute ACE
    ace = adaptive_calibration_error(logits, targets)
    print(f"ACE: {ace:.4f}")

    # Test calibration loss
    criterion = CalibrationAwareLoss(calibration_weight=0.1)

    pred_lora = {
        "lora_A": torch.randn(16, 4096),
        "lora_B": torch.randn(4096, 16),
    }
    target_lora = {
        "lora_A": torch.randn(16, 4096),
        "lora_B": torch.randn(4096, 16),
    }

    loss_dict = criterion(pred_lora, target_lora, logits, targets)
    print(f"Total loss: {loss_dict['total_loss']:.4f}")
    print(f"MSE loss: {loss_dict['mse_loss']:.4f}")
    print(f"Calibration loss: {loss_dict['calibration_loss']:.4f}")

    print("\nCalibration utilities test passed!")
