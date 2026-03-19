"""Loss functions for system identification."""

from typing import Literal

import torch
import torch.nn as nn



class MaskedLoss(nn.Module):
    """Wrap a base loss to ignore NaN targets.

    Loss is computed only on finite (non-NaN) target values. NaN values are
    completely ignored in the loss computation. This is used to exclude invalid
    or diverging trajectory segments from training.
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss on valid (non-NaN) target values only.

        Args:
            predictions: Predicted values, any shape
            targets: Target values, same shape as predictions; may contain NaN

        Returns:
            Scalar loss computed over finite target entries
        """
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions and targets must have the same shape, got "
                f"{predictions.shape} and {targets.shape}"
            )

        # Identify valid (finite) entries in targets
        valid = torch.isfinite(targets)

        if not valid.any():
            # No valid values to compute loss on
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

        # Compute loss only on valid entries
        return self.base_loss(predictions[valid], targets[valid])



def get_loss_function(loss_type: Literal["mse", "mae", "huber", "smooth_l1"]) -> nn.Module:
    """
    Get loss function by name.

    Returns a :class:`MaskedLoss` wrapper so that NaN-padded positions in the
    target (used for variable-length trajectory padding) are automatically
    excluded from the loss computation.

    Args:
        loss_type: Type of loss function

    Returns:
        Loss function module
    """
    if loss_type == "mse":
        base = nn.MSELoss()
    elif loss_type == "mae":
        base = nn.L1Loss()
    elif loss_type == "huber":
        base = nn.HuberLoss()
    elif loss_type == "smooth_l1":
        base = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return MaskedLoss(base)


class MultiStepLoss(nn.Module):
    """Multi-step prediction loss."""

    def __init__(self, base_loss: nn.Module, weights: list = None):
        """
        Initialize multi-step loss.

        Args:
            base_loss: Base loss function (e.g., MSE)
            weights: Weights for each time step (if None, use equal weights)
        """
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-step loss.

        Args:
            predictions: Predicted sequences (batch, seq_len, features)
            targets: Target sequences (batch, seq_len, features)

        Returns:
            Weighted loss over time steps
        """
        seq_len = predictions.shape[1]

        if self.weights is None:
            # Equal weights for all time steps
            loss = self.base_loss(predictions, targets)
        else:
            # Weighted sum over time steps
            loss = 0
            for t in range(seq_len):
                weight = self.weights[t] if t < len(self.weights) else self.weights[-1]
                loss = loss + weight * self.base_loss(predictions[:, t], targets[:, t])
            loss = loss / sum(self.weights[:seq_len])

        return loss
