"""Loss functions for system identification."""

import torch
import torch.nn as nn
from typing import Literal


def get_loss_function(loss_type: Literal["mse", "mae", "huber", "smooth_l1"]) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: Type of loss function
        
    Returns:
        Loss function module
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.HuberLoss()
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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
