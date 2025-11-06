"""Base model class for RNN-based system identification."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseRNN(nn.Module, ABC):
    """Base class for RNN models."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initialize the base RNN model.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            output_size: Dimension of output
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Initial hidden state (optional)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        pass
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute custom regularization loss on model parameters.
        This can be overridden by subclasses for specific constraints.
        
        Returns:
            Regularization loss tensor
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_parameter_dict(self) -> Dict[str, Any]:
        """
        Get dictionary of model parameters for logging/analysis.
        
        Returns:
            Dictionary of parameter names and values
        """
        param_dict = {}
        for name, param in self.named_parameters():
            param_dict[name] = {
                "shape": list(param.shape),
                "mean": param.data.mean().item(),
                "std": param.data.std().item(),
                "min": param.data.min().item(),
                "max": param.data.max().item(),
            }
        return param_dict
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save model state dict."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
        }, path)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "BaseRNN":
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        config = checkpoint["config"]
        config.update(kwargs)  # Allow override
        
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        
        return model
