"""Utility functions."""

import torch
import numpy as np
import random


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """
    Get torch device.
    
    Args:
        device: Device string ("auto", "cuda", "cpu", "mps")
        
    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module):
    """Print model summary."""
    print("=" * 60)
    print("Model Architecture")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 60)
