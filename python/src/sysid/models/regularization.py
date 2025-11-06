"""Custom regularization functions for model parameters."""

import torch
import torch.nn as nn
from typing import Optional, Callable


def parameter_regularization(
    model: nn.Module,
    reg_type: str = "l2",
    weight: float = 1e-3,
    parameter_filter: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Compute regularization loss on model parameters.
    
    Args:
        model: PyTorch model
        reg_type: Type of regularization ("l1", "l2", "elastic")
        weight: Regularization weight
        parameter_filter: Optional function to filter parameters (e.g., lambda name, param: "weight" in name)
        
    Returns:
        Regularization loss
    """
    reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    
    for name, param in model.named_parameters():
        if parameter_filter is not None and not parameter_filter(name, param):
            continue
        
        if reg_type == "l1":
            reg_loss = reg_loss + torch.abs(param).sum()
        elif reg_type == "l2":
            reg_loss = reg_loss + torch.square(param).sum()
        elif reg_type == "elastic":
            # Elastic net: combination of L1 and L2
            reg_loss = reg_loss + 0.5 * torch.abs(param).sum() + 0.5 * torch.square(param).sum()
    
    return weight * reg_loss


def lipschitz_regularization(
    weight_matrix: torch.Tensor,
    target_lipschitz: float = 1.0,
    method: str = "spectral_norm",
) -> torch.Tensor:
    """
    Regularization to constrain Lipschitz constant of weight matrices.
    
    Args:
        weight_matrix: Weight matrix to regularize
        target_lipschitz: Target Lipschitz constant
        method: Method to estimate Lipschitz constant ("spectral_norm", "frobenius")
        
    Returns:
        Regularization loss
    """
    if method == "spectral_norm":
        # Approximate largest singular value via power iteration
        # For efficiency, you might want to cache this computation
        _, s, _ = torch.svd(weight_matrix)
        spectral_norm = s[0]
        loss = torch.relu(spectral_norm - target_lipschitz) ** 2
    elif method == "frobenius":
        # Use Frobenius norm as upper bound on spectral norm
        frobenius_norm = torch.norm(weight_matrix, p="fro")
        loss = torch.relu(frobenius_norm - target_lipschitz) ** 2
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return loss


def bounded_parameters_regularization(
    param: torch.Tensor,
    lower_bound: float,
    upper_bound: float,
    penalty_type: str = "quadratic",
) -> torch.Tensor:
    """
    Regularization to keep parameters within bounds.
    
    Args:
        param: Parameter tensor
        lower_bound: Lower bound
        upper_bound: Upper bound
        penalty_type: Type of penalty ("quadratic", "linear")
        
    Returns:
        Regularization loss
    """
    lower_violation = torch.relu(lower_bound - param)
    upper_violation = torch.relu(param - upper_bound)
    
    if penalty_type == "quadratic":
        loss = (lower_violation ** 2).sum() + (upper_violation ** 2).sum()
    elif penalty_type == "linear":
        loss = lower_violation.sum() + upper_violation.sum()
    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}")
    
    return loss


def stability_regularization(
    recurrent_weight: torch.Tensor,
    target_spectral_radius: float = 0.9,
) -> torch.Tensor:
    """
    Regularization to encourage stability by constraining spectral radius.
    
    Args:
        recurrent_weight: Recurrent weight matrix
        target_spectral_radius: Target spectral radius (< 1 for stability)
        
    Returns:
        Regularization loss
    """
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(recurrent_weight)
    spectral_radius = torch.max(torch.abs(eigenvalues))
    
    # Penalize if spectral radius exceeds target
    loss = torch.relu(spectral_radius - target_spectral_radius) ** 2
    
    return loss
