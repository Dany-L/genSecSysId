"""Optimizer and learning rate scheduler utilities."""

from typing import Literal

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
)


def get_optimizer(
    parameters,
    optimizer_type: Literal["adam", "sgd", "rmsprop", "adamw"] = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs,
):
    """
    Get optimizer by name.

    Args:
        parameters: Model parameters
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance
    """
    if optimizer_type == "adam":
        return optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )
    elif optimizer_type == "adamw":
        return optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )
    elif optimizer_type == "sgd":
        return optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
        )
    elif optimizer_type == "rmsprop":
        return optim.RMSprop(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.0),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(
    optimizer,
    scheduler_type: Literal[
        "step", "exponential", "reduce_on_plateau", "cosine"
    ] = "reduce_on_plateau",
    **kwargs,
):
    """
    Get learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler
        **kwargs: Scheduler-specific arguments

    Returns:
        Scheduler instance or None
    """
    if scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_type == "exponential":
        return ExponentialLR(
            optimizer,
            gamma=kwargs.get("gamma", 0.95),
        )
    elif scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
            verbose=True,
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 100),
            eta_min=kwargs.get("eta_min", 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
