"""Evaluation metrics for system identification."""

import numpy as np
from typing import Dict
import torch


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute various evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        Dictionary of metrics
    """
    # Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # R-squared score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Normalized RMSE (NRMSE)
    nrmse = rmse / (np.max(targets) - np.min(targets) + 1e-10)
    
    # Max error
    max_error = np.max(np.abs(predictions - targets))
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "nrmse": float(nrmse),
        "max_error": float(max_error),
    }


def compute_simulation_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    horizon: int = None,
) -> Dict[str, float]:
    """
    Compute metrics for multi-step simulation.
    
    Args:
        predictions: Predicted sequences (n_samples, seq_len, features)
        targets: Target sequences
        horizon: Prediction horizon to evaluate (if None, use full sequence)
        
    Returns:
        Dictionary of metrics per time step
    """
    if horizon is None:
        horizon = predictions.shape[1]
    
    metrics_per_step = {}
    
    for t in range(horizon):
        step_metrics = compute_metrics(
            predictions[:, t, :],
            targets[:, t, :],
        )
        for key, value in step_metrics.items():
            if key not in metrics_per_step:
                metrics_per_step[key] = []
            metrics_per_step[key].append(value)
    
    # Average metrics
    avg_metrics = {
        f"{key}_avg": np.mean(values)
        for key, values in metrics_per_step.items()
    }
    
    # Final step metrics
    final_metrics = {
        f"{key}_final": values[-1]
        for key, values in metrics_per_step.items()
    }
    
    return {**avg_metrics, **final_metrics, "per_step": metrics_per_step}
