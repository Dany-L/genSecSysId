"""Evaluation metrics for system identification."""

from typing import Dict

import numpy as np


def compute_metrics(e_hat: np.ndarray, e: np.ndarray) -> Dict[str, float]:
    """
    Compute various evaluation metrics, ignoring NaN values.

    Args:
        e_hat: Predicted output values (may contain NaN for invalid regions)
        e: Output (target) values (may contain NaN for invalid regions)

    Returns:
        Dictionary of metrics (computed only on finite values)
    """
    # Identify valid (finite) entries
    valid = np.isfinite(e) & np.isfinite(e_hat)
    
    if not valid.any():
        # No valid values to compute metrics on
        return {
            "mse": float('nan'),
            "rmse": float('nan'),
            "mae": float('nan'),
            "r2": float('nan'),
            "nrmse": float('nan'),
            "max_error": float('nan'),
        }
    
    # Extract valid entries only
    e_hat_valid = e_hat[valid]
    e_valid = e[valid]

    # Mean Squared Error
    mse = np.mean((e_hat_valid - e_valid) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(e_hat_valid - e_valid))

    # R-squared score
    ss_res = np.sum((e_valid - e_hat_valid) ** 2)
    ss_tot = np.sum((e_valid - np.mean(e_valid)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # Normalized RMSE (NRMSE)
    nrmse = rmse / (np.max(e_valid) - np.min(e_valid) + 1e-10)

    # Max error
    max_error = np.max(np.abs(e_hat_valid - e_valid))

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "nrmse": float(nrmse),
        "max_error": float(max_error),
    }


def compute_simulation_metrics(
    e_hat: np.ndarray,  # predicted output
    e: np.ndarray,  # output (target)
    horizon: int = None,
) -> Dict[str, float]:
    """
    Compute metrics for multi-step simulation.

    Args:
        e_hat: Predicted output sequences (n_samples, seq_len, features)
        e: Output (target) sequences
        horizon: Prediction horizon to evaluate (if None, use full sequence)

    Returns:
        Dictionary of metrics per time step
    """
    if horizon is None:
        horizon = e_hat.shape[1]

    metrics_per_step = {}

    for t in range(horizon):
        step_metrics = compute_metrics(
            e_hat[:, t, :],
            e[:, t, :],
        )
        for key, value in step_metrics.items():
            if key not in metrics_per_step:
                metrics_per_step[key] = []
            metrics_per_step[key].append(value)

    # Average metrics
    avg_metrics = {f"{key}_avg": np.mean(values) for key, values in metrics_per_step.items()}

    # Final step metrics
    final_metrics = {f"{key}_final": values[-1] for key, values in metrics_per_step.items()}

    return {**avg_metrics, **final_metrics, "per_step": metrics_per_step}
