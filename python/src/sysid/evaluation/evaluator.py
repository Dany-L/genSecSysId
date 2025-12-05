"""Evaluator class for trained models."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.base import BaseRNN
from .metrics import compute_metrics, compute_simulation_metrics


class Evaluator:
    """Evaluator for RNN models."""

    def __init__(
        self,
        model: BaseRNN,
        device: str = "cuda",
        output_dir: str = "evaluation_results",
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            device: Device to evaluate on
            output_dir: Directory for evaluation results
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        test_loader: DataLoader,
        normalizer: Optional[Any] = None,
        print_results: bool = True,
        save_files: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.

        Args:
            test_loader: Test data loader
            normalizer: Data normalizer (for denormalization)
            print_results: Whether to print results to console
            save_files: Whether to save prediction/target files to disk

        Returns:
            Dictionary of evaluation results
        """
        all_predictions = []
        all_targets = []
        all_inputs = []
        all_states = []

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch (states may be None)
                if len(batch) == 3:
                    d, e, x = batch  # d: input, e: output, x: states (optional)
                else:
                    d, e = batch
                    x = None

                d = d.to(self.device)
                e = e.to(self.device)

                # Forward pass
                e_hat = self.model(d, x)  # e_hat: predicted output

                all_predictions.append(e_hat.cpu().numpy())
                all_targets.append(e.cpu().numpy())
                all_inputs.append(d.cpu().numpy())
                if x is not None:
                    all_states.append(x.cpu().numpy())

        # Concatenate all batches
        e_hat = np.concatenate(all_predictions, axis=0)
        e = np.concatenate(all_targets, axis=0)
        d = np.concatenate(all_inputs, axis=0)
        x = np.concatenate(all_states, axis=0) if len(all_states) > 0 else None

        # Denormalize if normalizer provided
        if normalizer is not None:
            e_hat = normalizer.inverse_transform_outputs(e_hat)
            e = normalizer.inverse_transform_outputs(e)
            d = normalizer.inverse_transform_inputs(d)

        # Compute metrics
        metrics = compute_metrics(
            e_hat.reshape(-1, e_hat.shape[-1]),
            e.reshape(-1, e.shape[-1]),
        )

        # Compute per-step metrics for sequences
        if e_hat.ndim == 3:
            sim_metrics = compute_simulation_metrics(e_hat, e)
            metrics.update(sim_metrics)

        # Save results
        results = {
            "metrics": metrics,
            "predictions_shape": e_hat.shape,
            "targets_shape": e.shape,
            "e_hat": e_hat,
            "e": e,
        }
        if x is not None:
            results["states_shape"] = x.shape

        if save_files:
            with open(self.output_dir / "evaluation_results.json", "w") as f:
                # Remove per_step from saved metrics (too verbose)
                save_metrics = {k: v for k, v in metrics.items() if k != "per_step"}
                json.dump({"metrics": save_metrics}, f, indent=2)

            # Save predictions, targets, inputs, and states
            np.save(self.output_dir / "predictions.npy", e_hat)
            np.save(self.output_dir / "targets.npy", e)
            np.save(self.output_dir / "inputs.npy", d)
            if x is not None:
                np.save(self.output_dir / "states.npy", x)

        if print_results:
            print("Evaluation Results:")
            for key, value in metrics.items():
                if key != "per_step":
                    print(f"  {key}: {value:.6f}")

        return results

    def analyze_errors(
        self,
        e_hat: np.ndarray,  # predicted output
        e: np.ndarray,  # output (target)
        save_path: Optional[str] = None,
    ):
        """
        Analyze prediction errors - simplified to show error over time.

        Args:
            e_hat: Predicted output values
            e: Output (target) values
            save_path: Path to save figure
        """
        errors = e_hat - e

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Error over time (for sequences)
        if e_hat.ndim == 3:
            abs_errors_per_step = np.mean(np.abs(errors), axis=(0, 2))
            ax.plot(abs_errors_per_step, linewidth=2)
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
            ax.set_title("Prediction Error Over Time", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Add statistical info
            mean_error = np.mean(abs_errors_per_step)
            max_error = np.max(abs_errors_per_step)
            ax.axhline(
                y=mean_error,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean MAE: {mean_error:.4f}",
                alpha=0.7,
            )
            ax.axhline(
                y=max_error,
                color="orange",
                linestyle="--",
                linewidth=1.5,
                label=f"Max MAE: {max_error:.4f}",
                alpha=0.7,
            )
            ax.legend(fontsize=10)
        else:
            # For non-sequence data, show error distribution
            ax.hist(np.abs(errors).flatten(), bins=50, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Absolute Error", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title("Prediction Error Distribution", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "error_analysis.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Error analysis plot saved to {save_path}")
