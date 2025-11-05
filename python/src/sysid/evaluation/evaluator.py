"""Evaluator class for trained models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

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
    ) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_loader: Test data loader
            normalizer: Data normalizer (for denormalization)
            
        Returns:
            Dictionary of evaluation results
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Denormalize if normalizer provided
        if normalizer is not None:
            predictions = normalizer.inverse_transform_outputs(predictions)
            targets = normalizer.inverse_transform_outputs(targets)
        
        # Compute metrics
        metrics = compute_metrics(
            predictions.reshape(-1, predictions.shape[-1]),
            targets.reshape(-1, targets.shape[-1]),
        )
        
        # Compute per-step metrics for sequences
        if predictions.ndim == 3:
            sim_metrics = compute_simulation_metrics(predictions, targets)
            metrics.update(sim_metrics)
        
        # Save results
        results = {
            "metrics": metrics,
            "predictions_shape": predictions.shape,
            "targets_shape": targets.shape,
        }
        
        with open(self.output_dir / "evaluation_results.json", "w") as f:
            # Remove per_step from saved metrics (too verbose)
            save_metrics = {k: v for k, v in metrics.items() if k != "per_step"}
            json.dump({"metrics": save_metrics}, f, indent=2)
        
        # Save predictions and targets
        np.save(self.output_dir / "predictions.npy", predictions)
        np.save(self.output_dir / "targets.npy", targets)
        
        print("Evaluation Results:")
        for key, value in metrics.items():
            if key != "per_step":
                print(f"  {key}: {value:.6f}")
        
        return results
    
    def plot_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        num_samples: int = 5,
        save_path: Optional[str] = None,
    ):
        """
        Plot predictions vs targets.
        
        Args:
            predictions: Predicted values
            targets: Target values
            num_samples: Number of samples to plot
            save_path: Path to save figure
        """
        num_samples = min(num_samples, predictions.shape[0])
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            ax = axes[i]
            
            if predictions.ndim == 3:
                # Sequence data
                seq_len = predictions.shape[1]
                for feat in range(predictions.shape[2]):
                    ax.plot(predictions[i, :, feat], label=f"Predicted (feat {feat})", alpha=0.7)
                    ax.plot(targets[i, :, feat], label=f"Target (feat {feat})", linestyle="--", alpha=0.7)
            else:
                # Single-step data
                ax.plot(predictions[i], label="Predicted", alpha=0.7)
                ax.plot(targets[i], label="Target", linestyle="--", alpha=0.7)
            
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.set_title(f"Sample {i + 1}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "predictions_plot.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Predictions plot saved to {save_path}")
    
    def analyze_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        Analyze prediction errors.
        
        Args:
            predictions: Predicted values
            targets: Target values
            save_path: Path to save figure
        """
        errors = predictions - targets
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Error distribution
        axes[0, 0].hist(errors.flatten(), bins=50, edgecolor="black", alpha=0.7)
        axes[0, 0].set_xlabel("Error")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Error Distribution")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Absolute error over time (for sequences)
        if predictions.ndim == 3:
            abs_errors_per_step = np.mean(np.abs(errors), axis=(0, 2))
            axes[0, 1].plot(abs_errors_per_step)
            axes[0, 1].set_xlabel("Time Step")
            axes[0, 1].set_ylabel("Mean Absolute Error")
            axes[0, 1].set_title("Error Over Time")
            axes[0, 1].grid(True, alpha=0.3)
        
        # Predictions vs targets scatter
        axes[1, 0].scatter(targets.flatten(), predictions.flatten(), alpha=0.3, s=1)
        axes[1, 0].plot(
            [targets.min(), targets.max()],
            [targets.min(), targets.max()],
            "r--",
            linewidth=2,
            label="Perfect prediction"
        )
        axes[1, 0].set_xlabel("Target")
        axes[1, 0].set_ylabel("Prediction")
        axes[1, 0].set_title("Predictions vs Targets")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error vs target value
        axes[1, 1].scatter(targets.flatten(), errors.flatten(), alpha=0.3, s=1)
        axes[1, 1].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[1, 1].set_xlabel("Target")
        axes[1, 1].set_ylabel("Error")
        axes[1, 1].set_title("Error vs Target Value")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "error_analysis.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Error analysis plot saved to {save_path}")
