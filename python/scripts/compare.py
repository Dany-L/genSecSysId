#!/usr/bin/env python
"""
Compare multiple MLflow runs and generate comparison tables and visualizations.

This script loads multiple trained models from MLflow runs and compares their:
- Evaluation metrics (MSE, RMSE, MAE, R², NRMSE, FIT)
- Training/validation loss curves
- Model parameters
- Stability constraints (for SimpleLure models)

Usage:
    python scripts/compare.py --run-ids <run_id_1> <run_id_2> ... [--test-data <path>] [--output-dir <path>]

Examples:
    # Compare two runs with custom test data
    python scripts/compare.py --run-ids abc123 def456 --test-data data/prepared/test

    # Compare multiple runs and save to custom directory
    python scripts/compare.py --run-ids abc123 def456 ghi789 --output-dir comparisons/my_comparison
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib
import torch

from sysid.config import resolve_run_artifacts, setup_mlflow_tracking
from sysid.data import DataNormalizer
from sysid.data.direct_loader import load_csv_folder
from sysid.evaluation import compute_metrics
from sysid.models import SimpleLure, load_model

UNSTAB_STAB_ZERO = [93, 4, 122]

DEFAULT_DATA_ROOT = "~/genSecSysId-Data"

torch.set_default_dtype(torch.float64)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class RunComparator:
    """Compare multiple MLflow runs."""

    def __init__(
        self,
        run_ids: List[str],
        test_data_path: Optional[str] = None,
        data_root: str = DEFAULT_DATA_ROOT,
    ):
        """
        Initialize the comparator.

        Args:
            run_ids: List of MLflow run IDs to compare
            test_data_path: Path to test data (folder). If None, defaults to
                the test/ split under the first run's config.data.train_path.
            data_root: Base directory under which per-run artefacts live
                (config.yaml, best_model.pt, normalizer.json, training_history.json).
        """
        self.run_ids = run_ids
        self.data_root = data_root
        self.test_data_path = test_data_path
        self.runs_info = []
        self.configs: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.normalizers: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.histories: Dict[str, Any] = {}

    def load_runs(self):
        """Load configs, models, normalizers, and training histories from disk.

        MLflow metadata (params, metrics, run name) is fetched best-effort —
        if the tracking server is unreachable, we still proceed with the
        artefacts that are available locally.
        """
        logger.info(f"Loading {len(self.run_ids)} runs from {self.data_root} ...")

        for run_id in self.run_ids:
            try:
                config, model_path, normalizer_path, run_info_disk = resolve_run_artifacts(
                    run_id, data_root=self.data_root
                )
            except Exception as e:
                logger.error(f"Failed to resolve run {run_id}: {e}")
                continue

            self.configs[run_id] = config

            # MLflow metadata is fetched separately in fetch_mlflow_metadata()
            # so the tracking URI can be configured first.
            self.runs_info.append({
                "run_id": run_id,
                "run_name": (
                    run_info_disk.get("run_name") if run_info_disk else None
                ) or run_id[:8],
                "params": {},
                "metrics": {},
            })

            # Load model + normalizer from local disk.
            try:
                model = load_model(str(model_path), config, device="cpu")
                model.eval()
                self.models[run_id] = model
                logger.info(f"✓ Loaded model for run {run_id[:8]}")
            except Exception as e:
                logger.warning(f"Could not load model for run {run_id[:8]}: {e}")
                continue

            if normalizer_path is not None:
                try:
                    self.normalizers[run_id] = DataNormalizer.load(str(normalizer_path))
                except Exception as e:
                    logger.warning(f"Could not load normalizer for run {run_id[:8]}: {e}")

            # Load training history from the per-run output dir.
            history_path = (
                Path(self.data_root).expanduser()
                / "outputs" / config.model.model_type / run_id / "training_history.json"
            )
            if history_path.exists():
                try:
                    with open(history_path) as f:
                        history = json.load(f)
                    self.histories[run_id] = {
                        "train_losses": history.get("train_losses"),
                        "val_losses": history.get("val_losses"),
                        "train_pred_losses": history.get("train_pred_losses"),
                        "train_reg_losses": history.get("train_reg_losses"),
                    }
                    logger.info(f"✓ Loaded training history for run {run_id[:8]}")
                except Exception as e:
                    logger.warning(f"Could not load training history for {run_id[:8]}: {e}")
            else:
                logger.warning(f"No training_history.json at {history_path}")

        logger.info(f"Successfully loaded {len(self.configs)} runs")

    def fetch_mlflow_metadata(self):
        """Best-effort fetch of run params/metrics from MLflow.

        Call this AFTER `setup_mlflow_tracking` so the tracking URI points
        at the server the runs were logged to. Failures are logged and
        skipped; summary fields fall back to "N/A".
        """
        for info in self.runs_info:
            run_id = info["run_id"]
            try:
                run = mlflow.get_run(run_id)
                info["run_name"] = run.data.tags.get("mlflow.runName", info["run_name"])
                info["params"] = dict(run.data.params)
                info["metrics"] = dict(run.data.metrics)
            except Exception as e:
                logger.warning(f"Could not fetch MLflow metadata for {run_id[:8]}: {e}")

    def _first_config(self):
        """Return any one of the loaded configs (used as the column-name reference)."""
        if not self.configs:
            return None
        return next(iter(self.configs.values()))

    def _predict(self, run_id: str, test_inputs: np.ndarray, test_states: Optional[np.ndarray]):
        """Run a single model end-to-end with its own normalizer.

        Returns predictions in physical (denormalized) units.
        """
        model = self.models[run_id]
        normalizer = self.normalizers.get(run_id)

        inputs = test_inputs
        if normalizer is not None:
            inputs = normalizer.transform_inputs(inputs)
        inputs_tensor = torch.as_tensor(inputs, dtype=torch.float64)
        if test_states is not None:
            x0 = torch.as_tensor(test_states[:, 0, :], dtype=torch.float64)
        else:
            x0 = None

        with torch.no_grad():
            out = model(inputs_tensor, x0=x0)
        # Some models return tuples — first element is e_hat.
        e_hat = out[0] if isinstance(out, tuple) else out
        predictions = e_hat.cpu().numpy()
        if normalizer is not None:
            predictions = normalizer.inverse_transform_outputs(predictions)
        return predictions

    def _load_test_data(self):
        """Load test data using column names from the first loaded run's config.

        Defaults the test path to `<first_config.data.train_path>/test` when
        none was provided on the CLI. Returns (inputs, outputs, states) or
        (None, None, None) on failure.
        """
        ref_config = self._first_config()
        if ref_config is None:
            logger.warning("No configs loaded — can't determine column names.")
            return None, None, None

        if self.test_data_path is None:
            data_base = Path(ref_config.data.train_path).expanduser()
            test_path = data_base / "test"
            logger.info(f"--test-data not provided, defaulting to {test_path}")
        else:
            test_path = Path(self.test_data_path).expanduser()

        if not test_path.is_dir():
            logger.error(f"Test data path is not a directory: {test_path}")
            return None, None, None

        state_col = getattr(ref_config.data, "state_col", None)
        if state_col and len(state_col) == 0:
            state_col = None
        try:
            inputs, outputs, states, filenames = load_csv_folder(
                folder_path=str(test_path),
                input_col=ref_config.data.input_col,
                output_col=ref_config.data.output_col,
                state_col=state_col,
                pattern=getattr(ref_config.data, "pattern", "*.csv"),
            )
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return None, None, None

        logger.info(f"Loaded {len(filenames)} files from {test_path}")
        if states is not None:
            logger.info(f"  states: {states.shape}")
        return inputs, outputs, states

    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate all models on test data and return comparison table.

        Each model is fed through its own normalizer so predictions are
        computed (and compared to ground truth) in physical units.
        """
        if not self.models:
            logger.warning("No models loaded, skipping evaluation")
            return pd.DataFrame()

        logger.info("Evaluating models on test data...")
        test_inputs, test_outputs, test_states = self._load_test_data()
        if test_inputs is None:
            return pd.DataFrame()

        results = []
        for run_id, model in self.models.items():
            try:
                run_info = next(r for r in self.runs_info if r["run_id"] == run_id)
                predictions = self._predict(run_id, test_inputs, test_states)
                metrics = compute_metrics(predictions, test_outputs)

                result = {
                    "run_id": run_id[:8],
                    "run_name": run_info["run_name"],
                    "model_type": run_info["params"].get("model_type", "unknown"),
                    "hidden_size": run_info["params"].get("hidden_size", "N/A"),
                    **metrics,
                }

                if isinstance(model, SimpleLure):
                    result["constraints_satisfied"] = model.check_constraints()
                    result["alpha"] = model.alpha.item()
                    result["s"] = model.s.item()

                results.append(result)
                self.metrics[run_id] = metrics
                logger.info(f"✓ Evaluated run {run_id[:8]}")
            except Exception as e:
                logger.error(f"Failed to evaluate run {run_id}: {e}")

        return pd.DataFrame(results)

    def plot_training_curves(self, output_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot training and validation loss curves for all runs.

        Args:
            output_path: Path to save the plot. If None, only returns the figure.

        Returns:
            Matplotlib figure
        """
        if not self.histories:
            logger.warning("No training histories available")
            return None

        logger.info("Plotting training curves...")

        # Determine subplot layout
        n_runs = len(self.histories)
        n_cols = min(2, n_runs)
        n_rows = (n_runs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        if n_runs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_runs > 1 else [axes]

        for idx, (run_id, history) in enumerate(self.histories.items()):
            ax = axes[idx]
            run_info = next(r for r in self.runs_info if r["run_id"] == run_id)

            epochs = np.arange(1, len(history["train_losses"]) + 1)

            # Plot losses
            ax.plot(epochs, history["train_losses"], label="Train Loss", linewidth=2)
            ax.plot(epochs, history["val_losses"], label="Val Loss", linewidth=2)

            # Plot prediction and regularization losses if available
            if history.get("train_pred_losses") is not None:
                ax.plot(
                    epochs,
                    history["train_pred_losses"],
                    label="Train Pred Loss",
                    linestyle="--",
                    alpha=0.7,
                )
            if history.get("train_reg_losses") is not None:
                ax.plot(
                    epochs,
                    history["train_reg_losses"],
                    label="Train Reg Loss",
                    linestyle="--",
                    alpha=0.7,
                )

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title(f"{run_info['run_name']}\n(Run ID: {run_id[:8]})", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

        # Hide unused subplots
        for idx in range(n_runs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved training curves to {output_path}")

        return fig

    def plot_validation_comparison(self, output_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot validation loss curves for all runs on a single plot.

        Args:
            output_path: Path to save the plot. If None, only returns the figure.

        Returns:
            Matplotlib figure
        """
        if not self.histories:
            logger.warning("No training histories available")
            return None

        logger.info("Plotting validation loss comparison...")

        fig, ax = plt.subplots(figsize=(10, 6))

        for run_id, history in self.histories.items():
            run_info = next(r for r in self.runs_info if r["run_id"] == run_id)
            epochs = np.arange(1, len(history["val_losses"]) + 1)

            ax.plot(
                epochs,
                history["val_losses"],
                label=f"{run_info['run_name']} ({run_id[:8]})",
                linewidth=2,
                marker="o",
                markersize=3,
                markevery=max(1, len(epochs) // 20),
            )

        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Validation Loss", fontsize=14)
        ax.set_title("Validation Loss Comparison", fontsize=16, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved validation comparison to {output_path}")
            try:
                tikzplotlib.save(output_path.with_suffix(".tex"), figure=fig)
            except Exception as e:
                logger.warning(f"Could not save TikZ plot: {e}")

        return fig

    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table with run parameters and final metrics.

        Returns:
            DataFrame with summary information
        """
        logger.info("Creating summary table...")

        summary_data = []
        for run_info in self.runs_info:
            run_id = run_info["run_id"]

            row = {
                "Run ID": run_id[:8],
                "Name": run_info["run_name"],
                "Model Type": run_info["params"].get("model_type", "N/A"),
                "Hidden Size": run_info["params"].get("hidden_size", "N/A"),
                "Learning Rate": run_info["params"].get("learning_rate", "N/A"),
                "Batch Size": run_info["params"].get("batch_size", "N/A"),
                "Epochs": run_info["params"].get("num_epochs", "N/A"),
            }

            # Add final validation loss from history
            if run_id in self.histories:
                row["Final Val Loss"] = f"{self.histories[run_id]['val_losses'][-1]:.6f}"

            # Add test metrics if available
            if run_id in self.metrics:
                for key, value in self.metrics[run_id].items():
                    if key.startswith("test_"):
                        metric_name = key.replace("test_", "").upper()
                        row[metric_name] = f"{value:.6f}"

            summary_data.append(row)

        return pd.DataFrame(summary_data)

    def plot_trajectory_comparison(
        self,
        test_inputs: np.ndarray,
        test_outputs: np.ndarray,
        test_states: Optional[np.ndarray],
        num_samples: int = 3,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot comparison of ground truth vs predictions for random trajectories.

        Args:
            test_inputs: Test input data
            test_outputs: Test output data (ground truth)
            test_states: Test state data (for initial conditions)
            num_samples: Number of random trajectories to plot (default: 5)
            output_path: Path to save the plot. If None, only returns the figure.

        Returns:
            Matplotlib figure
        """
        if not self.models:
            logger.warning("No models loaded, skipping trajectory comparison")
            return None

        logger.info(f"Plotting trajectory comparison for {num_samples} random samples...")

        # Select random sample indices
        # num_sequences = test_inputs.shape[0]
        # if num_sequences < num_samples:
        #     sample_indices = list(range(num_sequences))
        # else:
        #     sample_indices = np.random.choice(num_sequences, size=num_samples, replace=False).tolist()
        sample_indices = UNSTAB_STAB_ZERO

        # Get predictions from all models (each through its own normalizer).
        predictions_dict = {
            run_id: self._predict(run_id, test_inputs, test_states)
            for run_id in self.models
        }

        # Determine output dimensions
        output_dim = test_outputs.shape[2]

        # Create subplots
        fig, axes = plt.subplots(num_samples, output_dim, figsize=(7 * output_dim, 4 * num_samples))
        if num_samples == 1 and output_dim == 1:
            axes = np.array([[axes]])
        elif num_samples == 1:
            axes = axes.reshape(1, -1)
        elif output_dim == 1:
            axes = axes.reshape(-1, 1)

        # Plot each sample
        for sample_idx, seq_idx in enumerate(sample_indices):
            for dim in range(output_dim):
                ax = axes[sample_idx, dim]

                # Time steps
                time_steps = np.arange(test_outputs.shape[1])

                # Plot ground truth (dashed line)
                ax.plot(
                    time_steps,
                    test_outputs[seq_idx, :, dim],
                    "k--",
                    linewidth=2,
                    label="Ground Truth",
                    alpha=0.7,
                )

                # Plot predictions from each model
                for run_id, predictions in predictions_dict.items():
                    short_id = run_id[:8]
                    ax.plot(
                        time_steps,
                        predictions[seq_idx, :, dim],
                        linewidth=1.5,
                        label=short_id,
                        alpha=0.8,
                    )

                ax.set_xlabel("Time Step", fontsize=11)
                ax.set_ylabel(f"Output {dim+1}", fontsize=11)
                ax.set_title(f"Sequence {seq_idx}, Output Dim {dim+1}", fontsize=12)
                ax.grid(True, alpha=0.3)

                # Only show legend on first subplot
                if sample_idx == 0 and dim == 0:
                    ax.legend(fontsize=9, loc="best")

        plt.suptitle(
            "Trajectory Comparison: Ground Truth vs Model Predictions",
            fontsize=14,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved trajectory comparison to {output_path}")
            try:
                tikzplotlib.save(output_path.with_suffix(".tex"), figure=fig)
            except Exception as e:
                logger.warning(f"Could not save TikZ plot: {e}")

        return fig

    def plot_error_comparison(
        self,
        test_inputs: np.ndarray,
        test_outputs: np.ndarray,
        test_states: Optional[np.ndarray],
        num_samples: int = 3,
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot prediction errors over time for random trajectories.

        Args:
            test_inputs: Test input data
            test_outputs: Test output data (ground truth)
            test_states: Test state data (for initial conditions)
            num_samples: Number of random trajectories to plot (default: 3)
            output_path: Path to save the plot. If None, only returns the figure.

        Returns:
            Matplotlib figure
        """
        if not self.models:
            logger.warning("No models loaded, skipping error comparison")
            return None

        logger.info(f"Plotting error comparison for {num_samples} random samples...")

        # Select random sample indices
        num_sequences = test_inputs.shape[0]
        if num_sequences < num_samples:
            sample_indices = list(range(num_sequences))
        else:
            sample_indices = np.random.choice(
                num_sequences, size=num_samples, replace=False
            ).tolist()

        # Get predictions from all models (each through its own normalizer).
        predictions_dict = {
            run_id: self._predict(run_id, test_inputs, test_states)
            for run_id in self.models
        }

        # Determine output dimensions
        output_dim = test_outputs.shape[2]

        # Create subplots
        fig, axes = plt.subplots(num_samples, output_dim, figsize=(7 * output_dim, 4 * num_samples))
        if num_samples == 1 and output_dim == 1:
            axes = np.array([[axes]])
        elif num_samples == 1:
            axes = axes.reshape(1, -1)
        elif output_dim == 1:
            axes = axes.reshape(-1, 1)

        # Plot each sample
        for sample_idx, seq_idx in enumerate(sample_indices):
            for dim in range(output_dim):
                ax = axes[sample_idx, dim]

                # Time steps
                time_steps = np.arange(test_outputs.shape[1])

                # Plot absolute errors from each model
                for run_id, predictions in predictions_dict.items():
                    short_id = run_id[:8]
                    error = np.abs(predictions[seq_idx, :, dim] - test_outputs[seq_idx, :, dim])
                    ax.plot(time_steps, error, linewidth=1.5, label=short_id, alpha=0.8)

                ax.set_xlabel("Time Step", fontsize=11)
                ax.set_ylabel(f"Absolute Error (Output {dim+1})", fontsize=11)
                ax.set_title(f"Sequence {seq_idx}, Abs Error Dim {dim+1}", fontsize=12)
                ax.grid(True, alpha=0.3)

                # Only show legend on first subplot
                if sample_idx == 0 and dim == 0:
                    ax.legend(fontsize=9, loc="best")

        plt.suptitle(
            "Absolute Prediction Error Comparison Over Time", fontsize=14, fontweight="bold", y=1.00
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved error comparison to {output_path}")
            try:
                tikzplotlib.save(output_path.with_suffix(".tex"), figure=fig)
            except Exception as e:
                logger.warning(f"Could not save TikZ plot: {e}")

        return fig

    def generate_report(self, output_dir: Path):
        """
        Generate a complete comparison report with tables and plots.

        Args:
            output_dir: Directory to save the report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating comparison report in {output_dir}")

        # 1. Summary table
        summary_df = self.create_summary_table()
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Saved summary table to {summary_path}")

        # Print summary to console
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80 + "\n")

        # Load test data for trajectory comparison (uses column names from the
        # first loaded run's config).
        test_inputs, test_outputs, test_states = self._load_test_data()

        # 2. Evaluation metrics table
        eval_df = self.evaluate_models()
        if not eval_df.empty:
            eval_path = output_dir / "evaluation_metrics.csv"
            eval_df.to_csv(eval_path, index=False)
            logger.info(f"✓ Saved evaluation metrics to {eval_path}")

            print("\n" + "=" * 80)
            print("EVALUATION METRICS")
            print("=" * 80)
            print(eval_df.to_string(index=False))
            print("=" * 80 + "\n")

        # 3. Trajectory comparison plot
        if test_inputs is not None and test_outputs is not None:
            fig_traj = self.plot_trajectory_comparison(
                test_inputs,
                test_outputs,
                test_states,
                num_samples=3,
                output_path=output_dir / "outputs.png",
            )
            if fig_traj:
                plt.close(fig_traj)

            # 3b. Error comparison plot
            fig_error = self.plot_error_comparison(
                test_inputs,
                test_outputs,
                test_states,
                num_samples=3,
                output_path=output_dir / "error.png",
            )
            if fig_error:
                plt.close(fig_error)

        # 4. Training curves (individual)
        fig_individual = self.plot_training_curves(output_dir / "training_curves.png")
        if fig_individual:
            plt.close(fig_individual)

        # 5. Validation comparison (combined)
        fig_comparison = self.plot_validation_comparison(output_dir / "validation-loss.png")
        if fig_comparison:
            plt.close(fig_comparison)

        logger.info(f"✓ Report generation complete! Results saved to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare multiple MLflow runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--run-ids", nargs="+", required=True, help="MLflow run IDs to compare")

    parser.add_argument(
        "--data-root", type=str, default=DEFAULT_DATA_ROOT,
        help=f"Base directory for per-run artefacts (default: {DEFAULT_DATA_ROOT}).",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data folder. Defaults to <config.data.train_path>/test "
             "of the first loaded run.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparisons",
        help="Directory to save comparison results (default: comparisons)",
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="Override MLflow tracking URI from the first run's config "
             "(default: use config).",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create comparator and load runs (configs + models + normalizers from disk).
    comparator = RunComparator(
        run_ids=args.run_ids,
        test_data_path=args.test_data,
        data_root=args.data_root,
    )
    comparator.load_runs()

    # Set up MLflow using the first loaded run's config (with optional CLI override),
    # then enrich each run with params/metrics from the server.
    ref_config = comparator._first_config()
    if ref_config is not None:
        setup_mlflow_tracking(ref_config, override_uri=args.mlflow_tracking_uri)
    elif args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    logger.info(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")
    comparator.fetch_mlflow_metadata()

    # Generate report
    output_dir = Path(args.output_dir)
    comparator.generate_report(output_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
