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
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
import warnings
import tikzplotlib

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
 
from sysid.data import DataLoader, DataNormalizer
from sysid.data.direct_loader import load_csv_folder, load_split_data
from sysid.models import SimpleLure
from sysid.evaluation import compute_metrics

torch.set_default_dtype(torch.float64)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class RunComparator:
    """Compare multiple MLflow runs."""
    
    def __init__(self, run_ids: List[str], test_data_path: Optional[str] = None):
        """
        Initialize the comparator.
        
        Args:
            run_ids: List of MLflow run IDs to compare
            test_data_path: Path to test data (folder or CSV). If None, uses data from run artifacts.
        """
        self.run_ids = run_ids
        self.test_data_path = test_data_path
        self.runs_info = []
        self.models = {}
        self.metrics = {}
        self.histories = {}
        
    def load_runs(self):
        """Load information, models, and metrics from all runs."""
        logger.info(f"Loading {len(self.run_ids)} MLflow runs...")
        
        for run_id in self.run_ids:
            try:
                run = mlflow.get_run(run_id)
                run_info = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', run_id[:8]),
                    'experiment_id': run.info.experiment_id,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'status': run.info.status,
                    'params': run.data.params,
                    'metrics': run.data.metrics,
                }
                self.runs_info.append(run_info)
                
                # Load model if available
                try:
                    model_uri = f"runs:/{run_id}/model"
                    model = mlflow.pytorch.load_model(model_uri)
                    self.models[run_id] = model
                    logger.info(f"✓ Loaded model for run {run_id[:8]}")
                except Exception as e:
                    logger.warning(f"Could not load model for run {run_id[:8]}: {e}")
                
                # Load training history if available
                try:
                    history_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path="outputs/training_history.json"
                    )
                    history = json.load(open(history_path, 'r'))
                    self.histories[run_id] = {
                        'train_losses': history['train_losses'],
                        'val_losses': history['val_losses'],
                        'train_pred_losses': history.get('train_pred_losses', None),
                        'train_reg_losses': history.get('train_reg_losses', None),
                    }
                    logger.info(f"✓ Loaded training history for run {run_id[:8]}")
                except Exception as e:
                    logger.warning(f"Could not load training history for run {run_id[:8]}: {e}")
                    
            except Exception as e:
                logger.error(f"Failed to load run {run_id}: {e}")
                
        logger.info(f"Successfully loaded {len(self.runs_info)} runs")
        
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate all models on test data and return comparison table.
        
        Returns:
            DataFrame with evaluation metrics for each run
        """
        if not self.models:
            logger.warning("No models loaded, skipping evaluation")
            return pd.DataFrame()
            
        logger.info("Evaluating models on test data...")
        
        # Load test data using the same approach as evaluate.py
        test_inputs = None
        test_outputs = None
        test_states = None
        
        if self.test_data_path:
            try:
                test_path = Path(self.test_data_path)
                
                if test_path.is_dir():
                    # Check if it's a structured data directory (with train/test/validation subfolders)

                    # Single folder with CSV files (backward compatibility)
                    logger.info("Loading directly from CSV folder...")
                    logger.info(f"Test data directory: {self.test_data_path}")
                    
                    test_inputs, test_outputs, test_states, filenames = load_csv_folder(
                        folder_path=str(test_path),
                        input_col=['d'],
                        output_col=['e'],
                        state_col=['x_1', 'x_2'],  # Adjust based on your data
                        pattern='*.csv'
                    )
                    logger.info(f"Loaded {len(filenames)} files from test set")
                    if test_states is not None:
                        logger.info(f"State information loaded: {test_states.shape}")
                else:
                    raise ValueError(
                        f"Unsupported data format: {self.test_data_path}\n"
                        f"Use either:\n"
                        f"  1. Prepared data folder: 'data/prepared' with test/ subfolder\n"
                        f"  2. Test folder: 'data/prepared/test' with CSV files\n"
                        f"  3. Single CSV file: 'data/test.csv'\n"
                        f"  4. NPY file: 'data/test_inputs.npy'"
                    )
                
                logger.info(f"Test data loaded: inputs={test_inputs.shape}, outputs={test_outputs.shape}")
                
            except Exception as e:
                logger.error(f"Failed to load test data: {e}")
                logger.exception("Full traceback:")
                return pd.DataFrame()
        
        # Evaluate each model
        results = []
        for run_id, model in self.models.items():
            try:
                run_info = next(r for r in self.runs_info if r['run_id'] == run_id)
                
                # Set model to evaluation mode
                model.eval()
                with torch.no_grad():
                    # Convert to tensors
                    inputs_tensor = torch.Tensor(test_inputs)
                    outputs_tensor = torch.Tensor(test_outputs)
                    
                    # Get initial states
                    if test_states is not None:
                        states_tensor = torch.Tensor(test_states)
                        x0 = states_tensor[:, 0, :]
                    else:
                        x0 = None
                    
                    # Forward pass
                    predictions = model(inputs_tensor, x0=x0)
                    
                    # Compute metrics
                    metrics = compute_metrics(
                        predictions.numpy(),
                        test_outputs,
                    )
                    
                    # Add run information
                    result = {
                        'run_id': run_id[:8],
                        'run_name': run_info['run_name'],
                        'model_type': run_info['params'].get('model_type', 'unknown'),
                        'hidden_size': run_info['params'].get('hidden_size', 'N/A'),
                        **metrics
                    }
                    
                    # Add stability info for SimpleLure models
                    if isinstance(model, SimpleLure):
                        result['constraints_satisfied'] = model.check_constraints()
                        result['alpha'] = model.alpha.item()
                        result['s'] = model.s.item()
                    
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
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        if n_runs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_runs > 1 else [axes]
        
        for idx, (run_id, history) in enumerate(self.histories.items()):
            ax = axes[idx]
            run_info = next(r for r in self.runs_info if r['run_id'] == run_id)
            
            epochs = np.arange(1, len(history['train_losses']) + 1)
            
            # Plot losses
            ax.plot(epochs, history['train_losses'], label='Train Loss', linewidth=2)
            ax.plot(epochs, history['val_losses'], label='Val Loss', linewidth=2)
            
            # Plot prediction and regularization losses if available
            if history.get('train_pred_losses') is not None:
                ax.plot(epochs, history['train_pred_losses'], 
                       label='Train Pred Loss', linestyle='--', alpha=0.7)
            if history.get('train_reg_losses') is not None:
                ax.plot(epochs, history['train_reg_losses'], 
                       label='Train Reg Loss', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f"{run_info['run_name']}\n(Run ID: {run_id[:8]})", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # Hide unused subplots
        for idx in range(n_runs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
            run_info = next(r for r in self.runs_info if r['run_id'] == run_id)
            epochs = np.arange(1, len(history['val_losses']) + 1)
            
            ax.plot(epochs, history['val_losses'], 
                   label=f"{run_info['run_name']} ({run_id[:8]})", 
                   linewidth=2, marker='o', markersize=3, markevery=max(1, len(epochs)//20))
        
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Validation Loss', fontsize=14)
        ax.set_title('Validation Loss Comparison', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved validation comparison to {output_path}")
            try:
                tikzplotlib.save(output_path.with_suffix('.tex'), figure=fig)
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
            run_id = run_info['run_id']
            
            row = {
                'Run ID': run_id[:8],
                'Name': run_info['run_name'],
                'Model Type': run_info['params'].get('model_type', 'N/A'),
                'Hidden Size': run_info['params'].get('hidden_size', 'N/A'),
                'Learning Rate': run_info['params'].get('learning_rate', 'N/A'),
                'Batch Size': run_info['params'].get('batch_size', 'N/A'),
                'Epochs': run_info['params'].get('num_epochs', 'N/A'),
            }
            
            # Add final validation loss from history
            if run_id in self.histories:
                row['Final Val Loss'] = f"{self.histories[run_id]['val_losses'][-1]:.6f}"
            
            # Add test metrics if available
            if run_id in self.metrics:
                for key, value in self.metrics[run_id].items():
                    if key.startswith('test_'):
                        metric_name = key.replace('test_', '').upper()
                        row[metric_name] = f"{value:.6f}"
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def plot_trajectory_comparison(self, test_inputs: np.ndarray, test_outputs: np.ndarray, 
                                   test_states: Optional[np.ndarray], num_samples: int = 3,
                                   output_path: Optional[Path] = None) -> plt.Figure:
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
        num_sequences = test_inputs.shape[0]
        if num_sequences < num_samples:
            sample_indices = list(range(num_sequences))
        else:
            sample_indices = np.random.choice(num_sequences, size=num_samples, replace=False).tolist()
        
        # Get predictions from all models
        predictions_dict = {}
        for run_id, model in self.models.items():
            model.eval()
            with torch.no_grad():
                inputs_tensor = torch.Tensor(test_inputs)
                
                if test_states is not None:
                    states_tensor = torch.Tensor(test_states)
                    x0 = states_tensor[:, 0, :]
                else:
                    x0 = None
                
                predictions = model(inputs_tensor, x0=x0)
                predictions_dict[run_id] = predictions.numpy()
        
        # Determine output dimensions
        output_dim = test_outputs.shape[2]
        
        # Create subplots
        fig, axes = plt.subplots(num_samples, output_dim, figsize=(7*output_dim, 4*num_samples))
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
                ax.plot(time_steps, test_outputs[seq_idx, :, dim], 'k--', 
                       linewidth=2, label='Ground Truth', alpha=0.7)
                
                # Plot predictions from each model
                for run_id, predictions in predictions_dict.items():
                    run_info = next(r for r in self.runs_info if r['run_id'] == run_id)
                    short_id = run_id[:8]
                    ax.plot(time_steps, predictions[seq_idx, :, dim], 
                           linewidth=1.5, label=short_id, alpha=0.8)
                
                ax.set_xlabel('Time Step', fontsize=11)
                ax.set_ylabel(f'Output {dim+1}', fontsize=11)
                ax.set_title(f'Sequence {seq_idx}, Output Dim {dim+1}', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Only show legend on first subplot
                if sample_idx == 0 and dim == 0:
                    ax.legend(fontsize=9, loc='best')
        
        plt.suptitle('Trajectory Comparison: Ground Truth vs Model Predictions', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trajectory comparison to {output_path}")
            try:
                tikzplotlib.save(output_path.with_suffix('.tex'), figure=fig)
            except Exception as e:
                logger.warning(f"Could not save TikZ plot: {e}")
        
        return fig
    
    def plot_error_comparison(self, test_inputs: np.ndarray, test_outputs: np.ndarray, 
                             test_states: Optional[np.ndarray], num_samples: int = 3,
                             output_path: Optional[Path] = None) -> plt.Figure:
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
            sample_indices = np.random.choice(num_sequences, size=num_samples, replace=False).tolist()
        
        # Get predictions from all models
        predictions_dict = {}
        for run_id, model in self.models.items():
            model.eval()
            with torch.no_grad():
                inputs_tensor = torch.Tensor(test_inputs)
                
                if test_states is not None:
                    states_tensor = torch.Tensor(test_states)
                    x0 = states_tensor[:, 0, :]
                else:
                    x0 = None
                
                predictions = model(inputs_tensor, x0=x0)
                predictions_dict[run_id] = predictions.numpy()
        
        # Determine output dimensions
        output_dim = test_outputs.shape[2]
        
        # Create subplots
        fig, axes = plt.subplots(num_samples, output_dim, figsize=(7*output_dim, 4*num_samples))
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
                
                ax.set_xlabel('Time Step', fontsize=11)
                ax.set_ylabel(f'Absolute Error (Output {dim+1})', fontsize=11)
                ax.set_title(f'Sequence {seq_idx}, Abs Error Dim {dim+1}', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Only show legend on first subplot
                if sample_idx == 0 and dim == 0:
                    ax.legend(fontsize=9, loc='best')
        
        plt.suptitle('Absolute Prediction Error Comparison Over Time', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved error comparison to {output_path}")
            try:
                tikzplotlib.save(output_path.with_suffix('.tex'), figure=fig)
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
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80 + "\n")
        
        # Load test data for trajectory comparison
        test_inputs = None
        test_outputs = None
        test_states = None
        
        if self.test_data_path:
            try:
                test_path = Path(self.test_data_path)
                
                if test_path.is_dir():
                    logger.info("Loading test data for trajectory comparison...")
                    test_inputs, test_outputs, test_states, filenames = load_csv_folder(
                        folder_path=str(test_path),
                        input_col=['d'],
                        output_col=['e'],
                        state_col=['x_1', 'x_2'],
                        pattern='*.csv'
                    )
                    logger.info(f"Loaded test data: {test_inputs.shape}")
            except Exception as e:
                logger.warning(f"Could not load test data for trajectory comparison: {e}")
        
        # 2. Evaluation metrics table
        eval_df = self.evaluate_models()
        if not eval_df.empty:
            eval_path = output_dir / "evaluation_metrics.csv"
            eval_df.to_csv(eval_path, index=False)
            logger.info(f"✓ Saved evaluation metrics to {eval_path}")
            
            print("\n" + "="*80)
            print("EVALUATION METRICS")
            print("="*80)
            print(eval_df.to_string(index=False))
            print("="*80 + "\n")
        
        # 3. Trajectory comparison plot
        if test_inputs is not None and test_outputs is not None:
            fig_traj = self.plot_trajectory_comparison(
                test_inputs, test_outputs, test_states,
                num_samples=3,
                output_path=output_dir / "outputs.png"
            )
            if fig_traj:
                plt.close(fig_traj)
            
            # 3b. Error comparison plot
            fig_error = self.plot_error_comparison(
                test_inputs, test_outputs, test_states,
                num_samples=3,
                output_path=output_dir / "error.png"
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
        epilog=__doc__
    )
    
    parser.add_argument(
        '--run-ids',
        nargs='+',
        required=True,
        help='MLflow run IDs to compare'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test data (folder or CSV). If not provided, uses data from run artifacts.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='comparisons',
        help='Directory to save comparison results (default: comparisons)'
    )
    
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: uses current tracking URI)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    logger.info(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Create comparator
    comparator = RunComparator(
        run_ids=args.run_ids,
        test_data_path=args.test_data
    )
    
    # Load runs
    comparator.load_runs()
    
    # Generate report
    output_dir = Path(args.output_dir)
    comparator.generate_report(output_dir)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
