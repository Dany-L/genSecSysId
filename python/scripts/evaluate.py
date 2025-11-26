"""Evaluation script."""

import argparse
from pathlib import Path
import logging
import sys
from datetime import datetime
import json
import numpy as np
import torch
import mlflow

from sysid.config import Config
from sysid.data import create_dataloaders, DataLoader, DataNormalizer
from sysid.data.direct_loader import load_csv_folder, load_split_data
from sysid.models import load_model
from sysid.evaluation import Evaluator
from sysid.utils import plot_predictions


def setup_console_logging() -> logging.Logger:
    """Setup console-only logging (before run_id is known)."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    return logging.getLogger(__name__)


def setup_file_logging(log_dir: Path, log_prefix: str) -> logging.Logger:
    """Setup logging with both file and console output (after run_id is known)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = log_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file_path}")
    
    return logger


def filter_metrics(metrics: dict, allowed_metrics: list) -> dict:
    """
    Filter metrics to only include those in the allowed list.
    
    Args:
        metrics: Dictionary of all computed metrics
        allowed_metrics: List of metric names to keep
        
    Returns:
        Filtered metrics dictionary
    """
    filtered = {}
    
    for key, value in metrics.items():
        # Always skip per_step (internal detail, not logged)
        if key == "per_step":
            continue
        
        # If no filtering specified, include all metrics (except per_step)
        if allowed_metrics is None:
            filtered[key] = value
            continue
            
        # For suffixed metrics (_avg, _final), extract base metric name
        if '_avg' in key:
            base_metric = key.replace('_avg', '')
        elif '_final' in key:
            base_metric = key.replace('_final', '')
        else:
            base_metric = key
        
        # Include metric if base is in allowed list
        if base_metric in allowed_metrics:
            filtered[key] = value
    
    return filtered


def load_run_info(model_path: str):
    """
    Load MLflow run info from model directory.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        dict: Run info containing run_id, experiment_name, run_name, or None if not found
    """
    model_dir = Path(model_path).parent
    run_info_path = model_dir / "run_info.json"
    
    if run_info_path.exists():
        with open(run_info_path, 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RNN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data (folder or CSV file)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Load configuration early so we can derive directories from config.root_dir
    config_path = Path(args.config)
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        config = Config.from_yaml(args.config)
    elif config_path.suffix == ".json":
        config = Config.from_json(args.config)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    # Try to load run info from model directory
    run_info = load_run_info(args.model)

    # Setup console-only logging initially
    logger = setup_console_logging()

    # If a root_dir is configured, derive output/log dirs from it
    if getattr(config, 'root_dir', None):
        base = Path(config.root_dir)
        model_type = config.model.model_type
        if run_info:
            run_id = run_info['run_id']
            output_dir = base / "outputs" / model_type / run_id
            log_dir = base / "logs" / model_type / run_id
            # Setup file logging now that we have the log directory
            logger = setup_file_logging(log_dir, "evaluation")
            logger.info(f"Found MLflow run ID: {run_id}")
            logger.info("Will log evaluation results to the same run")
        else:
            # No run info - create timestamped folder under outputs/model_type
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = base / "outputs" / model_type / f"eval_{ts}"
            log_dir = base / "logs" / model_type / f"eval_{ts}"
            logger = setup_file_logging(log_dir, "evaluation")
            logger.warning("No run_info.json found - evaluation results will not be logged to MLflow")
    else:
        # Fallback: use CLI-provided output dir
        if run_info:
            run_id = run_info['run_id']
            output_dir = Path(args.output_dir) / run_id
            logger = setup_file_logging(output_dir, "evaluation")
            logger.info(f"Found MLflow run ID: {run_id}")
            logger.info(f"Will log evaluation results to the same run")
        else:
            output_dir = Path(args.output_dir)
            model_name = Path(args.model).stem
            logger = setup_file_logging(output_dir, "evaluation")
            logger.warning("No run_info.json found - evaluation results will not be logged to MLflow")
    
    logger.info("=" * 70)
    logger.info("Model Evaluation")
    logger.info("=" * 70)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Model checkpoint: {args.model}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output directory: {output_dir}")
    
    logger.info(f"Model type: {config.model.model_type}")
    logger.info(f"Evaluation metrics: {config.evaluation.metrics}")
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")
    
    # Load test data
    logger.info("Loading test data...")
    print("Loading test data...")
    try:
        # Auto-detect loading method based on path
        test_path = Path(args.test_data)
        
        if test_path.is_dir():
            # Check if it's a structured data directory (with train/test/validation subfolders)
            if (test_path / "test").exists():
                # Use load_split_data to load only test data from prepared structure
                logger.info("Loading from prepared data structure (test split only)...")
                logger.info(f"Data directory: {args.test_data}")
                
                # Get state column if provided
                state_col = getattr(config.data, 'state_col', None)
                if state_col and len(state_col) == 0:  # Empty list means no state
                    state_col = None
                
                # Load only test data
                _, _, _, _, test_inputs, test_outputs, _, _, test_states = load_split_data(
                    data_dir=str(test_path),
                    input_col=getattr(config.data, 'input_col', ['d']),
                    output_col=getattr(config.data, 'output_col', ['e']),
                    state_col=state_col,
                    pattern=getattr(config.data, 'pattern', '*.csv'),
                    load_train=False,  # Don't load training data
                    load_val=False,    # Don't load validation data
                    load_test=True     # Only load test data
                )
                logger.info(f"Loaded test data: {test_inputs.shape[0]} sequences")
                if test_states is not None:
                    logger.info(f"State information loaded: {test_states.shape}")
            else:
                # Single folder with CSV files (backward compatibility)
                logger.info("Loading directly from CSV folder...")
                logger.info(f"Test data directory: {args.test_data}")
                
                # Get state column if provided
                state_col = getattr(config.data, 'state_col', None)
                if state_col and len(state_col) == 0:  # Empty list means no state
                    state_col = None
                
                test_inputs, test_outputs, test_states, filenames = load_csv_folder(
                    folder_path=str(test_path),
                    input_col=getattr(config.data, 'input_col', ['d']),
                    output_col=getattr(config.data, 'output_col', ['e']),
                    state_col=state_col,
                    pattern=getattr(config.data, 'pattern', '*.csv')
                )
                logger.info(f"Loaded {len(filenames)} files from test set")
                if test_states is not None:
                    logger.info(f"State information loaded: {test_states.shape}")
            
        elif test_path.suffix == '.csv':
            # Fallback: Load from single CSV file
            logger.info("Loading from single CSV file...")
            test_inputs, test_outputs = DataLoader.load_from_csv(args.test_data, delimiter=",")
            test_states = None
            
        elif test_path.suffix == '.npy':
            # Legacy: Load from NPY files
            logger.info("Loading NPY files...")
            test_inputs, test_outputs = DataLoader.load_from_npy(
                args.test_data,
                args.test_data.replace('_inputs.npy', '_outputs.npy')
            )
            test_states = None
        else:
            raise ValueError(
                f"Unsupported data format: {args.test_data}\n"
                f"Use either:\n"
                f"  1. Prepared data folder: 'data/prepared' with test/ subfolder\n"
                f"  2. Test folder: 'data/prepared/test' with CSV files\n"
                f"  3. Single CSV file: 'data/test.csv'\n"
                f"  4. NPY file: 'data/test_inputs.npy'"
            )
        
        logger.info(f"Test data loaded: inputs={test_inputs.shape}, outputs={test_outputs.shape}")
        print(f"Test data: {test_inputs.shape}, {test_outputs.shape}")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        logger.exception("Full traceback:")
        raise
    
    # Load normalizer (check in model directory, which may include run_id)
    model_dir = Path(args.model).parent
    normalizer_path = model_dir / "normalizer.json"
    normalizer = None
    if normalizer_path.exists():
        normalizer = DataNormalizer.load(str(normalizer_path))
        logger.info(f"Loaded normalizer from {normalizer_path}")
        print(f"Loaded normalizer from {normalizer_path}")
    else:
        # Fallback to config model_dir. Also check derived root model folder if configured.
        normalizer_path_fallback = Path(config.model_dir) / "normalizer.json"
        if getattr(config, 'root_dir', None):
            derived = Path(config.root_dir) / "models" / config.model.model_type / "normalizer.json"
            if derived.exists():
                normalizer_path_fallback = derived
        if normalizer_path_fallback.exists():
            normalizer = DataNormalizer.load(str(normalizer_path_fallback))
            logger.info(f"Loaded normalizer from {normalizer_path_fallback}")
            print(f"Loaded normalizer from {normalizer_path_fallback}")
        else:
            logger.warning(f"Normalizer not found at {normalizer_path} or {normalizer_path_fallback}")
    
    # Create test loader
    if normalizer is not None:
        test_inputs_norm = normalizer.transform_inputs(test_inputs)
        test_outputs_norm = normalizer.transform_outputs(test_outputs)
    else:
        test_inputs_norm = test_inputs
        test_outputs_norm = test_outputs
    
    from sysid.data import TimeSeriesDataset
    from torch.utils.data import DataLoader as TorchDataLoader
    
    test_dataset = TimeSeriesDataset(
        test_inputs_norm,
        test_outputs_norm,
        test_states,
        sequence_length=config.data.sequence_length,
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
    )
    
    # Load model
    logger.info("Loading model...")
    print("Loading model...")
    try:
        model = load_model(args.model, config, str(device))
        logger.info(f"Model loaded from {args.model}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        print(f"Model loaded from {args.model}")
        
        # Generate ellipse/polytope plot for SimpleLure models
        from sysid.models import SimpleLure
        from sysid.utils import plot_ellipse_and_parallelogram
        import matplotlib.pyplot as plt
        
        if isinstance(model, SimpleLure) and model.nx == 2:
            logger.info("Generating ellipse and polytope visualization...")
            try:
                X = np.linalg.inv(model.P.cpu().detach().numpy())
                H = model.L.cpu().detach().numpy() @ X
                s = model.s.cpu().detach().numpy()
                max_norm_x0 = model.max_norm_x0 if hasattr(model, 'max_norm_x0') else None
                
                fig, ax = plot_ellipse_and_parallelogram(X, H, s, max_norm_x0, show=False)
                
                # Save to output directory
                ellipse_plot_path = output_dir / "ellipse_polytope.png"
                fig.savefig(ellipse_plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Ellipse/polytope plot saved to {ellipse_plot_path}")
            except Exception as e:
                logger.warning(f"Failed to generate ellipse/polytope plot: {e}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=str(device),
        output_dir=str(output_dir),
    )
    
    # Evaluate (with MLflow logging if run_info exists)
    logger.info("=" * 70)
    logger.info("Starting evaluation...")
    logger.info(f"Test batches: {len(test_loader)}")
    logger.info("=" * 70)
    print("\nEvaluating model...")
    
    try:
        # Start MLflow run if we have run_info
        if run_info:
            # Reopen the training run to add evaluation metrics
            mlflow.set_experiment(run_info['experiment_name'])
            with mlflow.start_run(run_id=run_id):
                logger.info(f"Logging evaluation to MLflow run: {run_id}")
                
                results = evaluator.evaluate(
                    test_loader=test_loader,
                    normalizer=normalizer,
                )
                
                # Filter metrics based on config
                all_metrics = results.get("metrics", {})
                metrics = filter_metrics(all_metrics, config.evaluation.metrics)
                
                # Log evaluation metrics to MLflow
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and metric != "per_step":
                        mlflow.log_metric(f"eval_{metric}", value)
                        logger.info(f"{metric}: {value:.6f}")
                
                # Generate plots
                logger.info("Generating plots and analysis...")
                print("\nGenerating plots...")
                
                # Load predictions, targets, and inputs from saved files
                try:
                    e_hat = np.load(output_dir / "predictions.npy")  # predicted output
                    e = np.load(output_dir / "targets.npy")  # output (target)
                    d = np.load(output_dir / "inputs.npy")  # input
                    
                    # Select sample indices: always include sample 0, plus 4 random samples
                    num_sequences = e_hat.shape[0]
                    sample_indices = [0]  # Always include sample 0
                    
                    if num_sequences > 1:
                        # Select 4 random samples (excluding sample 0)
                        other_indices = list(range(1, num_sequences))
                        num_random = min(4, len(other_indices))
                        random_indices = np.random.choice(other_indices, size=num_random, replace=False).tolist()
                        sample_indices.extend(random_indices)

                    plot_predictions(evaluator.output_dir, e_hat, e, d, sample_indices=sample_indices)
                    # evaluator.plot_predictions(e_hat, e, d, sample_indices=sample_indices)
                    evaluator.analyze_errors(e_hat, e)
                    logger.info("Plots generated successfully")
                except Exception as e_err:
                    logger.warning(f"Failed to generate plots: {e_err}")
                
                # Log evaluation artifacts to MLflow
                mlflow.log_artifacts(str(output_dir), "evaluation")
                logger.info("Evaluation artifacts logged to MLflow")
        else:
            # No run_info - just evaluate without MLflow
            results = evaluator.evaluate(
                test_loader=test_loader,
                normalizer=normalizer,
            )
            
            # Filter metrics based on config
            all_metrics = results.get("metrics", {})
            metrics = filter_metrics(all_metrics, config.evaluation.metrics)
            
            # Log results to console/file
            logger.info("=" * 70)
            logger.info("Evaluation Results:")
            logger.info("=" * 70)
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.6f}")
            
            # Generate plots
            logger.info("Generating plots and analysis...")
            print("\nGenerating plots...")
            
            # Load predictions, targets, and inputs from saved files
            try:
                e_hat = np.load(output_dir / "predictions.npy")  # predicted output
                e = np.load(output_dir / "targets.npy")  # output (target)
                d = np.load(output_dir / "inputs.npy")  # input
                
                # Select sample indices: always include sample 0, plus 4 random samples
                num_sequences = e_hat.shape[0]
                sample_indices = [0]  # Always include sample 0
                
                if num_sequences > 1:
                    # Select 4 random samples (excluding sample 0)
                    other_indices = list(range(1, num_sequences))
                    num_random = min(4, len(other_indices))
                    random_indices = np.random.choice(other_indices, size=num_random, replace=False).tolist()
                    sample_indices.extend(random_indices)
                
                evaluator.plot_predictions(e_hat, e, d, sample_indices=sample_indices)
                evaluator.analyze_errors(e_hat, e)
                logger.info("Plots generated successfully")
            except Exception as e_err:
                logger.warning(f"Failed to generate plots: {e_err}")
        
        logger.info("Plots and analysis generated")
        logger.info(f"Predictions plot: {output_dir / 'predictions_plot.png'}")
        logger.info(f"Error analysis: {output_dir / 'error_analysis.png'}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        raise
    
    logger.info("=" * 70)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Metrics file: {output_dir / 'evaluation_results.json'}")
    
    if run_info:
        logger.info(f"Evaluation metrics logged to training run: {run_id}")
        print(f"\nEvaluation completed! Metrics logged to MLflow run {run_id[:8]}")
    else:
        print(f"\nEvaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
