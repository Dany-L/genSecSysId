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
from sysid.data.direct_loader import load_csv_folder
from sysid.models import SimpleRNN, LSTM, GRU
from sysid.evaluation import Evaluator


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


def setup_logging(output_dir: Path, model_name: str) -> logging.Logger:
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model: {model_name}")
    
    return logger


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


def load_model(model_path: str, config: Config, device: str):
    """Load trained model."""
    model_config = config.model
    
    if model_config.model_type == "rnn":
        model = SimpleRNN(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            output_size=model_config.output_size,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
            activation=model_config.activation,
        )
    elif model_config.model_type == "lstm":
        model = LSTM(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            output_size=model_config.output_size,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
        )
    elif model_config.model_type == "gru":
        model = GRU(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            output_size=model_config.output_size,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_config.model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RNN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data (folder or CSV file)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Try to load run info from model directory
    run_info = load_run_info(args.model)
    
    # Setup logging (use run_id if available)
    if run_info:
        run_id = run_info['run_id']
        output_dir = Path(args.output_dir) / run_id
        logger = setup_logging(output_dir, f"eval_{run_id[:8]}")
        logger.info(f"Found MLflow run ID: {run_id}")
        logger.info(f"Will log evaluation results to the same run")
    else:
        output_dir = Path(args.output_dir)
        model_name = Path(args.model).stem
        logger = setup_logging(output_dir, model_name)
        logger.warning("No run_info.json found - evaluation results will not be logged to MLflow")
    
    logger.info("=" * 70)
    logger.info("Model Evaluation")
    logger.info("=" * 70)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Model checkpoint: {args.model}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        config = Config.from_yaml(args.config)
    elif config_path.suffix == ".json":
        config = Config.from_json(args.config)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
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
            # Primary method: Load directly from CSV folder
            logger.info("Loading directly from CSV folder...")
            logger.info(f"Test data directory: {args.test_data}")
            
            test_inputs, test_outputs, filenames = load_csv_folder(
                folder_path=str(test_path),
                input_col=getattr(config.data, 'input_col', 'd'),
                output_col=getattr(config.data, 'output_col', 'e'),
                pattern=getattr(config.data, 'pattern', '*.csv')
            )
            logger.info(f"Loaded {len(filenames)} files from test set")
            
        elif test_path.suffix == '.csv':
            # Fallback: Load from single CSV file
            logger.info("Loading from single CSV file...")
            test_inputs, test_outputs = DataLoader.load_from_csv(args.test_data, delimiter=",")
            
        elif test_path.suffix == '.npy':
            # Legacy: Load from NPY files
            logger.info("Loading NPY files...")
            test_inputs, test_outputs = DataLoader.load_from_npy(
                args.test_data,
                args.test_data.replace('_inputs.npy', '_outputs.npy')
            )
        else:
            raise ValueError(
                f"Unsupported data format: {args.test_data}\n"
                f"Use either:\n"
                f"  1. Folder path (recommended): 'data/prepared/test' with CSV files\n"
                f"  2. Single CSV file: 'data/test.csv'\n"
                f"  3. NPY file: 'data/test_inputs.npy'"
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
        # Fallback to config model_dir
        normalizer_path_fallback = Path(config.model_dir) / "normalizer.json"
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
                
                # Load predictions and targets from saved files
                try:
                    predictions = np.load(output_dir / "predictions.npy")
                    targets = np.load(output_dir / "targets.npy")
                    
                    evaluator.plot_predictions(predictions, targets, num_samples=5)
                    evaluator.analyze_errors(predictions, targets)
                    logger.info("Plots generated successfully")
                except Exception as e:
                    logger.warning(f"Failed to generate plots: {e}")
                
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
            
            # Load predictions and targets from saved files
            try:
                predictions = np.load(output_dir / "predictions.npy")
                targets = np.load(output_dir / "targets.npy")
                
                evaluator.plot_predictions(predictions, targets, num_samples=5)
                evaluator.analyze_errors(predictions, targets)
                logger.info("Plots generated successfully")
            except Exception as e:
                logger.warning(f"Failed to generate plots: {e}")
        
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
