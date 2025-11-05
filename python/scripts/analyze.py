"""Model analysis script."""

import argparse
from pathlib import Path
import logging
import sys
from datetime import datetime
import torch
import json
import numpy as np
import mlflow

from sysid.config import Config
from sysid.models import SimpleRNN, LSTM, GRU


def setup_logging(output_dir: Path, model_name: str) -> logging.Logger:
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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


def analyze_parameters(model, output_dir: Path, logger: logging.Logger):
    """Analyze model parameters."""
    param_dict = model.get_parameter_dict()
    
    logger.info("=" * 60)
    logger.info("Parameter Analysis")
    logger.info("=" * 60)
    
    print("=" * 60)
    print("Parameter Analysis")
    print("=" * 60)
    
    for name, stats in param_dict.items():
        msg = f"\n{name}:"
        print(msg)
        logger.info(msg.strip())
        
        for key, value in stats.items():
            if key != 'shape':
                msg = f"  {key}: {value:.6f}"
            else:
                msg = f"  {key}: {value}"
            print(msg)
            logger.info(msg)
    
    # Save parameter statistics
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / "parameter_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(param_dict, f, indent=2)
    
    logger.info(f"\nParameter statistics saved to {stats_file}")
    print(f"\nParameter statistics saved to {stats_file}")


def check_stability(model, logger: logging.Logger):
    """Check stability conditions on model parameters."""
    logger.info("\n" + "=" * 60)
    logger.info("Stability Analysis")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("Stability Analysis")
    print("=" * 60)
    
    stability_results = {}
    
    # For RNN models, check spectral radius of recurrent weights
    for name, param in model.named_parameters():
        if "rnn" in name.lower() or "lstm" in name.lower() or "gru" in name.lower():
            if "weight_hh" in name or "weight" in name:
                # Compute eigenvalues
                eigenvalues = torch.linalg.eigvals(param.data)
                spectral_radius = torch.max(torch.abs(eigenvalues)).item()
                
                stability_results[name] = {
                    "spectral_radius": spectral_radius,
                    "stable": spectral_radius < 1.0
                }
                
                logger.info(f"\n{name}:")
                logger.info(f"  Spectral radius: {spectral_radius:.6f}")
                
                print(f"\n{name}:")
                print(f"  Spectral radius: {spectral_radius:.6f}")
                
                if spectral_radius < 1.0:
                    logger.info(f"  ✓ Stable (spectral radius < 1)")
                    print(f"  ✓ Stable (spectral radius < 1)")
                else:
                    logger.warning(f"  ✗ Potentially unstable (spectral radius >= 1)")
                    print(f"  ✗ Potentially unstable (spectral radius >= 1)")
    
    return stability_results


def check_parameter_bounds(model, lower_bound: float = None, upper_bound: float = None, logger: logging.Logger = None):
    """Check if parameters are within specified bounds."""
    logger.info("\n" + "=" * 60)
    logger.info("Parameter Bounds Check")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("Parameter Bounds Check")
    print("=" * 60)
    
    violations = []
    
    for name, param in model.named_parameters():
        param_min = param.data.min().item()
        param_max = param.data.max().item()
        
        logger.info(f"\n{name}:")
        logger.info(f"  Min: {param_min:.6f}")
        logger.info(f"  Max: {param_max:.6f}")
        
        print(f"\n{name}:")
        print(f"  Min: {param_min:.6f}")
        print(f"  Max: {param_max:.6f}")
        
        if lower_bound is not None and param_min < lower_bound:
            violations.append((name, "lower", param_min, lower_bound))
            logger.warning(f"  ✗ Lower bound violation: {param_min:.6f} < {lower_bound:.6f}")
            print(f"  ✗ Lower bound violation: {param_min:.6f} < {lower_bound:.6f}")
        
        if upper_bound is not None and param_max > upper_bound:
            violations.append((name, "upper", param_max, upper_bound))
            logger.warning(f"  ✗ Upper bound violation: {param_max:.6f} > {upper_bound:.6f}")
            print(f"  ✗ Upper bound violation: {param_max:.6f} > {upper_bound:.6f}")
        
        if (lower_bound is None or param_min >= lower_bound) and \
           (upper_bound is None or param_max <= upper_bound):
            logger.info(f"  ✓ Within bounds")
            print(f"  ✓ Within bounds")
    
    if violations:
        logger.warning(f"\n⚠ Found {len(violations)} bound violations")
        print(f"\n⚠ Found {len(violations)} bound violations")
    else:
        logger.info(f"\n✓ All parameters within bounds")
        print(f"\n✓ All parameters within bounds")
    
    return violations


def main():
    parser = argparse.ArgumentParser(description="Analyze trained RNN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="analysis_results", help="Output directory")
    parser.add_argument("--check-stability", action="store_true", help="Check stability conditions")
    parser.add_argument("--lower-bound", type=float, default=None, help="Lower bound for parameters")
    parser.add_argument("--upper-bound", type=float, default=None, help="Upper bound for parameters")
    args = parser.parse_args()
    
    # Try to load run info from model directory
    run_info = load_run_info(args.model)
    
    # Setup logging (use run_id if available)
    if run_info:
        run_id = run_info['run_id']
        output_dir = Path(args.output_dir) / run_id
        logger = setup_logging(output_dir, f"analysis_{run_id[:8]}")
        logger.info(f"Found MLflow run ID: {run_id}")
        logger.info(f"Will log analysis results to the same run")
    else:
        output_dir = Path(args.output_dir)
        model_name = Path(args.model).stem
        logger = setup_logging(output_dir, model_name)
        logger.warning("No run_info.json found - analysis results will not be logged to MLflow")
    
    logger.info("=" * 70)
    logger.info("Model Analysis")
    logger.info("=" * 70)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Model checkpoint: {args.model}")
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
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    print("Loading model...")
    try:
        model = load_model(args.model, config, str(device))
        total_params = model.count_parameters()
        logger.info(f"Model loaded from {args.model}")
        logger.info(f"Total parameters: {total_params:,}")
        print(f"Model loaded from {args.model}")
        print(f"Total parameters: {total_params:,}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Start MLflow run if we have run_info
    if run_info:
        # Reopen the training run to add analysis metrics
        mlflow.set_experiment(run_info['experiment_name'])
        mlflow.start_run(run_id=run_id)
        logger.info(f"Logging analysis to MLflow run: {run_id}")
    
    try:
        # Analyze parameters
        logger.info("\nStarting parameter analysis...")
        analyze_parameters(model, output_dir, logger)
        
        # Load parameter statistics to log to MLflow
        param_stats_file = output_dir / 'parameter_statistics.json'
        if run_info and param_stats_file.exists():
            with open(param_stats_file, 'r') as f:
                param_stats = json.load(f)
                # Log key statistics as metrics
                if 'total_parameters' in param_stats:
                    mlflow.log_metric("analysis_total_params", param_stats['total_parameters'])
        
        # Check stability
        if args.check_stability:
            logger.info("\nPerforming stability analysis...")
            try:
                stability_results = check_stability(model, logger)
                
                # Save stability results
                stability_file = output_dir / "stability_check.json"
                with open(stability_file, "w") as f:
                    json.dump(stability_results, f, indent=2)
                logger.info(f"Stability results saved to {stability_file}")
                
                # Log stability metrics to MLflow
                if run_info:
                    for layer_name, results in stability_results.items():
                        if 'spectral_radius' in results:
                            mlflow.log_metric(f"stability_{layer_name}_spectral_radius", results['spectral_radius'])
                        if 'is_stable' in results:
                            mlflow.log_metric(f"stability_{layer_name}_is_stable", 1.0 if results['is_stable'] else 0.0)
            except Exception as e:
                logger.error(f"Stability check failed: {e}")
        
        # Check parameter bounds
        if args.lower_bound is not None or args.upper_bound is not None:
            logger.info(f"\nChecking parameter bounds (lower={args.lower_bound}, upper={args.upper_bound})...")
            try:
                violations = check_parameter_bounds(model, args.lower_bound, args.upper_bound, logger)
                
                # Save violations
                if violations:
                    violations_file = output_dir / "bound_violations.json"
                    with open(violations_file, "w") as f:
                        json.dump([
                            {
                                "parameter": v[0],
                                "type": v[1],
                                "value": v[2],
                                "bound": v[3],
                            }
                            for v in violations
                        ], f, indent=2)
                    logger.warning(f"Bound violations saved to {violations_file}")
                    
                    # Log violation count to MLflow
                    if run_info:
                        mlflow.log_metric("analysis_bound_violations", len(violations))
                else:
                    logger.info("No bound violations found")
                    if run_info:
                        mlflow.log_metric("analysis_bound_violations", 0)
            except Exception as e:
                logger.error(f"Bound check failed: {e}")
        
        # Log analysis artifacts to MLflow
        if run_info:
            mlflow.log_artifacts(str(output_dir), "analysis")
            logger.info("Analysis artifacts logged to MLflow")
    
    finally:
        # End MLflow run if we started one
        if run_info:
            mlflow.end_run()
    
    logger.info("=" * 70)
    logger.info("Analysis completed successfully!")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Parameter statistics: {output_dir / 'parameter_statistics.json'}")
    
    if run_info:
        logger.info(f"Analysis metrics logged to training run: {run_id}")
        print(f"\nAnalysis completed! Metrics logged to MLflow run {run_id[:8]}")
    else:
        print(f"\nAnalysis completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
