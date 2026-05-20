"""Evaluation script.

Usage:
    python scripts/evaluate.py --run-id <mlflow_run_id> [--test-data <path>]

Config, checkpoint, and normalizer are all resolved from the standard
training layout under --data-root (default: ~/genSecSysId-Data). Test
data defaults to the test/ split of the dataset the run was trained on
(config.data.train_path) unless overridden with --test-data.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch

from sysid.config import resolve_run_artifacts, setup_mlflow_tracking
from sysid.data import DataLoader, DataNormalizer
from sysid.data.direct_loader import load_csv_folder, load_split_data
from sysid.data.loader import collate_with_optional_states
from sysid.evaluation import Evaluator
from sysid.models import load_model
from sysid.utils import plot_predictions


DEFAULT_DATA_ROOT = "~/genSecSysId-Data"


def setup_console_logging() -> logging.Logger:
    """Setup console-only logging (before run_id-based dirs exist)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger(__name__)


def setup_file_logging(log_dir: Path, log_prefix: str) -> logging.Logger:
    """Setup logging with both file and console output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = log_dir / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file_path}")

    return logger


def filter_metrics(metrics: dict, allowed_metrics: list) -> dict:
    """Filter metrics to only include those in the allowed list."""
    filtered = {}
    for key, value in metrics.items():
        if key == "per_step":
            continue
        if allowed_metrics is None:
            filtered[key] = value
            continue
        if "_avg" in key:
            base_metric = key.replace("_avg", "")
        elif "_final" in key:
            base_metric = key.replace("_final", "")
        else:
            base_metric = key
        if base_metric in allowed_metrics:
            filtered[key] = value
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RNN model")
    parser.add_argument(
        "--run-id", type=str, required=True,
        help="MLflow training-run id. Config, checkpoint, and normalizer are "
             "resolved from <data-root>/{outputs,models}/<model_type>/<run_id>/.",
    )
    parser.add_argument(
        "--data-root", type=str, default=DEFAULT_DATA_ROOT,
        help=f"Base directory for run artefacts (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--test-data", type=str, default=None,
        help="Path to test data (folder with test/ subfolder, folder of CSVs, "
             "or single .csv/.npy file). Defaults to config.data.train_path.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Initial console-only logger (used until log_dir is known).
    console_logger = setup_console_logging()

    # Resolve config + checkpoint + normalizer + run_info from the run id alone.
    try:
        config, model_path, normalizer_path, run_info = resolve_run_artifacts(
            args.run_id, data_root=args.data_root
        )
    except Exception as e:
        console_logger.error(f"Failed to resolve run_id={args.run_id}: {e}")
        raise

    # Determine output/log directories — prefer config.root_dir, fall back to data-root.
    if getattr(config, "root_dir", None):
        base = Path(os.path.expanduser(config.root_dir))
    else:
        base = Path(os.path.expanduser(args.data_root))
    model_type = config.model.model_type
    output_dir = base / "outputs" / model_type / args.run_id
    log_dir = base / "logs" / model_type / args.run_id

    # Now switch to file+console logging in the run directory.
    logger = setup_file_logging(log_dir, "evaluation")

    logger.info("=" * 70)
    logger.info("Model Evaluation")
    logger.info("=" * 70)
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Model checkpoint: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model type: {config.model.model_type}")
    logger.info(f"Evaluation metrics: {config.evaluation.metrics}")
    if run_info is None:
        logger.warning("No run_info.json found alongside the checkpoint")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")

    # Resolve test data path. Default to the dataset the run was trained on.
    if args.test_data is None:
        test_data_arg = config.data.train_path
        logger.info(f"--test-data not provided, defaulting to {test_data_arg}")
    else:
        test_data_arg = args.test_data
    test_path = Path(os.path.expanduser(test_data_arg))

    # Load test data
    logger.info("Loading test data...")
    print("Loading test data...")
    try:
        if test_path.is_dir():
            if (test_path / "test").exists():
                logger.info("Loading from prepared data structure (test split only)...")
                logger.info(f"Data directory: {test_path}")

                state_col = getattr(config.data, "state_col", None)
                if state_col and len(state_col) == 0:
                    state_col = None

                # Load only test data (plus optional test_div)
                use_div = getattr(config.data, "use_diverging_trajectories", False)
                result = load_split_data(
                    data_dir=str(test_path),
                    input_col=getattr(config.data, "input_col", ["d"]),
                    output_col=getattr(config.data, "output_col", ["e"]),
                    state_col=state_col,
                    pattern=getattr(config.data, "pattern", "*.csv"),
                    load_train=False,  # Don't load training data
                    load_val=False,  # Don't load validation data
                    load_test=True,  # Only load test data
                    load_div=use_div,
                )
                (
                    _, _,
                    _, _,
                    test_inputs, test_outputs,
                    _, _, test_states,
                    _, _, _,
                    _, _, _,
                    test_div_inputs, test_div_outputs, test_div_states,
                ) = result
                logger.info(f"Loaded test data: {test_inputs.shape[0]} sequences")
                if test_states is not None:
                    logger.info(f"State information loaded: {test_states.shape}")
                if use_div and test_div_inputs is not None:
                    logger.info(f"Loaded test_div data: {len(test_div_inputs)} sequences")
            else:
                logger.info("Loading directly from CSV folder...")
                logger.info(f"Test data directory: {test_path}")

                state_col = getattr(config.data, "state_col", None)
                if state_col and len(state_col) == 0:
                    state_col = None

                test_inputs, test_outputs, test_states, filenames = load_csv_folder(
                    folder_path=str(test_path),
                    input_col=getattr(config.data, "input_col", ["d"]),
                    output_col=getattr(config.data, "output_col", ["e"]),
                    state_col=state_col,
                    pattern=getattr(config.data, "pattern", "*.csv"),
                )
                logger.info(f"Loaded {len(filenames)} files from test set")
                if test_states is not None:
                    logger.info(f"State information loaded: {test_states.shape}")
                test_div_inputs = test_div_outputs = test_div_states = None

        elif test_path.suffix == ".csv":
            logger.info("Loading from single CSV file...")
            test_inputs, test_outputs = DataLoader.load_from_csv(str(test_path), delimiter=",")
            test_states = None
            test_div_inputs = test_div_outputs = test_div_states = None

        elif test_path.suffix == ".npy":
            logger.info("Loading NPY files...")
            test_inputs, test_outputs = DataLoader.load_from_npy(
                str(test_path), str(test_path).replace("_inputs.npy", "_outputs.npy")
            )
            test_states = None
            test_div_inputs = test_div_outputs = test_div_states = None
        else:
            raise ValueError(
                f"Unsupported data format: {test_path}\n"
                f"Use either:\n"
                f"  1. Prepared data folder with test/ subfolder\n"
                f"  2. Folder of CSV files\n"
                f"  3. Single CSV file\n"
                f"  4. NPY file"
            )

        logger.info(f"Test data loaded: inputs={test_inputs.shape}, outputs={test_outputs.shape}")
        print(f"Test data: {test_inputs.shape}, {test_outputs.shape}")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        logger.exception("Full traceback:")
        raise

    # Load normalizer from the path resolved above.
    normalizer = None
    if normalizer_path is not None:
        normalizer = DataNormalizer.load(str(normalizer_path))
        logger.info(f"Loaded normalizer from {normalizer_path}")
    else:
        logger.warning("Normalizer not found next to checkpoint — proceeding unnormalized.")

    # Create test loader
    if normalizer is not None:
        test_inputs_norm = normalizer.transform_inputs(test_inputs)
        test_outputs_norm = normalizer.transform_outputs(test_outputs)
    else:
        test_inputs_norm = test_inputs
        test_outputs_norm = test_outputs

    from torch.utils.data import DataLoader as TorchDataLoader

    from sysid.data import TimeSeriesDataset, VariableLengthDataset

    test_dataset = TimeSeriesDataset(
        test_inputs_norm,
        test_outputs_norm,
        test_states,
        sequence_length=None,
        sequence_stride=None,
    )

    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_with_optional_states,
    )

    # Optional diverging test loader: variable-length trajectories, batch_size=1.
    # test_div_inputs is a list of per-trajectory 2D arrays.
    test_div_loader = None
    if test_div_inputs is not None:
        if normalizer is not None:
            test_div_inputs_norm = [normalizer.transform_inputs(a) for a in test_div_inputs]
            test_div_outputs_norm = [normalizer.transform_outputs(a) for a in test_div_outputs]
        else:
            test_div_inputs_norm = test_div_inputs
            test_div_outputs_norm = test_div_outputs
        test_div_dataset = VariableLengthDataset(
            test_div_inputs_norm,
            test_div_outputs_norm,
            test_div_states,
        )
        test_div_loader = TorchDataLoader(
            test_div_dataset,
            batch_size=getattr(config.data, "diverging_batch_size", 1),
            shuffle=False,
            collate_fn=collate_with_optional_states,
        )

    # Load model
    logger.info("Loading model...")
    print("Loading model...")
    try:
        model = load_model(str(model_path), config, str(device))
        logger.info(f"Model loaded from {model_path}")

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        print(f"Model loaded from {model_path}")

        # Generate ellipse/polytope plot for SimpleLure models
        import matplotlib.pyplot as plt

        from sysid.models import SimpleLure
        from sysid.utils import plot_ellipse_and_parallelogram

        if isinstance(model, SimpleLure) and model.nx == 2:
            logger.info("Generating ellipse and polytope visualization...")
            try:
                X = np.linalg.inv(model.P.cpu().detach().numpy())
                H = model.L.cpu().detach().numpy() @ X
                s = model.s.cpu().detach().numpy()
                max_norm_x0 = model.max_norm_x0 if hasattr(model, "max_norm_x0") else None

                fig, ax = plot_ellipse_and_parallelogram(X, H, s, max_norm_x0, show=False)

                ellipse_plot_path = output_dir / "ellipse_polytope.png"
                ellipse_plot_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(ellipse_plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Ellipse/polytope plot saved to {ellipse_plot_path}")
            except Exception as e:
                logger.warning(f"Failed to generate ellipse/polytope plot: {e}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Set up MLflow tracking (must happen before mlflow.start_run).
    setup_mlflow_tracking(config)

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=str(device),
        output_dir=str(output_dir),
        warmup_steps=config.training.warmup_steps,
    )

    # Evaluate — always log to the existing MLflow run.
    logger.info("=" * 70)
    logger.info("Starting evaluation...")
    logger.info(f"Test batches: {len(test_loader)}")
    logger.info("=" * 70)
    print("\nEvaluating model...")

    try:
        with mlflow.start_run(run_id=args.run_id):
            logger.info(f"Logging evaluation to MLflow run: {args.run_id}")

            results = evaluator.evaluate(
                test_loader=test_loader,
                normalizer=normalizer,
            )
            if test_div_loader is not None:
                evaluator.evaluate_diverging(
                    test_div_loader=test_div_loader,
                    normalizer=normalizer,
                )

            all_metrics = results.get("metrics", {})
            metrics = filter_metrics(all_metrics, config.evaluation.metrics)

            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and metric != "per_step":
                    mlflow.log_metric(f"eval_{metric}", value)
                    logger.info(f"{metric}: {value:.6f}")

            logger.info("Generating plots and analysis...")
            print("\nGenerating plots...")

            try:
                e_hat = np.load(output_dir / "predictions.npy")
                e = np.load(output_dir / "targets.npy")
                d = np.load(output_dir / "inputs.npy")

                num_sequences = e_hat.shape[0]
                sample_indices = [0]
                if num_sequences > 1:
                    other_indices = list(range(1, num_sequences))
                    num_random = min(4, len(other_indices))
                    random_indices = np.random.choice(
                        other_indices, size=num_random, replace=False
                    ).tolist()
                    sample_indices.extend(random_indices)

                plot_predictions(
                    evaluator.output_dir, e_hat, e, d,
                    sample_indices=sample_indices,
                    warmup_steps=config.training.warmup_steps,
                )
                evaluator.analyze_errors(e_hat, e)
                logger.info("Plots generated successfully")
            except Exception as e_err:
                logger.warning(f"Failed to generate plots: {e_err}")

            mlflow.log_artifacts(str(output_dir), "evaluation")
            logger.info("Evaluation artifacts logged to MLflow")

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
    logger.info(f"Evaluation metrics logged to training run: {args.run_id}")
    print(f"\nEvaluation completed! Metrics logged to MLflow run {args.run_id[:8]}")


if __name__ == "__main__":
    main()
