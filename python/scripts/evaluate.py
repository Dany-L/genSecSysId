"""Evaluation script.

Usage:
    python scripts/evaluate.py --run-id <mlflow_run_id> [--test-data <path>]

Config, checkpoint, and normalizer are all resolved from the standard
training layout under --data-root (default: ~/genSecSysId-Data). Test
data defaults to <config.data.train_path>/test unless overridden with
--test-data.
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
from sysid.evaluation import Evaluator, compute_metrics
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
        help="Path to test data (folder of CSVs, folder with a test/ subfolder, "
             "or single .csv/.npy file). Defaults to <config.data.train_path>/test.",
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

    # Resolve test data path. Default to the test/ split of the dataset the
    # run was trained on; --test-data overrides.
    if args.test_data is None:
        test_path = Path(os.path.expanduser(config.data.train_path)) / "test"
        logger.info(f"--test-data not provided, defaulting to {test_path}")
    else:
        test_path = Path(os.path.expanduser(args.test_data))

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

                # Load test/ (and test_div/ if present on disk — independent of
                # how the run was trained, so divergent metrics can be reported
                # for any run as long as a test_div/ folder exists).
                result = load_split_data(
                    data_dir=str(test_path),
                    input_col=getattr(config.data, "input_col", ["d"]),
                    output_col=getattr(config.data, "output_col", ["e"]),
                    state_col=state_col,
                    pattern=getattr(config.data, "pattern", "*.csv"),
                    load_train=False,  # Don't load training data
                    load_val=False,  # Don't load validation data
                    load_test=True,  # Only load test data
                    load_div=True,  # Auto-skip if test_div/ is missing.
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
                if test_div_inputs is not None:
                    logger.info(f"Loaded test_div data: {len(test_div_inputs)} sequences")
            else:
                logger.info("Loading directly from CSV folder...")
                logger.info(f"Test data directory: {test_path}")

                state_col = getattr(config.data, "state_col", None)
                if state_col and len(state_col) == 0:
                    state_col = None

                test_in_l, test_out_l, test_st_l, filenames = load_csv_folder(
                    folder_path=str(test_path),
                    input_col=getattr(config.data, "input_col", ["d"]),
                    output_col=getattr(config.data, "output_col", ["e"]),
                    state_col=state_col,
                    pattern=getattr(config.data, "pattern", "*.csv"),
                )
                # load_csv_folder returns per-trajectory lists; stack into the
                # (n_files, T, n_features) tensor downstream code expects.
                test_inputs = np.stack(test_in_l, axis=0)
                test_outputs = np.stack(test_out_l, axis=0)
                test_states = np.stack(test_st_l, axis=0) if test_st_l is not None else None
                logger.info(f"Loaded {len(filenames)} files from test set")
                if test_states is not None:
                    logger.info(f"State information loaded: {test_states.shape}")

                # Sibling diverging folder (e.g. .../test/ → .../test_div/).
                # Variable-length trajectories stay as lists for the downstream
                # batch_size=1 div loader.
                test_div_inputs = test_div_outputs = test_div_states = None
                div_sibling = test_path.parent / f"{test_path.name}_div"
                if div_sibling.is_dir():
                    try:
                        div_in_l, div_out_l, div_st_l, div_files = load_csv_folder(
                            folder_path=str(div_sibling),
                            input_col=getattr(config.data, "input_col", ["d"]),
                            output_col=getattr(config.data, "output_col", ["e"]),
                            state_col=state_col,
                            pattern=getattr(config.data, "pattern", "*.csv"),
                        )
                        test_div_inputs = div_in_l
                        test_div_outputs = div_out_l
                        test_div_states = div_st_l
                        logger.info(
                            f"Loaded test_div data from {div_sibling}: "
                            f"{len(div_files)} sequences"
                        )
                    except Exception as e:
                        logger.warning(f"Could not load {div_sibling}: {e}")

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
            # The normalizer promotes 2D (T, n) → 3D (1, T, n); reshape back so
            # VariableLengthDataset stores per-trajectory (T, n) tensors.
            test_div_inputs_norm = [
                np.asarray(normalizer.transform_inputs(a)).reshape(a.shape)
                for a in test_div_inputs
            ]
            test_div_outputs_norm = [
                np.asarray(normalizer.transform_outputs(a)).reshape(a.shape)
                for a in test_div_outputs
            ]
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

                ellipse_plot_path = eval_dir / "ellipse_polytope.png"
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

    # Eval artefacts go into a dedicated subfolder so they don't mingle with
    # the training-time artefacts (e.g. <output_dir>/predictions/epoch_*.png)
    # that were already logged to MLflow during training. Logging the whole
    # output_dir under "evaluation" would otherwise duplicate every training
    # plot under evaluation/predictions/.
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=str(device),
        output_dir=str(eval_dir),
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

            # --- Converging pass --------------------------------------------------
            results = evaluator.evaluate(
                test_loader=test_loader,
                normalizer=normalizer,
            )
            metrics_conv = filter_metrics(results.get("metrics", {}), config.evaluation.metrics)
            for metric, value in metrics_conv.items():
                if isinstance(value, (int, float)) and metric != "per_step":
                    mlflow.log_metric(f"eval_conv_{metric}", value)
                    # Also log under the legacy `eval_<metric>` name (historically
                    # that was the conv-only metric) so existing dashboards keep
                    # working and old runs compare directly to new ones.
                    mlflow.log_metric(f"eval_{metric}", value)
                    logger.info(f"conv {metric}: {value:.6f}")

            # Conv pooled samples (post-warmup) so we can combine with div.
            warmup = config.training.warmup_steps
            e_hat_full = results["e_hat"]
            e_full = results["e"]
            n_out = e_hat_full.shape[-1]
            e_hat_conv_pool = e_hat_full[:, warmup:, :].reshape(-1, n_out)
            e_conv_pool = e_full[:, warmup:, :].reshape(-1, n_out)

            # --- Diverging pass (if available) ------------------------------------
            if test_div_loader is not None:
                div_results = evaluator.evaluate_diverging(
                    test_div_loader=test_div_loader,
                    normalizer=normalizer,
                )
                metrics_div = filter_metrics(
                    div_results["metrics_div"], config.evaluation.metrics
                )
                for metric, value in metrics_div.items():
                    if isinstance(value, (int, float)) and metric != "per_step":
                        mlflow.log_metric(f"eval_div_{metric}", value)
                        logger.info(f"div {metric}: {value:.6f}")
                mlflow.log_metric(
                    "eval_div_num_trajectories", div_results["num_trajectories_div"]
                )

                # --- Overall (conv post-warmup + div, pooled) ---------------------
                e_hat_div_pool = div_results["e_hat_div"]
                e_div_pool = div_results["e_div"]
                e_hat_all = np.concatenate([e_hat_conv_pool, e_hat_div_pool], axis=0)
                e_all = np.concatenate([e_conv_pool, e_div_pool], axis=0)
                metrics_overall = filter_metrics(
                    compute_metrics(e_hat_all, e_all), config.evaluation.metrics
                )
                for metric, value in metrics_overall.items():
                    if isinstance(value, (int, float)) and metric != "per_step":
                        mlflow.log_metric(f"eval_overall_{metric}", value)
                        logger.info(f"overall {metric}: {value:.6f}")
            else:
                logger.info(
                    "No diverging test data — div/overall metrics not computed."
                )

            logger.info("Generating plots and analysis...")
            print("\nGenerating plots...")

            def _pick_indices(n, max_extra=4):
                idx = [0]
                if n > 1:
                    others = list(range(1, n))
                    k = min(max_extra, len(others))
                    idx.extend(np.random.choice(others, size=k, replace=False).tolist())
                return idx

            # Converging prediction plot + error analysis.
            try:
                e_hat = np.load(eval_dir / "predictions.npy")
                e = np.load(eval_dir / "targets.npy")
                d = np.load(eval_dir / "inputs.npy")
                plot_predictions(
                    evaluator.output_dir, e_hat, e, d,
                    sample_indices=_pick_indices(e_hat.shape[0]),
                    warmup_steps=config.training.warmup_steps,
                )
                evaluator.analyze_errors(e_hat, e)
                logger.info("Conv plots generated successfully")
            except Exception as e_err:
                logger.warning(f"Failed to generate conv plots: {e_err}")

            # Diverging prediction plot (variable-length trajectories, NaN-padded
            # to a common length; matplotlib silently skips NaN points so each
            # trajectory ends at its real final step).
            div_pred_path = eval_dir / "predictions_div.npy"
            if div_pred_path.exists():
                try:
                    e_hat_d = np.load(div_pred_path)
                    e_d = np.load(eval_dir / "targets_div.npy")
                    d_d = np.load(eval_dir / "inputs_div.npy")
                    plot_predictions(
                        evaluator.output_dir, e_hat_d, e_d, d_d,
                        sample_indices=_pick_indices(e_hat_d.shape[0]),
                        save_path=str(eval_dir / "predictions_div_plot.png"),
                        warmup_steps=0,
                    )
                    logger.info("Div plot generated successfully")
                except Exception as e_err:
                    logger.warning(f"Failed to generate div plot: {e_err}")

            mlflow.log_artifacts(str(eval_dir), "evaluation")
            logger.info("Evaluation artifacts logged to MLflow")

        logger.info(f"Predictions plot: {eval_dir / 'predictions_plot.png'}")
        logger.info(f"Error analysis: {eval_dir / 'error_analysis.png'}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        raise

    logger.info("=" * 70)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {eval_dir}")
    logger.info(f"Metrics file: {eval_dir / 'evaluation_results.json'}")
    logger.info(f"Evaluation metrics logged to training run: {args.run_id}")
    print(f"\nEvaluation completed! Metrics logged to MLflow run {args.run_id[:8]}")


if __name__ == "__main__":
    main()
