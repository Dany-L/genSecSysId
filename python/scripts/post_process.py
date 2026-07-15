#!/usr/bin/env python
"""
Post-process a trained model by solving an SDP to find optimal Lyapunov certificate (P and L)
while keeping system matrices (A, B, C, D) fixed.

This script loads a trained SimpleLure model and calls its post_process() method
to solve a semidefinite program (SDP) for optimal P and L matrices.

Usage:
    python scripts/post_process.py --run-id <run_id>

Config, checkpoint, and normalizer are all resolved from the run id via the
standard training layout (see sysid.config.resolve_run_artifacts).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import matplotlib.pyplot as plt

from sysid.config import resolve_run_artifacts, setup_mlflow_tracking
from sysid.data.direct_loader import load_csv_folder
from sysid.evaluation import (
    check_input_condition,
    list_true_dynamics,
    plot_post_process_trajectories,
    regional_verification,
    simulate_model,
)
from sysid.models import SimpleLure, load_model
from sysid.data import DataNormalizer
from sysid.utils import max_abs_output

torch.set_default_dtype(torch.float64)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "~/genSecSysId-Data"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-process a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--run-id", type=str, required=True, help="MLflow run ID of the trained model"
    )

    parser.add_argument(
        "--data-root", type=str, default=DEFAULT_DATA_ROOT,
        help=f"Base directory for run artefacts (default: {DEFAULT_DATA_ROOT}).",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Small positive constant for strict inequalities (default: 1e-3)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for post-processed results (default: derived from the run's config root_dir).",
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="Override MLflow tracking URI from the run's config (default: use config).",
    )

    parser.add_argument(
        "--true-dynamics",
        type=str,
        default=None,
        choices=list_true_dynamics(),
        help=(
            "Name of registered true-dynamics module to compare against during "
            "regional verification (e.g. 'duffing'). If omitted, regional "
            "verification still runs but without ground-truth comparison."
        ),
    )
    parser.add_argument(
        "--rv-violation-factors",
        type=float,
        nargs="+",
        default=[1.5],
        help=(
            "Peak-||u_n|| / s factors used for the input-violation regime. "
            "<1 stays inside the input bound (sanity baseline); >=1 violates."
        ),
    )
    parser.add_argument(
        "--rv-num-trajectories",
        type=int,
        default=4,
        help="Number of trajectories per regime/factor in regional verification.",
    )
    parser.add_argument(
        "--rv-horizon",
        type=int,
        default=400,
        help="Trajectory length (steps) for regional verification.",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data folder (default: <config.data.train_path>/test).",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to train data folder (default: <config.data.train_path>/train).",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    run_id = args.run_id

    # Resolve config, checkpoint, normalizer, and run_info from the run id.
    try:
        config, model_path, normalizer_path, run_info = resolve_run_artifacts(
            run_id, data_root=args.data_root
        )
    except Exception as e:
        logger.error(f"Failed to resolve run_id={run_id}: {e}")
        sys.exit(1)

    # Setup MLflow tracking from the run's own config (with optional CLI override).
    setup_mlflow_tracking(config, override_uri=args.mlflow_tracking_uri)
    logger.info(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Resolve output directory.
    if args.output_dir is not None:
        output_dir = Path(os.path.expanduser(args.output_dir))
    elif getattr(config, "root_dir", None):
        base = Path(os.path.expanduser(config.root_dir))
        output_dir = base / "outputs" / config.model.model_type
    else:
        output_dir = Path(os.path.expanduser(config.output_dir))
    run_output_dir = output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Load model from the resolved checkpoint path.
    logger.info(f"Loading model from {model_path}")
    try:
        model = load_model(str(model_path), config, device="cpu")
        if not isinstance(model, SimpleLure):
            logger.error(
                "Model is not a SimpleLure model. Post-processing only supports SimpleLure."
            )
            sys.exit(1)
        constraints_satisfied = model.check_constraints()
        logger.info(
            f"Best model weights loaded (constraints satisfied? {constraints_satisfied})"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Load normalizer from the resolved path.
    normalizer = None
    if normalizer_path is not None:
        normalizer = DataNormalizer.load(str(normalizer_path))
        logger.info(f"Normalizer loaded from {normalizer_path}")
    else:
        logger.warning("Normalizer not found next to checkpoint.")

    # Use the SAME run (not a new one) - just add artifacts and metrics
    logger.info(f"Logging post-processing results to original run: {run_id}")

    with mlflow.start_run(run_id=args.run_id):
        # Call model's post_process method
        logger.info("Calling model.post_process()...")

        # Get state column if provided
        state_col = getattr(config.data, "state_col", None)
        if state_col and len(state_col) == 0:  # Empty list means no state
            state_col = None

        data_base = Path(os.path.expanduser(config.data.train_path))
        test_path = (
            Path(os.path.expanduser(args.test_data))
            if args.test_data is not None
            else data_base / "test"
        )
        train_path = (
            Path(os.path.expanduser(args.train_data))
            if args.train_data is not None
            else data_base / "train"
        )

        test_inputs, test_outputs, test_states, filenames = load_csv_folder(
            folder_path=str(test_path),
            input_col=getattr(config.data, "input_col", ["d"]),
            output_col=getattr(config.data, "output_col", ["e"]),
            state_col=state_col,
            pattern=getattr(config.data, "pattern", "*.csv"),
        )
        train_inputs, train_outputs, train_states, _ = load_csv_folder(
            folder_path=str(train_path),
            input_col=getattr(config.data, "input_col", ["d"]),
            output_col=getattr(config.data, "output_col", ["e"]),
            state_col=state_col,
            pattern=getattr(config.data, "pattern", "*.csv"),
        )

        if test_states is not None:
            test_inputs, test_outputs, test_states = (
                np.stack(test_inputs),
                np.stack(test_outputs),
                np.stack(test_states),
            )
        else:
            test_inputs, test_outputs = (
                np.stack(test_inputs),
                np.stack(test_outputs),
            )

        if train_states is not None:
            train_inputs, train_outputs, train_states = (
                np.stack(train_inputs),
                np.stack(train_outputs),
                np.stack(train_states),
            )
        else:
            train_inputs, train_outputs = (
                np.stack(train_inputs),
                np.stack(train_outputs),
            )
        
        u_train_n = torch.tensor(normalizer.transform_inputs(train_inputs))
        b, N, _ = u_train_n.shape
        x0_train = torch.zeros((b, model.nx))  # start from origin
        warmup_steps = getattr(config.training, "warmup_steps", 0)

        # --- Physical safe output level from the training data ----------------
        # y_max is physical (has meaning); output_std relates the model's
        # normalized C/P/s to physical units for the coverage machinery.
        y_max_train = max_abs_output(train_outputs)
        output_std = float(np.asarray(normalizer.output_std).reshape(-1)[0])
        model.set_output_coverage_level(y_max_train, output_std)
        logger.info(f"Maximum |output| in training data (physical y_max): {y_max_train:.4f}")
        L_orig = model.L.cpu().detach().numpy()
        P_orig = model.P.cpu().detach().numpy()
        H_orig = L_orig @ np.linalg.inv(P_orig)
        C_orig = model.C.cpu().detach().numpy()
        s_orig = float(model.s.cpu().detach().numpy())
        y_bar_orig = float(output_std * s_orig * np.sqrt((C_orig @ P_orig @ C_orig.T).item()))
        logger.info(f"original s: {s_orig:.4f}, original ||H||: {float(np.linalg.norm(H_orig)):.4f} original y_bar: {y_bar_orig:.4f}")
        mlflow.log_metric("data/max_output_train", y_max_train)

        # --- Baseline: input condition under the ORIGINAL (trained) certificate
        n_stable_orig, n_unstable_orig = check_input_condition(
            model, u_train_n, x0_train, warmup_steps, run_output_dir,
            tag="orig", title="Original training trajectories",
        )
        mlflow.log_metric("post_process/orig/stable_train_trajectories", n_stable_orig)
        mlflow.log_metric("post_process/orig/unstable_train_trajectories", n_unstable_orig)
        mlflow.log_metric("post_process/orig/s", float(model.s.cpu().detach().numpy()))
        mlflow.log_metric("post_process/orig/norm_H", float(np.linalg.norm(H_orig)))
        mlflow.log_metric("post_process/orig/y_bar", y_bar_orig)

        # ------------------------------------------------------------------
        # Post-processing: solve the two (now cleanly separated) certificate
        # SDPs and set the model to the LARGEST invariant set. See
        # SimpleLure.post_process for the full description:
        #   Problem 1 (MaxS, _max_s_sdp): max feasible s -> the operative
        #       certificate written back into the model. Reports ȳ_c, ‖H‖, s
        #       and whether the coverage floor (σ·s)²·CPCᵀ ≥ y_max² holds.
        #   Problem 2 (coverage sweep, _coverage_sdp over a grid of s): the
        #       tightest coverage ȳ_f (reported only, not applied).
        # ------------------------------------------------------------------
        logger.info("Calling model.post_process()...")
        result = model.post_process(y_max=y_max_train, n_grid=20)
        if not result["success"]:
            logger.error(f"Post-processing failed: {result.get('status', 'unknown')}")
            sys.exit(1)

        max_s = result["max_s"]
        cov = result["coverage"]

        # Problem 1 — max-feasible-s certificate (operative; largest invariant set).
        logger.info(
            f"[Problem 1: MaxS] y_bar (ȳ_c)={max_s['y_bar']}, s={max_s['s']:.4f}, "
            f"norm_H={max_s['norm_H']:.4f}, coverage_ok={max_s['coverage_ok']}"
        )
        mlflow.log_metric("post_process/max_s/s", max_s["s"])
        mlflow.log_metric("post_process/max_s/norm_H", max_s["norm_H"])
        mlflow.log_metric("post_process/max_s/max_eig_F", max_s["max_eig_F"])
        if max_s["y_bar"] is not None:
            mlflow.log_metric("post_process/max_s/y_bar", max_s["y_bar"])  # ȳ_c
        if max_s["coverage_ok"] is not None:
            mlflow.log_metric("post_process/max_s/coverage_ok", int(max_s["coverage_ok"]))

        # Problem 2 — tightest coverage over the s-grid (report only).
        logger.info(
            f"[Problem 2: coverage] y_bar (ȳ_f)={cov['y_bar']}, s={cov['s']}, "
            f"reason={cov['reason']} "
            f"(band [{cov['s_min']:.1f}, {cov['s_max']:.1f}], n_grid={cov['n_grid']})"
        )
        if cov["y_bar"] is not None:
            mlflow.log_metric("post_process/coverage/y_bar", cov["y_bar"])  # ȳ_f
            mlflow.log_metric("post_process/coverage/s", cov["s"])
        mlflow.log_param("post_process/coverage_reason", cov["reason"])

        mlflow.log_metric(
            "post_process/constraints_satisfied", int(result["constraints_satisfied"])
        )
        mlflow.log_metric("post_process/y_max", result["y_max"])

        # --- Test on the training data under the APPLIED (MaxS) certificate ----
        n_stable_opt, n_unstable_opt = check_input_condition(
            model, u_train_n, x0_train, warmup_steps, run_output_dir,
            tag="opt", title="Post-processed (MaxS) training trajectories",
        )
        mlflow.log_metric("post_process/opt_stable_train_trajectories", n_stable_opt)
        mlflow.log_metric("post_process/opt_unstable_train_trajectories", n_unstable_opt)

        # Log parameters
        mlflow.log_param("post_processing", True)
        mlflow.log_param("post_process_eps", args.eps)

        # --- Save the post-processed certificate + fixed dynamics -------------
        alpha = 1.0 / (1.0 + np.exp(-model.tau.cpu().detach().numpy()))  # sigmoid(tau)
        results_path = run_output_dir / "post_processing.npz"
        np.savez(
            results_path,
            P=model.P.cpu().detach().numpy(),
            L=(model.L.cpu().detach().numpy() if model.learn_L
               else np.zeros((model.nz, model.nx))),
            s=model.s.cpu().detach().numpy(),
            alpha=alpha,
            y_c=np.nan if max_s["y_bar"] is None else max_s["y_bar"],
            y_f=np.nan if cov["y_bar"] is None else cov["y_bar"],
            norm_H=max_s["norm_H"],
            y_max=y_max_train,
            A=model.A.cpu().detach().numpy(),
            B=model.B.cpu().detach().numpy(),
            B2=model.B2.cpu().detach().numpy(),
            C=model.C.cpu().detach().numpy(),
            C2=model.C2.cpu().detach().numpy(),
            D21=model.D21.cpu().detach().numpy(),
        )
        logger.info(f"Saved results to {results_path}")
        mlflow.log_artifact(str(results_path), artifact_path="post_processing")

        # --- Simulate test trajectories (needed for the plots below) ----------
        with torch.no_grad():
            b, N, _ = test_inputs.shape
            x0 = torch.zeros((b, model.nx))  # start from origin
            u_n = torch.tensor(normalizer.transform_inputs(test_inputs))
            e_hat_n, (xs, _), _ = simulate_model(model, u_n, x0, warmup_steps)
            e_hat = normalizer.inverse_transform_outputs(e_hat_n.cpu().detach().numpy())
            xs = xs[:, :N]  # strip last state

        # plot some prediction for handpicked trajectories
        logger.info(f"Generating prediction plot for trajectory...")
        try:
            # import matplotlib.pyplot as plt
            import tikzplotlib

            from sysid.utils import plot_predictions

            pred_plot_name = Path("prediction_trajectory_post.png")
            pred_plot_path = Path(run_output_dir / pred_plot_name)

            fig, axes = plot_predictions(
                output_dir=run_output_dir,
                e_hat=e_hat,
                e=test_outputs,
                num_samples=3,
                # sample_indices=UNSTAB_STAB_ZERO,
                save_path=pred_plot_path,
                return_axes=True,
                warmup_steps=warmup_steps,
            )

            mlflow.log_figure(fig, f'post_processing/{str(pred_plot_name.with_suffix(".png"))}')
            try:
                tikzplotlib.save(str(pred_plot_path.with_suffix(".tex")))
                mlflow.log_artifact(
                    str(pred_plot_path.with_suffix(".tex")), artifact_path="post_processing"
                )
            except Exception as e:
                logger.warning(f"Failed to save TikZ plot: {e}")
            plt.close(fig)
            logger.info(f"Prediction plot saved to {pred_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate prediction plot for trajectory")

        # ------------------------------------------------------------------
        # Regional verification
        # ------------------------------------------------------------------
        logger.info("Running regional verification...")
        try:
            regional_verification(
                model=model,
                normalizer=normalizer,
                run_output_dir=run_output_dir,
                true_dynamics_name=args.true_dynamics,
                config=config,
                factors=list(args.rv_violation_factors),
                n_traj=args.rv_num_trajectories,
                horizon=args.rv_horizon,
            )
        except Exception as e:
            logger.warning(f"Regional verification failed: {e}", exc_info=True)

        logger.info(f"✓ Post-processing complete! Results saved to run: {args.run_id}")


if __name__ == "__main__":
    main()
