#!/usr/bin/env python
"""
Post-process a trained model by solving an SDP to find optimal Lyapunov certificate (P and L)
while keeping system matrices (A, B, C, D) fixed.

This script loads a trained SimpleLure model and calls its post_process() method
to solve a semidefinite program (SDP) for optimal P and L matrices.

Usage:
    python scripts/post_process.py --run-id <run_id> [--optimize-s]

Examples:
    # Post-process a model, keeping s fixed
    python scripts/post_process.py --run-id abc123def456

    # Post-process and optimize for minimum s
    python scripts/post_process.py --run-id abc123 --optimize-s
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

from sysid.config import Config
from sysid.data import DataNormalizer
from sysid.data.direct_loader import load_csv_folder
from sysid.models import SimpleLure, SimpleLureSafe
from sysid.utils import (
    plot_ellipse,
    plot_ellipse_and_parallelogram,
    plot_polytope,
    plot_safe_set_trajectories,
)

torch.set_default_dtype(torch.float64)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default configuration values (can be overridden via command line)
DEFAULT_TEST_DATA_PATH = "/Users/jack/genSecSysId-Data/data/Duffing/test"
DEFAULT_TRAIN_DATA_PATH = "/Users/jack/genSecSysId-Data/data/Duffing/train"
# DEFAULT_TRAJECTORY_LIST = [0, 1]
DEFAULT_CONFIG_PATH = "~/genSecSysId-Data/configs/crnn_gen-sec.yaml"


def _simulate(model, u, x0, warmup_steps):
    """Run model dynamics for diagnostic plots.

    For SimpleLureSafe we explicitly bypass the safety filter so that the
    constraint margin c reflects raw behavior (the filter would otherwise
    prevent any violation by construction, making the plots uninformative).
    """
    if isinstance(model, SimpleLureSafe):
        return model.forward_unfiltered(u, x0)
    return model(u, x0, warmup_steps=warmup_steps)


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
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file (YAML or JSON)",
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
        default="post_processing",
        help="Output directory for post-processed results (default: post_processing)",
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: uses current tracking URI)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    logger.info(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Load configuration early so we can derive directories from config.root_dir
    config_path = Path(args.config)
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        config = Config.from_yaml(os.path.expanduser(config_path))
    elif config_path.suffix == ".json":
        config = Config.from_json(os.path.expanduser(config_path))
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    if getattr(config, "root_dir", None):
        base = Path(os.path.expanduser(config.root_dir))
        output_dir = base / "outputs" / config.model.model_type
    else:
        output_dir = Path(os.path.expanduser(config.output_dir))

    # Load model from MLflow
    logger.info(f"Loading model from run {args.run_id}")
    try:
        model_uri = f"runs:/{args.run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)

        if not isinstance(model, SimpleLure):
            logger.error(
                "Model is not a SimpleLure model. Post-processing only supports SimpleLure."
            )
            sys.exit(1)

        # Load best model weights (overwrites final model weights with best checkpoint)
        try:
            best_model_path = mlflow.artifacts.download_artifacts(
                run_id=args.run_id, artifact_path="models/best_model.pt"
            )
            checkpoint = torch.load(best_model_path, map_location="cpu")
            # model.check_constraints()
            model.load_state_dict(checkpoint["model_state_dict"])
            constraints_satisfied = model.check_constraints()
            # model.check_constraints()
            best_epoch = checkpoint.get("best_epoch", "?")
            best_val_loss = checkpoint.get("best_val_loss", float("nan"))
            logger.info(
                f"Best model weights loaded (best epoch: {best_epoch}, best val loss: {best_val_loss:.6f} constraints satisfied? {constraints_satisfied})"
            )
        except Exception as e:
            logger.warning(f"Could not load best model weights, using final model: {e}")

        # Load normalizer from MLflow
        normalizer = None
        try:
            normalizer_path = mlflow.artifacts.download_artifacts(
                run_id=args.run_id, artifact_path="models/normalizer.json"
            )
            normalizer = DataNormalizer.load(normalizer_path)
            logger.info(f"Normalizer loaded from MLflow")
        except Exception as e:
            logger.warning(f"Could not load normalizer from MLflow: {e}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Get the original run
    run = mlflow.get_run(args.run_id)
    experiment_id = run.info.experiment_id

    # Use the SAME run (not a new one) - just add artifacts and metrics
    logger.info(f"Logging post-processing results to original run: {args.run_id}")

    with mlflow.start_run(run_id=args.run_id):
        # Call model's post_process method
        logger.info("Calling model.post_process()...")

        # Get state column if provided
        state_col = getattr(config.data, "state_col", None)
        if state_col and len(state_col) == 0:  # Empty list means no state
            state_col = None

        test_path = Path(os.path.expanduser(DEFAULT_TEST_DATA_PATH))
        train_path = Path(os.path.expanduser(DEFAULT_TRAIN_DATA_PATH))

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

        # output dir
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        u_train_n = torch.tensor(normalizer.transform_inputs(train_inputs))
        b, N, _ = u_train_n.shape
        x0_train = torch.zeros((b, model.nx))  # Start from origin

        # check input condition on training trajectories before post-processing
        warmup_steps = config.training.warmup_steps if hasattr(config.training, "warmup_steps") else 0
        with torch.no_grad():
            _, (x_hat_train, _), u_safe = _simulate(model, u_train_n, x0_train, warmup_steps)
            _, c_train = model.get_regularization_input(u_safe, x_hat_train, return_c=True, warmup_steps=warmup_steps)

        c_train_np = c_train.cpu().detach().numpy()
        x_hat_train_np = x_hat_train.cpu().detach().numpy()

        fig, ax, count_stable_orig, count_unstable_orig = plot_safe_set_trajectories(
            P=model.P.cpu().detach().numpy(),
            L=model.L.cpu().detach().numpy(),
            s=model.s.cpu().detach().numpy(),
            x_traj=x_hat_train_np,
            c=c_train_np,
            warmup_steps=warmup_steps,
            horizon=200,
        )
        ellipse_plot_path = output_dir / f"ellipse_polytope_post_train_orig_{args.run_id[:8]}.png"
        fig.savefig(ellipse_plot_path, dpi=150, bbox_inches="tight")
        mlflow.log_figure(fig, str(ellipse_plot_path.with_suffix(".png")))
        plt.close(fig)

        logger.info(f"Original training trajectories: total={b}, stable={count_stable_orig}, unstable={count_unstable_orig}")
        mlflow.log_metric("post_process/orig_stable_train_trajectories", count_stable_orig)
        mlflow.log_metric("post_process/orig_unstable_train_trajectories", count_unstable_orig)

        result = model.post_process()
        
        if not result["success"]:
            logger.error(
                f"Post-processing failed: {result.get('error', result.get('status', 'unknown'))}"
            )
            sys.exit(1)

        # Extract results
        summary = result["summary"]

        # Log metrics to MLflow with post_process prefix
        mlflow.log_metric(
            "post_process/constraints_satisfied", int(result["constraints_satisfied"])
        )
        mlflow.log_metric("post_process/s_original", summary["original"]["s"])
        mlflow.log_metric("post_process/s_optimized", summary["optimized"]["s"])
        # mlflow.log_metric("post_process/max_eig_F", summary["optimized"]["max_eig_F"])
        mlflow.log_metric("post_process/norm_P_original", summary["original"]["norm_P"])
        mlflow.log_metric("post_process/norm_L_original", summary["original"]["norm_L"])
        mlflow.log_metric("post_process/norm_H_original", summary["original"]["norm_H"])
        
        # mlflow.log_metric("post_process/norm_P_optimized", summary["optimized"]["norm_P"])
        # mlflow.log_metric("post_process/norm_L_optimized", summary["optimized"]["norm_L"])
        # mlflow.log_metric("post_process/norm_H_optimized", summary["optimized"]["norm_H"])
        # y_bar = summary["optimized"]["output_range"] * normalizer.output_std
        # mlflow.log_metric("post_process/y_bar", y_bar)

        # logger.info(f"Maximum output range (y_bar) after post-processing: {y_bar}")

        # check how many training trajectories satisfy the input condition after post-processing
        with torch.no_grad():
            _, (x_hat_train, _), u_safe = _simulate(model, u_train_n, x0_train, warmup_steps)
            _, c_train = model.get_regularization_input(u_safe, x_hat_train, return_c=True, warmup_steps=warmup_steps)

        fig, ax, count_stable_post, count_unstable_post = plot_safe_set_trajectories(
            P=model.P.cpu().detach().numpy(),
            L=model.L.cpu().detach().numpy(),
            s=model.s.cpu().detach().numpy(),
            x_traj=x_hat_train.cpu().detach().numpy(),
            c=c_train.cpu().detach().numpy(),
            warmup_steps=warmup_steps,
            horizon=200,
        )
        logger.info(f"Post-processed training trajectories: total={b}, stable={count_stable_post}, unstable={count_unstable_post}")
        mlflow.log_metric("post_process/opt_stable_train_trajectories", count_stable_post)
        mlflow.log_metric("post_process/opt_unstable_train_trajectories", count_unstable_post)

        ellipse_plot_path = output_dir / f"ellipse_polytope_post_train_opt_{args.run_id[:8]}.png"
        fig.savefig(ellipse_plot_path, dpi=150, bbox_inches="tight")
        mlflow.log_figure(fig, str(ellipse_plot_path.with_suffix(".png")))
        plt.close(fig)

        # Log parameter
        mlflow.log_param("post_processing", True)
        mlflow.log_param("post_process_eps", args.eps)

        # Save results to file
        alpha = 1/(1 + np.exp(-model.tau.cpu().detach().numpy()))  # Sigmoid of tau

        results_path = output_dir / f"post_processing_{args.run_id[:8]}.npz"
        np.savez(
            results_path,
            P_original=model.P.cpu().detach().numpy(),
            # P_opt=result["P_opt"],
            L_original=model.L.cpu().detach().numpy() if model.learn_L else None,
            # L_opt=result["L_opt"],
            # m_opt=result["m_opt"],
            s_original=summary["original"]["s"],
            # s_opt=result["s_opt"],
            # S_hat_opt=result["S_hat_opt"],
            # max_eig_F=result["max_eig_F"],
            alpha=alpha,
            A=model.A.cpu().detach().numpy(),
            B=model.B.cpu().detach().numpy(),
            B2=model.B2.cpu().detach().numpy(),
            C2=model.C2.cpu().detach().numpy(),
            D21=model.D21.cpu().detach().numpy(),
        )
        logger.info(f"Saved results to {results_path}")

        # Log results file to MLflow
        mlflow.log_artifact(str(results_path), artifact_path="post_processing")

        # Save and log updated model with different name
        # model_path = output_dir / f"post_processing_model_{args.run_id[:8]}.pt"
        # torch.save(model.state_dict(), model_path)
        # mlflow.pytorch.log_model(model, name="model_post_processing")

        # Generate ellipse/polytope plot if model is 2D
        if model.nx == 2:
            logger.info("Generating ellipse and polytope visualization...")

            # generate trajectories
            with torch.no_grad():
                b, N, _ = test_inputs.shape
                x0= torch.zeros((b, model.nx))  # Start from origin for visualization
                u_n = torch.tensor(normalizer.transform_inputs(test_inputs))
                e_hat_n, (xs, ws), _ = _simulate(model, u_n, x0, warmup_steps)
                e_hat = normalizer.inverse_transform_outputs(e_hat_n.cpu().detach().numpy())
                xs = xs[:,:N] # strip last state

            try:

                import tikzplotlib

                # check if (u^k)^T u^k <= s^2 - alpha^2 (x^k)^T X x^k holds for all test trajectories
                # c = (u^k)^T u^k - s^2 + alpha^2 (x^k)^T X x^k
                _, cs = model.get_regularization_input(u_n, xs, return_c=True)
                cs = cs.cpu().detach().numpy()

                ellipse_plot_path = output_dir / f"ellipse_polytope_post_{args.run_id[:8]}.png"

                if model.learn_L and config.training.use_custom_regularization:
                    fig, ax, count_stable, count_unstable = plot_safe_set_trajectories(
                        P=model.P.cpu().detach().numpy(),
                        L=model.L.cpu().detach().numpy(),
                        s=model.s.cpu().detach().numpy(),
                        x_traj=xs.cpu().detach().numpy(),
                        c=cs,
                        warmup_steps=warmup_steps,
                        horizon=100,
                        figsize=(8, 8),
                    )
                    logger.info(f'total:{b}, stable: {count_stable}, count unstable: {count_unstable}')

                    # Save to output directory
                    fig.savefig(ellipse_plot_path, dpi=150, bbox_inches="tight")

                    try:
                        tikzplotlib.save(str(ellipse_plot_path.with_suffix(".tex")))
                        mlflow.log_artifact(
                            str(ellipse_plot_path.with_suffix(".tex")),
                            artifact_path="post_processing",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save TikZ plot: {e}")
                    logger.info(f"Ellipse/polytope plot saved to {ellipse_plot_path}")

                    # Log to MLflow
                    mlflow.log_figure(fig, str(ellipse_plot_path.with_suffix(".png")))
                    plt.close(fig)
                    logger.info("Ellipse/polytope plot logged to MLflow")
                else:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    count_stable, count_unstable = 0, 0
                    xs_np = xs.cpu().detach().numpy()
                    for x_hat, c in zip(xs_np, cs):
                        M = warmup_steps + 100
                        if np.any(c > 0):
                            ax.plot(x_hat[warmup_steps, 0], x_hat[warmup_steps, 1], "rx")
                            ax.plot(x_hat[warmup_steps:M, 0], x_hat[warmup_steps:M, 1], '--')
                            count_unstable += 1
                        else:
                            ax.plot(x_hat[warmup_steps, 0], x_hat[warmup_steps, 1], "go")
                            ax.plot(x_hat[warmup_steps:M, 0], x_hat[warmup_steps:M, 1])
                            count_stable += 1
                    logger.info(f'total:{b}, stable: {count_stable}, count unstable: {count_unstable}')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel(r"$x_1$", fontsize=12)
                    ax.set_ylabel(r"$x_2$", fontsize=12)
                    mlflow.log_figure(fig, str(ellipse_plot_path.with_suffix(".png")))
                    tikzplotlib.save(str(ellipse_plot_path.with_suffix(".tex")))
                    mlflow.log_artifact(
                        str(ellipse_plot_path.with_suffix(".tex")), artifact_path="post_processing"
                    )
                    plt.close(fig)

            except Exception as e:
                logger.warning(f"Failed to generate ellipse/polytope plot: {e}")

        # plot some prediction for handpicked trajectories
        logger.info(f"Generating prediction plot for trajectory...")
        try:
            # import matplotlib.pyplot as plt
            import tikzplotlib

            from sysid.utils import plot_predictions

            pred_plot_path = output_dir / f"prediction_trajectory_post_{args.run_id[:8]}.png"

            fig, axes = plot_predictions(
                output_dir=output_dir,
                e_hat=e_hat,
                e=test_outputs,
                num_samples=3,
                # sample_indices=UNSTAB_STAB_ZERO,
                save_path=pred_plot_path,
                return_axes=True,
                warmup_steps=warmup_steps,
            )

            try:
                mlflow.log_figure(fig, str(pred_plot_path.with_suffix(".png")))
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

        logger.info(f"✓ Post-processing complete! Results saved to run: {args.run_id}")


if __name__ == "__main__":
    main()
