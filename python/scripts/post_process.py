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
from sysid.evaluation import get_true_dynamics, list_true_dynamics
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



def _simulate(model, u, x0, warmup_steps):
    """Run model dynamics for diagnostic plots.

    For SimpleLureSafe we explicitly bypass the safety filter so that the
    constraint margin c reflects raw behavior (the filter would otherwise
    prevent any violation by construction, making the plots uninformative).
    """
    if isinstance(model, SimpleLureSafe):
        return model.forward_unfiltered(u, x0)
    return model(u, x0, warmup_steps=warmup_steps)


def _make_lp_noise(rng, T, amp_max, f_cut=2.0, order=4, Ts=0.05):
    """Butterworth-LP-filtered white noise, peak-normalised to ``amp_max``.

    Mirrors the input excitation used during data generation in
    ``scripts/duffing/duffing_benchmark.ipynb`` so regional verification stays
    consistent with how training trajectories were created.
    """
    from scipy.signal import butter, filtfilt

    b, a = butter(order, f_cut / (0.5 / Ts), btype="low")
    pad = 4 * order
    noise = rng.standard_normal(T + pad)
    u = filtfilt(b, a, noise)[pad:]
    peak = float(np.max(np.abs(u)))
    if peak <= 0.0:
        return u
    return (u / peak) * amp_max


def _sample_on_ellipsoid(rng, X, radius, n):
    """Sample ``n`` points ``x`` uniformly on ``{x : x^T X x = radius^2}``."""
    nx = X.shape[0]
    L = np.linalg.cholesky(X)  # X = L @ L.T (lower)
    z = rng.standard_normal((n, nx))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z *= radius
    # solve L.T @ x = z.T  =>  x = solve(L.T, z.T).T
    return np.linalg.solve(L.T, z.T).T


def _fidelity_check(
    model, normalizer, spec, run_output_dir, run_id, *,
    P, L, X, s, alpha, Ts, n_traj, horizon, rng,
):
    """Sanity-check overlap between model and true dynamics on SAFE inputs.

    Generates trajectories with small initial state (well inside the safe
    ellipse) and modest LP-filtered input (well inside the input bound), then
    overlays model and true-dynamics state trajectories so the user can verify
    the identification is faithful before reading the divergence experiments.
    """
    rad_x0 = 0.8 * s / max(alpha, 1e-12)
    amp = 0.1 * s
    x0 = _sample_on_ellipsoid(rng, X, radius=rad_x0, n=n_traj)
    u_n = np.stack(
        [_make_lp_noise(rng, horizon, amp_max=amp, Ts=Ts) for _ in range(n_traj)]
    )

    u_t = torch.tensor(u_n[..., None], dtype=torch.float64)
    x0_t = torch.tensor(x0, dtype=torch.float64)
    with torch.no_grad():
        _, (xs_model_t, _), _ = _simulate(model, u_t, x0_t, warmup_steps=0)
    xs_model = xs_model_t.cpu().detach().numpy()

    u_phys = normalizer.inverse_transform_inputs(u_n[..., None]).squeeze(-1)
    xs_true = []
    for x0_p, u_p in zip(x0, u_phys):
        X_true, _, _ = spec.simulate(x0_p, u_p)
        xs_true.append(X_true)

    rmses = []
    for x_m, x_t in zip(xs_model, xs_true):
        T = min(len(x_m), len(x_t))
        rmses.append(float(np.sqrt(np.mean((x_m[:T] - x_t[:T]) ** 2))))
    rmse_mean = float(np.mean(rmses))
    mlflow.log_metric("regional_verification/fidelity/state_rmse", rmse_mean)
    logger.info(
        f"  [fidelity] mean state RMSE (model vs true, safe regime): {rmse_mean:.4f}"
    )

    if model.nx != 2:
        return rmse_mean

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_phase = axes[0]
    H = L @ X
    plot_ellipse_and_parallelogram(
        X, H, s, None, ax=ax_phase, show=False, fill_polytope=True,
    )
    n_label = min(3, n_traj)
    for i, (x_m, x_t) in enumerate(zip(xs_model, xs_true)):
        color = f"C{i}"
        T = min(len(x_m), len(x_t))
        ax_phase.plot(
            x_m[:T, 0], x_m[:T, 1],
            color=color, lw=1.5,
            label=(f"model #{i}" if i < n_label else None),
        )
        ax_phase.plot(
            x_t[:T, 0], x_t[:T, 1],
            color=color, lw=1.0, ls="--",
            label=(f"true #{i}" if i < n_label else None),
        )
        ax_phase.plot(x_m[0, 0], x_m[0, 1], "o", color=color, ms=5)
    ax_phase.set_xlabel(r"$x_1$")
    ax_phase.set_ylabel(r"$x_2$")
    ax_phase.set_title(
        f"Fidelity check – safe regime\n"
        f"|x0|_X={rad_x0:.3g}, peak‖u_n‖={amp:.3g}, mean RMSE={rmse_mean:.4f}"
    )
    ax_phase.legend(loc="best", fontsize=8)
    ax_phase.grid(alpha=0.3)

    ax_ts = axes[1]
    for i, (x_m, x_t) in enumerate(zip(xs_model, xs_true)):
        color = f"C{i}"
        T = min(len(x_m), len(x_t))
        t = np.arange(T) * Ts
        ax_ts.plot(
            t, x_m[:T, 0], color=color, lw=1.5,
            label=(f"model #{i}" if i < n_label else None),
        )
        ax_ts.plot(
            t, x_t[:T, 0], color=color, lw=1.0, ls="--",
            label=(f"true #{i}" if i < n_label else None),
        )
    ax_ts.set_xlabel("time [s]")
    ax_ts.set_ylabel(r"$x_1$")
    ax_ts.set_title("Position vs time (solid: model, dashed: true)")
    ax_ts.legend(loc="best", fontsize=8)
    ax_ts.grid(alpha=0.3)

    plot_path = run_output_dir / f"rv_fidelity_{run_id[:8]}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    mlflow.log_figure(fig, f"regional_verification/{plot_path.name}")
    plt.close(fig)
    logger.info(f"  [fidelity] plot saved to {plot_path}")
    return rmse_mean


def regional_verification(
    model,
    normalizer,
    run_output_dir,
    run_id,
    true_dynamics_name,
    config,
    factors,
    n_traj,
    horizon,
):
    """Verify the model's regional-stability character.

    Drives the (post-processed) model with input/initial-state combinations
    that violate the learned regional constraint
    ``α² xᵀ X x + ‖u‖² ≤ s²`` (with ``X = P⁻¹``, ``α = σ(τ)``) and reports
    whether the model diverges. When ``true_dynamics_name`` is provided, the
    same trajectories are simulated through the registered ground-truth model
    and divergence agreement is logged as well.

    Two regimes are run:
      * **Input violation** — ``x0`` inside the ellipse, LP-filtered noise
        excitation with peak ``factor · s`` (``factor < 1`` is a sanity
        baseline, ``factor ≥ 1`` violates the input bound).
      * **Initial-state violation** — ``x0`` outside the ellipse (radius
        ``2 · s/α`` along the ellipse axes), modest LP-filtered excitation
        within the input bound.
    """
    nx = model.nx
    nd = getattr(model, "nd", 1)
    if nd != 1:
        logger.warning(
            f"Regional verification currently builds scalar LP-noise excitation; "
            f"model has nd={nd} input channels. Skipping."
        )
        return

    s = float(model.s.detach().cpu().numpy())
    alpha = float(torch.sigmoid(model.tau.detach()).cpu().numpy())
    P = model.P.detach().cpu().numpy()
    L = model.L.detach().cpu().numpy()
    X = np.linalg.inv(P)

    Ts = getattr(config.data, "sampling_time", 0.05)
    seed = getattr(config, "seed", 0) or 0
    rng = np.random.default_rng(seed + 17)

    # Validate true-dynamics compatibility, if requested.
    spec = None
    if true_dynamics_name is not None:
        spec = get_true_dynamics(true_dynamics_name)
        if spec.state_dim != nx:
            logger.warning(
                f"True-dynamics '{true_dynamics_name}' has state_dim={spec.state_dim} "
                f"but model.nx={nx}. Skipping ground-truth comparison."
            )
            spec = None

    # ------------------------------------------------------------------
    # Fidelity sanity-check (only when true dynamics are available):
    # confirm that on safe (non-violating) trajectories the identified
    # model overlaps the true system. If the dashed (true) and solid
    # (model) curves disagree here, the divergence comparisons below
    # cannot be interpreted as evidence of (mis)matched regional
    # stability — fix the identification first.
    # ------------------------------------------------------------------
    if spec is not None:
        try:
            _fidelity_check(
                model, normalizer, spec, run_output_dir, run_id,
                P=P, L=L, X=X, s=s, alpha=alpha, Ts=Ts,
                n_traj=min(5, n_traj), horizon=horizon, rng=rng,
            )
        except Exception as e:
            logger.warning(f"Fidelity check failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Build trajectories (normalized input space; states are physical)
    # ------------------------------------------------------------------
    in_amps = [float(f) * s for f in factors]                  # per-factor peak ‖u_n‖
    in_x0 = _sample_on_ellipsoid(rng, X, radius=0.2 * s / max(alpha, 1e-12), n=n_traj)
    in_u_per_factor = [
        np.stack([_make_lp_noise(rng, horizon, amp_max=amp, Ts=Ts) for _ in range(n_traj)])
        for amp in in_amps
    ]  # list of (n_traj, horizon)

    st_x0 = _sample_on_ellipsoid(rng, X, radius=2.5 * s / max(alpha, 1e-12), n=n_traj)
    st_u = np.stack(
        [_make_lp_noise(rng, horizon, amp_max=0.01 * s, Ts=Ts) for _ in range(n_traj)]
    )

    DIVERGE_THRESHOLD = 2

    def _run(model, u_n, x0):
        """Simulate model on (u_n, x0) and return (xs, c, diverged_flag) per traj."""
        u_t = torch.tensor(u_n, dtype=torch.float64)
        if u_t.dim() == 2:
            u_t = u_t.unsqueeze(-1)  # (B, T, 1)
        x0_t = torch.tensor(x0, dtype=torch.float64)
        with torch.no_grad():
            _, (xs, _), u_used = _simulate(model, u_t, x0_t, warmup_steps=0)
            _, c = model.get_regularization_input(u_used, xs, return_c=True)
        xs_np = xs.cpu().detach().numpy()
        c_np = c.cpu().detach().numpy()
        # Divergence: any |state| > threshold OR non-finite anywhere.
        max_abs = np.nanmax(np.abs(np.where(np.isfinite(xs_np), xs_np, np.nan)),
                            axis=(1, 2))
        any_nan = ~np.isfinite(xs_np).all(axis=(1, 2))
        diverged = (max_abs > DIVERGE_THRESHOLD) | any_nan
        return xs_np, c_np, diverged

    # Concatenate input-violation trajectories across factors for a single
    # combined plot, while keeping per-factor metrics.
    in_results = []
    for amp, u_n in zip(in_amps, in_u_per_factor):
        xs_np, c_np, diverged = _run(model, u_n, in_x0)
        in_results.append((amp, u_n, xs_np, c_np, diverged))
    st_xs, st_c, st_diverged = _run(model, st_u, st_x0)

    # ------------------------------------------------------------------
    # Optional: simulate the same (x0, u) through the true dynamics
    # ------------------------------------------------------------------
    in_true = None  # list of (true_xs (n_traj, T+1, nx), true_diverged (n_traj,))
    st_true = None
    if spec is not None:
        in_true = []
        for amp, u_n, *_ in in_results:
            u_phys = normalizer.inverse_transform_inputs(u_n[..., None]).squeeze(-1)
            xs_list, div_list = [], []
            for x0_p, u_p in zip(in_x0, u_phys):
                X_true, _, div = spec.simulate(x0_p, u_p)
                xs_list.append(X_true)
                div_list.append(div)
            in_true.append((np.array(div_list), xs_list))

        u_phys_st = normalizer.inverse_transform_inputs(st_u[..., None]).squeeze(-1)
        xs_list, div_list = [], []
        for x0_p, u_p in zip(st_x0, u_phys_st):
            X_true, _, div = spec.simulate(x0_p, u_p)
            xs_list.append(X_true)
            div_list.append(div)
        st_true = (np.array(div_list), xs_list)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    for idx, ((amp, _, _, _, diverged), factor) in enumerate(zip(in_results, factors)):
        tag = f"f{factor:.2f}".replace(".", "p")
        mlflow.log_metric(
            f"regional_verification/input/{tag}/model_diverged_frac",
            float(diverged.mean()),
        )
        if in_true is not None:
            true_div = in_true[idx][0]
            mlflow.log_metric(
                f"regional_verification/input/{tag}/true_diverged_frac",
                float(true_div.mean()),
            )
            mlflow.log_metric(
                f"regional_verification/input/{tag}/agreement",
                float((diverged == true_div).mean()),
            )
        logger.info(
            f"  [input violation, factor={factor:.2f}] model_diverged="
            f"{int(diverged.sum())}/{n_traj}"
            + (
                f", true_diverged={int(in_true[idx][0].sum())}/{n_traj}"
                if in_true is not None
                else ""
            )
        )

    mlflow.log_metric(
        "regional_verification/state/model_diverged_frac",
        float(st_diverged.mean()),
    )
    if st_true is not None:
        true_div_st, _ = st_true
        mlflow.log_metric(
            "regional_verification/state/true_diverged_frac",
            float(true_div_st.mean()),
        )
        mlflow.log_metric(
            "regional_verification/state/agreement",
            float((st_diverged == true_div_st).mean()),
        )
    logger.info(
        f"  [state violation] model_diverged={int(st_diverged.sum())}/{n_traj}"
        + (
            f", true_diverged={int(st_true[0].sum())}/{n_traj}"
            if st_true is not None
            else ""
        )
    )

    # ------------------------------------------------------------------
    # Plots (only meaningful for nx == 2)
    # ------------------------------------------------------------------
    if nx != 2:
        logger.info(
            f"Skipping regional-verification plots (nx={nx}, plots are 2D only)."
        )
        return

    N = 400
    # Combine all input-violation trajectories into one figure.
    xs_all = np.concatenate([r[2][:,:N,:] for r in in_results], axis=0)
    c_all = np.concatenate([r[3][:,:N] for r in in_results], axis=0)
    fig_in, ax_in, n_stab_in, n_unst_in = plot_safe_set_trajectories(
        P=P, L=L, s=s,
        x_traj=xs_all,
        c=c_all,
        warmup_steps=0,
        horizon=horizon,
    )
    if in_true is not None:
        labelled = False
        for (true_div, xs_list) in in_true:
            for X_true in xs_list:
                ax_in.plot(
                    X_true[:N, 0], X_true[:N, 1],
                    color="k", lw=1.0, alpha=0.5,
                    label=("true dynamics" if not labelled else None),
                )
                labelled = True
        ax_in.legend(loc="upper right", fontsize=8)
    factor_summary = ",".join(f"{f:g}" for f in factors)
    ax_in.set_title(f"Regional verification – input violation (factors {factor_summary})")
    ax_in.set_xlim(-1.5, 1.5)
    ax_in.set_ylim(-1.5, 1.5)
    in_plot = run_output_dir / f"rv_input_{run_id[:8]}.png"
    fig_in.savefig(in_plot, dpi=150, bbox_inches="tight")
    mlflow.log_figure(fig_in, f"regional_verification/{in_plot.name}")
    plt.close(fig_in)

    fig_st, ax_st, n_stab_st, n_unst_st = plot_safe_set_trajectories(
        P=P, L=L, s=s,
        x_traj=st_xs[:,:N,:],
        c=st_c,
        warmup_steps=0,
        horizon=horizon,
    )
    if st_true is not None:
        labelled = False
        for X_true in st_true[1]:
            ax_st.plot(
                X_true[:N, 0], X_true[:N, 1],
                color="k", lw=1.0, alpha=0.5,
                label=("true dynamics" if not labelled else None),
            )
            labelled = True
        ax_st.legend(loc="upper right", fontsize=8)
    ax_st.set_title("Regional verification – initial-state violation")
    ax_st.set_xlim(-3.5, 3.5)
    ax_st.set_ylim(-3.5, 3.5)
    st_plot = run_output_dir / f"rv_state_{run_id[:8]}.png"
    fig_st.savefig(st_plot, dpi=150, bbox_inches="tight")
    mlflow.log_figure(fig_st, f"regional_verification/{st_plot.name}")
    plt.close(fig_st)

    logger.info(
        f"Regional verification: input plot {in_plot.name} "
        f"({n_stab_in} stable, {n_unst_in} violating); "
        f"state plot {st_plot.name} ({n_stab_st} stable, {n_unst_st} violating)."
    )


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
        default='~/genSecSysId-Data/configs/crnn_gen-sec_duffing.yaml',
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
        default=None,
        help="Output directory for post-processed results (default: derived from config root_dir or config output_dir)",
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: uses current tracking URI)",
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
        default=[0.2],
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

    if args.output_dir is not None:
        output_dir = Path(os.path.expanduser(args.output_dir))
    elif getattr(config, "root_dir", None):
        base = Path(os.path.expanduser(config.root_dir))
        output_dir = Path(base / "outputs" / config.model.model_type)

    else:
        output_dir = Path(os.path.expanduser(config.output_dir))
    run_id = args.run_id
    run_output_dir = Path(output_dir) / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

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
        ellipse_plot_name= Path(f"ellipse_polytope_post_train_orig_{args.run_id[:8]}.png")
        ellipse_plot_path = Path(run_output_dir / ellipse_plot_name)
        fig.savefig(ellipse_plot_path, dpi=150, bbox_inches="tight")
        mlflow.log_figure(fig, f'post_processing/{ellipse_plot_name}')
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
        mlflow.log_metric("post_process/norm_P_original", summary["original"]["norm_P"])
        mlflow.log_metric("post_process/norm_L_original", summary["original"]["norm_L"])
        mlflow.log_metric("post_process/norm_H_original", summary["original"]["norm_H"])
        mlflow.log_metric("post_process/max_eig_F", summary["optimized"]["max_eig_F"])
        mlflow.log_metric("post_process/norm_P_optimized", summary["optimized"]["norm_P"])
        mlflow.log_metric("post_process/norm_L_optimized", summary["optimized"]["norm_L"])
        mlflow.log_metric("post_process/norm_H_optimized", summary["optimized"]["norm_H"])
        y_bar = summary["optimized"]["y_bar_n"] * normalizer.output_std
        mlflow.log_metric("post_process/y_bar", y_bar)

        logger.info(f"Maximum output range (y_bar) after post-processing: {y_bar}")
        # compute maximum output value from training dataset
        y_max_train = np.max(np.abs(train_outputs))
        logger.info(f"Maximum output value in training data: {y_max_train}")
        mlflow.log_metric("data/max_output_train", y_max_train)

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

    
        ellipse_plot_name = Path(f"ellipse_polytope_post_train_opt_{args.run_id[:8]}.png")
        ellipse_plot_path = Path(run_output_dir / ellipse_plot_name)
        fig.savefig(ellipse_plot_path, dpi=150, bbox_inches="tight")
        mlflow.log_figure(fig, f'post_processing/{ellipse_plot_name}')
        plt.close(fig)

        # Log parameter
        mlflow.log_param("post_processing", True)
        mlflow.log_param("post_process_eps", args.eps)

        # Save results to file
        alpha = 1/(1 + np.exp(-model.tau.cpu().detach().numpy()))  # Sigmoid of tau

        results_path = run_output_dir / f"post_processing_{args.run_id[:8]}.npz"
        np.savez(
            results_path,
            P_original=model.P.cpu().detach().numpy(),
            P_opt=result["P_opt"],
            L_original=model.L.cpu().detach().numpy() if model.learn_L else None,
            L_opt=result["L_opt"],
            s_original=summary["original"]["s"],
            s_opt=result["s_opt"],
            max_eig_F=result["max_eig_F"],
            y_bar=y_bar,
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
        # model_path = run_output_dir / f"post_processing_model_{args.run_id[:8]}.pt"
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

                ellipse_plot_name = Path(f"ellipse_polytope_post_{args.run_id[:8]}.png")
                ellipse_plot_path = Path(run_output_dir / ellipse_plot_name)

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
                    mlflow.log_figure(fig, f'post_processing/{str(ellipse_plot_name.with_suffix(".png"))}')
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
                    mlflow.log_figure(fig, f'post_processing/{ellipse_plot_name}')
                    tikzplotlib.save(str(ellipse_plot_path.with_suffix(".tex")))
                    mlflow.log_artifact(
                        str(ellipse_plot_name.with_suffix(".tex")), artifact_path="post_processing"
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

            pred_plot_name = Path(f"prediction_trajectory_post_{args.run_id[:8]}.png")
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
                run_id=args.run_id,
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
