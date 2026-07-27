"""Regional (input/initial-state) verification for post-processed Lure models.

Drives a post-processed :class:`~sysid.models.SimpleLure` with input / initial-
state combinations that violate the learned regional constraint and reports
whether the model diverges, optionally comparing against registered true
dynamics. Extracted from ``scripts/post_process.py`` so the CLI stays thin.

``simulate_model`` is the shared diagnostic rollout helper (it bypasses the
safety filter for :class:`~sysid.models.SimpleLureSafe` so constraint margins
reflect raw behaviour); ``regional_verification`` is the top-level routine.
"""

import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from ..models import SimpleLureSafe
from ..utils import plot_ellipse_and_parallelogram, plot_safe_set_trajectories
from .true_dynamics import get_true_dynamics

logger = logging.getLogger(__name__)


def simulate_model(model, u, x0, warmup_steps):
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
    model, normalizer, spec, run_output_dir, *,
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
        _, (xs_model_t, _), _ = simulate_model(model, u_t, x0_t, warmup_steps=0)
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

    plot_path = run_output_dir / "rv_fidelity.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    mlflow.log_figure(fig, f"regional_verification/{plot_path.name}")
    plt.close(fig)
    logger.info(f"  [fidelity] plot saved to {plot_path}")
    return rmse_mean


def regional_verification(
    model,
    normalizer,
    run_output_dir,
    true_dynamics_name,
    config,
    factors,
    n_traj,
    horizon,
    initial_state_scale=2.0,
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
        baseline, ``factor ≥ 1`` violates the input bound), followed by a
        zero-input tail so the state can settle before the final-state
        feasibility check.
      * **Initial-state violation** — ``x0`` outside the ellipse (radius
        ``initial_state_scale · s/α`` along the ellipse axes), modest
        LP-filtered excitation within the input bound.

    ``initial_state_scale`` sets how far outside the safe ellipse the
    initial-state-violation samples are placed (``> 1`` violates the bound;
    the analogue of ``factors`` for the input regime).
    """
    nx = model.nx
    nd = getattr(model, "nd", 1)
    if nd != 1:
        logger.warning(
            f"Regional verification currently builds scalar LP-noise excitation; "
            f"model has nd={nd} input channels. Skipping."
        )
        return

    if model.learn_L:
        s = float(model.s.detach().cpu().numpy())
    else:
        s = 5.0
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
                model, normalizer, spec, run_output_dir,
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
    # For each factor, build one LP-noise excitation per trajectory and append
    # 400 zero-input steps. The trailing zero tail lets the state settle after
    # the excitation stops, so the final-state feasibility check (c[-1] > 0)
    # distinguishes trajectories that return to the safe set from those that
    # have genuinely diverged.
    in_u_per_factor = []
    for amp in in_amps:
        us = []
        for _ in range(n_traj):
            u_i = _make_lp_noise(rng, horizon, amp_max=amp, Ts=Ts)
            u_i = np.hstack((u_i, np.zeros(400)))
            us.append(u_i)
        us = np.stack(us)
        in_u_per_factor.append(us)
    # list of (n_traj, horizon + 400)



    st_x0 = _sample_on_ellipsoid(
        rng, X, radius=initial_state_scale * s / max(alpha, 1e-12), n=n_traj
    )
    st_u = np.stack(
        [_make_lp_noise(rng, horizon, amp_max=0.01 * s, Ts=Ts) for _ in range(n_traj)]
    )

    DIVERGE_THRESHOLD = 10

    def _run(model, u_n, x0):
        """Simulate model on (u_n, x0) and return (xs, c, diverged_flag) per traj."""
        u_t = torch.tensor(u_n, dtype=torch.float64)
        if u_t.dim() == 2:
            u_t = u_t.unsqueeze(-1)  # (B, T, 1)
        x0_t = torch.tensor(x0, dtype=torch.float64)
        with torch.no_grad():
            _, (xs, _), u_used = simulate_model(model, u_t, x0_t, warmup_steps=0)
            _, c = model.get_regularization_input(u_used, xs, return_c=True)
        xs_np = xs.cpu().detach().numpy()
        c_np = c.cpu().detach().numpy()
        # Divergence: the trajectory is considered diverging if its *last* state
        # leaves the feasibility region, i.e. the constraint margin c
        # (from α² xᵀ X x + ‖u‖² ≤ s²) is positive at the final timestep.
        diverged = c_np[:, -1] > 0
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
                X_true, _, div = spec.simulate(x0_p, u_p, diverge_thresh=DIVERGE_THRESHOLD)
                xs_list.append(X_true)
                div_list.append(div)
            in_true.append((np.array(div_list), xs_list))

        u_phys_st = normalizer.inverse_transform_inputs(st_u[..., None]).squeeze(-1)
        xs_list, div_list = [], []
        for x0_p, u_p in zip(st_x0, u_phys_st):
            X_true, _, div = spec.simulate(x0_p, u_p, diverge_thresh=DIVERGE_THRESHOLD)
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

    # N = 800
    # Combine all input-violation trajectories into one figure.
    lim_xs = 1.5 * np.max(np.abs(st_xs[:,0,:]),axis=0)
    xs_all = np.concatenate([r[2][:,:,:] for r in in_results], axis=0)
    c_all = np.concatenate([r[3][:,:] for r in in_results], axis=0)
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
                    X_true[:, 0], X_true[:, 1],
                    color="k", lw=1.0, alpha=0.5,
                    label=("true dynamics" if not labelled else None),
                )
                labelled = True
        ax_in.legend(loc="upper right", fontsize=8)
    factor_summary = ",".join(f"{f:g}" for f in factors)
    ax_in.set_title(f"Regional verification – input violation (factors {factor_summary})")
    ax_in.set_xlim(-lim_xs[0], lim_xs[0])
    ax_in.set_ylim(-lim_xs[1], lim_xs[1])
    in_plot = run_output_dir / "rv_input.png"
    fig_in.savefig(in_plot, dpi=150, bbox_inches="tight")
    mlflow.log_figure(fig_in, f"regional_verification/{in_plot.name}")
    plt.close(fig_in)

    fig_st, ax_st, n_stab_st, n_unst_st = plot_safe_set_trajectories(
        P=P, L=L, s=s,
        x_traj=st_xs[:,:,:],
        c=st_c,
        warmup_steps=0,
        horizon=horizon,
    )
    if st_true is not None:
        labelled = False
        for X_true in st_true[1]:
            ax_st.plot(
                X_true[:, 0], X_true[:, 1],
                color="k", lw=1.0, alpha=0.5,
                label=("true dynamics" if not labelled else None),
            )
            labelled = True
        ax_st.legend(loc="upper right", fontsize=8)
    ax_st.set_title(
        f"Regional verification – initial-state violation (scale {initial_state_scale:g})"
    )
    ax_st.set_xlim(-lim_xs[0], lim_xs[0])
    ax_st.set_ylim(-lim_xs[1], lim_xs[1])
    st_plot = run_output_dir / "rv_state.png"
    fig_st.savefig(st_plot, dpi=150, bbox_inches="tight")
    mlflow.log_figure(fig_st, f"regional_verification/{st_plot.name}")
    plt.close(fig_st)

    logger.info(
        f"Regional verification: input plot {in_plot.name} "
        f"({n_stab_in} stable, {n_unst_in} violating); "
        f"state plot {st_plot.name} ({n_stab_st} stable, {n_unst_st} violating)."
    )
