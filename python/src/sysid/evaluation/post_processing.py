"""Post-processing diagnostics for Lure models (input-condition check + plots).

Extracted from ``scripts/post_process.py`` so the CLI stays thin:

* :func:`check_input_condition` — roll the model out on a set of inputs and count
  how many trajectories breach the learned input constraint, optionally saving a
  2D safe-set plot.
* :func:`plot_post_process_trajectories` — the ellipse/polytope safe-set
  trajectory figure (with a TikZ export) for the post-processed model.

Both save their figures to ``run_output_dir`` and log them to the active MLflow
run under the ``post_processing/`` artifact path.
"""

import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from ..utils import plot_safe_set_trajectories
from .regional_verification import simulate_model

logger = logging.getLogger(__name__)


def check_input_condition(
    model,
    inputs_n,
    x0,
    warmup_steps,
    run_output_dir,
    tag,
    title,
    *,
    horizon=200,
):
    """Roll the model out on ``inputs_n`` and count input-constraint violations.

    A trajectory counts as 'unstable' iff its constraint margin
    ``c_k = ‖u_k‖² − s² + α² xₖᵀ P⁻¹ xₖ`` ever exceeds 0. When ``nx == 2`` a
    safe-set trajectory plot ``ellipse_polytope_<tag>.png`` is saved to
    ``run_output_dir`` and logged to MLflow. For ``SimpleLureSafe``,
    :func:`simulate_model` bypasses the safety filter so the margin reflects the
    raw (unprotected) behaviour.

    Args:
        model: the (post-processed) SimpleLure model.
        inputs_n: normalized input trajectories ``(B, N, nd)``.
        x0: initial states ``(B, nx)``.
        warmup_steps: leading steps excluded from the constraint check.
        run_output_dir: directory the plot is written to.
        tag: short slug used in the plot file name and metric grouping.
        title: human-readable label for the log line.
        horizon: number of steps to draw in the safe-set plot.

    Returns:
        ``(n_stable, n_unstable)``.
    """
    b = inputs_n.shape[0]
    with torch.no_grad():
        _, (x_hat, _), u_safe = simulate_model(model, inputs_n, x0, warmup_steps)
        _, c = model.get_regularization_input(
            u_safe, x_hat, return_c=True, warmup_steps=warmup_steps
        )
    c_np = c.cpu().detach().numpy()
    n_unstable = int(np.any(c_np > 0, axis=1).sum())
    n_stable = b - n_unstable
    logger.info(f"{title}: total={b}, stable={n_stable}, unstable={n_unstable}")

    if model.nx == 2:
        fig, _, _, _ = plot_safe_set_trajectories(
            P=model.P.cpu().detach().numpy(),
            L=model.L.cpu().detach().numpy(),
            s=model.s.cpu().detach().numpy(),
            x_traj=x_hat.cpu().detach().numpy(),
            c=c_np,
            warmup_steps=warmup_steps,
            horizon=horizon,
        )
        plot_name = f"ellipse_polytope_{tag}.png"
        fig.savefig(run_output_dir / plot_name, dpi=150, bbox_inches="tight")
        mlflow.log_figure(fig, f"post_processing/{plot_name}")
        plt.close(fig)

    return n_stable, n_unstable


def plot_post_process_trajectories(
    model,
    inputs_n,
    x_traj,
    run_output_dir,
    config,
    *,
    warmup_steps=0,
    horizon=100,
    name="ellipse_polytope_post",
    figsize=(8, 8),
):
    """Ellipse/polytope safe-set trajectory plot for the post-processed model.

    Computes the input-constraint margin ``c`` for the given (normalized) inputs
    and state trajectories and renders the 2D safe set (ellipse + input polytope)
    with trajectories coloured by violation. When the model does not use the
    custom regularization (no learned ``L``/polytope) a plain state-space scatter
    is drawn instead. Saves ``<name>.png`` (plus a TikZ ``.tex``) to
    ``run_output_dir`` and logs both to MLflow. Only meaningful for ``nx == 2``.

    Returns:
        ``(count_stable, count_unstable)``.
    """
    _, cs = model.get_regularization_input(inputs_n, x_traj, return_c=True)
    cs = cs.cpu().detach().numpy()
    xs_np = x_traj.cpu().detach().numpy()
    b = xs_np.shape[0]
    plot_path = run_output_dir / f"{name}.png"

    if model.learn_L and config.training.use_custom_regularization:
        fig, _, count_stable, count_unstable = plot_safe_set_trajectories(
            P=model.P.cpu().detach().numpy(),
            L=model.L.cpu().detach().numpy(),
            s=model.s.cpu().detach().numpy(),
            x_traj=xs_np,
            c=cs,
            warmup_steps=warmup_steps,
            horizon=horizon,
            figsize=figsize,
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        count_stable, count_unstable = 0, 0
        M = warmup_steps + horizon
        for x_hat, c in zip(xs_np, cs):
            if np.any(c > 0):
                ax.plot(x_hat[warmup_steps, 0], x_hat[warmup_steps, 1], "rx")
                ax.plot(x_hat[warmup_steps:M, 0], x_hat[warmup_steps:M, 1], "--")
                count_unstable += 1
            else:
                ax.plot(x_hat[warmup_steps, 0], x_hat[warmup_steps, 1], "go")
                ax.plot(x_hat[warmup_steps:M, 0], x_hat[warmup_steps:M, 1])
                count_stable += 1
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"$x_1$", fontsize=12)
        ax.set_ylabel(r"$x_2$", fontsize=12)

    logger.info(f"total: {b}, stable: {count_stable}, unstable: {count_unstable}")

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    mlflow.log_figure(fig, f"post_processing/{plot_path.name}")
    try:
        import tikzplotlib

        tikzplotlib.save(str(plot_path.with_suffix(".tex")))
        mlflow.log_artifact(
            str(plot_path.with_suffix(".tex")), artifact_path="post_processing"
        )
    except Exception as e:
        logger.warning(f"Failed to save TikZ plot: {e}")
    plt.close(fig)
    logger.info(f"Ellipse/polytope plot saved to {plot_path}")

    return count_stable, count_unstable
