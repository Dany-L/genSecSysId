"""Utility functions."""

import logging
import math
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """
    Get torch device.

    Args:
        device: Device string ("auto", "cuda", "cpu", "mps")

    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Device 'cuda' was requested but no CUDA GPUs are available on this node. "
            "Check GPU allocation (nvidia-smi) or use --device auto to fall back to CPU."
        )
    else:
        return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module):
    """Print model summary."""
    print("=" * 60)
    print("Model Architecture")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 60)


def torch_bmat(mat: List[List[torch.Tensor]]) -> torch.Tensor:
    mat_list = []
    for col in mat:
        mat_list.append(torch.hstack(col))
    return torch.vstack(mat_list)


def as_model_tensor(model, array, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert ``array`` to a tensor on the MODEL's device and dtype.

    Anything fed into a model has to live where the model's parameters live.
    Building it with only ``dtype=model.P.dtype`` silently works on CPU and then
    fails on GPU with "Expected all tensors to be on the same device", because
    ``torch.as_tensor`` on a numpy array always lands on the CPU — numpy arrays are
    host memory, so no factory call can infer the device from them.

    Use this at every seam where numpy data enters a model. It deliberately does
    NOT apply to the data loaders: those build CPU tensors on purpose and the
    training loop moves each batch with ``.to(device)``.

    Args:
        model: anything exposing a ``P`` parameter (the Lure models do); its device
            and dtype are the target.
        array: numpy array, tensor, or anything ``torch.as_tensor`` accepts.
        dtype: override for the model's dtype; defaults to ``model.P.dtype``.
    """
    ref = model.P
    return torch.as_tensor(
        array, dtype=ref.dtype if dtype is None else dtype, device=ref.device
    )


def max_abs_output(outputs: np.ndarray) -> float:
    """Safe output level ``y_max = max_{i,k} |y_k^(i)|`` from a dataset.

    NaN entries (padded variable-length trajectories) are ignored, so this is
    robust to the ragged batches produced by the variable-length loaders.

    The result is in the *same units as ``outputs``*: pass raw outputs for the
    physical level, or ``normalizer.transform_outputs(outputs)`` for the
    normalized level ``y_max_n`` that the model's ``C``/``P``/``s`` live in.
    """
    return float(np.nanmax(np.abs(outputs)))


def plot_safe_set_trajectories(
    P: np.ndarray,
    L: np.ndarray,
    s: float,
    x_traj: np.ndarray,
    c: np.ndarray,
    warmup_steps: int = 0,
    horizon: int = 200,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    fill_polytope: bool = True,
) -> Tuple[plt.Figure, plt.Axes, int, int]:
    """Plot 2D state trajectories color-coded by the safe-set constraint and
    overlay the safe-set ellipse + input-constraint polytope.

    Trajectories whose constraint vector ``c`` ever exceeds 0 are drawn red
    (dashed); the rest are drawn green (solid). The starting point (at
    ``warmup_steps``) is marked ``rx`` / ``go`` accordingly.

    Args:
        P: Lyapunov matrix ``(nx, nx)``. The ellipse is ``{x : (1/s²) xᵀ P⁻¹ x ≤ 1}``.
        L: Coupling matrix ``(nz, nx)``. The polytope is ``{x : ‖L P⁻¹ x‖∞ ≤ 1}``.
        s: Sector bound.
        x_traj: State trajectories ``(batch, seq_len, nx)``. Only ``nx == 2``
            is supported (the plot is 2D).
        c: Constraint values ``(batch, seq_len)`` from
            ``model.get_regularization_input(..., return_c=True)``. Trajectory
            ``i`` is unstable iff ``np.any(c[i] > 0)``.
        warmup_steps: Index where the trajectory window starts.
        horizon: Number of steps after ``warmup_steps`` to draw.
        ax: Optional pre-built axis; a new figure is created if ``None``.
        figsize: Figure size used when creating a new axis.
        fill_polytope: Forwarded to ``plot_ellipse_and_parallelogram``.

    Returns:
        ``(fig, ax, count_stable, count_unstable)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    M = warmup_steps + horizon
    count_stable = 0
    count_unstable = 0
    for c_i, x_i in zip(c, x_traj):
        if np.any(c_i > 0):
            ax.plot(x_i[warmup_steps, 0], x_i[warmup_steps, 1], "rx")
            ax.plot(x_i[warmup_steps:M, 0],x_i[warmup_steps:M, 1], "--")
            count_unstable += 1
        else:
            ax.plot(x_i[warmup_steps, 0], x_i[warmup_steps, 1], "go")
            ax.plot(x_i[warmup_steps:, 0], x_i[warmup_steps:, 1])
            count_stable += 1

    X = np.linalg.inv(P)
    H = L @ X
    fig, ax = plot_ellipse_and_parallelogram(
        X, H, s, None, ax=ax, show=False, fill_polytope=fill_polytope,
    )
    return fig, ax, count_stable, count_unstable


def plot_ellipse_and_parallelogram(
    X: np.array,
    H: np.array,
    s: float,
    max_norm_x0: Optional[float] = None,
    ax=None,
    show: bool = False,
    fill_polytope: bool = True,
):
    """Plot ellipse and parallelogram given by X'HX <= s^2.

    Args:
        X: Inverse of Lyapunov matrix P (X = P^-1)
        H: Coupling matrix (H = L @ P^-1)
        s: Sector bound
        max_norm_x0: Optional maximum norm of initial condition
        ax: Matplotlib axis to plot on (creates new if None)
        show: Whether to show the plot
        fill_polytope: Whether to fill the polytope (default: True)

    Returns:
        (fig, ax) tuple so caller can further modify the figure.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    H = np.asarray(H)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_fig = True
    else:
        fig = ax.figure

    # Polytope vertices for ||H x||_infty <= 1
    # This works for H ∈ ℝ^(m×2) where m >= 2
    plot_polytope(ax, H, fill_polytope, "r-")
    plot_ellipse(ax, X, s, "b-", fill=fill_polytope)

    # Plot ellipse on top

    # if max_norm_x0 is not None:
    #     # allow scalar or vector max_norm_x0; use its 2-norm as radius
    #     r = max_norm_x0
    #     circle_x0 = np.array([r * np.cos(theta), r * np.sin(theta)])
    #     ax.plot(circle_x0[0, :], circle_x0[1, :], 'k:', linewidth=1.5,
    #            label=r"$\|x^0\|_2 \leq x^0_{max}$")

    # ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_title("Ellipse and Polytope Regions", fontsize=14)

    if show and created_fig:
        plt.show()

    return fig, ax

def get_volume_of_ellipsoid(P: np.ndarray, s: float) -> float:
    """Compute the volume of the ellipsoid defined by (1/s^2) x^T P^-1 x <= 1.

    Args:
        P: Lyapunov matrix (nx, nx)
        s: Sector bound

    Returns:
        Volume of the ellipsoid
    """
    nx = P.shape[0]
    volume = (s**nx) * np.sqrt(np.linalg.det(P)) * (np.pi ** (nx / 2)) / math.gamma(nx / 2 + 1)
    return volume

def plot_polytope(
    ax, H: np.array, fill: bool = True, linetype: str = "r--", name: str = r"$\|Hx\|_\infty \leq 1$"
):
    if H.shape[1] == 2:
        try:
            # Each row of H defines constraints: -1 <= H_i @ x <= 1
            # This gives us 2*m linear inequalities defining a polytope
            # We'll compute the vertices of this polytope

            m = H.shape[0]
            # Create inequality matrix: A @ x <= b
            # For each row h_i of H: h_i @ x <= 1 and -h_i @ x <= 1
            A = np.vstack([H, -H])  # Shape: (2m, 2)
            b = np.ones(2 * m)

            # Find vertices by solving for intersections of constraint boundaries
            # A vertex occurs where 2 constraints are active (since we're in 2D)
            vertices = []

            for i in range(2 * m):
                for j in range(i + 1, 2 * m):
                    # Solve system where constraints i and j are active (equality)
                    M = np.array([A[i], A[j]])
                    b_eq = np.array([b[i], b[j]])

                    # Check if system is solvable (constraints not parallel)
                    if np.abs(np.linalg.det(M)) > 1e-10:
                        vertex = np.linalg.solve(M, b_eq)

                        # Check if vertex satisfies ALL constraints
                        if np.all(A @ vertex <= b + 1e-9):  # Small tolerance for numerical errors
                            vertices.append(vertex)

            if len(vertices) > 0:
                # Remove duplicates
                vertices = np.array(vertices)
                vertices = np.unique(vertices, axis=0)

                # Sort vertices by angle to get proper polygon order
                center = np.mean(vertices, axis=0)
                angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
                sorted_indices = np.argsort(angles)
                V = vertices[sorted_indices]

                # Plot filled polytope or just edges
                if fill:
                    ax.fill(
                        V[:, 0],
                        V[:, 1],
                        color=[0.8, 0.9, 1.0],
                        edgecolor="r",
                        linewidth=1.5,
                        label=name,
                        alpha=0.7,
                    )
                else:
                    # Close the polygon by adding first vertex at the end
                    V_closed = np.vstack([V, V[0]])
                    ax.plot(V_closed[:, 0], V_closed[:, 1], linetype, label=name, linewidth=1.5)
            else:
                logging.getLogger(__name__).warning(
                    "No vertices found for ||Hx||_infty <= 1 polytope."
                )

        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to plot ||Hx||_infty polytope: {e}")
    else:
        logging.getLogger(__name__).warning(
            f"H has {H.shape[1]} columns, cannot plot in 2D (need 2 columns)."
        )


def plot_ellipse(
    ax,
    X: np.array,
    s: float,
    linetype: str = "b--",
    name: str = r"$\frac{1}{s^2} x^T X x \leq 1$",
    fill: bool = False,
):
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    # map unit circle through ellipse transform
    L = np.linalg.cholesky(1 / s**2 * X)
    X_half_inv = np.linalg.inv(L.T)
    ellipse = X_half_inv @ circle

    if fill:
        ax.fill(
            ellipse[0, :],
            ellipse[1, :],
            color=[1.0, 0.8, 0.8],
            edgecolor="b",
            linewidth=1.5,
            label=name,
            alpha=0.7,
        )
    else:
        ax.plot(ellipse[0, :], ellipse[1, :], linetype, linewidth=2, label=name)


def _mask_nonfinite(arr: np.ndarray) -> np.ndarray:
    """Return a float copy with non-finite (NaN/Inf) entries set to NaN.

    matplotlib skips NaN points when drawing and autoscaling, so masking a
    diverged trajectory keeps its finite portion visible instead of letting
    Inf leak into the axis limits.
    """
    arr = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def _ensure_finite_axis_limits(axes, default: Tuple[float, float] = (-1.0, 1.0)) -> None:
    """Force finite limits on any axis left with no finite data.

    When a model diverges, a whole trajectory can be non-finite, leaving the
    axis with no data to autoscale from. matplotlib then produces non-finite
    limits and older versions crash inside the tick locator
    (``MaxNLocator._raw_ticks``: ``np.nonzero(steps >= raw_step)[0][0]`` on an
    empty array). Plotting must never abort a training run, so we clamp such
    axes to a finite default. Axes that already have finite data are untouched.
    """
    for ax in axes:
        x0, y0, x1, y1 = ax.dataLim.extents  # [xmin, ymin, xmax, ymax]
        if not np.isfinite([x0, x1]).all():
            ax.set_xlim(*default)
        if not np.isfinite([y0, y1]).all():
            ax.set_ylim(*default)


def plot_predictions(
    output_dir: str,
    e_hat: np.ndarray,  # predicted output
    e: np.ndarray,  # output (target)
    d: Optional[np.ndarray] = None,  # input
    num_samples: int = 5,
    sample_indices: Optional[list] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_axes: bool = False,
    warmup_steps: int = 0,
) -> plt.Axes:
    """
    Plot predictions vs targets and optionally inputs.

    Args:
        e_hat: Predicted output values
        e: Output (target) values
        d: Input values (optional)
        num_samples: Number of samples to plot (used if sample_indices is None)
        sample_indices: Specific sample indices to plot (overrides num_samples)
        save_path: Path to save figure
        warmup_steps: Number of steps to warm up before plotting
    """
    if sample_indices is not None:
        indices = sample_indices
        num_samples = len(indices)
    else:
        num_samples = min(num_samples, e_hat.shape[0])
        indices = list(range(num_samples))

    # Determine number of subplots
    num_plots = 2 if d is not None else 1

    if ax is None:
        fig, axes = plt.subplots(num_samples, num_plots, figsize=(12 * num_plots, 3 * num_samples))
    else:
        axes = ax

    if num_samples == 1:
        axes = axes.reshape(1, -1) if num_plots > 1 else [[axes]]
    elif num_plots == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_samples):
        # Use the actual index from the indices list
        idx = indices[i]

        # Plot predictions vs targets
        ax_pred = axes[i, 0] if num_plots > 1 else axes[i, 0]

        # dashes vertical line at warmup_steps if specified
        ax_pred.axvline(x=warmup_steps, color="k", linestyle=":", label="Warmup Steps")

        if e_hat.ndim == 3:
            # Sequence data (unused: seq_len = e_hat.shape[1])
            for feat in range(e_hat.shape[2]):
                ax_pred.plot(
                    _mask_nonfinite(e_hat[idx, :, feat]),
                    label=f"e_hat (predicted output, feat {feat})", alpha=0.7
                )
                ax_pred.plot(
                    _mask_nonfinite(e[idx, :, feat]),
                    label=f"e (output, feat {feat})", linestyle="--", alpha=0.7
                )
        else:
            # Single-step data
            ax_pred.plot(_mask_nonfinite(e_hat[idx]), label="e_hat (predicted output)", alpha=0.7)
            ax_pred.plot(_mask_nonfinite(e[idx]), label="e (output)", linestyle="--", alpha=0.7)

        ax_pred.set_xlabel("Time Step")
        ax_pred.set_ylabel("Output (e)")
        ax_pred.set_title(f"Sample {idx}: Output Prediction")
        ax_pred.legend()
        ax_pred.grid(True, alpha=0.3)

        # Plot inputs if provided
        if d is not None:
            ax_input = axes[i, 1]

            if d.ndim == 3:
                # Sequence data
                for feat in range(d.shape[2]):
                    ax_input.plot(_mask_nonfinite(d[idx, :, feat]),
                                  label=f"d (input, feat {feat})", alpha=0.7)
            else:
                ax_input.plot(_mask_nonfinite(d[idx]), label="d (input)", alpha=0.7)

            ax_input.set_xlabel("Time Step")
            ax_input.set_ylabel("Input (d)")
            ax_input.set_title(f"Sample {idx}: Input Signal")
            ax_input.legend()
            ax_input.grid(True, alpha=0.3)

    # Guard against fully non-finite trajectories leaving an axis with no data:
    # otherwise tight_layout() crashes in the tick locator on older matplotlib.
    _ensure_finite_axis_limits(np.atleast_1d(np.asarray(axes, dtype=object)).ravel())

    plt.tight_layout()

    if save_path is None:
        save_path = output_dir / "predictions_plot.png"

    if return_axes:
        return fig, axes
    else:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # print(f"Predictions plot saved to {save_path}")
