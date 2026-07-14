"""Ship-rolling demo: true dynamics vs. learned constrained model.

Inputs come from two Duffing CSV files (columns u, q, q_dot): one stable
trajectory and one unstable one. The same `u` sequence drives both the true
dynamics (taken straight from the CSV's q, q_dot columns — this is the
ground-truth simulation) and, optionally, the learned model, loaded by
training-run id (config + checkpoint + normalizer all resolve from the standard
layout under ~/genSecSysId-Data, or from a remote MLflow server via
``--mlflow-uri``).

With a ``--run-id`` there are four side-by-side columns:

  - True dynamics  (stable input)
  - True dynamics  (unstable input)
  - Learned model  (stable input)
  - Learned model  (unstable input)

Without a ``--run-id`` only the two true-dynamics columns are shown.

Each column animates a tilting ship, a time-series, and the phase portrait.
The capsize boundary is the model's certified output bound ``y_bar`` (from
post_process, or ``--y-bar``); the display is rescaled so that bound sits at
``--phi-v-deg`` degrees. When a model is loaded, its safe-set ellipse and
input-constraint polytope are overlaid on the learned-model phase plots.

Rendering can be slow. Full CSVs are ~4000 steps: ``--n-steps`` (clip length,
optionally offset by ``--start-step``) trims the trajectory before anything is
simulated, which is the biggest win. On top of that, ``--speed`` (compress
playback time), ``--fps`` (fewer frames), and ``--dpi`` (lower resolution) all
make videos/gifs faster to produce and smaller on disk. mp4 (FFMpeg) saves
much faster than .gif.

Run (true dynamics vs. learned model):
  python python/notebooks/duffing/ship_rolling_simulation.py \\
      --stable-csv   notebooks/duffing/datasets/Duffing/test/zero_conv_000.csv \\
      --unstable-csv notebooks/duffing/datasets/Duffing/test_div/zero_div_000.csv \\
      --run-id       5296a077a5074cf9b9cab0ca56fdfa0c

Run (model from a remote MLflow server):
  python python/notebooks/duffing/ship_rolling_simulation.py \\
      --stable-csv   notebooks/duffing/datasets/Duffing/test/zero_conv_000.csv \\
      --unstable-csv notebooks/duffing/datasets/Duffing/test_div/zero_div_000.csv \\
      --run-id       5296a077a5074cf9b9cab0ca56fdfa0c \\
      --mlflow-uri   http://mlflowui.informatik.uni-stuttgart.de/

Run (true dynamics only, short + fast gif):
  python python/notebooks/duffing/ship_rolling_simulation.py \\
      --stable-csv   notebooks/duffing/datasets/Duffing/test/zero_conv_000.csv \\
      --unstable-csv notebooks/duffing/datasets/Duffing/test_div/zero_div_000.csv \\
      --save out.gif --no-show --n-steps 500 --speed 4 --fps 20

Run (one video per subplot, e.g. for a slide deck):
  python python/notebooks/duffing/ship_rolling_simulation.py \\
      --stable-csv   notebooks/duffing/datasets/Duffing/test/zero_conv_000.csv \\
      --unstable-csv notebooks/duffing/datasets/Duffing/test_div/zero_div_000.csv \\
      --run-id       5296a077a5074cf9b9cab0ca56fdfa0c \\
      --split-dir    out/panels --no-show --n-steps 200 --speed 3
  # -> out/panels/{true,learned}_{stable,unstable}_{ship,timeseries,phase}.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import (
    FuncAnimation,
    PillowWriter,
    FFMpegWriter,
    adjusted_figsize,
)
from matplotlib.patches import Polygon

REPO_PY = Path(__file__).resolve().parents[2]  # .../python
if str(REPO_PY / "src") not in sys.path:
    sys.path.insert(0, str(REPO_PY / "src"))

from sysid.config import (  # noqa: E402
    resolve_run_artifacts,
    resolve_run_artifacts_mlflow,
)
from sysid.data import DataNormalizer  # noqa: E402
from sysid.evaluation.true_dynamics import DUFFING_TS, DUFFING_U_C  # noqa: E402
from sysid.models import load_model  # noqa: E402
from sysid.utils import plot_ellipse, plot_polytope  # noqa: E402

DEFAULT_DATA_ROOT = "~/genSecSysId-Data"


def load_csv(path):
    """Read u, q, q_dot columns from a Duffing dataset CSV."""
    df = pd.read_csv(path)
    return (
        df["u"].to_numpy(dtype=float),
        df["q"].to_numpy(dtype=float),
        df["q_dot"].to_numpy(dtype=float),
    )


def pad_1d(arr, n):
    """Pad a 1D array to length n by repeating its last value."""
    k = len(arr)
    if k >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - k, arr[-1])])


def pad_zeros(arr, n):
    """Pad a 1D array to length n with trailing zeros (or truncate to n).

    Used for the *input* ``u``: when a diverging trajectory's CSV ends early,
    the forcing goes to zero rather than freezing at its last value, so the
    learned model keeps evolving under zero input (cf. compare_run_ids.ipynb,
    which appends ``np.zeros`` to the diverging input before the forward pass).
    """
    k = len(arr)
    if k >= n:
        return arr[:n]
    return np.concatenate([arr, np.zeros(n - k)])


def window_slice(n_len, start, n_steps):
    """Slice object for a window of a length-``n_len`` trajectory.

    ``start`` is the first sample to keep (>= 0); ``n_steps`` is how many
    samples to keep (``None`` keeps everything from ``start`` to the end).
    Use it to cut long (e.g. 4000-step) trajectories down to a shorter clip
    so the animation is quick to build and render. Raises ``ValueError`` on
    an out-of-range ``start`` or a non-positive ``n_steps``.
    """
    if start < 0:
        raise ValueError("start-step must be >= 0")
    if start >= n_len:
        raise ValueError(
            f"start-step {start} is beyond the trajectory length {n_len}"
        )
    if n_steps is None:
        return slice(start, n_len)
    if n_steps <= 0:
        raise ValueError("n-steps must be positive")
    return slice(start, start + n_steps)


def load_learned_model(run_id, data_root=DEFAULT_DATA_ROOT, device="cpu",
                       mlflow_uri=None):
    """Resolve a run id and return (model, normalizer).

    When ``mlflow_uri`` is given, artefacts (config, checkpoint, normalizer)
    are downloaded from that MLflow tracking server via
    sysid.config.resolve_run_artifacts_mlflow — use it for models that live on
    a remote server. Otherwise they are resolved from the local ``data_root``
    layout via sysid.config.resolve_run_artifacts.

    Both helpers read the per-run YAML with a restricted SafeLoader subclass
    (only ``!!python/tuple`` is recognised, mapped to a list) rather than
    ``yaml.full_load`` — so a tampered config can't construct arbitrary Python
    objects.
    """
    if mlflow_uri is not None:
        config, model_path, normalizer_path, _ = resolve_run_artifacts_mlflow(
            run_id, mlflow_uri
        )
    else:
        config, model_path, normalizer_path, _ = resolve_run_artifacts(
            run_id, data_root=data_root
        )
    model = load_model(str(model_path), config, device=device)
    model.eval()
    normalizer = (
        DataNormalizer.load(str(normalizer_path)) if normalizer_path is not None else None
    )
    return model, normalizer


def _output_std(normalizer):
    """Physical scale of the model output (1.0 if unnormalized)."""
    if normalizer is None or getattr(normalizer, "output_std", None) is None:
        return 1.0
    return float(np.asarray(normalizer.output_std).ravel()[0])


def extract_certificate(model, normalizer):
    """Return the model's stability/safety certificate, or None.

    Calls ``model.post_process()`` — which re-optimises and writes back the
    Lyapunov certificate (``P``, ``L``, ``s``, ``la``) but leaves the forward
    predictions unchanged — and reads:

      - ``y_bar``: the certified output bound |y| ≤ y_bar in *physical* units
        (``y_bar_n * output_std``), used as the capsize boundary. When the
        output-bound SDP is infeasible (``y_bar_n`` ≤ 0) we fall back to the
        closed form ``s·√(C P Cᵀ)`` and set ``y_bar_reliable`` to False.
      - ``y_bar_reliable``: False when ``y_bar`` came from that infeasible-SDP
        fallback (it scales with ``s`` and may be vacuously large, so it should
        not be used to rescale the display); True otherwise.
      - ``P``, ``s``, ``L``: the safe-set matrices for the phase-plot overlay,
        in the model's physical (q, q̇) state coordinates.

    Returns None for models without a certificate (plain RNN/LSTM/GRU) or if
    the certificate SDP fails outright (no ``summary`` in the result).
    """
    if not hasattr(model, "post_process"):
        return None
    try:
        result = model.post_process()
    except Exception:
        return None
    if not result.get("success") or "summary" not in result:
        return None

    P = model.P.detach().cpu().numpy()
    s = float(model.s.detach().cpu().numpy())
    L = model.L.detach().cpu().numpy()

    y_bar_n = result["summary"]["optimized"].get("y_bar_n")
    y_bar_reliable = True
    if y_bar_n is None or y_bar_n <= 0:
        # Output-bound SDP was infeasible; use the exact closed form instead.
        # It scales with s, which the certificate repair may have pinned at its
        # cap, so the bound can be vacuously large — flag it as unreliable so
        # callers don't rescale the display by it (which collapses it to ~0°).
        C = model.C.detach().cpu().numpy()
        y_bar_n = float(s * np.sqrt(float((C @ P @ C.T).ravel()[0])))
        y_bar_reliable = False
    y_bar = y_bar_n * _output_std(normalizer)

    return {"y_bar": y_bar, "P": P, "s": s, "L": L,
            "y_bar_reliable": y_bar_reliable}


def choose_capsize_scale(y_bar_override, cert):
    """Pick the capsize boundary ``q_cap`` (physical q units) and its label.

    The display is rescaled so this boundary sits at ``--phi-v-deg`` degrees
    (φ = phi_v·q with ``phi_v = deg2rad(phi_v_deg) / q_cap``). Priority:

      1. an explicit ``--y-bar`` (``y_bar_override``);
      2. the model's certified bound ``cert["y_bar"]`` — but only when it is a
         usable number. A bound from an infeasible output-bound SDP
         (``y_bar_reliable`` False) comes from the closed form ``s·√(C P Cᵀ)``,
         which scales with an ``s`` the certificate repair may have pinned at
         its cap; it is then vacuously large and would rescale every trajectory
         to ~0° (an "empty"-looking animation), so it is skipped;
      3. otherwise the physical hilltop ``q = 1``.

    Returns ``(q_cap, boundary_label)``; ``boundary_label`` is ``None`` to keep
    the default ``±φ_v`` time-series label.
    """
    if y_bar_override is not None:
        return y_bar_override, rf"$\pm\bar y = \pm{y_bar_override:.2f}$"
    if (cert is not None and cert["y_bar"] > 0
            and cert.get("y_bar_reliable", True)):
        return cert["y_bar"], rf"capsize $\bar y = {cert['y_bar']:.2f}$"
    return 1.0, None


def run_learned_model(model, u_seq, normalizer, x0=None, device="cpu"):
    """Run the learned model on a 1D input sequence and return predicted
    (q, q_dot) in physical units.

    Normalizes the input, runs the model with no warmup skipping, then
    denormalizes the output. The internal state's second component is
    taken as q_dot_hat — for the constrained Lure-type model the state is
    [q, q_dot] in physical units. When provided, `x0` is the initial state
    in physical units (shape (2,)); pass it for trajectories that don't
    start from rest so the model output isn't dominated by the catch-up
    transient.
    """
    u = np.asarray(u_seq, dtype=np.float64).reshape(1, -1, 1)
    if normalizer is not None:
        u = normalizer.transform_inputs(u)
    d = torch.from_numpy(u).to(device)
    x0_tensor = None
    if x0 is not None:
        x0_tensor = torch.from_numpy(
            np.asarray(x0, dtype=np.float64).reshape(1, -1)
        ).to(device)
    with torch.no_grad():
        e_hat, (x, _w), _ = model(d, x0_tensor, warmup_steps=0)
    e_hat_np = e_hat.cpu().numpy()
    if normalizer is not None:
        e_hat_np = normalizer.inverse_transform_outputs(e_hat_np)
    q_hat = e_hat_np[0, :, 0]
    # The state array includes the post-final-step value, so it is one
    # sample longer than the output. Trim to align with q_hat.
    x_np = x.cpu().numpy()
    q_dot_hat = x_np[0, : q_hat.shape[0], 1]
    return q_hat, q_dot_hat


def rotate(points, phi):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T


def make_ship_artists(ax, color):
    hull_w_top = 1.6
    hull_w_bot = 0.7
    hull_h = 0.5
    base_hull = np.array(
        [
            [-hull_w_bot / 2, 0.0],
            [hull_w_bot / 2, 0.0],
            [hull_w_top / 2, hull_h],
            [-hull_w_top / 2, hull_h],
        ]
    )
    hull_poly = Polygon(
        base_hull, closed=True, facecolor=color, edgecolor="black", lw=1.5, zorder=3
    )
    ax.add_patch(hull_poly)

    mast_h = 1.4
    base_mast = np.array([[0.0, hull_h], [0.0, hull_h + mast_h]])
    (mast_line,) = ax.plot(
        base_mast[:, 0], base_mast[:, 1], color="black", lw=2.5, zorder=4
    )

    base_flag = np.array(
        [
            [0.0, hull_h + mast_h],
            [0.5, hull_h + mast_h - 0.15],
            [0.0, hull_h + mast_h - 0.30],
        ]
    )
    flag_poly = Polygon(
        base_flag, closed=True, facecolor="crimson", edgecolor="crimson", zorder=4
    )
    ax.add_patch(flag_poly)

    return hull_poly, mast_line, flag_poly, base_hull, base_mast, base_flag


def update_ship(art, phi):
    hull_poly, mast_line, flag_poly, base_hull, base_mast, base_flag = art
    hull_poly.set_xy(rotate(base_hull, phi))
    mast_xy = rotate(base_mast, phi)
    mast_line.set_data(mast_xy[:, 0], mast_xy[:, 1])
    flag_poly.set_xy(rotate(base_flag, phi))


def setup_ship_axis(ax, title):
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.2, 2.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.axhspan(-1.2, 0.0, color="#a8d0e6", zorder=0)
    ax.axhspan(0.0, 2.6, color="#f0f7ff", zorder=0)
    xs = np.linspace(-2.5, 2.5, 200)
    ax.plot(xs, 0.05 * np.sin(2 * xs), color="#3a7ca5", lw=0.8, zorder=1)


def hamiltonian(q, dq):
    """Mechanical energy H = KE + V with V(q) = q^2/2 - q^4/4."""
    return 0.5 * dq ** 2 + 0.5 * q ** 2 - 0.25 * q ** 4


def setup_phase_axis(ax, q_full, dq_full, phi_v, phi_v_deg, color, title):
    phi_max_deg = 1.5 * phi_v_deg
    dphi_max_deg = 1.6 * phi_v_deg
    ax.set_xlim(-phi_max_deg, phi_max_deg)
    ax.set_ylim(-dphi_max_deg, dphi_max_deg)
    ax.set_xlabel(r"$\varphi$ (deg)", fontsize=9)
    ax.set_ylabel(r"$\dot\varphi$ (deg/s)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)

    phi_grid = np.linspace(-phi_max_deg, phi_max_deg, 220)
    dphi_grid = np.linspace(-dphi_max_deg, dphi_max_deg, 220)
    PHI, DPHI = np.meshgrid(phi_grid, dphi_grid)
    Q = np.deg2rad(PHI) / phi_v
    DQ = np.deg2rad(DPHI) / phi_v
    H_grid = hamiltonian(Q, DQ)
    bg_levels = np.array([-0.5, -0.25, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0])
    ax.contour(PHI, DPHI, H_grid, levels=bg_levels,
               colors="lightgrey", linewidths=0.6, alpha=0.7, zorder=1)
    # ax.contour(PHI, DPHI, H_grid, levels=[0.25],
    #            colors="red", linewidths=1.1, linestyles="--", zorder=2)

    # ax.axvline(phi_v_deg, color="red", ls=":", lw=0.6, alpha=0.5)
    # ax.axvline(-phi_v_deg, color="red", ls=":", lw=0.6, alpha=0.5)
    ax.plot(0, 0, "o", color="green", markersize=7, zorder=5)
    # ax.plot([phi_v_deg, -phi_v_deg], [0, 0], "x", color="red",
    #         markersize=10, mew=2.0, zorder=5)

    phi_full_deg = np.rad2deg(np.clip(phi_v * q_full, -np.pi, np.pi))
    dphi_full_deg = np.clip(np.rad2deg(phi_v * dq_full), -dphi_max_deg, dphi_max_deg)
    ax.plot(phi_full_deg, dphi_full_deg, color=color, lw=0.9, alpha=0.35, zorder=3)

    (trail,) = ax.plot([], [], color=color, lw=2.0, alpha=0.95, zorder=6)
    (dot,) = ax.plot([], [], "o", color=color, markersize=9, zorder=7,
                     markeredgecolor="black", markeredgewidth=0.8)
    return dot, trail, dphi_max_deg


def draw_certificate_region(ax, cert, phi_v):
    """Overlay the model's certified safe set on a phase axis (in degrees).

    The safe set is ``{x : (1/s²) xᵀ P⁻¹ x ≤ 1}`` and the input-constraint
    polytope is ``{x : ‖L P⁻¹ x‖∞ ≤ 1}`` — both in the model's physical
    (q, q̇) state coordinates. The phase axis shows those coordinates scaled to
    degrees by ``k = rad2deg(phi_v)`` (φ = phi_v·q), so the ellipse maps with
    ``s → s·k`` and the polytope with ``H → H/k``. Only the constrained
    (Lure-type) models have a certificate; true-dynamics columns pass None.
    """
    if cert is None:
        return
    X = np.linalg.inv(cert["P"])
    k = np.rad2deg(phi_v)
    plot_polytope(ax, (cert["L"] @ X) / k, fill=False, linetype="m-",
                  name=r"input constr. $\|Lx\|_\infty \leq 1$")
    plot_ellipse(ax, X, cert["s"] * k, linetype="b-",
                 name=r"safe set $\frac{1}{s^2}x^\top P^{-1}x \leq 1$", fill=False)
    ax.legend(loc="lower right", fontsize=6, framealpha=0.7)


def setup_timeseries_axis(ax, t, phi_deg, u_seq, T_total, phi_v_deg,
                          boundary_label=None):
    if boundary_label is None:
        boundary_label = f"$\\pm\\varphi_v = \\pm{phi_v_deg:.0f}^\\circ$"
    ax.set_xlim(0, T_total)
    ax.set_ylim(-1.4 * phi_v_deg, 1.4 * phi_v_deg)
    ax.axhline(phi_v_deg, color="red", ls="--", lw=0.8, label=boundary_label)
    ax.axhline(-phi_v_deg, color="red", ls="--", lw=0.8)
    ax.plot(t, phi_deg, color="black", lw=1.2, label=r"$\varphi(t)$")
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("roll angle $\\varphi$ (deg)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(t, u_seq, color="steelblue", lw=0.8, alpha=0.8)
    ax2.axhline(DUFFING_U_C, color="steelblue", ls=":", lw=0.7, alpha=0.6)
    ax2.axhline(-DUFFING_U_C, color="steelblue", ls=":", lw=0.7, alpha=0.6)
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_ylabel("input $u$", color="steelblue", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=8)
    return ax2


def ensure_even_frame_size(fig, dpi):
    """Snap ``fig`` to render at ``dpi`` with even pixel width/height.

    The H.264 encoder requires even dimensions; an odd figure size makes
    ffmpeg abort with "height not divisible by 2". We fix that by adjusting
    the *figure* size (matplotlib's own approach, via ``adjusted_figsize``)
    rather than an ffmpeg ``-vf pad`` filter. Padding the piped rawvideo
    shifts every frame by a row, so the encoded clip slowly rolls vertically.
    Setting the figure dpi to the save dpi keeps the declared frame size and
    the rendered buffer in lock-step. Returns the new (w_in, h_in).
    """
    fig.set_dpi(dpi)
    w_in, h_in = adjusted_figsize(*fig.get_size_inches(), dpi, 2)
    fig.set_size_inches(w_in, h_in, forward=True)
    return w_in, h_in


def frame_times(T_total, fps, speed):
    """Simulation time sampled at the animation frame rate.

    Each frame advances ``speed / fps`` seconds of simulation, so a ``speed``
    of s produces a video s times shorter (and ~s times faster to render)
    while ``fps`` fixes the playback smoothness. Returns a 1D array of times.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if speed <= 0:
        raise ValueError("speed must be positive")
    return np.arange(0.0, T_total, speed / fps)


def status_str(phi, u, t_now, phi_v_deg):
    """One-line status text shown on a ship axis."""
    deg = np.rad2deg(phi)
    mark = "CAPSIZED" if abs(deg) >= phi_v_deg else "rolling "
    return f"t = {t_now:5.2f} s\nphi = {deg:+6.1f} deg\nu   = {u:+5.2f}\n{mark}"


def _update_phase(dot, trail, phi_arr, dphi_arr, i, dphi_clip, trail_window):
    phi_now = np.rad2deg(phi_arr[i])
    dphi_now = float(np.clip(np.rad2deg(dphi_arr[i]), -dphi_clip, dphi_clip))
    dot.set_data([phi_now], [dphi_now])
    i0 = max(0, i - trail_window)
    seg_phi = np.rad2deg(phi_arr[i0:i + 1])
    seg_dphi = np.clip(np.rad2deg(dphi_arr[i0:i + 1]), -dphi_clip, dphi_clip)
    trail.set_data(seg_phi, seg_dphi)


# The three subplot kinds each column is made of, in top-to-bottom order.
SUBPLOT_KINDS = ("ship", "timeseries", "phase")


def build_subplot(kind, ax, col, ctx):
    """Set up one subplot (``kind`` in :data:`SUBPLOT_KINDS`) for ``col`` on
    ``ax`` and return an ``update(i)`` callable that advances it to frame ``i``.

    ``ctx`` carries the shared animation context: ``t``, ``T_total``,
    ``t_anim``, ``phi_v``, ``phi_v_deg``, ``boundary_label`` and
    ``trail_window``. The same builder drives both the combined figure and the
    per-subplot split videos, so they stay pixel-for-pixel consistent.
    """
    if kind == "ship":
        setup_ship_axis(ax, col["label"])
        art = make_ship_artists(ax, color=col["color"])
        txt = ax.text(
            0.02, 0.97, "", transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85),
        )

        def update(i):
            phi_now = col["phi_anim"][i]
            update_ship(art, phi_now)
            txt.set_text(status_str(phi_now, col["u_anim"][i],
                                    ctx["t_anim"][i], ctx["phi_v_deg"]))
        return update

    if kind == "timeseries":
        setup_timeseries_axis(ax, ctx["t"], np.rad2deg(col["phi"]), col["u"],
                              ctx["T_total"], ctx["phi_v_deg"],
                              boundary_label=ctx["boundary_label"])
        cursor = ax.axvline(0.0, color="grey", lw=1.0)

        def update(i):
            cursor.set_xdata([ctx["t_anim"][i], ctx["t_anim"][i]])
        return update

    if kind == "phase":
        dot, trail, dphi_clip = setup_phase_axis(
            ax, col["q"], col["dq"], ctx["phi_v"], ctx["phi_v_deg"],
            col["color"], title="",
        )
        draw_certificate_region(ax, col.get("cert"), ctx["phi_v"])

        def update(i):
            _update_phase(dot, trail, col["phi_anim"], col["dphi_anim"],
                          i, dphi_clip, ctx["trail_window"])
        return update

    raise ValueError(f"unknown subplot kind: {kind!r}")


def save_animation(anim, fig, path, fps, dpi):
    """Save an animation to ``path`` (.gif via Pillow, else mp4 via FFMpeg).

    For mp4 the figure is snapped to an even pixel size first (H.264 needs
    even width/height; padding the video instead makes it roll vertically).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        ensure_even_frame_size(fig, dpi)
        writer = FFMpegWriter(fps=fps)
    anim.save(str(path), writer=writer, dpi=dpi)


def render_split_videos(columns, ctx, out_dir, fmt, fps, dpi):
    """Render each subplot as its own video into ``out_dir``.

    One file per (column, kind) named ``<tag>_<kind>.<fmt>`` (e.g.
    ``learned_unstable_phase.mp4``), so the panels can be arranged individually
    (e.g. in a slide deck). Returns the list of written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figsizes = {"ship": (5.0, 4.5), "timeseries": (6.0, 3.4), "phase": (5.0, 5.0)}
    n_frames = len(ctx["t_anim"])
    paths = []
    total = len(columns) * len(SUBPLOT_KINDS)
    for col in columns:
        for kind in SUBPLOT_KINDS:
            fig, ax = plt.subplots(figsize=figsizes[kind])
            update = build_subplot(kind, ax, col, ctx)
            fig.tight_layout()

            def animate(i, _update=update):
                _update(i)
                return ()

            anim = FuncAnimation(fig, animate, frames=n_frames,
                                 interval=1000.0 / fps, blit=False)
            path = out_dir / f"{col['tag']}_{kind}.{fmt}"
            print(f"  [{len(paths) + 1}/{total}] {path.name} ...")
            save_animation(anim, fig, path, fps, dpi)
            paths.append(path)
            plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stable-csv", type=Path, required=True,
                        help="CSV with a converging trajectory (columns u, q, q_dot).")
    parser.add_argument("--unstable-csv", type=Path, required=True,
                        help="CSV with a diverging trajectory (columns u, q, q_dot).")
    parser.add_argument("--run-id", type=str, default=None,
                        help="MLflow training-run id. Config, checkpoint, and "
                             "normalizer are resolved from <data-root>/outputs|models/"
                             "<model_type>/<run_id>/ (or from --mlflow-uri if set). "
                             "If omitted, only the true dynamics (from the CSVs) "
                             "are shown.")
    parser.add_argument("--mlflow-uri", type=str, default=None,
                        help="MLflow tracking URI (e.g. http://host/ or "
                             "file:///path/to/mlruns). When set, the --run-id "
                             "artefacts are downloaded from this server instead "
                             "of the local --data-root layout — use it for models "
                             "on a remote server.")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT),
                        help=f"Base directory for run artefacts (default: {DEFAULT_DATA_ROOT}).")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save the combined animation to this path (.gif or .mp4).")
    parser.add_argument("--split-dir", type=Path, default=None,
                        help="Also save each subplot as its own video into this "
                             "directory, named <tag>_<kind>.<ext> (e.g. "
                             "learned_unstable_phase.mp4), so the panels can be "
                             "arranged individually, e.g. in a slide deck. "
                             "Produces 3xN videos (ship, roll time-series, phase "
                             "per column).")
    parser.add_argument("--split-format", type=str, default="mp4",
                        choices=["mp4", "gif"],
                        help="Container/format for --split-dir videos (default: mp4).")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip plt.show(); useful when only saving.")
    parser.add_argument("--phi-v-deg", type=float, default=60.0,
                        help="Display roll angle (deg) shown AT the capsize "
                             "boundary (default: 60). The boundary itself is the "
                             "certified bound y_bar (or --y-bar); without a model "
                             "it is the physical hilltop q=1.")
    parser.add_argument("--y-bar", type=float, default=None,
                        help="Certified output bound |y| <= y_bar in physical q "
                             "units, used as the capsize boundary. Overrides the "
                             "value from the model's post_process(). The phase "
                             "plots also show the model's safe-set ellipse and "
                             "input-constraint polytope.")
    parser.add_argument("--start-step", type=int, default=0,
                        help="First CSV sample to include (default: 0). Combine "
                             "with --n-steps to animate a range [start, start+n).")
    parser.add_argument("--n-steps", "--max-steps", dest="n_steps", type=int,
                        default=None,
                        help="Number of samples to keep from --start-step "
                             "(the clip length). Trajectories are ~4000 steps; "
                             "e.g. --n-steps 500 animates a short window. "
                             "Default: through the end.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Animation frame rate. Lower renders faster and "
                             "yields smaller files (default: 30).")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed-up factor: e.g. 4 plays 4x faster, "
                             "so the video is 4x shorter and renders ~4x quicker "
                             "(default: 1.0 = real time).")
    parser.add_argument("--dpi", type=int, default=110,
                        help="Resolution for saved animations. Lower renders "
                             "faster and yields smaller files (default: 110).")
    args = parser.parse_args(argv)

    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    phi_v_deg = args.phi_v_deg
    Ts = DUFFING_TS

    # 1. Load CSV inputs and the CSV's q, q_dot as ground truth.
    u_st, q_st_true, dq_st_true = load_csv(args.stable_csv)
    u_un, q_un_true, dq_un_true = load_csv(args.unstable_csv)

    # 1b. Optionally window the trajectories: full CSVs are ~4000 steps, which
    #     makes for a long, slow animation. --start-step / --n-steps cut each
    #     one down to the same clip before the model is run.
    try:
        sl_st = window_slice(len(u_st), args.start_step, args.n_steps)
        sl_un = window_slice(len(u_un), args.start_step, args.n_steps)
    except ValueError as exc:
        parser.error(str(exc))
    u_st, q_st_true, dq_st_true = u_st[sl_st], q_st_true[sl_st], dq_st_true[sl_st]
    u_un, q_un_true, dq_un_true = u_un[sl_un], q_un_true[sl_un], dq_un_true[sl_un]

    # 2. Optionally load model + normalizer from the run id. Without a run id
    #    we fall back to the true dynamics only.
    model = normalizer = cert = None
    if args.run_id is not None:
        model, normalizer = load_learned_model(
            args.run_id, data_root=args.data_root, device="cpu",
            mlflow_uri=args.mlflow_uri,
        )
        cert = extract_certificate(model, normalizer)

    # 2b. Capsize boundary. Rescale the display so the boundary sits at the
    #     configured angle (--phi-v-deg): φ = phi_v·q maps q = q_cap → phi_v_deg.
    #     q_cap is the certified output bound ȳ (from the model, or --y-bar),
    #     falling back to the physical hilltop q = 1 when no model is loaded or
    #     the model's bound is degenerate (see choose_capsize_scale).
    q_cap, boundary_label = choose_capsize_scale(args.y_bar, cert)
    if (args.y_bar is None and cert is not None
            and not cert.get("y_bar_reliable", True)):
        print(
            f"Warning: the model's certified output bound is degenerate "
            f"(y_bar={cert['y_bar']:.3g} from an infeasible SDP with "
            f"s={cert['s']:.3g}); it would rescale every trajectory to ~0°. "
            f"Using the physical hilltop q=1 as the capsize boundary instead — "
            f"pass --y-bar to set it explicitly.",
            file=sys.stderr,
        )
    phi_v = np.deg2rad(phi_v_deg) / q_cap

    # 3. Pad shorter sequences to the longest length so the animation runs
    #    continuously. The *input* u is padded with zeros — a diverging
    #    trajectory dies early (its CSV ends when the simulator hits the
    #    divergence threshold), and after that the forcing is off rather than
    #    frozen at its last value. The true-dynamics q/q_dot freeze at their
    #    final state (we have no ground-truth simulator to continue them), but
    #    the learned model is run on the zero-padded input below so it keeps
    #    evolving to the end of the clip.
    n_max = max(len(u_st), len(u_un))
    t = np.arange(n_max) * Ts
    T_total = n_max * Ts

    u_st_full = pad_zeros(u_st, n_max)
    u_un_full = pad_zeros(u_un, n_max)

    columns = [
        {
            "tag": "true_stable",
            "label": f"True dynamics — stable\n{Path(args.stable_csv).name}",
            "color": "#5a7a3f",
            "u": u_st_full,
            "q": pad_1d(q_st_true, n_max),
            "dq": pad_1d(dq_st_true, n_max),
        },
        {
            "tag": "true_unstable",
            "label": f"True dynamics — unstable\n{Path(args.unstable_csv).name}",
            "color": "#7a3f3f",
            "u": u_un_full,
            "q": pad_1d(q_un_true, n_max),
            "dq": pad_1d(dq_un_true, n_max),
        },
    ]

    # 4. If a model was loaded, run it on the zero-padded inputs (full clip
    #    length) and add the learned-model columns. Seed x0 from the CSV's
    #    first row so the model isn't penalised by an initial-state mismatch
    #    when the CSV doesn't start from rest.
    if model is not None:
        x0_st = np.array([q_st_true[0], dq_st_true[0]])
        x0_un = np.array([q_un_true[0], dq_un_true[0]])
        q_st_hat, dq_st_hat = run_learned_model(model, u_st_full, normalizer, x0=x0_st)
        q_un_hat, dq_un_hat = run_learned_model(model, u_un_full, normalizer, x0=x0_un)
        columns += [
            {
                "tag": "learned_stable",
                "label": f"Learned model — stable\n{Path(args.stable_csv).name}",
                "color": "#3f5a7a",
                "u": u_st_full,
                "q": pad_1d(q_st_hat, n_max),
                "dq": pad_1d(dq_st_hat, n_max),
                "cert": cert,
            },
            {
                "tag": "learned_unstable",
                "label": f"Learned model — unstable\n{Path(args.unstable_csv).name}",
                "color": "#7a5a3f",
                "u": u_un_full,
                "q": pad_1d(q_un_hat, n_max),
                "dq": pad_1d(dq_un_hat, n_max),
                "cert": cert,
            },
        ]

    # Resample to a fixed animation rate (interpolation from Ts=0.05s). A
    # --speed of s advances s*Ts of simulation time per frame, so the video is
    # s times shorter (and renders ~s times faster).
    fps = args.fps
    t_anim = frame_times(T_total, fps, args.speed)
    for col in columns:
        col["phi"] = np.clip(phi_v * col["q"], -np.pi, np.pi)
        col["dphi"] = phi_v * col["dq"]
        col["u_anim"] = np.interp(t_anim, t, col["u"])
        col["phi_anim"] = np.interp(t_anim, t, col["phi"])
        col["dphi_anim"] = np.interp(t_anim, t, col["dphi"])

    # Shared animation context consumed by build_subplot for both the combined
    # figure and the per-subplot split videos.
    # ~2 s of recent simulation history; each frame covers speed/fps sim-seconds.
    trail_window = max(1, int(2.0 * fps / args.speed))
    ctx = {
        "t": t, "T_total": T_total, "t_anim": t_anim,
        "phi_v": phi_v, "phi_v_deg": phi_v_deg,
        "boundary_label": boundary_label, "trail_window": trail_window,
    }

    # 5a. Optionally render each subplot as its own video (for slide decks).
    if args.split_dir is not None:
        print(f"Rendering {len(columns) * len(SUBPLOT_KINDS)} split videos "
              f"to {args.split_dir} ...")
        render_split_videos(columns, ctx, args.split_dir, args.split_format,
                            fps, args.dpi)
        print("Done.")

    # 5b. Combined figure: 3 rows × one column per trajectory (2 without a
    #     model, 4 with a learned model).
    n_cols = len(columns)
    fig = plt.figure(figsize=(5.0 * n_cols, 11.5))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[3, 2, 3],
                          hspace=0.55, wspace=0.30)
    updates = []
    for c, col in enumerate(columns):
        # Create the three axes first (ship, time-series, phase) so the input
        # twin axis — created while building the time-series — is appended
        # after the phase axis, keeping fig.axes in [ship, ts, ph, twin] order.
        axes = [fig.add_subplot(gs[r, c]) for r in range(3)]
        for kind, ax in zip(SUBPLOT_KINDS, axes):
            updates.append(build_subplot(kind, ax, col, ctx))

    fig.suptitle(
        r"Softening Duffing as ship rolling: true dynamics vs. learned model"
        r"   ($\varphi = \varphi_v\,q$)",
        fontsize=12,
    )

    def animate(i):
        for update in updates:
            update(i)
        return ()

    # Only build the combined animation when it will actually be rendered
    # (saved or shown) — otherwise it would be created and immediately garbage
    # collected (e.g. in split-only runs), which matplotlib warns about.
    anim = None
    if args.save is not None or not args.no_show:
        anim = FuncAnimation(
            fig, animate, frames=len(t_anim), interval=1000.0 / fps, blit=False,
        )

    if args.save is not None:
        print(f"Saving animation to {args.save} ...")
        save_animation(anim, fig, args.save, fps, args.dpi)
        print("Done.")

    if not args.no_show:
        plt.show()

    # Keep a handle to the animation on the figure so it isn't garbage
    # collected before rendering, and so callers/tests can inspect the result.
    fig._ship_anim = anim
    return fig


if __name__ == "__main__":
    main()
