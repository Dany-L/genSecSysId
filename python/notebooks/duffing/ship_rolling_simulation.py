"""Ship-rolling demo: true dynamics vs. learned constrained model.

Inputs come from two Duffing CSV files (columns u, q, q_dot): one stable
trajectory and one unstable one. The same `u` sequence drives both the true
dynamics (taken straight from the CSV's q, q_dot columns — this is the
ground-truth simulation) and the learned model, loaded by training-run id
(config + checkpoint + normalizer all resolve from the standard layout
under ~/genSecSysId-Data). Four side-by-side columns:

  - True dynamics  (stable input)
  - True dynamics  (unstable input)
  - Learned model  (stable input)
  - Learned model  (unstable input)

Each column animates a tilting ship, a time-series, and the phase portrait.

Run:
  python python/notebooks/duffing/ship_rolling_simulation.py \\
      --stable-csv   notebooks/duffing/datasets/Duffing/test/zero_conv_000.csv \\
      --unstable-csv notebooks/duffing/datasets/Duffing/test_div/zero_div_000.csv \\
      --run-id       5296a077a5074cf9b9cab0ca56fdfa0c
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Polygon

REPO_PY = Path(__file__).resolve().parents[2]  # .../python
if str(REPO_PY / "src") not in sys.path:
    sys.path.insert(0, str(REPO_PY / "src"))

from sysid.config import resolve_run_artifacts  # noqa: E402
from sysid.data import DataNormalizer  # noqa: E402
from sysid.evaluation.true_dynamics import DUFFING_TS, DUFFING_U_C  # noqa: E402
from sysid.models import load_model  # noqa: E402

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


def load_learned_model(run_id, data_root=DEFAULT_DATA_ROOT, device="cpu"):
    """Resolve a run id and return (model, normalizer).

    Uses the shared sysid.config.resolve_run_artifacts helper, which reads
    the per-run YAML with a restricted SafeLoader subclass (only ``!!python/tuple``
    is recognised, mapped to a list) rather than ``yaml.full_load`` — so a
    tampered config can't construct arbitrary Python objects.
    """
    config, model_path, normalizer_path, _ = resolve_run_artifacts(
        run_id, data_root=data_root
    )
    model = load_model(str(model_path), config, device=device)
    model.eval()
    normalizer = (
        DataNormalizer.load(str(normalizer_path)) if normalizer_path is not None else None
    )
    return model, normalizer


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
    ax.contour(PHI, DPHI, H_grid, levels=[0.25],
               colors="red", linewidths=1.1, linestyles="--", zorder=2)

    ax.axvline(phi_v_deg, color="red", ls=":", lw=0.6, alpha=0.5)
    ax.axvline(-phi_v_deg, color="red", ls=":", lw=0.6, alpha=0.5)
    ax.plot(0, 0, "o", color="green", markersize=7, zorder=5)
    ax.plot([phi_v_deg, -phi_v_deg], [0, 0], "x", color="red",
            markersize=10, mew=2.0, zorder=5)

    phi_full_deg = np.rad2deg(np.clip(phi_v * q_full, -np.pi, np.pi))
    dphi_full_deg = np.clip(np.rad2deg(phi_v * dq_full), -dphi_max_deg, dphi_max_deg)
    ax.plot(phi_full_deg, dphi_full_deg, color=color, lw=0.9, alpha=0.35, zorder=3)

    (trail,) = ax.plot([], [], color=color, lw=2.0, alpha=0.95, zorder=6)
    (dot,) = ax.plot([], [], "o", color=color, markersize=9, zorder=7,
                     markeredgecolor="black", markeredgewidth=0.8)
    return dot, trail, dphi_max_deg


def setup_timeseries_axis(ax, t, phi_deg, u_seq, T_total, phi_v_deg):
    ax.set_xlim(0, T_total)
    ax.set_ylim(-1.4 * phi_v_deg, 1.4 * phi_v_deg)
    ax.axhline(phi_v_deg, color="red", ls="--", lw=0.8,
               label=f"$\\pm\\varphi_v = \\pm{phi_v_deg:.0f}^\\circ$")
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stable-csv", type=Path, required=True,
                        help="CSV with a converging trajectory (columns u, q, q_dot).")
    parser.add_argument("--unstable-csv", type=Path, required=True,
                        help="CSV with a diverging trajectory (columns u, q, q_dot).")
    parser.add_argument("--run-id", type=str, required=True,
                        help="MLflow training-run id. Config, checkpoint, and "
                             "normalizer are resolved from <data-root>/outputs|models/"
                             "<model_type>/<run_id>/.")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT),
                        help=f"Base directory for run artefacts (default: {DEFAULT_DATA_ROOT}).")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save the animation to this path (.gif or .mp4).")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip plt.show(); useful when only saving.")
    parser.add_argument("--phi-v-deg", type=float, default=60.0,
                        help="Angle of vanishing stability in degrees (visual scaling).")
    args = parser.parse_args()

    phi_v = np.deg2rad(args.phi_v_deg)
    phi_v_deg = args.phi_v_deg
    Ts = DUFFING_TS

    # 1. Load CSV inputs and the CSV's q, q_dot as ground truth.
    u_st, q_st_true, dq_st_true = load_csv(args.stable_csv)
    u_un, q_un_true, dq_un_true = load_csv(args.unstable_csv)

    # 2. Load model + normalizer from the run id.
    model, normalizer = load_learned_model(
        args.run_id, data_root=args.data_root, device="cpu"
    )

    # 3. Run the model on each input (each sequence keeps its own length).
    #    Seed x0 from the CSV's first row so the model isn't penalised by an
    #    initial-state mismatch when the CSV doesn't start from rest.
    x0_st = np.array([q_st_true[0], dq_st_true[0]])
    x0_un = np.array([q_un_true[0], dq_un_true[0]])
    q_st_hat, dq_st_hat = run_learned_model(model, u_st, normalizer, x0=x0_st)
    q_un_hat, dq_un_hat = run_learned_model(model, u_un, normalizer, x0=x0_un)

    # 4. Pad shorter sequences to the longest length so the animation runs
    #    continuously. Once a trajectory's CSV ends, the ship freezes on its
    #    final state — the unstable trajectory dies early when the simulator
    #    hits the divergence threshold.
    n_max = max(len(u_st), len(u_un))
    t = np.arange(n_max) * Ts
    T_total = n_max * Ts

    columns = [
        {
            "label": f"True dynamics — stable\n{Path(args.stable_csv).name}",
            "color": "#5a7a3f",
            "u": pad_1d(u_st, n_max),
            "q": pad_1d(q_st_true, n_max),
            "dq": pad_1d(dq_st_true, n_max),
        },
        {
            "label": f"True dynamics — unstable\n{Path(args.unstable_csv).name}",
            "color": "#7a3f3f",
            "u": pad_1d(u_un, n_max),
            "q": pad_1d(q_un_true, n_max),
            "dq": pad_1d(dq_un_true, n_max),
        },
        {
            "label": f"Learned model — stable\n{Path(args.stable_csv).name}",
            "color": "#3f5a7a",
            "u": pad_1d(u_st, n_max),
            "q": pad_1d(q_st_hat, n_max),
            "dq": pad_1d(dq_st_hat, n_max),
        },
        {
            "label": f"Learned model — unstable\n{Path(args.unstable_csv).name}",
            "color": "#7a5a3f",
            "u": pad_1d(u_un, n_max),
            "q": pad_1d(q_un_hat, n_max),
            "dq": pad_1d(dq_un_hat, n_max),
        },
    ]

    # Resample to a fixed animation rate (interpolation from Ts=0.05s).
    fps = 30
    t_anim = np.arange(0.0, T_total, 1.0 / fps)
    for col in columns:
        col["phi"] = np.clip(phi_v * col["q"], -np.pi, np.pi)
        col["dphi"] = phi_v * col["dq"]
        col["u_anim"] = np.interp(t_anim, t, col["u"])
        col["phi_anim"] = np.interp(t_anim, t, col["phi"])
        col["dphi_anim"] = np.interp(t_anim, t, col["dphi"])

    # 5. Figure layout: 3 rows × 4 cols.
    fig = plt.figure(figsize=(20, 11.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[3, 2, 3],
                          hspace=0.55, wspace=0.30)
    for c, col in enumerate(columns):
        ax_ship = fig.add_subplot(gs[0, c])
        ax_ts = fig.add_subplot(gs[1, c])
        ax_ph = fig.add_subplot(gs[2, c])
        setup_ship_axis(ax_ship, col["label"])
        col["art"] = make_ship_artists(ax_ship, color=col["color"])
        setup_timeseries_axis(ax_ts, t, np.rad2deg(col["phi"]), col["u"],
                              T_total, phi_v_deg)
        col["cursor"] = ax_ts.axvline(0.0, color="grey", lw=1.0)
        col["dot"], col["trail"], col["dphi_clip"] = setup_phase_axis(
            ax_ph, col["q"], col["dq"], phi_v, phi_v_deg, col["color"], title="",
        )
        col["txt"] = ax_ship.text(
            0.02, 0.97, "", transform=ax_ship.transAxes,
            va="top", ha="left", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85),
        )

    fig.suptitle(
        r"Softening Duffing as ship rolling: true dynamics vs. learned model"
        r"   ($\varphi = \varphi_v\,q$)",
        fontsize=12,
    )

    trail_window = int(2.0 * fps)  # ~2 s of recent history

    def status_str(phi, u, t_now):
        deg = np.rad2deg(phi)
        mark = "CAPSIZED" if abs(deg) >= phi_v_deg else "rolling "
        return f"t = {t_now:5.2f} s\nphi = {deg:+6.1f} deg\nu   = {u:+5.2f}\n{mark}"

    def update_phase(dot, trail, phi_arr, dphi_arr, i, dphi_clip):
        phi_now = np.rad2deg(phi_arr[i])
        dphi_now = float(np.clip(np.rad2deg(dphi_arr[i]), -dphi_clip, dphi_clip))
        dot.set_data([phi_now], [dphi_now])
        i0 = max(0, i - trail_window)
        seg_phi = np.rad2deg(phi_arr[i0:i + 1])
        seg_dphi = np.clip(np.rad2deg(dphi_arr[i0:i + 1]), -dphi_clip, dphi_clip)
        trail.set_data(seg_phi, seg_dphi)

    def init():
        for col in columns:
            update_ship(col["art"], 0.0)
            col["cursor"].set_xdata([0.0, 0.0])
            col["txt"].set_text("")
            col["dot"].set_data([], [])
            col["trail"].set_data([], [])
        return ()

    def animate(i):
        for col in columns:
            phi_now = col["phi_anim"][i]
            u_now = col["u_anim"][i]
            update_ship(col["art"], phi_now)
            col["cursor"].set_xdata([t_anim[i], t_anim[i]])
            col["txt"].set_text(status_str(phi_now, u_now, t_anim[i]))
            update_phase(col["dot"], col["trail"],
                         col["phi_anim"], col["dphi_anim"],
                         i, col["dphi_clip"])
        return ()

    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t_anim), interval=1000.0 / fps, blit=False,
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if args.save.suffix.lower() == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps)
        print(f"Saving animation to {args.save} ...")
        anim.save(str(args.save), writer=writer, dpi=110)
        print("Done.")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
