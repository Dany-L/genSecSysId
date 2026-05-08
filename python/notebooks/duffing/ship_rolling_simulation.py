"""Ship-rolling demo using the softening Duffing oscillator.

Maps the dimensionless Duffing state q to a roll angle phi = phi_v * q,
where phi_v is the angle of vanishing stability (capsize boundary).
Two side-by-side animations:
  - left:  small wave forcing -> stable rolling within the basin
  - right: a strong pulse     -> trajectory crosses the hilltop -> capsize

Run:
  python python/notebooks/duffing/ship_rolling_simulation.py
  python python/notebooks/duffing/ship_rolling_simulation.py \\
      --save python/notebooks/duffing/figs/ship_rolling.gif --no-show
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Polygon

REPO_PY = Path(__file__).resolve().parents[2]  # .../python
if str(REPO_PY / "src") not in sys.path:
    sys.path.insert(0, str(REPO_PY / "src"))

from sysid.evaluation.true_dynamics import (  # noqa: E402
    DUFFING_DELTA_D,
    DUFFING_TS,
    DUFFING_U_C,
    duffing_dt,
)


def simulate(x0, u_seq, Ts=DUFFING_TS, delta_d=DUFFING_DELTA_D, q_clip=4.0):
    """Step the Duffing system forward. Stops early once |q| > q_clip
    so the trajectory stays bounded for visualization (the polynomial
    blows up super-exponentially past the hilltop)."""
    X = [np.asarray(x0, dtype=float)]
    for u in u_seq:
        x_next = duffing_dt(X[-1], u=float(u), Ts=Ts, delta_d=delta_d)
        X.append(x_next)
        if not np.all(np.isfinite(x_next)) or abs(x_next[0]) > q_clip:
            break
    return np.asarray(X)


def build_inputs(T_total, Ts):
    n = int(round(T_total / Ts))
    t = np.arange(n) * Ts

    u_stable = 0.25 * np.sin(0.4 * t)

    u_capsize = np.zeros_like(t)
    pulse_mask = (t >= 4.0) & (t < 6.0)
    u_capsize[pulse_mask] = 1.5

    return t, u_stable, u_capsize


def pad_to_length(traj, n):
    k = traj.shape[0]
    if k >= n:
        return traj[:n]
    pad = np.tile(traj[-1], (n - k, 1))
    return np.vstack([traj, pad])


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


def setup_phase_axis(ax, X_traj, phi_v, phi_v_deg, color, title):
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

    phi_full_deg = np.rad2deg(np.clip(phi_v * X_traj[:, 0], -np.pi, np.pi))
    dphi_full_deg = np.rad2deg(phi_v * X_traj[:, 1])
    dphi_full_deg = np.clip(dphi_full_deg, -dphi_max_deg, dphi_max_deg)
    ax.plot(phi_full_deg, dphi_full_deg, color=color, lw=0.9, alpha=0.35, zorder=3)

    (trail,) = ax.plot([], [], color=color, lw=2.0, alpha=0.95, zorder=6)
    (dot,) = ax.plot([], [], "o", color=color, markersize=9, zorder=7,
                     markeredgecolor="black", markeredgewidth=0.8)
    return dot, trail, dphi_max_deg


def setup_timeseries_axis(ax, t, phi_deg, u_seq, T_total, phi_v_deg, label):
    ax.set_xlim(0, T_total)
    ax.set_ylim(-1.4 * phi_v_deg, 1.4 * phi_v_deg)
    ax.axhline(phi_v_deg, color="red", ls="--", lw=0.8,
               label=f"$\\pm\\varphi_v = \\pm{phi_v_deg:.0f}^\\circ$")
    ax.axhline(-phi_v_deg, color="red", ls="--", lw=0.8)
    ax.plot(t, phi_deg, color="black", lw=1.2, label=f"roll angle $\\varphi(t)$ ({label})")
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("roll angle $\\varphi$ (deg)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(t, u_seq, color="steelblue", lw=0.8, alpha=0.8)
    ax2.axhline(DUFFING_U_C, color="steelblue", ls=":", lw=0.7, alpha=0.6)
    ax2.axhline(-DUFFING_U_C, color="steelblue", ls=":", lw=0.7, alpha=0.6)
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_ylabel("input $u$ (wave moment)", color="steelblue", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=8)
    return ax2


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--save", type=Path, default=None,
                        help="Save the animation to this path (.gif or .mp4).")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip plt.show(); useful when only saving.")
    parser.add_argument("--phi-v-deg", type=float, default=60.0,
                        help="Angle of vanishing stability in degrees (visual scaling).")
    parser.add_argument("--T-total", type=float, default=30.0,
                        help="Total simulated time (seconds).")
    args = parser.parse_args()

    phi_v = np.deg2rad(args.phi_v_deg)
    phi_v_deg = args.phi_v_deg
    Ts = DUFFING_TS

    t, u_stable, u_capsize = build_inputs(args.T_total, Ts)
    n = len(t)

    X_stable = simulate((0.3, 0.0), u_stable, Ts=Ts)
    X_capsize = simulate((0.0, 0.0), u_capsize, Ts=Ts)

    X_stable = pad_to_length(X_stable, n + 1)
    X_capsize = pad_to_length(X_capsize, n + 1)

    phi_stable = np.clip(phi_v * X_stable[:n, 0], -np.pi, np.pi)
    phi_capsize = np.clip(phi_v * X_capsize[:n, 0], -np.pi, np.pi)
    dphi_stable = phi_v * X_stable[:n, 1]
    dphi_capsize = phi_v * X_capsize[:n, 1]

    fps = 30
    t_anim = np.arange(0.0, args.T_total, 1.0 / fps)
    phi_stable_anim = np.interp(t_anim, t, phi_stable)
    phi_capsize_anim = np.interp(t_anim, t, phi_capsize)
    dphi_stable_anim = np.interp(t_anim, t, dphi_stable)
    dphi_capsize_anim = np.interp(t_anim, t, dphi_capsize)
    u_stable_anim = np.interp(t_anim, t, u_stable)
    u_capsize_anim = np.interp(t_anim, t, u_capsize)

    fig = plt.figure(figsize=(11, 11.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 2, 3],
                          hspace=0.45, wspace=0.22)
    ax_ship_L = fig.add_subplot(gs[0, 0])
    ax_ship_R = fig.add_subplot(gs[0, 1])
    ax_ts_L = fig.add_subplot(gs[1, 0])
    ax_ts_R = fig.add_subplot(gs[1, 1])
    ax_ph_L = fig.add_subplot(gs[2, 0])
    ax_ph_R = fig.add_subplot(gs[2, 1])

    setup_ship_axis(ax_ship_L,
                    f"Stable rolling   peak $|u|=0.25 < u_c={DUFFING_U_C:.3f}$")
    setup_ship_axis(ax_ship_R,
                    f"Capsize   pulse $u=1.5 > u_c={DUFFING_U_C:.3f}$  ($t\\in[4,6]$ s)")

    art_L = make_ship_artists(ax_ship_L, color="#5a7a3f")
    art_R = make_ship_artists(ax_ship_R, color="#7a3f3f")

    setup_timeseries_axis(ax_ts_L, t, np.rad2deg(phi_stable), u_stable,
                          args.T_total, phi_v_deg, "stable")
    setup_timeseries_axis(ax_ts_R, t, np.rad2deg(phi_capsize), u_capsize,
                          args.T_total, phi_v_deg, "capsize")

    cursor_L = ax_ts_L.axvline(0.0, color="grey", lw=1.0)
    cursor_R = ax_ts_R.axvline(0.0, color="grey", lw=1.0)

    dot_L, trail_L, dphi_max_L = setup_phase_axis(
        ax_ph_L, X_stable[:n], phi_v, phi_v_deg, "#5a7a3f",
        r"Phase space (stable). Red dashed: $H = 1/4$ (saddle level)",
    )
    dot_R, trail_R, dphi_max_R = setup_phase_axis(
        ax_ph_R, X_capsize[:n], phi_v, phi_v_deg, "#7a3f3f",
        r"Phase space (capsize). Trajectory exits the basin",
    )
    trail_window = int(2.0 * fps)  # ~2 s of recent history

    txt_L = ax_ship_L.text(
        0.02, 0.97, "", transform=ax_ship_L.transAxes,
        va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85),
    )
    txt_R = ax_ship_R.text(
        0.02, 0.97, "", transform=ax_ship_R.transAxes,
        va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85),
    )

    fig.suptitle(
        r"Softening Duffing as ship rolling: $\ddot q = -\delta\dot q - q + q^3 + u$,"
        r"   $\varphi = \varphi_v\,q$",
        fontsize=12,
    )

    def status_str(phi, u, t_now):
        deg = np.rad2deg(phi)
        mark = "CAPSIZED" if abs(deg) >= phi_v_deg else "rolling "
        return f"t = {t_now:5.2f} s\nphi = {deg:+6.1f} deg\nu   = {u:+5.2f}\n{mark}"

    def init():
        update_ship(art_L, 0.0)
        update_ship(art_R, 0.0)
        cursor_L.set_xdata([0.0, 0.0])
        cursor_R.set_xdata([0.0, 0.0])
        txt_L.set_text("")
        txt_R.set_text("")
        for d in (dot_L, dot_R, trail_L, trail_R):
            d.set_data([], [])
        return ()

    def update_phase(dot, trail, phi_arr, dphi_arr, i, dphi_clip):
        phi_now = np.rad2deg(phi_arr[i])
        dphi_now = float(np.clip(np.rad2deg(dphi_arr[i]), -dphi_clip, dphi_clip))
        dot.set_data([phi_now], [dphi_now])
        i0 = max(0, i - trail_window)
        seg_phi = np.rad2deg(phi_arr[i0 : i + 1])
        seg_dphi = np.clip(np.rad2deg(dphi_arr[i0 : i + 1]), -dphi_clip, dphi_clip)
        trail.set_data(seg_phi, seg_dphi)

    def animate(i):
        update_ship(art_L, phi_stable_anim[i])
        update_ship(art_R, phi_capsize_anim[i])
        cursor_L.set_xdata([t_anim[i], t_anim[i]])
        cursor_R.set_xdata([t_anim[i], t_anim[i]])
        txt_L.set_text(status_str(phi_stable_anim[i], u_stable_anim[i], t_anim[i]))
        txt_R.set_text(status_str(phi_capsize_anim[i], u_capsize_anim[i], t_anim[i]))
        update_phase(dot_L, trail_L, phi_stable_anim, dphi_stable_anim, i, dphi_max_L)
        update_phase(dot_R, trail_R, phi_capsize_anim, dphi_capsize_anim, i, dphi_max_R)
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
