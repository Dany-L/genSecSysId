"""Ship-rolling presentation animation (2 columns, no phase portrait).

A slimmed-down variant of the original ``ship_rolling_simulation.py`` meant
for slides: two tilting ships side by side over the *true* softening-Duffing
dynamics, and below each the forcing input ``u(t)`` with its critical bounds
``±u_c``.

  - left:  small wave forcing  ->  stays inside the basin   (stable rolling)
  - right: a strong pulse       ->  crosses the hilltop      (capsize)

The narrative the bottom panels carry: the stable input never leaves the
safe band ``[-u_c, u_c]``, while the capsize pulse punches through it.

Maps the dimensionless Duffing state ``q`` to a roll angle ``phi = phi_v q``,
where ``phi_v`` is the angle of vanishing stability (capsize boundary).

Run (write a GIF without opening a window):
  python notebooks/duffing/ship_rolling_presentation.py \\
      --save notebooks/duffing/figs/ship_rolling_presentation.gif --no-show

Export PNG frames for LaTeX/beamer ``\\animategraphics`` (see header of the
generated frames dir, and the README note this script prints):
  python notebooks/duffing/ship_rolling_presentation.py \\
      --frames-dir notebooks/duffing/figs/ship_frames --no-show
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
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
    """Stable (small sine, inside the bounds) and capsize (pulse over the
    bound) forcing on a common time grid."""
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
    ax.set_title(title, fontsize=12)
    ax.axhspan(-1.2, 0.0, color="#a8d0e6", zorder=0)
    ax.axhspan(0.0, 2.6, color="#f0f7ff", zorder=0)
    xs = np.linspace(-2.5, 2.5, 200)
    ax.plot(xs, 0.05 * np.sin(2 * xs), color="#3a7ca5", lw=0.8, zorder=1)


def setup_timeseries_axis(ax, t, phi_deg, u_seq, T_total, phi_v_deg):
    """Bottom panel: roll angle phi(t) on the left axis (with the capsize
    boundary +/- phi_v), and the forcing u(t) with its bounds +/- u_c on a
    right twin axis. The twin axis is transparent, so the black angle trace
    shows through underneath the blue input."""
    ax.set_xlim(0, T_total)
    ax.set_ylim(-1.4 * phi_v_deg, 1.4 * phi_v_deg)
    ax.axhline(phi_v_deg, color="red", ls="--", lw=0.9,
               label=fr"$\pm\varphi_v = \pm{phi_v_deg:.0f}^\circ$ (capsize)")
    ax.axhline(-phi_v_deg, color="red", ls="--", lw=0.9)
    ax.plot(t, phi_deg, color="black", lw=1.4, label=r"roll angle $\varphi(t)$")
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_ylabel(r"roll angle $\varphi$ (deg)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax2 = ax.twinx()
    u_lim = max(2.0, 1.25 * float(np.max(np.abs(u_seq))))
    ax2.set_ylim(-u_lim, u_lim)
    ax2.plot(t, u_seq, color="steelblue", lw=1.1, alpha=0.9)
    ax2.axhline(DUFFING_U_C, color="steelblue", ls=":", lw=0.9, alpha=0.8)
    ax2.axhline(-DUFFING_U_C, color="steelblue", ls=":", lw=0.9, alpha=0.8)
    ax2.set_ylabel(fr"input $u$ ($\pm u_c={DUFFING_U_C:.3f}$ dotted)",
                   color="steelblue", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=9)
    return ax2


def build_animation(T_total=30.0, phi_v_deg=60.0, fps=30):
    """Construct the figure, artists and FuncAnimation.

    Returns ``(fig, anim, animate, init, t_anim)``. ``animate``/``init`` are
    exposed so a test can render single frames without driving the writer.
    """
    phi_v = np.deg2rad(phi_v_deg)
    Ts = DUFFING_TS

    t, u_stable, u_capsize = build_inputs(T_total, Ts)
    n = len(t)

    X_stable = pad_to_length(simulate((0.3, 0.0), u_stable, Ts=Ts), n + 1)
    X_capsize = pad_to_length(simulate((0.0, 0.0), u_capsize, Ts=Ts), n + 1)

    phi_stable = np.clip(phi_v * X_stable[:n, 0], -np.pi, np.pi)
    phi_capsize = np.clip(phi_v * X_capsize[:n, 0], -np.pi, np.pi)

    t_anim = np.arange(0.0, T_total, 1.0 / fps)
    phi_stable_anim = np.interp(t_anim, t, phi_stable)
    phi_capsize_anim = np.interp(t_anim, t, phi_capsize)
    u_stable_anim = np.interp(t_anim, t, u_stable)
    u_capsize_anim = np.interp(t_anim, t, u_capsize)

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.32, wspace=0.20)
    ax_ship_L = fig.add_subplot(gs[0, 0])
    ax_ship_R = fig.add_subplot(gs[0, 1])
    ax_in_L = fig.add_subplot(gs[1, 0])
    ax_in_R = fig.add_subplot(gs[1, 1])

    setup_ship_axis(
        ax_ship_L, fr"Stable rolling   peak $|u|=0.25 < u_c={DUFFING_U_C:.3f}$"
    )
    setup_ship_axis(
        ax_ship_R,
        fr"Capsize   pulse $u=1.5 > u_c={DUFFING_U_C:.3f}$  ($t\in[4,6]$ s)",
    )

    art_L = make_ship_artists(ax_ship_L, color="#5a7a3f")
    art_R = make_ship_artists(ax_ship_R, color="#7a3f3f")

    setup_timeseries_axis(ax_in_L, t, np.rad2deg(phi_stable), u_stable,
                          T_total, phi_v_deg)
    setup_timeseries_axis(ax_in_R, t, np.rad2deg(phi_capsize), u_capsize,
                          T_total, phi_v_deg)
    cursor_L = ax_in_L.axvline(0.0, color="grey", lw=1.0, zorder=4)
    cursor_R = ax_in_R.axvline(0.0, color="grey", lw=1.0, zorder=4)

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
        fontsize=13,
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
        return ()

    def animate(i):
        update_ship(art_L, phi_stable_anim[i])
        update_ship(art_R, phi_capsize_anim[i])
        cursor_L.set_xdata([t_anim[i], t_anim[i]])
        cursor_R.set_xdata([t_anim[i], t_anim[i]])
        txt_L.set_text(status_str(phi_stable_anim[i], u_stable_anim[i], t_anim[i]))
        txt_R.set_text(status_str(phi_capsize_anim[i], u_capsize_anim[i], t_anim[i]))
        return ()

    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t_anim), interval=1000.0 / fps, blit=False,
    )
    return fig, anim, animate, init, t_anim


def dump_frames(fig, animate, init, t_anim, frames_dir, dpi=110, fps=30):
    """Write one PNG per frame for LaTeX/beamer ``\\animategraphics``.

    ``fps`` is only used for the printed snippet: \\animategraphics must be
    told the same rate the frames were rendered at, or playback runs at the
    wrong speed.
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    init()
    for i in range(len(t_anim)):
        animate(i)
        fig.savefig(frames_dir / f"frame_{i:04d}.png", dpi=dpi)
    n = len(t_anim) - 1
    print(f"Wrote {len(t_anim)} frames to {frames_dir}")
    print("LaTeX (preamble: \\usepackage{animate}):")
    print(
        f"  \\animategraphics[autoplay,loop,width=\\linewidth]"
        f"{{{fps}}}{{{frames_dir.name}/frame_}}{{0}}{{{n}}}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--save", type=Path, default=None,
                        help="Save the animation to this path (.gif or .mp4).")
    parser.add_argument("--frames-dir", type=Path, default=None,
                        help="Also dump one PNG per frame here (for \\animategraphics).")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip plt.show(); useful when only saving.")
    parser.add_argument("--phi-v-deg", type=float, default=60.0,
                        help="Angle of vanishing stability in degrees (visual scaling).")
    parser.add_argument("--T-total", type=float, default=30.0,
                        help="Total simulated time (seconds).")
    parser.add_argument("--fps", type=int, default=30, help="Animation frame rate.")
    parser.add_argument("--dpi", type=int, default=110,
                        help="Output resolution for the GIF/MP4/frames.")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    fig, anim, animate, init, t_anim = build_animation(
        T_total=args.T_total, phi_v_deg=args.phi_v_deg, fps=args.fps
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if args.save.suffix.lower() == ".gif":
            writer = PillowWriter(fps=args.fps)
        else:
            writer = FFMpegWriter(fps=args.fps)
        print(f"Saving animation to {args.save} ...")
        anim.save(str(args.save), writer=writer, dpi=args.dpi)
        print("Done.")

    if args.frames_dir is not None:
        dump_frames(fig, animate, init, t_anim, args.frames_dir, dpi=args.dpi,
                    fps=args.fps)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
