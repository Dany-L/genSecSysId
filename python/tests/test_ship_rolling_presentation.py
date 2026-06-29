"""Tests for the ship-rolling presentation animation script.

The script lives under notebooks/ (not the installed package), so it is
loaded by file path. We never render the full GIF here: we check the input
generation / simulation invariants and that the figure builds and a couple
of frames render without raising.
"""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless: no display needed for frame rendering

REPO_PY = Path(__file__).resolve().parents[1]
SCRIPT = REPO_PY / "notebooks" / "duffing" / "ship_rolling_presentation.py"


@pytest.fixture(scope="module")
def srp():
    """Import the script module by path."""
    spec = importlib.util.spec_from_file_location("ship_rolling_presentation", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_inputs_bounds(srp):
    """Stable forcing stays inside the safe band; the capsize pulse exceeds it."""
    t, u_stable, u_capsize = srp.build_inputs(T_total=30.0, Ts=srp.DUFFING_TS)
    assert t.shape == u_stable.shape == u_capsize.shape
    assert np.max(np.abs(u_stable)) < srp.DUFFING_U_C
    assert np.max(np.abs(u_capsize)) > srp.DUFFING_U_C


def test_simulate_stable_vs_capsize(srp):
    """Stable input stays bounded; the pulse drives |q| past the clip."""
    t, u_stable, u_capsize = srp.build_inputs(T_total=30.0, Ts=srp.DUFFING_TS)
    X_stable = srp.simulate((0.3, 0.0), u_stable)
    X_capsize = srp.simulate((0.0, 0.0), u_capsize)
    assert np.all(np.isfinite(X_stable))
    assert np.max(np.abs(X_stable[:, 0])) < 1.0
    # The capsize run stops early (clipped) once it crosses the hilltop.
    assert X_capsize.shape[0] < len(u_capsize) + 1
    assert np.max(np.abs(X_capsize[:, 0])) > 1.0


def test_build_animation_renders_frames(srp, tmp_path):
    """Figure builds and first/last frames render to PNG without error."""
    fig, anim, animate, init, t_anim = srp.build_animation(
        T_total=2.0, phi_v_deg=60.0, fps=10
    )
    # 2 ship axes + 2 timeseries axes + 2 twin (input) axes.
    assert len(fig.axes) == 6
    assert len(t_anim) > 0
    init()
    animate(0)
    animate(len(t_anim) - 1)
    out = tmp_path / "frame.png"
    fig.savefig(out)
    assert out.exists()
    matplotlib.pyplot.close(fig)


def test_dump_frames(srp, tmp_path):
    """dump_frames writes one PNG per animation frame."""
    fig, anim, animate, init, t_anim = srp.build_animation(
        T_total=1.0, phi_v_deg=60.0, fps=8
    )
    frames_dir = tmp_path / "frames"
    srp.dump_frames(fig, animate, init, t_anim, frames_dir, dpi=60)
    pngs = sorted(frames_dir.glob("frame_*.png"))
    assert len(pngs) == len(t_anim)
    matplotlib.pyplot.close(fig)
