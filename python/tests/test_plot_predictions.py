"""Plotting must never abort a training run on non-finite predictions.

A diverged model can hand plot_predictions Inf/NaN trajectories. Older
matplotlib then crashes in the tick locator on non-finite axis limits
(MaxNLocator._raw_ticks). These tests pin the guard: non-finite inputs are
masked and every axis ends up with finite limits.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sysid.utils import (  # noqa: E402
    _ensure_finite_axis_limits,
    _mask_nonfinite,
    plot_predictions,
)


def test_mask_nonfinite_replaces_inf_and_nan():
    arr = np.array([1.0, np.inf, -np.inf, np.nan, 2.0])
    out = _mask_nonfinite(arr)
    assert np.array_equal(np.isnan(out), [False, True, True, True, False])
    assert out[0] == 1.0 and out[4] == 2.0


def test_ensure_finite_axis_limits_fixes_empty_axis():
    fig, ax = plt.subplots()
    # An axis with only non-finite data has no finite dataLim.
    ax.plot([np.inf, np.nan, np.inf])
    _ensure_finite_axis_limits([ax], default=(-1.0, 1.0))
    assert np.isfinite(ax.get_xlim()).all()
    assert np.isfinite(ax.get_ylim()).all()
    plt.close(fig)


def test_ensure_finite_axis_limits_leaves_good_axis():
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0, 4.0])
    _ensure_finite_axis_limits([ax])
    # A populated axis keeps its autoscaled (finite) limits, not the default.
    assert np.isfinite(ax.get_ylim()).all()
    assert ax.get_ylim() != (-1.0, 1.0)
    plt.close(fig)


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_plot_predictions_survives_nonfinite(tmp_path, bad):
    """Mixed finite / all-Inf / all-NaN samples must not raise, and every
    axis must end up with finite limits."""
    n, T = 3, 40
    e_hat = np.random.randn(n, T, 1)
    e_hat[0] = bad          # one fully non-finite predicted trajectory
    e = np.random.randn(n, T, 1)
    d = np.random.randn(n, T, 1)
    d[1] = bad              # one fully non-finite input

    fig, axes = plot_predictions(
        output_dir=tmp_path,
        e_hat=e_hat,
        e=e,
        d=d,
        sample_indices=[0, 1, 2],
        return_axes=True,
        warmup_steps=5,
    )
    for ax in fig.axes:
        assert np.isfinite(ax.get_xlim()).all(), "non-finite xlim survived"
        assert np.isfinite(ax.get_ylim()).all(), "non-finite ylim survived"
    plt.close(fig)


def test_plot_predictions_saves_file_with_nonfinite(tmp_path):
    """Non-return path writes a file instead of raising in tight_layout."""
    n, T = 2, 30
    e_hat = np.full((n, T, 1), np.inf)
    e = np.random.randn(n, T, 1)
    out = tmp_path / "pred.png"
    plot_predictions(
        output_dir=tmp_path,
        e_hat=e_hat,
        e=e,
        d=None,
        sample_indices=[0, 1],
        save_path=str(out),
    )
    assert out.exists()
