"""Plotting must survive a diverged model, on any matplotlib.

The failure these pin came off the cluster: a model that diverged at
initialization produced predictions around 1e308, and ``plt.tight_layout()``
died inside the tick locator::

    File "matplotlib/ticker.py", line 2100, in _raw_ticks
        istep = np.nonzero(steps >= raw_step)[0][0]
    IndexError: index 0 is out of bounds for axis 0 with size 0

It happened at ``trainer.train()``'s ``plot_trajectories("initial_trajectories")``
call, BEFORE epoch 1, so the run died without training at all.

Two things made it possible and both are covered here:

1. The old guard tested ``np.isfinite`` alone. 1e308 is finite, so it passed
   through, and matplotlib's own ``(x1t - x0t) * margin`` then overflowed.
2. matplotlib < 3.8 has no fallback in ``MaxNLocator._raw_ticks``. Pinning a
   newer one is not portable — 3.8.0 requires Python >= 3.9, and the cluster
   venv is Python 3.8, whose ceiling is the unguarded 3.7.5.

So the local matplotlib (guarded) cannot reproduce the crash on its own. These
tests assert the property that makes the crash impossible — nothing
astronomical reaches the axis limits — rather than asserting "no exception",
which would pass vacuously here and still fail on the cluster.
"""

import logging

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sysid.utils import (  # noqa: E402
    _PLOT_ABS_MAX,
    _ensure_finite_axis_limits,
    _mask_nonfinite,
    plot_predictions,
    plot_safe_set_trajectories,
    plot_state_trajectory,
)

# What the cluster actually held: finite, astronomical, and one step from Inf.
HUGE = 1e308


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _diverging(n=200, peak=HUGE):
    """A trajectory that starts sane and blows up to `peak`, then to Inf.

    Mirrors a real divergence: the informative part is the finite head, which
    the masking must keep.
    """
    out = np.geomspace(1e-3, peak, n)
    out[-3:] = np.inf
    out[:20] = np.linspace(-0.5, 0.5, 20)  # plausible early behaviour
    return out


class TestMaskNonfinite:
    def test_masks_finite_but_astronomical_values(self):
        """The whole bug: 1e308 is finite, so isfinite() alone let it through."""
        arr = np.array([1.0, HUGE, -HUGE, np.inf, np.nan])
        out = _mask_nonfinite(arr)

        assert out[0] == 1.0
        assert np.isnan(out[1:]).all(), "huge values must be masked, not just Inf"

    def test_keeps_the_finite_head_of_a_divergence(self):
        out = _mask_nonfinite(_diverging())
        finite = out[np.isfinite(out)]
        assert len(finite) > 10, "masking must not throw away the usable part"
        assert np.abs(finite).max() <= _PLOT_ABS_MAX

    def test_threshold_leaves_real_data_untouched(self):
        """Nothing physical in this package approaches 1e100."""
        arr = np.array([-1e6, -1.0, 0.0, 1e-12, 1e6, 1e30])
        np.testing.assert_array_equal(_mask_nonfinite(arr), arr)


class TestEnsureFiniteAxisLimits:
    def test_clamps_an_axis_with_no_finite_data(self):
        fig, ax = plt.subplots()
        ax.plot([np.nan, np.nan])
        _ensure_finite_axis_limits([ax])
        assert np.isfinite(ax.get_xlim()).all() and np.isfinite(ax.get_ylim()).all()

    def test_clamps_an_axis_whose_span_overflows(self):
        """Endpoints finite, but x1 - x0 is not — the case the old guard missed."""
        fig, ax = plt.subplots()
        ax.plot([-HUGE, HUGE], [-HUGE, HUGE])
        with np.errstate(over="ignore"):
            assert not np.isfinite(np.float64(HUGE) - np.float64(-HUGE)), (
                "precondition: this span must overflow"
            )
        _ensure_finite_axis_limits([ax])
        for lo, hi in (ax.get_xlim(), ax.get_ylim()):
            with np.errstate(over="ignore"):
                assert np.isfinite(np.float64(hi) - np.float64(lo))

    def test_leaves_a_healthy_axis_alone(self):
        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [-2.0, 3.0])
        before = (ax.get_xlim(), ax.get_ylim())
        _ensure_finite_axis_limits([ax])
        assert (ax.get_xlim(), ax.get_ylim()) == before


def _axis_spans_are_safe(fig) -> bool:
    """No axis on `fig` has limits whose span overflows or is non-finite."""
    for ax in fig.get_axes():
        for lo, hi in (ax.get_xlim(), ax.get_ylim()):
            if not np.isfinite([lo, hi]).all():
                return False
            with np.errstate(over="ignore", invalid="ignore"):
                if not np.isfinite(np.float64(hi) - np.float64(lo)):
                    return False
    return True


class TestPlotPredictions:
    def test_diverged_predictions_leave_usable_axis_limits(self, tmp_path):
        """The exact cluster shape: e_hat diverged, e and d healthy."""
        n = 200
        e_hat = _diverging(n).reshape(1, n, 1)
        e = np.sin(np.linspace(0, 10, n)).reshape(1, n, 1)
        d = np.cos(np.linspace(0, 10, n)).reshape(1, n, 1)

        fig, axes = plot_predictions(
            output_dir=str(tmp_path), e_hat=e_hat, e=e, d=d,
            sample_indices=[0], return_axes=True,
        )
        assert _axis_spans_are_safe(fig)
        # tight_layout is what crashed; it must now complete.
        fig.tight_layout()

    def test_fully_nonfinite_predictions_still_render(self, tmp_path):
        n = 50
        e_hat = np.full((1, n, 1), np.inf)
        e = np.full((1, n, 1), np.nan)
        plot_predictions(
            output_dir=str(tmp_path), e_hat=e_hat, e=e, d=None,
            sample_indices=[0], save_path=str(tmp_path / "p.png"),
        )
        assert (tmp_path / "p.png").exists()


class TestTrajectoryPlots:
    def test_safe_set_trajectories_masks_divergence(self):
        """Saved with bbox_inches='tight', so it runs the same locator path."""
        n = 120
        x = np.stack([_diverging(n), _diverging(n)], axis=1)[None, ...]  # (1, n, 2)
        c = np.zeros((1, n))
        fig, ax, n_stable, n_unstable = plot_safe_set_trajectories(
            P=np.eye(2), L=np.zeros((2, 2)), s=1.0, x_traj=x, c=c,
            warmup_steps=0, horizon=n,
        )
        assert _axis_spans_are_safe(fig)
        fig.tight_layout()

    def test_state_trajectory_masks_divergence(self):
        fig, ax = plt.subplots()
        n = 120
        plot_state_trajectory(ax, np.stack([_diverging(n), _diverging(n)], axis=1))
        assert _axis_spans_are_safe(fig)


class TestSafePlot:
    """Belt and braces: plotting must never abort a run, whatever goes wrong."""

    def _trainer_stub(self):
        from sysid.training.trainer import Trainer

        stub = object.__new__(Trainer)  # no __init__: _safe_plot needs no state
        return stub

    def test_a_failing_plot_is_downgraded_to_a_warning(self, caplog):
        stub = self._trainer_stub()
        calls = []

        def boom(**kwargs):
            calls.append(kwargs)
            raise IndexError("index 0 is out of bounds for axis 0 with size 0")

        with caplog.at_level(logging.WARNING):
            stub._safe_plot(boom, name="initial_trajectories")

        assert calls == [{"name": "initial_trajectories"}], "the call still happened"
        assert "Plotting" in caplog.text and "boom" in caplog.text
        # The traceback must survive, or the cause becomes invisible.
        assert "IndexError" in caplog.text

    def test_a_successful_plot_is_untouched(self):
        stub = self._trainer_stub()
        seen = {}
        stub._safe_plot(lambda **kw: seen.update(kw), name="epoch_3", normalizer=None)
        assert seen == {"name": "epoch_3", "normalizer": None}
