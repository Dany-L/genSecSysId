"""Safe-set plotting for scalar states (``nx == 1``).

With one state neither safe set is a 2-D shape any more: the input polytope
``{x : ||H x||_inf <= 1}`` collapses to ``|x| <= 1 / max_i |h_i|`` and the
ellipse ``{x : (1/s^2) x^T X x <= 1}`` to ``|x| <= s / sqrt(X)``. Both are drawn
as a band (``fill=True``) or as their two boundary lines (``fill=False``).

The band is HORIZONTAL by default because a scalar state has no phase plane:
``plot_state_trajectory`` draws it against the time step, which puts the state
on the y-axis. ``orientation="vertical"`` is there for axes that put the state
on the x-axis instead.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sysid.utils import plot_ellipse, plot_polytope, plot_state_trajectory  # noqa: E402


def _bbox(ax):
    """(x0, x1, y0, y1) of the single patch drawn on ``ax``.

    ``axhspan``/``axvspan`` return a ``Rectangle`` while the 2-D paths return a
    ``Polygon``; going through the path works for both.
    """
    (patch,) = ax.patches
    bb = patch.get_path().get_extents(patch.get_patch_transform())
    return bb.x0, bb.x1, bb.y0, bb.y1


# --------------------------------------------------------------------------
# polytope
# --------------------------------------------------------------------------
def test_polytope_1d_band_spans_the_interval():
    """The band covers exactly |x| <= 1 / max_i |h_i|, on the y-axis by default."""
    fig, ax = plt.subplots()
    H = np.array([[2.0], [-0.5]])  # max |h_i| = 2 -> |x| <= 0.5

    plot_polytope(ax, H, fill=True)

    x0, x1, y0, y1 = _bbox(ax)
    assert np.isclose(y0, -0.5) and np.isclose(y1, 0.5)
    # A horizontal band spans the full axis width in axes coordinates.
    assert np.isclose(x0, 0.0) and np.isclose(x1, 1.0)
    plt.close(fig)


def test_polytope_1d_vertical_orientation_puts_the_state_on_the_x_axis():
    fig, ax = plt.subplots()

    plot_polytope(ax, np.array([[2.0]]), fill=True, orientation="vertical")

    x0, x1, y0, y1 = _bbox(ax)
    assert np.isclose(x0, -0.5) and np.isclose(x1, 0.5)
    assert np.isclose(y0, 0.0) and np.isclose(y1, 1.0)
    plt.close(fig)


def test_polytope_1d_unfilled_draws_both_boundaries():
    """fill=False gives the two boundary lines at -r and +r, styled by linetype."""
    fig, ax = plt.subplots()
    H = np.array([[4.0]])  # |x| <= 0.25

    plot_polytope(ax, H, fill=False, linetype="m-.")

    assert len(ax.lines) == 2
    ys = sorted(float(np.unique(line.get_ydata())[0]) for line in ax.lines)
    assert np.allclose(ys, [-0.25, 0.25])
    for line in ax.lines:
        assert line.get_color() == "m"
        assert line.get_linestyle() == "-."
    # Only one of the two lines carries the legend label.
    labels = [line.get_label() for line in ax.lines]
    assert sum(not lbl.startswith("_") for lbl in labels) == 1
    plt.close(fig)


def test_polytope_1d_zero_H_is_unbounded_and_draws_nothing(caplog):
    """H = 0 means no constraint at all: warn instead of dividing by zero."""
    fig, ax = plt.subplots()

    with caplog.at_level("WARNING"):
        plot_polytope(ax, np.zeros((3, 1)), fill=True)

    assert not ax.patches and not ax.lines
    assert "unbounded" in caplog.text
    plt.close(fig)


def test_polytope_2d_polygon_is_unchanged():
    """Regression: the 2-D path still draws the four vertices of the square."""
    fig, ax = plt.subplots()
    H = np.eye(2)  # |x1| <= 1, |x2| <= 1

    plot_polytope(ax, H, fill=True)

    (poly,) = ax.patches
    V = np.unique(np.round(poly.get_xy(), 9), axis=0)
    assert len(V) == 4
    assert np.allclose(np.abs(V), 1.0)
    plt.close(fig)


# --------------------------------------------------------------------------
# ellipse
# --------------------------------------------------------------------------
def test_ellipse_1d_band_spans_s_over_sqrt_X():
    """(1/s^2) x X x <= 1 with scalar X is the interval |x| <= s / sqrt(X)."""
    fig, ax = plt.subplots()
    X = np.array([[4.0]])

    plot_ellipse(ax, X, s=3.0, fill=True)  # 3 / 2 = 1.5

    _, _, y0, y1 = _bbox(ax)
    assert np.isclose(y0, -1.5) and np.isclose(y1, 1.5)
    plt.close(fig)


def test_ellipse_1d_accepts_the_array_s_that_models_hand_out():
    """``model.s.detach().numpy()`` is an array, not a float."""
    fig, ax = plt.subplots()

    plot_ellipse(ax, np.array([[4.0]]), s=np.array([3.0]), fill=False, orientation="vertical")

    xs = sorted(float(np.unique(line.get_xdata())[0]) for line in ax.lines)
    assert np.allclose(xs, [-1.5, 1.5])
    plt.close(fig)


def test_ellipse_1d_non_positive_X_draws_nothing(caplog):
    fig, ax = plt.subplots()

    with caplog.at_level("WARNING"):
        plot_ellipse(ax, np.array([[-1.0]]), s=1.0, fill=True)

    assert not ax.patches and not ax.lines
    assert "positive definite" in caplog.text
    plt.close(fig)


def test_ellipse_2d_is_unchanged():
    """Regression: X = I, s = 1 still traces the unit circle."""
    fig, ax = plt.subplots()

    plot_ellipse(ax, np.eye(2), s=1.0, fill=False)

    (line,) = ax.lines
    radii = np.hypot(*line.get_data())
    assert np.allclose(radii, 1.0)
    plt.close(fig)


def test_ellipse_3d_is_skipped_with_a_warning(caplog):
    """More than two states cannot be drawn -- warn instead of raising."""
    fig, ax = plt.subplots()

    with caplog.at_level("WARNING"):
        plot_ellipse(ax, np.eye(3), s=1.0)

    assert not ax.patches and not ax.lines
    assert "cannot plot" in caplog.text
    plt.close(fig)


def test_invalid_orientation_raises():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="orientation"):
        plot_polytope(ax, np.array([[1.0]]), orientation="diagonal")
    with pytest.raises(ValueError, match="orientation"):
        plot_ellipse(ax, np.array([[1.0]]), 1.0, orientation="diagonal")
    plt.close(fig)


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------
def test_state_trajectory_2d_is_the_phase_plane():
    fig, ax = plt.subplots()
    x = np.array([[0.0, 1.0], [0.5, 0.8], [1.0, 0.2]])

    plot_state_trajectory(ax, x, color="green", marker="o")

    start, traj = ax.lines
    assert np.allclose(start.get_data(), [[0.0], [1.0]])  # x0 marker
    assert np.allclose(traj.get_xdata(), x[:, 0])
    assert np.allclose(traj.get_ydata(), x[:, 1])
    plt.close(fig)


def test_state_trajectory_1d_is_the_state_over_time():
    """nx == 1: k on the x-axis, x on the y-axis -- the band's default axis."""
    fig, ax = plt.subplots()
    x = np.array([[0.3], [0.27], [0.24]])

    plot_state_trajectory(ax, x, color="red", marker="x")

    start, traj = ax.lines
    assert np.allclose(start.get_data(), [[0.0], [0.3]])
    assert np.allclose(traj.get_xdata(), [0, 1, 2])
    assert np.allclose(traj.get_ydata(), x[:, 0])
    plt.close(fig)


def test_state_trajectory_1d_shares_its_axis_with_the_safe_set_band():
    """The whole point of the horizontal default: both land on the y-axis."""
    fig, ax = plt.subplots()
    x = np.full((10, 1), 0.3)

    plot_state_trajectory(ax, x, color="green")
    plot_polytope(ax, np.array([[2.0]]), fill=True)  # |x| <= 0.5, contains 0.3

    _, _, y0, y1 = _bbox(ax)
    assert y0 < x[0, 0] < y1
    plt.close(fig)


def test_state_trajectory_rejects_a_batched_array():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"\(T, nx\)"):
        plot_state_trajectory(ax, np.zeros((2, 10, 1)))
    plt.close(fig)


def test_state_trajectory_3d_is_skipped_with_a_warning(caplog):
    fig, ax = plt.subplots()
    with caplog.at_level("WARNING"):
        plot_state_trajectory(ax, np.zeros((10, 3)))
    assert not ax.lines
    assert "cannot plot" in caplog.text
    plt.close(fig)
