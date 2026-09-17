"""The 1-D Lur'e benchmark as a registered reference system.

The plant is the one ``scripts/generate_one_d_dataset.py`` samples::

    x+ = 0.9 x + 1.0 u + 1.1 dzn(x),   dzn(z) = max(|z| - 1, 0) sign(z)

Its autonomous map is contracting inside the dead zone (``x+ = 0.9 x``) and
``x+ = 2.0 x - 1.1`` outside it, so the true basin of attraction is |x| < 1.1 --
which is what ``converges_to_origin`` has to reproduce for the notebook to
colour its random initial conditions the same way it does for the Duffing set.
"""

import numpy as np

from sysid.evaluation import get_true_dynamics, list_true_dynamics
from sysid.evaluation.true_dynamics import (
    dead_zone,
    one_d_converges_to_origin,
    simulate_one_d,
)


def test_one_d_is_registered_as_a_scalar_system():
    assert "one_d" in list_true_dynamics()
    spec = get_true_dynamics("one_d")
    assert spec.state_dim == 1
    assert spec.state_labels == ("x",)
    assert spec.converges_to_origin is one_d_converges_to_origin


def test_duffing_keeps_its_entry_and_gains_the_verdict():
    spec = get_true_dynamics("duffing")
    assert spec.state_dim == 2
    assert spec.converges_to_origin is not None
    assert spec.converges_to_origin([0.1, 0.0])       # inside the separatrix
    assert not spec.converges_to_origin([1.5, 0.5])   # outside it


def test_dead_zone_breaks_at_one():
    assert dead_zone(0.5) == 0.0
    assert np.isclose(dead_zone(1.3), 0.3)
    assert np.isclose(dead_zone(-1.3), -0.3)


def test_simulate_one_d_matches_the_linear_recurrence_inside_the_dead_zone():
    """Below the breakpoint the plant is just x+ = 0.9 x."""
    X, y, diverged = simulate_one_d(0.5, np.zeros(10))

    assert X.shape == (11, 1)
    assert y.shape == (10,)
    assert not diverged
    assert np.allclose(X[:, 0], 0.5 * 0.9 ** np.arange(11))
    assert np.allclose(y, X[:-1, 0])


def test_simulate_one_d_fires_the_nonlinearity_above_the_breakpoint():
    """Above |x| = 1 the map becomes x+ = 2.0 x - 1.1."""
    X, _, _ = simulate_one_d(1.2, np.zeros(1))
    assert np.isclose(X[1, 0], 2.0 * 1.2 - 1.1)


def test_simulate_one_d_reports_divergence_and_stops_early():
    X, y, diverged = simulate_one_d(1.5, np.zeros(200))

    assert diverged
    assert len(X) < 201  # stopped at the threshold instead of running to the end
    assert abs(X[-1, 0]) > 5.0
    assert len(y) == len(X) - 1


def test_simulate_one_d_responds_to_the_input():
    X, _, _ = simulate_one_d(0.0, [0.3])
    assert np.isclose(X[1, 0], 0.3)  # B = 1, and dzn(0) = 0


def test_converges_to_origin_reproduces_the_basin():
    """|x0| < 1.1 comes back, beyond it the state runs away."""
    assert one_d_converges_to_origin(0.0)
    assert one_d_converges_to_origin(0.3)
    assert one_d_converges_to_origin(-0.99)
    assert one_d_converges_to_origin(1.05)   # one step above the breakpoint,
    assert one_d_converges_to_origin(-1.05)  # then back into the linear region
    assert not one_d_converges_to_origin(1.2)
    assert not one_d_converges_to_origin(-1.2)


def test_converges_to_origin_accepts_the_array_x0_the_notebook_draws():
    """``rng.uniform(-1.5, 1.5, nx)`` is a length-1 array, not a float."""
    assert one_d_converges_to_origin(np.array([0.3]))
    assert not one_d_converges_to_origin(np.array([1.3]))
