"""Regional-verification initial-state-violation scaling.

The initial-state-violation regime samples ``x0`` on the safe ellipse scaled
by ``initial_state_scale`` (radius ``= scale * s / alpha``). This used to be a
hard-coded factor of ``2``; it is now a parameter threaded from
``scripts/post_process.py``. These tests pin:

  * ``_sample_on_ellipsoid`` places points exactly on ``{x : xᵀ X x = r²}``,
    so scaling the radius scales how far the samples sit outside the ellipse;
  * ``regional_verification`` exposes ``initial_state_scale`` with the default
    (``2.0``) that preserves the previous behaviour.
"""

import inspect

import numpy as np

from sysid.evaluation.regional_verification import (  # noqa: E402
    _sample_on_ellipsoid,
    regional_verification,
)


def test_sample_on_ellipsoid_lies_on_scaled_ellipsoid():
    """Samples satisfy xᵀ X x = radius² for an arbitrary radius, so the radius
    (and hence initial_state_scale) linearly controls the sampling shell."""
    rng = np.random.default_rng(0)
    # A fixed SPD matrix X (= P⁻¹ in the caller).
    A = np.array([[2.0, 0.5], [0.5, 1.0]])
    X = A @ A.T

    for radius in (0.5, 2.0, 7.3):
        pts = _sample_on_ellipsoid(rng, X, radius=radius, n=64)
        quad = np.einsum("ni,ij,nj->n", pts, X, pts)
        np.testing.assert_allclose(quad, radius**2, rtol=1e-10, atol=1e-10)


def test_sample_on_ellipsoid_scale_multiplies_radius():
    """Doubling the radius doubles ‖x‖_X for matched samples: xᵀ X x scales by
    scale², which is exactly how initial_state_scale pushes x0 outside the set."""
    A = np.array([[2.0, 0.5], [0.5, 1.0]])
    X = A @ A.T
    base, scale = 1.3, 3.0

    pts_base = _sample_on_ellipsoid(np.random.default_rng(1), X, radius=base, n=32)
    pts_scaled = _sample_on_ellipsoid(
        np.random.default_rng(1), X, radius=scale * base, n=32
    )

    quad_base = np.einsum("ni,ij,nj->n", pts_base, X, pts_base)
    quad_scaled = np.einsum("ni,ij,nj->n", pts_scaled, X, pts_scaled)
    np.testing.assert_allclose(quad_scaled / quad_base, scale**2, rtol=1e-9)


def test_regional_verification_default_initial_state_scale_is_two():
    """The parameter default preserves the previous hard-coded factor of 2."""
    sig = inspect.signature(regional_verification)
    assert sig.parameters["initial_state_scale"].default == 2.0
