"""Shared test helper: is the MOSEK SDP solver available and licensed?

The certificate-synthesis SDPs (``optimization/synthesizer.py``) solve with
``cp.MOSEK`` exclusively, so any test that drives that path — the SDP unit
tests and the end-to-end train/sweep smoke pipelines — must skip when MOSEK is
absent (e.g. on CI runners without an academic license). Import ``requires_mosek``
and apply it as a marker (or ``pytestmark``) to gate those tests.
"""

import cvxpy as cp
import pytest


def mosek_available() -> bool:
    """True iff MOSEK is installed and can actually solve a trivial problem."""
    if "MOSEK" not in cp.installed_solvers():
        return False
    try:
        x = cp.Variable()
        cp.Problem(cp.Minimize((x - 1) ** 2), [x >= 0]).solve(solver=cp.MOSEK, verbose=False)
        return True
    except Exception:
        return False


requires_mosek = pytest.mark.skipif(
    not mosek_available(), reason="MOSEK solver not available/licensed"
)
