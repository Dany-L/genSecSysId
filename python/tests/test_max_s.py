"""Tests for the certificate SDPs on SimpleLure.

The clean design has three SDPs (see the wiki certificate-synthesis notes):
  * MaxS  (``_max_s_sdp``)        — max feasible s; brackets the coverage sweep.
  * Feasibility (``feasibility_problem`` / ``_feasibility_sdp``) — repair P,M,L
    at a FIXED s (the within-epoch step 3.1); s is never changed here.
  * MinTrProb (``solve_output_coverage_certificate``) — tested in
    tests/test_output_coverage.py.

These need MOSEK and are skipped otherwise.
"""

import numpy as np
import pytest
import torch
import cvxpy as cp

from sysid.models.constrained_rnn import SimpleLure


def _mosek_available() -> bool:
    if "MOSEK" not in cp.installed_solvers():
        return False
    try:
        x = cp.Variable()
        cp.Problem(cp.Minimize((x - 1) ** 2), [x >= 0]).solve(solver=cp.MOSEK, verbose=False)
        return True
    except Exception:
        return False


MOSEK = _mosek_available()
requires_mosek = pytest.mark.skipif(not MOSEK, reason="MOSEK solver not available/licensed")


def _make_model(s_value: float = 0.05) -> SimpleLure:
    """A small, stable Lure system for which the SDPs are feasible."""
    m = SimpleLure(nd=1, ne=1, nx=2, nw=1, activation="dzn", custom_params={"learn_L": True})
    with torch.no_grad():
        m.A.data = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=m.A.dtype)
        m.B.data = torch.tensor([[0.1], [0.1]], dtype=m.B.dtype)
        m.B2.data = torch.zeros_like(m.B2)
        m.C2.data = torch.tensor([[0.1, 0.1]], dtype=m.C2.dtype)
        m.D21.data = torch.tensor([[0.1]], dtype=m.D21.dtype)
        m.tau.data = torch.tensor(float(np.log(0.9 / 0.1)))  # alpha = 0.9
        m.s.data = torch.tensor(float(s_value))
    return m


def _make_moderate_model() -> SimpleLure:
    """A feasible system whose max feasible s is moderate (not the EPS-capped
    near-global value of the degenerate _make_model)."""
    m = SimpleLure(nd=1, ne=1, nx=2, nw=2, activation="dzn", custom_params={"learn_L": True})
    with torch.no_grad():
        m.A.data = torch.tensor([[0.9, 0.1], [0.0, 0.9]], dtype=m.A.dtype)
        m.B.data = torch.tensor([[0.1], [0.1]], dtype=m.B.dtype)
        m.B2.data = 0.3 * torch.ones_like(m.B2)
        m.C2.data = 0.5 * torch.ones_like(m.C2)
        m.D21.data = 0.1 * torch.ones_like(m.D21)
        m.tau.data = torch.tensor(float(np.log(0.99 / 0.01)))  # alpha = 0.99
    return m


@requires_mosek
class TestMaxS:
    """MaxS (``_max_s_sdp``): pure, returns the max feasible s + certificate."""

    def test_max_s_sdp_is_feasible(self):
        m = _make_model()
        sol = m._max_s_sdp()
        assert sol is not None and sol["s"] > 0
        m._apply_certificate_solution(sol)
        assert m.check_constraints()

    def test_post_process_matches_max_s(self):
        """post_process routes through the same _max_s_sdp core."""
        s_direct = _make_model()._max_s_sdp()["s"]
        out = _make_model().post_process()
        assert out["success"]
        assert out["s_opt"] == pytest.approx(s_direct, rel=1e-4)

    def test_multiplier_is_nonnegative(self):
        m = _make_moderate_model()
        sol = m._max_s_sdp()
        assert sol is not None
        m._apply_certificate_solution(sol)
        assert float(m.la.min()) >= -1e-9  # M = diag(la) >= 0


@requires_mosek
class TestFeasibilityProblem:
    """Feasibility: repair P, M, L at a FIXED s; s is never modified."""

    def test_repairs_at_fixed_s_without_changing_s(self):
        """Corrupt P so the certificate breaks, then repair at the current s.
        s must be unchanged and the constraints must hold again."""
        m = _make_moderate_model()
        s_max = m._max_s_sdp()["s"]
        s_target = 0.5 * s_max  # comfortably feasible scale
        with torch.no_grad():
            m.s.data = torch.tensor(s_target, dtype=m.s.dtype)
            m.P.data = torch.diag(torch.tensor([1.0, -1.0], dtype=m.P.dtype))  # indefinite
        assert not m.check_constraints()

        assert m.feasibility_problem() is True
        assert m.check_constraints()
        assert float(m.s) == pytest.approx(s_target, rel=1e-9)  # s untouched

    def test_returns_false_when_s_above_s_max(self):
        """No feasible P, M, L exists above the regionality ceiling -> False
        (the trainer then rolls the update back). s is not clamped."""
        m = _make_moderate_model()
        s_max = m._max_s_sdp()["s"]
        with torch.no_grad():
            m.s.data = torch.tensor(10.0 * s_max, dtype=m.s.dtype)
        assert m.feasibility_problem() is False

    def test_returns_false_on_genuine_infeasibility(self):
        """Uncertifiable dynamics (unstable A) -> False at any s."""
        m = _make_moderate_model()
        with torch.no_grad():
            m.A.data = torch.tensor([[1.5, 0.0], [0.0, 1.5]], dtype=m.A.dtype)  # unstable
        assert m.feasibility_problem() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
