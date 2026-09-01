"""Tests for the certificate SDPs on SimpleLure.

The certificate SDPs live on ``LureCertificateSynthesizer``:
  * MaxS  (``synthesizer.max_s``) — max feasible s; the OPERATIVE certificate
    (largest regional invariant set, well conditioned).
  * Feasibility (``feasibility_problem`` / ``synthesizer.feasibility``) — repair
    P,M,L at a FIXED s (the within-epoch step 3.1); s is never changed here.
  * MinTrProb (``solve_output_coverage_certificate``) — tested in
    tests/test_output_coverage.py.

These need MOSEK and are skipped otherwise.
"""

import numpy as np
import pytest
import torch
import cvxpy as cp

from sysid.models.constrained_rnn import SimpleLure
from sysid.optimization import LureCertificateSynthesizer


def _synth(model) -> LureCertificateSynthesizer:
    """The certificate synthesizer for a model's current dynamics (the SDP home)."""
    return LureCertificateSynthesizer.from_model(model)


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


def _make_global_model(s_value: float = 1.0) -> SimpleLure:
    """A globally stable Lure system with non-learnable L (L = 0) and fixed s.

    ``learn_L=False`` fixes the coupling to L = 0 (global stability) and freezes
    ``s`` / ``alpha``; the certificate SDPs must stay well-posed in this regime
    (no locality LMIs, s not an optimization variable)."""
    m = SimpleLure(nd=1, ne=1, nx=2, nw=1, activation="dzn", custom_params={"learn_L": False})
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
    """MaxS (``synthesizer.max_s``): pure, returns the max feasible s + certificate."""

    def test_max_s_is_feasible(self):
        m = _make_model()
        sol = _synth(m).max_s()
        assert sol is not None and sol.s > 0
        m._apply_certificate_solution(sol)
        assert m.check_constraints()

    def test_post_process_applies_max_s(self):
        """post_process now applies MaxS as the operative certificate, so its
        operative s equals the direct MaxS s."""
        s_direct = _synth(_make_model()).max_s().s
        out = _make_model().post_process()
        assert out["success"]
        assert out["max_s"]["s"] == pytest.approx(s_direct, rel=1e-4)
        assert out["s_opt"] == pytest.approx(out["max_s"]["s"], rel=1e-12)

    def test_multiplier_is_nonnegative(self):
        m = _make_moderate_model()
        sol = _synth(m).max_s()
        assert sol is not None
        m._apply_certificate_solution(sol)
        assert float(m.la.min()) >= -1e-9  # M = diag(la) >= 0

    def test_max_s_runs_for_global_model(self):
        """learn_L=False (L=0, globally stable): the SDP must solve without
        treating s as a variable or building the degenerate locality LMIs, and
        must return the fixed s (not an EPS-capped near-global value)."""
        m = _make_global_model(s_value=1.0)
        sol = _synth(m).max_s()
        assert sol is not None
        assert sol.s == pytest.approx(1.0, rel=1e-9)  # s stays fixed
        assert np.allclose(sol.L, 0.0)  # L stays fixed at 0
        assert sol.locality_min_eigs == []  # no locality LMIs
        m._apply_certificate_solution(sol)
        assert m.check_constraints()

    def test_post_process_runs_for_global_model(self):
        """post_process routes through MaxS for the global (L=0) model;
        s stays fixed."""
        out = _make_global_model(s_value=1.0).post_process()
        assert out["success"]
        assert out["s_opt"] == pytest.approx(1.0, rel=1e-9)
        assert out["constraints_satisfied"]

    def test_gamma_zero_is_pure_max_s(self):
        """gamma=0 (default) must reproduce the pure MaxS certificate exactly —
        the ceiling semantics (feasibility bracket / s_feas) rely on it."""
        synth = _synth(_make_moderate_model())
        base = synth.max_s()
        g0 = synth.max_s(gamma=0.0)
        assert base is not None and g0 is not None
        assert g0.s == pytest.approx(base.s, rel=1e-9)
        assert np.allclose(g0.P, base.P, rtol=1e-6, atol=1e-9)

    def test_gamma_slides_certificate_along_s_P_gauge(self):
        """The -gamma*log det P pull is a conditioning knob: larger gamma keeps P
        off zero (||P|| up, min-eig up) and pulls s down off the MaxS extreme,
        while the stability LMI stays satisfied (F < 0)."""
        synth = _synth(_make_moderate_model())
        sols = [synth.max_s(gamma=g) for g in (0.0, 1.0, 10.0)]
        assert all(s is not None for s in sols)
        s_vals = [s.s for s in sols]
        normP = [float(np.linalg.norm(s.P, 2)) for s in sols]
        mineigP = [float(np.min(np.linalg.eigvalsh(s.P))) for s in sols]
        # monotone: gamma up -> ||P|| up, min-eig(P) up, s down
        assert normP[0] < normP[1] < normP[2]
        assert mineigP[0] < mineigP[1] < mineigP[2]
        assert s_vals[0] >= s_vals[1] >= s_vals[2] - 1e-9
        # stability preserved throughout
        assert all(s.max_eig_F < 0 for s in sols)



@requires_mosek
class TestFeasibilityProblem:
    """Repair after a step that broke the LMIs: theta and alpha are held, the
    certificate is re-solved. Fixed ``s`` first (smallest repair, and it leaves the
    scale where the gradient/barrier put it); ``s`` freed only if that is
    infeasible; ``False`` — meaning the trainer rolls back — only if neither works.
    """

    def test_repairs_at_fixed_s_without_changing_s(self):
        """Corrupt P so the certificate breaks, then repair at the current s.
        s must be unchanged and the constraints must hold again."""
        m = _make_moderate_model()
        s_max = _synth(m).max_s().s
        s_target = 0.5 * s_max  # comfortably feasible scale
        with torch.no_grad():
            m.s.data = torch.tensor(s_target, dtype=m.s.dtype)
            m.P.data = torch.diag(torch.tensor([1.0, -1.0], dtype=m.P.dtype))  # indefinite
        assert not m.check_constraints()

        assert m.feasibility_problem() is True
        assert m.check_constraints()
        assert float(m.s) == pytest.approx(s_target, rel=1e-9)  # s untouched

    def test_theta_and_alpha_are_never_touched_by_a_repair(self):
        """The repair owns the certificate only; the prediction dynamics are the
        optimizer's. If it moved theta it would silently undo a gradient step."""
        m = _make_moderate_model()
        before = {n: p.detach().clone()
                  for n, p in m.named_parameters()
                  if n in ("A", "B", "B2", "C", "C2", "D", "D12", "D21", "tau")}
        with torch.no_grad():
            m.P.data = torch.diag(torch.tensor([1.0, -1.0], dtype=m.P.dtype))
        assert m.feasibility_problem() is True
        for name, value in before.items():
            assert torch.equal(getattr(m, name).detach(), value), name

    def test_frees_s_when_the_fixed_s_repair_is_infeasible(self):
        """Above the regionality ceiling no certificate exists at that s, but one
        does at a smaller s — so the repair succeeds and pulls s down instead of
        forcing the trainer to throw the whole step away."""
        m = _make_moderate_model()
        s_max = _synth(m).max_s().s
        with torch.no_grad():
            m.s.data = torch.tensor(10.0 * s_max, dtype=m.s.dtype)
        assert _synth(m).feasibility(10.0 * s_max) is None    # fixed-s tier fails

        assert m.feasibility_problem() is True
        assert m.check_constraints()
        assert float(m.s) < 10.0 * s_max                      # the scale was pulled back

    def test_returns_false_on_genuine_infeasibility(self):
        """Uncertifiable dynamics (unstable A): no certificate exists at ANY s, so
        neither tier can help and the trainer must roll back."""
        m = _make_moderate_model()
        with torch.no_grad():
            m.A.data = torch.tensor([[1.5, 0.0], [0.0, 1.5]], dtype=m.A.dtype)  # unstable
        assert _synth(m).feasibility(None) is None
        assert m.feasibility_problem() is False




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
