"""Tests for the certificate SDPs on SimpleLure.

The clean design has these SDPs (see the wiki certificate-synthesis notes):
  * MaxVol (``_max_vol_sdp`` / ``_max_vol_at_s``) — max invariant-ellipsoid
    *volume* sⁿˣ·√(det P) over the s-sweep; the OPERATIVE certificate.
  * MaxS  (``_max_s_sdp``)        — max feasible s; the feasibility ceiling that
    brackets the MaxVol sweep and the coverage sweep (no longer applied).
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
from sysid.optimization import LureCertificateSynthesizer
from sysid.utils import get_volume_of_ellipsoid


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

    def test_post_process_ceiling_matches_max_s(self):
        """post_process (MaxVol) reports max_s's s as the feasibility ceiling
        s_feas, and its operative (volume-optimal) s never exceeds that ceiling."""
        s_direct = _synth(_make_model()).max_s().s
        out = _make_model().post_process()
        assert out["success"]
        assert out["max_vol"]["s_feas"] == pytest.approx(s_direct, rel=1e-4)
        assert 0.0 < out["s_opt"] <= s_direct * (1.0 + 1e-6)

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
        """post_process routes through MaxVol for the global (L=0) model;
        s stays fixed (nothing to sweep)."""
        out = _make_global_model(s_value=1.0).post_process()
        assert out["success"]
        assert out["s_opt"] == pytest.approx(1.0, rel=1e-9)
        assert out["constraints_satisfied"]


@requires_mosek
class TestMaxVol:
    """MaxVol (``synthesizer.max_vol`` / ``max_vol_at_s``): the operative
    certificate — maximizes the ellipsoid volume sⁿˣ·√(det P) over the s-sweep."""

    def test_max_vol_is_feasible(self):
        m = _make_moderate_model()
        sol = _synth(m).max_vol()
        assert sol is not None
        assert sol.s > 0 and sol.volume > 0
        m._apply_certificate_solution(sol)
        assert m.check_constraints()

    def test_operative_s_within_feasibility_bracket(self):
        """The volume-optimal s lives in (0, s_feas]; s_feas is the MaxS ceiling."""
        m = _make_moderate_model()
        synth = _synth(m)
        sol = synth.max_vol()
        assert sol is not None
        assert sol.s_feas == pytest.approx(synth.max_s().s, rel=1e-4)
        assert 0.0 < sol.s <= sol.s_feas * (1.0 + 1e-6)

    def test_selected_point_maximizes_volume_over_sweep(self):
        """The returned (s, P) is the argmax of the recorded per-grid volumes."""
        sol = _synth(_make_moderate_model()).max_vol(n_grid=15)
        assert sol is not None
        assert sol.volume == pytest.approx(
            max(pt.volume for pt in sol.sweep), rel=1e-9
        )
        assert len(sol.sweep) >= 1

    def test_max_vol_beats_max_s_volume(self):
        """MaxVol must certify a volume at least as large as the MaxS solution's
        (it optimizes exactly that objective, so it can never be smaller)."""
        synth = _synth(_make_moderate_model())
        max_s = synth.max_s()
        vol_max_s = get_volume_of_ellipsoid(max_s.P, max_s.s)
        max_vol = synth.max_vol()
        assert max_vol is not None
        assert max_vol.volume >= vol_max_s * (1.0 - 1e-6)

    def test_max_vol_at_s_is_convex_and_feasible(self):
        """The fixed-s slice (a convex max-det SDP) is feasible below the ceiling
        and returns a strictly positive-volume certificate."""
        m = _make_moderate_model()
        synth = _synth(m)
        s_feas = synth.max_s().s
        sol = synth.max_vol_at_s(0.5 * s_feas)
        assert sol is not None
        assert sol.volume > 0 and np.isfinite(sol.logdet_P)
        m._apply_certificate_solution(sol)
        assert m.check_constraints()

    def test_max_vol_runs_for_global_model(self):
        """learn_L=False (L=0): no sweep (s fixed); returns the fixed s, L=0 and a
        positive-volume largest ellipsoid at that s."""
        m = _make_global_model(s_value=1.0)
        sol = _synth(m).max_vol()
        assert sol is not None
        assert sol.s == pytest.approx(1.0, rel=1e-9)
        assert sol.s_feas == pytest.approx(1.0, rel=1e-9)
        assert np.allclose(sol.L, 0.0)
        assert sol.locality_min_eigs == []  # no locality LMIs
        assert sol.volume > 0
        m._apply_certificate_solution(sol)
        assert m.check_constraints()

    def test_max_vol_is_pure(self):
        """max_vol operates on a synthesizer snapshot and must not mutate the model."""
        m = _make_moderate_model()
        before = (m.P.detach().clone(), m.s.detach().clone(), m.L.detach().clone())
        sol = _synth(m).max_vol(n_grid=8)
        assert sol is not None
        assert torch.equal(m.P, before[0])
        assert torch.equal(m.s, before[1])
        assert torch.equal(m.L, before[2])


def _make_coverage_model(y_max: float = 1.0) -> SimpleLure:
    """Regional model with a non-trivial output map C = [1, 0] and a physical
    output level set, so the coverage sweep (hence rho) is well defined."""
    m = _make_moderate_model()
    with torch.no_grad():
        m.C.data = torch.tensor([[1.0, 0.0]], dtype=m.C.dtype)
    m.set_output_coverage_level(y_max, 1.0)  # physical y_max, output_std = 1
    return m


@requires_mosek
class TestC2Calibration:
    """C2 calibration: scale C2 so the max-volume set *just* covers the coverage
    set (rho = vol(MaxVol)/vol(tightest coverage) driven toward 1 from above)."""

    def test_rho_is_monotone_decreasing_in_c2(self):
        """More C2 coupling -> smaller certifiable set -> lower volume and lower
        rho. This is the premise the search relies on."""
        synth = _synth(_make_coverage_model(y_max=1.0))
        lo = synth.coverage_ratio_at_c2(0.5, 1.0, mv_n_grid=10, cov_n_grid=10)
        hi = synth.coverage_ratio_at_c2(4.0, 1.0, mv_n_grid=10, cov_n_grid=10)
        assert lo.max_vol.volume > hi.max_vol.volume  # volume drops
        assert lo.rho > hi.rho                        # rho drops

    def test_calibration_returns_covering_regional_factor(self):
        """Calibration returns the smallest covering set: a finite rho >= 1
        (regional, still covers y_max), reached by growing C2. The synthesizer is
        pure, so the caller scales C2 by the winning factor to apply it."""
        m = _make_coverage_model(y_max=1.0)
        cal = _synth(m).calibrate_c2(
            1.0, eps=0.05, max_iter=15, mv_n_grid=10, cov_n_grid=10
        )
        assert cal is not None
        assert 1e-3 <= cal.f <= 1e3
        # Smallest covering set: rho finite and >= 1 (max set still contains the
        # coverage requirement); the base was well above the band, so C2 grew.
        assert np.isfinite(cal.rho) and cal.rho >= 1.0 - 1e-6
        assert cal.f > 1.0
        # Apply the winning factor + certificate (what initialize_parameters does)
        # and verify the model is feasible.
        m.C2.data = m.C2.data * cal.f
        m._apply_certificate_solution(cal.max_vol)
        assert m.check_constraints()

    def test_calibration_in_band_flag_matches_rho(self):
        """in_band is exactly the 0 <= rho-1 < eps test on the returned rho."""
        eps = 0.05
        cal = _synth(_make_coverage_model(y_max=2.0)).calibrate_c2(
            2.0, eps=eps, max_iter=15, mv_n_grid=10, cov_n_grid=10
        )
        assert cal is not None
        expected = bool(np.isfinite(cal.rho) and 0.0 <= cal.rho - 1.0 < eps)
        assert cal.in_band == expected


@requires_mosek
class TestFeasibilityProblem:
    """Feasibility: repair P, M, L at a FIXED s; s is never modified."""

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

    def test_returns_false_when_s_above_s_max(self):
        """No feasible P, M, L exists above the regionality ceiling -> False
        (the trainer then rolls the update back). s is not clamped."""
        m = _make_moderate_model()
        s_max = _synth(m).max_s().s
        with torch.no_grad():
            m.s.data = torch.tensor(10.0 * s_max, dtype=m.s.dtype)
        assert m.feasibility_problem() is False

    def test_returns_false_on_genuine_infeasibility(self):
        """Uncertifiable dynamics (unstable A) -> False at any s."""
        m = _make_moderate_model()
        with torch.no_grad():
            m.A.data = torch.tensor([[1.5, 0.0], [0.0, 1.5]], dtype=m.A.dtype)  # unstable
        assert m.feasibility_problem() is False


class _MockNormalizer:
    """Minimal normalizer stand-in for ``initialize_parameters``/``_init_identity``."""

    def __init__(self, input_std=1.0, output_std=1.0):
        self.input_std = np.array([[input_std]])
        self.output_std = np.array([[output_std]])

    def transform_inputs(self, u):
        return u  # identity: the mock data is already "normalized"


class TestInitializationInfeasible:
    """During initialization a failing certificate SDP means no feasible parameter
    set exists, so it must raise (not crash on a None solution)."""

    def test_initialize_parameters_raises_when_max_s_fails(self, monkeypatch):
        m = _make_model()
        # Simulate an infeasible/failed certificate SDP at its root (MaxS): the
        # calibration, MaxVol and the ceiling all bottom out here.
        monkeypatch.setattr(LureCertificateSynthesizer, "max_s", lambda self: None)

        u = np.zeros((2, 4, 1))
        y = np.zeros((2, 4, 1))
        with pytest.raises(RuntimeError, match="no feasible parameter set"):
            m.initialize_parameters(
                train_inputs=u,
                train_states=None,
                train_outputs=y,
                normalizer=_MockNormalizer(),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
