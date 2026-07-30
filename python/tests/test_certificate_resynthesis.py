"""Tests for certificate re-synthesis — the SDP-owned certificate scheme.

Wiki: ``training/certificate-resynthesis``. The pieces under test:

- ``LureCertificateSynthesizer.tight_cert`` — the ρ-pinned solve with ``ŝ = 1/s²``
  as a decision variable and the coverage band as hard LMIs (needs MOSEK).
- ``LureCertificateSynthesizer.feasibility(s, y_max=...)`` — the repair with the
  hard coverage floor (needs MOSEK).
- ``SimpleLure.freeze_certificate`` / ``coverage_ratio`` /
  ``resynthesize_certificate`` — the model-side ownership split.
- ``get_regularization_loss`` dropping the now-constant locality barriers.
- ``Trainer._repair_certificate`` (two tiers) and
  ``Trainer._resynthesize_certificate`` (trigger, guard, β anneal).
"""

import numpy as np
import pytest
import torch
import cvxpy as cp
from torch.utils.data import DataLoader, TensorDataset

from sysid.models.constrained_rnn import SimpleLure
from sysid.optimization import LureCertificateSynthesizer
from sysid.training import get_loss_function, get_optimizer
from sysid.training.trainer import Trainer


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


def _make_model(s_value: float = 0.05, learn_L: bool = True) -> SimpleLure:
    """The shared small, stable Lure system (mirrors tests/test_max_s.py)."""
    m = SimpleLure(
        nd=1, ne=1, nx=2, nw=1, activation="dzn", custom_params={"learn_L": learn_L}
    )
    with torch.no_grad():
        m.A.data = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=m.A.dtype)
        m.B.data = torch.tensor([[0.1], [0.1]], dtype=m.B.dtype)
        m.B2.data = torch.zeros_like(m.B2)
        m.C.data = torch.tensor([[1.0, 0.0]], dtype=m.C.dtype)
        m.C2.data = torch.tensor([[0.1, 0.1]], dtype=m.C2.dtype)
        m.D21.data = torch.tensor([[0.1]], dtype=m.D21.dtype)
        m.tau.data = torch.tensor(float(np.log(0.9 / 0.1)))  # alpha = 0.9
        m.s.data = torch.tensor(float(s_value))
        m.P.data = torch.eye(2, dtype=m.P.dtype)
    return m


def _synth(model) -> LureCertificateSynthesizer:
    return LureCertificateSynthesizer.from_model(model)


def _y_bar(model, P, s) -> float:
    """Physical certified half-width σ·s·√(λ_min(C P Cᵀ)) for a solved (P, s)."""
    C = model.C.detach().numpy()
    lam_min = max(float(np.min(np.linalg.eigvalsh(C @ P @ C.T))), 0.0)
    return float(float(model.output_std) * s * np.sqrt(lam_min))


def _reachable_y_max(model, fraction: float = 0.5) -> float:
    """A y_max the model can certify: a fraction of the MaxS certified width."""
    cert = _synth(model).max_s()
    assert cert is not None
    return fraction * _y_bar(model, cert.P, cert.s)


# --------------------------------------------------------------------- TightCert
class TestTightCert:
    @requires_mosek
    def test_pins_rho_in_band(self):
        """ρ = (ȳ/y_max)^nx lands inside [1, β^nx] — the whole point of the solve."""
        m = _make_model()
        y_max = _reachable_y_max(m)
        beta = 1.5
        sol = _synth(m).tight_cert(y_max=y_max, beta=beta)
        assert sol is not None
        assert sol.band_enforced
        assert sol.rho >= 1.0 - 1e-6
        assert sol.rho <= beta ** m.nx + 1e-6
        # ȳ recomputed from the returned (P, s) agrees with the reported one.
        assert _y_bar(m, sol.P, sol.s) == pytest.approx(sol.y_bar, rel=1e-6)

    @requires_mosek
    def test_tighter_beta_gives_tighter_rho(self):
        """Shrinking the over-claim budget shrinks the achieved ρ."""
        m = _make_model()
        y_max = _reachable_y_max(m)
        loose = _synth(m).tight_cert(y_max=y_max, beta=4.0)
        tight = _synth(m).tight_cert(y_max=y_max, beta=1.05)
        assert loose is not None and tight is not None
        assert tight.rho <= loose.rho + 1e-6
        assert tight.rho <= 1.05 ** m.nx + 1e-6

    @requires_mosek
    def test_solution_satisfies_the_lmis(self):
        """The returned certificate is feasible for stability + locality."""
        m = _make_model()
        sol = _synth(m).tight_cert(y_max=_reachable_y_max(m), beta=2.0)
        assert sol is not None
        assert sol.max_eig_F < 0.0
        m._apply_certificate_solution(sol)
        assert m.check_constraints()

    @requires_mosek
    def test_no_y_max_degenerates_to_max_s(self):
        """Without a level there is no band; the solve falls back to MaxS."""
        m = _make_model()
        sol = _synth(m).tight_cert(y_max=None)
        ref = _synth(m).max_s()
        assert sol is not None and ref is not None
        assert not sol.band_enforced
        assert sol.rho is None and sol.y_bar is None
        assert sol.s == pytest.approx(ref.s, rel=1e-6)

    @requires_mosek
    def test_unreachable_level_is_infeasible(self):
        """A y_max far above what θ can certify has no banded certificate."""
        m = _make_model()
        huge = 1e6 * _reachable_y_max(m)
        assert _synth(m).tight_cert(y_max=huge, beta=1.5) is None

    @requires_mosek
    def test_floor_only_when_beta_none(self):
        """β=None drops the ceiling: coverage still holds, ρ may exceed the band."""
        m = _make_model()
        y_max = _reachable_y_max(m)
        sol = _synth(m).tight_cert(y_max=y_max, beta=None)
        assert sol is not None
        assert sol.rho >= 1.0 - 1e-6
        assert sol.beta is None


# -------------------------------------------------- feasibility + coverage floor
class TestRepairWithCoverageFloor:
    @requires_mosek
    def test_floor_is_enforced(self):
        """The repaired certificate covers y_max when the floor is passed."""
        m = _make_model()
        s = 3.0
        base = _synth(m).feasibility(s)
        assert base is not None
        y_max = 0.9 * _y_bar(m, base.P, s)  # reachable but above the min-norm repair
        repaired = _synth(m).feasibility(s, y_max=y_max)
        assert repaired is not None
        assert _y_bar(m, repaired.P, s) >= y_max - 1e-6

    @requires_mosek
    def test_floor_free_repair_unchanged(self):
        """Default (no y_max) keeps the historical min-norm repair."""
        m = _make_model()
        sol = _synth(m).feasibility(3.0)
        assert sol is not None
        assert sol.s == pytest.approx(3.0)

    @requires_mosek
    def test_floor_binds_and_costs_norm(self):
        """The floor is a real constraint: it lifts ‖P‖ above the min-norm repair.

        Note it is a *mild* one at fixed ``s`` — the stability LMI bounds ``P``
        from below (the non-homogeneous disturbance blocks) but not from above
        whenever a global certificate exists, so the floor can usually be met by
        inflating ``P``. The ``min t`` objective keeps that inflation minimal.
        """
        m = _make_model()
        s = 3.0
        base = _synth(m).feasibility(s)
        assert base is not None
        y_max = 2.0 * _y_bar(m, base.P, s)  # more than the min-norm repair yields
        repaired = _synth(m).feasibility(s, y_max=y_max)
        assert repaired is not None
        assert _y_bar(m, base.P, s) < y_max  # the floor-free repair under-covers
        assert _y_bar(m, repaired.P, s) >= y_max - 1e-6  # the floored one covers
        assert np.linalg.norm(repaired.P, 2) > np.linalg.norm(base.P, 2)


# ------------------------------------------------------------ model-side wiring
class TestFreezeCertificate:
    def test_freezes_certificate_params_only(self):
        m = _make_model()
        frozen = m.freeze_certificate(freeze_alpha=True)
        assert set(frozen) == {"P", "la", "s", "L", "tau"}
        for name in ("P", "la", "s", "L", "tau"):
            assert not getattr(m, name).requires_grad
        # theta keeps its gradient
        for name in ("A", "B", "B2", "C", "C2", "D21"):
            assert getattr(m, name).requires_grad
        assert m.certificate_frozen

    def test_freeze_alpha_false_keeps_tau_trainable(self):
        m = _make_model()
        frozen = m.freeze_certificate(freeze_alpha=False)
        assert "tau" not in frozen
        assert m.tau.requires_grad

    def test_barrier_drops_constant_locality_terms(self):
        """With κ frozen the nz locality LMIs contain no θ, so their barrier terms
        are constants and are skipped — the loss reduces to the stability term."""
        m = _make_model(s_value=1.0)
        m.freeze_certificate()
        loss = m.get_regularization_loss()
        stability_only = -torch.logdet(m.get_lmis()[0]())
        assert float(loss) == pytest.approx(float(stability_only), rel=1e-6)
        assert loss.requires_grad  # still a live penalty on theta

    def test_barrier_gradient_reaches_theta_not_certificate(self):
        m = _make_model(s_value=1.0)
        m.freeze_certificate()
        m.get_regularization_loss().backward()
        assert m.A.grad is not None and torch.any(m.A.grad != 0)
        assert m.s.grad is None and m.P.grad is None

    def test_unfrozen_barrier_keeps_every_term(self):
        """Backward compatibility: nothing is skipped while κ is trainable."""
        m = _make_model(s_value=1.0)
        loss = m.get_regularization_loss()
        total = sum(-torch.logdet(f()) for f in m.get_lmis())
        assert float(loss) == pytest.approx(float(total), rel=1e-6)

    def test_frozen_params_do_not_move_under_a_step(self):
        m = _make_model(s_value=1.0)
        m.freeze_certificate()
        opt = get_optimizer(m.parameters(), learning_rate=0.1)
        s0, P0 = float(m.s), m.P.detach().clone()
        opt.zero_grad()
        m.get_regularization_loss().backward()
        opt.step()
        assert float(m.s) == pytest.approx(s0)
        assert torch.allclose(m.P, P0)


class TestCoverageRatio:
    def test_matches_the_formula(self):
        m = _make_model(s_value=2.0)
        m.set_output_coverage_level(0.5, output_std=1.0)
        expected = (_y_bar(m, m.P.detach().numpy(), 2.0) / 0.5) ** m.nx
        assert m.coverage_ratio() == pytest.approx(expected, rel=1e-6)

    def test_none_without_y_max(self):
        assert _make_model().coverage_ratio() is None

    def test_below_one_when_undercovering(self):
        m = _make_model(s_value=0.05)
        m.set_output_coverage_level(100.0, output_std=1.0)
        assert m.coverage_ratio() < 1.0


class TestResynthesizeCertificate:
    @requires_mosek
    def test_applies_and_lands_in_band(self):
        m = _make_model()
        y_max = _reachable_y_max(m)
        m.set_output_coverage_level(y_max, output_std=1.0)
        out = m.resynthesize_certificate(beta=1.5)
        assert out["success"] and out["applied"]
        assert 1.0 - 1e-6 <= m.coverage_ratio() <= 1.5 ** m.nx + 1e-6
        assert float(m.s) == pytest.approx(out["s"], rel=1e-6)

    @requires_mosek
    def test_guard_rejects_clean_to_dirty(self, monkeypatch):
        """A new certificate that breaks a CLEAN rollout is rolled back."""
        m = _make_model()
        m.set_output_coverage_level(_reachable_y_max(m), output_std=1.0)
        before = {"P": m.P.detach().clone(), "s": float(m.s)}
        counts = iter([0, 7])  # before, after
        monkeypatch.setattr(m, "_count_input_violations", lambda *a, **k: next(counts))
        out = m.resynthesize_certificate(
            beta=1.5, guard_inputs=torch.zeros(1, 4, 1)
        )
        assert out["success"] and not out["applied"]
        assert out["reason"] == "guard_rejected"
        assert torch.allclose(m.P, before["P"])
        assert float(m.s) == pytest.approx(before["s"])

    @requires_mosek
    def test_guard_accepts_when_not_worse(self, monkeypatch):
        m = _make_model()
        m.set_output_coverage_level(_reachable_y_max(m), output_std=1.0)
        counts = iter([3, 3])
        monkeypatch.setattr(m, "_count_input_violations", lambda *a, **k: next(counts))
        out = m.resynthesize_certificate(beta=1.5, guard_inputs=torch.zeros(1, 4, 1))
        assert out["applied"]

    @requires_mosek
    def test_guard_does_not_lock_up_during_a_blowup(self, monkeypatch):
        """The already-violating case MUST accept — a count comparison would
        freeze the certificate for the whole excursion (Duffing epoch 23)."""
        m = _make_model()
        m.set_output_coverage_level(_reachable_y_max(m), output_std=1.0)
        P0 = m.P.detach().clone()
        counts = iter([2, 50])  # already dirty, new one dirtier
        monkeypatch.setattr(m, "_count_input_violations", lambda *a, **k: next(counts))
        out = m.resynthesize_certificate(beta=1.5, guard_inputs=torch.zeros(1, 4, 1))
        assert out["applied"]
        assert not torch.allclose(m.P, P0)

    @requires_mosek
    def test_reports_failure_without_touching_the_model(self):
        m = _make_model()
        m.set_output_coverage_level(1e9, output_std=1.0)  # unreachable
        P0, s0 = m.P.detach().clone(), float(m.s)
        out = m.resynthesize_certificate(beta=1.2)
        assert not out["success"] and out["reason"] == "sdp_infeasible"
        assert torch.allclose(m.P, P0) and float(m.s) == pytest.approx(s0)

    @requires_mosek
    def test_without_y_max_falls_back_to_max_s(self):
        m = _make_model()
        ref = _synth(m).max_s()
        out = m.resynthesize_certificate()
        assert out["success"] and out["applied"]
        assert not out["band_enforced"]
        assert float(m.s) == pytest.approx(ref.s, rel=1e-6)


# ----------------------------------------------------------------- trainer wiring
def _make_loader(u_amp: float = 0.1, y_level: float = 0.9, N: int = 5, B: int = 4):
    d = u_amp * torch.ones(B, N, 1)
    e = y_level * torch.ones(B, N, 1)
    return DataLoader(TensorDataset(d, e), batch_size=2)


def _make_trainer(tmp_path, model, loader=None, lr=0.0, **kwargs) -> Trainer:
    loader = loader if loader is not None else _make_loader()
    return Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        loss_fn=get_loss_function("mse"),
        optimizer=get_optimizer(model.parameters(), learning_rate=lr),
        device="cpu",
        output_dir=str(tmp_path / "o"),
        model_dir=str(tmp_path / "m"),
        log_dir=str(tmp_path / "l"),
        mlflow_tracking=False,
        **kwargs,
    )


class TestTrainerOwnership:
    def test_freezes_at_construction(self, tmp_path):
        m = _make_model()
        _make_trainer(tmp_path, m, freeze_certificate=True)
        assert not m.s.requires_grad and not m.P.requires_grad
        assert not m.tau.requires_grad

    def test_no_freeze_by_default(self, tmp_path):
        m = _make_model()
        _make_trainer(tmp_path, m)
        assert m.s.requires_grad and m.P.requires_grad

    def test_y_max_derived_when_resynthesis_needs_it(self, tmp_path):
        """Re-synthesis needs the level even with every penalty weight at zero."""
        m = _make_model()
        _make_trainer(
            tmp_path, m, _make_loader(0.1, 0.9),
            regularization_weight=1e-8, resynthesize_certificate=True,
        )
        assert float(m.y_max) == pytest.approx(0.9)

    def test_two_tier_repair_falls_back(self, tmp_path, monkeypatch):
        """Tier 1 (with floor) infeasible -> tier 2 (floor-free) succeeds."""
        m = _make_model()
        t = _make_trainer(tmp_path, m, repair_enforce_coverage=True)
        calls = []

        def fake(enforce_coverage=False):
            calls.append(enforce_coverage)
            return not enforce_coverage  # only the floor-free repair works

        monkeypatch.setattr(m, "feasibility_problem", fake)
        assert t._repair_certificate() is True
        assert calls == [True, False]
        assert t.epoch_coverage_repair_fallbacks == 1

    def test_two_tier_repair_reports_total_failure(self, tmp_path, monkeypatch):
        m = _make_model()
        t = _make_trainer(tmp_path, m, repair_enforce_coverage=True)
        monkeypatch.setattr(m, "feasibility_problem", lambda **kw: False)
        assert t._repair_certificate() is False

    def test_single_tier_when_floor_disabled(self, tmp_path, monkeypatch):
        m = _make_model()
        t = _make_trainer(tmp_path, m, repair_enforce_coverage=False)
        calls = []
        monkeypatch.setattr(
            m, "feasibility_problem",
            lambda enforce_coverage=False: (calls.append(enforce_coverage), False)[1],
        )
        assert t._repair_certificate() is False
        assert calls == [False]  # no second attempt

    def test_resynthesis_off_is_a_noop(self, tmp_path):
        m = _make_model()
        t = _make_trainer(tmp_path, m)
        assert t._resynthesize_certificate(0) == {}

    def test_cadence_off_means_rho_only(self, tmp_path, monkeypatch):
        """resynthesis_every <= 0: nothing fires while rho stays inside the band."""
        m = _make_model(s_value=0.05)
        m.set_output_coverage_level(0.01, output_std=1.0)   # rho = 25, inside [1, 1e12]
        t = _make_trainer(
            tmp_path, m, resynthesize_certificate=True, resynthesis_every=0,
            resynthesis_beta=1e6, resynthesis_beta_decay=1.0, resynthesis_beta_min=1e6,
        )
        seen = []
        monkeypatch.setattr(m, "resynthesize_certificate",
                            lambda **kw: (seen.append(1), {"success": True, "applied": True,
                                                           "s": 1.0, "rho": 1.0, "reason": "ok",
                                                           "norm_P": 1.0})[1])
        for epoch in range(5):
            t._resynthesize_certificate(epoch)
        assert seen == []

    def test_logs_rho_before_and_after(self, tmp_path, monkeypatch):
        """The drift itself must be visible: `rho` alone is the POST value."""
        m = _make_model(s_value=0.05)
        m.set_output_coverage_level(100.0, output_std=1.0)   # rho << 1 -> triggers
        t = _make_trainer(tmp_path, m, resynthesize_certificate=True, resynthesis_every=0)
        monkeypatch.setattr(m, "resynthesize_certificate", lambda **kw: {
            "success": True, "applied": False, "reason": "guard_rejected"})
        out = t._resynthesize_certificate(3)
        assert "rho_before" in out and out["rho_before"] < 1.0
        assert out["resynth_trigger"] == 1.0

    def test_target_mid_inflates_the_solved_level(self, tmp_path, monkeypatch):
        """Restore to the middle of [1, beta^nx], not to its lower edge."""
        m = _make_model(s_value=0.05)
        m.set_output_coverage_level(1.0, output_std=1.0)
        t = _make_trainer(tmp_path, m, resynthesize_certificate=True, resynthesis_every=1,
                          resynthesis_beta=4.0, resynthesis_target_mid=True)
        seen = {}
        monkeypatch.setattr(m, "resynthesize_certificate", lambda **kw: (
            seen.update(kw), {"success": True, "applied": False, "reason": "guard_rejected"})[1])
        t._resynthesize_certificate(0)
        assert seen["y_max"] == pytest.approx(2.0)   # sqrt(beta) * y_max
        assert seen["beta"] == pytest.approx(2.0)    # sqrt(beta)

    def test_cadence_skips_epochs(self, tmp_path, monkeypatch):
        m = _make_model(s_value=0.05)
        # ȳ = 0.05 vs y_max = 0.01 -> rho = 25, inside the [1, (1e6)^2] band, so
        # only the cadence can trigger.
        m.set_output_coverage_level(0.01, output_std=1.0)
        t = _make_trainer(
            tmp_path, m, resynthesize_certificate=True, resynthesis_every=3,
            resynthesis_beta=1e6,
            resynthesis_beta_decay=1.0, resynthesis_beta_min=1e6,
        )
        seen = []
        monkeypatch.setattr(
            m, "resynthesize_certificate",
            lambda **kw: (seen.append(1), {"success": True, "applied": True, "s": 1.0,
                                           "rho": 1.0, "reason": "ok", "norm_P": 1.0})[1],
        )
        for epoch in range(6):
            t._resynthesize_certificate(epoch)
        assert len(seen) == 2  # epochs 0 and 3

    def test_drift_triggers_off_cadence(self, tmp_path, monkeypatch):
        """ρ outside the band re-solves even when the cadence would skip."""
        m = _make_model(s_value=0.05)
        m.set_output_coverage_level(100.0, output_std=1.0)  # rho << 1
        t = _make_trainer(
            tmp_path, m, resynthesize_certificate=True, resynthesis_every=1000,
            resynthesis_beta=1.5,
        )
        seen = []
        monkeypatch.setattr(
            m, "resynthesize_certificate",
            lambda **kw: (seen.append(1), {"success": True, "applied": False,
                                           "reason": "guard_rejected"})[1],
        )
        t._resynthesize_certificate(epoch=7)
        assert len(seen) == 1
        assert t.resynthesis_rejected == 1

    def test_beta_holds_still_on_a_triggered_epoch(self, tmp_path, monkeypatch):
        """The requirement must not move while the certificate is chasing it."""
        m = _make_model(s_value=0.05)
        m.set_output_coverage_level(100.0, output_std=1.0)   # rho << 1 -> out of band
        t = _make_trainer(tmp_path, m, resynthesize_certificate=True, resynthesis_every=0,
                          resynthesis_beta=2.0, resynthesis_beta_decay=0.5,
                          resynthesis_beta_min=1.0)
        monkeypatch.setattr(m, "resynthesize_certificate", lambda **kw: {
            "success": True, "applied": True, "s": 1.0, "rho": 1.0,
            "reason": "ok", "norm_P": 1.0})
        t._resynthesize_certificate(0)
        assert t.resynthesis_beta == pytest.approx(2.0)   # unchanged

    def test_beta_tightens_on_a_healthy_epoch(self, tmp_path, monkeypatch):
        """rho comfortably in band => the band is too loose => squeeze it."""
        m = _make_model(s_value=0.05)
        m.set_output_coverage_level(0.01, output_std=1.0)   # rho = 25, well inside
        t = _make_trainer(tmp_path, m, resynthesize_certificate=True, resynthesis_every=0,
                          resynthesis_beta=1e6, resynthesis_beta_decay=0.5,
                          resynthesis_beta_min=1.0)
        t._resynthesize_certificate(0)
        assert t.resynthesis_beta == pytest.approx(0.5e6)

    def test_beta_anneals_down_and_floors(self, tmp_path, monkeypatch):
        m = _make_model(s_value=2.0)
        # y_bar = 2.0 => rho = (2/1.3333)^2 = 2.25, inside [1, beta^2 = 4]
        m.set_output_coverage_level(1.33333333, output_std=1.0)
        t = _make_trainer(
            tmp_path, m, resynthesize_certificate=True, resynthesis_every=1,
            resynthesis_beta=2.0, resynthesis_beta_decay=0.5, resynthesis_beta_min=1.1,
        )
        monkeypatch.setattr(m, "resynthesize_certificate", lambda **kw: {
            "success": True, "applied": True, "s": 1.0, "rho": 1.0,
            "reason": "ok", "norm_P": 1.0,
        })
        t._resynthesize_certificate(1)
        assert t.resynthesis_beta == pytest.approx(1.1)  # 1.0 clipped to beta_min
        t._resynthesize_certificate(2)
        assert t.resynthesis_beta == pytest.approx(1.1)  # stays at the floor

    def test_beta_widens_after_all_rollback_epoch(self, tmp_path, monkeypatch):
        m = _make_model()
        m.set_output_coverage_level(0.9, output_std=1.0)
        t = _make_trainer(
            tmp_path, m, resynthesize_certificate=True, resynthesis_every=1000,
            resynthesis_beta=2.0, resynthesis_beta_grow=1.5,
        )
        monkeypatch.setattr(m, "resynthesize_certificate", lambda **kw: {
            "success": True, "applied": True, "s": 1.0, "rho": 1.0,
            "reason": "ok", "norm_P": 1.0,
        })
        t.epoch_rollback_count = len(t.train_loader)  # every batch rolled back
        t._resynthesize_certificate(1)
        assert t.resynthesis_beta == pytest.approx(3.0)

    def test_failure_is_counted_and_logged(self, tmp_path, monkeypatch):
        m = _make_model()
        m.set_output_coverage_level(0.9, output_std=1.0)
        t = _make_trainer(tmp_path, m, resynthesize_certificate=True)
        monkeypatch.setattr(m, "resynthesize_certificate", lambda **kw: {
            "success": False, "applied": False, "reason": "sdp_infeasible",
        })
        out = t._resynthesize_certificate(0)
        assert t.resynthesis_failed == 1
        assert out["resynth_failed"] == 1.0

    @requires_mosek
    def test_full_epoch_keeps_rho_in_band(self, tmp_path):
        """End-to-end: a real epoch with the scheme on leaves ρ inside the band."""
        m = _make_model()
        m.set_output_coverage_level(_reachable_y_max(m), output_std=1.0)
        t = _make_trainer(
            tmp_path, m, _make_loader(0.1, 0.9), lr=1e-3,
            regularization_weight=1e-6,
            freeze_certificate=True,
            repair_enforce_coverage=True,
            resynthesize_certificate=True,
            resynthesis_beta=1.5,
            resynthesis_beta_decay=1.0,
            resynthesis_beta_min=1.5,
        )
        t.train_epoch()
        metrics = t._resynthesize_certificate(epoch=0)
        assert metrics["rho"] <= 1.5 ** m.nx + 1e-6
        assert metrics["rho"] >= 1.0 - 1e-6
        assert m.check_constraints()
