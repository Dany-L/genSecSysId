"""Tests for the output-coverage certificate, as an ANALYSIS quantity.

Coverage is no longer imposed anywhere: the floor ``(sigma*s)^2 C P C^T >= y_max^2 I``
was dropped from training because the model class provably cannot reach it on this
benchmark (the certified set is an ellipsoid inscribed in a non-ellipsoidal basin —
see the wiki note certificate-synthesis/ellipsoidal-conservatism, and the fitted
reference model measures rho = 0.42 < 1). What remains, and is tested here, is
measuring and reporting it.

Covers:
- ``max_abs_output`` utility (the data-derived output level y_max).
- the trainer RECORDING y_max/output_std on the model for reporting.
- ``SimpleLure.solve_output_coverage_certificate`` binding SDP + s-sweep and the
  post-process coverage report (needs MOSEK; skipped otherwise).
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
from sysid.utils import max_abs_output


def _synth(model) -> LureCertificateSynthesizer:
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
    """A small, stable Lure system for which the certificate SDPs are feasible.

    Mirrors tests/test_max_s.py::_make_model so the coverage tests share a
    known-feasible starting point.
    """
    m = SimpleLure(nd=1, ne=1, nx=2, nw=1, activation="dzn", custom_params={"learn_L": True})
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


def _make_global_model(s_value: float = 1.0) -> SimpleLure:
    """A globally stable model (learn_L=False => L=0, fixed s/alpha).

    The coverage SDP must stay well-posed here: no locality LMIs and s is not a
    variable (mirrors tests/test_max_s.py::_make_global_model)."""
    m = SimpleLure(nd=1, ne=1, nx=2, nw=1, activation="dzn", custom_params={"learn_L": False})
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


class TestMaxAbsOutput:
    def test_matches_plain_max_abs(self):
        y = np.array([[[0.2], [-0.9], [0.5]], [[0.1], [0.3], [-0.4]]])
        assert max_abs_output(y) == pytest.approx(0.9)

    def test_ignores_nan_padding(self):
        y = np.array([[[0.2], [np.nan], [0.7]], [[np.nan], [-0.5], [np.nan]]])
        # plain np.max would return nan; the utility must ignore padding.
        assert max_abs_output(y) == pytest.approx(0.7)

    def test_normalized_level_scales_with_std(self):
        # scale_only-style: y_max_n = y_max_phys / output_std.
        y_phys = np.array([[[2.0], [-4.0]]])
        std = 2.0
        assert max_abs_output(y_phys / std) == pytest.approx(2.0)


def _make_loader(u_amp: float, y_level: float, N: int = 5, B: int = 4) -> DataLoader:
    """Constant-amplitude inputs with constant (normalized) target level."""
    d = u_amp * torch.ones(B, N, 1)
    e = y_level * torch.ones(B, N, 1)
    return DataLoader(TensorDataset(d, e), batch_size=2)


def _make_trainer(tmp_path, model, loader, lr=0.0, **kwargs) -> Trainer:
    optimizer = get_optimizer(model.parameters(), learning_rate=lr)
    return Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        loss_fn=get_loss_function("mse"),
        optimizer=optimizer,
        device="cpu",
        output_dir=str(tmp_path / "o"),
        model_dir=str(tmp_path / "m"),
        log_dir=str(tmp_path / "l"),
        mlflow_tracking=False,
        **kwargs,
    )


class TestTrainerWiring:
    """The trainer records y_max so rho can be reported; it never constrains it."""

    def test_y_max_recorded_from_the_loader(self, tmp_path):
        """Physical y_max = max|target| over the loader * output_std (=1 here)."""
        m = _make_model()
        _make_trainer(tmp_path, m, _make_loader(0.1, 0.9), regularization_weight=1e-8)
        assert float(m.y_max) == pytest.approx(0.9)

    def test_y_max_scaled_by_output_std(self, tmp_path):
        """The loader is normalized, so physical y_max = max|e| * output_std."""
        m = _make_model()
        _make_trainer(
            tmp_path, m, _make_loader(0.1, 0.9),
            regularization_weight=1e-8, output_std=3.0,
        )
        assert float(m.y_max) == pytest.approx(0.9 * 3.0)
        assert float(m.output_std) == pytest.approx(3.0)

    def test_recording_does_not_overwrite_an_existing_level(self, tmp_path):
        """initialize_parameters sets y_max from the RAW data; the trainer's
        loader-derived fallback must not clobber it."""
        m = _make_model()
        m.set_output_coverage_level(2.5, 1.0)
        _make_trainer(tmp_path, m, _make_loader(0.1, 0.9), regularization_weight=1e-8)
        assert float(m.y_max) == pytest.approx(2.5)

    def test_training_carries_no_coverage_term(self, tmp_path):
        """The objective is prediction loss + barrier (+ input) only — the
        coverage and tightness penalties are gone, so no epoch metric reports them."""
        m = _make_model(s_value=0.2)   # badly under-covering: would have been penalized
        t = _make_trainer(
            tmp_path, m, _make_loader(0.1, 0.9), lr=0.0, regularization_weight=1e-8,
        )
        out = t.train_epoch()
        assert "reg_output" not in out
        assert "reg_tightness" not in out
        assert not hasattr(m, "get_regularization_output")
        assert not hasattr(m, "get_regularization_tightness")

    def test_rho_is_still_reportable(self, tmp_path):
        """Dropping the constraint must not cost the diagnostic."""
        m = _make_model(s_value=0.2)
        _make_trainer(tmp_path, m, _make_loader(0.1, 0.9), regularization_weight=1e-8)
        rho = m.coverage_ratio()
        assert rho is not None and rho > 0.0


@requires_mosek
class TestBindingCertificate:
    """The exact binding-Corollary-1 certificate (solve_output_coverage_certificate)."""

    def test_tight_binds_ymax_operative_is_largest_set(self):
        """Coverage binds on the tight branch (y_tight = y_max, Corollary 1) while
        the operative certificate is the LARGEST invariant set (largest-s), so
        y_bar >= y_max and the written-back model reflects the operative (largest)
        certificate. output_std defaults to 1 here, so physical == normalized."""
        m = _make_model(s_value=0.5)
        res = m.solve_output_coverage_certificate(y_max=0.5, n_grid=15)
        assert res["success"]
        assert res["constraints_satisfied"]
        assert res["y_tight"] == pytest.approx(0.5, rel=5e-2)   # tight branch binds
        assert res["y_bar"] >= res["y_max"] - 1e-6              # operative covers
        assert res["s_min"] <= res["s"] <= res["s_max"] * (1 + 1e-6)
        # The written-back model reflects the selected (largest-set) certificate
        # (physical ȳ, output_std=1 so it equals s*sqrt(CPC^T)).
        ybar_model = float(m.s * torch.sqrt(m.C @ m.P @ m.C.T)[0, 0])
        assert ybar_model == pytest.approx(res["y_bar"], rel=1e-4)

    def test_uses_model_level_when_arg_omitted(self):
        m = _make_model(s_value=0.5)
        m.set_output_coverage_level(0.4)
        res = m.solve_output_coverage_certificate(n_grid=10)
        assert res["success"]
        assert res["y_max"] == pytest.approx(0.4)

    def test_coverage_unreachable_is_reported(self):
        """A level far beyond the model's reachable image -> honest failure,
        not a padded/vacuous 'success' (the informative-infeasibility signal).
        Pin a bounded s-band so the level stays beyond reach regardless of the
        (tunable) default s_max — at very large s_max MOSEK will marginally
        'solve' the coverage floor with an ill-conditioned huge P."""
        m = _make_model(s_value=0.5)
        res = m.solve_output_coverage_certificate(
            y_max=1e6, n_grid=10, s_min=0.1, s_max=20.0
        )
        assert not res["success"]
        assert res["reason"] == "coverage_unreachable"

    def test_selects_zero_violation_s_for_admissible_data(self):
        """With small (admissible) inputs, the sweep finds an s with zero input
        violations and marks the certificate violation-free."""
        m = _make_model(s_value=0.5)
        u = 0.02 * torch.ones(4, 8, 1, dtype=torch.float64)  # tiny -> admissible
        res = m.solve_output_coverage_certificate(y_max=0.5, inputs=u, n_grid=15)
        assert res["success"]
        assert res["violation_free"] is True
        assert res["n_input_violations"] == 0

    def test_physical_y_max_with_output_std(self):
        """y_max is physical: with output_std=2 the normalized floor is (y_max/2)^2.
        Coverage binds on the tight branch (y_tight == y_max) while the operative
        y_bar is the largest invariant set; both physical (output_std*s*sqrt(CPC^T))."""
        m = _make_model(s_value=0.5)
        m.set_output_coverage_level(1.0, output_std=2.0)
        res = m.solve_output_coverage_certificate(y_max=1.0, n_grid=15)
        assert res["success"]
        assert res["y_tight"] == pytest.approx(1.0, rel=5e-2)  # physical, tight binds
        assert res["y_bar"] >= res["y_max"] - 1e-6             # operative covers
        ybar_phys = float(2.0 * m.s * torch.sqrt(m.C @ m.P @ m.C.T)[0, 0])
        assert ybar_phys == pytest.approx(res["y_bar"], rel=1e-4)

    def test_does_not_mutate_model_on_failure(self):
        m = _make_model(s_value=0.5)
        before = (m.P.detach().clone(), m.s.detach().clone(), m.L.detach().clone())
        res = m.solve_output_coverage_certificate(
            y_max=1e6, n_grid=8, s_min=0.1, s_max=20.0  # bounded band -> unreachable
        )
        assert not res["success"]
        assert torch.equal(m.P, before[0])
        assert torch.equal(m.s, before[1])
        assert torch.equal(m.L, before[2])

    def test_reports_band_and_maxs_ceiling(self):
        """The summary carries the tight-branch band and the MaxS feasibility
        ceiling. y_tight (sweep min ȳ) ≤ y_bar; the operative ȳ covers y_max; and
        s_feas (MaxS) dominates any coverage-feasible s (same 5a/5b feasibility
        plus coverage), with a non-negative ‖H‖ diagnostic."""
        m = _make_model(s_value=0.5)
        res = m.solve_output_coverage_certificate(
            y_max=0.5, n_grid=15, s_min=0.1, s_max=20.0
        )
        assert res["success"]
        for key in ("y_feas", "s_feas", "norm_H_feas", "y_tight", "s_tight"):
            assert key in res and res[key] is not None
        tol = 1e-6
        assert res["y_tight"] <= res["y_bar"] + tol
        assert res["y_bar"] >= res["y_max"] - tol  # operative certificate covers
        assert res["s_feas"] >= res["s"] - tol      # MaxS s dominates operative s
        assert res["norm_H_feas"] >= 0.0
        sweep_ybars = [c["y_bar"] for c in res["sweep"] if c["y_bar"] is not None]
        assert res["y_tight"] == pytest.approx(min(sweep_ybars))

    def test_maxs_ceiling_independent_of_inputs(self):
        """The MaxS ceiling (y_feas/s_feas/norm_H_feas) is a pure certificate
        property (fixed θ, no coverage/input-violation constraint), so it does not
        depend on whether inputs are supplied."""
        m1 = _make_model(s_value=0.5)
        r1 = m1.solve_output_coverage_certificate(
            y_max=0.5, n_grid=15, s_min=0.1, s_max=20.0
        )
        m2 = _make_model(s_value=0.5)
        u = 0.02 * torch.ones(4, 8, 1, dtype=torch.float64)  # tiny -> admissible
        r2 = m2.solve_output_coverage_certificate(
            y_max=0.5, inputs=u, n_grid=15, s_min=0.1, s_max=20.0
        )
        assert r2["violation_free"] is True
        assert r1["s_feas"] == pytest.approx(r2["s_feas"], rel=1e-3)
        assert r1["y_feas"] == pytest.approx(r2["y_feas"], rel=1e-3)
        assert r1["norm_H_feas"] == pytest.approx(r2["norm_H_feas"], rel=1e-3, abs=1e-6)


@requires_mosek
class TestPostProcess:
    """post_process: the two cleanly separated certificate SDPs.

    Problem 1 (MaxS, ``max_s`` block) is the operative certificate written back
    into the model — the largest *regional* invariant set (reports volume, ȳ_c,
    ‖H‖, s and the tightness ratio ρ); Problem 2 (coverage sweep, ``coverage``
    block) reports the tightest coverage ȳ_f only.
    """

    def test_structure_and_applies_max_s(self):
        s_direct = _synth(_make_model(s_value=0.5)).max_s().s
        m = _make_model(s_value=0.5)
        res = m.post_process(y_max=0.5, n_grid=12)
        assert res["success"]
        assert res["constraints_satisfied"]
        # The operative (applied) certificate is the MaxS one; s == the direct MaxS.
        assert res["max_s"]["s"] == pytest.approx(s_direct, rel=1e-4)
        assert res["s_opt"] == pytest.approx(res["max_s"]["s"], rel=1e-12)
        assert float(m.s) == pytest.approx(res["max_s"]["s"], rel=1e-4)
        # Both problems are reported.
        assert res["max_s"]["volume"] > 0.0
        assert res["max_s"]["y_bar"] is not None       # ȳ_c
        assert res["max_s"]["norm_H"] >= 0.0
        assert res["coverage"]["reason"] == "ok"
        assert res["coverage"]["y_bar"] is not None    # ȳ_f
        assert res["max_s"]["rho"] is not None and res["max_s"]["rho"] > 0.0
        # ȳ_c = output_std * s * sqrt(CPCᵀ) matches the written-back model.
        ybar_model = float(m.output_std * m.s * torch.sqrt(m.C @ m.P @ m.C.T)[0, 0])
        assert ybar_model == pytest.approx(res["max_s"]["y_bar"], rel=1e-4)

    def test_coverage_tightest_matches_sweep(self):
        m = _make_model(s_value=0.5)
        res = m.post_process(y_max=0.5, n_grid=12, s_min=0.1, s_max=20.0)
        ybars = [c["y_bar"] for c in res["coverage"]["sweep"] if c["y_bar"] is not None]
        assert res["coverage"]["y_bar"] == pytest.approx(min(ybars))

    def test_coverage_skipped_when_y_max_unset(self):
        m = _make_model(s_value=0.5)  # y_max buffer left unset (nan)
        res = m.post_process(n_grid=8)
        assert res["success"]
        assert res["coverage"]["reason"] == "y_max_unset"
        assert res["coverage"]["y_bar"] is None
        assert res["max_s"]["coverage_ok"] is None

    def test_coverage_ok_when_image_covers_level(self):
        """The MaxS image easily covers a tiny demanded level -> coverage_ok."""
        m = _make_model(s_value=0.5)
        res = m.post_process(y_max=1e-3, n_grid=8)
        assert res["max_s"]["coverage_ok"] is True

    def test_coverage_unreachable_reported_but_maxvol_succeeds(self):
        """A level beyond reach on the band: MaxS (operative) still succeeds; the
        coverage sweep honestly reports it cannot certify y_max."""
        m = _make_model(s_value=0.5)
        res = m.post_process(y_max=1e6, n_grid=8, s_min=0.1, s_max=20.0)
        assert res["success"]
        assert res["coverage"]["reason"] == "coverage_unreachable"
        assert res["coverage"]["y_bar"] is None

    def test_coverage_sweep_is_pure(self):
        """coverage_sweep runs on a synthesizer snapshot and must not mutate the
        model (post_process applies MaxS, not the sweep result)."""
        m = _make_model(s_value=0.5)
        before = (m.P.detach().clone(), m.s.detach().clone(), m.L.detach().clone())
        out = _synth(m).coverage_sweep(0.5, n_grid=8, s_min=0.1, s_max=20.0)
        assert out is not None and out.y_f is not None
        assert out.y_f == pytest.approx(
            min(c.y_bar for c in out.sweep if c.y_bar is not None)
        )
        assert torch.equal(m.P, before[0])
        assert torch.equal(m.s, before[1])
        assert torch.equal(m.L, before[2])


@requires_mosek
class TestGlobalModelCoverage:
    """learn_L=False (L=0, globally stable): the coverage SDP must run without
    building the degenerate locality LMIs and must return the fixed L=0."""

    def test_coverage_sdp_runs_for_global_model(self):
        m = _make_global_model(s_value=1.0)
        sol = _synth(m).coverage_at_s(s=1.0, y_max=0.5)
        assert sol is not None
        assert np.allclose(sol.L, 0.0)  # fixed coupling returned unchanged
        assert sol.y_bar is not None and sol.y_bar > 0.0

    def test_post_process_with_coverage_for_global_model(self):
        m = _make_global_model(s_value=1.0)
        res = m.post_process(y_max=0.5, n_grid=8, s_min=0.1, s_max=5.0)
        assert res["success"]
        assert res["constraints_satisfied"]
        assert res["s_opt"] == pytest.approx(1.0, rel=1e-9)  # s stays fixed
        assert res["max_s"]["norm_H"] == pytest.approx(0.0, abs=1e-9)  # H = L P^-1 = 0
