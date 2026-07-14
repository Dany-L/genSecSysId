"""Tests for the output-coverage certificate (bind Corollary 1).

Covers:
- ``max_abs_output`` utility (the data-derived safe output level y_max).
- ``SimpleLure.get_regularization_output`` output-coverage penalty and its
  physical ``y_max`` / ``output_std`` buffers.
- ``SimpleLure.solve_output_coverage_certificate`` final binding SDP + s-sweep
  (needs MOSEK; skipped otherwise).
"""

import numpy as np
import pytest
import torch
import cvxpy as cp
from torch.utils.data import DataLoader, TensorDataset

from sysid.models.constrained_rnn import SimpleLure
from sysid.training import get_loss_function, get_optimizer
from sysid.training.trainer import Trainer
from sysid.utils import max_abs_output


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


class TestOutputCoverageRegularization:
    def test_zero_when_unset(self):
        """No coverage level -> no-op penalty (0), no gradient."""
        m = _make_model()
        loss = m.get_regularization_output()
        assert float(loss) == 0.0

    def test_penalty_positive_when_undercovered(self):
        """ybar = s*sqrt(CPC^T) with s,P small -> ybar << y_max -> penalty > 0."""
        m = _make_model(s_value=0.05)  # tiny s -> tiny output image
        m.set_output_coverage_level(1.0)  # demand a large certified output
        loss, margin = m.get_regularization_output(return_margin=True)
        assert float(loss) > 0.0
        assert float(margin) > 0.0  # deficit = y_max^2 - s^2 CPC^T > 0

    def test_zero_when_covered(self):
        """Large s -> output image reaches the (small) demanded level -> 0."""
        m = _make_model(s_value=5.0)  # s^2 = 25, C=[1,0], P=I -> s^2 CPC^T = 25
        m.set_output_coverage_level(1.0)  # demand ybar >= 1 (25 >= 1)
        loss = m.get_regularization_output()
        assert float(loss) == pytest.approx(0.0, abs=1e-9)

    def test_gradient_grows_the_output_image(self):
        """The penalty gradient must push s (and CPC^T) up to reduce the deficit."""
        m = _make_model(s_value=0.2)
        m.set_output_coverage_level(1.0)
        loss = m.get_regularization_output()
        loss.backward()
        # Increasing s reduces the deficit y_max^2 - s^2 CPC^T, so dL/ds < 0.
        assert m.s.grad is not None
        assert float(m.s.grad) < 0.0

    def test_buffer_survives_to_device_and_dtype(self):
        m = _make_model()
        m.set_output_coverage_level(0.75)
        assert float(m.y_max) == pytest.approx(0.75)  # stored physical
        assert m.y_max.dtype == m.P.dtype

    def test_physical_y_max_penalty_with_output_std(self):
        """y_max stays physical; the floor is (output_std*s)^2 CPC^T >= y_max^2."""
        m = _make_model(s_value=5.0)  # C=[1,0], P=I -> CPC^T = 1
        # output_std=2, s=5 -> physical image (2*5)^2 * 1 = 100.
        # y_max=2 -> 100 >= 4 -> covered.
        m.set_output_coverage_level(2.0, output_std=2.0)
        assert float(m.get_regularization_output()) == pytest.approx(0.0, abs=1e-9)
        # y_max=20 -> 100 < 400 -> undercovered.
        m.set_output_coverage_level(20.0, output_std=2.0)
        assert float(m.get_regularization_output()) > 0.0


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
    def test_y_max_set_from_loader(self, tmp_path):
        """Physical y_max = max|target| over the loader * output_std (=1 here)."""
        m = _make_model()
        _make_trainer(
            tmp_path, m, _make_loader(0.1, 0.9),
            regularization_weight=1e-8, output_regularization_weight=1.0,
        )
        assert float(m.y_max) == pytest.approx(0.9)

    def test_y_max_scaled_by_output_std(self, tmp_path):
        """The loader is normalized, so physical y_max = max|e| * output_std."""
        m = _make_model()
        _make_trainer(
            tmp_path, m, _make_loader(0.1, 0.9),
            regularization_weight=1e-8, output_regularization_weight=1.0,
            output_std=3.0,
        )
        assert float(m.y_max) == pytest.approx(0.9 * 3.0)
        assert float(m.output_std) == pytest.approx(3.0)

    def test_y_max_unset_when_weight_zero(self, tmp_path):
        """No output-coverage weight -> the level stays unset (penalty no-op)."""
        m = _make_model()
        _make_trainer(
            tmp_path, m, _make_loader(0.1, 0.9),
            regularization_weight=1e-8, output_regularization_weight=0.0,
        )
        assert bool(torch.isnan(m.y_max))

    def test_train_epoch_reports_positive_output_reg_when_undercovered(self, tmp_path):
        """With lr=0 the model can't move (stays feasible, no rollback/SDP), so
        this is MOSEK-free and just checks the term is wired and positive."""
        m = _make_model(s_value=0.2)  # tiny image vs demanded 0.9
        t = _make_trainer(
            tmp_path, m, _make_loader(0.1, 0.9), lr=0.0,
            regularization_weight=1e-8, output_regularization_weight=1.0,
        )
        out = t.train_epoch()
        assert "reg_output" in out
        assert out["reg_output"] > 0.0

    def test_train_epoch_output_reg_zero_when_weight_zero(self, tmp_path):
        m = _make_model(s_value=0.2)
        t = _make_trainer(
            tmp_path, m, _make_loader(0.1, 0.9), lr=0.0,
            regularization_weight=1e-8, output_regularization_weight=0.0,
        )
        out = t.train_epoch()
        assert out["reg_output"] == 0.0

    @requires_mosek
    def test_short_run_reduces_coverage_margin(self, tmp_path):
        """MSE (fit 0.9) and the coverage penalty both push the output image up,
        so a few epochs must shrink the coverage deficit."""
        m = _make_model(s_value=0.2)
        t = _make_trainer(
            tmp_path, m, _make_loader(0.1, 0.9), lr=1e-2,
            regularization_weight=1e-6, output_regularization_weight=1.0,
        )
        with torch.no_grad():
            _, margin_before = m.get_regularization_output(return_margin=True)
        for _ in range(10):
            t.train_epoch()
        with torch.no_grad():
            _, margin_after = m.get_regularization_output(return_margin=True)
        assert float(margin_after) < float(margin_before)


@requires_mosek
class TestBindingCertificate:
    """The exact binding-Corollary-1 certificate (solve_output_coverage_certificate)."""

    def test_binds_ybar_to_ymax(self):
        """When the level is reachable, the tightest certificate has ȳ = y_max
        (Corollary 1 binds), constraints hold, and s lies in [s_min, s_max].
        output_std defaults to 1 here, so physical == normalized."""
        m = _make_model(s_value=0.5)
        res = m.solve_output_coverage_certificate(y_max=0.5, n_grid=15)
        assert res["success"]
        assert res["constraints_satisfied"]
        assert res["y_bar"] == pytest.approx(0.5, rel=5e-2)
        assert res["s_min"] <= res["s"] <= res["s_max"] * (1 + 1e-6)
        # The written-back model reflects the selected certificate (physical ȳ,
        # output_std=1 so it equals s*sqrt(CPC^T)).
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
        not a padded/vacuous 'success' (the informative-infeasibility signal)."""
        m = _make_model(s_value=0.5)
        res = m.solve_output_coverage_certificate(y_max=1e6, n_grid=10)
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
        """y_max is physical: with output_std=2 the normalized floor is (y_max/2)^2
        and the reported ȳ is physical (== output_std * s * sqrt(CPC^T))."""
        m = _make_model(s_value=0.5)
        m.set_output_coverage_level(1.0, output_std=2.0)
        res = m.solve_output_coverage_certificate(y_max=1.0, n_grid=15)
        assert res["success"]
        assert res["y_bar"] == pytest.approx(1.0, rel=5e-2)  # physical, binds to y_max
        ybar_phys = float(2.0 * m.s * torch.sqrt(m.C @ m.P @ m.C.T)[0, 0])
        assert ybar_phys == pytest.approx(res["y_bar"], rel=1e-4)

    def test_does_not_mutate_model_on_failure(self):
        m = _make_model(s_value=0.5)
        before = (m.P.detach().clone(), m.s.detach().clone(), m.L.detach().clone())
        res = m.solve_output_coverage_certificate(y_max=1e6, n_grid=8)
        assert not res["success"]
        assert torch.equal(m.P, before[0])
        assert torch.equal(m.s, before[1])
        assert torch.equal(m.L, before[2])


def test_init_config_flag_defaults_on():
    """MinTrProb-at-init (output+input sweep) is the default; can be turned off."""
    from sysid.config import InitializationConfig
    assert InitializationConfig().init_s_from_conditions is True
    assert InitializationConfig(init_s_from_conditions=False).init_s_from_conditions is False


@requires_mosek
class TestInitSFromConditions:
    """initialize_s_from_conditions: pick s by the output+input sweep at init."""

    def test_reachable_level_sets_s_in_band(self):
        m = _make_model(s_value=0.05)  # arbitrary start; the sweep recomputes s
        u = 0.02 * np.ones((4, 8, 1))  # tiny -> admissible
        res = m.initialize_s_from_conditions(u, y_max=0.5, n_grid=12)
        assert res["success"]
        assert res["s_min"] <= res["s"] <= res["s_max"] * (1 + 1e-6)
        assert res["violation_free"] is True
        assert m.check_constraints()
        assert float(m.y_max) == pytest.approx(0.5)  # physical level stored

    def test_unreachable_level_falls_back_to_maxs(self):
        m = _make_model(s_value=0.05)
        u = 0.02 * np.ones((4, 8, 1))
        res = m.initialize_s_from_conditions(u, y_max=1e6, n_grid=8)
        # Coverage unreachable -> reported as not-success, but the model is left
        # with a feasible max-s certificate to start training from.
        assert not res["success"]
        assert res["reason"] == "coverage_unreachable"
        assert m.check_constraints()
        assert float(m.s) > 0.0
