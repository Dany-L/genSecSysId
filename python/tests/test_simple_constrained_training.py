"""The training loop after the ownership rollback.

The scheme is deliberately small, and these tests exist to keep it that way:

    loss   = prediction loss + lambda_F * log-barrier( LMIs )   [+ input term]
    step   = optimizer.step()
    check  = model.check_constraints()
    repair = re-solve the certificate with theta and alpha FIXED
    else   = roll the whole step back

Everything the certificate-ownership branch added on top — freezing (P, L, Lambda, s)
out of autograd, per-epoch TightCert re-synthesis, the beta anneal, the accept
guard, and the hard coverage floor ``(sigma*s)^2 C P C^T >= y_max^2 I`` — is gone.
The coverage floor went because the model class cannot satisfy it on this
benchmark (the fitted reference model measures rho = 0.42; see the wiki notes
running-example/reference-model and certificate-synthesis/ellipsoidal-conservatism),
so imposing it was asking training for something no theta in the class provides.

What is tested here is the *contract*, not the numbers: the objective has no
coverage term, every certificate parameter is back under gradient, and the
repair-or-rollback step is what keeps the LMIs true.
"""

import numpy as np
import pytest
import torch
import cvxpy as cp
from torch.utils.data import DataLoader, TensorDataset

from sysid.config import TrainingConfig
from sysid.models.constrained_rnn import SimpleLure
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


requires_mosek = pytest.mark.skipif(
    not _mosek_available(), reason="MOSEK solver not available/licensed"
)

CERT_PARAMS = ("P", "L", "la", "s", "tau")


def _make_model(s_value: float = 0.05) -> SimpleLure:
    """Strictly feasible at P = I, so the barrier is finite from the first step
    (an infeasible start gives -logdet of an indefinite matrix, i.e. nan, and
    every assertion below would degenerate)."""
    m = SimpleLure(nd=1, ne=1, nx=2, nw=2, activation="dzn", custom_params={"learn_L": True})
    with torch.no_grad():
        m.A.data = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=m.A.dtype)
        m.B.data = torch.tensor([[0.1], [0.1]], dtype=m.B.dtype)
        m.B2.data = 0.05 * torch.ones_like(m.B2)
        m.C.data = torch.tensor([[1.0, 0.0]], dtype=m.C.dtype)
        m.C2.data = 0.1 * torch.ones_like(m.C2)
        m.D21.data = 0.1 * torch.ones_like(m.D21)
        m.tau.data = torch.tensor(float(np.log(0.9 / 0.1)))
        m.s.data = torch.tensor(float(s_value))
        m.P.data = torch.eye(2, dtype=m.P.dtype)
    return m


def _loader(N: int = 6, B: int = 4) -> DataLoader:
    d = 0.1 * torch.ones(B, N, 1)
    e = 0.5 * torch.ones(B, N, 1)
    return DataLoader(TensorDataset(d, e), batch_size=2)


def _trainer(tmp_path, model, lr=0.0, **kwargs) -> Trainer:
    return Trainer(
        model=model,
        train_loader=_loader(),
        val_loader=_loader(),
        loss_fn=get_loss_function("mse"),
        optimizer=get_optimizer(model.parameters(), learning_rate=lr),
        device="cpu",
        output_dir=str(tmp_path / "o"),
        model_dir=str(tmp_path / "m"),
        log_dir=str(tmp_path / "l"),
        mlflow_tracking=False,
        **kwargs,
    )


class TestOwnershipMachineryIsGone:
    """The removed API must stay removed — a stale call would silently no-op or,
    worse, resurrect the drift the rollback was meant to undo."""

    @pytest.mark.parametrize("name", ["freeze_certificate", "resynthesize_certificate",
                                      "get_regularization_output",
                                      "get_regularization_tightness"])
    def test_model_methods_are_removed(self, name):
        assert not hasattr(_make_model(), name)

    def test_tight_cert_is_removed_from_the_synthesizer(self):
        from sysid.optimization import LureCertificateSynthesizer
        synth = LureCertificateSynthesizer.from_model(_make_model())
        assert not hasattr(synth, "tight_cert")

    @pytest.mark.parametrize("field", ["freeze_certificate", "freeze_alpha",
                                       "repair_enforce_coverage", "resynthesize_certificate",
                                       "resynthesis_every", "resynthesis_beta",
                                       "output_regularization_weight",
                                       "tightness_regularization_weight"])
    def test_training_config_fields_are_removed(self, field):
        assert not hasattr(TrainingConfig(), field)

    def test_stale_configs_still_load(self, tmp_path):
        """Old YAMLs carry the removed keys; they must warn and load, not raise —
        otherwise every archived config becomes unusable."""
        from sysid.config import Config
        import yaml
        cfg = tmp_path / "old.yaml"
        cfg.write_text(yaml.safe_dump({
            "data": {"train_path": str(tmp_path)},
            "model": {"model_type": "crnn", "nw": 2, "nx": 2, "activation": "dzn"},
            "optimizer": {"optimizer_type": "adam"},
            "training": {"max_epochs": 1, "freeze_certificate": True,
                         "resynthesize_certificate": True, "resynthesis_beta": 10.0,
                         "output_regularization_weight": 1.0, "warmup_steps": 7},
            "mlflow": {"experiment_name": "t"},
        }))
        loaded = Config.from_yaml(str(cfg))
        assert loaded.training.warmup_steps == 7      # the real keys still arrive


class TestObjective:
    def test_certificate_parameters_all_carry_gradient(self, tmp_path):
        """The rollback's premise: (P, L, Lambda, s, tau) are learned again, guided
        by the barrier and the local LMI, and s is *expected* to shrink."""
        m = _make_model()
        _trainer(tmp_path, m)
        for name in CERT_PARAMS:
            assert getattr(m, name).requires_grad, name

    def test_barrier_gradient_reaches_every_certificate_parameter(self):
        """Nothing is silently constant. The branch skipped barrier terms that did
        not require grad — an optimization that only made sense while the
        certificate was frozen, and that would now hide a dead parameter."""
        m = _make_model()
        m.get_regularization_loss().backward()
        for name in CERT_PARAMS:
            g = getattr(m, name).grad
            assert g is not None and torch.any(g != 0), name

    def test_epoch_metrics_carry_no_coverage_terms(self, tmp_path):
        m = _make_model()
        out = _trainer(tmp_path, m, regularization_weight=1e-8).train_epoch()
        assert {"loss", "pred_loss", "reg_feasibility", "reg_input"} <= set(out)
        assert "reg_output" not in out and "reg_tightness" not in out

    def test_loss_is_prediction_plus_barrier_only(self, tmp_path):
        """Reproduce the reported epoch loss from its two declared parts. If a
        third term were still wired in, this would not close."""
        m = _make_model()
        t = _trainer(tmp_path, m, regularization_weight=1e-3)
        out = t.train_epoch()   # lr = 0, so the parameters never move
        expected = (out["pred_loss"]
                    + 1e-3 * out["reg_feasibility"]
                    + t.input_regularization_weight * out["reg_input"])
        assert np.isfinite(out["loss"])
        assert out["loss"] == pytest.approx(expected, rel=1e-9)


@requires_mosek
class TestRepairOrRollback:
    def test_a_broken_step_is_repaired_and_theta_is_kept(self, tmp_path):
        """The point of repairing rather than rolling back: a step that improves
        the fit but breaks the LMIs is salvaged by moving only the certificate."""
        m = _make_model()
        t = _trainer(tmp_path, m, regularization_weight=1e-8)
        theta_before = m.A.detach().clone()
        with torch.no_grad():   # simulate a step that broke the certificate
            m.P.data = torch.diag(torch.tensor([1.0, -1.0], dtype=m.P.dtype))
        assert not m.check_constraints()

        assert t._repair_certificate() is True
        assert m.check_constraints()
        assert torch.equal(m.A.detach(), theta_before)

    def test_rollback_when_no_certificate_exists(self, tmp_path):
        """Unstable A admits no certificate at any s, so the repair must fail and
        report it — that False is what triggers the parameter rollback."""
        m = _make_model()
        t = _trainer(tmp_path, m, regularization_weight=1e-8)
        with torch.no_grad():
            m.A.data = torch.tensor([[1.5, 0.0], [0.0, 1.5]], dtype=m.A.dtype)
        assert t._repair_certificate() is False

    def test_training_ends_feasible(self, tmp_path):
        """The invariant the whole scheme buys: after any number of epochs the
        model still satisfies its own LMIs."""
        m = _make_model()
        t = _trainer(tmp_path, m, lr=1e-2, regularization_weight=1e-6)
        for _ in range(3):
            t.train_epoch()
        assert m.check_constraints()
