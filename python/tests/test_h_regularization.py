"""Tests for the anti-global-certificate (H) regularization.

The penalty pushes the coupling ``H = L P⁻¹`` away from zero
(``relu(h_star - ‖H‖_F)``), keeping the certificate LOCAL — ``H = 0`` is the
global sector condition, i.e. a globally stable, typically near-linear model.
Unlike the activity term it acts directly on the certificate params (L, P).

Covers:
- ``SimpleLure.get_regularization_H``: the hinge penalty, its no-op behavior
  (target <= 0, or learn_L=False), the ``‖H‖_F`` value, monotonicity, and the
  gradient direction (grows ‖H‖).
- Trainer wiring: the term is summed into the loss, reported as ``reg_H`` and
  gated by ``h_regularization_weight`` / ``h_target``.
- ``TrainingConfig`` defaults (off by default).

No SDP/MOSEK needed: the term reads the certificate params, and the trainer
tests use lr=0 (model can't move, stays feasible).
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sysid.config import TrainingConfig
from sysid.models.constrained_rnn import SimpleLure
from sysid.training import get_loss_function, get_optimizer
from sysid.training.trainer import Trainer


def _make_model(nw: int = 4, s_value: float = 0.5, learn_L: bool = True) -> SimpleLure:
    """A small dead-zone Lure system (mirrors tests/test_activity_regularization.py)."""
    m = SimpleLure(nd=1, ne=1, nx=2, nw=nw, activation="dzn", custom_params={"learn_L": learn_L})
    with torch.no_grad():
        m.A.data = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=m.A.dtype)
        m.B.data = torch.tensor([[0.1], [0.1]], dtype=m.B.dtype)
        m.B2.data = torch.zeros_like(m.B2)
        m.C.data = torch.tensor([[1.0, 0.0]], dtype=m.C.dtype)
        m.C2.data = 0.1 * torch.ones_like(m.C2)
        m.D21.data = 0.1 * torch.ones_like(m.D21)
        m.tau.data = torch.tensor(float(np.log(0.9 / 0.1)))  # alpha = 0.9
        m.s.data = torch.tensor(float(s_value))
        m.P.data = torch.eye(2, dtype=m.P.dtype)
    return m


class TestHPenalty:
    def test_no_op_when_target_nonpositive(self):
        """h_star <= 0 disables the term (exact 0, no gradient)."""
        m = _make_model()
        m.L.data = torch.ones_like(m.L)
        assert float(m.get_regularization_H(0.0)) == 0.0
        assert float(m.get_regularization_H(-1.0)) == 0.0

    def test_no_op_when_not_learn_L(self):
        """Without a learnable L, H is identically 0 and the term is a no-op."""
        m = _make_model(learn_L=False)
        assert float(m.get_regularization_H(5.0)) == 0.0

    def test_positive_when_below_target(self):
        """P = I -> ‖H‖_F = ‖L‖_F; a target above it gives a positive hinge."""
        m = _make_model()
        m.L.data = torch.zeros_like(m.L)
        m.L.data[0, 0] = 2.0  # ‖L‖_F = 2
        loss, norm_H = m.get_regularization_H(3.0, return_norm=True)
        assert float(norm_H) == pytest.approx(2.0)
        assert float(loss) == pytest.approx(1.0)  # relu(3 - 2) = 1

    def test_zero_when_norm_meets_target(self):
        """‖H‖_F at/above target -> hinge saturates to 0."""
        m = _make_model()
        m.L.data = torch.zeros_like(m.L)
        m.L.data[0, 0] = 2.0  # ‖L‖_F = 2
        assert float(m.get_regularization_H(1.0)) == 0.0
        assert float(m.get_regularization_H(2.0)) == pytest.approx(0.0, abs=1e-5)

    def test_gradient_grows_norm_H(self):
        """The penalty gradient is anti-parallel to L, so -grad grows ‖H‖."""
        m = _make_model()
        m.L.data = 0.1 * torch.ones_like(m.L)  # small, well below target
        loss = m.get_regularization_H(5.0)
        loss.backward()
        assert m.L.grad is not None
        # descent step (-grad) moves L in the direction it already points -> ‖H‖ up.
        assert float((m.L.grad * m.L).sum()) < 0.0

    def test_monotone_decreasing_in_norm(self):
        """Larger ‖H‖ -> smaller penalty (until it saturates at 0)."""
        m = _make_model()
        m.L.data = 0.5 * torch.ones_like(m.L)
        small = float(m.get_regularization_H(5.0))
        m.L.data = 2.0 * torch.ones_like(m.L)
        big = float(m.get_regularization_H(5.0))
        assert big < small

    def test_norm_H_matches_frobenius(self):
        """The reported norm equals ‖L P⁻¹‖_F (P = 2I here to exercise the inv)."""
        m = _make_model()
        m.P.data = 2.0 * torch.eye(2, dtype=m.P.dtype)
        m.L.data = torch.arange(8, dtype=m.L.dtype).reshape(4, 2)
        _, norm_H = m.get_regularization_H(1.0, return_norm=True)
        H = m.L.detach().numpy() @ np.linalg.inv(m.P.detach().numpy())
        assert float(norm_H) == pytest.approx(float(np.linalg.norm(H)))


def _make_loader(u_amp: float, y_level: float, N: int = 5, B: int = 4) -> DataLoader:
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


class TestHTrainerWiring:
    def test_reg_H_reported_and_positive_below_target(self, tmp_path):
        """H ~ 0 (L init zero) with a positive target -> the term is wired and > 0."""
        m = _make_model()
        t = _make_trainer(
            tmp_path, m, _make_loader(0.05, 0.9), lr=0.0,
            regularization_weight=1e-8,
            h_regularization_weight=1.0, h_target=1.0,
        )
        out = t.train_epoch()
        assert "reg_H" in out
        assert out["reg_H"] > 0.0

    def test_reg_H_zero_when_weight_zero(self, tmp_path):
        """Zero weight -> the term is a no-op and reg_H stays 0."""
        m = _make_model()
        t = _make_trainer(
            tmp_path, m, _make_loader(0.05, 0.9), lr=0.0,
            regularization_weight=1e-8,
            h_regularization_weight=0.0, h_target=1.0,
        )
        out = t.train_epoch()
        assert out["reg_H"] == 0.0

    def test_reg_H_zero_when_target_zero(self, tmp_path):
        """Weight on but target 0 -> hinge disabled, reg_H stays 0."""
        m = _make_model()
        t = _make_trainer(
            tmp_path, m, _make_loader(0.05, 0.9), lr=0.0,
            regularization_weight=1e-8,
            h_regularization_weight=1.0, h_target=0.0,
        )
        out = t.train_epoch()
        assert out["reg_H"] == 0.0


def test_h_config_defaults_off():
    """The H regularization is off by default."""
    cfg = TrainingConfig()
    assert cfg.h_regularization_weight == 0.0
    assert cfg.h_target == 0.0

