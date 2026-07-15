"""Tests for the dead-zone activity regularization.

The penalty rewards the dead-zone to fire (w != 0), preventing the degenerate
linear collapse (w == 0 -> pure LTI rollout). It is NOT a certificate-level
anti-global term — H=0 can hold with an active nonlinearity since tanh/dzn are
globally sector-bounded — so these tests only exercise the penalty mechanics.

Covers:
- ``SimpleLure.get_regularization_activity``: the hinge penalty
  ``relu(w_star - <||w||>)`` on the rollout nonlinearity output, its no-op
  behavior, warmup skipping, and gradient direction (pushes activity up).
- Trainer wiring: the term is summed into the loss, reported as ``reg_activity``,
  and gated by ``activity_regularization_weight`` / ``activity_target``.

No SDP/MOSEK needed: the activity term reads the rollout, not the certificate,
and the trainer tests use lr=0 (model can't move, stays feasible).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sysid.models.constrained_rnn import SimpleLure
from sysid.training import get_loss_function, get_optimizer
from sysid.training.trainer import Trainer


def _make_model(nw: int = 4, s_value: float = 0.5) -> SimpleLure:
    """A small dead-zone Lure system (mirrors tests/test_output_coverage.py)."""
    m = SimpleLure(nd=1, ne=1, nx=2, nw=nw, activation="dzn", custom_params={"learn_L": True})
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


class TestActivityPenalty:
    def test_no_op_when_target_nonpositive(self):
        """w_star <= 0 disables the term (no gradient, exact 0)."""
        m = _make_model(nw=4)
        w = torch.ones(2, 5, 4, dtype=m.P.dtype)
        assert float(m.get_regularization_activity(w, 0.0)) == 0.0
        assert float(m.get_regularization_activity(w, -1.0)) == 0.0

    def test_positive_when_below_target(self):
        """<||w||> = sqrt(nw) for all-ones w; target above it -> positive hinge."""
        m = _make_model(nw=4)
        w = torch.ones(2, 5, 4, dtype=m.P.dtype)  # per-step ||w|| = sqrt(4) = 2
        loss, activity = m.get_regularization_activity(w, w_star=3.0, return_activity=True)
        assert float(activity) == 2.0
        assert float(loss) == 1.0  # relu(3 - 2) = 1

    def test_zero_when_activity_meets_target(self):
        """Activity at/above target -> hinge saturates to 0."""
        m = _make_model(nw=4)
        w = torch.ones(2, 5, 4, dtype=m.P.dtype)  # activity = 2.0
        assert float(m.get_regularization_activity(w, w_star=1.0)) == 0.0
        assert float(m.get_regularization_activity(w, w_star=2.0)) == 0.0

    def test_warmup_steps_are_skipped(self):
        """Leading warmup steps are excluded from the activity mean."""
        m = _make_model(nw=4)
        w = torch.zeros(2, 5, 4, dtype=m.P.dtype)
        w[:, :2, :] = 10.0  # only the (skipped) warmup steps are active
        _, activity = m.get_regularization_activity(
            w, w_star=1.0, warmup_steps=2, return_activity=True
        )
        assert float(activity) == 0.0  # post-warmup steps are all zero
        # Without warmup skipping, the large leading steps dominate.
        _, activity_all = m.get_regularization_activity(w, w_star=1.0, return_activity=True)
        assert float(activity_all) > 0.0

    def test_gradient_pushes_activity_up(self):
        """The penalty gradient is anti-parallel to w, so -grad grows ||w||."""
        m = _make_model(nw=4)
        w = torch.full((2, 5, 4), 0.1, dtype=m.P.dtype, requires_grad=True)  # leaf
        loss = m.get_regularization_activity(w, w_star=5.0)  # far below target
        loss.backward()
        assert w.grad is not None
        # descent step (-grad) increases w in the direction it already points.
        assert float((w.grad * w).sum()) < 0.0

    def test_monotone_decreasing_in_activity(self):
        """Larger ||w|| -> smaller penalty (until it saturates at 0)."""
        m = _make_model(nw=4)
        w_small = 0.5 * torch.ones(2, 5, 4, dtype=m.P.dtype)
        w_big = 2.0 * torch.ones(2, 5, 4, dtype=m.P.dtype)
        assert float(m.get_regularization_activity(w_big, 5.0)) < float(
            m.get_regularization_activity(w_small, 5.0)
        )


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


class TestActivityTrainerWiring:
    def test_reg_activity_reported_and_positive_when_dead(self, tmp_path):
        """Tiny inputs keep the dead-zone in its dead band (w~0), so with a
        positive target the activity penalty is wired in and > 0."""
        m = _make_model(nw=4)
        t = _make_trainer(
            tmp_path, m, _make_loader(0.05, 0.9), lr=0.0,
            regularization_weight=1e-8,
            activity_regularization_weight=1.0, activity_target=1.0,
        )
        out = t.train_epoch()
        assert "reg_activity" in out
        assert out["reg_activity"] > 0.0

    def test_reg_activity_zero_when_weight_zero(self, tmp_path):
        """Zero weight -> the term is a no-op and reg_activity stays 0."""
        m = _make_model(nw=4)
        t = _make_trainer(
            tmp_path, m, _make_loader(0.05, 0.9), lr=0.0,
            regularization_weight=1e-8,
            activity_regularization_weight=0.0, activity_target=1.0,
        )
        out = t.train_epoch()
        assert out["reg_activity"] == 0.0

    def test_reg_activity_zero_when_target_zero(self, tmp_path):
        """Weight on but target 0 -> hinge disabled, reg_activity stays 0."""
        m = _make_model(nw=4)
        t = _make_trainer(
            tmp_path, m, _make_loader(0.05, 0.9), lr=0.0,
            regularization_weight=1e-8,
            activity_regularization_weight=1.0, activity_target=0.0,
        )
        out = t.train_epoch()
        assert out["reg_activity"] == 0.0
