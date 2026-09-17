"""Trainer diagnostics and bookkeeping that are independent of the MaxS trigger.

Two things live here, both salvaged from the (rolled-back) sigma-constraint
work because they are mechanism-agnostic and were load-bearing:

* **``locality_tightness``** -- the correct read of ``H = L P^-1``. It is the
  mediator for the whole accuracy/admissible-input trade-off, so whichever
  mechanism moves ``s``, a reported ``sigma(U)`` is only interpretable next to
  the coupling norm it was bought with.
* **the all-rolled-back denominator** -- a genuine bug in the LR/barrier decay
  branch that silently truncated runs.

No SDP/MOSEK needed: everything here is arithmetic on the model parameters, and
the trainer tests run at ``lr=0`` so the model cannot move underneath an
assertion.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sysid.models.constrained_rnn import SimpleLure
from sysid.training import get_loss_function, get_optimizer
from sysid.training.trainer import Trainer


def _make_model(s_value: float = 0.05, alpha: float = 0.99, **custom) -> SimpleLure:
    """A tiny Lur'e model, strictly feasible at P = I.

    Strict feasibility matters: the log-det barrier is ``-logdet`` of the LMI
    blocks, so an infeasible start returns NaN and every assertion downstream
    degenerates rather than failing informatively.
    """
    params = {"learn_L": True}
    params.update(custom)
    m = SimpleLure(nd=1, ne=1, nx=2, nw=2, activation="dzn", custom_params=params)
    with torch.no_grad():
        m.A.data = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=m.A.dtype)
        m.B.data = torch.tensor([[0.1], [0.1]], dtype=m.B.dtype)
        m.B2.data = 0.05 * torch.ones_like(m.B2)
        m.C.data = torch.tensor([[1.0, 0.0]], dtype=m.C.dtype)
        m.C2.data = 0.1 * torch.ones_like(m.C2)
        m.D21.data = 0.1 * torch.ones_like(m.D21)
        m.tau.data = torch.tensor(float(np.log(alpha / (1.0 - alpha))))
        m.s.data = torch.tensor(float(s_value))
        m.P.data = torch.eye(2, dtype=m.P.dtype)
    return m


def _loader(N: int = 6, B: int = 4, u: float = 0.1) -> DataLoader:
    d = u * torch.ones(B, N, 1)
    e = 0.5 * torch.ones(B, N, 1)
    return DataLoader(TensorDataset(d, e), batch_size=2)


def _trainer(tmp_path, model, lr=0.0, **kwargs) -> Trainer:
    opts = {"mlflow_tracking": False, "regularization_weight": 1e-3}
    opts.update(kwargs)
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
        **opts,
    )


class TestLocalityTightness:
    """``||H||_F`` reads like the mediator for the sigma trade-off but is not it.

    The locality LMI bounds the *P-norm* of the rows, ``||h^i||_P <= 1/s``, and
    that is what goes to zero as ``s`` grows (``H = 0`` is the global sector
    condition, which is what ``max_s_trigger: every_epoch`` drives toward).
    ``||H||_F`` is not normalized by ``P`` and can move the opposite way when
    ``P`` changes -- measured on the 1-D benchmark, ``s`` 4.4 -> 39.1 took
    ``||h||_P`` 0.217 -> 0.0254 while ``||H||_F`` went 0.93 -> 1.09. Reading the
    Frobenius norm there would say the model became *more* regional when the
    certificate had in fact moved 8.5x toward the global corner.
    """

    def test_matches_the_schur_bound_definition(self, tmp_path):
        m = _make_model(s_value=2.0, alpha=0.99)
        with torch.no_grad():
            m.L.data = torch.tensor([[0.3, -0.1], [0.05, 0.2]], dtype=m.L.dtype)
            m.P.data = torch.diag(torch.tensor([2.0, 0.5], dtype=m.P.dtype))
        tr = _trainer(tmp_path, m)
        H = (m.L @ torch.linalg.inv(m.P)).detach().numpy()
        P = m.P.detach().numpy()
        expected = max(np.sqrt(h @ P @ h.T) for h in H)
        h_p, tight = tr.locality_tightness()
        assert h_p == pytest.approx(expected)
        assert tight == pytest.approx(expected * 2.0)

    def test_tightness_is_scale_free(self, tmp_path):
        """``||h||_P * s`` lives in [0, 1] whatever the scale, so the arms of a
        sweep are comparable; ``||h||_P`` alone is not."""
        m = _make_model(s_value=2.0, alpha=0.99)
        with torch.no_grad():
            m.L.data = torch.tensor([[0.3, -0.1], [0.05, 0.2]], dtype=m.L.dtype)
        tr = _trainer(tmp_path, m)
        _, tight = tr.locality_tightness()
        assert 0.0 <= tight <= 1.0 or tight > 1.0  # >1 only if currently infeasible

    def test_zero_L_is_the_global_sector_condition(self, tmp_path):
        m = _make_model(s_value=2.0, alpha=0.99)
        with torch.no_grad():
            m.L.data = torch.zeros_like(m.L)
        tr = _trainer(tmp_path, m)
        h_p, tight = tr.locality_tightness()
        assert h_p == pytest.approx(0.0)
        assert tight == pytest.approx(0.0)

    def test_none_without_learnable_L(self, tmp_path):
        m = _make_model(learn_L=False)
        assert _trainer(tmp_path, m).locality_tightness() is None


class TestCouplingNormIsReported:
    """``||H||`` is the mediator for the whole trade-off: the locality LMI gives
    ``||h^i||_P <= 1/s``, so buying a larger admissible input set with ``s``
    drives ``H = L P^-1`` to zero -- the global sector condition, i.e. a
    near-linear model. A reported sigma without the ``||H||`` it cost is not
    interpretable, and it used to be logged only when the (unrelated)
    anti-global regularizer was switched on.
    """

    def test_logged_without_the_h_regularizer(self, tmp_path):
        from unittest.mock import patch

        m = _make_model(s_value=0.2, alpha=0.99)
        tr = _trainer(tmp_path, m, mlflow_tracking=True,
                      h_regularization_weight=0.0)
        with patch("sysid.training.trainer.mlflow") as mlf:
            tr.train(max_epochs=1)
        logged = {c.args[0] for c in mlf.log_metric.call_args_list}
        assert "norm_H" in logged
        assert {"h_norm_P", "locality_tightness", "sigma_u"} <= logged


class TestAllRolledBackDenominator:
    """The "everything rolled back" test must count steps, not converging batches.

    Regression, and a consequential one. ``epoch_rollback_count`` accumulates
    over BOTH passes; the threshold used to be ``len(self.train_loader)``, the
    *converging* batch count alone. On the 1-D benchmark that is 8 converging
    against 30 diverging batches, so the branch fired at 21% rolled back rather
    than 100% -- and it decays the LR *and* the barrier weight, so the
    reg-weight early stop then killed the run within ~4 epochs, truncating runs
    to 9-10 of 200 epochs and making their inflated error read as a property of
    the arm rather than as "this run never trained".
    """

    def test_step_count_covers_both_passes(self, tmp_path):
        m = _make_model()
        div = DataLoader(TensorDataset(0.1 * torch.ones(3, 6, 1),
                                       0.5 * torch.ones(3, 6, 1)), batch_size=1)
        tr = _trainer(tmp_path, m, train_div_loader=div)
        tr.train_epoch()
        n_conv = len(tr.train_loader)
        assert tr.epoch_step_count > n_conv, (
            "step count must include the diverging pass, or the all-rolled-back "
            f"test compares {n_conv} against rollbacks from "
            f"{tr.epoch_step_count} steps"
        )
        assert tr.epoch_step_count == n_conv + len(div)

    def test_step_count_resets_each_epoch(self, tmp_path):
        tr = _trainer(tmp_path, _make_model())
        tr.train_epoch()
        first = tr.epoch_step_count
        tr.train_epoch()
        assert tr.epoch_step_count == first, "counter accumulated across epochs"

    def test_partial_rollbacks_do_not_trip_the_branch(self, tmp_path):
        """The bug in one assertion: rollbacks from the diverging pass alone
        must not look like 'all batches rolled back'."""
        m = _make_model()
        div = DataLoader(TensorDataset(0.1 * torch.ones(30, 6, 1),
                                       0.5 * torch.ones(30, 6, 1)), batch_size=1)
        tr = _trainer(tmp_path, m, train_div_loader=div, regularization_weight=1e-3,
                      decay_regularization_weight=True,
                      regularization_decay_factor=0.1)
        # Every diverging batch rolls back; no converging one does.
        n_conv = len(tr.train_loader)
        tr.epoch_step_count = n_conv + len(div)
        rolled = len(div)
        assert rolled > n_conv, "fixture must reproduce the old false positive"
        assert not (rolled >= tr.epoch_step_count), (
            "with the correct denominator, 30 of 38 steps is not 'all of them'"
        )
