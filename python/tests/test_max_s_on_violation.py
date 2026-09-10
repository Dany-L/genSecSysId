"""Tests for the after-epoch MaxS repair (``Trainer._maybe_maximize_s``).

``s`` is learnable and the log-barrier only ever pushes it DOWN — nothing pushes
back. Once it drops below the input floor ``sqrt(u_max)`` the training data
breaches the input condition ``||u_k||^2 <= s^2 - alpha^2 V(x_k)``, every
optimizer step lands outside the feasible set, and the per-batch repair SDP fires
on every batch and mostly fails into a rollback.

The fix (restored from cf9cb54) is to scan the epoch for the peak margin and,
only when it is breached, re-solve MaxS once to grow the certified set back over
the data. That is now the ``max_s_trigger="on_violation"`` arm of the trigger
policy; the scan itself lives in ``Trainer.input_margin`` so it can run in every
arm. See test_max_s_trigger.py for the other two arms and the policy wiring.
"""

import logging

import cvxpy as cp
import numpy as np
import pytest
import torch

from sysid.config import Config, TrainingConfig
from sysid.models.constrained_rnn import SimpleLure
from sysid.training import Trainer


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


def _make_trainer(tmp_path, max_s_trigger, peak=6.0, n=3, T=25):
    torch.manual_seed(0)
    model = SimpleLure(nd=1, ne=1, nx=2, nw=4, activation="dzn", ts=0.05,
                       custom_params={"learn_L": True})
    with torch.no_grad():
        # Comfortably stable, so MaxS is feasible and the test exercises the
        # repair rather than an infeasible corner.
        model.A.data = torch.tensor([[0.8, 0.05], [0.0, 0.8]], dtype=model.A.dtype)
        model.B.data = torch.tensor([[0.0], [0.01]], dtype=model.B.dtype)
        model.C.data = torch.tensor([[1.0, 0.0]], dtype=model.C.dtype)
        model.B2.data = 0.01 * torch.ones_like(model.B2)
        model.C2.data = 0.01 * torch.ones_like(model.C2)
        model.D21.data = torch.zeros_like(model.D21)
        model.D.data = torch.zeros_like(model.D)
        model.D12.data = torch.zeros_like(model.D12)
        model.P.data = torch.eye(2, dtype=model.P.dtype)
        model.L.data = torch.zeros_like(model.L)
        model.la.data = torch.ones_like(model.la)
        model.tau.data = torch.tensor(float(np.log(0.9 / 0.1)))
        # s far below the input peak -> the input condition is breached.
        model.s.data = torch.tensor(0.5)

    d = torch.zeros(n, T, 1)
    d[:, ::5, 0] = peak  # spikes well above s
    e = torch.zeros(n, T, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(d, e), batch_size=2
    )
    trainer = Trainer(
        model=model, train_loader=loader, val_loader=loader,
        loss_fn=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device="cpu", output_dir=str(tmp_path), model_dir=str(tmp_path),
        log_dir=str(tmp_path), mlflow_tracking=False, warmup_steps=0,
        max_s_trigger=max_s_trigger,
        # Non-zero on purpose: with no barrier in the loss the trainer's repair and
        # rollback are both off, so there is no regional certificate to enlarge and
        # _maybe_maximize_s short-circuits in every arm (see
        # test_max_s_trigger.py::test_no_solve_without_the_barrier).
        regularization_weight=1e-3,
    )
    return trainer, model


class TestWiring:
    def test_defaults_off(self, tmp_path):
        """The default must stay "never", i.e. exactly the old
        ``solve_max_s_on_violation: False`` behaviour: a config that omits the key
        must not silently gain an SDP solve that moves ``s``."""
        assert TrainingConfig().max_s_trigger == "never"
        trainer, _ = _make_trainer(tmp_path, max_s_trigger="never")
        assert trainer.max_s_trigger == "never"

    def test_flag_is_stored(self, tmp_path):
        trainer, _ = _make_trainer(tmp_path, max_s_trigger="on_violation")
        assert trainer.max_s_trigger == "on_violation"

    def test_config_field_survives_yaml_roundtrip(self, tmp_path):
        """It was silently dropped before — a stale YAML key is a no-op."""
        p = tmp_path / "c.yaml"
        p.write_text(
            "data:\n  train_path: /nonexistent\n"
            "model:\n  model_type: crnn\n"
            "training:\n  max_s_trigger: on_violation\n"
        )
        assert Config.from_yaml(str(p)).training.max_s_trigger == "on_violation"

    def test_deprecated_bool_still_reaches_the_trigger(self, tmp_path):
        """Archived YAMLs spell it the old way; unknown keys are dropped silently,
        so the alias has to keep working or those configs become no-ops."""
        p = tmp_path / "old.yaml"
        p.write_text(
            "data:\n  train_path: /nonexistent\n"
            "model:\n  model_type: crnn\n"
            "training:\n  solve_max_s_on_violation: true\n"
        )
        assert Config.from_yaml(str(p)).training.max_s_trigger == "on_violation"


@requires_mosek
class TestRepair:
    def test_grows_s_when_input_condition_breached(self, tmp_path):
        trainer, model = _make_trainer(tmp_path, max_s_trigger="on_violation")
        s_before = float(model.s)

        c_max = trainer.input_margin()
        assert c_max > 0, "fixture must start with the input condition breached"
        new_s = trainer._maybe_maximize_s(epoch=0, c_max=c_max)

        assert new_s is not None, "breached input condition must trigger a solve"
        assert float(model.s) > s_before
        assert model.check_constraints()

    def test_noop_when_condition_already_holds(self, tmp_path):
        """No breach -> no SDP, and s is left exactly alone."""
        trainer, model = _make_trainer(tmp_path, max_s_trigger="on_violation", peak=0.0)
        with torch.no_grad():
            model.s.data = torch.tensor(50.0)
        s_before = float(model.s)

        c_max = trainer.input_margin()
        assert c_max <= 0, "fixture must start with the input condition satisfied"
        assert trainer._maybe_maximize_s(epoch=0, c_max=c_max) is None
        assert float(model.s) == s_before

    def test_returns_model_to_train_mode(self, tmp_path):
        trainer, model = _make_trainer(tmp_path, max_s_trigger="on_violation")
        model.train()
        trainer._maybe_maximize_s(epoch=0, c_max=trainer.input_margin())
        assert model.training

    def test_survives_a_failing_sdp(self, tmp_path, monkeypatch, caplog):
        """A failed solve must warn and leave s alone, not crash the epoch."""
        from sysid.optimization.synthesizer import LureCertificateSynthesizer

        trainer, model = _make_trainer(tmp_path, max_s_trigger="on_violation")
        c_max = trainer.input_margin()
        monkeypatch.setattr(LureCertificateSynthesizer, "max_s", lambda self: None)
        s_before = float(model.s)

        with caplog.at_level(logging.WARNING):
            assert trainer._maybe_maximize_s(epoch=3, c_max=c_max) is None
        assert float(model.s) == s_before
        assert "the SDP failed" in caplog.text
