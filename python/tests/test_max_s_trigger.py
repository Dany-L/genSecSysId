"""The MaxS trigger policy and the frozen contraction rate.

``s`` is learnable and the log-det barrier only ever pushes it DOWN, so the
epoch-boundary MaxS solve is the only thing that pushes it up. When that solve
fires therefore *sets* the size of the admissible input set

    sigma(U) = |s| sqrt(1 - alpha^2),

which is the radius of the input ball admissible from anywhere in the invariant
set X (on X, V(x) <= s^2, so ||u||^2 <= s^2 - alpha^2 V(x) worst-cases to
||u||^2 <= s^2 (1 - alpha^2)).

Three policies, which is what ``training.max_s_trigger`` selects:

    "never"        -- solve only in the initialization. sigma decays all run; the
                      LMIs stay true (the per-batch repair enforces them) but
                      Theorem 1's input hypothesis fails on the training data, so
                      the certificate is valid and vacuous.
    "on_violation" -- solve only when the training data breaches c_k <= 0, so s
                      tracks the data and settles near the input floor.
    "every_epoch"  -- solve unconditionally, driving s to the MaxS ceiling. Since
                      ||h^i||_P <= 1/s that pushes H = L P^-1 toward 0, i.e. the
                      global sector condition.

alpha is frozen (``custom_params.freeze_alpha``) because it is not in the rollout:
the prediction loss gives it no gradient, the barrier prefers alpha -> 1, and a
drifting alpha collapses sigma on its own, independently of s.
"""

import logging

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sysid.config import Config, TrainingConfig
from sysid.models.constrained_rnn import SimpleLure
from sysid.training import get_loss_function, get_optimizer
from sysid.training.trainer import Trainer


def _make_model(s_value: float = 0.05, **custom) -> SimpleLure:
    """Strictly feasible at P = I so the barrier is finite from the first step."""
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
        **{"mlflow_tracking": False, "regularization_weight": 1e-3, **kwargs},
    )


class TestTriggerConfig:
    """The three options must be reachable from a config file, and a bad value must
    fail loudly at load time rather than silently picking a policy."""

    def test_default_preserves_the_old_behaviour(self):
        """The old key defaulted to False (never solve). The new default must match
        it exactly, or every config that omits the key silently gains an SDP solve
        that moves ``s`` — and with it the admissible input set."""
        assert TrainingConfig().max_s_trigger == "never"
        assert TrainingConfig().max_s_every == 1

    @pytest.mark.parametrize("trigger", ["never", "on_violation", "every_epoch"])
    def test_all_three_load_from_yaml(self, tmp_path, trigger):
        import yaml
        cfg = tmp_path / "c.yaml"
        cfg.write_text(yaml.safe_dump({
            "data": {"train_path": str(tmp_path)},
            "model": {"model_type": "crnn", "nw": 2, "nx": 2, "activation": "dzn"},
            "optimizer": {"optimizer_type": "adam"},
            "training": {"max_epochs": 1, "max_s_trigger": trigger, "max_s_every": 5},
            "mlflow": {"experiment_name": "t"},
        }))
        loaded = Config.from_yaml(str(cfg))
        assert loaded.training.max_s_trigger == trigger
        assert loaded.training.max_s_every == 5

    def test_unknown_trigger_raises(self):
        """A typo must not fall back to a default: silently training the wrong arm
        would look like a null result in the sweep."""
        with pytest.raises(ValueError, match="max_s_trigger"):
            TrainingConfig(max_s_trigger="on_violaton")

    def test_bad_cadence_raises(self):
        with pytest.raises(ValueError, match="max_s_every"):
            TrainingConfig(max_s_trigger="every_epoch", max_s_every=0)


class TestDeprecatedAlias:
    """Unknown YAML keys are dropped *silently*, so dropping the old boolean would
    turn every archived config into a no-op that looks like a null result."""

    @pytest.mark.parametrize("flag,expected", [(True, "on_violation"), (False, "never")])
    def test_bool_maps_onto_the_trigger(self, flag, expected):
        assert TrainingConfig(solve_max_s_on_violation=flag).max_s_trigger == expected

    def test_absent_alias_leaves_the_default(self):
        assert TrainingConfig().solve_max_s_on_violation is None
        assert TrainingConfig().max_s_trigger == "never"

    def test_alias_deprecation_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="sysid.config"):
            cfg = TrainingConfig(solve_max_s_on_violation=True)
        assert cfg.max_s_trigger == "on_violation"
        assert "deprecated" in caplog.text

    def test_new_key_wins_over_the_alias(self, caplog):
        """A config setting both is contradictory; the new key must win and say so."""
        with caplog.at_level(logging.WARNING, logger="sysid.config"):
            cfg = TrainingConfig(
                solve_max_s_on_violation=False, max_s_trigger="every_epoch"
            )
        assert cfg.max_s_trigger == "every_epoch"
        assert "ignoring the deprecated" in caplog.text

    def test_old_yaml_still_loads(self, tmp_path):
        import yaml
        cfg = tmp_path / "old.yaml"
        cfg.write_text(yaml.safe_dump({
            "data": {"train_path": str(tmp_path)},
            "model": {"model_type": "crnn", "nw": 2, "nx": 2, "activation": "dzn"},
            "optimizer": {"optimizer_type": "adam"},
            "training": {"max_epochs": 1, "solve_max_s_on_violation": True},
            "mlflow": {"experiment_name": "t"},
        }))
        assert Config.from_yaml(str(cfg)).training.max_s_trigger == "on_violation"


class TestFreezeAlpha:
    """alpha must be holdable at a configured value. It lives in
    ``model.custom_params`` next to ``learn_L`` (a structural choice about the
    model), NOT in TrainingConfig — the ownership branch's ``training.freeze_alpha``
    stays removed, see test_simple_constrained_training.py."""

    def test_alpha_is_learnable_by_default(self):
        assert _make_model().tau.requires_grad is True
        assert _make_model().freeze_alpha is False

    def test_freeze_alpha_detaches_tau(self):
        m = _make_model(freeze_alpha=True)
        assert m.tau.requires_grad is False
        assert m.freeze_alpha is True

    @pytest.mark.parametrize("alpha_0", [0.9, 0.99, 0.9999])
    def test_alpha_0_sets_the_rate(self, alpha_0):
        m = _make_model(alpha_0=alpha_0)
        assert float(torch.sigmoid(m.tau)) == pytest.approx(alpha_0, rel=1e-6)

    @pytest.mark.parametrize("alpha_0", [0.0, 1.0, -0.5, 1.5])
    def test_alpha_0_outside_the_open_unit_interval_raises(self, alpha_0):
        """alpha = 1 makes sigma(U) = 0 and alpha = 0 is not representable by the
        sigmoid parameterization; both must fail loudly."""
        with pytest.raises(ValueError, match="alpha_0"):
            _make_model(alpha_0=alpha_0)

    def test_frozen_alpha_survives_a_training_step(self, tmp_path):
        """The barrier prefers alpha -> 1; frozen, it must not move at all."""
        m = _make_model(freeze_alpha=True, alpha_0=0.99)
        tau_0 = m.tau.detach().clone()
        _trainer(tmp_path, m, lr=1e-2).train_epoch()
        assert torch.equal(m.tau.detach(), tau_0)

    def test_learnable_alpha_does_move(self, tmp_path):
        """Guards the contrast: if alpha stopped moving for an unrelated reason,
        the frozen-alpha test above would pass vacuously."""
        m = _make_model(alpha_0=0.99)
        tau_0 = m.tau.detach().clone()
        _trainer(tmp_path, m, lr=1e-2).train_epoch()
        assert not torch.equal(m.tau.detach(), tau_0)


class TestSigmaU:
    """sigma(U) = |s| sqrt(1 - alpha^2) is the response variable of the experiment."""

    def test_matches_the_closed_form(self, tmp_path):
        m = _make_model(s_value=3.0, alpha_0=0.6)
        expected = 3.0 * np.sqrt(1 - 0.6 ** 2)
        assert _trainer(tmp_path, m).sigma_u() == pytest.approx(expected, rel=1e-6)

    def test_uses_the_magnitude_of_s(self, tmp_path):
        """Nothing bounds the sign of s (the -log s barrier was deliberately
        removed) and logged runs do reach s < 0. Every consumer uses only s**2, so
        a negative s is a parameterization artifact and sigma must not go negative
        — a negative 'size' would silently corrupt the whole trade-off plot."""
        m = _make_model(s_value=-3.0, alpha_0=0.6)
        expected = 3.0 * np.sqrt(1 - 0.6 ** 2)
        assert _trainer(tmp_path, m).sigma_u() == pytest.approx(expected, rel=1e-6)

    def test_alpha_near_one_beats_any_s(self, tmp_path):
        """The reason alpha must be frozen: the barrier drives it toward 1, and
        sigma collapses there however large s grows. Stated as a comparison rather
        than a tolerance, so it does not depend on the float width (the package
        sets float64 globally in data/loader.py)."""
        huge_s_dead_alpha = _trainer(
            tmp_path, _make_model(s_value=1e3, alpha_0=1 - 1e-12)
        ).sigma_u()
        tiny_s_live_alpha = _trainer(
            tmp_path, _make_model(s_value=1.0, alpha_0=0.5)
        ).sigma_u()
        assert huge_s_dead_alpha < tiny_s_live_alpha
        # ...and it is a 1/sqrt(1-alpha^2) effect, not a rounding artifact.
        assert huge_s_dead_alpha < 1e-2 * 1e3


class TestTriggerBehaviour:
    """The policy must decide whether the SDP runs. Patched so these stay fast and
    solver-independent: what is under test is the *trigger*, not MaxS itself."""

    def _spy(self, tmp_path, trigger, c_max, monkeypatch, **kw):
        m = _make_model()
        tr = _trainer(tmp_path, m, max_s_trigger=trigger, **kw)
        calls = []

        class _FakeSynth:
            def max_s(self_inner):
                calls.append(1)
                return None  # "solver failed" -> leaves s alone, still counted

        monkeypatch.setattr(m, "_synth", lambda: _FakeSynth())
        return tr, calls, m

    def test_never_does_not_solve_even_when_violated(self, tmp_path, monkeypatch):
        tr, calls, _ = self._spy(tmp_path, "never", 5.0, monkeypatch)
        assert tr._maybe_maximize_s(epoch=0, c_max=5.0) is None
        assert calls == []

    def test_on_violation_solves_only_when_breached(self, tmp_path, monkeypatch):
        tr, calls, _ = self._spy(tmp_path, "on_violation", None, monkeypatch)
        tr._maybe_maximize_s(epoch=0, c_max=-1.0)
        assert calls == [], "margin satisfied -> no solve"
        tr._maybe_maximize_s(epoch=1, c_max=+1.0)
        assert len(calls) == 1, "margin breached -> one solve"

    def test_every_epoch_solves_regardless_of_the_margin(self, tmp_path, monkeypatch):
        tr, calls, _ = self._spy(tmp_path, "every_epoch", None, monkeypatch)
        tr._maybe_maximize_s(epoch=0, c_max=-99.0)
        assert len(calls) == 1, "solves although the margin is satisfied"

    def test_every_epoch_honours_the_cadence(self, tmp_path, monkeypatch):
        """max_s_every gives the near-continuous axis between every_epoch and never."""
        tr, calls, _ = self._spy(
            tmp_path, "every_epoch", None, monkeypatch, max_s_every=3
        )
        for epoch in range(9):
            tr._maybe_maximize_s(epoch=epoch, c_max=-1.0)
        assert len(calls) == 3, "epochs 0, 3, 6"

    @pytest.mark.parametrize("trigger", ["never", "on_violation", "every_epoch"])
    def test_no_solve_without_the_barrier(self, tmp_path, monkeypatch, trigger):
        """regularization_weight == 0 means no LMI barrier and no repair, so there
        is no regional certificate to enlarge in any arm."""
        m = _make_model()
        tr = _trainer(tmp_path, m, max_s_trigger=trigger)
        tr.regularization_weight = 0.0
        calls = []
        monkeypatch.setattr(
            m, "_synth", lambda: type("F", (), {"max_s": lambda s: calls.append(1)})()
        )
        tr._maybe_maximize_s(epoch=0, c_max=10.0)
        assert calls == []

    def test_solve_count_is_tracked(self, tmp_path, monkeypatch):
        """The count is a response variable: it distinguishes 'the policy fired'
        from 'the policy fired and the SDP failed'."""
        m = _make_model()
        tr = _trainer(tmp_path, m, max_s_trigger="every_epoch")
        sol = tr.model._synth().max_s()
        if sol is None:
            pytest.skip("MaxS unavailable in this environment")
        monkeypatch.setattr(m, "_synth", lambda: type("F", (), {"max_s": lambda s: sol})())
        assert tr.max_s_solve_count == 0
        tr._maybe_maximize_s(epoch=0, c_max=-1.0)
        assert tr.max_s_solve_count == 1


class TestInputMargin:
    """The margin scan is split out of the trigger so it can run in every arm."""

    def test_returns_a_finite_scalar(self, tmp_path):
        tr = _trainer(tmp_path, _make_model())
        assert np.isfinite(tr.input_margin())

    def test_shrinking_s_breaches_the_margin(self, tmp_path):
        """c_k = ||u||^2 - s^2 + alpha^2 V(x), so a small s must show up as a
        positive margin — this is the mechanism the 'never' arm walks into."""
        m = _make_model(s_value=1e-3)
        assert _trainer(tmp_path, m).input_margin() > 0

    def test_large_s_satisfies_the_margin(self, tmp_path):
        m = _make_model(s_value=1e3)
        assert _trainer(tmp_path, m).input_margin() < 0

    def test_model_is_left_in_train_mode(self, tmp_path):
        """The scan flips to eval(); leaking that would silently change the rest of
        the epoch for every arm."""
        m = _make_model()
        tr = _trainer(tmp_path, m)
        m.train()
        tr.input_margin()
        assert m.training is True


class TestMarginScanCost:
    """The scan is a full extra no-grad rollout over the training set, so it must
    not appear on a path that previously did nothing."""

    def _epoch_with_counting_margin(self, tmp_path, trigger, mlflow_tracking):
        m = _make_model()
        tr = _trainer(tmp_path, m, max_s_trigger=trigger,
                      mlflow_tracking=mlflow_tracking)
        calls = []
        real = tr.input_margin
        tr.input_margin = lambda: (calls.append(1), real())[1]
        # train() does the epoch bookkeeping; one epoch is enough to see the call.
        tr.train(max_epochs=1)
        return len(calls)

    def test_untracked_never_arm_pays_nothing(self, tmp_path):
        """Matches the old solve_max_s_on_violation=False path exactly: no solve
        and no scan."""
        assert self._epoch_with_counting_margin(
            tmp_path, "never", mlflow_tracking=False) == 0

    def test_on_violation_always_scans(self, tmp_path):
        """It needs the margin to decide, tracked or not."""
        assert self._epoch_with_counting_margin(
            tmp_path, "on_violation", mlflow_tracking=False) == 1

    @pytest.mark.parametrize("trigger", ["never", "on_violation", "every_epoch"])
    def test_tracked_runs_scan_in_every_arm(self, tmp_path, trigger):
        """A tracked run is an experiment run: sigma and the margin have to be
        recorded in all arms, at equal per-epoch cost."""
        m = _make_model()
        tr = _trainer(tmp_path, m, max_s_trigger=trigger, mlflow_tracking=True)
        calls = []
        real = tr.input_margin
        tr.input_margin = lambda: (calls.append(1), real())[1]
        import unittest.mock as mock
        with mock.patch("sysid.training.trainer.mlflow"):
            tr.train(max_epochs=1)
        assert len(calls) == 1
