"""Constrained learning on the input-set size: sigma(U) >= c by dual ascent.

The training problem is

    min  prediction error   s.t.  F < 0,  G_j > 0,  sigma(U) >= c

with ``sigma(U) = |s| sqrt(1 - alpha^2)`` the radius of the input ball
admissible from anywhere in the invariant set X, and ``c`` the *worst allowed
input*. The LMIs keep their log-det interior-point barrier; this third
constraint gets a Lagrange multiplier under projected dual ascent, following
Chamon & Ribeiro, "Probably Approximately Correct Constrained Learning"
(NeurIPS 2020), Algorithm 1.

This module replaces ``test_max_s_trigger.py`` / ``test_max_s_on_violation.py``,
which covered the epoch-boundary MaxS trigger the multiplier supersedes.

Why the mechanism needs its own tests. Nothing else in the objective pushes
``s`` up: the barrier's locality term ``-logdet[[1/s^2, l], [l', P]]`` pushes it
strictly *down*, and the prediction loss does not see ``s`` at all. So if the
Lagrangian term is silently inert -- disabled, mis-signed, or gated behind the
barrier weight -- training still runs, the loss still falls, and the only symptom
is a certificate that has quietly gone vacuous. Every test here is a guard
against that failure being invisible.

No SDP/MOSEK needed: the multiplier update is arithmetic on ``sigma``, and the
trainer tests run at ``lr=0`` so the model cannot move underneath an assertion.
"""

import logging

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from sysid.config import Config, TrainingConfig
from sysid.models.constrained_rnn import SimpleLure
from sysid.training import get_loss_function, get_optimizer
from sysid.training.trainer import Trainer
from tests.solver_utils import requires_mosek


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


class TestDifferentiableSigma:
    """``sigma_u_t`` is the whole reason a multiplier is possible here."""

    def test_matches_the_float_reporter(self, tmp_path):
        m = _make_model(s_value=0.3, alpha=0.99)
        tr = _trainer(tmp_path, m)
        assert float(m.sigma_u_t()) == pytest.approx(tr.sigma_u())

    def test_equals_the_closed_form(self):
        m = _make_model(s_value=0.4166, alpha=0.99)
        expected = 0.4166 * np.sqrt(1 - 0.99 ** 2)
        assert float(m.sigma_u_t()) == pytest.approx(expected)

    def test_gradient_pushes_s_up(self):
        """The sign is the point. ``d sigma / d|s| > 0``, so minimizing
        ``lambda (c - sigma)`` grows ``|s|`` -- the counter-push the barrier
        lacks. A sign slip here would make the constraint term *shrink* the
        certified set while reporting that it is enforcing it."""
        m = _make_model(s_value=0.3, alpha=0.99)
        m.sigma_u_t().backward()
        assert float(m.s.grad) == pytest.approx(np.sqrt(1 - 0.99 ** 2))
        assert float(m.s.grad) > 0

    def test_uses_abs_s(self):
        """``s < 0`` happens (no ``-log s`` term) and is a sign artifact: every
        consumer uses ``s**2``. sigma must report the magnitude, not go
        negative."""
        pos, neg = _make_model(s_value=0.3), _make_model(s_value=-0.3)
        assert float(neg.sigma_u_t()) > 0
        assert float(neg.sigma_u_t()) == pytest.approx(float(pos.sigma_u_t()))

    def test_finite_gradient_as_alpha_approaches_one(self):
        """sqrt(1 - alpha^2) -> 0 has an infinite derivative; the clamp keeps
        the backward pass from returning NaN at the default alpha=0.9999."""
        m = _make_model(s_value=0.3, alpha=1 - 1e-12)
        m.sigma_u_t().backward()
        assert np.isfinite(float(m.s.grad))


class TestDualAscent:
    def test_multiplier_rises_while_the_constraint_is_violated(self, tmp_path):
        tr = _trainer(tmp_path, _make_model(s_value=1e-3),
                      sigma_constraint=True, sigma_target=1.0, sigma_dual_lr=0.1)
        before = tr.dual_lambda
        slack = tr._dual_ascent_step()
        assert slack > 0, "fixture must start with sigma below c"
        assert tr.dual_lambda > before

    def test_multiplier_falls_once_the_constraint_is_met(self, tmp_path):
        """This is what a fixed penalty weight cannot do: stop charging for a
        constraint that already holds."""
        tr = _trainer(tmp_path, _make_model(s_value=100.0),
                      sigma_constraint=True, sigma_target=0.01,
                      sigma_dual_lr=0.1, sigma_dual_init=5.0)
        before = tr.dual_lambda
        assert tr._dual_ascent_step() < 0
        assert tr.dual_lambda < before

    def test_multiplier_never_goes_negative(self, tmp_path):
        """The projection onto lambda >= 0. Without it a satisfied constraint
        turns into a *reward* for shrinking the certified input set."""
        tr = _trainer(tmp_path, _make_model(s_value=100.0),
                      sigma_constraint=True, sigma_target=0.01,
                      sigma_dual_lr=1e6, sigma_dual_init=1.0)
        tr._dual_ascent_step()
        assert tr.dual_lambda >= 0.0

    def test_respects_the_cap(self, tmp_path):
        tr = _trainer(tmp_path, _make_model(s_value=1e-6),
                      sigma_constraint=True, sigma_target=10.0,
                      sigma_dual_lr=1e3, sigma_dual_max=2.5)
        for _ in range(5):
            tr._dual_ascent_step()
        assert tr.dual_lambda == pytest.approx(2.5)

    def test_update_is_the_projected_ascent_rule(self, tmp_path):
        """lambda <- max(0, lambda + eta (c - sigma)), exactly."""
        m = _make_model(s_value=0.2, alpha=0.99)
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5,
                      sigma_dual_lr=0.25, sigma_dual_init=1.0)
        expected = 1.0 + 0.25 * (0.5 - float(m.sigma_u_t()))
        tr._dual_ascent_step()
        assert tr.dual_lambda == pytest.approx(expected)

    def test_disabled_leaves_the_multiplier_alone(self, tmp_path):
        tr = _trainer(tmp_path, _make_model(s_value=1e-3), sigma_constraint=False)
        assert tr._dual_ascent_step() is None
        assert tr.dual_lambda == 0.0


class TestTarget:
    def test_auto_resolves_to_the_input_floor(self, tmp_path):
        """c = max_k||u_k|| over the converging training split."""
        m = _make_model()
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target="auto")
        assert tr._resolve_sigma_target() == pytest.approx(tr.input_floor())
        # the loader is a constant u = 0.1, so the floor is exactly that
        assert tr.sigma_target == pytest.approx(0.1)

    def test_explicit_float_is_used_as_given(self, tmp_path):
        tr = _trainer(tmp_path, _make_model(), sigma_constraint=True, sigma_target=0.25)
        assert tr._resolve_sigma_target() == pytest.approx(0.25)

    def test_auto_without_a_floor_disables_the_constraint_loudly(self, tmp_path, caplog):
        """Rather than silently defaulting to some number -- an inactive
        constraint that still reports as enabled is the failure this module
        exists to prevent."""
        m = _make_model()
        with torch.no_grad():
            m.u_max.fill_(float("nan"))
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target="auto")
        tr.model.u_max.fill_(float("nan"))
        tr.sigma_target = None
        with caplog.at_level(logging.WARNING):
            assert tr._resolve_sigma_target() is None
        assert "INACTIVE" in caplog.text

    def test_target_is_resolved_once(self, tmp_path):
        tr = _trainer(tmp_path, _make_model(), sigma_constraint=True, sigma_target=0.25)
        tr._resolve_sigma_target()
        tr.sigma_target_spec = 99.0  # a later change must not re-resolve
        assert tr._resolve_sigma_target() == pytest.approx(0.25)


class TestLagrangianTerm:
    def test_term_value(self, tmp_path):
        m = _make_model(s_value=0.2, alpha=0.99)
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5,
                      sigma_dual_init=2.0)
        expected = 2.0 * (0.5 - float(m.sigma_u_t()))
        assert float(tr.sigma_lagrangian_term()) == pytest.approx(expected)

    def test_none_when_disabled(self, tmp_path):
        tr = _trainer(tmp_path, _make_model(), sigma_constraint=False)
        assert tr.sigma_lagrangian_term() is None

    def test_term_survives_a_zero_barrier_weight(self, tmp_path):
        """The gate bug this placement guards against.

        The barrier and input-hinge terms live inside ``if
        regularization_weight > 0``. The barrier weight *decays* during
        training, down to ``min_regularization_weight``. If the Lagrangian term
        were folded into that branch it would switch off exactly when the
        barrier stops holding ``s`` down -- i.e. precisely when it is needed.
        """
        m = _make_model(s_value=0.2, alpha=0.99)
        tr = _trainer(tmp_path, m, regularization_weight=0.0,
                      sigma_constraint=True, sigma_target=0.5, sigma_dual_init=2.0)
        term = tr.sigma_lagrangian_term()
        assert term is not None and float(term) > 0

        out = tr.train_epoch()
        # pred_loss is the whole barrier-free objective; the total must exceed
        # it by the Lagrangian term, i.e. the term really reached the loss.
        assert out["loss"] > out["pred_loss"]

    def test_term_reaches_the_gradient_on_s(self, tmp_path):
        """End to end: a training step with the constraint active must leave a
        gradient on ``s``, and it must point up."""
        m = _make_model(s_value=0.2, alpha=0.99)
        tr = _trainer(tmp_path, m, lr=0.0, regularization_weight=0.0,
                      sigma_constraint=True, sigma_target=10.0, sigma_dual_init=1.0)
        tr.train_epoch()
        assert m.s.grad is not None
        # loss carries -lambda*sigma, so dLoss/ds < 0 => gradient descent grows s
        assert float(m.s.grad) < 0


class TestEpochIntegration:
    def test_epoch_metrics_report_the_multiplier(self, tmp_path):
        out = _trainer(tmp_path, _make_model(), sigma_constraint=True,
                       sigma_target=0.5).train_epoch()
        assert "dual_lambda" in out

    def test_multiplier_is_not_logged_as_a_gradient_statistic(self, tmp_path):
        """Anything in train_epoch's dict that is missing from the exclusion
        list in ``train`` is blind-logged as a gradient metric."""
        import inspect
        src = inspect.getsource(Trainer.train)
        excl = src[src.index("grad_stats = {"):src.index("]", src.index("grad_stats = {"))]
        assert '"dual_lambda"' in excl

    def test_training_actually_grows_sigma(self, tmp_path):
        """The mechanism, end to end. With a target far above the current
        sigma and a real learning rate, sigma must increase over an epoch --
        against a barrier that is pushing it the other way."""
        m = _make_model(s_value=0.2, alpha=0.99)
        tr = _trainer(tmp_path, m, lr=1e-2, sigma_constraint=True,
                      sigma_target=5.0, sigma_dual_init=1.0)
        before = tr.sigma_u()
        tr.train_epoch()
        assert tr.sigma_u() > before

    def test_no_constraint_means_sigma_is_not_defended(self, tmp_path):
        """The control for the test above: with the multiplier off, nothing
        opposes the barrier and sigma does not grow."""
        m = _make_model(s_value=0.2, alpha=0.99)
        tr = _trainer(tmp_path, m, lr=1e-2, sigma_constraint=False)
        before = tr.sigma_u()
        tr.train_epoch()
        assert tr.sigma_u() <= before


class TestCheckpoint:
    def test_multiplier_round_trips(self, tmp_path):
        """lambda is trainer state, not model state, so a resume would restart
        it at lambda^(0) with the certificate already moved."""
        tr = _trainer(tmp_path, _make_model(), sigma_constraint=True,
                      sigma_target=0.5, sigma_dual_init=3.0)
        tr._resolve_sigma_target()
        tr.save_checkpoint("ck.pt")
        ck = torch.load(tmp_path / "m" / "ck.pt", weights_only=False)
        assert ck["dual_lambda"] == pytest.approx(3.0)
        assert ck["sigma_target"] == pytest.approx(0.5)


class TestConfig:
    def test_defaults_are_off(self):
        cfg = TrainingConfig()
        assert cfg.sigma_constraint is False
        assert cfg.sigma_target == "auto"
        assert cfg.sigma_dual_lr == 0.01
        assert cfg.sigma_dual_init == 1.0
        assert cfg.sigma_dual_max is None

    @pytest.mark.parametrize("bad", [
        {"sigma_target": "max"},
        {"sigma_target": -1.0},
        {"sigma_dual_lr": 0.0},
        {"sigma_dual_lr": -0.1},
        {"sigma_dual_init": -1.0},
        {"sigma_dual_max": 0.0},
    ])
    def test_a_typo_raises_rather_than_selecting_another_arm(self, bad):
        with pytest.raises(ValueError):
            TrainingConfig(**bad)

    def test_loads_from_yaml(self, tmp_path):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "data": {"train_path": str(tmp_path)},
            "model": {"model_type": "crnn", "nw": 2, "nx": 2, "activation": "dzn"},
            "training": {"max_epochs": 1, "sigma_constraint": True,
                         "sigma_target": 0.25, "sigma_dual_lr": 0.05},
        }))
        loaded = Config.from_yaml(str(cfg_path))
        assert loaded.training.sigma_constraint is True
        assert loaded.training.sigma_target == 0.25
        assert loaded.training.sigma_dual_lr == 0.05

    def test_survives_a_save_load_roundtrip(self, tmp_path):
        """train.py writes config.yaml per run; the keys must come back."""
        cfg = Config.from_dict({
            "data": {"train_path": str(tmp_path)},
            "training": {"sigma_constraint": True, "sigma_target": 0.25},
        })
        out = tmp_path / "out.yaml"
        cfg.save_yaml(str(out))
        back = Config.from_yaml(str(out))
        assert back.training.sigma_constraint is True
        assert back.training.sigma_target == 0.25


@requires_mosek
class TestWarmStart:
    """``s`` starts at a hardcoded 1.0 unless the init's own SDP fires, and Adam
    is scale invariant, so without this the dual spends the whole run in
    transient instead of enforcing anything."""

    def _model(self):
        m = _make_model(s_value=1.0, alpha=0.99)
        with torch.no_grad():
            m.u_max.fill_(0.25 ** 2)  # input floor 0.25
        return m

    def test_lands_on_the_constraint_boundary(self, tmp_path):
        m = self._model()
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5)
        tr._sigma_warm_start()
        assert tr.sigma_u() == pytest.approx(0.5, rel=1e-6)

    def test_does_not_go_to_the_maxs_ceiling(self, tmp_path):
        """MaxS would answer 1/sqrt(EPS) = 1e3 -- a solver constant that drives
        H -> 0 (the global sector condition) and leaves the constraint so slack
        the multiplier has nothing to do. The boundary is the target, not the
        maximum."""
        m = self._model()
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5)
        tr._sigma_warm_start()
        assert float(m.s) < 100.0, "warm start ran away toward the MaxS ceiling"
        assert float(m.s) == pytest.approx(0.5 / np.sqrt(1 - 0.99 ** 2), rel=1e-6)

    def test_leaves_the_certificate_feasible(self, tmp_path):
        m = self._model()
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5)
        tr._sigma_warm_start()
        assert m.check_constraints()

    def test_does_not_shrink_an_already_sufficient_s(self, tmp_path):
        m = _make_model(s_value=100.0, alpha=0.99)
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5)
        tr._sigma_warm_start()
        assert float(m.s) == pytest.approx(100.0)

    def test_is_a_no_op_when_the_constraint_is_off(self, tmp_path):
        m = self._model()
        tr = _trainer(tmp_path, m, sigma_constraint=False)
        assert tr._sigma_warm_start() is None
        assert float(m.s) == pytest.approx(1.0)

    def test_can_be_disabled(self, tmp_path):
        m = self._model()
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5,
                      sigma_warm_start=False)
        assert tr._sigma_warm_start() is None
        assert float(m.s) == pytest.approx(1.0)

    def test_unreachable_target_warns_and_keeps_training(self, tmp_path, caplog):
        """An unreachable c must be reported as infeasibility, not silently
        absorbed -- and must not abort the run."""
        m = self._model()
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=1e9)
        with caplog.at_level(logging.WARNING):
            tr._sigma_warm_start()
        assert "NOT reachable" in caplog.text
        assert m.check_constraints()


class TestCouplingNormIsReported:
    """``||H||`` is the mediator for the whole trade-off: the locality LMI gives
    ``||h^i||_P <= 1/s``, so buying sigma with s drives ``H = L P^-1`` to zero --
    the global sector condition, i.e. a near-linear model. A reported sigma
    without the ``||H||`` it cost is not interpretable, and it used to be logged
    only when the (unrelated) anti-global regularizer was switched on.
    """

    def test_logged_without_the_h_regularizer(self, tmp_path):
        from unittest.mock import patch

        m = _make_model(s_value=0.2, alpha=0.99)
        tr = _trainer(tmp_path, m, mlflow_tracking=True,
                      h_regularization_weight=0.0,
                      sigma_constraint=True, sigma_target=0.5,
                      sigma_warm_start=False)
        with patch("sysid.training.trainer.mlflow") as mlf:
            tr.train(max_epochs=1)
        logged = {c.args[0] for c in mlf.log_metric.call_args_list}
        assert "norm_H" in logged
        assert {"dual_lambda", "sigma_u", "sigma_target", "sigma_slack"} <= logged


class TestRepairPreservesTheCertifiedSet:
    """A repair only has to restore feasibility; *which* feasible point it picks
    is free. The free-s tier minimizes a conditioning objective that has no
    reason to respect ``sigma(U) >= c``, so it can silently undo the constraint.

    Measured on the 1-D benchmark before this tier existed: one free-s repair at
    epoch 8 reset ``s`` from 38.77 to 1.05 (a 37x collapse of the certified
    input set), and the dual then spent 90 epochs climbing back.
    """

    def test_floor_is_set_from_the_target(self, tmp_path):
        m = _make_model(alpha=0.99)
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5)
        tr._resolve_sigma_target()
        assert m._sigma_s_floor == pytest.approx(0.5 / np.sqrt(1 - 0.99 ** 2))

    def test_no_floor_when_the_constraint_is_off(self, tmp_path):
        m = _make_model()
        tr = _trainer(tmp_path, m, sigma_constraint=False)
        tr._resolve_sigma_target()
        assert getattr(m, "_sigma_s_floor", None) is None

    def test_repair_prefers_the_floor_over_freeing_s(self, tmp_path, monkeypatch):
        """Tier 1 infeasible -> tier 2 (pinned at the floor) must be tried
        before tier 3 frees the scale."""
        from sysid.optimization.solutions import CertificateSolution

        m = _make_model(s_value=40.0, alpha=0.99)
        m.set_sigma_floor(35.0)
        calls = []

        class _Fake:
            def feasibility(self, s):
                calls.append(s)
                if s is None:
                    return CertificateSolution(P=np.eye(2), L=np.zeros((2, 2)),
                                               M=np.eye(2), s=0.001)
                if s == pytest.approx(35.0):
                    return CertificateSolution(P=np.eye(2), L=np.zeros((2, 2)),
                                               M=np.eye(2), s=35.0)
                return None  # tier 1 infeasible

        monkeypatch.setattr(m, "_synth", lambda: _Fake())
        assert m.feasibility_problem() is True
        assert calls == [pytest.approx(40.0), pytest.approx(35.0)], \
            "tier 3 (s=None) must not be reached while the floor is certifiable"
        assert float(m.s) == pytest.approx(35.0)
        assert m.last_repair_freed_s is False

    def test_falls_through_to_free_s_when_the_floor_is_infeasible(self, tmp_path, monkeypatch):
        """The floor is a preference, not a hard gate: a repair that cannot hold
        it must still repair rather than force a rollback."""
        from sysid.optimization.solutions import CertificateSolution

        m = _make_model(s_value=40.0, alpha=0.99)
        m.set_sigma_floor(35.0)
        calls = []

        class _Fake:
            def feasibility(self, s):
                calls.append(s)
                if s is None:
                    return CertificateSolution(P=np.eye(2), L=np.zeros((2, 2)),
                                               M=np.eye(2), s=0.5)
                return None

        monkeypatch.setattr(m, "_synth", lambda: _Fake())
        assert m.feasibility_problem() is True
        assert calls == [pytest.approx(40.0), pytest.approx(35.0), None]
        assert m.last_repair_freed_s is True

    def test_floor_is_skipped_when_s_is_already_below_it(self, tmp_path, monkeypatch):
        """No point re-solving at a scale larger than the one that just failed."""
        from sysid.optimization.solutions import CertificateSolution

        m = _make_model(s_value=1.0, alpha=0.99)
        m.set_sigma_floor(35.0)
        calls = []

        class _Fake:
            def feasibility(self, s):
                calls.append(s)
                return None if s is not None else CertificateSolution(
                    P=np.eye(2), L=np.zeros((2, 2)), M=np.eye(2), s=0.5)

        monkeypatch.setattr(m, "_synth", lambda: _Fake())
        m.feasibility_problem()
        assert calls == [pytest.approx(1.0), None]


class TestProtectS:
    """Refusing the free-s repair tier: roll the step back rather than buy
    feasibility with the certified input set.

    The infeasibility a repair reacts to is usually transient -- on the 1-D
    benchmark the FINAL theta certified s = 52.7, above the 38.5 a single
    free-s repair discarded at epoch 8. A rollback costs one batch; freeing s
    cost 90 epochs of dual ascent.
    """

    @staticmethod
    def _fake_synth(results):
        """`results` maps the `s` passed to feasibility() -> solution or None."""
        from sysid.optimization.solutions import CertificateSolution

        calls = []

        class _Fake:
            def feasibility(self, s):
                calls.append(s)
                out = results.get("free" if s is None else round(float(s), 6))
                if out is None:
                    return None
                return CertificateSolution(P=np.eye(2), L=np.zeros((2, 2)),
                                           M=np.eye(2), s=out)
        return _Fake(), calls

    def test_refuses_to_free_s_and_reports_failure(self, monkeypatch):
        m = _make_model(s_value=40.0, alpha=0.99)
        m.set_sigma_floor(35.0, protect_s=True)
        fake, calls = self._fake_synth({"free": 0.001})  # only free-s would work
        monkeypatch.setattr(m, "_synth", lambda: fake)

        assert m.feasibility_problem() is False, "must fail so the trainer rolls back"
        assert None not in calls, "the free-s tier must not even be solved"
        assert float(m.s) == pytest.approx(40.0), "s must be left untouched"

    def test_still_takes_a_repair_that_keeps_the_scale(self, monkeypatch):
        """protect_s forbids tier 3, not repair as such."""
        m = _make_model(s_value=40.0, alpha=0.99)
        m.set_sigma_floor(35.0, protect_s=True)
        fake, calls = self._fake_synth({40.0: 40.0})  # tier 1 succeeds
        monkeypatch.setattr(m, "_synth", lambda: fake)

        assert m.feasibility_problem() is True
        assert calls == [pytest.approx(40.0)]
        assert float(m.s) == pytest.approx(40.0)

    def test_tier_2_still_runs_under_protect_s(self, monkeypatch):
        m = _make_model(s_value=40.0, alpha=0.99)
        m.set_sigma_floor(35.0, protect_s=True)
        fake, calls = self._fake_synth({35.0: 35.0})  # only the floor works
        monkeypatch.setattr(m, "_synth", lambda: fake)

        assert m.feasibility_problem() is True
        assert calls == [pytest.approx(40.0), pytest.approx(35.0)]
        assert float(m.s) == pytest.approx(35.0)

    def test_off_restores_the_free_s_fallback(self, monkeypatch):
        m = _make_model(s_value=40.0, alpha=0.99)
        m.set_sigma_floor(35.0, protect_s=False)
        fake, calls = self._fake_synth({"free": 0.001})
        monkeypatch.setattr(m, "_synth", lambda: fake)

        assert m.feasibility_problem() is True
        assert None in calls
        assert m.last_repair_freed_s is True

    def test_default_is_off_for_a_model_that_never_set_a_floor(self, monkeypatch):
        """Runs without the sigma constraint must behave exactly as before."""
        m = _make_model(s_value=40.0, alpha=0.99)
        fake, calls = self._fake_synth({"free": 0.001})
        monkeypatch.setattr(m, "_synth", lambda: fake)

        assert m.feasibility_problem() is True
        assert m.last_repair_freed_s is True

    def test_trainer_passes_the_flag_through(self, tmp_path):
        m = _make_model(alpha=0.99)
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5,
                      sigma_protect_s=True)
        tr._resolve_sigma_target()
        assert m._sigma_protect_s is True

        m2 = _make_model(alpha=0.99)
        tr2 = _trainer(tmp_path, m2, sigma_constraint=True, sigma_target=0.5,
                       sigma_protect_s=False)
        tr2._resolve_sigma_target()
        assert m2._sigma_protect_s is False

    def test_a_refused_repair_rolls_the_step_back(self, tmp_path, monkeypatch):
        """End to end: the trainer must restore the parameters, not proceed with
        an infeasible iterate."""
        m = _make_model(s_value=40.0, alpha=0.99)
        tr = _trainer(tmp_path, m, lr=1e-2, sigma_constraint=True,
                      sigma_target=0.5, sigma_protect_s=True,
                      sigma_warm_start=False)
        tr._resolve_sigma_target()
        monkeypatch.setattr(m, "check_constraints", lambda: False)
        fake, _ = self._fake_synth({})  # every tier infeasible
        monkeypatch.setattr(m, "_synth", lambda: fake)

        before = {k: v.detach().clone() for k, v in m.named_parameters()}
        out = tr.train_epoch()
        assert out["rollback_count"] > 0
        for k, v in m.named_parameters():
            assert torch.allclose(v.detach(), before[k]), f"{k} was not rolled back"


class TestLocalityTightness:
    """``||H||_F`` reads like the mediator for the sigma trade-off but is not it.

    The locality LMI bounds the *P-norm* of the rows, ``||h^i||_P <= 1/s``, and
    that is what goes to zero as ``s`` grows (``H = 0`` is the global sector
    condition). ``||H||_F`` is not normalized by ``P`` and can move the opposite
    way when ``P`` changes -- measured on the 1-D benchmark, ``s`` 4.4 -> 39.1
    took ``||h||_P`` 0.217 -> 0.0254 while ``||H||_F`` went 0.93 -> 1.09.
    Reading the Frobenius norm there would say the model became *more* regional
    when the certificate had in fact moved 8.5x toward the global corner.
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

    def test_is_logged(self, tmp_path):
        from unittest.mock import patch

        m = _make_model(s_value=0.2, alpha=0.99)
        tr = _trainer(tmp_path, m, mlflow_tracking=True, sigma_constraint=True,
                      sigma_target=0.5, sigma_warm_start=False)
        with patch("sysid.training.trainer.mlflow") as mlf:
            tr.train(max_epochs=1)
        logged = {c.args[0] for c in mlf.log_metric.call_args_list}
        assert {"h_norm_P", "locality_tightness"} <= logged


class TestProtectSDefaultsOff:
    """``sigma_protect_s`` must be opt-in.

    Against a REACHABLE target it is clearly better (measured: sigma >= c on
    97% of epochs vs 4%). Against an unreachable one it converts "the constraint
    cannot be met" into "training cannot proceed": every repair is refused,
    every batch rolls back, and the run ends with its parameters exactly where
    they started -- losing the model as well as the constraint. That is strictly
    worse than reporting an unmet constraint, so it cannot be the default.
    """

    def test_config_default_is_off(self):
        assert TrainingConfig().sigma_protect_s is False

    def test_trainer_default_is_off(self, tmp_path):
        tr = _trainer(tmp_path, _make_model(), sigma_constraint=True,
                      sigma_target=0.5)
        tr._resolve_sigma_target()
        assert tr.sigma_protect_s is False
        assert tr.model._sigma_protect_s is False

    @requires_mosek
    def test_an_unreachable_target_warns_about_the_stall(self, tmp_path, caplog):
        """The one combination that deadlocks must announce itself up front,
        naming the three ways out."""
        m = _make_model(s_value=1.0, alpha=0.99)
        with torch.no_grad():
            m.u_max.fill_(0.25 ** 2)
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=1e9,
                      sigma_protect_s=True)
        with caplog.at_level(logging.WARNING):
            tr._sigma_warm_start()
        assert "NOT reachable" in caplog.text
        assert "STALL" in caplog.text
        assert "sigma_protect_s: false" in caplog.text

    @requires_mosek
    def test_no_stall_warning_when_the_target_is_reachable(self, tmp_path, caplog):
        m = _make_model(s_value=1.0, alpha=0.99)
        with torch.no_grad():
            m.u_max.fill_(0.25 ** 2)
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5,
                      sigma_protect_s=True)
        with caplog.at_level(logging.WARNING):
            tr._sigma_warm_start()
        assert "STALL" not in caplog.text


class TestAllRolledBackDenominator:
    """The "everything rolled back" test must count steps, not converging batches.

    Regression, and a consequential one. ``epoch_rollback_count`` accumulates
    over BOTH passes; the threshold used to be ``len(self.train_loader)``, the
    *converging* batch count alone. On the 1-D benchmark that is 8 converging
    against 30 diverging batches, so the branch fired at 21% rolled back rather
    than 100% -- and it decays the LR *and* the barrier weight, so the
    reg-weight early stop then killed the run within ~4 epochs.

    On the sigma sweep this truncated every ``c >= 6.5`` run to 9-10 of 200
    epochs, and their inflated error read as "certifying larger inputs costs
    accuracy" when it was really "this run never trained".
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

    def test_partial_rollbacks_do_not_trip_the_branch(self, tmp_path, monkeypatch):
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


class TestInertConfigurationsWarn:
    """``sigma_constraint`` can be switched on in combinations where it cannot
    act, while still logging a rising ``dual_lambda`` -- which reads exactly like
    an enforced constraint. Both are legitimate arms on their own, so these warn
    rather than raise; what must not happen is silence.
    """

    def test_learn_L_false_freezes_s_and_warns(self, tmp_path, caplog):
        """``learn_L: false`` creates s and tau with requires_grad=False, so the
        Lagrangian term is a constant. Measured on Duffing: s pinned at 1.0,
        sigma 0.0894 vs c = 0.15, lambda climbing past 1.005 regardless."""
        m = _make_model(learn_L=False)
        assert m.s.requires_grad is False, "fixture must reproduce the frozen s"
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5,
                      sigma_warm_start=False)
        with caplog.at_level(logging.WARNING):
            tr._resolve_sigma_target()
        assert "learn_L is false" in caplog.text
        assert "lambda will grow without bound" in caplog.text

    def test_no_barrier_warns_that_sigma_certifies_nothing(self, tmp_path, caplog):
        m = _make_model()
        tr = _trainer(tmp_path, m, regularization_weight=0.0,
                      sigma_constraint=True, sigma_target=0.5,
                      sigma_warm_start=False)
        with caplog.at_level(logging.WARNING):
            tr._resolve_sigma_target()
        assert "use_custom_regularization is false" in caplog.text

    def test_the_healthy_combination_is_silent(self, tmp_path, caplog):
        m = _make_model()
        tr = _trainer(tmp_path, m, regularization_weight=1e-3,
                      sigma_constraint=True, sigma_target=0.5,
                      sigma_warm_start=False)
        with caplog.at_level(logging.WARNING):
            tr._resolve_sigma_target()
        assert "learn_L is false" not in caplog.text
        assert "use_custom_regularization is false" not in caplog.text

    def test_warning_is_emitted_once_not_per_batch(self, tmp_path, caplog):
        """The target resolves once and caches, so the warning must not repeat
        on every batch and drown the log."""
        m = _make_model(learn_L=False)
        tr = _trainer(tmp_path, m, sigma_constraint=True, sigma_target=0.5,
                      sigma_warm_start=False)
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                tr._resolve_sigma_target()
        assert caplog.text.count("learn_L is false") == 1
