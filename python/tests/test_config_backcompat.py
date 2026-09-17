"""Backward compatibility: a config that still carries keys from removed or
renamed features (e.g. ``esn_n_restarts`` after the ESN init was dropped) must
still load, dropping the stale keys with a warning rather than raising."""

import logging

import pytest

from sysid.config import Config


def _min_dict(**sections):
    d = {"data": {"train_path": "/tmp/data"}}
    d.update(sections)
    return d


def test_stale_initialization_key_loads_with_warning(caplog):
    cfg_dict = _min_dict(
        model={"initialization": {"method": "identity", "esn_n_restarts": 5}}
    )
    with caplog.at_level(logging.WARNING):
        cfg = Config.from_dict(cfg_dict)

    # Loads fine; the stale key is ignored (not carried onto the dataclass).
    assert cfg.model.initialization.method == "identity"
    assert not hasattr(cfg.model.initialization, "esn_n_restarts")
    assert "esn_n_restarts" in caplog.text
    assert "model.initialization" in caplog.text


def test_stale_training_key_loads_with_warning(caplog):
    # NOTE: this used to use ``solve_max_s_on_violation`` as the stale example.
    # That key is a real TrainingConfig field again (the after-epoch MaxS repair
    # was restored), so it would no longer be dropped — hence a key that is
    # genuinely gone.
    cfg_dict = _min_dict(training={"use_dual_certificate": True, "max_epochs": 3})
    with caplog.at_level(logging.WARNING):
        cfg = Config.from_dict(cfg_dict)

    assert cfg.training.max_epochs == 3
    assert not hasattr(cfg.training, "use_dual_certificate")
    assert "use_dual_certificate" in caplog.text


def test_solve_max_s_on_violation_is_not_dropped(caplog):
    """Regression: the key was silently ignored while the feature was missing,
    so configs asking for the after-epoch MaxS repair got no repair and no
    warning. It must round-trip now.

    It is now the deprecated spelling of ``max_s_trigger``, so it DOES draw a
    deprecation warning — that is the opposite of being dropped. What must never
    come back is the *unknown field* path, which discards the key entirely.
    """
    cfg_dict = _min_dict(training={"solve_max_s_on_violation": True})
    with caplog.at_level(logging.WARNING):
        cfg = Config.from_dict(cfg_dict)

    assert cfg.training.solve_max_s_on_violation is True
    assert cfg.training.max_s_trigger == "on_violation", "must reach the trigger"
    assert "Ignoring unknown config field" not in caplog.text


def test_clean_config_produces_no_unknown_field_warning(caplog):
    cfg_dict = _min_dict(model={"initialization": {"method": "identity"}})
    with caplog.at_level(logging.WARNING):
        Config.from_dict(cfg_dict)
    assert "Ignoring unknown config field" not in caplog.text


def test_retired_sigma_keys_raise_instead_of_being_dropped():
    """The sigma-constraint keys are gone; they must not load quietly.

    The original bug was that ``solve_max_s_on_violation`` was silently ignored
    while the feature was missing, so a config asking for the after-epoch MaxS
    repair got no repair and no warning. The Lagrangian ``sigma(U) >= c``
    mechanism that briefly replaced the MaxS trigger has now itself been rolled
    back, and the same trap would reappear in a worse form: a config selecting
    the dual-ascent arm would load cleanly and quietly train under the MaxS
    trigger's default ("never"), which reads as a null result rather than a
    mistake.

    So the retired keys are rejected outright, and the error names the
    replacement.
    """
    for key, value in (
        ("sigma_constraint", True),
        ("sigma_target", "auto"),
        ("sigma_dual_lr", 0.01),
        ("sigma_protect_s", True),
    ):
        with pytest.raises(ValueError, match="removed field"):
            Config.from_dict(_min_dict(training={key: value}))


def test_the_retired_key_error_points_at_the_replacement():
    """An error that only says 'removed' makes the reader go digging."""
    with pytest.raises(ValueError, match="max_s_trigger"):
        Config.from_dict(_min_dict(training={"sigma_constraint": True}))


def test_an_archived_run_config_still_loads(caplog):
    """The other half of the retired-key rule, and the one that is easy to get
    wrong in the destructive direction.

    ``resolve_run_artifacts`` reloads the ``config.yaml`` a finished run wrote
    beside its checkpoint, and every consumer of a past run goes through it:
    ``evaluate.py``, ``post_process.py``, ``compare.py``,
    ``export_for_matlab.py``. The runs trained while the sigma constraint was in
    place recorded those keys. Rejecting them would make every one of those runs
    unevaluatable -- so an archived config warns and drops, while a config
    someone is about to *train* with still raises.
    """
    cfg_dict = _min_dict(training={
        "sigma_constraint": True, "sigma_target": "auto",
        "sigma_dual_lr": 0.1, "sigma_protect_s": True, "max_epochs": 7,
    })
    with caplog.at_level(logging.WARNING):
        cfg = Config.from_dict(cfg_dict, allow_removed=True)

    assert cfg.training.max_epochs == 7
    assert not hasattr(cfg.training, "sigma_constraint")
    assert "Archived config carries removed field(s)" in caplog.text


def test_the_tolerant_path_is_opt_in():
    """Default strict, so a hand-written config cannot slip through."""
    with pytest.raises(ValueError, match="removed field"):
        Config.from_dict(_min_dict(training={"sigma_constraint": False}))
