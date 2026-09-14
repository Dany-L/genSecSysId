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
    # NOTE: not ``solve_max_s_on_violation`` / ``max_s_trigger`` — those are
    # *retired* keys, which now raise (see below) rather than being dropped.
    # This needs a key that is genuinely gone and unremarkable.
    cfg_dict = _min_dict(training={"use_dual_certificate": True, "max_epochs": 3})
    with caplog.at_level(logging.WARNING):
        cfg = Config.from_dict(cfg_dict)

    assert cfg.training.max_epochs == 3
    assert not hasattr(cfg.training, "use_dual_certificate")
    assert "use_dual_certificate" in caplog.text


def test_retired_max_s_keys_raise_instead_of_being_dropped():
    """Regression, restated for the mechanism that replaced them.

    The original bug was that ``solve_max_s_on_violation`` was silently ignored
    while the feature was missing, so a config asking for the after-epoch MaxS
    repair got no repair and no warning. The MaxS trigger has since been
    replaced by the Lagrangian constraint ``sigma(U) >= c``, and the same trap
    would reappear in a worse form: an archived config selecting a trigger arm
    would load cleanly and quietly train a *different* arm, which reads as a
    null result rather than a mistake.

    So the retired keys are rejected outright, and the error names the
    replacement.
    """
    for key, value in (
        ("max_s_trigger", "on_violation"),
        ("max_s_every", 5),
        ("solve_max_s_on_violation", True),
    ):
        with pytest.raises(ValueError, match="removed field"):
            Config.from_dict(_min_dict(training={key: value}))


def test_the_retired_key_error_points_at_the_replacement():
    """An error that only says 'removed' makes the reader go digging."""
    with pytest.raises(ValueError, match="sigma_constraint"):
        Config.from_dict(_min_dict(training={"max_s_trigger": "never"}))


def test_clean_config_produces_no_unknown_field_warning(caplog):
    cfg_dict = _min_dict(model={"initialization": {"method": "identity"}})
    with caplog.at_level(logging.WARNING):
        Config.from_dict(cfg_dict)
    assert "Ignoring unknown config field" not in caplog.text


def test_an_archived_run_config_still_loads(caplog):
    """The other half of the retired-key rule, and the one that is easy to get
    wrong in the destructive direction.

    ``resolve_run_artifacts`` reloads the ``config.yaml`` a finished run wrote
    beside its checkpoint, and every consumer of a past run goes through it:
    ``evaluate.py``, ``post_process.py``, ``compare.py``,
    ``export_for_matlab.py``. Those configs record the trigger the run actually
    used. Rejecting them would make every previously-trained run
    unevaluatable -- so an archived config warns and drops, while a config
    someone is about to *train* with still raises.
    """
    cfg_dict = _min_dict(training={
        "max_s_trigger": "on_violation", "max_s_every": 1,
        "solve_max_s_on_violation": None, "max_epochs": 7,
    })
    with caplog.at_level(logging.WARNING):
        cfg = Config.from_dict(cfg_dict, allow_removed=True)

    assert cfg.training.max_epochs == 7
    assert not hasattr(cfg.training, "max_s_trigger")
    assert "Archived config carries removed field(s)" in caplog.text


def test_the_tolerant_path_is_opt_in():
    """Default strict, so a hand-written config cannot slip through."""
    with pytest.raises(ValueError, match="removed field"):
        Config.from_dict(_min_dict(training={"max_s_trigger": "never"}))
