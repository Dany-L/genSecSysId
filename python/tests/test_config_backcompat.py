"""Backward compatibility: a config that still carries keys from removed or
renamed features (e.g. ``esn_n_restarts`` after the ESN init was dropped) must
still load, dropping the stale keys with a warning rather than raising."""

import logging

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
    cfg_dict = _min_dict(training={"solve_max_s_on_violation": True, "max_epochs": 3})
    with caplog.at_level(logging.WARNING):
        cfg = Config.from_dict(cfg_dict)

    assert cfg.training.max_epochs == 3
    assert not hasattr(cfg.training, "solve_max_s_on_violation")
    assert "solve_max_s_on_violation" in caplog.text


def test_clean_config_produces_no_unknown_field_warning(caplog):
    cfg_dict = _min_dict(model={"initialization": {"method": "identity"}})
    with caplog.at_level(logging.WARNING):
        Config.from_dict(cfg_dict)
    assert "Ignoring unknown config field" not in caplog.text
