"""Backward compatibility after removing the dual constrained-learning method.

Two legacy artifacts must not raise now that the dual method is gone:
  * old configs may still carry ``dual_penalty`` / ``regularization_method`` in
    ``custom_params`` (a free-form dict), and
  * old checkpoints carry a ``dual_penalty`` buffer in their ``state_dict``.
"""

import pytest
import torch

from sysid.models.constrained_rnn import SimpleLure


def _make_model(extra_params=None) -> SimpleLure:
    cp = {"learn_L": True}
    if extra_params:
        cp.update(extra_params)
    return SimpleLure(nd=1, ne=1, nx=2, nw=1, activation="dzn", custom_params=cp)


def test_legacy_dual_custom_params_do_not_raise():
    """Legacy dual keys in custom_params are accepted and ignored."""
    m = _make_model(
        {
            "regularization_method": "dual",
            "dual_penalty_init": 2.0,
            "dual_penalty_growth": 1.2,
            "dual_penalty_shrink": 0.8,
            "l_nonzero_weight": 0.1,
        }
    )
    # The dual mechanism is gone: neither the attributes nor the buffer exist.
    assert not hasattr(m, "dual_penalty")
    assert not hasattr(m, "regularization_method")


def test_checkpoint_with_dual_penalty_buffer_loads():
    """A state_dict carrying the retired ``dual_penalty`` buffer still loads."""
    m = _make_model()
    state = m.state_dict()
    assert "dual_penalty" not in state  # buffer is gone from fresh models

    # Simulate an old checkpoint that still has the retired buffer.
    legacy = dict(state)
    legacy["dual_penalty"] = torch.tensor(1.0)

    # A strict (default) load must not raise on the retired key.
    _make_model().load_state_dict(legacy)


def test_unknown_key_still_raises():
    """Strictness is preserved: a genuinely unknown key still raises."""
    legacy = dict(_make_model().state_dict())
    legacy["bogus_param"] = torch.tensor(1.0)

    with pytest.raises(RuntimeError):
        _make_model().load_state_dict(legacy)
