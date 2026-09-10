"""Backward compatibility for checkpoints written before the ``u_max`` buffer.

``u_max`` (the input floor, ``sup_k ||u_k||^2`` in normalized units) is a
persistent buffer, so it is expected in the state_dict — but runs trained before
it was introduced have no entry for it. Those checkpoints must still load, with
``u_max`` left at its "unset" default (nan), which every consumer already tests
for. This is the mirror of ``test_dual_removal_backcompat.py``: that one covers
keys a checkpoint still carries, this one keys a checkpoint predates.
"""

import pytest
import torch

from sysid.models.constrained_rnn import SimpleLure, SimpleLureSafe


def _make_model(cls=SimpleLure) -> SimpleLure:
    return cls(nd=1, ne=1, nx=2, nw=1, activation="dzn",
               custom_params={"learn_L": True})


def _pre_u_max_state(model) -> dict:
    """A checkpoint from before ``u_max`` existed: same keys, minus ``u_max``."""
    state = dict(model.state_dict())
    assert "u_max" in state, "u_max should be a persistent buffer"
    del state["u_max"]
    return state


@pytest.mark.parametrize("cls", [SimpleLure, SimpleLureSafe])
def test_checkpoint_without_u_max_loads(cls):
    """A strict (default) load must not raise on the missing ``u_max``."""
    legacy = _pre_u_max_state(_make_model(cls))

    m = _make_model(cls)
    m.load_state_dict(legacy)  # strict=True by default

    assert torch.isnan(m.u_max), "u_max must stay unset, not be invented"


def test_other_weights_still_load_from_a_pre_u_max_checkpoint():
    """Filling in ``u_max`` must not disturb the weights that *are* present."""
    source = _make_model()
    with torch.no_grad():
        source.s.fill_(2.5)
        source.A.fill_(0.25)
    legacy = _pre_u_max_state(source)

    m = _make_model()
    m.load_state_dict(legacy)

    assert float(m.s) == pytest.approx(2.5)
    assert torch.allclose(m.A, torch.full_like(m.A, 0.25))


def test_recorded_u_max_is_still_restored():
    """A checkpoint that *does* carry ``u_max`` keeps its value — the default is
    only a fallback for the ones that predate the buffer."""
    source = _make_model()
    source.set_input_bound(99.68)
    state = dict(source.state_dict())
    assert float(state["u_max"]) == pytest.approx(99.68)

    m = _make_model()
    m.load_state_dict(state)
    assert float(m.u_max) == pytest.approx(99.68)


def test_missing_non_optional_key_still_raises():
    """Strictness is preserved: a genuinely missing weight still raises."""
    state = dict(_make_model().state_dict())
    del state["A"]

    with pytest.raises(RuntimeError):
        _make_model().load_state_dict(state)
