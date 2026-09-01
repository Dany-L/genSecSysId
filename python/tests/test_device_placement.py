"""Device placement at the numpy -> model seams.

Anything fed into a model has to live where the model's parameters live. The bug
this guards is silent on CPU and only appears on GPU: ``torch.as_tensor`` on a
numpy array ALWAYS produces a CPU tensor — numpy arrays are host memory, so no
factory call can infer the device from them — so building inputs with only
``dtype=model.P.dtype`` works fine locally and then fails with "Expected all
tensors to be on the same device" the moment the model is on a GPU.

These tests run on CPU-only machines by using the ``meta`` device, which enforces
the same device checks as a real accelerator without needing one:

    >>> m = SimpleLure(...).to("meta")
    >>> m(torch.zeros(1, 5, 1, dtype=torch.float64))
    RuntimeError: Tensor on device cpu is not on the expected device meta!

Data loaders are deliberately NOT covered: they build CPU tensors on purpose and
the training loop moves each batch with ``.to(device)``.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from sysid.evaluation.divergence_checks import _rollout
from sysid.models.constrained_rnn import SimpleLure
from sysid.utils import as_model_tensor


def _model(**custom):
    params = {"learn_L": True}
    params.update(custom)
    return SimpleLure(nd=1, ne=1, nx=2, nw=3, activation="dzn", ts=0.05,
                      custom_params=params)


class _Stop(Exception):
    """Abort the rollout once we have seen what device it built its inputs on."""


class TestMetaReproducesTheBug:
    """Guards the guard: if 'meta' stopped enforcing device checks, every test
    below would pass vacuously."""

    def test_a_cpu_tensor_into_a_meta_model_still_raises(self):
        m = _model().to("meta")
        with pytest.raises(RuntimeError, match="device"):
            with torch.no_grad():
                m(torch.zeros(1, 5, 1, dtype=torch.float64), None, warmup_steps=0)


class TestAsModelTensor:
    def test_uses_the_models_device_and_dtype(self):
        m = _model().to("meta")
        t = as_model_tensor(m, np.zeros((1, 4, 1)))
        assert t.device == m.P.device
        assert t.dtype == m.P.dtype

    def test_dtype_can_be_overridden_without_losing_the_device(self):
        m = _model().to("meta")
        t = as_model_tensor(m, np.zeros((1, 4, 1)), dtype=torch.float32)
        assert t.device == m.P.device and t.dtype == torch.float32

    def test_cpu_model_still_gets_cpu_tensors(self):
        m = _model()
        t = as_model_tensor(m, np.zeros((2, 3, 1)))
        assert t.device == m.P.device == torch.device("cpu")
        assert torch.equal(t, torch.zeros(2, 3, 1, dtype=m.P.dtype))


class TestRolloutSeam:
    """``divergence_checks._rollout`` — the site the review flagged."""

    def _spy(self, m, seen):
        """Intercept the forward so we can read the device of what it received.

        _forward prefers ``forward_unfiltered`` when present, so attaching one is
        enough to capture the call without touching the class.
        """
        def fake(u_t, x0=None, *a, **kw):
            seen["u"] = u_t.device
            seen["x0"] = None if x0 is None else x0.device
            raise _Stop
        m.forward_unfiltered = fake
        return m

    def test_inputs_are_built_on_the_model_device(self):
        m = self._spy(_model().to("meta"), seen := {})
        with pytest.raises(_Stop):
            _rollout(m, np.zeros((1, 5, 1)))
        assert seen["u"] == m.P.device, (
            f"_rollout built its input on {seen['u']}, model is on {m.P.device}"
        )

    def test_x0_is_built_on_the_model_device(self):
        m = self._spy(_model().to("meta"), seen := {})
        with pytest.raises(_Stop):
            _rollout(m, np.zeros((1, 5, 1)), x0=np.zeros((1, 2, 1)))
        assert seen["x0"] == m.P.device

    def test_two_dimensional_input_is_still_placed_correctly(self):
        """The (T, nd) branch unsqueezes after construction — the device must
        survive that path too."""
        m = self._spy(_model().to("meta"), seen := {})
        with pytest.raises(_Stop):
            _rollout(m, np.zeros((5, 1)))
        assert seen["u"] == m.P.device

    def test_it_still_works_end_to_end_on_cpu(self):
        """The fix must not disturb the ordinary CPU path."""
        m = _model()
        y, xs, u_used, c = _rollout(m, np.zeros((2, 6, 1)))
        for arr in (y, xs, u_used, c):
            assert isinstance(arr, np.ndarray)
        assert y.shape[0] == 2


class TestRegionalVerificationSeam:
    """The same defect existed in regional_verification, which additionally
    hardcoded float64 instead of following the model."""

    def test_it_uses_the_shared_helper(self):
        """Its two rollout sites are nested inside regional_verification() and
        verify_regional(), so they cannot be imported directly; assert instead that
        the module now routes through the helper that carries the contract."""
        # sysid.evaluation re-exports the regional_verification FUNCTION under the
        # same name, which shadows the module on a plain import.
        rv = importlib.import_module("sysid.evaluation.regional_verification")
        assert rv.as_model_tensor is as_model_tensor
        source = Path(rv.__file__).read_text()
        assert "torch.tensor(u_n" not in source, (
            "a rollout site still builds its input with a bare torch.tensor"
        )
        assert "dtype=torch.float64" not in source, (
            "a rollout site still hardcodes float64 instead of the model's dtype"
        )


class TestGradientMaskHook:
    """Structural-constraint masks are closure-captured plain tensors, so
    ``nn.Module.to()`` does not move them: after ``model.to("cuda")`` the parameter
    and its gradient are on the GPU while the mask is still on the CPU."""

    def _masked_model(self):
        return _model(structural_constraints={"C": {"learnable_cols": [0],
                                                    "fixed_value": 0.0}})

    def test_a_mask_is_actually_registered(self):
        """Guards the test below: no hook, nothing to check."""
        m = self._masked_model()
        assert m.C._backward_hooks, "expected a gradient mask hook on C"

    def test_the_hook_accepts_a_gradient_from_another_device(self):
        """The real scenario: the mask was built on the CPU at __init__ time and the
        gradient arrives from wherever the parameter now lives.

        Tested by feeding the hook a foreign-device gradient directly rather than by
        calling ``.to("meta")`` on the model — meta conversion swaps the parameter
        objects and drops their hooks, which a real ``.to("cuda")`` does not do.
        """
        m = self._masked_model()
        assert m.C.device == torch.device("cpu")            # mask is on the CPU
        grad = torch.ones(m.C.shape, device="meta", dtype=m.C.dtype)
        for hook in m.C._backward_hooks.values():
            out = hook(grad)
            assert out.device == grad.device, (
                f"mask produced a {out.device} gradient for a {grad.device} one"
            )

    def test_the_mask_still_zeroes_the_fixed_entries_on_cpu(self):
        """The device fix must not change what the mask does."""
        m = self._masked_model()
        grad = torch.ones(m.C.shape, dtype=m.C.dtype)
        for hook in m.C._backward_hooks.values():
            out = hook(grad)
            assert out[0, 0] == 1.0      # learnable column survives
            assert out[0, 1] == 0.0      # fixed column is zeroed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
