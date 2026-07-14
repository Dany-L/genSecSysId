"""Tests for ``SimpleLure.get_feasibility_margins``.

The margins are the interior-point drift monitor: the smallest eigenvalue of
each LMI (distance to the constraint boundary) plus each scalar inequality
value, aggregated into ``min_eig``. These need no SDP solver — only an
eigendecomposition of the assembled LMIs — so they run without MOSEK.
"""

import numpy as np
import pytest
import torch

from sysid.models.constrained_rnn import SimpleLure


def _make_feasible_model(s_value: float = 0.05) -> SimpleLure:
    """A small, stable Lure system that satisfies its constraints."""
    m = SimpleLure(nd=1, ne=1, nx=2, nw=1, activation="dzn", custom_params={"learn_L": True})
    with torch.no_grad():
        m.A.data = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=m.A.dtype)
        m.B.data = torch.tensor([[0.1], [0.1]], dtype=m.B.dtype)
        m.B2.data = torch.zeros_like(m.B2)
        m.C2.data = torch.tensor([[0.1, 0.1]], dtype=m.C2.dtype)
        m.D21.data = torch.tensor([[0.1]], dtype=m.D21.dtype)
        m.tau.data = torch.tensor(float(np.log(0.9 / 0.1)))  # alpha = 0.9
        m.s.data = torch.tensor(float(s_value))
    return m


class TestFeasibilityMargins:
    def test_returns_expected_keys(self):
        """Dict exposes the aggregate plus one entry per constraint."""
        m = _make_feasible_model()
        margins = m.get_feasibility_margins()

        assert isinstance(margins, dict)
        assert "min_eig" in margins
        assert "lmi_0_min_eig" in margins  # stability LMI always present
        assert "scalar_0" in margins  # s > 0

    def test_min_eig_is_smallest_entry(self):
        """The aggregate equals the minimum over every per-constraint margin."""
        m = _make_feasible_model()
        margins = m.get_feasibility_margins()

        per_constraint = [v for k, v in margins.items() if k != "min_eig"]
        assert margins["min_eig"] == min(per_constraint)

    def test_lmi_min_eig_matches_independent_eig(self):
        """Per-LMI margin equals an independent eigendecomposition."""
        m = _make_feasible_model()
        margins = m.get_feasibility_margins()

        for i, f_i in enumerate(m.get_lmis()):
            expected = torch.linalg.eigvalsh(f_i()).min().item()
            assert margins[f"lmi_{i}_min_eig"] == pytest.approx(expected, abs=1e-9)

    def test_feasible_model_has_positive_margin(self):
        """A feasible model reports min_eig > 0, consistent with the check."""
        m = _make_feasible_model()
        assert m.check_constraints() is True
        assert m.get_feasibility_margins()["min_eig"] > 0

    def test_infeasible_scalar_gives_negative_margin(self):
        """Violating s > 0 flips min_eig negative, consistent with the check."""
        m = _make_feasible_model()
        with torch.no_grad():
            m.s.data = torch.tensor(-0.1)  # break the s > 0 inequality

        margins = m.get_feasibility_margins()
        assert margins["scalar_0"] < 0
        assert margins["min_eig"] < 0
        assert m.check_constraints() is False

    def test_min_eig_tracks_binding_constraint(self):
        """min_eig follows whichever constraint is closest to its boundary.

        Decreasing ``s`` below every LMI margin (the stability LMI is
        s-independent; the locality LMI only grows more definite as s shrinks)
        makes ``s > 0`` the binding constraint, so ``min_eig`` tracks it toward
        the boundary — the drift signal the monitor exposes, run in reverse.
        """
        m = _make_feasible_model()
        margins = m.get_feasibility_margins()
        lmi_min = min(v for k, v in margins.items() if k.startswith("lmi_"))
        assert lmi_min > 0  # feasible model

        s_small = lmi_min / 2.0  # strictly smaller than every LMI margin
        with torch.no_grad():
            m.s.data = torch.tensor(float(s_small))

        updated = m.get_feasibility_margins()
        assert updated["min_eig"] == updated["scalar_0"]
        assert abs(updated["min_eig"] - s_small) < 1e-6
