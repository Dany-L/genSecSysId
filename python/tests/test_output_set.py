"""Tests for the certified output set (sysid.optimization.output_set).

The set is defined geometrically -- the image of the invariant state ellipsoid
``{x : x^T P^-1 x <= s^2}`` under ``y = S C x`` -- so most of these check the
geometry directly against sampled points rather than re-deriving the algebra.

Two invariants carry the multi-output work:

1. At ``ne == 1`` every reported quantity collapses to the scalar
   ``sigma * s * sqrt(C P C^T)`` the SISO code used, so nothing about existing
   single-output runs moves.
2. ``output_std`` is applied PER CHANNEL. Scaling channel i must move only
   channel i's bound -- the bug this replaced put every channel in channel 0's
   units.
"""

import numpy as np
import pytest

from sysid.optimization.output_set import OutputEllipsoid, output_ellipsoid


def _ellipsoid_points(P, s, n=4000, seed=0):
    """Points on the boundary of ``{x : x^T P^-1 x <= s^2}``."""
    rng = np.random.default_rng(seed)
    nx = P.shape[0]
    v = rng.standard_normal((n, nx))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # x = s * P^{1/2} v
    w, V = np.linalg.eigh(P)
    P_half = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    return s * v @ P_half.T


def _spd(nx, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((nx, nx))
    return M @ M.T + nx * np.eye(nx)


# ── reduces to the SISO formula ───────────────────────────────────────────────
def test_ne_1_matches_the_scalar_formula():
    C = np.array([[0.7, -0.3]])
    P = _spd(2, 1)
    s, sigma = 2.5, 1.7
    e = output_ellipsoid(C, P, s, sigma)

    expected = sigma * s * np.sqrt((C @ P @ C.T).item())
    assert e.y_bar == pytest.approx(expected)
    assert e.y_bar_per_output == pytest.approx([expected])
    assert e.y_bar_min == pytest.approx(e.y_bar_max)
    assert e.Y == pytest.approx(np.array([[1.0 / expected ** 2]]))


def test_scalar_output_std_is_broadcast():
    C, P = np.eye(3, 4), _spd(4, 2)
    a = output_ellipsoid(C, P, 2.0, 1.5)
    b = output_ellipsoid(C, P, 2.0, [1.5, 1.5, 1.5])
    assert a.W == pytest.approx(b.W)


# ── the geometry ──────────────────────────────────────────────────────────────
def test_set_is_exactly_the_image_of_the_state_ellipsoid():
    # ne == nx: M is square, so every boundary point of the state ellipsoid maps
    # to a boundary point of the output set, y^T Y y == 1.
    C, P = np.array([[1.0, 0.4], [-0.2, 0.9]]), _spd(2, 3)
    s, sigma = 1.8, np.array([2.0, 0.5])
    e = output_ellipsoid(C, P, s, sigma)

    y = _ellipsoid_points(P, s) @ (np.diag(sigma) @ C).T
    quad = np.einsum("ni,ij,nj->n", y, e.Y, y)
    assert quad == pytest.approx(np.ones(len(y)), abs=1e-8)


def test_image_stays_inside_the_set_when_outputs_outnumber_states_is_flat():
    # ne < nx: the image is a strict subset, so y^T Y y <= 1 with slack.
    rng = np.random.default_rng(4)
    C, P = rng.standard_normal((2, 5)), _spd(5, 5)
    s, sigma = 1.2, np.array([1.0, 3.0])
    e = output_ellipsoid(C, P, s, sigma)

    y = _ellipsoid_points(P, s) @ (np.diag(sigma) @ C).T
    quad = np.einsum("ni,ij,nj->n", y, e.Y, y)
    assert quad.max() <= 1.0 + 1e-9
    assert quad.max() == pytest.approx(1.0, abs=1e-2)  # the boundary is attained


def test_per_channel_bound_is_the_support_function():
    # ybar_i is the largest |y_i| over the set -- the support function in e_i.
    C, P = np.array([[1.0, 0.4], [-0.2, 0.9]]), _spd(2, 6)
    s, sigma = 1.8, np.array([2.0, 0.5])
    e = output_ellipsoid(C, P, s, sigma)

    y = _ellipsoid_points(P, s, n=20000, seed=7) @ (np.diag(sigma) @ C).T
    sampled = np.abs(y).max(axis=0)
    assert sampled == pytest.approx(e.y_bar_per_output, rel=2e-2)
    # Sampling can only under-estimate a max.
    assert np.all(sampled <= e.y_bar_per_output + 1e-9)


def test_worst_direction_is_the_inscribed_ball_and_bounds_the_box():
    C, P = np.eye(3), _spd(3, 8)
    e = output_ellipsoid(C, P, 2.0, [1.0, 2.0, 4.0])
    # ball inside <= every box half-width <= ball outside
    assert e.y_bar_min <= e.y_bar_per_output.min() + 1e-12
    assert e.y_bar_per_output.max() <= e.y_bar_max + 1e-12
    assert e.y_bar == e.y_bar_min


def test_ball_and_box_coincide_only_for_a_spherical_set():
    e = output_ellipsoid(np.eye(3), np.eye(3), 2.0, 1.0)
    assert e.y_bar_min == pytest.approx(e.y_bar_max)
    assert e.y_bar_per_output == pytest.approx(np.full(3, 2.0))


# ── per-channel scaling ───────────────────────────────────────────────────────
def test_scaling_one_channel_moves_only_that_channel():
    C, P = np.eye(3, 4), _spd(4, 9)
    base = output_ellipsoid(C, P, 1.5, [1.0, 1.0, 1.0])
    scaled = output_ellipsoid(C, P, 1.5, [1.0, 10.0, 1.0])

    assert scaled.y_bar_per_output[0] == pytest.approx(base.y_bar_per_output[0])
    assert scaled.y_bar_per_output[2] == pytest.approx(base.y_bar_per_output[2])
    assert scaled.y_bar_per_output[1] == pytest.approx(10.0 * base.y_bar_per_output[1])


def test_bounds_are_linear_in_s_and_in_sigma():
    C, P = np.eye(2, 3), _spd(3, 10)
    a = output_ellipsoid(C, P, 1.0, [1.0, 2.0])
    b = output_ellipsoid(C, P, 3.0, [1.0, 2.0])
    assert b.y_bar_per_output == pytest.approx(3.0 * a.y_bar_per_output)
    c = output_ellipsoid(C, P, 1.0, [2.0, 4.0])
    assert c.y_bar_per_output == pytest.approx(2.0 * a.y_bar_per_output)


# ── degenerate sets ───────────────────────────────────────────────────────────
def test_more_outputs_than_states_gives_a_flat_set_and_no_Y():
    # 3 outputs read off a 2-state model: the image is a 2-D ellipse sitting in
    # R^3, so it is flat in one direction, W is singular and there is no Y
    # describing it. The per-channel bounds stay meaningful.
    C, P = np.eye(3, 2), _spd(2, 11)
    e = output_ellipsoid(C, P, 2.0, [1.0, 1.0, 1.0])

    assert e.Y is None
    assert e.y_bar_min == pytest.approx(0.0)
    assert e.y_bar_per_output[2] == pytest.approx(0.0)  # the unread channel
    assert np.all(e.y_bar_per_output[:2] > 0.0)
    assert e.covers(0.1) is False


def test_zero_rows_stay_at_zero_rather_than_going_negative():
    # P is only numerically PSD, so a direction the certificate does not extend
    # into can come back at -1e-18 and sqrt would give nan.
    C = np.array([[1.0, 0.0], [0.0, 0.0]])
    e = output_ellipsoid(C, np.eye(2), 1.0, 1.0)
    assert np.all(np.isfinite(e.y_bar_per_output))
    assert e.y_bar_per_output[1] == 0.0


# ── coverage + reporting ──────────────────────────────────────────────────────
def test_covers_is_the_worst_direction_test():
    e = output_ellipsoid(np.eye(2), np.eye(2), 1.0, [1.0, 5.0])
    assert e.y_bar_min == pytest.approx(1.0)
    assert e.covers(0.9) and e.covers(1.0)
    # Reaching 5.0 on channel 2 does not cover 2.0 in every direction.
    assert not e.covers(2.0)


def test_to_dict_is_json_friendly():
    import json

    e = output_ellipsoid(np.eye(2), np.eye(2), 1.0, [1.0, 2.0])
    d = e.to_dict()
    assert json.loads(json.dumps(d))["y_bar_per_output"] == pytest.approx([1.0, 2.0])
    assert isinstance(d["W"], list) and isinstance(d["W"][0], list)
    assert d["Y"] is not None


def test_to_dict_reports_null_Y_for_a_flat_set():
    import json

    d = output_ellipsoid(np.eye(3, 2), np.eye(2), 1.0, 1.0).to_dict()
    assert json.loads(json.dumps(d))["Y"] is None


def test_ne_and_frozen():
    e = output_ellipsoid(np.eye(3, 4), np.eye(4), 1.0, 1.0)
    assert e.ne == 3
    assert isinstance(e, OutputEllipsoid)
    with pytest.raises(Exception):
        e.W = np.eye(3)  # frozen dataclass


# ── input validation ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("sigma", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
def test_wrong_number_of_scales_is_rejected(sigma):
    with pytest.raises(ValueError, match="output_std has"):
        output_ellipsoid(np.eye(3, 4), np.eye(4), 1.0, sigma)


@pytest.mark.parametrize("sigma", [[1.0, 0.0, 1.0], [1.0, -2.0, 1.0], [1.0, np.nan, 1.0]])
def test_non_positive_scales_are_rejected(sigma):
    with pytest.raises(ValueError, match="finite and positive"):
        output_ellipsoid(np.eye(3, 4), np.eye(4), 1.0, sigma)


def test_accepts_the_normalizers_1_1_ne_shape():
    # DataNormalizer stores output_std with keepdims, i.e. (1, 1, ne).
    e = output_ellipsoid(np.eye(2, 3), np.eye(3), 1.0, np.array([[[2.0, 4.0]]]))
    assert e.y_bar_per_output == pytest.approx([2.0, 4.0])


# ── the model path (ne > 1 used to raise before epoch 0) ──────────────────────
class TestMultiOutputModel:
    """`ne > 1` end to end on the model, not just the geometry helper.

    Identity initialization used to do ``(1.0 / output_std) * C_init`` with a
    numpy per-channel vector and a torch tensor, which raised TypeError before
    the first epoch; and post_process guarded ȳ behind ``if self.ne == 1``.
    """

    @staticmethod
    def _model(ne=3, nx=4, nw=8):
        from sysid.models.constrained_rnn import SimpleLure

        return SimpleLure(
            nd=1, ne=ne, nx=nx, nw=nw, activation="dzn",
            custom_params={"learn_L": True, "freeze_alpha": True, "alpha_0": 0.99},
        )

    @staticmethod
    def _normalizer(ne, scales):
        from sysid.data import DataNormalizer

        rng = np.random.default_rng(0)
        norm = DataNormalizer(method="scale_only")
        norm.fit(rng.standard_normal((2, 300, 1)),
                 rng.standard_normal((2, 300, ne)) * np.asarray(scales))
        return norm

    def test_output_std_buffer_is_per_channel(self):
        m = self._model(ne=3)
        assert tuple(m.output_std.shape) == (3,)

    def test_set_output_coverage_level_accepts_a_vector(self):
        m = self._model(ne=3)
        m.set_output_coverage_level(2.0, np.array([[[1.0, 2.0, 3.0]]]))
        assert m.output_std.numpy() == pytest.approx([1.0, 2.0, 3.0])
        assert float(m.y_max) == pytest.approx(2.0)

    def test_set_output_coverage_level_broadcasts_a_scalar(self):
        m = self._model(ne=3)
        m.set_output_coverage_level(1.0, 2.5)
        assert m.output_std.numpy() == pytest.approx([2.5, 2.5, 2.5])

    def test_set_output_coverage_level_rejects_a_wrong_length_vector(self):
        m = self._model(ne=3)
        with pytest.raises(ValueError, match="ne=3"):
            m.set_output_coverage_level(1.0, [1.0, 2.0])

    def test_identity_init_scales_each_row_by_its_own_channel(self):
        # The regression: a (ne,) numpy vector times a (ne, nx) tensor. It used
        # to raise; worse, where it would have broadcast it scaled COLUMNS.
        scales = [1.0, 2.0, 4.0]
        m = self._model(ne=3, nx=4)
        m._init_identity(self._normalizer(3, scales))

        C = m.C.detach().numpy()
        # Row i reads state i, scaled by 1/sigma_i.
        fitted = np.asarray(self._normalizer(3, scales).output_std).reshape(-1)
        for i in range(3):
            assert C[i, i] == pytest.approx(1.0 / fitted[i], rel=1e-6)
        assert np.count_nonzero(C) == 3

    def test_forward_and_output_set_agree_on_shapes(self):
        import torch

        m = self._model(ne=3, nx=4)
        y, (x, w), _ = m(torch.randn(2, 40, 1, dtype=m.P.dtype))
        assert y.shape == (2, 40, 3)
        e = m.certified_output_set()
        assert e.ne == 3 and e.y_bar_per_output.shape == (3,)

    def test_certified_output_set_uses_the_stored_per_channel_scale(self):
        m = self._model(ne=3, nx=4)
        m.set_output_coverage_level(1.0, [1.0, 1.0, 1.0])
        base = m.certified_output_set()
        m.set_output_coverage_level(1.0, [1.0, 10.0, 1.0])
        scaled = m.certified_output_set()
        assert scaled.y_bar_per_output[1] == pytest.approx(
            10.0 * base.y_bar_per_output[1]
        )
        assert scaled.y_bar_per_output[0] == pytest.approx(base.y_bar_per_output[0])

    def test_coverage_ratio_is_defined_for_ne_gt_1(self):
        # It used to call float(self.output_std) on a vector.
        m = self._model(ne=3, nx=4)
        m.set_output_coverage_level(1.0, [1.0, 2.0, 3.0])
        ratio = m.coverage_ratio()
        assert ratio is not None and np.isfinite(ratio)

    def test_flat_set_when_outputs_outnumber_states(self):
        # 3 outputs off 2 states: the certified output set is flat, so there is
        # no Y and the worst direction is 0. nx >= ne is a real requirement for
        # a non-degenerate output set, not a tuning preference.
        m = self._model(ne=3, nx=2)
        m.set_output_coverage_level(1.0, [1.0, 1.0, 1.0])
        e = m.certified_output_set()
        assert e.Y is None
        assert e.y_bar == pytest.approx(0.0)
