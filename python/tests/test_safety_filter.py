"""Tests for LureSystem.forward, with and without the safety input filter.

The safe set is the ellipsoid {x : alpha^2 * x^T X x <= s^2}, and the filter
clamps the input d_k into the interval [-d_max, d_max] with
    d_max = sqrt(max(0, s^2 - alpha^2 * x_k^T X x_k)).
With x0 = 0, X = I, s = 0.5, alpha = 1.0 the budget at step 0 is exactly s,
so we can hand-check what the filter is doing.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from sysid.models.base import LureSystem, LureSystemClass, LureSystemSafe


SAFE_X = torch.eye(2, dtype=torch.float64)
SAFE_S = 0.5
SAFE_ALPHA = 1.0


def _make_lure(safety_filter, requires_grad=False):
    """A minimal Lure system whose dynamics can be traced by hand.

    Setup keeps state contractive (A = 0.5 I) and routes the input through D21
    into the static nonlinearity, so that the lure feedback w_k is sensitive to
    whether d_k was clamped or not.
    """

    def t(values):
        return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)

    A = t([[0.5, 0.0], [0.0, 0.5]])
    B = t([[0.1], [0.0]])
    B2 = t([[0.0], [0.1]])
    C = t([[1.0, 0.0]])
    D = t([[0.0]])
    D12 = t([[0.3]])
    C2 = t([[0.0, 0.0]])
    D21 = t([[1.0]])
    D22 = t([[0.0]])

    sys = LureSystemClass(
        A=A, B=B, B2=B2, C=C, D=D, D12=D12, C2=C2, D21=D21, D22=D22,
        Delta=nn.Hardtanh(min_val=-1.0, max_val=1.0),
    )
    return LureSystemSafe(sys) if safety_filter else LureSystem(sys)


def _zeros_x0(n_batch):
    return torch.zeros(n_batch, 2, 1, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Shape / contract tests
# ---------------------------------------------------------------------------

def test_forward_shapes_without_filter():
    model = _make_lure(safety_filter=False)
    n_batch, N = 3, 5
    d = torch.zeros(n_batch, N, 1, 1, dtype=torch.float64)

    e_hat, (x_full, w_full), d_out = model(d, x0=_zeros_x0(n_batch))
    assert e_hat.shape == (n_batch, N, 1, 1)
    assert x_full.shape == (n_batch, N + 1, 2, 1)
    assert w_full.shape == (n_batch, N, 1, 1)
    # Plain LureSystem applies no filter; inputs come back unchanged.
    assert d_out.shape == d.shape
    torch.testing.assert_close(d_out, d)


def test_forward_shapes_with_filter():
    model = _make_lure(safety_filter=True)
    n_batch, N = 3, 5
    d = torch.zeros(n_batch, N, 1, 1, dtype=torch.float64)

    e_hat, (x_full, w_full), d_filtered = model(
        d, x0=_zeros_x0(n_batch),
        X=SAFE_X, s=SAFE_S, alpha=SAFE_ALPHA,
    )
    assert e_hat.shape == (n_batch, N, 1, 1)
    assert x_full.shape == (n_batch, N + 1, 2, 1)
    assert w_full.shape == (n_batch, N, 1, 1)
    assert d_filtered.shape == (n_batch, N, 1, 1)


def test_filter_requires_safe_set_args():
    """LureSystemSafe.forward demands X/s/alpha; calling without them is a TypeError."""
    model = _make_lure(safety_filter=True)
    d = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
    with pytest.raises(TypeError):
        model(d, x0=_zeros_x0(1))


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------

def test_filter_is_identity_when_input_within_bounds():
    """If |d_k| stays below d_max for the whole trajectory, the filtered
    forward must coincide exactly with the unfiltered one."""
    safe_model = _make_lure(safety_filter=True)
    raw_model = _make_lure(safety_filter=False)

    n_batch, N = 2, 4
    # x0=0 gives d_max=s=0.5; |d|=0.2 leaves plenty of slack and the
    # contractive A keeps the state small for the rest of the trajectory.
    d = 0.2 * torch.ones(n_batch, N, 1, 1, dtype=torch.float64)
    x0 = _zeros_x0(n_batch)

    e_safe, (x_safe, w_safe), _ = safe_model(
        d, x0=x0, X=SAFE_X, s=SAFE_S, alpha=SAFE_ALPHA,
    )
    e_raw, (x_raw, w_raw), _ = raw_model(d, x0=x0)

    torch.testing.assert_close(e_safe, e_raw)
    torch.testing.assert_close(x_safe, x_raw)
    torch.testing.assert_close(w_safe, w_raw)


def test_filter_clamps_positive_oversized_input():
    """Single step from x0=0: d_max ≈ s. A large positive d must be clamped
    to +s, so the filtered run should match an unfiltered run fed +s.

    The filter intentionally subtracts a small eps from d_max for numerical
    safety (gradient through sqrt near the boundary), so we compare with a
    matching ``atol`` rather than bit-exact equality.
    """
    safe_model = _make_lure(safety_filter=True)
    raw_model = _make_lure(safety_filter=False)

    n_batch, N = 1, 1
    x0 = _zeros_x0(n_batch)
    d_big = 5.0 * torch.ones(n_batch, N, 1, 1, dtype=torch.float64)
    d_clamped = SAFE_S * torch.ones(n_batch, N, 1, 1, dtype=torch.float64)

    e_safe, (x_safe, w_safe), _ = safe_model(
        d_big, x0=x0, X=SAFE_X, s=SAFE_S, alpha=SAFE_ALPHA,
    )
    e_ref, (x_ref, w_ref), _ = raw_model(d_clamped, x0=x0)

    torch.testing.assert_close(e_safe, e_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_safe, x_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w_safe, w_ref, atol=1e-5, rtol=1e-5)

    # And filtering must actually have changed something vs feeding d_big raw.
    e_unfiltered, _, _ = raw_model(d_big, x0=x0)
    assert not torch.allclose(e_safe, e_unfiltered)


def test_filter_clamps_negative_oversized_input():
    """Same as the positive case but with d well below -d_max."""
    safe_model = _make_lure(safety_filter=True)
    raw_model = _make_lure(safety_filter=False)

    n_batch, N = 1, 1
    x0 = _zeros_x0(n_batch)
    d_big = -5.0 * torch.ones(n_batch, N, 1, 1, dtype=torch.float64)
    d_clamped = -SAFE_S * torch.ones(n_batch, N, 1, 1, dtype=torch.float64)

    e_safe, _, _ = safe_model(
        d_big, x0=x0, X=SAFE_X, s=SAFE_S, alpha=SAFE_ALPHA,
    )
    e_ref, _, _ = raw_model(d_clamped, x0=x0)

    torch.testing.assert_close(e_safe, e_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.xfail(
    strict=True,
    reason="Warmup window intentionally disabled in d684ddf: the filter now "
    "clamps from step 0 and ignores warmup_steps. Re-enable the "
    "'if k < warmup_steps' bypass in LureSystemSafe.clamp_step to restore this "
    "(and remove this marker) when the window comes back.",
)
def test_warmup_steps_disables_filter():
    """While k < warmup_steps the filter must be bypassed entirely."""
    safe_model = _make_lure(safety_filter=True)
    raw_model = _make_lure(safety_filter=False)

    n_batch, N = 1, 2
    x0 = _zeros_x0(n_batch)
    d_big = 5.0 * torch.ones(n_batch, N, 1, 1, dtype=torch.float64)

    e_safe, _, _ = safe_model(
        d_big, x0=x0,
        X=SAFE_X, s=SAFE_S, alpha=SAFE_ALPHA,
        warmup_steps=N,  # filter never engages
    )
    e_raw, _, _ = raw_model(d_big, x0=x0)

    torch.testing.assert_close(e_safe, e_raw)


def test_backward_through_filter_produces_finite_gradients():
    """The clamped path must not poison gradients with NaN/Inf."""
    model = _make_lure(safety_filter=True, requires_grad=True)

    n_batch, N = 2, 3
    # d well outside [-s, s] forces the filter to clamp at every step.
    d = 5.0 * torch.ones(n_batch, N, 1, 1, dtype=torch.float64)
    x0 = _zeros_x0(n_batch)

    e_hat, _, _ = model(
        d, x0=x0, X=SAFE_X, s=SAFE_S, alpha=SAFE_ALPHA,
    )
    loss = (e_hat ** 2).mean()
    loss.backward()

    grads = {name: t.grad for name, t in [
        ("A", model.A), ("B", model.B), ("B2", model.B2),
        ("C", model.C), ("D", model.D), ("D12", model.D12),
        ("C2", model.C2), ("D21", model.D21),
    ]}
    for name, g in grads.items():
        assert g is not None, f"{name} received no gradient"
        assert torch.isfinite(g).all(), f"{name} gradient has NaN/Inf: {g}"


# ---------------------------------------------------------------------------
# End-to-end test on a real Duffing trajectory
# ---------------------------------------------------------------------------

DUFFING_DATA = Path(os.path.expanduser("~/genSecSysId-Data/data/Duffing/id"))
# Dedicated test config so this test is independent of the user's working
# config (which they toggle while developing).
DUFFING_CONFIG = Path(__file__).parent / "resoruces" / "safety_filter_duffing.yaml"
PLOT_DIR = Path(__file__).parent / "outputs"


@pytest.mark.skipif(
    not DUFFING_DATA.exists() or not DUFFING_CONFIG.exists(),
    reason="Duffing dataset or config not available on this machine",
)
def test_safety_filter_clamps_real_duffing_trajectory():
    """Initialize the CRNN exactly as ``train.py`` does (identity init, real
    data-driven s), feed one Duffing trajectory through it, and verify the
    safety filter actually clamps. A plot of the original vs. filtered input
    is saved alongside the test for visual inspection.

    Background: after ``initialize_parameters``, the MinTrProb sweep sets ``s``
    so that the training data fits inside the safe set — meaning training inputs
    never trigger the filter. We therefore scale the input to ``2×s`` to
    guarantee clamping at ``x0=0`` (where ``d_max = s``), independently of the
    chosen ``s`` value.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sysid.config import Config
    from sysid.data import create_dataloaders
    from sysid.data.direct_loader import load_split_data
    from sysid.models import create_model

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    np.random.seed(42)
    try:
        config = Config.from_yaml(str(DUFFING_CONFIG))

        # 1. Load data exactly like train.py
        result = load_split_data(
            data_dir=str(DUFFING_DATA),
            input_col=config.data.input_col,
            output_col=config.data.output_col,
            state_col=config.data.state_col,
            pattern=config.data.pattern,
            load_test=False,
        )
        (train_inputs, train_outputs, val_inputs, val_outputs,
         _, _, train_states, val_states, _,
         _, _, _, _, _, _, _, _, _) = result

        # 2. Compute delta / max_norm_x0 the same way train.py does
        max_norm_x0 = float(np.max(np.linalg.norm(train_states[:, 0, :], 2, axis=1)))
        delta = np.max(np.abs(train_inputs), axis=(0, 1))

        # 3. Build dataloaders to get the normalizer
        *_, normalizer = create_dataloaders(
            train_inputs=train_inputs, train_outputs=train_outputs,
            train_states=train_states,
            val_inputs=val_inputs, val_outputs=val_outputs,
            val_states=val_states,
            batch_size=config.data.batch_size,
            sequence_length=config.data.train_sequence_length,
            sequence_stride=getattr(config.data, "sequence_stride", None),
            normalize=config.data.normalize,
            normalization_method=config.data.normalization_method,
            shuffle=False,
            num_workers=0,
        )
        if normalizer is not None:
            delta = normalizer.transform_inputs(delta).squeeze()

        # 4. Build the CRNN and run the same data-driven init
        model = create_model(config, delta, max_norm_x0)
        model.initialize_parameters(
            train_inputs, train_states, train_outputs,
            init_config=config.model.initialization,
            normalizer=normalizer,
        )
        model.eval()
        assert isinstance(model.lure, LureSystemSafe), (
            "Config (model_type: crnn_safe) should produce a safety-filtered model"
        )

        # 5. Pick a real Duffing input trajectory (shape: N × nd after normalization).
        TRAJ_IDX = 1
        seq = train_inputs[TRAJ_IDX]
        if normalizer is not None:
            seq = normalizer.transform_inputs(seq[None, ...]).squeeze(0)

        d_raw = torch.as_tensor(seq, dtype=torch.float64)  # (N, nd)

        # 6. The SDP sets s so that training data fits the safe set, so
        # unscaled training inputs won't trigger the filter.  Scale to 2×s so
        # the peak input always exceeds d_max (= s at x0 = 0), making the
        # filter engage regardless of the SDP-chosen s.
        with torch.no_grad():
            s = model.s.detach()
            X = torch.linalg.inv(model.P)
            alpha = (1.0 / (1.0 + torch.exp(-model.tau))).detach()

        s_val = float(s)
        scale_factor = s_val * 2.0 / float(d_raw.abs().max())
        d_scaled = d_raw * scale_factor  # (N, nd)

        # SimpleLureSafe.forward expects (B, N, nd) — no trailing dimension.
        d_3d = d_scaled.unsqueeze(0)  # (1, N, nd)
        x0 = torch.zeros(1, model.nx, 1, dtype=torch.float64)

        # 7. Filtered run via SimpleLureSafe.forward; unfiltered via forward_unfiltered.
        with torch.no_grad():
            _, (x_safe, _), d_applied = model(d_3d, x0=x0, warmup_steps=0)
            _, (x_unsafe, _), _ = model.forward_unfiltered(d_3d, x0=x0)

        # 8. Assert clamping engaged at least once.
        diff = (d_3d - d_applied).abs()
        n_clamped = int((diff > 1e-9).sum().item())
        assert n_clamped > 0, (
            "Safety filter did not engage on this trajectory. "
            f"max|d - d_applied| = {diff.max().item():.3e}, s={s_val:.3e}"
        )

        # 9. Plots in the style of the duffing_analysis notebook.
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        alpha_val = float(alpha)
        X_np = X.cpu().numpy()
        # (N, nd) and (N+1, nx) — squeeze out the batch dimension
        d_orig_np = d_3d.squeeze(0).cpu().numpy()
        d_safe_np = d_applied.squeeze(0).cpu().numpy()
        x_safe_np = x_safe.squeeze(0).cpu().numpy()
        x_unsafe_np = x_unsafe.squeeze(0).cpu().numpy()

        # --- Plot 1: original vs filtered input over time ---------------------
        TS = 0.1  # sampling time used in the notebook
        N = d_orig_np.shape[0]
        t_axis = np.arange(N) * TS

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_axis, d_safe_np[:, 0], label="u filtered")
        ax.plot(t_axis, d_orig_np[:, 0], label="u")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("input")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        input_plot_path = PLOT_DIR / "safety_filter_duffing_input.png"
        fig.savefig(input_plot_path, dpi=120)
        plt.close(fig)

        # --- Plot 2: phase-space safe vs unsafe + ellipsoid x^T X / s^2 x = 1 -
        # Bounds are driven by the safe trajectory + the ellipsoid, since the
        # unsafe trajectory diverges several orders of magnitude. We truncate
        # the unsafe line at the first sample that leaves the view.
        ellipse_axes = s_val / np.sqrt(np.linalg.eigvalsh(X_np))
        bound = max(np.abs(x_safe_np).max(), float(ellipse_axes.max())) * 1.3

        n_grid = 300
        g1 = np.linspace(-bound, bound, n_grid)
        g2 = np.linspace(-bound, bound, n_grid)
        G1, G2 = np.meshgrid(g1, g2)
        X_norm = X_np / (s_val ** 2)
        quad_form = (
            X_norm[0, 0] * G1 * G1
            + (X_norm[0, 1] + X_norm[1, 0]) * G1 * G2
            + X_norm[1, 1] * G2 * G2
        )

        unsafe_in_view = (np.abs(x_unsafe_np[:, 0]) <= bound) & (
            np.abs(x_unsafe_np[:, 1]) <= bound
        )
        if unsafe_in_view.all():
            unsafe_end = x_unsafe_np.shape[0]
        else:
            unsafe_end = int(np.argmax(~unsafe_in_view)) + 1  # one step past

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.contourf(G1, G2, quad_form, levels=[0, 1], colors=["lightblue"], alpha=0.6)
        contour = ax.contour(G1, G2, quad_form, levels=[1], colors=["blue"], linewidths=2.5)
        ax.clabel(contour, inline=True, fontsize=10, fmt="$x^T X/s^2\\, x = 1$")
        ax.plot(x_safe_np[:, 0], x_safe_np[:, 1], label="safe")
        ax.plot(x_unsafe_np[:unsafe_end, 0], x_unsafe_np[:unsafe_end, 1], label="unsafe")
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        phase_plot_path = PLOT_DIR / "safety_filter_duffing_phase_space.png"
        fig.savefig(phase_plot_path, dpi=120)
        plt.close(fig)

        print(
            f"\n[safety filter] plots written to:\n"
            f"  {input_plot_path}\n"
            f"  {phase_plot_path}\n"
            f"  s={s_val:.3f}, alpha={alpha_val:.4f}, "
            f"clamped {n_clamped}/{d_3d.numel()} samples"
        )
    finally:
        torch.set_default_dtype(prev_dtype)
