"""Tests for the ship-rolling simulation script.

The script lives under notebooks/ (not the installed package), so it is loaded
by file path. We never render a full GIF here: we check the pure helpers and
that the figure builds (without a run id, i.e. true dynamics only) with the
right number of columns. The animation object is attached but not rendered.
"""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

REPO_PY = Path(__file__).resolve().parents[1]
SCRIPT = REPO_PY / "notebooks" / "duffing" / "ship_rolling_simulation.py"

# Each column contributes 4 axes: ship, time-series, its input twin, phase.
AXES_PER_COLUMN = 4


@pytest.fixture(scope="module")
def srs():
    """Import the script module by path."""
    spec = importlib.util.spec_from_file_location("ship_rolling_simulation", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path, n, q_scale=0.3):
    """Write a tiny Duffing CSV (columns u, q, q_dot) with n rows."""
    t = np.arange(n) * 0.05
    df = {
        "u": 0.1 * np.sin(t),
        "q": q_scale * np.sin(t),
        "q_dot": q_scale * np.cos(t),
    }
    import pandas as pd

    pd.DataFrame(df).to_csv(path, index=False)
    return path


def test_pad_1d(srs):
    """pad_1d truncates when longer and repeats the last value when shorter."""
    arr = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(srs.pad_1d(arr, 2), [1.0, 2.0])
    assert np.array_equal(srs.pad_1d(arr, 5), [1.0, 2.0, 3.0, 3.0, 3.0])
    assert np.array_equal(srs.pad_1d(arr, 3), arr)


def test_pad_zeros(srs):
    """pad_zeros truncates when longer and pads with trailing zeros when shorter."""
    arr = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(srs.pad_zeros(arr, 2), [1.0, 2.0])
    assert np.array_equal(srs.pad_zeros(arr, 5), [1.0, 2.0, 3.0, 0.0, 0.0])
    assert np.array_equal(srs.pad_zeros(arr, 3), arr)


def test_main_zero_pads_short_input(srs, tmp_path):
    """A shorter (diverging) trajectory's input goes to zero, not held frozen."""
    stable = _write_csv(tmp_path / "s.csv", n=100)
    unstable = _write_csv(tmp_path / "u.csv", n=40)  # shorter -> zero-padded tail
    fig = srs.main([
        "--stable-csv", str(stable), "--unstable-csv", str(unstable),
        "--no-show", "--fps", "8",
    ])
    # Unstable column (index 1); its input trace is on the twin axis (4th axis).
    twin = fig.axes[1 * AXES_PER_COLUMN + 3]
    u_tail = twin.lines[0].get_ydata()[-1]
    assert u_tail == pytest.approx(0.0)  # forcing off after the CSV ends
    plt.close(fig)


def test_frame_times_speed_shortens(srs):
    """A speed of s yields ~1/s as many frames; fps fixes smoothness."""
    slow = srs.frame_times(T_total=10.0, fps=30, speed=1.0)
    fast = srs.frame_times(T_total=10.0, fps=30, speed=4.0)
    assert len(slow) == 300
    assert len(fast) == 75  # 4x fewer frames -> ~4x faster / shorter
    # More fps at the same speed -> smoother (more frames).
    assert len(srs.frame_times(10.0, 60, 1.0)) == 600


def test_frame_times_rejects_bad_args(srs):
    for bad in ({"fps": 0}, {"fps": -1}, {"speed": 0}, {"speed": -2}):
        kwargs = {"T_total": 5.0, "fps": 30, "speed": 1.0, **bad}
        with pytest.raises(ValueError):
            srs.frame_times(**kwargs)


def test_window_slice(srs):
    """window_slice trims to a clip length, an offset range, or the full tail."""
    # Sequence length: first 500 of 4000.
    assert srs.window_slice(4000, 0, 500) == slice(0, 500)
    # Range: [1000, 1500).
    assert srs.window_slice(4000, 1000, 500) == slice(1000, 1500)
    # n_steps=None keeps everything from start to the end.
    assert srs.window_slice(4000, 1000, None) == slice(1000, 4000)


def test_window_slice_rejects_bad_args(srs):
    with pytest.raises(ValueError):
        srs.window_slice(100, -1, 10)  # negative start
    with pytest.raises(ValueError):
        srs.window_slice(100, 100, 10)  # start beyond length
    with pytest.raises(ValueError):
        srs.window_slice(100, 0, 0)  # non-positive n_steps


def test_ensure_even_frame_size(srs):
    """The figure is snapped so width*dpi and height*dpi are both even.

    H.264 rejects odd dimensions; we fix it by resizing the figure (not by
    padding the video, which makes it roll vertically).
    """
    fig = plt.figure(figsize=(10.0, 11.5))  # 11.5*110 = 1265 px -> odd
    dpi = 110
    srs.ensure_even_frame_size(fig, dpi)
    w_in, h_in = fig.get_size_inches()
    assert int(round(w_in * dpi)) % 2 == 0
    assert int(round(h_in * dpi)) % 2 == 0
    assert fig.dpi == dpi
    plt.close(fig)


def test_main_true_dynamics_only(srs, tmp_path):
    """Without --run-id, only the two true-dynamics columns are built."""
    stable = _write_csv(tmp_path / "stable.csv", n=20)
    unstable = _write_csv(tmp_path / "unstable.csv", n=12)  # shorter -> exercises pad
    fig = srs.main([
        "--stable-csv", str(stable),
        "--unstable-csv", str(unstable),
        "--no-show",
        "--fps", "10",
        "--speed", "4",
    ])
    # 2 columns (true stable + true unstable), no learned model.
    assert len(fig.axes) == 2 * AXES_PER_COLUMN
    # Figure renders without raising.
    out = tmp_path / "frame.png"
    fig.savefig(out, dpi=50)
    assert out.exists()
    plt.close(fig)


def test_main_windows_trajectory(srs, tmp_path):
    """--start-step / --n-steps trim the trajectory to the requested clip."""
    stable = _write_csv(tmp_path / "stable.csv", n=400)
    unstable = _write_csv(tmp_path / "unstable.csv", n=400)
    fig = srs.main([
        "--stable-csv", str(stable),
        "--unstable-csv", str(unstable),
        "--no-show", "--fps", "10",
        "--start-step", "50", "--n-steps", "100",
    ])
    # The clip is 100 samples at Ts=DUFFING_TS; the time axis reflects that.
    n_max = 100
    assert fig.axes[1].get_xlim()[1] == pytest.approx(n_max * srs.DUFFING_TS)
    plt.close(fig)


def test_main_out_of_range_start_errors(srs, tmp_path):
    """A start-step past the end of the CSV is a clean argparse error."""
    stable = _write_csv(tmp_path / "stable.csv", n=30)
    unstable = _write_csv(tmp_path / "unstable.csv", n=30)
    with pytest.raises(SystemExit):
        srs.main([
            "--stable-csv", str(stable),
            "--unstable-csv", str(unstable),
            "--no-show", "--start-step", "999",
        ])


def _anim_column(srs, n=6, tag="c", with_cert=False):
    """A minimal column dict with the fields build_subplot consumes."""
    q = np.linspace(0.0, 0.2, n)
    dq = np.linspace(0.0, 0.1, n)
    u = np.zeros(n)
    col = {
        "tag": tag, "label": "L", "color": "#334455",
        "u": u, "q": q, "dq": dq,
        "phi": 0.5 * q, "dphi": 0.5 * dq,
        "u_anim": u, "phi_anim": 0.5 * q, "dphi_anim": 0.5 * dq,
    }
    if with_cert:
        col["cert"] = {"P": np.eye(2) * 8e-4, "s": 8.0, "L": np.eye(2) * 10.0}
    return col


def _anim_ctx(n=6):
    return {
        "t": np.arange(n) * 0.05, "T_total": n * 0.05,
        "t_anim": np.arange(n) * 0.05, "phi_v": np.deg2rad(60),
        "phi_v_deg": 60.0, "boundary_label": None, "trail_window": 3,
    }


def test_build_subplot_each_kind_updates(srs):
    """Every subplot kind builds on a standalone axis and advances frames."""
    n = 6
    col = _anim_column(srs, n=n, with_cert=True)
    ctx = _anim_ctx(n)
    for kind in srs.SUBPLOT_KINDS:
        fig, ax = plt.subplots()
        update = srs.build_subplot(kind, ax, col, ctx)
        update(0)
        update(n - 1)  # no error at first/last frame
        plt.close(fig)


def test_build_subplot_rejects_unknown_kind(srs):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        srs.build_subplot("nope", ax, _anim_column(srs), _anim_ctx())
    plt.close(fig)


def test_main_split_dir_writes_one_video_per_panel(srs, tmp_path):
    """--split-dir writes 3 videos (ship/timeseries/phase) per column."""
    stable = _write_csv(tmp_path / "s.csv", n=15)
    unstable = _write_csv(tmp_path / "u.csv", n=15)
    outdir = tmp_path / "panels"
    srs.main([
        "--stable-csv", str(stable), "--unstable-csv", str(unstable),
        "--no-show", "--fps", "5", "--split-dir", str(outdir),
        "--split-format", "gif",
    ])
    produced = sorted(p.name for p in outdir.glob("*.gif"))
    expected = sorted(f"{tag}_{kind}.gif"
                      for tag in ("true_stable", "true_unstable")
                      for kind in srs.SUBPLOT_KINDS)
    assert produced == expected
    plt.close("all")


class _DummyModel:
    def eval(self):
        return self


def _patch_resolvers(srs, monkeypatch, used):
    """Stub both artifact resolvers + model/normalizer loading."""
    def fake_mlflow(run_id, uri):
        used["mlflow"] = (run_id, uri)
        return ("CFG", "model.pt", "norm.json", None)

    def fake_local(run_id, data_root=None):
        used["local"] = (run_id, data_root)
        return ("CFG", "model.pt", "norm.json", None)

    monkeypatch.setattr(srs, "resolve_run_artifacts_mlflow", fake_mlflow)
    monkeypatch.setattr(srs, "resolve_run_artifacts", fake_local)
    monkeypatch.setattr(srs, "load_model",
                        lambda path, config, device="cpu": _DummyModel())
    monkeypatch.setattr(srs.DataNormalizer, "load",
                        staticmethod(lambda path: "NORM"))


def test_load_learned_model_uses_mlflow_when_uri_set(srs, monkeypatch):
    """A --mlflow-uri routes to the remote resolver, not the local one."""
    used = {}
    _patch_resolvers(srs, monkeypatch, used)
    _, norm = srs.load_learned_model("rid", mlflow_uri="http://server/")
    assert used.get("mlflow") == ("rid", "http://server/")
    assert "local" not in used
    assert norm == "NORM"


def test_load_learned_model_uses_local_by_default(srs, monkeypatch):
    """Without a --mlflow-uri the local data-root resolver is used."""
    used = {}
    _patch_resolvers(srs, monkeypatch, used)
    srs.load_learned_model("rid", data_root="/some/root")
    assert used.get("local") == ("rid", "/some/root")
    assert "mlflow" not in used


# --- certified capsize bound (y_bar) + safe-set overlay ---------------------

import torch  # noqa: E402


class _FakeCertModel:
    """Minimal constrained-model stand-in exposing what extract_certificate reads."""

    def __init__(self, y_bar_n, P, s, L, C):
        self.P = torch.tensor(np.asarray(P, dtype=float))
        self.s = torch.tensor(float(s))
        self.L = torch.tensor(np.asarray(L, dtype=float))
        self.C = torch.tensor(np.asarray(C, dtype=float))
        self._ybn = y_bar_n

    def post_process(self):
        return {"success": True,
                "summary": {"optimized": {"y_bar_n": self._ybn}}}


class _NormStub:
    def __init__(self, output_std):
        self.output_std = np.array([[[output_std]]])


def test_extract_certificate_reads_ybar(srs):
    """y_bar is the normalized bound times the output scale; P/s/L are returned."""
    P = [[8e-4, 0.0], [0.0, 8e-4]]
    m = _FakeCertModel(1.5, P, 8.0, np.zeros((3, 2)), [[6.0, 0.0]])
    cert = srs.extract_certificate(m, _NormStub(0.1))
    assert cert["y_bar"] == pytest.approx(1.5 * 0.1)
    assert cert["s"] == pytest.approx(8.0)
    assert cert["P"].shape == (2, 2)
    assert cert["y_bar_reliable"] is True  # came straight from a feasible SDP


def test_extract_certificate_fallback_closed_form(srs):
    """When y_bar_n <= 0 (infeasible SDP), fall back to s*sqrt(C P C^T)."""
    P = np.array([[4e-4, 0.0], [0.0, 4e-4]])
    s, C = 10.0, np.array([[5.0, 0.0]])
    m = _FakeCertModel(-1.0, P, s, np.zeros((2, 2)), C)
    cert = srs.extract_certificate(m, _NormStub(0.2))
    expected = s * np.sqrt(float((C @ P @ C.T).ravel()[0])) * 0.2
    assert cert["y_bar"] == pytest.approx(expected)
    # A fallback bound scales with s (possibly pinned at its cap) -> unreliable.
    assert cert["y_bar_reliable"] is False


def test_choose_capsize_scale_prefers_ybar_override(srs):
    """An explicit --y-bar wins over any certificate bound."""
    cert = {"y_bar": 3.0, "s": 5.0, "y_bar_reliable": True}
    q_cap, label = srs.choose_capsize_scale(0.5, cert)
    assert q_cap == pytest.approx(0.5)
    assert "0.50" in label


def test_choose_capsize_scale_uses_reliable_cert(srs):
    """A reliable certificate bound sets the capsize scale and its label."""
    cert = {"y_bar": 0.8, "s": 5.0, "y_bar_reliable": True}
    q_cap, label = srs.choose_capsize_scale(None, cert)
    assert q_cap == pytest.approx(0.8)
    assert "0.80" in label


def test_choose_capsize_scale_skips_degenerate_cert(srs):
    """A degenerate (infeasible-SDP) bound is ignored: fall back to q=1.

    Regression for the run whose repair pinned s at ~1000, giving a vacuous
    y_bar (~282) that rescaled every trajectory to ~0° (an "empty" animation).
    """
    cert = {"y_bar": 281.7, "s": 1000.0, "y_bar_reliable": False}
    q_cap, label = srs.choose_capsize_scale(None, cert)
    assert q_cap == pytest.approx(1.0)
    assert label is None  # keeps the default ±phi_v label


def test_choose_capsize_scale_no_cert_uses_hilltop(srs):
    """Without a certificate (or a non-positive bound) the hilltop q=1 is used."""
    assert srs.choose_capsize_scale(None, None) == (1.0, None)
    cert = {"y_bar": 0.0, "s": 5.0, "y_bar_reliable": True}
    assert srs.choose_capsize_scale(None, cert) == (1.0, None)


def test_extract_certificate_none_without_postprocess(srs):
    class M:  # plain RNN/LSTM/GRU have no post_process
        pass
    assert srs.extract_certificate(M(), None) is None


def test_extract_certificate_none_on_failed_sdp(srs):
    class M:  # certificate SDP failed outright -> no "summary"
        def post_process(self):
            return {"success": False, "status": "max_s_sdp_failed"}
    assert srs.extract_certificate(M(), None) is None


def test_draw_certificate_region_none_is_noop(srs):
    fig, ax = plt.subplots()
    srs.draw_certificate_region(ax, None, np.deg2rad(60))
    assert ax.get_legend() is None
    assert len(ax.lines) == 0
    plt.close(fig)


def test_draw_certificate_region_draws_ellipse_and_polytope(srs):
    fig, ax = plt.subplots()
    cert = {"P": np.array([[8e-4, 0.0], [0.0, 8e-4]]), "s": 8.0,
            "L": np.array([[10.0, 0.0], [0.0, 10.0]])}
    srs.draw_certificate_region(ax, cert, np.deg2rad(60))
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("safe set" in l for l in labels)
    assert any("input constr" in l for l in labels)
    plt.close(fig)


def test_main_ybar_rescales_capsize_boundary(srs, tmp_path):
    """--y-bar labels the capsize boundary with the certified bound value."""
    stable = _write_csv(tmp_path / "s.csv", n=80)
    unstable = _write_csv(tmp_path / "u.csv", n=80)
    fig = srs.main([
        "--stable-csv", str(stable), "--unstable-csv", str(unstable),
        "--no-show", "--fps", "8", "--y-bar", "0.5",
    ])
    labels = [t.get_text() for ax in fig.axes if ax.get_legend()
              for t in ax.get_legend().get_texts()]
    assert any("0.50" in l for l in labels)
    plt.close(fig)
