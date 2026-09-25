"""Unconstrained RNN baselines (LSTM / GRU / RNN) through the CRNN pipeline.

The trainer, evaluator and scripts call every model as
``model(d, x0, warmup_steps=...) -> (e_hat, (x, w), d)`` and ask it for
``get_regularization_input(..., return_c=True)``. The baselines used to return
a bare ``e_hat`` and had no ``c``, so they crashed on the first batch. These
tests pin the interface and run the scripts end to end on an LSTM config.

No MOSEK needed (no certificate), so unlike test_scripts_smoke.py this module
also runs on CI.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml

from sysid.config import Config
from sysid.models import GRU, LSTM, SimpleRNN, create_model, load_model
from tests.test_scripts_smoke import _run_script, _write_csvs

REPO_PY = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_PY / "configs" / "lstm_duffing.yaml"

BASELINES = [SimpleRNN, LSTM, GRU]


# ── model interface ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", BASELINES)
def test_forward_matches_trainer_call_convention(cls):
    model = cls(input_size=1, hidden_size=8, output_size=2, num_layers=2)
    d = torch.randn(3, 20, 1)

    e_hat, (x, w), d_out = model(d, None, warmup_steps=5)

    assert e_hat.shape == (3, 20, 2)
    assert x.shape == (3, 20, 8)  # last layer's hidden sequence
    assert w is None
    assert d_out is d
    assert model.check_constraints()


@pytest.mark.parametrize("cls", BASELINES)
def test_physical_x0_is_ignored(cls):
    """A black-box hidden state has no physical meaning; washout handles x0."""
    torch.manual_seed(0)
    model = cls(input_size=1, hidden_size=8, output_size=1)
    d = torch.randn(2, 20, 1)

    e_none, _, _ = model(d, None)
    e_x0, _, _ = model(d, x0=torch.randn(2, 1, 2))

    assert torch.equal(e_none, e_x0)


@pytest.mark.parametrize("cls", BASELINES)
def test_no_constraint_trajectory(cls):
    """Evaluator asks for c; a baseline has none and must say so with None."""
    model = cls(input_size=1, hidden_size=8, output_size=1)
    d = torch.randn(2, 20, 1)
    _, (x, _), _ = model(d)

    loss, c = model.get_regularization_input(d, x, return_c=True)
    assert c is None
    assert loss.item() == 0.0
    assert model.get_regularization_input(d, x).item() == 0.0


def test_state_dict_keys_unchanged():
    """Checkpoints saved before the refactor use lstm./gru./rnn. prefixes."""
    assert any(k.startswith("lstm.") for k in LSTM(1, 4, 1).state_dict())
    assert any(k.startswith("gru.") for k in GRU(1, 4, 1).state_dict())
    assert any(k.startswith("rnn.") for k in SimpleRNN(1, 4, 1).state_dict())


# ── factory / template config ─────────────────────────────────────────────────


def test_factory_reads_nw():
    """nw is the width key for every model type, baselines included."""
    config = Config.from_yaml(str(TEMPLATE))
    assert config.model.nw == 32

    model = create_model(config)

    assert isinstance(model, LSTM)
    assert model.hidden_size == 32
    assert model.input_size == len(config.data.input_col)
    assert model.output_size == len(config.data.output_col)


def test_factory_requires_nw():
    """Without nw torch would fail deep inside nn.LSTM with a NoneType error."""
    config = Config.from_yaml(str(TEMPLATE))
    config.model.nw = None
    with pytest.raises(ValueError, match="model.nw"):
        create_model(config)


def test_template_has_no_certificate_terms():
    cfg = yaml.safe_load(TEMPLATE.read_text())
    assert cfg["model"]["model_type"] == "lstm"
    assert cfg["training"]["use_custom_regularization"] is False
    assert cfg["training"]["regularization_weight"] == 0.0


# ── scripts end to end ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def lstm_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("lstm_root")
    data_dir = root / "data" / "SmokeData"
    _write_csvs(data_dir / "train", n_files=4, n_steps=200, seed=0)
    _write_csvs(data_dir / "validation", n_files=2, n_steps=200, seed=1)
    _write_csvs(data_dir / "test", n_files=2, n_steps=200, seed=2)
    return root


@pytest.fixture(scope="module")
def lstm_config(lstm_root):
    """The shipped template, shrunk and pointed at the synthetic data."""
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg["data"].update(
        train_path=str(lstm_root / "data" / "SmokeData"),
        batch_size=2,
        train_sequence_length=50,
        sequence_stride=50,
        use_diverging_trajectories=False,
    )
    cfg["model"]["nw"] = 8
    cfg["training"].update(max_epochs=2, warmup_steps=10)
    cfg["mlflow"].update(tracking_uri=f"file:{lstm_root}/mlruns", experiment_name="lstm-smoke")
    cfg["root_dir"] = str(lstm_root)
    path = lstm_root / "lstm_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _train(root, config, tmp, *extra):
    run_id_out = tmp / "run_id.txt"
    _run_script("train.py", "--config", config, "--run-id-out", run_id_out, *extra, cwd=root)
    run_id = run_id_out.read_text().strip()
    assert run_id, "train.py did not write a run_id"
    return run_id


@pytest.fixture(scope="module")
def lstm_run_id(lstm_root, lstm_config, tmp_path_factory):
    return _train(lstm_root, lstm_config, tmp_path_factory.mktemp("lstm_train"))


def test_train_lstm(lstm_root, lstm_run_id):
    run_dir = lstm_root / "models" / "lstm" / lstm_run_id
    assert (run_dir / "best_model.pt").exists()
    assert (lstm_root / "outputs" / "lstm" / lstm_run_id / "config.yaml").exists()

    config = Config.from_yaml(str(lstm_root / "outputs" / "lstm" / lstm_run_id / "config.yaml"))
    model = load_model(str(run_dir / "best_model.pt"), config)
    assert isinstance(model, LSTM) and model.hidden_size == 8


def test_evaluate_lstm(lstm_root, lstm_run_id):
    _run_script(
        "evaluate.py",
        "--run-id", lstm_run_id,
        "--data-root", lstm_root,
        "--test-data", lstm_root / "data" / "SmokeData",
        cwd=lstm_root,
    )
    eval_dir = lstm_root / "outputs" / "lstm" / lstm_run_id / "evaluation"
    assert any(eval_dir.rglob("evaluation_results.json"))


def test_compare_lstm(lstm_root, lstm_config, lstm_run_id, tmp_path_factory, tmp_path):
    second = _train(lstm_root, lstm_config, tmp_path_factory.mktemp("lstm_train2"), "--seed", "7")
    output_dir = tmp_path / "comparison"
    _run_script(
        "compare.py",
        "--run-ids", lstm_run_id, second,
        "--data-root", lstm_root,
        "--output-dir", output_dir,
        cwd=lstm_root,
    )
    assert (output_dir / "summary.csv").exists()


def test_compare_imports_without_tikzplotlib(monkeypatch):
    """CI has no tikzplotlib (the PyPI release breaks on current matplotlib);
    compare.py must still load and only skip the .tex exports."""
    monkeypatch.setitem(sys.modules, "tikzplotlib", None)  # makes `import` raise
    spec = importlib.util.spec_from_file_location("compare", REPO_PY / "scripts" / "compare.py")
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
