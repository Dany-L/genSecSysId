"""Tests for sysid.config.resolve_run_artifacts_mlflow (remote-model loading).

We never hit a real MLflow server: ``mlflow.set_tracking_uri`` and
``mlflow.artifacts.download_artifacts`` are monkeypatched to serve a local
temp directory laid out the way train.py logs artefacts (outputs/config.yaml,
models/best_model.pt, models/normalizer.json, models/run_info.json).
"""

from pathlib import Path

import mlflow
import mlflow.artifacts
import pytest

from sysid.config import Config, resolve_run_artifacts_mlflow

RES = Path(__file__).parent / "resoruces" / "safety_filter_duffing.yaml"


def _make_run_dir(tmp_path, model=True, normalizer=True, info=True):
    """Lay out a fake MLflow artifact download and return (outputs, models)."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    Config.from_yaml(str(RES)).save_yaml(str(outputs / "config.yaml"))
    if model:
        (models / "best_model.pt").write_bytes(b"")
    if normalizer:
        (models / "normalizer.json").write_text("{}")
    if info:
        (models / "run_info.json").write_text('{"run_id": "abc"}')
    return outputs, models


@pytest.fixture
def patched_mlflow(monkeypatch, tmp_path):
    """Patch mlflow so download_artifacts serves the local temp layout."""
    outputs = models = None
    calls = {}

    def fake_set_uri(uri):
        calls["uri"] = uri

    def fake_download(run_id, artifact_path):
        calls.setdefault("run_ids", []).append(run_id)
        return str({"outputs": outputs, "models": models}[artifact_path])

    monkeypatch.setattr(mlflow, "set_tracking_uri", fake_set_uri)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", fake_download)

    def _install(o, m):
        nonlocal outputs, models
        outputs, models = o, m
        return calls

    return _install


def test_resolve_downloads_and_parses(tmp_path, patched_mlflow):
    outputs, models = _make_run_dir(tmp_path)
    calls = patched_mlflow(outputs, models)

    config, model_path, normalizer_path, run_info = resolve_run_artifacts_mlflow(
        "run123", "http://fake-server/"
    )

    assert isinstance(config, Config)
    assert calls["uri"] == "http://fake-server/"  # URI was set on the client
    assert calls["run_ids"] == ["run123", "run123"]  # outputs + models
    assert Path(model_path).name == "best_model.pt" and Path(model_path).exists()
    assert Path(normalizer_path).name == "normalizer.json"
    assert run_info == {"run_id": "abc"}


def test_missing_normalizer_and_info_return_none(tmp_path, patched_mlflow):
    outputs, models = _make_run_dir(tmp_path, normalizer=False, info=False)
    patched_mlflow(outputs, models)

    _, _, normalizer_path, run_info = resolve_run_artifacts_mlflow("r", "uri")

    assert normalizer_path is None
    assert run_info is None


def test_missing_checkpoint_raises(tmp_path, patched_mlflow):
    outputs, models = _make_run_dir(tmp_path, model=False)
    patched_mlflow(outputs, models)

    with pytest.raises(FileNotFoundError):
        resolve_run_artifacts_mlflow("r", "uri")
