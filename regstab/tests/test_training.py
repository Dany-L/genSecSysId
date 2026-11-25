"""End-to-end: config -> data -> ESN init -> training -> test metrics -> saved model."""

import json
import math

import numpy as np
import pytest
import torch

from regstab.config import config_from_dict
from regstab.data import pad
from regstab.parametrization import certificate_margins
from regstab.trainer import load_model, run

from test_data_config import make_dataset


def _cfg(root, out, div_loss="relative", **model):
    return config_from_dict({
        "data": {"path": str(root), "inputs": ["u"], "outputs": ["q"], "sequence_length": 60,
                 "stride": 30, "batch_size": 4, "washout": 10, "diverging": True,
                 "diverging_batch_size": 2, "diverging_loss": div_loss},
        "model": {"nx": 2, "nz": 5, **model},
        "init": {"restarts": 2},
        "regularization": {"input": {"measure": "sigma", "weight": 1e-3},
                           "output": {"measure": "trace", "weight": 1e-3}},
        "train": {"epochs": 2},
        "mlflow": {"enabled": False},
        "output_dir": str(out),
    })


@pytest.mark.parametrize("model", [
    {"certificate": "regional"},
    {"certificate": "global"},
    {"certificate": "none"},
    {"type": "lstm", "hidden_size": 4},
])
def test_run_end_to_end(tmp_path, model):
    root = make_dataset(tmp_path / "ds")
    result = run(_cfg(root, tmp_path / "out", **model))
    m = result["metrics"]
    for k in ("test_mse", "test_rmse", "test_nrmse", "val_rmse"):
        assert math.isfinite(m[k]), k
    out = result["out_dir"]
    saved = json.load(open(f"{out}/results.json"))
    assert len(saved["history"]) == 2

    # the saved model reproduces the trained one
    reloaded, norm, _ = load_model(f"{out}/model.pt")
    u = torch.randn(2, 30, 1, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(reloaded(u), result["model"](u))

    trained = result["model"]
    if getattr(trained, "certified", False):
        # the certificate still holds after training, by construction
        mats = {k: v.detach().numpy() for k, v in trained.matrices().items()}
        mg = certificate_margins(*(mats[k] for k in ("A", "B", "B2", "C2", "D21", "Lam", "H")),
                                 alpha=float(trained.alpha), s=float(trained.s))
        assert mg["max_eig_F"] < 1e-9 and mg["max_h_norm"] <= 1.0
        for k in ("test_input_coverage", "test_output_coverage", "s", "alpha",
                  "size_in_sigma", "size_out_trace"):
            assert k in m


def test_s_init_from_data_and_fixed_scalars(tmp_path):
    root = make_dataset(tmp_path / "ds")
    cfg = _cfg(root, tmp_path / "out",
               alpha={"init": 0.97, "trainable": False}, s={"init": "max_input", "trainable": False})
    result = run(cfg)
    model, norm = result["model"], result["normalizer"]
    from regstab.data import load_dataset, max_input_norm
    train = norm.apply(load_dataset(str(root), ["u"], ["q"], False)["train"].conv)
    assert np.isclose(float(model.s), max_input_norm(train))
    assert np.isclose(float(model.alpha), 0.97)


def test_run_with_mlflow_file_store(tmp_path):
    root = make_dataset(tmp_path / "ds")
    cfg = _cfg(root, tmp_path / "out")
    cfg.mlflow.enabled = True
    cfg.mlflow.tracking_uri = (tmp_path / "mlruns").as_uri()
    cfg.mlflow.experiment = "test"
    run(cfg)
    import mlflow

    runs = mlflow.search_runs(experiment_names=["test"])
    assert len(runs) == 1 and "metrics.test_rmse" in runs.columns


@pytest.mark.parametrize("div_loss", ["relative", "mse"])
def test_diverging_loss_options(tmp_path, div_loss):
    root = make_dataset(tmp_path / "ds")
    result = run(_cfg(root, tmp_path / "out", div_loss=div_loss))
    hist = result["history"]
    assert all("val_relative_div" in h and "val_mse_div" in h for h in hist)
    key = "val_relative_div" if div_loss == "relative" else "val_mse_div"
    h = hist[0]
    # objective = conv mse + weight (1) · selected div loss + ν (1e-3) · inverse-square penalties
    reg = 1e-3 / h["size_in_sigma"] ** 2 + 1e-3 / h["size_out_trace"] ** 2
    assert math.isclose(h["val_objective"], h["val_mse"] + h[key] + reg, rel_tol=1e-9)


def test_unknown_diverging_loss_is_rejected(tmp_path):
    root = make_dataset(tmp_path / "ds")
    with pytest.raises(ValueError, match="diverging_loss"):
        run(_cfg(root, tmp_path / "out", div_loss="huber"))
