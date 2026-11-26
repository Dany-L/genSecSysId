"""Data loading, batching and configuration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from regstab.config import Config, apply_overrides, config_from_dict, load_config
from regstab.data import (Normalizer, Trajectories, load_dataset, masked_mse, max_input_norm,
                          pad, windows)

REPO = Path(__file__).resolve().parents[1]


def _write(folder: Path, lengths, rng, scale=1.0):
    folder.mkdir(parents=True)
    for i, T in enumerate(lengths):
        pd.DataFrame({"u": rng.normal(size=T) * scale, "q": rng.normal(size=T),
                      "q_dot": rng.normal(size=T)}).to_csv(folder / f"traj_{i:03d}.csv", index=False)


def make_dataset(root: Path, with_div=True, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", 4), ("validation", 2), ("test", 2)):
        _write(root / split, [120] * n, rng)
        if with_div:
            _write(root / f"{split}_div", [15, 40, 27][: n + 1], rng, scale=5.0)
    return root


def test_load_dataset_with_and_without_div(tmp_path):
    root = make_dataset(tmp_path / "ds")
    data = load_dataset(str(root), ["u"], ["q"], diverging=True)
    assert len(data["train"].conv) == 4 and len(data["train"].div) == 3
    assert data["train"].conv.u[0].shape == (120, 1)
    assert [len(a) for a in data["train"].div.u] == [15, 40, 27]
    no_div = load_dataset(str(root), ["u"], ["q", "q_dot"], diverging=False)
    assert len(no_div["test"].div) == 0 and no_div["test"].conv.y[0].shape == (120, 2)


def test_missing_div_folder_raises(tmp_path):
    root = make_dataset(tmp_path / "ds", with_div=False)
    with pytest.raises(FileNotFoundError):
        load_dataset(str(root), ["u"], ["q"], diverging=True)


def test_windows_and_pad():
    traj = Trajectories([np.arange(10.0)[:, None], np.arange(4.0)[:, None]],
                        [np.arange(10.0)[:, None], np.arange(4.0)[:, None]])
    w = windows(traj, length=4, stride=3)
    assert [a[0, 0] for a in w.u] == [0, 3, 6, 0]
    assert windows(traj, None, None) is traj
    u, y = pad(traj)
    assert u.shape == (2, 10, 1)
    assert torch.isnan(y[1, 4:]).all() and (u[1, 4:] == 0).all()


def test_masked_mse_ignores_nan_targets():
    pred = torch.tensor([[1.0, 2.0, 100.0]])
    target = torch.tensor([[0.0, 2.0, float("nan")]])
    assert float(masked_mse(pred, target)) == 0.5


@pytest.mark.parametrize("method", ["scale_only", "standard", "none"])
def test_normalizer_roundtrip(method):
    rng = np.random.default_rng(0)
    traj = Trajectories([rng.normal(2, 3, (50, 2))], [rng.normal(-1, 5, (50, 1))])
    n = Normalizer(method).fit(traj)
    y = traj.y[0]
    assert np.allclose(n.y_inverse(n.y(y)), y)
    n2 = Normalizer.from_state_dict(n.state_dict())
    assert np.allclose(n2.u(traj.u[0]), n.u(traj.u[0]))
    if method == "scale_only":
        assert np.allclose(n.apply(traj).u[0].std(0), 1.0) and n.u(np.zeros(2)).sum() == 0


def test_max_input_norm():
    traj = Trajectories([np.array([[3.0, 4.0], [0.0, 1.0]])], [np.zeros((2, 1))])
    assert max_input_norm(traj) == 5.0


def test_unknown_config_keys_are_errors():
    with pytest.raises(ValueError, match="unknown key"):
        config_from_dict({"trian": {}})
    with pytest.raises(ValueError, match="config.model.alpha"):
        config_from_dict({"model": {"alpha": {"init": 0.9, "learnable": True}}})


def test_overrides():
    d = apply_overrides({"model": {"nx": 2}}, ["model.nx=8", "train.lr=1e-4",
                                              "init.scale=[0.1, 1.0]"])
    cfg = config_from_dict(d)
    assert cfg.model.nx == 8 and cfg.train.lr == 1e-4 and cfg.init.scale == [0.1, 1.0]
    with pytest.raises(ValueError):
        apply_overrides({}, ["model.nx"])


def test_defaults_and_example_configs_parse():
    cfg = Config()
    assert cfg.model.s.init == "max_input" and cfg.regularization.form == "inverse_square"
    for path in sorted((REPO / "configs").glob("*.yaml")):
        load_config(path)


def test_numeric_strings_are_coerced():
    cfg = config_from_dict({"train": {"lr": "1e-3", "epochs": "5"},
                            "model": {"s": {"init": "2e1"}, "alpha": {"init": "0.9"}}})
    assert cfg.train.lr == 1e-3 and cfg.train.epochs == 5
    assert cfg.model.s.init == 20.0 and cfg.model.alpha.init == 0.9
    assert config_from_dict({"model": {"s": {"init": "max_input"}}}).model.s.init == "max_input"
    with pytest.raises(TypeError):
        config_from_dict({"train": {"lr": "fast"}})


def test_relative_mse_is_scale_free_and_per_trajectory():
    from regstab.data import relative_mse

    target = torch.tensor([[[1.0], [2.0], [float("nan")]], [[10.0], [20.0], [40.0]]])
    pred = torch.tensor([[[1.0], [1.0], [123.0]], [[10.0], [20.0], [0.0]]])
    # traj 0: 1/5, traj 1: 1600/2100; padded step of traj 0 ignored
    expected = 0.5 * (1 / 5 + 1600 / 2100)
    assert torch.isclose(relative_mse(pred, target), torch.tensor(expected))
    assert torch.isclose(relative_mse(1e3 * pred, 1e3 * target), torch.tensor(expected))
    assert float(relative_mse(target.nan_to_num(), target)) == 0.0


def test_relative_mse_gradient_ignores_nonfinite_padding():
    from regstab.data import relative_mse

    pred = torch.tensor([[[1.0], [float("inf")]]], requires_grad=True)
    target = torch.tensor([[[2.0], [float("nan")]]])
    relative_mse(pred, target).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[0, 1, 0] == 0
