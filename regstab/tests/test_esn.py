"""Echo-state-network initialization."""

import numpy as np
import torch

from regstab.data import Trajectories, masked_mse, pad
from regstab.esn import esn_init, ridge_readout
from regstab.model import DRN


def _data(n_traj=6, T=200, seed=0):
    """Trajectories from a random regional DRN, so a good reservoir exists."""
    g = torch.Generator().manual_seed(seed)
    true = DRN(1, 1, 2, 6, alpha=0.98, s=3.0)
    true.sample_free_parameters(0.05, generator=g)
    u = torch.randn(n_traj, T, 1, generator=g, dtype=torch.float64)
    with torch.no_grad():
        y = true(u)
    return Trajectories(list(u.numpy()), list(y.numpy()))


def test_ridge_readout_matches_lstsq():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 4, generator=g, dtype=torch.float64)
    y = torch.randn(500, 2, generator=g, dtype=torch.float64)
    C = ridge_readout(x, y, ridge=0.0)
    C_ls = np.linalg.lstsq(x.numpy(), y.numpy(), rcond=None)[0].T
    assert np.allclose(C.numpy(), C_ls)


def test_esn_keeps_best_restart_and_its_readout():
    train, val = _data(seed=0), _data(n_traj=3, seed=1)
    model = DRN(1, 1, 2, 6, alpha=0.98, s=3.0)
    info = esn_init(model, train, washout=20, val=val, restarts=6, scale=[0.01, 1.0], seed=3)
    errs = [h["mse"] for h in info["restarts"]]
    assert info["best_mse"] == min(errs)
    # the restored model reproduces the selected error
    u, y = pad(val)
    with torch.no_grad():
        err = float(masked_mse(model(u)[:, 20:], y[:, 20:]))
    assert np.isclose(err, info["best_mse"])
    # α and s are not touched by the restarts
    assert np.isclose(float(model.alpha), 0.98) and np.isclose(float(model.s), 3.0)


def test_esn_readout_beats_random_output_layer():
    train = _data(seed=2)
    model = DRN(1, 1, 2, 6, alpha=0.98, s=3.0)
    u, y = pad(train)
    with torch.no_grad():
        err_random = float(masked_mse(model(u)[:, 20:], y[:, 20:]))
    info = esn_init(model, train, washout=20, restarts=3, seed=0)
    assert info["best_mse"] < err_random


def test_esn_is_deterministic_given_seed():
    train = _data(seed=4)
    a, b = DRN(1, 1, 2, 6), DRN(1, 1, 2, 6)
    ia = esn_init(a, train, washout=10, restarts=3, seed=7)
    ib = esn_init(b, train, washout=10, restarts=3, seed=7)
    assert ia["best_mse"] == ib["best_mse"]
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(va, vb), ka


def test_esn_for_uncertified_model():
    train = _data(seed=5)
    model = DRN(1, 1, 2, 6, certificate="none")
    info = esn_init(model, train, washout=10, restarts=2, seed=0)
    assert np.isfinite(info["best_mse"])
