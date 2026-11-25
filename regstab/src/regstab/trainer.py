"""Training: unconstrained minimization of the scalarized objective.

    min_θ̂  L_conv + w_div L_div + ν₁ pen(size_U) + ν₂ pen(size_Y)

The certificate holds by construction (Γ), so there is no barrier, feasibility check,
rollback or SDP. The checkpoint with the lowest validation objective is kept.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from .config import Config
from .data import (DIVERGING_LOSSES, Normalizer, Split, Trajectories, load_dataset, masked_mse,
                   max_input_norm, pad, relative_mse, save_json, windows)
from .esn import esn_init
from .measures import all_sizes, input_size, output_size, penalty
from .model import DRN, LSTM

logger = logging.getLogger(__name__)
Tensor = torch.Tensor


# --------------------------------------------------------------------------- model


def build_model(cfg: Config, nu: int, ny: int, max_input: float) -> torch.nn.Module:
    m = cfg.model
    if m.type == "lstm":
        return LSTM(nu, ny, m.hidden_size, m.num_layers)
    if m.type != "drn":
        raise ValueError(f"model.type must be 'drn' or 'lstm', got {m.type!r}")
    s0 = m.s.init
    if s0 == "max_input":
        s0 = max_input
    elif isinstance(s0, str):
        raise ValueError(f"model.s.init must be a number or 'max_input', got {s0!r}")
    if m.certificate == "global" and m.s.trainable and (
        cfg.regularization.input.weight > 0 or cfg.regularization.output.weight > 0
    ):
        logger.warning(
            "certificate 'global' with size regularization: with H = 0 the state can be "
            "rescaled without changing the input-output map, so s and C grow along that "
            "gauge without bound. Consider regularization weights 0 for the global arm."
        )
    return DRN(nu, ny, m.nx, m.nz, m.certificate, alpha=float(m.alpha.init), s=float(s0),
               train_alpha=m.alpha.trainable, train_s=m.s.trainable, eps=m.eps)


def regularization(model, cfg: Config) -> Tensor:
    """ν₁ pen(size_U) + ν₂ pen(size_Y); zero for uncertified models."""
    rc = cfg.regularization
    if not getattr(model, "certified", False):
        return torch.zeros((), dtype=torch.float64)
    reg = model.C.new_zeros(())
    if rc.input.weight:
        size = input_size(rc.input.measure, model.s, model.alpha, model.nx)
        reg = reg + rc.input.weight * penalty(size, rc.form)
    if rc.output.weight:
        reg = reg + rc.output.weight * penalty(output_size(rc.output.measure, model.C), rc.form)
    return reg


# --------------------------------------------------------------------------- evaluation


@torch.no_grad()
def coverage(model: DRN, u: Tensor, y: Tensor, x: Tensor) -> Dict[str, float]:
    """How much of the data the certificate (U, Y) covers.

    * ``input_coverage``: fraction of steps with ‖x_k‖ ≤ 1 and ‖u_k‖² ≤ s²(1 - α²‖x_k‖²)
      along the model's own rollout; ``input_coverage_traj``: fraction of trajectories
      where this holds at every step.
    * ``output_coverage``: fraction of measured outputs inside Y = E((CCᵀ)⁻¹).
    * ``s_over_umax`` etc.: size of U and Y relative to the largest ‖u_k‖ and ‖y_k‖.
    """
    valid = torch.isfinite(y).all(-1)
    s, a = model.s, model.alpha
    xn2, un2 = (x**2).sum(-1), (u**2).sum(-1)
    ok = (xn2 <= 1.0) & (un2 <= s**2 * (1.0 - a**2 * xn2))
    out = {
        "input_coverage": float(ok[valid].double().mean()),
        "input_coverage_traj": float((ok | ~valid).all(1).double().mean()),
    }
    umax = float(un2[valid].max().sqrt())
    yv = y[valid]
    ymax = float(yv.norm(dim=-1).max())
    CCt = model.C @ model.C.T
    try:
        q = (yv * torch.linalg.solve(CCt, yv.T).T).sum(-1)
        out["output_coverage"] = float((q <= 1.0).double().mean())
    except RuntimeError:  # CCᵀ singular (ny > nx): Y is degenerate
        out["output_coverage"] = float("nan")
    ybar = float(torch.linalg.eigvalsh(CCt)[0].clamp_min(0).sqrt())
    out.update({
        "s_over_umax": float(s) / umax,
        "sigma_over_umax": float(s * torch.sqrt(1 - a**2)) / umax,
        "ybar_over_ymax": ybar / ymax,
        "max_state_norm": float(xn2[valid].max().sqrt()),
    })
    return out


@torch.no_grad()
def evaluate(model, split: Split, washout: int, device: str = "cpu") -> Dict[str, float]:
    """Normalized-unit errors on one split, plus coverage for certified models."""
    model.eval()
    u, y = pad(split.conv, device=device)
    if getattr(model, "certified", False):
        y_hat, x = model.simulate(u)
    else:
        y_hat, x = model(u), None
    out = {"mse": float(masked_mse(y_hat[:, washout:], y[:, washout:]))}
    if len(split.div):
        ud, yd = pad(split.div, device=device)
        yd_hat = model(ud)
        out["mse_div"] = float(masked_mse(yd_hat, yd))
        out["relative_div"] = float(relative_mse(yd_hat, yd))
    if x is not None:
        out.update(coverage(model, u, y, x))
    return out


@torch.no_grad()
def physical_errors(model, traj: Trajectories, norm: Normalizer, y_scale: np.ndarray,
                    washout: int, prefix: str) -> Dict[str, float]:
    """RMSE in physical units and NRMSE = RMSE / std(training outputs), per channel mean."""
    if not len(traj):
        return {}
    u, y = pad(norm.apply(traj))
    y_hat = model(u).cpu().numpy()[:, washout:]
    y_true = norm.y_inverse(y.cpu().numpy()[:, washout:])
    err = norm.y_inverse(y_hat) - y_true
    with np.errstate(invalid="ignore", over="ignore"):
        rmse_c = np.sqrt(np.nanmean(err**2, axis=(0, 1)))
    return {f"{prefix}rmse": float(np.sqrt(np.mean(rmse_c**2))),
            f"{prefix}nrmse": float(np.mean(rmse_c / y_scale))}


# --------------------------------------------------------------------------- training


class _Tracker:
    """Thin MLflow wrapper that is a no-op when disabled."""

    def __init__(self, cfg: Config):
        self.enabled = cfg.mlflow.enabled
        if not self.enabled:
            return
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        self.mlflow = mlflow
        if cfg.mlflow.tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment)
        self.run = mlflow.start_run(run_name=cfg.mlflow.run_name)

    def params(self, d: Dict[str, Any]):
        if self.enabled:
            self.mlflow.log_params({k: str(v)[:500] for k, v in d.items()})

    def metrics(self, d: Dict[str, float], step: Optional[int] = None):
        if self.enabled:
            self.mlflow.log_metrics({k: float(v) for k, v in d.items()}, step=step)

    def artifacts(self, folder: Path):
        if self.enabled:
            self.mlflow.log_artifacts(str(folder))

    def end(self):
        if self.enabled:
            self.mlflow.end_run()


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def fit(model, cfg: Config, train: Split, val: Split, tracker: Optional[_Tracker] = None
        ) -> Dict[str, Any]:
    """Train ``model`` in place and restore the best validation checkpoint."""
    dc, tc = cfg.data, cfg.train
    device, washout = cfg.device, dc.washout
    win = windows(train.conv, dc.sequence_length, dc.stride)
    U, Y = pad(win, device=device)
    use_div = dc.diverging and len(train.div) > 0
    if dc.diverging_loss not in DIVERGING_LOSSES:
        raise ValueError(f"data.diverging_loss must be one of {sorted(DIVERGING_LOSSES)}, "
                         f"got {dc.diverging_loss!r}")
    div_loss_fn = DIVERGING_LOSSES[dc.diverging_loss]
    div_key = "relative_div" if dc.diverging_loss == "relative" else "mse_div"
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=tc.lr_factor, patience=tc.lr_patience, min_lr=tc.min_lr)
    g = torch.Generator().manual_seed(cfg.seed)
    best_obj, best_state, best_epoch, bad = math.inf, copy.deepcopy(model.state_dict()), 0, 0
    history = []

    for epoch in range(1, tc.epochs + 1):
        t0 = time.time()
        model.train()
        sums = {"loss": 0.0, "loss_div": 0.0, "reg": 0.0}
        n_steps = n_skipped = 0
        perm = torch.randperm(len(U), generator=g)
        for start in range(0, len(U), dc.batch_size):
            idx = perm[start : start + dc.batch_size]
            loss = masked_mse(model(U[idx])[:, washout:], Y[idx, washout:])
            total = loss
            if use_div:
                didx = torch.randperm(len(train.div), generator=g)[: dc.diverging_batch_size]
                ud, yd = pad(train.div, didx.tolist(), device=device)
                loss_div = div_loss_fn(model(ud), yd)
                total = total + dc.diverging_weight * loss_div
            reg = regularization(model, cfg)
            total = total + reg
            if not torch.isfinite(total):
                n_skipped += 1
                continue
            opt.zero_grad()
            total.backward()
            if tc.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
            opt.step()
            n_steps += 1
            sums["loss"] += float(loss)
            sums["loss_div"] += float(loss_div) if use_div else 0.0
            sums["reg"] += float(reg)

        v = evaluate(model, val, washout, device)
        with torch.no_grad():
            reg_now = float(regularization(model, cfg))
        objective = v["mse"] + (dc.diverging_weight * v.get(div_key, 0.0) if use_div else 0.0)
        objective += reg_now
        if not math.isfinite(objective):
            objective = math.inf
        sched.step(objective)

        rec = {f"train_{k}": s / max(n_steps, 1) for k, s in sums.items()}
        rec.update({f"val_{k}": val_ for k, val_ in v.items()})
        rec.update({"val_objective": objective, "lr": opt.param_groups[0]["lr"],
                    "skipped_steps": n_skipped, "epoch_time": time.time() - t0})
        if getattr(model, "certified", False):
            rec.update({"alpha": float(model.alpha), "s": float(model.s), **all_sizes(model)})
        history.append(rec)
        if tracker:
            tracker.metrics(rec, step=epoch)
        logger.info(
            "epoch %4d  train %.4g  val %.4g  obj %.4g  %s lr %.1e  (%.1fs)", epoch,
            rec["train_loss"], v["mse"], objective,
            f"s {rec['s']:.3g} α {rec['alpha']:.4f} " if "s" in rec else "",
            rec["lr"], rec["epoch_time"])

        if objective < best_obj:
            best_obj, best_state, best_epoch, bad = objective, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= tc.early_stopping_patience:
                logger.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                break

    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_objective": best_obj, "history": history}


# --------------------------------------------------------------------------- pipeline


def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run(cfg: Config) -> Dict[str, Any]:
    """Load data, build and initialize the model, train, test, and save everything."""
    _seed_everything(cfg.seed)
    dc = cfg.data
    raw = load_dataset(dc.path, dc.inputs, dc.outputs, dc.diverging)
    norm = Normalizer(dc.normalization).fit(raw["train"].conv)
    data = {k: Split(norm.apply(v.conv), norm.apply(v.div)) for k, v in raw.items()}
    nu, ny = len(dc.inputs), len(dc.outputs)
    y_scale = np.concatenate(raw["train"].conv.y).std(0)
    y_scale[y_scale == 0] = 1.0

    model = build_model(cfg, nu, ny, max_input_norm(data["train"].conv)).to(cfg.device)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = cfg.mlflow.run_name or f"{cfg.model.type}-{cfg.model.certificate}-{stamp}"
    out_dir = Path(cfg.output_dir).expanduser() / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = _Tracker(cfg)
    try:
        tracker.params(_flatten(cfg.to_dict()))
        init_info: Dict[str, Any] = {"method": "random"}
        if isinstance(model, DRN) and cfg.init.method == "esn":
            init_info = esn_init(model, data["train"].conv, dc.washout, data["validation"].conv,
                                 cfg.init.restarts, cfg.init.ridge, cfg.init.scale, cfg.seed)
            init_info["method"] = "esn"
        elif cfg.init.method not in ("esn", "random"):
            raise ValueError(f"init.method must be 'esn' or 'random', got {cfg.init.method!r}")
        init_eval = evaluate(model, data["validation"], dc.washout, cfg.device)
        tracker.metrics({f"init_val_{k}": v for k, v in init_eval.items()})
        logger.info("initial validation: %s", init_eval)

        result = fit(model, cfg, data["train"], data["validation"], tracker)

        test = evaluate(model, data["test"], dc.washout, cfg.device)
        metrics = {f"test_{k}": v for k, v in test.items()}
        metrics.update(physical_errors(model, raw["test"].conv, norm, y_scale, dc.washout, "test_"))
        metrics.update(physical_errors(model, raw["test"].div, norm, y_scale, 0, "test_div_"))
        metrics.update(physical_errors(model, raw["validation"].conv, norm, y_scale, dc.washout, "val_"))
        if getattr(model, "certified", False):
            metrics.update({"alpha": float(model.alpha), "s": float(model.s), **all_sizes(model)})
        metrics["best_epoch"] = result["best_epoch"]
        tracker.metrics(metrics)
        logger.info("test: %s", metrics)

        torch.save({"config": cfg.to_dict(), "nu": nu, "ny": ny,
                    "state_dict": model.state_dict(), "normalizer": norm.state_dict()},
                   out_dir / "model.pt")
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        save_json({"metrics": metrics, "init": init_info, "history": result["history"]},
                  out_dir / "results.json")
        tracker.artifacts(out_dir)
    finally:
        tracker.end()
    return {"metrics": metrics, "init": init_info, "out_dir": str(out_dir), "model": model,
            "normalizer": norm, **result}


def load_model(path):
    """Rebuild a saved model; returns (model, normalizer, config dict)."""
    from .config import config_from_dict

    ckpt = torch.load(Path(path).expanduser(), weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    model = build_model(cfg, ckpt["nu"], ckpt["ny"], max_input=1.0)
    model.load_state_dict(ckpt["state_dict"])
    return model, Normalizer.from_state_dict(ckpt["normalizer"]), ckpt["config"]
