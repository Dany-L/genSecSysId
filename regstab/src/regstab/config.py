"""Experiment configuration.

A YAML file maps one-to-one onto these dataclasses. Unknown keys are an error, so a typo or
a stale key cannot silently become a no-op.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class DataConfig:
    path: str = ""
    inputs: List[str] = field(default_factory=lambda: ["u"])
    outputs: List[str] = field(default_factory=lambda: ["y"])
    normalization: str = "scale_only"  # scale_only | standard | none
    sequence_length: Optional[int] = None  # training windows; None = full trajectories
    stride: Optional[int] = None  # None = sequence_length
    batch_size: int = 32
    washout: int = 0  # steps excluded from the converging loss (unknown x0)
    diverging: bool = False  # also train on <split>_div (x0 = 0, no washout)
    diverging_batch_size: int = 8
    diverging_weight: float = 1.0
    diverging_loss: str = "relative"  # relative (Σ‖ŷ-y‖²/Σ‖y‖² per trajectory) | mse


@dataclass
class ScalarParam:
    init: Union[float, str] = 1.0  # a number, or "max_input" for s (max ‖u_k‖ in train)
    trainable: bool = True


@dataclass
class ModelConfig:
    type: str = "drn"  # drn | lstm
    certificate: str = "regional"  # regional | global | none (drn only)
    nx: int = 2
    nz: int = 20
    alpha: ScalarParam = field(default_factory=lambda: ScalarParam(0.99, True))
    s: ScalarParam = field(default_factory=lambda: ScalarParam("max_input", True))
    eps: float = 1e-6  # Cayley margin
    hidden_size: int = 32  # lstm
    num_layers: int = 1  # lstm


@dataclass
class InitConfig:
    method: str = "esn"  # esn | random
    restarts: int = 10
    ridge: float = 1e-8  # relative to the state energy
    scale: Union[float, List[float]] = field(default_factory=lambda: [0.01, 1.0])


@dataclass
class RegTerm:
    measure: str
    weight: float = 0.0


@dataclass
class RegularizationConfig:
    form: str = "inverse_square"  # inverse_square | neg_log
    input: RegTerm = field(default_factory=lambda: RegTerm("s"))
    output: RegTerm = field(default_factory=lambda: RegTerm("trace"))


@dataclass
class TrainConfig:
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    lr_patience: int = 20
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    early_stopping_patience: int = 100


@dataclass
class MlflowConfig:
    enabled: bool = True
    tracking_uri: Optional[str] = None
    experiment: str = "regstab"
    run_name: Optional[str] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    init: InitConfig = field(default_factory=InitConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "outputs"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Convert numeric strings to the annotated type.

    YAML 1.1 reads ``1e-3`` (no dot) as a string, which would otherwise reach the optimizer.
    """
    if not isinstance(value, str):
        return value
    args = typing.get_args(hint) or (hint,)
    for t in (int, float):
        if t in args:
            try:
                return t(value)
            except ValueError:
                pass
    if str in args or Any in args:
        return value
    if float in args or int in args:
        raise TypeError(f"{where}: expected a number, got {value!r}")
    return value


def _build(cls, values: Any, where: str):
    if not dataclasses.is_dataclass(cls):
        return values
    if dataclasses.is_dataclass(values):
        return values
    if not isinstance(values, dict):
        raise TypeError(f"{where}: expected a mapping, got {values!r}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; valid: {sorted(fields)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, val in values.items():
        sub = _NESTED.get((cls, name))
        kwargs[name] = (_build(sub, val, f"{where}.{name}") if sub
                        else _coerce(val, hints[name], f"{where}.{name}"))
    return cls(**kwargs)


_NESTED = {
    (Config, "data"): DataConfig,
    (Config, "model"): ModelConfig,
    (Config, "init"): InitConfig,
    (Config, "regularization"): RegularizationConfig,
    (Config, "train"): TrainConfig,
    (Config, "mlflow"): MlflowConfig,
    (ModelConfig, "alpha"): ScalarParam,
    (ModelConfig, "s"): ScalarParam,
    (RegularizationConfig, "input"): RegTerm,
    (RegularizationConfig, "output"): RegTerm,
}


def config_from_dict(d: Dict[str, Any]) -> Config:
    return _build(Config, d or {}, "config")


def apply_overrides(d: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML) to a nested dict."""
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not of the form key=value")
        node = d
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return d


def load_config(path: Union[str, Path], overrides: Optional[List[str]] = None) -> Config:
    d = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    return config_from_dict(apply_overrides(d, overrides or []))
