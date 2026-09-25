"""Hyperparameter table for the runs of the evaluation (NRMSE) table.

Driven by ``results/hyperparameter_config.yaml``, which names the columns; the
rows are the best run per (experiment, model class) exactly as
:mod:`sysid.reporting.eval_table` selects them from ``results/results_confg.yaml``.

As in :mod:`sysid.reporting.eval_table` the MLflow lookup stays in the script,
and this module is pure so it can be tested offline:

  * :func:`load_hyperparameter_config` — parse / validate the column YAML
  * :func:`column_value`               — one column's value off one run
  * :func:`format_value`               — value -> LaTeX cell
  * :func:`build_hyperparameter_table` — rows -> LaTeX

Layout: one row per model, grouped by experiment (``\\multirow`` on the
experiment name), one column per hyperparameter. With four model classes and
six experiments that is 24 rows by ~10 columns -- the transpose would be 24
columns wide. (LaTeX needs ``booktabs`` and ``multirow``.)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from .eval_table import MODEL_TO_LATEX, ROW_ORDER, _dig, _escape, _trace, significant

SOURCES = ("param", "metric", "tag", "config")
FORMATS = ("int", "sig", "sci", "duration")


@dataclass
class HyperRow:
    """One table row: a model class of one experiment, plus its provenance."""

    experiment: str
    model_name: str
    run_id: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    values: Dict[str, Optional[float]] = field(default_factory=dict)  # label -> value
    note: Optional[str] = None


# ── config ────────────────────────────────────────────────────────────────────
def load_hyperparameter_config(path) -> Dict[str, Any]:
    """Parse the column YAML into ``{"results_config": ..., "columns": [...]}``.

    Every column comes back with ``key`` as a list, ``offset`` (default 0) and
    ``blank_for`` (default empty) filled in.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping) or not raw.get("columns"):
        raise ValueError(f"{path}: expected a mapping with a non-empty 'columns' list")

    columns = []
    for i, col in enumerate(raw["columns"]):
        where = f"{path}: columns[{i}]"
        if not isinstance(col, Mapping):
            raise ValueError(f"{where}: expected a mapping")
        for required in ("label", "source", "key", "format"):
            if not col.get(required):
                raise ValueError(f"{where}: missing '{required}'")
        if col["source"] not in SOURCES:
            raise ValueError(f"{where}: source must be one of {SOURCES}, got {col['source']!r}")
        if col["format"] not in FORMATS:
            raise ValueError(f"{where}: format must be one of {FORMATS}, got {col['format']!r}")
        keys = col["key"] if isinstance(col["key"], (list, tuple)) else [col["key"]]
        columns.append(
            {
                "label": str(col["label"]),
                "source": col["source"],
                "keys": [str(k) for k in keys],
                "format": col["format"],
                "offset": float(col.get("offset", 0)),
                "blank_for": list(col.get("blank_for") or []),
            }
        )

    return {"results_config": raw.get("results_config"), "columns": columns}


def needs_logged_config(columns: Sequence[Mapping[str, Any]]) -> bool:
    """True when some column reads the run's ``outputs/config.yaml``."""
    return any(col["source"] == "config" for col in columns)


# ── values ────────────────────────────────────────────────────────────────────
def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def column_value(
    column: Mapping[str, Any],
    run: Mapping[str, Any],
    run_config: Optional[Mapping[str, Any]] = None,
) -> Optional[float]:
    """The column's value for one ``search_runs`` row, ``None`` when absent.

    ``param``/``metric``/``tag`` read the ``params.``/``metrics.``/``tags.``
    columns of the row; ``config`` looks the dotted key up in ``run_config``.
    The first key that yields a number wins, then ``offset`` is added.
    """
    prefix = {"param": "params.", "metric": "metrics.", "tag": "tags."}
    for key in column["keys"]:
        if column["source"] == "config":
            raw = _dig(run_config, key) if run_config else None
        else:
            raw = run.get(prefix[column["source"]] + key)
        value = _as_float(raw)
        if value is not None:
            return value + column.get("offset", 0.0)
    return None


def row_values(
    columns: Sequence[Mapping[str, Any]],
    model_name: str,
    run: Mapping[str, Any],
    run_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """``{label: value}`` for one run, with ``blank_for`` columns set to None."""
    return {
        col["label"]: (
            None if model_name in col.get("blank_for", ()) else column_value(col, run, run_config)
        )
        for col in columns
    }


# ── formatting ────────────────────────────────────────────────────────────────
def format_duration(seconds: Optional[float]) -> str:
    """Wall time in the largest unit that keeps it >= 1: ``42 s``, ``3.5 min``, ``6.4 h``."""
    if seconds is None:
        return "--"
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def format_sci(value: Optional[float], digits: int = 2) -> str:
    r"""``5e-3`` -> ``$5 \cdot 10^{-3}$``; a trailing-zero mantissa is dropped."""
    if value is None:
        return "--"
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = f"{value / 10 ** exponent:.{digits - 1}f}".rstrip("0").rstrip(".")
    if mantissa in ("10", "-10"):  # rounding pushed it up a decade
        mantissa, exponent = mantissa[:-1], exponent + 1
    if mantissa in ("1", "-1"):
        return rf"${'-' if mantissa == '-1' else ''}10^{{{exponent}}}$"
    return rf"${mantissa} \cdot 10^{{{exponent}}}$"


def format_value(value: Optional[float], fmt: str, digits: int = 3) -> str:
    """One cell of the table; ``--`` for a missing value."""
    if value is None:
        return "--"
    if fmt == "int":
        return str(int(round(value)))
    if fmt == "duration":
        return format_duration(value)
    if fmt == "sci":
        return format_sci(value)
    return significant(value, digits=digits)


# ── LaTeX assembly ────────────────────────────────────────────────────────────
def build_hyperparameter_table(
    rows: Sequence[HyperRow],
    columns: Sequence[Mapping[str, Any]],
    comment: Optional[str] = None,
    significant_digits: int = 3,
) -> str:
    """Render rows as a ``booktabs`` table, one block of model rows per experiment.

    Experiments keep their first-seen order, models follow
    :data:`sysid.reporting.eval_table.ROW_ORDER`, and a model class the
    experiment does not list is skipped rather than rendered empty. Every row
    ends in a LaTeX comment with the run it was read from.
    """
    experiments: List[str] = []
    by_experiment: Dict[str, Dict[str, HyperRow]] = {}
    for row in rows:
        if row.experiment not in by_experiment:
            experiments.append(row.experiment)
            by_experiment[row.experiment] = {}
        by_experiment[row.experiment][row.model_name] = row
    if not experiments:
        raise ValueError("no rows to render")

    known = set(ROW_ORDER)
    body: List[str] = []
    for i, experiment in enumerate(experiments):
        group = by_experiment[experiment]
        # Unknown model classes go after the known ones, in first-seen order.
        models = [m for m in ROW_ORDER if m in group] + [m for m in group if m not in known]
        if i:
            body.append(r"\midrule")
        for j, model in enumerate(models):
            row = group[model]
            name = (
                rf"\multirow{{{len(models)}}}{{*}}{{{_escape(experiment)}}}" if j == 0 else ""
            )
            cells = [
                format_value(row.values.get(col["label"]), col["format"], significant_digits)
                for col in columns
            ]
            label = MODEL_TO_LATEX.get(model, _escape(model))
            body.append(
                f"{name} & {label} & " + " & ".join(cells) + r" \\ % " + _trace(row, experiment)
            )

    header = "Dataset & Model & " + " & ".join(col["label"] for col in columns)
    lines = [f"% {comment}"] if comment else []
    lines += [
        rf"\begin{{tabular}}{{ll{'r' * len(columns)}}}",
        r"\toprule",
        rf"{header} \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)
