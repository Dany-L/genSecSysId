"""Per-experiment evaluation table (``results/eval_table_template.tex``).

Driven by ``results/results_confg.yaml``: every experiment contributes one
block of rows, one row per model class, filled from the best run of that class
as ranked by the experiment's ``criterion`` metric.

Most columns are read straight off the MLflow run. The ``# div/total`` column
is not logged anywhere and is recomputed here: each diverging test input is
replayed through the model, the input is then held at zero for ``padding``
further steps, and a trajectory counts as diverged when its final state has
not returned to the origin (``||x_N|| > eps``). That is the same test as
``results/tables/divergence-admissibility.tex``.

As elsewhere in this package the MLflow lookup, the model rollout and the pure
LaTeX assembly are kept apart so the config handling and the formatting can be
unit-tested without a tracking server or a checkpoint:

  * :func:`load_table_config`  — parse / normalise / validate the YAML
  * :func:`matches_selector`   — subset match of a run against a selector
  * :func:`resolve_run_config` — dotted key -> tag, else logged config.yaml
  * :func:`select_best_run`    — best run of a model class by the criterion
  * :func:`count_diverged`     — the zero-padded rollout
  * :func:`build_eval_table`   — rows -> LaTeX

(LaTeX needs ``booktabs`` and ``multirow``; the row labels are the project's
``\\MLtiRnn`` / ``\\MStdSec`` / ``\\MGenSec`` macros and ``\\nw``.)
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

# ── layout ────────────────────────────────────────────────────────────────────
# Row order matches results/eval_table_template.tex (LtiRnn, StdSec, GenSec),
# which is *not* the order the model classes appear in the YAML.
MODEL_TO_LATEX: Dict[str, str] = {
    "NoSec": r"\MLtiRnn{}",
    "StdSec": r"\MStdSec{}",
    "GenSec": r"\MGenSec{}",
}
ROW_ORDER: List[str] = ["NoSec", "StdSec", "GenSec"]

HEADER: List[str] = [
    "Experiment",
    "Model",
    "NRMSE",
    r"$\bar y$",
    r"$y_{\text{max}}$",
    r"\# div/ total",
    r"\# pars",
    r"$\nw$",
]
COLUMN_SPEC = "rrllllll"

# ── where each column comes from ──────────────────────────────────────────────
# ȳ moved from a flat key to the per-certificate namespace part-way through the
# project (duffing-soft-7 predates the move), so both spellings are accepted.
Y_BAR_KEYS: Tuple[str, ...] = ("post_process/max_s/y_bar", "post_process/y_bar")
Y_MAX_KEY = "data/max_output_train"
# config.model.nw is an alias for hidden_size (see sysid.config), and
# hidden_size is the one logged on every experiment.
NW_PARAM = "hidden_size"
N_PARS_PARAM = "trainable_parameters"

DEFAULT_CRITERION = "id/conv/eval_nrmse"
DEFAULT_DIVERGENCE: Dict[str, Any] = {
    "diverging_inputs": "evaluation/id/inputs_div.npy",
    "padding": 500,
    "eps": 0.1,
}


@dataclass
class EvalRow:
    """One table row: a model class of one experiment, plus its provenance."""

    experiment: str
    model_name: str
    run_id: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    nrmse: Optional[float] = None
    y_bar: Optional[float] = None
    y_max: Optional[float] = None
    diverged: Optional[Tuple[int, int]] = None  # (n_diverged, n_total)
    n_pars: Optional[int] = None
    nw: Optional[int] = None
    note: Optional[str] = None  # why a cell (or the whole row) is empty

    @property
    def model_label(self) -> str:
        return MODEL_TO_LATEX.get(self.model_name, self.model_name)


# ── config ────────────────────────────────────────────────────────────────────
def _as_selector(raw: Any, where: str) -> Dict[str, Any]:
    """A ``config:`` entry as a flat dict of dotted key -> value.

    Accepts either a mapping or a list of mappings (the original file wrapped
    every selector in a one-element list); a list is merged left to right.
    """
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        merged: Dict[str, Any] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"{where}: 'config' list entries must be mappings, got {type(item).__name__}"
                )
            merged.update(item)
        return merged
    raise ValueError(
        f"{where}: 'config' must be a mapping or a list of mappings, got {type(raw).__name__}"
    )


def load_table_config(path) -> Dict[str, Any]:
    """Parse the evaluation-table YAML into a normalised, validated structure.

    Returns ``{"experiments": [...]}`` where each experiment carries a resolved
    ``criterion`` and ``divergence`` block (experiment value, else the file's
    ``defaults``, else the module default) and each run carries a flat
    ``selector`` dict.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, Mapping) or "experiments" not in raw:
        raise ValueError(f"{path}: expected a mapping with an 'experiments' key")

    defaults = raw.get("defaults") or {}
    base_criterion = defaults.get("criterion", DEFAULT_CRITERION)
    base_divergence = {**DEFAULT_DIVERGENCE, **(defaults.get("divergence") or {})}

    experiments: List[Dict[str, Any]] = []
    for i, exp in enumerate(raw["experiments"] or []):
        where = f"{path}: experiments[{i}]"
        if not isinstance(exp, Mapping):
            raise ValueError(f"{where}: expected a mapping")
        for required in ("name", "mlflow_experiment_name"):
            if not exp.get(required):
                raise ValueError(f"{where}: missing '{required}'")

        runs = []
        for j, run in enumerate(exp.get("runs") or []):
            run_where = f"{where}.runs[{j}]"
            if not isinstance(run, Mapping) or not run.get("model_name"):
                raise ValueError(f"{run_where}: missing 'model_name'")
            selector = _as_selector(run.get("config"), run_where)
            if not selector:
                raise ValueError(f"{run_where}: empty 'config' selector")
            runs.append({"model_name": run["model_name"], "selector": selector})

        experiments.append(
            {
                "name": exp["name"],
                "mlflow_server_uri": exp.get("mlflow_server_uri"),
                "mlflow_experiment_name": exp["mlflow_experiment_name"],
                "criterion": exp.get("criterion", base_criterion),
                "divergence": {**base_divergence, **(exp.get("divergence") or {})},
                "runs": runs,
            }
        )

    return {"experiments": experiments}


# ── run matching ──────────────────────────────────────────────────────────────
def _canon(value: Any) -> Optional[str]:
    """Canonical string for comparing a YAML value with an MLflow tag.

    Tags are always strings (``"True"``, ``"20"``), the YAML gives real bools
    and ints, and a logged config gives bools and ints again — so everything is
    compared as text with booleans normalised to ``True``/``False``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in ("true", "false"):
            return stripped.lower().capitalize()
        return stripped
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
    return str(value)


def matches_selector(run_config: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    """True when every key of ``selector`` is present in ``run_config`` and agrees.

    Subset semantics: settings the selector does not mention are free. A key
    that is missing or ``None`` on the run never matches — an absent tag is
    treated as unknown rather than as "false".
    """
    for key, wanted in selector.items():
        got = _canon(run_config.get(key))
        if got is None or got != _canon(wanted):
            return False
    return True


def _dig(tree: Any, dotted_key: str) -> Any:
    """``a.b.c`` lookup in a nested mapping, ``None`` when any level is absent."""
    node = tree
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def resolve_run_config(
    row: Mapping[str, Any],
    keys: Iterable[str],
    config_loader: Optional[Callable[[str], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Values of ``keys`` for one ``search_runs`` row.

    A key is read from the run's ``tags.<key>`` column first. Only if that is
    absent or null does this fall back to ``config_loader(run_id)`` — the run's
    logged ``outputs/config.yaml``, looked up by dotted path. Sweep-tagged
    experiments therefore never download anything, while the untagged local
    one-D experiment resolves entirely from its logged configs.
    """
    resolved: Dict[str, Any] = {}
    pending: List[str] = []
    for key in keys:
        value = row.get(f"tags.{key}")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            pending.append(key)
        else:
            resolved[key] = value

    if pending and config_loader is not None:
        run_id = row.get("run_id")
        cfg = config_loader(run_id) if run_id else None
        if cfg:
            for key in pending:
                resolved[key] = _dig(cfg, key)
    return resolved


def select_best_run(
    runs_df,
    selector: Mapping[str, Any],
    criterion: str,
    config_loader: Optional[Callable[[str], Mapping[str, Any]]] = None,
    lower_is_better: bool = True,
):
    """The row of the best run matching ``selector``, or ``None``.

    ``runs_df`` is an ``mlflow.search_runs`` frame for a single experiment.
    Runs without a value for ``criterion`` are dropped before ranking.
    """
    metric_col = f"metrics.{criterion}"
    if runs_df is None or len(runs_df) == 0 or metric_col not in runs_df.columns:
        return None

    candidates = runs_df.dropna(subset=[metric_col])
    keep = [
        idx
        for idx, row in candidates.iterrows()
        if matches_selector(resolve_run_config(row, selector.keys(), config_loader), selector)
    ]
    if not keep:
        return None

    matched = candidates.loc[keep]
    best_idx = matched[metric_col].idxmin() if lower_is_better else matched[metric_col].idxmax()
    return matched.loc[best_idx]


def first_metric(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    """First of ``keys`` that the run carries as a metric, else ``None``."""
    for key in keys:
        value = row.get(f"metrics.{key}")
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return float(value)
    return None


def param_int(row: Mapping[str, Any], name: str) -> Optional[int]:
    """``params.<name>`` as an int, ``None`` when absent or not a number."""
    value = row.get(f"params.{name}")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# ── divergence count ──────────────────────────────────────────────────────────
def count_diverged(
    model,
    normalizer,
    diverging_inputs: np.ndarray,
    padding: int = 500,
    eps: float = 0.1,
    batch_size: int = 16,
) -> Tuple[int, int]:
    """``(n_diverged, n_total)`` over the zero-padded diverging test inputs.

    Each input is normalised, replayed through ``model``, then the input is held
    at zero for ``padding`` steps; a trajectory counts as diverged when
    ``||x_N|| > eps`` at the end of that tail, i.e. the state has not returned
    to the origin. NaNs (the ragged-length padding of the stacked test array)
    become zeros, matching the evaluation path.

    Trajectories are replayed in batches of ``batch_size``: the Lur'e rollout is
    sequential in time but vectorised over the batch, so this is much faster
    than one forward per trajectory while keeping the peak state tensor bounded.
    """
    import torch

    inputs = np.asarray(diverging_inputs, dtype=float)
    if inputs.ndim != 3:
        raise ValueError(
            f"diverging inputs must be (n_traj, n_steps, nd), got shape {inputs.shape}"
        )
    n_total, _, nd = inputs.shape
    if n_total == 0:
        return 0, 0

    # transform_inputs broadcasts against the fitted (1, 1, nd) statistics, so
    # the result can pick up a leading axis -- reshape back to a clean batch.
    normalized = np.asarray(normalizer.transform_inputs(np.nan_to_num(inputs)))
    normalized = normalized.reshape(n_total, -1, nd)

    proto = next(model.parameters())
    was_training = model.training
    model.eval()
    n_diverged = 0
    try:
        with torch.no_grad():
            for start in range(0, n_total, max(1, batch_size)):
                chunk = normalized[start : start + max(1, batch_size)]
                driven = torch.as_tensor(chunk, dtype=proto.dtype, device=proto.device)
                tail = torch.zeros(
                    (driven.shape[0], padding, nd), dtype=proto.dtype, device=proto.device
                )
                _, (states, _), _ = model(torch.cat([driven, tail], dim=1))
                final = states[:, -1, :].cpu().numpy()
                n_diverged += int((np.linalg.norm(final, axis=1) > eps).sum())
    finally:
        if was_training:
            model.train()

    return n_diverged, n_total


# ── LaTeX assembly ────────────────────────────────────────────────────────────
def _escape(text: str) -> str:
    """Escape the LaTeX specials that can show up in an experiment name."""
    out = str(text)
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        out = out.replace(char, rf"\{char}" if char != "\\" else r"\textbackslash{}")
    return out


def _number(value: Optional[float], decimals: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{decimals}f}"


def _integer(value: Optional[int]) -> str:
    return "--" if value is None else str(int(value))


def _div_cell(diverged: Optional[Tuple[int, int]]) -> str:
    if diverged is None:
        return "--"
    n_diverged, n_total = diverged
    return f"{n_diverged}/{n_total}"


def build_eval_table(
    rows: Sequence[EvalRow],
    decimals: int = 2,
    comment: Optional[str] = None,
) -> str:
    """Render rows as the LaTeX table of ``results/eval_table_template.tex``.

    ``rows`` are grouped by :attr:`EvalRow.experiment` in first-seen order; the
    group's experiment name spans its rows via ``\\multirow`` and groups are
    separated by ``\\midrule``. Within a group the rows are ordered by
    :data:`ROW_ORDER`. Every row ends in a LaTeX comment carrying the run id and
    the MLflow experiment name it was filled from, so a number in the paper can
    be traced back to its run.
    """
    grouped: Dict[str, List[EvalRow]] = {}
    for row in rows:
        grouped.setdefault(row.experiment, []).append(row)

    order = {name: i for i, name in enumerate(ROW_ORDER)}
    body: List[str] = []
    for block_i, (experiment, block) in enumerate(grouped.items()):
        block = sorted(block, key=lambda r: order.get(r.model_name, len(order)))
        if block_i:
            body.append(r"\midrule")
        for row_i, row in enumerate(block):
            lead = (
                rf"\multirow{{{len(block)}}}{{*}}{{{_escape(experiment)}}}"
                if row_i == 0
                else ""
            )
            cells = [
                lead,
                row.model_label,
                _number(row.nrmse, decimals),
                _number(row.y_bar, decimals),
                _number(row.y_max, decimals),
                _div_cell(row.diverged),
                _integer(row.n_pars),
                _integer(row.nw),
            ]
            if row.run_id:
                trace = f"run_id={row.run_id} experiment={row.mlflow_experiment_name}"
                # A '--' on a row that *does* have a run needs its reason on the
                # line, so "no diverging set logged" is not read as "0 diverged".
                if row.note:
                    trace += f" [{row.note}]"
            else:
                trace = row.note or "no matching run"
            # lstrip so continuation rows read "& \MStdSec{} & ..." as in the template.
            body.append(" & ".join(cells).lstrip() + rf"\\ % {trace}")

    lines = [f"% {comment}"] if comment else []
    lines += [
        rf"\begin{{tabular}}{{{COLUMN_SPEC}}}",
        r"\toprule",
        " & ".join(HEADER) + r"\\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)
