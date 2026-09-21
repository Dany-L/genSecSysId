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

# Per-experiment column group: the two metrics that matter, side by side.
SUB_HEADER: List[str] = ["NRMSE", r"\# div/ total"]
# Rows carried above the GenSec block (one metric pair per experiment).
PLAIN_ROWS: List[str] = ["NoSec", "StdSec"]
# Rows that only GenSec has: a certificate quantity spanning the whole column
# group, because it is not one of the two per-model metrics.
GENSEC_EXTRA_ROWS: List[Tuple[str, str]] = [
    (r"$\bar y$", "y_bar"),
    (r"$\bar \sigma(\theta)$", "sigma_u"),
]

# ── where each column comes from ──────────────────────────────────────────────
# ȳ moved from a flat key to the per-certificate namespace part-way through the
# project (duffing-soft-7 predates the move), so both spellings are accepted.
Y_BAR_KEYS: Tuple[str, ...] = ("post_process/max_s/y_bar", "post_process/y_bar")
Y_MAX_KEY = "data/max_output_train"
# sigma(U) = |s|*sqrt(1 - alpha^2), the size of the admissible input set --
# logged every epoch by the trainer (see Trainer.sigma_u). Reported for GenSec
# only: without a regional certificate there is no admissible input set to size.
SIGMA_KEY = "sigma_u"
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
    sigma_u: Optional[float] = None  # GenSec only; see SIGMA_KEY
    # Collected and logged but no longer tabulated -- the table now carries the
    # two metrics that matter per model, and the parameter counts crowded them
    # out without separating the arms (they differ by ~50 out of ~780).
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


def significant(value: Optional[float], digits: int = 3) -> str:
    r"""A certificate quantity at ``digits`` significant figures.

    Used for ``y_max``, ``ȳ`` and ``σ(θ)``, which span a whole column group and
    so need no decimal alignment, but DO span orders of magnitude: on the
    shipped experiments ``σ`` runs from 5e-5 (a collapsed admissible input set)
    to 19.7, and ``ȳ`` reaches 335. One shared decimal count then prints the
    others at a precision they do not carry -- ``y_max = 0.934495``.

    Outside ``[1e-3, 1e5)`` the result is LaTeX scientific notation, wrapped in
    math mode; inside it, a plain decimal.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    value = float(value)
    if value == 0.0:
        return "0"
    if 1e-3 <= abs(value) < 1e5:
        return f"{value:.{digits}g}"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf"${mantissa:.{digits - 1}f} \cdot 10^{{{exponent}}}$"


def _integer(value: Optional[int]) -> str:
    return "--" if value is None else str(int(value))


def _div_cell(diverged: Optional[Tuple[int, int]]) -> str:
    if diverged is None:
        return "--"
    n_diverged, n_total = diverged
    return f"{n_diverged}/{n_total}"


def adaptive_decimals(
    values: Sequence[Optional[float]],
    significant: int = 2,
    min_decimals: int = 2,
    max_decimals: int = 6,
) -> int:
    """Decimals that give the SMALLEST value in ``values`` ``significant`` digits.

    One count per column, chosen from the column's own numbers, so every cell in
    a column lines up on the decimal point while none of them rounds away. The
    1-D benchmark is the case that forces this: its NRMSEs are ~2e-3, and the
    two decimals that suit the Duffing columns print all of them as ``0.00``.

    Values that are ``None``/NaN/zero are ignored; an empty column falls back to
    ``min_decimals``.
    """
    finite = [
        abs(float(v))
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and v != 0
    ]
    if not finite:
        return min_decimals
    smallest = min(finite)
    needed = int(math.ceil(-math.log10(smallest))) + significant - 1
    return max(min_decimals, min(max_decimals, needed))


def rank_emphasis(
    values: Sequence[Optional[float]], lower_is_better: bool = True
) -> Dict[int, Tuple[bool, bool]]:
    """``{index: (bold, italic)}`` marking the best and second-best entries.

    Best is bold, second best bold-italic — the convention
    :func:`sysid.reporting.tables.build_table` already uses, so the two tables in
    the paper read the same way. ``lower_is_better`` because these are error
    metrics. Ties share a rank: two identical best values are both bold and
    nothing is marked second.
    """
    scored = [
        (float(v), i)
        for i, v in enumerate(values)
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not scored:
        return {}
    ordered = sorted(scored, key=lambda t: t[0] if lower_is_better else -t[0])
    best = ordered[0][0]
    marks: Dict[int, Tuple[bool, bool]] = {i: (True, False) for v, i in scored if v == best}
    runners = [v for v, _ in ordered if v != best]
    if runners:
        second = runners[0]
        marks.update({i: (True, True) for v, i in scored if v == second})
    return marks


def _emphasize(text: str, bold: bool, italic: bool) -> str:
    """Text-mode emphasis, matching ``tables._emphasize``."""
    if bold and italic:
        return rf"\textbf{{\textit{{{text}}}}}"
    if bold:
        return rf"\textbf{{{text}}}"
    return text


def _trace(row: Optional[EvalRow], experiment_key: str) -> str:
    """``<mlflow experiment>=<run id>`` for one cell, or why it is empty."""
    if row is None:
        return f"{experiment_key}=<no row>"
    # Name the MLflow experiment whether or not a run was found, so a reader
    # chasing an empty cell knows which experiment to go looking in.
    name = row.mlflow_experiment_name or experiment_key
    if row.run_id is None:
        return f"{name}=<{row.note or 'no matching run'}>"
    trace = f"{name}={row.run_id}"
    return f"{trace} [{row.note}]" if row.note else trace


def build_eval_table(
    rows: Sequence[EvalRow],
    comment: Optional[str] = None,
    nrmse_significant: int = 2,
    min_decimals: int = 2,
    max_decimals: int = 6,
    significant_digits: int = 3,
) -> str:
    r"""Render rows as the LaTeX table of ``results/eval_table_template.tex``.

    EXPERIMENTS ARE COLUMNS and models are rows, so one model can be read across
    every benchmark at a glance — the transpose of the earlier layout, which put
    one experiment per block and made that comparison a vertical scan.

    Each experiment owns a two-column group ``(NRMSE, # div/total)`` under a
    spanning header carrying its name and its ``y_max``. ``\MLtiRnn`` and
    ``\MStdSec`` take one row each; ``\MGenSec`` takes a row plus two more,
    ``$\bar y$`` and ``$\bar\sigma(\theta)$``, which span the whole group
    because they are certificate quantities rather than per-model metrics and
    only the regional arm has them.

    NRMSE decimals are chosen per column by :func:`adaptive_decimals`; the
    certificate quantities use :func:`significant` instead, because they span
    a column group and orders of magnitude. Per NRMSE
    column the lowest is bold, the second lowest bold-italic
    (:func:`rank_emphasis`). Every row ends in a LaTeX comment mapping each
    experiment to the run it was filled from.

    (LaTeX needs ``booktabs``; the labels are the project's ``\MLtiRnn`` /
    ``\MStdSec`` / ``\MGenSec`` macros.)
    """
    # Experiments in first-seen order -> column groups.
    experiments: List[str] = []
    by_experiment: Dict[str, Dict[str, EvalRow]] = {}
    for row in rows:
        if row.experiment not in by_experiment:
            experiments.append(row.experiment)
            by_experiment[row.experiment] = {}
        by_experiment[row.experiment][row.model_name] = row
    if not experiments:
        raise ValueError("no rows to render")

    def cell(experiment: str, model: str) -> Optional[EvalRow]:
        return by_experiment[experiment].get(model)

    # One decimal count and one emphasis map per experiment column.
    decimals: Dict[str, int] = {}
    emphasis: Dict[str, Dict[int, Tuple[bool, bool]]] = {}
    for experiment in experiments:
        nrmse = [
            (cell(experiment, m).nrmse if cell(experiment, m) else None) for m in ROW_ORDER
        ]
        decimals[experiment] = adaptive_decimals(
            nrmse,
            significant=nrmse_significant,
            min_decimals=min_decimals,
            max_decimals=max_decimals,
        )
        emphasis[experiment] = rank_emphasis(nrmse)

    # y_max belongs to the dataset, not the model, so it goes in the header.
    # Any arm of the experiment carries it; take the first that does.
    y_max: Dict[str, Optional[float]] = {}
    for experiment in experiments:
        levels = [
            cell(experiment, m).y_max
            for m in ROW_ORDER
            if cell(experiment, m) is not None and cell(experiment, m).y_max is not None
        ]
        y_max[experiment] = levels[0] if levels else None

    header_names = " ".join(
        rf"& \multicolumn{{2}}{{c}}{{{_escape(e)}}}" for e in experiments
    )
    header_levels = " ".join(
        rf"& \multicolumn{{2}}{{l}}{{$y_{{\text{{max}}}} = "
        rf"{significant(y_max[e], digits=significant_digits)}$}}"
        for e in experiments
    )
    header_metrics = "Model  & " + " & ".join(
        " & ".join(SUB_HEADER) for _ in experiments
    )

    body: List[str] = []

    def metric_row(model: str) -> str:
        cells: List[str] = []
        traces: List[str] = []
        for i_exp, experiment in enumerate(experiments):
            row = cell(experiment, model)
            i_model = ROW_ORDER.index(model)
            bold, italic = emphasis[experiment].get(i_model, (False, False))
            value = _number(row.nrmse if row else None, decimals[experiment])
            cells += [
                _emphasize(value, bold, italic) if value != "--" else value,
                _div_cell(row.diverged if row else None),
            ]
            traces.append(_trace(row, experiment))
        return (
            f"{MODEL_TO_LATEX[model]} & " + " & ".join(cells) + r" \\ % " + ", ".join(traces)
        )

    body.append(metric_row("NoSec"))
    body.append(metric_row("StdSec"))
    body.append(r"\midrule")
    body.append(metric_row("GenSec"))

    # GenSec-only certificate rows, each spanning its experiment's whole group.
    for label, attribute in GENSEC_EXTRA_ROWS:
        cells = []
        for experiment in experiments:
            row = cell(experiment, "GenSec")
            value = getattr(row, attribute) if row else None
            cells.append(
                rf"\multicolumn{{2}}{{l}}{{{significant(value, digits=significant_digits)}}}"
            )
        body.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines = [f"% {comment}"] if comment else []
    lines += [
        rf"\begin{{tabular}}{{r{'ll' * len(experiments)}}}",
        r"\toprule",
        f"{header_names} \\\\",
        f"{header_levels} \\\\",
        rf"{header_metrics} \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)
