"""LaTeX evaluation tables for the comparison/reporting notebook.

Two products:
  * ``build_table`` — the NRMSE table matching
    ``results/tables/_template-mse-eval.tex`` (training group x distribution x
    trajectory type), per-column highlighting (lowest mean bold, second-lowest
    bold-italic). Fed by ``collect_nrmse_cells`` which pulls the per-cell
    mean/std from MLflow.
  * ``build_summary_frames`` / ``summary_tables_latex`` — the compact
    best-individual / best-HP-group summary tables.

Pure assembly is separated from the MLflow lookup so the formatting can be
unit-tested without a tracking server. Required LaTeX packages: ``booktabs``
and ``multirow`` (cell emphasis uses plain ``\\textbf``/``\\textit``).
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── NRMSE table layout (matches the template header) ──────────────────────────
#   train_div in {"no", "yes"}  -> "Only conv." vs "Conv. and div." in training
#   dist      in {"id", "ood"}
#   traj      in {"conv", "div", "overall"}  -> "conv." / "div." / "all"
TRAIN_GROUPS: List[str] = ["no", "yes"]
DISTS: List[str] = ["id", "ood"]
TRAJS: List[str] = ["conv", "div", "overall"]
COLUMNS: List[Tuple[str, str, str]] = [
    (tg, dist, traj) for tg in TRAIN_GROUPS for dist in DISTS for traj in TRAJS
]

# Row order matches the template (\MLtiRnn, \MGenSec, \MStdSec).
MODEL_ORDER: List[str] = [r"\MLtiRnn{}", r"\MGenSec{}", r"\MGenSec{}_s", r"\MStdSec{}", r"\MStdSec{}_s"]

# stab_type tag value -> template row label (default; the notebook can override).
STAB_TO_MODEL: Dict[str, str] = {
    "none": r"\MLtiRnn{}",
    "regional": r"\MGenSec{}",
    "regional_s": r"\MGenSec{}_s",
    "global": r"\MStdSec{}",
    "global_s": r"\MGenSec{}_s",
}

Cell = Optional[Tuple[float, float, int]]  # (mean, std, n) or None


def metric_key(dist: str, traj: str) -> str:
    """MLflow metric key for a (distribution, trajectory-type) column."""
    return f"{dist}/{traj}/eval_nrmse"


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _emphasize(s: str, bold: bool, italic: bool) -> str:
    """Text-mode emphasis (robust across math fonts where ``\\mathbf`` and
    ``\\boldsymbol`` look identical for digits, e.g. ICLR's Times template)."""
    if bold and italic:
        return rf"\textbf{{\textit{{{s}}}}}"
    if bold:
        return rf"\textbf{{{s}}}"
    return s


def format_cell(
    mean: Optional[float],
    std: Optional[float] = None,
    decimals: int = 4,
    bold: bool = False,
    italic: bool = False,
    std_style: str = "paren",
    std_decimals: Optional[int] = None,
) -> str:
    """Render one table cell as ``mean (std)`` with optional emphasis.

    ``std_style`` controls how the spread is shown:
      * ``"paren"`` (default) — ``mean (std)``, no ± symbol; compact and
        readable.
      * ``"pm"`` — inline ``mean \\ensuremath{\\pm} std``.
      * ``"sub"`` — ``±std`` as a small ``\\textsubscript`` after the mean.
    ``std_decimals`` (default = ``decimals``) sets the spread's precision
    independently. Emphasis: bold -> ``\\textbf`` (lowest in a column),
    bold+italic -> ``\\textbf{\\textit{...}}`` (second-lowest). Missing -> ``--``.
    """
    if _is_missing(mean):
        return "--"
    sd = decimals if std_decimals is None else std_decimals
    mean_s = f"{mean:.{decimals}f}"
    if _is_missing(std):
        return _emphasize(mean_s, bold, italic)
    std_s = f"{std:.{sd}f}"
    if std_style == "sub":
        return _emphasize(mean_s, bold, italic) + rf"\textsubscript{{\ensuremath{{\pm}}{std_s}}}"
    if std_style == "pm":
        return _emphasize(rf"{mean_s} \ensuremath{{\pm}} {std_s}", bold, italic)
    return _emphasize(rf"{mean_s} ({std_s})", bold, italic)


def _column_emphasis(means: List[Optional[float]]) -> Dict[int, Tuple[bool, bool]]:
    """Row index -> (bold, italic): lowest -> (True, False), second -> (True, True)."""
    flags = {i: (False, False) for i in range(len(means))}
    finite = [(i, m) for i, m in enumerate(means) if not _is_missing(m)]
    order = sorted(finite, key=lambda t: t[1])
    if len(order) >= 1:
        flags[order[0][0]] = (True, False)
    if len(order) >= 2:
        flags[order[1][0]] = (True, True)
    return flags


_NRMSE_HEADER = "\n".join(
    [
        r"\begin{tabular}{rccc ccc ccc ccc}",
        r"\toprule",
        r"\multirow{3}{*}{Model} & \multicolumn{6}{c}{Only conv. traj. in training} & "
        r"\multicolumn{6}{c}{Conv. and div. traj. in training}\\",
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-13}",
        r"& \multicolumn{3}{c}{ID} & \multicolumn{3}{c}{OOD} & "
        r"\multicolumn{3}{c}{ID} & \multicolumn{3}{c}{OOD}\\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}",
        r"& conv. & div. & all & conv. & div. & all & conv. & div. & all & conv. & div. & all \\",
        r"\midrule",
    ]
)


def build_table(
    cells: Dict[str, Dict[Tuple[str, str, str], Cell]],
    decimals: int = 2,
    std_style: str = "paren",
    std_decimals: Optional[int] = None,
) -> str:
    """Turn per-(model, column) ``(mean, std, n)`` cells into a 12-column LaTeX
    tabular (both training conditions side by side).

    ``cells[model_label][(train_div, dist, traj)] = (mean, std, n) | None``.
    Models/columns absent from ``cells`` render ``--``. ``std_style`` /
    ``std_decimals`` are forwarded to :func:`format_cell`. For a narrower
    layout, prefer :func:`build_split_tables` (one 6-column table per training
    condition).
    """
    col_flags: Dict[Tuple[str, str, str], Dict[int, Tuple[bool, bool]]] = {}
    for col in COLUMNS:
        means = [
            (cells.get(model, {}).get(col) or (None,))[0] for model in MODEL_ORDER
        ]
        col_flags[col] = _column_emphasis(means)

    body_lines = []
    for mi, model in enumerate(MODEL_ORDER):
        rendered = []
        for col in COLUMNS:
            c = cells.get(model, {}).get(col)
            bold, italic = col_flags[col][mi]
            if c is None:
                rendered.append("--")
            else:
                mean, std, _n = c
                rendered.append(
                    format_cell(mean, std, decimals, bold, italic, std_style, std_decimals)
                )
        body_lines.append(f"{model}  & " + " & ".join(rendered) + r" \\")

    return "\n".join([_NRMSE_HEADER, *body_lines, r"\bottomrule", r"\end{tabular}"])


# ── Split tables: one 6-column table per training condition ───────────────────
TRAIN_LABELS: Dict[str, str] = {
    "no": "Only conv. traj. in training",
    "yes": "Conv. and div. traj. in training",
}

_SINGLE_HEADER = "\n".join(
    [
        r"\begin{tabular}{rccc ccc}",
        r"\toprule",
        r"\multirow{2}{*}{Model} & \multicolumn{3}{c}{ID} & \multicolumn{3}{c}{OOD}\\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"& conv. & div. & all & conv. & div. & all \\",
        r"\midrule",
    ]
)


def _train_columns(train_div: str) -> List[Tuple[str, str, str]]:
    return [(train_div, dist, traj) for dist in DISTS for traj in TRAJS]


def build_table_train(
    cells: Dict[str, Dict[Tuple[str, str, str], Cell]],
    train_div: str,
    decimals: int = 2,
    std_style: str = "paren",
    std_decimals: Optional[int] = None,
) -> str:
    """A 6-column NRMSE table (ID/OOD x conv/div/all) for one training condition.

    ``train_div`` is ``"no"`` (conv-only training) or ``"yes"`` (conv+div). The
    lowest mean per column is bold, the second-lowest bold-italic. A leading
    LaTeX comment names the training condition.
    """
    cols = _train_columns(train_div)
    col_flags = {
        col: _column_emphasis(
            [(cells.get(m, {}).get(col) or (None,))[0] for m in MODEL_ORDER]
        )
        for col in cols
    }

    body_lines = []
    for mi, model in enumerate(MODEL_ORDER):
        rendered = []
        for col in cols:
            c = cells.get(model, {}).get(col)
            bold, italic = col_flags[col][mi]
            if c is None:
                rendered.append("--")
            else:
                mean, std, _n = c
                rendered.append(
                    format_cell(mean, std, decimals, bold, italic, std_style, std_decimals)
                )
        body_lines.append(f"{model}  & " + " & ".join(rendered) + r" \\")

    comment = f"% {TRAIN_LABELS.get(train_div, train_div)}"
    return "\n".join([comment, _SINGLE_HEADER, *body_lines, r"\bottomrule", r"\end{tabular}"])


def build_split_tables(
    cells: Dict[str, Dict[Tuple[str, str, str], Cell]],
    decimals: int = 2,
    std_style: str = "paren",
    std_decimals: Optional[int] = None,
) -> Dict[str, str]:
    """Two 6-column tables keyed by training condition (``"no"`` / ``"yes"``)."""
    return {
        tg: build_table_train(cells, tg, decimals, std_style, std_decimals)
        for tg in TRAIN_GROUPS
    }


def collect_nrmse_cells(
    experiment_name: str,
    stab_type_dict: Dict[str, Dict],
    use_div_traj: Dict[str, Dict],
    stab_to_model: Dict[str, str],
    rank_metric: str = "id/overall/eval_nrmse",
    search_fn: Optional[Callable] = None,
    verbose: bool = True,
    extra_tags: Optional[Dict[str, object]] = None,
    extra_filter: Optional[str] = None,
    aggregate: str = "best_group",
) -> Dict[str, Dict[Tuple[str, str, str], Cell]]:
    """Pull per-cell ``(mean, std, n)`` NRMSE from MLflow for ``build_table``.

    Per (model x training condition) ``runs.select_best_group`` chooses which
    runs to aggregate — the same selector ``collect_best_runs`` uses:
      * ``aggregate="best_group"`` (default): the HP group with the lowest mean
        ``rank_metric``.
      * ``aggregate="all"``: every matched run (stab/div tags = whole class).
    The six categorized NRMSE metrics are then averaged over the selected runs.

    ``extra_tags`` / ``extra_filter`` restrict the search further, e.g.
    ``extra_tags={"model.nw": 16}`` to build the table for a single
    architecture size (see ``runs.fetch_runs``).
    """
    from .runs import fetch_runs, select_best_group

    all_stab_keys = set(k for tags in stab_type_dict.values() for k in tags)
    cells: Dict[str, Dict[Tuple[str, str, str], Cell]] = {m: {} for m in MODEL_ORDER}

    for stab_key, stab_tags in stab_type_dict.items():
        model = stab_to_model[stab_key]
        for div_key, div_tags in use_div_traj.items():
            runs = fetch_runs(
                experiment_name, stab_tags, div_tags, all_stab_keys,
                rank_metric, search_fn=search_fn,
                extra_tags=extra_tags, extra_filter=extra_filter,
            )
            if runs is None:
                if verbose:
                    print(f"  [{stab_key}/{div_key}] no runs — cells left as '--'")
                continue
            grp = select_best_group(
                runs, f"metrics.{rank_metric}", aggregate=aggregate
            )
            if grp is None or grp.empty:
                if verbose:
                    print(f"  [{stab_key}/{div_key}] no usable HP group")
                continue
            for (tg, dist, traj) in COLUMNS:
                if tg != div_key:
                    continue
                col = f"metrics.{metric_key(dist, traj)}"
                vals = grp[col].dropna().values if col in grp.columns else np.array([])
                if len(vals) == 0:
                    cells[model][(tg, dist, traj)] = None
                else:
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    cells[model][(tg, dist, traj)] = (mean, std, len(vals))
            if verbose:
                scope = "class (all)" if aggregate == "all" else "best group"
                print(f"  [{stab_key}/{div_key}] {scope}: n={len(grp)} runs")
    return cells


# ── Compact summary tables (best individual / best HP group) ──────────────────
COLS_INDIV = ["Model", "Reg.", "Learn L", "Div. Traj.", "NRMSE"]
COLS_MEAN = ["Model", "Reg.", "Learn L", "Div. Traj.", "Mean", "Std", "n"]


def build_summary_frames(
    best_runs: Dict[str, Optional[dict]],
    stab_type_dict: Dict[str, Dict],
    use_div_traj: Dict[str, Dict],
    model_name_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build (df_indiv, df_mean) summary DataFrames from ``collect_best_runs``."""
    indiv_records, mean_records = [], []
    for info in best_runs.values():
        if info is None:
            continue
        stab_tags = stab_type_dict[info["stab_key"]]
        base = {
            "Model": model_name_map[info["stab_key"]],
            "Reg.": stab_tags.get("training.use_custom_regularization", None),
            "Learn L": stab_tags.get("model.custom_params.learn_L", None),
            "Div. Traj.": use_div_traj[info["div_key"]]["data.use_diverging_trajectories"],
            "_div": info["div_key"],
        }
        indiv_records.append({**base, "NRMSE": info["indiv_nrmse"]})
        mean_records.append(
            {**base, "Mean": info["mean"], "Std": info["std"], "n": len(info["all_nrmse"])}
        )
    return pd.DataFrame(indiv_records), pd.DataFrame(mean_records)


def _val_to_latex(val) -> str:
    if pd.isna(val):
        return "--"
    if isinstance(val, (bool, np.bool_)):
        return r"\checkmark" if val else r"$\times$"
    if isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    return str(val)


def _rows_indiv(frame: pd.DataFrame) -> str:
    return "\n".join(
        " & ".join(_val_to_latex(row[c]) for c in COLS_INDIV) + r" \\"
        for _, row in frame.iterrows()
    )


def _rows_mean(frame: pd.DataFrame) -> str:
    lines = []
    for _, row in frame.iterrows():
        std_str = f'{row["Std"]:.4f}' if not pd.isna(row["Std"]) else "--"
        cells = [
            _val_to_latex(row["Model"]),
            _val_to_latex(row["Reg."]),
            _val_to_latex(row["Learn L"]),
            _val_to_latex(row["Div. Traj."]),
            f'${_val_to_latex(row["Mean"])} \\pm {std_str}$',
            str(int(row["n"])),
        ]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def _tabular(fmt: str, header: str, body: str) -> str:
    return "\n".join(
        [rf"\begin{{tabular}}{{{fmt}}}", r"\toprule", header, r"\midrule", body,
         r"\bottomrule", r"\end{tabular}"]
    )


def _grouped_body(df: pd.DataFrame, sort_col: str, row_fn, use_div_traj) -> str:
    blocks = []
    for div_key in use_div_traj:
        grp = df[df["_div"] == div_key].sort_values(sort_col).reset_index(drop=True)
        blocks.append(row_fn(grp))
    if len(blocks) <= 1:
        return blocks[0] if blocks else ""
    return r"\hline" + "\n" + (r" \\" + "\n" + r"\hline" + "\n").join(blocks)


def summary_tables_latex(
    df_indiv: pd.DataFrame, df_mean: pd.DataFrame, use_div_traj: Dict[str, Dict]
) -> Dict[str, str]:
    """Build the four summary LaTeX tables (flat + grouped, indiv + mean)."""
    hdr_i = r"Model & Reg. & Learn $L$ & Div.\ Traj. & NRMSE \\"
    hdr_m = r"Model & Reg. & Learn $L$ & Div.\ Traj. & NRMSE (mean $\pm$ std) & $n$ \\"
    fmt_i, fmt_m = "lcccc", "lccccc"

    df1 = df_indiv.sort_values("NRMSE").reset_index(drop=True)
    df2 = df_mean.sort_values("Mean").reset_index(drop=True)
    return {
        "table1_best_individual": _tabular(fmt_i, hdr_i, _rows_indiv(df1)),
        "table2_best_mean": _tabular(fmt_m, hdr_m, _rows_mean(df2)),
        "table3_best_individual_grouped": _tabular(
            fmt_i, hdr_i, _grouped_body(df_indiv, "NRMSE", _rows_indiv, use_div_traj)
        ),
        "table4_best_mean_grouped": _tabular(
            fmt_m, hdr_m, _grouped_body(df_mean, "Mean", _rows_mean, use_div_traj)
        ),
    }
