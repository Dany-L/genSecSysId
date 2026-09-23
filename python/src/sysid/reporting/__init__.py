"""Reporting helpers for the comparison notebook.

Keeps the notebook slim: project config (stab_type_dict, model names,
use_div_traj) stays in the notebook and is passed into these functions.

  * runs   — MLflow run discovery and best-run / best-HP-group selection
  * eval_table — per-experiment evaluation table driven by results_confg.yaml
  * tables — NRMSE table (template layout) + compact summary tables
  * plots  — val-loss training curves + multi-run trajectory comparison
"""

from .runs import collect_best_runs, fetch_runs, select_best_group
from .eval_table import (
    MODEL_TO_LATEX,
    ROW_ORDER,
    EvalRow,
    build_eval_table,
    count_diverged,
    load_table_config,
    matches_selector,
    resolve_run_config,
    select_best_run,
)
from .tables import (
    COLUMNS,
    DIVERGENCE_GROUPS,
    MODEL_ORDER,
    STAB_TO_MODEL,
    build_divergence_table,
    build_split_tables,
    build_summary_frames,
    build_table,
    build_table_train,
    collect_nrmse_cells,
    format_cell,
    metric_key,
    summary_tables_latex,
)
from .plots import (
    load_run_comparison,
    plot_comparison_all_runs,
    plot_val_loss,
    smooth_ema,
)

__all__ = [
    "collect_best_runs",
    "fetch_runs",
    "select_best_group",
    "MODEL_TO_LATEX",
    "ROW_ORDER",
    "EvalRow",
    "build_eval_table",
    "count_diverged",
    "load_table_config",
    "matches_selector",
    "resolve_run_config",
    "select_best_run",
    "COLUMNS",
    "DIVERGENCE_GROUPS",
    "MODEL_ORDER",
    "STAB_TO_MODEL",
    "build_table",
    "build_table_train",
    "build_split_tables",
    "build_divergence_table",
    "collect_nrmse_cells",
    "format_cell",
    "metric_key",
    "build_summary_frames",
    "summary_tables_latex",
    "plot_val_loss",
    "load_run_comparison",
    "plot_comparison_all_runs",
    "smooth_ema",
]
