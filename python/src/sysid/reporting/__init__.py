"""Reporting helpers for the comparison notebook.

Keeps the notebook slim: project config (stab_type_dict, model names,
use_div_traj) stays in the notebook and is passed into these functions.

  * runs   — MLflow run discovery and best-run / best-HP-group selection
  * tables — NRMSE table (template layout) + compact summary tables
  * plots  — val-loss training curves + multi-run trajectory comparison
"""

from .runs import collect_best_runs, fetch_runs, select_best_group
from .tables import (
    COLUMNS,
    MODEL_ORDER,
    STAB_TO_MODEL,
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
    "COLUMNS",
    "MODEL_ORDER",
    "STAB_TO_MODEL",
    "build_table",
    "build_table_train",
    "build_split_tables",
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
