"""MLflow run discovery for the comparison/reporting notebook.

Fetches training runs by stability type and diverging-trajectory setting, then
picks the best individual run and the best hyper-parameter group per
combination. The MLflow ``search_runs`` call is injected via ``search_fn`` so
the selection logic is unit-testable without a tracking server.

Project-specific config (the ``stab_type_dict`` / ``use_div_traj`` mappings)
stays in the notebook and is passed in — this module makes no assumptions about
which tags exist.
"""

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


def build_filter_string(tags_dict: Dict[str, object]) -> str:
    r"""MLflow filter string: ``tags.`k` = "v"`` conditions joined by ``and``."""
    return " and ".join(f'tags.`{k}` = "{v}"' for k, v in tags_dict.items())


def _default_search_fn() -> Callable:
    import mlflow

    return mlflow.search_runs


def fetch_runs(
    experiment_name: str,
    stab_tags: Dict[str, object],
    div_tags: Dict[str, object],
    all_stab_keys,
    metric: str,
    search_fn: Optional[Callable] = None,
    extra_tags: Optional[Dict[str, object]] = None,
    extra_filter: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Return runs matching the tags that carry ``metric``, else ``None``.

    Runs carrying a stability tag *not* in ``stab_tags`` are dropped so e.g.
    ``none`` runs (no ``learn_L`` tag) don't leak into ``regional``/``global``.

    ``extra_tags`` adds tag equality conditions (same form as the stab/div
    tags), e.g. ``{"model.nw": 16}`` -> ``tags.`model.nw` = "16"``. Sweep
    overrides are logged as tags under their dotted key, so this is the way to
    restrict to one architecture size. ``extra_filter`` is a raw MLflow filter
    fragment ANDed in for non-tag fields, e.g. ``'params.hidden_size = "16"'``
    when ``nw`` was fixed in the base config rather than swept.
    """
    if search_fn is None:
        search_fn = _default_search_fn()

    required_absent = set(all_stab_keys) - set(stab_tags)
    tag_filter = build_filter_string({**stab_tags, **div_tags, **(extra_tags or {})})
    filter_str = " and ".join(p for p in (tag_filter, extra_filter) if p)
    try:
        runs = search_fn(experiment_names=[experiment_name], filter_string=filter_str)
    except Exception as e:  # pragma: no cover - network/credential failures
        print(f"  Search error: {e}")
        return None

    if runs is None or runs.empty:
        return None

    for key in required_absent:
        col = f"tags.{key}"
        if col in runs.columns:
            runs = runs[runs[col].isna()]

    metric_col = f"metrics.{metric}"
    if metric_col not in runs.columns:
        return None

    runs = runs.dropna(subset=[metric_col])
    return runs if not runs.empty else None


def get_best_individual(runs: pd.DataFrame, metric_col: str):
    """``(run_id, value)`` of the single run with the lowest ``metric_col``."""
    idx = runs[metric_col].idxmin()
    return runs.loc[idx, "run_id"], float(runs.loc[idx, metric_col])


def _group_keys(runs: pd.DataFrame) -> pd.Series:
    """Group runs by parent run id, falling back to the full param tuple."""
    parent_col = "tags.mlflow.parentRunId"
    param_cols = [c for c in runs.columns if c.startswith("params.")]
    if parent_col in runs.columns and runs[parent_col].notna().any():
        return runs[parent_col].fillna("__NA__")
    if param_cols:
        return runs[param_cols].fillna("__NA__").apply(
            lambda r: "|".join(r.astype(str)), axis=1
        )
    return pd.Series("all", index=runs.index)


def get_best_mean_group(
    runs: pd.DataFrame, metric_col: str, aggregate: str = "best_group"
):
    """Best HP group's summary, or the whole class when ``aggregate="all"``.

    Returns ``(run_id, best_val, vals, all_run_ids, mean, std, group_key)`` where
    ``run_id`` is the best single run inside the selected group and ``std`` is
    ``None`` for n=1.

    ``aggregate``:
      * ``"best_group"`` (default): partition runs into HP groups (see
        ``_group_keys``) and pick the group with the lowest mean ``metric_col``.
      * ``"all"``: treat every matched run as one class and aggregate over all of
        them. Use when the leftover, non-filtered tags are nuisance variation
        (e.g. initial-parameter scale) rather than tuned hyper-parameters, so the
        HP sub-grouping would split the class spuriously.

    ``group_key`` identifies the selected group — the ``parentRunId``/param-tuple
    for ``"best_group"``, the literal ``"all"`` for ``"all"``.
    """
    best_group = select_best_group(runs, metric_col, aggregate=aggregate)
    group_key = (
        "all" if aggregate == "all"
        else _group_keys(runs).loc[best_group.index[0]]
    )

    vals = best_group[metric_col].tolist()
    all_run_ids = best_group["run_id"].tolist()
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    idx = best_group[metric_col].idxmin()
    return (
        best_group.loc[idx, "run_id"],
        float(best_group.loc[idx, metric_col]),
        vals,
        all_run_ids,
        mean,
        std,
        group_key,
    )


def select_best_group(
    runs: pd.DataFrame, rank_col: str, aggregate: str = "best_group"
) -> Optional[pd.DataFrame]:
    """Run DataFrame of the group to aggregate — the shared selector.

    ``aggregate="best_group"`` (default) returns the HP group with the lowest
    mean ``rank_col``; ``aggregate="all"`` returns every matched run as a single
    group (the stab/div tags define the whole class, so the HP sub-grouping is
    skipped). Used by both ``get_best_mean_group`` and
    ``tables.collect_nrmse_cells`` so the two stay in sync.
    """
    if aggregate not in ("best_group", "all"):
        raise ValueError(
            f"aggregate must be 'best_group' or 'all', got {aggregate!r}"
        )
    if aggregate == "all":
        return runs
    keys = _group_keys(runs)
    best_grp, best_mean = None, float("inf")
    for gk in keys.unique():
        grp = runs[keys == gk]
        col_vals = (
            grp[rank_col].dropna() if rank_col in grp.columns else pd.Series(dtype=float)
        )
        if col_vals.empty:
            continue
        m = float(col_vals.mean())
        if m < best_mean:
            best_mean, best_grp = m, grp
    return best_grp


def collect_best_runs(
    experiment_name: str,
    stab_type_dict: Dict[str, Dict],
    use_div_traj: Dict[str, Dict],
    eval_metric: str,
    search_fn: Optional[Callable] = None,
    verbose: bool = True,
    extra_tags: Optional[Dict[str, object]] = None,
    extra_filter: Optional[str] = None,
    aggregate: str = "best_group",
) -> Dict[str, Optional[dict]]:
    """Best individual run + aggregated group per (stab_type x div) combination.

    Returns a dict keyed ``"stab=<k>_div=<k>"`` mirroring the notebook's prior
    ``best_runs`` structure (value ``None`` when nothing matched). Each entry also
    carries ``group_key`` identifying the aggregated group.

    ``aggregate`` (forwarded to ``get_best_mean_group``): ``"best_group"``
    (default) aggregates the single best HP sub-group; ``"all"`` aggregates over
    every matched run, treating the stab/div tags as the whole class definition —
    use when the leftover tags are nuisance variation (e.g. initial-parameter
    scale) rather than tuned hyper-parameters. The ``indiv_*`` fields report the
    best single run over the whole class either way.

    ``extra_tags`` / ``extra_filter`` restrict the search further, e.g.
    ``extra_tags={"model.nw": 16}`` to only consider one architecture size
    (see ``fetch_runs``).
    """
    all_stab_keys = set(k for tags in stab_type_dict.values() for k in tags)
    metric_col = f"metrics.{eval_metric}"
    best_runs: Dict[str, Optional[dict]] = {}

    if verbose:
        print(f"Searching '{experiment_name}'...\n")

    for stab_key, stab_tags in stab_type_dict.items():
        for div_key, div_tags in use_div_traj.items():
            label = f"stab={stab_key}_div={div_key}"
            runs = fetch_runs(
                experiment_name, stab_tags, div_tags, all_stab_keys,
                eval_metric, search_fn=search_fn,
                extra_tags=extra_tags, extra_filter=extra_filter,
            )
            if runs is None:
                best_runs[label] = None
                if verbose:
                    print(f"  [{stab_key}/{div_key}] no runs")
                continue

            i_run_id, i_val = get_best_individual(runs, metric_col)
            m_run_id, m_val, vals, all_run_ids, mean, std, group_key = (
                get_best_mean_group(runs, metric_col, aggregate=aggregate)
            )
            if verbose:
                std_str = f"{std:.6f}" if std is not None else "N/A"
                print(
                    f"  [{stab_key}/{div_key}]  indiv={i_val:.6f}  "
                    f"mean={mean:.6f}  std={std_str}  n={len(vals)}"
                )
                print(f"      selected group [{group_key}]: {all_run_ids}")

            best_runs[label] = {
                "stab_key": stab_key,
                "div_key": div_key,
                "indiv_run_id": i_run_id,
                "indiv_nrmse": i_val,
                "run_id": m_run_id,
                "nrmse": m_val,
                "group_key": group_key,
                "all_run_ids": all_run_ids,
                "all_nrmse": vals,
                "mean": mean,
                "std": std,
            }
    return best_runs
