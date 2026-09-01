"""Tests for the sysid.reporting package (run discovery + tables + plots).

MLflow is never contacted: search results are injected as synthetic
DataFrames, so the selection/aggregation/formatting logic is covered offline.
"""

import numpy as np
import pandas as pd

from sysid.reporting import runs, tables
from sysid.reporting.plots import mean_std_per_step, smooth_ema


# ── tables: NRMSE table assembly ──────────────────────────────────────────────
def test_metric_key_and_columns():
    assert tables.metric_key("id", "overall") == "id/overall/eval_nrmse"
    assert tables.metric_key("ood", "conv") == "ood/conv/eval_nrmse"
    assert len(tables.COLUMNS) == 12
    assert tables.COLUMNS[0] == ("no", "id", "conv")
    assert tables.COLUMNS[-1] == ("yes", "ood", "overall")


def test_format_cell_paren_default():
    # Default "paren" style: mean (std), no ± symbol.
    assert tables.format_cell(0.1234, 0.0056) == r"0.1234 (0.0056)"
    assert tables.format_cell(0.12, 0.01, decimals=2) == r"0.12 (0.01)"
    assert tables.format_cell(0.1234, 0.0056, bold=True) == r"\textbf{0.1234 (0.0056)}"
    assert (
        tables.format_cell(0.1234, 0.0056, bold=True, italic=True)
        == r"\textbf{\textit{0.1234 (0.0056)}}"
    )
    assert tables.format_cell(None) == "--"
    assert tables.format_cell(float("nan")) == "--"
    assert tables.format_cell(0.5, None) == r"0.5000"


def test_format_cell_other_styles_and_std_decimals():
    assert tables.format_cell(0.1234, 0.0056, std_style="pm") == r"0.1234 \ensuremath{\pm} 0.0056"
    assert tables.format_cell(0.1234, 0.0056, std_style="sub") == r"0.1234\textsubscript{\ensuremath{\pm}0.0056}"
    # std precision independent of the mean's.
    assert tables.format_cell(0.1234, 0.0056, decimals=4, std_decimals=2) == r"0.1234 (0.01)"


def test_build_table_per_column_emphasis():
    means = {r"\MLtiRnn{}": 0.3, r"\MGenSec{}": 0.1, r"\MStdSec{}": 0.2,
             r"\MGenSec{}_s": 0.4, r"\MStdSec{}_s": 0.5}
    cells = {m: {c: (means[m], 0.01, 3) for c in tables.COLUMNS} for m in tables.MODEL_ORDER}
    tex = tables.build_table(cells)
    lines = {m: next(l for l in tex.splitlines() if l.startswith(m)) for m in tables.MODEL_ORDER}
    # Lowest -> bold; second -> bold+italic; third -> plain. Default: paren, 2 dp.
    assert r"\textbf{0.10 (0.01)}" in lines[r"\MGenSec{}"]
    assert r"\textit" not in lines[r"\MGenSec{}"]
    assert r"\textbf{\textit{0.20 (0.01)}}" in lines[r"\MStdSec{}"]
    assert r"\textbf" not in lines[r"\MLtiRnn{}"]
    for m in tables.MODEL_ORDER:
        assert lines[m].count("&") == 12
    assert r"\multicolumn{6}{c}{Only conv. traj. in training}" in tex


def test_build_table_train_six_columns():
    means = {r"\MLtiRnn{}": 0.3, r"\MGenSec{}": 0.1, r"\MStdSec{}": 0.2,
             r"\MGenSec{}_s": 0.4, r"\MStdSec{}_s": 0.5}
    cells = {
        m: {("yes", d, t): (means[m], 0.01, 3) for d in tables.DISTS for t in tables.TRAJS}
        for m in tables.MODEL_ORDER
    }
    tex = tables.build_table_train(cells, "yes")
    assert tex.startswith("% Conv. and div. traj. in training")
    assert r"\begin{tabular}{rccc ccc}" in tex
    assert r"\multicolumn{3}{c}{ID}" in tex and r"\multicolumn{3}{c}{OOD}" in tex
    lines = {m: next(l for l in tex.splitlines() if l.startswith(m)) for m in tables.MODEL_ORDER}
    for m in tables.MODEL_ORDER:
        assert lines[m].count("&") == 6           # 6 data columns
    assert r"\textbf{0.10 (0.01)}" in lines[r"\MGenSec{}"]
    assert r"\textbf{\textit{0.20 (0.01)}}" in lines[r"\MStdSec{}"]
    assert r"\textbf" not in lines[r"\MLtiRnn{}"]


def test_build_split_tables_keys_and_labels():
    tabs = tables.build_split_tables({m: {} for m in tables.MODEL_ORDER})
    assert set(tabs) == {"no", "yes"}
    assert tabs["no"].startswith("% Only conv. traj. in training")
    assert tabs["yes"].startswith("% Conv. and div. traj. in training")
    line = next(l for l in tabs["no"].splitlines() if l.startswith(r"\MLtiRnn{}"))
    assert line.count("--") == 6                  # empty cells, still 6 columns


# ── runs: pure selection logic ────────────────────────────────────────────────
def test_build_filter_string():
    s = runs.build_filter_string({"tags.a": True, "tags.b": False})
    assert s == 'tags.`tags.a` = "True" and tags.`tags.b` = "False"'


def test_get_best_individual_and_mean_group():
    df = pd.DataFrame({
        "run_id": ["r1", "r2", "r3", "r4"],
        "tags.mlflow.parentRunId": ["A", "A", "B", "B"],
        "metrics.m": [0.5, 0.7, 0.2, 0.4],
    })
    assert runs.get_best_individual(df, "metrics.m") == ("r3", 0.2)
    run_id, best, vals, ids, mean, std, group_key = runs.get_best_mean_group(
        df, "metrics.m"
    )
    # group B has lower mean (0.3) than A (0.6).
    assert sorted(ids) == ["r3", "r4"]
    assert run_id == "r3" and best == 0.2
    assert group_key == "B"
    assert abs(mean - 0.3) < 1e-9
    assert abs(std - np.std([0.2, 0.4], ddof=1)) < 1e-9


def test_get_best_mean_group_aggregate_all():
    # Same runs, but treat the whole set as one class (leftover tags = nuisance).
    df = pd.DataFrame({
        "run_id": ["r1", "r2", "r3", "r4"],
        "tags.mlflow.parentRunId": ["A", "A", "B", "B"],
        "metrics.m": [0.5, 0.7, 0.2, 0.4],
    })
    run_id, best, vals, ids, mean, std, group_key = runs.get_best_mean_group(
        df, "metrics.m", aggregate="all"
    )
    # No sub-grouping: every run is aggregated, mean/std span all four.
    assert sorted(ids) == ["r1", "r2", "r3", "r4"]
    assert group_key == "all"
    assert run_id == "r3" and best == 0.2                 # best single run overall
    assert abs(mean - np.mean([0.5, 0.7, 0.2, 0.4])) < 1e-9
    assert abs(std - np.std([0.5, 0.7, 0.2, 0.4], ddof=1)) < 1e-9


def test_get_best_mean_group_rejects_bad_aggregate():
    df = pd.DataFrame({"run_id": ["r1"], "metrics.m": [0.1]})
    try:
        runs.get_best_mean_group(df, "metrics.m", aggregate="everything")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "aggregate" in str(e)


def test_collect_best_runs_aggregate_all_uses_whole_class():
    # Two parent groups; aggregate='all' must report all four runs, not the best 2.
    df = pd.DataFrame({
        "run_id": ["r1", "r2", "r3", "r4"],
        "tags.mlflow.parentRunId": ["A", "A", "B", "B"],
        "metrics.eval_nrmse": [0.5, 0.7, 0.2, 0.4],
    })
    search = lambda experiment_names, filter_string: df.copy()
    out = runs.collect_best_runs(
        "exp",
        stab_type_dict={"cls": {}},
        use_div_traj={"all": {}},
        eval_metric="eval_nrmse",
        search_fn=search, verbose=False, aggregate="all",
    )
    entry = out["stab=cls_div=all"]
    assert entry["group_key"] == "all"
    assert sorted(entry["all_run_ids"]) == ["r1", "r2", "r3", "r4"]
    assert abs(entry["mean"] - np.mean([0.5, 0.7, 0.2, 0.4])) < 1e-9
    # Default best-group mode on the same data collapses to the lower-mean pair.
    out_bg = runs.collect_best_runs(
        "exp",
        stab_type_dict={"cls": {}},
        use_div_traj={"all": {}},
        eval_metric="eval_nrmse",
        search_fn=search, verbose=False,
    )
    assert sorted(out_bg["stab=cls_div=all"]["all_run_ids"]) == ["r3", "r4"]


def test_fetch_runs_filters_absent_tags_and_missing_metric():
    # 'none' should exclude runs that carry the learn_L tag.
    df = pd.DataFrame({
        "run_id": ["r1", "r2"],
        "tags.training.use_custom_regularization": ["False", "False"],
        "tags.model.custom_params.learn_L": [np.nan, "True"],
        "metrics.eval_nrmse": [0.1, 0.2],
    })
    search = lambda experiment_names, filter_string: df.copy()
    all_keys = {"training.use_custom_regularization", "model.custom_params.learn_L"}
    out = runs.fetch_runs("exp", {"training.use_custom_regularization": False},
                          {}, all_keys, "eval_nrmse", search_fn=search)
    assert list(out["run_id"]) == ["r1"]  # r2 dropped (has learn_L tag)

    # Missing metric column -> None.
    out2 = runs.fetch_runs("exp", {}, {}, set(), "does_not_exist", search_fn=search)
    assert out2 is None


def test_fetch_runs_merges_extra_tags_and_filter():
    captured = {}

    def search(experiment_names, filter_string):
        captured["f"] = filter_string
        return pd.DataFrame({"run_id": ["r1"], "metrics.m": [0.1]})

    runs.fetch_runs(
        "exp", {"training.use_custom_regularization": True}, {"data.x": False},
        {"training.use_custom_regularization"}, "m", search_fn=search,
        extra_tags={"model.nw": 16}, extra_filter='params.hidden_size = "16"',
    )
    f = captured["f"]
    assert 'tags.`model.nw` = "16"' in f          # extra tag merged
    assert 'tags.`training.use_custom_regularization` = "True"' in f
    assert 'params.hidden_size = "16"' in f         # raw extra filter ANDed
    assert f.count(" and ") == 3                    # 4 conditions joined


def test_select_best_group_by_params_fallback():
    df = pd.DataFrame({
        "run_id": ["r1", "r2", "r3"],
        "params.lr": ["0.1", "0.1", "0.9"],
        "metrics.rank": [0.30, 0.10, 0.05],
    })
    grp = runs.select_best_group(df, "metrics.rank")
    # group lr=0.9 has the single lowest value (0.05) and lowest mean.
    assert list(grp["run_id"]) == ["r3"]


def test_select_best_group_aggregate_all_returns_every_run():
    df = pd.DataFrame({
        "run_id": ["r1", "r2", "r3"],
        "params.lr": ["0.1", "0.1", "0.9"],
        "metrics.rank": [0.30, 0.10, 0.05],
    })
    grp = runs.select_best_group(df, "metrics.rank", aggregate="all")
    assert list(grp["run_id"]) == ["r1", "r2", "r3"]      # no sub-grouping


def test_collect_nrmse_cells_aggregate_all_uses_whole_class():
    # Two HP groups (by lr); 'all' must average over all four runs, not the best.
    cols = {f"metrics.{tables.metric_key(d, t)}": [0.1, 0.2, 0.3, 0.4]
            for d in tables.DISTS for t in tables.TRAJS}
    df = pd.DataFrame({
        "run_id": ["r1", "r2", "r3", "r4"],
        "params.lr": ["0.1", "0.1", "0.9", "0.9"],
        **cols,
    })
    search = lambda experiment_names, filter_string: df.copy()
    common = dict(
        stab_type_dict={"regional": {"training.use_custom_regularization": True,
                                     "model.custom_params.learn_L": True}},
        use_div_traj={"no": {"data.use_diverging_trajectories": False}},
        stab_to_model={"regional": r"\MGenSec{}"},
        search_fn=search, verbose=False,
    )
    cells_all = tables.collect_nrmse_cells("exp", aggregate="all", **common)
    cell = cells_all[r"\MGenSec{}"][("no", "id", "conv")]
    assert cell[2] == 4                                    # n = all runs
    assert abs(cell[0] - 0.25) < 1e-9                      # mean over all four
    # Default best-group mode collapses to the lower-mean pair (lr=0.1).
    cells_bg = tables.collect_nrmse_cells("exp", **common)
    assert cells_bg[r"\MGenSec{}"][("no", "id", "conv")][2] == 2


def test_collect_nrmse_cells_aggregates_group():
    # One model class, one training condition; three repeated runs (one group).
    cols = {f"metrics.{tables.metric_key(d, t)}": [0.1, 0.2, 0.3]
            for d in tables.DISTS for t in tables.TRAJS}
    df = pd.DataFrame({"run_id": ["r1", "r2", "r3"], **cols})
    search = lambda experiment_names, filter_string: df.copy()

    cells = tables.collect_nrmse_cells(
        "exp",
        stab_type_dict={"regional": {"training.use_custom_regularization": True,
                                     "model.custom_params.learn_L": True}},
        use_div_traj={"no": {"data.use_diverging_trajectories": False}},
        stab_to_model={"regional": r"\MGenSec{}"},
        search_fn=search, verbose=False,
    )
    cell = cells[r"\MGenSec{}"][("no", "id", "conv")]
    assert abs(cell[0] - 0.2) < 1e-9                       # mean
    assert abs(cell[1] - np.std([0.1, 0.2, 0.3], ddof=1)) < 1e-9  # std
    assert cell[2] == 3                                    # n
    # 'yes' columns were never populated for this run.
    assert ("yes", "id", "conv") not in cells[r"\MGenSec{}"]


# ── plots: pure aggregation helpers ───────────────────────────────────────────
def test_smooth_ema():
    assert smooth_ema([]) == []
    assert smooth_ema([5.0]) == [5.0]
    out = smooth_ema([0.0, 1.0], alpha=0.5)
    assert out[0] == 0.0 and abs(out[1] - 0.5) < 1e-9


def test_mean_std_per_step():
    histories = [{0: 1.0, 1: 2.0}, {0: 3.0, 1: 4.0}]
    steps, means, stds = mean_std_per_step(histories)
    assert steps == [0, 1]
    assert means == [2.0, 3.0]
    assert abs(stds[0] - np.std([1.0, 3.0], ddof=1)) < 1e-9
