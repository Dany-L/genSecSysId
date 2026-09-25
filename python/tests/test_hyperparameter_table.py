"""Tests for sysid.reporting.hyperparameter_table and its script.

MLflow is never contacted: runs are plain dicts shaped like ``search_runs``
rows, the logged config a nested dict.
"""

import importlib.util
from pathlib import Path

import pytest

from sysid.reporting import hyperparameter_table as hp

REPO_PY = Path(__file__).resolve().parents[1]


def _write(tmp_path, text):
    path = tmp_path / "hyperparameter_config.yaml"
    path.write_text(text)
    return path


# ── config ────────────────────────────────────────────────────────────────────
def test_load_normalises_columns(tmp_path):
    cfg = hp.load_hyperparameter_config(
        _write(
            tmp_path,
            """
results_config: results/results_confg.yaml
columns:
  - {label: Epochs, source: metric, key: final_epoch, format: int, offset: 1}
  - {label: lr, source: param, key: [lr, learning_rate], format: sci, blank_for: [LSTM]}
""",
        )
    )
    assert cfg["results_config"] == "results/results_confg.yaml"
    epochs, lr = cfg["columns"]
    assert epochs["keys"] == ["final_epoch"] and epochs["offset"] == 1.0
    assert epochs["blank_for"] == []
    assert lr["keys"] == ["lr", "learning_rate"] and lr["blank_for"] == ["LSTM"]


@pytest.mark.parametrize(
    "column, match",
    [
        ("{label: x, source: nope, key: k, format: int}", "source"),
        ("{label: x, source: param, key: k, format: nope}", "format"),
        ("{label: x, source: param, format: int}", "key"),
    ],
)
def test_load_rejects_bad_columns(tmp_path, column, match):
    with pytest.raises(ValueError, match=match):
        hp.load_hyperparameter_config(_write(tmp_path, f"columns:\n  - {column}\n"))


def test_shipped_config_loads():
    cfg = hp.load_hyperparameter_config(REPO_PY / "results" / "hyperparameter_config.yaml")
    labels = [c["label"] for c in cfg["columns"]]
    assert labels == [
        r"$n_\theta$", "Time", "Warmup", "Batch", "$n_w$", r"$\lambda$",
    ]
    # T_s (the only config-sourced column) is commented out, so no run's
    # outputs/config.yaml has to be downloaded
    assert not hp.needs_logged_config(cfg["columns"])


# ── values ────────────────────────────────────────────────────────────────────
RUN = {
    "params.trainable_parameters": "782",
    "params.learning_rate": "0.005",
    "params.regularization_weight": "0.001",
    "metrics.final_epoch": 999.0,
    "metrics.total_train_time_sec": float("nan"),
    "tags.model.type": "crnn",
}
RUN_CONFIG = {"data": {"sampling_time": 0.05}}


def _col(source, key, fmt="sig", **kw):
    keys = key if isinstance(key, list) else [key]
    return {"label": kw.pop("label", str(key)), "source": source, "keys": keys,
            "format": fmt, "offset": kw.pop("offset", 0.0), "blank_for": kw.pop("blank_for", [])}


def test_column_value_reads_each_source():
    assert hp.column_value(_col("param", "trainable_parameters"), RUN) == 782.0
    assert hp.column_value(_col("metric", "final_epoch", offset=1.0), RUN) == 1000.0
    assert hp.column_value(_col("config", "data.sampling_time"), RUN, RUN_CONFIG) == 0.05
    # A tag that is not a number is not a value.
    assert hp.column_value(_col("tag", "model.type"), RUN) is None


def test_column_value_treats_nan_and_absent_as_missing_and_falls_back():
    assert hp.column_value(_col("metric", "total_train_time_sec"), RUN) is None
    assert hp.column_value(_col("config", "data.sampling_time"), RUN, None) is None
    assert hp.column_value(_col("param", ["lr", "learning_rate"]), RUN) == 0.005


def test_row_values_blanks_the_columns_a_model_has_no_use_for():
    cols = [_col("param", "regularization_weight", "sci", label="reg",
                 blank_for=["NoSec", "LSTM"])]
    assert hp.row_values(cols, "GenSec", RUN) == {"reg": 0.001}
    assert hp.row_values(cols, "LSTM", RUN) == {"reg": None}


# ── formatting ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "seconds, text",
    [(None, "--"), (42.4, "42 s"), (90, "1.5 min"), (3599, "60.0 min"), (23040, "6.4 h")],
)
def test_format_duration(seconds, text):
    assert hp.format_duration(seconds) == text


@pytest.mark.parametrize(
    "value, text",
    [
        (0.005, r"$5 \cdot 10^{-3}$"),
        (0.001, r"$10^{-3}$"),
        (2.5e-4, r"$2.5 \cdot 10^{-4}$"),
        (0.00999, r"$10^{-2}$"),  # rounding carries into the next decade
        (0.0, "0"),
        (None, "--"),
    ],
)
def test_format_sci(value, text):
    assert hp.format_sci(value) == text


def test_format_value_dispatches_on_format():
    assert hp.format_value(12961.0, "int") == "12961"
    assert hp.format_value(0.001638404194, "sig") == "0.00164"
    assert hp.format_value(0.05, "sig") == "0.05"
    assert hp.format_value(None, "int") == "--"


# ── LaTeX assembly ────────────────────────────────────────────────────────────
COLUMNS = [
    _col("param", "trainable_parameters", "int", label=r"$n_\theta$"),
    _col("param", "learning_rate", "sci", label="lr"),
]


def _row(experiment, model, n=100.0, run_id=None, note=None):
    return hp.HyperRow(
        experiment=experiment,
        model_name=model,
        run_id=run_id if run_id is not None else f"rid-{model}",
        mlflow_experiment_name=experiment.lower(),
        values={r"$n_\theta$": n, "lr": 0.005},
        note=note,
    )


def test_table_groups_models_per_experiment_in_row_order():
    rows = [_row("Duffing", m) for m in ("GenSec", "LSTM", "NoSec", "StdSec")]
    rows += [_row("One_D", "GenSec")]
    tex = hp.build_hyperparameter_table(rows, COLUMNS, comment="hi")
    lines = tex.splitlines()
    assert lines[0] == "% hi"
    assert lines[1] == r"\begin{tabular}{llrr}"
    assert lines[3] == r"Dataset & Model & $n_\theta$ & lr \\"
    body = lines[5:-2]
    assert body[0] == (
        r"\multirow{4}{*}{Duffing} & \MLtiRnn{} & 100 & $5 \cdot 10^{-3}$ \\ % duffing=rid-NoSec"
    )
    assert [line.split(" & ")[1] for line in body[:4]] == [
        r"\MLtiRnn{}", r"\MStdSec{}", r"\MLstm{}", r"\MGenSec{}",
    ]
    assert all(line.startswith(" & ") for line in body[1:4])
    # A \midrule between experiments, the name escaped, a one-row multirow.
    assert body[4] == r"\midrule"
    assert body[5].startswith(r"\multirow{1}{*}{One\_D} & \MGenSec{}")
    assert lines[-2:] == [r"\bottomrule", r"\end{tabular}"]


def test_table_renders_missing_runs_as_dashes_with_the_reason():
    row = hp.HyperRow("Exp", "NoSec", mlflow_experiment_name="exp", note="no run matching x")
    tex = hp.build_hyperparameter_table([row], COLUMNS)
    assert r"\MLtiRnn{} & -- & -- \\ % exp=<no run matching x>" in tex


def test_table_rejects_empty_input():
    with pytest.raises(ValueError, match="no rows"):
        hp.build_hyperparameter_table([], COLUMNS)


# ── the script ────────────────────────────────────────────────────────────────
def _load_script(monkeypatch):
    # The script imports its sibling generate_eval_table by name, as it does
    # when run from the command line.
    monkeypatch.syspath_prepend(str(REPO_PY / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "generate_hyperparameter_table", REPO_PY / "scripts" / "generate_hyperparameter_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_builds_rows_from_the_shared_run_selection(monkeypatch):
    script = _load_script(monkeypatch)
    experiment = {"name": "Duffing", "mlflow_experiment_name": "duffing-soft-12"}
    best = {"run_id": "abc", **RUN}
    monkeypatch.setattr(
        script,
        "iter_best_runs",
        lambda config, only=None: iter(
            [(experiment, "GenSec", best, None), (experiment, "LSTM", None, "no run matching y")]
        ),
    )
    monkeypatch.setattr(script.ConfigLoader, "__call__", lambda self, run_id: RUN_CONFIG)
    cols = [
        _col("param", "trainable_parameters", "int", label="n"),
        _col("config", "data.sampling_time", label="Ts"),
        _col("metric", "total_train_time_sec", "duration", label="t"),
    ]
    gensec, lstm = script.build_rows({}, cols)
    assert gensec.run_id == "abc" and gensec.values == {"n": 782.0, "Ts": 0.05, "t": None}
    assert gensec.note == "not logged: t"
    assert lstm.run_id is None and lstm.note == "no run matching y"
