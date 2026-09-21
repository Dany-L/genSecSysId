"""Tests for sysid.reporting.eval_table (the per-experiment evaluation table).

MLflow is never contacted: search results are injected as synthetic DataFrames
and the logged-config fallback as a plain dict, so config parsing, run matching
and LaTeX assembly are all covered offline. The divergence rollout is exercised
against a tiny stand-in model with the same call signature as the real one.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from sysid.reporting import eval_table


# ── config parsing ────────────────────────────────────────────────────────────
def _write(tmp_path, text):
    path = tmp_path / "results_confg.yaml"
    path.write_text(text)
    return path


def test_load_table_config_accepts_mapping_and_list_selectors(tmp_path):
    # The original file wrapped every selector in a one-element list; both that
    # and a plain mapping must normalise to the same flat dict.
    path = _write(
        tmp_path,
        """
experiments:
  - name: Exp
    mlflow_experiment_name: exp
    runs:
      - model_name: GenSec
        config:
          training.use_custom_regularization: true
          model.custom_params.learn_L: true
      - model_name: StdSec
        config:
          - training.use_custom_regularization: true
          - model.custom_params.learn_L: false
""",
    )
    cfg = eval_table.load_table_config(path)
    runs = cfg["experiments"][0]["runs"]
    assert runs[0]["selector"] == {
        "training.use_custom_regularization": True,
        "model.custom_params.learn_L": True,
    }
    assert runs[1]["selector"] == {
        "training.use_custom_regularization": True,
        "model.custom_params.learn_L": False,
    }


def test_load_table_config_defaults_and_overrides(tmp_path):
    path = _write(
        tmp_path,
        """
defaults:
  criterion: id/conv/eval_nrmse
  divergence:
    padding: 500
    eps: 0.1
experiments:
  - name: Inherits
    mlflow_experiment_name: a
    runs:
      - {model_name: GenSec, config: {k: true}}
  - name: Overrides
    mlflow_experiment_name: b
    criterion: ood/overall/eval_nrmse
    divergence:
      eps: 0.5
    runs:
      - {model_name: GenSec, config: {k: true}}
""",
    )
    a, b = eval_table.load_table_config(path)["experiments"]

    assert a["criterion"] == "id/conv/eval_nrmse"
    assert a["divergence"] == {
        "diverging_inputs": "evaluation/id/inputs_div.npy",  # module default
        "padding": 500,
        "eps": 0.1,
    }
    # A per-experiment block overrides key by key, it does not replace the block.
    assert b["criterion"] == "ood/overall/eval_nrmse"
    assert b["divergence"]["eps"] == 0.5
    assert b["divergence"]["padding"] == 500
    assert a["mlflow_server_uri"] is None


@pytest.mark.parametrize(
    "body, message",
    [
        ("experiments:\n  - {mlflow_experiment_name: a, runs: []}", "missing 'name'"),
        ("experiments:\n  - {name: A, runs: []}", "missing 'mlflow_experiment_name'"),
        (
            "experiments:\n  - name: A\n    mlflow_experiment_name: a\n"
            "    runs:\n      - {model_name: GenSec}",
            "empty 'config' selector",
        ),
        ("defaults: {}\n", "expected a mapping with an 'experiments' key"),
    ],
)
def test_load_table_config_rejects_malformed(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        eval_table.load_table_config(_write(tmp_path, body))


# ── run matching ──────────────────────────────────────────────────────────────
def test_matches_selector_coerces_tag_strings():
    # MLflow tags come back as strings, the YAML gives real bools/ints.
    run = {"a": "True", "b": "False", "nw": "20"}
    assert eval_table.matches_selector(run, {"a": True, "b": False, "nw": 20})
    assert not eval_table.matches_selector(run, {"a": False})


def test_matches_selector_is_subset_and_treats_absent_as_no_match():
    run = {"reg": True, "learn_L": True, "extra": 1}
    # Unmentioned keys are free -- this is what lets the NoSec selector omit
    # learn_L, which carries no meaning without a certificate.
    assert eval_table.matches_selector(run, {"reg": True})
    # A key the run does not carry never matches, rather than counting as false.
    assert not eval_table.matches_selector(run, {"missing": False})
    assert not eval_table.matches_selector({"reg": None}, {"reg": False})


def test_resolve_run_config_prefers_tags_and_falls_back_to_logged_config():
    calls = []

    def loader(run_id):
        calls.append(run_id)
        return {"model": {"custom_params": {"learn_L": True}}, "training": {"x": 1}}

    row = {"run_id": "r1", "tags.training.x": "9"}
    resolved = eval_table.resolve_run_config(
        row, ["training.x", "model.custom_params.learn_L"], loader
    )
    assert resolved["training.x"] == "9"          # tag wins
    assert resolved["model.custom_params.learn_L"] is True  # dug out of the config
    assert calls == ["r1"]


def test_resolve_run_config_skips_loader_when_all_tags_present():
    def loader(run_id):  # pragma: no cover - must not be reached
        raise AssertionError("should not download a config when tags suffice")

    row = {"run_id": "r1", "tags.a": "True", "tags.b": "2"}
    assert eval_table.resolve_run_config(row, ["a", "b"], loader) == {"a": "True", "b": "2"}


def test_resolve_run_config_handles_nan_tag_and_missing_loader():
    row = {"run_id": "r1", "tags.a": float("nan")}
    assert eval_table.resolve_run_config(row, ["a"], None) == {}


def _runs_frame():
    return pd.DataFrame(
        {
            "run_id": ["r1", "r2", "r3", "r4"],
            "metrics.id/conv/eval_nrmse": [0.5, 0.2, 0.9, float("nan")],
            "tags.training.use_custom_regularization": ["True", "True", "False", "True"],
            "tags.model.custom_params.learn_L": ["True", "True", "True", "True"],
            "params.trainable_parameters": ["532", "540", "500", "510"],
            "params.hidden_size": ["20", "20", "20", "20"],
        }
    )


def test_select_best_run_picks_lowest_criterion_among_matches():
    best = eval_table.select_best_run(
        _runs_frame(),
        {"training.use_custom_regularization": True, "model.custom_params.learn_L": True},
        "id/conv/eval_nrmse",
    )
    # r3 has the wrong reg tag, r4 has no metric -> r2 wins on 0.2 < 0.5.
    assert best["run_id"] == "r2"


def test_select_best_run_returns_none_when_nothing_matches_or_metric_absent():
    frame = _runs_frame()
    assert eval_table.select_best_run(frame, {"nope": True}, "id/conv/eval_nrmse") is None
    assert eval_table.select_best_run(frame, {}, "no/such/metric") is None
    assert eval_table.select_best_run(frame.iloc[:0], {}, "id/conv/eval_nrmse") is None


def test_first_metric_prefers_earlier_key_and_param_int():
    row = {"metrics.post_process/y_bar": 0.59, "params.trainable_parameters": "532"}
    assert eval_table.first_metric(row, eval_table.Y_BAR_KEYS) == pytest.approx(0.59)
    # The namespaced key wins when both spellings are present.
    row["metrics.post_process/max_s/y_bar"] = 0.42
    assert eval_table.first_metric(row, eval_table.Y_BAR_KEYS) == pytest.approx(0.42)
    assert eval_table.first_metric({}, eval_table.Y_BAR_KEYS) is None
    assert eval_table.param_int(row, "trainable_parameters") == 532
    assert eval_table.param_int(row, "missing") is None


# ── divergence rollout ────────────────────────────────────────────────────────
class _ExpandingModel(torch.nn.Module):
    """x_{k+1} = gain * x_k + d_k, with the real model's return signature.

    The gain is a scalar so the dynamics do not depend on where a trajectory
    sits in the batch -- divergence is decided purely by whether the input
    kicked the state off the origin, which keeps the chunked and unchunked
    paths comparable.
    """

    def __init__(self, gain=1.05):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(gain, dtype=torch.float64))

    def forward(self, d, x0=None, warmup_steps=0):
        batch, steps, _ = d.shape
        x = torch.zeros((batch, steps + 1, 1), dtype=d.dtype)
        for k in range(steps):
            x[:, k + 1, 0] = self.gain * x[:, k, 0] + d[:, k, 0]
        return x[:, :-1, :], (x, torch.zeros_like(d)), d


class _IdentityNormalizer:
    def transform_inputs(self, inputs):
        return inputs


def test_count_diverged_counts_states_that_do_not_return_to_origin():
    # Expanding dynamics: the trajectory that never leaves the origin stays
    # inside eps over the zero-input tail, the kicked one grows away from it.
    model = _ExpandingModel()
    inputs = np.zeros((2, 3, 1))
    inputs[1, 0, 0] = 1.0

    n_div, n_total = eval_table.count_diverged(
        model, _IdentityNormalizer(), inputs, padding=50, eps=0.1
    )
    assert (n_div, n_total) == (1, 2)


def test_count_diverged_handles_nans_and_batching_and_restores_mode():
    model = _ExpandingModel()
    model.train()
    inputs = np.zeros((4, 3, 1))
    inputs[2:, 0, 0] = 1.0  # only the last two get kicked off the origin
    inputs[:, 2, 0] = np.nan  # ragged-length padding in the stacked test array

    # batch_size < n_traj exercises the chunking path.
    n_div, n_total = eval_table.count_diverged(
        model, _IdentityNormalizer(), inputs, padding=50, eps=0.1, batch_size=2
    )
    assert (n_div, n_total) == (2, 4)
    assert model.training is True  # training mode restored


def test_count_diverged_rejects_non_3d_and_handles_empty():
    model = _ExpandingModel()
    with pytest.raises(ValueError, match=r"n_traj, n_steps, nd"):
        eval_table.count_diverged(model, _IdentityNormalizer(), np.zeros((3, 1)))
    assert eval_table.count_diverged(model, _IdentityNormalizer(), np.zeros((0, 3, 1))) == (0, 0)


def test_count_diverged_reshapes_broadcast_normalizer_output():
    # DataNormalizer broadcasts against fitted (1, 1, nd) stats, which can add a
    # leading axis; count_diverged must reshape back to a clean batch.
    class _Broadcasting:
        def transform_inputs(self, inputs):
            return np.asarray(inputs)[None, ...].reshape(1, *inputs.shape)

    model = _ExpandingModel()
    inputs = np.zeros((2, 3, 1))
    inputs[1, 0, 0] = 1.0
    assert eval_table.count_diverged(
        model, _Broadcasting(), inputs, padding=50, eps=0.1
    ) == (1, 2)


# ── LaTeX assembly ────────────────────────────────────────────────────────────
def _body_lines(tex):
    """The model rows of a rendered table (the header also has `\\` and `&`)."""
    macros = tuple(eval_table.MODEL_TO_LATEX.values())
    return [line for line in tex.splitlines() if any(m in line for m in macros)]


def _row(model_name, **kw):
    base = dict(
        experiment="Exp A",
        model_name=model_name,
        run_id=f"rid-{model_name}",
        mlflow_experiment_name="exp-a",
        nrmse=0.1234,
        y_bar=0.5,
        y_max=0.93,
        diverged=(12, 15),
        n_pars=532,
        nw=20,
    )
    base.update(kw)
    return eval_table.EvalRow(**base)


def test_build_eval_table_structure_and_template_header():
    tex = eval_table.build_eval_table([_row("GenSec")], comment="generated")
    assert tex.startswith("% generated\n")
    assert r"\begin{tabular}{rrllllll}" in tex
    assert (
        r"Experiment & Model & NRMSE & $\bar y$ & $y_{\text{max}}$ "
        r"& \# div/ total & \# pars & $\nw$\\" in tex
    )
    assert tex.rstrip().endswith(r"\end{tabular}")
    assert r"\toprule" in tex and r"\bottomrule" in tex


def test_build_eval_table_row_order_multirow_and_trace_comments():
    # Supplied in YAML order (GenSec, StdSec, NoSec); rendered in template order.
    rows = [_row("GenSec"), _row("StdSec"), _row("NoSec")]
    body = _body_lines(eval_table.build_eval_table(rows))

    assert r"\MLtiRnn{}" in body[0] and r"\MStdSec{}" in body[1] and r"\MGenSec{}" in body[2]
    # The experiment name spans the block from its first row only.
    assert body[0].startswith(r"\multirow{3}{*}{Exp A} & \MLtiRnn{}")
    assert body[1].startswith(r"& \MStdSec{}")
    # Every row is traceable back to the run it was filled from.
    assert body[0].endswith(r"\\ % run_id=rid-NoSec experiment=exp-a")
    assert "0.12 & 0.50 & 0.93 & 12/15 & 532 & 20" in body[0]


def test_build_eval_table_groups_experiments_with_midrule():
    rows = [
        _row("GenSec"),
        _row("GenSec", experiment="Exp B", mlflow_experiment_name="exp-b"),
    ]
    tex = eval_table.build_eval_table(rows)
    lines = tex.splitlines()
    # One \midrule under the header, one more between the two blocks.
    assert lines.count(r"\midrule") == 2
    assert r"\multirow{1}{*}{Exp A}" in tex and r"\multirow{1}{*}{Exp B}" in tex


def test_build_eval_table_renders_missing_values_as_dashes():
    rows = [
        _row("GenSec", diverged=None, y_bar=None),
        _row(
            "NoSec",
            run_id=None,
            nrmse=None,
            y_bar=None,
            y_max=None,
            diverged=None,
            n_pars=None,
            nw=None,
            note="no run matching {'x': False}",
        ),
    ]
    body = _body_lines(eval_table.build_eval_table(rows))

    # A run with no diverging test set still reports its other columns.
    gensec = next(line for line in body if r"\MGenSec{}" in line)
    assert "0.12 & -- & 0.93 & -- & 532 & 20" in gensec
    # A model class with no matching run at all is kept as an empty row, and the
    # comment says why instead of pointing at a run.
    nosec = next(line for line in body if r"\MLtiRnn{}" in line)
    assert nosec.count("--") == 6
    assert nosec.endswith(r"\\ % no run matching {'x': False}")


def test_build_eval_table_notes_why_a_populated_row_has_no_count():
    # A "--" on a row that has a run must say why, so "no diverging set logged"
    # is never read as "nothing diverged".
    rows = [_row("GenSec", diverged=None, note="no diverging test set logged")]
    line = _body_lines(eval_table.build_eval_table(rows))[0]
    assert line.endswith(
        r"\\ % run_id=rid-GenSec experiment=exp-a [no diverging test set logged]"
    )


def test_build_eval_table_escapes_experiment_names():
    tex = eval_table.build_eval_table([_row("GenSec", experiment="Sanity: a_b & c")])
    assert r"Sanity: a\_b \& c" in tex
