"""Tests for sysid.reporting.eval_table (the per-experiment evaluation table).

MLflow is never contacted: search results are injected as synthetic DataFrames
and the logged-config fallback as a plain dict, so config parsing, run matching
and LaTeX assembly are all covered offline. The divergence rollout is exercised
against a tiny stand-in model with the same call signature as the real one.
"""

from pathlib import Path

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


# ── number formatting ─────────────────────────────────────────────────────────
def test_adaptive_decimals_keeps_the_smallest_value_readable():
    # The 1-D benchmark's NRMSEs are ~2e-3: at the two decimals that suit the
    # Duffing columns every one of them prints as "0.00".
    assert eval_table.adaptive_decimals([0.00205, 0.0192, 0.0246]) == 4
    assert eval_table.adaptive_decimals([0.0643, 0.191, 0.0712]) == 3
    assert eval_table.adaptive_decimals([0.22, 1.84, 0.55]) == 2


def test_adaptive_decimals_honours_its_bounds_and_ignores_holes():
    assert eval_table.adaptive_decimals([1e-9], max_decimals=4) == 4
    assert eval_table.adaptive_decimals([123.0], min_decimals=2) == 2
    # 0.05 needs 3 decimals for two significant digits (0.050).
    assert eval_table.adaptive_decimals([None, float("nan"), 0.0, 0.05]) == 3
    assert eval_table.adaptive_decimals([]) == 2
    assert eval_table.adaptive_decimals([None, None]) == 2


def test_rank_emphasis_marks_best_then_second_best():
    # Lower is better, matching tables.build_table.
    marks = eval_table.rank_emphasis([0.3, 0.1, 0.2])
    assert marks[1] == (True, False)    # best -> bold
    assert marks[2] == (True, True)     # second -> bold italic
    assert 0 not in marks


def test_rank_emphasis_shares_a_rank_on_ties_and_skips_holes():
    marks = eval_table.rank_emphasis([0.1, 0.1, 0.5])
    assert marks[0] == (True, False) and marks[1] == (True, False)
    assert marks[2] == (True, True)     # the only runner-up value
    assert eval_table.rank_emphasis([None, 0.4]) == {1: (True, False)}
    assert eval_table.rank_emphasis([None, float("nan")]) == {}


# ── LaTeX assembly ────────────────────────────────────────────────────────────
def _rows(experiment="Exp A", mlflow_name="exp-a", nrmse=(0.3, 0.2, 0.1), y_max=0.9, **kw):
    """One full column group: NoSec, StdSec, GenSec."""
    out = []
    for model, value in zip(eval_table.ROW_ORDER, nrmse):
        base = dict(
            experiment=experiment,
            model_name=model,
            run_id=f"rid-{model}",
            mlflow_experiment_name=mlflow_name,
            nrmse=value,
            y_max=y_max,
            diverged=(12, 15),
        )
        if model == "GenSec":
            base.update(y_bar=0.5, sigma_u=1.25)
        base.update(kw.get(model, {}))
        out.append(eval_table.EvalRow(**base))
    return out


def _line(tex, prefix):
    return next(line for line in tex.splitlines() if line.startswith(prefix))


def test_experiments_are_columns_and_models_are_rows():
    tex = eval_table.build_eval_table(_rows() + _rows("Exp B", "exp-b", (0.9, 0.8, 0.7), 1.4))

    # Two experiments -> r + 2 two-column groups, matching the template's "rll".
    assert r"\begin{tabular}{rllll}" in tex
    assert r"& \multicolumn{2}{c}{Exp A} & \multicolumn{2}{c}{Exp B} \\" in tex
    assert r"Model  & NRMSE & \# div/ total & NRMSE & \# div/ total \\" in tex
    # One row per model, each carrying both experiments.
    for macro in (r"\MLtiRnn{}", r"\MStdSec{}", r"\MGenSec{}"):
        assert len([line for line in tex.splitlines() if line.startswith(macro)]) == 1


def test_single_experiment_matches_the_template_column_spec():
    tex = eval_table.build_eval_table(_rows())
    assert r"\begin{tabular}{rll}" in tex


def test_y_max_is_a_header_row_not_a_column():
    tex = eval_table.build_eval_table(_rows(y_max=0.29))
    assert r"& \multicolumn{2}{l}{$y_{\text{max}} = 0.29$} \\" in tex
    # It is no longer one of the per-model metrics.
    assert r"Model  & NRMSE & \# div/ total \\" in tex


def test_gensec_gets_its_own_spanning_certificate_rows():
    tex = eval_table.build_eval_table(_rows())
    # ybar and sigma span the whole two-column group -- they are certificate
    # quantities, not one of the per-model metrics.
    assert _line(tex, r"$\bar y$") == r"$\bar y$ & \multicolumn{2}{l}{0.5} \\"
    assert _line(tex, r"$\bar \sigma(\theta)$") == (
        r"$\bar \sigma(\theta)$ & \multicolumn{2}{l}{1.25} \\"
    )
    # A \midrule separates the GenSec block from the other two arms.
    lines = tex.splitlines()
    assert lines[lines.index(_line(tex, r"\MGenSec{}")) - 1] == r"\midrule"


def test_certificate_rows_are_dashes_without_a_gensec_run():
    rows = _rows(GenSec={"y_bar": None, "sigma_u": None})
    tex = eval_table.build_eval_table(rows)
    assert _line(tex, r"$\bar y$").endswith(r"\multicolumn{2}{l}{--} \\")
    assert _line(tex, r"$\bar \sigma(\theta)$").endswith(r"\multicolumn{2}{l}{--} \\")


def test_parameter_counts_are_no_longer_tabulated():
    tex = eval_table.build_eval_table(_rows())
    assert r"\# pars" not in tex and r"$\nw$" not in tex


def test_nrmse_precision_is_chosen_per_column():
    # Small-valued column keeps 4 decimals; the larger one stays at 2. Both in
    # the same table, because the decimals are a property of the column.
    tex = eval_table.build_eval_table(
        _rows(nrmse=(0.00205, 0.0192, 0.0246)) + _rows("Exp B", "exp-b", (0.22, 1.84, 0.55), 1.4)
    )
    nosec = _line(tex, r"\MLtiRnn{}")
    assert "0.0021" in nosec and "0.00 " not in nosec
    assert "0.22" in nosec


def test_lowest_nrmse_per_column_is_bold_and_second_lowest_bold_italic():
    tex = eval_table.build_eval_table(_rows(nrmse=(0.30, 0.20, 0.10)))
    assert r"\textbf{0.10}" in _line(tex, r"\MGenSec{}")          # best
    assert r"\textbf{\textit{0.20}}" in _line(tex, r"\MStdSec{}")  # second
    assert r"\textbf" not in _line(tex, r"\MLtiRnn{}")             # worst: plain


def test_emphasis_is_independent_per_experiment_column():
    tex = eval_table.build_eval_table(
        _rows(nrmse=(0.30, 0.20, 0.10)) + _rows("Exp B", "exp-b", (0.10, 0.20, 0.30), 1.4)
    )
    # NoSec is worst in A but best in B, so its single row carries both.
    nosec = _line(tex, r"\MLtiRnn{}")
    assert nosec.count(r"\textbf{0.10}") == 1
    assert "0.30" in nosec


def test_missing_values_render_as_dashes_and_are_not_emphasized():
    rows = _rows(NoSec={"run_id": None, "nrmse": None, "diverged": None,
                        "note": "no run matching {'x': False}"})
    tex = eval_table.build_eval_table(rows)
    nosec = _line(tex, r"\MLtiRnn{}")
    assert nosec.startswith(r"\MLtiRnn{} & -- & -- \\")
    assert r"\textbf{--}" not in tex


def test_every_row_traces_each_experiment_to_its_run():
    tex = eval_table.build_eval_table(_rows() + _rows("Exp B", "exp-b", (0.9, 0.8, 0.7), 1.4))
    assert _line(tex, r"\MGenSec{}").endswith(
        r"\\ % exp-a=rid-GenSec, exp-b=rid-GenSec"
    )


def test_trace_reports_why_a_cell_is_empty():
    rows = _rows(NoSec={"run_id": None, "nrmse": None, "note": "no run matching {'x': False}"})
    tex = eval_table.build_eval_table(rows)
    assert _line(tex, r"\MLtiRnn{}").endswith("% exp-a=<no run matching {'x': False}>")


def test_trace_keeps_a_note_alongside_a_run_that_did_produce_one():
    rows = _rows(GenSec={"note": "no diverging test set logged"})
    tex = eval_table.build_eval_table(rows)
    assert _line(tex, r"\MGenSec{}").endswith(
        "% exp-a=rid-GenSec [no diverging test set logged]"
    )


def test_escapes_experiment_names_and_rejects_empty_input():
    tex = eval_table.build_eval_table(_rows(experiment="Sanity: a_b & c"))
    assert r"Sanity: a\_b \& c" in tex
    with pytest.raises(ValueError, match="no rows"):
        eval_table.build_eval_table([])


def test_structure_matches_the_shipped_template():
    template = (
        Path(__file__).resolve().parents[1] / "results" / "eval_table_template.tex"
    )
    if not template.exists():
        pytest.skip(f"{template} not present")
    wanted = template.read_text()
    tex = eval_table.build_eval_table(_rows())
    # Every structural line of the template appears in the rendered table.
    for token in (r"\begin{tabular}{rll}", r"\toprule", r"\midrule", r"\bottomrule",
                  r"Model  & NRMSE & \# div/ total \\", r"$\bar \sigma(\theta)$",
                  r"$\bar y$", r"\MLtiRnn{}", r"\MStdSec{}", r"\MGenSec{}"):
        assert token in wanted, f"{token} missing from the template itself"
        assert token in tex, f"{token} missing from the rendered table"


# ── certificate-quantity formatting ───────────────────────────────────────────
def test_significant_keeps_three_figures_across_magnitudes():
    # These three live in the same table: y_max ~ 0.9, ybar ~ 335, sigma ~ 5e-5.
    # One shared decimal count printed y_max as 0.934495.
    assert eval_table.significant(0.934495) == "0.934"
    assert eval_table.significant(0.358288) == "0.358"
    assert eval_table.significant(335.46) == "335"
    assert eval_table.significant(19.7) == "19.7"


def test_significant_switches_to_latex_scientific_at_the_extremes():
    # A collapsed admissible input set really is ~5e-5; "0.000050" is noise.
    assert eval_table.significant(5.0e-5) == r"$5.00 \cdot 10^{-5}$"
    assert eval_table.significant(-5.0e-5) == r"$-5.00 \cdot 10^{-5}$"
    assert eval_table.significant(2.5e6) == r"$2.50 \cdot 10^{6}$"
    # ... but stays plain inside [1e-3, 1e5).
    assert eval_table.significant(1.0e-3) == "0.001"
    assert eval_table.significant(99999.0) == "1e+05"


def test_significant_handles_holes_and_zero():
    assert eval_table.significant(None) == "--"
    assert eval_table.significant(float("nan")) == "--"
    assert eval_table.significant(0.0) == "0"


def test_a_tiny_sigma_does_not_drag_the_other_certificate_cells():
    rows = _rows(y_max=0.934495, GenSec={"y_bar": 0.358288, "sigma_u": 5.0e-5})
    tex = eval_table.build_eval_table(rows)
    assert r"$y_{\text{max}} = 0.934$" in tex
    assert _line(tex, r"$\bar y$").endswith(r"{0.358} \\")
    assert r"$5.00 \cdot 10^{-5}$" in _line(tex, r"$\bar \sigma(\theta)$")
