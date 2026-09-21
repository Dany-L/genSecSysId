"""Tests for the generic nonlinear_benchmarks adapter.

Nothing here downloads: the registry's `fetch` hooks are the only part that
touches the network, and every test below drives `prepare` with synthetic
records instead. The one exception is a test marked `benchmark_data`, which
checks the real package's return shapes and is skipped when it is unavailable.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sysid.data import benchmark_prep as bp
from sysid.data.benchmark_registry import (
    REGISTRY,
    SUPPORTED,
    BenchmarkSpec,
    Record,
    apply_declared_split,
    get,
    normalize_records,
    rename,
)
from sysid.data.direct_loader import load_split_data

REPO_PY = Path(__file__).resolve().parents[1]
prep = importlib.util.spec_from_file_location(
    "prepare_benchmark", REPO_PY / "scripts" / "prepare_benchmark.py"
)
prepare_benchmark = importlib.util.module_from_spec(prep)
prep.loader.exec_module(prepare_benchmark)


def _rec(name, n, nd=1, ne=1, seed=0):
    rng = np.random.default_rng(seed)
    return Record(name, rng.standard_normal((n, nd)), rng.standard_normal((n, ne)))


# ── the registry itself ───────────────────────────────────────────────────────
class TestRegistry:
    def test_every_entry_is_either_supported_or_explains_itself(self):
        for name, spec in REGISTRY.items():
            assert spec.name == name
            if not spec.supported:
                # An unsupported entry exists precisely to make the gap visible,
                # so the reason is the whole point of it.
                assert spec.unsupported and len(spec.unsupported) > 20
            else:
                assert spec.fetch is not None

    def test_get_rejects_unknown_and_unsupported_with_a_reason(self):
        with pytest.raises(KeyError, match="unknown benchmark"):
            get("NoSuchBenchmark")
        with pytest.raises(ValueError, match="not zero-mean"):
            get("Cascaded_Tanks")
        with pytest.raises(ValueError, match="NO estimation data"):
            get("BoucWen")

    def test_supported_set_is_what_we_expect(self):
        assert set(SUPPORTED) == {
            "CED", "EMPS", "Silverbox", "WienerHammerBenchMark", "ParWHF", "F16"
        }

    def test_benchmarks_without_an_official_split_are_flagged(self):
        # Reading a number off a self-declared split as if it were the
        # benchmark's own is the mistake this flag exists to prevent.
        assert REGISTRY["F16"].official_split is False
        assert REGISTRY["ParWHF"].official_split is False
        assert REGISTRY["Silverbox"].official_split is True


# ── flattening the package's return shapes ────────────────────────────────────
class TestNormalizeRecords:
    class _Fake:
        def __init__(self, u, y, ts=None):
            self.u, self.y, self.sampling_time, self.name = u, y, ts, "ignored"

    def test_pair_of_lists(self):
        f = self._Fake(np.zeros(10), np.zeros(10), ts=0.5)
        train, test, ts = normalize_records(([f, f], [f]), "X")
        assert len(train) == 2 and len(test) == 1 and ts == 0.5

    def test_flat_list_has_no_test_side(self):
        f = self._Fake(np.zeros(10), np.zeros(10))
        train, test, _ = normalize_records([f, f, f], "X")
        assert len(train) == 3 and test == []

    def test_single_record_and_nesting_are_both_flattened(self):
        f = self._Fake(np.zeros(10), np.zeros(10))
        assert len(normalize_records(f, "X")[0]) == 1
        assert len(normalize_records(([[f, f], [f]], [f]), "X")[0]) == 3

    def test_1d_signals_are_promoted_to_2d(self):
        f = self._Fake(np.zeros(10), np.zeros(10))
        train, _, _ = normalize_records([f], "X")
        assert train[0].u.shape == (10, 1) and train[0].y.shape == (10, 1)

    def test_names_come_from_position_not_the_name_attribute(self):
        # v0.1.2 labels Silverbox's second test record 'test SB multisine' when
        # it is the arrow record, so .name cannot be trusted.
        f = self._Fake(np.zeros(10), np.zeros(10))
        train, _, _ = normalize_records([f, f], "X")
        assert [r.name for r in train] == ["X_0000", "X_0001"]
        assert "ignored" not in train[0].name

    def test_rename_checks_the_count(self):
        recs = [_rec("a", 5), _rec("b", 5)]
        assert [r.name for r in rename(recs, ["x", "y"])] == ["x", "y"]
        with pytest.raises(ValueError, match="2 records"):
            rename(recs, ["only_one"])


# ── split policy ──────────────────────────────────────────────────────────────
class TestSplitPolicy:
    def test_single_record_splits_within_itself(self):
        train, val = bp.split_records([_rec("solo", 1000)], 0.2)
        assert len(train) == 1 and len(val) == 1
        assert len(train[0].u) == 800 and len(val[0].u) == 200

    def test_many_records_split_by_record(self):
        # ParWHF's realizations are independent experiments: cutting inside one
        # would straddle the split while leaving the other 199 untouched.
        recs = [_rec(f"r{i}", 100, seed=i) for i in range(10)]
        train, val = bp.split_records(recs, 0.2)
        assert len(train) == 8 and len(val) == 2
        assert all(len(r.u) == 100 for r in train + val)
        assert [r.name for r in val] == ["r8", "r9"]

    def test_record_split_is_the_tail_and_keeps_at_least_one_each_side(self):
        recs = [_rec(f"r{i}", 50, seed=i) for i in range(2)]
        train, val = bp.split_records(recs, 0.9)
        assert len(train) == 1 and len(val) == 1

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="no records"):
            bp.split_records([], 0.2)


# ── the ragged-test rule ──────────────────────────────────────────────────────
class TestRaggedTestRule:
    def test_uniform_records_can_share_a_folder(self, tmp_path):
        recs = [_rec("a", 50), _rec("b", 50)]
        n = bp.write_records(tmp_path / "test", recs, ["u"], ["y"])
        assert n == 2

    def test_ragged_records_raise_with_the_folder_named(self, tmp_path):
        # load_split_data's own failure is a bare "all input arrays must have
        # the same shape" that names no folder, which is the trap this replaces.
        with pytest.raises(ValueError, match=r"test/ would hold records of different"):
            bp.write_records(tmp_path / "test", [_rec("a", 50), _rec("b", 60)],
                             ["u"], ["y"])

    def test_every_record_gets_a_sibling_and_the_selected_one_lands_in_test(self, tmp_path):
        recs = [_rec("multisine", 50), _rec("arrow", 60)]
        files = bp.write_test_records(tmp_path, recs, ["u"], ["y"], selected="arrow")
        assert (tmp_path / "test_multisine").is_dir()
        assert (tmp_path / "test_arrow").is_dir()
        assert files["test"] == 1
        # test/ holds the selected record, ragged siblings notwithstanding.
        import pandas as pd
        assert len(pd.read_csv(next((tmp_path / "test").glob("*.csv")))) == 60

    def test_unknown_selection_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            bp.write_test_records(tmp_path, [_rec("a", 50)], ["u"], ["y"], selected="nope")


# ── declared splits ───────────────────────────────────────────────────────────
class TestDeclaredSplit:
    def test_f16_style_declaration_picks_named_records(self):
        spec = BenchmarkSpec(name="X", declared_train_record="tr",
                             declared_test_records=("te1", "te2"))
        recs = [_rec(n, 10) for n in ("tr", "te1", "te2", "unused")]
        train, test = apply_declared_split(spec, recs)
        assert [r.name for r in train] == ["tr"]
        assert [r.name for r in test] == ["te1", "te2"]

    def test_only_declared_records_are_materialized(self):
        # The F-16 has 20 records of 49k-117k samples; writing them all as
        # siblings would cost hundreds of MB for records nothing references.
        spec = BenchmarkSpec(name="X", declared_train_record="tr",
                             declared_test_records=("te1",))
        recs = [_rec(n, 10) for n in ("tr", "te1", "other")]
        train, test = apply_declared_split(spec, recs)
        assert len(train) + len(test) == 2

    def test_a_missing_declared_record_names_itself(self):
        spec = BenchmarkSpec(name="X", declared_train_record="absent")
        with pytest.raises(KeyError, match="absent"):
            apply_declared_split(spec, [_rec("tr", 10)])


# ── end-to-end layout ─────────────────────────────────────────────────────────
@pytest.fixture
def prepared(tmp_path):
    spec = BenchmarkSpec(name="Synthetic", output_cols=("y",), init_window=7,
                         notes="synthetic", reference="none")
    train = [_rec("run", 1000)]
    test = [_rec("t_short", 200, seed=1), _rec("t_long", 400, seed=2)]
    meta = prepare_benchmark.prepare(spec, tmp_path, train, test, 0.01,
                                     val_fraction=0.2, test_record="t_short")
    return tmp_path, meta


class TestPreparedLayout:
    def test_loader_reads_it_back(self, prepared):
        out, _ = prepared
        res = load_split_data(str(out), input_col=["u"], output_col=["y"])
        assert res[0].shape == (1, 800, 1)     # train
        assert res[2].shape == (1, 200, 1)     # validation
        assert res[4].shape == (1, 200, 1)     # test == t_short

    def test_metadata_records_what_a_reader_needs(self, prepared):
        out, meta = prepared
        assert meta["test_record_in_test_folder"] == "t_short"
        assert meta["benchmark_state_initialization_window_length"] == 7
        # tests/test_benchmark_datasets.py reads n_samples off this.
        assert meta["records"]["train"]["n_samples"] == 800
        assert "no validation set" in meta["validation_note"]
        assert json.loads(json.dumps(meta))  # JSON-serializable

    def test_ragged_siblings_all_exist(self, prepared):
        out, _ = prepared
        assert (out / "test_t_short").is_dir() and (out / "test_t_long").is_dir()

    def test_rerun_does_not_leave_stale_files(self, prepared, tmp_path):
        out, _ = prepared
        spec = BenchmarkSpec(name="Synthetic")
        before = len(list((out / "train").glob("*.csv")))
        prepare_benchmark.prepare(spec, out, [_rec("run", 1000)],
                                  [_rec("t_short", 200)], 0.01,
                                  subsequence_length=100)
        after = len(list((out / "train").glob("*.csv")))
        assert before == 1 and after == 8  # 800 // 100, cleanly replaced


# ── against the real package ──────────────────────────────────────────────────
@pytest.mark.benchmark_data
class TestAgainstThePackage:
    """Shapes the registry assumes, checked against nonlinear_benchmarks itself."""

    @pytest.mark.parametrize("name", SUPPORTED)
    def test_fetch_returns_usable_records(self, name):
        pytest.importorskip("nonlinear_benchmarks")
        spec = REGISTRY[name]
        try:
            train, test, ts = spec.fetch()
        except Exception as exc:  # not cached / no network
            pytest.skip(f"{name} unavailable: {exc}")
        assert train, f"{name} returned no training records"
        for rec in train[:3] + test[:3]:
            assert rec.u.ndim == 2 and rec.y.ndim == 2
            assert len(rec.u) == len(rec.y)
            assert rec.u.shape[1] == len(spec.input_cols)
            assert rec.y.shape[1] == len(spec.output_cols)
        if not test:
            assert spec.declared_train_record, f"{name} has no test side and no declaration"


# ── generated sweep files ─────────────────────────────────────────────────────
class TestGeneratedSweep:
    """The sweep file has to be valid input to scripts/sweep.py, not just YAML."""

    @staticmethod
    def _sweep(tmp_path, n_seeds=3):
        import yaml
        from sysid.data import benchmark_config as bc

        spec = REGISTRY["Silverbox"]
        text = bc.render_sweep(spec, "/data/work/me/benchmarks/configs/crnn_silverbox.yaml",
                               b2c2=0.075, n_seeds=n_seeds)
        path = tmp_path / "sweep_silverbox.yaml"
        path.write_text(text)
        return path, yaml.safe_load(text)

    def test_three_arms_in_separate_groups(self, tmp_path):
        _, cfg = self._sweep(tmp_path)
        groups = cfg["search_space"]
        assert len(groups) == 3
        nosec, gensec, stdsec = groups
        # NoSec must NOT carry learn_L: with the barrier off it changes nothing,
        # so including it would just duplicate every NoSec run.
        assert nosec["training.use_custom_regularization"] == [False]
        assert "model.custom_params.learn_L" not in nosec
        assert gensec["model.custom_params.learn_L"] == [True]
        assert stdsec["model.custom_params.learn_L"] == [False]
        assert gensec["training.use_custom_regularization"] == [True]

    def test_sweep_py_counts_the_tasks_we_expect(self, tmp_path):
        # Groups CONCATENATE rather than multiply, so 3 arms x n_seeds.
        path, _ = self._sweep(tmp_path, n_seeds=4)
        sweep = importlib.util.spec_from_file_location(
            "sweep", REPO_PY / "scripts" / "sweep.py"
        )
        mod = importlib.util.module_from_spec(sweep)
        sweep.loader.exec_module(mod)
        import yaml
        assert mod.n_tasks(yaml.safe_load(path.read_text())) == 12

    def test_base_config_and_seeds_are_carried_through(self, tmp_path):
        _, cfg = self._sweep(tmp_path, n_seeds=5)
        assert cfg["base_config"] == "/data/work/me/benchmarks/configs/crnn_silverbox.yaml"
        assert cfg["n_seeds"] == 5
        assert cfg["sweep_name"] == "Silverbox"
