"""Tests for evaluation metrics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sysid.data.normalizer import DataNormalizer
from sysid.evaluation.metrics import compute_metrics

_EVALUATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("evaluate", _EVALUATE_PATH)
_evaluate = importlib.util.module_from_spec(_spec)
# execute_module would trigger main(); we only need the helpers, so we import
# the module but guard against the argparse call that runs at module level.
_spec.loader.exec_module(_evaluate)
metric_category_for_path = _evaluate.metric_category_for_path


class TestMetricCategoryForPath:
    """metric_category_for_path must use exact component matching, not substring."""

    def test_ood_component_classified_as_ood(self):
        assert metric_category_for_path(Path("/data/SmokeData/ood")) == "ood"

    def test_id_component_classified_as_id(self):
        assert metric_category_for_path(Path("/data/SmokeData/id")) == "id"

    def test_good_runs_not_misclassified_as_ood(self):
        """'good' contains 'ood' as a substring — must not be classified as OOD."""
        assert metric_category_for_path(Path("/data/good_runs/test")) == "id"

    def test_ood_case_insensitive(self):
        assert metric_category_for_path(Path("/data/OOD/test")) == "ood"
        assert metric_category_for_path(Path("/data/Ood/test")) == "ood"

    def test_ood_nested_inside_path(self):
        assert metric_category_for_path(Path("/root/SmokeData/ood/test")) == "ood"

    def test_no_ood_component_is_id(self):
        assert metric_category_for_path(Path("/data/SmokeData/train")) == "id"


class TestComputeMetricsOutputScale:
    """Tests for training-data-based NRMSE normalization."""

    def _perfect_pred(self, n=50, n_feat=1):
        e = np.random.randn(n, n_feat)
        return e.copy(), e.copy()

    def test_fallback_uses_test_range(self):
        """Without output_scale, NRMSE denominator is max-min of test data."""
        e = np.array([[0.0], [1.0], [2.0]])
        e_hat = np.array([[0.1], [1.1], [2.1]])
        metrics = compute_metrics(e_hat, e)
        rmse = np.sqrt(np.mean(0.01))
        expected_nrmse = rmse / (2.0 - 0.0)
        assert metrics["nrmse"] == pytest.approx(expected_nrmse, rel=1e-6)

    def test_output_scale_overrides_test_range(self):
        """With output_scale provided, NRMSE uses that denominator instead."""
        e = np.array([[0.0], [1.0], [2.0]])
        e_hat = np.array([[0.1], [1.1], [2.1]])
        training_scale = 10.0
        metrics = compute_metrics(e_hat, e, output_scale=training_scale)
        rmse = np.sqrt(np.mean(0.01))
        expected_nrmse = rmse / training_scale
        assert metrics["nrmse"] == pytest.approx(expected_nrmse, rel=1e-6)

    def test_nrmse_lower_for_large_output_scale(self):
        """Larger output_scale (e.g. from wide-range training data) yields smaller NRMSE."""
        e = np.array([[0.0], [1.0]])
        e_hat = np.array([[0.5], [1.5]])
        m_small = compute_metrics(e_hat, e, output_scale=1.0)
        m_large = compute_metrics(e_hat, e, output_scale=10.0)
        assert m_large["nrmse"] < m_small["nrmse"]

    def test_rmse_unaffected_by_output_scale(self):
        """output_scale must not affect RMSE or other non-normalized metrics."""
        e = np.random.randn(20, 2)
        e_hat = e + 0.1
        m_no_scale = compute_metrics(e_hat, e)
        m_scaled = compute_metrics(e_hat, e, output_scale=5.0)
        assert m_no_scale["rmse"] == pytest.approx(m_scaled["rmse"], rel=1e-9)
        assert m_no_scale["mse"] == pytest.approx(m_scaled["mse"], rel=1e-9)
        assert m_no_scale["mae"] == pytest.approx(m_scaled["mae"], rel=1e-9)


class TestNormalizerGetOutputScale:
    """Tests for DataNormalizer.get_output_scale()."""

    def _fit(self, method, outputs):
        n = DataNormalizer(method=method)
        inputs = np.zeros_like(outputs)
        n.fit(inputs, outputs)
        return n

    def test_scale_only_returns_mean_std(self):
        outputs = np.random.randn(100, 10, 2) * np.array([[[2.0, 4.0]]])
        n = self._fit("scale_only", outputs)
        expected = float(np.mean(n.output_std))
        assert n.get_output_scale() == pytest.approx(expected, rel=1e-9)

    def test_standard_returns_mean_std(self):
        outputs = np.random.randn(100, 10, 2) * 3.0
        n = self._fit("standard", outputs)
        expected = float(np.mean(n.output_std))
        assert n.get_output_scale() == pytest.approx(expected, rel=1e-9)

    def test_minmax_returns_mean_range(self):
        outputs = np.random.randn(100, 10, 2)
        n = self._fit("minmax", outputs)
        expected = float(np.mean(n.output_max - n.output_min))
        assert n.get_output_scale() == pytest.approx(expected, rel=1e-9)

    def test_not_fitted_raises(self):
        n = DataNormalizer(method="scale_only")
        with pytest.raises(RuntimeError, match="fitted"):
            n.get_output_scale()

    def test_scale_is_positive(self):
        for method in ("scale_only", "standard", "minmax"):
            outputs = np.random.randn(50, 5, 3)
            n = self._fit(method, outputs)
            assert n.get_output_scale() > 0
