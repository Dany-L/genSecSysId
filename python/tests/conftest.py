"""Pytest configuration."""

import os

import pytest
import torch
import numpy as np

# Newer MLflow versions raise on the filesystem tracking backend unless this
# opt-out is set. The suite (and the scripts it invokes as subprocesses, which
# inherit this env) intentionally uses file-based tracking, so opt out here.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def pytest_configure(config):
    """Register the marks the suite uses, so -m filtering is not a typo risk."""
    config.addinivalue_line(
        "markers",
        "benchmark_data: needs the nonlinear_benchmarks package and its cached "
        "downloads; skips itself when they are unavailable",
    )


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
