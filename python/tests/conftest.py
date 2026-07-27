"""Pytest configuration."""

import os

import pytest
import torch
import numpy as np

# Newer MLflow versions raise on the filesystem tracking backend unless this
# opt-out is set. The suite (and the scripts it invokes as subprocesses, which
# inherit this env) intentionally uses file-based tracking, so opt out here.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
