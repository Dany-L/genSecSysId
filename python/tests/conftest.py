"""Pytest configuration."""

import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
