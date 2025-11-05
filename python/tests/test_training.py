"""Tests for training utilities."""

import pytest
import torch
import numpy as np

from sysid.training import get_loss_function, get_optimizer, get_scheduler
from sysid.models import SimpleRNN


class TestLossFunctions:
    """Test loss functions."""
    
    def test_get_mse_loss(self):
        """Test MSE loss function."""
        loss_fn = get_loss_function("mse")
        
        pred = torch.randn(10, 5)
        target = torch.randn(10, 5)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
    
    def test_get_mae_loss(self):
        """Test MAE loss function."""
        loss_fn = get_loss_function("mae")
        
        pred = torch.randn(10, 5)
        target = torch.randn(10, 5)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0


class TestOptimizers:
    """Test optimizers."""
    
    @pytest.fixture
    def model(self):
        return SimpleRNN(input_size=2, hidden_size=8, output_size=1)
    
    def test_adam_optimizer(self, model):
        """Test Adam optimizer."""
        optimizer = get_optimizer(
            model.parameters(),
            optimizer_type="adam",
            learning_rate=1e-3,
        )
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 1
    
    def test_sgd_optimizer(self, model):
        """Test SGD optimizer."""
        optimizer = get_optimizer(
            model.parameters(),
            optimizer_type="sgd",
            learning_rate=1e-2,
            momentum=0.9,
        )
        
        assert optimizer is not None
        assert optimizer.param_groups[0]["momentum"] == 0.9


class TestSchedulers:
    """Test learning rate schedulers."""
    
    @pytest.fixture
    def optimizer(self):
        model = SimpleRNN(input_size=2, hidden_size=8, output_size=1)
        return get_optimizer(model.parameters(), learning_rate=1e-3)
    
    def test_step_scheduler(self, optimizer):
        """Test step scheduler."""
        scheduler = get_scheduler(
            optimizer,
            scheduler_type="step",
            step_size=10,
            gamma=0.1,
        )
        
        assert scheduler is not None
        
        # Test step
        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(10):
            scheduler.step()
        
        # LR should have decreased
        assert optimizer.param_groups[0]["lr"] < initial_lr
    
    def test_reduce_on_plateau_scheduler(self, optimizer):
        """Test ReduceLROnPlateau scheduler."""
        scheduler = get_scheduler(
            optimizer,
            scheduler_type="reduce_on_plateau",
            patience=5,
            factor=0.5,
        )
        
        assert scheduler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
