"""Tests for training utilities."""

import pytest
import torch
import numpy as np

from sysid.training import get_loss_function, get_optimizer, get_scheduler
from sysid.training.trainer import Trainer
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


class TestRegularizationDecay:
    """Test interior-point-method decay of regularization weights."""

    def _make_trainer(self, tmp_path, **kwargs):
        model = SimpleRNN(input_size=2, hidden_size=8, output_size=1)
        optimizer = get_optimizer(model.parameters(), learning_rate=1e-3)
        return Trainer(
            model=model,
            train_loader=None,
            val_loader=None,
            loss_fn=get_loss_function("mse"),
            optimizer=optimizer,
            device="cpu",
            output_dir=str(tmp_path / "outputs"),
            model_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
            mlflow_tracking=False,
            **kwargs,
        )

    def test_decays_both_feasibility_and_input_weights(self, tmp_path):
        """decay_regularization should decay the feasibility AND input weights."""
        trainer = self._make_trainer(
            tmp_path,
            regularization_weight=1.0,
            input_regularization_weight=0.1,
            decay_regularization_weight=True,
            regularization_decay_factor=0.5,
            min_regularization_weight=1e-7,
        )

        trainer.decay_regularization()

        assert trainer.regularization_weight == pytest.approx(0.5)
        assert trainer.input_regularization_weight == pytest.approx(0.05)

    def test_input_weight_clamped_to_minimum(self, tmp_path):
        """Input weight should not decay below the minimum threshold."""
        trainer = self._make_trainer(
            tmp_path,
            regularization_weight=1.0,
            input_regularization_weight=1e-7,
            decay_regularization_weight=True,
            regularization_decay_factor=0.5,
            min_regularization_weight=1e-6,
        )

        trainer.decay_regularization()

        assert trainer.input_regularization_weight == pytest.approx(1e-6)

    def test_no_decay_when_flag_disabled(self, tmp_path):
        """Neither weight changes when decay is disabled."""
        trainer = self._make_trainer(
            tmp_path,
            regularization_weight=1.0,
            input_regularization_weight=0.1,
            decay_regularization_weight=False,
            regularization_decay_factor=0.5,
        )

        trainer.decay_regularization()

        assert trainer.regularization_weight == pytest.approx(1.0)
        assert trainer.input_regularization_weight == pytest.approx(0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
