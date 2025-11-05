"""Tests for models."""

import pytest
import torch
import numpy as np

from sysid.models import SimpleRNN, LSTM, GRU
from sysid.models.regularization import (
    parameter_regularization,
    lipschitz_regularization,
    bounded_parameters_regularization,
    stability_regularization,
)


class TestModels:
    """Test model classes."""
    
    @pytest.fixture
    def model_params(self):
        return {
            "input_size": 2,
            "hidden_size": 16,
            "output_size": 1,
            "num_layers": 2,
            "dropout": 0.1,
        }
    
    @pytest.fixture
    def sample_input(self):
        batch_size = 4
        seq_len = 10
        input_size = 2
        return torch.randn(batch_size, seq_len, input_size)
    
    def test_simple_rnn(self, model_params, sample_input):
        """Test SimpleRNN forward pass."""
        model = SimpleRNN(**model_params)
        output = model(sample_input)
        
        assert output.shape == (sample_input.shape[0], sample_input.shape[1], model_params["output_size"])
        assert not torch.isnan(output).any()
    
    def test_lstm(self, model_params, sample_input):
        """Test LSTM forward pass."""
        model = LSTM(**model_params)
        output = model(sample_input)
        
        assert output.shape == (sample_input.shape[0], sample_input.shape[1], model_params["output_size"])
        assert not torch.isnan(output).any()
    
    def test_gru(self, model_params, sample_input):
        """Test GRU forward pass."""
        model = GRU(**model_params)
        output = model(sample_input)
        
        assert output.shape == (sample_input.shape[0], sample_input.shape[1], model_params["output_size"])
        assert not torch.isnan(output).any()
    
    def test_model_save_load(self, model_params, tmp_path):
        """Test model save and load."""
        model = SimpleRNN(**model_params)
        
        # Save model
        save_path = tmp_path / "model.pt"
        model.save(str(save_path))
        
        # Load model
        loaded_model = SimpleRNN.load(str(save_path))
        
        # Compare parameters
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_parameter_count(self, model_params):
        """Test parameter counting."""
        model = SimpleRNN(**model_params)
        count = model.count_parameters()
        
        assert count > 0
        assert isinstance(count, int)


class TestRegularization:
    """Test regularization functions."""
    
    @pytest.fixture
    def sample_model(self):
        return SimpleRNN(input_size=2, hidden_size=8, output_size=1)
    
    def test_parameter_regularization_l2(self, sample_model):
        """Test L2 parameter regularization."""
        reg_loss = parameter_regularization(sample_model, reg_type="l2", weight=0.01)
        
        assert reg_loss.item() >= 0
        assert not torch.isnan(reg_loss)
    
    def test_parameter_regularization_l1(self, sample_model):
        """Test L1 parameter regularization."""
        reg_loss = parameter_regularization(sample_model, reg_type="l1", weight=0.01)
        
        assert reg_loss.item() >= 0
        assert not torch.isnan(reg_loss)
    
    def test_lipschitz_regularization(self):
        """Test Lipschitz regularization."""
        weight_matrix = torch.randn(10, 10)
        reg_loss = lipschitz_regularization(weight_matrix, target_lipschitz=1.0)
        
        assert reg_loss.item() >= 0
        assert not torch.isnan(reg_loss)
    
    def test_bounded_parameters_regularization(self):
        """Test bounded parameters regularization."""
        param = torch.randn(10, 10)
        reg_loss = bounded_parameters_regularization(
            param,
            lower_bound=-1.0,
            upper_bound=1.0,
        )
        
        assert reg_loss.item() >= 0
        assert not torch.isnan(reg_loss)
    
    def test_stability_regularization(self):
        """Test stability regularization."""
        # Create a stable matrix (spectral radius < 1)
        recurrent_weight = 0.5 * torch.eye(10)
        reg_loss = stability_regularization(recurrent_weight, target_spectral_radius=0.9)
        
        # Loss should be zero since spectral radius is less than target
        assert reg_loss.item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
