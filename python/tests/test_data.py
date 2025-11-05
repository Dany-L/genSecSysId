"""Tests for data loading and preprocessing."""

import pytest
import numpy as np
import torch
from pathlib import Path

from sysid.data import DataLoader, DataNormalizer, TimeSeriesDataset, create_dataloaders


class TestDataLoader:
    """Test DataLoader class."""
    
    def test_load_from_csv(self, tmp_path):
        """Test loading data from CSV file."""
        # Create temporary CSV file
        csv_path = tmp_path / "test_data.csv"
        data = np.random.randn(100, 3)
        np.savetxt(csv_path, data, delimiter=",")
        
        # Load data
        inputs, outputs = DataLoader.load_from_csv(str(csv_path))
        
        assert inputs.shape == (100, 2)
        assert outputs.shape == (100, 1)
    
    def test_load_from_npy(self, tmp_path):
        """Test loading data from NPY file."""
        # Create temporary NPY file
        npy_path = tmp_path / "test_data.npy"
        data = {
            "inputs": np.random.randn(100, 2),
            "outputs": np.random.randn(100, 1),
        }
        np.save(npy_path, data)
        
        # Load data
        inputs, outputs = DataLoader.load_from_npy(str(npy_path))
        
        assert inputs.shape == (100, 2)
        assert outputs.shape == (100, 1)


class TestDataNormalizer:
    """Test DataNormalizer class."""
    
    @pytest.fixture
    def sample_data(self):
        inputs = np.random.randn(100, 10, 2)
        outputs = np.random.randn(100, 10, 1)
        return inputs, outputs
    
    def test_minmax_normalization(self, sample_data):
        """Test min-max normalization."""
        inputs, outputs = sample_data
        
        normalizer = DataNormalizer(method="minmax", feature_range=(-1, 1))
        normalizer.fit(inputs, outputs)
        
        # Transform
        inputs_norm = normalizer.transform_inputs(inputs)
        outputs_norm = normalizer.transform_outputs(outputs)
        
        # Check range
        assert inputs_norm.min() >= -1.0
        assert inputs_norm.max() <= 1.0
        assert outputs_norm.min() >= -1.0
        assert outputs_norm.max() <= 1.0
        
        # Inverse transform
        outputs_denorm = normalizer.inverse_transform_outputs(outputs_norm)
        assert np.allclose(outputs, outputs_denorm, rtol=1e-5)
    
    def test_standard_normalization(self, sample_data):
        """Test standard normalization."""
        inputs, outputs = sample_data
        
        normalizer = DataNormalizer(method="standard")
        normalizer.fit(inputs, outputs)
        
        # Transform
        inputs_norm = normalizer.transform_inputs(inputs)
        outputs_norm = normalizer.transform_outputs(outputs)
        
        # Check mean and std (approximately)
        assert abs(inputs_norm.mean()) < 0.1
        assert abs(inputs_norm.std() - 1.0) < 0.1
        
        # Inverse transform
        outputs_denorm = normalizer.inverse_transform_outputs(outputs_norm)
        assert np.allclose(outputs, outputs_denorm, rtol=1e-5)
    
    def test_save_load(self, sample_data, tmp_path):
        """Test saving and loading normalizer."""
        inputs, outputs = sample_data
        
        normalizer = DataNormalizer(method="minmax")
        normalizer.fit(inputs, outputs)
        
        # Save
        save_path = tmp_path / "normalizer.json"
        normalizer.save(str(save_path))
        
        # Load
        loaded_normalizer = DataNormalizer.load(str(save_path))
        
        # Transform with both
        inputs_norm1 = normalizer.transform_inputs(inputs)
        inputs_norm2 = loaded_normalizer.transform_inputs(inputs)
        
        assert np.allclose(inputs_norm1, inputs_norm2)


class TestTimeSeriesDataset:
    """Test TimeSeriesDataset class."""
    
    def test_full_sequences(self):
        """Test dataset with full sequences."""
        inputs = np.random.randn(10, 20, 2)
        outputs = np.random.randn(10, 20, 1)
        
        dataset = TimeSeriesDataset(inputs, outputs)
        
        assert len(dataset) == 10
        
        input_seq, output_seq = dataset[0]
        assert input_seq.shape == (20, 2)
        assert output_seq.shape == (20, 1)
    
    def test_sliding_window(self):
        """Test dataset with sliding window."""
        inputs = np.random.randn(10, 100, 2)
        outputs = np.random.randn(10, 100, 1)
        
        sequence_length = 20
        dataset = TimeSeriesDataset(inputs, outputs, sequence_length=sequence_length)
        
        # Each sample generates (100 - 20 + 1) = 81 sequences
        assert len(dataset) == 10 * 81
        
        input_seq, output_seq = dataset[0]
        assert input_seq.shape == (20, 2)
        assert output_seq.shape == (20, 1)


class TestCreateDataloaders:
    """Test create_dataloaders function."""
    
    def test_create_dataloaders(self):
        """Test creating data loaders."""
        train_inputs = np.random.randn(100, 50, 2)
        train_outputs = np.random.randn(100, 50, 1)
        val_inputs = np.random.randn(20, 50, 2)
        val_outputs = np.random.randn(20, 50, 1)
        
        train_loader, val_loader, test_loader, normalizer = create_dataloaders(
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            val_inputs=val_inputs,
            val_outputs=val_outputs,
            batch_size=16,
            normalize=True,
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert test_loader is None
        assert normalizer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
