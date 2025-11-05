"""PyTorch Dataset for time series data."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional


class TimeSeriesDataset(Dataset):
    """Dataset for time series data with input-output pairs."""
    
    def __init__(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        sequence_length: Optional[int] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            inputs: Input data of shape (n_samples, n_timesteps, n_features) or (n_samples, n_features)
            outputs: Output data of shape (n_samples, n_timesteps, n_outputs) or (n_samples, n_outputs)
            sequence_length: Length of sequences to extract (for sliding window). If None, use full sequences.
        """
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)
        self.sequence_length = sequence_length
        
        # Ensure 3D shape (n_samples, n_timesteps, n_features)
        if self.inputs.ndim == 2:
            self.inputs = self.inputs.unsqueeze(1)
        if self.outputs.ndim == 2:
            self.outputs = self.outputs.unsqueeze(1)
        
        assert self.inputs.shape[0] == self.outputs.shape[0], "Input and output must have same number of samples"
        assert self.inputs.shape[1] == self.outputs.shape[1], "Input and output must have same sequence length"
        
        self.n_samples = self.inputs.shape[0]
        self.n_timesteps = self.inputs.shape[1]
        
        # Calculate number of sequences if using sliding window
        if self.sequence_length is not None and self.sequence_length < self.n_timesteps:
            self.n_sequences = self.n_samples * (self.n_timesteps - self.sequence_length + 1)
            self.use_sliding_window = True
        else:
            self.n_sequences = self.n_samples
            self.use_sliding_window = False
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self.n_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence from the dataset.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (input_sequence, output_sequence)
        """
        if self.use_sliding_window:
            # Convert flat index to (sample_idx, start_idx)
            sample_idx = idx // (self.n_timesteps - self.sequence_length + 1)
            start_idx = idx % (self.n_timesteps - self.sequence_length + 1)
            end_idx = start_idx + self.sequence_length
            
            input_seq = self.inputs[sample_idx, start_idx:end_idx]
            output_seq = self.outputs[sample_idx, start_idx:end_idx]
        else:
            input_seq = self.inputs[idx]
            output_seq = self.outputs[idx]
        
        return input_seq, output_seq
