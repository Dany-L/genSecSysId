"""PyTorch Dataset for time series data."""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Dataset for time series data with input-output pairs and optional state information."""

    def __init__(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        states: Optional[np.ndarray] = None,
        sequence_length: Optional[int] = None,
        sequence_stride: Optional[int] = None,
    ):
        """
        Initialize the dataset.

        Args:
            inputs: Input data of shape (n_samples, n_timesteps, n_features) or (n_samples, n_features)
            outputs: Output data of shape (n_samples, n_timesteps, n_outputs) or (n_samples, n_outputs)
            states: Optional state data of shape (n_samples, n_timesteps, n_states) or (n_samples, n_states)
            sequence_length: Length of sequences to extract (for sliding window). If None, use full sequences.
            sequence_stride: Step size between consecutive sequence windows. If None,
                defaults to 1 in sliding-window mode.
        """
        self.inputs = torch.Tensor(inputs)
        self.outputs = torch.Tensor(outputs)
        self.states = torch.Tensor(states) if states is not None else None
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

        # Ensure 3D shape (n_samples, n_timesteps, n_features)
        if self.inputs.ndim == 2:
            self.inputs = self.inputs.unsqueeze(1)
        if self.outputs.ndim == 2:
            self.outputs = self.outputs.unsqueeze(1)
        if self.states is not None and self.states.ndim == 2:
            self.states = self.states.unsqueeze(1)

        assert (
            self.inputs.shape[0] == self.outputs.shape[0]
        ), "Input and output must have same number of samples"
        assert (
            self.inputs.shape[1] == self.outputs.shape[1]
        ), "Input and output must have same sequence length"
        if self.states is not None:
            assert (
                self.states.shape[0] == self.inputs.shape[0]
            ), "States must have same number of samples as inputs"
            assert (
                self.states.shape[1] == self.inputs.shape[1]
            ), "States must have same sequence length as inputs"

        self.n_samples = self.inputs.shape[0]
        self.n_timesteps = self.inputs.shape[1]
        self.windows_per_sample = 1

        # Calculate number of sequences if using sliding window
        if self.sequence_length is not None and self.sequence_length < self.n_timesteps:
            stride = self.sequence_stride if self.sequence_stride is not None else 1
            if stride <= 0:
                raise ValueError("sequence_stride must be a positive integer")

            self.windows_per_sample = ((self.n_timesteps - self.sequence_length) // stride) + 1
            self.n_sequences = self.n_samples * self.windows_per_sample
            self.use_sliding_window = True
        else:
            self.n_sequences = self.n_samples
            self.use_sliding_window = False

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a sequence from the dataset.

        Args:
            idx: Index of the sequence

        Returns:
            Tuple of (input_sequence, output_sequence, initial_state)
            Note: initial_state is the state at the beginning of the sequence, or None if no states were provided
        """
        if self.use_sliding_window:
            # Convert flat index to (sample_idx, start_idx)
            stride = self.sequence_stride if self.sequence_stride is not None else 1
            sample_idx = idx // self.windows_per_sample
            window_idx = idx % self.windows_per_sample
            start_idx = window_idx * stride
            end_idx = start_idx + self.sequence_length

            input_seq = self.inputs[sample_idx, start_idx:end_idx]
            output_seq = self.outputs[sample_idx, start_idx:end_idx]
            # Only return the initial state at start_idx
            initial_state = self.states[sample_idx, start_idx] if self.states is not None else None
        else:
            input_seq = self.inputs[idx]
            output_seq = self.outputs[idx]
            # Only return the initial state (first timestep)
            initial_state = self.states[idx, 0] if self.states is not None else None

        return input_seq, output_seq, initial_state
