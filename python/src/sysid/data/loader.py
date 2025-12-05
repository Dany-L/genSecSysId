"""Data loading utilities.

For loading data from CSV files directly, use direct_loader.py instead.
This module is kept for backward compatibility with single CSV files.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from .dataset import TimeSeriesDataset
from .normalizer import DataNormalizer

torch.set_default_dtype(torch.float64)


def collate_with_optional_states(batch: List[Tuple]) -> Tuple:
    """Custom collate function that handles optional initial state tensors.

    When states are None, the third element in the returned tuple is None
    instead of trying to stack None values.

    Args:
        batch: List of tuples (inputs, outputs, initial_state) where initial_state can be None

    Returns:
        Tuple of (batched_inputs, batched_outputs, batched_initial_states or None)
    """
    # Separate the elements
    inputs = [item[0] for item in batch]
    outputs = [item[1] for item in batch]
    initial_states = [item[2] for item in batch]

    # Stack inputs and outputs
    batched_inputs = torch.stack(inputs)
    batched_outputs = torch.stack(outputs)

    # Stack initial states only if they're not None
    if initial_states[0] is not None:
        batched_initial_states = torch.stack(initial_states)
    else:
        batched_initial_states = None

    return batched_inputs, batched_outputs, batched_initial_states


class DataLoader:
    """Load and preprocess time series data from single CSV files.

    For loading from folder structures, use direct_loader.load_split_data() instead.
    """

    @staticmethod
    def load_from_csv(
        path: str,
        input_columns: Optional[list] = None,
        output_columns: Optional[list] = None,
        delimiter: str = ",",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from a single CSV file.

        Note: For loading from folder structures with multiple CSV files,
        use direct_loader.load_split_data() instead.

        Args:
            path: Path to CSV file
            input_columns: List of column indices for inputs (if None, use all but last)
            output_columns: List of column indices for outputs (if None, use last column)
            delimiter: CSV delimiter

        Returns:
            Tuple of (inputs, outputs)
        """
        data = np.loadtxt(path, delimiter=delimiter)

        if input_columns is None:
            inputs = data[:, :-1]
        else:
            inputs = data[:, input_columns]

        if output_columns is None:
            outputs = data[:, -1:]
        else:
            outputs = data[:, output_columns]

        # Ensure proper shape
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        if outputs.ndim == 1:
            outputs = outputs.reshape(-1, 1)

        return inputs, outputs


def create_dataloaders(
    train_inputs: np.ndarray,
    train_outputs: np.ndarray,
    val_inputs: np.ndarray,
    val_outputs: np.ndarray,
    test_inputs: Optional[np.ndarray] = None,
    test_outputs: Optional[np.ndarray] = None,
    train_states: Optional[np.ndarray] = None,
    val_states: Optional[np.ndarray] = None,
    test_states: Optional[np.ndarray] = None,
    batch_size: int = 32,
    sequence_length: Optional[int] = None,
    normalize: bool = True,
    normalization_method: str = "minmax",
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[TorchDataLoader, TorchDataLoader, Optional[TorchDataLoader], Optional[DataNormalizer]]:
    """
    Create train, validation, and test data loaders.

    Args:
        train_inputs: Training input data
        train_outputs: Training output data
        val_inputs: Validation input data
        val_outputs: Validation output data
        test_inputs: Test input data (optional)
        test_outputs: Test output data (optional)
        batch_size: Batch size
        sequence_length: Sequence length for sliding window
        normalize: Whether to normalize data
        normalization_method: Normalization method
        shuffle: Whether to shuffle training data
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader, normalizer)
    """
    normalizer = None

    if normalize:
        normalizer = DataNormalizer(method=normalization_method)
        normalizer.fit(train_inputs, train_outputs)

        train_inputs = normalizer.transform_inputs(train_inputs)
        train_outputs = normalizer.transform_outputs(train_outputs)
        val_inputs = normalizer.transform_inputs(val_inputs)
        val_outputs = normalizer.transform_outputs(val_outputs)

        if test_inputs is not None:
            test_inputs = normalizer.transform_inputs(test_inputs)
            test_outputs = normalizer.transform_outputs(test_outputs)

    # Create datasets
    train_dataset = TimeSeriesDataset(train_inputs, train_outputs, train_states, sequence_length)
    val_dataset = TimeSeriesDataset(val_inputs, val_outputs, val_states, sequence_length)

    # Create data loaders
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_with_optional_states,
    )

    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_optional_states,
    )

    test_loader = None
    if test_inputs is not None:
        test_dataset = TimeSeriesDataset(test_inputs, test_outputs, test_states, sequence_length)
        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_with_optional_states,
        )

    return train_loader, val_loader, test_loader, normalizer
