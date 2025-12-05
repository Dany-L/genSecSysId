"""Direct data loading from CSV folders (no preprocessing needed)."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv_folder(
    folder_path: str,
    input_col: Union[str, List[str]] = "d",
    output_col: Union[str, List[str]] = "e",
    state_col: Union[str, List[str], None] = None,
    pattern: str = "*.csv",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Load all CSV files from a folder directly.

    Supports MIMO (Multiple Input Multiple Output) systems by accepting lists of column names.

    Args:
        folder_path: Path to folder containing CSV files
        input_col: Name(s) of input column(s). Can be string or list of strings.
        output_col: Name(s) of output column(s). Can be string or list of strings.
        state_col: Name(s) of state column(s). Can be string, list of strings, or None.
        pattern: Glob pattern for CSV files

    Returns:
        Tuple of (inputs, outputs, states, filenames) where:
        - inputs shape: (n_files, seq_len, n_inputs)
        - outputs shape: (n_files, seq_len, n_outputs)
        - states shape: (n_files, seq_len, n_states) or None if no state columns
        - filenames: List of file names
    """
    folder = Path(folder_path)
    csv_files = sorted(folder.glob(pattern))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {folder_path} matching pattern {pattern}")

    # Convert to lists if single strings provided
    input_cols = [input_col] if isinstance(input_col, str) else input_col
    output_cols = [output_col] if isinstance(output_col, str) else output_col

    # Handle state columns - None or empty list means no states
    if state_col is not None and len(state_col) > 0 if isinstance(state_col, list) else True:
        state_cols = [state_col] if isinstance(state_col, str) else state_col
    else:
        state_cols = None

    logger.info(f"Loading {len(csv_files)} CSV files from {folder_path}")
    logger.info(f"  Input columns: {input_cols}")
    logger.info(f"  Output columns: {output_cols}")
    if state_cols:
        logger.info(f"  State columns: {state_cols}")

    all_inputs = []
    all_outputs = []
    all_states = [] if state_cols else None
    filenames = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Extract input columns (support MIMO)
            inputs = df[input_cols].values  # shape: (seq_len, n_inputs)

            # Extract output columns (support MIMO)
            outputs = df[output_cols].values  # shape: (seq_len, n_outputs)

            # Extract state columns if provided
            if state_cols:
                states = df[state_cols].values  # shape: (seq_len, n_states)
                all_states.append(states)

            all_inputs.append(inputs)
            all_outputs.append(outputs)
            filenames.append(csv_file.name)

        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")
            continue

    # Stack all sequences
    all_inputs = np.array(all_inputs)  # shape: (n_files, seq_len, n_inputs)
    all_outputs = np.array(all_outputs)  # shape: (n_files, seq_len, n_outputs)

    if state_cols:
        all_states = np.array(all_states)  # shape: (n_files, seq_len, n_states)
        logger.info(
            f"Loaded: inputs={all_inputs.shape}, outputs={all_outputs.shape}, states={all_states.shape}"
        )
    else:
        logger.info(f"Loaded: inputs={all_inputs.shape}, outputs={all_outputs.shape}")

    return all_inputs, all_outputs, all_states, filenames


def load_split_data(
    data_dir: str,
    input_col: Union[str, List[str]] = "d",
    output_col: Union[str, List[str]] = "e",
    state_col: Union[str, List[str], None] = None,
    pattern: str = "*.csv",
    load_train: bool = True,
    load_val: bool = True,
    load_test: bool = True,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Load train, validation, and/or test data from folder structure.

    Supports MIMO systems and optional state information.
    Flexible loading: can load only specific splits (useful for training vs evaluation).

    Expects folder structure:
        data_dir/
            train/
                *.csv
            test/
                *.csv
            validation/
                *.csv

    Args:
        data_dir: Base directory containing train/test/validation folders
        input_col: Name(s) of input column(s). Can be string or list of strings.
        output_col: Name(s) of output column(s). Can be string or list of strings.
        state_col: Name(s) of state column(s). Can be string, list of strings, or None.
        pattern: Glob pattern for CSV files
        load_train: Whether to load training data (default: True)
        load_val: Whether to load validation data (default: True)
        load_test: Whether to load test data (default: True)

    Returns:
        Tuple of (train_inputs, train_outputs, val_inputs, val_outputs, test_inputs, test_outputs,
                  train_states, val_states, test_states)
        Note: Arrays are None for splits that are not loaded
        Note: state arrays are None if state_col is None

    Examples:
        # For training (skip test data)
        train_in, train_out, val_in, val_out, _, _, train_s, val_s, _ = load_split_data(
            data_dir, load_test=False
        )

        # For evaluation (only test data)
        _, _, _, _, test_in, test_out, _, _, test_s = load_split_data(
            data_dir, load_train=False, load_val=False
        )
    """
    data_dir = Path(data_dir)

    logger.info(f"Loading data from {data_dir}")

    # Initialize outputs
    train_inputs = train_outputs = train_states = None
    val_inputs = val_outputs = val_states = None
    test_inputs = test_outputs = test_states = None

    # Load requested splits
    if load_train:
        train_inputs, train_outputs, train_states, _ = load_csv_folder(
            data_dir / "train", input_col, output_col, state_col, pattern
        )
        logger.info(f"Train: {train_inputs.shape[0]} sequences")

    if load_val:
        val_inputs, val_outputs, val_states, _ = load_csv_folder(
            data_dir / "validation", input_col, output_col, state_col, pattern
        )
        logger.info(f"Val: {val_inputs.shape[0]} sequences")

    if load_test:
        test_inputs, test_outputs, test_states, _ = load_csv_folder(
            data_dir / "test", input_col, output_col, state_col, pattern
        )
        logger.info(f"Test: {test_inputs.shape[0]} sequences")

    return (
        train_inputs,
        train_outputs,
        val_inputs,
        val_outputs,
        test_inputs,
        test_outputs,
        train_states,
        val_states,
        test_states,
    )


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Test direct CSV loading")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--input-col", type=str, default="d", help="Input column")
    parser.add_argument("--output-col", type=str, default="e", help="Output column")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train_in, train_out, val_in, val_out, test_in, test_out = load_split_data(
        args.data_dir, args.input_col, args.output_col
    )

    print("\nData loaded successfully!")
    print(f"Train: {train_in.shape}")
    print(f"Val: {val_in.shape}")
    print(f"Test: {test_in.shape}")
