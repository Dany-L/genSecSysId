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
) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]], List[str]]:
    """
    Load all CSV files from a folder as a list of per-trajectory arrays.

    Supports MIMO (Multiple Input Multiple Output) systems by accepting lists
    of column names. Variable-length trajectories are kept as-is — no padding
    is applied. Callers that need a uniform (n_files, T, n_features) tensor
    can stack the returned lists with np.stack themselves.

    Args:
        folder_path: Path to folder containing CSV files
        input_col: Name(s) of input column(s). Can be string or list of strings.
        output_col: Name(s) of output column(s). Can be string or list of strings.
        state_col: Name(s) of state column(s). Can be string, list of strings, or None.
        pattern: Glob pattern for CSV files

    Returns:
        Tuple of (inputs_list, outputs_list, states_list, filenames):
        - inputs_list:  list of np.ndarray, each shape (T_i, n_inputs)
        - outputs_list: list of np.ndarray, each shape (T_i, n_outputs)
        - states_list:  list of np.ndarray, each shape (T_i, n_states), or None
        - filenames:    list of file names (length = number of trajectories)
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

    all_inputs: List[np.ndarray] = []
    all_outputs: List[np.ndarray] = []
    all_states: Optional[List[np.ndarray]] = [] if state_cols else None
    filenames: List[str] = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            inputs = df[input_cols].values  # shape: (seq_len, n_inputs)
            outputs = df[output_cols].values  # shape: (seq_len, n_outputs)

            all_inputs.append(inputs)
            all_outputs.append(outputs)
            if state_cols:
                states = df[state_cols].values  # shape: (seq_len, n_states)
                all_states.append(states)
            filenames.append(csv_file.name)

        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")
            continue

    if not all_inputs:
        raise ValueError(f"No valid CSV files loaded from {folder_path}")

    lengths = [arr.shape[0] for arr in all_inputs]
    if len(set(lengths)) > 1:
        logger.info(
            f"Variable-length trajectories: min={min(lengths)}, max={max(lengths)} "
            f"({len(all_inputs)} files)."
        )
    else:
        logger.info(f"Loaded {len(all_inputs)} trajectories of length {lengths[0]}.")

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
    load_div: bool = False,
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
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
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
        load_div: Whether to additionally load diverging sibling folders
            (train_div/, validation_div/, test_div/). Default: False.
            Missing _div folders are tolerated (the corresponding arrays
            return None).

    Returns:
        Tuple of 18 entries (any may be None):
            train_inputs, train_outputs,
            val_inputs,   val_outputs,
            test_inputs,  test_outputs,
            train_states, val_states, test_states,
            train_div_inputs, train_div_outputs, train_div_states,
            val_div_inputs,   val_div_outputs,   val_div_states,
            test_div_inputs,  test_div_outputs,  test_div_states,
        Converging splits are stacked into 3D arrays of shape
        (n_files, T, n_features) — this requires uniform length within a
        folder. Diverging splits stay as lists of per-trajectory 2D arrays
        because their lengths differ. The trailing nine _div entries are
        always None when load_div=False.
    """
    data_dir = Path(data_dir)

    logger.info(f"Loading data from {data_dir}")

    # Initialize outputs
    train_inputs = train_outputs = train_states = None
    val_inputs = val_outputs = val_states = None
    test_inputs = test_outputs = test_states = None
    train_div_inputs = train_div_outputs = train_div_states = None
    val_div_inputs = val_div_outputs = val_div_states = None
    test_div_inputs = test_div_outputs = test_div_states = None

    def _stack_uniform(arrs, name):
        """Stack a list of (T, n) arrays into (len, T, n); error if T varies."""
        if arrs is None:
            return None
        return np.stack(arrs, axis=0)

    # Load requested splits. Converging splits are stacked (uniform length).
    if load_train:
        train_in_l, train_out_l, train_st_l, _ = load_csv_folder(
            data_dir / "train", input_col, output_col, state_col, pattern
        )
        train_inputs = _stack_uniform(train_in_l, "train")
        train_outputs = _stack_uniform(train_out_l, "train")
        train_states = _stack_uniform(train_st_l, "train")
        logger.info(f"Train: {train_inputs.shape[0]} sequences")

    if load_val:
        val_in_l, val_out_l, val_st_l, _ = load_csv_folder(
            data_dir / "validation", input_col, output_col, state_col, pattern
        )
        val_inputs = _stack_uniform(val_in_l, "validation")
        val_outputs = _stack_uniform(val_out_l, "validation")
        val_states = _stack_uniform(val_st_l, "validation")
        logger.info(f"Val: {val_inputs.shape[0]} sequences")

    if load_test:
        test_in_l, test_out_l, test_st_l, _ = load_csv_folder(
            data_dir / "test", input_col, output_col, state_col, pattern
        )
        test_inputs = _stack_uniform(test_in_l, "test")
        test_outputs = _stack_uniform(test_out_l, "test")
        test_states = _stack_uniform(test_st_l, "test")
        logger.info(f"Test: {test_inputs.shape[0]} sequences")

    # Diverging splits stay as lists — they have variable length.
    if load_div:
        for split_name, do_load in (
            ("train", load_train),
            ("validation", load_val),
            ("test", load_test),
        ):
            if not do_load:
                continue
            div_folder = data_dir / f"{split_name}_div"
            if not div_folder.exists():
                logger.info(
                    f"{split_name}_div folder not found at {div_folder}; "
                    f"no diverging trajectories will be used for this split."
                )
                continue
            div_inputs, div_outputs, div_states, _ = load_csv_folder(
                div_folder, input_col, output_col, state_col, pattern
            )
            logger.info(f"{split_name}_div: {len(div_inputs)} sequences")
            if split_name == "train":
                train_div_inputs, train_div_outputs, train_div_states = (
                    div_inputs, div_outputs, div_states,
                )
            elif split_name == "validation":
                val_div_inputs, val_div_outputs, val_div_states = (
                    div_inputs, div_outputs, div_states,
                )
            elif split_name == "test":
                test_div_inputs, test_div_outputs, test_div_states = (
                    div_inputs, div_outputs, div_states,
                )

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
        train_div_inputs,
        train_div_outputs,
        train_div_states,
        val_div_inputs,
        val_div_outputs,
        val_div_states,
        test_div_inputs,
        test_div_outputs,
        test_div_states,
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

    result = load_split_data(args.data_dir, args.input_col, args.output_col)
    train_in, _, val_in, _, test_in, *_ = result

    print("\nData loaded successfully!")
    print(f"Train: {train_in.shape}")
    print(f"Val: {val_in.shape}")
    print(f"Test: {test_in.shape}")
