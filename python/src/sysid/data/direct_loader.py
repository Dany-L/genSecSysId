"""Direct data loading from CSV folders (no preprocessing needed)."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import logging


logger = logging.getLogger(__name__)


def load_csv_folder(
    folder_path: str,
    input_col: str = "d",
    output_col: str = "e",
    pattern: str = "*.csv",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all CSV files from a folder directly.
    
    Args:
        folder_path: Path to folder containing CSV files
        input_col: Name of input column
        output_col: Name of output column
        pattern: Glob pattern for CSV files
        
    Returns:
        Tuple of (inputs, outputs, filenames) where inputs/outputs have shape (n_files, seq_len, 1)
    """
    folder = Path(folder_path)
    csv_files = sorted(folder.glob(pattern))
    
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {folder_path} matching pattern {pattern}")
    
    logger.info(f"Loading {len(csv_files)} CSV files from {folder_path}")
    
    all_inputs = []
    all_outputs = []
    filenames = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract columns
            inputs = df[input_col].values.reshape(-1, 1)
            outputs = df[output_col].values.reshape(-1, 1)
            
            all_inputs.append(inputs)
            all_outputs.append(outputs)
            filenames.append(csv_file.name)
            
        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")
            continue
    
    # Stack all sequences - shape: (n_files, seq_len, 1)
    all_inputs = np.array(all_inputs)
    all_outputs = np.array(all_outputs)
    
    logger.info(f"Loaded: inputs={all_inputs.shape}, outputs={all_outputs.shape}")
    
    return all_inputs, all_outputs, filenames


def load_split_data(
    data_dir: str,
    input_col: str = "d",
    output_col: str = "e",
    pattern: str = "*.csv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train, validation, and test data from folder structure.
    
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
        input_col: Name of input column
        output_col: Name of output column
        pattern: Glob pattern for CSV files
        
    Returns:
        Tuple of (train_inputs, train_outputs, val_inputs, val_outputs, test_inputs, test_outputs)
    """
    data_dir = Path(data_dir)
    
    # Load each split
    logger.info(f"Loading data from {data_dir}")
    
    train_inputs, train_outputs, _ = load_csv_folder(
        data_dir / "train", input_col, output_col, pattern
    )
    
    val_inputs, val_outputs, _ = load_csv_folder(
        data_dir / "validation", input_col, output_col, pattern
    )
    
    test_inputs, test_outputs, _ = load_csv_folder(
        data_dir / "test", input_col, output_col, pattern
    )
    
    logger.info(f"Train: {train_inputs.shape[0]} sequences")
    logger.info(f"Val: {val_inputs.shape[0]} sequences")
    logger.info(f"Test: {test_inputs.shape[0]} sequences")
    
    return train_inputs, train_outputs, val_inputs, val_outputs, test_inputs, test_outputs


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
    
    print(f"\nData loaded successfully!")
    print(f"Train: {train_in.shape}")
    print(f"Val: {val_in.shape}")
    print(f"Test: {test_in.shape}")
