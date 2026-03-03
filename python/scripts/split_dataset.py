"""Split dataset trajectories into train/validation/test sets."""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm


def split_dataset(
    raw_dir,
    prepared_dir,
    train_ratio=0.6,
    val_ratio=0.1,
    test_ratio=0.3,
    seed=None,
):
    """
    Split trajectory files into train/validation/test sets.
    
    Args:
        raw_dir: Directory containing raw trajectory CSV files
        prepared_dir: Directory to create train/val/test subdirectories
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility
    """
    raw_dir = Path(raw_dir)
    prepared_dir = Path(prepared_dir)
    
    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative, and train_ratio must be positive")
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed set to: {seed}")
    
    # Find all trajectory files
    trajectory_files = sorted(raw_dir.glob("trajectory_*.csv"))
    
    if len(trajectory_files) == 0:
        raise ValueError(f"No trajectory files found in {raw_dir}")
    
    print(f"Found {len(trajectory_files)} trajectory files in {raw_dir}")
    
    # Shuffle trajectories
    trajectory_files = list(trajectory_files)
    np.random.shuffle(trajectory_files)
    
    # Calculate split sizes
    n_total = len(trajectory_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Remaining goes to test
    
    print(f"\nSplit sizes:")
    print(f"  Train: {n_train} ({n_train/n_total*100:.1f}%)")
    print(f"  Val:   {n_val} ({n_val/n_total*100:.1f}%)")
    print(f"  Test:  {n_test} ({n_test/n_total*100:.1f}%)")
    print()
    
    # Create subdirectories
    train_dir = prepared_dir / "train"
    val_dir = prepared_dir / "validation"
    test_dir = prepared_dir / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Split trajectories
    train_files = trajectory_files[:n_train]
    val_files = trajectory_files[n_train:n_train + n_val]
    test_files = trajectory_files[n_train + n_val:]
    
    # Copy files to respective directories
    print("Copying files to train directory...")
    for src_file in tqdm(train_files, desc="Train"):
        dst_file = train_dir / src_file.name
        shutil.copy2(src_file, dst_file)
    
    print("Copying files to validation directory...")
    for src_file in tqdm(val_files, desc="Validation"):
        dst_file = val_dir / src_file.name
        shutil.copy2(src_file, dst_file)
    
    print("Copying files to test directory...")
    for src_file in tqdm(test_files, desc="Test"):
        dst_file = test_dir / src_file.name
        shutil.copy2(src_file, dst_file)
    
    # Copy metadata if it exists
    metadata_file = raw_dir / "metadata.json"
    if metadata_file.exists():
        print("\nCopying metadata file...")
        shutil.copy2(metadata_file, prepared_dir / "metadata.json")
        
        # Update metadata with split information
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata['split'] = {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'n_total': n_total,
            'seed': seed,
        }
        
        with open(prepared_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Updated metadata saved to {prepared_dir / 'metadata.json'}")
    
    # Create split summary
    split_info = {
        'raw_dir': str(raw_dir),
        'prepared_dir': str(prepared_dir),
        'n_total': n_total,
        'split': {
            'train': {'n': n_train, 'ratio': train_ratio, 'files': [f.name for f in train_files]},
            'validation': {'n': n_val, 'ratio': val_ratio, 'files': [f.name for f in val_files]},
            'test': {'n': n_test, 'ratio': test_ratio, 'files': [f.name for f in test_files]},
        },
        'seed': seed,
    }
    
    split_info_path = prepared_dir / "split_info.json"
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✓ Split information saved to {split_info_path}")
    
    print("\n" + "="*70)
    print("Dataset split complete!")
    print("="*70)
    print(f"Source: {raw_dir}")
    print(f"Destination: {prepared_dir}")
    print(f"\nDirectory structure:")
    print(f"  {train_dir}: {n_train} files")
    print(f"  {val_dir}: {n_val} files")
    print(f"  {test_dir}: {n_test} files")
    
    return split_info


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset trajectories into train/validation/test sets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--raw-dir', type=str, required=True,
                       help='Directory containing raw trajectory CSV files')
    parser.add_argument('--prepared-dir', type=str, required=True,
                       help='Output directory for train/val/test subdirectories')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                       help='Fraction of data for training (e.g., 0.6 for 60%%)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Fraction of data for validation (e.g., 0.1 for 10%%)')
    parser.add_argument('--test-ratio', type=float, default=0.3,
                       help='Fraction of data for testing (e.g., 0.3 for 30%%)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run split
    split_dataset(
        raw_dir=args.raw_dir,
        prepared_dir=args.prepared_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
