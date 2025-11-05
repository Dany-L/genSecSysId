"""Data loading and preprocessing utilities.

For most use cases, use direct_loader.load_split_data() to load directly
from CSV folder structures. The DataLoader class is kept for backward
compatibility with single CSV files.
"""

from .loader import DataLoader, create_dataloaders
from .direct_loader import load_split_data, load_csv_folder
from .normalizer import DataNormalizer
from .dataset import TimeSeriesDataset

__all__ = [
    # Primary (recommended)
    "load_split_data",
    "load_csv_folder",
    "create_dataloaders",
    "DataNormalizer",
    "TimeSeriesDataset",
    # Legacy (backward compatibility)
    "DataLoader",
]
