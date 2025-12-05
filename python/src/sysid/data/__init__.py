"""Data loading and preprocessing utilities.

For most use cases, use direct_loader.load_split_data() to load directly
from CSV folder structures. The DataLoader class is kept for backward
compatibility with single CSV files.
"""

from .dataset import TimeSeriesDataset
from .direct_loader import load_csv_folder, load_split_data
from .loader import DataLoader, collate_with_optional_states, create_dataloaders
from .normalizer import DataNormalizer

__all__ = [
    # Primary (recommended)
    "load_split_data",
    "load_csv_folder",
    "create_dataloaders",
    "collate_with_optional_states",
    "DataNormalizer",
    "TimeSeriesDataset",
    # Legacy (backward compatibility)
    "DataLoader",
]
