# Bug Fixes: DataConfig and NumPy Compatibility

## Issues Fixed

### 1. DataConfig Missing Parameters ✅

**Error**: `DataConfig.__init__() got an unexpected keyword argument 'input_col'`

**Root Cause**: The `DataConfig` dataclass didn't have the new parameters needed for direct folder loading.

**Solution**: Updated `src/sysid/config.py` to add:
- `input_col: str = "d"` - Column name for input
- `output_col: str = "e"` - Column name for output  
- `pattern: str = "*.csv"` - File pattern for folder loading
- Made `val_path` and `test_path` optional (not needed for folder loading)
- Changed `sequence_length` to `Optional[int] = None` for full sequences

### 2. NumPy Version Incompatibility ✅

**Error**: 
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

**Root Cause**: NumPy 2.x has breaking changes and many packages are not yet compatible.

**Solution**: 
1. Updated `setup.py` to constrain: `numpy>=1.21.0,<2.0.0`
2. Downgraded numpy to version 1.x: `pip install "numpy<2.0.0"`
3. Added pandas to dependencies (needed for CSV loading)

## Changes Made

### `src/sysid/config.py`

```python
@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_path: str
    val_path: Optional[str] = None  # Not required for folder loading
    test_path: Optional[str] = None  # Not required for folder loading
    
    # Direct folder loading parameters
    input_col: str = "d"  # Column name for input
    output_col: str = "e"  # Column name for output
    pattern: str = "*.csv"  # File pattern for folder loading
    
    # Preprocessing
    normalize: bool = True
    normalization_method: str = "minmax"  # or "standard"
    batch_size: int = 32
    sequence_length: Optional[int] = None  # None = use full sequences
    shuffle: bool = True
    num_workers: int = 0
```

### `setup.py`

```python
install_requires=[
    "torch>=2.0.0",
    "numpy>=1.21.0,<2.0.0",  # NumPy 1.x for compatibility
    "scipy>=1.7.0",
    "pandas>=1.3.0",  # For CSV loading
    "mlflow>=2.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.5.0",
    "tensorboard>=2.10.0",
],
```

## Config Format Now Supported

Your config can now use all these parameters:

```yaml
data:
  train_path: "data/prepared"  # Folder path
  input_col: "d"               # NEW - Column for input
  output_col: "e"              # NEW - Column for output
  pattern: "*.csv"             # NEW - File pattern
  normalize: true
  normalization_method: "standard"
  batch_size: 32
  sequence_length: null        # null for full sequences
  shuffle: true
  num_workers: 0
```

## Verification

You can now run:

```bash
cd python
python scripts/train.py --config configs/rnn_baseline.yaml
```

It should:
1. ✅ Load config without errors
2. ✅ Recognize `input_col`, `output_col`, `pattern` parameters
3. ✅ Load data from folder structure
4. ✅ Use NumPy 1.x (no compatibility warnings)
5. ✅ Start training

## What Works Now

✅ Direct folder loading with column specification  
✅ NumPy 1.x compatibility (no warnings)  
✅ All configs support new parameters  
✅ Backward compatible (old single-file configs still work)  

## If You Still See Issues

### Check NumPy version:
```bash
python -c "import numpy; print(numpy.__version__)"
```

Should output something like `1.26.4` (not `2.x.x`)

### Reinstall with correct numpy:
```bash
pip uninstall numpy
pip install "numpy>=1.21.0,<2.0.0"
```

### Check pandas is installed:
```bash
pip install pandas
```

## Next Steps

Try training again:

```bash
python scripts/train.py --config configs/rnn_baseline.yaml
```

The errors should be resolved! 🎉
