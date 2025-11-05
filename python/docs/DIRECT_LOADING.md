# Direct Loading from Original CSV Files (Recommended!)

## Why Direct Loading?

For your use case, **you don't need the preprocessing step at all!** 

### Benefits:
✅ **Simpler workflow** - No extra preprocessing step  
✅ **No storage duplication** - Work with original files  
✅ **Always fresh data** - No sync issues  
✅ **More flexible** - Easy to change data  
✅ **Less to maintain** - Fewer moving parts  

### When to preprocess (rare):
- Very large datasets (>100k sequences) where load time matters
- Training hundreds of models on same data
- Need to validate data once upfront

## Simplified Config

Instead of pointing to preprocessed NPY files, **point directly to your folder structure**:

```yaml
# Direct loading from original CSV files
data:
  train_path: "data/prepared"  # Points to folder containing train/test/validation
  input_col: "d"               # Column name for input
  output_col: "e"              # Column name for output
  pattern: "*.csv"             # Which files to load
  normalize: true
  normalization_method: "minmax"
  batch_size: 32
  sequence_length: null
  shuffle: true
  num_workers: 0
```

The script will automatically:
1. Find `data/prepared/train/`, `data/prepared/test/`, `data/prepared/validation/`
2. Load all CSV files matching the pattern
3. Extract columns `d` and `e`
4. Normalize during training (not before!)
5. Start training

## How It Works

### Old Workflow (Preprocessing)
```
Original CSV files
    ↓ python scripts/prepare_data.py
Preprocessed NPY files  
    ↓ python scripts/train.py
Training
```

### New Workflow (Direct Loading)
```
Original CSV files
    ↓ python scripts/train.py (loads directly!)
Training
```

## Example Configs

### Option 1: Direct Folder Loading (Recommended!)

```yaml
data:
  train_path: "data/prepared"  # Base folder with train/test/validation subfolders
  input_col: "d"
  output_col: "e"
  pattern: "*.csv"
  normalize: true
  normalization_method: "minmax"
  batch_size: 32
```

**Expected structure:**
```
data/prepared/
├── train/
│   ├── init_rand-output_noise_1.csv
│   ├── init_rand-output_noise_2.csv
│   └── ...
├── test/
│   └── ...
└── validation/
    └── ...
```

### Option 2: Preprocessed NPY Files (If you already have them)

```yaml
data:
  train_path: "data/prepared/train_inputs.npy"
  val_path: "data/prepared/val_inputs.npy"
  test_path: "data/prepared/test_inputs.npy"
  normalize: true
  batch_size: 32
```

### Option 3: Single CSV Files (Alternative)

```yaml
data:
  train_path: "data/prepared/train.csv"
  val_path: "data/prepared/val.csv"
  test_path: "data/prepared/test.csv"
  normalize: true
  batch_size: 32
```

## Quick Start with Direct Loading

### 1. Update your config

```yaml
# configs/rnn_baseline.yaml
data:
  train_path: "data/prepared"  # Change to folder path
  input_col: "d"               # Add column names
  output_col: "e"
  pattern: "*.csv"             # Add pattern
  normalize: true
  normalization_method: "minmax"
  batch_size: 32
  sequence_length: null
  shuffle: true
  num_workers: 0
```

### 2. Train directly!

```bash
cd python
python scripts/train.py --config configs/rnn_baseline.yaml
```

**That's it!** No preprocessing needed.

## Normalization

Normalization happens **during training**, not before:

1. **Load raw data** from CSV files
2. **Create normalizer** (fits on training data)
3. **Transform** batches during training
4. **Save normalizer** with model for later use

This is actually **better** because:
- Normalizer is always consistent with loaded data
- No risk of using wrong normalization
- Normalizer saved with model for evaluation

## Performance Comparison

For your dataset size:

| Method | Load Time | Storage | Complexity |
|--------|-----------|---------|------------|
| **Direct CSV** | ~2 seconds | 1× (original) | Simple |
| Preprocessed NPY | ~0.1 seconds | 2× (original + NPY) | Complex |

**Verdict**: Direct loading is perfect! The 2-second overhead is negligible.

## Which Format Should You Use?

### For Development/Small Datasets (Your Case!)
```yaml
# Use direct folder loading
data:
  train_path: "data/prepared"
  input_col: "d"
  output_col: "e"
```

**Why**: Simplest, no preprocessing, no duplication

### For Production/Large Datasets
```yaml
# Use preprocessed NPY
data:
  train_path: "data/prepared/train_inputs.npy"
```

**Why**: Faster loading when training many models

## Migration Guide

### If you've been using preprocessing:

**Before** (with preprocessing):
```bash
# Step 1: Preprocess
python scripts/prepare_data.py --data-dir data/prepared --output-dir data/prepared

# Step 2: Train
python scripts/train.py --config configs/rnn_baseline.yaml
```

**After** (direct loading):
```bash
# Just train! (one step)
python scripts/train.py --config configs/rnn_baseline.yaml
```

**Update config from**:
```yaml
data:
  train_path: "data/prepared/train_inputs.npy"
```

**To**:
```yaml
data:
  train_path: "data/prepared"
  input_col: "d"
  output_col: "e"
  pattern: "*.csv"
```

## Advantages Summary

| Feature | Preprocessing | Direct Loading |
|---------|--------------|----------------|
| Steps required | 2 | **1** |
| Storage | 2× | **1×** |
| Data sync | Manual | **Automatic** |
| Flexibility | Low | **High** |
| Load speed | Fast (0.1s) | Good (2s) |
| Simplicity | Complex | **Simple** |
| **Best for** | Large datasets | **Your use case!** |

## Complete Example

```yaml
# configs/rnn_baseline_direct.yaml
data:
  # Point to folder with train/test/validation subfolders
  train_path: "data/prepared"
  
  # Specify which columns to use
  input_col: "d"
  output_col: "e"
  
  # File pattern
  pattern: "*.csv"
  
  # Normalization (happens during training)
  normalize: true
  normalization_method: "minmax"
  
  # Training settings
  batch_size: 32
  sequence_length: null
  shuffle: true
  num_workers: 0

model:
  model_type: "rnn"
  input_size: 1
  hidden_size: 32
  output_size: 1
  num_layers: 3
  dropout: 0.0
  activation: "tanh"

optimizer:
  optimizer_type: "adam"
  learning_rate: 0.001
  weight_decay: 0.00001
  use_scheduler: true
  scheduler_type: "reduce_on_plateau"
  scheduler_patience: 10
  scheduler_factor: 0.5

training:
  max_epochs: 500
  early_stopping_patience: 30
  checkpoint_frequency: 100
  gradient_clip_value: 1.0
  loss_type: "mse"
  use_custom_regularization: false
  device: "cpu"
  log_interval: 10
  use_tensorboard: true

mlflow:
  tracking_uri: "http://127.0.0.1:5000"
  experiment_name: "rnn_baseline_direct"
  run_name: null
  log_models: true
  log_artifacts: true

output_dir: "outputs/rnn_baseline_direct"
model_dir: "models/rnn_baseline_direct"
log_dir: "logs/rnn_baseline_direct"

seed: 42
```

## Summary

**For your use case with ~400 CSV files**:

✅ **Use direct loading** - Simpler, no preprocessing needed  
✅ **Point config to folder** - Let script find the files  
✅ **Normalization during training** - More robust  
✅ **Save 50% storage** - No duplication  
✅ **One less step** - Fewer things to break  

The `prepare_data.py` script is now **optional** - use it only if you want preprocessed NPY files for faster loading when training many models!
