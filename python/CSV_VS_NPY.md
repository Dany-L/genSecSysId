# Data Loading: Original CSV vs Preprocessed NPY

## TL;DR - Which Should You Use?

**For your use case (small dataset, ~400 sequences):**

✅ **Use Direct CSV Loading** - Skip preprocessing entirely!

```bash
python scripts/train.py --config configs/rnn_direct.yaml
```

## Comparison

| Feature | Direct CSV Loading | Preprocessed NPY |
|---------|-------------------|------------------|
| **Storage** | 1× (original only) | 2× (original + NPY) |
| **Steps** | 1 (train) | 2 (preprocess + train) |
| **Load time** | ~2 seconds | ~0.1 seconds |
| **Flexibility** | High (edit CSVs anytime) | Low (need to reprocess) |
| **Data sync** | Automatic | Manual |
| **Complexity** | Simple | More complex |
| **Best for** | Small datasets, development | Large datasets, production |

## Your Workflow

### Before (with preprocessing)
```bash
# Step 1: Preprocess
python scripts/prepare_data.py --data-dir data/prepared

# Step 2: Train
python scripts/train.py --config configs/rnn_baseline.yaml
```

### After (direct loading)
```bash
# Just train!
python scripts/train.py --config configs/rnn_direct.yaml
```

**Savings:**
- ✅ 1 step instead of 2
- ✅ 50% less storage (no NPY duplication)
- ✅ No sync issues
- ✅ More flexible

## When to Use Preprocessing

Use `prepare_data.py` to create NPY files **only if**:

1. **Large dataset**: >10,000 sequences where load time matters
2. **Many models**: Training 10+ models on same data
3. **Production**: Optimizing for speed over flexibility
4. **Data validation**: Want to validate data structure upfront

## Implementation

The training script **auto-detects** your data format:

```python
# In train.py
if train_path.is_dir():
    # Load directly from CSV folders
    data = load_split_data(train_path, input_col="d", output_col="e")
elif train_path.suffix == '.npy':
    # Load preprocessed NPY files
    data = DataLoader.load_from_npy(train_path)
elif train_path.suffix == '.csv':
    # Load single CSV file
    data = DataLoader.load_from_csv(train_path)
```

## Config Examples

### Direct Loading (Recommended)
```yaml
# configs/rnn_direct.yaml
data:
  train_path: "data/prepared"  # Folder with train/test/validation
  input_col: "d"
  output_col: "e"
  pattern: "*.csv"
```

### Preprocessed NPY
```yaml
# configs/rnn_baseline.yaml
data:
  train_path: "data/prepared/train_inputs.npy"
  val_path: "data/prepared/val_inputs.npy"
  test_path: "data/prepared/test_inputs.npy"
```

## Normalization

Both methods normalize **during training** (not before):

1. Load raw data
2. Fit normalizer on training data
3. Transform batches during training
4. Save normalizer with model

This is better because:
- Normalizer consistent with loaded data
- No risk of using wrong normalization
- Normalizer saved for evaluation

## Bottom Line

**Skip preprocessing for your use case!**

The 2-second load time overhead is negligible compared to:
- Time saved not preprocessing
- Storage saved (50% less)
- Flexibility gained (edit CSVs anytime)
- Simplicity gained (one less step to maintain)

Just use `configs/rnn_direct.yaml` and train!
