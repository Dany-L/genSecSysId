# Quick Start: Direct CSV Loading

## What You Need

```
data/prepared/
├── train/
│   ├── sequence_001.csv  (columns: time, d, e, x_1, x_2, ...)
│   ├── sequence_002.csv
│   └── ...
├── test/
│   └── ...
└── validation/
    └── ...
```

## Step 1: Train

```bash
cd python
python scripts/train.py --config configs/rnn_direct.yaml
```

**That's it!** No preprocessing needed.

## Step 2: Monitor Training

### Option A: Check logs
```bash
tail -f outputs/rnn_direct/training_*.log
```

### Option B: TensorBoard
```bash
tensorboard --logdir=logs/rnn_direct
```

### Option C: MLflow UI
```bash
mlflow ui
# Open http://localhost:5000
```

## Step 3: Evaluate Model

```bash
python scripts/evaluate.py \
    --config configs/rnn_direct.yaml \
    --model models/rnn_direct/best_model.pt
```

## Config Overview

```yaml
# configs/rnn_direct.yaml

# Data - Point to folder with train/test/validation subfolders
data:
  train_path: "data/prepared"  # Base folder
  input_col: "d"               # Input column name
  output_col: "e"              # Output column name
  pattern: "*.csv"             # File pattern
  normalize: true
  batch_size: 32

# Model
model:
  model_type: "rnn"            # or "lstm", "gru"
  hidden_size: 32
  num_layers: 3

# Training
training:
  max_epochs: 500
  early_stopping_patience: 30
  learning_rate: 0.001
```

## Customization

### Different columns
```yaml
data:
  input_col: "my_input"
  output_col: "my_output"
```

### Different file pattern
```yaml
data:
  pattern: "experiment_*.csv"
```

### Different model type
```yaml
model:
  model_type: "lstm"  # or "gru"
  hidden_size: 64
  num_layers: 2
```

## Troubleshooting

### "No CSV files found"
Check that `data/prepared/train/` exists and contains CSV files.

### "Column 'd' not found"
Check your CSV columns and update `input_col` and `output_col` in config.

### "Out of memory"
Reduce `batch_size` in config:
```yaml
data:
  batch_size: 16  # or 8
```

## What Happens Automatically

✅ Loads all CSV files from train/test/validation folders  
✅ Extracts specified columns (`d` for input, `e` for output)  
✅ Normalizes data using training statistics  
✅ Creates batches for training  
✅ Tracks experiments with MLflow  
✅ Logs progress to console and file  
✅ Saves best model automatically  
✅ Creates checkpoints  

## Performance

For your dataset (~400 sequences):
- **Load time**: ~2 seconds
- **Training time**: Depends on epochs, ~1-5 minutes
- **Storage**: No duplication (uses original CSV files only)

## Next Steps

After training:

### 1. Evaluate on test set
```bash
python scripts/evaluate.py \
    --config configs/rnn_direct.yaml \
    --model models/rnn_direct/best_model.pt
```

### 2. Try different architectures
```bash
# Copy config and modify
cp configs/rnn_direct.yaml configs/lstm_direct.yaml

# Edit lstm_direct.yaml:
# - Change model_type to "lstm"
# - Adjust hidden_size, num_layers
# - Change experiment name

# Train
python scripts/train.py --config configs/lstm_direct.yaml
```

### 4. Compare experiments in MLflow
```bash
mlflow ui
# Open http://localhost:5000
# Compare runs, metrics, parameters
```

## FAQ

**Q: Do I need to preprocess the data?**  
A: No! Direct loading handles everything automatically.

**Q: Can I use NPY files instead?**  
A: Yes, change `train_path` to point to `.npy` files. But for your dataset size, CSV is simpler.

**Q: Where are the trained models saved?**  
A: In `models/rnn_direct/best_model.pt`

**Q: Where are the logs?**  
A: In `outputs/rnn_direct/training_*.log`

**Q: How do I use a different model type?**  
A: Change `model_type` in config to "lstm" or "gru"

**Q: How do I speed up training?**  
A: Use GPU if available: `device: "cuda"` or `device: "mps"` (Mac M1/M2)

**Q: Can I train on multiple GPUs?**  
A: Not currently supported, but you can train multiple models in parallel with different configs.
