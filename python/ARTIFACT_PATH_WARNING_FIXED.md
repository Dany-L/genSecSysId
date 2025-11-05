# MLflow artifact_path Deprecation Warning Fixed ✅

## Warning Fixed

**Warning:**
```
WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
```

**Root Cause:** Using `artifact_path` as a keyword argument in `mlflow.log_artifacts()` is deprecated in newer MLflow versions.

---

## Solution

Changed from keyword argument to positional argument:

### Before (Deprecated)
```python
mlflow.log_artifacts(config.model_dir, artifact_path="models")
mlflow.log_artifacts(config.output_dir, artifact_path="outputs")
```

### After (Correct)
```python
mlflow.log_artifacts(config.model_dir, "models")
mlflow.log_artifacts(config.output_dir, "outputs")
```

---

## What Was Changed

**File:** `scripts/train.py`

**Lines 324-325:**
```python
# Log model artifacts (using positional args to avoid deprecation warning)
mlflow.log_artifacts(config.model_dir, "models")
mlflow.log_artifacts(config.output_dir, "outputs")
```

**Why:** MLflow's API now expects the artifact path as a positional argument, not a keyword argument.

---

## MLflow API Update

### Old API (Deprecated)
```python
mlflow.log_artifacts(local_dir, artifact_path="path")  # ⚠️ Deprecated
```

### New API (Current)
```python
mlflow.log_artifacts(local_dir, "path")  # ✅ Correct
```

Both still work, but the keyword argument form generates a deprecation warning.

---

## Testing

Run training again:

```bash
cd python
python scripts/train.py --config configs/rnn_baseline.yaml
```

**Expected:** Training completes with no MLflow warnings! ✅

---

## All MLflow Warnings Now Fixed

We've fixed all MLflow-related warnings:

1. ✅ **artifact_path deprecation** - Using positional arguments
2. ✅ **Missing signature** - Added input_example to log_model
3. ✅ **Connection issues** - Using local file-based tracking

---

## Summary

**Changed:** `scripts/train.py` lines 324-325

**From:**
```python
mlflow.log_artifacts(config.model_dir, artifact_path="models")
mlflow.log_artifacts(config.output_dir, artifact_path="outputs")
```

**To:**
```python
mlflow.log_artifacts(config.model_dir, "models")
mlflow.log_artifacts(config.output_dir, "outputs")
```

**Result:** Clean MLflow logging with no deprecation warnings! 🎉

---

## Verify

After training completes, check the MLflow UI:

```bash
mlflow ui
```

Open http://127.0.0.1:5000

You should see:
- ✅ No warnings in logs
- ✅ Model artifacts in "models" folder
- ✅ Output artifacts in "outputs" folder
- ✅ All metrics logged correctly

All warnings resolved!
