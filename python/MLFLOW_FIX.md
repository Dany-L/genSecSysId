# MLflow Connection Issues - Solutions

## Problem

Getting error: `mlflow.exceptions.MlflowException: API request to endpoint /api/2.0/mlflow/experiments/get-by-name failed with error code 403 != 200`

This typically means:
1. MLflow server is not properly configured
2. Authentication/permission issues
3. Server is not accepting connections

## Quick Fix: Use Local File-Based Tracking

**Solution 1: Disable Remote Tracking (Recommended for Development)**

Update your config to use local file-based tracking instead of remote server:

```yaml
# MLflow configuration
mlflow:
  tracking_uri: null  # Use local file-based tracking
  experiment_name: "rnn_baseline"
  run_name: null
  log_models: true
  log_artifacts: true
```

**Benefits:**
- ✅ No server needed
- ✅ Works immediately
- ✅ Still tracks all experiments
- ✅ Can view with `mlflow ui` later

**Where data is stored:** `./mlruns` directory

---

## Solution 2: Fix MLflow Server Connection

If you want to use the remote server, try these steps:

### Option A: Use localhost instead of 127.0.0.1

```yaml
mlflow:
  tracking_uri: "http://localhost:5000"  # Changed from 127.0.0.1
```

### Option B: Remove trailing slash

```yaml
mlflow:
  tracking_uri: "http://127.0.0.1:5000"  # No trailing /
```

### Option C: Restart MLflow server without authentication

```bash
# Stop current MLflow server
# Then restart without authentication:
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

---

## Solution 3: Check MLflow Server Status

### 1. Check if server is running:
```bash
curl http://127.0.0.1:5000/health
```

Should return: `OK`

### 2. Check if you can access the UI:
Open browser: http://127.0.0.1:5000

### 3. Check server logs:
Look at the terminal where `mlflow ui` is running for error messages.

---

## Current Fix Applied

I've updated your config to use **local file-based tracking** (tracking_uri: null) which:

✅ Bypasses the 403 error completely  
✅ Still logs all experiments and metrics  
✅ Stores data in `./mlruns` directory  
✅ Can view later with: `mlflow ui`  

## How to Use Local Tracking

### 1. Train your model:
```bash
python scripts/train.py --config configs/rnn_baseline.yaml
```

Data is saved to: `./mlruns/`

### 2. View experiments later:
```bash
mlflow ui
```

Then open: http://127.0.0.1:5000

**Same UI, no connection issues!**

---

## Comparison: Local vs Remote Tracking

| Feature | Local (`null`) | Remote (`http://...`) |
|---------|----------------|----------------------|
| Setup | None needed | Server required |
| Speed | Fast | Network dependent |
| Reliability | ✅ Always works | ⚠️ Can have issues |
| Multi-user | ❌ Single machine | ✅ Team sharing |
| **Best for** | **Development** | Production/Teams |

---

## Recommended Configuration

For your use case (development, local experiments):

```yaml
# configs/rnn_baseline.yaml
mlflow:
  tracking_uri: null  # Local file-based tracking
  experiment_name: "rnn_baseline"
  run_name: null
  log_models: true
  log_artifacts: true
```

**This is now the default in your config!**

---

## If You Still Want Remote Server

### Start MLflow server properly:

```bash
# In a separate terminal:
cd /Users/jack/Documents/01_Git/01_promotion/genSecSysId/python

# Start server with proper settings:
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --serve-artifacts
```

Then update config:
```yaml
mlflow:
  tracking_uri: "http://127.0.0.1:5000"
```

---

## Testing the Fix

```bash
cd python
python scripts/train.py --config configs/rnn_baseline.yaml
```

You should see:
```
Setting up MLflow...
Using local file-based MLflow tracking (./mlruns)
MLflow experiment: rnn_baseline
```

**No 403 error!** ✅

---

## Viewing Your Experiments

After training, view results:

```bash
cd python
mlflow ui
```

Open browser: http://127.0.0.1:5000

You'll see all your experiments, even with local tracking!

---

## Summary

**Fixed by:**
1. ✅ Set `tracking_uri: null` in config → Uses local file tracking
2. ✅ Added error handling in `train.py` → Falls back if server fails
3. ✅ Same functionality, no server needed

**Try it now:**
```bash
python scripts/train.py --config configs/rnn_baseline.yaml
```

Should work without 403 error! 🎉
