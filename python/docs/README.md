# Documentation

This folder contains detailed guides and troubleshooting documentation for the System Identification package.

## Quick Start

- **[QUICKSTART_DIRECT.md](QUICKSTART_DIRECT.md)** - Quick start guide for direct CSV loading (recommended)
- **[DIRECT_LOADING.md](DIRECT_LOADING.md)** - Comprehensive guide for loading data directly from CSV folders

## Data & Configuration

- **[CSV_VS_NPY.md](CSV_VS_NPY.md)** - Comparison of CSV vs NPY file formats
- **[BUGFIX_CONFIG_NUMPY.md](BUGFIX_CONFIG_NUMPY.md)** - DataConfig parameters and NumPy compatibility fixes

## Evaluation & Analysis

- **[EVALUATION_METRICS.md](EVALUATION_METRICS.md)** - **NEW!** Configure which evaluation metrics to compute and log

## MLflow Integration

- **[MLFLOW_RUN_ORGANIZATION.md](MLFLOW_RUN_ORGANIZATION.md)** - Complete guide for run-based organization
- **[MLFLOW_FIX.md](MLFLOW_FIX.md)** - Solutions for MLflow connection issues (403 errors)
- **[MLFLOW_WARNINGS_FIXED.md](MLFLOW_WARNINGS_FIXED.md)** - Fixed MLflow model signature warnings
- **[ARTIFACT_PATH_WARNING_FIXED.md](ARTIFACT_PATH_WARNING_FIXED.md)** - Fixed artifact_path deprecation warning

## Bug Fixes & Changes

- **[KEYERROR_FIXED.md](KEYERROR_FIXED.md)** - Fixed missing 'best_epoch' key in training history
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** - Summary of codebase cleanup (removed preprocessing)
- **[CLEANUP_COMPLETE.md](CLEANUP_COMPLETE.md)** - Completion status of cleanup tasks

## Documentation Structure

```
docs/
├── README.md                          # This file
├── QUICKSTART_DIRECT.md              # Quick start guide
├── DIRECT_LOADING.md                 # Direct CSV loading guide
├── CSV_VS_NPY.md                     # Format comparison
├── BUGFIX_CONFIG_NUMPY.md            # Config & NumPy fixes
├── MLFLOW_FIX.md                     # MLflow troubleshooting
├── MLFLOW_WARNINGS_FIXED.md          # MLflow signature fix
├── ARTIFACT_PATH_WARNING_FIXED.md    # MLflow deprecation fix
├── KEYERROR_FIXED.md                 # Training history fix
├── CLEANUP_SUMMARY.md                # Cleanup details
└── CLEANUP_COMPLETE.md               # Cleanup status
```

## Main Documentation

For general package documentation, see:
- **[../README.md](../README.md)** - Main package README
- **[../LOGGING.md](../LOGGING.md)** - Logging system documentation

## Quick Reference

### Most Commonly Needed

1. **Getting Started**: [QUICKSTART_DIRECT.md](QUICKSTART_DIRECT.md)
2. **Data Loading**: [DIRECT_LOADING.md](DIRECT_LOADING.md)
3. **MLflow Issues**: [MLFLOW_FIX.md](MLFLOW_FIX.md)

### By Topic

- **Data Issues**: CSV_VS_NPY.md, DIRECT_LOADING.md
- **Configuration**: BUGFIX_CONFIG_NUMPY.md
- **MLflow**: MLFLOW_FIX.md, MLFLOW_WARNINGS_FIXED.md, ARTIFACT_PATH_WARNING_FIXED.md
- **Training**: KEYERROR_FIXED.md
- **Project Structure**: CLEANUP_SUMMARY.md, CLEANUP_COMPLETE.md
