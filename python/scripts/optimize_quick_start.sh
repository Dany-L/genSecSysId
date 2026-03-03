#!/bin/bash
# Quick start script for hyperparameter optimization

set -e

echo "=============================================="
echo "Hyperparameter Optimization Quick Start"
echo "=============================================="

# Check if config provided
if [ -z "$1" ]; then
    echo "Usage: ./scripts/optimize_quick_start.sh <config.yaml> [n_trials]"
    echo "Example: ./scripts/optimize_quick_start.sh ~/genSecSysId-Data/configs/crnn_gen-sec_silverbox.yaml 20"
    exit 1
fi

CONFIG=$1
N_TRIALS=${2:-20}
STUDY_NAME="quick-opt-$(date +%Y%m%d-%H%M%S)"

# Check if optuna is installed
if ! python -c "import optuna" 2>/dev/null; then
    echo "Installing optuna..."
    pip install optuna optuna-dashboard
fi

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA GPU available: {torch.cuda.get_device_name(0)}')
    print(f'  Device count: {torch.cuda.device_count()}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ Apple MPS (GPU) available')
else:
    print('⚠ No GPU available - will use CPU')
    print('  Consider using a remote GPU server for faster optimization')
"

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Number of trials: $N_TRIALS"
echo "  Study name: $STUDY_NAME"
echo ""

# Create storage directory
STORAGE_DIR="results/optimization"
mkdir -p $STORAGE_DIR
STORAGE_PATH="sqlite:///$STORAGE_DIR/optuna.db"

echo "Running optimization..."
echo "  Storage: $STORAGE_PATH"
echo ""

# Run optimization
python scripts/optimize_hyperparameters.py \
    --config "$CONFIG" \
    --n-trials $N_TRIALS \
    --study-name "$STUDY_NAME" \
    --storage "$STORAGE_PATH"

echo ""
echo "=============================================="
echo "Optimization Complete!"
echo "=============================================="
echo ""
echo "View results in dashboard:"
echo "  optuna-dashboard $STORAGE_PATH"
echo ""
echo "Best config saved to:"
echo "  $STORAGE_DIR/best_config_${STUDY_NAME}.yaml"
echo ""
echo "To train with best hyperparameters:"
echo "  python scripts/train.py --config $STORAGE_DIR/best_config_${STUDY_NAME}.yaml"
echo ""
