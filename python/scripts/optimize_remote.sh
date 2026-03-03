#!/bin/bash
# Script to deploy and run optimization on remote GPU server

# Configuration
REMOTE_USER="your-username"
REMOTE_HOST="gpu-server.example.com"
REMOTE_DIR="~/genSecSysId"
LOCAL_CODE_DIR="~/Documents/01_Git/01_promotion/genSecSysId/python"
LOCAL_DATA_DIR="~/genSecSysId-Data"

echo "=============================================="
echo "Remote GPU Hyperparameter Optimization"
echo "=============================================="
echo ""
echo "Remote: $REMOTE_USER@$REMOTE_HOST"
echo "Target: $REMOTE_DIR"
echo ""

# Check if remote host is provided
if [ "$REMOTE_HOST" == "gpu-server.example.com" ]; then
    echo "ERROR: Please edit this script and set REMOTE_USER and REMOTE_HOST"
    echo "Example:"
    echo "  REMOTE_USER=\"jsmith\""
    echo "  REMOTE_HOST=\"ml-server.university.edu\""
    exit 1
fi

# Parse arguments
CONFIG_NAME=${1:-"crnn_gen-sec_silverbox.yaml"}
N_TRIALS=${2:-50}

echo "Step 1: Syncing code to remote server..."
rsync -avz --exclude='mlruns/' --exclude='*.pt' --exclude='__pycache__/' \
    --exclude='results/' --exclude='.git/' \
    $LOCAL_CODE_DIR/ \
    $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/python/

echo ""
echo "Step 2: Syncing data (configs only, not large data files)..."
rsync -avz --include='configs/***' --include='n4sid_params.mat' \
    --exclude='data/' --exclude='models/' --exclude='*.db' \
    $LOCAL_DATA_DIR/ \
    $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/data/

# Note: If you need full data, remove the --exclude='data/' flag above
# Or manually sync large data files once:
# rsync -avz ~/genSecSysId-Data/data/ user@host:~/genSecSysId/data/data/

echo ""
echo "Step 3: Setting up environment on remote..."
ssh $REMOTE_USER@$REMOTE_HOST << 'ENDSSH'
cd ~/genSecSysId/python

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install optuna optuna-dashboard

# Verify GPU
echo ""
echo "GPU Status:"
nvidia-smi || echo "No NVIDIA GPU detected"
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"

ENDSSH

echo ""
echo "Step 4: Starting optimization in tmux session..."

# Create tmux session name based on config
SESSION_NAME="optuna-$(basename $CONFIG_NAME .yaml)"

ssh -t $REMOTE_USER@$REMOTE_HOST << ENDSSH
cd ~/genSecSysId/python
source venv/bin/activate

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Start new tmux session
tmux new-session -d -s $SESSION_NAME

# Run optimization in tmux
tmux send-keys -t $SESSION_NAME "cd ~/genSecSysId/python" C-m
tmux send-keys -t $SESSION_NAME "source venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "python scripts/optimize_hyperparameters.py \
    --config ~/genSecSysId/data/configs/$CONFIG_NAME \
    --n-trials $N_TRIALS \
    --study-name ${SESSION_NAME} \
    --storage sqlite:///~/genSecSysId/data/optuna.db" C-m

echo ""
echo "=============================================="
echo "Optimization started!"
echo "=============================================="
echo ""
echo "Tmux session: $SESSION_NAME"
echo ""
echo "To check progress:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from tmux: Ctrl+B, then D"
echo ""
echo "To monitor GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "To sync results back:"
echo "  rsync -avz $REMOTE_USER@$REMOTE_HOST:~/genSecSysId/data/optuna.db $LOCAL_DATA_DIR/"
echo "  rsync -avz $REMOTE_USER@$REMOTE_HOST:~/genSecSysId/python/results/ $LOCAL_CODE_DIR/results/"
echo ""

ENDSSH

echo ""
echo "Connecting to view progress..."
echo "(Exit with Ctrl+D or type 'exit')"
sleep 2
ssh -t $REMOTE_USER@$REMOTE_HOST "tmux attach -t $SESSION_NAME"
