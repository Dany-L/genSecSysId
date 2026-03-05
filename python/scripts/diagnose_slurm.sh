#!/bin/bash
# Diagnostic script to identify SLURM test job issues

echo "======================================"
echo "SLURM Test Diagnostics"
echo "======================================"
echo ""

# Check 1: Did the job submit?
echo "1. Checking recent jobs..."
squeue -u $USER --sort=-submit | head -5
echo ""

# Check 2: Check job status in detail
echo "2. Most recent job details:"
LATEST_JOB=$(squeue -u $USER -h -o %i --sort=-submit | head -1)
if [ ! -z "$LATEST_JOB" ]; then
    echo "Job ID: $LATEST_JOB"
    scontrol show job $LATEST_JOB | grep -E "JobState|ExitCode|WorkDir|StdOut|StdErr"
else
    echo "No recent jobs found"
fi
echo ""

# Check 3: Check for log files
echo "3. Checking for log files..."
echo "Current directory: $(pwd)"
echo "Content of ./logs/slurm/:"
if [ -d "./logs/slurm/" ]; then
    ls -lh ./logs/slurm/ | head -20
else
    echo "Directory ./logs/slurm/ does not exist"
    echo "Creating it..."
    mkdir -p ./logs/slurm/
fi
echo ""

# Check 4: Check venv exists
echo "4. Checking virtual environment paths..."
echo "Checking: /home/ac137967/venv/genSecSysId/bin/activate"
if [ -f "/home/ac137967/venv/genSecSysId/bin/activate" ]; then
    echo "✓ Found"
else
    echo "✗ NOT FOUND"
    echo "Available venv directories:"
    find /home/ac137967 -maxdepth 3 -name "activate" -type f 2>/dev/null
fi
echo ""

# Check 5: Check workspace
echo "5. Checking workspace..."
echo "Current directory: $(pwd)"
echo "Does 'scripts/train.py' exist? $([ -f scripts/train.py ] && echo '✓ Yes' || echo '✗ No')"
echo "Does 'scripts/optimize_hyperparameters.py' exist? $([ -f scripts/optimize_hyperparameters.py ] && echo '✓ Yes' || echo '✗ No')"
echo ""

# Check 6: Partition availability
echo "6. Checking available partitions..."
sinfo -o "%P %A %l" | head -10
echo ""

echo "======================================"
echo "Next steps:"
echo "1. Check squeue output above"
echo "2. If job shows in squeue, check log file path from 'scontrol show job'"
echo "3. Check venv path and update slurm_test_single_run.sh"
echo "4. Update partition if needed (currently: gpu)"
echo "======================================"
