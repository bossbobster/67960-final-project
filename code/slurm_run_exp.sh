#!/bin/bash

# Job flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 4
#SBATCH -G h200:1
#SBATCH --time=02:00:00

# Determine script directory (handle SLURM vs local execution)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # Running under SLURM - use submission directory
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
    echo "Running under SLURM from: $SCRIPT_DIR"
else
    # Running locally - use actual script location
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    echo "Running locally from: $SCRIPT_DIR"
fi

# append also 67960-final-project to the project root
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT="$PROJECT_ROOT/67960-final-project"

# Activate virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "WARNING: Virtual environment not found at $PROJECT_ROOT/.venv"
fi

python -u "$SCRIPT_DIR/code/exp.py"