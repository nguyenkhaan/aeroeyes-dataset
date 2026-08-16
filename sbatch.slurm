#!/bin/bash
#SBATCH --job-name=aeroeyes_job
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=mps:a100:2
#SBATCH --mem=12G
#SBATCH --time=24:00:00

# ===============================
# Load environment
# ===============================

module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1
module load shared python312


# ===============================
# Python environment
# ===============================

PYTHON=/datastore/cndt_khanhnd/aeroeyes_cloudian/aeroeyes-dataset/venv/bin/python

if [ ! -x "$PYTHON" ]; then
    echo "Configured Python is not executable: $PYTHON" >&2
    exit 1
fi


# ===============================
# Project environment
# ===============================

export CMMD_REPO_DIR=/datastore/cndt_khanhnd/aeroeyes_cloudian/aeroeyes-dataset/cmmd-pytorch
export SDQM_REPO_DIR=/datastore/cndt_khanhnd/aeroeyes_cloudian/aeroeyes-dataset/third_party/SDQM

CACHE_ROOT=/datastore/cndt_khanhnd/aeroeyes_cloudian/aeroeyes-dataset/.cache

mkdir -p "$CACHE_ROOT/huggingface" \
         "$CACHE_ROOT/ultralytics" \
         "$CACHE_ROOT/matplotlib"

export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export YOLO_CONFIG_DIR="$CACHE_ROOT/ultralytics"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"

export SDQM_ENABLED=true
export SDQM_VINFO_ENABLED=false


# ===============================
# GPU check
# ===============================

REQUIRED_VRAM=30720

CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
fi

if [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$CHECK_OUT
fi

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-gpu$CHECK_OUT

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | Pipe: gpu$CHECK_OUT"


# ===============================
# Check Python / PyTorch / CUDA
# ===============================

$PYTHON -u -c '
import torch
print("Python   :", __import__("sys").executable)
print("PyTorch  :", torch.__version__)
print("CUDA     :", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
'


# ===============================
# Preflight
# ===============================

$PYTHON scripts/preflight_evaluation.py --require-cuda


# ===============================
# Run project
# ===============================

$PYTHON main.py

echo "Done."