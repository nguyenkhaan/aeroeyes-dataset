#!/bin/bash
#SBATCH --job-name=aeroeyes_job
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=mps:a100:2
#SBATCH --mem=12G
#SBATCH --time=24:60:00


# ===============================
# Load environment
# ===============================

module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1
module load shared python312


# ===============================
# Conda environment
# ===============================

PYTHON=/datastore/cndt_khanhnd/aeroeyes_cloudian/aeroeyes-dataset/venv/bin/python
REQUIRED_VRAM=30720
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then echo "$CHECK_OUT"; exit 0; fi
if [ $EXIT_CODE -eq 11 ]; then echo "$CHECK_OUT"; exit 1; fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$CHECK_OUT
fi
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-gpu$CHECK_OUT

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | Pipe: gpu$CHECK_OUT"

$PYTHON -u -c '
import torch
print("PyTorch  :", torch.__version__)
print("CUDA     :", torch.cuda.is_available())
print("GPU name :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
'
# ===============================
# Run project
# ===============================

$PYTHON main.py

echo "Done."
