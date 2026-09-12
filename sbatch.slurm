#!/bin/bash
#SBATCH --job-name=aeroeyes_job
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=mps:a100:2
#SBATCH --mem=20G
#SBATCH --time=72:00:00

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1


# ===============================
# Helper
# ===============================

run_step() {
    local name="$1"
    local status

    shift

    echo "[$(date -Is)] START $name" >&2

    if "$@"; then
        echo "[$(date -Is)] END $name" >&2
    else
        status=$?
        echo "[$(date -Is)] FAILED $name (exit=$status)" >&2
        return "$status"
    fi
}


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

PYTHON=${PYTHON:-/datastore/cndt_khanhnd/aeroeyes_cloudian/aeroeyes-dataset/venv/bin/python}

if [ ! -x "$PYTHON" ]; then
    echo "Configured Python is not executable: $PYTHON" >&2
    exit 1
fi


# ===============================
# Project environment
# ===============================

export CMMD_REPO_DIR=/datastore/cndt_khanhnd/aeroeyes_cloudian/aeroeyes-dataset/cmmd-pytorch
export SDQM_REPO_DIR=/datastore/cndt_khanhnd/aeroeyes_cloudian/aeroeyes-dataset/third_party/SDQM

export AEROEYES_MODEL_DIR="${AEROEYES_MODEL_DIR:-/datastore/cndt_khanhnd/models/aeroeyes_model}"

export HF_HOME="$AEROEYES_MODEL_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"

export TORCH_HOME="$AEROEYES_MODEL_DIR/torch"
export XDG_CACHE_HOME="$AEROEYES_MODEL_DIR/xdg"

export YOLO_CONFIG_DIR="$AEROEYES_MODEL_DIR/ultralytics/config"
export YOLO_WEIGHTS_DIR="$AEROEYES_MODEL_DIR/ultralytics/weights"
export YOLO_RUNS_DIR="$AEROEYES_MODEL_DIR/ultralytics/runs"

export SDQM_VINFO_MODEL_PATH="$YOLO_WEIGHTS_DIR/yolo11n.pt"

export MPLCONFIGDIR="$AEROEYES_MODEL_DIR/matplotlib"


mkdir -p \
    "$HF_HUB_CACHE" \
    "$HF_ASSETS_CACHE" \
    "$HF_XET_CACHE" \
    "$TORCH_HOME" \
    "$XDG_CACHE_HOME" \
    "$YOLO_CONFIG_DIR" \
    "$YOLO_WEIGHTS_DIR" \
    "$YOLO_RUNS_DIR" \
    "$MPLCONFIGDIR"


export SDQM_ENABLED=true
export SDQM_VINFO_ENABLED=false
export EXPECTED_PYTORCH_CUDA=12.8


# ===============================
# GPU selection
# ===============================

# Force GPU 2.
#
# Current AsusL40 snapshot:
# GPU 2 has the most free VRAM.
#
SELECTED_GPU=2
REQUIRED_VRAM=24576


echo
echo "============================================================"
echo "AeroEyes GPU Selection"
echo "============================================================"

echo "Job ID               : ${SLURM_JOB_ID:-<not-set>}"
echo "Node                 : ${SLURMD_NODENAME:-<not-set>}"
echo "SLURM_JOB_GPUS       : ${SLURM_JOB_GPUS:-<not-set>}"
echo "Original CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<not-set>}"
echo "Requested GPU        : $SELECTED_GPU"
echo "Required VRAM        : ${REQUIRED_VRAM} MiB"


# ===============================
# Check GPU 2
# ===============================

GPU_INFO=$(
    timeout 15s \
    nvidia-smi \
        --id="$SELECTED_GPU" \
        --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits
)


if [ -z "$GPU_INFO" ]; then
    echo "ERROR: Cannot inspect GPU $SELECTED_GPU." >&2
    exit 1
fi


echo
echo "============================================================"
echo "Selected GPU"
echo "============================================================"

echo "index, name, memory.total [MiB], memory.used [MiB], memory.free [MiB]"
echo "$GPU_INFO"


# ===============================
# Correctly parse memory.free
# ===============================

GPU_FREE_MIB=$(
    echo "$GPU_INFO" |
    awk -F',' '
    {
        gsub(/^[ \t]+|[ \t]+$/, "", $5)
        print $5
    }'
)


if ! [[ "$GPU_FREE_MIB" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid VRAM value: $GPU_FREE_MIB" >&2
    exit 1
fi


echo
echo "Free VRAM            : ${GPU_FREE_MIB} MiB"
echo "Required VRAM        : ${REQUIRED_VRAM} MiB"


if [ "$GPU_FREE_MIB" -lt "$REQUIRED_VRAM" ]; then

    echo
    echo "============================================================"
    echo "INSUFFICIENT VRAM"
    echo "============================================================"

    echo "GPU                  : $SELECTED_GPU"
    echo "Free VRAM            : ${GPU_FREE_MIB} MiB"
    echo "Required VRAM        : ${REQUIRED_VRAM} MiB"

    exit 1
fi


# ===============================
# CUDA device
# ===============================

#
# NOTE:
# This overrides the value injected by Slurm.
#
# We deliberately do this because the purpose of this test
# is to determine whether GPU 2 is actually usable from this
# job under the current MPS/cgroup configuration.
#

export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"


# ===============================
# MPS directories
# ===============================

export CUDA_MPS_PIPE_DIRECTORY="/tmp/aeroeyes-mps-gpu${SELECTED_GPU}"
export CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY/log"

mkdir -p \
    "$CUDA_MPS_PIPE_DIRECTORY" \
    "$CUDA_MPS_LOG_DIRECTORY"


echo
echo "============================================================"
echo "GPU / MPS Configuration"
echo "============================================================"

echo "Selected GPU            : $SELECTED_GPU"
echo "CUDA_VISIBLE_DEVICES    : $CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_PIPE_DIRECTORY : $CUDA_MPS_PIPE_DIRECTORY"
echo "CUDA_MPS_LOG_DIRECTORY  : $CUDA_MPS_LOG_DIRECTORY"


# ===============================
# NVIDIA SMI
# ===============================

echo
echo "============================================================"
echo "NVIDIA SMI"
echo "============================================================"

run_step \
    "nvidia-smi" \
    timeout 15s \
    nvidia-smi \
        --id="$SELECTED_GPU"


# ===============================
# Check Python / PyTorch / CUDA
# ===============================

echo
echo "============================================================"
echo "PyTorch CUDA Check"
echo "============================================================"

run_step \
    "CUDA probe" \
    timeout 60s \
    "$PYTHON" \
    -u \
    -c \
'
import sys
import torch

print("Python       :", sys.executable)
print("PyTorch      :", torch.__version__)
print("Torch CUDA   :", torch.version.cuda)
print("CUDA         :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable for GPU 2. "
        "This may mean Slurm/cgroup/MPS did not allocate GPU 2 "
        "to this job."
    )

count = torch.cuda.device_count()

print("GPU count    :", count)

if count < 1:
    raise RuntimeError("No CUDA GPU visible to PyTorch")

for i in range(count):
    props = torch.cuda.get_device_properties(i)

    print(
        f"GPU {i}: "
        f"{torch.cuda.get_device_name(i)} | "
        f"VRAM: {props.total_memory / (1024 ** 3):.2f} GB"
    )

print("Testing CUDA allocation...", flush=True)

x = torch.empty(
    (1,),
    dtype=torch.float32,
    device="cuda"
)

torch.cuda.synchronize()

del x

print("CUDA probe passed.", flush=True)
'


# ===============================
# Preflight
# ===============================

echo
echo "============================================================"
echo "Preflight"
echo "============================================================"

run_step \
    "Preflight" \
    timeout 300s \
    "$PYTHON" \
    scripts/preflight_evaluation.py \
    --require-cuda


# ===============================
# Run project
# ===============================

echo
echo "============================================================"
echo "Run AeroEyes"
echo "============================================================"

run_step \
    "Pipeline" \
    timeout 72h \
    "$PYTHON" \
    main.py


# ===============================
# Done
# ===============================

echo
echo "============================================================"
echo "Done"
echo "============================================================"

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $CUDA_VISIBLE_DEVICES"

echo "============================================================"
