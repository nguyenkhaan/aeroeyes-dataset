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

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
JOB_TIMEOUT_SECONDS=${JOB_TIMEOUT_SECONDS:-82800}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-120}
PREFLIGHT_TIMEOUT_SECONDS=${PREFLIGHT_TIMEOUT_SECONDS:-300}
for duration in "$JOB_TIMEOUT_SECONDS" "$STARTUP_TIMEOUT_SECONDS" "$PREFLIGHT_TIMEOUT_SECONDS"; do
    if [[ ! "$duration" =~ ^[1-9][0-9]{0,5}$ ]]; then
        echo "Timeouts must be positive integer seconds (at most 999999)." >&2
        exit 2
    fi
done
JOB_DEADLINE=$((SECONDS + JOB_TIMEOUT_SECONDS))

run_timed() {
    local name=$1 duration=$2 status remaining
    shift 2
    remaining=$((JOB_DEADLINE - SECONDS))
    if (( remaining <= 0 )); then
        echo "Job runtime budget exhausted before $name" >&2
        return 124
    fi
    if (( duration > remaining )); then
        duration=$remaining
    fi
    echo "[$(date -Is)] START $name (timeout=${duration}s)" >&2
    if timeout --verbose --kill-after=10s "${duration}s" "$@"; then
        echo "[$(date -Is)] END $name" >&2
    else
        status=$?
        echo "[$(date -Is)] FAILED $name (exit=$status; 124/137 indicates timeout or forced kill)" >&2
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

mkdir -p "$HF_HUB_CACHE" \
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
# GPU check
# ===============================

REQUIRED_VRAM=30720

if CHECK_OUT=$(run_timed "GPU selection" "$STARTUP_TIMEOUT_SECONDS" \
    "${GPU_CHECK_SCRIPT:-/usr/local/bin/gpu_check.sh}" "$REQUIRED_VRAM" "$SLURM_JOB_ID"); then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
fi

if [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "$CHECK_OUT" >&2
    exit "$EXIT_CODE"
fi

if [[ ! "$CHECK_OUT" =~ ^[0-9]+$ ]]; then
    echo "GPU check did not return a numeric GPU index: $CHECK_OUT" >&2
    exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$CHECK_OUT
fi

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-gpu$CHECK_OUT

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | Pipe: gpu$CHECK_OUT"


# ===============================
# Check Python / PyTorch / CUDA
# ===============================

run_timed "CUDA probe" "$STARTUP_TIMEOUT_SECONDS" "$PYTHON" -u -c '
import torch
print("Python   :", __import__("sys").executable)
print("PyTorch  :", torch.__version__)
print("Checking CUDA availability...", flush=True)
available = torch.cuda.is_available()
print("CUDA     :", available)
if not available:
    raise SystemExit("CUDA unavailable; stopping before model loading")
print("GPU count:", torch.cuda.device_count())
print("GPU name :", torch.cuda.get_device_name(0))
print("Checking CUDA allocation and synchronization...", flush=True)
torch.empty(1, device="cuda")
torch.cuda.synchronize()
print("CUDA probe passed.", flush=True)
'


# ===============================
# Preflight
# ===============================

run_timed "Preflight" "$PREFLIGHT_TIMEOUT_SECONDS" "$PYTHON" scripts/preflight_evaluation.py --require-cuda


# ===============================
# Run project
# ===============================

run_timed "Pipeline" "$JOB_TIMEOUT_SECONDS" "$PYTHON" main.py

echo "Done."
