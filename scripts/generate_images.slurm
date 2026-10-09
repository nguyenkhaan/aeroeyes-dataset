#!/bin/bash
#SBATCH --job-name=aeroeyes_generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="$PROJECT_DIR/venv/bin/python"
SUMMARY=/datastore/khanhnd/models/aeroeyes_output/image_summary.json
LIMIT_IMAGES="${1:-5}"

cd "$PROJECT_DIR"

if [[ "$PROJECT_DIR" != /datastore/khanhnd/* ]]; then
  echo "Project must be stored below /datastore/khanhnd: $PROJECT_DIR" >&2
  exit 1
fi

if [[ ! "$LIMIT_IMAGES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Image target must be a positive integer: $LIMIT_IMAGES" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Generation environment is missing. Run scripts/setup_generation.slurm first." >&2
  exit 1
fi

if [[ ! -f .env ]] || ! grep -Eq '^HF_TOKEN=hf_.+' .env; then
  echo "HF_TOKEN is missing from .env." >&2
  exit 1
fi
chmod 600 .env

module clear -f
module load slurm/slurm/25.05
module load cuda12.9/toolkit/12.9.1
module load shared python312

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export AEROEYES_MODEL_DIR=/datastore/khanhnd/models/aeroeyes_model
export HF_HOME="$AEROEYES_MODEL_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME="$AEROEYES_MODEL_DIR/torch"
export XDG_CACHE_HOME="$AEROEYES_MODEL_DIR/xdg"
export IMAGE_SIZE="${IMAGE_SIZE:-512}"
export LIMIT_IMAGES
export WATERMARK_REMOVAL_ENABLED="${WATERMARK_REMOVAL_ENABLED:-false}"

mkdir -p "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME"

"$PYTHON" - "$SUMMARY" "$LIMIT_IMAGES" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
target = int(sys.argv[2])
if not summary_path.is_file():
    raise SystemExit(f"Input summary is missing: {summary_path}")

with summary_path.open(encoding="utf-8") as summary_file:
    records = json.load(summary_file)

if not isinstance(records, dict):
    raise SystemExit("image_summary.json must contain a JSON object")
if len(records) < target:
    raise SystemExit(f"Only {len(records)} inputs are ready; requested {target}")

print(f"Input images ready: {len(records)}; generation target: {target}")
PY

"$PYTHON" - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}; compiled CUDA: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the Slurm GPU job")
print(f"GPU: {torch.cuda.get_device_name(0)}")
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"GPU memory: {free_bytes / 2**30:.1f} GiB free / {total_bytes / 2**30:.1f} GiB")
PY

echo "Starting generation for $LIMIT_IMAGES image(s) at ${IMAGE_SIZE}x${IMAGE_SIZE}."
"$PYTHON" main.py
