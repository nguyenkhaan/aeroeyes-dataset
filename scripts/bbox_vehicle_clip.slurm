#!/bin/bash
#SBATCH --job-name=bbox_vehicle_clip
#SBATCH --output=logs/bbox_vehicle_clip_%j.out
#SBATCH --error=logs/bbox_vehicle_clip_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BENCH_DIR=/datastore/khanhnd/models/aeroeyes_output/bbox_benchmark
BASELINE_JOB="${1:?Pass the baseline experiment job ID}"
if [[ ! "$BASELINE_JOB" =~ ^[0-9]+$ ]]; then
  echo "Baseline job ID must be numeric" >&2
  exit 1
fi
cd "$PROJECT_DIR"
module clear -f
module load slurm/slurm/25.05
module load cuda12.9/toolkit/12.9.1
module load shared python312
export PYTHONUNBUFFERED=1
export AEROEYES_MODEL_DIR=/datastore/khanhnd/models/aeroeyes_model
export HF_HOME="$AEROEYES_MODEL_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME="$AEROEYES_MODEL_DIR/torch"
export XDG_CACHE_HOME="$AEROEYES_MODEL_DIR/xdg"

"$PROJECT_DIR/venv/bin/python" scripts/score_bbox_vehicle_conflicts.py \
  --run-dir "$BENCH_DIR/runs/bbox-experiment-$BASELINE_JOB" \
  --metrics "$BENCH_DIR/strict-v2-results/metrics-$BASELINE_JOB.json" \
  --output "$BENCH_DIR/strict-v2-results/vehicle-clip-$SLURM_JOB_ID.json"
