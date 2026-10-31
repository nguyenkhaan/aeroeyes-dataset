#!/bin/bash
#SBATCH --job-name=bbox_experiment
#SBATCH --output=logs/bbox_experiment_%j.out
#SBATCH --error=logs/bbox_experiment_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="$PROJECT_DIR/venv/bin/python"
CONFIG_PATH="${1:?Pass the experiment config JSON path}"
LIMIT_PAIRS="${2:-5}"
BENCH_DIR=/datastore/khanhnd/models/aeroeyes_output/bbox_benchmark
RUN_DIR="$BENCH_DIR/runs/bbox-experiment-$SLURM_JOB_ID"

cd "$PROJECT_DIR"
if [[ ! "$LIMIT_PAIRS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Pair limit must be a positive integer: $LIMIT_PAIRS" >&2
  exit 1
fi

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

"$PYTHON" scripts/run_bbox_experiment.py \
  --manifest "$BENCH_DIR/bbox-gt-v1-manifest.json" \
  --ground-truth "$BENCH_DIR/bbox-gt-v1.validation-review.json" \
  --config "$CONFIG_PATH" \
  --output-dir "$RUN_DIR" \
  --limit "$LIMIT_PAIRS"

"$PYTHON" scripts/evaluate_bounding_boxes.py \
  --ground-truth "$RUN_DIR/ground-truth.selected.json" \
  --predictions "$RUN_DIR/reports" \
  --split validation \
  --run-manifest "$RUN_DIR/run-manifest.json" \
  --output "$RUN_DIR/metrics.json"

echo "Experiment output: $RUN_DIR"
