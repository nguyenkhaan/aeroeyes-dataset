#!/usr/bin/env bash
#SBATCH --job-name=incidents_download
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00

set -euo pipefail
umask 077

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch; downloads must run on a compute node." >&2
    exit 1
fi
case "${SLURMD_NODENAME:-$(hostname -s)}" in
    login*|*headnode*)
        echo "Downloads must run on a compute node, not the login node." >&2
        exit 1
        ;;
esac

cd "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"
PROJECT_DIR="$(pwd -P)"
case "$PROJECT_DIR/" in
    "/datastore/${USER}/"*) ;;
    *) echo "Submit from your project under /datastore/$USER." >&2; exit 1 ;;
esac

LIMIT="${1:-100}"
DATASET_JSON="${2:-data/input/eccv_train.json}"
if [[ $# -gt 2 || ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: sbatch scripts/download_incidents.slurm [positive-target] [dataset-json]" >&2
    exit 1
fi
if [[ ! -f "$DATASET_JSON" ]]; then
    echo "Dataset JSON not found: $DATASET_JSON" >&2
    exit 1
fi
if [[ ! -f "$PROJECT_DIR/requirements-download.txt" ]]; then
    echo "Download requirements are missing; apply the source patch first." >&2
    exit 1
fi
if [[ ! -d "$PROJECT_DIR/downloads/wheels" ]]; then
    echo "Download wheels are missing; copy the bundle's wheels into downloads/wheels." >&2
    exit 1
fi

module clear -f
module load slurm/slurm/25.05
module load shared python312
BOOTSTRAP_PYTHON="$(command -v python3.12)"
"$BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache/xdg"
export TMPDIR="$PROJECT_DIR/.cache/tmp/download_$SLURM_JOB_ID"
OUTPUT_DIR=/datastore/khanhnd/models/aeroeyes_output
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$OUTPUT_DIR"

# Each project copy shares this output directory; serialize writes across jobs.
exec 9>"$OUTPUT_DIR/.download.lock"
if ! flock -n 9; then
    echo "Another download job is writing the same dataset. Wait for it to finish." >&2
    exit 1
fi

VENV_DIR="$PROJECT_DIR/downloads/venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"$PYTHON" -m pip install --no-index --find-links "$PROJECT_DIR/downloads/wheels" \
    -r "$PROJECT_DIR/requirements-download.txt"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze >"$PROJECT_DIR/logs/download_env_$SLURM_JOB_ID.txt"

echo "Job $SLURM_JOB_ID | Node ${SLURMD_NODENAME:-unknown} | Target $LIMIT | JSON $DATASET_JSON"
"$PYTHON" -u "$PROJECT_DIR/main_down.py" --json-path "$DATASET_JSON" --limit "$LIMIT"
