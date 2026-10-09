#!/bin/bash
#SBATCH --job-name=aeroeyes_setup
#SBATCH --output=logs/setup_generation_%j.out
#SBATCH --error=logs/setup_generation_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

module clear -f
module load slurm/slurm/25.05
module load shared python312

export PIP_CACHE_DIR=/datastore/khanhnd/models/aeroeyes_model/pip
export VENV_DIR="$PROJECT_DIR/venv"
mkdir -p "$PIP_CACHE_DIR"

bash scripts/setup_generation_env.sh
