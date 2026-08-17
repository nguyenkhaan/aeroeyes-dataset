# Generating Rescue Images from Natural Disaster Images

This project transforms natural-disaster images into rescue-oriented images,
then evaluates the generated dataset with per-image quality metrics, CMMD, and
SDQM.

## Requirements

- NVIDIA GPU with at least 32 GB available VRAM
- Python 3.12 and the VPS `module` command
- Hugging Face token with access to the configured models

CMMD and SDQM source trees are already included in this repository. Do not
clone them separately.

## VPS setup

```bash
git clone <repository-url> aeroeyes-dataset
cd aeroeyes-dataset
conda create --prefix venv/ python=3.12
conda activate venv/
```

Set `HF_TOKEN` in `.env` once:

```bash
nano .env
```

### Install dependencies

Use the bootstrap script. It removes incompatible CUDA 13 packages, installs
the official CUDA 12.8 PyTorch wheels, and verifies the installed Torch build
before installing the remaining dependencies.

```bash
bash scripts/setup_vps.sh
```

The Slurm environment uses CUDA 12.8, so do not install a `+cu130` PyTorch
wheel. For a manual repair, run the bootstrap script with the same interpreter
used by Slurm:

```bash 
VENV_DIR=/datastore/cndt_khanhnd/aeroeyes_cloudian/aeroeyes-dataset/venv \
  bash scripts/setup_vps.sh
```

### Run code  
```bash 
sbatch sbatch.slurm 
```
## Preflight (optional for testing)

```bash
export PYTHON="$PWD/venv/bin/python"
export CACHE_ROOT="$PWD/.cache"
"$PYTHON" scripts/preflight_evaluation.py --require-cuda
```

## Calculate reports for existing images

```bash
"$PYTHON" scripts/preflight_evaluation.py --require-cuda --require-images
"$PYTHON" scripts/run_evaluation.py \
  --real-dir data/real_reference \
  --synthetic-dir data/gen_reference
```

## Generate images and reports with Slurm

```bash
sbatch --export=ALL,PYTHON="$PYTHON",CACHE_ROOT="$CACHE_ROOT" sbatch.slurm
```

Outputs:

```text
data/output/evaluation_report.csv
data/output/evaluation_metadata.jsonl
data/output/sdqm/sdqm_report.json
data/output/sdqm/sdqm_values.csv
reports/sdqm_summary.md
```
