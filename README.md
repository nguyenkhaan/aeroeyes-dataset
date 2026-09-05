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
mkdir -p logs
sbatch sbatch.slurm 
```

## Runtime limits and hang diagnostics

The default Slurm allocation is **24 hours**. The script uses GNU `timeout`
for GPU selection, a CUDA allocation/synchronization probe, preflight, and the
Python pipeline. A timed-out command receives TERM, followed by KILL after
10 seconds if necessary. Python logs are unbuffered.

| Environment variable | Default | Meaning |
|---|---:|---|
| `JOB_TIMEOUT_SECONDS` | 82800 | Shared command budget from script startup (23 hours) |
| `STARTUP_TIMEOUT_SECONDS` | 120 | Each GPU selection/CUDA probe |
| `PREFLIGHT_TIMEOUT_SECONDS` | 300 | Evaluation prerequisite checks |
| `LIMIT_IMAGES` | 1 | Newly accepted images per run |
| `MAX_ATTEMPTS` | 10 | Images attempted, including download failures and quality rejections |
| `MAX_CONSECUTIVE_ERRORS` | 3 | Consecutive processing errors before stopping |
| `GENERATION_TIMEOUT_SECONDS` | 3600 | Stop starting new images after this time, excluding model loading |
| `MODEL_LOAD_TIMEOUT_SECONDS` | 1800 | Each model-loading stage |
| `STAGE_TIMEOUT_SECONDS` | 900 | Each Gemma, FLUX, quality, or cleanup stage |
| `EVALUATION_TIMEOUT_SECONDS` | 3600 | Each CMMD/SDQM report stage |

All limits must be positive integer seconds/counts. Export shell timeout
settings before submission; Python settings can also be placed in `.env`.
If changing Slurm's `--time`, keep `JOB_TIMEOUT_SECONDS` below the allocation
with room for termination. It is not automatically derived from Slurm.

Each download has a 120-second total watchdog in addition to request retries.
Python stages log `START`, `END` or `FAILED` with elapsed time. A stuck stage
dumps thread tracebacks to `.err` and immediately exits with code 1 using
[Python's faulthandler watchdog](https://docs.python.org/3/library/faulthandler.html#dumping-the-tracebacks-after-a-timeout).
This is a hard stop: the current image and unfinished aggregate reports may
not be saved. Previously completed image/metadata files remain on disk.
These process limits cannot repair a GPU driver or kernel stuck in
uninterruptible I/O; that requires the cluster administrator.

Existing output images and ineligible records do not consume attempts.
A quality rejection or successfully saved image resets the error streak.
If the target is not reached because of limits or dataset exhaustion,
the pipeline writes available reports, skips CMMD/SDQM, and exits with code 2.
After successful generation it releases generation models before bounded
dataset evaluation. Shell timeout exits are normally 124, or 137 after KILL.

For a small diagnostic run:

```bash
mkdir -p logs
sbatch --export=ALL,LIMIT_IMAGES=1,MAX_ATTEMPTS=10,MAX_CONSECUTIVE_ERRORS=3 sbatch.slurm
```

Use the same job ID for `logs/job_<id>.out` and `logs/job_<id>.err`. The last
`START` without an `END`/`FAILED`, together with the traceback, identifies the
stage to investigate. Increase a stage budget only after confirming that it
is making progress (first-time model downloads may need a larger budget).
MPS allocation and GPU mapping remain cluster-specific; the script does not
change the requested `--gres` resource or restart the MPS server.

CPU-only guard checks (no model downloads):

```bash
python3 -m unittest discover -s tests -p 'test_sbatch*.py'
python3 -m unittest discover -s tests -p test_runtime.py
python3 -m unittest discover -s tests -p test_pipeline_limits.py
```

## Model storage

All downloaded model weights and model caches are stored below:

```text
/datastore/cndt_khanhnd/models/aeroeyes_model/
├── huggingface/
├── torch/
└── ultralytics/
```

Existing files in this directory are reused automatically by later jobs. The
project does not use its local `.cache` directory for model storage.

## Preflight (optional for testing)

```bash
export PYTHON="$PWD/venv/bin/python"
export AEROEYES_MODEL_DIR="/datastore/cndt_khanhnd/models/aeroeyes_model"
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
sbatch --export=ALL,PYTHON="$PYTHON",AEROEYES_MODEL_DIR="$AEROEYES_MODEL_DIR" sbatch.slurm
```

Outputs:

```text
data/output/evaluation_report.csv
data/output/evaluation_metadata.jsonl
data/output/sdqm/sdqm_report.json
data/output/sdqm/sdqm_values.csv
reports/sdqm_summary.md
```
