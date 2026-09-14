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
Run the pipeline in two separate jobs. In the final Python command under
`Run project` in `sbatch.slurm`, change `main.py` to `main_down.py` and submit
the download job. After it finishes, change it back to `main.py` and submit
the generation job using the same command below.

`main_down.py` reads `JSON_PATH` (default: `data/input/eccv_train.json`);
set `JSON_PATH` if your dataset JSON is in the output directory. It downloads
all records, including those without positive labels, to
`data/input/download_images/` as lossless RGB PNG files. The original dataset
keys and metadata (including labels and source URLs) are stored together with
`downloaded_file` in `data/input/image_summary.json`. Failed downloads or
invalid images are logged and skipped without stopping the remaining downloads.
The summary is replaced atomically when the loop finishes or unwinds through
a Python exception; a forced process kill cannot save the current summary.
Rerunning this step downloads the dataset again and rebuilds the summary.

`main.py` reads only this summary and the local images, then applies the existing
positive-label filter, watermark removal, generation, and evaluation steps.
Missing or unreadable local images are skipped. Downloading does not use the
generation count/error limits. The existing Slurm GPU checks and preflight
still run for both jobs.

```bash 
mkdir -p logs
sbatch sbatch.slurm 
```

## Runtime limits and hang diagnostics

The default Slurm allocation is **72 hours**, controlled by `#SBATCH --time`,
with 20 GB of system memory.
The batch script runs GPU selection, the CUDA allocation/synchronization probe,
preflight, and the pipeline with a 72-hour shell timeout. Slow CUDA initialization
can finish without being killed after 120 seconds. Each step logs
`START`, `END`, or `FAILED`; command failures stop the job. Python logs are
unbuffered. A stuck CUDA probe can wait until Slurm ends the allocation.

The former `JOB_TIMEOUT_SECONDS`, `STARTUP_TIMEOUT_SECONDS`, and
`PREFLIGHT_TIMEOUT_SECONDS` variables are no longer used by the batch script.
The Python pipeline retains its own limits:

| Environment variable | Default | Meaning |
|---|---:|---|
| `LIMIT_IMAGES` | 500 | Newly accepted images per run |
| `MAX_ATTEMPTS` | `4 × LIMIT_IMAGES` | Local images attempted, including quality rejections |
| `MAX_CONSECUTIVE_ERRORS` | 10 | Consecutive processing errors before stopping |
| `GENERATION_TIMEOUT_SECONDS` | 244800 (68 hours) | Stop starting new images after this time |
| `MODEL_LOAD_TIMEOUT_SECONDS` | 1800 | Each model-loading stage |
| `STAGE_TIMEOUT_SECONDS` | 900 | Each Gemma, FLUX, quality, or cleanup stage |
| `EVALUATION_TIMEOUT_SECONDS` | 3600 | Each CMMD/SDQM report stage |

All Python limits must be non-negative integer seconds/counts and can be placed
in `.env`; zero disables the corresponding guard. Change Slurm's `--time` and
the shell timeout together to adjust the overall job allocation.

Downloads use request timeouts and retries without a process-exiting watchdog.
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
the pipeline writes per-image reports, runs CMMD/SDQM on available saved images,
then exits with code 2 to indicate the incomplete generation target. SDQM needs
at least two real and two synthetic images; with fewer images it writes a
`skipped` status and reason to `data/output/sdqm/sdqm_report.json`.
Gemma and FLUX are each loaded once and reused throughout generation. After
the generation loop ends, both models are released before bounded dataset
evaluation. The shell hard timeout is 72 hours; timeout exits are normally
124, or 137 after KILL.

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
