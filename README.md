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

On UIT HPC, prepare source files on the login node and run environment setup,
CPU/GPU processing, and CUDA checks inside a Slurm allocation. Store the project,
virtual environment, package caches, model weights, datasets, and logs under
`/datastore/khanhnd`. Direct execution examples below apply inside an allocated
compute node.

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

The project pins PyTorch wheels built for CUDA 12.8. The migrated UIT HPC server
exposes `slurm/slurm/25.05`, `cuda12.9/toolkit/12.9.1`, and `python312` modules.
Keep the `cu128` wheel requirements and verify the NVIDIA driver and CUDA runtime
on the allocated compute node before running workloads; the toolkit module
version alone does not establish runtime compatibility. See
[NVIDIA's CUDA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html).
For a manual repair, run the bootstrap script with the same interpreter used by
Slurm:

```bash 
VENV_DIR=/datastore/khanhnd/aeroeyes_cloudian/aeroeyes-dataset/venv \
  bash scripts/setup_vps.sh
```

### Run code  
Run the download once, then generation and evaluation as separate processes:

```bash
python main_down.py
python main.py
python main_evaluation.py
```

The local `sbatch.slurm` currently runs evaluation only (`main_evaluation.py`)
after its CUDA probe and preflight. This file is Git-ignored; copy its changes
to the server separately. Downloading and generation require separate Slurm
jobs invoking `main_down.py` and `main.py` with the configured Python interpreter.

`main_down.py` reads `JSON_PATH` (default: `data/input/eccv_train.json`);
set `JSON_PATH` or pass `--json-path` for a different dataset JSON. It downloads
records accepted by the disaster metadata whitelist and damage filter to
`/datastore/khanhnd/models/aeroeyes_output/download_images/` as lossless RGB PNG files. The original dataset
keys and metadata (including labels and source URLs) are stored together with
`downloaded_file` in `/datastore/khanhnd/models/aeroeyes_output/image_summary.json`. Failed downloads or
invalid images are logged and skipped without stopping the remaining downloads.
Non-disaster records and records marked only `little_or_no_damage` are skipped.
Use `--limit 5000` to stop once 5,000 readable images are available, including
cached images. Failed downloads do not count toward the target. Without a limit,
the script scans the whole JSON; `LIMIT_IMAGES` still controls only generation.
Rerunning preserves valid cached records, verifies images, and downloads only
missing or corrupt images. A smaller target does not trim existing valid records.
Cached records absent from the selected JSON or rejected by the current metadata
filter are removed from the summary; image files are not deleted.
The summary is checkpointed atomically every 25 completed records and at exit.
Images saved before a forced process kill can be recovered on the next run.
A requested target that cannot be reached returns exit code 2 and retains progress.
An invalid existing summary stops the job without replacing that file.

### Rebuild an input subset on the migrated UIT HPC server

Incidents dataset metadata supplies image URLs and labels; see the
[authors' dataset instructions](https://github.com/ethanweber/IncidentsDataset#obtain-the-data).
Upload your existing metadata JSON to `data/input/` under the project.
The data-recovery bundle supplies Linux Python 3.12 wheels for a small download
environment using Pillow, requests, and python-dotenv. Copy its `wheels/*.whl`
files to `downloads/wheels/` before submitting the download job.

From the project directory on the login node:

```bash
mkdir -p data/input logs downloads/wheels
sbatch --test-only scripts/download_incidents.slurm 100 data/input/eccv_train.json
sbatch scripts/download_incidents.slurm 100 data/input/eccv_train.json
```

Use your actual JSON filename. `--test-only` validates the allocation without
submitting a job. The job requests one CPU and 8 GB RAM in `normal`, account
`uit`, QOS `mig2-35`; it requests no GPU. Scheduler acceptance must be confirmed
on the server. It refuses execution outside Slurm or on a login node, keeps its
environment and caches in your datastore, and locks the shared output directory
to prevent concurrent download jobs from rewriting the summary.
Logs are `logs/download_<job-id>.out` and `.err`. If the job reaches its six-hour
wall time, submitting it again resumes saved images. After checking the 100-image
trial, use target `5000` for the input pool. Honor the school rule requiring one
hour between job submissions, even when a preceding job has finished.

Downloaded input images do not replace missing generated before/after pairs.
Without the previous results, generate a small new set of pairs and verify the
bounding-box stage before scaling to 500. The current evaluation entry point
computes delta statistics and the quality gate; object matching/refinement is
not yet connected to it. Grounding DINO's existing YOLO export does not apply
`match_and_refine_objects` to its detections.

If the download job was interrupted, run `python main_label.py` to rebuild
`/datastore/khanhnd/models/aeroeyes_output/image_summary.json` from the images already on disk without
downloading them again. Alternatively, replace `main.py` with `main_label.py`
in the final Python command in `sbatch.slurm`. Use the same `JSON_PATH` dataset
as the download job: the script matches each original key to its SHA-256 PNG
filename, keeps only positive-label records with readable images, and preserves
their metadata and source URLs. It scans the full JSON regardless of
`LIMIT_IMAGES` and atomically replaces any existing summary after the scan.
Temporary `.tmp` images are ignored. After recovery, run `main.py` as usual.

`main.py` reads only the summary and local images. It first prepares cleaned
images and Gemma prompts in `output/_prepared`, then releases Gemma and watermark
tools once and loads FLUX. FLUX stays resident in VRAM throughout rendering;
there is no CPU offload or per-image model swapping. Both phases checkpoint to
disk, so interrupted work can resume without repeating completed samples.
Generation artifacts, reference pairs, and metadata live under
`/datastore/khanhnd/models/aeroeyes_output/output`.
These storage paths are fixed in `src/core/config.py`; old `OUTPUT_DIR`,
`REAL_IMAGES_DIR`, and `GEN_IMAGES_DIR` environment values do not redirect them.
SDQM report, history, and summary paths also stay under this output directory.
Existing files in the old download location are not moved automatically; copy
them and their summary to the new paths before generating, or rerun downloading.

`main_evaluation.py` reads committed `output/_metadata/*.json` files, computes
SC/PQ/O-score/SSIM, records `quality_passed`, and produces CSV, JSONL, CMMD, and
SDQM reports. It evaluates all saved images, including resumed runs. Generation
retains all images; quality rejection is now a report field, not a regeneration
loop. The target counts generated images, not quality-approved images.

```bash 
mkdir -p logs
sbatch sbatch.slurm 
```

## Runtime limits and hang diagnostics

The default Slurm allocation is **72 hours**, controlled by `#SBATCH --time`,
with 20 GB of system memory.
The batch script runs `nvidia-smi` (15-second timeout), the CUDA
allocation/synchronization probe (60-second timeout), preflight (5-minute
timeout), and evaluation (72-hour shell timeout). Slurm enforces the total job
wall time, including the preceding checks. Each step logs `START`, `END`, or
`FAILED`; command failures stop the job. Python logs are unbuffered.

The former `JOB_TIMEOUT_SECONDS`, `STARTUP_TIMEOUT_SECONDS`, and
`PREFLIGHT_TIMEOUT_SECONDS` variables are no longer used by the batch script.
The Python pipeline retains its own limits:

| Environment variable | Default | Meaning |
|---|---:|---|
| `LIMIT_IMAGES` | 500 | Target saved images, including resumed samples |
| `MAX_ATTEMPTS` | `4 × LIMIT_IMAGES` | Attempts per preparation/rendering phase |
| `MAX_CONSECUTIVE_ERRORS` | 10 | Consecutive processing errors before stopping |
| `GENERATION_TIMEOUT_SECONDS` | 244800 (68 hours) | Stop starting new images after this time |
| `MODEL_LOAD_TIMEOUT_SECONDS` | 1800 | Each model-loading stage |
| `STAGE_TIMEOUT_SECONDS` | 900 | Each Gemma, FLUX, watermark, or quality stage |
| `EVALUATION_TIMEOUT_SECONDS` | 3600 | Each CMMD/SDQM report stage |

All Python limits must be non-negative integer seconds/counts and can be placed
in `.env`; zero disables count/error/overall generation guards. Stage timeouts must be positive. Change Slurm's `--time` and
the shell timeout together to adjust the overall job allocation.

Downloads use request timeouts and retries without a process-exiting watchdog.
Python stages log `START`, `END` or `FAILED` with elapsed time. A stuck stage
dumps thread tracebacks to `.err` and immediately exits with code 1 using
[Python's faulthandler watchdog](https://docs.python.org/3/library/faulthandler.html#dumping-the-tracebacks-after-a-timeout).
This is a hard stop: the current image and unfinished aggregate reports may
not be saved. Previously completed image/metadata files remain on disk.
These process limits cannot repair a GPU driver or kernel stuck in
uninterruptible I/O; that requires the cluster administrator.

Completed samples and ineligible records do not consume attempts. A successful
sample resets the error streak. Generation exits with code 2 if its target is
not reached. Evaluation can run independently on partial output; SDQM requires
at least two pairs. Evaluation exits with code 1 on metric failures and code 2
when no images can be evaluated.

FLUX uses bfloat16 on CUDA and tiled VAE encoding/decoding. Gemma uses the
requested dtype on one device, without automatic CPU/GPU dispatch. Cached
allocator memory is cleared at phase boundaries and once after an OOM traceback
has been released, not after every successful image. A second FLUX OOM stops
the run immediately, retaining checkpoints. This avoids retrying ten images
under the same memory pressure. It cannot make model weights fit on an
undersized or busy GPU: use an allocation with sufficient free VRAM, or lower
`IMAGE_SIZE` (default 1024; at least 256, a multiple of 32) for activation-memory
pressure. Resolution is explicitly passed to FLUX and never silently reduced.
Slurm now respects scheduler GPU visibility and MPS settings instead of forcing
physical GPU 2. Validate peak memory and image quality on your server.

The lifecycle follows [PyTorch's allocator guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management):
emptying the cache cannot free live model tensors. FLUX's VAE tiling uses the
[Diffusers Flux2 implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py).

For an evaluation run with a shorter Slurm allocation:

```bash
mkdir -p logs
sbatch --time=01:00:00 sbatch.slurm
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
/datastore/khanhnd/models/aeroeyes_model/
├── huggingface/
├── torch/
└── ultralytics/
```

Existing files in this directory are reused automatically by later jobs. The
project does not use its local `.cache` directory for model storage.

## Preflight (optional for testing)

```bash
export PYTHON="$PWD/venv/bin/python"
export AEROEYES_MODEL_DIR="/datastore/khanhnd/models/aeroeyes_model"
"$PYTHON" scripts/preflight_evaluation.py --require-cuda
```

## Calculate reports for existing images

```bash
"$PYTHON" scripts/preflight_evaluation.py --require-cuda --require-images
"$PYTHON" main_evaluation.py
```

## Generate images and reports with Slurm

```bash
sbatch --export=ALL,PYTHON="$PYTHON",AEROEYES_MODEL_DIR="$AEROEYES_MODEL_DIR" sbatch.slurm
```

Outputs:

```text
/datastore/khanhnd/models/aeroeyes_output/output/evaluation_report.csv
/datastore/khanhnd/models/aeroeyes_output/output/evaluation_metadata.jsonl
/datastore/khanhnd/models/aeroeyes_output/output/sdqm/sdqm_report.json
/datastore/khanhnd/models/aeroeyes_output/output/sdqm/sdqm_values.csv
/datastore/khanhnd/models/aeroeyes_output/output/sdqm/sdqm_summary.md
```
