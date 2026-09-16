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
Run the download once, then generation and evaluation as separate processes:

```bash
python main_down.py
python main.py
python main_evaluation.py
```

The local `sbatch.slurm` runs generation followed by evaluation. This file is
Git-ignored; copy its changes to the server separately. It also evaluates saved
partial results when generation exits with code 2, preserving that exit status.
For downloading under Slurm, replace its generation/evaluation commands with
`python main_down.py` using the configured Python interpreter.

`main_down.py` reads `JSON_PATH` (default: `data/input/eccv_train.json`);
set `JSON_PATH` if your dataset JSON is in the output directory. It downloads
all records with at least one positive disaster label (`incidents` value `1`) to
`/datastore/cndt_khanhnd/models/aeroeyes_output/download_images/` as lossless RGB PNG files. The original dataset
keys and metadata (including labels and source URLs) are stored together with
`downloaded_file` in `/datastore/cndt_khanhnd/models/aeroeyes_output/image_summary.json`. Failed downloads or
invalid images are logged and skipped without stopping the remaining downloads.
Records with missing or empty `incidents`, or no value equal to `1`, are skipped
before downloading and are not included in the summary. Every download run scans
the entire JSON; `LIMIT_IMAGES` applies only to AI generation in `main.py`.
The summary is replaced atomically when the loop finishes or unwinds through
a Python exception; a forced process kill cannot save the current summary.
Rerunning this step downloads the dataset again and rebuilds the summary.

If the download job was interrupted, run `python main_label.py` to rebuild
`/datastore/cndt_khanhnd/models/aeroeyes_output/image_summary.json` from the images already on disk without
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
`/datastore/cndt_khanhnd/models/aeroeyes_output/output`.
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
The batch script runs GPU selection, the CUDA allocation/synchronization probe,
preflight, generation (68-hour shell timeout), and evaluation (4-hour shell
timeout). Each step logs
`START`, `END`, or `FAILED`; command failures stop the job. Python logs are
unbuffered. The CUDA probe has a 60-second timeout.

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
"$PYTHON" main_evaluation.py
```

## Generate images and reports with Slurm

```bash
sbatch --export=ALL,PYTHON="$PYTHON",AEROEYES_MODEL_DIR="$AEROEYES_MODEL_DIR" sbatch.slurm
```

Outputs:

```text
/datastore/cndt_khanhnd/models/aeroeyes_output/output/evaluation_report.csv
/datastore/cndt_khanhnd/models/aeroeyes_output/output/evaluation_metadata.jsonl
/datastore/cndt_khanhnd/models/aeroeyes_output/output/sdqm/sdqm_report.json
/datastore/cndt_khanhnd/models/aeroeyes_output/output/sdqm/sdqm_values.csv
/datastore/cndt_khanhnd/models/aeroeyes_output/output/sdqm/sdqm_summary.md
```
