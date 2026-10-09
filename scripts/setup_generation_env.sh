#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/venv}"
VENV_PYTHON="$VENV_DIR/bin/python"
EXPECTED_PYTORCH_CUDA="${EXPECTED_PYTORCH_CUDA:-12.8}"

if [[ "$ROOT_DIR" != /datastore/khanhnd/* ]]; then
  echo "Project must be stored below /datastore/khanhnd: $ROOT_DIR" >&2
  exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run environment setup through scripts/setup_generation.slurm." >&2
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install \
  --upgrade --force-reinstall --no-cache-dir \
  -r "$ROOT_DIR/requirements-pytorch-cu128.txt"
"$VENV_PYTHON" -m pip install \
  -r "$ROOT_DIR/requirements-generation.txt"
"$VENV_PYTHON" -m pip check

"$VENV_PYTHON" - "$EXPECTED_PYTORCH_CUDA" <<'PY'
import sys

import diffusers
import torch
import transformers

expected_cuda = sys.argv[1]
if torch.version.cuda != expected_cuda:
    raise SystemExit(
        f"PyTorch CUDA mismatch: {torch.version.cuda}; expected {expected_cuda}"
    )

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"Transformers: {transformers.__version__}")
print(f"Diffusers: {diffusers.__version__}")
PY

echo "Generation environment is ready: $VENV_PYTHON"
