#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/venv}"
VENV_PYTHON="$VENV_DIR/bin/python"
EXPECTED_PYTORCH_CUDA="${EXPECTED_PYTORCH_CUDA:-12.8}"
INCOMPATIBLE_CUDA_PACKAGES=(
  cuda-bindings
  cuda-pathfinder
  cuda-toolkit
  nvidia-cublas
  nvidia-cuda-cupti
  nvidia-cuda-nvrtc
  nvidia-cuda-runtime
  nvidia-cudnn-cu13
  nvidia-cufft
  nvidia-cufile
  nvidia-curand
  nvidia-cusolver
  nvidia-cusparse
  nvidia-cusparselt-cu13
  nvidia-nccl-cu13
  nvidia-nvjitlink
  nvidia-nvshmem-cu13
  nvidia-nvtx
  torch
  torchvision
  triton
)

if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
  echo "Bootstrap Python was not found: $BOOTSTRAP_PYTHON" >&2
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

if [[ ! -f "$ROOT_DIR/cmmd-pytorch/main.py" ]]; then
  echo "CMMD source is missing: $ROOT_DIR/cmmd-pytorch/main.py" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/third_party/SDQM/sdqm.py" ]]; then
  echo "SDQM source is missing: $ROOT_DIR/third_party/SDQM/sdqm.py" >&2
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip uninstall --yes "${INCOMPATIBLE_CUDA_PACKAGES[@]}"
"$VENV_PYTHON" -m pip install --upgrade --force-reinstall --no-cache-dir -r "$ROOT_DIR/requirements-pytorch-cu128.txt"
"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/cmmd-pytorch/requirements.txt"
PYTHON="$VENV_PYTHON" bash "$ROOT_DIR/scripts/setup_sdqm_vinfo.sh"
"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" - "$EXPECTED_PYTORCH_CUDA" <<'PY'
import sys

import torch

expected_cuda = sys.argv[1]
if torch.version.cuda != expected_cuda:
    raise SystemExit(
        f"PyTorch was compiled for CUDA {torch.version.cuda}, expected {expected_cuda}."
    )

print(f"PyTorch {torch.__version__} compiled for CUDA {torch.version.cuda}.")
PY

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/data/input" "$ROOT_DIR/data/output" \
  "$ROOT_DIR/data/real_reference" "$ROOT_DIR/data/gen_reference" "$ROOT_DIR/.cache"

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
fi

echo "VPS setup complete. Set HF_TOKEN in $ROOT_DIR/.env before submitting a job."
echo "Use this interpreter: $VENV_PYTHON"
