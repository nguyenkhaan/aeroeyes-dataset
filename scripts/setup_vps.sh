#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/venv}"
VENV_PYTHON="$VENV_DIR/bin/python"

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
"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/cmmd-pytorch/requirements.txt"
PYTHON="$VENV_PYTHON" bash "$ROOT_DIR/scripts/setup_sdqm_vinfo.sh"

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/data/input" "$ROOT_DIR/data/output" \
  "$ROOT_DIR/data/real_reference" "$ROOT_DIR/data/gen_reference" "$ROOT_DIR/.cache"

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
fi

echo "VPS setup complete. Set HF_TOKEN in $ROOT_DIR/.env before submitting a job."
echo "Use this interpreter: $VENV_PYTHON"
