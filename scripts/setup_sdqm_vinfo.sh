#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDQM_DIR="${SDQM_REPO_DIR:-$ROOT_DIR/third_party/SDQM}"
ULTRALYTICS_DIR="$SDQM_DIR/dataset_interpretability/v_info/ultralytics"
PYTHON="${PYTHON:-python}"

if [[ ! -d "$SDQM_DIR" ]]; then
  echo "SDQM repo not found at $SDQM_DIR"
  exit 1
fi

if [[ ! -d "$ULTRALYTICS_DIR" ]]; then
  echo "Custom ultralytics not found at $ULTRALYTICS_DIR"
  exit 1
fi

"$PYTHON" -m pip install -r "$ROOT_DIR/requirements-sdqm.txt"
"$PYTHON" -m pip install --no-deps --force-reinstall -e "$ULTRALYTICS_DIR"

echo "SDQM V-Info setup complete."
