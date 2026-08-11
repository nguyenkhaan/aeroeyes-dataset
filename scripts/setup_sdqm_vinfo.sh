#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDQM_DIR="${SDQM_REPO_DIR:-$ROOT_DIR/third_party/SDQM}"
ULTRALYTICS_DIR="$SDQM_DIR/dataset_interpretability/v_info/ultralytics"

if [[ ! -d "$SDQM_DIR" ]]; then
  echo "SDQM repo not found at $SDQM_DIR"
  echo "Run: git clone https://github.com/ayushzenith/SDQM.git third_party/SDQM"
  exit 1
fi

if [[ ! -d "$ULTRALYTICS_DIR" ]]; then
  echo "Custom ultralytics not found at $ULTRALYTICS_DIR"
  exit 1
fi

pip install -r "$ROOT_DIR/requirements-sdqm.txt"
pip install -e "$ULTRALYTICS_DIR"

echo "SDQM Phase 3 setup complete."
