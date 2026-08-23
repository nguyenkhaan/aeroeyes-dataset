from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.preflight import (
    PreflightOptions,
    collect_preflight_errors,
    preflight_summary,
)


def parse_arguments() -> PreflightOptions:
    parser = argparse.ArgumentParser(
        description="Validate VPS evaluation prerequisites without loading models."
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when the configured Python interpreter cannot access CUDA.",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Fail when fewer than two real or synthetic images are available.",
    )
    arguments = parser.parse_args()
    return PreflightOptions(
        require_cuda=arguments.require_cuda,
        require_images=arguments.require_images,
    )


def main() -> int:
    options = parse_arguments()
    errors = collect_preflight_errors(options)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    for line in preflight_summary():
        print(line)
    print("Evaluation preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
