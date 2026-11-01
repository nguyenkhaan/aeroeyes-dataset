#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bbox_experiment import compare_experiment_runs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two compatible validation experiments.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--vary", default="box_threshold",
        choices=("box_threshold", "text_threshold", "rescue_boat_threshold", "nms_iou_threshold", "prompt", "label_mapping_mode", "supplemental_prompts", "cross_class_iou_threshold", "vehicle_clip_filter"),
    )
    args = parser.parse_args()
    result = compare_experiment_runs(args.baseline, args.candidate, args.vary)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
