#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cvat_bbox import (  # noqa: E402
    ConversionConfig,
    CvatGroundTruthConverter,
    load_json_object,
    write_json_atomic,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert native CVAT image exports into paired bbox ground truth."
    )
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument(
        "--cvat-export",
        required=True,
        type=Path,
        action="append",
        dest="cvat_exports",
        help="CVAT for images 1.1 ZIP/XML; repeat for multiple tasks.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("validation", "test", "all"),
        default="validation",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Create an in-progress pilot artifact and report missing/ambiguous pairs.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    converter = CvatGroundTruthConverter(
        load_json_object(arguments.template, "ground-truth template"),
        load_json_object(arguments.mapping, "annotation mapping"),
        ConversionConfig(
            split=arguments.split,
            allow_incomplete=arguments.allow_incomplete,
        ),
    )
    artifact, summary = converter.convert(arguments.cvat_exports)
    write_json_atomic(arguments.output, artifact)
    summary_path = arguments.output.with_suffix(".summary.json")
    write_json_atomic(summary_path, summary)
    print(f"Annotation status: {summary['annotation_status']}")
    print(
        "Pairs: "
        f"{summary['completed_pairs']}/{summary['selected_pairs']} completed, "
        f"{summary['unresolved_pairs']} unresolved"
    )
    print(f"Ground truth: {arguments.output}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
