#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bbox_benchmark import (  # noqa: E402
    BenchmarkSelectionConfig,
    build_benchmark_manifest,
    load_generation_pairs,
    load_selection_config,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible paired Before/After bounding-box benchmark."
    )
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "bounding_box" / "benchmark.yaml",
    )
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def write_json_atomically(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> int:
    args = parse_arguments()
    records = load_generation_pairs(args.metadata_dir)
    protocol_config = load_selection_config(args.config)
    config = BenchmarkSelectionConfig(
        validation_size=(
            args.validation_size
            if args.validation_size is not None
            else protocol_config.validation_size
        ),
        test_size=(
            args.test_size
            if args.test_size is not None
            else protocol_config.test_size
        ),
        seed=args.seed if args.seed is not None else protocol_config.seed,
    )
    manifest = build_benchmark_manifest(records, config)
    write_json_atomically(args.output_manifest, manifest)
    print(
        "Bounding-box benchmark created: "
        f"{manifest['counts']['validation']} validation, "
        f"{manifest['counts']['test']} test. "
        f"Output: {args.output_manifest}",
        flush=True,
    )
    fallback_count = sum(
        item["source_group_method"] == "pair_id_fallback"
        for item in manifest["pairs"]
    )
    if fallback_count:
        print(
            f"Warning: {fallback_count} pair(s) lack an explicit source/event group; "
            "pair_id fallback was used. Record this limitation before training a detector.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
