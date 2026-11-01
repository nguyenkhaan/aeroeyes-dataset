#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bbox_annotation import (  # noqa: E402
    AnnotationBundleConfig,
    create_bundle,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "bounding_box" / "benchmark.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare reproducible CVAT bundles for paired bounding boxes."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--double-annotation-fraction", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _load_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid benchmark manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark manifest must contain an object: {path}")
    return value


def _load_protocol(path: Path) -> tuple[float, int]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        fraction = float(value["quality_control"]["double_annotation_fraction"])
        seed = int(value["selection"]["seed"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid bounding-box protocol: {path}") from exc
    return fraction, seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    arguments = parse_arguments()
    protocol_fraction, protocol_seed = _load_protocol(arguments.config)
    config = AnnotationBundleConfig(
        double_annotation_fraction=(
            arguments.double_annotation_fraction
            if arguments.double_annotation_fraction is not None
            else protocol_fraction
        ),
        seed=arguments.seed if arguments.seed is not None else protocol_seed,
        source_manifest_sha256=_sha256(arguments.manifest),
    )
    summary = create_bundle(
        _load_manifest(arguments.manifest),
        arguments.output_dir,
        config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
