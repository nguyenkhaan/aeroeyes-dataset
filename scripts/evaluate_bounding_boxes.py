#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.bbox_metrics import EvaluationConfig, evaluate_benchmark


DEFAULT_ONTOLOGY = REPOSITORY_ROOT / "config" / "bounding_box" / "ontology.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rescue-object bounding boxes against paired ground truth."
    )
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument(
        "--split",
        choices=("validation", "test", "all"),
        default="validation",
        help="Evaluate validation by default so test is not opened during tuning.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="Optional JSON or YAML file describing the frozen prediction run.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}") from exc


def load_ground_truth(path: Path, split: str) -> list[dict]:
    document = _load_json(path)
    if isinstance(document, dict):
        pairs = document.get("pairs")
    else:
        pairs = document
    if not isinstance(pairs, list) or not all(isinstance(item, dict) for item in pairs):
        raise ValueError(
            "Ground truth must be a list or an object containing a pairs list"
        )
    if split == "all":
        return pairs
    selected = [item for item in pairs if item.get("split") == split]
    if not selected:
        raise ValueError(f"Ground truth contains no pairs for split {split}")
    return selected


def load_prediction_reports(directory: Path) -> dict[str, dict]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Prediction report directory is missing: {directory}")
    reports: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        report = _load_json(path)
        if not isinstance(report, dict) or not report.get("stem"):
            continue
        pair_id = str(report["stem"])
        if pair_id in reports:
            raise ValueError(f"Duplicate prediction report for pair_id {pair_id}")
        reports[pair_id] = report
    return reports


def load_class_names(path: Path) -> tuple[str, ...]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        classes = document["classes"]
        names = tuple(str(item["name"]) for item in classes)
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid bounding-box ontology: {path}") from exc
    if not names:
        raise ValueError(f"Bounding-box ontology has no classes: {path}")
    return names


def load_run_manifest(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        if path.suffix.lower() == ".json":
            document = json.loads(path.read_text(encoding="utf-8"))
        else:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid run manifest: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Run manifest must contain an object: {path}")
    return json.loads(json.dumps(document, default=str))


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> int:
    arguments = parse_arguments()
    result = evaluate_benchmark(
        load_ground_truth(arguments.ground_truth, arguments.split),
        load_prediction_reports(arguments.predictions),
        EvaluationConfig(class_names=load_class_names(arguments.ontology)),
    )
    result["evaluation_protocol"]["split"] = arguments.split
    result["run_manifest"] = load_run_manifest(arguments.run_manifest)
    write_json_atomic(arguments.output, result)
    overall = result["final_pipeline"]["overall"]
    print(f"Evaluated pairs: {result['generation']['total_pairs']}")
    print(
        "Final pipeline: "
        f"P={overall['precision']} R={overall['recall']} "
        f"F1={overall['f1']} AP50={overall['ap50']}"
    )
    print(f"Metrics: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
