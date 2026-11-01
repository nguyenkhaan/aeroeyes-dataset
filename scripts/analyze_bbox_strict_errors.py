#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.bbox_error_analysis import analyze_strict_errors


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose strict-v2 FP/FN from a verified existing run.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    gt_bytes = args.ground_truth.read_bytes()
    manifest_bytes = (args.run_dir / "run-manifest.json").read_bytes()
    document, manifest = json.loads(gt_bytes), json.loads(manifest_bytes)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    provenance = metrics.get("provenance", {})
    if provenance.get("ground_truth_sha256") != digest(gt_bytes) or provenance.get("run_manifest_sha256") != digest(manifest_bytes):
        raise ValueError("Ground truth or run manifest differs from evaluated metrics")
    if manifest.get("status") != "completed" or manifest.get("split") != "validation":
        raise ValueError("Expected a completed validation run")
    for key, path in (("evaluator_logic_sha256", ROOT / "src/evaluation/bbox_strict_metrics.py"),
                      ("matching_logic_sha256", ROOT / "src/evaluation/bbox_metrics.py")):
        if provenance.get(key) != digest(path.read_bytes()):
            raise ValueError(f"Evaluator code differs from stored metrics: {key}")
    reports, hashes = {}, {}
    for path in sorted((args.run_dir / "reports").glob("*.json")):
        data = path.read_bytes()
        report = json.loads(data)
        pair_id = report.get("stem")
        if not isinstance(pair_id, str) or not pair_id or pair_id in reports:
            raise ValueError(f"Invalid or duplicate prediction report: {path}")
        reports[pair_id], hashes[pair_id] = report, digest(data)
    if hashes != provenance.get("prediction_report_sha256"):
        raise ValueError("Prediction reports differ from stored metrics")
    result = analyze_strict_errors(document, reports)
    if result["coverage"] != metrics.get("coverage"):
        raise ValueError("Ground truth coverage differs from stored metrics")
    for tier, summary in result["summary"].items():
        for name, counts in summary["per_class"].items():
            expected = metrics[tier]["per_class"][name]
            for key, metric_key in (("tp", "tp"), ("fp", "fp"), ("fn", "fn"), ("ignored", "ignored_prediction_count")):
                if counts[key] != expected[metric_key]:
                    raise ValueError(f"Diagnostic differs from stored metrics: {tier}/{name}/{key}")
    image_records = {item["stem"]: item for item in manifest.get("selected_images", [])}
    for row in result["rows"]:
        images = image_records.get(row["pair_id"], {})
        row["before_path"] = images.get("reference_path")
        row["after_path"] = images.get("output_path")
    result["provenance"] = {"ground_truth_sha256": digest(gt_bytes), "run_manifest_sha256": digest(manifest_bytes),
                            "metrics_sha256": digest(args.metrics.read_bytes()), "run_dir": str(args.run_dir)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "analysis.json"
    csv_path = args.output_dir / "fp_fn.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ("tier", "outcome", "class_name", "reason", "source_pair_id", "pair_id", "confidence",
              "prediction_index", "object_id", "bbox", "related_class", "related_object_id", "related_iou",
              "generation_status", "before_file", "after_file", "before_path", "after_path")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"coverage": result["coverage"], "summary": result["summary"],
                      "note": result["note"], "analysis_json": str(json_path), "fp_fn_csv": str(csv_path)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
