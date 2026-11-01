#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.bbox_strict_metrics import evaluate_strict_v2


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen bounding-box reports against strict validation v2.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    gt_bytes = args.ground_truth.read_bytes()
    ground_truth = json.loads(gt_bytes)
    manifest_path = args.run_dir / "run-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    ids = [item["pair_id"] for item in ground_truth["pairs"]]
    if manifest.get("status") != "completed" or manifest.get("split") != "validation":
        raise ValueError("Run must be a completed validation experiment")
    if manifest.get("benchmark_manifest_sha256") != ground_truth.get("source_manifest_sha256") or not ground_truth.get("source_manifest_sha256"):
        raise ValueError("Run and ground truth benchmark manifest checksums differ")
    run_ids = manifest.get("selected_pair_ids")
    if not isinstance(run_ids, list) or len(run_ids) != len(ids) or set(run_ids) != set(ids):
        raise ValueError("Run and ground truth validation pair IDs differ")
    if manifest.get("result", {}).get("processed_pairs") != len(ids):
        raise ValueError("Run did not process every selected pair")
    reports, report_hashes = {}, {}
    for path in sorted((args.run_dir / "reports").glob("*.json")):
        report_bytes = path.read_bytes()
        report = json.loads(report_bytes)
        if not isinstance(report, dict) or not isinstance(report.get("stem"), str):
            raise ValueError(f"Invalid prediction report: {path}")
        pair_id = report["stem"]
        if pair_id in reports:
            raise ValueError(f"Duplicate prediction report: {pair_id}")
        reports[pair_id] = report
        report_hashes[pair_id] = _digest(report_bytes)
    result = evaluate_strict_v2(ground_truth, reports)
    result["provenance"] = {
        "ground_truth_sha256": _digest(gt_bytes),
        "source_strict_manifest_sha256": ground_truth.get("source_strict_manifest_sha256"),
        "benchmark_manifest_sha256": ground_truth.get("source_manifest_sha256"),
        "run_manifest_sha256": _digest(manifest_bytes), "run_dir": str(args.run_dir),
        "run_git": manifest.get("git"), "run_config": manifest.get("config"),
        "run_result": manifest.get("result"), "prediction_report_sha256": report_hashes,
        "evaluator_script_sha256": _digest(Path(__file__).read_bytes()),
        "evaluator_logic_sha256": _digest((ROOT / "src/evaluation/bbox_strict_metrics.py").read_bytes()),
        "matching_logic_sha256": _digest((ROOT / "src/evaluation/bbox_metrics.py").read_bytes()),
        "prediction_report_note": "Report hashes recorded at evaluation; original inference did not bind individual report hashes.",
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Validation pairs: {result['coverage']['evaluated_pairs']}")
    for tier in ("detector", "final_pipeline"):
        metrics = result[tier]["overall"]
        print(f"{tier}: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
              f"P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']} "
              f"ignored={metrics['ignored_prediction_count']}")
    print(f"Metrics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
