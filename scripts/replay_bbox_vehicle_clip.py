#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_bbox_vehicle_conflicts import CONTEXT, MODEL_ID, PROMPTS, load_images, load_run
from src.evaluation.bbox_clip_replay import filter_conflicts, validate_scores
from src.evaluation.bbox_strict_metrics import evaluate_strict_v2
from src.evaluation.bbox_vehicle_clip import padded_crop_box


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_inputs(args: argparse.Namespace) -> tuple:
    gt_bytes = args.ground_truth.read_bytes()
    metrics_bytes = args.baseline_metrics.read_bytes()
    score_bytes = args.clip_scores.read_bytes()
    ground_truth = json.loads(gt_bytes)
    metrics = json.loads(metrics_bytes)
    score_file = json.loads(score_bytes)
    manifest, reports = load_run(args.baseline_run, args.baseline_metrics)
    provenance = metrics.get("provenance", {})
    if (provenance.get("ground_truth_sha256") != digest(gt_bytes)
            or ground_truth.get("source_manifest_sha256") != manifest.get("benchmark_manifest_sha256")
            or not ground_truth.get("source_manifest_sha256")):
        raise ValueError("Ground truth differs from baseline metrics or benchmark")
    for field, path in (("evaluator_logic_sha256", ROOT / "src/evaluation/bbox_strict_metrics.py"),
                        ("matching_logic_sha256", ROOT / "src/evaluation/bbox_metrics.py")):
        if provenance.get(field) != digest(path.read_bytes()):
            raise ValueError(f"Evaluator differs from saved baseline: {field}")
    if (score_file.get("schema_version") != "vehicle-conflict-clip-scores-1"
            or score_file.get("metrics_sha256") != digest(metrics_bytes)
            or score_file.get("run_manifest_sha256") != provenance.get("run_manifest_sha256")
            or score_file.get("script_sha256") != digest((ROOT / "scripts/score_bbox_vehicle_conflicts.py").read_bytes())
            or score_file.get("selection_logic_sha256") != digest((ROOT / "src/evaluation/bbox_vehicle_clip.py").read_bytes())
            or score_file.get("pairs") != len(reports)
            or score_file.get("conflict_iou") != 0.75 or score_file.get("crop_context") != CONTEXT
            or score_file.get("prompts") != PROMPTS or score_file.get("model", {}).get("model_id") != MODEL_ID
            or not score_file.get("model", {}).get("model_revision")):
        raise ValueError("CLIP scores do not match the verified baseline or scoring protocol")
    scores = validate_scores(reports, score_file.get("conflicts"))
    for item in score_file["conflicts"]:
        report = reports[item["pair_id"]]
        size = (report["image_width"], report["image_height"])
        if item.get("crop_box") != list(padded_crop_box(item["box"], size, CONTEXT)):
            raise ValueError("CLIP crop coordinates differ from scoring protocol")
    baseline = evaluate_strict_v2(ground_truth, reports)
    if any(metrics.get(key) != value for key, value in baseline.items()):
        raise ValueError("Baseline metrics cannot be reproduced from frozen reports")
    images = load_images(manifest)
    return ground_truth, manifest, reports, scores, images, baseline, score_file


def objects(records: list) -> list:
    from src.evaluation.change_detection import BoundingBox
    from src.evaluation.object_matching import DetectedObject

    return [DetectedObject(class_id=item["class_id"], label=item["label"],
                           box=BoundingBox(**item["box"]), confidence=item["confidence"])
            for item in records]


def replay_report(report: dict, filtered: dict, image_paths: dict) -> dict:
    from PIL import Image
    from src.evaluation.bounding_box_pipeline import _matching_record
    from src.evaluation.change_detection import detect_change_delta
    from src.evaluation.object_matching import match_and_refine_objects

    with Image.open(image_paths["before"]) as source:
        before = source.convert("RGB")
    with Image.open(image_paths["after"]) as source:
        after = source.convert("RGB")
    if before.size != after.size or before.size != (report["image_width"], report["image_height"]):
        raise ValueError(f"Image dimensions differ: {report['stem']}")
    delta = detect_change_delta(before, after)
    original_delta = report["delta"]
    if (round(delta.mean_delta, 6) != original_delta["mean"]
            or round(delta.delta_area_ratio, 6) != original_delta["area_ratio"]
            or delta.is_valid_change != original_delta["is_valid_change"]):
        raise ValueError(f"Original delta cannot be reproduced: {report['stem']}")
    original = match_and_refine_objects(objects(report["before_detections"]),
                                        objects(report["after_detections"]), delta)
    if (_matching_record(original) != report["matching"]
            or (delta.is_valid_change and original.is_consistent_rescue) != report["bbox_passed"]):
        raise ValueError(f"Original matching cannot be reproduced: {report['stem']}")
    candidate = match_and_refine_objects(objects(filtered["before_detections"]),
                                         objects(filtered["after_detections"]), delta)
    return {**filtered, "matching": _matching_record(candidate),
            "bbox_passed": delta.is_valid_change and candidate.is_consistent_rescue}


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay baseline pipeline using frozen CLIP conflict scores.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument("--clip-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gt, manifest, reports, scores, images, baseline, clip = validate_inputs(args)
    candidate_reports = {}
    for pair_id in manifest["selected_pair_ids"]:
        filtered = filter_conflicts(reports[pair_id], scores)
        candidate_reports[pair_id] = replay_report(reports[pair_id], filtered, images[pair_id])
        print(f"Replayed {pair_id}", flush=True)
    candidate = evaluate_strict_v2(gt, candidate_reports)
    if candidate["coverage"] != baseline["coverage"] or candidate["evaluation_protocol"] != baseline["evaluation_protocol"]:
        raise ValueError("Baseline and candidate evaluation coverage differs")
    removed = {role: sum(margin <= 0 for (pair_id, name, index), margin in scores.items() if name == role)
               for role in ("before", "after")}
    import numpy
    try:
        import cv2
        opencv_version = cv2.__version__
    except ImportError:
        opencv_version = None
    result = {"schema_version": "bbox-clip-replay-1", "ground_truth_sha256": digest(args.ground_truth.read_bytes()),
              "baseline_metrics_sha256": digest(args.baseline_metrics.read_bytes()),
              "baseline_run_manifest_sha256": clip["run_manifest_sha256"],
              "clip_scores_sha256": digest(args.clip_scores.read_bytes()), "clip_model": clip["model"],
              "rule": "suppress conflicting ambulance when cosine(ambulance)-cosine(fire_truck) <= 0",
              "conflict_iou": 0.75, "removed": removed,
              "baseline": baseline, "candidate": candidate,
              "limitations": ["Exploratory rule selected after inspecting validation; not an independent test.",
                              "Frozen predictions and CLIP scores are replayed; no new model inference was run.",
                              "Original inference did not bind individual report hashes."],
              "runtime": {"python": platform.python_version(), "numpy": numpy.__version__, "opencv": opencv_version},
              "pipeline_source_sha256": {name: digest((ROOT / "src/evaluation" / name).read_bytes())
                                         for name in ("bounding_box_pipeline.py", "change_detection.py", "object_matching.py")},
              "replay_script_sha256": digest(Path(__file__).read_bytes()),
              "replay_logic_sha256": digest((ROOT / "src/evaluation/bbox_clip_replay.py").read_bytes())}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    for name in ("detector", "final_pipeline"):
        for label, metrics in (("baseline", baseline), ("candidate", candidate)):
            overall = metrics[name]["overall"]
            ambulance = metrics[name]["per_class"]["ambulance"]
            print(f"{name} {label}: P={overall['precision']} R={overall['recall']} F1={overall['f1']} "
                  f"ambulance TP={ambulance['tp']} FP={ambulance['fp']} FN={ambulance['fn']}")
    print(f"Removed: {removed}; output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
