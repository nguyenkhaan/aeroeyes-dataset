#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.bbox_visual_comparison import select_review_cases
from src.evaluation.bbox_visual_render import render_comparison


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified_run(run_dir: Path, metrics_path: Path, gt_hash: str, benchmark_hash: str) -> tuple[dict, dict, dict]:
    manifest_bytes = (run_dir / "run-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance = metrics.get("provenance", {})
    if (manifest.get("status") != "completed" or manifest.get("split") != "validation"
        or manifest.get("benchmark_manifest_sha256") != benchmark_hash
        or provenance.get("ground_truth_sha256") != gt_hash
        or provenance.get("run_manifest_sha256") != _hash(manifest_bytes)):
        raise ValueError("Run, benchmark, or ground truth differs from saved metrics")
    for key, path in (("evaluator_logic_sha256", ROOT / "src/evaluation/bbox_strict_metrics.py"),
                      ("matching_logic_sha256", ROOT / "src/evaluation/bbox_metrics.py")):
        if provenance.get(key) != _hash(path.read_bytes()):
            raise ValueError(f"Evaluator code differs from saved metrics: {key}")
    reports, hashes = {}, {}
    for path in sorted((run_dir / "reports").glob("*.json")):
        data = path.read_bytes()
        report = json.loads(data)
        stem = report.get("stem")
        if not isinstance(stem, str) or not stem or stem in reports:
            raise ValueError(f"Invalid or duplicate report: {path}")
        reports[stem], hashes[stem] = report, _hash(data)
    if hashes != provenance.get("prediction_report_sha256"):
        raise ValueError("Prediction reports differ from saved metrics")
    return manifest, metrics, reports


def _verify_runs(baseline: dict, candidate: dict, first_metrics: dict, second_metrics: dict) -> None:
    protocol = first_metrics.get("evaluation_protocol", {})
    if (protocol.get("version") != "bbox-strict-validation-v2-iou50-1"
        or protocol != second_metrics.get("evaluation_protocol")
        or first_metrics.get("coverage") != second_metrics.get("coverage")):
        raise ValueError("Evaluation protocols or coverage differ")
    first, second = dict(baseline["config"]), dict(candidate["config"])
    first.pop("name", None)
    second.pop("name", None)
    if first.pop("cross_class_iou_threshold", None) is not None or second.pop("cross_class_iou_threshold", None) is None or first != second:
        raise ValueError("Runs must differ only by enabled cross-class conflict threshold")
    if not baseline.get("git", {}).get("commit") or baseline.get("git", {}).get("commit") != candidate.get("git", {}).get("commit"):
        raise ValueError("Inference commits do not match")
    if any(run.get("git", {}).get("tracked_files_dirty") is not False for run in (baseline, candidate)):
        raise ValueError("Inference working tree was dirty or unknown")
    for field in ("model_revision", "ontology_sha256", "class_map_sha256"):
        left, right = baseline.get("result", {}).get(field), candidate.get("result", {}).get(field)
        if not left or left != right:
            raise ValueError(f"Inference provenance differs: {field}")


def _image_map(manifest: dict, pair_ids: set[str]) -> dict:
    records = manifest.get("selected_images")
    if not isinstance(records, list) or len(records) != len(pair_ids):
        raise ValueError("Run image records are incomplete")
    images = {item["stem"]: item for item in records}
    if len(images) != len(records) or set(images) != pair_ids:
        raise ValueError("Run image IDs do not match GT")
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description="Render verified before/GT/baseline/candidate conflict-review figures.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--candidate-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-extra-fp-pairs", type=int, default=5)
    args = parser.parse_args()
    gt_bytes = args.ground_truth.read_bytes()
    document = json.loads(gt_bytes)
    benchmark_hash = document.get("source_manifest_sha256")
    if not benchmark_hash:
        raise ValueError("Ground truth has no benchmark checksum")
    base, base_metrics, base_reports = _verified_run(args.baseline_run, args.baseline_metrics, _hash(gt_bytes), benchmark_hash)
    candidate, candidate_metrics, candidate_reports = _verified_run(args.candidate_run, args.candidate_metrics, _hash(gt_bytes), benchmark_hash)
    _verify_runs(base, candidate, base_metrics, candidate_metrics)
    pair_ids = {record["pair_id"] for record in document["pairs"]}
    base_images, candidate_images = _image_map(base, pair_ids), _image_map(candidate, pair_ids)
    cases = select_review_cases(document, base_reports, candidate_reports, args.max_extra_fp_pairs)
    for manifest in (base, candidate):
        selected = manifest.get("selected_pair_ids")
        if not isinstance(selected, list) or len(selected) != len(pair_ids) or set(selected) != pair_ids:
            raise ValueError("Run selection differs from GT pairs")
        if manifest.get("result", {}).get("processed_pairs") != len(pair_ids):
            raise ValueError("Run did not process every GT pair")
    records = {record["pair_id"]: record for record in document["pairs"]}
    prepared = []
    source_ids = [case["source_pair_id"] for case in cases]
    if len(set(source_ids)) != len(source_ids) or any(not isinstance(value, str) or not value.isascii() or not value.isdigit() for value in source_ids):
        raise ValueError("Selected source pair IDs are unsafe or duplicated")
    for case in cases:
        pair_id = case["pair_id"]
        first, second = base_images[pair_id], candidate_images[pair_id]
        paths = []
        for role, field in (("before", "reference_path"), ("after", "output_path")):
            key = f"{role}_sha256"
            if not first.get(key) or first[key] != second.get(key):
                raise ValueError(f"Image checksum differs between runs: {pair_id} {role}")
            path = Path(first[field])
            if _hash(path.read_bytes()) != first[key]:
                raise ValueError(f"Image checksum differs from run manifest: {pair_id} {role}")
            paths.append(path)
            case[key] = first[key]
        prepared.append((case, paths))
    from PIL import Image  # noqa: F401

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for case, paths in prepared:
        pair_id = case["pair_id"]
        name = f"pair-{case['source_pair_id']}.png"
        render_comparison(case, records[pair_id], (base_reports[pair_id], candidate_reports[pair_id]), paths, args.output_dir / name)
        case["figure"] = name
        case["figure_sha256"] = _hash((args.output_dir / name).read_bytes())
    index = {"ground_truth_sha256": _hash(gt_bytes), "baseline_run": str(args.baseline_run),
             "candidate_run": str(args.candidate_run),
             "baseline_metrics_sha256": _hash(args.baseline_metrics.read_bytes()),
             "candidate_metrics_sha256": _hash(args.candidate_metrics.read_bytes()), "cases": cases,
             "note": "Lost means a GT object matched in baseline but not candidate; cause is not established. FP overlaps are geometric hints."}
    (args.output_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    entries = "\n".join(f'<h2>Pair {html.escape(case["source_pair_id"])}</h2><img src="{html.escape(case["figure"], quote=True)}" style="max-width:100%">' for case in cases)
    page = "<!doctype html><meta charset='utf-8'><title>BBox conflict review</title><h1>BBox conflict review</h1><p>Before, GT After, baseline, candidate. Cyan: ambulance; orange: emergency vehicle; purple: ambiguous GT.</p>" + entries
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    archive_path = args.output_dir / "review-bundle.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("index.html", "index.json", *(case["figure"] for case in cases)):
            archive.write(args.output_dir / name, arcname=name)
    print(f"Lost GT ambulance objects: {sum(len(item['lost_ambulance_gt_ids']) for item in cases)}")
    print(f"Selected pairs: {len(cases)}")
    print(f"Output: {args.output_dir}")
    print(f"Review archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
