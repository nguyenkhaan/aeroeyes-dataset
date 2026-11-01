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

from src.evaluation.bbox_vehicle_clip import conflict_crops, padded_crop_box

MODEL_ID = "openai/clip-vit-base-patch16"
PROMPTS = ["a photo of an ambulance", "a photo of a fire truck"]
CONTEXT = 0.1


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_run(run_dir: Path, metrics_path: Path) -> tuple[dict, dict]:
    manifest_bytes = (run_dir / "run-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance = metrics.get("provenance", {})
    if (manifest.get("status") != "completed" or manifest.get("split") != "validation"
            or manifest.get("config", {}).get("cross_class_iou_threshold") is not None
            or provenance.get("run_manifest_sha256") != digest(manifest_bytes)):
        raise ValueError("Expected a completed validation baseline without conflict suppression, bound to metrics")
    ids = manifest.get("selected_pair_ids")
    images = manifest.get("selected_images")
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or not isinstance(images, list) or len(images) != len(ids)
            or {item.get("stem") for item in images} != set(ids)
            or manifest.get("result", {}).get("processed_pairs") != len(ids)):
        raise ValueError("Run selection or image records are incomplete")
    reports, hashes = {}, {}
    for path in sorted((run_dir / "reports").glob("*.json")):
        data = path.read_bytes()
        report = json.loads(data)
        stem = report.get("stem")
        if not isinstance(stem, str) or not stem or stem in reports:
            raise ValueError(f"Invalid or duplicate report: {path}")
        reports[stem], hashes[stem] = report, digest(data)
    if set(reports) != set(ids) or hashes != provenance.get("prediction_report_sha256"):
        raise ValueError("Run reports differ from the saved metrics")
    return manifest, reports


def load_images(manifest: dict) -> dict:
    images = {}
    for record in manifest["selected_images"]:
        pair_images = {}
        for role, field in (("before", "reference_path"), ("after", "output_path")):
            path = Path(record[field])
            if digest(path.read_bytes()) != record.get(f"{role}_sha256"):
                raise ValueError(f"Image checksum mismatch: {record['stem']} {role}")
            pair_images[role] = path
        images[record["stem"]] = pair_images
    return images


def score_crops(manifest: dict, reports: dict, images: dict) -> tuple[list, dict]:
    import torch
    import transformers
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; submit this script as a GPU job")
    model = CLIPModel.from_pretrained(MODEL_ID).to("cuda").eval()
    revision = getattr(model.config, "_commit_hash", None)
    processor = CLIPProcessor.from_pretrained(MODEL_ID, revision=revision)
    scores = []
    for pair_id in manifest["selected_pair_ids"]:
        report = reports[pair_id]
        cases = conflict_crops(report, 0.75)
        for role in ("before", "after"):
            role_cases = [case for case in cases if case["role"] == role]
            if not role_cases:
                continue
            with Image.open(images[pair_id][role]) as source:
                image = source.convert("RGB")
            if image.size != (report["image_width"], report["image_height"]):
                raise ValueError(f"Image dimensions differ from report: {pair_id} {role}")
            crops = []
            for case in role_cases:
                case["crop_box"] = padded_crop_box(case["box"], image.size, CONTEXT)
                crops.append(image.crop(case["crop_box"]))
            for start in range(0, len(crops), 16):
                inputs = processor(text=PROMPTS, images=crops[start:start + 16],
                                   return_tensors="pt", padding=True).to("cuda")
                with torch.inference_mode():
                    outputs = model(**inputs)
                    similarities = outputs.logits_per_image.float() / model.logit_scale.exp().float()
                for case, values in zip(role_cases[start:start + 16], similarities.cpu().tolist(), strict=True):
                    case["cosine_similarity"] = dict(zip(("ambulance", "fire_truck"), values, strict=True))
                    case["ambulance_minus_fire_truck"] = values[0] - values[1]
                    scores.append(case)
        print(f"Scored {pair_id}: {len(cases)} conflicts", flush=True)
    return scores, {"model_id": MODEL_ID, "model_revision": revision,
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "gpu": torch.cuda.get_device_name()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score overlapping vehicle predictions with CLIP without changing boxes.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, reports = load_run(args.run_dir, args.metrics)
    images = load_images(manifest)
    scores, model = score_crops(manifest, reports, images)
    result = {"schema_version": "vehicle-conflict-clip-scores-1", "model": model,
              "prompts": PROMPTS, "conflict_iou": 0.75, "crop_context": CONTEXT,
              "run_manifest_sha256": digest((args.run_dir / "run-manifest.json").read_bytes()),
              "metrics_sha256": digest(args.metrics.read_bytes()),
              "script_sha256": digest(Path(__file__).read_bytes()),
              "selection_logic_sha256": digest((ROOT / "src/evaluation/bbox_vehicle_clip.py").read_bytes()),
              "pairs": len(reports), "conflicts": scores,
              "note": "Cosine similarities are not calibrated probabilities; no labels or detections were changed."}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Scored {len(scores)} conflicts across {len(reports)} pairs: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
