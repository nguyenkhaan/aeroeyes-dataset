#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bbox_experiment import (  # noqa: E402
    ExperimentConfig,
    load_validation_pairs,
    read_json_object,
    select_ground_truth,
    sha256_file,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated Grounding DINO experiment on validation pairs."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Use a reproducible smoke-test subset.")
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Check inputs and write the selected ground truth without loading a GPU model.",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_files_dirty": None}
    return {"commit": commit, "tracked_files_dirty": dirty}


def run_inference(records: list[dict], config: ExperimentConfig, output_dir: Path) -> dict:
    import numpy as np
    import torch
    import transformers
    import yaml
    from PIL import Image

    from src.core.config import SDQM_YOLO_DATA_YAML
    from src.evaluation.bounding_box_pipeline import _process_record
    from src.evaluation.bbox_aliases import build_alias_map, experiment_alias_map
    from src.evaluation.bbox_prompt_supplement import validate_prompt_classes
    from src.evaluation.grounding_postprocess import grounding_postprocess_parameters
    from src.evaluation.yolo_export import RescueDetector, load_class_map

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; submit this experiment as a GPU job")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    for record in records:
        with (
            Image.open(record["reference_path"]) as before,
            Image.open(record["output_path"]) as after,
        ):
            if before.size != after.size:
                raise ValueError(f"Before/After image sizes differ for {record['stem']}")

    ontology_path = PROJECT_ROOT / "config" / "bounding_box" / "ontology.yaml"
    class_map_path = Path(SDQM_YOLO_DATA_YAML)
    class_map_hash = sha256_file(class_map_path)
    class_map = load_class_map(class_map_path)
    if sha256_file(class_map_path) != class_map_hash:
        raise ValueError("Detector class map changed while loading")
    ontology_bytes = ontology_path.read_bytes()
    ontology_hash = hashlib.sha256(ontology_bytes).hexdigest()
    ontology = yaml.safe_load(ontology_bytes)
    aliases = experiment_alias_map(build_alias_map(ontology, class_map), config.label_mapping_mode)
    validate_prompt_classes(config.supplemental_prompts, aliases)
    detector = RescueDetector(
        model_name=config.model_id,
        text_prompt=config.prompt,
        supplemental_prompts=config.supplemental_prompts,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
        class_thresholds={"rescue_boat": config.rescue_boat_threshold},
        nms_iou_threshold=config.nms_iou_threshold,
        class_map=class_map,
        aliases=aliases,
        device="cuda",
    )
    detector.load()
    _box_parameter, supports_text = grounding_postprocess_parameters(detector._processor)
    if not supports_text:
        raise RuntimeError(
            "Installed Transformers does not support text_threshold; "
            "this experiment cannot be compared safely."
        )
    torch.cuda.reset_peak_memory_stats()
    results = []
    for index, record in enumerate(records, start=1):
        print(f"Bounding boxes: {index}/{len(records)} | {record['stem']}", flush=True)
        results.append(_process_record(record, detector, output_dir))
    if detector.postprocess_mode != "box_and_text_threshold":
        raise RuntimeError(
            "Installed Transformers did not apply text_threshold; "
            "this experiment cannot be compared safely."
        )
    model_config = getattr(detector._model, "config", None)
    return {
        "processed_pairs": len(results),
        "passed_pairs": sum(bool(item["bbox_passed"]) for item in results),
        "confirmed_rescue_objects": sum(
            int(item["matching"]["confirmed_rescue_count"]) for item in results
        ),
        "model_revision": getattr(model_config, "_commit_hash", None),
        "postprocess_mode": detector.postprocess_mode,
        "ontology_sha256": ontology_hash,
        "class_map_sha256": class_map_hash,
        "label_mapping_counts": {
            "raw_labels": dict(sorted(detector.raw_label_counts.items())),
            "automatic_aliases": dict(sorted(detector.alias_label_counts.items())),
            "unmapped_labels": dict(sorted(detector.unmapped_label_counts.items())),
        },
        "supplemental_prompt_counts": {
            prompt: dict(counts) for prompt, counts in sorted(detector.supplemental_prompt_counts.items())
        },
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def main() -> int:
    args = parse_arguments()
    config = ExperimentConfig.from_dict(read_json_object(args.config))
    manifest_hash = sha256_file(args.manifest)
    records = load_validation_pairs(args.manifest, args.limit)
    gt = select_ground_truth(read_json_object(args.ground_truth), records, manifest_hash)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected_path = args.output_dir / "ground-truth.selected.json"
    write_json(selected_path, gt)
    run_manifest = {
        "schema_version": 1,
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.as_dict(),
        "config_sha256": sha256_file(args.config),
        "benchmark_manifest_sha256": manifest_hash,
        "source_ground_truth_sha256": sha256_file(args.ground_truth),
        "selected_ground_truth_sha256": sha256_file(selected_path),
        "split": "validation",
        "selected_pair_ids": [record["stem"] for record in records],
        "selected_images": records,
        "git": git_info(),
        "python": platform.python_version(),
        "ground_truth_status": gt.get("annotation_status"),
    }
    manifest_path = args.output_dir / "run-manifest.json"
    write_json(manifest_path, run_manifest)
    print(f"Prepared {len(records)} validation pairs: {args.output_dir}", flush=True)
    if args.prepare_only:
        return 0
    started = datetime.now(timezone.utc)
    try:
        run_manifest["result"] = run_inference(records, config, args.output_dir)
        run_manifest["status"] = "completed"
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run_manifest["elapsed_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
        write_json(manifest_path, run_manifest)
    print(f"Experiment completed: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
