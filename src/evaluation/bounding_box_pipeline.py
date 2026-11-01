from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from src.core.config import OUTPUT_DIR, SDQM_YOLO_DATA_YAML
from src.evaluation.change_detection import BoundingBox, detect_change_delta
from src.evaluation.object_matching import (
    DetectedObject,
    ObjectMatchingReport,
    match_and_refine_objects,
)
from src.evaluation.yolo_export import RescueDetector, load_class_map


MIN_VISIBLE_DELTA_AREA_RATIO = 0.005


@dataclass(frozen=True)
class BoundingBoxRunConfig:
    metadata_dir: Path = Path(OUTPUT_DIR) / "_metadata"
    output_dir: Path = Path(OUTPUT_DIR) / "bounding_boxes"
    limit: int = 5


def load_generation_records(metadata_dir: Path, limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("Bounding-box pair limit must be positive")
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Generation metadata directory is missing: {metadata_dir}")

    records: list[dict] = []
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"Metadata must contain a JSON object: {metadata_path}")

        before_path = Path(str(record.get("reference_path", "")))
        after_path = Path(str(record.get("output_path", "")))
        if not record.get("stem") or not before_path.is_file() or not after_path.is_file():
            continue

        records.append(record)
        if len(records) >= limit:
            break
    return records


def _box_record(box: BoundingBox) -> dict:
    return {
        "x_min": box.x_min,
        "y_min": box.y_min,
        "x_max": box.x_max,
        "y_max": box.y_max,
        "area": box.area,
        "mean_delta": round(box.mean_delta, 6),
    }


def _object_record(detected: DetectedObject) -> dict:
    return {
        "class_id": detected.class_id,
        "label": detected.label,
        "confidence": round(detected.confidence, 6),
        "box": _box_record(detected.box),
    }


def _matching_record(report: ObjectMatchingReport) -> dict:
    return {
        "preservation_score": round(report.preservation_score, 6),
        "confirmed_rescue_count": report.confirmed_rescue_count,
        "is_consistent_rescue": report.is_consistent_rescue,
        "preserved_objects": [
            {
                "before": _object_record(before),
                "after": _object_record(after),
                "iou": round(iou, 6),
            }
            for before, after, iou in report.preserved_objects
        ],
        "added_rescue_objects": [
            {"detection": _object_record(detected), "delta_overlap": round(overlap, 6)}
            for detected, overlap in report.added_rescue_objects
        ],
        "refined_rescue_objects": [
            _object_record(detected) for detected in report.refined_rescue_objects
        ],
        "rejected_detections": [
            {"detection": _object_record(detected), "reason": reason}
            for detected, reason in report.rejected_detections
        ],
    }


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _draw_visualization(
    image: Image.Image,
    delta_boxes: list[BoundingBox],
    confirmed_objects: list[DetectedObject],
) -> Image.Image:
    visualization = image.convert("RGB").copy()
    draw = ImageDraw.Draw(visualization)
    image_area = image.width * image.height
    for box in delta_boxes:
        if box.area / image_area < MIN_VISIBLE_DELTA_AREA_RATIO:
            continue
        draw.rectangle(
            (box.x_min, box.y_min, box.x_max, box.y_max),
            outline=(255, 165, 0),
            width=2,
        )
    for detected in confirmed_objects:
        box = detected.box
        draw.rectangle(
            (box.x_min, box.y_min, box.x_max, box.y_max),
            outline=(0, 255, 80),
            width=3,
        )
        draw.text(
            (box.x_min + 2, max(0, box.y_min - 12)),
            f"{detected.label} {detected.confidence:.2f}",
            fill=(0, 255, 80),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    return visualization


def _prepare_output_directories(output_dir: Path) -> dict[str, Path]:
    directories = {
        "labels": output_dir / "labels",
        "reports": output_dir / "reports",
        "visualizations": output_dir / "visualizations",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _process_record(record: dict, detector: RescueDetector, output_dir: Path) -> dict:
    stem = str(record["stem"])
    before_path = Path(record["reference_path"])
    after_path = Path(record["output_path"])
    directories = _prepare_output_directories(output_dir)

    with Image.open(before_path) as before_source, Image.open(after_path) as after_source:
        before_image = before_source.convert("RGB")
        after_image = after_source.convert("RGB")
    if after_image.size != before_image.size:
        after_image = after_image.resize(before_image.size, Image.Resampling.BILINEAR)

    delta_result = detect_change_delta(before_image, after_image)
    before_objects = detector.detect_objects(before_image)
    before_clip_decisions = list(getattr(detector, "last_clip_decisions", []))
    after_objects = detector.detect_objects(after_image)
    after_clip_decisions = list(getattr(detector, "last_clip_decisions", []))
    matching = match_and_refine_objects(before_objects, after_objects, delta_result)

    label_lines = [
        detected.to_yolo(*after_image.size)
        for detected in matching.refined_rescue_objects
    ]
    (directories["labels"] / f"{stem}.txt").write_text(
        "".join(label_lines),
        encoding="utf-8",
    )
    visualization = _draw_visualization(
        after_image,
        delta_result.bounding_boxes,
        matching.refined_rescue_objects,
    )
    visualization.save(directories["visualizations"] / f"{stem}.png")

    result = {
        "stem": stem,
        "before_path": str(before_path),
        "after_path": str(after_path),
        "image_width": after_image.width,
        "image_height": after_image.height,
        "delta": {
            "mean": round(delta_result.mean_delta, 6),
            "area_ratio": round(delta_result.delta_area_ratio, 6),
            "is_valid_change": delta_result.is_valid_change,
            "boxes": [_box_record(box) for box in delta_result.bounding_boxes],
        },
        "before_detections": [_object_record(item) for item in before_objects],
        "after_detections": [_object_record(item) for item in after_objects],
        "matching": _matching_record(matching),
        "bbox_passed": delta_result.is_valid_change and matching.is_consistent_rescue,
        "label_path": str(directories["labels"] / f"{stem}.txt"),
        "visualization_path": str(directories["visualizations"] / f"{stem}.png"),
    }
    if getattr(detector, "vehicle_clip_filter", None) is not None:
        result["vehicle_clip_decisions"] = {"before": before_clip_decisions, "after": after_clip_decisions}
    _write_json(directories["reports"] / f"{stem}.json", result)
    return result


def run_bounding_box_pipeline(
    config: BoundingBoxRunConfig,
    detector: RescueDetector | None = None,
) -> dict:
    records = load_generation_records(config.metadata_dir, config.limit)
    if len(records) < config.limit:
        raise ValueError(
            f"Only {len(records)} complete generated pair(s) found in "
            f"{config.metadata_dir}; requested {config.limit}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    active_detector = detector or RescueDetector(
        class_map=load_class_map(SDQM_YOLO_DATA_YAML)
    )
    active_detector.load()

    results: list[dict] = []
    for index, record in enumerate(records, start=1):
        print(f"Bounding boxes: {index}/{len(records)} | {record['stem']}", flush=True)
        results.append(_process_record(record, active_detector, config.output_dir))

    summary = {
        "requested_pairs": config.limit,
        "processed_pairs": len(results),
        "passed_pairs": sum(bool(item["bbox_passed"]) for item in results),
        "confirmed_rescue_objects": sum(
            int(item["matching"]["confirmed_rescue_count"]) for item in results
        ),
        "results": results,
    }
    _write_json(config.output_dir / "summary.json", summary)
    return summary
