from __future__ import annotations

import math

from src.evaluation.bbox_metrics import Box


def _box(record: dict, size: tuple[int, int]) -> Box:
    raw = record.get("box", {})
    values = [raw.get(key) for key in ("x_min", "y_min", "x_max", "y_max")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Invalid detection coordinates")
    box = Box(*values)
    box.validate()
    if box.x_min < 0 or box.y_min < 0 or box.x_max > size[0] or box.y_max > size[1]:
        raise ValueError("Detection box lies outside image")
    return box


def padded_crop_box(coords: list, size: tuple[int, int], context: float) -> tuple[int, int, int, int]:
    if not math.isfinite(context) or not 0 <= context <= 1:
        raise ValueError("Crop context must be in [0, 1]")
    left, top, right, bottom = coords
    dx, dy = (right - left) * context, (bottom - top) * context
    return (max(0, math.floor(left - dx)), max(0, math.floor(top - dy)),
            min(size[0], math.ceil(right + dx)), min(size[1], math.ceil(bottom + dy)))


def conflict_crops(report: dict, threshold: float) -> list[dict]:
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Conflict IoU must be in (0, 1]")
    size = (report.get("image_width"), report.get("image_height"))
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size):
        raise ValueError("Invalid report image size")
    conflicts = []
    for role in ("before", "after"):
        detections = report.get(f"{role}_detections")
        if not isinstance(detections, list):
            raise ValueError(f"Missing {role} detections")
        vehicles = [(index, _box(item, size)) for index, item in enumerate(detections)
                    if item.get("label") == "emergency_vehicle"]
        for index, item in enumerate(detections):
            if item.get("label") != "ambulance":
                continue
            box = _box(item, size)
            overlapping = [(other_index, box.iou(vehicle)) for other_index, vehicle in vehicles
                           if box.iou(vehicle) >= threshold]
            if overlapping:
                conflicts.append({"pair_id": report["stem"], "role": role, "detection_index": index,
                                  "box": [box.x_min, box.y_min, box.x_max, box.y_max],
                                  "confidence": item.get("confidence"),
                                  "overlap_vehicle_indices": [entry[0] for entry in overlapping],
                                  "max_conflict_iou": max(entry[1] for entry in overlapping)})
    return conflicts
