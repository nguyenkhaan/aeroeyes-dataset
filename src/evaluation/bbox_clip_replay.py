from __future__ import annotations

import math

from src.evaluation.bbox_vehicle_clip import conflict_crops


def validate_scores(reports: dict, scores: list) -> dict:
    expected = {(item["pair_id"], item["role"], item["detection_index"]): item
                for report in reports.values() for item in conflict_crops(report, 0.75)}
    if not isinstance(scores, list):
        raise ValueError("Conflict scores must be a list")
    validated = {}
    for item in scores:
        if not isinstance(item, dict):
            raise ValueError("Invalid conflict score")
        key = (item.get("pair_id"), item.get("role"), item.get("detection_index"))
        if key in validated or key not in expected:
            raise ValueError("Duplicate or unexpected conflict score")
        target = expected[key]
        if any(item.get(field) != target[field] for field in
               ("box", "confidence", "overlap_vehicle_indices", "max_conflict_iou")):
            raise ValueError("Conflict score does not match the original detection")
        similarities = item.get("cosine_similarity")
        if not isinstance(similarities, dict):
            raise ValueError("Missing CLIP similarities")
        values = [similarities.get("ambulance"), similarities.get("fire_truck"),
                  item.get("ambulance_minus_fire_truck")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in values):
            raise ValueError("Invalid CLIP score")
        if not math.isclose(values[0] - values[1], values[2], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("CLIP margin differs from similarities")
        validated[key] = values[2]
    if set(validated) != set(expected):
        raise ValueError("Conflict scores do not cover the original reports")
    return validated


def filter_conflicts(report: dict, scores: dict) -> dict:
    filtered = dict(report)
    for role in ("before", "after"):
        filtered[f"{role}_detections"] = [
            detection for index, detection in enumerate(report[f"{role}_detections"])
            if scores.get((report["stem"], role, index), 1.0) > 0
        ]
    return filtered
