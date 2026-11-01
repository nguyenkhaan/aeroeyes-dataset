from __future__ import annotations

import math
from pathlib import PurePosixPath


CLASSES = {"firefighter", "rescue_boat", "helicopter", "ambulance", "emergency_vehicle"}
STATUSES = {"valid", "ambiguous", "missing_requested_object", "invalid_object", "severe_scene_corruption"}


def _objects(values: object, frame: dict, role: str) -> list[dict]:
    if not isinstance(values, list):
        raise ValueError(f"{role} objects must be a list")
    width, height = frame["width"], frame["height"]
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    result = []
    for item in values:
        index = item.get("box_index_in_frame")
        label = item.get("label")
        box = item.get("bbox_xyxy")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or label not in CLASSES:
            raise ValueError("Invalid box index or label")
        if not isinstance(box, list) or len(box) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box
        ):
            raise ValueError("Invalid coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Box is outside image bounds")
        new, valid = item.get("is_newly_added"), item.get("is_valid_rescue")
        if new not in {"true", "false", "ambiguous"} or valid not in {"true", "false", "ambiguous"}:
            raise ValueError("Invalid annotation attributes")
        result.append({"object_id": f"{role}-{index}", "class_name": label,
                       "bbox_xyxy": box, "is_newly_added": new, "is_valid_rescue": valid,
                       "source_box_index": index, "annotation_notes": item.get("annotation_notes")})
    if len({x["object_id"] for x in result}) != len(result):
        raise ValueError("Duplicate object index")
    return result


def _frame_name(frame: dict, prefix: str, pair_hash: str) -> None:
    name = frame.get("file")
    if not isinstance(name, str) or PurePosixPath(name).name != name:
        raise ValueError("Invalid frame filename")
    stem = PurePosixPath(name).stem
    if stem != f"{prefix}__{pair_hash}":
        raise ValueError("Frame filename does not match pair")


def _adapt_pair(pair: dict) -> dict:
    number, pair_hash = pair["pair_id"], pair["pair_hash"]
    if not isinstance(number, str) or not number.isdigit() or not isinstance(pair_hash, str) or not pair_hash.isalnum():
        raise ValueError("Invalid pair identifiers")
    if pair["generation_status"] not in STATUSES:
        raise ValueError("Invalid generation status")
    before, after = pair["before"], pair["after"]
    _frame_name(before, f"{number}__00_before", pair_hash)
    _frame_name(after, f"{number}__01_after", pair_hash)
    if (before["width"], before["height"]) != (after["width"], after["height"]):
        raise ValueError("Image dimensions differ")
    positives = _objects(after["confirmed_rescue_objects"], after, "after")
    new = _objects(after["confirmed_new_rescue_objects"], after, "after")
    old = _objects(after["confirmed_existing_rescue_objects"], after, "after")
    ignored = _objects(after["ignore_regions"], after, "after")
    if any(x["is_valid_rescue"] != "true" or x["is_newly_added"] not in {"true", "false"} for x in positives):
        raise ValueError("Invalid confirmed positive")
    if {x["object_id"] for x in positives} & {x["object_id"] for x in ignored}:
        raise ValueError("Ignore region overlaps a positive object ID")
    expected_new = [x for x in positives if x["is_newly_added"] == "true"]
    expected_old = [x for x in positives if x["is_newly_added"] == "false"]
    if sorted(new, key=lambda x: x["object_id"]) != sorted(expected_new, key=lambda x: x["object_id"]) or sorted(old, key=lambda x: x["object_id"]) != sorted(expected_old, key=lambda x: x["object_id"]):
        raise ValueError("Confirmed lists disagree")
    if any("ambiguous" not in (x["is_valid_rescue"], x["is_newly_added"]) for x in ignored):
        raise ValueError("Ignore region has no ambiguous attribute")
    return {"pair_id": pair_hash, "source_pair_id": number, "split": "validation",
            "generation_status": pair["generation_status"],
            "before_objects": _objects(before["objects"], before, "before"),
            "after_objects": positives, "ignore_regions": ignored,
            "pair_notes": pair.get("pair_notes"), "before_file": before["file"], "after_file": after["file"]}


def adapt_strict_manifest(strict: dict, benchmark: dict) -> dict:
    if strict.get("split") != "validation" or not isinstance(strict.get("pairs"), list):
        raise ValueError("Expected a strict validation manifest")
    pairs = [_adapt_pair(pair) for pair in strict["pairs"]]
    ids = [pair["pair_id"] for pair in pairs]
    source_ids = [pair["source_pair_id"] for pair in pairs]
    benchmark_ids = [pair["pair_id"] for pair in benchmark["pairs"] if pair.get("split") == "validation"]
    if not ids or len(set(ids)) != len(ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("Duplicate or empty validation pairs")
    if len(set(benchmark_ids)) != len(benchmark_ids) or set(ids) != set(benchmark_ids):
        raise ValueError("Strict and benchmark validation pair IDs differ")
    return {"schema_version": "strict-validation-v2-adapter-1", "annotation_status": "single_annotator_reviewed",
            "source_dataset_id": strict.get("dataset_id"), "pairs": pairs}
