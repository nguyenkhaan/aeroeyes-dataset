from __future__ import annotations

from collections import Counter
from statistics import mean

from src.evaluation.bbox_metrics import (
    EvaluationConfig, MatchTargets, _match, _parse_ground_truth_object,
    _prepare_pair, _validate_classes,
)


CLASS_NAMES = ("firefighter", "rescue_boat", "helicopter", "ambulance", "emergency_vehicle")


def _numbers(targets, predictions, ignored):
    matched = _match(MatchTargets(targets, ignored), predictions, 0.5)
    tp, fp, fn = matched.true_positives, matched.false_positives, matched.false_negatives
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    ious = [overlap for _, _, overlap in matched.matches]
    return {"gt_count": len(targets), "raw_prediction_count": len(predictions),
            "prediction_count": matched.prediction_count, "ignored_gt_count": len(ignored),
            "ignored_prediction_count": matched.ignored_prediction_count,
            "tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "mean_matched_iou": mean(ious) if ious else None}


def _tier(pairs, ignored, field, novel_only):
    targets = [obj for pair in pairs for obj in pair.ground_truth
               if not novel_only or obj.is_newly_added == "true"]
    predictions = [obj for pair in pairs for obj in getattr(pair, field)]
    classes = {}
    for name in CLASS_NAMES:
        classes[name] = _numbers([obj for obj in targets if obj.class_name == name],
                                 [obj for obj in predictions if obj.class_name == name],
                                 [obj for obj in ignored if obj.class_name == name])
    supported = [item for item in classes.values() if item["gt_count"] > 0]
    # A supported class without predictions has F1=0, not an omitted macro entry.
    macro_f1 = mean(item["f1"] if item["f1"] is not None else 0.0 for item in supported) if supported else None
    return {"overall": _numbers(targets, predictions, ignored), "per_class": classes,
            "macro_f1": macro_f1, "macro_f1_scope": "classes_with_ground_truth"}


def evaluate_strict_v2(document: dict, reports: dict[str, dict]) -> dict:
    if document.get("schema_version") != "strict-validation-v2-adapter-1":
        raise ValueError("Expected adapted strict validation v2 ground truth")
    records = document.get("pairs")
    if not isinstance(records, list) or not records:
        raise ValueError("Ground truth pairs are missing")
    ids = [record.get("pair_id") for record in records]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Ground truth pair IDs are invalid")
    if set(ids) != set(reports):
        raise ValueError("Prediction reports do not match ground truth pairs")
    pairs, ignored = [], []
    for record in records:
        if record.get("split") != "validation" or not isinstance(record.get("ignore_regions"), list):
            raise ValueError("Expected validation pairs with explicit ignore regions")
        report = reports[record["pair_id"]]
        matching = report.get("matching")
        if not isinstance(report.get("after_detections"), list) or not isinstance(matching, dict) or not isinstance(matching.get("refined_rescue_objects"), list):
            raise ValueError("Prediction report is missing detector or final output")
        original = {key: value for key, value in record.items() if key != "ignore_regions"}
        pair = _prepare_pair(original, report)
        pair_ignored = [_parse_ground_truth_object(record["pair_id"], item) for item in record["ignore_regions"]]
        if any(item.is_valid_rescue != "true" or item.is_newly_added not in {"true", "false"} for item in pair.ground_truth):
            raise ValueError("Invalid confirmed positive")
        if any("ambiguous" not in (item.is_valid_rescue, item.is_newly_added) for item in pair_ignored):
            raise ValueError("Invalid ignore region")
        positive_ids = {item.object_id for item in pair.ground_truth}
        ignore_ids = [item.object_id for item in pair_ignored]
        if len(ignore_ids) != len(set(ignore_ids)) or positive_ids.intersection(ignore_ids):
            raise ValueError("Duplicate positive or ignore object IDs")
        pairs.append(pair)
        ignored.extend(pair_ignored)
    _validate_classes(pairs, EvaluationConfig(class_names=CLASS_NAMES))
    if any(item.class_name not in CLASS_NAMES for item in ignored):
        raise ValueError("Unknown ignore-region class")
    detector = _tier(pairs, ignored, "detector_predictions", False)
    final = _tier(pairs, ignored, "final_predictions", True)
    return {
        "evaluation_protocol": {"version": "bbox-strict-validation-v2-iou50-1",
            "pair_scope": "all_validation_pairs_including_ambiguous_and_generation_failures",
            "detector_gt": "confirmed_after_including_existing", "final_gt": "confirmed_new_after",
            "positive_iou": 0.5, "ignore_iou": 0.5,
            "ignore_matching": "same_pair_same_class_one_to_one",
            "priority": "positive_then_duplicate_positive_fp_then_ignore",
            "final_predictions": "matching.refined_rescue_objects",
            "f1_overall": "micro", "macro_f1": "mean_over_classes_with_gt",
            "annotation_status": document.get("annotation_status")},
        "coverage": {"evaluated_pairs": len(pairs), "status_counts": dict(sorted(Counter(p.generation_status for p in pairs).items())),
                     "confirmed_after": detector["overall"]["gt_count"], "confirmed_new": final["overall"]["gt_count"],
                     "ignore_regions": len(ignored)},
        "detector": detector, "final_pipeline": final,
        "sensitivity_no_ignore": {"detector": _tier(pairs, [], "detector_predictions", False),
                                  "final_pipeline": _tier(pairs, [], "final_predictions", True)},
    }
