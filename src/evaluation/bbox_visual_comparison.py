from __future__ import annotations

from collections import Counter

from src.evaluation.bbox_error_analysis import analyze_strict_errors
from src.evaluation.bbox_metrics import MatchTargets, _match, _prepare_pair
from src.evaluation.bbox_strict_metrics import evaluate_strict_v2


def _matched_new_ambulances(record: dict, report: dict) -> set[str]:
    original = {key: value for key, value in record.items() if key != "ignore_regions"}
    pair = _prepare_pair(original, report)
    targets = [item for item in pair.ground_truth
               if item.class_name == "ambulance" and item.is_newly_added == "true"]
    predictions = [item for item in pair.final_predictions if item.class_name == "ambulance"]
    return {target.object_id for target, _, _ in _match(MatchTargets(targets), predictions, 0.5).matches}


def select_review_cases(document: dict, baseline: dict, candidate: dict, max_fp_pairs: int = 5) -> list[dict]:
    if isinstance(max_fp_pairs, bool) or not isinstance(max_fp_pairs, int) or max_fp_pairs < 0:
        raise ValueError("max_fp_pairs must be a nonnegative integer")
    errors = analyze_strict_errors(document, baseline)
    evaluate_strict_v2(document, candidate)
    overlap_counts = Counter(
        row["pair_id"] for row in errors["rows"]
        if row["tier"] == "final_pipeline" and row["outcome"] == "fp"
        and row["class_name"] == "ambulance" and row["reason"] == "overlap_other_class"
        and row["related_class"] == "emergency_vehicle"
    )
    selected = {}
    for record in document["pairs"]:
        pair_id = record["pair_id"]
        lost = sorted(_matched_new_ambulances(record, baseline[pair_id]) -
                      _matched_new_ambulances(record, candidate[pair_id]))
        if lost:
            selected[pair_id] = {"pair_id": pair_id, "source_pair_id": record["source_pair_id"],
                                 "lost_ambulance_gt_ids": lost,
                                 "baseline_fp_overlap_emergency": overlap_counts[pair_id]}
    records_by_id = {record["pair_id"]: record for record in document["pairs"]}
    additional = sorted((pair_id for pair_id in overlap_counts if pair_id not in selected),
                        key=lambda pair_id: (-overlap_counts[pair_id], pair_id))[:max_fp_pairs]
    for pair_id in additional:
        selected[pair_id] = {"pair_id": pair_id, "source_pair_id": records_by_id[pair_id]["source_pair_id"],
                             "lost_ambulance_gt_ids": [],
                             "baseline_fp_overlap_emergency": overlap_counts[pair_id]}
    return sorted(selected.values(), key=lambda item: (not bool(item["lost_ambulance_gt_ids"]), item["source_pair_id"], item["pair_id"]))
