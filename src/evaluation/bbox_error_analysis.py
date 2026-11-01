from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from src.evaluation.bbox_metrics import _best_target, _parse_ground_truth_object, _prepare_pair
from src.evaluation.bbox_strict_metrics import CLASS_NAMES, evaluate_strict_v2


def _best_overlap(prediction, targets):
    if not targets:
        return None
    target = max(targets, key=lambda item: item.box.iou(prediction.box))
    return target, target.box.iou(prediction.box)


def _fp_reason(prediction, positives, ignored, confirmed, tier):
    duplicate = _best_target(positives, prediction, 0.5)
    if duplicate:
        return "duplicate_confirmed", duplicate
    already_ignored = _best_target(ignored, prediction, 0.5)
    if already_ignored:
        return "overlap_ignore_already_used", already_ignored
    if tier == "final_pipeline":
        existing = _best_target([item for item in confirmed if item.is_newly_added == "false"], prediction, 0.5)
        if existing:
            return "overlap_existing", existing
    other = _best_overlap(prediction, [item for item in confirmed if item.class_name != prediction.class_name])
    if other and other[1] >= 0.5:
        return "overlap_other_class", other
    near = _best_target(positives, prediction, 0.3)
    if near:
        return "same_class_iou_030_050", near
    return "unmatched", None


def _pair_tier(pair, ignored, tier):
    positives = [item for item in pair.ground_truth if tier == "detector" or item.is_newly_added == "true"]
    predictions = pair.detector_predictions if tier == "detector" else pair.final_predictions
    used_positive, used_ignore = set(), set()
    rows, counts = [], Counter()
    for index, prediction in sorted(enumerate(predictions), key=lambda item: -item[1].confidence):
        matched = _best_target((item for item in positives if item.object_id not in used_positive), prediction, 0.5)
        if matched:
            used_positive.add(matched[0].object_id)
            counts[(prediction.class_name, "tp")] += 1
            continue
        duplicate = _best_target(positives, prediction, 0.5)
        ignored_match = None if duplicate else _best_target(
            (item for item in ignored if item.object_id not in used_ignore), prediction, 0.5
        )
        if ignored_match:
            used_ignore.add(ignored_match[0].object_id)
            counts[(prediction.class_name, "ignored")] += 1
            continue
        reason, related = _fp_reason(prediction, positives, ignored, pair.ground_truth, tier)
        counts[(prediction.class_name, "fp")] += 1
        rows.append({"pair_id": pair.pair_id, "tier": tier, "outcome": "fp", "reason": reason,
                     "class_name": prediction.class_name, "prediction_index": index,
                     "confidence": prediction.confidence, "bbox": asdict(prediction.box),
                     "related_class": related[0].class_name if related else None,
                     "related_object_id": related[0].object_id if related else None,
                     "related_iou": related[1] if related else None})
    for target in positives:
        if target.object_id not in used_positive:
            counts[(target.class_name, "fn")] += 1
            rows.append({"pair_id": pair.pair_id, "tier": tier, "outcome": "fn", "reason": "unmatched_ground_truth",
                         "class_name": target.class_name, "object_id": target.object_id, "bbox": asdict(target.box)})
    return rows, counts


def analyze_strict_errors(document: dict, reports: dict[str, dict]) -> dict:
    metrics = evaluate_strict_v2(document, reports)
    rows, totals = [], {"detector": Counter(), "final_pipeline": Counter()}
    for record in document["pairs"]:
        original = {key: value for key, value in record.items() if key != "ignore_regions"}
        pair = _prepare_pair(original, reports[record["pair_id"]])
        ignored = [_parse_ground_truth_object(pair.pair_id, item) for item in record["ignore_regions"]]
        for tier in totals:
            pair_rows, counts = _pair_tier(pair, ignored, tier)
            for row in pair_rows:
                row["source_pair_id"] = record["source_pair_id"]
                row["generation_status"] = record["generation_status"]
                row["before_file"] = record["before_file"]
                row["after_file"] = record["after_file"]
            rows.extend(pair_rows)
            totals[tier].update(counts)
    summary = {}
    for tier, counts in totals.items():
        for name in CLASS_NAMES:
            expected = metrics[tier]["per_class"][name]
            for key, metric_key in (("tp", "tp"), ("fp", "fp"), ("fn", "fn"), ("ignored", "ignored_prediction_count")):
                if counts[(name, key)] != expected[metric_key]:
                    raise ValueError(f"Diagnostic matching differs from evaluator for {tier}/{name}/{key}")
        by_class = {}
        for name in CLASS_NAMES:
            by_class[name] = {key: counts[(name, key)] for key in ("tp", "fp", "fn", "ignored")}
            failures = [row for row in rows if row["tier"] == tier and row["class_name"] == name and row["outcome"] == "fp"]
            by_class[name]["fp_reasons"] = dict(Counter(row["reason"] for row in failures))
            pair_counts = Counter((row["source_pair_id"], row["pair_id"]) for row in failures)
            by_class[name]["top_fp_pairs"] = [
                {"source_pair_id": key[0], "pair_id": key[1], "fp": count}
                for key, count in pair_counts.most_common(5)
            ]
        summary[tier] = {**{key: sum(counts[(name, key)] for name in CLASS_NAMES) for key in ("tp", "fp", "fn", "ignored")},
                         "per_class": by_class}
    return {"protocol": "strict-v2-iou50", "note": "FP reasons are overlap heuristics, not verified error causes.",
            "coverage": metrics["coverage"], "summary": summary, "rows": rows}
