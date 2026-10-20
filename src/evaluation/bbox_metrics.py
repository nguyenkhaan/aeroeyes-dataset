from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Callable, Iterable, Sequence


DEFAULT_IOU_THRESHOLDS = tuple(round(0.50 + step * 0.05, 2) for step in range(10))
VALID_GENERATION_STATUS = "valid"
AMBIGUOUS_STATE = "ambiguous"
GENERATION_STATUSES = {
    "valid",
    "missing_requested_object",
    "invalid_object",
    "severe_scene_corruption",
    "ambiguous",
}
TRISTATE_VALUES = {"true", "false", "ambiguous"}


@dataclass(frozen=True)
class EvaluationConfig:
    class_names: tuple[str, ...]
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS

    def validate(self) -> None:
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Evaluation classes must be unique and non-empty")
        if 0.5 not in self.iou_thresholds:
            raise ValueError("IoU thresholds must include 0.50")
        if any(threshold <= 0 or threshold > 1 for threshold in self.iou_thresholds):
            raise ValueError("IoU thresholds must be in the interval (0, 1]")


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def validate(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(f"Bounding box has non-positive area: {self}")

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def iou(self, other: Box) -> float:
        intersection_width = max(
            0.0,
            min(self.x_max, other.x_max) - max(self.x_min, other.x_min),
        )
        intersection_height = max(
            0.0,
            min(self.y_max, other.y_max) - max(self.y_min, other.y_min),
        )
        intersection = intersection_width * intersection_height
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class GroundTruthObject:
    pair_id: str
    object_id: str
    class_name: str
    box: Box
    is_newly_added: str
    is_valid_rescue: str


@dataclass(frozen=True)
class Prediction:
    pair_id: str
    class_name: str
    box: Box
    confidence: float


@dataclass(frozen=True)
class PairData:
    pair_id: str
    generation_status: str
    before_class_names: tuple[str, ...]
    ground_truth: tuple[GroundTruthObject, ...]
    detector_predictions: tuple[Prediction, ...]
    added_predictions: tuple[Prediction, ...]
    rejected_predictions: tuple[Prediction, ...]
    final_predictions: tuple[Prediction, ...]
    bbox_passed: bool
    has_report: bool
    performance: dict


@dataclass(frozen=True)
class MatchResult:
    ground_truth_count: int
    prediction_count: int
    true_positive_flags: tuple[int, ...]
    matches: tuple[tuple[GroundTruthObject, Prediction, float], ...]

    @property
    def true_positives(self) -> int:
        return sum(self.true_positive_flags)

    @property
    def false_positives(self) -> int:
        return self.prediction_count - self.true_positives

    @property
    def false_negatives(self) -> int:
        return self.ground_truth_count - self.true_positives


@dataclass(frozen=True)
class TierDefinition:
    ground_truth_selector: Callable[[PairData], Sequence[GroundTruthObject]]
    prediction_selector: Callable[[PairData], Sequence[Prediction]]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _parse_ground_truth_box(value: object) -> Box:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Ground-truth bbox_xyxy must contain four coordinates")
    try:
        box = Box(*(float(coordinate) for coordinate in value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Ground-truth bbox_xyxy contains a non-numeric value") from exc
    box.validate()
    return box


def _parse_prediction_box(value: object) -> Box:
    if not isinstance(value, dict):
        raise ValueError("Prediction box must be an object")
    try:
        box = Box(
            x_min=float(value["x_min"]),
            y_min=float(value["y_min"]),
            x_max=float(value["x_max"]),
            y_max=float(value["y_max"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Prediction box is missing a numeric coordinate") from exc
    box.validate()
    return box


def _parse_ground_truth_object(pair_id: str, value: object) -> GroundTruthObject:
    if not isinstance(value, dict):
        raise ValueError(f"Ground-truth object for {pair_id} must be an object")
    try:
        result = GroundTruthObject(
            pair_id=pair_id,
            object_id=str(value["object_id"]),
            class_name=str(value["class_name"]),
            box=_parse_ground_truth_box(value["bbox_xyxy"]),
            is_newly_added=str(value["is_newly_added"]).lower(),
            is_valid_rescue=str(value["is_valid_rescue"]).lower(),
        )
    except KeyError as exc:
        raise ValueError(f"Ground-truth object for {pair_id} is incomplete") from exc
    if not result.object_id:
        raise ValueError(f"Ground-truth object for {pair_id} has an empty object_id")
    if result.is_newly_added not in TRISTATE_VALUES:
        raise ValueError(
            f"Ground-truth object for {pair_id} has invalid is_newly_added"
        )
    if result.is_valid_rescue not in TRISTATE_VALUES:
        raise ValueError(
            f"Ground-truth object for {pair_id} has invalid is_valid_rescue"
        )
    return result


def _validate_before_objects(pair_id: str, values: list) -> tuple[str, ...]:
    object_ids: list[str] = []
    class_names: list[str] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"Before object for {pair_id} must be an object")
        try:
            object_id = str(value["object_id"])
            class_name = str(value["class_name"])
            _parse_ground_truth_box(value["bbox_xyxy"])
        except KeyError as exc:
            raise ValueError(f"Before object for {pair_id} is incomplete") from exc
        if not object_id or not class_name:
            raise ValueError(f"Before object for {pair_id} has an empty identifier")
        object_ids.append(object_id)
        class_names.append(class_name)
    if len(object_ids) != len(set(object_ids)):
        raise ValueError(
            f"Ground truth contains duplicate Before object_id in {pair_id}"
        )
    return tuple(class_names)


def _parse_prediction(pair_id: str, value: object) -> Prediction:
    if not isinstance(value, dict):
        raise ValueError(f"Prediction for {pair_id} must be an object")
    try:
        prediction = Prediction(
            pair_id=pair_id,
            class_name=str(value["label"]),
            box=_parse_prediction_box(value["box"]),
            confidence=float(value["confidence"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Prediction for {pair_id} is incomplete") from exc
    if not isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1:
        raise ValueError(f"Prediction for {pair_id} has invalid confidence")
    return prediction


def _prediction_list(pair_id: str, values: object) -> tuple[Prediction, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError(f"Predictions for {pair_id} must be a list")
    return tuple(_parse_prediction(pair_id, item) for item in values)


def _matching_predictions(report: dict, field_name: str) -> tuple[Prediction, ...]:
    pair_id = str(report["stem"])
    matching = report.get("matching", {})
    if not isinstance(matching, dict):
        raise ValueError(f"Matching report for {pair_id} must be an object")
    values = matching.get(field_name, [])
    if field_name in {"added_rescue_objects", "rejected_detections"}:
        if not isinstance(values, list):
            raise ValueError(f"Matching predictions for {pair_id} must be a list")
        if not all(isinstance(item, dict) and "detection" in item for item in values):
            raise ValueError(f"Matching prediction for {pair_id} is incomplete")
        values = [item["detection"] for item in values]
    return _prediction_list(pair_id, values)


def _prepare_pair(ground_truth: dict, report: dict | None) -> PairData:
    try:
        pair_id = str(ground_truth["pair_id"])
        generation_status = str(ground_truth["generation_status"])
        before_objects = ground_truth["before_objects"]
        after_objects = ground_truth["after_objects"]
    except KeyError as exc:
        raise ValueError("Ground-truth pair is missing a required field") from exc
    if (
        not pair_id
        or not isinstance(before_objects, list)
        or not isinstance(after_objects, list)
    ):
        raise ValueError(
            "Ground-truth pair has an invalid pair_id, before_objects, or after_objects"
        )

    objects = tuple(
        _parse_ground_truth_object(pair_id, item) for item in after_objects
    )
    object_ids = [item.object_id for item in objects]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError(f"Ground truth contains duplicate object_id in {pair_id}")
    if generation_status not in GENERATION_STATUSES:
        raise ValueError(f"Ground-truth pair {pair_id} has invalid generation_status")
    if report is None:
        return PairData(
            pair_id=pair_id,
            generation_status=generation_status,
            before_class_names=_validate_before_objects(pair_id, before_objects),
            ground_truth=objects,
            detector_predictions=(),
            added_predictions=(),
            rejected_predictions=(),
            final_predictions=(),
            bbox_passed=False,
            has_report=False,
            performance={},
        )
    if str(report.get("stem", "")) != pair_id:
        raise ValueError(f"Prediction report stem does not match pair_id {pair_id}")
    performance = report.get("performance", {})
    if not isinstance(performance, dict):
        performance = {}
    return PairData(
        pair_id=pair_id,
        generation_status=generation_status,
        before_class_names=_validate_before_objects(pair_id, before_objects),
        ground_truth=objects,
        detector_predictions=_prediction_list(
            pair_id,
            report.get("after_detections", []),
        ),
        added_predictions=_matching_predictions(report, "added_rescue_objects"),
        rejected_predictions=_matching_predictions(report, "rejected_detections"),
        final_predictions=_matching_predictions(report, "refined_rescue_objects"),
        bbox_passed=bool(report.get("bbox_passed", False)),
        has_report=True,
        performance=performance,
    )


def _validate_classes(pairs: Iterable[PairData], config: EvaluationConfig) -> None:
    allowed = set(config.class_names)
    observed = {
        item.class_name
        for pair in pairs
        for item in (
            *pair.ground_truth,
            *pair.detector_predictions,
            *pair.added_predictions,
            *pair.rejected_predictions,
            *pair.final_predictions,
        )
    }
    observed.update(
        class_name for pair in pairs for class_name in pair.before_class_names
    )
    unknown = sorted(observed - allowed)
    if unknown:
        raise ValueError(f"Classes are absent from the evaluation ontology: {unknown}")


def _valid_ground_truth(pair: PairData) -> tuple[GroundTruthObject, ...]:
    if pair.generation_status != VALID_GENERATION_STATUS:
        return ()
    return tuple(
        item
        for item in pair.ground_truth
        if item.is_valid_rescue == "true"
    )


def _new_ground_truth(pair: PairData) -> tuple[GroundTruthObject, ...]:
    return tuple(
        item for item in _valid_ground_truth(pair) if item.is_newly_added == "true"
    )


def _valid_detector_predictions(pair: PairData) -> tuple[Prediction, ...]:
    if pair.generation_status != VALID_GENERATION_STATUS:
        return ()
    return pair.detector_predictions


def _valid_final_predictions(pair: PairData) -> tuple[Prediction, ...]:
    if pair.generation_status != VALID_GENERATION_STATUS:
        return ()
    return pair.final_predictions


def _match(
    ground_truth: Sequence[GroundTruthObject],
    predictions: Sequence[Prediction],
    threshold: float,
) -> MatchResult:
    available_by_pair: dict[str, set[int]] = defaultdict(set)
    indexed_ground_truth: dict[str, list[GroundTruthObject]] = defaultdict(list)
    for item in ground_truth:
        index = len(indexed_ground_truth[item.pair_id])
        indexed_ground_truth[item.pair_id].append(item)
        available_by_pair[item.pair_id].add(index)

    flags: list[int] = []
    matches: list[tuple[GroundTruthObject, Prediction, float]] = []
    ordered_predictions = sorted(
        predictions,
        key=lambda item: item.confidence,
        reverse=True,
    )
    for predicted in ordered_predictions:
        candidates = indexed_ground_truth.get(predicted.pair_id, [])
        best_index: int | None = None
        best_iou = -1.0
        for index in available_by_pair[predicted.pair_id]:
            candidate = candidates[index]
            if candidate.class_name != predicted.class_name:
                continue
            overlap = candidate.box.iou(predicted.box)
            if overlap > best_iou:
                best_index = index
                best_iou = overlap
        if best_index is not None and best_iou >= threshold:
            matched = candidates[best_index]
            available_by_pair[predicted.pair_id].remove(best_index)
            flags.append(1)
            matches.append((matched, predicted, best_iou))
        else:
            flags.append(0)
    return MatchResult(
        ground_truth_count=len(ground_truth),
        prediction_count=len(predictions),
        true_positive_flags=tuple(flags),
        matches=tuple(matches),
    )


def _average_precision(result: MatchResult) -> float | None:
    if result.ground_truth_count == 0:
        return None
    cumulative_tp = 0
    cumulative_fp = 0
    precision_at_rank: list[float] = []
    recall_at_rank: list[float] = []
    for is_true_positive in result.true_positive_flags:
        cumulative_tp += is_true_positive
        cumulative_fp += 1 - is_true_positive
        precision_at_rank.append(cumulative_tp / (cumulative_tp + cumulative_fp))
        recall_at_rank.append(cumulative_tp / result.ground_truth_count)

    interpolated: list[float] = []
    for recall_level in (step / 100 for step in range(101)):
        eligible = [
            precision
            for recall, precision in zip(recall_at_rank, precision_at_rank)
            if recall >= recall_level
        ]
        interpolated.append(max(eligible, default=0.0))
    return mean(interpolated)


def _class_metrics(
    ground_truth: Sequence[GroundTruthObject],
    predictions: Sequence[Prediction],
    config: EvaluationConfig,
) -> dict:
    primary = _match(ground_truth, predictions, 0.5)
    diagnostic = _match(ground_truth, predictions, 0.3)
    if primary.ground_truth_count == 0:
        return {
            "status": "insufficient_data",
            "gt_count": 0,
            "prediction_count": primary.prediction_count,
            "tp": 0,
            "fp": primary.prediction_count,
            "fn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "ap50": None,
            "map50_95": None,
            "recall_iou_0_30": None,
            "mean_matched_iou": None,
        }

    precision = _ratio(primary.true_positives, primary.prediction_count)
    if precision is None:
        precision = 0.0
    recall = primary.true_positives / primary.ground_truth_count
    average_precisions = [
        _average_precision(_match(ground_truth, predictions, threshold))
        for threshold in config.iou_thresholds
    ]
    matched_ious = [overlap for _gt, _prediction, overlap in primary.matches]
    return {
        "status": "ok",
        "gt_count": primary.ground_truth_count,
        "prediction_count": primary.prediction_count,
        "tp": primary.true_positives,
        "fp": primary.false_positives,
        "fn": primary.false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "ap50": average_precisions[config.iou_thresholds.index(0.5)],
        "map50_95": mean(value for value in average_precisions if value is not None),
        "recall_iou_0_30": diagnostic.true_positives / diagnostic.ground_truth_count,
        "mean_matched_iou": mean(matched_ious) if matched_ious else None,
    }


def _tier_metrics(
    pairs: Sequence[PairData],
    config: EvaluationConfig,
    definition: TierDefinition,
) -> dict:
    all_ground_truth = [
        item for pair in pairs for item in definition.ground_truth_selector(pair)
    ]
    all_predictions = [
        item for pair in pairs for item in definition.prediction_selector(pair)
    ]
    per_class: dict[str, dict] = {}
    for class_name in config.class_names:
        class_ground_truth = [
            item for item in all_ground_truth if item.class_name == class_name
        ]
        class_predictions = [
            item for item in all_predictions if item.class_name == class_name
        ]
        per_class[class_name] = _class_metrics(
            class_ground_truth,
            class_predictions,
            config,
        )

    supported = [value for value in per_class.values() if value["status"] == "ok"]
    total_gt = sum(value["gt_count"] for value in per_class.values())
    total_predictions = sum(value["prediction_count"] for value in per_class.values())
    total_tp = sum(value["tp"] for value in per_class.values())
    total_fp = total_predictions - total_tp
    total_fn = total_gt - total_tp
    precision = _ratio(total_tp, total_predictions)
    if total_gt and precision is None:
        precision = 0.0
    recall = _ratio(total_tp, total_gt)
    matched = _match(all_ground_truth, all_predictions, 0.5)
    matched_ious = [overlap for _gt, _prediction, overlap in matched.matches]
    overall = {
        "status": "ok" if total_gt else "insufficient_data",
        "gt_count": total_gt,
        "prediction_count": total_predictions,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision if total_gt else None,
        "recall": recall,
        "f1": _f1(precision, recall) if total_gt else None,
        "ap50": mean(value["ap50"] for value in supported) if supported else None,
        "map50_95": (
            mean(value["map50_95"] for value in supported) if supported else None
        ),
        "recall_iou_0_30": (
            sum(value["recall_iou_0_30"] * value["gt_count"] for value in supported)
            / total_gt
            if total_gt
            else None
        ),
        "mean_matched_iou": mean(matched_ious) if matched_ious else None,
        "ap_aggregation": "macro_over_classes_with_ground_truth",
    }
    return {"overall": overall, "per_class": per_class}


def _is_added_prediction(pair: PairData, predicted: Prediction) -> bool:
    return any(
        candidate.class_name == predicted.class_name
        and candidate.box.iou(predicted.box) >= 0.99
        for candidate in pair.added_predictions
    )


def _is_rejected_prediction(pair: PairData, predicted: Prediction) -> bool:
    return any(
        candidate.class_name == predicted.class_name
        and candidate.box.iou(predicted.box) >= 0.99
        for candidate in pair.rejected_predictions
    )


def _change_verification(pairs: Sequence[PairData]) -> dict:
    ground_truth = [
        item
        for pair in pairs
        for item in _valid_ground_truth(pair)
        if item.is_newly_added in {"true", "false"}
    ]
    predictions = [
        item for pair in pairs for item in _valid_detector_predictions(pair)
    ]
    matched = _match(ground_truth, predictions, 0.5)
    pairs_by_id = {pair.pair_id: pair for pair in pairs}
    tp_new = fp_new = tn_existing = fn_new = 0
    for target, predicted, _overlap in matched.matches:
        predicted_new = _is_added_prediction(pairs_by_id[target.pair_id], predicted)
        actual_new = target.is_newly_added == "true"
        if actual_new and predicted_new:
            tp_new += 1
        elif not actual_new and predicted_new:
            fp_new += 1
        elif actual_new:
            fn_new += 1
        else:
            tn_existing += 1

    evaluated = len(matched.matches)
    precision = _ratio(tp_new, tp_new + fp_new)
    recall = _ratio(tp_new, tp_new + fn_new)
    return {
        "status": "ok" if ground_truth else "insufficient_data",
        "eligible_ground_truth": len(ground_truth),
        "detected_and_evaluated": evaluated,
        "coverage": _ratio(evaluated, len(ground_truth)),
        "tp_new": tp_new,
        "fp_new": fp_new,
        "tn_existing": tn_existing,
        "fn_new": fn_new,
        "precision_new": precision,
        "recall_new": recall,
        "f1_new": _f1(precision, recall),
        "accuracy": _ratio(tp_new + tn_existing, evaluated),
    }


def _generation_metrics(pairs: Sequence[PairData]) -> dict:
    counts = Counter(pair.generation_status for pair in pairs)
    valid_pairs = counts[VALID_GENERATION_STATUS]
    ambiguous_pairs = counts[AMBIGUOUS_STATE]
    failure_pairs = len(pairs) - valid_pairs - ambiguous_pairs
    resolved_pairs = valid_pairs + failure_pairs
    return {
        "total_pairs": len(pairs),
        "valid_pairs": valid_pairs,
        "failure_pairs": failure_pairs,
        "ambiguous_pairs": ambiguous_pairs,
        "success_rate": _ratio(valid_pairs, resolved_pairs),
        "status_counts": dict(sorted(counts.items())),
    }


def _pair_pass_metrics(pairs: Sequence[PairData]) -> dict:
    processed = [pair for pair in pairs if pair.has_report]
    passed = sum(pair.bbox_passed for pair in processed)
    return {
        "annotated_pairs": len(pairs),
        "processed_pairs": len(processed),
        "passed_pairs": passed,
        "pass_rate": _ratio(passed, len(processed)),
        "missing_reports": len(pairs) - len(processed),
        "role": "supplementary_operational_metric",
    }


def _performance_metrics(pairs: Sequence[PairData]) -> dict:
    runtime_values: list[float] = []
    vram_values: list[float] = []
    for pair in pairs:
        runtime = pair.performance.get("runtime_seconds")
        peak_vram = pair.performance.get("peak_vram_mb")
        if isinstance(runtime, (int, float)):
            runtime_values.append(float(runtime))
        if isinstance(peak_vram, (int, float)):
            vram_values.append(float(peak_vram))
    return {
        "pairs_with_runtime": len(runtime_values),
        "total_runtime_seconds": sum(runtime_values) if runtime_values else None,
        "mean_runtime_seconds": mean(runtime_values) if runtime_values else None,
        "pairs_with_peak_vram": len(vram_values),
        "peak_vram_mb": max(vram_values) if vram_values else None,
    }


def _best_overlap(
    target: GroundTruthObject,
    predictions: Sequence[Prediction],
) -> float:
    return max(
        (
            target.box.iou(predicted.box)
            for predicted in predictions
            if predicted.class_name == target.class_name
        ),
        default=0.0,
    )


def _has_wrong_class_overlap(
    target: GroundTruthObject,
    predictions: Sequence[Prediction],
) -> bool:
    return any(
        predicted.class_name != target.class_name
        and target.box.iou(predicted.box) >= 0.5
        for predicted in predictions
    )


def _failure_taxonomy(pairs: Sequence[PairData]) -> dict:
    counts: Counter[str] = Counter()
    final_ground_truth = [item for pair in pairs for item in _new_ground_truth(pair)]
    final_predictions = [
        item for pair in pairs for item in _valid_final_predictions(pair)
    ]
    matched = _match(final_ground_truth, final_predictions, 0.5)
    matched_ids = {
        (target.pair_id, target.object_id)
        for target, _prediction, _iou in matched.matches
    }
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    for target in final_ground_truth:
        if (target.pair_id, target.object_id) in matched_ids:
            continue
        pair = pair_by_id[target.pair_id]
        raw_overlap = _best_overlap(target, pair.detector_predictions)
        if raw_overlap >= 0.5:
            matching_raw = [
                predicted
                for predicted in pair.detector_predictions
                if predicted.class_name == target.class_name
                and target.box.iou(predicted.box) >= 0.5
            ]
            added = [item for item in matching_raw if _is_added_prediction(pair, item)]
            if not added and any(
                _is_rejected_prediction(pair, item) for item in matching_raw
            ):
                counts["delta_rejection"] += 1
            elif not added:
                counts["before_after_matching_error"] += 1
            elif _best_overlap(target, pair.final_predictions) > 0:
                counts["localization_failure"] += 1
            else:
                counts["delta_rejection"] += 1
        elif _has_wrong_class_overlap(target, pair.detector_predictions):
            counts["wrong_class"] += 1
        elif raw_overlap >= 0.3:
            counts["localization_failure"] += 1
        else:
            counts["detector_miss"] += 1
    counts["false_positive"] = matched.false_positives
    missed_names = (
        "detector_miss",
        "wrong_class",
        "before_after_matching_error",
        "delta_rejection",
        "localization_failure",
    )
    return {
        "generation_failure_pairs": sum(
            pair.generation_status not in {VALID_GENERATION_STATUS, AMBIGUOUS_STATE}
            for pair in pairs
        ),
        "missed_new_objects": {
            "total": matched.false_negatives,
            **{name: counts[name] for name in missed_names},
        },
        "false_positive_predictions": counts["false_positive"],
    }


def evaluate_benchmark(
    ground_truth_pairs: Sequence[dict],
    reports_by_pair_id: dict[str, dict],
    config: EvaluationConfig,
) -> dict:
    config.validate()
    pair_ids = [str(item.get("pair_id", "")) for item in ground_truth_pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Ground truth contains duplicate pair_id values")
    pairs = tuple(
        _prepare_pair(item, reports_by_pair_id.get(str(item.get("pair_id", ""))))
        for item in ground_truth_pairs
    )
    _validate_classes(pairs, config)
    detector = _tier_metrics(
        pairs,
        config,
        TierDefinition(
            ground_truth_selector=_valid_ground_truth,
            prediction_selector=_valid_detector_predictions,
        ),
    )
    final_pipeline = _tier_metrics(
        pairs,
        config,
        TierDefinition(
            ground_truth_selector=_new_ground_truth,
            prediction_selector=_valid_final_predictions,
        ),
    )
    return {
        "schema_version": 1,
        "evaluation_protocol": {
            "primary_iou": 0.5,
            "ap_iou_thresholds": list(config.iou_thresholds),
            "diagnostic_iou": 0.3,
            "class_names": list(config.class_names),
        },
        "generation": _generation_metrics(pairs),
        "detector": detector,
        "change_verification": _change_verification(pairs),
        "final_pipeline": final_pipeline,
        "pair_pass_rate": _pair_pass_metrics(pairs),
        "failure_taxonomy": _failure_taxonomy(pairs),
        "performance": _performance_metrics(pairs),
        "unused_prediction_reports": len(set(reports_by_pair_id) - set(pair_ids)),
    }
