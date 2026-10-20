import unittest

from src.evaluation.bbox_metrics import EvaluationConfig, evaluate_benchmark


CLASSES = ("firefighter", "helicopter", "ambulance")


def ground_truth_pair(
    pair_id: str,
    after_objects: list[dict],
    generation_status: str = "valid",
) -> dict:
    return {
        "pair_id": pair_id,
        "generation_status": generation_status,
        "before_objects": [],
        "after_objects": after_objects,
    }


def ground_truth_object(
    object_id: str,
    class_name: str,
    box: list[float],
    is_newly_added: str = "true",
) -> dict:
    return {
        "object_id": object_id,
        "class_name": class_name,
        "bbox_xyxy": box,
        "is_newly_added": is_newly_added,
        "is_valid_rescue": "true",
        "occlusion": "none",
    }


def prediction(class_name: str, box: list[float], confidence: float = 0.9) -> dict:
    return {
        "class_id": 0,
        "label": class_name,
        "confidence": confidence,
        "box": {
            "x_min": box[0],
            "y_min": box[1],
            "x_max": box[2],
            "y_max": box[3],
        },
    }


def prediction_report(
    pair_id: str,
    raw: list[dict],
    added: list[dict] | None = None,
    refined: list[dict] | None = None,
    bbox_passed: bool = True,
) -> dict:
    added = raw if added is None else added
    refined = added if refined is None else refined
    return {
        "stem": pair_id,
        "after_detections": raw,
        "matching": {
            "added_rescue_objects": [
                {"detection": item, "delta_overlap": 0.8} for item in added
            ],
            "refined_rescue_objects": refined,
        },
        "bbox_passed": bbox_passed,
    }


class BoundingBoxMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EvaluationConfig(class_names=CLASSES)

    def test_perfect_prediction_has_perfect_metrics(self) -> None:
        gt = [
            ground_truth_pair(
                "pair-1",
                [ground_truth_object("gt-1", "firefighter", [0, 0, 10, 10])],
            )
        ]
        detected = prediction("firefighter", [0, 0, 10, 10])

        result = evaluate_benchmark(
            gt,
            {"pair-1": prediction_report("pair-1", [detected])},
            self.config,
        )

        detector = result["detector"]["overall"]
        final = result["final_pipeline"]["overall"]
        self.assertEqual(detector["tp"], 1)
        self.assertEqual(detector["fp"], 0)
        self.assertEqual(detector["fn"], 0)
        self.assertEqual(detector["recall"], 1.0)
        self.assertEqual(detector["ap50"], 1.0)
        self.assertEqual(final["f1"], 1.0)

    def test_duplicate_prediction_counts_as_false_positive(self) -> None:
        gt = [
            ground_truth_pair(
                "pair-1",
                [ground_truth_object("gt-1", "firefighter", [0, 0, 10, 10])],
            )
        ]
        predictions = [
            prediction("firefighter", [0, 0, 10, 10], 0.9),
            prediction("firefighter", [0, 0, 10, 10], 0.8),
        ]

        result = evaluate_benchmark(
            gt,
            {"pair-1": prediction_report("pair-1", predictions)},
            self.config,
        )

        detector = result["detector"]["overall"]
        self.assertEqual(detector["tp"], 1)
        self.assertEqual(detector["fp"], 1)
        self.assertEqual(detector["fn"], 0)
        self.assertEqual(detector["precision"], 0.5)

    def test_wrong_class_is_false_positive_and_false_negative(self) -> None:
        gt = [
            ground_truth_pair(
                "pair-1",
                [ground_truth_object("gt-1", "firefighter", [0, 0, 10, 10])],
            )
        ]
        detected = prediction("helicopter", [0, 0, 10, 10])

        result = evaluate_benchmark(
            gt,
            {"pair-1": prediction_report("pair-1", [detected])},
            self.config,
        )

        detector = result["detector"]["overall"]
        self.assertEqual(detector["tp"], 0)
        self.assertEqual(detector["fp"], 1)
        self.assertEqual(detector["fn"], 1)
        failures = result["failure_taxonomy"]["missed_new_objects"]
        self.assertEqual(failures["total"], 1)
        self.assertEqual(failures["wrong_class"], 1)

    def test_missing_report_counts_valid_ground_truth_as_false_negative(self) -> None:
        gt = [
            ground_truth_pair(
                "pair-1",
                [ground_truth_object("gt-1", "firefighter", [0, 0, 10, 10])],
            )
        ]

        result = evaluate_benchmark(gt, {}, self.config)

        self.assertEqual(result["detector"]["overall"]["fn"], 1)
        self.assertEqual(result["pair_pass_rate"]["missing_reports"], 1)

    def test_class_without_ground_truth_is_insufficient_data(self) -> None:
        gt = [ground_truth_pair("pair-1", [])]
        detected = prediction("ambulance", [0, 0, 10, 10])

        result = evaluate_benchmark(
            gt,
            {"pair-1": prediction_report("pair-1", [detected])},
            self.config,
        )

        ambulance = result["detector"]["per_class"]["ambulance"]
        self.assertEqual(ambulance["status"], "insufficient_data")
        self.assertEqual(ambulance["prediction_count"], 1)
        self.assertIsNone(ambulance["recall"])
        self.assertIsNone(ambulance["ap50"])

    def test_iou_boundary_is_inclusive_and_just_below_is_rejected(self) -> None:
        gt = [
            ground_truth_pair(
                "at-threshold",
                [ground_truth_object("gt-1", "firefighter", [0, 0, 10, 10])],
            ),
            ground_truth_pair(
                "below-threshold",
                [ground_truth_object("gt-2", "firefighter", [0, 0, 10, 10])],
            ),
        ]
        at_threshold = prediction("firefighter", [0, 0, 5, 10])
        below_threshold = prediction("firefighter", [0, 0, 4.99, 10])
        reports = {
            "at-threshold": prediction_report("at-threshold", [at_threshold]),
            "below-threshold": prediction_report("below-threshold", [below_threshold]),
        }

        result = evaluate_benchmark(gt, reports, self.config)

        detector = result["detector"]["overall"]
        self.assertEqual(detector["tp"], 1)
        self.assertEqual(detector["fp"], 1)
        self.assertEqual(detector["fn"], 1)

    def test_one_prediction_cannot_match_two_ground_truth_objects(self) -> None:
        gt = [
            ground_truth_pair(
                "pair-1",
                [
                    ground_truth_object("gt-1", "firefighter", [0, 0, 10, 10]),
                    ground_truth_object("gt-2", "firefighter", [0, 0, 10, 10]),
                ],
            )
        ]
        detected = prediction("firefighter", [0, 0, 10, 10])

        result = evaluate_benchmark(
            gt,
            {"pair-1": prediction_report("pair-1", [detected])},
            self.config,
        )

        detector = result["detector"]["overall"]
        self.assertEqual(detector["tp"], 1)
        self.assertEqual(detector["fn"], 1)

    def test_change_verification_is_measured_only_on_detected_objects(self) -> None:
        gt = [
            ground_truth_pair(
                "pair-1",
                [
                    ground_truth_object(
                        "new", "firefighter", [0, 0, 10, 10], "true"
                    ),
                    ground_truth_object(
                        "existing", "helicopter", [20, 20, 30, 30], "false"
                    ),
                ],
            )
        ]
        new_detection = prediction("firefighter", [0, 0, 10, 10])
        existing_detection = prediction("helicopter", [20, 20, 30, 30])
        report = prediction_report(
            "pair-1",
            [new_detection, existing_detection],
            added=[existing_detection],
            refined=[existing_detection],
        )

        result = evaluate_benchmark(gt, {"pair-1": report}, self.config)

        verification = result["change_verification"]
        self.assertEqual(verification["eligible_ground_truth"], 2)
        self.assertEqual(verification["detected_and_evaluated"], 2)
        self.assertEqual(verification["tp_new"], 0)
        self.assertEqual(verification["fp_new"], 1)
        self.assertEqual(verification["fn_new"], 1)
        self.assertEqual(verification["tn_existing"], 0)

    def test_generation_failures_are_excluded_from_detector_recall(self) -> None:
        gt = [
            ground_truth_pair(
                "valid",
                [ground_truth_object("gt-1", "firefighter", [0, 0, 10, 10])],
            ),
            ground_truth_pair(
                "failed",
                [ground_truth_object("gt-2", "firefighter", [0, 0, 10, 10])],
                generation_status="invalid_object",
            ),
        ]

        invalid_detection = prediction("firefighter", [0, 0, 10, 10])
        result = evaluate_benchmark(
            gt,
            {
                "failed": prediction_report(
                    "failed",
                    [invalid_detection],
                )
            },
            self.config,
        )

        self.assertEqual(result["detector"]["overall"]["gt_count"], 1)
        self.assertEqual(result["detector"]["overall"]["prediction_count"], 0)
        self.assertEqual(result["generation"]["valid_pairs"], 1)
        self.assertEqual(result["generation"]["failure_pairs"], 1)

    def test_failure_taxonomy_separates_delta_rejection_from_matching(self) -> None:
        gt = [
            ground_truth_pair(
                "delta",
                [ground_truth_object("gt-1", "firefighter", [0, 0, 10, 10])],
            ),
            ground_truth_pair(
                "matching",
                [ground_truth_object("gt-2", "firefighter", [0, 0, 10, 10])],
            ),
        ]
        delta_detection = prediction("firefighter", [0, 0, 10, 10])
        matching_detection = prediction("firefighter", [0, 0, 10, 10])
        delta_report = prediction_report(
            "delta",
            [delta_detection],
            added=[],
            refined=[],
        )
        delta_report["matching"]["rejected_detections"] = [
            {"detection": delta_detection, "reason": "Low delta overlap"}
        ]
        matching_report = prediction_report(
            "matching",
            [matching_detection],
            added=[],
            refined=[],
        )

        result = evaluate_benchmark(
            gt,
            {"delta": delta_report, "matching": matching_report},
            self.config,
        )

        failures = result["failure_taxonomy"]["missed_new_objects"]
        self.assertEqual(failures["total"], 2)
        self.assertEqual(failures["delta_rejection"], 1)
        self.assertEqual(failures["before_after_matching_error"], 1)

    def test_invalid_annotation_state_is_rejected(self) -> None:
        invalid = ground_truth_object(
            "gt-1",
            "firefighter",
            [0, 0, 10, 10],
            is_newly_added="maybe",
        )
        gt = [ground_truth_pair("pair-1", [invalid])]

        with self.assertRaisesRegex(ValueError, "invalid is_newly_added"):
            evaluate_benchmark(gt, {}, self.config)


if __name__ == "__main__":
    unittest.main()
