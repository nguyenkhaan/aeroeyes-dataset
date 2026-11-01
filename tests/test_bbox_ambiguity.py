import unittest

from src.evaluation.bbox_metrics import EvaluationConfig, evaluate_benchmark
from tests.test_bbox_metrics import (
    CLASSES,
    ground_truth_object,
    ground_truth_pair,
    prediction,
    prediction_report,
)


def uncertain_object(object_id: str, box: list[float]) -> dict:
    result = ground_truth_object(object_id, "firefighter", box)
    result["is_valid_rescue"] = "ambiguous"
    return result


class BoundingBoxAmbiguityTests(unittest.TestCase):
    def evaluate(self, objects: list[dict], detections: list[dict]) -> dict:
        return evaluate_benchmark(
            [ground_truth_pair("pair", objects)],
            {"pair": prediction_report("pair", detections)},
            EvaluationConfig(class_names=CLASSES),
        )

    def test_uncertain_validity_prediction_is_ignored_in_both_tiers(self) -> None:
        objects = [
            ground_truth_object("known", "firefighter", [0, 0, 10, 10]),
            uncertain_object("uncertain", [20, 0, 30, 10]),
        ]
        result = self.evaluate(objects, [
            prediction("firefighter", [20, 0, 30, 10], 0.99),
            prediction("firefighter", [0, 0, 10, 10], 0.8),
        ])

        for tier in ("detector", "final_pipeline"):
            overall = result[tier]["overall"]
            self.assertEqual(overall["tp"], 1)
            self.assertEqual(overall["fp"], 0)
            self.assertEqual(overall["prediction_count"], 1)
            self.assertEqual(overall["ignored_prediction_count"], 1)
            self.assertEqual(overall["ignored_gt_count"], 1)
            self.assertEqual(overall["ap50"], 1.0)
            self.assertEqual(overall["map50_95"], 1.0)
        self.assertEqual(result["failure_taxonomy"]["false_positive_predictions"], 0)

    def test_unknown_newness_only_affects_final_tier(self) -> None:
        obj = ground_truth_object("uncertain", "firefighter", [0, 0, 10, 10], "ambiguous")
        result = self.evaluate([obj], [prediction("firefighter", [0, 0, 10, 10])])

        self.assertEqual(result["detector"]["overall"]["tp"], 1)
        final = result["final_pipeline"]["overall"]
        self.assertEqual(final["status"], "insufficient_data")
        self.assertEqual(final["fp"], 0)
        self.assertEqual(final["ignored_prediction_count"], 1)
        self.assertEqual(result["change_verification"]["eligible_ground_truth"], 0)

    def test_resolved_invalid_object_does_not_hide_false_positive(self) -> None:
        obj = ground_truth_object("invalid", "firefighter", [0, 0, 10, 10])
        obj["is_valid_rescue"] = "false"
        result = self.evaluate([obj], [prediction("firefighter", [0, 0, 10, 10])])

        self.assertEqual(result["detector"]["overall"]["fp"], 1)
        self.assertEqual(result["final_pipeline"]["overall"]["fp"], 1)

    def test_known_existing_object_is_not_a_final_ignore_region(self) -> None:
        obj = uncertain_object("existing", [0, 0, 10, 10])
        obj["is_newly_added"] = "false"
        result = self.evaluate([obj], [prediction("firefighter", [0, 0, 10, 10])])

        self.assertEqual(result["detector"]["overall"]["ignored_prediction_count"], 1)
        self.assertEqual(result["final_pipeline"]["overall"]["ignored_prediction_count"], 0)
        self.assertEqual(result["final_pipeline"]["overall"]["fp"], 1)

    def test_wrong_class_prediction_is_not_ignored(self) -> None:
        result = self.evaluate(
            [uncertain_object("uncertain", [0, 0, 10, 10])],
            [prediction("helicopter", [0, 0, 10, 10])],
        )

        self.assertEqual(result["detector"]["overall"]["fp"], 1)
        self.assertEqual(result["detector"]["overall"]["ignored_prediction_count"], 0)

    def test_ignore_region_only_absorbs_one_prediction(self) -> None:
        result = self.evaluate(
            [uncertain_object("uncertain", [0, 0, 10, 10])],
            [prediction("firefighter", [0, 0, 10, 10], 0.9),
             prediction("firefighter", [0, 0, 10, 10], 0.8)],
        )

        self.assertEqual(result["detector"]["overall"]["ignored_prediction_count"], 1)
        self.assertEqual(result["detector"]["overall"]["fp"], 1)

    def test_known_positive_and_its_duplicate_take_priority_over_ignore(self) -> None:
        result = self.evaluate(
            [ground_truth_object("known", "firefighter", [0, 0, 10, 10]),
             uncertain_object("uncertain", [0, 0, 10, 10])],
            [prediction("firefighter", [0, 0, 10, 10], 0.9),
             prediction("firefighter", [0, 0, 10, 10], 0.8)],
        )

        self.assertEqual(result["detector"]["overall"]["tp"], 1)
        self.assertEqual(result["detector"]["overall"]["fp"], 1)
        self.assertEqual(result["detector"]["overall"]["ignored_prediction_count"], 0)

    def test_ignore_matching_uses_each_ap_threshold(self) -> None:
        result = self.evaluate(
            [ground_truth_object("known", "firefighter", [0, 0, 10, 10]),
             uncertain_object("uncertain", [20, 0, 30, 10])],
            [prediction("firefighter", [23, 0, 33, 10], 0.99),
             prediction("firefighter", [0, 0, 10, 10], 0.8)],
        )

        self.assertEqual(result["detector"]["overall"]["ignored_prediction_count"], 1)
        overall = result["detector"]["overall"]
        self.assertEqual(overall["ap50"], 1.0)
        self.assertAlmostEqual(overall["map50_95"], 0.55)

    def test_prediction_below_ignore_overlap_threshold_remains_false_positive(self) -> None:
        result = self.evaluate(
            [uncertain_object("uncertain", [0, 0, 10, 10])],
            [prediction("firefighter", [4, 0, 14, 10])],
        )

        self.assertEqual(result["detector"]["overall"]["fp"], 1)
        self.assertEqual(result["detector"]["overall"]["ignored_prediction_count"], 0)

    def test_coverage_exposes_excluded_pairs_and_objects(self) -> None:
        pairs = [
            ground_truth_pair("valid", [uncertain_object("uncertain", [0, 0, 10, 10])]),
            ground_truth_pair("unknown", [], "ambiguous"),
            ground_truth_pair("failed", [], "missing_requested_object"),
        ]
        result = evaluate_benchmark(pairs, {}, EvaluationConfig(class_names=CLASSES))

        coverage = result["annotation_coverage"]
        self.assertEqual(coverage["evaluated_pairs"], 1)
        self.assertEqual(coverage["excluded_ambiguous_pairs"], 1)
        self.assertEqual(coverage["excluded_generation_failure_pairs"], 1)
        self.assertEqual(coverage["ambiguous_validity_objects_in_evaluated_pairs"], 1)
        self.assertEqual(result["schema_version"], 2)

    def test_ignore_regions_do_not_apply_to_other_pairs(self) -> None:
        pairs = [
            ground_truth_pair("one", [uncertain_object("uncertain", [0, 0, 10, 10])]),
            ground_truth_pair("two", []),
        ]
        result = evaluate_benchmark(
            pairs,
            {"two": prediction_report("two", [prediction("firefighter", [0, 0, 10, 10])])},
            EvaluationConfig(class_names=CLASSES),
        )

        self.assertEqual(result["detector"]["overall"]["ignored_prediction_count"], 0)
        self.assertEqual(result["detector"]["overall"]["fp"], 1)


if __name__ == "__main__":
    unittest.main()
