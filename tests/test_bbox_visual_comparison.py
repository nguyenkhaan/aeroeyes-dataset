import unittest

from scripts.render_bbox_conflict_review import _verify_runs
from src.evaluation.bbox_strict_v2 import adapt_strict_manifest
from src.evaluation.bbox_visual_comparison import select_review_cases
from tests.test_bbox_metrics import prediction, prediction_report
from tests.test_bbox_strict_v2 import sample


class VisualComparisonTests(unittest.TestCase):
    def test_run_comparison_requires_same_code_and_only_threshold_change(self):
        common = {"git": {"commit": "same", "tracked_files_dirty": False},
                  "result": {"model_revision": "m", "ontology_sha256": "o", "class_map_sha256": "c"}}
        control = {**common, "config": {"name": "control", "prompt": "a", "cross_class_iou_threshold": None}}
        candidate = {**common, "config": {"name": "candidate", "prompt": "a", "cross_class_iou_threshold": .75}}
        metrics = {"evaluation_protocol": {"version": "bbox-strict-validation-v2-iou50-1"}, "coverage": {"evaluated_pairs": 1}}
        _verify_runs(control, candidate, metrics, metrics)
        with self.assertRaises(ValueError):
            _verify_runs(control, {**candidate, "config": {**candidate["config"], "prompt": "changed"}}, metrics, metrics)
        with self.assertRaises(ValueError):
            _verify_runs(control, {**candidate, "git": {"commit": "other", "tracked_files_dirty": False}}, metrics, metrics)

    def fixture(self):
        strict, benchmark = sample()
        after = strict["pairs"][0]["after"]
        after["confirmed_rescue_objects"][0]["label"] = "ambulance"
        after["confirmed_rescue_objects"][0]["bbox_xyxy"] = [0, 0, 10, 10]
        after["confirmed_rescue_objects"][1]["label"] = "emergency_vehicle"
        after["confirmed_rescue_objects"][1]["bbox_xyxy"] = [12, 0, 20, 10]
        gt = adapt_strict_manifest(strict, benchmark)
        base = [prediction("ambulance", [0, 0, 10, 10]),
                prediction("ambulance", [12, 0, 20, 10]),
                prediction("emergency_vehicle", [12, 0, 20, 10])]
        candidate = [prediction("emergency_vehicle", [12, 0, 20, 10])]
        return gt, {"abcd": prediction_report("abcd", base)}, {"abcd": prediction_report("abcd", candidate)}

    def test_selects_lost_true_positive_and_overlapping_false_positive(self):
        gt, baseline, candidate = self.fixture()
        cases = select_review_cases(gt, baseline, candidate, max_fp_pairs=5)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["source_pair_id"], "0000")
        self.assertEqual(cases[0]["lost_ambulance_gt_ids"], ["after-0"])
        self.assertEqual(cases[0]["baseline_fp_overlap_emergency"], 1)

    def test_rejects_missing_reports_and_negative_limit(self):
        gt, baseline, candidate = self.fixture()
        with self.assertRaises(ValueError):
            select_review_cases(gt, baseline, {}, max_fp_pairs=5)
        with self.assertRaises(ValueError):
            select_review_cases(gt, baseline, candidate, max_fp_pairs=-1)

    def test_does_not_select_false_positive_when_limit_zero(self):
        gt, baseline, candidate = self.fixture()
        cases = select_review_cases(gt, baseline, baseline, max_fp_pairs=0)
        self.assertEqual(cases, [])
