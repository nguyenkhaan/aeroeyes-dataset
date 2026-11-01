import unittest

from src.evaluation.bbox_strict_v2 import adapt_strict_manifest
from src.evaluation.bbox_metrics import EvaluationConfig, evaluate_benchmark


def obj(index, label="firefighter", new="true", valid="true"):
    return {"box_index_in_frame": index, "label": label, "bbox_xyxy": [1, 2, 10, 20],
            "is_newly_added": new, "is_valid_rescue": valid}


def sample():
    known = obj(0)
    old = obj(1, new="false")
    uncertain = obj(2, valid="ambiguous")
    pair = {"pair_id": "0000", "pair_hash": "abcd", "generation_status": "ambiguous",
            "before": {"file": "0000__00_before__abcd.png", "width": 32, "height": 32,
                       "objects": [old]},
            "after": {"file": "0000__01_after__abcd.png", "width": 32, "height": 32,
                      "confirmed_rescue_objects": [known, old],
                      "confirmed_new_rescue_objects": [known],
                      "confirmed_existing_rescue_objects": [old],
                      "ignore_regions": [uncertain]}}
    strict = {"split": "validation", "pairs": [pair]}
    benchmark = {"pairs": [{"pair_id": "abcd", "split": "validation"}]}
    return strict, benchmark


class StrictV2Tests(unittest.TestCase):
    def test_preserves_pair_status_and_separates_positive_and_ignore(self):
        strict, benchmark = sample()
        result = adapt_strict_manifest(strict, benchmark)
        pair = result["pairs"][0]
        self.assertEqual(pair["pair_id"], "abcd")
        self.assertEqual(pair["generation_status"], "ambiguous")
        self.assertEqual(len(pair["after_objects"]), 2)
        self.assertEqual(len(pair["ignore_regions"]), 1)
        self.assertEqual(pair["ignore_regions"][0]["class_name"], "firefighter")
        self.assertEqual(result["annotation_status"], "single_annotator_reviewed")

    def test_rejects_mismatched_pair_or_conflicting_positive_lists(self):
        strict, benchmark = sample()
        strict["pairs"][0]["after"]["confirmed_new_rescue_objects"] = []
        with self.assertRaises(ValueError):
            adapt_strict_manifest(strict, benchmark)
        strict, benchmark = sample()
        benchmark["pairs"][0]["pair_id"] = "different"
        with self.assertRaises(ValueError):
            adapt_strict_manifest(strict, benchmark)

    def test_rejects_duplicate_or_invalid_boxes(self):
        strict, benchmark = sample()
        strict["pairs"][0]["after"]["ignore_regions"][0]["box_index_in_frame"] = 0
        with self.assertRaises(ValueError):
            adapt_strict_manifest(strict, benchmark)
        strict, benchmark = sample()
        strict["pairs"][0]["after"]["ignore_regions"][0]["bbox_xyxy"] = [0, 0, 100, 5]
        with self.assertRaises(ValueError):
            adapt_strict_manifest(strict, benchmark)

    def test_existing_evaluator_rejects_separate_ignore_regions(self):
        strict, benchmark = sample()
        pairs = adapt_strict_manifest(strict, benchmark)["pairs"]
        with self.assertRaisesRegex(ValueError, "strict-v2 evaluator"):
            evaluate_benchmark(pairs, {}, EvaluationConfig(class_names=("firefighter",)))
