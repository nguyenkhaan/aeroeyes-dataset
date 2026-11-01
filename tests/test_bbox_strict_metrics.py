import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bbox_strict_metrics import evaluate_strict_v2
from tests.test_bbox_metrics import prediction, prediction_report
from tests.test_bbox_strict_v2 import adapt_strict_manifest, sample


class StrictMetricsTests(unittest.TestCase):
    def fixture(self):
        strict, benchmark = sample()
        after = strict["pairs"][0]["after"]
        after["confirmed_rescue_objects"][0]["bbox_xyxy"] = [0, 0, 10, 10]
        after["confirmed_new_rescue_objects"][0]["bbox_xyxy"] = [0, 0, 10, 10]
        after["confirmed_rescue_objects"][1]["bbox_xyxy"] = [12, 0, 20, 10]
        after["confirmed_existing_rescue_objects"][0]["bbox_xyxy"] = [12, 0, 20, 10]
        after["ignore_regions"][0]["bbox_xyxy"] = [21, 0, 31, 10]
        return adapt_strict_manifest(strict, benchmark)

    def test_all_statuses_and_separate_positive_ignore(self):
        document = self.fixture()
        good = prediction("firefighter", [0, 0, 10, 10], 0.9)
        old = prediction("firefighter", [12, 0, 20, 10], 0.8)
        uncertain = prediction("firefighter", [21, 0, 31, 10], 0.7)
        report = prediction_report("abcd", [good, old, uncertain], refined=[good, old, uncertain])
        result = evaluate_strict_v2(document, {"abcd": report})
        self.assertEqual(result["coverage"]["evaluated_pairs"], 1)
        self.assertEqual(result["coverage"]["status_counts"], {"ambiguous": 1})
        detector = result["detector"]["overall"]
        final = result["final_pipeline"]["overall"]
        self.assertEqual((detector["tp"], detector["fp"], detector["fn"]), (2, 0, 0))
        self.assertEqual((final["tp"], final["fp"], final["fn"]), (1, 1, 0))
        self.assertEqual(final["ignored_prediction_count"], 1)
        self.assertEqual(result["sensitivity_no_ignore"]["detector"]["overall"]["fp"], 1)

    def test_wrong_class_and_duplicate_are_false_positives(self):
        doc = self.fixture()
        predictions = [prediction("firefighter", [0, 0, 10, 10], 0.9),
                       prediction("firefighter", [0, 0, 10, 10], 0.8),
                       prediction("ambulance", [21, 0, 31, 10], 0.7)]
        result = evaluate_strict_v2(doc, {"abcd": prediction_report("abcd", predictions)})
        overall = result["detector"]["overall"]
        self.assertEqual((overall["tp"], overall["fp"], overall["fn"]), (1, 2, 1))
        self.assertEqual(overall["ignored_prediction_count"], 0)

    def test_missing_or_unexpected_report_is_rejected(self):
        doc = self.fixture()
        with self.assertRaisesRegex(ValueError, "reports"):
            evaluate_strict_v2(doc, {})
        report = prediction_report("abcd", [])
        with self.assertRaisesRegex(ValueError, "reports"):
            evaluate_strict_v2(doc, {"abcd": report, "extra": report})

    def test_ignore_boundary_and_one_to_one(self):
        doc = self.fixture()
        predictions = [prediction("firefighter", [21, 0, 26, 10], 0.9),
                       prediction("firefighter", [21, 0, 26, 10], 0.8)]
        result = evaluate_strict_v2(doc, {"abcd": prediction_report("abcd", predictions)})
        self.assertEqual(result["detector"]["overall"]["ignored_prediction_count"], 1)
        self.assertEqual(result["detector"]["overall"]["fp"], 1)

    def test_positive_has_priority_over_overlapping_ignore(self):
        doc = self.fixture()
        doc["pairs"][0]["ignore_regions"][0]["bbox_xyxy"] = [0, 0, 10, 10]
        predictions = [prediction("firefighter", [0, 0, 10, 10], 0.9)]
        result = evaluate_strict_v2(doc, {"abcd": prediction_report("abcd", predictions)})
        self.assertEqual(result["detector"]["overall"]["tp"], 1)
        self.assertEqual(result["detector"]["overall"]["ignored_prediction_count"], 0)

    def test_cli_checks_benchmark_and_refuses_overwrite(self):
        doc = self.fixture()
        doc["source_manifest_sha256"] = "benchmark-checksum"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_path = root / "gt.json"
            gt_path.write_text(json.dumps(doc), encoding="utf-8")
            run = root / "run"
            (run / "reports").mkdir(parents=True)
            manifest = {"status": "completed", "split": "validation", "selected_pair_ids": ["abcd"],
                        "benchmark_manifest_sha256": "benchmark-checksum", "result": {"processed_pairs": 1}}
            (run / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run / "reports" / "abcd.json").write_text(json.dumps(prediction_report("abcd", [])), encoding="utf-8")
            output = root / "metrics.json"
            command = [sys.executable, "scripts/evaluate_bbox_strict_v2.py", "--ground-truth", str(gt_path),
                       "--run-dir", str(run), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["coverage"]["evaluated_pairs"], 1)
            comparison = [sys.executable, "scripts/compare_bbox_strict_v2.py", str(output), str(output)]
            self.assertEqual(subprocess.run(comparison, capture_output=True).returncode, 0)
            altered = json.loads(output.read_text(encoding="utf-8"))
            altered["provenance"]["ground_truth_sha256"] = "different"
            other = root / "different.json"
            other.write_text(json.dumps(altered), encoding="utf-8")
            self.assertNotEqual(subprocess.run(comparison[:-1] + [str(other)], capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            manifest["benchmark_manifest_sha256"] = "different"
            (run / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            failed = subprocess.run(command[:-1] + [str(root / "bad.json")], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("checksums differ", failed.stderr)
