import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bbox_error_analysis import analyze_strict_errors
from src.evaluation.bbox_strict_v2 import adapt_strict_manifest
from tests.test_bbox_metrics import prediction, prediction_report
from tests.test_bbox_strict_v2 import sample


class ErrorAnalysisTests(unittest.TestCase):
    def fixture(self):
        strict, benchmark = sample()
        after = strict["pairs"][0]["after"]
        after["confirmed_rescue_objects"][0]["bbox_xyxy"] = [0, 0, 10, 10]
        after["confirmed_rescue_objects"][1]["bbox_xyxy"] = [12, 0, 20, 10]
        after["ignore_regions"][0]["bbox_xyxy"] = [21, 0, 31, 10]
        return adapt_strict_manifest(strict, benchmark)

    def test_accounts_for_duplicate_ignore_existing_and_wrong_class(self):
        gt = self.fixture()
        predictions = [prediction("firefighter", [0, 0, 10, 10], .9),
                       prediction("firefighter", [0, 0, 10, 10], .8),
                       prediction("firefighter", [12, 0, 20, 10], .7),
                       prediction("firefighter", [21, 0, 31, 10], .6),
                       prediction("ambulance", [0, 0, 10, 10], .5)]
        result = analyze_strict_errors(gt, {"abcd": prediction_report("abcd", predictions)})
        final = result["summary"]["final_pipeline"]
        self.assertEqual((final["tp"], final["fp"], final["fn"], final["ignored"]), (1, 3, 0, 1))
        reasons = [row["reason"] for row in result["rows"] if row["tier"] == "final_pipeline" and row["outcome"] == "fp"]
        self.assertCountEqual(reasons, ["duplicate_confirmed", "overlap_existing", "overlap_other_class"])

    def test_diagnostic_never_matches_ignore_from_another_pair(self):
        gt = self.fixture()
        other = dict(gt["pairs"][0], pair_id="other", source_pair_id="0001", after_objects=[], ignore_regions=[], before_objects=[])
        gt["pairs"].append(other)
        reports = {"abcd": prediction_report("abcd", []),
                   "other": prediction_report("other", [prediction("firefighter", [21, 0, 31, 10])])}
        result = analyze_strict_errors(gt, reports)
        fp = [r for r in result["rows"] if r["tier"] == "detector" and r["outcome"] == "fp"]
        self.assertEqual(len(fp), 1)
        self.assertEqual(fp[0]["pair_id"], "other")

    def test_reports_must_be_complete(self):
        with self.assertRaisesRegex(ValueError, "reports"):
            analyze_strict_errors(self.fixture(), {})

    def test_mixed_overlaps_match_evaluator_across_many_cases(self):
        randomizer = random.Random(42)
        boxes = ([0, 0, 10, 10], [0, 0, 5, 10], [12, 0, 20, 10],
                 [21, 0, 31, 10], [3, 0, 13, 10], [0, 15, 10, 25])
        for _ in range(30):
            values = [prediction(randomizer.choice(("firefighter", "ambulance")),
                                 randomizer.choice(boxes), randomizer.choice((.2, .5, .9)))
                      for _ in range(12)]
            result = analyze_strict_errors(self.fixture(), {"abcd": prediction_report("abcd", values)})
            self.assertEqual(result["summary"]["detector"]["tp"] + result["summary"]["detector"]["fn"], 2)

    def test_duplicate_positive_has_priority_over_overlapping_ignore(self):
        gt = self.fixture()
        gt["pairs"][0]["ignore_regions"][0]["bbox_xyxy"] = [0, 0, 10, 10]
        predictions = [prediction("firefighter", [0, 0, 10, 10], .9),
                       prediction("firefighter", [0, 0, 10, 10], .8)]
        result = analyze_strict_errors(gt, {"abcd": prediction_report("abcd", predictions)})
        fp = [row for row in result["rows"] if row["tier"] == "detector" and row["outcome"] == "fp"]
        self.assertEqual([row["reason"] for row in fp], ["duplicate_confirmed"])
        self.assertEqual(result["summary"]["detector"]["ignored"], 0)

    def test_cli_checks_report_hashes_and_refuses_overwrite(self):
        gt = self.fixture()
        gt["source_manifest_sha256"] = "benchmark"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ground_truth = root / "gt.json"
            ground_truth.write_text(json.dumps(gt), encoding="utf-8")
            run = root / "run"
            (run / "reports").mkdir(parents=True)
            manifest = {"status": "completed", "split": "validation", "selected_pair_ids": ["abcd"],
                        "benchmark_manifest_sha256": "benchmark", "result": {"processed_pairs": 1}}
            (run / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report_path = run / "reports" / "abcd.json"
            report_path.write_text(json.dumps(prediction_report("abcd", [])), encoding="utf-8")
            metrics = root / "metrics.json"
            evaluation = [sys.executable, "scripts/evaluate_bbox_strict_v2.py", "--ground-truth", str(ground_truth),
                          "--run-dir", str(run), "--output", str(metrics)]
            self.assertEqual(subprocess.run(evaluation, capture_output=True).returncode, 0)
            output = root / "analysis"
            analysis = [sys.executable, "scripts/analyze_bbox_strict_errors.py", "--ground-truth", str(ground_truth),
                        "--run-dir", str(run), "--metrics", str(metrics), "--output-dir", str(output)]
            first = subprocess.run(analysis, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue((output / "fp_fn.csv").is_file())
            self.assertNotEqual(subprocess.run(analysis, capture_output=True).returncode, 0)
            report_path.write_text(json.dumps(prediction_report("abcd", [prediction("ambulance", [0, 0, 10, 10])])), encoding="utf-8")
            failed = subprocess.run(analysis[:-1] + [str(root / "bad")], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("reports differ", failed.stderr)
