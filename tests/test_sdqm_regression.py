import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.sdqm_regression import run_sdqm_regression


class SdqmRegressionTests(unittest.TestCase):
    def test_skips_regression_when_history_has_no_map_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "sdqm_history.csv"
            pd.DataFrame(
                [
                    {"similarity": 0.10},
                    {"similarity": 0.20},
                    {"similarity": 0.30},
                ]
            ).to_csv(history_path, index=False)

            result = run_sdqm_regression(history_path)

            self.assertIsNone(result)
            self.assertFalse((history_path.parent / "regression_report.json").exists())

    def test_writes_correlations_when_history_has_map_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "sdqm_history.csv"
            report_path = history_path.parent / "regression_report.json"
            pd.DataFrame(
                [
                    {"map": 0.10, "similarity": 0.15},
                    {"map": 0.20, "similarity": 0.30},
                    {"map": 0.30, "similarity": 0.45},
                ]
            ).to_csv(history_path, index=False)

            result = run_sdqm_regression(history_path, output_path=report_path)

            self.assertIn("similarity", result)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["row_count"], 3)

    def test_skips_metric_when_too_few_paired_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "sdqm_history.csv"
            pd.DataFrame(
                [
                    {"map": 0.10, "similarity": 0.15},
                    {"map": 0.20, "similarity": None},
                    {"map": 0.30, "similarity": 0.45},
                ]
            ).to_csv(history_path, index=False)

            result = run_sdqm_regression(history_path)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertIsNone(result["similarity"]["pearson_r"])


if __name__ == "__main__":
    unittest.main()
