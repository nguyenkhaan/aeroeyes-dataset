import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.sdqm_regression import (
    append_sdqm_history_row,
    run_sdqm_regression,
)


class SdqmRegressionTests(unittest.TestCase):
    def test_rejects_history_rows_without_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "sdqm_history.csv"

            with self.assertRaisesRegex(ValueError, "at least one metric"):
                append_sdqm_history_row(history_path, {})

            self.assertFalse(history_path.exists())

    def test_replaces_an_existing_empty_history_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "sdqm_history.csv"
            history_path.touch()

            append_sdqm_history_row(history_path, {"similarity": 0.5})

            history_df = pd.read_csv(history_path)

        self.assertEqual(history_df.to_dict("records"), [{"similarity": 0.5}])

    def test_skips_regression_for_an_empty_history_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "sdqm_history.csv"
            history_path.touch()

            result = run_sdqm_regression(history_path)

        self.assertIsNone(result)

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
