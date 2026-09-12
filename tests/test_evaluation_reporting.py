import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation import reporting


class EvaluationReportingTests(unittest.TestCase):
    def test_runs_cmmd_and_sdqm_for_existing_image_sets(self) -> None:
        with patch.object(
            reporting,
            "list_images",
            side_effect=[[Path("real-1.jpg"), Path("real-2.jpg")], [Path("gen-1.jpg"), Path("gen-2.jpg")]],
        ), patch.object(
            reporting,
            "compute_dataset_cmmd",
            return_value=0.42,
        ) as compute_cmmd, patch.object(
            reporting,
            "compute_dataset_sdqm",
            return_value={"similarity": 0.8},
        ) as compute_sdqm:
            result = reporting.run_dataset_evaluation("real", "synthetic")

        self.assertEqual(result.cmmd_score, 0.42)
        self.assertEqual(result.sdqm_metrics, {"similarity": 0.8})
        compute_cmmd.assert_called_once_with("real", "synthetic")
        compute_sdqm.assert_called_once_with("real", "synthetic")

    def test_requires_two_images_for_sdqm(self) -> None:
        with patch.object(
            reporting,
            "list_images",
            side_effect=[[Path("real-1.jpg")], [Path("gen-1.jpg")]],
        ), self.assertRaisesRegex(ValueError, "at least 2"):
            reporting.run_dataset_evaluation("real", "synthetic")


if __name__ == "__main__":
    unittest.main()
