import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation import sdqm


class SdqmTests(unittest.TestCase):
    def test_writes_failed_run_report_when_output_directory_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "sdqm"

            report_path = sdqm.write_sdqm_status_report(
                output_dir,
                {
                    "status": "failed",
                    "reason": "SDQM repo missing",
                    "real_image_count": 2,
                    "synthetic_image_count": 2,
                },
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason"], "SDQM repo missing")
        self.assertEqual(report["real_image_count"], 2)
        self.assertEqual(report["synthetic_image_count"], 2)

    def test_checks_sdqm_repo_before_starting_yolo_export(self) -> None:
        image_paths = [Path("first.jpg"), Path("second.jpg")]
        with patch.object(sdqm, "list_images", return_value=image_paths), patch.object(
            sdqm,
            "_load_calculate_sdqm",
            side_effect=FileNotFoundError("SDQM repo missing"),
        ), patch.object(sdqm, "export_yolo_pair") as export_yolo_pair:
            with self.assertRaisesRegex(FileNotFoundError, "SDQM repo missing"):
                sdqm.compute_dataset_sdqm("real", "synthetic")

        export_yolo_pair.assert_not_called()

    def test_adds_sdqm_subdirectories_to_import_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory) / "SDQM"
            subdir = repo_root / "labels_and_characteristics"
            subdir.mkdir(parents=True)

            added_paths = [str(repo_root.resolve()), str(subdir.resolve())]
            for path_text in added_paths:
                if path_text in sys.path:
                    sys.path.remove(path_text)

            try:
                sdqm._ensure_sdqm_import_paths(repo_root)

                for path_text in added_paths:
                    self.assertIn(path_text, sys.path)
            finally:
                for path_text in added_paths:
                    if path_text in sys.path:
                        sys.path.remove(path_text)

    def test_writes_dataset_summary(self) -> None:
        report = {
            "real_image_count": 2,
            "synthetic_image_count": 2,
            "embedding_model": "test-model",
            "yolo_export_enabled": False,
            "vinfo_status": "skipped",
            "metrics": {"similarity": 0.125},
            "regression_ran": False,
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sdqm,
            "SDQM_SUMMARY_PATH",
            str(Path(directory) / "sdqm_summary.md"),
        ):
            summary_path = sdqm.write_sdqm_summary(report)

            summary = summary_path.read_text(encoding="utf-8")

        self.assertIn("| similarity | 0.1250 |", summary)
        self.assertIn("- V-Info: skipped", summary)


if __name__ == "__main__":
    unittest.main()
