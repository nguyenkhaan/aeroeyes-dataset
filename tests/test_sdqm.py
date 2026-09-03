import csv
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
        with (
            patch.object(sdqm, "list_images", return_value=image_paths),
            patch.object(
                sdqm,
                "_load_calculate_sdqm",
                side_effect=FileNotFoundError("SDQM repo missing"),
            ),
            patch.object(sdqm, "export_yolo_pair") as export_yolo_pair,
            self.assertRaisesRegex(FileNotFoundError, "SDQM repo missing"),
        ):
            sdqm.compute_dataset_sdqm("real", "synthetic")

        export_yolo_pair.assert_not_called()

    def test_adds_only_sdqm_repository_root_to_import_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory) / "SDQM"
            subdir = repo_root / "labels_and_characteristics"
            subdir.mkdir(parents=True)

            repository_path = str(repo_root.resolve())
            subdirectory_path = str(subdir.resolve())
            for path_text in (repository_path, subdirectory_path):
                if path_text in sys.path:
                    sys.path.remove(path_text)

            try:
                sdqm._ensure_sdqm_import_paths(repo_root)

                self.assertIn(repository_path, sys.path)
                self.assertNotIn(subdirectory_path, sys.path)
            finally:
                for path_text in (repository_path, subdirectory_path):
                    if path_text in sys.path:
                        sys.path.remove(path_text)

    def test_explains_how_to_install_a_missing_sdqm_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory) / "SDQM"
            repo_root.mkdir()
            (repo_root / "sdqm.py").write_text("import missing_sdqm_dependency\n")

            with (
                patch.object(sdqm, "SDQM_REPO_DIR", str(repo_root)),
                patch.object(
                    sdqm,
                    "check_custom_ultralytics",
                    return_value=(True, "ready"),
                ),
                self.assertRaisesRegex(
                    ModuleNotFoundError,
                    "pip install -r requirements-sdqm.txt",
                ),
            ):
                sdqm._load_calculate_sdqm()

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
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                sdqm,
                "SDQM_SUMMARY_PATH",
                str(Path(directory) / "sdqm_summary.md"),
            ),
        ):
            summary_path = sdqm.write_sdqm_summary(report)

            summary = summary_path.read_text(encoding="utf-8")

        self.assertIn("| similarity | 0.1250 |", summary)
        self.assertIn("- V-Info: skipped", summary)

    def test_uses_auto_dataset_and_writes_numeric_metrics_csv(self) -> None:
        upstream_arguments = {}

        def calculate_sdqm(**kwargs):
            upstream_arguments.update(kwargs)
            return [{"Dataset Similarity_mauve": 0.75}]

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                sdqm,
                "list_images",
                side_effect=[
                    [Path("real-1.jpg"), Path("real-2.jpg")],
                    [Path("synthetic-1.jpg"), Path("synthetic-2.jpg")],
                ],
            ),
            patch.object(
                sdqm,
                "_load_calculate_sdqm",
                return_value=calculate_sdqm,
            ),
            patch.object(
                sdqm,
                "export_yolo_pair",
                return_value=(Path("real-yolo"), Path("synthetic-yolo")),
            ),
            patch.object(sdqm, "embed_image_directory"),
            patch.object(sdqm, "_resolve_image_size", return_value=(512, 512)),
            patch.object(sdqm, "_maybe_append_history"),
            patch.object(sdqm, "_maybe_run_regression", return_value=None),
            patch.object(
                sdqm,
                "SDQM_SUMMARY_PATH",
                str(Path(directory) / "sdqm_summary.md"),
            ),
        ):
            output_dir = Path(directory) / "sdqm"
            metrics = sdqm.compute_dataset_sdqm(
                "real",
                "synthetic",
                output_dir=str(output_dir),
                export_yolo=True,
                include_vinfo=False,
            )
            with (output_dir / "sdqm_values.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(upstream_arguments["dataset"], "auto")
        self.assertEqual(metrics, {"Dataset Similarity_mauve": 0.75})
        self.assertEqual(rows, [{"Dataset Similarity_mauve": "0.75"}])

    def test_rejects_empty_upstream_metrics_before_writing_history(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                sdqm,
                "list_images",
                side_effect=[
                    [Path("real-1.jpg"), Path("real-2.jpg")],
                    [Path("synthetic-1.jpg"), Path("synthetic-2.jpg")],
                ],
            ),
            patch.object(
                sdqm,
                "_load_calculate_sdqm",
                return_value=lambda **kwargs: [],
            ),
            patch.object(sdqm, "embed_image_directory"),
            patch.object(sdqm, "_resolve_image_size", return_value=(512, 512)),
            patch.object(sdqm, "_maybe_append_history") as append_history,
            self.assertRaisesRegex(RuntimeError, "no numeric metrics"),
        ):
            sdqm.compute_dataset_sdqm(
                "real",
                "synthetic",
                output_dir=str(Path(directory) / "sdqm"),
                export_yolo=False,
                include_vinfo=False,
            )

        append_history.assert_not_called()


if __name__ == "__main__":
    unittest.main()
