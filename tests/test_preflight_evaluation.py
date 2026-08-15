import tempfile
import unittest
from pathlib import Path

from src.evaluation.preflight import EvaluationPaths, collect_path_errors


class PreflightEvaluationTests(unittest.TestCase):
    def test_accepts_complete_evaluation_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = EvaluationPaths(
                cmmd_main=root / "cmmd" / "main.py",
                sdqm_main=root / "SDQM" / "sdqm.py",
                data_yaml=root / "config" / "data.yaml",
            )
            for path in (paths.cmmd_main, paths.sdqm_main, paths.data_yaml):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            self.assertEqual(collect_path_errors(paths), [])

    def test_reports_each_missing_evaluation_asset(self) -> None:
        root = Path("missing")
        paths = EvaluationPaths(
            cmmd_main=root / "cmmd" / "main.py",
            sdqm_main=root / "SDQM" / "sdqm.py",
            data_yaml=root / "config" / "data.yaml",
        )

        errors = collect_path_errors(paths)

        self.assertEqual(len(errors), 3)
        self.assertIn("CMMD", errors[0])
        self.assertIn("SDQM", errors[1])
        self.assertIn("YOLO data YAML", errors[2])


if __name__ == "__main__":
    unittest.main()
