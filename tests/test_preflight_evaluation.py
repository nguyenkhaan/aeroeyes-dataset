import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.preflight import (
    EvaluationPaths,
    PreflightOptions,
    collect_path_errors,
    collect_preflight_errors,
)


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

    def test_requires_the_expected_pytorch_cuda_build(self) -> None:
        with (
            patch("src.evaluation.preflight.check_custom_ultralytics", return_value=(True, "")),
            patch("src.evaluation.preflight.torch.cuda.is_available", return_value=True),
            patch("src.evaluation.preflight.torch.version.cuda", "13.0"),
        ):
            with patch(
                "src.evaluation.preflight.collect_path_errors",
                return_value=[],
            ):
                errors = collect_preflight_errors(PreflightOptions(require_cuda=True))

        self.assertEqual(
            errors,
            ["PyTorch CUDA build mismatch: expected 12.8, found 13.0."],
        )


if __name__ == "__main__":
    unittest.main()
