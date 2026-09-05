import ast
import contextlib
import io
import json
import os
import re
import tempfile
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class PipelineLimitTests(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.download = Mock(return_value=b"image")
        self.quality_gate = Mock(return_value=False)
        self.cmmd = Mock(return_value=None)
        self.sdqm = Mock(return_value=None)
        fake_image = Mock()
        fake_image.convert.return_value = fake_image
        self.environment = {
            "os": os,
            "json": json,
            "re": re,
            "traceback": traceback,
            "BytesIO": io.BytesIO,
            "time": time,
            "monotonic": time.monotonic,
            "stage": lambda *args: contextlib.nullcontext(),
            "OUTPUT_DIR": self.output.name,
            "LIMIT_IMAGES": 1,
            "MAX_ATTEMPTS": 3,
            "MAX_CONSECUTIVE_ERRORS": 2,
            "GENERATION_TIMEOUT_SECONDS": 3600,
            "STAGE_TIMEOUT_SECONDS": 600,
            "EVALUATION_TIMEOUT_SECONDS": 600,
            "HEADERS": {},
            "REQUEST_TIMEOUT": 10,
            "DOWNLOAD_RETRIES": 1,
            "IMAGE_SIZE": 512,
            "BASE_SEED": 1,
            "GUIDANCE_SCALE": 2.5,
            "NUM_INFERENCE_STEPS": 20,
            "vision_model": None,
            "vision_processor": None,
            "pipe": None,
            "evaluators": None,
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: False,
                OutOfMemoryError=MemoryError,
            )),
            "Image": SimpleNamespace(open=Mock(return_value=fake_image)),
            "download_image": self.download,
            "resize_center_crop": Mock(return_value=fake_image),
            "generate_scene_description": Mock(return_value="flood scene"),
            "generate_rescue_instruction": Mock(return_value="add rescuers"),
            "build_flux_prompt": Mock(return_value="rescue in flood"),
            "generate_rescue_image": Mock(return_value=fake_image),
            "evaluate_quality": Mock(return_value=(0.8, 0.9)),
            "compute_o_score": Mock(return_value=0.85),
            "compute_ssim": Mock(return_value=0.7),
            "passes_quality_gate": self.quality_gate,
            "save_generated_image": Mock(return_value="generated.png"),
            "save_reference_images": Mock(),
            "cleanup": Mock(),
            "export_evaluation_report": Mock(return_value=None),
            "run_cmmd_report": self.cmmd,
            "run_sdqm_report": self.sdqm,
            "write_metadata_jsonl": Mock(return_value=None),
            "data": {
                f"image_{index}": {"incidents": {"flood": 1}, "url": f"url_{index}"}
                for index in range(12)
            },
        }

    def run_pipeline(self):
        # Execute the production loop without loading GPU models at module import.
        source_path = Path(__file__).resolve().parents[1] / "main.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        start = next(
            index for index, node in enumerate(module.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "count"
                    for target in node.targets)
        )
        helpers = [node for node in module.body
                   if isinstance(node, ast.FunctionDef) and node.name == "make_safe_stem"]
        program = ast.Module(body=helpers + module.body[start:], type_ignores=[])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                exec(compile(program, str(source_path), "exec"), self.environment)
            except SystemExit as exc:
                return exc.code
        return 0

    def test_quality_rejections_stop_at_attempt_limit_without_dataset_evaluation(self):
        exit_code = self.run_pipeline()
        self.assertEqual(self.download.call_count, 3)
        self.assertEqual(exit_code, 2)
        self.cmmd.assert_not_called()
        self.sdqm.assert_not_called()

    def test_expired_time_budget_prevents_the_next_attempt(self):
        self.environment["monotonic"] = Mock(side_effect=[0, 0, 3600])
        exit_code = self.run_pipeline()
        self.assertEqual(self.download.call_count, 1)
        self.assertEqual(exit_code, 2)
        self.cmmd.assert_not_called()
        self.sdqm.assert_not_called()

    def test_dataset_exhaustion_without_target_is_reported_as_incomplete(self):
        self.environment["data"] = {"only_image": {
            "incidents": {"flood": 1}, "url": "only_url",
        }}
        exit_code = self.run_pipeline()
        self.assertEqual(self.download.call_count, 1)
        self.assertEqual(exit_code, 2)
        self.cmmd.assert_not_called()

    def test_repeated_download_errors_stop_before_attempt_limit(self):
        self.download.return_value = None
        exit_code = self.run_pipeline()
        self.assertEqual(self.download.call_count, 2)
        self.assertEqual(exit_code, 2)
        self.cmmd.assert_not_called()

    def test_repeated_model_and_image_errors_stop_early(self):
        error_cases = [
            ("Image", ValueError("invalid image")),
            ("generate_scene_description", RuntimeError("scene failed")),
            ("generate_rescue_instruction", RuntimeError("instruction failed")),
            ("generate_rescue_image", MemoryError("CUDA out of memory")),
            ("generate_rescue_image", RuntimeError("generation failed")),
            ("evaluate_quality", RuntimeError("quality failed")),
        ]
        for dependency, error in error_cases:
            with self.subTest(dependency=dependency, error=type(error).__name__):
                failing_call = self.environment[dependency]
                if dependency == "Image":
                    failing_call = failing_call.open
                failing_call.side_effect = error
                self.download.reset_mock()
                self.cmmd.reset_mock()
                try:
                    exit_code = self.run_pipeline()
                finally:
                    failing_call.side_effect = None
                self.assertEqual(self.download.call_count, 2)
                self.assertEqual(exit_code, 2)
                self.cmmd.assert_not_called()

    def test_repeated_save_errors_do_not_reset_error_streak(self):
        self.quality_gate.return_value = True
        self.environment["save_generated_image"].side_effect = OSError("disk full")
        exit_code = self.run_pipeline()
        self.assertEqual(self.download.call_count, 2)
        self.assertEqual(exit_code, 2)
        self.cmmd.assert_not_called()

    def test_quality_rejection_resets_consecutive_error_streak(self):
        self.environment["MAX_ATTEMPTS"] = 8
        self.download.side_effect = [None, b"image", None, None, b"unused"]
        exit_code = self.run_pipeline()
        self.assertEqual(self.download.call_count, 4)
        self.assertEqual(exit_code, 2)

    def test_success_stops_at_target_and_persists_metadata(self):
        self.quality_gate.return_value = True
        exit_code = self.run_pipeline()
        self.assertEqual(exit_code, 0)
        self.assertEqual(self.download.call_count, 1)
        metadata_path = Path(self.output.name) / "_metadata" / "image_0.json"
        self.assertEqual(json.loads(metadata_path.read_text())["image_key"], "image_0")

    def test_ineligible_and_existing_images_do_not_consume_attempts(self):
        self.environment["data"]["image_0"]["incidents"] = {}
        self.environment["data"]["image_1"].pop("url")
        (Path(self.output.name) / "image_2.png").touch()
        exit_code = self.run_pipeline()
        self.assertEqual(self.download.call_count, 3)
        self.assertEqual(self.download.call_args_list[0].args[0], "url_3")
        self.assertEqual(exit_code, 2)

    def test_metadata_survives_later_attempt_limit_stop(self):
        self.environment["LIMIT_IMAGES"] = 2
        self.quality_gate.side_effect = [True] + [False] * 11
        exit_code = self.run_pipeline()
        self.assertEqual(exit_code, 2)
        metadata_path = Path(self.output.name) / "_metadata" / "image_0.json"
        self.assertEqual(json.loads(metadata_path.read_text())["image_key"], "image_0")
        self.cmmd.assert_not_called()


if __name__ == "__main__":
    unittest.main()
