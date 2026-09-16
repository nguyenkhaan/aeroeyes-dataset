import ast
import contextlib
import gc
import io
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class GenerationMemoryTests(unittest.TestCase):
    def setUp(self):
        source_path = Path(__file__).resolve().parents[1] / "main.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        function = next(node for node in module.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "generate_flux_safe")
        self.environment = {
            "Image": SimpleNamespace(Image=object),
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: False, OutOfMemoryError=MemoryError)),
            "stage": lambda *args: contextlib.nullcontext(),
            "STAGE_TIMEOUT_SECONDS": 30,
            "GUIDANCE_SCALE": 2.5,
            "NUM_INFERENCE_STEPS": 4,
            "BASE_SEED": 123,
            "cleanup_cuda": gc.collect,
            "print_gpu_memory": Mock(),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(source_path), "exec"), self.environment)
        self.generate = self.environment["generate_flux_safe"]

    def test_retry_releases_failed_call_frame_and_preserves_inputs(self):
        calls = []
        failed_allocations = []
        retry_state = []
        result = object()

        class Allocation:
            pass

        def generate(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                allocation = Allocation()
                failed_allocations.append(weakref.ref(allocation))
                raise MemoryError("CUDA out of memory")
            retry_state.append((sys.exc_info()[1], failed_allocations[0]()))
            return result

        self.environment["generate_rescue_image"] = generate
        with contextlib.redirect_stdout(io.StringIO()):
            actual = self.generate(object(), object(), "flood rescue", 7)

        self.assertIs(actual, result)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(retry_state, [(None, None)])

    def test_persistent_oom_is_retried_only_once(self):
        generate = Mock(side_effect=MemoryError("CUDA out of memory"))
        self.environment["generate_rescue_image"] = generate
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(MemoryError):
                self.generate(object(), object(), "flood rescue", 0)
        self.assertEqual(generate.call_count, 2)

    def test_non_memory_failures_are_not_retried(self):
        generate = Mock(side_effect=ValueError("invalid input"))
        self.environment["generate_rescue_image"] = generate
        with self.assertRaises(ValueError):
            self.generate(object(), object(), "flood rescue", 0)
        self.assertEqual(generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
