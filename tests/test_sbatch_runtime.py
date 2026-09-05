import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class SbatchRuntimeTests(unittest.TestCase):
    def run_job(self, overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python"
            python.write_text(
                '#!/bin/bash\n'
                'case "$1" in\n'
                '  -u) step=cuda ;;\n'
                '  scripts/preflight_evaluation.py) step=preflight ;;\n'
                '  main.py) step=pipeline ;;\n'
                '  *) exit 99 ;;\n'
                'esac\n'
                'if [ "${SLOW_STEP:-}" = "$step" ]; then sleep 3; fi\n'
                'echo "ran $step"\n'
                'if [ "${FAIL_STEP:-}" = "$step" ]; then exit 42; fi\n'
            )
            gpu_check = root / "gpu_check.sh"
            gpu_check.write_text(
                '#!/bin/bash\n'
                'if [ "${SLOW_STEP:-}" = gpu ]; then sleep 3; fi\n'
                'echo "${GPU_RESULT:-3}"\n'
                'exit "${GPU_EXIT:-0}"\n'
            )
            python.chmod(0o755)
            gpu_check.chmod(0o755)
            environment = {
                **os.environ,
                "PYTHON": str(python), "GPU_CHECK_SCRIPT": str(gpu_check),
                "AEROEYES_MODEL_DIR": str(root / "models"),
                "SLURM_JOB_ID": "123", "SLURMD_NODENAME": "test-node",
                "JOB_TIMEOUT_SECONDS": "2", "STARTUP_TIMEOUT_SECONDS": "1",
                "PREFLIGHT_TIMEOUT_SECONDS": "1", **overrides,
            }
            environment.pop("CUDA_VISIBLE_DEVICES", None)
            return subprocess.run(
                ["bash", "-c", "module() { :; }; export -f module; exec bash sbatch.slurm"],
                env=environment, capture_output=True, text=True, timeout=20,
            )

    def test_success_with_unset_cuda_visible_devices(self):
        result = self.run_job({})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GPU: 3 | Pipe: gpu3", result.stdout)
        self.assertIn("ran pipeline", result.stdout)
        self.assertIn("Done.", result.stdout)

    def test_gpu_check_status_is_preserved_before_python_runs(self):
        for gpu_exit, expected in [(10, 0), (11, 1), (42, 42)]:
            with self.subTest(gpu_exit=gpu_exit):
                result = self.run_job({"GPU_EXIT": str(gpu_exit)})
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertNotIn("ran cuda", result.stdout)

    def test_slow_commands_finish_without_shell_timeouts(self):
        for step in ["gpu", "cuda", "preflight", "pipeline"]:
            with self.subTest(step=step):
                result = self.run_job({"SLOW_STEP": step})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Done.", result.stdout)

    def test_invalid_gpu_index_fails_before_pipeline(self):
        result = self.run_job({"GPU_RESULT": "bad output"})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("ran pipeline", result.stdout)

    def test_command_errors_stop_job_with_stage_name(self):
        for step, label, next_step in [
            ("cuda", "CUDA probe", "preflight"),
            ("preflight", "Preflight", "pipeline"),
            ("pipeline", "Pipeline", None),
        ]:
            with self.subTest(step=step):
                result = self.run_job({"FAIL_STEP": step})
                self.assertEqual(result.returncode, 42, result.stderr)
                self.assertIn(f"FAILED {label}", result.stderr)
                self.assertNotIn("Done.", result.stdout)
                if next_step:
                    self.assertNotIn(f"ran {next_step}", result.stdout)


if __name__ == "__main__":
    unittest.main()
