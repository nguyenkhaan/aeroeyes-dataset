import unittest
from pathlib import Path


class SbatchConfigurationTests(unittest.TestCase):
    def test_keeps_sbatch_directives_before_shell_commands(self) -> None:
        lines = Path("sbatch.slurm").read_text(encoding="utf-8").splitlines()
        strict_mode_index = lines.index("set -euo pipefail")
        last_directive_index = max(
            index for index, line in enumerate(lines) if line.startswith("#SBATCH")
        )

        self.assertGreater(strict_mode_index, last_directive_index)

    def test_runs_preflight_with_the_configured_python(self) -> None:
        script = Path("sbatch.slurm").read_text(encoding="utf-8")

        self.assertIn("set -euo pipefail", script)
        self.assertIn('"$PYTHON" scripts/preflight_evaluation.py --require-cuda', script)
        self.assertIn('"$PYTHON" main.py', script)

    def test_sets_evaluation_paths_and_writable_caches(self) -> None:
        script = Path("sbatch.slurm").read_text(encoding="utf-8")

        self.assertIn("export CMMD_REPO_DIR=", script)
        self.assertIn("export SDQM_REPO_DIR=", script)
        self.assertIn("export YOLO_CONFIG_DIR=", script)
        self.assertIn("export MPLCONFIGDIR=", script)
        self.assertIn("export SDQM_ENABLED=", script)
        self.assertNotIn("SDQL_ENABLED", script)


if __name__ == "__main__":
    unittest.main()
