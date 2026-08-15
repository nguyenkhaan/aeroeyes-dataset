import unittest
from pathlib import Path


class VpsSetupScriptTests(unittest.TestCase):
    def test_bootstrap_uses_one_virtual_environment_for_all_dependencies(self) -> None:
        script = Path("scripts/setup_vps.sh").read_text(encoding="utf-8")

        self.assertIn("set -euo pipefail", script)
        self.assertIn('"$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"', script)
        self.assertIn('"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"', script)
        self.assertIn('"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/cmmd-pytorch/requirements.txt"', script)
        self.assertIn('PYTHON="$VENV_PYTHON" bash "$ROOT_DIR/scripts/setup_sdqm_vinfo.sh"', script)

    def test_bootstrap_creates_runtime_directories_without_overwriting_env(self) -> None:
        script = Path("scripts/setup_vps.sh").read_text(encoding="utf-8")

        self.assertIn('mkdir -p "$ROOT_DIR/logs"', script)
        self.assertIn('if [[ ! -f "$ROOT_DIR/.env" ]]', script)
        self.assertIn('cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"', script)


if __name__ == "__main__":
    unittest.main()
