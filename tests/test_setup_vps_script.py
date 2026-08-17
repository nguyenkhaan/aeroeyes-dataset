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

    def test_bootstrap_reinstalls_the_cuda_12_8_pytorch_wheels_before_base_dependencies(
        self,
    ) -> None:
        script = Path("scripts/setup_vps.sh").read_text(encoding="utf-8")

        pytorch_install = (
            '"$VENV_PYTHON" -m pip install --upgrade --force-reinstall '
            '--no-cache-dir -r "$ROOT_DIR/requirements-pytorch-cu128.txt"'
        )
        base_install = '"$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"'

        self.assertIn(pytorch_install, script)
        self.assertLess(script.index(pytorch_install), script.index(base_install))
        self.assertIn('"$VENV_PYTHON" -m pip check', script)

    def test_cuda_12_8_requirement_file_pins_matching_official_wheels(self) -> None:
        requirements = Path("requirements-pytorch-cu128.txt").read_text(encoding="utf-8")

        self.assertIn("--index-url https://download.pytorch.org/whl/cu128", requirements)
        self.assertIn("torch==2.8.0", requirements)
        self.assertIn("torchvision==0.23.0", requirements)

    def test_base_requirements_do_not_reinstall_cuda_13_pytorch(self) -> None:
        requirements = Path("requirements.txt").read_text(encoding="utf-8")

        self.assertNotIn("torch==2.13.0", requirements)
        self.assertNotIn("torchvision==0.28.0", requirements)
        self.assertNotIn("cuda-toolkit==13.0.3.0", requirements)
        self.assertNotIn("triton==3.7.1", requirements)

    def test_bootstrap_creates_runtime_directories_without_overwriting_env(self) -> None:
        script = Path("scripts/setup_vps.sh").read_text(encoding="utf-8")

        self.assertIn('mkdir -p "$ROOT_DIR/logs"', script)
        self.assertIn('if [[ ! -f "$ROOT_DIR/.env" ]]', script)
        self.assertIn('cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"', script)


if __name__ == "__main__":
    unittest.main()
