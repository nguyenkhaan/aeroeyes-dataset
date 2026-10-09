import unittest
from pathlib import Path


class GenerationEnvironmentTests(unittest.TestCase):
    def test_generation_requirements_leave_pytorch_to_the_cuda_manifest(self) -> None:
        requirements = Path("requirements-generation.txt").read_text(encoding="utf-8")

        self.assertNotIn("torch==", requirements)
        self.assertNotIn("torchvision==", requirements)
        self.assertIn("diffusers==", requirements)
        self.assertIn("transformers==", requirements)

    def test_setup_script_uses_the_generation_requirements(self) -> None:
        script = Path("scripts/setup_generation_env.sh").read_text(encoding="utf-8")

        pytorch_manifest = '"$ROOT_DIR/requirements-pytorch-cu128.txt"'
        generation_manifest = '"$ROOT_DIR/requirements-generation.txt"'
        self.assertIn('set -euo pipefail', script)
        self.assertIn(pytorch_manifest, script)
        self.assertIn(generation_manifest, script)
        self.assertLess(script.index(pytorch_manifest), script.index(generation_manifest))
        self.assertIn('torch.version.cuda', script)
        self.assertIn('pip check', script)

    def test_setup_job_does_not_request_a_gpu(self) -> None:
        script = Path("scripts/setup_generation.slurm").read_text(encoding="utf-8")

        self.assertIn("#SBATCH --account=uit", script)
        self.assertIn("#SBATCH --partition=normal", script)
        self.assertNotIn("#SBATCH --gres", script)
        self.assertIn("bash scripts/setup_generation_env.sh", script)


class GenerationJobTests(unittest.TestCase):
    def test_generation_job_requests_the_school_gpu_partition(self) -> None:
        script = Path("scripts/generate_images.slurm").read_text(encoding="utf-8")

        self.assertIn("#SBATCH --account=uit", script)
        self.assertIn("#SBATCH --partition=normal", script)
        self.assertIn("#SBATCH --gres=gpu:2g.35gb:1", script)
        self.assertIn("#SBATCH --qos=mig2-35", script)

    def test_generation_job_uses_a_small_explicit_target(self) -> None:
        script = Path("scripts/generate_images.slurm").read_text(encoding="utf-8")

        self.assertIn('LIMIT_IMAGES="${1:-5}"', script)
        self.assertIn('export LIMIT_IMAGES', script)
        self.assertIn('"$PYTHON" main.py', script)
        self.assertIn('WATERMARK_REMOVAL_ENABLED="${WATERMARK_REMOVAL_ENABLED:-false}"', script)

    def test_generation_job_checks_inputs_without_printing_the_token(self) -> None:
        script = Path("scripts/generate_images.slurm").read_text(encoding="utf-8")

        self.assertIn('chmod 600 .env', script)
        self.assertIn("HF_TOKEN is missing", script)
        self.assertIn("image_summary.json", script)
        self.assertNotIn("cat .env", script)


if __name__ == "__main__":
    unittest.main()
