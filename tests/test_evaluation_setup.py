import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation import cmmd
from src.evaluation.sdqm_vinfo import validate_vinfo_dataset


class CmmdSetupTests(unittest.TestCase):
    def test_missing_cmmd_message_uses_canonical_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            cmmd,
            "CMMD_REPO_DIR",
            str(Path(directory) / "missing"),
        ), self.assertRaisesRegex(FileNotFoundError, "cmmd-pytorch"):
            cmmd._load_compute_cmmd()


class VinfoSetupTests(unittest.TestCase):
    def test_accepts_supported_vinfo_dataset(self) -> None:
        self.assertEqual(validate_vinfo_dataset("rareplanes"), "rareplanes")

    def test_rejects_rescue_vinfo_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "rescue"):
            validate_vinfo_dataset("rescue")


if __name__ == "__main__":
    unittest.main()
