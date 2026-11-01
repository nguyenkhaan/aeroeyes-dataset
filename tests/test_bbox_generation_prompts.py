import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_bbox_generation_prompts import export_prompts


class GenerationPromptsTest(unittest.TestCase):
    def test_export_maps_frames_and_preserves_missing_instruction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pair_id = "a" * 64
            manifest = {"pairs": [{"pair_id": pair_id, "split": "test"}]}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            mapping = {"source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       "pairs": [{"pair_id": pair_id, "split": "test", "frames": [
                           {"role": "before", "bundle_name": "0000__before.png"},
                           {"role": "after", "bundle_name": "0000__after.png"}]}]}
            mapping_path = root / "mapping.json"
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
            (root / f"{pair_id}.json").write_text(json.dumps({"stem": pair_id, "flux_prompt": "full"}), encoding="utf-8")
            output = root / "prompts.json"
            export_prompts(manifest_path, mapping_path, root, output, "test")
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result[0]["cvat_files"], ["0000__before.png", "0000__after.png"])
            self.assertIsNone(result[0]["editing_instruction"])
            self.assertEqual(result[0]["flux_prompt"], "full")
            with self.assertRaises(FileExistsError):
                export_prompts(manifest_path, mapping_path, root, output, "test")
            (root / f"{pair_id}.json").unlink()
            missing_output = root / "missing.json"
            with self.assertRaises(FileNotFoundError):
                export_prompts(manifest_path, mapping_path, root, missing_output, "test")
            self.assertFalse(missing_output.exists())

    def test_wrong_mapping_hash_fails_without_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            mapping = root / "mapping.json"
            output = root / "prompts.json"
            manifest.write_text('{"pairs": []}', encoding="utf-8")
            mapping.write_text('{"source_manifest_sha256": "wrong", "pairs": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                export_prompts(manifest, mapping, root, output, "test")
            self.assertFalse(output.exists())
