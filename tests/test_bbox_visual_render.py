import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.evaluation.bbox_strict_v2 import adapt_strict_manifest
from tests.test_bbox_metrics import prediction, prediction_report
from tests.test_bbox_strict_v2 import sample


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required to render images")
class VisualRenderTests(unittest.TestCase):
    def test_generates_verified_four_panel_png_and_html(self):
        from PIL import Image

        strict, benchmark = sample()
        after = strict["pairs"][0]["after"]
        after["confirmed_rescue_objects"][0]["label"] = "ambulance"
        gt = adapt_strict_manifest(strict, benchmark)
        gt["source_manifest_sha256"] = "benchmark"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ground_truth = root / "gt.json"
            ground_truth.write_text(json.dumps(gt), encoding="utf-8")
            before, image_after = root / "before.png", root / "after.png"
            Image.new("RGB", (128, 128), "white").save(before)
            Image.new("RGB", (128, 128), "white").save(image_after)
            images = [{"stem": "abcd", "reference_path": str(before), "output_path": str(image_after),
                       "before_sha256": hashlib.sha256(before.read_bytes()).hexdigest(),
                       "after_sha256": hashlib.sha256(image_after.read_bytes()).hexdigest()}]
            paths = []
            for label, threshold in (("base", None), ("candidate", .75)):
                run = root / label
                (run / "reports").mkdir(parents=True)
                manifest = {"status": "completed", "split": "validation", "selected_pair_ids": ["abcd"],
                            "selected_images": images, "benchmark_manifest_sha256": "benchmark",
                            "git": {"commit": "same", "tracked_files_dirty": False},
                            "config": {"name": label, "cross_class_iou_threshold": threshold},
                            "result": {"processed_pairs": 1, "model_revision": "model", "ontology_sha256": "ontology", "class_map_sha256": "map"}}
                (run / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                objects = [prediction("ambulance", [1, 2, 10, 20])] if label == "base" else []
                (run / "reports" / "abcd.json").write_text(json.dumps(prediction_report("abcd", objects)), encoding="utf-8")
                metrics = root / f"{label}-metrics.json"
                command = [sys.executable, "scripts/evaluate_bbox_strict_v2.py", "--ground-truth", str(ground_truth),
                           "--run-dir", str(run), "--output", str(metrics)]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                paths.extend((run, metrics))
            output = root / "review"
            command = [sys.executable, "scripts/render_bbox_conflict_review.py", "--ground-truth", str(ground_truth),
                       "--baseline-run", str(paths[0]), "--baseline-metrics", str(paths[1]),
                       "--candidate-run", str(paths[2]), "--candidate-metrics", str(paths[3]),
                       "--output-dir", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            index = json.loads((output / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["cases"][0]["lost_ambulance_gt_ids"], ["after-0"])
            with Image.open(output / "pair-0000.png") as figure:
                self.assertEqual(figure.size, (512, 220))
            self.assertTrue((output / "index.html").is_file())
            with zipfile.ZipFile(output / "review-bundle.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"index.html", "index.json", "pair-0000.png"})
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
