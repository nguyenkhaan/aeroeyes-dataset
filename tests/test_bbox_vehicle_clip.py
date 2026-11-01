import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bbox_vehicle_clip import conflict_crops, padded_crop_box
from scripts.score_bbox_vehicle_conflicts import load_run


def detection(label, coords):
    return {"label": label, "confidence": 0.4, "box": dict(zip(
        ("x_min", "y_min", "x_max", "y_max"), coords, strict=True
    ))}


class VehicleClipTest(unittest.TestCase):
    def test_selects_only_ambulance_with_overlapping_vehicle_and_preserves_index(self):
        report = {"stem": "pair-a", "image_width": 100, "image_height": 100,
                  "before_detections": [], "after_detections": [
                      detection("ambulance", (10, 10, 30, 30)),
                      detection("emergency_vehicle", (10, 10, 30, 30)),
                      detection("ambulance", (60, 60, 80, 80)),
                  ]}
        crops = conflict_crops(report, 0.75)
        self.assertEqual(len(crops), 1)
        self.assertEqual((crops[0]["role"], crops[0]["detection_index"]), ("after", 0))
        self.assertEqual(crops[0]["overlap_vehicle_indices"], [1])

    def test_crop_context_is_clamped_to_image(self):
        self.assertEqual(padded_crop_box([0, 2, 20, 22], (100, 100), 0.1), (0, 0, 22, 24))

    def test_invalid_boxes_and_threshold_are_rejected(self):
        report = {"stem": "a", "image_width": 100, "image_height": 100,
                  "before_detections": [], "after_detections": [detection("ambulance", (4, 4, 4, 9))]}
        with self.assertRaises(ValueError):
            conflict_crops(report, 0.75)
        with self.assertRaises(ValueError):
            conflict_crops(report, float("nan"))

    def test_frozen_report_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reports = root / "reports"
            reports.mkdir()
            report = reports / "a.json"
            report.write_text(json.dumps({"stem": "a"}), encoding="utf-8")
            manifest = {"status": "completed", "split": "validation", "config": {},
                        "selected_pair_ids": ["a"], "selected_images": [{"stem": "a"}],
                        "result": {"processed_pairs": 1}}
            manifest_path = root / "run-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            sha = lambda data: hashlib.sha256(data).hexdigest()
            metrics = {"provenance": {"run_manifest_sha256": sha(manifest_path.read_bytes()),
                                      "prediction_report_sha256": {"a": sha(report.read_bytes())}}}
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            self.assertEqual(set(load_run(root, metrics_path)[1]), {"a"})
            report.write_text(json.dumps({"stem": "a", "changed": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reports differ"):
                load_run(root, metrics_path)


if __name__ == "__main__":
    unittest.main()
