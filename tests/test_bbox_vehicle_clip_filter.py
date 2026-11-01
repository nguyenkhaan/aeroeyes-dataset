import unittest
from dataclasses import dataclass
import json
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.evaluation.bbox_experiment import ExperimentConfig
from src.evaluation.bbox_metrics import Box
from src.evaluation.bbox_vehicle_clip_filter import VehicleConflictClipFilter, conflict_indices, filter_scored_conflicts


@dataclass
class Detection:
    label: str
    box: Box
    confidence: float = 0.4


class VehicleClipFilterTests(unittest.TestCase):
    def test_config_is_opt_in_and_rejects_conflicting_policies(self):
        data = json.loads(Path("config/bounding_box/dino-ambulance-min030.json").read_text())
        self.assertFalse(ExperimentConfig.from_dict(data).vehicle_clip_filter)
        data["vehicle_clip_filter"] = True
        self.assertTrue(ExperimentConfig.from_dict(data).vehicle_clip_filter)
        data["cross_class_iou_threshold"] = 0.75
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict(data)
        data["cross_class_iou_threshold"] = None
        data["vehicle_clip_filter"] = "true"
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict(data)

    def test_candidate_config_changes_only_opt_in_policy(self):
        control = ExperimentConfig.from_dict(json.loads(
            Path("config/bounding_box/dino-ambulance-min030.json").read_text())).as_dict()
        candidate = ExperimentConfig.from_dict(json.loads(
            Path("config/bounding_box/dino-ambulance-clip.json").read_text())).as_dict()
        different = {key for key in control if control[key] != candidate[key]}
        self.assertEqual(different, {"name", "vehicle_clip_filter"})

    def test_filters_only_conflicts_with_nonpositive_margin(self):
        vehicle = Detection("emergency_vehicle", Box(0, 0, 4, 4))
        keep = Detection("ambulance", Box(0, 0, 4, 4))
        reject = Detection("ambulance", Box(0, 0, 3, 4))
        unrelated = Detection("ambulance", Box(10, 10, 12, 12))
        detections = [vehicle, keep, reject, unrelated]
        indices = conflict_indices(detections)
        self.assertEqual(indices, [1, 2])
        filtered, decisions = filter_scored_conflicts(detections, indices, [(0.4, 0.3), (0.3, 0.3)])
        self.assertEqual(filtered, [vehicle, keep, unrelated])
        self.assertEqual([d["suppressed"] for d in decisions], [False, True])
        self.assertEqual(len(detections), 4)

    def test_rejects_missing_duplicate_and_nonfinite_scores(self):
        detections = [Detection("ambulance", Box(0, 0, 4, 4)),
                      Detection("emergency_vehicle", Box(0, 0, 4, 4))]
        for indices, values in [([0], []), ([0, 0], [(0.1, 0.2)] * 2), ([0], [(float("nan"), 0.2)])]:
            with self.subTest(indices=indices, values=values), self.assertRaises(ValueError):
                filter_scored_conflicts(detections, indices, values)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required")
    def test_clip_application_crops_and_passes_model_scores_to_rule(self):
        from PIL import Image

        class Tensor:
            def float(self): return self
            def exp(self): return self
            def __truediv__(self, other): return self
            def cpu(self): return self
            def tolist(self): return [[0.2, 0.3]]
        class Inputs(dict):
            def to(self, device): return self
        class Model:
            logit_scale = Tensor()
            def __call__(self, **kwargs): return SimpleNamespace(logits_per_image=Tensor())
        class Processor:
            def __call__(self, **kwargs):
                self.crop_sizes = [image.size for image in kwargs["images"]]
                self.prompts = kwargs["text"]
                return Inputs()
        clip_filter = VehicleConflictClipFilter.__new__(VehicleConflictClipFilter)
        clip_filter.device = "cpu"
        clip_filter.model = Model()
        clip_filter.processor = Processor()
        clip_filter.torch = SimpleNamespace(inference_mode=nullcontext)
        detections = [Detection("ambulance", Box(0, 0, 4, 4)),
                      Detection("emergency_vehicle", Box(0, 0, 4, 4))]
        retained, decisions = clip_filter.apply(Image.new("RGB", (10, 10)), detections)
        self.assertEqual(retained, detections[1:])
        self.assertTrue(decisions[0]["suppressed"])
        self.assertEqual(clip_filter.processor.crop_sizes, [(5, 5)])
        self.assertEqual(clip_filter.processor.prompts, ["a photo of an ambulance", "a photo of a fire truck"])

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                             ("numpy", "torch", "transformers", "yaml", "PIL")),
                         "Detector dependencies are required for integration")
    def test_detector_calls_opt_in_filter_and_records_decisions(self):
        from PIL import Image
        from src.evaluation.yolo_export import RescueDetector

        item = Detection("ambulance", Box(0, 0, 4, 4))
        item.class_id = 1
        class FakeFilter:
            def apply(self, image, detections):
                return [], [{"suppressed": True}]

        detector = RescueDetector(text_prompt="ambulance .", vehicle_clip_filter=FakeFilter())
        detector._model, detector._processor = object(), object()
        with patch.object(detector, "_detect_prompt", return_value=[item]):
            self.assertEqual(detector.detect_objects(Image.new("RGB", (10, 10))), [])
        self.assertEqual(detector.vehicle_clip_counts["ambulance_suppressed"], 1)
        self.assertEqual(detector.last_clip_decisions, [{"suppressed": True}])


if __name__ == "__main__":
    unittest.main()
