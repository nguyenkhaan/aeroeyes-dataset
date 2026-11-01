import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bbox_clip_replay import filter_conflicts, validate_scores
from src.evaluation.bbox_vehicle_clip import conflict_crops


def detection(label, coords):
    return {"label": label, "confidence": 0.4, "box": dict(zip(
        ("x_min", "y_min", "x_max", "y_max"), coords, strict=True
    ))}


def fixture():
    report = {"stem": "pair-a", "image_width": 100, "image_height": 100,
              "before_detections": [detection("ambulance", (10, 10, 30, 30)),
                                    detection("emergency_vehicle", (10, 10, 30, 30))],
              "after_detections": [detection("ambulance", (10, 10, 30, 30)),
                                   detection("emergency_vehicle", (10, 10, 30, 30)),
                                   detection("ambulance", (60, 60, 80, 80))]}
    scores = []
    for case, margin in zip(conflict_crops(report, 0.75), (-0.1, 0.2), strict=True):
        scores.append({**case, "cosine_similarity": {"ambulance": 0.3 + margin, "fire_truck": 0.3},
                       "ambulance_minus_fire_truck": margin})
    return report, scores


class ClipReplayTest(unittest.TestCase):
    def test_filter_applies_to_both_roles_only_for_conflicts(self):
        report, scores = fixture()
        before = copy.deepcopy(report)
        chosen = validate_scores({"pair-a": report}, scores)
        result = filter_conflicts(report, chosen)
        self.assertEqual([x["label"] for x in result["before_detections"]], ["emergency_vehicle"])
        self.assertEqual(len(result["after_detections"]), 3)
        self.assertEqual(report, before)

    def test_zero_margin_is_removed(self):
        report, scores = fixture()
        scores[1]["ambulance_minus_fire_truck"] = 0.0
        scores[1]["cosine_similarity"] = {"ambulance": 0.3, "fire_truck": 0.3}
        result = filter_conflicts(report, validate_scores({"pair-a": report}, scores))
        self.assertEqual([x["label"] for x in result["after_detections"]], ["emergency_vehicle", "ambulance"])

    def test_missing_duplicate_modified_or_nonfinite_scores_are_rejected(self):
        report, scores = fixture()
        bad_sets = [scores[:1], scores + scores[:1]]
        modified = copy.deepcopy(scores)
        modified[0]["box"][0] += 1
        bad_sets.append(modified)
        nonfinite = copy.deepcopy(scores)
        nonfinite[0]["ambulance_minus_fire_truck"] = float("nan")
        bad_sets.append(nonfinite)
        for bad in bad_sets:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_scores({"pair-a": report}, bad)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                             ("numpy", "torch", "transformers", "yaml", "PIL")),
                         "Pipeline dependencies are required for the replay integration test")
    def test_replay_recomputes_matching_when_before_detection_is_removed(self):
        from PIL import Image, ImageDraw
        from scripts.replay_bbox_vehicle_clip import objects, replay_report
        from src.evaluation.bounding_box_pipeline import _matching_record
        from src.evaluation.change_detection import detect_change_delta
        from src.evaluation.object_matching import match_and_refine_objects

        report, scores = fixture()
        for role in ("before", "after"):
            for item in report[f"{role}_detections"]:
                item["class_id"] = 1 if item["label"] == "ambulance" else 2
                box = item["box"]
                box["area"] = (box["x_max"] - box["x_min"]) * (box["y_max"] - box["y_min"])
                box["mean_delta"] = 0.0
        before = Image.new("RGB", (100, 100), "black")
        after = before.copy()
        ImageDraw.Draw(after).rectangle((0, 0, 35, 35), fill="white")
        delta = detect_change_delta(before, after)
        matching = match_and_refine_objects(objects(report["before_detections"]), objects(report["after_detections"]), delta)
        report["matching"] = _matching_record(matching)
        report["bbox_passed"] = delta.is_valid_change and matching.is_consistent_rescue
        report["delta"] = {"mean": round(delta.mean_delta, 6), "area_ratio": round(delta.delta_area_ratio, 6),
                           "is_valid_change": delta.is_valid_change}
        filtered = filter_conflicts(report, validate_scores({"pair-a": report}, scores))
        with tempfile.TemporaryDirectory() as folder:
            paths = {role: Path(folder) / f"{role}.png" for role in ("before", "after")}
            before.save(paths["before"])
            after.save(paths["after"])
            result = replay_report(report, filtered, paths)
        self.assertEqual(len(report["matching"]["preserved_objects"]), 2)
        self.assertEqual(len(result["matching"]["preserved_objects"]), 1)
        self.assertTrue(any(x["label"] == "ambulance" for x in result["matching"]["refined_rescue_objects"]))


if __name__ == "__main__":
    unittest.main()
