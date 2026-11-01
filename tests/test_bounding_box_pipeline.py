import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.bounding_box_pipeline import (
    BoundingBoxRunConfig,
    load_generation_records,
    run_bounding_box_pipeline,
)
from src.evaluation.change_detection import BoundingBox
from src.evaluation.object_matching import DetectedObject
from src.evaluation.yolo_export import (
    DetectionConversionContext,
    RescueDetector,
    _class_aware_nms,
    _processed_result_to_objects,
)


class BoxValue:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class BoundingBoxPipelineTests(unittest.TestCase):
    def test_load_generation_records_skips_incomplete_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_before = root / "before.png"
            valid_after = root / "after.png"
            valid_before.touch()
            valid_after.touch()

            (root / "valid.json").write_text(
                '{"stem":"valid","reference_path":"%s","output_path":"%s"}'
                % (valid_before.as_posix(), valid_after.as_posix()),
                encoding="utf-8",
            )
            (root / "missing.json").write_text(
                '{"stem":"missing","reference_path":"missing.png",'
                '"output_path":"also-missing.png"}',
                encoding="utf-8",
            )

            records = load_generation_records(root, limit=5)

            self.assertEqual([record["stem"] for record in records], ["valid"])

    def test_load_generation_records_rejects_non_positive_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            load_generation_records(Path("unused"), limit=0)

    def test_pipeline_rejects_fewer_complete_pairs_than_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "Only 0 complete"):
                run_bounding_box_pipeline(
                    BoundingBoxRunConfig(
                        metadata_dir=metadata_dir,
                        output_dir=root / "boxes",
                        limit=1,
                    )
                )


class BoundingBoxSlurmTests(unittest.TestCase):
    def test_job_requests_gpu_and_runs_dedicated_entrypoint(self) -> None:
        script = Path("scripts/detect_bounding_boxes.slurm").read_text(encoding="utf-8")

        self.assertIn("#SBATCH --account=uit", script)
        self.assertIn("#SBATCH --partition=normal", script)
        self.assertIn("#SBATCH --qos=mig2-35", script)
        self.assertIn("#SBATCH --gres=gpu:2g.35gb:1", script)
        self.assertIn('LIMIT_PAIRS="${1:-5}"', script)
        self.assertIn('"$PYTHON" main_bounding_boxes.py', script)
        self.assertIn("test_bounding_box_pipeline.py", script)
        self.assertIn("test_change_detection.py", script)
        self.assertNotIn("cat .env", script)

    def test_detector_does_not_import_heavy_sdqm_embedding_dependencies(self) -> None:
        source = Path("src/evaluation/yolo_export.py").read_text(encoding="utf-8")

        self.assertNotIn("src.evaluation.sdqm_embedding", source)
        self.assertIn("src.evaluation.image_files", source)


class GroundingDinoConversionTests(unittest.TestCase):
    def test_supplement_confidence_does_not_filter_base_detection(self) -> None:
        base = DetectedObject(3, "ambulance", BoundingBox(0, 0, 20, 20, 400, 0), 0.2)
        supplemental = DetectedObject(3, "ambulance", BoundingBox(30, 30, 50, 50, 400, 0), 0.39)
        detector = RescueDetector(text_prompt="base", class_map={"ambulance": 3},
                                  aliases={"ambulance": "ambulance"},
                                  supplemental_prompts=[{"class_name": "ambulance", "prompt": "ambulance .", "min_confidence": 0.4}])
        detector._model = object()
        detector._processor = object()
        with patch.object(detector, "_detect_prompt", side_effect=[[base], [supplemental]]):
            self.assertEqual(detector.detect_objects(object()), [base])
        self.assertEqual(detector.supplemental_prompt_counts["ambulance ."]["after_confidence"], 0)

    def test_supplemental_prompts_merge_after_class_filter_and_nms(self) -> None:
        from src.evaluation.change_detection import BoundingBox

        base = DetectedObject(3, "ambulance", BoundingBox(0, 0, 20, 20, 400, 0), 0.7)
        better = DetectedObject(3, "ambulance", BoundingBox(0, 0, 20, 20, 400, 0), 0.9)
        wrong_class = DetectedObject(0, "firefighter", BoundingBox(30, 30, 50, 50, 400, 0), 0.8)
        detector = RescueDetector(text_prompt="base", class_map={"ambulance": 3, "firefighter": 0},
                                  aliases={"ambulance": "ambulance"},
                                  supplemental_prompts=[{"class_name": "ambulance", "prompt": "ambulance ."}])
        detector._model = object()
        detector._processor = object()
        with patch.object(detector, "_detect_prompt", side_effect=[[base], [better, wrong_class]]) as detect:
            result = detector.detect_objects(object())

        self.assertEqual(result, [better])
        self.assertEqual(detect.call_count, 2)
        self.assertEqual(detector.supplemental_prompt_counts["ambulance ."],
                         {"raw": 2, "target_class": 1, "after_confidence": 1})
    def test_aliases_return_canonical_labels_and_keep_class_thresholds(self) -> None:
        objects = _processed_result_to_objects(
            {
                "boxes": [BoxValue([0, 0, 20, 20]), BoxValue([30, 30, 50, 50]),
                          BoxValue([60, 60, 80, 80])],
                "scores": [0.4, 0.31, 0.4],
                "text_labels": ["fire truck", "emergency boat", "rescue worker"],
            },
            DetectionConversionContext(
                class_map={"emergency_vehicle": 4, "rescue_boat": 1, "firefighter": 0},
                image_width=100,
                image_height=100,
                box_threshold=0.18,
                class_thresholds={"rescue_boat": 0.33},
                aliases={"fire_truck": "emergency_vehicle", "emergency_boat": "rescue_boat"},
            ),
        )

        self.assertEqual([(item.label, item.class_id) for item in objects],
                         [("emergency_vehicle", 4)])

    def test_processed_detection_keeps_class_confidence_and_clips_box(self) -> None:
        objects = _processed_result_to_objects(
            {
                "boxes": [BoxValue([-4.2, 10.1, 105.8, 91.2])],
                "scores": [0.91],
                "text_labels": ["rescue boat"],
            },
            DetectionConversionContext(
                class_map={"rescue_boat": 1},
                image_width=100,
                image_height=100,
                box_threshold=0.25,
                class_thresholds={},
            ),
        )

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].class_id, 1)
        self.assertEqual(objects[0].label, "rescue_boat")
        self.assertAlmostEqual(objects[0].confidence, 0.91)
        self.assertEqual(
            (objects[0].box.x_min, objects[0].box.y_min),
            (0, 10),
        )
        self.assertEqual(
            (objects[0].box.x_max, objects[0].box.y_max),
            (100, 92),
        )

    def test_class_threshold_rejects_low_confidence_rescue_boat_only(self) -> None:
        objects = _processed_result_to_objects(
            {
                "boxes": [BoxValue([0, 0, 20, 20]), BoxValue([30, 30, 50, 50])],
                "scores": [0.31, 0.31],
                "text_labels": ["rescue boat", "firefighter"],
            },
            DetectionConversionContext(
                class_map={"rescue_boat": 1, "firefighter": 0},
                image_width=100,
                image_height=100,
                box_threshold=0.25,
                class_thresholds={"rescue_boat": 0.33},
            ),
        )

        self.assertEqual([item.label for item in objects], ["firefighter"])

    def test_class_aware_nms_keeps_highest_confidence_duplicate(self) -> None:
        box = BoundingBox(10, 10, 50, 50, 1600, 0.0)
        detections = [
            DetectedObject(0, "firefighter", box, 0.40),
            DetectedObject(0, "firefighter", box, 0.80),
            DetectedObject(1, "rescue_boat", box, 0.60),
        ]

        kept = _class_aware_nms(detections, iou_threshold=0.50)

        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0].confidence, 0.80)
        self.assertEqual(kept[1].class_id, 1)


if __name__ == "__main__":
    unittest.main()
