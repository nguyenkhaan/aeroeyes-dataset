import unittest
import numpy as np
from PIL import Image

from src.evaluation.change_detection import (
    BoundingBox,
    compute_otsu_threshold,
    detect_change_delta,
)
from src.evaluation.object_matching import (
    DetectedObject,
    match_and_refine_objects,
)


class ChangeDetectionTests(unittest.TestCase):
    def test_bounding_box_iou(self) -> None:
        box1 = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, area=100, mean_delta=50.0)
        box2 = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, area=100, mean_delta=50.0)
        self.assertAlmostEqual(box1.iou(box2), 1.0)

        box3 = BoundingBox(x_min=10, y_min=10, x_max=20, y_max=20, area=100, mean_delta=50.0)
        self.assertAlmostEqual(box1.iou(box3), 0.0)

    def test_bounding_box_to_yolo(self) -> None:
        box = BoundingBox(x_min=20, y_min=20, x_max=40, y_max=40, area=400, mean_delta=50.0)
        yolo_str = box.to_yolo(image_width=100, image_height=100, class_id=1)
        parts = yolo_str.strip().split()
        self.assertEqual(parts[0], "1")
        self.assertAlmostEqual(float(parts[1]), 0.3, places=4)
        self.assertAlmostEqual(float(parts[2]), 0.3, places=4)
        self.assertAlmostEqual(float(parts[3]), 0.2, places=4)
        self.assertAlmostEqual(float(parts[4]), 0.2, places=4)

    def test_compute_otsu_threshold(self) -> None:
        arr = np.zeros((50, 50), dtype=np.float32)
        arr[20:40, 20:40] = 200.0
        thresh = compute_otsu_threshold(arr)
        self.assertTrue(50.0 < thresh < 200.0)

    def test_detect_change_delta_finds_edited_region(self) -> None:
        width, height = 200, 200
        before_arr = np.full((height, width, 3), 100, dtype=np.uint8)
        before_img = Image.fromarray(before_arr)

        after_arr = before_arr.copy()
        after_arr[75:125, 75:125, :] = 240
        after_img = Image.fromarray(after_arr)

        result = detect_change_delta(
            before_img,
            after_img,
            intensity_threshold=30.0,
            min_area_ratio=0.01,
            max_area_ratio=0.60,
        )

        self.assertGreater(result.mean_delta, 50.0)
        self.assertGreater(result.delta_area_ratio, 0.01)
        self.assertTrue(result.is_valid_change)
        self.assertGreaterEqual(len(result.bounding_boxes), 1)

        top_box = result.bounding_boxes[0]
        self.assertLessEqual(top_box.x_min, 80)
        self.assertGreaterEqual(top_box.x_max, 120)

    def test_object_matching_identifies_added_rescue_and_preserved_context(self) -> None:
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[80:120, 80:120] = 255
        delta_res = detect_change_delta(
            Image.new("RGB", (200, 200), (100, 100, 100)),
            Image.new("RGB", (200, 200), (100, 100, 100)),
        )
        delta_res.delta_mask = mask
        delta_res.delta_area_ratio = 0.05

        building_box = BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50, area=1600, mean_delta=0.0)
        before_obj = DetectedObject(class_id=0, label="building", box=building_box, confidence=0.9)
        after_building = DetectedObject(class_id=0, label="building", box=building_box, confidence=0.9)

        rescue_box = BoundingBox(x_min=80, y_min=80, x_max=120, y_max=120, area=1600, mean_delta=100.0)
        after_boat = DetectedObject(class_id=1, label="rescue boat", box=rescue_box, confidence=0.88)

        fake_box = BoundingBox(x_min=150, y_min=150, x_max=180, y_max=180, area=900, mean_delta=0.0)
        after_fake = DetectedObject(class_id=2, label="firefighter", box=fake_box, confidence=0.7)

        report = match_and_refine_objects(
            before_objects=[before_obj],
            after_objects=[after_building, after_boat, after_fake],
            delta_result=delta_res,
            min_delta_overlap=0.20,
        )

        self.assertEqual(len(report.preserved_objects), 1)
        self.assertEqual(report.preserved_objects[0][0].label, "building")
        self.assertEqual(len(report.added_rescue_objects), 1)
        self.assertEqual(report.added_rescue_objects[0][0].label, "rescue boat")
        self.assertEqual(len(report.rejected_detections), 1)
        self.assertEqual(report.rejected_detections[0][0].label, "firefighter")


if __name__ == "__main__":
    unittest.main()
