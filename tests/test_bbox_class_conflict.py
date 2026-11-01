import unittest
from dataclasses import dataclass

from src.evaluation.bbox_class_conflict import suppress_ambulance_conflicts
from src.evaluation.bbox_metrics import Box


@dataclass
class Detection:
    label: str
    box: Box


class CrossClassConflictTests(unittest.TestCase):
    def test_keeps_original_detections_when_disabled(self):
        detections = [Detection("ambulance", Box(0, 0, 4, 4)),
                      Detection("emergency_vehicle", Box(0, 0, 4, 4))]
        result, removed = suppress_ambulance_conflicts(detections, None)
        self.assertEqual(result, detections)
        self.assertEqual(removed, 0)

    def test_only_overlapping_ambulances_are_removed_at_boundary(self):
        emergency = Detection("emergency_vehicle", Box(0, 0, 4, 4))
        ambulance_at = Detection("ambulance", Box(0, 0, 3, 4))
        ambulance_near = Detection("ambulance", Box(0, 0, 2.9, 4))
        person = Detection("firefighter", Box(0, 0, 4, 4))
        detections = [ambulance_at, person, emergency, ambulance_near]
        result, removed = suppress_ambulance_conflicts(detections, .75)
        self.assertEqual(result, [person, emergency, ambulance_near])
        self.assertEqual(removed, 1)
        self.assertEqual(len(detections), 4)

    def test_nearby_distinct_boxes_are_not_removed(self):
        detections = [Detection("ambulance", Box(0, 0, 4, 4)),
                      Detection("emergency_vehicle", Box(3, 0, 7, 4))]
        self.assertEqual(suppress_ambulance_conflicts(detections, .75), (detections, 0))
