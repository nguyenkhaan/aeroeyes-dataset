import unittest
from dataclasses import dataclass

from src.evaluation.bbox_prompt_supplement import detect_with_supplemental_prompts, validate_prompt_classes


@dataclass
class Detection:
    label: str
    confidence: float
    box: str


class SupplementalPromptTests(unittest.TestCase):
    def test_prompt_must_resolve_to_its_declared_class(self):
        aliases = {"ambulance": "ambulance", "fire_truck": "emergency_vehicle"}
        validate_prompt_classes([{"class_name": "ambulance", "prompt": "ambulance ."}], aliases)
        for prompt in ("boat .", "ambulance . boat .", "fire truck ."):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                validate_prompt_classes([{"class_name": "ambulance", "prompt": prompt}], aliases)
    def test_supplements_keep_base_filter_classes_and_apply_combined_nms(self):
        calls = []
        def detect(prompt):
            calls.append(prompt)
            if prompt == "base":
                return [Detection("firefighter", .9, "a"), Detection("ambulance", .7, "b")]
            return [Detection("ambulance", .8, "b"), Detection("firefighter", .9, "c")]
        def nms(items):
            chosen = {}
            for item in sorted(items, key=lambda x: x.confidence, reverse=True):
                chosen.setdefault((item.label, item.box), item)
            return list(chosen.values())
        prompts = [{"class_name": "ambulance", "prompt": "ambulance ."}]
        result, counts = detect_with_supplemental_prompts("base", prompts, detect, nms)
        self.assertEqual(calls, ["base", "ambulance ."])
        self.assertEqual([(x.label, x.box) for x in result], [("firefighter", "a"), ("ambulance", "b")])
        self.assertEqual(counts, {"ambulance .": {"raw": 2, "target_class": 1, "after_confidence": 1}})

    def test_empty_supplements_preserve_baseline(self):
        result, counts = detect_with_supplemental_prompts(
            "base", [], lambda prompt: [Detection("firefighter", .8, "a")], lambda x: x)
        self.assertEqual(len(result), 1)
        self.assertEqual(counts, {})

    def test_threshold_filters_only_supplemental_class_before_nms(self):
        base = Detection("ambulance", .20, "base")
        low = Detection("ambulance", .39, "low")
        at_threshold = Detection("ambulance", .40, "threshold")
        wrong_class = Detection("firefighter", .95, "wrong")
        def detect(prompt):
            return [base] if prompt == "base" else [low, at_threshold, wrong_class]
        prompts = [{"class_name": "ambulance", "prompt": "ambulance .", "min_confidence": .40}]
        result, counts = detect_with_supplemental_prompts("base", prompts, detect, lambda x: x)
        self.assertEqual(result, [base, at_threshold])
        self.assertEqual(counts["ambulance ."]["after_confidence"], 1)
