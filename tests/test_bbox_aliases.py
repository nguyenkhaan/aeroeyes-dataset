import unittest

from src.evaluation.bbox_aliases import build_alias_map, normalize_label, resolve_label, experiment_alias_map


class BoundingBoxAliasTests(unittest.TestCase):
    def test_compound_mapping_is_opt_in_and_only_maps_the_explicit_label(self) -> None:
        base = {"firefighter": "firefighter", "fire_fighter": "firefighter"}
        control = experiment_alias_map(base, "exact")
        candidate = experiment_alias_map(base, "firefighter_compound")
        self.assertEqual(resolve_label("firefighter fire fighter", control), "firefighter_fire_fighter")
        self.assertEqual(resolve_label("firefighter fire fighter", candidate), "firefighter")
        self.assertEqual(resolve_label("firefighter rescue boat", candidate), "firefighter_rescue_boat")
        self.assertNotIn("firefighter_fire_fighter", base)

    def setUp(self) -> None:
        self.classes = {"firefighter": 0, "rescue_boat": 1, "emergency_vehicle": 4}
        self.ontology = {
            "classes": [
                {"id": 0, "name": "firefighter", "automatic_aliases": ["fire fighter"],
                 "review_terms": ["rescue worker"]},
                {"id": 1, "name": "rescue_boat", "automatic_aliases": ["rescue boat", "emergency boat"],
                 "review_terms": ["boat"]},
                {"id": 4, "name": "emergency_vehicle", "automatic_aliases": ["fire truck", "fire engine"],
                 "review_terms": ["emergency car"]},
            ]
        }

    def test_maps_only_explicit_automatic_aliases_to_canonical_labels(self) -> None:
        aliases = build_alias_map(self.ontology, self.classes)

        self.assertEqual(resolve_label("Fire truck", aliases), "emergency_vehicle")
        self.assertEqual(resolve_label("fire-fighter", aliases), "firefighter")
        self.assertEqual(resolve_label("RESCUE BOAT", aliases), "rescue_boat")
        self.assertEqual(resolve_label("rescue worker", aliases), "rescue_worker")
        self.assertNotIn("rescue_worker", aliases)
        self.assertNotIn("boat", aliases)
        self.assertEqual(normalize_label("Emergency vehicle."), "emergency_vehicle")

    def test_rejects_conflicting_aliases_and_class_ids(self) -> None:
        self.ontology["classes"][1]["automatic_aliases"].append("fire truck")
        with self.assertRaisesRegex(ValueError, "conflict"):
            build_alias_map(self.ontology, self.classes)
        self.ontology["classes"][1]["automatic_aliases"].remove("fire truck")
        self.ontology["classes"][0]["id"] = 9
        with self.assertRaisesRegex(ValueError, "class ID"):
            build_alias_map(self.ontology, self.classes)

    def test_rejects_extra_or_missing_classes_and_non_string_aliases(self) -> None:
        with self.assertRaisesRegex(ValueError, "classes"):
            build_alias_map(self.ontology, {"firefighter": 0})
        self.ontology["classes"][0]["automatic_aliases"].append(17)
        with self.assertRaisesRegex(ValueError, "alias"):
            build_alias_map(self.ontology, self.classes)


if __name__ == "__main__":
    unittest.main()
