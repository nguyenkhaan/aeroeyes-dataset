import unittest

from src.vision.disaster_filter import is_valid_disaster_metadata


class DisasterFilterTests(unittest.TestCase):
    def test_rejects_empty_incidents(self) -> None:
        is_valid, reason = is_valid_disaster_metadata({})
        self.assertFalse(is_valid)
        self.assertIn("Missing", reason)

    def test_rejects_non_disaster_classes(self) -> None:
        incidents = {"traffic_jam": 1, "blocked_road": 1}
        is_valid, reason = is_valid_disaster_metadata(incidents)
        self.assertFalse(is_valid)
        self.assertIn("Non-disaster incident only", reason)

    def test_accepts_valid_natural_disaster(self) -> None:
        incidents = {"flood": 1}
        is_valid, reason = is_valid_disaster_metadata(incidents)
        self.assertTrue(is_valid)
        self.assertIn("flood", reason)

    def test_accepts_wildfire_disaster(self) -> None:
        incidents = {"wildfire": 1}
        is_valid, reason = is_valid_disaster_metadata(incidents)
        self.assertTrue(is_valid)
        self.assertIn("wildfire", reason)

    def test_rejects_when_metadata_indicates_no_damage(self) -> None:
        incidents = {"flood": 1}
        damage = {"little_or_no_damage": 1, "severe_damage": 0, "mild_damage": 0}
        is_valid, reason = is_valid_disaster_metadata(incidents, damage)
        self.assertFalse(is_valid)
        self.assertIn("little or no physical damage", reason)

    def test_accepts_when_severe_damage_is_present(self) -> None:
        incidents = {"earthquake": 1}
        damage = {"severe_damage": 1}
        is_valid, reason = is_valid_disaster_metadata(incidents, damage)
        self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
