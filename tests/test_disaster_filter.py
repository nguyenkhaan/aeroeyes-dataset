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

    def test_accepts_flooded_and_on_fire_labels(self) -> None:
        # Exact labels from Incidents-1M dataset
        self.assertTrue(is_valid_disaster_metadata({"flooded": 1})[0])
        self.assertTrue(is_valid_disaster_metadata({"on fire": 1})[0])
        self.assertTrue(is_valid_disaster_metadata({"on_fire": 1})[0])
        self.assertTrue(is_valid_disaster_metadata({"building collapse": 1})[0])

    def test_rejects_traffic_jam_with_spaces_or_underscores(self) -> None:
        self.assertFalse(is_valid_disaster_metadata({"traffic jam": 1})[0])
        self.assertFalse(is_valid_disaster_metadata({"traffic_jam": 1})[0])
        self.assertFalse(is_valid_disaster_metadata({"car accident": 1})[0])

    def test_accepts_disaster_names_observed_in_download_logs(self) -> None:
        for label in ("tropical cyclone", "snowslide avalanche", "collapsed"):
            with self.subTest(label=label):
                self.assertTrue(is_valid_disaster_metadata({label: 1})[0])

    def test_new_aliases_still_require_positive_labels_and_physical_damage(self) -> None:
        for label in ("tropical cyclone", "snowslide avalanche", "collapsed"):
            with self.subTest(label=label):
                self.assertFalse(is_valid_disaster_metadata({label: 0})[0])
                self.assertFalse(is_valid_disaster_metadata(
                    {label: 1}, {"little_or_no_damage": 1},
                )[0])

    def test_smoke_or_drought_alone_do_not_count_as_whitelisted_disasters(self) -> None:
        for label in ("with smoke", "drought"):
            with self.subTest(label=label):
                self.assertFalse(is_valid_disaster_metadata({label: 1})[0])

    def test_accepts_dataset_names_for_mudslides_and_rockslides(self) -> None:
        for label in ("mudslide mudflow", "rockslide rockfall"):
            with self.subTest(label=label):
                self.assertTrue(is_valid_disaster_metadata({label: 1})[0])


if __name__ == "__main__":
    unittest.main()
