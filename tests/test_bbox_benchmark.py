import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.evaluation.bbox_benchmark import (
    BenchmarkSelectionConfig,
    build_benchmark_manifest,
    load_selection_config,
    load_generation_pairs,
)


class BoundingBoxBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.metadata_dir = self.root / "metadata"
        self.metadata_dir.mkdir()

    def write_pair(
        self,
        index: int,
        label: str,
        source_group: str | None = None,
    ) -> None:
        stem = f"stem-{index:03d}"
        before_path = self.root / f"before-{index:03d}.png"
        after_path = self.root / f"after-{index:03d}.png"
        before_path.write_bytes(f"before-{index}".encode())
        after_path.write_bytes(f"after-{index}".encode())
        record = {
            "stem": stem,
            "image_key": f"{label}/image-{index:03d}.jpg",
            "labels": [label],
            "url": f"https://example.test/{label}/{index}",
            "reference_path": str(before_path),
            "output_path": str(after_path),
        }
        if source_group is not None:
            record["source_group"] = source_group
        (self.metadata_dir / f"{stem}.json").write_text(
            json.dumps(record),
            encoding="utf-8",
        )

    def test_selection_is_deterministic_and_splits_are_disjoint(self) -> None:
        labels = ["flood", "wildfire", "earthquake", "landslide"]
        for index in range(24):
            self.write_pair(index, labels[index % len(labels)])

        records = load_generation_pairs(self.metadata_dir)
        config = BenchmarkSelectionConfig(
            validation_size=8,
            test_size=12,
            seed=20261020,
        )

        first = build_benchmark_manifest(records, config)
        second = build_benchmark_manifest(records, config)

        self.assertEqual(first, second)
        validation_ids = {
            item["pair_id"] for item in first["pairs"]
            if item["split"] == "validation"
        }
        test_ids = {
            item["pair_id"] for item in first["pairs"]
            if item["split"] == "test"
        }
        self.assertEqual(len(validation_ids), 8)
        self.assertEqual(len(test_ids), 12)
        self.assertTrue(validation_ids.isdisjoint(test_ids))

    def test_source_group_is_never_split_between_validation_and_test(self) -> None:
        for index in range(12):
            self.write_pair(
                index,
                "flood" if index < 6 else "wildfire",
                source_group=f"event-{index // 2}",
            )

        records = load_generation_pairs(self.metadata_dir)
        manifest = build_benchmark_manifest(
            records,
            BenchmarkSelectionConfig(
                validation_size=4,
                test_size=6,
                seed=7,
            ),
        )

        split_by_group: dict[str, set[str]] = {}
        for item in manifest["pairs"]:
            split_by_group.setdefault(item["source_group"], set()).add(item["split"])

        self.assertTrue(all(len(splits) == 1 for splits in split_by_group.values()))
        self.assertEqual(manifest["counts"]["validation"], 4)
        self.assertEqual(manifest["counts"]["test"], 6)

    def test_manifest_records_input_checksums_and_sampling_metadata(self) -> None:
        for index in range(6):
            self.write_pair(index, "flood")

        records = load_generation_pairs(self.metadata_dir)
        manifest = build_benchmark_manifest(
            records,
            BenchmarkSelectionConfig(validation_size=2, test_size=3, seed=11),
        )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["selection"]["seed"], 11)
        self.assertEqual(manifest["selection"]["strategy"], "grouped_stratified")
        for item in manifest["pairs"]:
            self.assertRegex(item["before_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(item["after_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn(item["source_group_method"], {"explicit", "pair_id_fallback"})

    def test_selection_preserves_label_proportions(self) -> None:
        for index in range(20):
            self.write_pair(index, "flood")
        for index in range(20, 30):
            self.write_pair(index, "wildfire")

        manifest = build_benchmark_manifest(
            load_generation_pairs(self.metadata_dir),
            BenchmarkSelectionConfig(validation_size=6, test_size=9, seed=5),
        )

        selected_counts = Counter(
            label
            for item in manifest["pairs"]
            for label in item["labels"]
        )
        self.assertEqual(selected_counts, Counter({"flood": 10, "wildfire": 5}))

    def test_loader_rejects_corrupt_or_incomplete_metadata(self) -> None:
        (self.metadata_dir / "corrupt.json").write_text("", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid generation metadata"):
            load_generation_pairs(self.metadata_dir)

        (self.metadata_dir / "corrupt.json").unlink()
        (self.metadata_dir / "incomplete.json").write_text(
            json.dumps({"stem": "incomplete"}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            load_generation_pairs(self.metadata_dir)

    def test_selection_rejects_insufficient_pairs(self) -> None:
        for index in range(4):
            self.write_pair(index, "flood")

        with self.assertRaisesRegex(ValueError, "requires 5 pairs"):
            build_benchmark_manifest(
                load_generation_pairs(self.metadata_dir),
                BenchmarkSelectionConfig(validation_size=2, test_size=3, seed=1),
            )

    def test_selection_config_is_loaded_from_protocol_yaml(self) -> None:
        protocol_path = self.root / "benchmark.yaml"
        protocol_path.write_text(
            "selection:\n"
            "  seed: 19\n"
            "  validation_pairs: 3\n"
            "  test_pairs: 7\n",
            encoding="utf-8",
        )

        config = load_selection_config(protocol_path)

        self.assertEqual(config.seed, 19)
        self.assertEqual(config.validation_size, 3)
        self.assertEqual(config.test_size, 7)

    def test_selection_config_rejects_missing_protocol_fields(self) -> None:
        protocol_path = self.root / "benchmark.yaml"
        protocol_path.write_text("selection:\n  seed: 19\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid benchmark protocol"):
            load_selection_config(protocol_path)


if __name__ == "__main__":
    unittest.main()
