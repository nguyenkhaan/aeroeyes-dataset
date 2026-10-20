import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.evaluation.bbox_annotation import AnnotationBundleConfig, create_bundle


class BoundingBoxAnnotationBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.images_dir = self.root / "source"
        self.images_dir.mkdir()

    def manifest_pair(self, index: int, split: str) -> dict:
        before_path = self.images_dir / f"before-{index}.jpg"
        after_path = self.images_dir / f"after-{index}.jpg"
        before_path.write_bytes(f"before-{index}".encode())
        after_path.write_bytes(f"after-{index}".encode())
        return {
            "pair_id": f"pair-{index}",
            "split": split,
            "image_key": f"flood/image-{index}.jpg",
            "labels": ["flooded"],
            "url": f"https://example.test/{index}",
            "source_group": f"pair:pair-{index}",
            "source_group_method": "pair_id_fallback",
            "before_path": str(before_path),
            "after_path": str(after_path),
            "before_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
            "after_sha256": hashlib.sha256(after_path.read_bytes()).hexdigest(),
            "sampling_reason": "representative_stratified_sample",
        }

    def manifest(self) -> dict:
        pairs = [
            self.manifest_pair(0, "validation"),
            self.manifest_pair(1, "validation"),
            self.manifest_pair(2, "test"),
            self.manifest_pair(3, "test"),
        ]
        return {
            "schema_version": 1,
            "selection": {"seed": 7},
            "counts": {"validation": 2, "test": 2, "total": 4},
            "pairs": pairs,
        }

    def test_bundle_keeps_before_and_after_frames_adjacent(self) -> None:
        output_dir = self.root / "bundle"

        summary = create_bundle(
            self.manifest(),
            output_dir,
            AnnotationBundleConfig(double_annotation_fraction=0.5, seed=11),
        )

        with zipfile.ZipFile(output_dir / "validation-images.zip") as archive:
            names = archive.namelist()
        self.assertEqual(
            names,
            [
                "0000__00_before__pair-0.jpg",
                "0000__01_after__pair-0.jpg",
                "0001__00_before__pair-1.jpg",
                "0001__01_after__pair-1.jpg",
            ],
        )
        self.assertEqual(summary["counts"]["pairs"], 4)
        self.assertEqual(summary["counts"]["frames"], 8)
        self.assertEqual(len(summary["artifact_sha256"]), 7)
        self.assertTrue((output_dir / "bundle-summary.json").is_file())

    def test_bundle_selects_exact_double_annotation_fraction_per_split(self) -> None:
        output_dir = self.root / "bundle"

        summary = create_bundle(
            self.manifest(),
            output_dir,
            AnnotationBundleConfig(double_annotation_fraction=0.5, seed=11),
        )

        self.assertEqual(summary["double_annotation_counts"]["validation"], 1)
        self.assertEqual(summary["double_annotation_counts"]["test"], 1)
        mapping = json.loads(
            (output_dir / "annotation-mapping.json").read_text(encoding="utf-8")
        )
        selected = [pair for pair in mapping["pairs"] if pair["double_annotation"]]
        self.assertEqual(len(selected), 2)
        second_output_dir = self.root / "second-bundle"
        create_bundle(
            self.manifest(),
            second_output_dir,
            AnnotationBundleConfig(double_annotation_fraction=0.5, seed=11),
        )
        second_mapping = json.loads(
            (second_output_dir / "annotation-mapping.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(mapping, second_mapping)
        with zipfile.ZipFile(
            output_dir / "validation-double-annotation-images.zip"
        ) as archive:
            self.assertEqual(len(archive.namelist()), 2)

    def test_bundle_writes_ground_truth_template_and_cvat_labels(self) -> None:
        output_dir = self.root / "bundle"

        create_bundle(
            self.manifest(),
            output_dir,
            AnnotationBundleConfig(double_annotation_fraction=0.5, seed=11),
        )

        template = json.loads(
            (output_dir / "bbox-gt-v1.template.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(template["pairs"]), 4)
        self.assertIsNone(template["pairs"][0]["generation_status"])
        self.assertEqual(template["pairs"][0]["before_objects"], [])
        self.assertEqual(template["pairs"][0]["after_objects"], [])
        labels = json.loads(
            (output_dir / "cvat-labels.json").read_text(encoding="utf-8")
        )
        object_labels = [label for label in labels if label["type"] == "rectangle"]
        tag_labels = [label for label in labels if label["type"] == "tag"]
        self.assertEqual(len(object_labels), 5)
        self.assertEqual([label["name"] for label in tag_labels], ["pair_status"])

    def test_bundle_rejects_an_image_that_changed_after_sampling(self) -> None:
        manifest = self.manifest()
        Path(manifest["pairs"][0]["before_path"]).write_bytes(b"modified")

        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            create_bundle(
                manifest,
                self.root / "bundle",
                AnnotationBundleConfig(double_annotation_fraction=0.5, seed=11),
            )


if __name__ == "__main__":
    unittest.main()
