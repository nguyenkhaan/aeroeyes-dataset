import tempfile
import unittest
import zipfile
from pathlib import Path

from src.evaluation.cvat_bbox import ConversionConfig, CvatGroundTruthConverter


PAIR_IDS = ("pair-0", "pair-1")


def ground_truth_template() -> dict:
    return {
        "schema_version": 1,
        "artifact_version": "bbox-gt-v1",
        "source_manifest_sha256": "manifest-digest",
        "annotation_status": "template",
        "pairs": [
            {
                "pair_id": pair_id,
                "split": "validation",
                "generation_status": None,
                "before_objects": [],
                "after_objects": [],
                "pair_notes": "",
            }
            for pair_id in PAIR_IDS
        ],
    }


def annotation_mapping() -> dict:
    pairs = []
    for index, pair_id in enumerate(PAIR_IDS):
        prefix = f"{index:04d}"
        pairs.append(
            {
                "pair_id": pair_id,
                "split": "validation",
                "frames": [
                    {
                        "role": "before",
                        "bundle_name": f"{prefix}__00_before__{pair_id}.png",
                    },
                    {
                        "role": "after",
                        "bundle_name": f"{prefix}__01_after__{pair_id}.png",
                    },
                ],
            }
        )
    return {
        "schema_version": 1,
        "source_manifest_sha256": "manifest-digest",
        "pairs": pairs,
    }


def image_xml(
    frame_id: int,
    name: str,
    body: str = "",
) -> str:
    return (
        f'<image id="{frame_id}" name="{name}" width="512" height="512">'
        f"{body}</image>"
    )


def pair_status(status: str = "valid", notes: str = "none") -> str:
    return (
        '<tag label="pair_status" source="manual">'
        f'<attribute name="generation_status">{status}</attribute>'
        f'<attribute name="pair_notes">{notes}</attribute>'
        "</tag>"
    )


def emergency_vehicle_box(new_state: str = "true") -> str:
    return (
        '<box label="emergency_vehicle" source="manual" occluded="0" '
        'xtl="397.90" ytl="244.20" xbr="467.10" ybr="315.20">'
        f'<attribute name="is_newly_added">{new_state}</attribute>'
        '<attribute name="is_valid_rescue">true</attribute>'
        '<attribute name="occlusion_level">partial</attribute>'
        '<attribute name="annotation_notes">'
        "Fire truck partly occluded by collapsed shelves"
        "</attribute></box>"
    )


def cvat_xml(images: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<annotations><version>1.1</version><meta><task>"
        "<id>2686278</id><name>bbox-gt-v1-validation-main</name>"
        "</task><dumped>2026-10-20 17:35:43+00:00</dumped></meta>"
        f"{''.join(images)}</annotations>"
    )


class CvatBoundingBoxConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

    def write_export(self, xml: str, name: str = "annotations.zip") -> Path:
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("annotations.xml", xml)
        return path

    def converter(self, allow_incomplete: bool) -> CvatGroundTruthConverter:
        return CvatGroundTruthConverter(
            ground_truth_template(),
            annotation_mapping(),
            ConversionConfig(allow_incomplete=allow_incomplete),
        )

    def sample_images(self, status: str = "valid") -> list[str]:
        return [
            image_xml(0, "0000__00_before__pair-0.png"),
            image_xml(
                1,
                "0000__01_after__pair-0.png",
                emergency_vehicle_box() + pair_status(status),
            ),
            image_xml(2, "0001__00_before__pair-1.png"),
            image_xml(3, "0001__01_after__pair-1.png"),
        ]

    def test_converts_native_cvat_attributes_and_normalizes_default_note(self) -> None:
        export_path = self.write_export(cvat_xml(self.sample_images()))

        artifact, summary = self.converter(allow_incomplete=True).convert(
            [export_path]
        )

        first = artifact["pairs"][0]
        detected = first["after_objects"][0]
        self.assertEqual(first["generation_status"], "valid")
        self.assertEqual(first["pair_notes"], "")
        self.assertEqual(detected["class_name"], "emergency_vehicle")
        self.assertEqual(detected["bbox_xyxy"], [397.9, 244.2, 467.1, 315.2])
        self.assertEqual(detected["is_newly_added"], "true")
        self.assertEqual(detected["is_valid_rescue"], "true")
        self.assertEqual(detected["occlusion"], "partial")
        self.assertEqual(summary["completed_pairs"], 1)
        self.assertEqual(summary["pending_pairs"], 1)
        self.assertEqual(summary["annotation_status"], "in_progress")
        self.assertEqual(summary["annotation_sources"][0]["task_id"], "2686278")

    def test_strict_conversion_locks_a_complete_split(self) -> None:
        images = self.sample_images()
        images[-1] = image_xml(
            3,
            "0001__01_after__pair-1.png",
            pair_status("missing_requested_object", "No rescue object was added"),
        )
        export_path = self.write_export(cvat_xml(images))

        artifact, summary = self.converter(allow_incomplete=False).convert(
            [export_path]
        )

        self.assertEqual(artifact["annotation_status"], "validation_locked")
        self.assertEqual(summary["completed_pairs"], 2)
        self.assertEqual(summary["pending_pairs"], 0)
        self.assertEqual(
            summary["generation_status_counts"],
            {"missing_requested_object": 1, "valid": 1},
        )

    def test_strict_conversion_rejects_missing_completion_tag(self) -> None:
        export_path = self.write_export(cvat_xml(self.sample_images()))

        with self.assertRaisesRegex(ValueError, "1 incomplete pair"):
            self.converter(allow_incomplete=False).convert([export_path])

    def test_ambiguous_fields_are_reported_and_block_locking(self) -> None:
        images = self.sample_images(status="ambiguous")
        images[1] = image_xml(
            1,
            "0000__01_after__pair-0.png",
            emergency_vehicle_box("ambiguous") + pair_status("ambiguous"),
        )
        images[-1] = image_xml(
            3,
            "0001__01_after__pair-1.png",
            pair_status(),
        )
        export_path = self.write_export(cvat_xml(images))

        _, summary = self.converter(allow_incomplete=True).convert([export_path])
        self.assertEqual(summary["unresolved_pairs"], 1)
        self.assertEqual(
            summary["unresolved"]["pair-0"],
            ["generation_status", "after-0001.is_newly_added"],
        )
        with self.assertRaisesRegex(ValueError, "1 pair.*ambiguous"):
            self.converter(allow_incomplete=False).convert([export_path])

    def test_duplicate_frames_across_exports_are_rejected(self) -> None:
        export_path = self.write_export(cvat_xml(self.sample_images()), "first.zip")
        duplicate_path = self.write_export(
            cvat_xml(self.sample_images()),
            "second.zip",
        )

        with self.assertRaisesRegex(ValueError, "multiple CVAT exports"):
            self.converter(allow_incomplete=True).convert(
                [export_path, duplicate_path]
            )

    def test_unknown_frames_and_unsafe_xml_are_rejected(self) -> None:
        unknown = self.write_export(
            cvat_xml([image_xml(0, "renamed.png")]),
            "unknown.zip",
        )
        with self.assertRaisesRegex(ValueError, "unknown frame"):
            self.converter(allow_incomplete=True).convert([unknown])

        unsafe = self.write_export(
            '<!DOCTYPE annotations [<!ENTITY x "bad">]><annotations/>',
            "unsafe.zip",
        )
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.converter(allow_incomplete=True).convert([unsafe])


if __name__ == "__main__":
    unittest.main()
