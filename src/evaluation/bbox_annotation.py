from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path


SPLITS = ("validation", "test")
OBJECT_CLASSES = (
    "firefighter",
    "rescue_boat",
    "helicopter",
    "ambulance",
    "emergency_vehicle",
)
GENERATION_ATTRIBUTE_VALUES = (
    "ambiguous",
    "valid",
    "missing_requested_object",
    "invalid_object",
    "severe_scene_corruption",
)
LABEL_COLORS = (
    "#00C853",
    "#2979FF",
    "#AA00FF",
    "#FF1744",
    "#FF9100",
)


@dataclass(frozen=True)
class AnnotationBundleConfig:
    double_annotation_fraction: float
    seed: int
    source_manifest_sha256: str | None = None

    def validate(self) -> None:
        if not 0 <= self.double_annotation_fraction <= 1:
            raise ValueError("Double-annotation fraction must be between 0 and 1")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_json_atomic(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(_json_bytes(value))
    temporary_path.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(manifest: dict, configured_digest: str | None) -> str:
    if configured_digest:
        return configured_digest
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _safe_pair_id(pair_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", pair_id).strip("._")
    if not cleaned:
        raise ValueError(f"pair_id cannot be represented safely: {pair_id!r}")
    return cleaned


def _validate_image(pair: dict, role: str) -> Path:
    path_key = f"{role}_path"
    checksum_key = f"{role}_sha256"
    try:
        path = Path(str(pair[path_key]))
        expected_checksum = str(pair[checksum_key])
    except KeyError as exc:
        raise ValueError(
            f"Pair {pair.get('pair_id')} is missing {exc.args[0]}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"Annotation image is missing: {path}")
    actual_checksum = _sha256_file(path)
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Image checksum mismatch for {pair.get('pair_id')} {role}: {path}"
        )
    return path


def _validate_manifest(manifest: dict) -> dict[str, list[dict]]:
    pairs = manifest.get("pairs")
    if (
        not isinstance(pairs, list)
        or not pairs
        or not all(isinstance(pair, dict) for pair in pairs)
    ):
        raise ValueError("Benchmark manifest must contain a non-empty pairs list")
    pair_ids = [str(pair.get("pair_id", "")) for pair in pairs]
    if any(not pair_id for pair_id in pair_ids):
        raise ValueError("Every benchmark pair must have a pair_id")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Benchmark manifest contains duplicate pair_id values")

    pairs_by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for pair in pairs:
        split = pair.get("split")
        if split not in pairs_by_split:
            raise ValueError(f"Unsupported benchmark split: {split}")
        _validate_image(pair, "before")
        _validate_image(pair, "after")
        pairs_by_split[split].append(pair)
    for split in SPLITS:
        pairs_by_split[split].sort(key=lambda pair: str(pair["pair_id"]))
        if not pairs_by_split[split]:
            raise ValueError(f"Benchmark split is empty: {split}")
    declared_counts = manifest.get("counts", {})
    actual_counts = {
        "validation": len(pairs_by_split["validation"]),
        "test": len(pairs_by_split["test"]),
        "total": len(pairs),
    }
    if declared_counts != actual_counts:
        raise ValueError(
            f"Benchmark manifest counts do not match its pairs: {actual_counts}"
        )
    return pairs_by_split


def _double_annotation_ids(
    pairs_by_split: dict[str, list[dict]],
    config: AnnotationBundleConfig,
) -> dict[str, set[str]]:
    selections: dict[str, set[str]] = {}
    for split, pairs in pairs_by_split.items():
        count = math.ceil(len(pairs) * config.double_annotation_fraction)
        ranked = sorted(
            pairs,
            key=lambda pair: hashlib.sha256(
                f"{config.seed}:{split}:{pair['pair_id']}".encode()
            ).hexdigest(),
        )
        selections[split] = {
            str(pair["pair_id"]) for pair in ranked[:count]
        }
    return selections


def _frame_record(pair: dict, role: str, bundle_name: str) -> dict:
    return {
        "role": role,
        "bundle_name": bundle_name,
        "source_path": str(pair[f"{role}_path"]),
        "sha256": str(pair[f"{role}_sha256"]),
    }


def _mapping_pairs(
    pairs_by_split: dict[str, list[dict]],
    double_ids: dict[str, set[str]],
) -> list[dict]:
    mapping: list[dict] = []
    for split, pairs in pairs_by_split.items():
        for index, pair in enumerate(pairs):
            pair_id = str(pair["pair_id"])
            safe_pair_id = _safe_pair_id(pair_id)
            before_path = Path(str(pair["before_path"]))
            after_path = Path(str(pair["after_path"]))
            prefix = f"{index:04d}"
            before_name = (
                f"{prefix}__00_before__{safe_pair_id}{before_path.suffix.lower()}"
            )
            after_name = (
                f"{prefix}__01_after__{safe_pair_id}{after_path.suffix.lower()}"
            )
            mapping.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "source_group": pair.get("source_group"),
                    "source_group_method": pair.get("source_group_method"),
                    "double_annotation": pair_id in double_ids[split],
                    "frames": [
                        _frame_record(pair, "before", before_name),
                        _frame_record(pair, "after", after_name),
                    ],
                }
            )
    return mapping


def _write_image_archive(path: Path, mapping_pairs: list[dict]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary_path,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for pair in mapping_pairs:
            for frame in pair["frames"]:
                archive.write(frame["source_path"], arcname=frame["bundle_name"])
    temporary_path.replace(path)


def _attribute(
    name: str,
    input_type: str,
    values: list[str],
) -> dict:
    return {
        "name": name,
        "mutable": False,
        "input_type": input_type,
        "default_value": values[0] if values else "",
        "values": values,
    }


def _object_attributes() -> list[dict]:
    return [
        _attribute(
            "is_newly_added",
            "select",
            ["ambiguous", "true", "false"],
        ),
        _attribute(
            "is_valid_rescue",
            "select",
            ["ambiguous", "true", "false"],
        ),
        _attribute(
            "occlusion_level",
            "select",
            ["none", "partial", "heavy"],
        ),
        _attribute("annotation_notes", "text", []),
    ]


def _cvat_labels() -> list[dict]:
    labels = [
        {
            "name": class_name,
            "color": color,
            "type": "rectangle",
            "attributes": _object_attributes(),
        }
        for class_name, color in zip(OBJECT_CLASSES, LABEL_COLORS, strict=True)
    ]
    labels.append(
        {
            "name": "pair_status",
            "color": "#607D8B",
            "type": "tag",
            "attributes": [
                _attribute(
                    "generation_status",
                    "select",
                    list(GENERATION_ATTRIBUTE_VALUES),
                ),
                _attribute("pair_notes", "text", []),
            ],
        }
    )
    return labels


def _ground_truth_template(manifest: dict, manifest_digest: str) -> dict:
    template_pairs = []
    for pair in manifest["pairs"]:
        template_pairs.append(
            {
                **pair,
                "generation_status": None,
                "before_objects": [],
                "after_objects": [],
                "pair_notes": "",
            }
        )
    return {
        "schema_version": 1,
        "artifact_version": "bbox-gt-v1",
        "source_manifest_sha256": manifest_digest,
        "annotation_status": "template",
        "pairs": template_pairs,
    }


def create_bundle(
    manifest: dict,
    output_dir: Path,
    config: AnnotationBundleConfig,
) -> dict:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_by_split = _validate_manifest(manifest)
    double_ids = _double_annotation_ids(pairs_by_split, config)
    mapping_pairs = _mapping_pairs(pairs_by_split, double_ids)
    manifest_digest = _manifest_digest(manifest, config.source_manifest_sha256)

    for split in SPLITS:
        split_pairs = [pair for pair in mapping_pairs if pair["split"] == split]
        double_pairs = [pair for pair in split_pairs if pair["double_annotation"]]
        _write_image_archive(output_dir / f"{split}-images.zip", split_pairs)
        _write_image_archive(
            output_dir / f"{split}-double-annotation-images.zip",
            double_pairs,
        )

    mapping = {
        "schema_version": 1,
        "source_manifest_sha256": manifest_digest,
        "double_annotation_fraction": config.double_annotation_fraction,
        "double_annotation_seed": config.seed,
        "pairs": mapping_pairs,
    }
    _write_json_atomic(output_dir / "annotation-mapping.json", mapping)
    _write_json_atomic(output_dir / "cvat-labels.json", _cvat_labels())
    _write_json_atomic(
        output_dir / "bbox-gt-v1.template.json",
        _ground_truth_template(manifest, manifest_digest),
    )

    artifact_names = [
        "validation-images.zip",
        "test-images.zip",
        "validation-double-annotation-images.zip",
        "test-double-annotation-images.zip",
        "annotation-mapping.json",
        "cvat-labels.json",
        "bbox-gt-v1.template.json",
    ]
    summary = {
        "output_dir": str(output_dir),
        "source_manifest_sha256": manifest_digest,
        "counts": {
            "pairs": len(mapping_pairs),
            "frames": len(mapping_pairs) * 2,
            **{split: len(pairs_by_split[split]) for split in SPLITS},
        },
        "double_annotation_counts": {
            split: len(double_ids[split]) for split in SPLITS
        },
        "artifact_sha256": {
            name: _sha256_file(output_dir / name) for name in artifact_names
        },
    }
    _write_json_atomic(output_dir / "bundle-summary.json", summary)
    return summary
