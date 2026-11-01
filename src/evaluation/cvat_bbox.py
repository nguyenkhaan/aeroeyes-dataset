from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


OBJECT_CLASSES = {
    "firefighter",
    "rescue_boat",
    "helicopter",
    "ambulance",
    "emergency_vehicle",
}
GENERATION_STATUSES = {
    "valid",
    "missing_requested_object",
    "invalid_object",
    "severe_scene_corruption",
    "ambiguous",
}
BOOLEAN_STATES = {"true", "false", "ambiguous"}
OCCLUSION_LEVELS = {"none", "partial", "heavy"}
MAX_XML_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ConversionConfig:
    split: str = "validation"
    allow_incomplete: bool = False

    def validate(self) -> None:
        if self.split not in {"validation", "test", "all"}:
            raise ValueError(f"Unsupported annotation split: {self.split}")


@dataclass(frozen=True)
class FrameMapping:
    pair_id: str
    split: str
    role: str
    bundle_name: str


@dataclass(frozen=True)
class ParsedExport:
    source: dict
    frames: dict[str, dict]


def load_json_object(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description.capitalize()} must contain an object: {path}")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_xml(path: Path) -> bytes:
    if path.suffix.lower() != ".zip":
        data = path.read_bytes()
    else:
        with zipfile.ZipFile(path) as archive:
            candidates = [
                item
                for item in archive.infolist()
                if not item.is_dir() and Path(item.filename).name == "annotations.xml"
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"CVAT archive must contain exactly one annotations.xml: {path}"
                )
            item = candidates[0]
            if item.flag_bits & 0x1:
                raise ValueError(f"Encrypted CVAT archive is unsupported: {path}")
            if item.file_size > MAX_XML_BYTES:
                raise ValueError(f"CVAT annotations.xml is too large: {path}")
            data = archive.read(item)
    if len(data) > MAX_XML_BYTES:
        raise ValueError(f"CVAT XML is too large: {path}")
    lowered = data.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ValueError(f"CVAT XML declarations are unsafe: {path}")
    return data


def _text(element: ElementTree.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _task_metadata(root: ElementTree.Element) -> dict:
    task = root.find("./meta/task")
    if task is None:
        raise ValueError("CVAT XML is missing task metadata")
    return {
        "task_id": _text(task.find("id")),
        "task_name": _text(task.find("name")),
        "dumped_at": _text(root.find("./meta/dumped")),
        "format_version": _text(root.find("version")),
    }


def _attributes(element: ElementTree.Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for attribute in element.findall("attribute"):
        name = attribute.get("name", "").strip()
        if not name:
            raise ValueError("CVAT attribute is missing its name")
        if name in values:
            raise ValueError(f"Duplicate CVAT attribute: {name}")
        values[name] = _text(attribute)
    return values


def _note(value: str) -> str:
    return "" if value.strip().lower() == "none" else value.strip()


def _coordinate(box: ElementTree.Element, name: str) -> float:
    try:
        value = float(box.attrib[name])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"CVAT box has invalid {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"CVAT box has non-finite {name}")
    return value


def _parse_box(
    box: ElementTree.Element,
    role: str,
    image_size: tuple[int, int],
    object_index: int,
) -> dict:
    class_name = box.get("label", "")
    if class_name not in OBJECT_CLASSES:
        raise ValueError(f"Unknown CVAT rectangle label: {class_name!r}")
    x_min = _coordinate(box, "xtl")
    y_min = _coordinate(box, "ytl")
    x_max = _coordinate(box, "xbr")
    y_max = _coordinate(box, "ybr")
    width, height = image_size
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError(
            f"CVAT box is outside the {width}x{height} image or has zero area"
        )

    attributes = _attributes(box)
    unknown = set(attributes) - {
        "is_newly_added",
        "is_valid_rescue",
        "occlusion_level",
        "annotation_notes",
    }
    if unknown:
        raise ValueError(f"Unknown CVAT object attributes: {sorted(unknown)}")
    occlusion = attributes.get("occlusion_level", "none")
    if occlusion not in OCCLUSION_LEVELS:
        raise ValueError(f"Invalid occlusion_level: {occlusion!r}")

    result = {
        "object_id": f"{role}-{object_index:04d}",
        "class_name": class_name,
        "bbox_xyxy": [x_min, y_min, x_max, y_max],
        "occlusion": occlusion,
        "annotation_notes": _note(attributes.get("annotation_notes", "")),
    }
    if role == "after":
        for name in ("is_newly_added", "is_valid_rescue"):
            state = attributes.get(name, "")
            if state not in BOOLEAN_STATES:
                raise ValueError(f"After object has invalid {name}: {state!r}")
            result[name] = state
    return result


def _parse_pair_status(tag: ElementTree.Element) -> dict:
    if tag.get("label") != "pair_status":
        raise ValueError(f"Unknown CVAT tag label: {tag.get('label')!r}")
    attributes = _attributes(tag)
    unknown = set(attributes) - {"generation_status", "pair_notes"}
    if unknown:
        raise ValueError(f"Unknown pair_status attributes: {sorted(unknown)}")
    generation_status = attributes.get("generation_status", "")
    if generation_status not in GENERATION_STATUSES:
        raise ValueError(f"Invalid generation_status: {generation_status!r}")
    return {
        "generation_status": generation_status,
        "pair_notes": _note(attributes.get("pair_notes", "")),
    }


def _parse_frame(image: ElementTree.Element, mapping: FrameMapping) -> dict:
    try:
        width = int(image.attrib["width"])
        height = int(image.attrib["height"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"CVAT image has invalid dimensions: {mapping.bundle_name}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"CVAT image has invalid dimensions: {mapping.bundle_name}")

    unsupported = [
        child.tag
        for child in image
        if child.tag not in {"box", "tag"}
    ]
    if unsupported:
        raise ValueError(
            f"Unsupported CVAT shapes in {mapping.bundle_name}: {sorted(set(unsupported))}"
        )
    boxes = [
        _parse_box(box, mapping.role, (width, height), index)
        for index, box in enumerate(image.findall("box"), start=1)
    ]
    tags = image.findall("tag")
    if mapping.role == "before" and tags:
        raise ValueError(f"pair_status is only allowed on After: {mapping.bundle_name}")
    if len(tags) > 1:
        raise ValueError(f"After frame has multiple pair_status tags: {mapping.bundle_name}")
    status = _parse_pair_status(tags[0]) if tags else None
    return {"objects": boxes, "status": status}


def _parse_export(path: Path, frame_mappings: dict[str, FrameMapping]) -> ParsedExport:
    try:
        root = ElementTree.fromstring(_read_xml(path))
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise ValueError(f"Invalid CVAT export: {path}") from exc
    if root.tag != "annotations":
        raise ValueError(f"CVAT XML root must be annotations: {path}")
    if _text(root.find("version")) != "1.1":
        raise ValueError(f"CVAT export must use native image format 1.1: {path}")
    if root.find("track") is not None:
        raise ValueError(f"CVAT video annotations are unsupported: {path}")

    frames: dict[str, dict] = {}
    for image in root.findall("image"):
        name = image.get("name", "")
        mapping = frame_mappings.get(name)
        if mapping is None:
            raise ValueError(f"CVAT export contains an unknown frame: {name!r}")
        if name in frames:
            raise ValueError(f"CVAT export contains a duplicate frame: {name}")
        frames[name] = _parse_frame(image, mapping)
    source = {
        "file_name": path.name,
        "sha256": _sha256(path),
        **_task_metadata(root),
    }
    return ParsedExport(source=source, frames=frames)


class CvatGroundTruthConverter:
    def __init__(self, template: dict, mapping: dict, config: ConversionConfig):
        config.validate()
        self.template = template
        self.mapping = mapping
        self.config = config
        self.template_pairs = self._template_pairs()
        self.frame_mappings = self._frame_mappings()

    def _is_selected(self, split: str) -> bool:
        return self.config.split == "all" or split == self.config.split

    def _template_pairs(self) -> dict[str, dict]:
        pairs = self.template.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("Ground-truth template must contain a non-empty pairs list")
        indexed: dict[str, dict] = {}
        for pair in pairs:
            if not isinstance(pair, dict) or not pair.get("pair_id"):
                raise ValueError("Ground-truth template contains an invalid pair")
            pair_id = str(pair["pair_id"])
            if pair_id in indexed:
                raise ValueError(f"Duplicate template pair_id: {pair_id}")
            indexed[pair_id] = pair
        return indexed

    def _frame_mappings(self) -> dict[str, FrameMapping]:
        if self.template.get("source_manifest_sha256") != self.mapping.get(
            "source_manifest_sha256"
        ):
            raise ValueError("Template and annotation mapping use different manifests")
        pairs = self.mapping.get("pairs")
        if not isinstance(pairs, list):
            raise ValueError("Annotation mapping must contain a pairs list")
        mappings: dict[str, FrameMapping] = {}
        mapped_pair_ids: set[str] = set()
        for pair in pairs:
            pair_id = str(pair.get("pair_id", ""))
            split = str(pair.get("split", ""))
            if pair_id not in self.template_pairs:
                raise ValueError(f"Mapping pair is absent from template: {pair_id}")
            if pair_id in mapped_pair_ids:
                raise ValueError(f"Duplicate mapping pair_id: {pair_id}")
            if split != str(self.template_pairs[pair_id].get("split", "")):
                raise ValueError(f"Template and mapping split differ for: {pair_id}")
            mapped_pair_ids.add(pair_id)
            if not self._is_selected(split):
                continue
            frames = pair.get("frames")
            if not isinstance(frames, list) or len(frames) != 2:
                raise ValueError(f"Mapping pair must contain two frames: {pair_id}")
            roles: set[str] = set()
            for frame in frames:
                role = str(frame.get("role", ""))
                name = str(frame.get("bundle_name", ""))
                if role not in {"before", "after"} or not name:
                    raise ValueError(f"Invalid frame mapping for pair: {pair_id}")
                if name in mappings:
                    raise ValueError(f"Duplicate mapped frame name: {name}")
                roles.add(role)
                mappings[name] = FrameMapping(pair_id, split, role, name)
            if roles != {"before", "after"}:
                raise ValueError(f"Mapping lacks Before or After frame: {pair_id}")
        if mapped_pair_ids != set(self.template_pairs):
            raise ValueError("Template and annotation mapping contain different pairs")
        return mappings

    def convert(self, export_paths: list[Path]) -> tuple[dict, dict]:
        if not export_paths:
            raise ValueError("At least one CVAT export is required")
        merged_frames: dict[str, dict] = {}
        sources: list[dict] = []
        for path in export_paths:
            parsed = _parse_export(path, self.frame_mappings)
            duplicates = set(merged_frames) & set(parsed.frames)
            if duplicates:
                raise ValueError(
                    f"Frames occur in multiple CVAT exports: {sorted(duplicates)[:3]}"
                )
            merged_frames.update(parsed.frames)
            sources.append(parsed.source)

        pairs_by_id: dict[str, dict[str, dict]] = {}
        for frame_name, frame in merged_frames.items():
            mapping = self.frame_mappings[frame_name]
            pairs_by_id.setdefault(mapping.pair_id, {})[mapping.role] = frame

        completed: set[str] = set()
        unresolved: dict[str, list[str]] = {}
        converted_pairs = []
        for pair_id, template_pair in self.template_pairs.items():
            pair = dict(template_pair)
            if not self._is_selected(str(pair.get("split", ""))):
                converted_pairs.append(pair)
                continue
            frames = pairs_by_id.get(pair_id, {})
            after = frames.get("after")
            if after is None or after["status"] is None:
                converted_pairs.append(pair)
                continue
            before = frames.get("before")
            if before is None:
                raise ValueError(f"Completed pair lacks its Before frame: {pair_id}")
            completed.add(pair_id)
            status = after["status"]
            pair["before_objects"] = before["objects"]
            pair["after_objects"] = after["objects"]
            pair["generation_status"] = status["generation_status"]
            pair["pair_notes"] = status["pair_notes"]
            reasons = []
            if status["generation_status"] == "ambiguous":
                reasons.append("generation_status")
            for obj in after["objects"]:
                for field in ("is_newly_added", "is_valid_rescue"):
                    if obj[field] == "ambiguous":
                        reasons.append(f"{obj['object_id']}.{field}")
            if reasons:
                unresolved[pair_id] = reasons
            converted_pairs.append(pair)

        selected_ids = {
            pair_id
            for pair_id, pair in self.template_pairs.items()
            if self._is_selected(str(pair.get("split", "")))
        }
        pending = sorted(selected_ids - completed)
        if not self.config.allow_incomplete:
            if pending:
                raise ValueError(
                    f"Selected split has {len(pending)} incomplete pair(s); "
                    "use --allow-incomplete only for pilot review"
                )
            if unresolved:
                raise ValueError(
                    f"Selected split has {len(unresolved)} pair(s) with ambiguous labels"
                )

        status_name = (
            "in_progress"
            if pending or unresolved
            else f"{self.config.split}_locked"
        )
        artifact = {
            **self.template,
            "annotation_status": status_name,
            "annotation_split": self.config.split,
            "annotation_sources": sources,
            "pairs": converted_pairs,
        }
        annotated_pairs = [
            pair for pair in converted_pairs if str(pair.get("pair_id")) in completed
        ]
        summary = {
            "annotation_status": status_name,
            "split": self.config.split,
            "selected_pairs": len(selected_ids),
            "completed_pairs": len(completed),
            "pending_pairs": len(pending),
            "unresolved_pairs": len(unresolved),
            "pending_pair_ids": pending,
            "unresolved": unresolved,
            "generation_status_counts": dict(
                sorted(
                    Counter(
                        pair["generation_status"] for pair in annotated_pairs
                    ).items()
                )
            ),
            "before_object_count": sum(
                len(pair["before_objects"]) for pair in annotated_pairs
            ),
            "after_object_count": sum(
                len(pair["after_objects"]) for pair in annotated_pairs
            ),
            "after_class_counts": dict(
                sorted(
                    Counter(
                        obj["class_name"]
                        for pair in annotated_pairs
                        for obj in pair["after_objects"]
                    ).items()
                )
            ),
            "annotation_sources": sources,
        }
        return artifact, summary
