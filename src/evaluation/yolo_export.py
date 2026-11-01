from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path

from src.core.config import (
    HF_HUB_CACHE,
    SDQM_BOX_THRESHOLD,
    SDQM_GROUNDING_DINO_MODEL,
    SDQM_MODEL_TEXT,
    SDQM_NMS_IOU_THRESHOLD,
    SDQM_RESCUE_BOAT_THRESHOLD,
    SDQM_TEXT_THRESHOLD,
    SDQM_YOLO_DATA_YAML,
    ensure_model_storage,
)

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.evaluation.change_detection import BoundingBox
from src.evaluation.bbox_aliases import normalize_label, resolve_label
from src.evaluation.bbox_class_conflict import suppress_ambulance_conflicts
from src.evaluation.bbox_prompt_supplement import detect_with_supplemental_prompts, validate_prompt_classes
from src.evaluation.image_files import list_images
from src.evaluation.grounding_postprocess import GroundingPostprocessRequest, postprocess_grounding
from src.evaluation.object_matching import DetectedObject


DEFAULT_CLASS_THRESHOLDS = {"rescue_boat": SDQM_RESCUE_BOAT_THRESHOLD}


def load_class_map(yaml_path: str | Path) -> dict[str, int]:
    with Path(yaml_path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    names = data.get("names", {})
    if isinstance(names, dict):
        return {str(name).lower(): int(class_id) for class_id, name in names.items()}

    return {str(name).lower(): index for index, name in enumerate(names)}


def _label_to_class_id(label: str, class_map: dict[str, int]) -> int | None:
    normalized = normalize_label(label)
    candidates = [
        normalized,
        normalized.replace("_", " "),
        label.strip().lower(),
    ]
    for candidate in candidates:
        if candidate in class_map:
            return class_map[candidate]
    return None


@dataclass(frozen=True)
class DetectionConversionContext:
    class_map: dict[str, int]
    image_width: int
    image_height: int
    box_threshold: float
    class_thresholds: dict[str, float]
    aliases: dict[str, str] = field(default_factory=dict)


def _processed_result_to_objects(
    result: dict,
    context: DetectionConversionContext,
) -> list[DetectedObject]:
    labels = result.get("text_labels", result.get("labels", []))
    objects: list[DetectedObject] = []
    for box, score, label in zip(
        result.get("boxes", []),
        result.get("scores", []),
        labels,
        strict=False,
    ):
        label_text = str(label)
        normalized_label = resolve_label(label_text, context.aliases)
        class_id = _label_to_class_id(normalized_label, context.class_map)
        if class_id is None:
            continue
        confidence = float(score)
        required_confidence = max(
            context.box_threshold,
            context.class_thresholds.get(normalized_label, context.box_threshold),
        )
        if confidence < required_confidence:
            continue

        raw_x_min, raw_y_min, raw_x_max, raw_y_max = box.tolist()
        x_min = max(0, min(context.image_width - 1, floor(raw_x_min)))
        y_min = max(0, min(context.image_height - 1, floor(raw_y_min)))
        x_max = max(x_min + 1, min(context.image_width, ceil(raw_x_max)))
        y_max = max(y_min + 1, min(context.image_height, ceil(raw_y_max)))
        objects.append(
            DetectedObject(
                class_id=class_id,
                label=normalized_label,
                box=BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    area=(x_max - x_min) * (y_max - y_min),
                    mean_delta=0.0,
                ),
                confidence=confidence,
            )
        )
    return objects


def _class_aware_nms(
    detections: list[DetectedObject],
    iou_threshold: float,
) -> list[DetectedObject]:
    kept: list[DetectedObject] = []
    for candidate in sorted(
        detections,
        key=lambda detected: detected.confidence,
        reverse=True,
    ):
        overlaps_existing = any(
            candidate.class_id == existing.class_id
            and candidate.box.iou(existing.box) >= iou_threshold
            for existing in kept
        )
        if not overlaps_existing:
            kept.append(candidate)
    return kept


@dataclass
class RescueDetector:
    model_name: str = SDQM_GROUNDING_DINO_MODEL
    text_prompt: str = SDQM_MODEL_TEXT
    box_threshold: float = SDQM_BOX_THRESHOLD
    text_threshold: float = SDQM_TEXT_THRESHOLD
    class_map: dict[str, int] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    class_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_THRESHOLDS)
    )
    nms_iou_threshold: float = SDQM_NMS_IOU_THRESHOLD
    device: str | None = None
    supplemental_prompts: list[dict] = field(default_factory=list)
    cross_class_iou_threshold: float | None = None
    vehicle_clip_filter: object | None = None
    postprocess_mode: str | None = field(default=None, init=False)
    raw_label_counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    alias_label_counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    unmapped_label_counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    supplemental_prompt_counts: dict[str, Counter] = field(default_factory=dict, init=False, repr=False)
    cross_class_conflict_counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    vehicle_clip_counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    last_clip_decisions: list[dict] = field(default_factory=list, init=False, repr=False)
    _processor: AutoProcessor | None = field(default=None, init=False, repr=False)
    _model: AutoModelForZeroShotObjectDetection | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def load(self) -> None:
        if self._model is not None:
            return

        ensure_model_storage()
        eval_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = eval_device
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=str(HF_HUB_CACHE),
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name,
            cache_dir=str(HF_HUB_CACHE),
        ).to(eval_device)
        self._model.eval()

    def detect_objects(self, image: Image.Image) -> list[DetectedObject]:
        if self._model is None or self._processor is None:
            self.load()

        assert self._model is not None
        assert self._processor is not None

        validate_prompt_classes(self.supplemental_prompts, self.aliases)
        detections, counts = detect_with_supplemental_prompts(
            self.text_prompt, self.supplemental_prompts,
            lambda prompt: self._detect_prompt(image, prompt),
            lambda items: _class_aware_nms(items, self.nms_iou_threshold),
        )
        for prompt, values in counts.items():
            self.supplemental_prompt_counts.setdefault(prompt, Counter()).update(values)
        if self.cross_class_iou_threshold is not None:
            detections, removed = suppress_ambulance_conflicts(detections, self.cross_class_iou_threshold)
            self.cross_class_conflict_counts.update({"images_processed": 1, "ambulance_suppressed": removed})
        self.last_clip_decisions = []
        if self.vehicle_clip_filter is not None:
            if self.cross_class_iou_threshold is not None:
                raise ValueError("CLIP and unconditional cross-class suppression cannot be combined")
            detections, self.last_clip_decisions = self.vehicle_clip_filter.apply(image, detections)
            self.vehicle_clip_counts.update({"images_processed": 1,
                                             "conflicts": len(self.last_clip_decisions),
                                             "ambulance_suppressed": sum(item["suppressed"] for item in self.last_clip_decisions)})
        return detections

    def _detect_prompt(self, image: Image.Image, prompt: str) -> list[DetectedObject]:
        assert self._model is not None
        assert self._processor is not None

        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        inputs = self._processor(
            images=rgb_image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[height, width]], device=self.device)
        processed, self.postprocess_mode = postprocess_grounding(
            self._processor,
            GroundingPostprocessRequest(
                outputs=outputs,
                input_ids=inputs.input_ids,
                target_sizes=target_sizes,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            ),
        )
        for label in processed[0].get("text_labels", processed[0].get("labels", [])):
            normalized = normalize_label(str(label))
            canonical = resolve_label(str(label), self.aliases)
            self.raw_label_counts[normalized] += 1
            if _label_to_class_id(canonical, self.class_map) is None:
                self.unmapped_label_counts[normalized] += 1
            elif canonical != normalized:
                self.alias_label_counts[normalized] += 1
        detections = _processed_result_to_objects(
            processed[0],
            DetectionConversionContext(
                class_map=self.class_map,
                image_width=width,
                image_height=height,
                box_threshold=self.box_threshold,
                class_thresholds=self.class_thresholds,
                aliases=self.aliases,
            ),
        )
        return detections

    def detect(self, image: Image.Image) -> list[str]:
        width, height = image.size
        return [detected.to_yolo(width, height) for detected in self.detect_objects(image)]


def _write_dataset_yaml(output_dir: Path, template_path: Path) -> Path:
    with template_path.open(encoding="utf-8") as handle:
        yaml_data = yaml.safe_load(handle)

    yaml_data["path"] = str(output_dir.resolve())
    yaml_data["train"] = "images/train"
    yaml_data["val"] = "images/train"

    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.dump(yaml_data, handle, sort_keys=False)

    return yaml_path


def export_yolo_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    data_yaml_template: str | Path = SDQM_YOLO_DATA_YAML,
    detector: RescueDetector | None = None,
    split: str = "train",
) -> Path:
    """
    Auto-label images with Grounding DINO and export a YOLO dataset layout.

    Output structure:
        {output_dir}/images/{split}/
        {output_dir}/labels/{split}/
        {output_dir}/data.yaml
    """
    source_root = Path(source_dir)
    export_root = Path(output_dir)
    images_dir = export_root / "images" / split
    labels_dir = export_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_map = load_class_map(data_yaml_template)
    rescue_detector = detector or RescueDetector(class_map=class_map)
    rescue_detector.class_map = class_map
    rescue_detector.load()

    image_paths = list_images(source_root)
    if not image_paths:
        raise ValueError(f"No images found in {source_root}")

    for image_path in tqdm(image_paths, desc=f"YOLO export {export_root.name}"):
        destination_image = images_dir / image_path.name
        shutil.copy2(image_path, destination_image)

        image = Image.open(image_path)
        label_lines = rescue_detector.detect(image)

        label_path = labels_dir / f"{image_path.stem}.txt"
        label_path.write_text("".join(label_lines), encoding="utf-8")

    _write_dataset_yaml(export_root, Path(data_yaml_template))
    return export_root


def export_yolo_pair(
    ref_dir: str | Path,
    eval_dir: str | Path,
    output_dir: str | Path,
    data_yaml_template: str | Path = SDQM_YOLO_DATA_YAML,
    detector: RescueDetector | None = None,
) -> tuple[Path, Path]:
    """Export real and synthetic image directories into YOLO layouts."""
    sdqm_yolo_root = Path(output_dir)
    real_root = sdqm_yolo_root / "real"
    synthetic_root = sdqm_yolo_root / "synthetic"

    class_map = load_class_map(data_yaml_template)
    shared_detector = detector or RescueDetector(class_map=class_map)

    export_yolo_dataset(
        ref_dir,
        real_root,
        data_yaml_template=data_yaml_template,
        detector=shared_detector,
    )
    export_yolo_dataset(
        eval_dir,
        synthetic_root,
        data_yaml_template=data_yaml_template,
        detector=shared_detector,
    )

    return real_root, synthetic_root
