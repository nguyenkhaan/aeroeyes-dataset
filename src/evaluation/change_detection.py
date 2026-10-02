from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from src.core.config import (
    DELTA_INTENSITY_THRESHOLD,
    DELTA_MAX_AREA_RATIO,
    DELTA_MIN_AREA_RATIO,
    DELTA_MORPH_KERNEL_SIZE,
)


@dataclass
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    area: int
    mean_delta: float

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    def to_yolo(self, image_width: int, image_height: int, class_id: int = 0) -> str:
        """Format as YOLO line: class_id center_x center_y width height"""
        if image_width <= 0 or image_height <= 0:
            return ""
        center_x = ((self.x_min + self.x_max) / 2.0) / image_width
        center_y = ((self.y_min + self.y_max) / 2.0) / image_height
        norm_w = self.width / float(image_width)
        norm_h = self.height / float(image_height)
        return f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"

    def iou(self, other: BoundingBox) -> float:
        inter_x_min = max(self.x_min, other.x_min)
        inter_y_min = max(self.y_min, other.y_min)
        inter_x_max = min(self.x_max, other.x_max)
        inter_y_max = min(self.y_max, other.y_max)

        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)


@dataclass
class ChangeDeltaResult:
    mean_delta: float
    delta_area_ratio: float
    is_valid_change: bool
    bounding_boxes: list[BoundingBox]
    delta_mask: np.ndarray
    delta_intensity_map: np.ndarray

    @property
    def box_count(self) -> int:
        return len(self.bounding_boxes)


def compute_otsu_threshold(intensity_map: np.ndarray) -> float:
    flat = np.clip(intensity_map.ravel(), 0, 255).astype(np.uint8)
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    total = float(flat.size)
    if total == 0:
        return 0.0

    current_max = -1.0
    best_thresholds: list[int] = []
    weight_background = 0.0
    sum_background = 0.0
    total_mean = float(np.dot(np.arange(256), hist))

    for t in range(256):
        weight_background += float(hist[t])
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += float(t * hist[t])
        mean_background = sum_background / weight_background
        mean_foreground = (total_mean - sum_background) / weight_foreground

        between_variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        if between_variance > current_max + 1e-4:
            current_max = between_variance
            best_thresholds = [t]
        elif abs(between_variance - current_max) <= 1e-4:
            best_thresholds.append(t)

    if not best_thresholds:
        return 0.0
    return float(np.mean(best_thresholds))


def _morphological_close_open(
    binary_mask: np.ndarray,
    kernel_size: int = DELTA_MORPH_KERNEL_SIZE,
) -> np.ndarray:
    try:
        import cv2

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size),
        )
        closed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return opened > 0
    except ImportError:
        try:
            from scipy import ndimage

            structure = np.ones((kernel_size, kernel_size), dtype=bool)
            closed = ndimage.binary_closing(binary_mask, structure=structure)
            opened = ndimage.binary_opening(closed, structure=structure)
            return opened
        except ImportError:
            return binary_mask


def _find_connected_components(
    binary_mask: np.ndarray,
    min_area: int = 100,
) -> list[tuple[int, int, int, int, int]]:
    try:
        import cv2

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask.astype(np.uint8),
            connectivity=8,
        )
        boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= min_area:
                boxes.append((int(x), int(y), int(x + w), int(y + h), int(area)))
        return boxes
    except ImportError:
        try:
            from scipy import ndimage

            labeled, num_features = ndimage.label(binary_mask)
            boxes = []
            slices = ndimage.find_objects(labeled)
            for i, slice_tuple in enumerate(slices, start=1):
                if slice_tuple is None:
                    continue
                slice_y, slice_x = slice_tuple
                comp_mask = labeled[slice_y, slice_x] == i
                area = int(comp_mask.sum())
                if area >= min_area:
                    boxes.append(
                        (
                            int(slice_x.start),
                            int(slice_y.start),
                            int(slice_x.stop),
                            int(slice_y.stop),
                            area,
                        )
                    )
            return boxes
        except ImportError:
            if not np.any(binary_mask):
                return []
            ys, xs = np.where(binary_mask)
            area = int(len(ys))
            if area >= min_area:
                return [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), area)]
            return []


def detect_change_delta(
    before_image: Image.Image,
    after_image: Image.Image,
    intensity_threshold: float = DELTA_INTENSITY_THRESHOLD,
    min_area_ratio: float = DELTA_MIN_AREA_RATIO,
    max_area_ratio: float = DELTA_MAX_AREA_RATIO,
    morph_kernel_size: int = DELTA_MORPH_KERNEL_SIZE,
    use_otsu: bool = True,
) -> ChangeDeltaResult:
    before_rgb = before_image.convert("RGB")
    after_rgb = after_image.convert("RGB")

    if before_rgb.size != after_rgb.size:
        after_rgb = after_rgb.resize(before_rgb.size, Image.Resampling.BILINEAR)

    width, height = before_rgb.size
    total_pixels = width * height

    before_arr = np.array(before_rgb, dtype=np.float32)
    after_arr = np.array(after_rgb, dtype=np.float32)

    diff_rgb = np.abs(after_arr - before_arr)

    intensity_delta = (
        0.299 * diff_rgb[..., 0]
        + 0.587 * diff_rgb[..., 1]
        + 0.114 * diff_rgb[..., 2]
    )

    if use_otsu:
        otsu_val = compute_otsu_threshold(intensity_delta)
        applied_threshold = max(intensity_threshold, min(otsu_val, 60.0))
    else:
        applied_threshold = intensity_threshold

    raw_binary = intensity_delta >= applied_threshold
    clean_mask = _morphological_close_open(raw_binary, kernel_size=morph_kernel_size)

    min_box_pixels = max(100, int(total_pixels * (min_area_ratio / 5.0)))
    raw_boxes = _find_connected_components(clean_mask, min_area=min_box_pixels)

    bounding_boxes: list[BoundingBox] = []
    for x_min, y_min, x_max, y_max, area in raw_boxes:
        box_delta = intensity_delta[y_min:y_max, x_min:x_max]
        mean_box_delta = float(np.mean(box_delta)) if box_delta.size > 0 else 0.0
        bounding_boxes.append(
            BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                area=area,
                mean_delta=mean_box_delta,
            )
        )

    bounding_boxes.sort(key=lambda b: b.area, reverse=True)

    changed_pixels = int(np.sum(clean_mask))
    delta_area_ratio = float(changed_pixels / total_pixels) if total_pixels > 0 else 0.0
    mean_delta = float(np.mean(intensity_delta[clean_mask])) if changed_pixels > 0 else 0.0

    is_valid_change = (
        min_area_ratio <= delta_area_ratio <= max_area_ratio
        and len(bounding_boxes) > 0
    )

    return ChangeDeltaResult(
        mean_delta=mean_delta,
        delta_area_ratio=delta_area_ratio,
        is_valid_change=is_valid_change,
        bounding_boxes=bounding_boxes,
        delta_mask=clean_mask.astype(np.uint8) * 255,
        delta_intensity_map=intensity_delta,
    )
