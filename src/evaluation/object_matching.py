from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from src.evaluation.change_detection import BoundingBox, ChangeDeltaResult


@dataclass
class DetectedObject:
    class_id: int
    label: str
    box: BoundingBox
    confidence: float

    def to_yolo(self, image_width: int, image_height: int) -> str:
        return self.box.to_yolo(image_width, image_height, class_id=self.class_id)


@dataclass
class ObjectMatchingReport:
    preserved_objects: list[tuple[DetectedObject, DetectedObject, float]]
    added_rescue_objects: list[tuple[DetectedObject, float]]
    rejected_detections: list[tuple[DetectedObject, str]]
    preservation_score: float
    confirmed_rescue_count: int
    refined_boxes: list[BoundingBox]

    @property
    def is_consistent_rescue(self) -> bool:
        return self.preservation_score >= 0.5 and self.confirmed_rescue_count > 0


def match_and_refine_objects(
    before_objects: Sequence[DetectedObject],
    after_objects: Sequence[DetectedObject],
    delta_result: ChangeDeltaResult,
    match_iou_threshold: float = 0.40,
    min_delta_overlap: float = 0.20,
) -> ObjectMatchingReport:
    matched_before_indices: set[int] = set()
    matched_after_indices: set[int] = set()
    preserved_pairs: list[tuple[DetectedObject, DetectedObject, float]] = []

    for i, before_obj in enumerate(before_objects):
        best_iou = 0.0
        best_j = -1
        for j, after_obj in enumerate(after_objects):
            if j in matched_after_indices:
                continue
            iou_val = before_obj.box.iou(after_obj.box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j

        if best_iou >= match_iou_threshold and best_j >= 0:
            matched_before_indices.add(i)
            matched_after_indices.add(best_j)
            preserved_pairs.append((before_obj, after_objects[best_j], best_iou))

    total_before = len(before_objects)
    if total_before > 0:
        preservation_score = float(len(matched_before_indices) / total_before)
    else:
        preservation_score = 1.0 if delta_result.delta_area_ratio < 0.4 else 0.5

    added_rescue: list[tuple[DetectedObject, float]] = []
    rejected: list[tuple[DetectedObject, str]] = []
    refined_boxes: list[BoundingBox] = []

    mask_h, mask_w = delta_result.delta_mask.shape

    for j, after_obj in enumerate(after_objects):
        if j in matched_after_indices:
            continue

        box = after_obj.box
        x_min = max(0, min(box.x_min, mask_w - 1))
        x_max = max(0, min(box.x_max, mask_w))
        y_min = max(0, min(box.y_min, mask_h - 1))
        y_max = max(0, min(box.y_max, mask_h))

        box_crop = delta_result.delta_mask[y_min:y_max, x_min:x_max]
        box_area = (x_max - x_min) * (y_max - y_min)

        if box_area <= 0:
            rejected.append((after_obj, "Invalid zero box area"))
            continue

        changed_pixels_inside = int(np.sum(box_crop > 0))
        overlap_ratio = changed_pixels_inside / float(box_area)

        if overlap_ratio >= min_delta_overlap:
            added_rescue.append((after_obj, overlap_ratio))

            ys, xs = np.where(box_crop > 0)
            if len(ys) > 0:
                refined_x_min = int(x_min + xs.min())
                refined_x_max = int(x_min + xs.max() + 1)
                refined_y_min = int(y_min + ys.min())
                refined_y_max = int(y_min + ys.max() + 1)
                refined_boxes.append(
                    BoundingBox(
                        x_min=refined_x_min,
                        y_min=refined_y_min,
                        x_max=refined_x_max,
                        y_max=refined_y_max,
                        area=int(changed_pixels_inside),
                        mean_delta=after_obj.box.mean_delta,
                    )
                )
            else:
                refined_boxes.append(box)
        else:
            rejected.append(
                (
                    after_obj,
                    f"Low delta overlap ({overlap_ratio:.2f} < {min_delta_overlap}) - unchanged background",
                )
            )

    return ObjectMatchingReport(
        preserved_objects=preserved_pairs,
        added_rescue_objects=added_rescue,
        rejected_detections=rejected,
        preservation_score=preservation_score,
        confirmed_rescue_count=len(added_rescue),
        refined_boxes=refined_boxes,
    )
