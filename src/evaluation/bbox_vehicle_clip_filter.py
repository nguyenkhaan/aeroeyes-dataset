from __future__ import annotations

import math

from src.evaluation.bbox_vehicle_clip import padded_crop_box


MODEL_ID = "openai/clip-vit-base-patch16"
MODEL_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
PROMPTS = ["a photo of an ambulance", "a photo of a fire truck"]
CONTEXT = 0.1
CONFLICT_IOU = 0.75


def conflict_indices(detections: list) -> list[int]:
    vehicles = [item for item in detections if item.label == "emergency_vehicle"]
    return [index for index, item in enumerate(detections)
            if item.label == "ambulance"
            and any(item.box.iou(vehicle.box) >= CONFLICT_IOU for vehicle in vehicles)]


def filter_scored_conflicts(detections: list, indices: list[int], scores: list) -> tuple[list, list]:
    if indices != conflict_indices(detections) or len(indices) != len(scores):
        raise ValueError("CLIP scores do not cover the conflicting detections")
    decisions = []
    for index, score in zip(indices, scores, strict=True):
        if (not isinstance(score, (list, tuple)) or len(score) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in score)):
            raise ValueError("CLIP similarities must be finite numeric pairs")
        ambulance, fire_truck = score
        box = detections[index].box
        decisions.append({"detection_index": index,
                          "box": [box.x_min, box.y_min, box.x_max, box.y_max],
                          "ambulance_similarity": ambulance, "fire_truck_similarity": fire_truck,
                          "margin": ambulance - fire_truck, "suppressed": ambulance <= fire_truck})
    removed = {decision["detection_index"] for decision in decisions if decision["suppressed"]}
    return [item for index, item in enumerate(detections) if index not in removed], decisions


class VehicleConflictClipFilter:
    def __init__(self, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.torch = torch

    def apply(self, image, detections: list) -> tuple[list, list]:
        indices = conflict_indices(detections)
        rgb = image.convert("RGB")
        crops = []
        for index in indices:
            box = detections[index].box
            coords = [box.x_min, box.y_min, box.x_max, box.y_max]
            crops.append(rgb.crop(padded_crop_box(coords, rgb.size, CONTEXT)))
        scores = []
        for start in range(0, len(crops), 16):
            inputs = self.processor(text=PROMPTS, images=crops[start:start + 16],
                                    return_tensors="pt", padding=True).to(self.device)
            with self.torch.inference_mode():
                output = self.model(**inputs)
                similarities = output.logits_per_image.float() / self.model.logit_scale.exp().float()
            scores.extend(similarities.cpu().tolist())
        return filter_scored_conflicts(detections, indices, scores)
