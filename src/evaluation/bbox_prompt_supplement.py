from __future__ import annotations

from src.evaluation.bbox_aliases import normalize_label


def validate_prompt_classes(prompts: list[dict], aliases: dict[str, str]) -> None:
    for item in prompts:
        if aliases.get(normalize_label(item["prompt"])) != item["class_name"]:
            raise ValueError(f"Supplemental prompt does not map to its target class: {item['prompt']}")


def detect_with_supplemental_prompts(base_prompt, prompts, detect, nms):
    detections = list(detect(base_prompt))
    counts = {}
    for item in prompts:
        raw = list(detect(item["prompt"]))
        selected = [detection for detection in raw if detection.label == item["class_name"]]
        retained = [
            detection for detection in selected
            if detection.confidence >= item.get("min_confidence", 0.0)
        ]
        counts[item["prompt"]] = {
            "raw": len(raw),
            "target_class": len(selected),
            "after_confidence": len(retained),
        }
        detections.extend(retained)
    return nms(detections), counts
