from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    torch = None
    CLIPModel = None
    CLIPProcessor = None

try:
    from PIL import Image
except ImportError:
    Image = None

from src.core.config import (
    CLIP_MODEL_ID,
    DISCARD_INCIDENT_CLASSES,
    DISASTER_CLIP_THRESHOLD,
    HF_HUB_CACHE,
    VALID_DISASTER_CLASSES,
    ensure_model_storage,
)


def is_valid_disaster_metadata(
    incidents: dict[str, Any] | None,
    damage: dict[str, Any] | None = None,
    valid_classes: Sequence[str] = VALID_DISASTER_CLASSES,
    discard_classes: Sequence[str] = DISCARD_INCIDENT_CLASSES,
) -> tuple[bool, str]:
    """
    Stage 1 filter: Inspect metadata dictionary from dataset (e.g. Incidents-1M).

    Returns:
        (is_valid, reason_str)
    """
    if not incidents or not isinstance(incidents, dict):
        return False, "Missing or empty incidents metadata"

    active_incidents = {
        cls.lower().strip()
        for cls, val in incidents.items()
        if val == 1 or val is True or str(val).lower() == "true"
    }

    if not active_incidents:
        return False, "No positive incident labels"

    discard_matches = active_incidents.intersection(
        {c.lower().strip() for c in discard_classes}
    )
    valid_matches = active_incidents.intersection(
        {c.lower().strip() for c in valid_classes}
    )

    if discard_matches and not valid_matches:
        return False, f"Non-disaster incident only: {', '.join(sorted(discard_matches))}"

    if not valid_matches:
        return False, f"No recognised natural disaster classes: {', '.join(sorted(active_incidents))}"

    if damage and isinstance(damage, dict):
        no_damage = damage.get("little_or_no_damage", 0) == 1
        has_severe = damage.get("severe_damage", 0) == 1
        has_mild = damage.get("mild_damage", 0) == 1
        if no_damage and not (has_severe or has_mild):
            return False, "Metadata indicates little or no physical damage"

    matched_str = ", ".join(sorted(valid_matches))
    return True, f"Valid disaster: {matched_str}"


DEFAULT_POSITIVE_PROMPTS = [
    "a close-up photo of severe natural disaster destruction, flooded streets, earthquake rubble, collapsed buildings, active wildfire devastation",
    "an emergency disaster impact zone showing physical destruction, flood waters, or fire damage",
]

DEFAULT_NEGATIVE_PROMPTS = [
    "a normal paved highway, everyday street view, traffic jam with cars waiting, distant horizon landscape",
    "an ordinary urban or residential road with undamaged vehicles and normal traffic flow",
    "a news headline infographic, text banner, map, or computer diagram",
]


@dataclass
class DisasterVisualFilter:
    """
    Stage 2 filter: Zero-shot visual classification using CLIP to reject
    traffic jams, highway distant shots, and normal urban photos.
    """
    model_name: str = CLIP_MODEL_ID
    threshold: float = DISASTER_CLIP_THRESHOLD
    device: str | None = None
    positive_prompts: Sequence[str] = tuple(DEFAULT_POSITIVE_PROMPTS)
    negative_prompts: Sequence[str] = tuple(DEFAULT_NEGATIVE_PROMPTS)
    _model: CLIPModel | None = None
    _processor: CLIPProcessor | None = None

    def load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        ensure_model_storage()
        device_str = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device_str

        self._processor = CLIPProcessor.from_pretrained(
            self.model_name,
            cache_dir=str(HF_HUB_CACHE),
        )
        self._model = CLIPModel.from_pretrained(
            self.model_name,
            cache_dir=str(HF_HUB_CACHE),
        ).to(device_str)
        self._model.eval()

    def classify_image(
        self,
        image: Image.Image,
    ) -> tuple[bool, float, dict[str, float]]:
        if self._model is None or self._processor is None:
            self.load()

        assert self._model is not None
        assert self._processor is not None

        prompts = list(self.positive_prompts) + list(self.negative_prompts)
        num_pos = len(self.positive_prompts)

        inputs = self._processor(
            text=prompts,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1)[0].cpu().numpy()

        pos_prob = float(probs[:num_pos].sum())
        neg_prob = float(probs[num_pos:].sum())

        details = {
            "disaster_prob": pos_prob,
            "non_disaster_prob": neg_prob,
            "threshold": self.threshold,
        }

        is_disaster = pos_prob >= self.threshold and pos_prob > neg_prob
        return is_disaster, pos_prob, details
