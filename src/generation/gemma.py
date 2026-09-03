import torch
from src.core.config import (
    GENERAL_MODEL,
    HF_HUB_CACHE,
    HF_TOKEN,
    ensure_model_storage,
)

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError:  # pragma: no cover - compatibility fallback
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration as AutoModelForImageTextToText

def loading_model(
    model_id: str = GENERAL_MODEL,
    token: str | None = HF_TOKEN,
    device_map: str = "auto",
    torch_dtype: torch.dtype | None = None,
):
    ensure_model_storage()
    resolved_dtype = torch_dtype or (
        torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    resolved_token = token or None

    vision_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype="auto",
        device_map=device_map,
        token=resolved_token,
        cache_dir=str(HF_HUB_CACHE),
    ).eval()
    vision_processor = AutoProcessor.from_pretrained(
        model_id,
        token=resolved_token,
        cache_dir=str(HF_HUB_CACHE),
    )
    return vision_model, vision_processor
