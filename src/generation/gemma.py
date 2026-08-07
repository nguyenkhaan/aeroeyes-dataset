import torch
from src.core.config import GENERAL_MODEL, HF_TOKEN

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
    resolved_dtype = torch_dtype or (
        torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    resolved_token = token or None

    vision_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=resolved_dtype,
        device_map=device_map,
        token=resolved_token,
    ).eval()
    vision_processor = AutoProcessor.from_pretrained(
        model_id,
        token=resolved_token,
    )
    return vision_model, vision_processor
