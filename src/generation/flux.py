import torch
try:
    from diffusers import Flux2KleinPipeline
except ImportError:  # pragma: no cover - compatibility fallback
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

from src.core.config import FLUX_MODEL, HF_TOKEN


def loading_model(
    model_id: str = FLUX_MODEL,
    token: str | None = HF_TOKEN,
    device: str | None = None,
    torch_dtype: torch.dtype | None = None,
):
    """
    Load FLUX2-klein-4B once.
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_dtype = torch_dtype or (
        torch.float16 if resolved_device == "cuda" else torch.float32
    )
    resolved_token = token or None

    print("=" * 60)
    print(f"Loading {model_id}")
    print("=" * 60)
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=resolved_dtype,
        token=resolved_token,
    )
    pipe.to(resolved_device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe.set_progress_bar_config(
        disable=False,
    )
    print("=" * 60)
    print("FLUX Loaded Successfully")
    print("=" * 60)
    return pipe
