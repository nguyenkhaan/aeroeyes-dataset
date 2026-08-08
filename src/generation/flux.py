import torch

from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

from src.core.config import FLUX_MODEL, HF_TOKEN


def loading_model(
    model_id: str = FLUX_MODEL,
    token: str | None = HF_TOKEN,
    device: str | None = None,
    torch_dtype: torch.dtype | None = None,
):

    resolved_device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    resolved_dtype = torch_dtype or (
        torch.bfloat16
        if resolved_device == "cuda"
        else torch.float32
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 60)
    print(f"Loading {model_id}")
    print(f"dtype: {resolved_dtype}")
    print("=" * 60)

    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=resolved_dtype,
        token=token or None,
    )

    pipe.to(resolved_device)

    print(
        "VRAM:",
        round(
            torch.cuda.memory_allocated()/1024**3,
            2
        ),
        "GB"
    )

    pipe.set_progress_bar_config(
        disable=False,
    )

    print("=" * 60)
    print("FLUX Loaded Successfully")
    print("=" * 60)

    return pipe