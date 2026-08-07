import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from src.core.config import FLUX_MODEL, HF_TOKEN
def loading_model():
    """
    Load FLUX2-klein-4B once.
    """
    print("=" * 60)
    print(f"Loading {FLUX_MODEL}")
    print("=" * 60)
    pipe = Flux2KleinPipeline.from_pretrained(
        FLUX_MODEL,
        torch_dtype=torch.float16,
        # Neu chi co 1 GPU thi bo phan device_map, max_memory 
        token=HF_TOKEN
    )
    # Chi co 1 GPU: pipe.to("cuda")
    pipe.to("cuda") 
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
