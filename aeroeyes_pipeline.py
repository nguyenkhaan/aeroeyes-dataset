"""aeroeyes_pipeline.py - single-file, Kaggle-friendly runner for the rescue-image pipeline.

Why this file exists
--------------------
`main.py` + `src/` target the university Slurm/VPS box (CUDA 12.8, >=32 GB VRAM,
shared cache under /datastore, faulthandler stage watchdogs). That stack is
currently too unstable to produce a report, so this script reproduces the same
pipeline in one file that runs on a stock Kaggle GPU notebook (2x T4 16 GB, or
1x P100). It is standalone (no import from `src/`), but the CSV / JSONL / per-image
metadata field names match `main.py` so the outputs stay comparable.

Pipeline (unchanged from the VPS version, incl. the fix/watermark merge):
    dataset -> filter positive incident labels -> download
    -> watermark / caption removal (EasyOCR + Simple LaMa) -> resize/center-crop
    -> Gemma scene description -> Gemma rescue instruction -> build FLUX prompt
    -> FLUX image edit -> quality gate (O-score, SSIM; SC uses Long-CLIP, 248 tok)
    -> save image + metadata -> optional dataset CMMD.

Toggles: WATERMARK_REMOVAL_ENABLED, LONG_CLIP_ENABLED (both default true; set to
"0" if EasyOCR / Simple LaMa / the Long-CLIP checkpoint cannot be installed).

How to run on Kaggle
--------------------
1. Notebook settings: Accelerator = "GPU T4 x2", Internet = ON.
2. Add-ons -> Secrets: add ``HF_TOKEN`` (token from an account that already
   accepted the google/gemma-* and black-forest-labs/FLUX.2-* licenses).
3. Attach the incidents-1M dataset that contains ``eccv_train.json`` (or export
   ``JSON_PATH``). Attach this repo too if you want CMMD (needs ``cmmd-pytorch/``).
4. Run one cell:
       !python aeroeyes_pipeline.py
   or paste the file into a cell and call ``main()``.
5. Outputs land in ``/kaggle/working/aeroeyes/`` (images, rejected/, CSV, JSONL,
   reports/kaggle_run_summary.md, config_used.json).

Every setting below is overridable with an environment variable - see CONFIG.
SDQM is intentionally not run here (it needs third_party/SDQM + the custom
ultralytics fork on Linux); use ``scripts/run_evaluation.py`` on the VPS for that.
"""

from __future__ import annotations

import contextlib
import gc
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Environment detection + config
# ---------------------------------------------------------------------------

ON_KAGGLE = Path("/kaggle").is_dir() or os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in ("1", "true", "yes", "on")


if ON_KAGGLE:
    WORK_DIR = Path(_env("AEROEYES_WORK", "/kaggle/working/aeroeyes"))
    MODEL_DIR = Path(_env("AEROEYES_MODEL_DIR", "/kaggle/temp/aeroeyes_models"))
else:
    WORK_DIR = Path(_env("AEROEYES_WORK", "data")).resolve()
    MODEL_DIR = Path(_env("AEROEYES_MODEL_DIR", str(WORK_DIR / "models")))

OUTPUT_DIR = Path(_env("OUTPUT_DIR", str(WORK_DIR / "output")))
REAL_IMAGES_DIR = Path(_env("REAL_IMAGES_DIR", str(WORK_DIR / "real_reference")))
GEN_IMAGES_DIR = Path(_env("GEN_IMAGES_DIR", str(WORK_DIR / "gen_reference")))
REPORT_DIR = Path(_env("AEROEYES_REPORT_DIR", str(WORK_DIR / "reports")))

# HF caches must be set before transformers / diffusers import.
os.environ.setdefault("HF_HOME", str(MODEL_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_DIR / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(MODEL_DIR / "torch"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Models (Kaggle defaults follow humaninstruction-gamm3-4.ipynb, which ran on 2x T4).
GEMMA_MODEL = _env("GENERAL_MODEL", "google/gemma-3-4b-it")
FLUX_MODEL = _env("FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B")
CLIP_MODEL_ID = _env("CLIP_MODEL_ID", "openai/clip-vit-base-patch16")

# Generation params
IMAGE_SIZE = _int("IMAGE_SIZE", 1024)
LIMIT_IMAGES = _int("LIMIT_IMAGES", 3 if ON_KAGGLE else 1)
MAX_ATTEMPTS = _int("MAX_ATTEMPTS", 12)
MAX_CONSECUTIVE_ERRORS = _int("MAX_CONSECUTIVE_ERRORS", 4)
GENERATION_TIMEOUT_SECONDS = _int("GENERATION_TIMEOUT_SECONDS", 3600)
REQUEST_TIMEOUT = _int("REQUEST_TIMEOUT", 30)
DOWNLOAD_RETRIES = _int("DOWNLOAD_RETRIES", 3)
MAX_NEW_TOKENS = _int("MAX_NEW_TOKENS", 256)
NUM_INFERENCE_STEPS = _int("NUM_INFERENCE_STEPS", 8)
GUIDANCE_SCALE = _float("GUIDANCE_SCALE", 2.5)
BASE_SEED = _int("BASE_SEED", 50)

# Quality gate (matches src/core/config.py)
QUALITY_GATE_ENABLED = _bool("QUALITY_GATE_ENABLED", True)
O_SCORE_THRESHOLD = _float("O_SCORE_THRESHOLD", 0.55)
SSIM_MAX_THRESHOLD = _float("SSIM_MAX_THRESHOLD", 0.90)
SC_NORM_DIVISOR = _float("SC_NORM_DIVISOR", 40.0)
KEEP_REJECTED = _bool("KEEP_REJECTED", True)

# Long-CLIP for the SC score (248 tokens instead of 77, so the full FLUX prompt
# is scored). Mirrors src/evaluation/quality.py from the fix/watermark merge.
LONG_CLIP_ENABLED = _bool("LONG_CLIP_ENABLED", True)
LONG_CLIP_MODEL_ID = _env("LONG_CLIP_MODEL_ID", "zer0int/LongCLIP-L-Diffusers")
LONG_CLIP_MAX_TOKENS = _int("LONG_CLIP_MAX_TOKENS", 248)
_BASE_CLIP_MAX_TOKENS = 77

# Watermark / caption removal (EasyOCR detection + Simple LaMa inpainting) on the
# freshly downloaded image, before resize/crop. Mirrors src/helper/watermark.py.
WATERMARK_REMOVAL_ENABLED = _bool("WATERMARK_REMOVAL_ENABLED", True)
WATERMARK_OCR_LANGUAGES = tuple(
    lang.strip() for lang in _env("WATERMARK_OCR_LANGUAGES", "en").split(",") if lang.strip()
) or ("en",)
WATERMARK_OCR_USE_GPU = _bool("WATERMARK_OCR_USE_GPU", True)
WATERMARK_MIN_TEXT_CONFIDENCE = _float("WATERMARK_MIN_TEXT_CONFIDENCE", 0.2)
WATERMARK_EDGE_MARGIN_RATIO = _float("WATERMARK_EDGE_MARGIN_RATIO", 0.12)
WATERMARK_WIDE_ASPECT_RATIO = _float("WATERMARK_WIDE_ASPECT_RATIO", 3.5)
WATERMARK_DILATE_KERNEL = _int("WATERMARK_DILATE_KERNEL", 7)
WATERMARK_DILATE_ITERATIONS = _int("WATERMARK_DILATE_ITERATIONS", 2)

# Model-loading / memory knobs
GEMMA_DTYPE = _env("GEMMA_DTYPE", "bfloat16")
GEMMA_ATTN = _env("GEMMA_ATTN", "eager")
GEMMA_DEVICE_MAP = _env("GEMMA_DEVICE_MAP", "")
QUANTIZE_GEMMA = _bool("QUANTIZE_GEMMA", False)
FLUX_MODE = _env("FLUX_MODE", "")  # "" auto | balanced | cpu_offload | sequential_offload | cuda

# Dataset-level metrics
RUN_CMMD = _bool("RUN_CMMD", True)
CMMD_REPO_DIR = _env("CMMD_REPO_DIR", "cmmd-pytorch")
CMMD_BATCH_SIZE = _int("CMMD_BATCH_SIZE", 8)

AUTO_PIP = _bool("AEROEYES_AUTO_PIP", ON_KAGGLE)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/137.0 Safari/537.36"
    )
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


@contextlib.contextmanager
def stage(name: str):
    """Log START / END / FAILED with elapsed time. No hard process exit (unlike the VPS)."""
    started = time.monotonic()
    log(f"START {name}")
    try:
        yield
    except Exception:
        log(f"FAILED {name} ({time.monotonic() - started:.1f}s)")
        raise
    log(f"END {name} ({time.monotonic() - started:.1f}s)")


# ---------------------------------------------------------------------------
# Package / token / dataset bootstrap
# ---------------------------------------------------------------------------

def ensure_packages() -> None:
    wanted = [("pyiqa", "pyiqa"), ("skimage", "scikit-image"), ("diffusers", "diffusers")]
    if WATERMARK_REMOVAL_ENABLED:
        wanted += [("easyocr", "easyocr"), ("simple_lama_inpainting", "simple-lama-inpainting")]
    missing = []
    for module_name, pip_name in wanted:
        try:
            __import__(module_name)
        except Exception:
            missing.append(pip_name)
    if missing and AUTO_PIP:
        log(f"pip install {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)


def resolve_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as exc:
            log(f"No HF token from Kaggle secrets ({exc}).")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        log("HF token resolved.")
    else:
        log("WARNING: no HF token; gated model downloads will fail.")
    return token


def resolve_json_path() -> Path:
    explicit = os.getenv("JSON_PATH")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    roots = ["/kaggle/input", str(WORK_DIR / "input"), "data/input", "."]
    for root in roots:
        matches = glob.glob(f"{root}/**/eccv_train.json", recursive=True)
        if matches:
            return Path(sorted(matches, key=len)[0])
    raise FileNotFoundError(
        "eccv_train.json not found. Set JSON_PATH or attach the incidents-1M dataset."
    )


def load_dataset(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("Expected the dataset JSON to be an object keyed by image id.")
    log(f"Dataset: {json_path} ({len(data)} samples)")
    only = _env("SAMPLE_KEYS", "")
    if only:
        keys = [k.strip() for k in only.split(",") if k.strip()]
        data = {k: data[k] for k in keys if k in data}
        log(f"SAMPLE_KEYS filter -> {len(data)} samples")
    return data


# ---------------------------------------------------------------------------
# Image helpers (mirror src/helper/image.py)
# ---------------------------------------------------------------------------

def download_image(url: str, retries: int = DOWNLOAD_RETRIES) -> bytes | None:
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            log(f"[download retry {attempt + 1}/{retries}] {exc}")
    return None


def resize_center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = max(size / width, size / height)
    resized = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def make_safe_stem(image_key: str) -> str:
    normalized = image_key.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_")


# ---------------------------------------------------------------------------
# Watermark / caption removal (mirror src/helper/watermark.py)
# EasyOCR locates text, a heuristic keeps only watermark-like boxes (near an
# edge or a long thin strip), Simple LaMa inpaints them. Tools load lazily and
# stay cached; unload_watermark_tools() frees their VRAM before CMMD.
# ---------------------------------------------------------------------------

_wm_reader = None
_wm_inpainter = None


def _load_watermark_tools():
    global _wm_reader, _wm_inpainter
    if _wm_reader is None or _wm_inpainter is None:
        import easyocr
        from simple_lama_inpainting import SimpleLama

        log("Loading watermark tools (EasyOCR + Simple LaMa)...")
        _wm_reader = easyocr.Reader(list(WATERMARK_OCR_LANGUAGES), gpu=WATERMARK_OCR_USE_GPU)
        _wm_inpainter = SimpleLama()
        log("Watermark tools ready.")
    return _wm_reader, _wm_inpainter


def unload_watermark_tools() -> None:
    global _wm_reader, _wm_inpainter
    _wm_reader = None
    _wm_inpainter = None


def create_auto_watermark_mask(image: Image.Image) -> Image.Image:
    """Return an ``L`` mask (255 = remove) covering detected watermark text."""
    import cv2

    reader, _ = _load_watermark_tools()
    image_np = np.array(image.convert("RGB"))
    height, width = image_np.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for bbox, _text, confidence in reader.readtext(image_np):
        if confidence < WATERMARK_MIN_TEXT_CONFIDENCE:
            continue
        points = np.array(bbox, dtype=np.int32)
        x_min, x_max = int(points[:, 0].min()), int(points[:, 0].max())
        y_min, y_max = int(points[:, 1].min()), int(points[:, 1].max())
        box_width = x_max - x_min
        box_height = y_max - y_min

        near_edge = (
            x_min < WATERMARK_EDGE_MARGIN_RATIO * width
            or x_max > (1.0 - WATERMARK_EDGE_MARGIN_RATIO) * width
            or y_min < WATERMARK_EDGE_MARGIN_RATIO * height
            or y_max > (1.0 - WATERMARK_EDGE_MARGIN_RATIO) * height
        )
        wide_strip = (box_width / (box_height + 1e-5)) > WATERMARK_WIDE_ASPECT_RATIO
        if near_edge or wide_strip:
            cv2.fillPoly(mask, [points], 255)

    if WATERMARK_DILATE_KERNEL > 0 and WATERMARK_DILATE_ITERATIONS > 0:
        kernel = np.ones((WATERMARK_DILATE_KERNEL, WATERMARK_DILATE_KERNEL), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=WATERMARK_DILATE_ITERATIONS)

    return Image.fromarray(mask).convert("L")


def remove_watermark(image: Image.Image) -> Image.Image:
    """Inpaint detected watermark text; returns the input unchanged when none is found."""
    mask = create_auto_watermark_mask(image)
    if int(np.asarray(mask).max()) == 0:
        return image

    _, inpainter = _load_watermark_tools()
    rgb_image = image.convert("RGB")
    restored = inpainter(rgb_image, mask)
    if not isinstance(restored, Image.Image):
        restored = Image.fromarray(np.asarray(restored))
    restored = restored.convert("RGB")
    if restored.size != rgb_image.size:
        restored = restored.crop((0, 0, rgb_image.size[0], rgb_image.size[1]))
    return restored


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _from_pretrained(cls, model_id, *, dtype=None, **kwargs):
    """`from_pretrained` tolerant of the dtype kwarg name and unsupported device_map."""
    dtype_variants = [{}] if dtype is None else [{"dtype": dtype}, {"torch_dtype": dtype}]
    kwarg_variants = [kwargs]
    trimmed = {k: v for k, v in kwargs.items() if k not in ("device_map", "max_memory")}
    if trimmed != kwargs:
        kwarg_variants.append(trimmed)
    last_error: Exception | None = None
    for extra_kwargs in kwarg_variants:
        for dtype_kwargs in dtype_variants:
            try:
                return cls.from_pretrained(model_id, **dtype_kwargs, **extra_kwargs)
            except TypeError as exc:
                last_error = exc
    raise last_error if last_error else RuntimeError("from_pretrained failed")


def load_gemma(token: str | None):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as GemmaModel
    except ImportError:  # older transformers
        from transformers import Gemma3ForConditionalGeneration as GemmaModel

    gpu_count = torch.cuda.device_count()
    kwargs: dict = {"token": token or None, "low_cpu_mem_usage": True}
    if GEMMA_ATTN:
        kwargs["attn_implementation"] = GEMMA_ATTN

    if QUANTIZE_GEMMA:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = GEMMA_DEVICE_MAP or "auto"
        model = _from_pretrained(GemmaModel, GEMMA_MODEL, dtype=None, **kwargs)
    else:
        dtype = getattr(torch, GEMMA_DTYPE, torch.bfloat16)
        if GEMMA_DEVICE_MAP:
            kwargs["device_map"] = (
                GEMMA_DEVICE_MAP
                if GEMMA_DEVICE_MAP in ("auto", "balanced", "balanced_low_0", "sequential")
                else {"": GEMMA_DEVICE_MAP}
            )
        elif gpu_count:
            kwargs["device_map"] = {"": 0}  # keep the whole model on cuda:0, leave cuda:1 for FLUX
        model = _from_pretrained(GemmaModel, GEMMA_MODEL, dtype=dtype, **kwargs)
        if not gpu_count:
            model.to("cpu")

    model.eval()
    processor = AutoProcessor.from_pretrained(GEMMA_MODEL, token=token or None)
    return model, processor


def _flux_pipeline_cls():
    try:
        from diffusers import Flux2KleinPipeline

        return Flux2KleinPipeline
    except Exception:
        pass
    try:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

        return Flux2KleinPipeline
    except Exception:
        pass
    if AUTO_PIP:
        log("Flux2KleinPipeline missing - upgrading diffusers")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers"], check=False)
        with contextlib.suppress(Exception):
            from diffusers import Flux2KleinPipeline

            return Flux2KleinPipeline
    from diffusers import DiffusionPipeline

    log("Falling back to DiffusionPipeline for FLUX (Flux2KleinPipeline unavailable).")
    return DiffusionPipeline


def load_flux(token: str | None):
    import torch

    pipeline_cls = _flux_pipeline_cls()
    gpu_count = torch.cuda.device_count()
    mode = FLUX_MODE or ("balanced" if gpu_count >= 2 else "cpu_offload" if gpu_count == 1 else "cpu")
    dtype = torch.bfloat16 if gpu_count else torch.float32
    log(f"FLUX load mode: {mode} (gpus={gpu_count})")

    if mode == "balanced":
        max_memory = {
            index: _env(f"FLUX_MAX_MEM_{index}", "12GiB") for index in range(gpu_count)
        }
        pipe = _from_pretrained(
            pipeline_cls,
            FLUX_MODEL,
            dtype=dtype,
            token=token or None,
            device_map="balanced",
            max_memory=max_memory,
        )
    else:
        pipe = _from_pretrained(pipeline_cls, FLUX_MODEL, dtype=dtype, token=token or None)
        if mode == "cpu_offload":
            pipe.enable_model_cpu_offload()
        elif mode == "sequential_offload":
            pipe.enable_sequential_cpu_offload()
        elif mode == "cuda":
            pipe.to("cuda")

    for method in ("enable_vae_slicing", "enable_vae_tiling", "enable_attention_slicing"):
        with contextlib.suppress(Exception):
            getattr(pipe, method)()
    with contextlib.suppress(Exception):
        pipe.set_progress_bar_config(disable=False)
    return pipe


# ---------------------------------------------------------------------------
# Vision steps (mirror src/vision/*)
# ---------------------------------------------------------------------------

SCENE_SYSTEM = (
    "You are a Vision-Language AI assistant specialized in disaster scene "
    "understanding and image editing instruction generation."
)
SCENE_PROMPT = """
You are a professional disaster scene analysis assistant.
Your task is ONLY to describe what is directly visible in the image.
Rules:
- Describe only visible objects.
- Do not infer hidden information.
- Do not speculate.
- Do not explain the cause of the disaster.
- Do not suggest rescue actions.
- Do not mention anything not visible.
- Return a single factual paragraph.
""".strip()

INSTRUCTION_SYSTEM = """
You are an expert emergency rescue planner.

Your task is to generate editing instructions for an image editing model.
Requirements:
1. Preserve the original disaster scene.
2. Preserve damaged buildings and existing objects.
3. Do not change the disaster type.
4. Add only realistic rescue operations.
5. Add rescue personnel when appropriate.
6. Add rescue vehicles when appropriate.
7. Add emergency equipment when appropriate.
8. Maintain realistic object scale.
9. Maintain realistic lighting.
10. Maintain realistic perspective.
11. Keep all newly added objects consistent with the existing environment.

Return ONLY the editing instructions.
Do not explain your reasoning.
Do not describe the original image.
Do not use markdown.
""".strip()


def _gemma_generate(model, processor, messages, max_new_tokens: int) -> str:
    import torch

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()


def generate_scene_description(model, processor, image: Image.Image) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SCENE_SYSTEM}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": SCENE_PROMPT}],
        },
    ]
    return _gemma_generate(model, processor, messages, MAX_NEW_TOKENS)


def generate_rescue_instruction(model, processor, scene_description: str) -> str:
    user_prompt = f"Disaster Scene:\n{scene_description}\nGenerate image editing instructions."
    messages = [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    return _gemma_generate(model, processor, messages, MAX_NEW_TOKENS)


def build_flux_prompt(scene_description: str, rescue_instruction: str) -> str:
    return f"""
You are editing an existing disaster photograph.

Original Scene
--------------
{scene_description}
Editing Instructions
--------------------
{rescue_instruction}
Requirements
    - Preserve the original disaster scene.
    - Preserve all existing buildings, vehicles, roads and environmental objects.
    - Do not change the disaster type.
    - Add only realistic rescue operations.
    - Blend newly added rescue personnel, vehicles and equipment naturally.
    - Maintain realistic lighting, shadows and perspective.
    - Maintain correct object proportions.
    - Generate anatomically correct humans.
    - Produce seamless image editing without visible artifacts.
    Style
    - Documentary disaster photography
    - Photojournalism
    - Real-world emergency response
    - Natural color grading
    - Authentic textures
    - High realism
    - Non-cinematic
""".strip()


def generate_rescue_image(pipe, image: Image.Image, prompt: str, seed: int) -> Image.Image:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=prompt,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            output_type="pil",
        )
    generated = result.images[0]
    del result, generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated


# ---------------------------------------------------------------------------
# Quality metrics (mirror src/evaluation/quality.py)
# ---------------------------------------------------------------------------

def load_evaluators():
    import torch
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_text_tokens = _BASE_CLIP_MAX_TOKENS

    if LONG_CLIP_ENABLED:
        try:
            config = CLIPConfig.from_pretrained(LONG_CLIP_MODEL_ID)
            config.text_config.max_position_embeddings = LONG_CLIP_MAX_TOKENS
            clip_model = CLIPModel.from_pretrained(LONG_CLIP_MODEL_ID, config=config)
            clip_processor = CLIPProcessor.from_pretrained(
                LONG_CLIP_MODEL_ID, padding="max_length", max_length=LONG_CLIP_MAX_TOKENS
            )
            max_text_tokens = LONG_CLIP_MAX_TOKENS
            log(f"SC score: Long-CLIP enabled ({LONG_CLIP_MODEL_ID}, {max_text_tokens} tokens).")
        except Exception as exc:
            log(f"Long-CLIP unavailable, falling back to {CLIP_MODEL_ID} ({exc}).")
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
            clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
            max_text_tokens = _BASE_CLIP_MAX_TOKENS
    else:
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    clip_model = clip_model.to(device).eval()
    clip_iqa = None
    try:
        import pyiqa

        clip_iqa = pyiqa.create_metric("clipiqa", device=device)
    except Exception as exc:
        log(f"pyiqa unavailable - PQ score disabled ({exc}).")
    return {
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "clip_iqa": clip_iqa,
        "device": device,
        "max_text_tokens": max_text_tokens,
    }


def evaluate_quality(evaluators, image: Image.Image, prompt: str) -> tuple[float, float]:
    import torch

    pq_score = float("nan")
    if evaluators["clip_iqa"] is not None:
        from torchvision import transforms

        tensor = transforms.ToTensor()(image).unsqueeze(0).to(evaluators["device"])
        with torch.no_grad():
            pq_score = float(evaluators["clip_iqa"](tensor).item())

    inputs = evaluators["clip_processor"](
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding="max_length",
        max_length=evaluators.get("max_text_tokens", _BASE_CLIP_MAX_TOKENS),
        truncation=True,
    ).to(evaluators["device"])
    with torch.no_grad():
        outputs = evaluators["clip_model"](**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        sc_raw = float(torch.matmul(image_embeds, text_embeds.T).item()) * 100.0
    return sc_raw, pq_score


def compute_o_score(sc_score: float, pq_score: float) -> float:
    sc_norm = max(0.0, min(1.0, sc_score / SC_NORM_DIVISOR))
    if pq_score != pq_score:  # NaN -> fall back to the semantic-consistency term only
        return sc_norm
    return min(sc_norm, pq_score)


def compute_ssim(original: Image.Image, generated: Image.Image) -> float:
    try:
        from skimage.metrics import structural_similarity

        if generated.size != original.size:
            generated = generated.resize(original.size, Image.Resampling.LANCZOS)
        return float(
            structural_similarity(
                np.array(original), np.array(generated), channel_axis=-1, data_range=255
            )
        )
    except Exception as exc:
        log(f"SSIM unavailable ({exc}).")
        return float("nan")


def passes_quality_gate(o_score: float, ssim_value: float) -> bool:
    if o_score < O_SCORE_THRESHOLD:
        return False
    if ssim_value == ssim_value and ssim_value > SSIM_MAX_THRESHOLD:
        return False
    return True


# ---------------------------------------------------------------------------
# Optional dataset CMMD (mirror src/evaluation/cmmd.py)
# ---------------------------------------------------------------------------

def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def maybe_run_cmmd() -> float | None:
    if not RUN_CMMD:
        return None
    if _count_images(REAL_IMAGES_DIR) < 2 or _count_images(GEN_IMAGES_DIR) < 2:
        log("CMMD skipped: need >=2 real and >=2 generated reference images.")
        return None
    cmmd_main = Path(CMMD_REPO_DIR) / "main.py"
    if not cmmd_main.is_file():
        log(f"CMMD skipped: {cmmd_main} not found (attach the repo or set CMMD_REPO_DIR).")
        return None
    try:
        import importlib.util

        repo_root = str(Path(CMMD_REPO_DIR).resolve())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        spec = importlib.util.spec_from_file_location("cmmd_pytorch_main", cmmd_main)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with stage("CMMD report"):
            score = float(
                module.compute_cmmd(
                    ref_dir=str(REAL_IMAGES_DIR),
                    eval_dir=str(GEN_IMAGES_DIR),
                    batch_size=CMMD_BATCH_SIZE,
                    max_count=-1,
                )
            )
        log(f"CMMD score: {score:.4f}")
        return score
    except Exception as exc:
        log(f"CMMD failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def free_memory(*objects) -> None:
    for obj in objects:
        with contextlib.suppress(Exception):
            del obj
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def runtime_banner() -> dict:
    info: dict = {"on_kaggle": ON_KAGGLE, "python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def config_snapshot() -> dict:
    return {
        "gemma_model": GEMMA_MODEL,
        "flux_model": FLUX_MODEL,
        "clip_model": CLIP_MODEL_ID,
        "image_size": IMAGE_SIZE,
        "limit_images": LIMIT_IMAGES,
        "max_attempts": MAX_ATTEMPTS,
        "max_consecutive_errors": MAX_CONSECUTIVE_ERRORS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "base_seed": BASE_SEED,
        "quality_gate_enabled": QUALITY_GATE_ENABLED,
        "o_score_threshold": O_SCORE_THRESHOLD,
        "ssim_max_threshold": SSIM_MAX_THRESHOLD,
        "keep_rejected": KEEP_REJECTED,
        "flux_mode": FLUX_MODE or "auto",
        "quantize_gemma": QUANTIZE_GEMMA,
        "long_clip_enabled": LONG_CLIP_ENABLED,
        "long_clip_model": LONG_CLIP_MODEL_ID if LONG_CLIP_ENABLED else None,
        "long_clip_max_tokens": LONG_CLIP_MAX_TOKENS if LONG_CLIP_ENABLED else _BASE_CLIP_MAX_TOKENS,
        "watermark_removal_enabled": WATERMARK_REMOVAL_ENABLED,
        "watermark_ocr_gpu": WATERMARK_OCR_USE_GPU,
        "run_cmmd": RUN_CMMD,
        "output_dir": str(OUTPUT_DIR),
    }


def write_report(records: list[dict], counts: dict, cmmd_score: float | None, runtime: dict) -> Path:
    import pandas as pd

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(records)
    if not frame.empty:
        frame.to_csv(OUTPUT_DIR / "evaluation_report.csv", index=False)
    with (OUTPUT_DIR / "evaluation_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (REPORT_DIR / "config_used.json").write_text(
        json.dumps({"config": config_snapshot(), "runtime": runtime}, indent=2), encoding="utf-8"
    )

    gpus = ", ".join(runtime.get("gpus", [])) or "none"
    lines = [
        "# AeroEyes - Kaggle run summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Environment: Kaggle={ON_KAGGLE}, GPUs={runtime.get('gpu_count', 0)} ({gpus})",
        f"- Torch {runtime.get('torch', '?')} / CUDA {runtime.get('torch_cuda', '?')}",
        f"- Gemma: `{GEMMA_MODEL}`",
        f"- FLUX: `{FLUX_MODEL}` (mode={FLUX_MODE or 'auto'})",
        f"- Params: steps={NUM_INFERENCE_STEPS}, guidance={GUIDANCE_SCALE}, "
        f"image_size={IMAGE_SIZE}, seed={BASE_SEED}",
        f"- Limits: LIMIT_IMAGES={LIMIT_IMAGES}, MAX_ATTEMPTS={MAX_ATTEMPTS}, "
        f"MAX_CONSECUTIVE_ERRORS={MAX_CONSECUTIVE_ERRORS}",
        f"- Quality gate: enabled={QUALITY_GATE_ENABLED}, O>={O_SCORE_THRESHOLD}, "
        f"SSIM<={SSIM_MAX_THRESHOLD}",
        f"- Watermark removal: {WATERMARK_REMOVAL_ENABLED} (EasyOCR + Simple LaMa)",
        f"- SC score: {'Long-CLIP ' + LONG_CLIP_MODEL_ID + f' ({LONG_CLIP_MAX_TOKENS} tok)' if LONG_CLIP_ENABLED else CLIP_MODEL_ID + ' (77 tok)'}",
        "",
        "## Counts",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| attempted | {counts['attempted']} |",
        f"| accepted | {counts['accepted']} |",
        f"| rejected by gate | {counts['rejected']} |",
        f"| skipped (ineligible/exists) | {counts['skipped']} |",
        f"| errors | {counts['errors']} |",
        f"| stop reason | {counts['stop_reason']} |",
        "",
        "## Per-image",
        "",
        "| image_key | labels | sc_score | pq_score | o_score | ssim | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| {image_key} | {labels} | {sc} | {pq} | {o} | {ssim} | {status} |".format(
                image_key=record.get("image_key", ""),
                labels=",".join(record.get("labels", [])),
                sc=_fmt(record.get("sc_score")),
                pq=_fmt(record.get("pq_score")),
                o=_fmt(record.get("o_score")),
                ssim=_fmt(record.get("ssim")),
                status=record.get("quality_status", ""),
            )
        )
    lines += ["", "## Aggregate", ""]
    metric_cols = [c for c in ("sc_score", "pq_score", "o_score", "ssim") if c in frame.columns]
    if metric_cols:
        lines += ["```", frame[metric_cols].describe().to_string(), "```", ""]
    else:
        lines += ["_No per-image metrics recorded._", ""]

    lines += [
        "## Dataset metrics",
        "",
        f"- CMMD: {cmmd_score:.4f}" if cmmd_score is not None else "- CMMD: skipped",
        "- SDQM: not run on Kaggle (needs third_party/SDQM + custom ultralytics on Linux). "
        "Use `scripts/run_evaluation.py` on the VPS.",
        "",
        "## Outputs",
        "",
        f"- Accepted images: `{OUTPUT_DIR}`",
        f"- Rejected images: `{OUTPUT_DIR / 'rejected'}`",
        f"- Real / generated references: `{REAL_IMAGES_DIR}` / `{GEN_IMAGES_DIR}`",
        f"- CSV: `{OUTPUT_DIR / 'evaluation_report.csv'}`",
        f"- JSONL: `{OUTPUT_DIR / 'evaluation_metadata.jsonl'}`",
        "",
    ]
    report_path = REPORT_DIR / "kaggle_run_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _fmt(value) -> str:
    try:
        if value is None or value != value:
            return "n/a"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    print("Hello, Cloudian - Kaggle run")
    for directory in (WORK_DIR, OUTPUT_DIR, REAL_IMAGES_DIR, GEN_IMAGES_DIR, REPORT_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "rejected").mkdir(parents=True, exist_ok=True)
    metadata_dir = OUTPUT_DIR / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    ensure_packages()
    token = resolve_hf_token()
    runtime = runtime_banner()
    log(f"Runtime: {runtime}")
    log(f"Config: {config_snapshot()}")

    seed_everything(BASE_SEED)
    data = load_dataset(resolve_json_path())

    with stage("Load Gemma"):
        gemma_model, gemma_processor = load_gemma(token)
    with stage("Load FLUX"):
        flux_pipe = load_flux(token)
    evaluators = None
    if QUALITY_GATE_ENABLED:
        with stage("Load quality evaluators"):
            evaluators = load_evaluators()

    records: list[dict] = []
    counts = {
        "attempted": 0,
        "accepted": 0,
        "rejected": 0,
        "skipped": 0,
        "errors": 0,
        "stop_reason": "target reached",
    }
    consecutive_errors = 0
    started = time.monotonic()

    print("=" * 80)
    print(f"Pipeline started - target: {LIMIT_IMAGES} images")
    print("=" * 80)

    for image_key, image_info in data.items():
        if counts["accepted"] >= LIMIT_IMAGES:
            break
        if counts["attempted"] >= MAX_ATTEMPTS:
            counts["stop_reason"] = f"MAX_ATTEMPTS reached ({counts['attempted']})"
            break
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            counts["stop_reason"] = f"MAX_CONSECUTIVE_ERRORS reached ({consecutive_errors})"
            break
        if time.monotonic() - started >= GENERATION_TIMEOUT_SECONDS:
            counts["stop_reason"] = "generation time budget exhausted"
            break

        print("=" * 80)
        print(f"Processing: {image_key}")
        print("=" * 80)

        try:
            incidents = image_info.get("incidents", {})
            positive_incidents = [name for name, value in incidents.items() if value == 1]
            if not positive_incidents:
                log(f"Skip {image_key}: no positive labels")
                counts["skipped"] += 1
                continue

            safe_name = make_safe_stem(image_key)
            if (OUTPUT_DIR / f"{safe_name}.png").exists():
                log(f"Skip {image_key}: already generated")
                counts["skipped"] += 1
                continue

            url = image_info.get("url")
            if not url:
                log(f"Skip {image_key}: missing URL")
                counts["skipped"] += 1
                continue

            counts["attempted"] += 1
            with stage("Download image"):
                content = download_image(url)
            if content is None:
                log("Skip: download failed")
                consecutive_errors += 1
                counts["errors"] += 1
                continue

            try:
                original_image = Image.open(BytesIO(content)).convert("RGB")
            except Exception as exc:
                log(f"Skip invalid image: {exc}")
                consecutive_errors += 1
                counts["errors"] += 1
                continue

            if WATERMARK_REMOVAL_ENABLED:
                try:
                    with stage("Watermark removal"):
                        original_image = remove_watermark(original_image)
                except Exception as exc:
                    log(f"Watermark removal skipped: {exc}")

            original_image = resize_center_crop(original_image, IMAGE_SIZE)

            with stage("Gemma scene description"):
                scene_description = generate_scene_description(
                    gemma_model, gemma_processor, original_image
                )
            print("\nScene Description\n" + "-" * 60 + f"\n{scene_description}")

            with stage("Gemma rescue instruction"):
                rescue_instruction = generate_rescue_instruction(
                    gemma_model, gemma_processor, scene_description
                )
            print("\nEditing Instruction\n" + "-" * 60 + f"\n{rescue_instruction}")

            flux_prompt = build_flux_prompt(scene_description, rescue_instruction)

            with stage("FLUX generation"):
                generated_image = generate_rescue_image(
                    flux_pipe, original_image, flux_prompt, BASE_SEED + counts["accepted"]
                )

            sc_score, pq_score = (float("nan"), float("nan"))
            o_score, ssim_value = (float("nan"), float("nan"))
            if evaluators is not None:
                with stage("Quality evaluation"):
                    sc_score, pq_score = evaluate_quality(evaluators, generated_image, flux_prompt)
                    o_score = compute_o_score(sc_score, pq_score)
                    ssim_value = compute_ssim(original_image, generated_image)
                print(f"O_Score: {o_score:.4f} | SSIM: {ssim_value:.4f}")

            gate_ok = (not QUALITY_GATE_ENABLED) or passes_quality_gate(o_score, ssim_value)
            status = "accepted" if gate_ok else "rejected"
            target_dir = OUTPUT_DIR if gate_ok else (OUTPUT_DIR / "rejected")
            output_path = target_dir / f"{safe_name}.png"

            if gate_ok or KEEP_REJECTED:
                generated_image.save(output_path)
                if gate_ok:
                    original_image.save(REAL_IMAGES_DIR / f"{safe_name}_real.jpg", quality=95)
                    generated_image.save(GEN_IMAGES_DIR / f"{safe_name}_gen.jpg", quality=95)
                log(f"Saved ({status}) -> {output_path}")

            record = {
                "image_key": image_key,
                "labels": positive_incidents,
                "url": url,
                "scene_description": scene_description,
                "editing_instruction": rescue_instruction,
                "flux_prompt": flux_prompt,
                "sc_score": _round(sc_score),
                "pq_score": _round(pq_score),
                "o_score": _round(o_score),
                "ssim": _round(ssim_value),
                "quality_status": status,
                "passed_gate": gate_ok,
                "output_path": str(output_path) if (gate_ok or KEEP_REJECTED) else None,
            }
            records.append(record)
            (metadata_dir / f"{safe_name}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            if gate_ok:
                counts["accepted"] += 1
                consecutive_errors = 0
                print(f"Progress: {counts['accepted']}/{LIMIT_IMAGES}")
            else:
                counts["rejected"] += 1
                consecutive_errors = 0  # a completed generation is not a processing error

            free_memory(original_image, generated_image)

        except Exception:
            log("Unexpected error:")
            import traceback

            traceback.print_exc()
            consecutive_errors += 1
            counts["errors"] += 1
            free_memory()
            continue

    if counts["accepted"] < LIMIT_IMAGES and counts["stop_reason"] == "target reached":
        counts["stop_reason"] = "dataset exhausted before reaching target"

    print("\n" + "=" * 80)
    print(f"Attempted : {counts['attempted']}")
    print(f"Accepted  : {counts['accepted']}")
    print(f"Rejected  : {counts['rejected']}")
    print(f"Skipped   : {counts['skipped']}")
    print(f"Errors    : {counts['errors']}")
    print(f"Stop      : {counts['stop_reason']}")

    with stage("Release generation models"):
        unload_watermark_tools()
        free_memory(gemma_model, gemma_processor, flux_pipe, evaluators)
        gemma_model = gemma_processor = flux_pipe = evaluators = None

    cmmd_score = maybe_run_cmmd()
    report_path = write_report(records, counts, cmmd_score, runtime)
    log(f"Report: {report_path}")
    log(f"CSV: {OUTPUT_DIR / 'evaluation_report.csv'}")
    return 0 if counts["accepted"] >= LIMIT_IMAGES else 2


def _round(value) -> float | None:
    try:
        if value is None or value != value:
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(main())
