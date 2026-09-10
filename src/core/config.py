from dotenv import load_dotenv
from pathlib import Path
import os
import torch
import numpy as np
import random

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_STORAGE_DIR = Path(
    "/datastore/cndt_khanhnd/models/aeroeyes_model"
)
MODEL_STORAGE_DIR = Path(
    os.getenv("AEROEYES_MODEL_DIR", str(DEFAULT_MODEL_STORAGE_DIR))
).expanduser()
HF_HOME = MODEL_STORAGE_DIR / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
HF_ASSETS_CACHE = HF_HOME / "assets"
HF_XET_CACHE = HF_HOME / "xet"
TORCH_HOME = MODEL_STORAGE_DIR / "torch"
XDG_CACHE_HOME = MODEL_STORAGE_DIR / "xdg"
ULTRALYTICS_DIR = MODEL_STORAGE_DIR / "ultralytics"
ULTRALYTICS_WEIGHTS_DIR = ULTRALYTICS_DIR / "weights"
ULTRALYTICS_RUNS_DIR = ULTRALYTICS_DIR / "runs"
YOLO_CONFIG_DIR = ULTRALYTICS_DIR / "config"
SDQM_VINFO_MODEL_PATH = ULTRALYTICS_WEIGHTS_DIR / "yolo11n.pt"
MPLCONFIGDIR = MODEL_STORAGE_DIR / "matplotlib"

MODEL_STORAGE_DIRECTORIES = (
    MODEL_STORAGE_DIR,
    HF_HUB_CACHE,
    HF_ASSETS_CACHE,
    HF_XET_CACHE,
    TORCH_HOME,
    XDG_CACHE_HOME,
    ULTRALYTICS_WEIGHTS_DIR,
    ULTRALYTICS_RUNS_DIR,
    YOLO_CONFIG_DIR,
    MPLCONFIGDIR,
)

MODEL_CACHE_ENVIRONMENT = {
    "AEROEYES_MODEL_DIR": MODEL_STORAGE_DIR,
    "HF_HOME": HF_HOME,
    "HF_HUB_CACHE": HF_HUB_CACHE,
    "HF_ASSETS_CACHE": HF_ASSETS_CACHE,
    "HF_XET_CACHE": HF_XET_CACHE,
    "TORCH_HOME": TORCH_HOME,
    "XDG_CACHE_HOME": XDG_CACHE_HOME,
    "YOLO_CONFIG_DIR": YOLO_CONFIG_DIR,
    "YOLO_WEIGHTS_DIR": ULTRALYTICS_WEIGHTS_DIR,
    "YOLO_RUNS_DIR": ULTRALYTICS_RUNS_DIR,
    "SDQM_VINFO_MODEL_PATH": SDQM_VINFO_MODEL_PATH,
    "MPLCONFIGDIR": MPLCONFIGDIR,
}

for environment_name, directory in MODEL_CACHE_ENVIRONMENT.items():
    os.environ[environment_name] = str(directory)


def ensure_model_storage() -> None:
    for directory in MODEL_STORAGE_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


# READ ENVIRONMENT
HF_TOKEN = os.getenv('HF_TOKEN')
JSON_PATH = os.getenv(
    "JSON_PATH",
    str(PROJECT_ROOT / "data" / "input" / "eccv_train.json"),
)
OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    str(PROJECT_ROOT / "data" / "output"),
)
REAL_IMAGES_DIR = os.getenv(
    "REAL_IMAGES_DIR",
    str(PROJECT_ROOT / "data" / "real_reference"),
)
GEN_IMAGES_DIR = os.getenv(
    "GEN_IMAGES_DIR",
    str(PROJECT_ROOT / "data" / "gen_reference"),
)
CMMD_REPO_DIR = os.getenv(
    "CMMD_REPO_DIR",
    str(PROJECT_ROOT / "cmmd-pytorch"),
)
GENERAL_MODEL = "google/gemma-4-E4B-it"
FLUX_REPO = "black-forest-labs/FLUX.2-klein-4B"
# black-forest-labs/FLUX.2-dev
FLUX_MODEL = FLUX_REPO
# ----------------------------------------------------------
# Generation Parameters
# ----------------------------------------------------------
IMAGE_SIZE = 1024
LIMIT_IMAGES = int(os.getenv("LIMIT_IMAGES", "500"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "0"))
MAX_CONSECUTIVE_ERRORS = int(os.getenv("MAX_CONSECUTIVE_ERRORS", "0"))
GENERATION_TIMEOUT_SECONDS = int(os.getenv("GENERATION_TIMEOUT_SECONDS", "0"))
MODEL_LOAD_TIMEOUT_SECONDS = int(os.getenv("MODEL_LOAD_TIMEOUT_SECONDS", "1800"))
STAGE_TIMEOUT_SECONDS = int(os.getenv("STAGE_TIMEOUT_SECONDS", "900"))
EVALUATION_TIMEOUT_SECONDS = int(os.getenv("EVALUATION_TIMEOUT_SECONDS", "3600"))
for limit_name in (
    "LIMIT_IMAGES", "MAX_ATTEMPTS", "MAX_CONSECUTIVE_ERRORS",
    "GENERATION_TIMEOUT_SECONDS", "MODEL_LOAD_TIMEOUT_SECONDS",
    "STAGE_TIMEOUT_SECONDS", "EVALUATION_TIMEOUT_SECONDS",
):
    if globals()[limit_name] < 0:
        raise ValueError(f"{limit_name} must be non-negative")
REQUEST_TIMEOUT = 30
DOWNLOAD_RETRIES = 3
MAX_NEW_TOKENS = 256
NUM_INFERENCE_STEPS = 20 # 15
GUIDANCE_SCALE = 3.5 # 3.5 
BASE_SEED = 50
EXPECTED_PYTORCH_CUDA = os.getenv("EXPECTED_PYTORCH_CUDA", "12.8")
# ----------------------------------------------------------
# Quality Evaluation (humaninstruction-ver2-8)
# ----------------------------------------------------------
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
O_SCORE_THRESHOLD = 0.55
SSIM_MAX_THRESHOLD = 0.90
SC_NORM_DIVISOR = 40.0
CMMD_BATCH_SIZE = 16
CMMD_MAX_COUNT = 30000
# ----------------------------------------------------------
# Long-CLIP for the SC score (Zhang et al., ECCV 2024,
# https://arxiv.org/abs/2403.15378). The base CLIP text encoder
# truncates at 77 tokens, which drops most of the FLUX prompt.
# Long-CLIP interpolates the positional embeddings to 248 tokens.
# ----------------------------------------------------------
LONG_CLIP_ENABLED = os.getenv("LONG_CLIP_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
LONG_CLIP_MODEL_ID = os.getenv("LONG_CLIP_MODEL_ID", "zer0int/LongCLIP-L-Diffusers")
LONG_CLIP_MAX_TOKENS = int(os.getenv("LONG_CLIP_MAX_TOKENS", "248"))
# ----------------------------------------------------------
# Watermark / caption removal (EasyOCR detection + Simple LaMa inpainting).
# Runs on the freshly downloaded image, before resize/crop.
# ----------------------------------------------------------
WATERMARK_REMOVAL_ENABLED = os.getenv("WATERMARK_REMOVAL_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
WATERMARK_OCR_LANGUAGES = tuple(
    language.strip()
    for language in os.getenv("WATERMARK_OCR_LANGUAGES", "en").split(",")
    if language.strip()
) or ("en",)
WATERMARK_OCR_USE_GPU = os.getenv("WATERMARK_OCR_USE_GPU", "true").lower() in (
    "1",
    "true",
    "yes",
)
WATERMARK_MIN_TEXT_CONFIDENCE = float(
    os.getenv("WATERMARK_MIN_TEXT_CONFIDENCE", "0.2")
)
WATERMARK_EDGE_MARGIN_RATIO = float(os.getenv("WATERMARK_EDGE_MARGIN_RATIO", "0.12"))
WATERMARK_WIDE_ASPECT_RATIO = float(os.getenv("WATERMARK_WIDE_ASPECT_RATIO", "3.5"))
WATERMARK_DILATE_KERNEL = int(os.getenv("WATERMARK_DILATE_KERNEL", "7"))
WATERMARK_DILATE_ITERATIONS = int(os.getenv("WATERMARK_DILATE_ITERATIONS", "2"))
# ----------------------------------------------------------
# SDQM (Synthetic Dataset Quality Metric)
# See docs/pipeline/sdqm-integration-plan.md
# ----------------------------------------------------------
SDQM_REPO_DIR = os.getenv(
    "SDQM_REPO_DIR",
    str(PROJECT_ROOT / "third_party" / "SDQM"),
)
SDQM_OUTPUT_DIR = os.getenv(
    "SDQM_OUTPUT_DIR",
    str(Path(OUTPUT_DIR) / "sdqm"),
)
SDQM_ENABLED = os.getenv("SDQM_ENABLED", "true").lower() in ("1", "true", "yes")
SDQM_EMBEDDING_MODEL = os.getenv(
    "SDQM_EMBEDDING_MODEL",
    "facebook/dinov2-small",
)
SDQM_MODEL_TEXT = os.getenv(
    "SDQM_MODEL_TEXT",
    "firefighter . rescue boat . helicopter . ambulance . emergency vehicle .",
)
SDQM_YOLO_DATA_YAML = os.getenv(
    "SDQM_YOLO_DATA_YAML",
    str(PROJECT_ROOT / "config" / "sdqm" / "data.yaml"),
)
SDQM_YOLO_EXPORT = os.getenv("SDQM_YOLO_EXPORT", "true").lower() in ("1", "true", "yes")
SDQM_GROUNDING_DINO_MODEL = os.getenv(
    "SDQM_GROUNDING_DINO_MODEL",
    "IDEA-Research/grounding-dino-tiny",
)
SDQM_BOX_THRESHOLD = float(os.getenv("SDQM_BOX_THRESHOLD", "0.25"))
SDQM_TEXT_THRESHOLD = float(os.getenv("SDQM_TEXT_THRESHOLD", "0.25"))
SDQM_METRIC_TYPES = [
    "similarity",
    "fdg",
    "lcm",
    "separability",
    "distribution",
    "bounding_box",
    "label_overlap",
    "spatial",
]
SDQM_VINFO_ENABLED = os.getenv("SDQM_VINFO_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
SDQM_VINFO_DATASET = os.getenv("SDQM_VINFO_DATASET", "rescue")
SDQM_HISTORY_CSV = os.getenv(
    "SDQM_HISTORY_CSV",
    str(Path(SDQM_OUTPUT_DIR) / "sdqm_history.csv"),
)
SDQM_MAP_CSV = os.getenv("SDQM_MAP_CSV", "")
SDQM_MAP_COLUMN = os.getenv("SDQM_MAP_COLUMN", "map")
SDQM_MAP_VALUE = os.getenv("SDQM_MAP_VALUE")
SDQM_APPEND_HISTORY = os.getenv("SDQM_APPEND_HISTORY", "true").lower() in (
    "1",
    "true",
    "yes",
)
SDQM_RUN_REGRESSION = os.getenv("SDQM_RUN_REGRESSION", "true").lower() in (
    "1",
    "true",
    "yes",
)
SDQM_MIN_REGRESSION_ROWS = int(os.getenv("SDQM_MIN_REGRESSION_ROWS", "3"))
SDQM_MIN_IMAGES = 2
SDQM_SUMMARY_PATH = os.getenv(
    "SDQM_SUMMARY_PATH",
    str(PROJECT_ROOT / "reports" / "sdqm_summary.md"),
)
# ----------------------------------------------------------
# HTTP Headers
# ----------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/137.0 Safari/537.36"
    )
}
# ----------------------------------------------------------
# Random Seed
# ----------------------------------------------------------

def random_seed(): 
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_SEED)
