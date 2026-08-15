from dotenv import load_dotenv
from pathlib import Path
import os
import torch
import numpy as np
import random

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
GENERAL_MODEL = "google/gemma-4-12B-it" 
FLUX_REPO = "black-forest-labs/FLUX.2-klein-9B"
# black-forest-labs/FLUX.2-dev
FLUX_MODEL = FLUX_REPO
# ----------------------------------------------------------
# Generation Parameters
# ----------------------------------------------------------
IMAGE_SIZE = 1024
LIMIT_IMAGES = 5
REQUEST_TIMEOUT = 30
DOWNLOAD_RETRIES = 3
MAX_NEW_TOKENS = 256
NUM_INFERENCE_STEPS = 20 # 15
GUIDANCE_SCALE = 3.5 # 3.5 
BASE_SEED = 50
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
