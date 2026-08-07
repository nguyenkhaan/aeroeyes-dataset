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
GENERAL_MODEL = "google/gemma-4-12B" 
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
MAX_NEW_TOKENS = 128
NUM_INFERENCE_STEPS = 6 # 15
GUIDANCE_SCALE = 1.0 # 3.5 
BASE_SEED = 50
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
