from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from src.core.config import SDQM_EMBEDDING_MODEL

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".webp"}


def list_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {root}")

    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return files


def embed_image_directory(
    image_dir: str | Path,
    output_prefix: str | Path,
    model_name: str = SDQM_EMBEDDING_MODEL,
    device: str | None = None,
) -> tuple[str, str]:
    """
    Embed all images in a directory and save SDQM-compatible artifacts.

    Returns paths to (.pkl, .csv) files expected by SDQM's load_embedding_file().
    """
    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    eval_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(eval_device)
    model.eval()

    embeddings: list[np.ndarray] = []
    rows: list[dict[str, str]] = []

    for image_path in tqdm(image_paths, desc=f"Embedding {Path(image_dir).name}"):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(eval_device)

        with torch.no_grad():
            outputs = model(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

        embeddings.append(vector)
        rows.append({"file_path": str(image_path.resolve())})

    features = np.stack(embeddings, axis=0)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    pkl_path = output_prefix.with_suffix(".pkl")
    csv_path = output_prefix.with_suffix(".csv")
    npy_path = output_prefix.with_suffix(".npy")

    with pkl_path.open("wb") as handle:
        pickle.dump(features, handle)

    np.save(npy_path, features)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return str(pkl_path), str(csv_path)
