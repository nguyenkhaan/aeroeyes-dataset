from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyiqa
import torch
from PIL import Image
from skimage.metrics import structural_similarity as _ssim
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from src.core.config import (
    CLIP_MODEL_ID,
    O_SCORE_THRESHOLD,
    SC_NORM_DIVISOR,
    SSIM_MAX_THRESHOLD,
)


@dataclass
class QualityEvaluators:
    clip_model: CLIPModel
    clip_processor: CLIPProcessor
    clip_iqa: torch.nn.Module
    pq_transform: transforms.Compose
    device: str


def load_evaluators(device: str | None = None) -> QualityEvaluators:
    eval_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading evaluators on: {eval_device}...")

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(eval_device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_iqa = pyiqa.create_metric("clipiqa", device=eval_device)
    pq_transform = transforms.Compose([transforms.ToTensor()])

    return QualityEvaluators(
        clip_model=clip_model,
        clip_processor=clip_processor,
        clip_iqa=clip_iqa,
        pq_transform=pq_transform,
        device=eval_device,
    )


def evaluate_quality(
    evaluators: QualityEvaluators,
    generated_pil: Image.Image,
    prompt: str,
) -> tuple[float, float]:
    """
    Compute SC (CLIP score, raw × 100) and PQ (CLIP-IQA).

    Matches humaninstruction-ver2-8 Cell 11.5.
    """
    img_tensor = (
        evaluators.pq_transform(generated_pil)
        .unsqueeze(0)
        .to(evaluators.device)
    )
    with torch.no_grad():
        pq_score = evaluators.clip_iqa(img_tensor).item()

    inputs = evaluators.clip_processor(
        text=[prompt],
        images=generated_pil,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(evaluators.device)
    with torch.no_grad():
        outputs = evaluators.clip_model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(
            p=2, dim=-1, keepdim=True
        )
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(
            p=2, dim=-1, keepdim=True
        )
        sc_raw = torch.matmul(image_embeds, text_embeds.T).item() * 100.0

    return sc_raw, pq_score


def compute_o_score(sc_score: float, pq_score: float) -> float:
    sc_norm = max(0.0, min(1.0, sc_score / SC_NORM_DIVISOR))
    return min(sc_norm, pq_score)


def compute_ssim(
    original_image: Image.Image,
    generated_image: Image.Image,
) -> float:
    orig_np = np.array(original_image)
    gen_np = np.array(generated_image)
    return float(
        _ssim(orig_np, gen_np, channel_axis=-1, data_range=255)
    )


def passes_quality_gate(o_score: float, ssim_val: float) -> bool:
    if o_score < O_SCORE_THRESHOLD:
        return False
    if ssim_val > SSIM_MAX_THRESHOLD:
        return False
    return True
