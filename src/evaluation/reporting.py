from __future__ import annotations

from dataclasses import dataclass

from src.core.config import SDQM_MIN_IMAGES
from src.evaluation.cmmd import compute_dataset_cmmd
from src.evaluation.sdqm import compute_dataset_sdqm
from src.evaluation.sdqm_embedding import list_images


@dataclass(frozen=True)
class DatasetEvaluationResult:
    cmmd_score: float
    sdqm_metrics: dict[str, float]


def _validate_image_counts(ref_dir: str, eval_dir: str) -> None:
    real_count = len(list_images(ref_dir))
    synthetic_count = len(list_images(eval_dir))
    if real_count < SDQM_MIN_IMAGES or synthetic_count < SDQM_MIN_IMAGES:
        raise ValueError(
            f"SDQM requires at least {SDQM_MIN_IMAGES} images per side. "
            f"Found real={real_count}, synthetic={synthetic_count}."
        )


def run_dataset_evaluation(
    ref_dir: str,
    eval_dir: str,
) -> DatasetEvaluationResult:
    _validate_image_counts(ref_dir, eval_dir)
    return DatasetEvaluationResult(
        cmmd_score=compute_dataset_cmmd(ref_dir, eval_dir),
        sdqm_metrics=compute_dataset_sdqm(ref_dir, eval_dir),
    )
