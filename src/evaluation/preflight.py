from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.core.config import (
    CMMD_REPO_DIR,
    EXPECTED_PYTORCH_CUDA,
    GEN_IMAGES_DIR,
    REAL_IMAGES_DIR,
    SDQM_MIN_IMAGES,
    SDQM_REPO_DIR,
    SDQM_YOLO_DATA_YAML,
)
from src.evaluation.sdqm_embedding import IMAGE_EXTENSIONS
from src.evaluation.sdqm_vinfo import check_custom_ultralytics


@dataclass(frozen=True)
class EvaluationPaths:
    cmmd_main: Path
    sdqm_main: Path
    data_yaml: Path


@dataclass(frozen=True)
class PreflightOptions:
    require_cuda: bool = False
    require_images: bool = False


def configured_evaluation_paths() -> EvaluationPaths:
    return EvaluationPaths(
        cmmd_main=Path(CMMD_REPO_DIR) / "main.py",
        sdqm_main=Path(SDQM_REPO_DIR) / "sdqm.py",
        data_yaml=Path(SDQM_YOLO_DATA_YAML),
    )


def collect_path_errors(paths: EvaluationPaths) -> list[str]:
    required_paths = (
        ("CMMD", paths.cmmd_main),
        ("SDQM", paths.sdqm_main),
        ("YOLO data YAML", paths.data_yaml),
    )
    return [
        f"{name} is missing: {path}"
        for name, path in required_paths
        if not path.is_file()
    ]


def _count_images(image_dir: Path) -> int:
    if not image_dir.is_dir():
        return 0
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in image_dir.iterdir()
    )


def collect_image_errors() -> list[str]:
    image_directories = (
        ("real", Path(REAL_IMAGES_DIR)),
        ("synthetic", Path(GEN_IMAGES_DIR)),
    )
    errors: list[str] = []
    for name, directory in image_directories:
        image_count = _count_images(directory)
        if image_count < SDQM_MIN_IMAGES:
            errors.append(
                f"Need at least {SDQM_MIN_IMAGES} {name} images for SDQM; found "
                f"{image_count} in {directory}."
            )
    return errors


def collect_cuda_errors() -> list[str]:
    installed_cuda_version = torch.version.cuda
    if installed_cuda_version != EXPECTED_PYTORCH_CUDA:
        return [
            "PyTorch CUDA build mismatch: "
            f"expected {EXPECTED_PYTORCH_CUDA}, found {installed_cuda_version}."
        ]
    if not torch.cuda.is_available():
        return ["CUDA is unavailable to the configured Python interpreter."]
    return []


def collect_preflight_errors(options: PreflightOptions) -> list[str]:
    errors = collect_path_errors(configured_evaluation_paths())
    if not errors:
        is_ready, message = check_custom_ultralytics()
        if not is_ready:
            errors.append(f"Custom Ultralytics check failed: {message}")
    if options.require_images:
        errors.extend(collect_image_errors())
    if options.require_cuda:
        errors.extend(collect_cuda_errors())
    return errors


def preflight_summary() -> list[str]:
    paths = configured_evaluation_paths()
    return [
        f"Python CMMD path: {paths.cmmd_main}",
        f"Python SDQM path: {paths.sdqm_main}",
        f"YOLO data YAML: {paths.data_yaml}",
        f"PyTorch CUDA build: {torch.version.cuda}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
