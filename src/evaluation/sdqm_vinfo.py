from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import yaml

from src.core.config import (
    SDQM_REPO_DIR,
    SDQM_VINFO_DATASET,
    SDQM_YOLO_DATA_YAML,
)

VINFO_METRIC_KEYS = (
    "conditional_iou",
    "predictive_iou",
    "v_info_iou",
    "conditional_conf",
    "predictive_conf",
    "v_info_conf",
    "conditional_fusion",
    "predictive_fusion",
    "v_info_fusion",
)


def custom_ultralytics_path() -> Path:
    return (
        Path(SDQM_REPO_DIR)
        / "dataset_interpretability"
        / "v_info"
        / "ultralytics"
    )


def check_custom_ultralytics() -> tuple[bool, str]:
    repo_ultralytics = custom_ultralytics_path()
    if not repo_ultralytics.is_dir():
        return (
            False,
            "Custom ultralytics not found. Clone SDQM and run: "
            f"pip install -e {repo_ultralytics}",
        )

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return (
            False,
            "ultralytics is not installed. Run: "
            f"pip install -e {repo_ultralytics}",
        )

    return True, "ok"


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    try:
        os.symlink(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_tree(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source_dir.iterdir()):
        if not source_file.is_file():
            continue
        _link_or_copy(source_file, destination_dir / source_file.name)


def _write_vinfo_yaml(output_dir: Path, template_path: Path) -> Path:
    with template_path.open(encoding="utf-8") as handle:
        yaml_data = yaml.safe_load(handle)

    yaml_data["path"] = str(output_dir.resolve())
    yaml_data["train"] = "images/train"
    yaml_data["val"] = "images/val"

    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.dump(yaml_data, handle, sort_keys=False)

    return yaml_path


def prepare_vinfo_yolo_layout(
    real_yolo_root: str | Path,
    synthetic_yolo_root: str | Path,
    output_dir: str | Path,
    data_yaml_template: str | Path = SDQM_YOLO_DATA_YAML,
) -> Path:
    """
    Build a combined YOLO dataset for V-Info.

    Synthetic images are used for train; real images for val.
    Uses copy/symlink fallback for cross-platform compatibility.
    """
    real_root = Path(real_yolo_root)
    synthetic_root = Path(synthetic_yolo_root)
    vinfo_root = Path(output_dir)

    _copy_tree(
        synthetic_root / "images" / "train",
        vinfo_root / "images" / "train",
    )
    _copy_tree(
        synthetic_root / "labels" / "train",
        vinfo_root / "labels" / "train",
    )
    _copy_tree(
        real_root / "images" / "train",
        vinfo_root / "images" / "val",
    )
    _copy_tree(
        real_root / "labels" / "train",
        vinfo_root / "labels" / "val",
    )

    _write_vinfo_yaml(vinfo_root, Path(data_yaml_template))
    return vinfo_root


def _ensure_sdqm_repo_on_path() -> None:
    repo_path = str(Path(SDQM_REPO_DIR).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _load_get_v_info():
    _ensure_sdqm_repo_on_path()
    vinfo_module_path = (
        Path(SDQM_REPO_DIR) / "dataset_interpretability" / "run.py"
    )
    if not vinfo_module_path.is_file():
        raise FileNotFoundError(
            f"V-Info module not found at {vinfo_module_path}. "
            "Clone SDQM into third_party/SDQM."
        )

    spec = importlib.util.spec_from_file_location(
        "sdqm_vinfo_run",
        vinfo_module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load V-Info module from {vinfo_module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_v_info


def compute_vinfo_metrics(
    real_yolo_root: str | Path,
    synthetic_yolo_root: str | Path,
    output_dir: str | Path,
    image_size: int,
    dataset: str = SDQM_VINFO_DATASET,
) -> dict[str, float]:
    """
    Train YOLO on synthetic labels and validate on real labels (V-Info).

    Requires the customized ultralytics package from the SDQM repo.
    """
    is_ready, message = check_custom_ultralytics()
    if not is_ready:
        raise RuntimeError(message)

    vinfo_root = prepare_vinfo_yolo_layout(
        real_yolo_root=real_yolo_root,
        synthetic_yolo_root=synthetic_yolo_root,
        output_dir=output_dir,
    )
    yaml_path = str(vinfo_root / "data.yaml")

    get_v_info = _load_get_v_info()
    results = get_v_info(
        yaml_path,
        yaml_path,
        dataset,
        image_size,
    )

    return {
        key: float(value)
        for key, value in zip(VINFO_METRIC_KEYS, results, strict=True)
    }
