from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

from src.core.config import (
    SDQM_APPEND_HISTORY,
    SDQM_EMBEDDING_MODEL,
    SDQM_HISTORY_CSV,
    SDQM_MAP_COLUMN,
    SDQM_MAP_CSV,
    SDQM_MAP_VALUE,
    SDQM_METRIC_TYPES,
    SDQM_MIN_IMAGES,
    SDQM_OUTPUT_DIR,
    SDQM_REPO_DIR,
    SDQM_RUN_REGRESSION,
    SDQM_SUMMARY_PATH,
    SDQM_VINFO_DATASET,
    SDQM_VINFO_ENABLED,
    SDQM_YOLO_DATA_YAML,
    SDQM_YOLO_EXPORT,
)
from src.evaluation.sdqm_embedding import embed_image_directory, list_images
from src.evaluation.sdqm_regression import append_sdqm_history_row, run_sdqm_regression
from src.evaluation.sdqm_vinfo import (
    check_custom_ultralytics,
    compute_vinfo_metrics,
    validate_vinfo_dataset,
)
from src.evaluation.yolo_export import export_yolo_pair


def _ensure_sdqm_import_paths(repo_dir: str | Path) -> None:
    repo_root = Path(repo_dir).resolve()
    if not repo_root.is_dir():
        return

    path_text = str(repo_root)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _load_calculate_sdqm():
    sdqm_main = Path(SDQM_REPO_DIR) / "sdqm.py"
    if not sdqm_main.is_file():
        raise FileNotFoundError(
            f"SDQM repo not found at {sdqm_main}. "
            "Clone it with: git clone https://github.com/ayushzenith/SDQM.git third_party/SDQM"
        )

    _ensure_sdqm_import_paths(SDQM_REPO_DIR)
    ultralytics_ready, ultralytics_message = check_custom_ultralytics()
    if not ultralytics_ready:
        raise RuntimeError(
            "SDQM requires the custom ultralytics fork before it can load: "
            f"{ultralytics_message}"
        )

    spec = importlib.util.spec_from_file_location("sdqm_main", sdqm_main)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SDQM module from {sdqm_main}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "an SDQM dependency"
        raise ModuleNotFoundError(
            "SDQM dependency is missing: "
            f"{missing_module}. Install SDQM dependencies with: "
            "python -m pip install -r requirements-sdqm.txt"
        ) from exc
    return module.calculate_sdqm


def _flatten_metric_values(metric_values: list[dict]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for values in metric_values:
        for key, value in values.items():
            if isinstance(value, (int, float)):
                flattened[key] = float(value)
    return flattened


def _resolve_image_size(image_dir: str | Path) -> tuple[int, int]:
    first_image = list_images(image_dir)[0]
    with Image.open(first_image) as image:
        width, height = image.size
    return width, height


def _resolve_output_path(
    output_dir: Path,
    configured_path: str,
) -> Path:
    configured = Path(configured_path)
    if configured.is_absolute() or configured.parent != Path("."):
        return configured
    return output_dir / configured.name


def _parse_map_value(value: str | None) -> float | None:
    if not value:
        return None

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError("SDQM_MAP_VALUE must be a numeric mAP value.") from exc


def _maybe_append_history(
    flattened: dict[str, float],
    output_dir: Path,
) -> None:
    if not SDQM_APPEND_HISTORY:
        return

    history_row = dict(flattened)
    map_value = _parse_map_value(SDQM_MAP_VALUE)
    if map_value is not None:
        history_row[SDQM_MAP_COLUMN] = map_value

    history_csv = _resolve_output_path(output_dir, SDQM_HISTORY_CSV)
    append_sdqm_history_row(history_csv, history_row)


def _maybe_run_regression(
    output_dir: Path,
    map_csv: str,
) -> dict[str, dict[str, float | None]] | None:
    if not SDQM_RUN_REGRESSION:
        return None

    history_csv = _resolve_output_path(
        output_dir,
        map_csv or SDQM_HISTORY_CSV,
    )
    if not history_csv.is_file():
        return None

    return run_sdqm_regression(
        history_csv=history_csv,
        map_column=SDQM_MAP_COLUMN,
        output_path=output_dir / "regression_report.json",
    )


def write_sdqm_summary(report: dict[str, object]) -> Path:
    summary_path = Path(SDQM_SUMMARY_PATH)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    summary_lines = [
        "# SDQM Dataset Summary",
        "",
        "## Dataset",
        "",
        f"- Real images: {report['real_image_count']}",
        f"- Synthetic images: {report['synthetic_image_count']}",
        f"- Embedding model: {report['embedding_model']}",
        f"- YOLO export: {report['yolo_export_enabled']}",
        f"- V-Info: {report['vinfo_status']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    numeric_metrics = [
        (metric_name, metric_value)
        for metric_name, metric_value in sorted(metrics.items())
        if isinstance(metric_name, str) and isinstance(metric_value, (int, float))
    ]
    summary_lines.extend(
        f"| {metric_name} | {metric_value:.4f} |"
        for metric_name, metric_value in numeric_metrics
    )
    if not numeric_metrics:
        summary_lines.append("| No numeric metrics returned | N/A |")

    summary_lines.extend(
        [
            "",
            "## Regression",
            "",
            f"- Completed: {report['regression_ran']}",
        ]
    )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return summary_path


def compute_dataset_sdqm(
    ref_dir: str,
    eval_dir: str,
    output_dir: str | None = None,
    metric_types: list[str] | None = None,
    embedding_model: str = SDQM_EMBEDDING_MODEL,
    export_yolo: bool | None = None,
    include_vinfo: bool | None = None,
    map_csv: str | None = None,
) -> dict[str, float]:
    """
    Compute SDQM metrics between reference and generated image directories.

    Phase 3 adds V-Info (custom ultralytics) and optional SDQM-vs-mAP
    regression when history CSV contains map scores.
    See docs/pipeline/sdqm-integration-plan.md.
    """
    real_images = list_images(ref_dir)
    synthetic_images = list_images(eval_dir)

    if len(real_images) < SDQM_MIN_IMAGES or len(synthetic_images) < SDQM_MIN_IMAGES:
        raise ValueError(
            f"SDQM requires at least {SDQM_MIN_IMAGES} images per side. "
            f"Found real={len(real_images)}, synthetic={len(synthetic_images)}."
        )

    sdqm_dir = Path(output_dir or SDQM_OUTPUT_DIR)
    sdqm_dir.mkdir(parents=True, exist_ok=True)
    regression_csv = map_csv if map_csv is not None else SDQM_MAP_CSV
    calculate_sdqm = _load_calculate_sdqm()

    use_yolo_export = SDQM_YOLO_EXPORT if export_yolo is None else export_yolo
    use_vinfo = SDQM_VINFO_ENABLED if include_vinfo is None else include_vinfo
    if use_vinfo:
        validate_vinfo_dataset(SDQM_VINFO_DATASET)
    real_yolo_root: Path | None = None
    synthetic_yolo_root: Path | None = None

    if use_yolo_export:
        real_yolo_root, synthetic_yolo_root = export_yolo_pair(
            ref_dir=ref_dir,
            eval_dir=eval_dir,
            output_dir=sdqm_dir / "yolo",
            data_yaml_template=SDQM_YOLO_DATA_YAML,
        )
        real_embed_dir = real_yolo_root / "images" / "train"
        synthetic_embed_dir = synthetic_yolo_root / "images" / "train"
        yolo_layout = {
            "real": str(real_yolo_root.resolve()),
            "synthetic": str(synthetic_yolo_root.resolve()),
        }
    else:
        real_embed_dir = Path(ref_dir)
        synthetic_embed_dir = Path(eval_dir)
        yolo_layout = None

    real_prefix = sdqm_dir / "real_embeddings"
    synthetic_prefix = sdqm_dir / "synthetic_embeddings"

    embed_image_directory(real_embed_dir, real_prefix, model_name=embedding_model)
    embed_image_directory(
        synthetic_embed_dir,
        synthetic_prefix,
        model_name=embedding_model,
    )

    image_size = _resolve_image_size(real_embed_dir)

    selected_metrics = metric_types or SDQM_METRIC_TYPES

    metric_values = calculate_sdqm(
        real_files=[str(real_prefix.with_suffix(".pkl"))],
        synthetic_files=[str(synthetic_prefix.with_suffix(".pkl"))],
        image_size=image_size,
        output=str(sdqm_dir / "sdqm_values.csv"),
        metric_type=selected_metrics,
        dataset="N/A",
        temp_dir=str(sdqm_dir / "vinfo_temp"),
    )

    flattened = _flatten_metric_values(metric_values)
    vinfo_status = "skipped"

    if use_vinfo:
        if not use_yolo_export or real_yolo_root is None or synthetic_yolo_root is None:
            vinfo_status = "skipped_missing_yolo_export"
        else:
            ultralytics_ready, ultralytics_message = check_custom_ultralytics()
            if not ultralytics_ready:
                print(f"V-Info skipped: {ultralytics_message}")
                vinfo_status = "skipped_missing_ultralytics"
            else:
                try:
                    vinfo_metrics = compute_vinfo_metrics(
                        real_yolo_root=real_yolo_root,
                        synthetic_yolo_root=synthetic_yolo_root,
                        output_dir=sdqm_dir / "vinfo",
                        image_size=image_size[0],
                        dataset=SDQM_VINFO_DATASET,
                    )
                    flattened.update(vinfo_metrics)
                    vinfo_status = "completed"
                except Exception as exc:  # noqa: BLE001
                    # V-Info is optional; upstream validator failures must not discard base SDQM metrics.
                    print(f"V-Info calculation failed: {exc}")
                    vinfo_status = f"failed: {exc}"

    _maybe_append_history(flattened, sdqm_dir)
    regression_results = _maybe_run_regression(sdqm_dir, regression_csv)

    report = {
        "status": "completed",
        "ref_dir": str(Path(ref_dir).resolve()),
        "eval_dir": str(Path(eval_dir).resolve()),
        "real_image_count": len(real_images),
        "synthetic_image_count": len(synthetic_images),
        "embedding_model": embedding_model,
        "metric_types": selected_metrics,
        "vinfo_enabled": use_vinfo,
        "vinfo_status": vinfo_status,
        "yolo_export_enabled": use_yolo_export,
        "yolo_layout": yolo_layout,
        "image_size": list(image_size),
        "metrics": flattened,
        "regression_ran": regression_results is not None,
        "summary_path": str(Path(SDQM_SUMMARY_PATH).resolve()),
    }

    report_path = sdqm_dir / SDQM_REPORT_FILENAME
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sdqm_summary(report)

    return flattened
