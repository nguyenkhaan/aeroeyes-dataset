from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.core.config import SDQM_MAP_COLUMN, SDQM_MIN_REGRESSION_ROWS


def compute_metric_map_correlations(
    history_df: pd.DataFrame,
    map_column: str = SDQM_MAP_COLUMN,
) -> dict[str, dict[str, float | None]]:
    if map_column not in history_df.columns:
        raise ValueError(f"Column '{map_column}' not found in regression history.")

    metric_columns = [
        column
        for column in history_df.columns
        if column != map_column and pd.api.types.is_numeric_dtype(history_df[column])
    ]

    correlations: dict[str, dict[str, float | None]] = {}
    for column in metric_columns:
        paired_df = history_df[[column, map_column]].dropna()
        if len(paired_df) < SDQM_MIN_REGRESSION_ROWS:
            correlations[column] = {
                "pearson_r": None,
                "pearson_p": None,
                "spearman_r": None,
                "spearman_p": None,
            }
            continue

        values = paired_df[column].astype(float)
        target = paired_df[map_column].astype(float)
        if values.nunique() <= 1 or target.nunique() <= 1:
            correlations[column] = {
                "pearson_r": None,
                "pearson_p": None,
                "spearman_r": None,
                "spearman_p": None,
            }
            continue

        pearson = pearsonr(values, target)
        spearman = spearmanr(values, target)
        correlations[column] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_r": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }

    return correlations


def append_sdqm_history_row(
    history_csv: str | Path,
    row: dict[str, float | str | int],
) -> Path:
    history_path = Path(history_csv)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([row])
    if history_path.is_file():
        history_df = pd.read_csv(history_path)
        history_df = pd.concat([history_df, new_row], ignore_index=True)
    else:
        history_df = new_row

    history_df.to_csv(history_path, index=False)
    return history_path


def run_sdqm_regression(
    history_csv: str | Path,
    map_column: str = SDQM_MAP_COLUMN,
    output_path: str | Path | None = None,
) -> dict[str, dict[str, float | None]] | None:
    """
    Correlate SDQM metric history with detector mAP values.

    Requires a CSV with one row per experiment and at least
    SDQM_MIN_REGRESSION_ROWS rows containing map scores.
    """
    history_path = Path(history_csv)
    if not history_path.is_file():
        return None

    history_df = pd.read_csv(history_path)
    if len(history_df) < SDQM_MIN_REGRESSION_ROWS:
        return None

    if map_column not in history_df.columns:
        return None

    history_df = history_df.dropna(subset=[map_column])
    if len(history_df) < SDQM_MIN_REGRESSION_ROWS:
        return None

    correlations = compute_metric_map_correlations(history_df, map_column=map_column)
    report = {
        "history_csv": str(history_path.resolve()),
        "map_column": map_column,
        "row_count": len(history_df),
        "correlations": correlations,
    }

    destination = Path(output_path or history_path.with_name("regression_report.json"))
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return correlations
