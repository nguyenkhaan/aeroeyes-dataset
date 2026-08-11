from __future__ import annotations

import json
from pathlib import Path


def attach_sdqm_metadata(
    records: list[dict],
    sdqm_metrics: dict[str, float],
) -> list[dict]:
    sdqm_block = {
        "scope": "dataset",
        "metrics": dict(sdqm_metrics),
    }
    return [{**record, "sdqm": sdqm_block} for record in records]


def write_metadata_jsonl(
    records: list[dict],
    output_dir: str | Path,
) -> Path | None:
    if not records:
        return None

    output_path = Path(output_dir) / "evaluation_metadata.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path
