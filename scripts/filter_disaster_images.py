#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.core.config import (
    DOWNLOAD_IMAGES_DIR,
    IMAGE_SUMMARY_PATH,
    JSON_PATH,
)
from src.vision.disaster_filter import (
    DisasterVisualFilter,
    is_valid_disaster_metadata,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Incidents-1M dataset for genuine natural disaster scenes."
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=IMAGE_SUMMARY_PATH,
        help="Path to image_summary.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for cleaned summary",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Run only metadata whitelist filtering (fast, CPU only)",
    )
    parser.add_argument(
        "--use-clip",
        action="store_true",
        help="Run CLIP visual classifier on downloaded images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.55,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    summary_path = args.summary_path

    if not summary_path.is_file():
        raw_json = Path(JSON_PATH)
        if raw_json.is_file():
            print(f"Loading raw dataset from {raw_json}...")
            data = json.loads(raw_json.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(f"Neither {summary_path} nor {raw_json} exists.")
    else:
        print(f"Loading image summary from {summary_path}...")
        data = json.loads(summary_path.read_text(encoding="utf-8"))

    total = len(data)
    print(f"Total input samples: {total}")

    accepted_metadata: dict[str, dict] = {}
    rejected_metadata: dict[str, str] = {}

    for key, info in data.items():
        incidents = info.get("incidents") or {}
        damage = info.get("damage")
        is_valid, reason = is_valid_disaster_metadata(incidents, damage)
        if is_valid:
            accepted_metadata[key] = info
        else:
            rejected_metadata[key] = reason

    print("\n--- STAGE 1: METADATA WHITELIST SUMMARY ---")
    print(f"Passed: {len(accepted_metadata)} / {total}")
    print(f"Rejected: {len(rejected_metadata)} / {total}")

    reasons_count: dict[str, int] = {}
    for r in rejected_metadata.values():
        reasons_count[r] = reasons_count.get(r, 0) + 1
    for r, count in sorted(reasons_count.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {r}: {count} images")

    final_accepted: dict[str, dict] = accepted_metadata
    if args.use_clip and not args.metadata_only:
        print("\n--- STAGE 2: CLIP ZERO-SHOT VISUAL FILTERING ---")
        visual_filter = DisasterVisualFilter(threshold=args.clip_threshold)
        visual_filter.load()

        final_accepted = {}
        visual_rejected_count = 0
        missing_images = 0

        for key, info in tqdm(accepted_metadata.items(), desc="CLIP Filter"):
            downloaded_file = info.get("downloaded_file")
            if not downloaded_file:
                final_accepted[key] = info
                continue

            img_path = DOWNLOAD_IMAGES_DIR / downloaded_file
            if not img_path.is_file():
                missing_images += 1
                final_accepted[key] = info
                continue

            try:
                with Image.open(img_path) as img:
                    is_disaster, score, details = visual_filter.classify_image(img)
                if is_disaster:
                    info["clip_disaster_prob"] = score
                    final_accepted[key] = info
                else:
                    visual_rejected_count += 1
            except Exception:
                final_accepted[key] = info

        print(f"Passed: {len(final_accepted)}")
        print(f"Rejected as non-disaster scene: {visual_rejected_count}")

    output_path = args.output_path or summary_path
    if output_path == summary_path and summary_path.is_file():
        backup_path = summary_path.with_suffix(".json.bak")
        shutil.copy2(summary_path, backup_path)
        print(f"Backup created at: {backup_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(final_accepted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(output_path)
    print(f"Saved {len(final_accepted)} records to: {output_path}")


if __name__ == "__main__":
    main()
