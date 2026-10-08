import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.core.config import (
    DOWNLOAD_IMAGES_DIR,
    DOWNLOAD_RETRIES,
    HEADERS,
    IMAGE_SUMMARY_PATH,
    JSON_PATH,
    REQUEST_TIMEOUT,
)
from src.helper.image import download_image
from src.helper.loading_dataset import loading_dataset
from src.vision.disaster_filter import is_valid_disaster_metadata


CHECKPOINT_INTERVAL = 25


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and resume a natural-disaster image subset.")
    parser.add_argument("--json-path", type=Path, default=Path(JSON_PATH))
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Target total of readable disaster images, including cached images; 0 scans all records.",
    )
    arguments = parser.parse_args(argv)
    if arguments.limit < 0:
        parser.error("--limit must be non-negative")
    return arguments


def image_filename(image_key: str) -> str:
    return hashlib.sha256(image_key.encode("utf-8")).hexdigest() + ".png"


def cached_image_is_readable(filename: str) -> bool:
    try:
        with Image.open(DOWNLOAD_IMAGES_DIR / filename) as image:
            image.load()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def metadata_verdict(image_info: object) -> tuple[bool, str]:
    if not isinstance(image_info, dict):
        return False, "Invalid metadata"
    return is_valid_disaster_metadata(image_info.get("incidents"), image_info.get("damage"))


def load_cached_summary(data: dict) -> dict:
    if not IMAGE_SUMMARY_PATH.is_file():
        return {}
    saved = json.loads(IMAGE_SUMMARY_PATH.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ValueError("Existing image summary must be an object keyed by image id.")
    summary = {}
    for image_key, previous in saved.items():
        image_info = data.get(image_key)
        if not metadata_verdict(image_info)[0]:
            continue
        filename = image_filename(image_key)
        if cached_image_is_readable(filename):
            summary[image_key] = {
                **(previous if isinstance(previous, dict) else {}),
                **image_info,
                "downloaded_file": filename,
            }
    return summary


def save_summary(summary: dict) -> None:
    temporary = IMAGE_SUMMARY_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(IMAGE_SUMMARY_PATH)


def download_and_save(image_info: dict, filename: str) -> None:
    url = image_info.get("url")
    if not url:
        raise ValueError("Missing URL")
    content = download_image(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, retries=DOWNLOAD_RETRIES)
    if content is None:
        raise ValueError("Download failed after retries")
    image_path = DOWNLOAD_IMAGES_DIR / filename
    temporary = image_path.with_suffix(".tmp")
    with Image.open(BytesIO(content)) as downloaded_image:
        with downloaded_image.convert("RGB") as image:
            image.save(temporary, format="PNG")
    temporary.replace(image_path)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    data = loading_dataset(str(arguments.json_path))
    DOWNLOAD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = load_cached_summary(data)
    downloaded = failed = skipped = 0
    reused = len(summary)
    print(f"Target images: {arguments.limit or 'all'} | Already cached: {reused}", flush=True)

    try:
        for image_key, image_info in data.items():
            if arguments.limit and len(summary) >= arguments.limit:
                break
            if image_key in summary:
                continue
            try:
                is_valid, reason = metadata_verdict(image_info)
                if not is_valid:
                    skipped += 1
                    print(f"Skip: {image_key} ({reason})", flush=True)
                    continue

                image_name = image_filename(image_key)
                if cached_image_is_readable(image_name):
                    reused += 1
                else:
                    print(f"Downloading: {image_key}", flush=True)
                    download_and_save(image_info, image_name)
                    downloaded += 1
                summary[image_key] = {
                    **image_info,
                    "downloaded_file": image_name,
                }
                if len(summary) % CHECKPOINT_INTERVAL == 0:
                    save_summary(summary)
                print(f"Ready: {len(summary)} | {image_name}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"Skip: {image_key}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        save_summary(summary)
        print(
            f"Ready: {len(summary)} | Downloaded: {downloaded} | Reused: {reused} | "
            f"Failed: {failed} | Skipped/filtered: {skipped}",
            flush=True,
        )
        print(f"Image summary: {IMAGE_SUMMARY_PATH}", flush=True)
    if arguments.limit and len(summary) < arguments.limit:
        print("Target not reached; saved images can be reused by the next run.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
