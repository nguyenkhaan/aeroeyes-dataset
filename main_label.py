import hashlib
import json

from PIL import Image

from src.core.config import DOWNLOAD_IMAGES_DIR, IMAGE_SUMMARY_PATH
from src.helper.loading_dataset import loading_dataset


def main() -> None:
    if not DOWNLOAD_IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Download directory not found: {DOWNLOAD_IMAGES_DIR}")

    data = loading_dataset()
    summary = {}
    skipped = 0
    missing = 0
    invalid = 0

    for image_key, image_info in data.items():
        if not isinstance(image_info, dict):
            invalid += 1
            print(f"Skip: {image_key} (Invalid metadata)", flush=True)
            continue

        incidents = image_info.get("incidents") or {}
        if not isinstance(incidents, dict) or not any(
            value == 1 for value in incidents.values()
        ):
            skipped += 1
            continue

        image_name = hashlib.sha256(image_key.encode("utf-8")).hexdigest() + ".png"
        image_path = DOWNLOAD_IMAGES_DIR / image_name
        if not image_path.is_file():
            missing += 1
            continue

        try:
            with Image.open(image_path) as image:
                image.load()
        except Exception as exc:
            invalid += 1
            print(f"Skip: {image_name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        summary[image_key] = {
            **image_info,
            "downloaded_file": image_name,
        }

    temporary_summary_path = IMAGE_SUMMARY_PATH.with_suffix(".tmp")
    with temporary_summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")
    temporary_summary_path.replace(IMAGE_SUMMARY_PATH)
    print(
        f"Recovered: {len(summary)} | Missing: {missing} | "
        f"Invalid: {invalid} | No positive labels: {skipped}",
        flush=True,
    )
    print(f"Image summary: {IMAGE_SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
