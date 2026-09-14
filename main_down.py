import hashlib
import json
from io import BytesIO

from PIL import Image

from src.core.config import (
    DOWNLOAD_IMAGES_DIR,
    DOWNLOAD_RETRIES,
    HEADERS,
    IMAGE_SUMMARY_PATH,
    REQUEST_TIMEOUT,
)
from src.helper.image import download_image
from src.helper.loading_dataset import loading_dataset


def main() -> None:
    data = loading_dataset()
    DOWNLOAD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    failed = 0

    try:
        for image_key, image_info in data.items():
            print(f"Downloading: {image_key}", flush=True)
            try:
                url = image_info.get("url")
                if not url:
                    raise ValueError("Missing URL")

                content = download_image(
                    url,
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                    retries=DOWNLOAD_RETRIES,
                )
                if content is None:
                    raise ValueError("Download failed after retries")

                image_name = hashlib.sha256(image_key.encode("utf-8")).hexdigest() + ".png"
                image_path = DOWNLOAD_IMAGES_DIR / image_name
                temporary_image_path = image_path.with_suffix(".tmp")
                with Image.open(BytesIO(content)) as downloaded_image:
                    with downloaded_image.convert("RGB") as image:
                        image.save(temporary_image_path, format="PNG")
                temporary_image_path.replace(image_path)

                summary[image_key] = {
                    **image_info,
                    "downloaded_file": image_name,
                }
                print(f"Saved: {image_name}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"Skip: {image_key}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        temporary_summary_path = IMAGE_SUMMARY_PATH.with_suffix(".tmp")
        with temporary_summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)
            summary_file.write("\n")
        temporary_summary_path.replace(IMAGE_SUMMARY_PATH)
        print(f"Downloaded: {len(summary)} | Failed: {failed}", flush=True)
        print(f"Image summary: {IMAGE_SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
