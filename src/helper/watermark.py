"""
Watermark / caption removal.

EasyOCR locates text, a heuristic keeps only boxes that look like watermarks
(near an image edge or a long thin strip), and Simple LaMa inpaints them.
Ported from "Pipeline + Long Clip + Remove Watermark.ipynb" (Cell 4b).

The OCR reader and the inpainting model are loaded lazily on first use and
cached for the rest of the process. Call ``unload_watermark_tools()`` before
the dataset-level CMMD/SDQM stage to release the VRAM they hold.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.core.config import (
    WATERMARK_DILATE_ITERATIONS,
    WATERMARK_DILATE_KERNEL,
    WATERMARK_EDGE_MARGIN_RATIO,
    WATERMARK_MIN_TEXT_CONFIDENCE,
    WATERMARK_OCR_LANGUAGES,
    WATERMARK_OCR_USE_GPU,
    WATERMARK_WIDE_ASPECT_RATIO,
)

_reader = None
_inpainter = None


def _load_tools():
    """Load EasyOCR + Simple LaMa once and cache them."""
    global _reader, _inpainter
    if _reader is None or _inpainter is None:
        import easyocr
        from simple_lama_inpainting import SimpleLama

        print("Loading watermark tools (EasyOCR + Simple LaMa)...")
        _reader = easyocr.Reader(
            list(WATERMARK_OCR_LANGUAGES),
            gpu=WATERMARK_OCR_USE_GPU,
        )
        _inpainter = SimpleLama()
        print("Watermark tools ready.")
    return _reader, _inpainter


def unload_watermark_tools() -> None:
    """Drop the cached models so their VRAM can be reclaimed."""
    global _reader, _inpainter
    _reader = None
    _inpainter = None


def create_auto_watermark_mask(image: Image.Image) -> Image.Image:
    """
    Return an ``L`` mask (255 = remove) covering detected watermark text.

    A detection is kept when it sits within WATERMARK_EDGE_MARGIN_RATIO of an
    edge or when its width/height ratio exceeds WATERMARK_WIDE_ASPECT_RATIO.
    """
    reader, _ = _load_tools()

    image_np = np.array(image.convert("RGB"))
    height, width = image_np.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for bbox, _text, confidence in reader.readtext(image_np):
        if confidence < WATERMARK_MIN_TEXT_CONFIDENCE:
            continue

        points = np.array(bbox, dtype=np.int32)
        x_min, x_max = int(points[:, 0].min()), int(points[:, 0].max())
        y_min, y_max = int(points[:, 1].min()), int(points[:, 1].max())
        box_width = x_max - x_min
        box_height = y_max - y_min

        near_edge = (
            x_min < WATERMARK_EDGE_MARGIN_RATIO * width
            or x_max > (1.0 - WATERMARK_EDGE_MARGIN_RATIO) * width
            or y_min < WATERMARK_EDGE_MARGIN_RATIO * height
            or y_max > (1.0 - WATERMARK_EDGE_MARGIN_RATIO) * height
        )
        wide_strip = (
            box_width / (box_height + 1e-5)
        ) > WATERMARK_WIDE_ASPECT_RATIO

        if near_edge or wide_strip:
            cv2.fillPoly(mask, [points], 255)

    if WATERMARK_DILATE_KERNEL > 0 and WATERMARK_DILATE_ITERATIONS > 0:
        kernel = np.ones(
            (WATERMARK_DILATE_KERNEL, WATERMARK_DILATE_KERNEL),
            np.uint8,
        )
        mask = cv2.dilate(mask, kernel, iterations=WATERMARK_DILATE_ITERATIONS)

    return Image.fromarray(mask).convert("L")


def remove_watermark(image: Image.Image) -> Image.Image:
    """
    Inpaint detected watermark text. Returns the input unchanged when nothing
    watermark-like is found.
    """
    mask = create_auto_watermark_mask(image)
    if int(np.asarray(mask).max()) == 0:
        return image

    _, inpainter = _load_tools()
    rgb_image = image.convert("RGB")
    restored = inpainter(rgb_image, mask)

    if not isinstance(restored, Image.Image):
        restored = Image.fromarray(np.asarray(restored))
    restored = restored.convert("RGB")

    if restored.size != rgb_image.size:
        restored = restored.crop((0, 0, rgb_image.size[0], rgb_image.size[1]))

    return restored
