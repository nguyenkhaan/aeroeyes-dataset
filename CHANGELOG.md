# Changelog

## Unreleased

### Added
- Watermark / caption removal (`src/helper/watermark.py`): EasyOCR detection +
  Simple LaMa inpainting, applied to the downloaded image before resize/crop.
  Ported from "Pipeline + Long Clip + Remove Watermark.ipynb" (Cell 4b).
  Toggle with `WATERMARK_REMOVAL_ENABLED` (default `true`); tuning knobs
  `WATERMARK_OCR_LANGUAGES`, `WATERMARK_OCR_USE_GPU`,
  `WATERMARK_MIN_TEXT_CONFIDENCE`, `WATERMARK_EDGE_MARGIN_RATIO`,
  `WATERMARK_WIDE_ASPECT_RATIO`, `WATERMARK_DILATE_KERNEL`,
  `WATERMARK_DILATE_ITERATIONS`.
- Long-CLIP for the SC score (`src/evaluation/quality.py`): optional
  `zer0int/LongCLIP-L-Diffusers` text encoder (248 tokens) so the full FLUX
  prompt is scored instead of being truncated at 77 tokens. Toggle with
  `LONG_CLIP_ENABLED` (default `true`); `LONG_CLIP_MODEL_ID`,
  `LONG_CLIP_MAX_TOKENS` override the checkpoint and limit.
- `requirements.txt`: `easyocr`, `simple-lama-inpainting`.
