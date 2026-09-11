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

### Fixed
- SDQM crashed at the end of every run with `NameError: SDQM_REPORT_FILENAME`.
  Commit 49a7ae6 ("fix: sqdm import file") rewrote the top of
  `src/evaluation/sdqm.py` and dropped the `SDQM_REPORT_FILENAME` constant and
  `write_sdqm_status_report()` added in ece94a0, but left the reference at the
  bottom of `compute_dataset_sdqm()`. Both are restored.
- SDQM produced zero metrics because the wrapper called upstream
  `calculate_sdqm(..., dataset="N/A")`. Upstream only binds its internal
  `detected_dataset` inside the `if dataset == "auto"` branch, so any other
  value raised `UnboundLocalError` and the per-file `except` dropped every
  metric. Now passes `dataset="auto"`.
- `compute_dataset_sdqm()` now writes `sdqm_report.json` with
  `status: "failed"` and the exception before re-raising, so a headless run
  always leaves a diagnostic artifact.
