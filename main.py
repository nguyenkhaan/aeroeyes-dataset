"""Prepare prompts, then generate images with one resident model per phase."""
import hashlib
import json
from pathlib import Path
from time import monotonic

import torch
from PIL import Image

from src.core.config import (
    BASE_SEED, DOWNLOAD_IMAGES_DIR, GENERATION_TIMEOUT_SECONDS,
    GEN_IMAGES_DIR, GUIDANCE_SCALE, IMAGE_SIZE, IMAGE_SUMMARY_PATH,
    LIMIT_IMAGES, MAX_ATTEMPTS, MAX_CONSECUTIVE_ERRORS,
    MODEL_LOAD_TIMEOUT_SECONDS, NUM_INFERENCE_STEPS, OUTPUT_DIR,
    REAL_IMAGES_DIR, STAGE_TIMEOUT_SECONDS, WATERMARK_REMOVAL_ENABLED,
    random_seed,
)
from src.generation.flux import loading_model as loading_flux
from src.generation.gemma import loading_model as loading_gemma
from src.helper.image import generate_rescue_image, resize_center_crop
from src.helper.loading_dataset import loading_dataset
from src.helper.memory import cleanup as cleanup_cuda
from src.helper.runtime import stage
from src.helper.watermark import remove_watermark, unload_watermark_tools
from src.vision import build_flux_prompt
from src.vision.rescue_instruction import generate_rescue_instruction
from src.vision.scene_description import generate_scene_description


def write_json(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_image(image: Image.Image, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def budget_exhausted(started: float) -> bool:
    return GENERATION_TIMEOUT_SECONDS > 0 and monotonic() - started >= GENERATION_TIMEOUT_SECONDS


def error_limit_reached(errors: int) -> bool:
    return MAX_CONSECUTIVE_ERRORS > 0 and errors >= MAX_CONSECUTIVE_ERRORS


def load_gemma_model():
    with stage("Load Gemma", MODEL_LOAD_TIMEOUT_SECONDS):
        return loading_gemma()


def load_flux_memory_safe():
    with stage("Load resident FLUX", MODEL_LOAD_TIMEOUT_SECONDS):
        return loading_flux()


def generate_flux_safe(pipe, original_image: Image.Image, flux_prompt: str, count: int):
    for attempt in range(2):
        try:
            with stage("FLUX generation", STAGE_TIMEOUT_SECONDS):
                return generate_rescue_image(
                    pipe=pipe,
                    image=original_image,
                    prompt=flux_prompt,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    seed=BASE_SEED + count,
                )
        except torch.cuda.OutOfMemoryError:
            if attempt == 1:
                raise
            print("FLUX OOM; releasing failed-call tensors before one retry.", flush=True)
        # The exception traceback must be gone before collecting its tensors.
        cleanup_cuda()


def completed_sample(stem: str) -> dict | None:
    metadata_path = Path(OUTPUT_DIR) / "_metadata" / f"{stem}.json"
    if not metadata_path.is_file():
        return None
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    if all(Path(record[key]).is_file() for key in (
        "output_path", "reference_path", "generated_reference_path",
    )):
        return record
    return None


def prepare_samples(data: dict, started: float) -> list[dict]:
    records = []
    model = processor = None
    attempts = errors = 0
    prepared_dir = Path(OUTPUT_DIR) / "_prepared"
    try:
        for image_key, info in data.items():
            if len(records) >= LIMIT_IMAGES or budget_exhausted(started):
                break
            if error_limit_reached(errors):
                print("Preparation stopped: MAX_CONSECUTIVE_ERRORS reached.", flush=True)
                break
            if not isinstance(info, dict):
                continue
            incidents = info.get("incidents") or {}
            if not isinstance(incidents, dict):
                continue
            labels = [name for name, value in incidents.items() if value == 1]
            if not labels:
                continue
            stem = hashlib.sha256(image_key.encode("utf-8")).hexdigest()
            try:
                completed = completed_sample(stem)
                if completed is not None:
                    records.append(completed)
                    errors = 0
                    continue
                prepared_path = prepared_dir / f"{stem}.json"
                reference_path = prepared_dir / f"{stem}.png"
                if prepared_path.is_file() and reference_path.is_file():
                    record = json.loads(prepared_path.read_text(encoding="utf-8"))
                    if record.get("image_size") == IMAGE_SIZE:
                        records.append(record)
                        errors = 0
                        continue
                if MAX_ATTEMPTS > 0 and attempts >= MAX_ATTEMPTS:
                    print("Preparation stopped: MAX_ATTEMPTS reached.", flush=True)
                    break
                attempts += 1
                filename = info.get("downloaded_file")
                if not isinstance(filename, str) or Path(filename).name != filename:
                    raise ValueError("Invalid downloaded filename")
                with Image.open(DOWNLOAD_IMAGES_DIR / filename) as downloaded:
                    original = downloaded.convert("RGB")
                if WATERMARK_REMOVAL_ENABLED:
                    try:
                        with stage("Watermark removal", STAGE_TIMEOUT_SECONDS):
                            original = remove_watermark(original)
                    except torch.cuda.OutOfMemoryError:
                        raise
                    except Exception as exc:
                        print(f"Watermark removal skipped: {exc}", flush=True)
                original = resize_center_crop(original, IMAGE_SIZE)
                if model is None:
                    model, processor = load_gemma_model()
                with stage("Gemma scene description", STAGE_TIMEOUT_SECONDS):
                    description = generate_scene_description(original, model, processor)
                with stage("Gemma rescue instruction", STAGE_TIMEOUT_SECONDS):
                    instruction = generate_rescue_instruction(description, model, processor)
                record = {
                    "image_key": image_key,
                    "stem": stem,
                    "labels": labels,
                    "url": info.get("url"),
                    "scene_description": description,
                    "editing_instruction": instruction,
                    "flux_prompt": build_flux_prompt(description, instruction),
                    "image_size": IMAGE_SIZE,
                    "seed": BASE_SEED + len(records),
                    "prepared_image_path": str(reference_path),
                    "output_path": str(Path(OUTPUT_DIR) / f"{stem}.png"),
                    "reference_path": str(Path(REAL_IMAGES_DIR) / f"{stem}.png"),
                    "generated_reference_path": str(Path(GEN_IMAGES_DIR) / f"{stem}.png"),
                }
                save_image(original, reference_path)
                write_json(prepared_path, record)
                records.append(record)
                errors = 0
                print(f"Prepared: {len(records)}/{LIMIT_IMAGES}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print("Preparation OOM; stopping this phase. Completed prompts are saved.", flush=True)
                break
            except Exception as exc:
                errors += 1
                print(f"Preparation failed for {image_key}: {type(exc).__name__}: {exc}", flush=True)
            finally:
                original = None
    finally:
        # Release owners before clearing the allocator at the phase boundary.
        model = processor = None
        unload_watermark_tools()
        cleanup_cuda()
    return records


def render_samples(records: list[dict], started: float) -> int:
    count = errors = attempts = 0
    pipe = None
    try:
        for record in records:
            if count >= LIMIT_IMAGES or budget_exhausted(started) or error_limit_reached(errors):
                break
            if completed_sample(record["stem"]) is not None:
                count += 1
                continue
            if MAX_ATTEMPTS > 0 and attempts >= MAX_ATTEMPTS:
                break
            attempts += 1
            try:
                if pipe is None:
                    pipe = load_flux_memory_safe()
                with Image.open(record["prepared_image_path"]) as reference:
                    original = reference.convert("RGB")
                generated = generate_flux_safe(
                    pipe, original, record["flux_prompt"], record["seed"] - BASE_SEED,
                )
                save_image(original, Path(record["reference_path"]))
                save_image(generated, Path(record["generated_reference_path"]))
                save_image(generated, Path(record["output_path"]))
                # Metadata is the commit marker: incomplete samples are retried on resume.
                write_json(Path(OUTPUT_DIR) / "_metadata" / f'{record["stem"]}.json', record)
                count += 1
                errors = 0
                print(f"Generated: {count}/{LIMIT_IMAGES}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(
                    "Persistent CUDA OOM; stopping instead of repeating failing samples. "
                    "Free GPU memory or lower IMAGE_SIZE and rerun. Saved work is resumable.",
                    flush=True,
                )
                break
            except Exception as exc:
                errors += 1
                print(f'Generation failed for {record["image_key"]}: {type(exc).__name__}: {exc}', flush=True)
            finally:
                original = generated = None
    finally:
        pipe = None
        cleanup_cuda()
    return count


def main() -> int:
    for directory in (
        Path(OUTPUT_DIR), Path(REAL_IMAGES_DIR), Path(GEN_IMAGES_DIR),
        Path(OUTPUT_DIR) / "_prepared", Path(OUTPUT_DIR) / "_metadata",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    random_seed()
    started = monotonic()
    with stage("Dataset initialization", STAGE_TIMEOUT_SECONDS):
        data = loading_dataset(IMAGE_SUMMARY_PATH)
    records = prepare_samples(data, started)
    count = render_samples(records, started)
    print(f"Generation complete: {count}/{LIMIT_IMAGES}. Output: {OUTPUT_DIR}", flush=True)
    print("Run python main_evaluation.py to evaluate saved images.", flush=True)
    return 0 if count >= LIMIT_IMAGES else 2


if __name__ == "__main__":
    raise SystemExit(main())
