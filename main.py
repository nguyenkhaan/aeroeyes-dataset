import json
import os
import re
import traceback
from io import BytesIO

import pandas as pd
import torch
from PIL import Image

from src.core.config import (
    BASE_SEED,
    DOWNLOAD_RETRIES,
    GEN_IMAGES_DIR,
    GUIDANCE_SCALE,
    HEADERS,
    IMAGE_SIZE,
    LIMIT_IMAGES,
    NUM_INFERENCE_STEPS,
    OUTPUT_DIR,
    REAL_IMAGES_DIR,
    REQUEST_TIMEOUT,
    SDQM_ENABLED,
    SDQM_MIN_IMAGES,
    SDQM_VINFO_ENABLED,
    SDQM_YOLO_EXPORT,
    random_seed,
)
from src.evaluation import (
    QualityEvaluators,
    attach_sdqm_metadata,
    check_custom_ultralytics,
    compute_dataset_cmmd,
    compute_dataset_sdqm,
    compute_o_score,
    compute_ssim,
    evaluate_quality,
    load_evaluators,
    passes_quality_gate,
    write_metadata_jsonl,
)
from src.generation.flux import loading_model as loading_flux
from src.generation.gemma import loading_model as loading_gemma
from src.helper.image import (
    download_image,
    generate_rescue_image,
    resize_center_crop,
    save_generated_image,
)
from src.helper.loading_dataset import loading_dataset as load
from src.helper.memory import cleanup
from src.vision import build_flux_prompt
from src.vision.rescue_instruction import generate_rescue_instruction
from src.vision.scene_description import generate_scene_description

print("Hello, Cloudian 💙 Cloud")

for directory in (OUTPUT_DIR, REAL_IMAGES_DIR, GEN_IMAGES_DIR):
    os.makedirs(directory, exist_ok=True)

random_seed()
data = load()


def make_safe_stem(image_key: str) -> str:
    """
    Build a filesystem-safe stem from the full dataset key.

    Using the full key avoids collisions between samples that share the
    same basename in different folders.
    """
    normalized_key = image_key.replace("/", "_").replace("\\", "_")
    return re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        normalized_key,
    ).strip("_")


def save_reference_images(
    safe_name: str,
    original_image: Image.Image,
    generated_image: Image.Image,
) -> None:
    original_image.save(
        os.path.join(REAL_IMAGES_DIR, f"{safe_name}_real.jpg"),
        quality=95,
    )
    generated_image.save(
        os.path.join(GEN_IMAGES_DIR, f"{safe_name}_gen.jpg"),
        quality=95,
    )


def export_evaluation_report(records: list[dict]) -> str | None:
    if not records:
        return None

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_report.csv")
    df.to_csv(csv_path, index=False)

    metric_cols = ["sc_score", "pq_score", "o_score", "ssim"]
    print("\nMetric summary:")
    print(df[metric_cols].describe())

    return csv_path


def run_cmmd_report() -> float | None:
    if not os.path.isdir(REAL_IMAGES_DIR) or not os.path.isdir(GEN_IMAGES_DIR):
        return None

    real_images = [
        name
        for name in os.listdir(REAL_IMAGES_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    gen_images = [
        name
        for name in os.listdir(GEN_IMAGES_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not real_images or not gen_images:
        return None

    print("Computing CMMD score...")
    cleanup()
    try:
        cmmd_score = compute_dataset_cmmd(
            ref_dir=REAL_IMAGES_DIR,
            eval_dir=GEN_IMAGES_DIR,
        )
        print(f"CMMD Score: {cmmd_score:.4f}")
        return cmmd_score
    except Exception as exc:
        print(f"CMMD calculation failed: {exc}")
        return None


def run_sdqm_report() -> dict[str, float] | None:
    if not SDQM_ENABLED:
        print("SDQM disabled (SDQM_ENABLED=false).")
        return None

    if not os.path.isdir(REAL_IMAGES_DIR) or not os.path.isdir(GEN_IMAGES_DIR):
        return None

    real_images = [
        name
        for name in os.listdir(REAL_IMAGES_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    gen_images = [
        name
        for name in os.listdir(GEN_IMAGES_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(real_images) < SDQM_MIN_IMAGES or len(gen_images) < SDQM_MIN_IMAGES:
        print(
            "SDQM skipped: need at least "
            f"{SDQM_MIN_IMAGES} real and synthetic images."
        )
        return None

    print("Computing SDQM metrics...")
    if SDQM_YOLO_EXPORT:
        print("YOLO auto-labeling enabled (Grounding DINO).")
    if SDQM_VINFO_ENABLED:
        ultralytics_ready, ultralytics_message = check_custom_ultralytics()
        if ultralytics_ready:
            print("V-Info enabled (custom ultralytics detected).")
        else:
            print(f"V-Info will be skipped: {ultralytics_message}")
    cleanup()
    try:
        sdqm_metrics = compute_dataset_sdqm(
            ref_dir=REAL_IMAGES_DIR,
            eval_dir=GEN_IMAGES_DIR,
        )
        print("SDQM metrics:")
        for metric_name, metric_value in sorted(sdqm_metrics.items()):
            print(f"  {metric_name}: {metric_value:.4f}")
        return sdqm_metrics
    except FileNotFoundError as exc:
        print(f"SDQM setup incomplete: {exc}")
        return None
    except Exception as exc:
        print(f"SDQM calculation failed: {exc}")
        return None


vision_model, vision_processor = loading_gemma()
pipe = loading_flux()
evaluators: QualityEvaluators = load_evaluators()

count = 0
skipped = 0
evaluation_records = []
metadata_dir = os.path.join(OUTPUT_DIR, "_metadata")
os.makedirs(metadata_dir, exist_ok=True)

print("=" * 80)
print(f"Pipeline started - target: {LIMIT_IMAGES} images")
print("=" * 80)

for img_key, img_info in data.items():
    if count >= LIMIT_IMAGES:
        break

    print("=" * 80)
    print(f"Processing: {img_key}")
    print("=" * 80)

    try:
        incidents = img_info.get("incidents", {})
        positive_incidents = [
            incident
            for incident, value in incidents.items()
            if value == 1
        ]

        if not positive_incidents:
            print(f"Skip: {img_key} (No Positive Labels)")
            skipped += 1
            continue

        safe_name = make_safe_stem(img_key)
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")

        if os.path.exists(output_path):
            print(f"Skip: {img_key} (Already Generated)")
            skipped += 1
            continue

        url = img_info.get("url")
        if not url:
            print("Skip: Missing URL")
            skipped += 1
            continue

        content = download_image(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            retries=DOWNLOAD_RETRIES,
        )
        if content is None:
            print("Skip: Download Failed")
            skipped += 1
            continue

        try:
            original_image = Image.open(BytesIO(content)).convert("RGB")
        except Exception as exc:
            print(f"Skip Invalid Image: {exc}")
            skipped += 1
            continue

        original_image = resize_center_crop(original_image, IMAGE_SIZE)

        if torch.cuda.is_available():
            print(
                "GPU Memory:",
                round(torch.cuda.memory_allocated() / 1024**3, 2),
                "GB",
            )

        try:
            scene_description = generate_scene_description(
                image=original_image,
                vision_model=vision_model,
                vision_processor=vision_processor,
            )
        except Exception as exc:
            print(f"Generate Image Error: {exc}")
            skipped += 1
            del original_image
            cleanup()
            continue

        print("\nScene Description")
        print("-" * 60)
        print(scene_description)

        try:
            rescue_instruction = generate_rescue_instruction(
                scene_description=scene_description,
                vision_model=vision_model,
                vision_processor=vision_processor,
            )
        except Exception as exc:
            print(f"LLM Error: {exc}")
            skipped += 1
            del original_image
            cleanup()
            continue

        print("\nEditing Instruction")
        print("-" * 60)
        print(rescue_instruction)

        flux_prompt = build_flux_prompt(scene_description, rescue_instruction)
        print("\nFLUX Prompt")
        print("-" * 60)
        print(flux_prompt)

        try:
            generated_image = generate_rescue_image(
                pipe=pipe,
                image=original_image,
                prompt=flux_prompt,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                seed=BASE_SEED + count,
            )
        except torch.cuda.OutOfMemoryError:
            print("FLUX CUDA Out Of Memory")
            skipped += 1
            del original_image
            cleanup()
            continue

        sc_score, pq_score = evaluate_quality(
            evaluators,
            generated_image,
            flux_prompt,
        )
        o_score = compute_o_score(sc_score, pq_score)
        ssim_val = compute_ssim(original_image, generated_image)

        print(f"O_Score: {o_score:.4f} | SSIM: {ssim_val:.4f}")

        if not passes_quality_gate(o_score, ssim_val):
            print("Rejected by quality gate")
            skipped += 1
            del original_image
            del generated_image
            cleanup()
            continue

        output_path = save_generated_image(
            image=generated_image,
            output_dir=OUTPUT_DIR,
            image_name=f"{safe_name}.png",
        )
        save_reference_images(safe_name, original_image, generated_image)
        print(f"Saved -> {output_path}")

        metadata = {
            "image_key": img_key,
            "labels": positive_incidents,
            "url": url,
            "scene_description": scene_description,
            "editing_instruction": rescue_instruction,
            "flux_prompt": flux_prompt,
            "sc_score": round(sc_score, 4),
            "pq_score": round(pq_score, 4),
            "o_score": round(o_score, 4),
            "ssim": round(ssim_val, 4),
            "output_path": output_path,
        }
        evaluation_records.append(metadata)

        metadata_path = os.path.join(metadata_dir, f"{safe_name}.json")
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

        count += 1
        print(f"Progress: {count}/{LIMIT_IMAGES}")

        del original_image
        del generated_image
        del scene_description
        del rescue_instruction
        del flux_prompt
        cleanup()

    except Exception:
        print("\nUnexpected Error")
        traceback.print_exc()
        cleanup()
        skipped += 1
        continue

print("\n" + "=" * 80)
print("Finished")
print(f"Generated : {count}")
print(f"Skipped   : {skipped}")
print(f"Output    : {OUTPUT_DIR}")

csv_path = export_evaluation_report(evaluation_records)
if csv_path:
    print(f"CSV Report: {csv_path}")

run_cmmd_report()
sdqm_metrics = run_sdqm_report()
if sdqm_metrics:
    evaluation_records = attach_sdqm_metadata(evaluation_records, sdqm_metrics)

metadata_jsonl_path = write_metadata_jsonl(evaluation_records, OUTPUT_DIR)
if metadata_jsonl_path:
    print(f"Metadata JSONL: {metadata_jsonl_path}")
