import gc
import json
import os
import re
import traceback
from io import BytesIO
from time import monotonic

import pandas as pd
import torch
from PIL import Image
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2KleinPipeline,
)

from src.core.config import (
    BASE_SEED,
    DOWNLOAD_RETRIES,
    FLUX_MODEL,
    GEN_IMAGES_DIR,
    GUIDANCE_SCALE,
    HEADERS,
    HF_HUB_CACHE,
    HF_TOKEN,
    IMAGE_SIZE,
    LIMIT_IMAGES,
    MAX_ATTEMPTS,
    MAX_CONSECUTIVE_ERRORS,
    GENERATION_TIMEOUT_SECONDS,
    MODEL_LOAD_TIMEOUT_SECONDS,
    STAGE_TIMEOUT_SECONDS,
    EVALUATION_TIMEOUT_SECONDS,
    NUM_INFERENCE_STEPS,
    OUTPUT_DIR,
    REAL_IMAGES_DIR,
    REQUEST_TIMEOUT,
    SDQM_ENABLED,
    SDQM_MIN_IMAGES,
    SDQM_OUTPUT_DIR,
    SDQM_VINFO_ENABLED,
    SDQM_YOLO_EXPORT,
    WATERMARK_REMOVAL_ENABLED,
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
    write_sdqm_status_report,
    write_metadata_jsonl,
)
from src.generation.gemma import loading_model as loading_gemma
from src.helper.image import (
    download_image,
    generate_rescue_image,
    resize_center_crop,
    save_generated_image,
)
from src.helper.loading_dataset import loading_dataset as load
from src.helper.memory import cleanup
from src.helper.runtime import stage
from src.helper.watermark import remove_watermark, unload_watermark_tools
from src.vision import build_flux_prompt
from src.vision.rescue_instruction import generate_rescue_instruction
from src.vision.scene_description import generate_scene_description


print("Hello, Cloudian 💙 Cloud")


for directory in (
    OUTPUT_DIR,
    REAL_IMAGES_DIR,
    GEN_IMAGES_DIR,
):
    os.makedirs(directory, exist_ok=True)


# ============================================================
# Memory helpers
# ============================================================

def cleanup_cuda() -> None:
    """
    Release Python garbage and unused CUDA allocator memory.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_gpu_memory(label: str) -> None:
    """
    Print current PyTorch CUDA memory usage.
    """
    if not torch.cuda.is_available():
        print(f"{label}: CUDA unavailable")
        return

    allocated = (
        torch.cuda.memory_allocated() / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved() / 1024**3
    )

    print(
        f"{label}: "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB"
    )


def print_gpu_memory_detailed(label: str) -> None:
    """
    Print PyTorch memory and nvidia-smi-visible memory.
    """
    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print(
            f"PyTorch allocated : "
            f"{allocated:.2f} GB"
        )

        print(
            f"PyTorch reserved  : "
            f"{reserved:.2f} GB"
        )

    else:
        print("PyTorch CUDA unavailable")


def release_object(obj):
    """
    Delete an object if present and aggressively clean CUDA memory.
    """
    if obj is None:
        return None

    try:
        del obj
    except Exception:
        pass

    cleanup_cuda()

    return None


# ============================================================
# Gemma loading
# ============================================================

def load_gemma_model():
    """
    Load Gemma only when it is actually required.

    Keeping Gemma lazy-loaded prevents Gemma from occupying GPU
    memory while FLUX is being initialized/generating.
    """
    print("Loading Gemma...")

    cleanup_cuda()

    with stage(
        "Load Gemma",
        MODEL_LOAD_TIMEOUT_SECONDS,
    ):
        model, processor = loading_gemma()

    print_gpu_memory(
        "GPU after Gemma load"
    )

    return model, processor


def release_gemma(
    vision_model,
    vision_processor,
):
    """
    Completely release Gemma after prompt generation.

    We intentionally delete the model instead of relying only on
    torch.cuda.empty_cache(), because empty_cache() cannot release
    tensors still referenced by vision_model.
    """
    print(
        "Releasing Gemma before FLUX..."
    )

    del vision_model
    del vision_processor

    cleanup_cuda()

    print_gpu_memory(
        "GPU after Gemma release"
    )

    return None, None


# ============================================================
# FLUX loading
# ============================================================

def load_flux_memory_safe():
    """
    Load FLUX without moving the entire model to CUDA.

    Diffusers CPU offload keeps model weights on CPU and moves only
    the required components to GPU during execution.

    This avoids the large CUDA allocation caused by:
        pipe.to("cuda")
    """
    cleanup_cuda()

    print("=" * 60)
    print(f"Loading {FLUX_MODEL}")
    print("dtype: torch.bfloat16")
    print("mode: CPU offload")
    print("=" * 60)

    pipe = Flux2KleinPipeline.from_pretrained(
        FLUX_MODEL,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN or None,
        cache_dir=str(HF_HUB_CACHE),
    )

    # --------------------------------------------------------
    # Prefer model CPU offload.
    #
    # This keeps the majority of model weights on CPU and moves
    # components to GPU only when needed.
    # --------------------------------------------------------

    try:
        pipe.enable_model_cpu_offload()

        print(
            "FLUX CPU offload enabled."
        )

    except Exception as exc:

        print(
            "WARNING: model CPU offload failed:"
            f" {type(exc).__name__}: {exc}"
        )

        # ----------------------------------------------------
        # Fallback to sequential offload.
        #
        # Sequential offload is slower but uses less VRAM.
        # ----------------------------------------------------

        try:

            pipe.enable_sequential_cpu_offload()

            print(
                "FLUX sequential CPU offload enabled."
            )

        except Exception as fallback_exc:

            print(
                "ERROR: FLUX CPU offload unavailable:"
                f" {type(fallback_exc).__name__}: "
                f"{fallback_exc}"
            )

            del pipe
            cleanup_cuda()

            raise


    pipe.set_progress_bar_config(
        disable=False,
    )

    cleanup_cuda()

    print_gpu_memory(
        "GPU after FLUX initialization"
    )

    print("=" * 60)
    print("FLUX Loaded Successfully")
    print("=" * 60)

    return pipe


# ============================================================
# Safe FLUX generation
# ============================================================

def generate_flux_safe(
    pipe,
    original_image: Image.Image,
    flux_prompt: str,
    count: int,
):
    """
    Generate an image with OOM recovery.

    Normal execution uses the offloaded FLUX pipeline.

    On OOM:
      1. Release temporary CUDA memory.
      2. Retry once.
      3. Let the caller handle a second failure.
    """

    cleanup_cuda()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:

        with stage(
            "FLUX generation",
            STAGE_TIMEOUT_SECONDS,
        ):

            generated_image = (
                generate_rescue_image(
                    pipe=pipe,
                    image=original_image,
                    prompt=flux_prompt,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    seed=BASE_SEED + count,
                )
            )

        print_gpu_memory(
            "GPU after FLUX generation"
        )

        if torch.cuda.is_available():
            peak = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            print(
                f"FLUX peak allocated: "
                f"{peak:.2f} GB"
            )

        return generated_image

    except torch.cuda.OutOfMemoryError:

        print()
        print(
            "FLUX CUDA Out Of Memory."
        )

        cleanup_cuda()

        print_gpu_memory(
            "GPU after FLUX OOM"
        )

        # ----------------------------------------------------
        # Retry once after cleanup.
        # ----------------------------------------------------

        print(
            "Retrying FLUX generation "
            "after CUDA cleanup..."
        )

        cleanup_cuda()

        try:

            with stage(
                "FLUX generation retry",
                STAGE_TIMEOUT_SECONDS,
            ):

                generated_image = (
                    generate_rescue_image(
                        pipe=pipe,
                        image=original_image,
                        prompt=flux_prompt,
                        guidance_scale=GUIDANCE_SCALE,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        seed=BASE_SEED + count,
                    )
                )

            print(
                "FLUX retry succeeded."
            )

            return generated_image

        except torch.cuda.OutOfMemoryError:

            print(
                "FLUX retry also failed with "
                "CUDA Out Of Memory."
            )

            cleanup_cuda()

            raise


# ============================================================
# Utility
# ============================================================

def make_safe_stem(image_key: str) -> str:
    """
    Build a filesystem-safe stem from the full dataset key.

    Using the full key avoids collisions between samples that share
    the same basename in different folders.
    """
    normalized_key = (
        image_key
        .replace("/", "_")
        .replace("\\", "_")
    )

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
        os.path.join(
            REAL_IMAGES_DIR,
            f"{safe_name}_real.jpg",
        ),
        quality=95,
    )

    generated_image.save(
        os.path.join(
            GEN_IMAGES_DIR,
            f"{safe_name}_gen.jpg",
        ),
        quality=95,
    )


def export_evaluation_report(
    records: list[dict],
) -> str | None:

    if not records:
        return None

    df = pd.DataFrame(records)

    csv_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_report.csv",
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    metric_cols = [
        "sc_score",
        "pq_score",
        "o_score",
        "ssim",
    ]

    print("\nMetric summary:")
    print(df[metric_cols].describe())

    return csv_path


# ============================================================
# CMMD
# ============================================================

def run_cmmd_report() -> float | None:

    if not os.path.isdir(REAL_IMAGES_DIR):
        return None

    if not os.path.isdir(GEN_IMAGES_DIR):
        return None

    real_images = [
        name
        for name in os.listdir(REAL_IMAGES_DIR)
        if name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    gen_images = [
        name
        for name in os.listdir(GEN_IMAGES_DIR)
        if name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
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

        print(
            f"CMMD Score: {cmmd_score:.4f}"
        )

        return cmmd_score

    except Exception as exc:

        print(
            f"CMMD calculation failed: {exc}"
        )

        return None


# ============================================================
# SDQM failure report
# ============================================================

def write_sdqm_failure_report(
    reason: str,
    real_image_count: int,
    synthetic_image_count: int,
) -> None:

    try:

        report_path = (
            write_sdqm_status_report(
                SDQM_OUTPUT_DIR,
                {
                    "status": "failed",
                    "reason": reason,
                    "real_image_count":
                        real_image_count,
                    "synthetic_image_count":
                        synthetic_image_count,
                },
            )
        )

        print(
            f"SDQM failure report: "
            f"{report_path.resolve()}"
        )

    except OSError as exc:

        print(
            f"Could not write SDQM failure report: "
            f"{exc}"
        )


# ============================================================
# SDQM
# ============================================================

def run_sdqm_report() -> dict[str, float] | None:

    if not SDQM_ENABLED:

        print(
            "SDQM disabled "
            "(SDQM_ENABLED=false)."
        )

        return None

    if not os.path.isdir(REAL_IMAGES_DIR):
        return None

    if not os.path.isdir(GEN_IMAGES_DIR):
        return None

    real_images = [
        name
        for name in os.listdir(REAL_IMAGES_DIR)
        if name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    gen_images = [
        name
        for name in os.listdir(GEN_IMAGES_DIR)
        if name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if (
        len(real_images) < SDQM_MIN_IMAGES
        or len(gen_images) < SDQM_MIN_IMAGES
    ):

        print(
            "SDQM skipped: need at least "
            f"{SDQM_MIN_IMAGES} real and "
            "synthetic images."
        )

        return None


    print("Computing SDQM metrics...")


    if SDQM_YOLO_EXPORT:

        print(
            "YOLO auto-labeling enabled "
            "(Grounding DINO)."
        )


    if SDQM_VINFO_ENABLED:

        (
            ultralytics_ready,
            ultralytics_message,
        ) = check_custom_ultralytics()

        if ultralytics_ready:

            print(
                "V-Info enabled "
                "(custom ultralytics detected)."
            )

        else:

            print(
                "V-Info will be skipped: "
                f"{ultralytics_message}"
            )


    cleanup()


    try:

        sdqm_metrics = compute_dataset_sdqm(
            ref_dir=REAL_IMAGES_DIR,
            eval_dir=GEN_IMAGES_DIR,
        )

        print("SDQM metrics:")

        for (
            metric_name,
            metric_value,
        ) in sorted(
            sdqm_metrics.items()
        ):

            print(
                f"  {metric_name}: "
                f"{metric_value:.4f}"
            )

        return sdqm_metrics


    except FileNotFoundError as exc:

        reason = (
            f"SDQM setup incomplete: {exc}"
        )

        print(reason)

        write_sdqm_failure_report(
            reason,
            len(real_images),
            len(gen_images),
        )

        return None


    except Exception as exc:

        reason = (
            f"SDQM calculation failed: {exc}"
        )

        print(reason)

        write_sdqm_failure_report(
            reason,
            len(real_images),
            len(gen_images),
        )

        return None


# ============================================================
# Initialize
# ============================================================

data = None

vision_model = None
vision_processor = None
pipe = None

#
# Quality evaluators are CPU-only.
#
# This is intentional.
#
with stage(
    "Load quality evaluators",
    MODEL_LOAD_TIMEOUT_SECONDS,
):

    evaluators: QualityEvaluators = (
        load_evaluators(
            device="cpu"
        )
    )


# ============================================================
# Dataset
# ============================================================

with stage(
    "Dataset initialization",
    STAGE_TIMEOUT_SECONDS,
):

    random_seed()

    data = load()


count = 0
skipped = 0
attempts = 0
consecutive_errors = 0
stop_reason = None

generation_started = monotonic()

evaluation_records = []

metadata_dir = os.path.join(
    OUTPUT_DIR,
    "_metadata",
)

os.makedirs(
    metadata_dir,
    exist_ok=True,
)


print("=" * 80)
print(
    f"Pipeline started - target: "
    f"{LIMIT_IMAGES} images"
)
print("=" * 80)


# ============================================================
# Main pipeline
# ============================================================

for img_key, img_info in data.items():

    if count >= LIMIT_IMAGES:
        break


    if attempts >= MAX_ATTEMPTS:

        stop_reason = (
            f"MAX_ATTEMPTS reached ({attempts})"
        )

        break


    if (
        consecutive_errors
        >= MAX_CONSECUTIVE_ERRORS
    ):

        stop_reason = (
            "MAX_CONSECUTIVE_ERRORS reached "
            f"({consecutive_errors})"
        )

        break


    if (
        monotonic() - generation_started
        >= GENERATION_TIMEOUT_SECONDS
    ):

        stop_reason = (
            "Generation time budget exhausted"
        )

        break


    print("=" * 80)
    print(f"Processing: {img_key}")
    print("=" * 80)


    original_image = None
    generated_image = None
    scene_description = None
    rescue_instruction = None
    flux_prompt = None


    try:

        # ====================================================
        # Positive labels
        # ====================================================

        incidents = img_info.get(
            "incidents",
            {},
        )

        positive_incidents = [
            incident
            for incident, value
            in incidents.items()
            if value == 1
        ]


        if not positive_incidents:

            print(
                f"Skip: {img_key} "
                "(No Positive Labels)"
            )

            skipped += 1

            continue


        # ====================================================
        # Output
        # ====================================================

        safe_name = make_safe_stem(
            img_key
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{safe_name}.png",
        )


        if os.path.exists(output_path):

            print(
                f"Skip: {img_key} "
                "(Already Generated)"
            )

            skipped += 1

            continue


        # ====================================================
        # URL
        # ====================================================

        url = img_info.get("url")


        if not url:

            print(
                "Skip: Missing URL"
            )

            skipped += 1

            continue


        attempts += 1


        # ====================================================
        # Download
        # ====================================================

        with stage(
            "Download image",
            REQUEST_TIMEOUT
            * (DOWNLOAD_RETRIES + 1),
        ):

            content = download_image(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                retries=DOWNLOAD_RETRIES,
            )


        if content is None:

            print(
                "Skip: Download Failed"
            )

            consecutive_errors += 1
            skipped += 1

            continue


        # ====================================================
        # Decode image
        # ====================================================

        try:

            original_image = (
                Image.open(
                    BytesIO(content)
                )
                .convert("RGB")
            )

        except Exception as exc:

            print(
                f"Skip Invalid Image: {exc}"
            )

            consecutive_errors += 1
            skipped += 1

            continue

        if WATERMARK_REMOVAL_ENABLED:

            try:

                with stage(
                    "Watermark removal",
                    STAGE_TIMEOUT_SECONDS,
                ):

                    original_image = remove_watermark(
                        original_image
                    )

            except Exception as exc:

                print(
                    f"Watermark removal skipped: {exc}"
                )

        original_image = resize_center_crop(original_image, IMAGE_SIZE)

        if torch.cuda.is_available():
            print(
                "GPU Memory:",
                round(torch.cuda.memory_allocated() / 1024**3, 2),
                "GB",
            )

        if vision_model is None or vision_processor is None:
            try:
                vision_model, vision_processor = load_gemma_model()
            except torch.cuda.OutOfMemoryError:
                print("Gemma CUDA Out Of Memory.")
                cleanup_cuda()
                consecutive_errors += 1
                skipped += 1
                del original_image
                continue

        try:
            with stage("Gemma scene description", STAGE_TIMEOUT_SECONDS):
                scene_description = generate_scene_description(
                    image=original_image,
                    vision_model=vision_model,
                    vision_processor=vision_processor,
                )
        except Exception as exc:

            print(
                f"Generate Image Error: {exc}"
            )

            consecutive_errors += 1
            skipped += 1

            del original_image

            with stage(
                "Cleanup",
                STAGE_TIMEOUT_SECONDS,
            ):
                cleanup()

            continue


        print("\nScene Description")
        print("-" * 60)
        print(scene_description)


        # ====================================================
        # Gemma rescue instruction
        # ====================================================

        try:

            with stage(
                "Gemma rescue instruction",
                STAGE_TIMEOUT_SECONDS,
            ):

                rescue_instruction = (
                    generate_rescue_instruction(
                        scene_description=scene_description,
                        vision_model=vision_model,
                        vision_processor=vision_processor,
                    )
                )

        except Exception as exc:

            print(
                f"LLM Error: {exc}"
            )

            consecutive_errors += 1
            skipped += 1

            del original_image

            with stage(
                "Cleanup",
                STAGE_TIMEOUT_SECONDS,
            ):
                cleanup()

            continue


        print("\nEditing Instruction")
        print("-" * 60)
        print(rescue_instruction)


        # ====================================================
        # Build FLUX prompt
        # ====================================================

        flux_prompt = build_flux_prompt(
            scene_description,
            rescue_instruction,
        )

        print("\nFLUX Prompt")
        print("-" * 60)
        print(flux_prompt)


        # ====================================================
        # RELEASE GEMMA
        #
        # No FLUX can be initialized until Gemma is released.
        # ====================================================

        (
            vision_model,
            vision_processor,
        ) = release_gemma(
            vision_model,
            vision_processor,
        )


        # ====================================================
        # Load FLUX lazily
        #
        # FLUX is loaded ONLY after Gemma has been released.
        # ====================================================

        if pipe is None:

            try:

                with stage(
                    "Load FLUX",
                    MODEL_LOAD_TIMEOUT_SECONDS,
                ):

                    pipe = load_flux_memory_safe()

            except torch.cuda.OutOfMemoryError:

                print(
                    "FLUX CUDA Out Of Memory "
                    "during initialization."
                )

                cleanup_cuda()

                print(
                    "FLUX could not be initialized "
                    "with available GPU memory."
                )

                consecutive_errors += 1
                skipped += 1

                del original_image

                continue


        # ====================================================
        # FLUX generation
        # ====================================================

        try:

            generated_image = generate_flux_safe(
                pipe=pipe,
                original_image=original_image,
                flux_prompt=flux_prompt,
                count=count,
            )

        except torch.cuda.OutOfMemoryError:

            print(
                "FLUX CUDA Out Of Memory."
            )

            consecutive_errors += 1
            skipped += 1

            del original_image

            with stage(
                "Cleanup",
                STAGE_TIMEOUT_SECONDS,
            ):
                cleanup()

            continue


        # ====================================================
        # Quality evaluation
        #
        # Evaluators are CPU-only.
        #
        # FLUX remains offloaded, so quality evaluation does not
        # compete for the GPU.
        # ====================================================

        try:

            with stage(
                "Quality evaluation",
                STAGE_TIMEOUT_SECONDS,
            ):

                (
                    sc_score,
                    pq_score,
                ) = evaluate_quality(
                    evaluators,
                    generated_image,
                    flux_prompt,
                )

                o_score = compute_o_score(
                    sc_score,
                    pq_score,
                )

                ssim_val = compute_ssim(
                    original_image,
                    generated_image,
                )

        except Exception as exc:

            print(
                "Quality evaluation failed: "
                f"{type(exc).__name__}: {exc}"
            )

            consecutive_errors += 1
            skipped += 1

            del original_image
            del generated_image

            with stage(
                "Cleanup",
                STAGE_TIMEOUT_SECONDS,
            ):
                cleanup()

            continue


        print(
            f"O_Score: {o_score:.4f} | "
            f"SSIM: {ssim_val:.4f}"
        )


        # ====================================================
        # Quality gate
        # ====================================================

        if not passes_quality_gate(
            o_score,
            ssim_val,
        ):

            consecutive_errors = 0

            print(
                "Rejected by quality gate"
            )

            skipped += 1

            del original_image
            del generated_image

            with stage(
                "Cleanup",
                STAGE_TIMEOUT_SECONDS,
            ):
                cleanup()

            continue


        # ====================================================
        # Save generated image
        # ====================================================

        output_path = save_generated_image(
            image=generated_image,
            output_dir=OUTPUT_DIR,
            image_name=f"{safe_name}.png",
        )

        save_reference_images(
            safe_name,
            original_image,
            generated_image,
        )

        print(
            f"Saved -> {output_path}"
        )


        # ====================================================
        # Metadata
        # ====================================================

        metadata = {
            "image_key": img_key,
            "labels": positive_incidents,
            "url": url,
            "scene_description": scene_description,
            "editing_instruction": rescue_instruction,
            "flux_prompt": flux_prompt,
            "sc_score": round(
                sc_score,
                4,
            ),
            "pq_score": round(
                pq_score,
                4,
            ),
            "o_score": round(
                o_score,
                4,
            ),
            "ssim": round(
                ssim_val,
                4,
            ),
            "output_path": output_path,
        }


        evaluation_records.append(
            metadata
        )


        metadata_path = os.path.join(
            metadata_dir,
            f"{safe_name}.json",
        )


        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )


        # ====================================================
        # Progress
        # ====================================================

        consecutive_errors = 0
        count += 1

        print(
            f"Progress: "
            f"{count}/{LIMIT_IMAGES}"
        )


        # ====================================================
        # Per-sample cleanup
        # ====================================================

        del original_image
        del generated_image
        del scene_description
        del rescue_instruction
        del flux_prompt

        with stage(
            "Cleanup",
            STAGE_TIMEOUT_SECONDS,
        ):
            cleanup()


    except Exception:

        print(
            "\nUnexpected Error"
        )

        consecutive_errors += 1

        traceback.print_exc()


        try:

            with stage(
                "Cleanup",
                STAGE_TIMEOUT_SECONDS,
            ):
                cleanup()

        except Exception:
            cleanup_cuda()


        skipped += 1

        continue


# ============================================================
# Generation summary
# ============================================================

print("\n" + "=" * 80)
print("Generation loop finished")
print(f"Attempted : {attempts}")
print(f"Generated : {count}")
print(f"Skipped   : {skipped}")
print(f"Output    : {OUTPUT_DIR}")


# ============================================================
# Evaluation report
# ============================================================

csv_path = export_evaluation_report(
    evaluation_records
)

if csv_path:

    print(
        f"CSV Report: {csv_path}"
    )


# ============================================================
# Dataset not completed
# ============================================================

if count < LIMIT_IMAGES:

    stop_reason = (
        stop_reason
        or "Dataset exhausted before "
        "reaching target"
    )

    write_metadata_jsonl(
        evaluation_records,
        OUTPUT_DIR,
    )

    print(
        f"Stopped: {stop_reason}. "
        "Dataset evaluation skipped.",
        flush=True,
    )

    raise SystemExit(2)

with stage("Release generation models", STAGE_TIMEOUT_SECONDS):
    unload_watermark_tools()
    del vision_model, vision_processor, pipe, evaluators
    cleanup()


# ============================================================
# CMMD report
# ============================================================

with stage(
    "CMMD report",
    EVALUATION_TIMEOUT_SECONDS,
):

    run_cmmd_report()


# ============================================================
# SDQM report
# ============================================================

with stage(
    "SDQM report",
    EVALUATION_TIMEOUT_SECONDS,
):

    sdqm_metrics = run_sdqm_report()


if sdqm_metrics:

    evaluation_records = (
        attach_sdqm_metadata(
            evaluation_records,
            sdqm_metrics,
        )
    )


# ============================================================
# Final metadata
# ============================================================

metadata_jsonl_path = (
    write_metadata_jsonl(
        evaluation_records,
        OUTPUT_DIR,
    )
)


if metadata_jsonl_path:

    print(
        f"Metadata JSONL: "
        f"{metadata_jsonl_path}"
    )
