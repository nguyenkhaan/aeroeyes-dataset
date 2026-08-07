# Hello with everybody 
import os 
import torch 
import re 
import traceback
from io import BytesIO
import json 
from src.core.config import (
        OUTPUT_DIR, 
        BASE_SEED,
        random_seed, 
        LIMIT_IMAGES, 
        HEADERS, 
        DOWNLOAD_RETRIES, 
        IMAGE_SIZE, 
        NUM_INFERENCE_STEPS,
        GUIDANCE_SCALE, 
        REQUEST_TIMEOUT)
from PIL import Image
from src.helper.loading_dataset import loading_dataset as load 
from src.generation.gemma import loading_model as loading_gemma
from src.generation.flux import loading_model as loading_flux 
from src.vision.rescue_instruction import generate_rescue_instruction 
from src.vision.scene_description import generate_scene_description
from src.vision import build_flux_prompt
from src.helper.image import (
    download_image,
    resize_center_crop,
    generate_rescue_image,
    save_generated_image,
)
from src.helper.memory import cleanup
print('Hello, Cloudian 💙 Cloud') 

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

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

# Loading model 
vision_model, vision_processor = loading_gemma() 
pipe = loading_flux() 


# Starting pipeline 
count = 0
skipped = 0
evaluation_records = []
metadata_dir = os.path.join(
    OUTPUT_DIR,
    "_metadata",
)

os.makedirs(
    metadata_dir,
    exist_ok=True,
)

for img_key, img_info in data.items():

    if count >= LIMIT_IMAGES:
        break

    print("=" * 80)
    print(f"Processing: {img_key}")
    print("=" * 80)

    try:

        # ==================================================
        # 1. Check Positive Labels
        # ==================================================

        incidents = img_info.get(
            "incidents",
            {}
        )

        positive_incidents = [
            incident
            for incident, value in incidents.items()
            if value == 1
        ]

        if not positive_incidents:
            print(
                f"Skip: {img_key} (No Positive Labels)"
            )

            skipped += 1
            continue

        # ==================================================
        # 2. Resume Check
        # ==================================================

        safe_name = make_safe_stem(img_key)

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{safe_name}.png",
        )

        if os.path.exists(output_path):

            print(
                f"Skip: {img_key} (Already Generated)"
            )

            skipped += 1
            continue

        # ==================================================
        # 3. Download Image
        # ==================================================

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
            original_image = Image.open(
                BytesIO(content)
            ).convert("RGB")

        except Exception as e:

            print(
                f"Skip Invalid Image: {e}"
            )
            skipped += 1
            continue
        original_image = resize_center_crop(
            original_image,
            IMAGE_SIZE,
        )
        if torch.cuda.is_available():
            print(
                "GPU Memory:",
                round(
                    torch.cuda.memory_allocated()
                    / 1024 ** 3,
                    2,
                ),
                "GB",
            )
        # ==================================================
        # 4. Vision Understanding
        # ==================================================

        try:
            scene_description = generate_scene_description(
                image=original_image, 
                vision_model=vision_model, 
                vision_processor=vision_processor
            )
        except Exception as e:
            print(
                f"Generate Image Error: {e}"
            )
            skipped += 1
            del original_image
            cleanup()
            continue

        print("\nScene Description")
        print("-" * 60)
        print(scene_description)
        # ==================================================
        # 5. Generate Rescue Instruction
        # ==================================================
        try:

            rescue_instruction = generate_rescue_instruction(
                scene_description=scene_description,
                vision_model=vision_model, 
                vision_processor=vision_processor
            )
        except Exception as e:

            print(
                f"LLM Error: {e}"
            )
            skipped += 1
            del original_image
            cleanup()
            continue
        print("\nEditing Instruction")
        print("-" * 60)
        print(rescue_instruction)
        # ==================================================
        # 6. Build FLUX Prompt
        # ==================================================
        flux_prompt = build_flux_prompt(
            scene_description,
            rescue_instruction,
        )
        print("\nFLUX Prompt")
        print("-" * 60)
        print(flux_prompt)
        # ==================================================
        # 7. Generate Rescue Image
        # ==================================================
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

            print(
                "FLUX CUDA Out Of Memory"
            )
            skipped += 1
            del original_image
            cleanup()
            continue
        # ==================================================
        # 8. Save Image
        # ==================================================
        output_path = save_generated_image(
            image=generated_image,
            output_dir=OUTPUT_DIR,
            image_name=f"{safe_name}.png",
        )
        print(
            f"Saved -> {output_path}"
        )
        # ==================================================
        # 9. Save Metadata
        # ==================================================

        metadata = {
            "image_key": img_key,
            "labels": positive_incidents,
            "url": url,
            "scene_description": scene_description,
            "editing_instruction": rescue_instruction,
            "flux_prompt": flux_prompt,
            "output_path": output_path,
        }
        evaluation_records.append(metadata)
        metadata_path = os.path.join(
            metadata_dir,
            f"{safe_name}.json",
        )
        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
            )

        count += 1

        print(
            f"Progress: {count}/{LIMIT_IMAGES}"
        )
        # ==================================================
        # 10. Cleanup
        # ==================================================
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
