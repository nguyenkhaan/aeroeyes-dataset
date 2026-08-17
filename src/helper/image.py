import requests
from PIL import Image
import os 
import torch 
def download_image(
    url: str,
    headers=None,
    timeout: int = 30,
    retries: int = 3,
):
    """
    Download image from URL.
    Returns
    -------
    bytes | None
    """
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()

            return response.content

        except Exception as e:
            print(
                f"[Retry {attempt+1}/{retries}] {e}"
            )
    return None


def resize_center_crop(
    image: Image.Image,
    size: int = 512,
):
    """
    Resize while preserving aspect ratio,
    then center crop to (size x size).
    """
    width, height = image.size
    scale = max(
        size / width,
        size / height,
    )
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )
    left = (new_width - size) // 2
    top = (new_height - size) // 2
    image = image.crop(
        (
            left,
            top,
            left + size,
            top + size,
        )
    )
    return image

def save_generated_image(
    image: Image.Image,
    output_dir: str,
    image_name: str,
):
    """
    Save generated image.
    """
    os.makedirs(
        output_dir,
        exist_ok=True,
    )
    output_path = os.path.join(
        output_dir,
        image_name,
    )
    image.save(output_path)

    return output_path

def generate_rescue_image(
    pipe,
    image: Image.Image,
    prompt: str,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 30,
    seed: int = 42,
):
    """
    Generate a rescue simulation image using FLUX2-klein-4B.
    """
    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)
    with torch.inference_mode():

        result = pipe(
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
        )
    generated_image = result.images[0]
    del result
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated_image