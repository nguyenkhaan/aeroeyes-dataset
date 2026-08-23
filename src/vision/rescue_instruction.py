
import torch
from typing import Any, cast
from src.core.config import MAX_NEW_TOKENS
from src.vision.prompting import build_vision_inputs


def _decode_model_output(vision_processor, generation) -> str:
    """
    Decode model output with a fallback when the processor does not
    expose `parse_response`.
    """
    response = vision_processor.decode(
        generation,
        skip_special_tokens=False,
    )
    parse_response = getattr(vision_processor, "parse_response", None)
    if callable(parse_response):
        try:
            parsed = parse_response(response)
            if isinstance(parsed, str):
                return parsed.strip()
            return str(parsed).strip()
        except Exception:
            pass
    return response.strip()


def generate_rescue_instruction(
    scene_description: str, 
    vision_model, 
    vision_processor,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Generate rescue image editing instructions from
    the disaster scene description.
    """

    system_prompt = """
        You are an expert emergency rescue planner.
        
        Your task is to generate editing instructions for an image editing model.
        Requirements:
        1. Preserve the original disaster scene.
        2. Preserve damaged buildings and existing objects.
        3. Do not change the disaster type.
        4. Add only realistic rescue operations.
        5. Add rescue personnel when appropriate.
        6. Add rescue vehicles when appropriate.
        7. Add emergency equipment when appropriate.
        8. Maintain realistic object scale.
        9. Maintain realistic lighting.
        10. Maintain realistic perspective.
        11. Keep all newly added objects consistent with the existing environment.
        
        Return ONLY the editing instructions.
        Do not explain your reasoning.
        Do not describe the original image.
        Do not use markdown.
    """
    user_prompt = f"""
        Disaster Scene:
        {scene_description}
        Generate image editing instructions.
    """

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                }
            ],
        },
    ]

    inputs = build_vision_inputs(
        vision_processor,
        messages=messages,
        text=f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
    ).to(vision_model.device)
    inputs = cast(dict[str, Any], inputs)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode(): 
        outputs = vision_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generation = outputs[0][input_len:]
    return _decode_model_output(vision_processor, generation)
