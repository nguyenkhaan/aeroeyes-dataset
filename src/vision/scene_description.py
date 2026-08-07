import torch
from src.core.config import MAX_NEW_TOKENS


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


def generate_scene_description(
    image, 
    vision_model, 
    vision_processor,
    max_new_tokens: int = MAX_NEW_TOKENS,
): 
    """
        Analyze the image and generate the image description about the scene 
    
    """
    prompt = """
            You are a professional disaster scene analysis assistant.
        Your task is ONLY to describe what is directly visible in the image.
        Rules:
        - Describe only visible objects.
        - Do not infer hidden information.
        - Do not speculate.
        - Do not explain the cause of the disaster.
        - Do not suggest rescue actions.
        - Do not mention anything not visible.
        - Return a single factual paragraph.
    """
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text", 
                    "text": "You are a Vision-Language AI assistant specialized in disaster scene understanding and image editing instruction generation."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    # Tu cai document cua hugging face nhe may em :v 
    inputs = vision_processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt").to(vision_model.device)
    input_len = inputs["input_ids"].shape[-1]
    """
    # Gemma 3 
    with torch.inference_mode():
        generation = vision_model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
    
    decoded = vision_processor.decode(generation, skip_special_tokens=True).strip()
    return decoded 
    # Gemma 3 
    """
    ### Gemma 4 

    with torch.inference_mode(): 
        outputs = vision_model.generate(**inputs, max_new_tokens=max_new_tokens) 
        generation = outputs[0][input_len:]
    return _decode_model_output(vision_processor, generation)
