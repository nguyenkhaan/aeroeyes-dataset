
import torch
def generate_rescue_instruction(
    scene_description: str, 
    vision_model, 
    vision_processor
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

    # Gemma 3
    inputs = vision_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(vision_model.device)
    
    input_len = inputs["input_ids"].shape[-1]
    """
    # Gemma 3 
    with torch.inference_mode():
        generation = vision_model.generate(**inputs, max_new_tokens=320, do_sample=False)
    generation = generation[0][input_len:]
    instruction = vision_processor.decode(
        generation,
        skip_special_tokens=True,
    ).strip()
    return instruction

    # Gemma 3 
    """
    ### Gemma 4 

    with torch.inference_mode(): 
        outputs = vision_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generation = outputs[0][input_len:]
    instruction = vision_processor.decode(generation, skip_special_tokens=False).strip()
    return vision_processor.parse_response(instruction)
