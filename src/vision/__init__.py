def build_flux_prompt(
    scene_description: str,
    rescue_instruction: str,
) -> str:
    """
    Build the final prompt for FLUX image editing.
    """
    prompt = f"""
You are editing an existing disaster photograph.

Original Scene
--------------
{scene_description}
Editing Instructions
--------------------
{rescue_instruction}
Requirements
    - Preserve the original disaster scene.
    - Preserve all existing buildings, vehicles, roads and environmental objects.
    - Do not change the disaster type.
    - Add only realistic rescue operations.
    - Blend newly added rescue personnel, vehicles and equipment naturally.
    - Maintain realistic lighting, shadows and perspective.
    - Maintain correct object proportions.
    - Generate anatomically correct humans.
    - Produce seamless image editing without visible artifacts.
    Style
    - Documentary disaster photography
    - Photojournalism
    - Real-world emergency response
    - Natural color grading
    - Authentic textures
    - High realism
    - Non-cinematic
"""
    return prompt.strip()