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
        Bạn là một trợ lý phân tích cảnh thiên tai chuyên nghiệp.
        Nhiệm vụ của bạn CHỈ là mô tả những gì trực tiếp xuất hiện trong ảnh.

        Quy tắc:
        - Chỉ mô tả các vật thể thực sự nhìn thấy trong hình.
        - Không suy đoán thông tin ẩn bên trong.
        - Không suy luận quá mức.
        - Không giải thích nguyên nhân của thiên tai.
        - Không đề xuất hành động cứu hộ.
        - Không nhắc đến bất cứ thứ gì không xuất hiện trong ảnh.
        - Trả về một đoạn văn mô tả sự thật, ngắn gọn và chính xác.
    """
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Bạn là trợ lý Vision-Language chuyên về hiểu cảnh thiên tai và sinh hướng dẫn chỉnh sửa ảnh."
                }
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
    inputs = build_vision_inputs(
        vision_processor,
        messages=messages,
        text=prompt.strip(),
        image=image,
    ).to(vision_model.device)
    inputs = cast(dict[str, Any], inputs)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode(): 
        outputs = vision_model.generate(**inputs, max_new_tokens=max_new_tokens) 
        generation = outputs[0][input_len:]
    return _decode_model_output(vision_processor, generation)
