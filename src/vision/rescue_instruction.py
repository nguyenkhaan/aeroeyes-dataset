
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
        Bạn là chuyên gia lập kế hoạch cứu hộ khẩn cấp.
        
        Nhiệm vụ của bạn là tạo hướng dẫn chỉnh sửa cho một model chỉnh sửa ảnh.
        Yêu cầu:
        1. Giữ nguyên cảnh thiên tai ban đầu.
        2. Giữ nguyên các tòa nhà bị hư hại và những đối tượng hiện có.
        3. Không thay đổi loại thiên tai.
        4. Chỉ thêm các hoạt động cứu hộ thực tế.
        5. Thêm nhân viên cứu hộ khi phù hợp.
        6. Thêm phương tiện cứu hộ khi phù hợp.
        7. Thêm thiết bị khẩn cấp khi phù hợp.
        8. Duy trì tỷ lệ kích thước thực tế của các đối tượng.
        9. Duy trì ánh sáng thực tế.
        10. Duy trì phối cảnh thực tế.
        11. Giữ cho tất cả đối tượng mới thêm phù hợp với môi trường hiện có.
        
        CHỈ trả về hướng dẫn chỉnh sửa.
        Không giải thích quá trình suy luận.
        Không mô tả ảnh ban đầu.
        Không sử dụng markdown.
    """
    user_prompt = f"""
        Cảnh thiên tai:
        {scene_description}
        Hãy tạo hướng dẫn chỉnh sửa ảnh.
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
