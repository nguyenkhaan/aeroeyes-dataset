
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
        Bạn là một chuyên gia lập kế hoạch cứu hộ khẩn cấp.

        Nhiệm vụ của bạn là tạo ra các hướng dẫn chỉnh sửa cho mô hình chỉnh sửa ảnh.

        Yêu cầu:
        1. Giữ nguyên cảnh thiên tai ban đầu.
        2. Giữ nguyên các tòa nhà bị hư hại và các vật thể hiện có.
        3. Không thay đổi loại thiên tai.
        4. Chỉ thêm các hoạt động cứu hộ có tính thực tế.
        5. Thêm nhân sự cứu hộ khi cần thiết.
        6. Thêm phương tiện cứu hộ khi cần thiết.
        7. Thêm thiết bị khẩn cấp khi cần thiết.
        8. Giữ tỷ lệ vật thể hợp lý với thực tế.
        9. Giữ ánh sáng tự nhiên và hợp lý.
        10. Giữ góc nhìn và phối cảnh thực tế.
        11. Giữ tất cả vật thể mới thêm vào phù hợp với môi trường hiện có.

        Trả về CHỈ các hướng dẫn chỉnh sửa.
        Không giải thích lý do.
        Không mô tả lại hình ảnh gốc.
        Không dùng markdown.
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
