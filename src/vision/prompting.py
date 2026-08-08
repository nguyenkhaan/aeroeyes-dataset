from __future__ import annotations

from typing import Any, Protocol, cast


class VisionInputs(Protocol):
    def to(self, *args: Any, **kwargs: Any) -> "VisionInputs":
        ...


def _ensure_image_token(text: str, image_count: int = 1) -> str:
    """
    Make sure the prompt contains at least one image placeholder.

    Some vision processors expect a literal `<|image|>` token in the text
    prompt whenever images are passed separately.
    """
    if image_count <= 0:
        return text

    if "<|image|>" in text or "<image>" in text:
        return text

    return ("<|image|>\n" * image_count) + text.strip()


def build_vision_inputs(
    vision_processor,
    *,
    messages=None,
    text: str | None = None,
    image=None,
) -> VisionInputs:
    """
    Build model inputs with a chat-template path when available and a
    plain processor fallback otherwise.
    """
    apply_chat_template = getattr(vision_processor, "apply_chat_template", None)

    if messages is not None and callable(apply_chat_template):
        return cast(
            VisionInputs,
            apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ),
        )

    processor_kwargs = {
        "return_tensors": "pt",
    }
    if text is not None:
        if image is not None:
            text = _ensure_image_token(text)
        processor_kwargs["text"] = text
    if image is not None:
        processor_kwargs["images"] = image

    return cast(VisionInputs, vision_processor(**processor_kwargs))
