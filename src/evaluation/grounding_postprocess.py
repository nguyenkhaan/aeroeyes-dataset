from __future__ import annotations

import inspect
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingPostprocessRequest:
    outputs: object
    input_ids: object
    target_sizes: object
    box_threshold: float
    text_threshold: float


def grounding_postprocess_parameters(processor: object) -> tuple[str, bool]:
    method = processor.post_process_grounded_object_detection
    parameters = inspect.signature(method).parameters
    if "box_threshold" in parameters:
        box_parameter = "box_threshold"
    elif "threshold" in parameters:
        box_parameter = "threshold"
    else:
        raise ValueError("Unsupported Grounding DINO postprocess threshold API")
    return box_parameter, "text_threshold" in parameters


def postprocess_grounding(
    processor: object, request: GroundingPostprocessRequest
) -> tuple[object, str]:
    box_parameter, supports_text = grounding_postprocess_parameters(processor)
    kwargs = {
        box_parameter: request.box_threshold,
        "target_sizes": request.target_sizes,
    }
    if supports_text:
        kwargs["text_threshold"] = request.text_threshold
    processed = processor.post_process_grounded_object_detection(
        request.outputs, request.input_ids, **kwargs
    )
    mode = (
        "box_and_text_threshold"
        if supports_text
        else "box_threshold_only_compatibility_fallback"
    )
    return processed, mode
