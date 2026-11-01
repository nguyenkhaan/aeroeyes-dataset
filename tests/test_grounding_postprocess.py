import unittest

from src.evaluation.grounding_postprocess import (
    GroundingPostprocessRequest,
    grounding_postprocess_parameters,
    postprocess_grounding,
)


class OldProcessor:
    def post_process_grounded_object_detection(
        self, outputs, input_ids, *, box_threshold, text_threshold, target_sizes
    ):
        return (outputs, input_ids, box_threshold, text_threshold, target_sizes)


class NewProcessor:
    def post_process_grounded_object_detection(
        self, outputs, input_ids, *, threshold, text_threshold, target_sizes
    ):
        return (outputs, input_ids, threshold, text_threshold, target_sizes)


class BoxOnlyProcessor:
    def post_process_grounded_object_detection(
        self, outputs, input_ids, *, threshold, target_sizes
    ):
        return (outputs, input_ids, threshold, target_sizes)


class BrokenProcessor:
    def post_process_grounded_object_detection(
        self, outputs, input_ids, *, threshold, text_threshold, target_sizes
    ):
        raise TypeError("internal postprocessing error")


class GroundingPostprocessTests(unittest.TestCase):
    def test_both_transformers_signatures_apply_both_thresholds(self) -> None:
        request = GroundingPostprocessRequest("outputs", "ids", "sizes", 0.18, 0.25)
        for processor in (OldProcessor(), NewProcessor()):
            with self.subTest(processor=type(processor).__name__):
                result, mode = postprocess_grounding(processor, request)
                self.assertEqual(result, ("outputs", "ids", 0.18, 0.25, "sizes"))
                self.assertEqual(mode, "box_and_text_threshold")

    def test_missing_text_threshold_is_reported(self) -> None:
        request = GroundingPostprocessRequest("outputs", "ids", "sizes", 0.18, 0.25)
        result, mode = postprocess_grounding(BoxOnlyProcessor(), request)
        self.assertEqual(result, ("outputs", "ids", 0.18, "sizes"))
        self.assertEqual(mode, "box_threshold_only_compatibility_fallback")
        self.assertEqual(grounding_postprocess_parameters(BoxOnlyProcessor()), ("threshold", False))

    def test_internal_type_error_is_not_silently_swallowed(self) -> None:
        request = GroundingPostprocessRequest("outputs", "ids", "sizes", 0.18, 0.25)
        with self.assertRaisesRegex(TypeError, "internal postprocessing error"):
            postprocess_grounding(BrokenProcessor(), request)


if __name__ == "__main__":
    unittest.main()
