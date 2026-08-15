import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.sdqm_metadata import attach_sdqm_metadata, write_metadata_jsonl


class SdqmMetadataTests(unittest.TestCase):
    def test_attaches_dataset_scoped_metrics_to_each_record(self) -> None:
        records = [{"image_key": "example"}]

        enriched_records = attach_sdqm_metadata(records, {"similarity": 0.75})

        self.assertNotIn("sdqm", records[0])
        self.assertEqual(enriched_records[0]["sdqm"]["scope"], "dataset")
        self.assertEqual(enriched_records[0]["sdqm"]["metrics"]["similarity"], 0.75)

    def test_writes_jsonl_with_enriched_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = write_metadata_jsonl(
                [{"image_key": "example", "sdqm": {"scope": "dataset"}}],
                directory,
            )

            self.assertIsNotNone(output_path)
            row = json.loads(Path(output_path).read_text(encoding="utf-8"))

        self.assertEqual(row["sdqm"]["scope"], "dataset")


if __name__ == "__main__":
    unittest.main()
