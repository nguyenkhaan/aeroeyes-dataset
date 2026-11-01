import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.bbox_experiment import (
    ExperimentConfig,
    compare_experiment_runs,
    load_validation_pairs,
    select_ground_truth,
)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class BoundingBoxExperimentTests(unittest.TestCase):
    def test_cross_class_conflict_config_validation(self) -> None:
        base = json.loads(Path("config/bounding_box/dino-ambulance-min030.json").read_text(encoding="utf-8"))
        self.assertIsNone(ExperimentConfig.from_dict(base).cross_class_iou_threshold)
        self.assertEqual(ExperimentConfig.from_dict({**base, "cross_class_iou_threshold": .75}).cross_class_iou_threshold, .75)
        for value in (0, -0.1, 1.1, True, "0.75", float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({**base, "cross_class_iou_threshold": value})

    def test_conflict_config_only_changes_cross_class_threshold(self) -> None:
        root = Path("config/bounding_box")
        control = ExperimentConfig.from_dict(json.loads((root / "dino-ambulance-min030.json").read_text(encoding="utf-8"))).as_dict()
        candidate = ExperimentConfig.from_dict(json.loads((root / "dino-ambulance-conflict075.json").read_text(encoding="utf-8"))).as_dict()
        self.assertEqual({key for key in control if control[key] != candidate[key]}, {"name", "cross_class_iou_threshold"})
        self.assertEqual(candidate["cross_class_iou_threshold"], 0.75)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pairs = []
        for index, split in enumerate(("validation", "test", "validation")):
            before = self.root / f"before-{index}.png"
            after = self.root / f"after-{index}.png"
            before.write_bytes(f"before-{index}".encode())
            after.write_bytes(f"after-{index}".encode())
            self.pairs.append({
                "pair_id": f"pair-{index}",
                "split": split,
                "before_path": str(before),
                "after_path": str(after),
                "before_sha256": digest(before.read_bytes()),
                "after_sha256": digest(after.read_bytes()),
            })
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps({"pairs": self.pairs}), encoding="utf-8")

    def test_selects_only_validation_and_validates_hashes(self) -> None:
        records = load_validation_pairs(self.manifest_path, limit=None)

        self.assertEqual([item["stem"] for item in records], ["pair-0", "pair-2"])
        self.assertEqual(records[0]["reference_path"], self.pairs[0]["before_path"])

    def test_smoke_limit_is_deterministic(self) -> None:
        records = load_validation_pairs(self.manifest_path, limit=1)

        self.assertEqual([item["stem"] for item in records], ["pair-0"])

    def test_changed_image_fails_before_inference(self) -> None:
        Path(self.pairs[0]["before_path"]).write_bytes(b"modified")

        with self.assertRaisesRegex(ValueError, "checksum"):
            load_validation_pairs(self.manifest_path, limit=1)

    def test_duplicate_pair_ids_and_oversized_limit_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested"):
            load_validation_pairs(self.manifest_path, limit=3)
        self.pairs[1]["pair_id"] = self.pairs[0]["pair_id"]
        self.manifest_path.write_text(json.dumps({"pairs": self.pairs}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_validation_pairs(self.manifest_path, limit=None)

    def test_selected_ground_truth_matches_only_executed_pairs(self) -> None:
        records = load_validation_pairs(self.manifest_path, limit=1)
        source_hash = digest(self.manifest_path.read_bytes())
        gt = {"source_manifest_sha256": source_hash, "pairs": [
            {**item, "generation_status": "valid", "before_objects": [], "after_objects": []}
            for item in self.pairs
        ]}

        selected = select_ground_truth(gt, records, source_hash)

        self.assertEqual([item["pair_id"] for item in selected["pairs"]], ["pair-0"])

    def test_ground_truth_manifest_mismatch_and_missing_pairs_fail(self) -> None:
        records = load_validation_pairs(self.manifest_path, limit=1)
        with self.assertRaisesRegex(ValueError, "manifest"):
            select_ground_truth({"source_manifest_sha256": "bad", "pairs": []}, records, "expected")
        with self.assertRaisesRegex(ValueError, "missing"):
            select_ground_truth(
                {"source_manifest_sha256": "expected", "pairs": []}, records, "expected"
            )

    def test_rejects_invalid_configuration(self) -> None:
        good = {"name": "test", "model_id": "model", "prompt": "firefighter .",
                "box_threshold": 0.25, "text_threshold": 0.25,
                "rescue_boat_threshold": 0.33, "nms_iou_threshold": 0.5,
                "seed": 20261020}
        for field, value in (("box_threshold", -0.1), ("prompt", ""),
                             ("text_threshold", float("nan")), ("name", "../bad"),
                             ("seed", -1)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    ExperimentConfig.from_dict({**good, field: value})

    def test_supplemental_prompts_validation_and_old_defaults(self) -> None:
        path = Path("config/bounding_box/dino-prompt-control.json")
        base = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(ExperimentConfig.from_dict(base).supplemental_prompts, [])
        good = [{"class_name": "ambulance", "prompt": "ambulance ."}]
        self.assertEqual(ExperimentConfig.from_dict({**base, "supplemental_prompts": good}).supplemental_prompts, good)
        for invalid in ([{"class_name": "helicopter", "prompt": "ambulance ."}],
                        [{"class_name": "ambulance", "prompt": ""}], good + good, "ambulance"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({**base, "supplemental_prompts": invalid})

    def test_supplemental_confidence_validation_and_roundtrip(self) -> None:
        base = json.loads(Path("config/bounding_box/dino-prompt-control.json").read_text(encoding="utf-8"))
        prompt = {"class_name": "ambulance", "prompt": "ambulance .", "min_confidence": 0.4}
        config = ExperimentConfig.from_dict({**base, "supplemental_prompts": [prompt]})
        self.assertEqual(ExperimentConfig.from_dict(config.as_dict()).supplemental_prompts, [prompt])
        for value in (-0.1, 1.1, True, "0.4", float("nan"), None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({**base, "supplemental_prompts": [{**prompt, "min_confidence": value}]})

    def test_prompt_configs_change_only_prompt_and_run_name(self) -> None:
        baseline = ExperimentConfig.from_dict(json.loads(
            Path("config/bounding_box/dino-prompt-control.json").read_text(encoding="utf-8")
        )).as_dict()
        candidate = ExperimentConfig.from_dict(json.loads(
            Path("config/bounding_box/dino-prompt-aliases.json").read_text(encoding="utf-8")
        )).as_dict()
        differences = {key for key in baseline if baseline[key] != candidate[key]}

        self.assertEqual(differences, {"name", "prompt"})
        self.assertIn("fire truck", candidate["prompt"])

    def test_compound_experiment_changes_only_label_mapping(self) -> None:
        def config(name):
            path = Path("config/bounding_box") / name
            return ExperimentConfig.from_dict(json.loads(path.read_text(encoding="utf-8"))).as_dict()

        baseline = config("dino-prompt-aliases.json")
        candidate = config("dino-prompt-compound.json")
        self.assertEqual({key for key in baseline if baseline[key] != candidate[key]},
                         {"name", "label_mapping_mode"})
        self.assertEqual(baseline["label_mapping_mode"], "exact")
        self.assertEqual(candidate["label_mapping_mode"], "firefighter_compound")

    def test_vehicle_experiment_changes_only_supplemental_prompts(self) -> None:
        root = Path("config/bounding_box")
        control = ExperimentConfig.from_dict(json.loads((root / "dino-prompt-control.json").read_text(encoding="utf-8"))).as_dict()
        candidate = ExperimentConfig.from_dict(json.loads((root / "dino-supplemental-vehicles.json").read_text(encoding="utf-8"))).as_dict()
        self.assertEqual({key for key in control if control[key] != candidate[key]}, {"name", "supplemental_prompts"})
        self.assertEqual(len(candidate["supplemental_prompts"]), 3)

    def test_ambulance_threshold_configs_only_change_the_ambulance_threshold(self) -> None:
        root = Path("config/bounding_box")
        control = ExperimentConfig.from_dict(json.loads((root / "dino-supplemental-vehicles.json").read_text(encoding="utf-8"))).as_dict()
        for suffix, threshold in (("030", 0.30), ("045", 0.45), ("060", 0.60)):
            config = ExperimentConfig.from_dict(json.loads((root / f"dino-ambulance-min{suffix}.json").read_text(encoding="utf-8"))).as_dict()
            self.assertEqual({key for key in control if control[key] != config[key]}, {"name", "supplemental_prompts"})
            self.assertEqual(config["supplemental_prompts"][0], {**control["supplemental_prompts"][0], "min_confidence": threshold})
            self.assertEqual(config["supplemental_prompts"][1:], control["supplemental_prompts"][1:])

    def test_cli_help_does_not_load_gpu_dependencies(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_bbox_experiment.py", "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ground-truth", result.stdout)

    def test_prepare_only_selects_ground_truth_and_prevents_overwriting(self) -> None:
        source_hash = digest(self.manifest_path.read_bytes())
        ground_truth = self.root / "ground-truth.json"
        ground_truth.write_text(json.dumps({
            "source_manifest_sha256": source_hash,
            "pairs": [{**pair, "generation_status": "valid", "before_objects": [],
                       "after_objects": []} for pair in self.pairs],
        }), encoding="utf-8")
        output = self.root / "run"
        command = [sys.executable, "scripts/run_bbox_experiment.py",
                   "--manifest", str(self.manifest_path), "--ground-truth", str(ground_truth),
                   "--config", "config/bounding_box/dino-low-threshold.json",
                   "--output-dir", str(output), "--limit", "1", "--prepare-only"]

        first = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(first.returncode, 0, first.stderr)
        run = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
        gt = json.loads((output / "ground-truth.selected.json").read_text(encoding="utf-8"))
        self.assertEqual(run["selected_pair_ids"], ["pair-0"])
        self.assertEqual([pair["pair_id"] for pair in gt["pairs"]], ["pair-0"])
        self.assertEqual(run["config"]["box_threshold"], 0.18)
        second = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(
            json.loads((output / "run-manifest.json").read_text(encoding="utf-8")), run
        )

    def test_execution_records_success_and_failure_without_a_gpu(self) -> None:
        from scripts import run_bbox_experiment

        source_hash = digest(self.manifest_path.read_bytes())
        ground_truth = self.root / "ground-truth.json"
        ground_truth.write_text(json.dumps({
            "source_manifest_sha256": source_hash,
            "pairs": [{**pair, "generation_status": "valid", "before_objects": [],
                       "after_objects": []} for pair in self.pairs],
        }), encoding="utf-8")

        def arguments(output: Path) -> list[str]:
            return ["run_bbox_experiment.py", "--manifest", str(self.manifest_path),
                    "--ground-truth", str(ground_truth), "--config",
                    "config/bounding_box/dino-low-threshold.json", "--output-dir",
                    str(output), "--limit", "1"]

        output = self.root / "success"
        with patch.object(sys, "argv", arguments(output)), patch.object(
            run_bbox_experiment, "run_inference", return_value={"processed_pairs": 1}
        ) as inference:
            self.assertEqual(run_bbox_experiment.main(), 0)
        self.assertEqual(inference.call_args.args[1].box_threshold, 0.18)
        manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["result"]["processed_pairs"], 1)

        failed_output = self.root / "failed"
        with patch.object(sys, "argv", arguments(failed_output)), patch.object(
            run_bbox_experiment, "run_inference", side_effect=RuntimeError("CUDA test failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA test failure"):
                run_bbox_experiment.main()
        failed = json.loads((failed_output / "run-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(failed["status"], "failed")
        self.assertIn("CUDA test failure", failed["error"])

    def write_completed_run(self, name: str, threshold: float) -> Path:
        path = self.root / name
        path.mkdir()
        config_path = Path("config/bounding_box/dino-baseline.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["box_threshold"] = threshold
        selected_gt = path / "ground-truth.selected.json"
        selected_gt.write_text('{"pairs": []}', encoding="utf-8")
        manifest = {
            "status": "completed", "split": "validation", "config": config,
            "ground_truth_status": "validation_locked",
            "selected_pair_ids": ["pair-0"],
            "benchmark_manifest_sha256": "benchmark",
            "source_ground_truth_sha256": "ground-truth",
            "selected_ground_truth_sha256": digest(selected_gt.read_bytes()),
            "git": {"commit": "abc", "tracked_files_dirty": False},
            "result": {"postprocess_mode": "box_and_text_threshold", "processed_pairs": 1,
                       "model_revision": "model-revision", "gpu": "GPU"},
        }
        overall = {"precision": 0.5, "recall": threshold, "f1": 0.4,
                   "ap50": 0.3, "map50_95": 0.2, "tp": 1, "fp": 1,
                   "fn": 1, "ignored_prediction_count": 0}
        metrics = {
            "evaluation_protocol": {"version": "bbox-metrics-v2-ambiguity-aware"},
            "generation": {"total_pairs": 1}, "pair_pass_rate": {"missing_reports": 0},
            "detector": {"overall": overall}, "final_pipeline": {"overall": overall},
            "run_manifest": manifest,
        }
        (path / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return path

    def test_compare_reports_metric_deltas_for_compatible_runs(self) -> None:
        baseline = self.write_completed_run("baseline", 0.25)
        candidate = self.write_completed_run("candidate", 0.18)

        result = compare_experiment_runs(baseline, candidate, "box_threshold")

        self.assertEqual(result["selected_pairs"], 1)
        self.assertIn("adjudication is not verified", result["warnings"][0])
        self.assertAlmostEqual(result["tiers"]["detector"]["recall"]["delta"], -0.07)

    def test_compare_rejects_different_selection_and_other_changed_parameters(self) -> None:
        baseline = self.write_completed_run("baseline", 0.25)
        candidate = self.write_completed_run("candidate", 0.18)
        path = candidate / "run-manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["selected_pair_ids"] = ["different-pair"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        metrics_path = candidate / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["run_manifest"] = manifest
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different selected pair"):
            compare_experiment_runs(baseline, candidate, "box_threshold")
        manifest["selected_pair_ids"] = ["pair-0"]
        manifest["config"]["prompt"] = "another prompt ."
        path.write_text(json.dumps(manifest), encoding="utf-8")
        metrics["run_manifest"] = manifest
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-varying field: prompt"):
            compare_experiment_runs(baseline, candidate, "box_threshold")

    def test_compare_warns_when_model_revision_is_missing(self) -> None:
        baseline = self.write_completed_run("baseline", 0.25)
        candidate = self.write_completed_run("candidate", 0.18)
        path = candidate / "run-manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["result"]["model_revision"] = None
        path.write_text(json.dumps(manifest), encoding="utf-8")
        metrics_path = candidate / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["run_manifest"] = manifest
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

        result = compare_experiment_runs(baseline, candidate, "box_threshold")

        self.assertTrue(any("Model revision unavailable" in item for item in result["warnings"]))

    def test_compare_rejects_metrics_from_another_run(self) -> None:
        baseline = self.write_completed_run("baseline", 0.25)
        candidate = self.write_completed_run("candidate", 0.18)
        (candidate / "metrics.json").write_bytes((baseline / "metrics.json").read_bytes())

        with self.assertRaisesRegex(ValueError, "not bound"):
            compare_experiment_runs(baseline, candidate, "box_threshold")

    def test_compare_rejects_different_ontology_hashes(self) -> None:
        baseline = self.write_completed_run("baseline", 0.25)
        candidate = self.write_completed_run("candidate", 0.18)
        for path, hash_value in ((baseline, "hash-a"), (candidate, "hash-b")):
            manifest_path = path / "run-manifest.json"
            metrics_path = path / "metrics.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            manifest["result"]["ontology_sha256"] = hash_value
            metrics["run_manifest"] = manifest
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different ontology_sha256"):
            compare_experiment_runs(baseline, candidate, "box_threshold")


if __name__ == "__main__":
    unittest.main()
