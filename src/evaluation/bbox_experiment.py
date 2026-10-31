from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_id: str
    prompt: str
    box_threshold: float
    text_threshold: float
    rescue_boat_threshold: float
    nms_iou_threshold: float
    seed: int

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentConfig:
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Experiment config must contain exactly the documented fields")
        name, model_id, prompt = (data[key] for key in ("name", "model_id", "prompt"))
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("Experiment name must use only letters, digits, hyphen or underscore")
        if any(not isinstance(value, str) or not value.strip() for value in (model_id, prompt)):
            raise ValueError("Model ID and prompt must be non-empty strings")
        seed = data["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Experiment seed must be an integer in [0, 2**32)")
        thresholds = {}
        for key in (
            "box_threshold", "text_threshold", "rescue_boat_threshold", "nms_iou_threshold"
        ):
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number in [0, 1]")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be a number in [0, 1]")
            thresholds[key] = float(value)
        return cls(name=name, model_id=model_id, prompt=prompt, seed=seed, **thresholds)

    def as_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_image(pair: dict, role: str) -> str:
    path = pair.get(f"{role}_path")
    expected = pair.get(f"{role}_sha256")
    if not isinstance(path, str) or not path or not isinstance(expected, str):
        raise ValueError(f"Missing {role} image path or checksum for {pair.get('pair_id')}")
    try:
        actual = sha256_file(Path(path))
    except OSError as exc:
        raise ValueError(f"Cannot read {role} image for {pair.get('pair_id')}: {path}") from exc
    if actual != expected.lower():
        raise ValueError(f"Image checksum mismatch for {pair.get('pair_id')} {role}")
    return path


def load_validation_pairs(path: Path, limit: int | None) -> list[dict]:
    pairs = read_json_object(path).get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Benchmark manifest must have a non-empty pairs list")
    ids = [pair.get("pair_id") if isinstance(pair, dict) else None for pair in pairs]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Every benchmark pair requires a string pair_id")
    if len(set(ids)) != len(ids):
        raise ValueError("Benchmark manifest contains duplicate pair_id values")
    validation = sorted(
        (pair for pair in pairs if pair.get("split") == "validation"),
        key=lambda pair: pair["pair_id"],
    )
    if not validation:
        raise ValueError("Benchmark manifest contains no validation pairs")
    if limit is not None and (limit <= 0 or limit > len(validation)):
        raise ValueError(f"Invalid requested pair limit: {limit}; available: {len(validation)}")
    if limit is None or limit == len(validation):
        chosen = validation
    elif limit == 1:
        chosen = validation[:1]
    else:
        chosen = [
            validation[index * (len(validation) - 1) // (limit - 1)]
            for index in range(limit)
        ]
    return [
        {
            "stem": pair["pair_id"],
            "reference_path": _validate_image(pair, "before"),
            "output_path": _validate_image(pair, "after"),
            "before_sha256": pair["before_sha256"],
            "after_sha256": pair["after_sha256"],
        }
        for pair in chosen
    ]


def select_ground_truth(document: dict, records: list[dict], manifest_hash: str) -> dict:
    if document.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("Ground truth and benchmark manifest hashes do not match")
    pairs = document.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("Ground truth must contain a pairs list")
    selected_ids = {record["stem"] for record in records}
    selected = [
        pair for pair in pairs
        if isinstance(pair, dict) and pair.get("pair_id") in selected_ids
        and pair.get("split") == "validation"
    ]
    actual_ids = [pair["pair_id"] for pair in selected]
    if len(actual_ids) != len(selected_ids) or set(actual_ids) != selected_ids:
        raise ValueError("Selected validation ground truth has missing or duplicate pairs")
    valid_statuses = {
        "valid", "ambiguous", "missing_requested_object", "invalid_object",
        "severe_scene_corruption",
    }
    records_by_id = {record["stem"]: record for record in records}
    for pair in selected:
        if pair.get("generation_status") not in valid_statuses or not all(
            isinstance(pair.get(field), list) for field in ("before_objects", "after_objects")
        ):
            raise ValueError("Selected ground truth contains incomplete pair annotations")
        record = records_by_id[pair["pair_id"]]
        for role in ("before", "after"):
            if pair.get(f"{role}_sha256") != record[f"{role}_sha256"]:
                raise ValueError(f"Ground truth image checksum differs for {pair['pair_id']}")
    return {**document, "pairs": selected, "experiment_selection": "validation_only"}


def compare_experiment_runs(baseline: Path, candidate: Path, varying: str) -> dict:
    allowed = {
        "box_threshold", "text_threshold", "rescue_boat_threshold",
        "nms_iou_threshold", "prompt",
    }
    if varying not in allowed:
        raise ValueError(f"Unsupported varying field: {varying}")
    runs = []
    for path in (baseline, candidate):
        manifest = read_json_object(path / "run-manifest.json")
        metrics = read_json_object(path / "metrics.json")
        if manifest.get("status") != "completed" or manifest.get("split") != "validation":
            raise ValueError(f"Run is not a completed validation experiment: {path}")
        ids = manifest.get("selected_pair_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError(f"Run has invalid selected pair IDs: {path}")
        if metrics.get("run_manifest") != manifest:
            raise ValueError(f"Metrics are not bound to this run manifest: {path}")
        if sha256_file(path / "ground-truth.selected.json") != manifest.get(
            "selected_ground_truth_sha256"
        ):
            raise ValueError(f"Selected ground truth checksum differs: {path}")
        if manifest.get("result", {}).get("processed_pairs") != len(ids):
            raise ValueError(f"Run did not process all selected pairs: {path}")
        if metrics.get("evaluation_protocol", {}).get("version") != "bbox-metrics-v2-ambiguity-aware":
            raise ValueError(f"Run does not use ambiguity-aware evaluator v2: {path}")
        if metrics.get("generation", {}).get("total_pairs") != len(ids):
            raise ValueError(f"Metrics pair count differs from the run: {path}")
        if metrics.get("pair_pass_rate", {}).get("missing_reports") != 0:
            raise ValueError(f"Metrics include missing prediction reports: {path}")
        runs.append((manifest, metrics))
    first, second = (item[0] for item in runs)
    for field in ("benchmark_manifest_sha256", "source_ground_truth_sha256", "selected_ground_truth_sha256"):
        if not first.get(field) or first.get(field) != second.get(field):
            raise ValueError(f"Runs use different or missing {field}")
    if first["selected_pair_ids"] != second["selected_pair_ids"]:
        raise ValueError("Runs use different selected pair IDs")
    if runs[0][1].get("annotation_coverage") != runs[1][1].get("annotation_coverage"):
        raise ValueError("Runs have different annotation coverage")
    first_config = ExperimentConfig.from_dict(first.get("config"))
    second_config = ExperimentConfig.from_dict(second.get("config"))
    for key in first_config.__dataclass_fields__:
        if key not in {"name", varying} and getattr(first_config, key) != getattr(second_config, key):
            raise ValueError(f"Configurations differ in non-varying field: {key}")
    if first.get("git", {}).get("commit") != second.get("git", {}).get("commit"):
        raise ValueError("Runs use different git commits")
    first_result, second_result = first.get("result", {}), second.get("result", {})
    for result in (first_result, second_result):
        if result.get("postprocess_mode") != "box_and_text_threshold":
            raise ValueError("Run did not apply both box and text thresholds")
    revision_a, revision_b = first_result.get("model_revision"), second_result.get("model_revision")
    if revision_a and revision_b and revision_a != revision_b:
        raise ValueError("Runs use different model revisions")
    warnings = [
        "Annotation adjudication is not verified; treat results as exploratory until reviewed"
    ]
    if first.get("ground_truth_status") not in {"validation_locked", "all_locked"}:
        warnings.append("Ground truth is not locked; results are exploratory")
    if not revision_a or not revision_b:
        warnings.append("Model revision unavailable; identical model weights are not verified")
    if not first.get("git", {}).get("commit"):
        warnings.append("Git commit unavailable; identical code is not verified")
    dirty_states = [run.get("git", {}).get("tracked_files_dirty") for run in (first, second)]
    if any(dirty is not False for dirty in dirty_states):
        warnings.append("Tracked working tree was dirty or unknown during at least one run")
    if first_result.get("gpu") != second_result.get("gpu"):
        warnings.append("GPU hardware differs between runs")
    for field in ("ontology_sha256", "class_map_sha256"):
        left, right = first_result.get(field), second_result.get(field)
        if left != right:
            raise ValueError(f"Runs use different {field}")
        if not left:
            warnings.append(f"{field} unavailable; identical label mapping is not verified")
    tiers = {}
    for name in ("detector", "final_pipeline"):
        first_overall = runs[0][1][name]["overall"]
        second_overall = runs[1][1][name]["overall"]
        fields = ("precision", "recall", "f1", "ap50", "map50_95", "tp", "fp", "fn", "ignored_prediction_count")
        tiers[name] = {}
        for field in fields:
            a, b = first_overall.get(field), second_overall.get(field)
            delta = b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            tiers[name][field] = {"baseline": a, "candidate": b, "delta": delta}
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "varying_field": varying,
        "baseline_value": getattr(first_config, varying),
        "candidate_value": getattr(second_config, varying),
        "selected_pairs": len(first["selected_pair_ids"]),
        "annotation_coverage": runs[0][1].get("annotation_coverage"),
        "warnings": warnings,
        "resources": {
            "baseline": {
                "elapsed_seconds": first.get("elapsed_seconds"),
                "peak_vram_bytes": first_result.get("peak_vram_bytes"),
                "gpu": first_result.get("gpu"),
                "model_revision": revision_a,
                "label_mapping_counts": first_result.get("label_mapping_counts"),
            },
            "candidate": {
                "elapsed_seconds": second.get("elapsed_seconds"),
                "peak_vram_bytes": second_result.get("peak_vram_bytes"),
                "gpu": second_result.get("gpu"),
                "model_revision": revision_b,
                "label_mapping_counts": second_result.get("label_mapping_counts"),
            },
        },
        "tiers": tiers,
    }
