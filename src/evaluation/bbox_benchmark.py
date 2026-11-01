from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


EXPLICIT_SOURCE_GROUP_FIELDS = (
    "source_group",
    "source_event_id",
    "event_id",
    "incident_id",
)


@dataclass(frozen=True)
class BenchmarkSelectionConfig:
    validation_size: int
    test_size: int
    seed: int

    @property
    def total_size(self) -> int:
        return self.validation_size + self.test_size

    def validate(self) -> None:
        if self.validation_size <= 0 or self.test_size <= 0:
            raise ValueError("Validation and test sizes must both be positive")


@dataclass(frozen=True)
class GenerationPair:
    pair_id: str
    image_key: str
    labels: tuple[str, ...]
    url: str | None
    before_path: Path
    after_path: Path
    source_group: str
    source_group_method: str

    @property
    def stratum(self) -> str:
        return "+".join(self.labels) if self.labels else "unlabeled"


@dataclass(frozen=True)
class PairGroup:
    group_id: str
    records: tuple[GenerationPair, ...]

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def stratum(self) -> str:
        labels = sorted({record.stratum for record in self.records})
        return "+".join(labels)


def load_selection_config(protocol_path: Path) -> BenchmarkSelectionConfig:
    try:
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        selection = protocol["selection"]
        config = BenchmarkSelectionConfig(
            validation_size=int(selection["validation_pairs"]),
            test_size=int(selection["test_pairs"]),
            seed=int(selection["seed"]),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid benchmark protocol: {protocol_path}") from exc
    config.validate()
    return config


def _active_labels(record: dict) -> tuple[str, ...]:
    labels = record.get("labels")
    if isinstance(labels, list):
        cleaned = sorted({str(label).strip() for label in labels if str(label).strip()})
        if cleaned:
            return tuple(cleaned)

    incidents = record.get("incidents")
    if isinstance(incidents, dict):
        active = sorted(
            str(label).strip()
            for label, value in incidents.items()
            if value == 1 and str(label).strip()
        )
        if active:
            return tuple(active)
    return ("unlabeled",)


def _source_group(record: dict, pair_id: str) -> tuple[str, str]:
    for field_name in EXPLICIT_SOURCE_GROUP_FIELDS:
        value = record.get(field_name)
        if value is not None and str(value).strip():
            return f"{field_name}:{str(value).strip()}", "explicit"
    return f"pair:{pair_id}", "pair_id_fallback"


def _require_string(record: dict, field_name: str, metadata_path: Path) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Generation metadata missing required fields at {metadata_path}: "
            f"{field_name}"
        )
    return value


def _load_metadata_object(metadata_path: Path) -> dict:
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid generation metadata: {metadata_path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Invalid generation metadata: {metadata_path}")
    return value


def load_generation_pairs(metadata_dir: Path) -> list[GenerationPair]:
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory is missing: {metadata_dir}")

    pairs: list[GenerationPair] = []
    seen_pair_ids: set[str] = set()
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        record = _load_metadata_object(metadata_path)
        pair_id = _require_string(record, "stem", metadata_path)
        image_key = _require_string(record, "image_key", metadata_path)
        before_path = Path(_require_string(record, "reference_path", metadata_path))
        after_path = Path(_require_string(record, "output_path", metadata_path))

        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate pair_id in generation metadata: {pair_id}")
        if not before_path.is_file() or not after_path.is_file():
            raise ValueError(
                f"Generation metadata references missing Before/After image: {metadata_path}"
            )

        source_group, source_group_method = _source_group(record, pair_id)
        pairs.append(
            GenerationPair(
                pair_id=pair_id,
                image_key=image_key,
                labels=_active_labels(record),
                url=str(record["url"]) if record.get("url") is not None else None,
                before_path=before_path,
                after_path=after_path,
                source_group=source_group,
                source_group_method=source_group_method,
            )
        )
        seen_pair_ids.add(pair_id)
    return pairs


def _group_pairs(records: Iterable[GenerationPair]) -> list[PairGroup]:
    records_by_group: dict[str, list[GenerationPair]] = defaultdict(list)
    for record in records:
        records_by_group[record.source_group].append(record)
    return [
        PairGroup(
            group_id=group_id,
            records=tuple(sorted(items, key=lambda item: item.pair_id)),
        )
        for group_id, items in sorted(records_by_group.items())
    ]


def _stratified_group_order(groups: list[PairGroup], seed: int) -> list[PairGroup]:
    groups_by_stratum: dict[str, list[PairGroup]] = defaultdict(list)
    for group in groups:
        groups_by_stratum[group.stratum].append(group)

    ranked: list[tuple[float, str, PairGroup]] = []
    for stratum, stratum_groups in sorted(groups_by_stratum.items()):
        random.Random(f"{seed}:{stratum}").shuffle(stratum_groups)
        denominator = len(stratum_groups)
        for rank, group in enumerate(stratum_groups, start=1):
            ranked.append((rank / denominator, stratum, group))
    return [group for _rank, _stratum, group in sorted(ranked, key=lambda item: item[:2])]


def _select_exact_pair_count(groups: list[PairGroup], target: int) -> list[PairGroup]:
    selections: dict[int, list[PairGroup]] = {0: []}
    for group in groups:
        additions: dict[int, list[PairGroup]] = {}
        for selected_count, selected_groups in selections.items():
            next_count = selected_count + group.size
            count_is_available = (
                next_count <= target
                and next_count not in selections
                and next_count not in additions
            )
            if count_is_available:
                additions[next_count] = [*selected_groups, group]
        selections.update(additions)
    if target not in selections:
        raise ValueError(
            f"Cannot select exactly {target} pairs without splitting a source group"
        )
    return selections[target]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_pair(record: GenerationPair, split: str) -> dict:
    return {
        "pair_id": record.pair_id,
        "split": split,
        "image_key": record.image_key,
        "labels": list(record.labels),
        "url": record.url,
        "source_group": record.source_group,
        "source_group_method": record.source_group_method,
        "before_path": str(record.before_path),
        "after_path": str(record.after_path),
        "before_sha256": _sha256(record.before_path),
        "after_sha256": _sha256(record.after_path),
        "sampling_reason": "representative_stratified_sample",
    }


def _label_counts(records: Iterable[GenerationPair]) -> dict[str, int]:
    counts = Counter(label for record in records for label in record.labels)
    return dict(sorted(counts.items()))


def build_benchmark_manifest(
    records: list[GenerationPair],
    config: BenchmarkSelectionConfig,
) -> dict:
    config.validate()
    if len(records) < config.total_size:
        raise ValueError(
            f"Benchmark requires {config.total_size} pairs, but only {len(records)} are available"
        )

    ordered_groups = _stratified_group_order(_group_pairs(records), config.seed)
    benchmark_groups = _select_exact_pair_count(ordered_groups, config.total_size)
    validation_order = _stratified_group_order(benchmark_groups, config.seed + 1)
    validation_groups = _select_exact_pair_count(validation_order, config.validation_size)
    validation_group_ids = {group.group_id for group in validation_groups}
    test_groups = [
        group for group in benchmark_groups
        if group.group_id not in validation_group_ids
    ]

    validation_records = [
        record for group in validation_groups for record in group.records
    ]
    test_records = [record for group in test_groups for record in group.records]
    manifest_pairs = [
        *(_manifest_pair(record, "validation") for record in validation_records),
        *(_manifest_pair(record, "test") for record in test_records),
    ]
    manifest_pairs.sort(key=lambda item: (item["split"], item["pair_id"]))

    return {
        "schema_version": 1,
        "selection": {
            "seed": config.seed,
            "strategy": "grouped_stratified",
            "validation_size": config.validation_size,
            "test_size": config.test_size,
            "source_group_fields": list(EXPLICIT_SOURCE_GROUP_FIELDS),
        },
        "counts": {
            "validation": len(validation_records),
            "test": len(test_records),
            "total": len(manifest_pairs),
        },
        "label_counts": {
            "validation": _label_counts(validation_records),
            "test": _label_counts(test_records),
        },
        "pairs": manifest_pairs,
    }
