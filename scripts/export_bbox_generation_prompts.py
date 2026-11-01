#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def export_prompts(manifest_path: Path, mapping_path: Path, metadata_dir: Path,
                   output_path: Path, split: str) -> list[dict]:
    if split not in {"validation", "test"}:
        raise ValueError("Split must be validation or test")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if mapping.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("Mapping does not match the benchmark manifest")
    selected = [pair for pair in manifest["pairs"] if pair.get("split") == split]
    mapped = [pair for pair in mapping["pairs"] if pair.get("split") == split]
    ids = [pair.get("pair_id") for pair in selected]
    mapped_ids = [pair.get("pair_id") for pair in mapped]
    if (not ids or len(ids) != len(set(ids)) or len(mapped_ids) != len(set(mapped_ids))
            or set(ids) != set(mapped_ids)
            or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in ids)):
        raise ValueError("Missing, invalid or duplicated pair IDs")
    mapped_by_id = {pair["pair_id"]: pair for pair in mapped}
    records = []
    for pair_id in sorted(ids):
        frames = mapped_by_id[pair_id].get("frames")
        if not isinstance(frames, list) or len(frames) != 2:
            raise ValueError(f"Missing paired CVAT frames: {pair_id}")
        names = {item.get("role"): item.get("bundle_name") for item in frames}
        if (set(names) != {"before", "after"} or any(not isinstance(name, str) or not name
                or Path(name).name != name for name in names.values()) or names["before"] == names["after"]):
            raise ValueError(f"Invalid paired CVAT frames: {pair_id}")
        path = metadata_dir / f"{pair_id}.json"
        metadata_bytes = path.read_bytes()
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict) or metadata.get("stem") != pair_id:
            raise ValueError(f"Metadata stem does not match pair: {pair_id}")
        records.append({"pair_id": pair_id, "cvat_files": [names["before"], names["after"]],
                        "metadata_path": str(path), "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                        "scene_description": metadata.get("scene_description"),
                        "editing_instruction": metadata.get("editing_instruction"),
                        "flux_prompt": metadata.get("flux_prompt")})
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Export original generation prompts for a benchmark split.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    records = export_prompts(args.manifest, args.mapping, args.metadata_dir, args.output, args.split)
    missing = sum(not isinstance(item["editing_instruction"], str) or not item["editing_instruction"].strip()
                  for item in records)
    print(f"Split: {args.split}; pairs: {len(records)}; missing editing instructions: {missing}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
