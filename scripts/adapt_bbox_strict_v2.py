#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.bbox_strict_v2 import adapt_strict_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapt strict validation v2; this does not evaluate predictions.")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.bundle) as archive:
        matches = [x for x in archive.infolist() if x.filename == "GT_validation_v2_strict_manifest.json"]
        if len(matches) != 1 or matches[0].file_size > 16 * 1024 * 1024:
            raise ValueError("Bundle must contain exactly one suitably sized strict manifest")
        strict_bytes = archive.read(matches[0])
    benchmark_bytes = args.benchmark_manifest.read_bytes()
    result = adapt_strict_manifest(json.loads(strict_bytes), json.loads(benchmark_bytes))
    result["source_strict_manifest_sha256"] = hashlib.sha256(strict_bytes).hexdigest()
    result["source_bundle_sha256"] = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    result["source_manifest_sha256"] = hashlib.sha256(benchmark_bytes).hexdigest()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    pairs = result["pairs"]
    print(f"Validation pairs: {len(pairs)}")
    print(f"Confirmed after: {sum(len(p['after_objects']) for p in pairs)}")
    print(f"Confirmed new: {sum(o['is_newly_added'] == 'true' for p in pairs for o in p['after_objects'])}")
    print(f"Ignore regions: {sum(len(p['ignore_regions']) for p in pairs)}")
    print(f"Output: {args.output}")
    print("Adapter only: this output requires a strict-v2 evaluator and must not be scored by the existing evaluator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
