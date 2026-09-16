"""Score saved images and aggregate reports without loading generation models."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from PIL import Image

from src.core.config import (
    OUTPUT_DIR, SDQM_ENABLED, SDQM_MIN_IMAGES, SDQM_OUTPUT_DIR,
    EVALUATION_TIMEOUT_SECONDS, MODEL_LOAD_TIMEOUT_SECONDS, STAGE_TIMEOUT_SECONDS,
)
from src.evaluation import (
    attach_sdqm_metadata, compute_dataset_cmmd, compute_dataset_sdqm,
    compute_o_score, compute_ssim, evaluate_quality, load_evaluators,
    passes_quality_gate, write_metadata_jsonl, write_sdqm_status_report,
)
from src.helper.memory import cleanup
from src.helper.runtime import stage


def evaluate_samples(metadata_paths: list[Path]) -> tuple[list[dict], int]:
    records = []
    failures = 0
    evaluators = None
    try:
        with stage("Load quality evaluators", MODEL_LOAD_TIMEOUT_SECONDS):
            evaluators = load_evaluators(device="cpu")
        for metadata_path in metadata_paths:
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
                with stage("Quality evaluation", STAGE_TIMEOUT_SECONDS):
                    with Image.open(record["reference_path"]) as reference, Image.open(record["output_path"]) as output:
                        original = reference.convert("RGB")
                        generated = output.convert("RGB")
                    sc_score, pq_score = evaluate_quality(evaluators, generated, record["flux_prompt"])
                    o_score = compute_o_score(sc_score, pq_score)
                    ssim = compute_ssim(original, generated)
                records.append({
                    **record,
                    "sc_score": round(sc_score, 4),
                    "pq_score": round(pq_score, 4),
                    "o_score": round(o_score, 4),
                    "ssim": round(ssim, 4),
                    "quality_passed": passes_quality_gate(o_score, ssim),
                })
            except Exception as exc:
                failures += 1
                print(f"Evaluation failed for {metadata_path.name}: {type(exc).__name__}: {exc}", flush=True)
            finally:
                original = generated = None
    finally:
        evaluators = None
        cleanup()
    return records, failures


def evaluate_dataset(records: list[dict]) -> dict:
    summary = {"evaluated_images": len(records), "cmmd": None, "sdqm": None, "errors": []}
    # Only committed, evaluated pairs participate, even after an interrupted save.
    with TemporaryDirectory(prefix="_evaluation_", dir=OUTPUT_DIR) as directory:
        real_dir = Path(directory) / "real"
        generated_dir = Path(directory) / "generated"
        real_dir.mkdir()
        generated_dir.mkdir()
        for index, record in enumerate(records):
            (real_dir / f"{index}.png").symlink_to(Path(record["reference_path"]).resolve())
            (generated_dir / f"{index}.png").symlink_to(Path(record["output_path"]).resolve())
        try:
            with stage("CMMD report", EVALUATION_TIMEOUT_SECONDS):
                summary["cmmd"] = compute_dataset_cmmd(ref_dir=str(real_dir), eval_dir=str(generated_dir))
        except Exception as exc:
            summary["errors"].append(f"CMMD failed: {type(exc).__name__}: {exc}")
        cleanup()
        if not SDQM_ENABLED or len(records) < SDQM_MIN_IMAGES:
            write_sdqm_status_report(SDQM_OUTPUT_DIR, {
                "status": "skipped",
                "reason": "SDQM disabled" if not SDQM_ENABLED else f"Need at least {SDQM_MIN_IMAGES} pairs",
                "real_image_count": len(records),
                "synthetic_image_count": len(records),
            })
        else:
            try:
                with stage("SDQM report", EVALUATION_TIMEOUT_SECONDS):
                    summary["sdqm"] = compute_dataset_sdqm(ref_dir=str(real_dir), eval_dir=str(generated_dir))
            except Exception as exc:
                reason = f"SDQM failed: {type(exc).__name__}: {exc}"
                summary["errors"].append(reason)
                write_sdqm_status_report(SDQM_OUTPUT_DIR, {"status": "failed", "reason": reason})
            finally:
                cleanup()
    return summary


def main() -> int:
    metadata_paths = sorted((Path(OUTPUT_DIR) / "_metadata").glob("*.json"))
    if not metadata_paths:
        print(f"No generation metadata found in {OUTPUT_DIR}/_metadata", flush=True)
        return 2
    records, failures = evaluate_samples(metadata_paths)
    if not records:
        print("No images could be evaluated.", flush=True)
        return 2
    report = pd.DataFrame(records)
    report.to_csv(Path(OUTPUT_DIR) / "evaluation_report.csv", index=False)
    print(report[["sc_score", "pq_score", "o_score", "ssim"]].describe())
    # Preserve scores before a dataset metric can fail or hit its watchdog.
    write_metadata_jsonl(records, OUTPUT_DIR)
    summary = evaluate_dataset(records)
    summary["quality_passed"] = sum(record["quality_passed"] for record in records)
    summary["per_image_failures"] = failures
    if summary["sdqm"]:
        records = attach_sdqm_metadata(records, summary["sdqm"])
    write_metadata_jsonl(records, OUTPUT_DIR)
    (Path(OUTPUT_DIR) / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    for error in summary["errors"]:
        print(error, flush=True)
    print(f"Evaluation reports saved to {OUTPUT_DIR}", flush=True)
    return 1 if failures or summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
