from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import GEN_IMAGES_DIR, REAL_IMAGES_DIR
from src.evaluation.reporting import run_dataset_evaluation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate CMMD and SDQM without loading image-generation models."
    )
    parser.add_argument("--real-dir", default=REAL_IMAGES_DIR)
    parser.add_argument("--synthetic-dir", default=GEN_IMAGES_DIR)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    result = run_dataset_evaluation(arguments.real_dir, arguments.synthetic_dir)
    print(f"CMMD: {result.cmmd_score:.4f}")
    for metric_name, metric_value in sorted(result.sdqm_metrics.items()):
        print(f"{metric_name}: {metric_value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
