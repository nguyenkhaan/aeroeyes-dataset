from __future__ import annotations

import importlib.util
from pathlib import Path

from src.core.config import CMMD_BATCH_SIZE, CMMD_MAX_COUNT, CMMD_REPO_DIR


def _load_compute_cmmd():
    cmmd_main = Path(CMMD_REPO_DIR) / "main.py"
    if not cmmd_main.is_file():
        raise FileNotFoundError(
            f"CMMD repo not found at {cmmd_main}. "
            "Clone it with: git clone https://github.com/sayakpaul/cmmd-pytorch.git cmmd_pt"
        )

    spec = importlib.util.spec_from_file_location(
        "cmmd_pytorch_main",
        cmmd_main,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CMMD module from {cmmd_main}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_cmmd


def compute_dataset_cmmd(
    ref_dir: str,
    eval_dir: str,
    batch_size: int = CMMD_BATCH_SIZE,
    max_count: int = CMMD_MAX_COUNT,
) -> float:
    """
    Compute CMMD between reference and generated image directories.

    Matches humaninstruction-ver2-8 Cell 13.
    """
    compute_cmmd = _load_compute_cmmd()
    return float(
        compute_cmmd(
            ref_dir=ref_dir,
            eval_dir=eval_dir,
            batch_size=batch_size,
            max_count=max_count,
        )
    )
