"""Default locations for the novel-rows build. Every CLI takes these as arguments; the defaults are site conventions."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
#: The novelty ledger (one record per candidate with verdict, witnesses and document group).
DEFAULT_POOL = REPO_ROOT / "logs/next_round/unseen_v1/unseen_pool.jsonl"
#: Where candidate files of the ledger live (per-source extraction, incl. edition text).
DEFAULT_CANDIDATES_DIR = REPO_ROOT / "logs/next_round/unseen_v1/candidates"
#: Scratch and cache dir of the build (images, manifests, reads, stage files). Large; lives on the offload volume.
DEFAULT_WORK_DIR = Path(os.environ.get("NOVEL_BUILD_WORK_DIR", "/Volumes/home/studio_offload/datasets/genizah_novel_v1_work"))
#: The dataset this build writes (never an existing dataset).
DEFAULT_OUT_DIR = Path(os.environ.get("NOVEL_BUILD_OUT_DIR", "/Volumes/home/studio_offload/datasets/genizah_novel_v1"))
#: Datasets root (read-only inputs: earlier builds, work dirs with cached reads).
DATASETS_ROOT = Path(os.environ.get("NOVEL_BUILD_DATASETS_ROOT", "/Volumes/home/studio_offload/datasets"))

STAGE_FILES = {
    "candidates": "stages/stage0_candidates.jsonl",
    "images": "stages/stage1_images.jsonl",
    "reads": "reads/reads.jsonl",
    "sides": "stages/stage2_sides.jsonl",
    "quality": "stages/stage3_quality.jsonl",
    "holdout": "stages/stage5_holdout.jsonl",
    "rows": "stages/stage6_rows.jsonl",
    "decisions": "stages/stage6_decisions.jsonl",
}


def stage_path(work_dir: Path, stage: str) -> Path:
    """Return the stage file path inside a work dir.

    :param work_dir: The build's work dir.
    :param stage: A key of :data:`STAGE_FILES`.
    :return: The file path (its parent may not exist yet).
    """
    return work_dir / STAGE_FILES[stage]
