# File name: push_ktiv_dataset.py
# Date: 9/9/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Push a saved KTIV DatasetDict (arrow on the NAS) to a private hub repo.

Kept separate from the build so a hub failure (quota, network) never costs
the two-hour build. Re-shapes the train split into one split per task family
(``train_<task>``) so the notebook can stream each family on its own; ``val``
is pushed whole. Prints the resulting revision SHA for the notebook pin.

Usage (repo root):
    .venv/bin/python -m src.finetuning.qwen_hebrew.push_ktiv_dataset \\
        --src /Volumes/home/studio_offload/datasets/genizah_ktiv_v3 --repo isaacmg/genizah_ktiv_v3
"""
import argparse
import json
import os
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import HfApi


_BOX_TASKS = ("locate", "locate_word")
_BOX_TEXT_TASKS = ("line_index", "line_of_phrase")
_ARRAY_TASKS = ("grounded_page", "grounded_detect", "grounded_crop")


def _ok_box(b: object) -> bool:
    """A valid 0-1000 integer box with positive width and height."""
    return (isinstance(b, list) and len(b) == 4 and all(isinstance(v, int) and 0 <= v <= 1000 for v in b)
            and b[2] > b[0] and b[3] > b[1])


def _answer_ok(task: str, answer: str) -> bool:
    """Validate a grounding answer; non-grounding tasks always pass.

    :param task: Row task family.
    :param answer: Row answer string.
    :returns: False for malformed JSON, degenerate boxes or empty texts.
    """
    try:
        if task in _BOX_TASKS:
            return _ok_box(json.loads(answer)["bbox_2d"])
        if task in _BOX_TEXT_TASKS:
            o = json.loads(answer)
            return _ok_box(o["bbox_2d"]) and bool(str(o["text"]).strip())
        if task in _ARRAY_TASKS:
            arr = json.loads(answer)
            return isinstance(arr, list) and bool(arr) and all(
                _ok_box(e["bbox_2d"]) and bool(str(e["text"]).strip()) for e in arr)
    except (ValueError, KeyError, TypeError):
        return False
    return True


def main() -> None:
    """Load the arrow dataset, print its composition, push, print the revision."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="save_to_disk directory")
    ap.add_argument("--repo", required=True, help="hub dataset id (pushed PRIVATE)")
    ap.add_argument("--dry-run", action="store_true", help="only print composition and size")
    a = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    token = os.environ.get("HF1_TOKEN") or os.environ.get("HF_TOKEN")
    dsd = load_from_disk(str(a.src))
    # Hub layout: ONE TRAIN SPLIT PER TASK FAMILY (train_<task>) + the whole val
    # split. The v2.1 notebook streams the train families (the dataset is too
    # big for Colab's disk twice over), and per-family splits let each source
    # read only its own parquet shards instead of re-scanning everything.
    out = DatasetDict()
    for split_in, prefix in (("train", "train_"), ("val", None)):
        ds = dsd[split_in]
        meta = ds.select_columns(["task", "answer"]).to_pandas()
        keep = meta.index[[_answer_ok(t, a) for t, a in zip(meta["task"], meta["answer"])]]
        dropped = len(meta) - len(keep)
        if dropped:
            print(f"{split_in}: dropping {dropped} rows whose grounding answer failed validation")
        if prefix is None:
            out["val"] = ds.select([int(i) for i in keep])
            print(f"val: {len(keep)} rows")
            continue
        tasks = meta.loc[keep, "task"]
        for task in sorted(tasks.unique()):
            idx = [int(i) for i in tasks.index[tasks == task]]
            out[f"{prefix}{task}"] = ds.select(idx)        # index view, no copy
            print(f"{prefix}{task}: {len(idx)} rows")
    size = sum(p.stat().st_size for p in a.src.rglob("*.arrow")) / 1e9
    print(f"arrow bytes on disk: {size:.1f} GB (hub parquet will be similar)")
    if a.dry_run:
        return
    out.push_to_hub(a.repo, private=True, token=token)
    sha = HfApi(token=token).dataset_info(a.repo).sha
    print(f"pushed {a.repo} PRIVATE — revision {sha}")


if __name__ == "__main__":
    main()
