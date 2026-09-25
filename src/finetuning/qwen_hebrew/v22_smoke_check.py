# File name: v22_smoke_check.py
# Date: 9/24/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""CPU smoke test of a v22 images-once mixture through the real Qwen3-VL processor.

Runs on the Mac before a Colab launch and answers "does the data path work at all":
every sampled row decodes, the chat template renders, the processor packs the image at
the training resolution contract (6.5–7 MP ⇒ roughly 25k patch rows per page), the
patch rows equal the image grid product (the v2.1 truncation signature), the answer
tokens sit after the assistant header (so ``train_on_responses_only`` masks the prompt),
and the sequence fits ``MAX_SEQ``. It does not load model weights.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.v22_smoke_check \
        --data-dir /Volumes/home/studio_offload/datasets/genizah_v22_pilot --rows 12
"""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from src.finetuning.qwen_hebrew.images_once import ImagesOnceDataset

PROCESSOR_ID = "unsloth/Qwen3-VL-8B-Instruct"
MIN_PIX = 6_500_000
MAX_PIX = 7_000_000
MAX_SEQ = 12288
ASSISTANT_HEADER = "<|im_start|>assistant\n"


def to_conversation(sample: Dict) -> List[Dict]:
    """Chat messages for one row, identical to the notebook's ``to_conversation``.

    :param sample: Row dict with ``image``, ``question`` and ``answer``.
    :type sample: Dict
    :return: Two-turn message list (user: image + question; assistant: answer).
    :rtype: List[Dict]
    """
    return [
        {"role": "user", "content": [{"type": "image", "image": sample["image"]},
                                     {"type": "text", "text": sample["question"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
    ]


def pick_rows(ds: ImagesOnceDataset, n: int, seed: int) -> List[int]:
    """Row indices covering every source, then random fill.

    :param ds: The mixture split.
    :type ds: ImagesOnceDataset
    :param n: Total rows wanted.
    :type n: int
    :param seed: Random seed.
    :type seed: int
    :return: Distinct row indices.
    :rtype: List[int]
    """
    sources = ds._table["source"].to_pylist() if "source" in ds._table.column_names else [""] * len(ds)
    first: Dict[str, int] = {}
    for i, s in enumerate(sources):
        first.setdefault(s, i)
    chosen = list(first.values())
    rng = random.Random(seed)
    pool = [i for i in range(len(ds)) if i not in set(chosen)]
    chosen += rng.sample(pool, max(0, min(n, len(ds)) - len(chosen)))
    return chosen[:n]


def check_row(processor, row: Dict) -> Dict:
    """Run one row through the processor and return its measurements.

    :param processor: A Qwen3-VL ``AutoProcessor`` with the training pixel bounds.
    :param row: Decoded dataset row.
    :type row: Dict
    :return: Measurements (patch rows, grid product, sequence length, answer position, timings).
    :rtype: Dict
    :raises AssertionError: When a training invariant fails for the row.
    """
    t0 = time.perf_counter()
    text = processor.apply_chat_template(to_conversation(row), tokenize=False, add_generation_prompt=False)
    enc = processor(text=[text], images=[row["image"]], return_tensors="pt")
    dt = time.perf_counter() - t0
    pv, grid = enc["pixel_values"], enc["image_grid_thw"]
    need = int(grid.prod(dim=-1).sum())
    assert pv.shape[0] == need, f"pixel rows {pv.shape[0]} != grid product {need} (truncation)"
    seq = int(enc["input_ids"].shape[1])
    assert seq <= MAX_SEQ, f"sequence {seq} exceeds MAX_SEQ {MAX_SEQ}"
    assert ASSISTANT_HEADER in text, "assistant header missing from the rendered template"
    pos = text.index(ASSISTANT_HEADER) + len(ASSISTANT_HEADER)
    assert text[pos:].startswith(row["answer"][:40]), "answer does not follow the assistant header"
    assert row["question"][:40] in text[:pos], "prompt not in the user turn"
    w, h = row["image"].size
    return {"patch_rows": int(pv.shape[0]), "grid": grid[0].tolist(), "seq_len": seq,
            "image_px": w * h, "answer_chars": len(row["answer"]), "secs": round(dt, 2)}


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    :param argv: Arguments (None = ``sys.argv``).
    :type argv: Sequence[str] | None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args(argv)

    from transformers import AutoProcessor  # imported late: slow, and only needed here
    processor = AutoProcessor.from_pretrained(PROCESSOR_ID, min_pixels=MIN_PIX, max_pixels=MAX_PIX)
    ds = ImagesOnceDataset(args.data_dir / "rows" / f"{args.split}.parquet", args.data_dir / "images")
    idx = pick_rows(ds, args.rows, args.seed)
    print(f"{args.split}: {len(ds)} rows; checking {len(idx)}; processor pixels {MIN_PIX}-{MAX_PIX}")
    results = []
    for i in idx:
        row = ds[i]
        m = check_row(processor, row)
        m.update({"row": i, "source": row.get("source", "?"), "task": row["task"]})
        results.append(m)
        print(f"  row {i:5d} {m['source']:22} {m['task']:20} patches {m['patch_rows']:6d} grid {m['grid']} "
              f"seq {m['seq_len']:5d} answer {m['answer_chars']:4d} chars  {m['secs']}s")
    patches = [r["patch_rows"] for r in results]
    print(json.dumps({"rows_checked": len(results), "sources": dict(Counter(r["source"] for r in results)),
                      "patch_rows_min_max": [min(patches), max(patches)],
                      "seq_len_max": max(r["seq_len"] for r in results),
                      "collate_secs_mean": round(sum(r["secs"] for r in results) / len(results), 2)},
                     ensure_ascii=False))
    assert min(patches) > 15000, "some image packed far below the 6.5 MP contract"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
