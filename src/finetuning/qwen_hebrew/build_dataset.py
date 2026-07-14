"""Build the HF training dataset for Qwen3-VL Hebrew fine-tuning.

Pairs section crops (Task A) and full pages (Task B) from the full Talmud
corpus with their ground-truth transcriptions, holding out every page that
appears in the ``talmud_sample`` benchmark.

Schema (consumable by both mlx-vlm and Unsloth — each trainer builds its own
chat-message format from these columns):

- ``image``           — one PIL image (crop or full page)
- ``question``        — the training prompt
- ``answer``          — ground-truth transcription (NFC-normalized)
- ``task``            — ``crop_transcribe`` (A) or ``page_extract`` (B)
- ``section``         — gemara / rashi / tosafot
- ``stem``            — GT file stem, e.g. ``01_10b``
- ``label_source``    — ``gold`` here; Phase 2/3 adds ``kraken_consensus`` etc.
- ``target_chars`` / ``target_tokens`` / ``image_width`` / ``image_height``

Splits: ``train``, ``val`` (whole held-back pages, stratified by volume),
``smoke`` (32 fixed train crops; the first 8 are the overfit sanity set).

Usage
-----
  python -m src.finetuning.qwen_hebrew.build_dataset
  python -m src.finetuning.qwen_hebrew.build_dataset --push_to_hub user/repo
"""

import argparse
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

from datasets import Dataset, DatasetDict, Features, Image, Value

from src.datasets.document_models.talmud_gt_parser import load_gt_directory
from src.finetuning.qwen_hebrew.prompts import (
    CROP_TRANSCRIBE_PROMPTS,
    SECTION_EXTRACT_PROMPTS,
    SECTIONS,
)

_REPO = Path(__file__).resolve().parents[3]
_EVALS = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations"

FULL_TEXTS_DIR = _EVALS / "talmud_full/talmud_complete/texts"
FULL_IMAGES_DIR = _EVALS / "talmud_full/talmud_complete/converted_images"
FULL_CROPS_DIR = _EVALS / "talmud_full/talmud_complete/crops"
SAMPLE_TEXTS_DIR = _EVALS / "talmud_sample/texts"

DEFAULT_OUTPUT_DIR = _REPO / "src/datasets/processed/qwen_hebrew_talmud"
DEFAULT_TOKENIZER = "Qwen/Qwen3-VL-8B-Instruct"

MAX_TARGET_TOKENS = 4096
MIN_CROP_WIDTH_PX = 300
VAL_PAGES = 100
SMOKE_SIZE = 32
SPLIT_SEED = 20260714

FEATURES = Features(
    {
        # single Image column — Sequence(Image()) hits a datasets/pyarrow bug
        # in embed_storage during save_to_disk; both mlx-vlm and the notebook
        # accept a scalar "image" column.
        "image": Image(),
        "question": Value("string"),
        "answer": Value("string"),
        "task": Value("string"),
        "section": Value("string"),
        "stem": Value("string"),
        "label_source": Value("string"),
        "target_chars": Value("int32"),
        "target_tokens": Value("int32"),
        "image_width": Value("int32"),
        "image_height": Value("int32"),
    }
)


def normalize_target(text: str) -> str:
    """NFC-normalize a transcription target, preserving internal line breaks.

    :param text: Raw ground-truth section text.
    :returns: Normalized text with outer whitespace stripped.
    """
    return unicodedata.normalize("NFC", text).strip()


def benchmark_holdout_stems(sample_texts_dir: Path = SAMPLE_TEXTS_DIR) -> Set[str]:
    """Return the GT stems of every page in the benchmark sample.

    :param sample_texts_dir: Directory of benchmark ground-truth ``.txt`` files.
    :returns: Set of stems that must never appear in training data.
    """
    stems = {p.stem for p in sample_texts_dir.glob("*.txt")}
    if not stems:
        raise FileNotFoundError(f"No benchmark GT files found in {sample_texts_dir}")
    return stems


def load_crop_manifest(crops_dir: Path = FULL_CROPS_DIR) -> Dict[str, Dict]:
    """Load the crop manifest produced by ``crop_full_corpus``.

    :param crops_dir: Directory containing crops and ``crop_manifest.json``.
    :returns: Mapping from GT stem (page suffix stripped) to manifest entry.
    """
    manifest_path = crops_dir / "crop_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found — run "
            f"`python -m src.finetuning.qwen_hebrew.crop_full_corpus` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_stem: Dict[str, Dict] = {}
    for page in manifest["pages"]:
        gt_stem = page["stem"].replace("_page_001", "")
        by_stem[gt_stem] = page
    return by_stem


def _token_counter(tokenizer_name: str):
    """Build a target-token counting function from a HF tokenizer.

    :param tokenizer_name: HF hub id or local path of the tokenizer.
    :returns: Callable mapping text to its token count.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


def collect_records(
    gt_by_stem: Dict[str, Dict[str, str]],
    manifest_by_stem: Dict[str, Dict],
    count_tokens,
    page_task_stems: Set[str],
    stats: Counter,
) -> List[Dict]:
    """Assemble raw dataset records for both tasks.

    :param gt_by_stem: Parsed ground truth, ``{stem: {section: text}}``.
    :param manifest_by_stem: Crop manifest entries keyed by GT stem.
    :param count_tokens: Callable mapping text to token count.
    :param page_task_stems: Stems that also get Task B (full page) rows.
    :param stats: Counter mutated with skip/keep tallies.
    :returns: List of record dicts (images stored as file paths).
    """
    records: List[Dict] = []
    for stem, sections in sorted(gt_by_stem.items()):
        page_entry = manifest_by_stem.get(stem)
        if page_entry is None:
            stats["pages_missing_crops"] += 1
            continue

        page_image = Path(page_entry["source_image"])
        # Pages missing any section have a non-standard layout (e.g. Nedarim
        # prints the Ran where Rashi normally sits and swaps Tosafot's side),
        # so the fixed crop geometry would pair margin crops with the WRONG
        # text. Only the centre gemara column is layout-invariant.
        standard_layout = all(sections.get(s, "").strip() for s in SECTIONS)
        for section in SECTIONS:
            gt = sections.get(section)
            if not gt or not gt.strip():
                stats["sections_missing_gt"] += 1
                continue
            if section != "gemara" and not standard_layout:
                stats["sections_skipped_nonstandard_layout"] += 1
                continue
            answer = normalize_target(gt)
            n_tokens = count_tokens(answer)
            if n_tokens > MAX_TARGET_TOKENS:
                stats["sections_too_long"] += 1
                continue

            crop_info = page_entry["crops"][section]
            crop_path = Path(crop_info["path"])
            if not crop_path.exists():
                stats["crop_files_missing"] += 1
                continue
            if crop_info["width"] < MIN_CROP_WIDTH_PX:
                stats["crops_too_narrow"] += 1
                continue

            base = {
                "answer": answer,
                "section": section,
                "stem": stem,
                "label_source": "gold",
                "target_chars": len(answer),
                "target_tokens": n_tokens,
            }
            records.append(
                {
                    **base,
                    "image": str(crop_path),
                    "question": CROP_TRANSCRIBE_PROMPTS[section],
                    "task": "crop_transcribe",
                    "image_width": crop_info["width"],
                    "image_height": crop_info["height"],
                }
            )
            stats["crop_records"] += 1

            if stem in page_task_stems and page_image.exists():
                records.append(
                    {
                        **base,
                        "image": str(page_image),
                        "question": SECTION_EXTRACT_PROMPTS[section],
                        "task": "page_extract",
                        "image_width": page_entry["width"],
                        "image_height": page_entry["height"],
                    }
                )
                stats["page_records"] += 1
    return records


def pick_val_stems(stems: List[str], n_pages: int, seed: int) -> Set[str]:
    """Pick validation pages stratified by volume prefix.

    :param stems: All candidate GT stems.
    :param n_pages: Number of validation pages to select.
    :param seed: RNG seed for a deterministic split.
    :returns: Set of validation stems.
    """
    by_volume: Dict[str, List[str]] = {}
    for stem in sorted(stems):
        by_volume.setdefault(stem.split("_")[0], []).append(stem)

    rng = random.Random(seed)
    volumes = sorted(by_volume)
    picked: List[str] = []
    while len(picked) < n_pages and volumes:
        for vol in volumes:
            pool = [s for s in by_volume[vol] if s not in picked]
            if pool and len(picked) < n_pages:
                picked.append(rng.choice(pool))
    return set(picked)


def build(
    output_dir: Path,
    tokenizer_name: str,
    page_task_pages: int,
    push_to_hub: Optional[str] = None,
) -> None:
    """Build and save the dataset splits plus ``stats.json``.

    :param output_dir: Directory for ``save_to_disk`` output.
    :param tokenizer_name: HF tokenizer used for target-length filtering.
    :param page_task_pages: Number of pages that also get Task B rows
        (full-page images are large; keep this bounded).
    :param push_to_hub: Optional private HF hub repo id to push to.
    """
    stats: Counter = Counter()

    gt_by_stem = load_gt_directory(FULL_TEXTS_DIR)
    holdout = benchmark_holdout_stems()
    train_gt = {s: v for s, v in gt_by_stem.items() if s not in holdout}
    stats["holdout_pages_excluded"] = len(gt_by_stem) - len(train_gt)

    leaked = set(train_gt) & holdout
    assert not leaked, f"Benchmark leakage: {sorted(leaked)[:5]}"

    manifest_by_stem = load_crop_manifest()
    count_tokens = _token_counter(tokenizer_name)

    rng = random.Random(SPLIT_SEED)
    page_task_stems = set(
        rng.sample(sorted(train_gt), min(page_task_pages, len(train_gt)))
    )

    records = collect_records(
        train_gt, manifest_by_stem, count_tokens, page_task_stems, stats
    )

    val_stems = pick_val_stems(list(train_gt), VAL_PAGES, SPLIT_SEED)
    train_records = [r for r in records if r["stem"] not in val_stems]
    val_records = [r for r in records if r["stem"] in val_stems]

    smoke_records = [
        r for r in train_records if r["task"] == "crop_transcribe"
    ][:SMOKE_SIZE]

    splits = DatasetDict(
        {
            "train": Dataset.from_list(train_records, features=FEATURES),
            "val": Dataset.from_list(val_records, features=FEATURES),
            "smoke": Dataset.from_list(smoke_records, features=FEATURES),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(output_dir))

    token_counts = sorted(r["target_tokens"] for r in records)

    def pct(p: float) -> int:
        return token_counts[int(p * (len(token_counts) - 1))] if token_counts else 0

    stats_out = {
        "counts": {
            split: {
                "total": len(ds),
                "by_task": dict(Counter(ds["task"])),
                "by_section": dict(Counter(ds["section"])),
            }
            for split, ds in splits.items()
        },
        "skips": {k: v for k, v in stats.items() if k not in ("crop_records", "page_records")},
        "target_tokens": {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": pct(1.0)},
        "max_target_tokens": MAX_TARGET_TOKENS,
        "min_crop_width_px": MIN_CROP_WIDTH_PX,
        "holdout": {
            "benchmark_stems_excluded": sorted(holdout),
            "leak_check": "PASSED — zero benchmark stems in any split",
        },
        "val_pages": sorted(val_stems),
        "page_task_pages": len(page_task_stems),
        "tokenizer": tokenizer_name,
        "split_seed": SPLIT_SEED,
    }
    stats_path = output_dir / "stats.json"
    stats_path.write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n✅ Dataset saved → {output_dir}")
    for split, ds in splits.items():
        print(f"   {split:6s}: {len(ds):6d} rows  {dict(Counter(ds['task']))}")
    print(f"   Skips: {dict(stats_out['skips'])}")
    print(f"   Stats: {stats_path}")

    if push_to_hub:
        splits.push_to_hub(push_to_hub, private=True)
        print(f"   Pushed to hub (private): {push_to_hub}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    :param argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Build the Qwen3-VL Hebrew training dataset.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--page_task_pages", type=int, default=1000,
        help="Pages that also get full-page (Task B) rows — bounded because "
             "each adds ~3 copies of a ~1.5MB page image to the artifact.",
    )
    parser.add_argument("--push_to_hub", type=str, default=None,
                        help="Private HF hub repo id, e.g. user/qwen-hebrew-talmud.")
    args = parser.parse_args(argv)

    build(
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        page_task_pages=args.page_task_pages,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
