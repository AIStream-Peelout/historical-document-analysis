# File name: build_genizah_dataset.py
# Date: 7/22/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""Build the Genizah fragment-transcription fine-tuning dataset (Task C).

Consumes the conservative training-clean manifest produced by
``src.datasets.cleaning.clean_genizah_transcriptions`` (triage-A docs that
passed :func:`is_training_clean`: single image, single kept diplomatic
section, low reconstruction share, no mojibake/apparatus residue) plus the
locally downloaded primary images, and emits a ``DatasetDict`` with the same
column layout as the Talmud builder so the existing Colab notebooks can
interleave it without changes.

Gap policy: the cleaner's internal ``␣gap␣`` token is rendered as ``[...]``
in answers, matching :data:`FRAGMENT_TRANSCRIBE_PROMPT`'s instruction for
damaged/illegible passages (all other brackets were stripped upstream, so
``[...]`` is unambiguous).

Usage::

    python -m src.finetuning.qwen_hebrew.build_genizah_dataset \
        --push-to-hub isaacmg/genizah_clean_v1
"""
import argparse
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from src.datasets.cleaning.clean_genizah_transcriptions import GAP
from src.finetuning.qwen_hebrew.prompts import FRAGMENT_TRANSCRIBE_PROMPT

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = _REPO / "src/datasets/raw_data/genizah_cleanup/train_clean_manifest.jsonl"
DEFAULT_IMAGES_DIR = _REPO / "genizah_images/merged_v1/clean"
DEFAULT_OUTPUT_DIR = _REPO / "src/datasets/processed/qwen_hebrew_genizah"

VAL_FRACTION = 0.05
SPLIT_SEED = 20260722
GAP_RENDERING = "[...]"
MIN_IMAGE_SIDE_PX = 400

FEATURES = Features(
    {
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


def render_answer(cleaned_text: str) -> str:
    """Convert cleaner output to the training answer format.

    :param cleaned_text: Text with internal :data:`GAP` tokens.
    :returns: NFC-normalized answer with gaps rendered as ``[...]``.
    """
    text = cleaned_text.replace(GAP, GAP_RENDERING)
    return unicodedata.normalize("NFC", text).strip()


def iter_rows(manifest_path: Path, images_dir: Path,
              editors: Dict[str, str]) -> Iterator[Dict[str, Any]]:
    """Yield dataset rows for every manifest doc with a usable local image.

    :param manifest_path: JSONL manifest from the cleaning pipeline.
    :param images_dir: Directory of downloaded primary images (``<doc_id>.jpg``).
    :param editors: Mapping of doc_id to kept-section editor attribution.
    :returns: Iterator of feature dicts matching :data:`FEATURES`.
    """
    skipped_img = 0
    for line in open(manifest_path):
        rec = json.loads(line)
        doc_id = rec["doc_id"]
        matches = list(images_dir.glob(doc_id + ".*"))
        if not matches:
            skipped_img += 1
            continue
        img_path = matches[0]
        with PILImage.open(img_path) as im:
            width, height = im.size
        if min(width, height) < MIN_IMAGE_SIDE_PX:
            skipped_img += 1
            continue
        answer = render_answer(rec["cleaned_text"])
        yield {
            "image": str(img_path),
            "question": FRAGMENT_TRANSCRIBE_PROMPT,
            "answer": answer,
            "task": "fragment_transcribe",
            "section": "genizah",
            "stem": doc_id,
            "label_source": editors.get(doc_id, "unknown"),
            "target_chars": len(answer),
            "target_tokens": 0,
            "image_width": width,
            "image_height": height,
        }
    if skipped_img:
        logger.warning("Skipped %d docs (missing or too-small image)", skipped_img)


def load_editors(full_cleaned_path: Path) -> Dict[str, str]:
    """Map each doc to the editor of its kept transcription section.

    :param full_cleaned_path: ``full_cleaned.jsonl`` from the cleaning run.
    :returns: doc_id -> editor string.
    """
    editors: Dict[str, str] = {}
    for line in open(full_cleaned_path):
        rec = json.loads(line)
        kept = [s for s in rec["sections"] if s["kept"]]
        if kept:
            editors[rec["doc_id"]] = kept[0]["editor"] or "unknown"
    return editors


def build(manifest_path: Path, images_dir: Path, output_dir: Path,
          push_to_hub: Optional[str] = None) -> DatasetDict:
    """Build, save and optionally push the Task C dataset.

    :param manifest_path: Training-clean manifest JSONL.
    :param images_dir: Directory of downloaded primary images.
    :param output_dir: ``save_to_disk`` destination.
    :param push_to_hub: Private hub repo id to push to (skipped when ``None``).
    :returns: The train/val ``DatasetDict``.
    """
    editors = load_editors(manifest_path.parent / "full_cleaned.jsonl")
    rows: List[Dict[str, Any]] = list(iter_rows(manifest_path, images_dir, editors))
    if not rows:
        raise RuntimeError("No rows built — are the images downloaded?")
    ds = Dataset.from_list(rows, features=FEATURES)
    split = ds.train_test_split(test_size=VAL_FRACTION, seed=SPLIT_SEED)
    dsd = DatasetDict({"train": split["train"], "val": split["test"]})
    logger.info("train=%d val=%d", len(dsd["train"]), len(dsd["val"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(output_dir))
    if push_to_hub:
        dsd.push_to_hub(push_to_hub, private=True)
        logger.info("Pushed to %s (private)", push_to_hub)
    return dsd


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--push-to-hub", default=None,
                        help="private HF repo id, e.g. isaacmg/genizah_clean_v1")
    args = parser.parse_args()
    build(args.manifest, args.images_dir, args.output_dir, args.push_to_hub)
