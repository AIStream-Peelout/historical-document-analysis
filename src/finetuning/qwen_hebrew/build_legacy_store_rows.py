# File name: build_legacy_store_rows.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Page-transcription rows from the legacy store's diplomatic transcriptions (PRIVATE; no training use until reviewed).

The project's legacy image store carries transcriptions of documents whose images we hold. The legacy-store cleaner
turned the transcriptions of the documents in scope (one to four images each; codices with five or more images are out
of scope) into side records (``cleaned_sections.jsonl``: one record per side of a section, cleaned text with the
``[...]`` gap token). Few records say which image holds their text, so this builder places each record on an image by
*reading* the images: cached Kraken reads (KTIV-trained Hebrew-script model) and, for Arabic-script pages, cached
Gemini reads. Nothing is called here: no API, no LM Studio, no Kraken.

1. **Reads.** Per image (by sha256): the Gemini read when it says something other than ``NO TEXT``, else the Kraken
   lines joined. A read is unusable when Kraken found no line, when it is a decoder loop
   (``arabic_script.repeat_share`` >= ``side_match.MAX_REPEAT_SHARE``) or when it has fewer than
   :data:`MIN_READ_LETTERS` letters of its script.
2. **Skeletons.** Hebrew script: letters only, final letters folded to their base forms, points dropped
   (:func:`hebrew_skeleton`). Arabic script: the rasm (``arabic_bench_coverage.rasm_skeleton``).
3. **Recall and its null.** Clipped 4-gram recall of a side's skeleton in an image's read (3-grams as a second
   signal). Every side has its own null: the same recall against the usable reads of images of OTHER documents in the
   same script (Hebrew: the Kraken reads; Arabic: the Gemini reads with text), leaving out the side's own document,
   every document sharing its text and every image byte-identical to one of its own. The null is stratified by the
   read's length (:data:`BINS`); p95 / p99 with a bootstrap CI (:data:`N_BOOT` resamples) per bin; a bin with fewer
   than :data:`MIN_BIN_NULL` null reads falls back to the unstratified null.
4. **Duplicates and shared texts.** Duplicate versions of one text inside a document (cleaner decision
   ``duplicate_version``, paired by the cleaner's ``near_duplicates`` rule) keep the version with more letters.
   Records whose raw text also belongs to other documents (``shared_text_docs``: an upstream join bug through
   shelf-mark prefix collisions) are owned by the document whose image gives the best 4-gram recall clearing its p99,
   but only when it is the sole copy clearing p99 or its recall is at least :data:`MIN_OWNER_MARGIN` times the
   runner-up's; the other documents' copies are dropped (``shared_text_not_owner``); no document clearing, a tie, or a
   lead under that margin (``ownership_contested``) sends every copy to review.
5. **Assignment.** Per document, sides are matched one to one to distinct images by Hungarian assignment on 4-gram
   recall (side labels and the stored-order prior only break ties). An assigned pair is accepted when the recall
   clears the side's p99 for the image's read-length bin (p95 when the side label agrees with the image's file-name
   token; the order prior never relaxes the threshold), the side shares at least :data:`MIN_MATCHED_GRAMS` 4-grams
   with the read, its 3-gram recall clears the 3-gram p95, no other side of the document clears its own p95 on the
   image, and the side clears its p95 on no other image. Everything else goes to ``needs_review`` (some image clears
   p95) or ``unassigned`` (none does), with reasons. An image accepted for two documents goes to review.
6. **Coverage.** Flags: read letters / target letters outside :data:`LETTER_RATIO_RANGE`, a Kraken image without
   lines, other-script letters in the read above :data:`MAX_OTHER_SCRIPT_SHARE`, another image of the document
   without a usable read, byte-identical images in the document. Exclusions: the ``edition_apparatus`` family, the
   ``editorial_terms`` flag (editor prose kept as ink), gap density over :data:`MAX_GAP_DENSITY` gaps per 100 letters
   (30 %), fewer than :data:`MIN_TARGET_LETTERS` letters, a read with more than :data:`MAX_READ_OVER_TARGET` times
   the target's letters (Kraken under-reads, so such a page holds text the target lacks).
7. **Leakage** (before export). Identity: ``derive_v25a.held_out_reason`` on a ``DocRef`` built from the PGP ids of
   the merged-corpus record, the canonical id and the shelf-mark parts (split on ``" + "``); ``decontam_gate`` loose
   keys in the clean v2 and benchmark inventories (the frozen 131 are in both rules). Text: 25-letter Hebrew shingles
   (``DecontamGate.content_contaminated``) and 20-letter rasm shingles (at least :data:`ARABIC_MIN_SHARED_ONE_DOC`
   shared with one GT document, or :data:`ARABIC_MIN_SHARE_UNION` of the target's shingles in the union) against the
   GT of genizah_test_v1, genizah_religious_v1, sealed_v1 (documentary and literary), the two-edition footnote GT,
   the Arabic-script benchmark v0 and the clean v2 val texts. Any hit holds the row out with its reason.
8. **Rows.** KTIV ``FEATURES`` schema: ``question`` = ``SCRIPT_NEUTRAL_PROMPT``, ``answer`` = the cleaned text
   (explicit line breaks kept where the record has them, flat otherwise), ``task`` = ``fragment_transcribe``,
   ``section`` = :data:`SECTION`, stem ``legacy_<canonical_id>_<side>_c<k>_page`` (``<side>`` = the image's file-name
   token, the record's label only when the image has none). Splits ``train_page`` / ``val``
   by a hash of the document (``build_pgp_arabic_editions.split_of``, 5 %).

Output::

    <out>/              saved DatasetDict + manifest.jsonl + decisions.jsonl + stats.json + README.md
    <out>_images_once/  images-once export (``images_once.export_images_once``) + README.md

The dataset is PRIVATE: the images and transcriptions come from the legacy store; nothing is released, and no row is
used for training until the build report has been reviewed.

Usage::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.build_legacy_store_rows [--source DIR] [--out DIR]
        [--docs-list FILE] [--report FILE] [--eye-queue FILE] [--no-export]
"""
import argparse
import csv
import json
import shutil
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.datasets.cleaning.clean_genizah_transcriptions import _strip_letters, near_duplicates
from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations import benchmark_registry as br
from src.datasets.evaluations.helper_eval_scripts.arabic_bench_coverage import ngram_counts, rasm_skeleton
from src.datasets.evaluations.helper_eval_scripts.decontam_gate import _HEB, _SHINGLE_N, DecontamGate
from src.finetuning.qwen_hebrew import build_pgp_arabic_editions as pae
from src.finetuning.qwen_hebrew import side_match as sm
from src.finetuning.qwen_hebrew.colab import derive_v25a as d25
from src.finetuning.qwen_hebrew.prompts import SCRIPT_NEUTRAL_PROMPT

REPO = Path(__file__).resolve().parents[3]
NAS_DATASETS = Path("/Volumes/home/studio_offload/datasets")
SOURCE_DIR = NAS_DATASETS / "legacy_store_diplomatic"
DEFAULT_OUT = NAS_DATASETS / "legacy_store_diplomatic_v1"
PGP_EXPORT = REPO / "artifacts" / "pgp_export_b2343a17"
MERGED_JSONL = REPO / "src" / "datasets" / "raw_data" / "cairo_genizah" / "merged" / "merged_shelfmarks.jsonl"
EVAL_DIR = REPO / "src" / "datasets" / "raw_data" / "cairo_genizah" / "evaluations"
CLEAN_V2_VAL = REPO / "src" / "datasets" / "processed" / "qwen_hebrew_genizah_v2" / "val"
ARABIC_V0 = NAS_DATASETS / "arabic_script_benchmark_v0" / "benchmark.jsonl"

HEBREW, ARABIC = "hebrew", "arabic"
SCRIPTS = (HEBREW, ARABIC)
#: reader whose reads make the null pool of each script
POOL_SOURCE = {HEBREW: "kraken", ARABIC: "gemini"}
NGRAMS = (4, 3)
PRIMARY_N = 4
MIN_READ_LETTERS = 20
BINS = ("under_100", "100_300", "300_600", "over_600")
ALL_BINS = "all"
MIN_BIN_NULL = 20
N_BOOT = 500
SEED = 1030
#: tie-break weights added to the recall in the Hungarian assignment (priors never outweigh a recall difference)
PRIOR_EPS = {"side_label": 1e-6, "order_prior": 1e-7}
LETTER_RATIO_RANGE = (0.5, 1.5)
MAX_READ_OVER_TARGET = 2.0                 # read letters / target letters above this: the page holds untranscribed text
MIN_MATCHED_GRAMS = 5                      # an acceptance needs at least this many matched PRIMARY_N-grams
UNDER_MATCHED_REASON = f"under_{MIN_MATCHED_GRAMS}_matched_{PRIMARY_N}grams"
MIN_OWNER_MARGIN = 2.0                     # owner recall / runner-up recall when two copies of a shared text clear p99
OWNERSHIP_REVIEW_REASONS = {"no_owner": "shared_text_no_owner", "tie": "shared_text_tie",
                            "contested": "ownership_contested"}
MAX_OTHER_SCRIPT_SHARE = 0.10
MAX_GAP_DENSITY = 30.0                     # the cleaner's gap_density is gaps per 100 letters: 30 = 30 %
MIN_TARGET_LETTERS = pae.MIN_PAGE_LETTERS  # 40
EXCLUDED_FAMILIES = ("edition_apparatus",)
EXCLUDED_FLAGS = ("editorial_terms",)
HEBREW_SHINGLE = _SHINGLE_N                # 25 (decontam_gate)
ARABIC_SHINGLE = 20
ARABIC_MIN_SHARED_ONE_DOC = 3
ARABIC_MIN_SHARE_UNION = 0.30
VAL_SHARE = pae.VAL_SHARE
TASK = pae.TASK
SECTION = "legacy_store_diplomatic_page"
LABEL_SOURCE = "legacy_store_diplomatic"
PROMPT = SCRIPT_NEUTRAL_PROMPT
PROMPT_NAME = "SCRIPT_NEUTRAL_PROMPT"
DUPLICATE_DECISION = "duplicate_version"
NO_TEXT = re.compile(r"^\W*NO\s+TEXT\W*$", re.IGNORECASE)
HEBREW_LETTER = re.compile(r"[א-ת]")
FINAL_TO_BASE = str.maketrans("ךםןףץ", "כמנפצ")
CANONICAL_AT_START = re.compile(r'^\{"canonical_id": "((?:[^"\\]|\\.)*)"')
EDITION_RELATION = re.compile(r"\bEdition\b")
SPOT_CHECK = 10
QUOTE_WORDS = 8
OUTCOMES = ("accepted", "needs_review", "unassigned", "dropped")
#: review reasons a same-document background could explain (triage only; never an automatic acceptance)
RESOLVABLE_REASONS = ("side_clears_p95_on_another_image", "other_side_clears_p95_on_image")
#: triage bands of the dominance ratio (upper bound inclusive, band name)
TRIAGE_BANDS = ((0.2, "background_like"), (0.5, "unclear"), (float("inf"), "competing_text"))
GT_SETS = ("genizah_test_v1", "genizah_religious_v1", "sealed_v1_documentary", "sealed_v1_literary",
           "two_edition_footnotes", "arabic_script_benchmark_v0", "clean_v2_val")


# ===================================================================================================================
# Text
# ===================================================================================================================

def hebrew_skeleton(text: str) -> str:
    """Hebrew-script skeleton of a text: letters only, finals folded to base forms, points and everything else dropped.

    :param text: Any text (a target or a read).
    :type text: str
    :return: Hebrew letters in order, no spaces (``ך ם ן ף ץ`` written as ``כ מ נ פ צ``).
    :rtype: str
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(HEBREW_LETTER.findall(decomposed)).translate(FINAL_TO_BASE)


def skeleton_of(text: str, script: str) -> str:
    """Skeleton of a text in one script.

    :param text: Any text.
    :type text: str
    :param script: :data:`HEBREW` or :data:`ARABIC`.
    :type script: str
    :return: :func:`hebrew_skeleton` or ``rasm_skeleton``.
    :rtype: str
    :raises ValueError: For an unknown script.
    """
    if script == HEBREW:
        return hebrew_skeleton(text)
    if script == ARABIC:
        return rasm_skeleton(text or "")
    raise ValueError(f"unknown script {script!r}")


def majority_script(skeletons: Mapping[str, str]) -> str:
    """The script with more skeleton letters (Hebrew on a tie), or empty for a text without letters.

    :param skeletons: Skeleton per script.
    :type skeletons: Mapping[str, str]
    :return: :data:`HEBREW`, :data:`ARABIC` or ``""``.
    :rtype: str
    """
    he, arb = len(skeletons.get(HEBREW, "")), len(skeletons.get(ARABIC, ""))
    if he == 0 and arb == 0:
        return ""
    return HEBREW if he >= arb else ARABIC


def clipped_recall(side: Counter, read: Counter) -> float:
    """Share of a side's n-grams found in a read, each gram counted at most as often as the read has it.

    :param side: n-gram counts of the side's skeleton.
    :type side: Counter
    :param read: n-gram counts of the read's skeleton.
    :type read: Counter
    :return: Recall in [0, 1]; 0.0 for a side without n-grams.
    :rtype: float
    """
    total = sum(side.values())
    if not total:
        return 0.0
    return sum(min(count, read.get(gram, 0)) for gram, count in side.items()) / total


def length_bin(letters: int) -> str:
    """Read-length bin of a read with this many skeleton letters.

    :param letters: Skeleton letters of the read.
    :type letters: int
    :return: One of :data:`BINS`: under 100; 100-299; 300-600; over 600.
    :rtype: str
    """
    if letters < 100:
        return BINS[0]
    if letters < 300:
        return BINS[1]
    if letters <= 600:
        return BINS[2]
    return BINS[3]


def first_words(text: str, n: int = QUOTE_WORDS) -> str:
    """The first words of a text (for review queues; never more than ``n`` words).

    :param text: Text.
    :type text: str
    :param n: Words.
    :type n: int
    :return: Up to ``n`` whitespace-separated words joined by spaces.
    :rtype: str
    """
    return " ".join((text or "").split()[:n])


def round_floats(obj: Any, digits: int = 4) -> Any:
    """Round every float of a nested structure (for output; decisions are made at full precision).

    :param obj: Nested dicts / lists / scalars.
    :type obj: Any
    :param digits: Decimal places.
    :type digits: int
    :return: The same structure with rounded floats.
    :rtype: Any
    """
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, digits) for v in obj]
    return obj


def pct(part: int, whole: int) -> str:
    """``part/whole (x.x %)`` for reports.

    :param part: Numerator.
    :type part: int
    :param whole: Denominator.
    :type whole: int
    :return: Formatted share.
    :rtype: str
    """
    return f"{part:,}/{whole:,} ({100.0 * part / whole:.1f} %)" if whole else f"{part:,}/0"


# ===================================================================================================================
# Inputs
# ===================================================================================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Records of a JSONL file.

    :param path: JSONL file.
    :type path: Path
    :return: Records.
    :rtype: List[Dict[str, Any]]
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass(frozen=True)
class ImageInfo:
    """One image of a document.

    :ivar canonical_id: Document.
    :ivar index: Image index within the document (the order of its image list).
    :ivar file_name: File name in the legacy store.
    :ivar path: Local copy.
    :ivar url: Source URL (identity only; never written to outputs).
    :ivar sha256: Content hash.
    :ivar width: Pixels.
    :ivar height: Pixels.
    :ivar token: File-name side token with its leaf (``"1r"``, ``"2v"``), ``""`` when the name gives none.
    """
    canonical_id: str
    index: int
    file_name: str
    path: str
    url: str
    sha256: str
    width: int
    height: int
    token: str


def image_token(row: Mapping[str, Any]) -> str:
    """File-name side token of a manifest image (``"1r"``), or ``""``.

    :param row: ``images_manifest.jsonl`` row (``side_token`` r / v / none, ``folio``).
    :type row: Mapping[str, Any]
    :return: Leaf and side, the side alone without a leaf, or ``""``.
    :rtype: str
    """
    side = row.get("side_token") or ""
    if side not in ("r", "v"):
        return ""
    folio = row.get("folio")
    return f"{folio}{side}" if folio is not None else side


def load_images(manifest: Path) -> Dict[str, List[ImageInfo]]:
    """Images per document, in image order.

    :param manifest: ``images_manifest.jsonl``.
    :type manifest: Path
    :return: ``{canonical_id: [ImageInfo]}``.
    :rtype: Dict[str, List[ImageInfo]]
    """
    out: Dict[str, List[ImageInfo]] = defaultdict(list)
    for row in read_jsonl(manifest):
        out[row["canonical_id"]].append(ImageInfo(row["canonical_id"], int(row["image_index"]), row["file_name"],
                                                  row["path"], row["url"], row["sha256"], int(row["width"]),
                                                  int(row["height"]), image_token(row)))
    return {cid: sorted(images, key=lambda im: im.index) for cid, images in out.items()}


@dataclass
class ReadInfo:
    """The read of one image (by sha256).

    :ivar sha256: Image hash.
    :ivar source: ``kraken`` or ``gemini``.
    :ivar text: Read text (Kraken lines joined with newlines).
    :ivar n_lines: Kraken line count (None for a Gemini read).
    :ivar skeletons: Skeleton per script.
    :ivar script: Majority script of the read (``""`` without letters).
    :ivar repeat_share: ``arabic_script.repeat_share`` of the text.
    :ivar fingerprint: ``side_match.text_fingerprint`` of the text.
    :ivar unusable: Why the read cannot be used (``""`` when it can).
    """
    sha256: str
    source: str
    text: str
    n_lines: Optional[int]
    skeletons: Dict[str, str]
    script: str
    repeat_share: float
    fingerprint: str
    unusable: str = ""

    @property
    def letters(self) -> int:
        """Skeleton letters of the read in its own script.

        :return: Letter count (0 without a script).
        :rtype: int
        """
        return len(self.skeletons.get(self.script, "")) if self.script else 0

    def usable_for(self, script: str) -> bool:
        """Whether the read can be compared with a side of this script.

        :param script: Side script.
        :type script: str
        :return: True for a usable read whose majority script is ``script``.
        :rtype: bool
        """
        return not self.unusable and self.script == script


def make_read(sha256: str, source: str, text: str, n_lines: Optional[int]) -> ReadInfo:
    """A read with its skeletons and usability.

    :param sha256: Image hash.
    :type sha256: str
    :param source: ``kraken`` or ``gemini``.
    :type source: str
    :param text: Read text.
    :type text: str
    :param n_lines: Kraken line count, None for Gemini.
    :type n_lines: Optional[int]
    :return: The read; ``unusable`` is ``kraken_zero_lines``, ``no_text``, ``loop``,
        ``under_<MIN_READ_LETTERS>_letters`` or ``""``.
    :rtype: ReadInfo
    """
    text = text or ""
    skeletons = {script: skeleton_of(text, script) for script in SCRIPTS}
    read = ReadInfo(sha256, source, text, n_lines, skeletons, majority_script(skeletons), ar.repeat_share(text),
                    sm.text_fingerprint(text))
    if source == "kraken" and not n_lines:
        read.unusable = "kraken_zero_lines"
    elif not text.strip() or NO_TEXT.match(text.strip()):
        read.unusable = "no_text"
    elif read.repeat_share >= sm.MAX_REPEAT_SHARE:
        read.unusable = "loop"
    elif read.letters < MIN_READ_LETTERS:
        read.unusable = f"under_{MIN_READ_LETTERS}_letters"
    return read


def load_reads(kraken_reads: Path, gemini_dir: Path) -> Tuple[Dict[str, ReadInfo], Dict[str, Any]]:
    """The read of every image: the Gemini read when it is not NO TEXT, else the Kraken read.

    :param kraken_reads: ``kraken_reads.jsonl`` (``sha256``, ``lines`` of ``{text, ...}``, ``n_lines``).
    :type kraken_reads: Path
    :param gemini_dir: ``gemini_reads_arabic/`` (``index.jsonl`` + one JSON per key with ``text``).
    :type gemini_dir: Path
    :return: ``({sha256: ReadInfo}, counts)``.
    :rtype: Tuple[Dict[str, ReadInfo], Dict[str, Any]]
    """
    reads: Dict[str, ReadInfo] = {}
    kraken_rows = read_jsonl(kraken_reads)
    for row in kraken_rows:
        lines = row.get("lines") or []
        reads[row["sha256"]] = make_read(row["sha256"], "kraken", "\n".join(line["text"] for line in lines), len(lines))
    gemini_total = gemini_text = 0
    for row in read_jsonl(gemini_dir / "index.jsonl"):
        gemini_total += 1
        text = json.loads((gemini_dir / f"{row['key']}.json").read_text(encoding="utf-8")).get("text") or ""
        if NO_TEXT.match(text.strip()) or not text.strip():
            continue
        gemini_text += 1
        reads[row["sha256"]] = make_read(row["sha256"], "gemini", text, None)
    info = {"kraken_reads": len(kraken_rows), "kraken_zero_lines": sum(1 for r in kraken_rows if not r.get("lines")),
            "gemini_reads": gemini_total, "gemini_with_text": gemini_text,
            "gemini_no_text": gemini_total - gemini_text}
    return reads, info


@dataclass
class SideRecord:
    """One cleaned side record.

    :ivar key: ``<canonical_id>#<record index>``.
    :ivar canonical_id: Document.
    :ivar record_index: Position among the document's records (file order).
    :ivar record: The ``cleaned_sections.jsonl`` record.
    :ivar script: Majority script of the text (``""`` without letters).
    :ivar skeleton: Skeleton in that script.
    :ivar grams: n-gram counts of the skeleton per n.
    :ivar order_prior_index: Image index the stored section order points to (cleaner's ``order_prior_image``), or None.
    """
    key: str
    canonical_id: str
    record_index: int
    record: Dict[str, Any]
    script: str
    skeleton: str
    grams: Dict[int, Counter]
    order_prior_index: Optional[int] = None

    @property
    def label(self) -> str:
        """The record's side label (``"1r"``, ``"v"``) or ``""``.

        :return: Label.
        :rtype: str
        """
        return self.record.get("side") or ""


def make_side(record: Dict[str, Any], record_index: int, images: Sequence[ImageInfo]) -> SideRecord:
    """A side record with its skeleton, n-grams and order prior.

    :param record: ``cleaned_sections.jsonl`` record.
    :type record: Dict[str, Any]
    :param record_index: Position among the document's records.
    :type record_index: int
    :param images: The document's images (to map the order prior's URL to an image index).
    :type images: Sequence[ImageInfo]
    :return: The side.
    :rtype: SideRecord
    """
    text = record.get("text") or ""
    skeletons = {script: skeleton_of(text, script) for script in SCRIPTS}
    script = majority_script(skeletons)
    skeleton = skeletons.get(script, "") if script else ""
    prior = record.get("order_prior_image")
    prior_index = None
    if prior is not None and 0 <= prior < len(record.get("images") or []):
        url = record["images"][prior]["url"]
        prior_index = next((im.index for im in images if im.url == url), None)
    return SideRecord(f"{record['canonical_id']}#{record_index}", record["canonical_id"], record_index, record, script,
                      skeleton, {n: ngram_counts(skeleton, n) for n in NGRAMS}, prior_index)


def load_sides(path: Path, images: Mapping[str, Sequence[ImageInfo]]) -> List[SideRecord]:
    """Every side record, in file order.

    :param path: ``cleaned_sections.jsonl``.
    :type path: Path
    :param images: Images per document.
    :type images: Mapping[str, Sequence[ImageInfo]]
    :return: Side records.
    :rtype: List[SideRecord]
    """
    seen: Counter = Counter()
    sides = []
    for record in read_jsonl(path):
        cid = record["canonical_id"]
        sides.append(make_side(record, seen[cid], images.get(cid, [])))
        seen[cid] += 1
    return sides


def merged_pgpids(path: Path, wanted: Set[str]) -> Dict[str, List[str]]:
    """PGP ids of the merged-corpus records of some documents (streamed; the file is large).

    :param path: ``merged_shelfmarks.jsonl``.
    :type path: Path
    :param wanted: Canonical ids.
    :type wanted: Set[str]
    :return: ``{canonical_id: [pgpid, ...]}`` for the wanted ids that have a record.
    :rtype: Dict[str, List[str]]
    """
    out: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            match = CANONICAL_AT_START.match(line)
            if match is not None:
                cid = json.loads(f'"{match.group(1)}"')
                if cid not in wanted:
                    continue
                record = json.loads(line)
            else:
                record = json.loads(line)
                cid = record.get("canonical_id")
                if cid not in wanted:
                    continue
            out[cid] = [str(p) for p in record.get("pgpids") or []]
    return out


# ===================================================================================================================
# Null
# ===================================================================================================================

@dataclass
class NullPool:
    """The reads one script's sides are compared with for their null.

    :ivar script: Script.
    :ivar shas: Image hash of each pool read.
    :ivar owners: Documents whose images have that hash.
    :ivar letters: Skeleton letters of each read.
    :ivar bins: Length bin of each read.
    :ivar index: ``{n: {gram: (read positions, counts)}}``.
    :ivar dropped: Reads of the pool's reader left out, by reason.
    """
    script: str
    shas: List[str]
    owners: List[FrozenSet[str]]
    letters: np.ndarray
    bins: np.ndarray
    index: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]
    dropped: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        """Reads in the pool.

        :return: Count.
        :rtype: int
        """
        return len(self.shas)


def build_null_pool(script: str, reads: Mapping[str, ReadInfo], owners_by_sha: Mapping[str, Set[str]],
                    source: Optional[str] = None) -> NullPool:
    """The null pool of a script: usable reads of that script by its reader, exact duplicate texts once.

    :param script: :data:`HEBREW` or :data:`ARABIC`.
    :type script: str
    :param reads: Reads by image hash.
    :type reads: Mapping[str, ReadInfo]
    :param owners_by_sha: Documents per image hash.
    :type owners_by_sha: Mapping[str, Set[str]]
    :param source: Reader whose reads are pooled; default :data:`POOL_SOURCE`.
    :type source: Optional[str]
    :return: The pool.
    :rtype: NullPool
    """
    source = source or POOL_SOURCE[script]
    kept: List[ReadInfo] = []
    dropped: Counter = Counter()
    seen: Set[str] = set()
    for sha in sorted(reads):
        read = reads[sha]
        if read.source != source:
            continue
        if read.unusable:
            dropped[read.unusable] += 1
        elif read.script != script:
            dropped["other_script"] += 1
        elif read.fingerprint in seen:
            dropped["duplicate_text"] += 1
        else:
            seen.add(read.fingerprint)
            kept.append(read)
    index: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for n in NGRAMS:
        postings: Dict[str, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
        for i, read in enumerate(kept):
            for gram, count in ngram_counts(read.skeletons[script], n).items():
                postings[gram][0].append(i)
                postings[gram][1].append(count)
        index[n] = {g: (np.array(i, dtype=np.int64), np.array(c, dtype=np.float64)) for g, (i, c) in postings.items()}
    letters = np.array([len(r.skeletons[script]) for r in kept], dtype=np.int64)
    return NullPool(script, [r.sha256 for r in kept], [frozenset(owners_by_sha.get(r.sha256, ())) for r in kept],
                    letters, np.array([length_bin(int(x)) for x in letters]), index, dict(dropped))


def pool_recall(side: Counter, index: Mapping[str, Tuple[np.ndarray, np.ndarray]], n_reads: int) -> np.ndarray:
    """:func:`clipped_recall` of one side against every read of a pool at once.

    :param side: The side's n-gram counts.
    :type side: Counter
    :param index: The pool's inverted index for that n.
    :type index: Mapping[str, Tuple[np.ndarray, np.ndarray]]
    :param n_reads: Reads in the pool.
    :type n_reads: int
    :return: Recall per pool read.
    :rtype: np.ndarray
    """
    hits = np.zeros(n_reads)
    total = sum(side.values())
    if not total:
        return hits
    for gram, count in side.items():
        posting = index.get(gram)
        if posting is not None:
            hits[posting[0]] += np.minimum(posting[1], count)
    return hits / total


def threshold_stats(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> Optional[Dict[str, Any]]:
    """p95 / p99 of a null sample, with bootstrap 95 % CIs when ``n_boot`` > 0.

    :param values: Null recalls.
    :type values: np.ndarray
    :param rng: Random generator.
    :type rng: np.random.Generator
    :param n_boot: Bootstrap resamples (0: none).
    :type n_boot: int
    :return: ``{n, p95, p99, mean, std, max[, p95_ci, p99_ci]}``, or None for an empty sample.
    :rtype: Optional[Dict[str, Any]]
    """
    if values.size == 0:
        return None
    p95, p99 = np.percentile(values, [95, 99])
    out: Dict[str, Any] = {"n": int(values.size), "p95": float(p95), "p99": float(p99), "mean": float(values.mean()),
                           "std": float(values.std()), "max": float(values.max())}
    if n_boot:
        boot = np.percentile(values[rng.integers(0, values.size, (n_boot, values.size))], [95, 99], axis=1)
        out["p95_ci"] = [float(x) for x in np.percentile(boot[0], [2.5, 97.5])]
        out["p99_ci"] = [float(x) for x in np.percentile(boot[1], [2.5, 97.5])]
    return out


def side_null(side: SideRecord, pool: NullPool, excluded_docs: Set[str], excluded_shas: Set[str],
              rng: np.random.Generator, n_boot: int = N_BOOT) -> Dict[str, Any]:
    """The null of one side: its recall against every pool read of another document, per length bin.

    :param side: The side.
    :type side: SideRecord
    :param pool: Its script's pool.
    :type pool: NullPool
    :param excluded_docs: Documents whose reads never count (its own, those sharing its text or an image).
    :type excluded_docs: Set[str]
    :param excluded_shas: Image hashes that never count (its own images).
    :type excluded_shas: Set[str]
    :param rng: Random generator (bootstrap).
    :type rng: np.random.Generator
    :param n_boot: Bootstrap resamples for the primary n (the second signal gets none).
    :type n_boot: int
    :return: ``{"reads": k, "n4": {bin: stats}, "n3": {bin: stats}}`` with :data:`ALL_BINS` unstratified.
    :rtype: Dict[str, Any]
    """
    mask = np.array([not (owners & excluded_docs) and sha not in excluded_shas
                     for sha, owners in zip(pool.shas, pool.owners)], dtype=bool)
    out: Dict[str, Any] = {"reads": int(mask.sum())}
    for n in NGRAMS:
        values = pool_recall(side.grams[n], pool.index[n], len(pool))[mask] if len(pool) else np.zeros(0)
        bins = pool.bins[mask] if len(pool) else np.zeros(0, dtype=str)
        boot = n_boot if n == PRIMARY_N else 0
        per = {ALL_BINS: threshold_stats(values, rng, boot)}
        for b in BINS:
            per[b] = threshold_stats(values[bins == b], rng, boot)
        out[f"n{n}"] = per
    return out


def pick_threshold(null: Mapping[str, Any], n: int, bin_name: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """The null statistics for a read of this length bin (the unstratified null when the bin is too small).

    :param null: :func:`side_null` result.
    :type null: Mapping[str, Any]
    :param n: n-gram length.
    :type n: int
    :param bin_name: The image read's length bin.
    :type bin_name: str
    :return: ``(stats or None, the bin used)``.
    :rtype: Tuple[Optional[Dict[str, Any]], str]
    """
    per = null[f"n{n}"]
    stats = per.get(bin_name)
    if stats is not None and stats["n"] >= MIN_BIN_NULL:
        return stats, bin_name
    return per.get(ALL_BINS), ALL_BINS


# ===================================================================================================================
# Pairs, duplicates, ownership, assignment
# ===================================================================================================================

def label_agrees(label: str, token: str) -> Optional[bool]:
    """Whether a side label names the image's file-name token (``"r"`` matches ``"1r"``; ``"1r"`` only ``"1r"``).

    :param label: Side label of the record.
    :type label: str
    :param token: File-name token of the image.
    :type token: str
    :return: None when either is missing, else the agreement.
    :rtype: Optional[bool]
    """
    if not label or not token:
        return None
    if len(label) > 1:
        return token == label
    return token.endswith(label)


def priors_of(side: SideRecord, image: ImageInfo) -> Dict[str, Optional[bool]]:
    """The tie-break priors of placing a side on an image.

    :param side: Side.
    :type side: SideRecord
    :param image: Image.
    :type image: ImageInfo
    :return: ``{"side_label": bool or None, "order_prior": bool or None}``.
    :rtype: Dict[str, Optional[bool]]
    """
    order = None if side.order_prior_index is None else side.order_prior_index == image.index
    return {"side_label": label_agrees(side.label, image.token), "order_prior": order}


def matched_grams(side: SideRecord, pair: Mapping[str, Any]) -> int:
    """How many of the side's :data:`PRIMARY_N`-grams the read matched (clipped), from the pair's recall.

    :param side: Side.
    :type side: SideRecord
    :param pair: A usable :func:`score_pair` result.
    :type pair: Mapping[str, Any]
    :return: Recall times the side's gram count, rounded (the clipped match count is an integer).
    :rtype: int
    """
    return int(round(pair[f"n{PRIMARY_N}"] * max(len(side.skeleton) - (PRIMARY_N - 1), 0)))


def score_pair(side: SideRecord, read: Optional[ReadInfo], null: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recall of a side in one image's read against the side's null.

    :param side: Side.
    :type side: SideRecord
    :param read: The image's read (None: no read).
    :type read: Optional[ReadInfo]
    :param null: :func:`side_null` result (None: no null).
    :type null: Optional[Mapping[str, Any]]
    :return: ``usable`` False with ``why``, or the recalls ``n4`` / ``n3``, the bin, the thresholds of the bin used,
        ``z`` and the clears flags.
    :rtype: Dict[str, Any]
    """
    if read is None:
        return {"usable": False, "why": "no_read"}
    if read.unusable:
        return {"usable": False, "why": read.unusable, "read_source": read.source}
    if not side.script or not side.grams[PRIMARY_N]:
        return {"usable": False, "why": "side_without_grams"}
    if read.script != side.script:
        return {"usable": False, "why": f"read_in_{read.script or 'no'}_script", "read_source": read.source}
    if null is None:
        return {"usable": False, "why": "no_null"}
    skeleton = read.skeletons[side.script]
    bin_name = length_bin(len(skeleton))
    out: Dict[str, Any] = {"usable": True, "read_source": read.source, "read_letters": len(skeleton), "bin": bin_name}
    for n in NGRAMS:
        recall = clipped_recall(side.grams[n], ngram_counts(skeleton, n))
        stats, used = pick_threshold(null, n, bin_name)
        out[f"n{n}"] = recall
        if stats is None:
            return {"usable": False, "why": "empty_null", "read_source": read.source}
        out[f"null_n{n}"] = {"bin_used": used, "n": stats["n"], "p95": stats["p95"], "p99": stats["p99"],
                             "mean": stats["mean"], "std": stats["std"],
                             **({"p95_ci": stats["p95_ci"], "p99_ci": stats["p99_ci"]} if "p99_ci" in stats else {})}
    main = out[f"null_n{PRIMARY_N}"]
    out["z"] = (out[f"n{PRIMARY_N}"] - main["mean"]) / main["std"] if main["std"] > 0 else None
    out["clears_p95"] = out[f"n{PRIMARY_N}"] > main["p95"]
    out["clears_p99"] = out[f"n{PRIMARY_N}"] > main["p99"]
    out["clears_p99_ci_hi"] = out[f"n{PRIMARY_N}"] > main["p99_ci"][1] if "p99_ci" in main else None
    out["n3_clears_p95"] = out["n3"] > out["null_n3"]["p95"]
    return out


def duplicate_versions(sides: Sequence[SideRecord]) -> Dict[str, str]:
    """Duplicate versions inside each document: every version but the one with the most letters.

    A record the cleaner marked ``duplicate_version`` is grouped with the earlier records of its document whose text
    ``near_duplicates`` finds the same (the cleaner's own rule, on its letters-only form).

    :param sides: All side records.
    :type sides: Sequence[SideRecord]
    :return: ``{dropped key: kept key}``.
    :rtype: Dict[str, str]
    """
    by_doc: Dict[str, List[SideRecord]] = defaultdict(list)
    for side in sides:
        by_doc[side.canonical_id].append(side)
    dropped: Dict[str, str] = {}
    for doc_sides in by_doc.values():
        parent = {s.key: s.key for s in doc_sides}

        def find(key: str) -> str:
            """Union-find root.

            :param key: Side key.
            :type key: str
            :return: Root key.
            :rtype: str
            """
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        stripped = {s.key: _strip_letters(s.record.get("text") or "") for s in doc_sides}
        for i, side in enumerate(doc_sides):
            if side.record.get("side_decision") != DUPLICATE_DECISION:
                continue
            for earlier in doc_sides[:i]:
                if near_duplicates(stripped[side.key], stripped[earlier.key]):
                    parent[find(side.key)] = find(earlier.key)
        groups: Dict[str, List[SideRecord]] = defaultdict(list)
        for side in doc_sides:
            groups[find(side.key)].append(side)
        for group in groups.values():
            if len(group) < 2:
                continue
            keep = max(group, key=lambda s: (s.record.get("letters") or 0, -s.record_index))
            for side in group:
                if side is not keep:
                    dropped[side.key] = keep.key
    return dropped


def shared_text_groups(sides: Sequence[SideRecord], skip: Set[str]) -> List[List[SideRecord]]:
    """Groups of records holding the same raw text in different documents.

    :param sides: All side records.
    :type sides: Sequence[SideRecord]
    :param skip: Keys already dropped (duplicate versions).
    :type skip: Set[str]
    :return: One list per group (records with ``shared_text_docs``, keyed by section digest, side, column and text);
        a group may hold one record when its partners have no record here.
    :rtype: List[List[SideRecord]]
    """
    groups: Dict[Tuple[Any, ...], List[SideRecord]] = defaultdict(list)
    for side in sides:
        if side.key in skip or not side.record.get("shared_text_docs"):
            continue
        rec = side.record
        groups[(rec.get("section_digest"), rec.get("side"), rec.get("column"), rec.get("text"))].append(side)
    return list(groups.values())


def decide_ownership(group: Sequence[SideRecord], pairs: Mapping[str, Mapping[int, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Which document of a shared-text group owns the text.

    The copy with the best 4-gram recall clearing p99 owns the text when it is the only copy clearing p99 or its
    recall is at least :data:`MIN_OWNER_MARGIN` times the runner-up's. Equal bests are a ``tie``; a smaller lead makes
    every copy of the group ``contested`` (all go to review); no copy clearing p99 makes every copy ``no_owner``.

    :param group: Records of one shared text (one per document).
    :type group: Sequence[SideRecord]
    :param pairs: ``{side key: {image index: score_pair result}}``.
    :type pairs: Mapping[str, Mapping[int, Dict[str, Any]]]
    :return: Per key: ``role`` (``owner``, ``not_owner``, ``no_owner``, ``tie``, ``contested``), ``best_n4`` (best
        4-gram recall clearing p99, or None), ``owner`` (owner key or None) and ``group_docs``.
    :rtype: Dict[str, Dict[str, Any]]
    """
    best: Dict[str, Optional[float]] = {}
    for side in group:
        clearing = [p[f"n{PRIMARY_N}"] for p in pairs.get(side.key, {}).values() if p.get("usable") and p["clears_p99"]]
        best[side.key] = max(clearing) if clearing else None
    docs = sorted({s.canonical_id for s in group})
    scored = sorted(((v, k) for k, v in best.items() if v is not None), reverse=True)
    out: Dict[str, Dict[str, Any]] = {}
    if not scored:
        for side in group:
            out[side.key] = {"role": "no_owner", "best_n4": None, "owner": None, "group_docs": docs}
        return out
    top = scored[0][0]
    tied = [k for v, k in scored if abs(v - top) <= 1e-12]
    contested = len(tied) == 1 and len(scored) > 1 and not top >= MIN_OWNER_MARGIN * scored[1][0]
    owner = tied[0] if len(tied) == 1 and not contested else None
    for side in group:
        if contested:
            role = "contested"
        elif side.key == owner:
            role = "owner"
        else:
            role = "tie" if side.key in tied and owner is None else "not_owner"
        out[side.key] = {"role": role, "best_n4": best[side.key], "owner": owner, "group_docs": docs}
    return out


def distinct_images(images: Sequence[ImageInfo]) -> Tuple[List[ImageInfo], Dict[int, int]]:
    """The document's images with byte-identical copies folded onto the first.

    :param images: Images of one document.
    :type images: Sequence[ImageInfo]
    :return: ``(distinct images, {index of a copy: index kept})``.
    :rtype: Tuple[List[ImageInfo], Dict[int, int]]
    """
    kept: Dict[str, ImageInfo] = {}
    copies: Dict[int, int] = {}
    for image in images:
        if image.sha256 in kept:
            copies[image.index] = kept[image.sha256].index
        else:
            kept[image.sha256] = image
    return list(kept.values()), copies


def decide_document(sides: Sequence[SideRecord], images: Sequence[ImageInfo],
                    pairs: Mapping[str, Mapping[int, Dict[str, Any]]],
                    blocked: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Place the sides of one document on its images.

    :param sides: The document's records that still compete for its images (duplicate versions and shared texts
        owned by another document already removed).
    :type sides: Sequence[SideRecord]
    :param images: The document's distinct images.
    :type images: Sequence[ImageInfo]
    :param pairs: ``{side key: {image index: score_pair result}}``.
    :type pairs: Mapping[str, Mapping[int, Dict[str, Any]]]
    :param blocked: Sides that compete for images but may not be accepted (shared text without an owner), with their
        ``needs_review`` decision.
    :type blocked: Mapping[str, Dict[str, Any]]
    :return: Per side key: ``outcome`` (``accepted`` / ``needs_review`` / ``unassigned``), ``reasons``, ``image_index``
        (assigned image or None), ``basis`` (``p99`` or ``p95+side_label``), ``matched_n4`` (matched 4-grams on the
        image, when a basis holds), ``conflicts`` (side keys clearing p95 on the image), ``other_images`` (images where
        the side also clears p95).
    :rtype: Dict[str, Dict[str, Any]]
    """
    def pair(side: SideRecord, image: ImageInfo) -> Dict[str, Any]:
        """Pair score (unusable when missing).

        :param side: Side.
        :type side: SideRecord
        :param image: Image.
        :type image: ImageInfo
        :return: The score.
        :rtype: Dict[str, Any]
        """
        return pairs.get(side.key, {}).get(image.index) or {"usable": False, "why": "no_read"}

    def clears95(side: SideRecord, image: ImageInfo) -> bool:
        """Whether a side clears its p95 on an image.

        :param side: Side.
        :type side: SideRecord
        :param image: Image.
        :type image: ImageInfo
        :return: True when the pair is usable and clears p95.
        :rtype: bool
        """
        p = pair(side, image)
        return bool(p.get("usable") and p["clears_p95"])

    out: Dict[str, Dict[str, Any]] = {}
    candidates = [s for s in sides if s.key not in blocked]
    weights = np.zeros((len(candidates), len(images)))
    for r, side in enumerate(candidates):
        for c, image in enumerate(images):
            p = pair(side, image)
            if p.get("usable"):
                prior = priors_of(side, image)
                weights[r, c] = p[f"n{PRIMARY_N}"] + sum(PRIOR_EPS[k] for k, v in prior.items() if v)
    assigned: Dict[str, ImageInfo] = {}
    if weights.size:
        rows, cols = linear_sum_assignment(weights, maximize=True)
        assigned = {candidates[r].key: images[c] for r, c in zip(rows.tolist(), cols.tolist())}
    taken = {image.index: key for key, image in assigned.items()}
    for side in sides:
        if side.key in blocked:
            out[side.key] = dict(blocked[side.key])
            out[side.key].setdefault("image_index", None)
            continue
        usable = [im for im in images if pair(side, im).get("usable")]
        cleared = [im for im in usable if clears95(side, im)]
        image = assigned.get(side.key)
        decision: Dict[str, Any] = {"outcome": "", "reasons": [], "image_index": image.index if image else None,
                                    "basis": None, "conflicts": [], "other_images": []}
        out[side.key] = decision
        if not side.script or not side.grams[PRIMARY_N]:
            decision.update(outcome="unassigned", reasons=["side_without_grams"], image_index=None)
            continue
        if not usable:
            decision.update(outcome="unassigned", reasons=["no_usable_read"], image_index=None)
            continue
        p = pair(side, image) if image is not None else {"usable": False}
        prior = priors_of(side, image) if image is not None else {}
        basis = None
        if p.get("usable") and p["clears_p99"]:
            basis = "p99"
        elif p.get("usable") and p["clears_p95"] and prior.get("side_label"):
            basis = "p95+side_label"
        if basis is None:
            if not cleared:
                decision.update(outcome="unassigned", reasons=["not_found"], image_index=None)
                continue
            reasons = []
            if image is not None and image in cleared:
                reasons.append("clears_p95_not_p99_without_agreeing_prior")
            for other in cleared:
                if image is not None and other.index == image.index:
                    continue
                reasons.append("clears_p95_on_image_assigned_to_another_side" if other.index in taken
                               else "clears_p95_on_unassigned_image")
            decision.update(outcome="needs_review", reasons=sorted(set(reasons)),
                            other_images=[im.index for im in cleared])
            continue
        decision["basis"] = basis
        decision["matched_n4"] = matched_grams(side, p)
        conflicts = [o.key for o in sides if o.key != side.key and clears95(o, image)]
        others = [im.index for im in cleared if im.index != image.index]
        reasons = []
        if conflicts:
            reasons.append("other_side_clears_p95_on_image")
        if others:
            reasons.append("side_clears_p95_on_another_image")
        if not p["n3_clears_p95"]:
            reasons.append("n3_below_p95")
        if decision["matched_n4"] < MIN_MATCHED_GRAMS:
            reasons.append(UNDER_MATCHED_REASON)
        decision.update(outcome="needs_review" if reasons else "accepted", reasons=reasons, conflicts=conflicts,
                        other_images=others)
    return out


def shared_image_conflicts(decisions: Mapping[str, Dict[str, Any]], sides: Mapping[str, SideRecord],
                           images: Mapping[str, Sequence[ImageInfo]]) -> List[str]:
    """Send accepted pairs to review when one image is accepted for sides of different documents.

    :param decisions: Side decisions (modified in place).
    :type decisions: Mapping[str, Dict[str, Any]]
    :param sides: Sides by key.
    :type sides: Mapping[str, SideRecord]
    :param images: Images per document.
    :type images: Mapping[str, Sequence[ImageInfo]]
    :return: Keys moved to review.
    :rtype: List[str]
    """
    by_sha: Dict[str, List[str]] = defaultdict(list)
    for key, decision in decisions.items():
        if decision["outcome"] == "accepted":
            cid = sides[key].canonical_id
            image = next(im for im in images[cid] if im.index == decision["image_index"])
            by_sha[image.sha256].append(key)
    moved = []
    for keys in by_sha.values():
        if len({sides[k].canonical_id for k in keys}) > 1:
            for key in keys:
                decisions[key].update(outcome="needs_review",
                                      reasons=decisions[key]["reasons"] + ["image_accepted_for_another_document"])
                moved.append(key)
    return moved


def dominance_ratio(key: str, decision: Mapping[str, Any], pairs: Mapping[str, Mapping[int, Dict[str, Any]]]) -> Optional[float]:
    """How strongly a side held in review only by a conflict or a spill dominates the competing evidence.

    The ratio is the largest of: the side's 4-gram recall on each other image where it clears p95, divided by its
    recall on its assigned image (spill); and each competing side's recall on the assigned image divided by that
    side's best recall on any image of the document (conflict). Small ratios mean the competing evidence looks like
    the background the two sides of one document share (names, formulae, the same hand), large ones a real conflict.

    :param key: Side key.
    :type key: str
    :param decision: The side's decision (``needs_review`` with ``image_index``, ``other_images``, ``conflicts``).
    :type decision: Mapping[str, Any]
    :param pairs: ``{side key: {image index: score}}``.
    :type pairs: Mapping[str, Mapping[int, Dict[str, Any]]]
    :return: The ratio, or None when the decision has another reason or no assigned image.
    :rtype: Optional[float]
    """
    image = decision.get("image_index")
    reasons = set(decision.get("reasons") or [])
    if image is None or not reasons or not reasons <= set(RESOLVABLE_REASONS):
        return None
    own = pairs[key][image].get("n4") or 0.0
    if own <= 0:
        return None
    ratios = [pairs[key][j]["n4"] / own for j in decision.get("other_images") or []]
    for other in decision.get("conflicts") or []:
        best = max((p["n4"] for p in pairs[other].values() if p.get("usable")), default=0.0)
        ratios.append(pairs[other][image]["n4"] / best if best > 0 else 1.0)
    return max(ratios) if ratios else None


def triage_band(ratio: Optional[float]) -> Optional[str]:
    """Triage band of a dominance ratio.

    :param ratio: :func:`dominance_ratio` result.
    :type ratio: Optional[float]
    :return: A :data:`TRIAGE_BANDS` name, or None without a ratio.
    :rtype: Optional[str]
    """
    if ratio is None:
        return None
    return next(name for bound, name in TRIAGE_BANDS if ratio <= bound)


# ===================================================================================================================
# Coverage
# ===================================================================================================================

def coverage(side: SideRecord, read: ReadInfo, pair: Mapping[str, Any], other_images_unread: bool,
             has_copies: bool) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Coverage flags and exclusions of an accepted pair.

    :param side: Side.
    :type side: SideRecord
    :param read: The image's read.
    :type read: ReadInfo
    :param pair: :func:`score_pair` result.
    :type pair: Mapping[str, Any]
    :param other_images_unread: Another distinct image of the document has no usable read in the side's script.
    :type other_images_unread: bool
    :param has_copies: The document has byte-identical images.
    :type has_copies: bool
    :return: ``(flags, exclusions, info)``; info has ``letter_ratio`` and ``other_script_share``. A letter ratio
        outside :data:`LETTER_RATIO_RANGE` is a flag; one over :data:`MAX_READ_OVER_TARGET` is also an exclusion.
    :rtype: Tuple[List[str], List[str], Dict[str, Any]]
    """
    record = side.record
    flags: List[str] = []
    exclusions: List[str] = []
    target_letters = len(side.skeleton)
    ratio = pair["read_letters"] / target_letters if target_letters else None
    other = ARABIC if side.script == HEBREW else HEBREW
    own_letters, other_letters = len(read.skeletons[side.script]), len(read.skeletons[other])
    other_share = other_letters / (own_letters + other_letters) if own_letters + other_letters else 0.0
    if ratio is None or not LETTER_RATIO_RANGE[0] <= ratio <= LETTER_RATIO_RANGE[1]:
        flags.append("letter_ratio_out_of_range")
    if read.source == "kraken" and not read.n_lines:
        flags.append("kraken_zero_line_image")
    if other_share > MAX_OTHER_SCRIPT_SHARE:
        flags.append("read_other_script_over_10pct")
    if other_images_unread:
        flags.append("other_image_without_usable_read")
    if has_copies:
        flags.append("byte_identical_images_in_document")
    if record.get("family") in EXCLUDED_FAMILIES:
        exclusions.append(record["family"])
    for flag in EXCLUDED_FLAGS:
        if (record.get("flags") or {}).get(flag):
            exclusions.append(flag)
    if (record.get("gap_density") or 0.0) > MAX_GAP_DENSITY:
        exclusions.append("gap_density_over_30pct")
    if (record.get("letters") or 0) < MIN_TARGET_LETTERS:
        exclusions.append(f"under_{MIN_TARGET_LETTERS}_letters")
    if ratio is not None and ratio > MAX_READ_OVER_TARGET:
        exclusions.append(f"read_exceeds_target_{MAX_READ_OVER_TARGET:g}x")
    return flags, exclusions, {"letter_ratio": ratio, "target_letters": target_letters, "other_script_share": other_share}


# ===================================================================================================================
# Leakage
# ===================================================================================================================

def hebrew_shingles(text: str, n: int = HEBREW_SHINGLE) -> Set[str]:
    """Hebrew letter shingles of a text, as ``DecontamGate.content_contaminated`` slides them.

    :param text: Text.
    :type text: str
    :param n: Shingle length.
    :type n: int
    :return: Distinct shingles.
    :rtype: Set[str]
    """
    letters = "".join(_HEB.findall(text or ""))
    return {letters[i:i + n] for i in range(len(letters) - n + 1)}


def hebrew_hits(text: str, shingles: Set[str], n: int = HEBREW_SHINGLE) -> int:
    """Shingle positions of a text found in a shingle set (the count ``content_contaminated`` compares with 2).

    :param text: Text.
    :type text: str
    :param shingles: GT shingles.
    :type shingles: Set[str]
    :param n: Shingle length.
    :type n: int
    :return: Hits.
    :rtype: int
    """
    letters = "".join(_HEB.findall(text or ""))
    return sum(letters[i:i + n] in shingles for i in range(len(letters) - n + 1))


def arabic_shingles(text: str, n: int = ARABIC_SHINGLE) -> Set[str]:
    """Rasm shingles of a text (spaces and non-rasm characters dropped, step 1).

    :param text: Text.
    :type text: str
    :param n: Shingle length.
    :type n: int
    :return: Distinct shingles.
    :rtype: Set[str]
    """
    rasm = rasm_skeleton(text or "")
    return {rasm[i:i + n] for i in range(len(rasm) - n + 1)}


@dataclass
class GtIndex:
    """Shingles of every GT set.

    :ivar hebrew: GT set -> its 25-letter Hebrew shingles.
    :ivar arabic: GT set -> rasm shingle -> GT documents holding it.
    :ivar docs: GT set -> document labels.
    """
    hebrew: Dict[str, Set[str]]
    arabic: Dict[str, Dict[str, Set[int]]]
    docs: Dict[str, List[str]]


def build_gt_index(gt_sets: Mapping[str, Sequence[Tuple[str, str]]]) -> GtIndex:
    """Index the GT texts.

    :param gt_sets: ``{set: [(document label, text)]}``.
    :type gt_sets: Mapping[str, Sequence[Tuple[str, str]]]
    :return: The index.
    :rtype: GtIndex
    """
    hebrew: Dict[str, Set[str]] = {}
    arabic: Dict[str, Dict[str, Set[int]]] = {}
    docs: Dict[str, List[str]] = {}
    for name, items in gt_sets.items():
        hebrew[name] = set()
        arabic[name] = defaultdict(set)
        docs[name] = []
        for i, (label, text) in enumerate(items):
            docs[name].append(label)
            hebrew[name] |= hebrew_shingles(text)
            for shingle in arabic_shingles(text):
                arabic[name][shingle].add(i)
    return GtIndex(hebrew, {k: dict(v) for k, v in arabic.items()}, docs)


def text_leakage(text: str, gt: GtIndex, contaminated: Callable[[str, set], bool]) -> Dict[str, Any]:
    """Text rule: which GT sets a target shares shingles with.

    :param text: Target text.
    :type text: str
    :param gt: GT index.
    :type gt: GtIndex
    :param contaminated: ``DecontamGate.content_contaminated`` (the Hebrew decision).
    :type contaminated: Callable[[str, set], bool]
    :return: ``{"hebrew": {set: hits}, "arabic": {set: {max_one_doc, doc, share_union}}, "reasons": [...]}`` listing
        only the sets that hit.
    :rtype: Dict[str, Any]
    """
    out: Dict[str, Any] = {"hebrew": {}, "arabic": {}, "reasons": []}
    for name, shingles in gt.hebrew.items():
        if contaminated(text, shingles):
            out["hebrew"][name] = hebrew_hits(text, shingles)
            out["reasons"].append(f"text:{name}:hebrew_{HEBREW_SHINGLE}_letter_shingles")
    mine = arabic_shingles(text)
    if mine:
        for name, index in gt.arabic.items():
            per_doc: Counter = Counter()
            union = 0
            for shingle in mine:
                holders = index.get(shingle)
                if holders:
                    union += 1
                    per_doc.update(holders)
            top = per_doc.most_common(1)
            max_one = top[0][1] if top else 0
            share = union / len(mine)
            if max_one >= ARABIC_MIN_SHARED_ONE_DOC or share >= ARABIC_MIN_SHARE_UNION:
                out["arabic"][name] = {"max_one_doc": max_one, "doc": gt.docs[name][top[0][0]] if top else None,
                                       "share_union": share}
                out["reasons"].append(f"text:{name}:rasm_{ARABIC_SHINGLE}_shingles")
    return out


def identity_leakage(canonical_id: str, shelf_mark: str, pgpids: Sequence[str], held: d25.HeldOut,
                     pgp_index: Mapping[str, Set[str]], gate: DecontamGate) -> Dict[str, Any]:
    """Identity rule of one document.

    :param canonical_id: Canonical id.
    :type canonical_id: str
    :param shelf_mark: Shelf mark as recorded (joins split on ``" + "``).
    :type shelf_mark: str
    :param pgpids: PGP ids of the merged-corpus record.
    :type pgpids: Sequence[str]
    :param held: ``derive_v25a.held_out_sets`` result.
    :type held: d25.HeldOut
    :param pgp_index: ``derive_v25a.pgp_shelfmark_index`` of the pinned export.
    :type pgp_index: Mapping[str, Set[str]]
    :param gate: Decontamination gate (``clean_keys``, ``bench_keys``, ``loose_key``).
    :type gate: DecontamGate
    :return: ``{pgpids, keys, derive, clean_v2_keys, benchmark_keys, reasons}``.
    :rtype: Dict[str, Any]
    """
    source = d25.PoolSource(LABEL_SOURCE, "legacy_", 0, f"val/{LABEL_SOURCE}")
    record = {"stem": canonical_id, "pgpids": list(pgpids), "shelfmark": shelf_mark, "canonical_id": canonical_id,
              "doc_id": canonical_id}
    ref = d25.doc_ref(source, record, pgp_index)
    derive = d25.held_out_reason(ref, held)
    parts = [canonical_id] + [p.strip() for p in (shelf_mark or "").split(" + ") if p.strip()]
    loose = {gate.loose_key(p) for p in parts} - {None, ""}
    clean = sorted(loose & gate.clean_keys)
    bench = sorted(loose & gate.bench_keys)
    reasons = ([f"identity:derive_v25a:{derive}"] if derive else []) + \
              ([f"identity:clean_v2_key:{','.join(clean)}"] if clean else []) + \
              ([f"identity:benchmark_key:{','.join(bench)}"] if bench else [])
    return {"pgpids": sorted(ref.pgpids), "keys": sorted(ref.keys), "derive": derive, "clean_v2_keys": clean,
            "benchmark_keys": bench, "reasons": reasons}


def footnote_editions(pgp_export: Path, pgpids: Set[str]) -> List[Tuple[str, str]]:
    """Edition footnote texts of some PGP documents from the pinned export.

    :param pgp_export: Export directory (``footnotes.csv``).
    :type pgp_export: Path
    :param pgpids: Documents wanted.
    :type pgpids: Set[str]
    :return: ``[(label, content)]`` for footnotes whose relation names an Edition and whose content is not empty.
    :rtype: List[Tuple[str, str]]
    """
    csv.field_size_limit(1 << 30)
    out = []
    with open(pgp_export / "footnotes.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            pid = str(row["document_id"]).strip()
            if pid in pgpids and EDITION_RELATION.search(row.get("doc_relation") or "") and (row.get("content") or "").strip():
                out.append((f"pgpid {pid} ({row.get('source_slug') or ''})", row["content"]))
    return out


def saved_split_texts(split_dir: Path) -> List[Tuple[str, str]]:
    """``(stem, answer)`` of every row of a saved split (streamed with sequential reads).

    :param split_dir: ``<dataset>/<split>`` of a ``save_to_disk`` directory.
    :type split_dir: Path
    :return: Rows.
    :rtype: List[Tuple[str, str]]
    """
    from src.finetuning.qwen_hebrew.images_once import iter_saved_batches

    out = []
    for batch in iter_saved_batches(split_dir):
        out.extend(zip(batch.column("stem").to_pylist(), batch.column("answer").to_pylist()))
    return out


def load_gt_sets(eval_dir: Path = EVAL_DIR, pgp_export: Path = PGP_EXPORT, arabic_v0: Path = ARABIC_V0,
                 clean_v2_val: Path = CLEAN_V2_VAL) -> Dict[str, List[Tuple[str, str]]]:
    """GT texts of the leakage text rule (read only).

    :param eval_dir: ``raw_data/cairo_genizah/evaluations``.
    :type eval_dir: Path
    :param pgp_export: Pinned PGP export.
    :type pgp_export: Path
    :param arabic_v0: Arabic-script benchmark v0 ``benchmark.jsonl``.
    :type arabic_v0: Path
    :param clean_v2_val: Saved val split of clean v2.
    :type clean_v2_val: Path
    :return: ``{set name (GT_SETS): [(document label, text)]}``.
    :rtype: Dict[str, List[Tuple[str, str]]]
    """
    def docs_of(path: Path) -> List[Tuple[str, str]]:
        """``(doc_id, gt)`` of a benchmark JSON.

        :param path: Benchmark JSON with ``docs``.
        :type path: Path
        :return: Documents.
        :rtype: List[Tuple[str, str]]
        """
        return [(str(d.get("doc_id")), str(d.get("gt") or "")) for d in json.loads(path.read_text(encoding="utf-8"))["docs"]]

    _, two_edition = br.registered_benchmark_documents(["two_edition_documents_v0"])
    arabic = [(r["id"], "\n".join(line for sec in r["gt"]["sections"] for line in sec["lines"])) for r in read_jsonl(arabic_v0)]
    return {"genizah_test_v1": docs_of(eval_dir / "genizah_test_v1" / "genizah_test_v1.json"),
            "genizah_religious_v1": docs_of(eval_dir / "genizah_religious_v1" / "genizah_religious_v1.json"),
            "sealed_v1_documentary": docs_of(eval_dir / "sealed_v1" / "documentary" / "sealed_v1_documentary.json"),
            "sealed_v1_literary": docs_of(eval_dir / "sealed_v1" / "literary" / "sealed_v1_literary.json"),
            "two_edition_footnotes": footnote_editions(pgp_export, {str(p) for p in two_edition}),
            "arabic_script_benchmark_v0": arabic,
            "clean_v2_val": saved_split_texts(clean_v2_val)}


@dataclass
class LeakageResources:
    """Everything the leakage scan reads.

    :ivar held: ``derive_v25a.held_out_sets`` result.
    :ivar pgp_index: ``derive_v25a.pgp_shelfmark_index`` of the pinned export.
    :ivar gate: Decontamination gate (inventories and the Hebrew content rule).
    :ivar gt_sets: GT texts per set (:func:`load_gt_sets`).
    :ivar pgpids: PGP ids of the merged-corpus record per document.
    """
    held: d25.HeldOut
    pgp_index: Dict[str, Set[str]]
    gate: DecontamGate
    gt_sets: Dict[str, List[Tuple[str, str]]]
    pgpids: Dict[str, List[str]]


def load_leakage_resources(doc_ids: Set[str]) -> LeakageResources:
    """The real registries, inventories, GT texts and merged-corpus PGP ids (read only).

    :param doc_ids: Documents in scope.
    :type doc_ids: Set[str]
    :return: Resources.
    :rtype: LeakageResources
    """
    documents_csv = PGP_EXPORT / "documents.csv"
    return LeakageResources(d25.held_out_sets(None, documents_csv), d25.pgp_shelfmark_index(documents_csv),
                            DecontamGate(), load_gt_sets(), merged_pgpids(MERGED_JSONL, doc_ids))


# ===================================================================================================================
# Decisions over the whole set
# ===================================================================================================================

@dataclass
class Decisions:
    """Everything the matching decided.

    :ivar sides: Side records by key.
    :ivar nulls: Side nulls by key.
    :ivar pairs: ``{side key: {image index: score}}``.
    :ivar decisions: ``{side key: decision}``.
    :ivar duplicates: ``{dropped key: kept key}``.
    :ivar ownership: ``{side key: ownership}`` for shared-text records.
    :ivar pools: Null pools by script.
    """
    sides: Dict[str, SideRecord]
    nulls: Dict[str, Dict[str, Any]]
    pairs: Dict[str, Dict[int, Dict[str, Any]]]
    decisions: Dict[str, Dict[str, Any]]
    duplicates: Dict[str, str]
    ownership: Dict[str, Dict[str, Any]]
    pools: Dict[str, NullPool]


def decide_all(sides: Sequence[SideRecord], images: Mapping[str, Sequence[ImageInfo]], reads: Mapping[str, ReadInfo],
               rng: np.random.Generator, n_boot: int = N_BOOT) -> Decisions:
    """Nulls, pair scores, duplicate versions, shared-text ownership and the per-document assignment.

    :param sides: Every side record.
    :type sides: Sequence[SideRecord]
    :param images: Images per document.
    :type images: Mapping[str, Sequence[ImageInfo]]
    :param reads: Reads by image hash.
    :type reads: Mapping[str, ReadInfo]
    :param rng: Random generator (bootstrap).
    :type rng: np.random.Generator
    :param n_boot: Bootstrap resamples.
    :type n_boot: int
    :return: The decisions.
    :rtype: Decisions
    """
    owners_by_sha: Dict[str, Set[str]] = defaultdict(set)
    for cid, doc_images in images.items():
        for image in doc_images:
            owners_by_sha[image.sha256].add(cid)
    pools = {script: build_null_pool(script, reads, owners_by_sha) for script in SCRIPTS}
    by_key = {s.key: s for s in sides}
    duplicates = duplicate_versions(sides)
    nulls: Dict[str, Dict[str, Any]] = {}
    pairs: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for side in sides:
        if side.key in duplicates:
            continue
        doc_images = images.get(side.canonical_id, [])
        own_shas = {im.sha256 for im in doc_images}
        excluded = {side.canonical_id} | set(side.record.get("shared_text_docs") or [])
        excluded |= {d for sha in own_shas for d in owners_by_sha.get(sha, ())}
        null = None
        if side.script and side.grams[PRIMARY_N]:
            null = side_null(side, pools[side.script], excluded, own_shas, rng, n_boot)
            nulls[side.key] = null
        pairs[side.key] = {im.index: score_pair(side, reads.get(im.sha256), null) for im in doc_images}
    ownership: Dict[str, Dict[str, Any]] = {}
    for group in shared_text_groups(sides, set(duplicates)):
        ownership.update(decide_ownership(group, pairs))
    decisions: Dict[str, Dict[str, Any]] = {}
    for key, kept in duplicates.items():
        decisions[key] = {"outcome": "dropped", "reasons": ["duplicate_version_shorter"], "image_index": None,
                          "kept": kept}
    for key, own in ownership.items():
        if own["role"] == "not_owner":
            decisions[key] = {"outcome": "dropped", "reasons": ["shared_text_not_owner"], "image_index": None,
                              "owner": own["owner"]}
    by_doc: Dict[str, List[SideRecord]] = defaultdict(list)
    for side in sides:
        if side.key not in decisions:
            by_doc[side.canonical_id].append(side)
    for cid, doc_sides in by_doc.items():
        blocked = {s.key: {"outcome": "needs_review", "reasons": [OWNERSHIP_REVIEW_REASONS[ownership[s.key]["role"]]],
                           "image_index": None, "basis": None, "conflicts": [], "other_images": []}
                   for s in doc_sides if ownership.get(s.key, {}).get("role") in OWNERSHIP_REVIEW_REASONS}
        distinct, _ = distinct_images(images.get(cid, []))
        decisions.update(decide_document(doc_sides, distinct, pairs, blocked))
    shared_image_conflicts(decisions, by_key, images)
    for key, decision in decisions.items():
        if decision["outcome"] == "needs_review":
            decision["dominance_ratio"] = dominance_ratio(key, decision, pairs)
            decision["triage"] = triage_band(decision["dominance_ratio"])
    return Decisions(by_key, nulls, pairs, decisions, duplicates, ownership, pools)


# ===================================================================================================================
# Rows
# ===================================================================================================================

def safe_token(text: str) -> str:
    """A stem-safe token.

    :param text: Any id.
    :type text: str
    :return: Letters, digits and underscores only.
    :rtype: str
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_") or "x"


def language_group(main_language: Optional[str]) -> str:
    """Language group of a record's ``main_language``.

    :param main_language: As recorded (``judeo-arabic``, ``arabic, hebrew``, None ...).
    :type main_language: Optional[str]
    :return: ``unknown``, ``mixed``, ``judaeo-arabic`` or the language.
    :rtype: str
    """
    value = (main_language or "").strip().lower()
    if not value:
        return "unknown"
    if "," in value:
        return "mixed"
    if value in ("judeo-arabic", "judaeo-arabic"):
        return "judaeo-arabic"
    return value


def row_stem(side: SideRecord, image: ImageInfo) -> Tuple[str, str]:
    """The side name and stem of a row. The image's file-name token names the side (the record's label only when the
    image has no token), so the stem names the file the row sits on; the label stays in the manifest's ``side_label``.

    :param side: Side.
    :type side: SideRecord
    :param image: The row's image.
    :type image: ImageInfo
    :return: ``(side name, stem)``.
    :rtype: Tuple[str, str]
    """
    side_name = image.token or side.label or "none"
    return side_name, f"legacy_{safe_token(side.canonical_id)}_{safe_token(side_name)}_c{image.index}_page"


def page_row(image: ImageInfo, text: str, stem: str) -> Dict[str, Any]:
    """One page-transcription row in the KTIV ``FEATURES`` schema.

    :param image: Page image.
    :type image: ImageInfo
    :param text: Target (the cleaned side text).
    :type text: str
    :param stem: Row id.
    :type stem: str
    :return: Feature dict.
    :rtype: Dict[str, Any]
    """
    return {"image": image.path, "question": PROMPT, "answer": text, "task": TASK, "section": SECTION, "stem": stem,
            "label_source": LABEL_SOURCE, "target_chars": len(text), "target_tokens": 0, "image_width": image.width,
            "image_height": image.height}


def pair_summary(pair: Mapping[str, Any]) -> Dict[str, Any]:
    """The numbers of a pair score for manifests and review queues.

    :param pair: :func:`score_pair` result.
    :type pair: Mapping[str, Any]
    :return: Flat summary.
    :rtype: Dict[str, Any]
    """
    if not pair.get("usable"):
        return {"usable": False, "why": pair.get("why")}
    n4, n3 = pair[f"null_n{PRIMARY_N}"], pair["null_n3"]
    return {"usable": True, "read_source": pair["read_source"], "read_letters": pair["read_letters"],
            "bin": pair["bin"], "recall_n4": pair["n4"], "recall_n3": pair["n3"], "null_bin_used": n4["bin_used"],
            "null_n": n4["n"], "null_p95_n4": n4["p95"], "null_p99_n4": n4["p99"],
            "null_p95_ci_n4": n4.get("p95_ci"), "null_p99_ci_n4": n4.get("p99_ci"), "null_p95_n3": n3["p95"],
            "null_p99_n3": n3["p99"], "z": pair["z"], "clears_p95": pair["clears_p95"],
            "clears_p99": pair["clears_p99"], "clears_p99_ci_hi": pair["clears_p99_ci_hi"],
            "n3_clears_p95": pair["n3_clears_p95"]}


def check_out_dir(out: Path, source: Path) -> None:
    """Refuse an output that would land in or over the inputs.

    :param out: Output directory.
    :type out: Path
    :param source: Input directory.
    :type source: Path
    :raises ValueError: When ``out`` is the source, inside it or contains it, or is not a ``legacy_store_diplomatic_v*``
        directory.
    """
    o, s = out.resolve(), source.resolve()
    if o == s or s in o.parents or o in s.parents:
        raise ValueError(f"output {out} overlaps the input directory {source}")
    if not re.match(r"^legacy_store_diplomatic_v\d+", out.name):
        raise ValueError(f"output {out} is not a legacy_store_diplomatic_v<N> directory")


def readme_text(stats: Mapping[str, Any]) -> str:
    """README of the dataset (PRIVATE notice first).

    :param stats: Build statistics.
    :type stats: Mapping[str, Any]
    :return: Markdown.
    :rtype: str
    """
    rows = stats["rows"]
    return f"""# {stats['name']}

**PRIVATE. Never publish, push or release this dataset, its images-once export or anything derived from them.**
The images and the transcriptions come from the project's legacy image store.

**NO TRAINING USE until the user has reviewed the build report** (written with `--report`) and the eye queue of
sides sent to review (`--eye-queue`). Rows carry coverage flags (`coverage_flags` in `manifest.jsonl`) that the
review may turn into exclusions.

Built by `src/finetuning/qwen_hebrew/build_legacy_store_rows.py` on {stats['built']} from the cleaned side records of
the legacy store's diplomatic transcriptions; each side was placed on an image by matching its letter skeleton against
cached Kraken / Gemini reads of the images, accepted only above a per-side null (p99 of its recall against other
documents' reads, by read length). Prompt: `{PROMPT_NAME}` (script neutral).

- rows: train_page {rows.get('train_page', 0)}, val {rows.get('val', 0)} ({stats['row_documents']} documents, {stats['row_letters']} letters)
- side records: {stats['sides']['records']}; accepted pairs: {stats['sides']['outcomes'].get('accepted', 0)};
  needs review: {stats['sides']['outcomes'].get('needs_review', 0)}; unassigned: {stats['sides']['outcomes'].get('unassigned', 0)};
  dropped: {stats['sides']['outcomes'].get('dropped', 0)}
- coverage exclusions: {stats['coverage']['excluded_pairs']}; leakage hold-outs: {stats['leakage']['held_out_pairs']}

`manifest.jsonl` has one record per row (match numbers, null thresholds, flags, leakage status); `decisions.jsonl`
has the decision for every side record; `stats.json` has the funnel.
"""


def build(source: Path = SOURCE_DIR, out: Path = DEFAULT_OUT, docs_list: Optional[Path] = None,
          report: Optional[Path] = None, eye_queue: Optional[Path] = None, export: bool = True,
          n_boot: int = N_BOOT, seed: int = SEED, leakage: Optional[LeakageResources] = None) -> Dict[str, Any]:
    """Build the dataset (and its images-once export, the report and the eye queue).

    :param source: Input directory (cleaned sections, images manifest, cached reads).
    :type source: Path
    :param out: Output directory of the saved DatasetDict (a new ``legacy_store_diplomatic_v<N>``).
    :type out: Path
    :param docs_list: JSON list of the documents considered (``canonical_id``, ``image_urls``), only for the scope
        counts (documents with more than four images are out of scope); None: the images manifest is the scope.
    :type docs_list: Optional[Path]
    :param report: Where to write the build report (Markdown); None: not written.
    :type report: Optional[Path]
    :param eye_queue: Where to write the review queue (JSONL); None: not written.
    :type eye_queue: Optional[Path]
    :param export: Also write ``<out>_images_once``.
    :type export: bool
    :param n_boot: Bootstrap resamples of the null thresholds.
    :type n_boot: int
    :param seed: Seed of the bootstrap and the spot-check draw.
    :type seed: int
    :param leakage: Leakage resources; None loads the real ones (:func:`load_leakage_resources`).
    :type leakage: Optional[LeakageResources]
    :return: Statistics (also ``stats.json``).
    :rtype: Dict[str, Any]
    """
    from src.finetuning.qwen_hebrew.build_pgp_editions import save_dataset
    from src.finetuning.qwen_hebrew.images_once import export_images_once

    check_out_dir(out, source)
    images = load_images(source / "images_manifest.jsonl")
    reads, read_info = load_reads(source / "kraken_reads.jsonl", source / "gemini_reads_arabic")
    sides = load_sides(source / "cleaned_sections.jsonl", images)
    unknown = sorted({s.canonical_id for s in sides} - set(images))
    if unknown:
        raise ValueError(f"{len(unknown)} side records belong to documents without images, e.g. {unknown[0]}")
    rng = np.random.default_rng(seed)
    result = decide_all(sides, images, reads, rng, n_boot)
    decisions = result.decisions

    # ---- coverage -----------------------------------------------------------------------------------------------
    accepted = [k for k, d in decisions.items() if d["outcome"] == "accepted"]
    triaged = [k for k, d in decisions.items() if d["outcome"] == "needs_review" and d.get("triage")]
    cover: Dict[str, Dict[str, Any]] = {}
    for key in accepted + triaged:
        side = result.sides[key]
        doc_images = images[side.canonical_id]
        distinct, copies = distinct_images(doc_images)
        image = next(im for im in doc_images if im.index == decisions[key]["image_index"])
        unread = any(not result.pairs[key][im.index].get("usable") for im in distinct if im.index != image.index)
        flags, exclusions, info = coverage(side, reads[image.sha256], result.pairs[key][image.index], unread, bool(copies))
        cover[key] = {"flags": flags, "exclusions": exclusions, **info}

    # ---- leakage ------------------------------------------------------------------------------------------------
    doc_ids = sorted(images)
    res = leakage if leakage is not None else load_leakage_resources(set(doc_ids))
    pgpids, gt_sets = res.pgpids, res.gt_sets
    shelf = {s.canonical_id: s.record.get("shelf_mark") or "" for s in sides}
    identity = {cid: identity_leakage(cid, shelf.get(cid, ""), pgpids.get(cid, []), res.held, res.pgp_index, res.gate)
                for cid in doc_ids}
    gt = build_gt_index(gt_sets)
    text_leaks = {k: text_leakage(s.record.get("text") or "", gt, res.gate.content_contaminated)
                  for k, s in result.sides.items()}

    for key in triaged:                       # what a reviewed acceptance would give (no row is built from them)
        leak_reasons = identity[result.sides[key].canonical_id]["reasons"] + text_leaks[key]["reasons"]
        decisions[key].update(coverage=cover[key], leakage=leak_reasons,
                              would_be=("excluded:" + ",".join(cover[key]["exclusions"]) if cover[key]["exclusions"]
                                        else "held_out:" + ";".join(leak_reasons) if leak_reasons else "row"))

    # ---- rows ---------------------------------------------------------------------------------------------------
    splits: Dict[str, List[Dict[str, Any]]] = {"train_page": [], "val": []}
    manifest: List[Dict[str, Any]] = []
    for key in sorted(accepted, key=lambda k: (result.sides[k].canonical_id, result.sides[k].record_index)):
        side, decision = result.sides[key], decisions[key]
        rec = side.record
        image = next(im for im in images[side.canonical_id] if im.index == decision["image_index"])
        leak_reasons = identity[side.canonical_id]["reasons"] + text_leaks[key]["reasons"]
        decision["coverage"] = cover[key]
        decision["leakage"] = leak_reasons
        if cover[key]["exclusions"]:
            decision["row"] = "excluded:" + ",".join(cover[key]["exclusions"])
            continue
        if leak_reasons:
            decision["row"] = "held_out:" + ";".join(leak_reasons)
            continue
        side_name, stem = row_stem(side, image)
        split = pae.split_of(side.canonical_id, VAL_SHARE)
        text = (rec.get("text") or "").strip()
        splits[split].append(page_row(image, text, stem))
        pair = result.pairs[key][image.index]
        own = result.ownership.get(key)
        manifest.append(round_floats({
            "stem": stem, "split": split, "canonical_id": side.canonical_id, "doc_id": rec.get("doc_id"),
            "shelf_mark": rec.get("shelf_mark"), "pgpids": identity[side.canonical_id]["pgpids"],
            "merged_record_pgpids": pgpids.get(side.canonical_id, []), "main_language": rec.get("main_language"),
            "language_group": language_group(rec.get("main_language")), "section_index": rec.get("section_index"),
            "record_index": side.record_index, "column": rec.get("column"), "side": side_name, "side_label": side.label,
            "side_decision": rec.get("side_decision"), "image_index": image.index, "image_file": image.file_name,
            "image_token": image.token, "image_path": image.path, "image_sha256": image.sha256,
            "image_width": image.width, "image_height": image.height, "family": rec.get("family"),
            "flags": rec.get("flags") or {}, "letters": rec.get("letters"), "hebrew_letters": rec.get("hebrew_letters"),
            "arabic_letters": rec.get("arabic_letters"), "gaps": rec.get("gaps"), "gap_density": rec.get("gap_density"),
            "explicit_lines": "\n" in text, "first_words": first_words(text), "script": side.script, **pair_summary(pair),
            "assignment_basis": decision["basis"], "matched_n4": decision["matched_n4"], "priors": priors_of(side, image),
            "coverage_flags": cover[key]["flags"], "letter_ratio": cover[key]["letter_ratio"],
            "leakage": "clean", "shared_text": ({"role": own["role"], "group_docs": own["group_docs"]} if own else None),
            "legacy_store": True, "prompt_name": PROMPT_NAME, "prompt": PROMPT}))
        decision["row"] = stem

    # ---- statistics, outputs ------------------------------------------------------------------------------------
    stats = summarize(result, images, reads, read_info, cover, identity, text_leaks, gt_sets, manifest, docs_list)
    stats.update(name=out.name, built=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 rows={k: len(v) for k, v in splits.items()}, prompt_name=PROMPT_NAME,
                 inputs={"source": str(source), "pgp_export": str(PGP_EXPORT), "merged_corpus": str(MERGED_JSONL),
                         "n_boot": n_boot, "seed": seed})
    decision_rows = decision_records(result, images, cover, identity, text_leaks)
    readme = readme_text(stats)
    sidecars = {"manifest.jsonl": "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
                "decisions.jsonl": "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decision_rows),
                "stats.json": json.dumps(stats, ensure_ascii=False, indent=1), "README.md": readme}
    save_dataset(splits, out, sidecars)
    if export and any(splits.values()):
        export_dir = out.with_name(out.name + "_images_once")
        if export_dir.exists():
            shutil.rmtree(export_dir)          # a stale export would keep images of rows an earlier run exported
        exported = export_images_once(out, export_dir)
        (export_dir / "README.md").write_text(readme, encoding="utf-8")
        stats["images_once"] = {k: v for k, v in exported.items() if k != "features"}
        (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    queue = eye_queue_records(result, images, decision_rows)
    if eye_queue is not None:
        eye_queue.parent.mkdir(parents=True, exist_ok=True)
        eye_queue.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queue), encoding="utf-8")
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(stats, manifest, queue, out, eye_queue, seed), encoding="utf-8")
    return stats


# ===================================================================================================================
# Statistics, decisions file, eye queue, report
# ===================================================================================================================

def null_bin_summary(result: Decisions) -> Dict[str, Any]:
    """Null thresholds by read-length bin across sides, per script (4-grams).

    :param result: Decisions.
    :type result: Decisions
    :return: ``{script: {"pool": {...}, "bins": {bin: {...}}}}``.
    :rtype: Dict[str, Any]
    """
    out: Dict[str, Any] = {}
    for script, pool in result.pools.items():
        entry: Dict[str, Any] = {"pool_reads": len(pool), "pool_dropped": pool.dropped,
                                 "pool_reads_by_bin": {b: int((pool.bins == b).sum()) for b in BINS}, "bins": {}}
        keys = [k for k, s in result.sides.items() if s.script == script and k in result.nulls]
        for b in BINS + (ALL_BINS,):
            stats = [result.nulls[k][f"n{PRIMARY_N}"][b] for k in keys]
            usable = [s for s in stats if s is not None and (b == ALL_BINS or s["n"] >= MIN_BIN_NULL)]
            if not usable:
                entry["bins"][b] = {"sides": len(keys), "sides_with_bin_null": 0}
                continue
            p95 = np.array([s["p95"] for s in usable])
            p99 = np.array([s["p99"] for s in usable])
            width = np.array([s["p99_ci"][1] - s["p99_ci"][0] for s in usable if "p99_ci" in s])
            entry["bins"][b] = {"sides": len(keys), "sides_with_bin_null": len(usable),
                                "null_reads_median": float(np.median([s["n"] for s in usable])),
                                "p95": {"median": float(np.median(p95)), "p90": float(np.percentile(p95, 90)),
                                        "max": float(p95.max())},
                                "p99": {"median": float(np.median(p99)), "p90": float(np.percentile(p99, 90)),
                                        "max": float(p99.max())},
                                "p99_ci_width_median": float(np.median(width)) if width.size else None}
        out[script] = entry
    return out


def prior_vs_best_image(result: Decisions, images: Mapping[str, Sequence[ImageInfo]], prior: str) -> Dict[str, Any]:
    """Agreement of a prior with the image of best 4-gram recall, for every non-dropped side with two or more usable
    distinct images (independent of the acceptance rule).

    :param result: Decisions.
    :type result: Decisions
    :param images: Images per document.
    :type images: Mapping[str, Sequence[ImageInfo]]
    :param prior: ``side_label`` or ``order_prior``.
    :type prior: str
    :return: ``{family: {"best_clears_p99" | "best_below_p99": {agree, disagree}}}`` plus ``total``.
    :rtype: Dict[str, Any]
    """
    out: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for key, side in result.sides.items():
        if result.decisions[key]["outcome"] == "dropped":
            continue
        distinct, _ = distinct_images(images[side.canonical_id])
        scored = [(result.pairs[key][im.index], im) for im in distinct if result.pairs[key][im.index].get("usable")]
        if len(scored) < 2:
            continue
        best, image = max(scored, key=lambda t: t[0]["n4"])
        value = priors_of(side, image)[prior]
        if value is None:
            continue
        strength = "best_clears_p99" if best["clears_p99"] else "best_below_p99"
        out[side.record.get("family") or "none"][strength]["agree" if value else "disagree"] += 1
        out["total"][strength]["agree" if value else "disagree"] += 1
    return {family: {k: dict(v) for k, v in per.items()} for family, per in out.items()}


def summarize(result: Decisions, images: Mapping[str, Sequence[ImageInfo]], reads: Mapping[str, ReadInfo],
              read_info: Mapping[str, Any], cover: Mapping[str, Dict[str, Any]], identity: Mapping[str, Dict[str, Any]],
              text_leaks: Mapping[str, Dict[str, Any]], gt_sets: Mapping[str, Sequence[Tuple[str, str]]],
              manifest: Sequence[Dict[str, Any]], docs_list: Optional[Path]) -> Dict[str, Any]:
    """The funnel and every count of the report.

    :param result: Decisions.
    :type result: Decisions
    :param images: Images per document.
    :type images: Mapping[str, Sequence[ImageInfo]]
    :param reads: Reads by image hash.
    :type reads: Mapping[str, ReadInfo]
    :param read_info: :func:`load_reads` counts.
    :type read_info: Mapping[str, Any]
    :param cover: Coverage of accepted pairs.
    :type cover: Mapping[str, Dict[str, Any]]
    :param identity: Identity leakage per document.
    :type identity: Mapping[str, Dict[str, Any]]
    :param text_leaks: Text leakage per side.
    :type text_leaks: Mapping[str, Dict[str, Any]]
    :param gt_sets: GT texts per set.
    :type gt_sets: Mapping[str, Sequence[Tuple[str, str]]]
    :param manifest: Row manifest.
    :type manifest: Sequence[Dict[str, Any]]
    :param docs_list: Optional documents list (scope counts).
    :type docs_list: Optional[Path]
    :return: Statistics.
    :rtype: Dict[str, Any]
    """
    decisions = result.decisions
    sides = result.sides
    all_images = [im for doc in images.values() for im in doc]
    shas = {im.sha256 for im in all_images}
    usable_shas = {sha for sha in shas if sha in reads and not reads[sha].unusable}
    scope: Dict[str, Any] = {"documents_in_scope": len(images), "images": len(all_images), "distinct_images": len(shas)}
    if docs_list is not None:
        listed = json.loads(docs_list.read_text(encoding="utf-8"))
        scope["documents_listed"] = len(listed)
        scope["codices_out_of_scope"] = sum(1 for d in listed if len(d.get("image_urls") or []) > 4)
    docs_with_records = {s.canonical_id for s in sides.values()}
    docs_usable = {cid for cid, doc in images.items() if any(im.sha256 in usable_shas for im in doc)}
    scope.update(documents_with_side_records=len(docs_with_records), documents_with_usable_read=len(docs_usable),
                 documents_with_records_and_usable_read=len(docs_with_records & docs_usable))
    read_stats = {**read_info, "distinct_images_with_read": len(shas & set(reads)),
                  "usable": len(usable_shas), "unusable_by_reason": dict(Counter(reads[s].unusable for s in shas
                                                                                if s in reads and reads[s].unusable)),
                  "usable_by_source": dict(Counter(reads[s].source for s in usable_shas)),
                  "usable_by_script": dict(Counter(reads[s].script for s in usable_shas))}
    outcomes = Counter(d["outcome"] for d in decisions.values())
    reasons = {o: dict(Counter(r for d in decisions.values() if d["outcome"] == o for r in d["reasons"])) for o in OUTCOMES}
    docs_accepted = {sides[k].canonical_id for k, d in decisions.items() if d["outcome"] == "accepted"}
    groups = defaultdict(list)
    for key, own in result.ownership.items():
        groups[(tuple(own["group_docs"]), sides[key].record.get("section_digest"), sides[key].record.get("side"),
                sides[key].record.get("column"))].append(own["role"])
    group_outcome = Counter("owner_found" if "owner" in roles else "tie" if "tie" in roles
                            else "contested" if "contested" in roles else "no_owner" for roles in groups.values())
    accepted = [k for k, d in decisions.items() if d["outcome"] == "accepted"]
    basis = Counter(decisions[k]["basis"] for k in accepted)

    # priors
    def agreement(prior: str, keys: Iterable[str]) -> Dict[str, int]:
        """Agreement of one prior with the accepted images.

        :param prior: ``side_label`` or ``order_prior``.
        :type prior: str
        :param keys: Accepted side keys.
        :type keys: Iterable[str]
        :return: ``{with_prior, agree, disagree}``.
        :rtype: Dict[str, int]
        """
        c = Counter()
        for k in keys:
            side = sides[k]
            image = next(im for im in images[side.canonical_id] if im.index == decisions[k]["image_index"])
            value = priors_of(side, image)[prior]
            if value is None:
                continue
            c["with_prior"] += 1
            c["agree" if value else "disagree"] += 1
        return dict(c)

    p99_keys = [k for k in accepted if decisions[k]["basis"] == "p99"]
    labelled = [k for k, s in sides.items() if s.label]
    prior_stats = {
        "side_label": {"accepted_all": agreement("side_label", accepted), "accepted_on_p99": agreement("side_label", p99_keys),
                       "labelled_records": len(labelled),
                       "labelled_outcomes": dict(Counter(decisions[k]["outcome"] for k in labelled))},
        "order_prior": {"accepted_all": agreement("order_prior", accepted),
                        "accepted_on_p99": agreement("order_prior", p99_keys),
                        "records_with_order_prior": sum(1 for s in sides.values() if s.order_prior_index is not None)},
        "file_tokens": {"accepted_images_with_token": sum(1 for k in accepted if next(
            im for im in images[sides[k].canonical_id] if im.index == decisions[k]["image_index"]).token)},
        "side_label_vs_best_image": prior_vs_best_image(result, images, "side_label"),
        "order_prior_vs_best_image": prior_vs_best_image(result, images, "order_prior")}
    review = [k for k, d in decisions.items() if d["outcome"] == "needs_review"]
    triage: Dict[str, Any] = {"needs_review": len(review),
                              "held_only_by_conflict_or_spill": sum(1 for k in review if decisions[k].get("triage")),
                              "bands": {}}
    for _, band in TRIAGE_BANDS:
        keys = [k for k in review if decisions[k].get("triage") == band]
        would = Counter((decisions[k].get("would_be") or "").split(":")[0] for k in keys)
        triage["bands"][band] = {"sides": len(keys), "documents": len({sides[k].canonical_id for k in keys}),
                                 "would_be": dict(would),
                                 "would_be_row_letters": sum(sides[k].record.get("letters") or 0 for k in keys
                                                             if decisions[k].get("would_be") == "row")}

    excl = Counter(e for k in accepted for e in cover[k]["exclusions"])
    flagged = Counter(f for k in accepted if not cover[k]["exclusions"] for f in cover[k]["flags"])
    passing_cover = [k for k in accepted if not cover[k]["exclusions"]]
    id_rule = Counter()
    for cid, info in identity.items():
        for name in ("derive", "clean_v2_keys", "benchmark_keys"):
            if info[name]:
                id_rule[name] += 1
    text_rule = Counter(name for k in text_leaks for name in list(text_leaks[k]["hebrew"]) + list(text_leaks[k]["arabic"]))
    held_pairs = [k for k in passing_cover if decisions[k].get("leakage")]
    held_by_rule = Counter()
    for k in held_pairs:
        for reason in decisions[k]["leakage"]:
            held_by_rule[":".join(reason.split(":")[:2])] += 1
    row_letters = sum(m["letters"] or 0 for m in manifest)

    def by_field(name: str) -> Dict[str, Dict[str, int]]:
        """Rows, letters and documents per value of a manifest field.

        :param name: Field.
        :type name: str
        :return: ``{value: {rows, letters, documents, val_rows}}``.
        :rtype: Dict[str, Dict[str, int]]
        """
        out: Dict[str, Dict[str, int]] = {}
        for m in manifest:
            entry = out.setdefault(str(m[name]), {"rows": 0, "letters": 0, "documents": 0, "val_rows": 0})
            entry["rows"] += 1
            entry["letters"] += m["letters"] or 0
            entry["val_rows"] += m["split"] == "val"
        for value in out:
            out[value]["documents"] = len({m["canonical_id"] for m in manifest if str(m[name]) == value})
        return out

    return {
        "scope": scope, "reads": read_stats,
        "sides": {"records": len(sides), "documents": len(docs_with_records), "outcomes": dict(outcomes),
                  "reasons": reasons, "duplicate_versions_dropped": len(result.duplicates),
                  "documents_with_accepted_side": len(docs_accepted), "accepted_basis": dict(basis),
                  "accepted_clearing_p99_ci_hi": sum(1 for k in accepted
                                                     if result.pairs[k][decisions[k]["image_index"]].get("clears_p99_ci_hi")),
                  "by_script": dict(Counter(s.script or "none" for s in sides.values()))},
        "shared_text": {"records": len(result.ownership), "groups": len(groups), "group_outcomes": dict(group_outcome),
                        "record_roles": dict(Counter(o["role"] for o in result.ownership.values()))},
        "priors": prior_stats, "triage": triage, "null": null_bin_summary(result),
        "coverage": {"accepted_pairs": len(accepted), "excluded_pairs": sum(1 for k in accepted if cover[k]["exclusions"]),
                     "exclusions": dict(excl), "flags_on_kept_pairs": dict(flagged),
                     "kept_pairs": len(passing_cover)},
        "leakage": {"documents_checked": len(identity), "identity_documents_by_rule": dict(id_rule),
                    "identity_documents_any": sum(1 for v in identity.values() if v["reasons"]),
                    "text_records_checked": len(text_leaks), "text_records_by_set": dict(text_rule),
                    "text_records_any": sum(1 for v in text_leaks.values() if v["reasons"]),
                    "documents_with_merged_pgpids": sum(1 for v in identity.values() if v["pgpids"]),
                    "gt_sets": {k: len(v) for k, v in gt_sets.items()},
                    "held_out_pairs": len(held_pairs), "held_out_pairs_by_rule": dict(held_by_rule),
                    "pairs_checked": len(passing_cover)},
        "row_documents": len({m["canonical_id"] for m in manifest}), "row_letters": row_letters,
        "rows_by_family": by_field("family"), "rows_by_language_group": by_field("language_group"),
        "rows_by_basis": by_field("assignment_basis"),
        "rows_with_coverage_flags": sum(1 for m in manifest if m["coverage_flags"]),
        "val_documents": sorted({m["canonical_id"] for m in manifest if m["split"] == "val"}),
    }


def decision_records(result: Decisions, images: Mapping[str, Sequence[ImageInfo]], cover: Mapping[str, Dict[str, Any]],
                     identity: Mapping[str, Dict[str, Any]], text_leaks: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One record per side: its outcome, reasons, scores on every image of its document, null summary.

    :param result: Decisions.
    :type result: Decisions
    :param images: Images per document.
    :type images: Mapping[str, Sequence[ImageInfo]]
    :param cover: Coverage of accepted pairs.
    :type cover: Mapping[str, Dict[str, Any]]
    :param identity: Identity leakage per document.
    :type identity: Mapping[str, Dict[str, Any]]
    :param text_leaks: Text leakage per side.
    :type text_leaks: Mapping[str, Dict[str, Any]]
    :return: Records (texts quoted by their first words only).
    :rtype: List[Dict[str, Any]]
    """
    out = []
    for key, side in result.sides.items():
        rec = side.record
        decision = result.decisions[key]
        null = result.nulls.get(key)
        out.append(round_floats({
            "key": key, "canonical_id": side.canonical_id, "record_index": side.record_index,
            "section_index": rec.get("section_index"), "column": rec.get("column"), "side_label": side.label,
            "side_decision": rec.get("side_decision"), "family": rec.get("family"), "script": side.script,
            "letters": rec.get("letters"), "skeleton_letters": len(side.skeleton), "first_words": first_words(rec.get("text") or ""),
            "outcome": decision["outcome"], "reasons": decision["reasons"], "image_index": decision.get("image_index"),
            "basis": decision.get("basis"), "matched_n4": decision.get("matched_n4"),
            "conflicts": decision.get("conflicts", []),
            "other_images": decision.get("other_images", []), "kept_version": decision.get("kept"),
            "shared_text": result.ownership.get(key), "order_prior_index": side.order_prior_index,
            "images": [{"index": im.index, "file": im.file_name, "token": im.token, "sha256": im.sha256,
                        "priors": priors_of(side, im), **pair_summary(result.pairs.get(key, {}).get(im.index) or {})}
                       for im in images.get(side.canonical_id, [])],
            "null_reads": null["reads"] if null else None,
            "dominance_ratio": decision.get("dominance_ratio"), "triage": decision.get("triage"),
            "would_be": decision.get("would_be"),
            "coverage": cover.get(key), "leakage_identity": identity.get(side.canonical_id, {}).get("reasons"),
            "leakage_text": text_leaks.get(key, {}).get("reasons"), "row": decision.get("row")}))
    return out


def eye_queue_records(result: Decisions, images: Mapping[str, Sequence[ImageInfo]],
                      decision_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The review queue: every ``needs_review`` side with its images, candidate texts and numbers.

    :param result: Decisions.
    :type result: Decisions
    :param images: Images per document.
    :type images: Mapping[str, Sequence[ImageInfo]]
    :param decision_rows: :func:`decision_records` output.
    :type decision_rows: Sequence[Dict[str, Any]]
    :return: Queue records.
    :rtype: List[Dict[str, Any]]
    """
    by_key = {d["key"]: d for d in decision_rows}
    out = []
    for key, decision in result.decisions.items():
        if decision["outcome"] != "needs_review":
            continue
        side = result.sides[key]
        competitors = [k for k, s in result.sides.items() if s.canonical_id == side.canonical_id and k != key
                       and result.decisions[k]["outcome"] != "dropped"]
        out.append({"key": key, "canonical_id": side.canonical_id, "shelf_mark": side.record.get("shelf_mark"),
                    "family": side.record.get("family"), "side_label": side.label, "reasons": decision["reasons"],
                    "triage": decision.get("triage"), "dominance_ratio": round_floats(decision.get("dominance_ratio")),
                    "would_be": decision.get("would_be"),
                    "assigned_image_index": decision.get("image_index"),
                    "text_first_words": first_words(side.record.get("text") or ""),
                    "letters": side.record.get("letters"),
                    "images": [{"path": im.path, **{k: v for k, v in img.items() if k not in ("sha256",)}}
                               for im, img in zip(images[side.canonical_id], by_key[key]["images"])],
                    "other_sides": [{"key": k, "text_first_words": first_words(result.sides[k].record.get("text") or ""),
                                     "outcome": result.decisions[k]["outcome"],
                                     "recall_n4": {im.index: round_floats(result.pairs.get(k, {}).get(im.index, {}).get("n4"))
                                                   for im in images[side.canonical_id]}}
                                    for k in competitors],
                    "shared_text": result.ownership.get(key)})
    order = {name: i for i, (_, name) in enumerate(TRIAGE_BANDS)}
    out.sort(key=lambda q: (order.get(q["triage"], len(order)), q["dominance_ratio"] if q["dominance_ratio"] is not None
                            else 9.0, q["key"]))
    return out


def render_report(stats: Mapping[str, Any], manifest: Sequence[Dict[str, Any]], queue: Sequence[Dict[str, Any]],
                  out: Path, eye_queue: Optional[Path], seed: int) -> str:
    """The build report (Markdown, internal).

    :param stats: Statistics.
    :type stats: Mapping[str, Any]
    :param manifest: Row manifest.
    :type manifest: Sequence[Dict[str, Any]]
    :param queue: Eye queue.
    :type queue: Sequence[Dict[str, Any]]
    :param out: Dataset directory.
    :type out: Path
    :param eye_queue: Eye-queue path.
    :type eye_queue: Optional[Path]
    :param seed: Seed of the spot-check draw.
    :type seed: int
    :return: Markdown.
    :rtype: str
    """
    sc, sd, cov, leak = stats["scope"], stats["sides"], stats["coverage"], stats["leakage"]
    rd = stats["reads"]
    n_rec = sd["records"]
    oc = sd["outcomes"]
    assessed = n_rec - oc.get("dropped", 0)
    rows = stats["rows"]
    n_rows = sum(rows.values())
    lines = [f"# Legacy-store diplomatic rows: build report ({stats['name']})", "",
             f"Built {stats['built']} by `src/finetuning/qwen_hebrew/build_legacy_store_rows.py`. Dataset: `{out}`"
             f" (images-once export next to it). **PRIVATE; NO TRAINING USE until this report and the eye queue have"
             f" been reviewed.** Texts are quoted by their first {QUOTE_WORDS} words at most.", "",
             "## Funnel", ""]
    if "documents_listed" in sc:
        lines.append(f"- Documents listed: {sc['documents_listed']:,}; codices (5+ images) out of scope: "
                     f"{pct(sc['codices_out_of_scope'], sc['documents_listed'])}; in scope: "
                     f"{pct(sc['documents_in_scope'], sc['documents_listed'])}.")
    lines += [
        f"- In scope: {sc['documents_in_scope']:,} documents, {sc['images']:,} images ({sc['distinct_images']:,} distinct sha256).",
        f"- Documents with side records: {pct(sc['documents_with_side_records'], sc['documents_in_scope'])}; with at least one "
        f"usable read: {pct(sc['documents_with_usable_read'], sc['documents_in_scope'])}; with both: "
        f"{pct(sc['documents_with_records_and_usable_read'], sc['documents_in_scope'])}.",
        f"- Reads: Kraken {rd['kraken_reads']:,} (zero lines {rd['kraken_zero_lines']:,}); Gemini {rd['gemini_reads']:,} "
        f"(NO TEXT {rd['gemini_no_text']:,}, with text {rd['gemini_with_text']:,}). Distinct images with a read: "
        f"{pct(rd['distinct_images_with_read'], sc['distinct_images'])}; usable: {pct(rd['usable'], sc['distinct_images'])}; "
        f"unusable by reason: {rd['unusable_by_reason']}; usable by source {rd['usable_by_source']}, by script {rd['usable_by_script']}.",
        f"- Side records: {n_rec:,} from {sd['documents']:,} documents (script {sd['by_script']}).",
        f"  - dropped: {pct(oc.get('dropped', 0), n_rec)} ({stats['sides']['reasons'].get('dropped', {})});",
        f"  - assessed: {pct(assessed, n_rec)} -> accepted {pct(oc.get('accepted', 0), assessed)}, needs_review "
        f"{pct(oc.get('needs_review', 0), assessed)}, unassigned {pct(oc.get('unassigned', 0), assessed)}.",
        f"  - needs_review reasons (a side can have several): {sd['reasons'].get('needs_review', {})}",
        f"  - unassigned reasons: {sd['reasons'].get('unassigned', {})}",
        f"  - accepted by basis: {sd['accepted_basis']}; accepted pairs also above the p99 bootstrap CI upper bound: "
        f"{pct(sd['accepted_clearing_p99_ci_hi'], oc.get('accepted', 0))}; documents with an accepted side: "
        f"{pct(sd['documents_with_accepted_side'], sc['documents_in_scope'])}.",
        f"- Shared texts: {stats['shared_text']['records']:,} records in {stats['shared_text']['groups']:,} groups; group "
        f"outcomes {stats['shared_text']['group_outcomes']}; record roles {stats['shared_text']['record_roles']}.",
        f"- Coverage (accepted pairs {cov['accepted_pairs']:,}): excluded {pct(cov['excluded_pairs'], cov['accepted_pairs'])} "
        f"by reason {cov['exclusions']} (a pair can have several); kept {pct(cov['kept_pairs'], cov['accepted_pairs'])}; "
        f"flags on kept pairs {cov['flags_on_kept_pairs']}.",
        f"- Leakage (kept pairs {leak['pairs_checked']:,}): held out {pct(leak['held_out_pairs'], leak['pairs_checked'])}, by "
        f"rule {leak['held_out_pairs_by_rule']}.",
        f"- **Rows out: {n_rows:,}** (train_page {rows.get('train_page', 0):,}, val {rows.get('val', 0):,}) from "
        f"{stats['row_documents']:,} documents, {stats['row_letters']:,} letters; rows with a coverage flag: "
        f"{pct(stats['rows_with_coverage_flags'], n_rows)}.", ""]
    for title, key in (("family", "rows_by_family"), ("language group", "rows_by_language_group"),
                       ("assignment basis", "rows_by_basis")):
        lines += [f"Rows by {title}:", "", f"| {title} | rows | val rows | documents | letters |", "|---|---|---|---|---|"]
        for value, e in sorted(stats[key].items(), key=lambda kv: -kv[1]["rows"]):
            lines.append(f"| {value} | {pct(e['rows'], n_rows)} | {e['val_rows']} | {e['documents']} | {e['letters']:,} |")
        lines.append("")
    ratios = [m["letter_ratio"] for m in manifest if m["letter_ratio"] is not None]
    low = sum(1 for r in ratios if r < LETTER_RATIO_RANGE[0])
    high = sum(1 for r in ratios if r > LETTER_RATIO_RANGE[1])
    zs = sorted(m["z"] for m in manifest if m["z"] is not None)
    grams = sorted(m["matched_n4"] for m in manifest)
    lines += ["Row coverage detail:", "",
              f"- read letters / target letters: under {LETTER_RATIO_RANGE[0]} {pct(low, n_rows)}, "
              f"{LETTER_RATIO_RANGE[0]}-{LETTER_RATIO_RANGE[1]} {pct(n_rows - low - high, n_rows)}, over "
              f"{LETTER_RATIO_RANGE[1]} {pct(high, n_rows)} (pairs over {MAX_READ_OVER_TARGET:g} are excluded).",
              f"- matched {PRIMARY_N}-grams on the row's image: min {grams[0]}, median {grams[len(grams) // 2]} "
              f"(an acceptance needs {MIN_MATCHED_GRAMS})." if grams else "- matched grams: none.",
              f"- read-length bin of the row's image: {dict(Counter(m['bin'] for m in manifest))}; read source: "
              f"{dict(Counter(m['read_source'] for m in manifest))}; rows with explicit line breaks: "
              f"{pct(sum(1 for m in manifest if m['explicit_lines']), n_rows)}.",
              f"- z of the matched recall against the side's null (bin used): min {zs[0]:.1f}, median {zs[len(zs) // 2]:.1f}."
              if zs else "- z: none.", ""]
    pr = stats["priors"]
    lines += ["## Prior agreement", "",
              "Priors only break ties in the assignment; the side label also lets p95 stand in for p99 when it names the "
              "image (the order prior does not). Agreement is measured on the accepted pairs, and separately on the pairs "
              "accepted on p99 alone (independent of the prior).", ""]
    for name in ("side_label", "order_prior"):
        for scope_name in ("accepted_all", "accepted_on_p99"):
            a = pr[name][scope_name]
            lines.append(f"- {name}, {scope_name}: agree {pct(a.get('agree', 0), a.get('with_prior', 0))}, disagree "
                         f"{pct(a.get('disagree', 0), a.get('with_prior', 0))}.")
    lines += [f"- Labelled records: {pr['side_label']['labelled_records']:,}/{n_rec:,}; their outcomes "
              f"{pr['side_label']['labelled_outcomes']}.",
              f"- Records with an order prior: {pr['order_prior']['records_with_order_prior']:,}/{n_rec:,}.", "",
              "Independent of the acceptance rule: does the prior name the image of best 4-gram recall? (sides not "
              "dropped, with two or more usable distinct images; by whether that best recall clears the side's p99)", "",
              "| prior | family | best clears p99: agree | disagree | best below p99: agree | disagree |",
              "|---|---|---|---|---|---|"]
    for name in ("side_label_vs_best_image", "order_prior_vs_best_image"):
        for family, per in sorted(pr[name].items(), key=lambda kv: kv[0] != "total"):
            strong, weak = per.get("best_clears_p99", {}), per.get("best_below_p99", {})
            n_strong, n_weak = sum(strong.values()), sum(weak.values())
            lines.append(f"| {name.split('_vs_')[0]} | {family} | {pct(strong.get('agree', 0), n_strong)} | "
                         f"{pct(strong.get('disagree', 0), n_strong)} | {pct(weak.get('agree', 0), n_weak)} | "
                         f"{pct(weak.get('disagree', 0), n_weak)} |")
    lines.append("")
    tr = stats["triage"]
    lines += ["## Review triage (no automatic acceptance)", "",
              f"Of {tr['needs_review']:,} needs_review sides, {pct(tr['held_only_by_conflict_or_spill'], tr['needs_review'])} "
              f"are held ONLY by a conflict (another side of the document clears its p95 on the image) or a spill (the "
              f"side clears its p95 on another image). The two sides of one leaf share names, formulae and hand, so their "
              f"cross recall sits above the cross-document null. Dominance ratio = largest competing recall relative to the "
              f"matched recall (see `dominance_ratio`). Bands, and what each side would give after coverage and leakage:", "",
              "| band | ratio | sides | documents | would be row / excluded / held out | letters of would-be rows |",
              "|---|---|---|---|---|---|"]
    lo = 0.0
    for bound, band in TRIAGE_BANDS:
        e = tr["bands"][band]
        span = f"{lo:g}-{bound:g}" if bound != float("inf") else f"> {lo:g}"
        lines.append(f"| {band} | {span} | {pct(e['sides'], tr['held_only_by_conflict_or_spill'])} | {e['documents']} | "
                     f"{e['would_be'].get('row', 0)} / {e['would_be'].get('excluded', 0)} / {e['would_be'].get('held_out', 0)} | "
                     f"{e['would_be_row_letters']:,} |")
        lo = bound
    lines.append("")
    lines += ["## Null thresholds by read-length bin (4-grams)", "",
              f"Per side: recall against usable reads of OTHER documents' images in the same script; a bin needs "
              f"{MIN_BIN_NULL} null reads, else the unstratified null is used. Bootstrap: {stats['inputs']['n_boot']} "
              f"resamples. Values are the distribution over sides of each side's threshold.", ""]
    for script, entry in stats["null"].items():
        pool_line = (f"**{script}** pool: {entry['pool_reads']:,} reads (by bin {entry['pool_reads_by_bin']}; reader's "
                     f"reads left out {entry['pool_dropped']}).")
        if not entry["bins"][ALL_BINS].get("sides"):
            lines += [pool_line + f" No side record has a majority of {script} letters (0/{n_rec:,}), so this null was "
                                  f"never used.", ""]
            continue
        lines += [pool_line, "",
                  "| bin | sides with a bin null | median null reads | p95 median / p90 / max | p99 median / p90 / max | "
                  "median p99 CI width |", "|---|---|---|---|---|---|"]
        for b, e in entry["bins"].items():
            if not e.get("sides_with_bin_null"):
                lines.append(f"| {b} | 0/{e['sides']} | - | - | - | - |")
                continue
            lines.append(f"| {b} | {e['sides_with_bin_null']}/{e['sides']} | {e['null_reads_median']:.0f} | "
                         f"{e['p95']['median']:.3f} / {e['p95']['p90']:.3f} / {e['p95']['max']:.3f} | "
                         f"{e['p99']['median']:.3f} / {e['p99']['p90']:.3f} / {e['p99']['max']:.3f} | "
                         f"{e['p99_ci_width_median'] if e['p99_ci_width_median'] is None else round(e['p99_ci_width_median'], 3)} |")
        lines.append("")
    lines += ["## Leakage by rule", "",
              f"- Identity, documents (of {leak['documents_checked']:,} in scope): {leak['identity_documents_by_rule']}; any "
              f"rule {pct(leak['identity_documents_any'], leak['documents_checked'])}. Documents with merged-corpus PGP ids: "
              f"{pct(leak['documents_with_merged_pgpids'], leak['documents_checked'])}.",
              f"- Text, side records (of {leak['text_records_checked']:,}): by GT set {leak['text_records_by_set']}; any set "
              f"{pct(leak['text_records_any'], leak['text_records_checked'])}. GT documents per set: {leak['gt_sets']}.",
              f"- Kept pairs held out: {pct(leak['held_out_pairs'], leak['pairs_checked'])}, by rule {leak['held_out_pairs_by_rule']} "
              f"(a pair can hit several rules).", ""]
    reason_counts = Counter(r for q in queue for r in q["reasons"])
    lines += ["## Eye queue", "",
              f"`{eye_queue}`: {len(queue):,} needs_review sides (image paths, first {QUOTE_WORDS} words of the side and of "
              f"the document's other sides, recall and null numbers per image). Reasons: {dict(reason_counts)}.", ""]
    rng = random.Random(seed)
    sample = rng.sample(list(manifest), min(SPOT_CHECK, len(manifest)))
    lines += [f"## Spot check: {len(sample)} random rows (seed {seed})", "",
              "| # | image | side | recall n4 / p99 | first words |", "|---|---|---|---|---|"]
    for i, m in enumerate(sample, 1):
        lines.append(f"| {i} | `{m['image_path']}` | {m['side']} | {m['recall_n4']:.3f} / {m['null_p99_n4']:.3f} | "
                     f"{m['first_words']} |")
    lines.append("")
    lines += method_lines() + caveat_lines(stats, manifest)
    return "\n".join(lines)


def method_lines() -> List[str]:
    """The method and every threshold, for the report.

    :return: Markdown lines.
    :rtype: List[str]
    """
    return ["## Method and thresholds", "",
            f"- Skeletons: Hebrew letters only, finals folded, points dropped; Arabic rasm (`rasm_skeleton`). Reads under "
            f"{MIN_READ_LETTERS} letters of their script, Kraken images without lines, NO TEXT and loops (repeat share >= "
            f"{sm.MAX_REPEAT_SHARE}) are unusable; exact duplicate read texts enter a null pool once.",
            f"- Recall: clipped {PRIMARY_N}-gram recall of the side in the read; {NGRAMS[1]}-grams as the second signal "
            f"(must clear its own p95).",
            f"- Null per side: same recall against the pool's reads of other documents (own document, documents sharing its "
            f"text, documents sharing one of its images left out), bins {', '.join(BINS)} (letters of the read); a bin with "
            f"fewer than {MIN_BIN_NULL} reads uses the unstratified null; p95 / p99 (linear), {N_BOOT}-resample bootstrap CI.",
            "- Accept: Hungarian one-to-one on recall (priors add 1e-6 / 1e-7 as tie-breaks); recall > p99 (> p95 when the "
            f"side label names the image; the order prior relaxes nothing); at least {MIN_MATCHED_GRAMS} matched "
            f"{PRIMARY_N}-grams (recall x the side's gram count; fewer = `{UNDER_MATCHED_REASON}`, review); no other side of "
            "the document > its p95 on the image; the side > its p95 on no other distinct image (added beyond the plan: a "
            "record covering both leaf sides would otherwise become one page target); an image accepted for two documents "
            "goes to review.",
            "- Duplicate versions: the cleaner's `near_duplicates` rule pairs a `duplicate_version` record with earlier "
            "records of its document; the version with more letters stays.",
            "- Shared texts: records grouped by section digest, side, column and text; owner = best recall clearing p99 when "
            f"it is the only copy clearing p99 or at least {MIN_OWNER_MARGIN:g}x the runner-up's; a smaller lead = "
            "`ownership_contested` (all copies to review); equal bests = tie (review); none clearing = no owner (all copies "
            "to review).",
            f"- Coverage: flags for read/target letters outside {LETTER_RATIO_RANGE}, other-script letters in the read over "
            f"{MAX_OTHER_SCRIPT_SHARE:.0%}, another image without a usable read, byte-identical images; exclusions for "
            f"{', '.join(EXCLUDED_FAMILIES)}, {', '.join(EXCLUDED_FLAGS)}, gap density over {MAX_GAP_DENSITY:g} gaps per 100 "
            f"letters (the cleaner's `gap_density`), under {MIN_TARGET_LETTERS} letters, read letters over "
            f"{MAX_READ_OVER_TARGET:g}x the target's (`read_exceeds_target_{MAX_READ_OVER_TARGET:g}x`).",
            f"- Leakage: identity = `derive_v25a.held_out_reason` (registered PGP ids / canonical ids / shelf-mark keys, "
            f"frozen 131) on PGP ids from the merged corpus, the canonical id and the shelf-mark parts, plus the decontam "
            f"gate's clean v2 and benchmark loose keys; text = `DecontamGate.content_contaminated` ({HEBREW_SHINGLE}-letter "
            f"Hebrew shingles, 2 hits) and {ARABIC_SHINGLE}-letter rasm shingles ({ARABIC_MIN_SHARED_ONE_DOC} with one GT "
            f"document or {ARABIC_MIN_SHARE_UNION:.0%} of the target's shingles in the union) against {', '.join(GT_SETS)}.",
            f"- Rows: question `{PROMPT_NAME}`, answer = cleaned text (line breaks kept where present), stem side = the "
            f"image's file-name token (the record's label only without one), split by "
            f"`build_pgp_arabic_editions.split_of(canonical_id, {VAL_SHARE})`.", ""]


def caveat_lines(stats: Mapping[str, Any], manifest: Sequence[Dict[str, Any]]) -> List[str]:
    """Caveats the reviewer should weigh before any training use.

    :param stats: Statistics.
    :type stats: Mapping[str, Any]
    :param manifest: Row manifest.
    :type manifest: Sequence[Dict[str, Any]]
    :return: Markdown lines.
    :rtype: List[str]
    """
    n_rows = len(manifest)
    sd, tr, pr = stats["sides"], stats["triage"], stats["priors"]
    assessed = sd["records"] - sd["outcomes"].get("dropped", 0)
    low = sum(1 for m in manifest if m["letter_ratio"] is not None and m["letter_ratio"] < LETTER_RATIO_RANGE[0])
    high = sum(1 for m in manifest if m["letter_ratio"] is not None and m["letter_ratio"] > LETTER_RATIO_RANGE[1])
    over = stats["coverage"]["exclusions"].get(f"read_exceeds_target_{MAX_READ_OVER_TARGET:g}x", 0)
    thin = sd["reasons"].get("needs_review", {}).get(UNDER_MATCHED_REASON, 0)
    contested = stats["shared_text"]["record_roles"].get("contested", 0)
    contested_groups = stats["shared_text"]["group_outcomes"].get("contested", 0)
    short = sum(1 for m in manifest if m["bin"] == BINS[0])
    doc_label = pr["side_label_vs_best_image"].get("documentary_plain", {}).get("best_clears_p99", {})
    app_label = pr["side_label_vs_best_image"].get("edition_apparatus", {}).get("best_clears_p99", {})
    order = pr["order_prior_vs_best_image"].get("total", {}).get("best_clears_p99", {})
    bg = tr["bands"]["background_like"]
    arabic_rows = sum(1 for m in manifest if m["language_group"] == "arabic")
    return ["## Caveats (settle before any training use)", "",
            f"1. **Most sides are not rows yet.** {pct(sd['outcomes'].get('accepted', 0), assessed)} assessed sides were "
            f"accepted; {pct(sd['outcomes'].get('needs_review', 0), assessed)} wait in the eye queue. Of those, "
            f"{pct(tr['held_only_by_conflict_or_spill'], tr['needs_review'])} are held only by the conflict / spill checks, "
            f"which fire on the shared background of the two sides of one leaf; the `background_like` band "
            f"({bg['sides']} sides, {bg['documents']} documents) would add {bg['would_be'].get('row', 0)} rows if a reviewer "
            f"accepts it. Nothing is accepted automatically.",
            f"2. **Partial targets.** {pct(low, n_rows)} rows have a read with under half the target's letters and "
            f"{pct(short, n_rows)} rows' reads hold under 100 letters: Kraken misses lines on cursive hands, or the target "
            f"covers more than the image. Their recall still clears the side's p99 for that read length, but the flag "
            f"(`letter_ratio_out_of_range`) should be reviewed before training. The other direction is worse, since Kraken "
            f"under-reads: a read with more than {MAX_READ_OVER_TARGET:g}x the target's letters means the page holds text "
            f"the target lacks, so those pairs are excluded (`read_exceeds_target_{MAX_READ_OVER_TARGET:g}x`: {over} "
            f"accepted pairs); {pct(high, n_rows)} rows sit between {LETTER_RATIO_RANGE[1]:g}x and "
            f"{MAX_READ_OVER_TARGET:g}x and carry the same flag.",
            f"3. **Priors are weak.** For documentary sides whose best image clears p99 the side label names that image in "
            f"{pct(doc_label.get('agree', 0), sum(doc_label.values()))} cases (edition apparatus: "
            f"{pct(app_label.get('agree', 0), sum(app_label.values()))}); the stored-order prior in "
            f"{pct(order.get('agree', 0), sum(order.values()))}, a coin flip, so it no longer relaxes p99 to p95 (it "
            f"only breaks ties); the side-label relaxation accepted {sd['accepted_basis'].get('p95+side_label', 0)} sides.",
            f"4. **Thin evidence and contested ownership go to review.** {thin} sides whose pair clears its threshold share "
            f"fewer than {MIN_MATCHED_GRAMS} matched {PRIMARY_N}-grams with the read (`{UNDER_MATCHED_REASON}`); {contested} copies "
            f"in {contested_groups} shared-text groups have a best recall under {MIN_OWNER_MARGIN:g}x the runner-up's while "
            f"both clear p99 (`ownership_contested`). None of them is a row until a reviewer decides.",
            f"5. **Script.** No side record has an Arabic-script majority (0/{sd['records']:,}); the Arabic null pool was "
            f"built but never used. Documents catalogued as `arabic` give {pct(arabic_rows, n_rows)} rows whose targets and "
            f"reads are Hebrew script (Judaeo-Arabic). Gemini reads that came back in Hebrew letters are scored against the "
            f"Kraken null ({sum(1 for m in manifest if m['read_source'] == 'gemini')} row).",
            "6. **Text rule is conservative.** A 25-letter Hebrew shingle shared twice with literary GT (Bible, Talmud, "
            "liturgy) holds a documentary record out even when it only quotes the same passage.",
            f"7. **Small val.** The 5 % document hash puts {len(stats['val_documents'])} documents ({stats['rows'].get('val', 0)} "
            f"rows) in val.",
            "8. Gap density is read as the cleaner's `gap_density` (gaps per 100 letters) over 30.", ""]


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface.

    :return: Parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR, help=f"input directory (default {SOURCE_DIR})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"dataset directory (default {DEFAULT_OUT})")
    parser.add_argument("--docs-list", type=Path, default=None, help="JSON list of the documents considered (scope counts)")
    parser.add_argument("--report", type=Path, default=None, help="write the build report here (Markdown)")
    parser.add_argument("--eye-queue", type=Path, default=None, help="write the review queue here (JSONL)")
    parser.add_argument("--no-export", action="store_true", help="skip the images-once export")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    :return: None.
    :rtype: None
    """
    args = build_parser().parse_args(argv)
    stats = build(args.source, args.out, args.docs_list, args.report, args.eye_queue, not args.no_export,
                  args.n_boot, args.seed)
    print(json.dumps({k: stats[k] for k in ("scope", "rows", "row_documents", "row_letters")}, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in stats["sides"].items() if k != "reasons"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
