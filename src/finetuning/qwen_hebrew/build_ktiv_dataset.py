"""Build the KTIV Genizah training dataset (v1.9 data lever).

Input: KTIV viewer transcription bundles in API shape
(``ktiv_<pnx>_transcription.json`` with ``source == "nli_ktiv_viewer"``) plus
the per-manuscript image zips, both under ``raw_data/cairo_genizah/ktiv``.
Reading order, word merging and sigla policy come from
:mod:`src.finetuning.qwen_hebrew.ktiv_layout`; each page pairs with its image
exactly via the ``fl`` identifier embedded in the zip member name.

Task families (the notebook chooses the mixture; full-page transcription is
meant to be the bulk):

* ``fragment_transcribe`` — full page image -> full page text (the same
  prompt as genizah_clean_v2, so the row is a drop-in).
* ``region_transcribe`` — full page image + "transcribe ONLY <region>" ->
  the region's lines.  Regions are defined geometrically (first/last k
  lines, right/left column, top/bottom half) so targets are exact by
  construction.  This is the realistic "ask about part of the page" task.
* ``section_transcribe`` — crop of k consecutive lines (native resolution)
  -> those lines; trains the sections inference mode.
* ``line_transcribe`` — one full-line crop -> its text (small share).
* ``page_short`` (v4 ``short_fragment`` family) — the page-transcription row
  (same prompt, target and label source as ``fragment_transcribe``) for
  transcribed pages under the MIN_LETTERS gate: SHORT_MIN_LETTERS..149
  letters and a damage share <= SHORT_MAX_DAMAGE_SHARE (see
  :func:`page_damage_share`).  No other family is emitted for these pages;
  the task name keeps them in their own ``train_page_short`` hub split.

Gates per page: >= MIN_LETTERS Hebrew letters, gap words <= MAX_GAP_SHARE,
>= MIN_LINES lines, image present and boxes inside it.  Decontamination:
manuscripts sharing >= 2 25-letter shingles with any verified-benchmark GT
(short-fragment pages included) are dropped entirely.  Split by manuscript;
``--val-manuscripts <previous build>/val_manuscripts.json`` pins the val set
across rebuilds (only manuscripts new since then are drawn into val).

Usage (from repo root):
    PYTHONPATH=. python -m src.finetuning.qwen_hebrew.build_ktiv_dataset \\
        [--limit 0] [--no-short-fragments] [--val-manuscripts PATH] \\
        [--push-to-hub isaacmg/genizah_ktiv_v1]
"""

import argparse
import glob
import io
import json
import logging
import os
import random
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from src.finetuning.qwen_hebrew.ktiv_bundles import (
    bundle_shape,
    bundle_sys_num,
    select_bundles,
)
from src.finetuning.qwen_hebrew.ktiv_layout import (
    GAP_TOKEN,
    _median_height,
    cluster_lines,
    extract_words,
    hebrew_letters,
    reconstruct_page,
    split_columns,
)
from src.finetuning.qwen_hebrew.prompts import FRAGMENT_TRANSCRIBE_PROMPT

PILImage.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
KTIV_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/ktiv"
BENCH_PATH = (_REPO / "src/datasets/raw_data/cairo_genizah/evaluations/"
              "genizah_test_v1/genizah_test_v1_verified.json")
RELIGIOUS_BENCH_PATH = (_REPO / "src/datasets/raw_data/cairo_genizah/evaluations/"
                        "genizah_religious_v1/genizah_religious_v1.json")
DEFAULT_OUT = _REPO / "src/datasets/processed/genizah_ktiv_v1"
DEFAULT_IMAGES = _REPO / "src/datasets/raw_data/cairo_genizah/ktiv_dataset_v1/images"

MIN_LETTERS = 150
MAX_GAP_SHARE = 0.40
MIN_LINES = 3
MIN_IMAGE_SIDE_PX = 400
MAX_PAGE_PIXELS = 6_500_000      # == training min_pixels: no information lost
DECONTAM_SHINGLE = 25
DECONTAM_MIN_HITS = 2
VAL_FRACTION = 0.05
SPLIT_SEED = 20260818
REGION_ROWS_PER_PAGE = 1
SECTION_ROWS_PER_PAGE = 1
LINE_ROWS_PER_PAGE = 2
SECTION_LINES = (4, 8)
LINE_MIN_WORDS = 4
LINE_MIN_WIDTH_PX = 600
# v20 grounding/QA rows (design: docs/v20_grounding_qa_design.md). Emitted only
# for geometry-mode pages; all coordinates are 0-1000 normalized to the
# ORIGINAL scan frame (invariant under the aspect-preserving page rescale).
LOCATE_ROWS_PER_PAGE = 2
READBOX_ROWS_PER_PAGE = 1
LAYOUT_QA_ROWS_PER_PAGE = 1
GROUNDED_PAGE_FRACTION = 0.15
GROUNDED_PAGE_MAX_LINES = 20
LOCATE_PHRASE_WORDS = (2, 3)
LOCATE_MIN_PHRASE_LETTERS = 8
READBOX_MIN_LETTERS = 15
# v2.1 direct-grounding row families (design: docs/v21_grounding_boxes_design.md).
# Emitted only for geometry-mode pages, same original-scan 0-1000 frame.
LOCATE_WORD_ROWS_PER_PAGE = 3        # unique word -> box (the un-guessable target)
READBOX_WORD_ROWS_PER_PAGE = 1       # word box -> word text
LINE_INDEX_ROWS_PER_PAGE = 2         # "line N of the {right|left} column" -> box + text
LINE_OF_PHRASE_ROWS_PER_PAGE = 1     # phrase -> host line box + text
LOCATE_WORD_MIN_LETTERS = 4          # shorter words are too ambiguous to box
LINE_INDEX_MIN_LETTERS = 8           # target line needs clean text for its answer
GROUNDED_DETECT_MAX_LINES = 30       # detect-then-read emitted up to this many lines
# Random per-column band crop: a mid-column run of lines becomes its own image
# with boxes re-normalized to the crop, so a given line's y-position and the
# page margin vary between examples -> the box can no longer be a page template.
GROUNDED_CROP_ROWS_PER_PAGE = 1
GROUNDED_CROP_LINES = (4, 10)        # inclusive range of lines per crop band
# v4 short_fragment family (decided 2026-09-14): lightly damaged pages under the
# MIN_LETTERS gate get page-transcription rows only (task ``page_short``).
# Heavily damaged short pages stay out: their targets are mostly guesswork.
SHORT_MIN_LETTERS = 40
SHORT_MAX_DAMAGE_SHARE = 0.30
SHORT_MIN_LINES = 1
DAMAGE_MARK_CHARS = ".·…[](){}<>⟨⟩"  # illegible-letter dots and editorial brackets

FEATURES = Features({
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
})

_REGION_PROMPT = """This image is a manuscript fragment from the Cairo Genizah — \
handwritten Hebrew script (the language may be Hebrew, Judeo-Arabic, or Aramaic).

Transcribe ONLY {region}, exactly as written, in reading order. Mark unclear characters with [?].
Where text is lost or illegible due to damage, write [...].
Do NOT correct, restore, or complete from memory. Do not transcribe any other part of the page.

Return ONLY the transcription."""

_LOCATE_PROMPT = ('Locate the exact Hebrew phrase "{phrase}" on this manuscript '
                  'page. Respond with ONLY a JSON object '
                  '{{"bbox_2d": [x1, y1, x2, y2]}} giving the phrase\'s bounding '
                  'box, coordinates normalized to 0-1000. No other text.')
_READBOX_PROMPT = ('Transcribe ONLY the Hebrew text inside the region bbox_2d = '
                   '[{x0}, {y0}, {x1}, {y1}] (coordinates normalized 0-1000) on '
                   'this manuscript page, exactly as written. Mark unclear '
                   'characters with [?]. Return the text alone, no commentary.')
_GROUNDED_PROMPT = ('Transcribe this manuscript page line by line. Respond with '
                    'ONLY a JSON array; each element '
                    '{"text": "...", "bbox_2d": [x1, y1, x2, y2]} gives one '
                    'line\'s transcription and its bounding box, coordinates '
                    'normalized to 0-1000. Preserve reading order.')
_QA_COLUMNS_PROMPT = ('How many columns of text does this manuscript page have? '
                      'Answer with the number only.')
_QA_FIRST_LINE_PROMPT = ('Transcribe ONLY the {which} line of this manuscript '
                         'page, exactly as written. Return the text alone.')
_QA_FIND_LINE_PROMPT = ('Which line of this manuscript page contains the phrase '
                        '"{phrase}"? Return that full line\'s transcription '
                        'alone, exactly as written.')

# v2.1 direct-grounding prompts (design: docs/v21_grounding_boxes_design.md).
# These target the "template box" failure: v2.0 emitted one x-range repeated
# down the page. Word-level locate, index-addressed lines and detect-then-read
# make the box depend on the ink, not on the line's ordinal position.
_LOCATE_WORD_PROMPT = ('Locate the exact Hebrew word "{word}" on this manuscript '
                       'page. Respond with ONLY a JSON object '
                       '{{"bbox_2d": [x1, y1, x2, y2]}} giving that word\'s '
                       'bounding box, coordinates normalized to 0-1000. No other text.')
_READBOX_WORD_PROMPT = ('What single Hebrew word is written inside the region '
                        'bbox_2d = [{x0}, {y0}, {x1}, {y1}] (coordinates '
                        'normalized 0-1000) on this manuscript page? Return the '
                        'word alone, no commentary.')
_LINE_INDEX_PROMPT = ('Find line number {n}{col} of this manuscript page '
                      '(counting from the {frm}). Respond with ONLY a JSON object '
                      '{{"bbox_2d": [x1, y1, x2, y2], "text": "..."}} giving that '
                      'line\'s bounding box (coordinates normalized 0-1000) and '
                      'its transcription. No other text.')
_LINE_OF_PHRASE_BOX_PROMPT = ('Which line of this manuscript page contains the '
                              'phrase "{phrase}"? Respond with ONLY a JSON object '
                              '{{"bbox_2d": [x1, y1, x2, y2], "text": "..."}} giving '
                              'that line\'s bounding box (coordinates normalized '
                              '0-1000) and its full transcription. No other text.')
# Detect-then-read: bbox emitted BEFORE text, so localization precedes
# transcription. Distinct wording from _GROUNDED_PROMPT (which stays text-first
# for eval comparability across versions) so the two formats do not bleed.
_GROUNDED_DETECT_PROMPT = ('Find and transcribe every text line on this '
                           'manuscript page. Respond with ONLY a JSON array; each '
                           'element {"bbox_2d": [x1, y1, x2, y2], "text": "..."} '
                           'gives one line\'s bounding box (coordinates normalized '
                           '0-1000) FIRST, then its transcription. Preserve reading '
                           'order.')

_HEB_RE = re.compile(r"[א-ת]")


def region_prompt(region: str) -> str:
    """Build the region-conditioned transcription prompt.

    :param region: Natural-language region description.
    :type region: str
    :return: Prompt text.
    :rtype: str
    """
    return _REGION_PROMPT.format(region=region)


def load_bundles(ktiv_dir: Path) -> List[dict]:
    """Load API-shape transcription bundles, one per manuscript.

    Re-scrapes leave duplicate ``..._transcription(1).json`` siblings that a
    plain ``*_transcription.json`` glob would miss entirely; selection is
    delegated to :func:`ktiv_bundles.select_bundles` (API shape > newer file
    > richer content), so a re-scraped word-box bundle supersedes its older
    DOM-flat sibling.

    :param ktiv_dir: Directory holding ``*_transcription*.json`` files.
    :type ktiv_dir: Path
    :return: Winning bundles with ``sys_num`` attached, sorted by sys_num;
        manuscripts whose best bundle is DOM-shape are excluded.
    :rtype: List[dict]
    """
    out = []
    for sys_num, (path, doc) in sorted(select_bundles(ktiv_dir).items()):
        if bundle_shape(doc) != "api":
            continue
        doc["sys_num"] = bundle_sys_num(doc, path) or sys_num
        doc["bundle_file"] = path.name
        out.append(doc)
    return out


def find_zip_member(ktiv_dir: Path, sys_num: str, fl: str) -> Optional[Tuple[Path, str]]:
    """Locate the zip and member holding the page image for ``fl``.

    :param ktiv_dir: KTIV directory.
    :type ktiv_dir: Path
    :param sys_num: Manuscript system number.
    :type sys_num: str
    :param fl: Page FL identifier (e.g. ``FL202565086``).
    :type fl: str
    :return: (zip path, member name) or None.
    :rtype: Optional[Tuple[Path, str]]
    """
    for zpath in sorted(ktiv_dir.glob(f"*{sys_num}*_images*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                for name in zf.namelist():
                    if fl in name and name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                        return zpath, name
        except zipfile.BadZipFile:
            continue
    return None


def load_page_image(zpath: Path, member: str) -> PILImage.Image:
    """Read one page image out of its zip.

    :param zpath: Zip path.
    :type zpath: Path
    :param member: Member name.
    :type member: str
    :return: RGB image.
    :rtype: PILImage.Image
    """
    with zipfile.ZipFile(zpath) as zf:
        return PILImage.open(io.BytesIO(zf.read(member))).convert("RGB")


def scaled_copy(im: PILImage.Image, max_pixels: int) -> Tuple[PILImage.Image, float]:
    """Downscale so pixel count <= max_pixels (never upscale).

    :param im: Source image.
    :type im: PILImage.Image
    :param max_pixels: Pixel budget.
    :type max_pixels: int
    :return: (image, scale factor applied to coordinates).
    :rtype: Tuple[PILImage.Image, float]
    """
    px = im.width * im.height
    if px <= max_pixels:
        return im, 1.0
    s = (max_pixels / px) ** 0.5
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                     PILImage.LANCZOS), s


def benchmark_shingles(bench_path: Path, n: int = DECONTAM_SHINGLE) -> set:
    """Letter shingles of every verified-benchmark ground truth.

    :param bench_path: Verified benchmark JSON.
    :type bench_path: Path
    :param n: Shingle length in letters.
    :type n: int
    :return: Set of n-letter strings.
    :rtype: set
    """
    if not bench_path.exists():
        return set()
    spec = json.load(open(bench_path))
    shingles = set()
    for d in spec["docs"]:
        letters = "".join(_HEB_RE.findall(d.get("gt") or ""))
        shingles.update(letters[i:i + n] for i in range(len(letters) - n + 1))
    return shingles


def shingle_hits(text: str, shingles: set, n: int = DECONTAM_SHINGLE) -> int:
    """Count benchmark shingles present in a page text.

    :param text: Page text.
    :type text: str
    :param shingles: Benchmark shingle set.
    :type shingles: set
    :param n: Shingle length.
    :type n: int
    :return: Number of matching shingles.
    :rtype: int
    """
    letters = "".join(_HEB_RE.findall(text))
    return sum(letters[i:i + n] in shingles for i in range(len(letters) - n + 1))


def page_gate(page: Dict) -> Optional[str]:
    """Return a rejection reason for a reconstructed page, or None if it passes.

    :param page: Output of :func:`reconstruct_page`.
    :type page: Dict
    :return: Reason string or None.
    :rtype: Optional[str]
    """
    letters = hebrew_letters(page["text"])
    if letters < MIN_LETTERS:
        return "too_few_letters"
    if len(page["lines"]) < MIN_LINES:
        return "too_few_lines"
    total = page["n_words"] + page["n_gaps"]
    if total and page["n_gaps"] / total > MAX_GAP_SHARE:
        return "too_many_gaps"
    return None


def page_damage_share(text: str) -> float:
    """Share of a page's tokens that are damaged or editorially restored.

    A token counts as damaged when it is the gap token or contains any of
    :data:`DAMAGE_MARK_CHARS` (dots for illegible letters, brackets for
    restorations); each token counts once however many marks it carries.

    :param text: Reconstructed page text (whitespace-separated tokens).
    :type text: str
    :return: Damaged tokens / all tokens; 1.0 for a page without tokens.
    :rtype: float
    """
    tokens = text.split()
    if not tokens:
        return 1.0
    damaged = sum(1 for tok in tokens
                  if tok == GAP_TOKEN or any(ch in tok for ch in DAMAGE_MARK_CHARS))
    return damaged / len(tokens)


def short_fragment_reason(page_letters: int, damage_share: float,
                          n_lines: int) -> Optional[str]:
    """Return why a page is not a short fragment, or None when it is one.

    The family covers pages under the MIN_LETTERS gate only; pages at or
    above it belong to ``fragment_transcribe`` (``"not_short"``).

    :param page_letters: Hebrew letters on the page.
    :type page_letters: int
    :param damage_share: Output of :func:`page_damage_share`.
    :type damage_share: float
    :param n_lines: Reconstructed lines on the page.
    :type n_lines: int
    :return: ``"not_short"``, ``"too_few_letters"``, ``"too_few_lines"``,
        ``"too_damaged"`` or None.
    :rtype: Optional[str]
    """
    if page_letters >= MIN_LETTERS:
        return "not_short"
    if page_letters < SHORT_MIN_LETTERS:
        return "too_few_letters"
    if n_lines < SHORT_MIN_LINES:
        return "too_few_lines"
    if damage_share > SHORT_MAX_DAMAGE_SHARE:
        return "too_damaged"
    return None


def short_fragment_gate(page_letters: int, damage_share: float, n_lines: int) -> bool:
    """Whether a page under the MIN_LETTERS gate earns a ``page_short`` row.

    :param page_letters: Hebrew letters on the page.
    :type page_letters: int
    :param damage_share: Output of :func:`page_damage_share`.
    :type damage_share: float
    :param n_lines: Reconstructed lines on the page.
    :type n_lines: int
    :return: True for SHORT_MIN_LETTERS <= letters < MIN_LETTERS, at least
        SHORT_MIN_LINES lines and damage share <= SHORT_MAX_DAMAGE_SHARE.
    :rtype: bool
    """
    return short_fragment_reason(page_letters, damage_share, n_lines) is None


def image_frame_reason(width: int, height: int, lines: List[Dict]) -> Optional[str]:
    """Reject a page image that is too small or does not contain the page's boxes.

    :param width: Native page image width in pixels.
    :type width: int
    :param height: Native page image height in pixels.
    :type height: int
    :param lines: Reconstructed lines (``box`` in the native frame).
    :type lines: List[Dict]
    :return: ``"image_too_small"``, ``"boxes_out_of_frame"`` (2% slack) or None.
    :rtype: Optional[str]
    """
    if min(width, height) < MIN_IMAGE_SIDE_PX:
        return "image_too_small"
    max_x = max(l["box"][2] for l in lines)
    max_y = max(l["box"][3] for l in lines)
    if max_x > width * 1.02 or max_y > height * 1.02:
        return "boxes_out_of_frame"
    return None


def _row(image_path: Path, question: str, answer: str, task: str, section: str,
         stem: str, width: int, height: int) -> Dict:
    """Assemble one dataset row.

    :param image_path: Path of the image file.
    :type image_path: Path
    :param question: Prompt.
    :type question: str
    :param answer: Target text.
    :type answer: str
    :param task: Task family.
    :type task: str
    :param section: Region/section label.
    :type section: str
    :param stem: Row identifier.
    :type stem: str
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :return: Feature dict.
    :rtype: Dict
    """
    answer = unicodedata.normalize("NFC", answer).strip()
    return {
        "image": str(image_path), "question": question, "answer": answer,
        "task": task, "section": section, "stem": stem, "label_source": "ktiv_nli",
        "target_chars": len(answer), "target_tokens": 0,
        "image_width": width, "image_height": height,
    }


def region_candidates(page: Dict) -> List[Tuple[str, str, str]]:
    """Enumerate exact region tasks available for a page.

    :param page: Reconstructed page.
    :type page: Dict
    :return: List of (section label, region description, answer text).
    :rtype: List[Tuple[str, str, str]]
    """
    lines = page["lines"]
    n = len(lines)
    cands = []
    cands.append(("first_line", "the first line of text", lines[0]["text"]))
    cands.append(("last_line", "the last line of text", lines[-1]["text"]))
    for k in (2, 3, 4, 5):
        if n >= k + 2:
            cands.append((f"first_{k}_lines", f"the first {k} lines of text",
                          "\n".join(l["text"] for l in lines[:k])))
            cands.append((f"last_{k}_lines", f"the last {k} lines of text",
                          "\n".join(l["text"] for l in lines[-k:])))
    cols = [c for c in page["columns"] if c["lines"]]
    if len(cols) == 2:
        cands.append(("right_column", "the right-hand column",
                      "\n".join(l["text"] for l in cols[0]["lines"])))
        cands.append(("left_column", "the left-hand column",
                      "\n".join(l["text"] for l in cols[1]["lines"])))
    elif len(cols) == 1 and n >= 6:
        half = n // 2
        cands.append(("top_half", "the top half of the text",
                      "\n".join(l["text"] for l in lines[:half])))
        cands.append(("bottom_half", "the bottom half of the text",
                      "\n".join(l["text"] for l in lines[half:])))
    return cands


def _centre(line: Dict) -> float:
    """Vertical centre of a line box.

    :param line: Line entry with ``box``.
    :type line: Dict
    :return: y centre in pixels.
    :rtype: float
    """
    return (line["box"][1] + line["box"][3]) / 2


def crop_lines(im: PILImage.Image, col_lines: List[Dict], start: int, k: int,
               line_h: float, pad_x: float = 0.3) -> PILImage.Image:
    """Crop lines ``start .. start+k-1`` of one column between line centres.

    KTIV word boxes are taller than the line pitch (adjacent lines' boxes
    overlap vertically), so padding by box height leaks neighbouring lines
    into the crop and the target would then omit visible text.  Vertical
    bounds are therefore the midpoints between the outermost target lines and
    their neighbours (true inter-line boundaries); at the column's top/bottom
    edge, half a pitch (fallback 0.45 line heights) is used instead.

    :param im: Page image (native frame).
    :type im: PILImage.Image
    :param col_lines: All lines of the column, top to bottom.
    :type col_lines: List[Dict]
    :param start: Index of the first target line.
    :type start: int
    :param k: Number of target lines.
    :type k: int
    :param line_h: Median word-box height.
    :type line_h: float
    :param pad_x: Horizontal padding in line heights.
    :type pad_x: float
    :return: Cropped image.
    :rtype: PILImage.Image
    """
    seg = col_lines[start:start + k]
    centres = [_centre(l) for l in col_lines]
    pitches = [b - a for a, b in zip(centres, centres[1:]) if b > a]
    half_pitch = (sorted(pitches)[len(pitches) // 2] / 2) if pitches else 0.45 * line_h
    top_c, bot_c = centres[start], centres[start + k - 1]
    margin = 0.15 * 2 * half_pitch   # keep our own ascenders/descenders whole
    y0 = ((centres[start - 1] + top_c) / 2 if start > 0 else top_c - half_pitch) - margin
    y1 = ((bot_c + centres[start + k]) / 2 if start + k < len(col_lines)
          else bot_c + half_pitch) + margin
    x0 = min(l["box"][0] for l in seg) - pad_x * line_h
    x1 = max(l["box"][2] for l in seg) + pad_x * line_h
    # Boxes may poke past the frame within the page gate's 2% slack; a line
    # hugging an edge can then clamp inverted (right < left). Degenerate
    # intersections return a 1x1 crop, which every caller's minimum-size
    # check already rejects.
    left, top = int(max(x0, 0)), int(max(y0, 0))
    right, bottom = int(min(x1, im.width)), int(min(y1, im.height))
    if right <= left or bottom <= top:
        return im.crop((0, 0, 1, 1))
    return im.crop((left, top, right, bottom))


def norm_box(box: Tuple[float, float, float, float], width: int,
             height: int) -> List[int]:
    """Normalize a pixel box to 0-1000 integer coordinates.

    :param box: (x0, y0, x1, y1) in the original scan frame.
    :type box: Tuple[float, float, float, float]
    :param width: Original image width in pixels.
    :type width: int
    :param height: Original image height in pixels.
    :type height: int
    :return: ``[x0, y0, x1, y1]`` ints in 0-1000.
    :rtype: List[int]
    """
    x0, y0, x1, y1 = box
    # Clamp: KTIV boxes may poke past the frame within the page gate's 2%
    # slack; the model's coordinate space (and the site's loader) is 0-1000.
    def c(v: float, span: int) -> int:
        return max(0, min(1000, round(1000 * v / span)))
    return [c(x0, width), c(y0, height), c(x1, width), c(y1, height)]


def _positive(box: List[int]) -> bool:
    """True for a box with positive width and height after normalization."""
    return box[2] > box[0] and box[3] > box[1]


def locate_candidates(items: List[dict], page_text: str) -> List[Tuple[str, Tuple]]:
    """Phrases of consecutive words that are letter-unique on the page.

    Works from raw word geometry (fragments allowed — uniqueness is judged on
    the letters alone, so annotation word-splitting cannot break the match).

    :param items: AnnotationPage items of the page.
    :type items: List[dict]
    :param page_text: The page's reconstructed GT text.
    :type page_text: str
    :return: ``(phrase, pixel_box)`` pairs, at most one per line.
    :rtype: List[Tuple[str, Tuple]]
    """
    words = [w for w in extract_words(items)
             if not w["gap"] and len(_HEB_RE.findall(w["text"])) >= 2]
    if not words:
        return []
    page_letters = "".join(_HEB_RE.findall(page_text))
    lh = _median_height(words)
    out = []
    for col in split_columns(words, lh):
        for line in cluster_lines(col, lh):
            ws = [w for w in line if not w["gap"]
                  and len(_HEB_RE.findall(w["text"])) >= 2]
            found = None
            for i in range(len(ws) - 1):
                for span in (max(LOCATE_PHRASE_WORDS), min(LOCATE_PHRASE_WORDS)):
                    seg = ws[i:i + span]
                    if len(seg) < span:
                        continue
                    letters = "".join(_HEB_RE.findall("".join(w["text"] for w in seg)))
                    if len(letters) < LOCATE_MIN_PHRASE_LETTERS \
                            or page_letters.count(letters) != 1:
                        continue
                    box = (min(w["box"][0] for w in seg), min(w["box"][1] for w in seg),
                           max(w["box"][2] for w in seg), max(w["box"][3] for w in seg))
                    found = (" ".join(w["text"] for w in seg), box)
                    break
                if found:
                    break
            if found:
                out.append(found)
    return out


def word_candidates(items: List[dict], page_text: str) -> List[Tuple[str, Tuple]]:
    """Single words that are letter-unique on the page (word-level locate targets).

    Uniqueness is judged on Hebrew letters only, so annotation word-splitting
    cannot break the match; longer words are preferred (more likely to be a
    whole word rather than a fragment).

    :param items: AnnotationPage items of the page.
    :type items: List[dict]
    :param page_text: The page's reconstructed GT text.
    :type page_text: str
    :return: ``(word, pixel_box)`` pairs, longest first.
    :rtype: List[Tuple[str, Tuple]]
    """
    page_letters = "".join(_HEB_RE.findall(page_text))
    page_tokens = {"".join(_HEB_RE.findall(tok)) for tok in page_text.split()}
    out, seen = [], set()
    for w in extract_words(items):
        if w["gap"] or "[" in w["text"] or "]" in w["text"]:
            continue
        letters = "".join(_HEB_RE.findall(w["text"]))
        if len(letters) < LOCATE_WORD_MIN_LETTERS or letters in seen:
            continue
        # unique on the page AND a whole token of the reconstructed text (a
        # fragment that merge_line_words glued into a longer word is skipped)
        if page_letters.count(letters) != 1 or letters not in page_tokens:
            continue
        seen.add(letters)
        out.append((w["text"], tuple(w["box"])))
    out.sort(key=lambda t: -len("".join(_HEB_RE.findall(t[0]))))
    return out


def _eligible_columns(page: Dict) -> List[Tuple[str, List[Dict]]]:
    """Columns eligible for index-addressed line rows, with a reading-order label.

    ``page["columns"]`` is rightmost-first. Ordinal line references are only
    unambiguous on one- or two-column pages, so three+ columns are skipped.

    :param page: Reconstructed page.
    :type page: Dict
    :return: ``(column_clause, lines)`` per eligible column.
    :rtype: List[Tuple[str, List[Dict]]]
    """
    cols = [c for c in page["columns"] if c["lines"]]
    if len(cols) == 1:
        return [("", cols[0]["lines"])]
    if len(cols) == 2:
        return [(" of the right column", cols[0]["lines"]),
                (" of the left column", cols[1]["lines"])]
    return []


def line_index_rows(page: Dict, orig_w: int, orig_h: int, page_path: Path,
                    stem: str, page_w: int, page_h: int,
                    rng: random.Random) -> List[Dict]:
    """Index-addressed line rows: "line N of the {right|left} column" -> box + text.

    Forces the model to count lines from the page rather than emit a template
    box at the average pitch. The ordinal counts every reconstructed line of
    the column (from the top or the bottom, chosen at random); the answer's
    text target is required clean so both halves of the answer are reliable.

    :param page: Reconstructed page.
    :param orig_w: Original scan width (box frame).
    :param orig_h: Original scan height.
    :param page_path: Saved page JPEG path.
    :param stem: Row stem prefix.
    :param page_w: Saved image width.
    :param page_h: Saved image height.
    :param rng: Seeded RNG.
    :return: Up to :data:`LINE_INDEX_ROWS_PER_PAGE` rows.
    :rtype: List[Dict]
    """
    pool = []
    for clause, lines in _eligible_columns(page):
        for k, ln in enumerate(lines):
            if hebrew_letters(ln["text"]) < LINE_INDEX_MIN_LETTERS \
                    or GAP_TOKEN in ln["text"]:
                continue
            frm = rng.choice(("top", "bottom"))
            n = k + 1 if frm == "top" else len(lines) - k
            pool.append((clause, frm, n, ln))
    rng.shuffle(pool)
    rows = []
    pool = [x for x in pool if _positive(norm_box(x[3]["box"], orig_w, orig_h))]
    for j, (clause, frm, n, ln) in enumerate(pool[:LINE_INDEX_ROWS_PER_PAGE]):
        b = norm_box(ln["box"], orig_w, orig_h)
        answer = json.dumps({"bbox_2d": b, "text": ln["text"]}, ensure_ascii=False)
        rows.append(_row(page_path,
                         _LINE_INDEX_PROMPT.format(n=n, col=clause, frm=frm),
                         answer, "line_index", "line", f"{stem}_li{j}",
                         page_w, page_h))
    return rows


def line_of_phrase_row(page: Dict, cands: List[Tuple[str, Tuple]], orig_w: int,
                       orig_h: int, page_path: Path, stem: str, page_w: int,
                       page_h: int, rng: random.Random) -> Optional[Dict]:
    """Phrase -> host-line box + text (grounded upgrade of the find-line QA).

    :param page: Reconstructed page.
    :param cands: Unique phrases from :func:`locate_candidates`.
    :param orig_w: Original scan width.
    :param orig_h: Original scan height.
    :param page_path: Saved page JPEG path.
    :param stem: Row stem prefix.
    :param page_w: Saved image width.
    :param page_h: Saved image height.
    :param rng: Seeded RNG.
    :return: One row, or None when no phrase maps cleanly to a host line.
    :rtype: Optional[Dict]
    """
    if not cands:
        return None
    phrase, box = rng.choice(cands)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    host = min(page["lines"],
               key=lambda ln: abs((ln["box"][1] + ln["box"][3]) / 2 - cy)
               if ln["box"][0] <= cx <= ln["box"][2] else 1e12)
    letters = "".join(_HEB_RE.findall(phrase))
    if letters not in "".join(_HEB_RE.findall(host["text"])):
        return None
    b = norm_box(host["box"], orig_w, orig_h)
    if not _positive(b) or not host["text"].strip():
        return None
    answer = json.dumps({"bbox_2d": b, "text": host["text"]}, ensure_ascii=False)
    return _row(page_path, _LINE_OF_PHRASE_BOX_PROMPT.format(phrase=phrase),
                answer, "line_of_phrase", "line", f"{stem}_lop", page_w, page_h)


def _line_band_box(im: PILImage.Image, col_lines: List[Dict], start: int, k: int,
                   line_h: float, pad_x: float = 0.3) -> Tuple[int, int, int, int]:
    """Pixel bounds of a clean crop around lines ``start .. start+k-1`` of a column.

    Same inter-line-midpoint boundaries as :func:`crop_lines`, returned as
    coordinates so the caller can re-normalize the kept lines' boxes.

    :param im: Native page image.
    :param col_lines: All lines of the column, top to bottom.
    :param start: First target line index.
    :param k: Number of target lines.
    :param line_h: Median word-box height.
    :param pad_x: Horizontal padding in line heights.
    :return: ``(left, top, right, bottom)`` clamped to the image.
    :rtype: Tuple[int, int, int, int]
    """
    seg = col_lines[start:start + k]
    centres = [_centre(l) for l in col_lines]
    pitches = sorted(b - a for a, b in zip(centres, centres[1:]) if b > a)
    half_pitch = (pitches[len(pitches) // 2] / 2) if pitches else 0.45 * line_h
    top_c, bot_c = centres[start], centres[start + k - 1]
    margin = 0.15 * 2 * half_pitch
    y0 = ((centres[start - 1] + top_c) / 2 if start > 0 else top_c - half_pitch) - margin
    y1 = ((bot_c + centres[start + k]) / 2 if start + k < len(col_lines)
          else bot_c + half_pitch) + margin
    x0 = min(l["box"][0] for l in seg) - pad_x * line_h
    x1 = max(l["box"][2] for l in seg) + pad_x * line_h
    return (int(max(x0, 0)), int(max(y0, 0)),
            int(min(x1, im.width)), int(min(y1, im.height)))


def _norm_box_crop(box: Tuple[float, float, float, float], left: int, top: int,
                   width: int, height: int) -> List[int]:
    """Normalize a scan-frame box to 0-1000 within a crop window (clamped)."""
    def c(v: float, span: int) -> int:
        return max(0, min(1000, round(1000 * v / span))) if span > 0 else 0
    x0, y0, x1, y1 = box
    return [c(x0 - left, width), c(y0 - top, height),
            c(x1 - left, width), c(y1 - top, height)]


def grounded_crop_rows(page: Dict, im: PILImage.Image, ms_dir: Path, fl: str,
                       stem: str, rng: random.Random) -> List[Dict]:
    """Detect-then-read rows on a random per-column line band (anti-template aug).

    A mid-column run of lines becomes its own image with boxes re-normalized to
    the crop, so a given line's y-position and the page margin vary between
    examples and the box can no longer be predicted from the line's ordinal.

    :param page: Reconstructed page.
    :param im: Native page image.
    :param ms_dir: Manuscript image directory (crop is saved here).
    :param fl: Page image id (filename component).
    :param stem: Row stem prefix.
    :param rng: Seeded RNG.
    :return: Up to :data:`GROUNDED_CROP_ROWS_PER_PAGE` rows.
    :rtype: List[Dict]
    """
    if page.get("mode") != "geometry":
        return []
    line_h = page["line_h"] or 1.0
    cols = [c["lines"] for c in page["columns"]
            if len(c["lines"]) >= GROUNDED_CROP_LINES[0]]
    rng.shuffle(cols)
    rows, made = [], 0
    for ci, lines in enumerate(cols):
        if made >= GROUNDED_CROP_ROWS_PER_PAGE:
            break
        k = min(len(lines), rng.randint(*GROUNDED_CROP_LINES))
        start = rng.randint(0, len(lines) - k)
        seg = lines[start:start + k]
        left, top, right, bottom = _line_band_box(im, lines, start, k, line_h)
        if right - left < MIN_IMAGE_SIDE_PX or bottom - top < MIN_IMAGE_SIDE_PX:
            continue
        crop, _ = scaled_copy(im.crop((left, top, right, bottom)), MAX_PAGE_PIXELS)
        cw, ch = right - left, bottom - top
        payload = [{"bbox_2d": _norm_box_crop(ln["box"], left, top, cw, ch),
                    "text": ln["text"]} for ln in seg]
        payload = [e for e in payload if _positive(e["bbox_2d"]) and e["text"].strip()]
        if not payload:
            continue
        cpath = ms_dir / f"{fl}_gcrop{ci}_{start}_{k}.jpg"
        crop.save(cpath, "JPEG", quality=90)
        rows.append(_row(cpath, _GROUNDED_DETECT_PROMPT,
                         json.dumps(payload, ensure_ascii=False),
                         "grounded_crop", "band", f"{stem}_gcrop{ci}",
                         crop.width, crop.height))
        made += 1
    return rows


def grounding_rows(page: Dict, items: List[dict], page_path: Path, stem: str,
                   orig_w: int, orig_h: int, page_w: int, page_h: int,
                   rng: random.Random) -> List[Dict]:
    """Grounding + layout-QA rows for one geometry-mode page.

    v2.0 families: up to :data:`LOCATE_ROWS_PER_PAGE` ``locate``,
    :data:`READBOX_ROWS_PER_PAGE` ``read_box``, one ``layout_qa``, and (for a
    :data:`GROUNDED_PAGE_FRACTION` sample) one text-first ``grounded_page`` row
    (kept for eval comparability with v1.9/v2.0).

    v2.1 direct-grounding families (target the template-box failure):
    ``locate_word`` (unique word -> box), ``read_box_word`` (word box -> word),
    ``line_index`` (ordinal line -> box + text), ``line_of_phrase`` (phrase ->
    host-line box + text), and ``grounded_detect`` (whole page, bbox emitted
    before text). ``grounded_crop`` bands are added separately in
    :func:`build_rows` (they need the native image). Pages reconstructed
    without trustworthy geometry (served-breakline fallback) emit nothing.

    :param page: Output of :func:`reconstruct_page`.
    :type page: Dict
    :param items: The page's AnnotationPage items (word geometry source).
    :type items: List[dict]
    :param page_path: Saved page JPEG path.
    :type page_path: Path
    :param stem: Row stem prefix.
    :type stem: str
    :param orig_w: ORIGINAL scan width (the boxes' frame).
    :type orig_w: int
    :param orig_h: Original scan height.
    :type orig_h: int
    :param page_w: Saved (rescaled) page image width.
    :type page_w: int
    :param page_h: Saved page image height.
    :type page_h: int
    :param rng: Seeded RNG.
    :type rng: random.Random
    :return: Dataset rows.
    :rtype: List[Dict]
    """
    if page.get("mode") != "geometry":
        return []
    rows: List[Dict] = []
    cands = locate_candidates(items, page["text"])
    rng.shuffle(cands)
    for j, (phrase, box) in enumerate(cands[:LOCATE_ROWS_PER_PAGE]):
        answer = json.dumps({"bbox_2d": norm_box(box, orig_w, orig_h)})
        rows.append(_row(page_path, _LOCATE_PROMPT.format(phrase=phrase), answer,
                         "locate", "phrase", f"{stem}_loc{j}", page_w, page_h))
    readable = [ln for ln in page["lines"]
                if hebrew_letters(ln["text"]) >= READBOX_MIN_LETTERS
                and GAP_TOKEN not in ln["text"]]
    for j, ln in enumerate(rng.sample(readable,
                                      min(READBOX_ROWS_PER_PAGE, len(readable)))):
        b = norm_box(ln["box"], orig_w, orig_h)
        rows.append(_row(page_path,
                         _READBOX_PROMPT.format(x0=b[0], y0=b[1], x1=b[2], y1=b[3]),
                         ln["text"], "read_box", "line", f"{stem}_rb{j}",
                         page_w, page_h))
    qa = _layout_qa_row(page, cands, page_path, stem, page_w, page_h, rng)
    if qa is not None:
        rows.append(qa)
    if 4 <= len(page["lines"]) <= GROUNDED_PAGE_MAX_LINES \
            and rng.random() < GROUNDED_PAGE_FRACTION:
        payload = [{"text": ln["text"],
                    "bbox_2d": norm_box(ln["box"], orig_w, orig_h)}
                   for ln in page["lines"]]
        payload = [e for e in payload if _positive(e["bbox_2d"]) and e["text"].strip()]
        rows.append(_row(page_path, _GROUNDED_PROMPT,
                         json.dumps(payload, ensure_ascii=False),
                         "grounded_page", "page", f"{stem}_grounded",
                         page_w, page_h))
    # --- v2.1 direct-grounding families ---
    words = word_candidates(items, page["text"])
    rng.shuffle(words)
    for j, (word, box) in enumerate(words[:LOCATE_WORD_ROWS_PER_PAGE]):
        answer = json.dumps({"bbox_2d": norm_box(box, orig_w, orig_h)})
        rows.append(_row(page_path, _LOCATE_WORD_PROMPT.format(word=word), answer,
                         "locate_word", "word", f"{stem}_lw{j}", page_w, page_h))
    for j, (word, box) in enumerate(words[LOCATE_WORD_ROWS_PER_PAGE:
                                          LOCATE_WORD_ROWS_PER_PAGE + READBOX_WORD_ROWS_PER_PAGE]):
        b = norm_box(box, orig_w, orig_h)
        rows.append(_row(page_path,
                         _READBOX_WORD_PROMPT.format(x0=b[0], y0=b[1], x1=b[2], y1=b[3]),
                         word, "read_box_word", "word", f"{stem}_rbw{j}",
                         page_w, page_h))
    rows.extend(line_index_rows(page, orig_w, orig_h, page_path, stem,
                                page_w, page_h, rng))
    lop = line_of_phrase_row(page, cands, orig_w, orig_h, page_path, stem,
                             page_w, page_h, rng)
    if lop is not None:
        rows.append(lop)
    if 4 <= len(page["lines"]) <= GROUNDED_DETECT_MAX_LINES:
        payload = [{"bbox_2d": norm_box(ln["box"], orig_w, orig_h), "text": ln["text"]}
                   for ln in page["lines"]]
        payload = [e for e in payload if _positive(e["bbox_2d"]) and e["text"].strip()]
        rows.append(_row(page_path, _GROUNDED_DETECT_PROMPT,
                         json.dumps(payload, ensure_ascii=False),
                         "grounded_detect", "page", f"{stem}_gdet",
                         page_w, page_h))
    return rows


def _layout_qa_row(page: Dict, locate_cands: List[Tuple[str, Tuple]],
                   page_path: Path, stem: str, page_w: int, page_h: int,
                   rng: random.Random) -> Optional[Dict]:
    """One templated layout-QA row (answers derivable from reconstruction).

    :param page: Output of :func:`reconstruct_page`.
    :type page: Dict
    :param locate_cands: Unique phrases (reused for the find-line template).
    :type locate_cands: List[Tuple[str, Tuple]]
    :param page_path: Saved page JPEG path.
    :type page_path: Path
    :param stem: Row stem prefix.
    :type stem: str
    :param page_w: Saved page image width.
    :type page_w: int
    :param page_h: Saved page image height.
    :type page_h: int
    :param rng: Seeded RNG.
    :type rng: random.Random
    :return: One row, or None when no template applies.
    :rtype: Optional[Dict]
    """
    templates = ["columns", "edge_line"]
    if locate_cands:
        templates.append("find_line")
    kind = rng.choice(templates)
    if kind == "columns":
        q, a = _QA_COLUMNS_PROMPT, str(len(page["columns"]))
    elif kind == "edge_line":
        which = rng.choice(("first", "last"))
        line = page["lines"][0 if which == "first" else -1]
        q, a = _QA_FIRST_LINE_PROMPT.format(which=which), line["text"]
    else:
        phrase, box = rng.choice(locate_cands)
        cy = (box[1] + box[3]) / 2
        host = min(page["lines"],
                   key=lambda ln: abs((ln["box"][1] + ln["box"][3]) / 2 - cy)
                   if ln["box"][0] <= (box[0] + box[2]) / 2 <= ln["box"][2] else 1e12)
        letters = "".join(_HEB_RE.findall(phrase))
        if letters not in "".join(_HEB_RE.findall(host["text"])):
            return None
        q, a = _QA_FIND_LINE_PROMPT.format(phrase=phrase), host["text"]
    return _row(page_path, q, a, "layout_qa", kind, f"{stem}_qa", page_w, page_h)


def exclude_benchmark_manuscripts(bundles: List[dict],
                                  ktiv_dir: Path) -> Tuple[List[dict], Dict]:
    """Drop manuscripts that belong to any eval benchmark.

    Two id-level layers on top of the text-shingle check in
    :func:`build_rows` (which catches page-text overlap but not a benchmark
    manuscript's OTHER pages — same-hand leakage):

    * every religious-benchmark manuscript, by exact sys_num;
    * every manuscript whose loose shelfmark key appears in the frozen
      PGP-131 benchmark (same physical fragment reachable through two
      sources), via :class:`DecontamGate`'s inventory.

    :param bundles: API-shape bundles with ``sys_num`` attached.
    :type bundles: List[dict]
    :param ktiv_dir: KTIV raw directory (metadata JSONs for shelfmarks).
    :type ktiv_dir: Path
    :return: (kept bundles, counts-by-reason dict).
    :rtype: Tuple[List[dict], Dict]
    """
    from src.datasets.evaluations.helper_eval_scripts.decontam_gate import DecontamGate

    religious_sys = set()
    if RELIGIOUS_BENCH_PATH.exists():
        religious_sys = {d["sys_num"] for d in
                         json.load(open(RELIGIOUS_BENCH_PATH))["docs"]}
    shelf_by_sys: Dict[str, str] = {}
    for f in ktiv_dir.glob("ktiv_*.json"):
        if "_transcription" in f.name:
            continue
        try:
            meta = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        s = str(meta.get("sys_num")
                or (meta.get("shelfmarks") or {}).get("system_no") or "")
        sm = meta.get("shelf_mark") or (meta.get("shelfmarks") or {}).get("shelf_mark") or ""
        if s and sm and s not in shelf_by_sys:
            shelf_by_sys[s] = sm
    gate = DecontamGate()
    kept, excl = [], Counter()
    for doc in bundles:
        if doc["sys_num"] in religious_sys:
            excl["religious_benchmark_ms"] += 1
            continue
        key = gate.loose_key(shelf_by_sys.get(doc["sys_num"], ""))
        if key and key in gate.bench_keys:
            excl["pgp131_shelfmark"] += 1
            continue
        kept.append(doc)
    return kept, dict(excl)


def short_fragment_row(page: Dict, im: PILImage.Image, ms_dir: Path, fl: str,
                       stem: str) -> Dict:
    """Save a short fragment's page image and build its ``page_short`` row.

    Same prompt, target, label source and page-image handling as the
    ``fragment_transcribe`` row; only the task (``page_short``) and section
    (``page``) differ, so the notebook can weight the family on its own.

    :param page: Reconstructed page that passed :func:`short_fragment_gate`.
    :type page: Dict
    :param im: Native page image (already frame-checked).
    :type im: PILImage.Image
    :param ms_dir: Manuscript image directory (page JPEG is saved here).
    :type ms_dir: Path
    :param fl: Page image id (filename component).
    :type fl: str
    :param stem: Row stem (``ktiv_<sys_num>_<fl>``).
    :type stem: str
    :return: One dataset row.
    :rtype: Dict
    """
    page_im, _ = scaled_copy(im, MAX_PAGE_PIXELS)
    page_path = ms_dir / f"{fl}.jpg"
    if not page_path.exists():
        page_im.save(page_path, "JPEG", quality=90)
    return _row(page_path, FRAGMENT_TRANSCRIBE_PROMPT, page["text"], "page_short",
                "page", stem, page_im.width, page_im.height)


def build_rows(bundles: List[dict], ktiv_dir: Path, images_dir: Path,
               shingles: set, rng: random.Random, limit: int = 0,
               short_fragments: bool = True) -> Tuple[List[Dict], Dict]:
    """Process bundles into dataset rows plus a stats dict.

    Short-fragment pages (see :func:`short_fragment_gate`) take part in the
    manuscript-level decontamination like any other page but draw nothing
    from ``rng``, so every other family's sampling is identical with and
    without them.

    :param bundles: API-shape bundles.
    :type bundles: List[dict]
    :param ktiv_dir: KTIV directory (zips).
    :type ktiv_dir: Path
    :param images_dir: Where page/crop JPEGs are written.
    :type images_dir: Path
    :param shingles: Benchmark shingle set for decontamination.
    :type shingles: set
    :param rng: Seeded RNG for task sampling.
    :type rng: random.Random
    :param limit: Max manuscripts (0 = all).
    :type limit: int
    :param short_fragments: Emit the ``page_short`` family.
    :type short_fragments: bool
    :return: (rows, stats).
    :rtype: Tuple[List[Dict], Dict]
    """
    stats = Counter()
    rows: List[Dict] = []
    contaminated: List[str] = []
    short_ms = set()
    for i, doc in enumerate(bundles):
        if limit and i >= limit:
            break
        sys_num = doc["sys_num"]
        pages = []
        shorts = []
        for p in doc.get("pages") or []:
            items = ((p.get("annotation_page") or {}).get("items")) or []
            page = reconstruct_page(items)
            reason = page_gate(page) if page["lines"] else "empty"
            stats[f"page_{reason or 'pass'}"] += 1
            if reason == "too_few_letters" and short_fragments:
                short_reason = short_fragment_reason(
                    hebrew_letters(page["text"]), page_damage_share(page["text"]),
                    len(page["lines"]))
                stats[f"page_short_{short_reason or 'pass'}"] += 1
                if short_reason is None:
                    shorts.append((p.get("fl") or "", page))
            if reason:
                continue
            pages.append((p.get("fl") or "", page, items))
        if not pages and not shorts:
            continue
        # Decontamination at manuscript level; short pages count like any other.
        normal_hit = bool(shingles) and any(
            shingle_hits(pg["text"], shingles) >= DECONTAM_MIN_HITS for _, pg, _i in pages)
        short_hit = bool(shingles) and any(
            shingle_hits(pg["text"], shingles) >= DECONTAM_MIN_HITS for _, pg in shorts)
        if normal_hit or short_hit:
            contaminated.append(sys_num)
            if pages:
                stats["ms_contaminated"] += 1
                if not normal_hit:
                    stats["ms_contaminated_by_short"] += 1
            else:
                stats["ms_short_only_contaminated"] += 1
            if shorts:
                stats["page_short_contaminated"] += len(shorts)
            continue
        stats["ms_kept" if pages else "ms_short_only"] += 1
        for fl, page, items in pages:
            found = find_zip_member(ktiv_dir, sys_num, fl)
            if not found:
                stats["page_no_image"] += 1
                continue
            im = load_page_image(*found)
            # Sanity: image big enough, boxes inside its frame (2% slack).
            frame = image_frame_reason(im.width, im.height, page["lines"])
            if frame:
                stats[f"page_{frame}"] += 1
                continue
            stem = f"ktiv_{sys_num}_{fl}"
            ms_dir = images_dir / sys_num
            ms_dir.mkdir(parents=True, exist_ok=True)

            page_im, _ = scaled_copy(im, MAX_PAGE_PIXELS)
            page_path = ms_dir / f"{fl}.jpg"
            if not page_path.exists():
                page_im.save(page_path, "JPEG", quality=90)
            rows.append(_row(page_path, FRAGMENT_TRANSCRIBE_PROMPT, page["text"],
                             "fragment_transcribe", "ktiv_page", stem,
                             page_im.width, page_im.height))
            stats["rows_page"] += 1

            for g_row in grounding_rows(page, items, page_path, stem,
                                        im.width, im.height,
                                        page_im.width, page_im.height, rng):
                rows.append(g_row)
                stats[f"rows_{g_row['task']}"] += 1

            for c_row in grounded_crop_rows(page, im, ms_dir, fl, stem, rng):
                rows.append(c_row)
                stats[f"rows_{c_row['task']}"] += 1

            # Column tasks are the most valuable region rows and only exist on
            # multi-column pages, so they are always emitted; the remaining
            # region types are sampled REGION_ROWS_PER_PAGE per page.
            cands = region_candidates(page)
            column_cands = [c for c in cands if c[0] in ("right_column", "left_column")]
            other_cands = [c for c in cands if c[0] not in ("right_column", "left_column")]
            chosen = column_cands + rng.sample(
                other_cands, min(REGION_ROWS_PER_PAGE, len(other_cands)))
            for label, desc, answer in chosen:
                rows.append(_row(page_path, region_prompt(desc), answer,
                                 "region_transcribe", label, f"{stem}_{label}",
                                 page_im.width, page_im.height))
                stats["rows_region"] += 1
                if label.endswith("_column"):
                    stats["rows_region_column"] += 1

            lh = page["line_h"]
            for c_idx, col in enumerate(page["columns"]):
                ln = col["lines"]
                if len(ln) >= SECTION_LINES[0] and stats[f"_sec_{stem}"] < SECTION_ROWS_PER_PAGE:
                    k = min(len(ln), rng.randint(*SECTION_LINES))
                    start = rng.randint(0, len(ln) - k)
                    seg = ln[start:start + k]
                    crop = crop_lines(im, ln, start, k, lh)
                    if min(crop.size) >= MIN_IMAGE_SIDE_PX:
                        crop, _ = scaled_copy(crop, MAX_PAGE_PIXELS)
                        cpath = ms_dir / f"{fl}_sec{c_idx}_{start}_{k}.jpg"
                        crop.save(cpath, "JPEG", quality=90)
                        rows.append(_row(cpath, FRAGMENT_TRANSCRIBE_PROMPT,
                                         "\n".join(l["text"] for l in seg),
                                         "section_transcribe", f"lines_{k}",
                                         f"{stem}_sec{c_idx}_{start}", crop.width, crop.height))
                        stats["rows_section"] += 1
                        stats[f"_sec_{stem}"] += 1
            eligible = [(col["lines"], idx) for col in page["columns"]
                        for idx, l in enumerate(col["lines"])
                        if l["n_words"] >= LINE_MIN_WORDS
                        and (l["box"][2] - l["box"][0]) >= LINE_MIN_WIDTH_PX
                        and GAP_TOKEN not in l["text"]]
            for j, (col_lines, idx) in enumerate(
                    rng.sample(eligible, min(LINE_ROWS_PER_PAGE, len(eligible)))):
                line = col_lines[idx]
                crop = crop_lines(im, col_lines, idx, 1, lh, pad_x=0.2)
                if crop.height < 24 or crop.width < LINE_MIN_WIDTH_PX:
                    continue
                cpath = ms_dir / f"{fl}_line{j}.jpg"
                crop.save(cpath, "JPEG", quality=92)
                rows.append(_row(cpath, FRAGMENT_TRANSCRIBE_PROMPT, line["text"],
                                 "line_transcribe", "line", f"{stem}_line{j}",
                                 crop.width, crop.height))
                stats["rows_line"] += 1
        for fl, page in shorts:
            found = find_zip_member(ktiv_dir, sys_num, fl)
            if not found:
                stats["page_short_no_image"] += 1
                continue
            im = load_page_image(*found)
            frame = image_frame_reason(im.width, im.height, page["lines"])
            if frame:
                stats[f"page_short_{frame}"] += 1
                continue
            ms_dir = images_dir / sys_num
            ms_dir.mkdir(parents=True, exist_ok=True)
            rows.append(short_fragment_row(page, im, ms_dir, fl, f"ktiv_{sys_num}_{fl}"))
            stats["rows_page_short"] += 1
            short_ms.add(sys_num)
        if (i + 1) % 50 == 0:
            logger.info("processed %d manuscripts, %d rows", i + 1, len(rows))
    stats = {k: v for k, v in stats.items() if not k.startswith("_")}
    if short_fragments:
        stats["ms_page_short"] = len(short_ms)
    stats["contaminated_sys_nums"] = contaminated
    return rows, stats


def stem_manuscript(stem: str) -> str:
    """Manuscript sys_num of a row stem (``ktiv_<sys_num>_<fl>...``).

    :param stem: Row stem.
    :type stem: str
    :return: The sys_num.
    :rtype: str
    """
    return stem.split("_")[1]


def load_val_pin(path: Path) -> Tuple[set, Optional[set]]:
    """Read a val-manuscript pin file.

    Accepts a JSON list of sys_nums (exactly those go to val, everything else
    to train) or the object a build writes to ``val_manuscripts.json``:
    ``{"val_manuscripts": [...], "known_manuscripts": [...], ...}``, where
    ``known_manuscripts`` lists every manuscript of the build the pin came
    from, so manuscripts outside it are new and can be drawn into val.

    :param path: Pin file.
    :type path: Path
    :return: (pinned val sys_nums, known sys_nums or None).
    :rtype: Tuple[set, Optional[set]]
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return {str(s) for s in data}, None
    known = data.get("known_manuscripts")
    return ({str(s) for s in data["val_manuscripts"]},
            None if known is None else {str(s) for s in known})


def choose_val_manuscripts(manuscripts: Iterable[str], val_fraction: float, seed: int,
                           pinned: Optional[set] = None,
                           known: Optional[set] = None) -> set:
    """Pick the validation manuscripts.

    Without a pin, a seeded shuffle of all manuscripts takes ``val_fraction``
    (the historical rule; any change to the manuscript set reshuffles it).
    With a pin, the pinned manuscripts that are present form val, every
    ``known`` manuscript outside the pin stays in train, and a seeded
    ``val_fraction`` of the manuscripts new relative to ``known`` joins val,
    so a rebuild never moves a manuscript between splits.

    :param manuscripts: sys_nums present in this build (duplicates allowed).
    :type manuscripts: Iterable[str]
    :param val_fraction: Fraction of (new) manuscripts for validation.
    :type val_fraction: float
    :param seed: RNG seed.
    :type seed: int
    :param pinned: sys_nums forced into val, or None for the unpinned rule.
    :type pinned: Optional[set]
    :param known: Manuscripts of the build the pin came from; None = no
        manuscript counts as new (the pin is the whole val set).
    :type known: Optional[set]
    :return: Validation sys_nums.
    :rtype: set
    """
    ms = sorted(set(manuscripts))
    if pinned is None:
        rng = random.Random(seed)
        rng.shuffle(ms)
        return set(ms[:max(1, int(len(ms) * val_fraction))])
    val = {m for m in ms if m in pinned}
    new = [m for m in ms if known is not None and m not in known and m not in pinned]
    if new:
        rng = random.Random(seed)
        rng.shuffle(new)
        val.update(new[:max(1, int(len(new) * val_fraction))])
    return val


def split_by_manuscript(rows: List[Dict], val_fraction: float, seed: int,
                        pinned: Optional[set] = None,
                        known: Optional[set] = None) -> Tuple[List[Dict], List[Dict]]:
    """Split rows into train/val by manuscript so no page leaks across.

    All rows of a manuscript (every family, ``page_short`` included) land in
    the same split; see :func:`choose_val_manuscripts` for the val rule.

    :param rows: Dataset rows (stem starts with ``ktiv_<sys_num>_``).
    :type rows: List[Dict]
    :param val_fraction: Fraction of manuscripts for validation.
    :type val_fraction: float
    :param seed: RNG seed.
    :type seed: int
    :param pinned: sys_nums forced into val (``--val-manuscripts``), or None.
    :type pinned: Optional[set]
    :param known: Manuscripts of the pin's source build, or None.
    :type known: Optional[set]
    :return: (train rows, val rows).
    :rtype: Tuple[List[Dict], List[Dict]]
    """
    val_ms = choose_val_manuscripts((stem_manuscript(r["stem"]) for r in rows),
                                    val_fraction, seed, pinned, known)
    train = [r for r in rows if stem_manuscript(r["stem"]) not in val_ms]
    val = [r for r in rows if stem_manuscript(r["stem"]) in val_ms]
    return train, val


def val_pin_record(train: List[Dict], val: List[Dict], provenance: Dict) -> Dict:
    """The ``val_manuscripts.json`` a build writes, usable as its successor's pin.

    :param train: Train rows.
    :type train: List[Dict]
    :param val: Val rows.
    :type val: List[Dict]
    :param provenance: How this val set was chosen.
    :type provenance: Dict
    :return: ``{"val_manuscripts", "known_manuscripts", "provenance"}``.
    :rtype: Dict
    """
    val_ms = {stem_manuscript(r["stem"]) for r in val}
    known = val_ms | {stem_manuscript(r["stem"]) for r in train}
    return {"val_manuscripts": sorted(val_ms), "known_manuscripts": sorted(known),
            "provenance": provenance}


def build(ktiv_dir: Path, images_dir: Path, output_dir: Path, limit: int = 0,
          push_to_hub: Optional[str] = None, short_fragments: bool = True,
          val_manuscripts: Optional[Path] = None) -> DatasetDict:
    """Build, save and optionally push the dataset.

    Always writes ``output_dir/val_manuscripts.json`` (val + all manuscripts),
    which the next rebuild passes as ``val_manuscripts`` to keep the split.

    :param ktiv_dir: KTIV raw directory.
    :type ktiv_dir: Path
    :param images_dir: Output directory for JPEGs.
    :type images_dir: Path
    :param output_dir: ``save_to_disk`` destination.
    :type output_dir: Path
    :param limit: Max manuscripts (0 = all).
    :type limit: int
    :param push_to_hub: Private hub repo id, or None.
    :type push_to_hub: Optional[str]
    :param short_fragments: Emit the ``page_short`` family.
    :type short_fragments: bool
    :param val_manuscripts: Val pin file (see :func:`load_val_pin`), or None
        for the historical seeded shuffle.
    :type val_manuscripts: Optional[Path]
    :return: The DatasetDict.
    :rtype: DatasetDict
    :raises ValueError: When the pin leaves the val split empty.
    """
    pinned, known = load_val_pin(val_manuscripts) if val_manuscripts else (None, None)
    bundles = load_bundles(ktiv_dir)
    logger.info("API-shape bundles: %d", len(bundles))
    bundles, excl = exclude_benchmark_manuscripts(bundles, ktiv_dir)
    logger.info("benchmark exclusions: %s", excl)
    logger.info("short_fragment family: %s", "on" if short_fragments else "off")
    shingles = benchmark_shingles(BENCH_PATH) | benchmark_shingles(RELIGIOUS_BENCH_PATH)
    rows, stats = build_rows(bundles, ktiv_dir, images_dir, shingles,
                             random.Random(SPLIT_SEED), limit, short_fragments)
    stats["benchmark_exclusions"] = excl
    train, val = split_by_manuscript(rows, VAL_FRACTION, SPLIT_SEED, pinned, known)
    if pinned is not None:
        present = {stem_manuscript(r["stem"]) for r in rows}
        new = present - (known or set()) - pinned if known is not None else set()
        stats["val_pin"] = {
            "file": str(val_manuscripts), "pinned": len(pinned),
            "pinned_present": len(pinned & present), "new_manuscripts": len(new),
            "new_drawn_to_val": len({stem_manuscript(r["stem"]) for r in val} & new)}
        logger.info("val pin: %s", stats["val_pin"])
        if not val:
            raise ValueError(f"val pin {val_manuscripts} matches no manuscript of this build")
    stats.update(n_rows=len(rows), n_train=len(train), n_val=len(val),
                 letters_page_rows=sum(hebrew_letters(r["answer"]) for r in rows
                                       if r["task"] == "fragment_transcribe"))
    if short_fragments:
        stats["letters_page_short_rows"] = sum(hebrew_letters(r["answer"]) for r in rows
                                               if r["task"] == "page_short")
    logger.info("stats: %s", json.dumps({k: v for k, v in stats.items()
                                        if k != "contaminated_sys_nums"}, indent=1))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    rule = ("pinned: val = pinned manuscripts present + seeded val_fraction of new ones"
            if pinned is not None else "seeded shuffle of all manuscripts")
    with open(output_dir / "val_manuscripts.json", "w") as fh:
        json.dump(val_pin_record(train, val, {
            "rule": rule, "pin_file": str(val_manuscripts) if val_manuscripts else None,
            "seed": SPLIT_SEED, "val_fraction": VAL_FRACTION}), fh, indent=1)
    dsd = DatasetDict({
        "train": Dataset.from_list(train, features=FEATURES),
        "val": Dataset.from_list(val, features=FEATURES),
    })
    dsd.save_to_disk(str(output_dir))
    logger.info("saved to %s", output_dir)
    if push_to_hub:
        dsd.push_to_hub(push_to_hub, private=True)
        logger.info("pushed to %s (private)", push_to_hub)
    return dsd


def main() -> None:
    """CLI entry point."""
    import dotenv

    dotenv.load_dotenv(_REPO / ".env")
    os.environ.setdefault("HF_TOKEN", os.environ.get("HF1_TOKEN", ""))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ktiv-dir", type=Path, default=KTIV_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument("--short-fragments", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="emit the page_short family for lightly damaged pages "
                             "under the MIN_LETTERS gate (default: on)")
    parser.add_argument("--val-manuscripts", type=Path, default=None,
                        help="pin the val split: a JSON list of sys_nums, or a previous "
                             "build's val_manuscripts.json (its val stays val, its other "
                             "manuscripts stay train, VAL_FRACTION of new ones join val)")
    args = parser.parse_args()
    build(args.ktiv_dir, args.images_dir, args.output_dir, args.limit, args.push_to_hub,
          args.short_fragments, args.val_manuscripts)


if __name__ == "__main__":
    main()
