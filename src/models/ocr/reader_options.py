# File name: reader_options.py
# Date: 10/29/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Decode-time options of the production page reader (:mod:`src.models.ocr.page_reader`), all off by default.

The decode arms of 2026-10-29 (``logs/next_round/arabic_decode_1029/``) measure four changes that need no training.
Each is an option of :class:`page_reader.ReaderSettings`; this module holds their pure decisions and the image
views they read, so they are tested without a model.

1. **Marker-aware loop detector** (``loop_detector="markers"``). The production guard looks for repeated runs of
   Hebrew and Arabic letters only, so a read that floods ``[...]`` markers (one cached read held 13k characters of
   them) or repeats digits, punctuation or Latin fragments is never flagged. Here the guard runs on *units*: every
   non-space character, with a whole gap marker (``[...]``, ``[…]``) counted as one unit. A ``stop`` read whose unit
   repeat share exceeds the scorer's loop threshold, or whose units hold a window-rule loop made mostly of
   non-letters (:func:`non_letter_loop`), is cut where the loop starts and, like a ``length`` finish, retried once
   at temperature 0.
2. **Orientation probe** (``orientation="probe"``). The page is read with a short budget in each of the four
   orientations (the image turned 0, 90, 180 and 270 degrees counter-clockwise) at temperature 0 with token
   log-probabilities; the orientation with the highest mean token log-probability among the probes whose output is
   mostly Hebrew or Arabic letters wins, and the page leaves upright only when the winner beats the upright probe by
   ``orientation_margin`` nats per token. The full read is then made in the chosen orientation only.
   ``orientation="gated"`` reads the page upright first and probes only when that read is suspect
   (:func:`suspect_read`: the first request hit the token limit, the read looped, or it is mostly gap markers); a
   turned read then replaces it. Probing every page turned a third of the benchmark's upright pages (pages with lines
   in more than one direction, or fluent invented reads in any orientation), so ``"probe"`` is for measurement only.
3. **Band crops** (``crops="always"`` or ``"auto"``). The page's fragment box (the coverage tool's fragment mask) is
   cut into 2-4 horizontal bands that overlap by a share of the box height (the Hebrew rescue policy's halves,
   ``reread.half_boxes``, generalised); each band is read on its own and the band reads are joined line by line with
   the overlap dropped (``reread.join_halves``). The joined read replaces the whole-page read when no band failed,
   it is not a loop where the whole-page read was not, and it holds at least ``crop_min_ratio`` of the whole-page
   read's letters. ``"auto"`` crops only a large image (``crop_min_mp`` megapixels, which the model sees
   downscaled to 7 MP) or a read that looks short for the ink on the page (letters per ink unit under
   ``crop_max_density``).
4. **Script filter** (``script_filter=True``). LM Studio's MLX engine ignores ``logit_bias`` (tested 2026-10-29: a
   -100 bias on a word's tokens left the reply unchanged) and offers no character grammar for free text, so tokens
   cannot be masked at decode time. The filter is a post-filter: characters outside Arabic, Hebrew, digits,
   white space and punctuation (Latin letters, CJK, other scripts) are stripped from the stored read, and a read
   that lost at least ``filter_retry_chars`` characters is retried once at temperature 0.
"""
import hashlib
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from src.datasets.consensus.reread import join_halves
from src.datasets.evaluations import arabic_script as ar

Box = Tuple[int, int, int, int]

# --- loop detector -----------------------------------------------------------------------------------------------------

LOOP_LETTERS, LOOP_MARKERS = "letters", "markers"
GAP_MARKER = re.compile(r"\[\s*(?:\.\s*){2,}\]|\[\s*…\s*\]")      # [...], [. . .], [....], […]
MARKER_UNIT = ""                                             # one private-use character stands for a whole marker


def loop_units(text: str) -> Tuple[str, List[int]]:
    """The units the marker-aware loop detector runs on, with the character offset of each.

    :param text: A read.
    :type text: str
    :return: ``(units, offsets)``: every non-space character of ``text`` in order, a gap marker collapsed into
        :data:`MARKER_UNIT`; ``offsets[i]`` is the offset in ``text`` where unit ``i`` starts.
    :rtype: Tuple[str, List[int]]
    """
    units: List[str] = []
    offsets: List[int] = []
    position = 0
    for match in GAP_MARKER.finditer(text):
        for i in range(position, match.start()):
            if not text[i].isspace():
                units.append(text[i])
                offsets.append(i)
        units.append(MARKER_UNIT)
        offsets.append(match.start())
        position = match.end()
    for i in range(position, len(text)):
        if not text[i].isspace():
            units.append(text[i])
            offsets.append(i)
    return "".join(units), offsets


def unit_repeat_share(units: str, span: int = 12) -> float:
    """Share of the ``span``-unit runs that repeat an earlier run (``arabic_script.repeat_share`` on units).

    :param units: :func:`loop_units` string.
    :type units: str
    :param span: Run length.
    :type span: int
    :return: ``1 - distinct runs / runs``; 0.0 below three spans.
    :rtype: float
    """
    if len(units) < span * 3:
        return 0.0
    runs = [units[i:i + span] for i in range(len(units) - span + 1)]
    return 1.0 - len(set(runs)) / len(runs)


NON_LETTER_LOOP_SHARE = 0.5


def non_letter_loop(units: str, start: Optional[int], min_share: float = NON_LETTER_LOOP_SHARE) -> bool:
    """Whether a unit loop found by the loop guard's window rule is a flood of non-letters.

    :param units: :func:`loop_units` string.
    :type units: str
    :param start: Unit index where the window rule puts the loop start (None: no loop).
    :type start: Optional[int]
    :param min_share: Share of the units from the loop start on that must not be Hebrew or Arabic letters.
    :type min_share: float
    :return: True for a ``[...]`` flood or a repeat of digits, punctuation or Latin text; False for a letter loop
        (left to the production gate, which spares formula repeats inside a real text) or no loop.
    :rtype: bool
    """
    if start is None:
        return False
    tail = units[start:]
    return bool(tail) and 1 - len(ar.SCRIPT_LETTER.findall(tail)) / len(tail) >= min_share


def marker_count(text: str) -> int:
    """Number of gap markers (``[...]`` and variants) in a read.

    :param text: A read.
    :type text: str
    :return: Count.
    :rtype: int
    """
    return len(GAP_MARKER.findall(text))


# --- script filter -----------------------------------------------------------------------------------------------------

ALLOWED_CHAR = re.compile(
    r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿"   # Arabic, supplements, presentation forms
    r"֐-׿יִ-ﭏ"                                          # Hebrew and its presentation forms
    r"0-9\s"
    r"!-/:-@\[-`{-~"                                                       # ASCII punctuation and symbols
    r" -¿×÷ -⁯⸀-⹿‌-‏"    # Latin-1 punctuation, general punctuation, marks
    r"◌◦•]")
FILTER_RETRY_CHARS = 3


def filter_script(text: str) -> Tuple[str, int, str]:
    """Strip characters that are not Arabic, Hebrew, digits, white space or punctuation.

    :param text: A read.
    :type text: str
    :return: ``(filtered text, number of characters stripped, the stripped characters in order)``. Runs of spaces
        left by a removal are collapsed and lines are stripped at the ends; a line left empty by the filter is
        dropped. A text with nothing to strip comes back unchanged.
    :rtype: Tuple[str, int, str]
    """
    stripped = "".join(ch for ch in text if not ALLOWED_CHAR.match(ch))
    if not stripped:
        return text, 0, ""
    lines = []
    for line in text.split("\n"):
        kept = "".join(ch for ch in line if ALLOWED_CHAR.match(ch))
        kept = re.sub(r"[ \t]{2,}", " ", kept).strip()
        if kept or not line.strip():
            lines.append(kept)
    return "\n".join(lines).strip(), len(stripped), stripped


# --- orientation probe ---------------------------------------------------------------------------------------------------

ORIENTATION_OFF, ORIENTATION_PROBE, ORIENTATION_GATED = "off", "probe", "gated"
ROTATIONS = (0, 90, 180, 270)            # counter-clockwise turns of the image (PIL's ROTATE_90 is counter-clockwise)
PROBE_MAX_TOKENS = 48
ORIENTATION_MARGIN = 0.3                 # nats per token the winner must beat the upright probe by (calibrated per run)
MIN_PROBE_SCRIPT_SHARE = 0.5
SUSPECT_MARKER_SHARE = 0.15              # gated probe: gap markers per word that make an upright read suspect


def suspect_read(read: Mapping[str, Any], marker_share: float = SUSPECT_MARKER_SHARE) -> Tuple[bool, str]:
    """Whether an upright read looks broken enough to probe the page's orientation (``orientation="gated"``).

    A page read in the wrong orientation tends to run to the token limit, loop, or come back as gap markers; a fluent
    read is trusted even when a turned probe would score higher (on the Arabic benchmark, ungated probes turned 36 of
    108 upright pages: fluent invented reads in any orientation, lines in more than one direction).

    :param read: A settled read (``first_finish_reason``, ``finish_reason``, ``looped``, ``text``).
    :type read: Mapping[str, Any]
    :param marker_share: Gap markers per whitespace word at or above which the read is suspect.
    :type marker_share: float
    :return: ``(suspect, why)``: ``why`` is ``"length"`` (the first request hit the token limit), ``"looped"``,
        ``"markers"`` or ``"fluent"``.
    :rtype: Tuple[bool, str]
    """
    if (read.get("first_finish_reason") or read.get("finish_reason")) == "length":
        return True, "length"
    if read.get("looped"):
        return True, "looped"
    text = read.get("text") or ""
    if marker_count(text) / max(1, len(text.split())) >= marker_share:
        return True, "markers"
    return False, "fluent"
_TRANSPOSE = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}


def script_share(text: str) -> float:
    """Share of a text's non-space characters that are Hebrew or Arabic letters.

    :param text: A read.
    :type text: str
    :return: Share in [0, 1]; 0.0 for a text without non-space characters.
    :rtype: float
    """
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return len(ar.SCRIPT_LETTER.findall("".join(chars))) / len(chars)


def probe_features(meta: Optional[Mapping[str, Any]], first_k: int = 16) -> Dict[str, Any]:
    """What an orientation probe says about its orientation.

    :param meta: The probe's completion metadata (``text``, ``finish_reason``, ``token_logprobs``), None when it failed.
    :type meta: Optional[Mapping[str, Any]]
    :param first_k: Tokens for the early-tokens mean.
    :type first_k: int
    :return: ``ok`` (an answer with log-probabilities), ``mean_logprob`` (over every generated token),
        ``first_logprob`` (over the first ``first_k``), ``min_logprob``, ``tokens``, ``script_share``,
        ``repeat_share`` (``arabic_script.repeat_share``) and the probe ``text``.
    :rtype: Dict[str, Any]
    """
    text = (meta or {}).get("text") or ""
    logprobs = [float(v) for v in ((meta or {}).get("token_logprobs") or []) if v is not None]
    ok = meta is not None and bool(logprobs)
    return {"ok": ok, "mean_logprob": round(statistics.mean(logprobs), 4) if logprobs else None,
            "first_logprob": round(statistics.mean(logprobs[:first_k]), 4) if logprobs else None,
            "min_logprob": round(min(logprobs), 4) if logprobs else None, "tokens": len(logprobs),
            "script_share": round(script_share(text), 4), "repeat_share": round(ar.repeat_share(text), 4), "text": text}


def choose_orientation(probes: Mapping[int, Mapping[str, Any]], margin: float = ORIENTATION_MARGIN,
                       min_script_share: float = MIN_PROBE_SCRIPT_SHARE, statistic: str = "mean_logprob") -> Tuple[int, str]:
    """The orientation a page is read in, from its four probes.

    :param probes: :func:`probe_features` per rotation.
    :type probes: Mapping[int, Mapping[str, Any]]
    :param margin: Nats per token the best probe must beat the upright probe by to turn the page.
    :type margin: float
    :param min_script_share: A probe whose output holds fewer Hebrew and Arabic letters than this share of its
        characters cannot win (an orientation the model reads as Latin, digits or nothing).
    :type min_script_share: float
    :param statistic: The probe score compared (``mean_logprob`` or ``first_logprob``).
    :type statistic: str
    :return: ``(rotation, reason)``: ``reason`` is ``"upright"`` (the upright probe won or no probe could),
        ``"margin"`` (another probe won by less than the margin), or ``"turned"``.
    :rtype: Tuple[int, str]
    """
    candidates = {r: p for r, p in probes.items() if p.get("ok") and p.get(statistic) is not None
                  and p.get("script_share", 0.0) >= min_script_share}
    if not candidates:
        return 0, "upright"
    best = max(candidates, key=lambda r: (candidates[r][statistic], r == 0))
    if best == 0:
        return 0, "upright"
    upright = probes.get(0) or {}
    if upright.get("ok") and upright.get(statistic) is not None and 0 in candidates:
        if candidates[best][statistic] - upright[statistic] < margin:
            return 0, "margin"
    return best, "turned"


# --- image views -----------------------------------------------------------------------------------------------------------

VIEW_QUALITY = 95


def view_path(image_path: str, view_dir: Path, tag: str) -> Path:
    """Where a derived view of an image is written: the stem, a short hash of the source path, and the view tag.

    :param image_path: Source image.
    :type image_path: str
    :param view_dir: Directory of derived views.
    :type view_dir: Path
    :param tag: View tag (``r90``, ``r0_b1of3`` ...).
    :type tag: str
    :return: ``<view_dir>/<stem>__<hash8>__<tag>.jpg``.
    :rtype: Path
    """
    digest = hashlib.sha1(str(Path(image_path).resolve()).encode()).hexdigest()[:8]
    return view_dir / f"{Path(image_path).stem}__{digest}__{tag}.jpg"


def _save(image: Image.Image, path: Path) -> str:
    """Write a view once (JPEG, quality :data:`VIEW_QUALITY`); an existing file is reused so the bytes stay stable."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.jpg")
        image.convert("RGB").save(tmp, format="JPEG", quality=VIEW_QUALITY)
        tmp.rename(path)
    return str(path)


def rotated_view(image_path: str, rotation: int, view_dir: Path) -> str:
    """The image turned ``rotation`` degrees counter-clockwise, as a file the transcriber can send.

    :param image_path: Source image.
    :type image_path: str
    :param rotation: 0, 90, 180 or 270.
    :type rotation: int
    :param view_dir: Directory of derived views.
    :type view_dir: Path
    :return: ``image_path`` itself for 0 (the bytes the upright read sends), else the written view.
    :rtype: str
    :raises ValueError: For another angle.
    """
    if rotation == 0:
        return image_path
    if rotation not in _TRANSPOSE:
        raise ValueError(f"unsupported rotation {rotation}")
    path = view_path(image_path, view_dir, f"r{rotation}")
    if path.exists():
        return str(path)
    with Image.open(image_path) as im:
        return _save(im.transpose(_TRANSPOSE[rotation]), path)


# --- band crops ------------------------------------------------------------------------------------------------------------

CROPS_OFF, CROPS_ALWAYS, CROPS_AUTO = "off", "always", "auto"
CROP_OVERLAP = 0.12            # reread.OVERLAP_SHARE: neighbouring bands share 12 % of the box height
CROP_TARGET_MP = 7.0           # the processor's pixel budget: a band larger than this is still downscaled
CROP_MAX_BANDS = 4
CROP_MIN_RATIO = 0.8           # decode_policy.ASSEMBLED_MIN_RATIO: a joined read must keep >= 0.8x the whole-page letters
CROP_BOX_MARGIN = 0.02         # the fragment box is padded by 2 % of each side


def band_count(box_width: int, box_height: int, target_mp: float = CROP_TARGET_MP, max_bands: int = CROP_MAX_BANDS) -> int:
    """How many horizontal bands a fragment box is cut into: 2, or more for a box the model would see downscaled.

    :param box_width: Box width in pixels.
    :type box_width: int
    :param box_height: Box height in pixels.
    :type box_height: int
    :param target_mp: Megapixels one band may hold before it is cut further.
    :type target_mp: float
    :param max_bands: Upper bound.
    :type max_bands: int
    :return: ``clamp(ceil(box pixels / target), 2, max_bands)``.
    :rtype: int
    """
    return max(2, min(max_bands, math.ceil(box_width * box_height / (target_mp * 1e6))))


def band_boxes(box: Box, bands: int, overlap: float = CROP_OVERLAP) -> List[Box]:
    """Horizontal bands of a box, top to bottom, each sharing ``overlap`` of the box height with its neighbour.

    :param box: ``(left, top, right, bottom)``.
    :type box: Box
    :param bands: Number of bands (>= 1).
    :type bands: int
    :param overlap: Share of the box height two neighbouring bands both contain.
    :type overlap: float
    :return: PIL crop boxes covering the box.
    :rtype: List[Box]
    :raises ValueError: For an empty box, fewer than one band or an overlap outside [0, 1).
    """
    left, top, right, bottom = box
    height = bottom - top
    if right <= left or height <= 0 or bands < 1 or not 0 <= overlap < 1:
        raise ValueError(f"bad band split: box {box}, {bands} bands, overlap {overlap}")
    shared = overlap * height
    band_h = (height + (bands - 1) * shared) / bands
    out = []
    for i in range(bands):
        start = top + i * (band_h - shared)
        out.append((left, int(round(start)), right, min(bottom, int(round(start + band_h)))))
    out[-1] = (left, out[-1][1], right, bottom)
    return out


def fragment_box(image_path: str, long_side: int = 1000, margin: float = CROP_BOX_MARGIN) -> Optional[Box]:
    """The bounding box of the image's fragment mask (the coverage tool's), in original pixels, padded.

    :param image_path: Image file.
    :type image_path: str
    :param long_side: Working resolution of the mask.
    :type long_side: int
    :param margin: Padding as a share of each side.
    :type margin: float
    :return: ``(left, top, right, bottom)``, or None when the mask is empty.
    :rtype: Optional[Box]
    """
    import numpy as np

    from src.datasets.evaluations.helper_eval_scripts import arabic_bench_coverage as abc

    settings = abc.ImageSettings()
    rgb = abc.load_rgb(Path(image_path), long_side)
    mask = abc.fragment_mask(rgb, settings)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    with Image.open(image_path) as im:
        width, height = im.size
    sx, sy = width / mask.shape[1], height / mask.shape[0]
    pad_x, pad_y = margin * width, margin * height
    return (max(0, int(xs.min() * sx - pad_x)), max(0, int(ys.min() * sy - pad_y)),
            min(width, int((xs.max() + 1) * sx + pad_x)), min(height, int((ys.max() + 1) * sy + pad_y)))


def band_views(image_path: str, box: Optional[Box], view_dir: Path, overlap: float = CROP_OVERLAP,
               target_mp: float = CROP_TARGET_MP, max_bands: int = CROP_MAX_BANDS, tag: str = "") -> List[Tuple[Box, str]]:
    """The band crops of an image, written as files.

    :param image_path: Image (already in its reading orientation).
    :type image_path: str
    :param box: Fragment box to cut, None for the whole image.
    :type box: Optional[Box]
    :param view_dir: Directory of derived views.
    :type view_dir: Path
    :param overlap: Band overlap share.
    :type overlap: float
    :param target_mp: See :func:`band_count`.
    :type target_mp: float
    :param max_bands: See :func:`band_count`.
    :type max_bands: int
    :param tag: Prefix of the view tag (the rotation, for a turned page).
    :type tag: str
    :return: ``[(box, path)]`` top to bottom.
    :rtype: List[Tuple[Box, str]]
    """
    with Image.open(image_path) as im:
        im.load()
        box = box or (0, 0, im.size[0], im.size[1])
        n = band_count(box[2] - box[0], box[3] - box[1], target_mp, max_bands)
        out = []
        for i, b in enumerate(band_boxes(box, n, overlap)):
            path = view_path(image_path, view_dir, f"{tag}b{i + 1}of{n}_{b[0]}_{b[1]}_{b[2]}_{b[3]}")
            out.append((b, str(path) if path.exists() else _save(im.crop(b), path)))
    return out


def merge_bands(texts: Sequence[str]) -> Tuple[str, int]:
    """Join band reads top to bottom, dropping the lines a band repeats from the one above.

    :param texts: Band reads in order (empty strings allowed).
    :type texts: Sequence[str]
    :return: ``(joined text, lines dropped as overlap)``.
    :rtype: Tuple[str, int]
    """
    merged: List[Dict[str, Any]] = []
    dropped = 0
    for text in texts:
        lines = [{"text": ln} for ln in (text or "").split("\n") if ln.strip()]
        if not merged:
            merged = lines
            continue
        merged, n = join_halves(merged, lines)
        dropped += n
    return "\n".join(ln["text"] for ln in merged), dropped


def letters_per_ink(letters: int, ink: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Letters read per unit of ink on the page (``ink_permille`` x the fragment's share of the image).

    :param letters: Hebrew and Arabic letters of the whole-page read.
    :type letters: int
    :param ink: :func:`page_reader.measure_ink` result.
    :type ink: Optional[Mapping[str, Any]]
    :return: The density, None when the ink was not measured or the mask failed.
    :rtype: Optional[float]
    """
    if not ink or not ink.get("mask_ok") or not ink.get("ink_permille") or not ink.get("inner_share"):
        return None
    return round(letters / (float(ink["ink_permille"]) * float(ink["inner_share"])), 3)


def crop_decision(mode: str, image_pixels: int, letters: int, ink: Optional[Mapping[str, Any]], min_mp: float,
                  max_density: float) -> Tuple[bool, str]:
    """Whether a page is also read in band crops.

    :param mode: :data:`CROPS_OFF`, :data:`CROPS_ALWAYS` or :data:`CROPS_AUTO`.
    :type mode: str
    :param image_pixels: Pixels of the image in its reading orientation.
    :type image_pixels: int
    :param letters: Hebrew and Arabic letters of the whole-page read.
    :type letters: int
    :param ink: Ink measures (for the density).
    :type ink: Optional[Mapping[str, Any]]
    :param min_mp: ``auto``: crop an image of at least this many megapixels.
    :type min_mp: float
    :param max_density: ``auto``: crop a page whose read holds fewer letters per ink unit than this.
    :type max_density: float
    :return: ``(crop, reason)`` with reason ``off``, ``always``, ``size``, ``density`` or ``no``.
    :rtype: Tuple[bool, str]
    :raises ValueError: For an unknown mode.
    """
    if mode == CROPS_OFF:
        return False, "off"
    if mode == CROPS_ALWAYS:
        return True, "always"
    if mode != CROPS_AUTO:
        raise ValueError(f"unknown crop mode {mode!r}")
    if image_pixels >= min_mp * 1e6:
        return True, "size"
    density = letters_per_ink(letters, ink)
    if density is not None and density < max_density:
        return True, "density"
    return False, "no"


def accept_bands(whole: Mapping[str, Any], joined_letters: int, joined_looped: bool, any_failed: bool,
                 min_ratio: float = CROP_MIN_RATIO) -> Tuple[bool, str]:
    """Whether the joined band read replaces the whole-page read.

    :param whole: The whole-page read (``text``, ``looped``).
    :type whole: Mapping[str, Any]
    :param joined_letters: Hebrew and Arabic letters of the joined band read.
    :type joined_letters: int
    :param joined_looped: Whether any band read ended as a loop.
    :type joined_looped: bool
    :param any_failed: Whether any band request failed.
    :type any_failed: bool
    :param min_ratio: The joined read must keep at least this share of the whole-page read's letters.
    :type min_ratio: float
    :return: ``(accept, reason)``: ``failed``, ``looped`` (a band looped and the whole-page read did not),
        ``shorter`` or ``bands``.
    :rtype: Tuple[bool, str]
    """
    if any_failed:
        return False, "failed"
    if joined_looped and not whole.get("looped"):
        return False, "looped"
    whole_letters = len(ar.SCRIPT_LETTER.findall(ar.normalize_arabic(whole.get("text") or "")))
    if joined_letters < min_ratio * whole_letters:
        return False, "shorter"
    return True, "bands"
