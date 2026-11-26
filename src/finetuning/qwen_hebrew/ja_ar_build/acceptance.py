"""Acceptance test for gate v2 (GATE_V2_SPEC.md §8): audited fixtures, planted wrong pairs, and their scoring.

Two kinds of test material, both kept out of the repository (edition text never enters ``src/`` or ``tests/``):

* **Fixtures** (``logs/next_round/ja_ar_v1/build/acceptance/fixtures.jsonl``): the 80 pairs of the blind visual
  audit, each with its remedy class (keep / re_pair / trim / crop / trim_and_crop / drop / unsure, as in
  ``visual_audit/REMEDIES.md``), per-line truth for the audited photograph (``on`` / ``off`` / ``part`` /
  ``unknown``) and the expected gate outcome.
* **Planted negatives** (``W2/pool_planted/<class>.jsonl`` + ``planted_manifest.jsonl``): stage-1 candidate records
  built from stage-2 ``accepted`` sides of the frozen novel_v1 work dir, with wrong lines planted on purpose
  (cross-document and same-genre swaps, the other face appended, two documents concatenated, shuffled line order).
  They are test material only and must never reach a dataset; :func:`write_planted` refuses any directory that is not
  a ``pool_planted`` directory.

Scoring a gate ``pairs.jsonl`` (schema §4). The outcome of a fixture is read on its audited photograph (or a photograph
of the same face: same sha256, pHash distance <= 2, or listed in the record's ``duplicate_photos``):

``accept_as_is`` | ``accept_trim`` | ``accept_crop`` | ``accept_trim_crop``
    an ``accept`` record on the photo; trim = ``trim`` in remedies, trimmed or moved lines, or a fixture ``off`` line
    missing from the kept lines; crop = a crop box or ``crop`` in remedies.
``review`` | ``reject``
    the best record on the photo (or the side-level reject record when no photo supports the side).
``elsewhere``
    no record on the photo, but the side is accepted or under review on another photograph.
``missing``
    the gate wrote nothing for the side (not scored; listed).

§8.1 mapping: keep <-> accept as is; trim <-> accept with trim; crop <-> accept with crop; trim_and_crop <-> both;
re_pair <-> the text is not accepted on the photo and lands on another photo, or the photo gets the text the judges
named (``expected.photo_should_get``); drop <-> no accept on the photo; unsure is listed, never scored. Trim is judged
per side: in a merged record only this side's own trimmed or missing lines count.

**Harm** = an accept where the judge said drop or re_pair, or an accept without trim where the judge said trim
(trim_and_crop counted alongside, reported apart); an accepted record that also merges in lines of a text the judges
saw was not on the photo, or a second edition of the same text, is harm too (kinds ``merged_text_not_on_photo`` /
``merged_duplicate_edition``); a merge passes only with the partner lines the fixture allows
(``expected.merge_partners``). **Loss** = reject / review / elsewhere where the judge said keep, trim or crop
(trim_and_crop reported apart). Other disagreements are ``mismatch`` with a kind (``over_trim``, ``crop_missed``, ...).
Secondary checks per item: kept lines against the per-line truth (``strict_pass`` = pass and the kept lines match it),
other texts of the item that the judges saw were not on the photo (``siblings_not_on_photo``) accepted there, and the
rotation the judges asked for.

Planted negatives: an accept whose kept lines include a line the manifest does not place on the record's photograph
(``line_photos``) is a false accept; for ``shuffled`` an accept whose kept lines are out of their true order is one too.
False-accept rate per class = false accepts / planted records the gate wrote anything for. Run the gate on the planted
files apart from the real pool, with the real pool as null material, and keep the donors' photographs
(``notes.planted.null_exclude_sha256``) out of the planted targets' nulls.

Usage::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.ja_ar_build.acceptance \\
        --pairs W2/gate/pairs.jsonl --planted-pairs W2/gate_planted/pairs.jsonl --out <report.md> [--json <out.json>]
    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.ja_ar_build.acceptance --plant \\
        [--work-dir W1] [--out-dir W2/pool_planted] [--per-class 60] [--seed 20261112]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
#: The round's private folder (git-ignored).
ROUND_DIR = REPO_ROOT / "logs" / "next_round" / "ja_ar_v1"
DEFAULT_FIXTURES = ROUND_DIR / "build" / "acceptance" / "fixtures.jsonl"
#: Frozen novel_v1 work dir (read-only).
W1 = Path("/Volumes/home/studio_offload/datasets/genizah_novel_v1_work")
#: This round's work dir.
W2 = Path("/Volumes/home/studio_offload/datasets/genizah_ja_ar_v1_work")
DEFAULT_PLANTED_DIR = W2 / "pool_planted"
MANIFEST_NAME = "planted_manifest.jsonl"
DEFAULT_SEED = 20261112

REMEDY_CLASSES = ("keep", "re_pair", "trim", "crop", "trim_and_crop", "drop", "unsure")
#: Remedy classes whose pair is usable after the remedy (a reject / review there is a loss).
USABLE_REMEDIES = ("keep", "trim", "crop", "trim_and_crop")
ACCEPT_OUTCOMES = ("accept_as_is", "accept_trim", "accept_crop", "accept_trim_crop")
OUTCOMES = ACCEPT_OUTCOMES + ("review", "reject", "elsewhere", "missing")
#: Gate outcome that passes each usable remedy class (fixtures may list more under ``expected.acceptable``).
PASS_OUTCOME = {"keep": "accept_as_is", "trim": "accept_trim", "crop": "accept_crop",
                "trim_and_crop": "accept_trim_crop"}
#: Names of the remaining disagreements of a usable class.
MISMATCH_KINDS = {
    ("keep", "accept_trim"): "over_trim", ("keep", "accept_crop"): "over_crop",
    ("keep", "accept_trim_crop"): "over_trim_and_crop", ("trim", "accept_trim_crop"): "over_crop",
    ("crop", "accept_as_is"): "crop_missed", ("crop", "accept_trim"): "trim_instead_of_crop",
    ("crop", "accept_trim_crop"): "over_trim", ("trim_and_crop", "accept_trim"): "crop_missed",
}
#: Harm and loss as GATE_V2_SPEC §8.1 names them; trim_and_crop is counted beside them.
SPEC_HARM_REMEDIES = ("drop", "re_pair", "trim")
SPEC_LOSS_REMEDIES = ("keep", "trim", "crop")
LINE_STATES = ("on", "off", "part", "unknown")
PLANTED_CLASSES = ("cross_doc", "same_genre", "other_face_append", "concat_docs", "shuffled")
#: Planting order: the most constrained classes first, so they get first pick of the donor pool.
PLANTING_ORDER = ("same_genre", "other_face_append", "cross_doc", "concat_docs", "shuffled")
#: Image statuses that mean "file on disk".
ON_DISK_STATUS = ("present", "fetched", "duplicate")
DECISION_RANK = {"accept": 3, "review": 2, "reject": 1}
#: pHash distance up to which two photographs of one candidate are taken as the same face.
SAME_FACE_PHASH = 2
#: pHash distance above which two accepted photographs are taken as different faces (the gate clusters at <= 10).
OTHER_FACE_PHASH = 10
RNL_RE = re.compile(r"yevr|firk|antonin|\brnl\b|russian national|st\.? ?petersburg", re.IGNORECASE)
#: Judaeo-Arabic function words (the JA marker of orchestrator/alignment_profiles.py).
JA_MARKERS = frozenset(("אלדי אלתי פי מן אן הדא הדה דלך כאן ען עלי אלי לם קד ענד בעד קבל יא אללה אלא ואן פאן לאן כאנת "
                        "יכון אלמולי סידי מולאי כתאבי אטאל אדאם וצל וקד אנא אנת הו ליס גיר כל שי מא לא").split())
#: Formula marker tokens (letter skeletons) per genre family, used to label same-genre swaps.
GENRE_MARKERS = {
    "hebrew": {
        "letter": frozenset("אטאל בקאכ בקאה ואדאמ כתאבי יאסידי ומולאי סלאמתכ וסעאדתכ תאיידכ ושלומ".split()),
        "deed": frozenset("שריר ובריר וקיימ מודעינ ותנאינ קנינא הכשר לקנות ולראיה לזכו מותבה נילוס קדמנא וחתמנא "
                          "לשטרות העדימ החתומימ בפנינו".split()),
    },
    "arabic": {
        "letter": frozenset("اطال بقاك بقاءك بقاه كتابي سيدي ومولاي عزك عزه وادام".split()),
        "legal": frozenset("شهد اشهد بتاريخ الثمن حدود الاجره المستاجر باع اشتري اقر".split()),
    },
}

_HEX_RE = re.compile(r"[0-9a-fA-F]+")
_HEB_FINALS = str.maketrans("ךםןףץ", "כמנפצ")
_AR_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})


# --- text helpers (local, so the harness does not depend on the gate's own modules) --------------------------------

def _is_hebrew(ch: str) -> bool:
    """Whether a character is a Hebrew letter (U+05D0-U+05EA).

    :param ch: One character.
    :return: True for a Hebrew letter.
    """
    return "א" <= ch <= "ת"


def _is_arabic(ch: str) -> bool:
    """Whether a character is an Arabic letter of the skeleton blocks (tatweel excluded).

    :param ch: One character.
    :return: True for an Arabic letter.
    """
    return ("ء" <= ch <= "ي" or "ٱ" <= ch <= "ۓ") and ch != "ـ"


def skeleton(text: str) -> str:
    """Letters only, one character per letter (GATE_V2_SPEC §1): Hebrew finals folded, Arabic alef / hamza carriers /
    alef maqsura / ta marbuta folded, everything else dropped.

    :param text: Any text.
    :return: The letter skeleton.
    """
    out: List[str] = []
    for ch in text:
        if _is_hebrew(ch):
            out.append(ch.translate(_HEB_FINALS))
        elif _is_arabic(ch):
            out.append(ch.translate(_AR_FOLD))
    return "".join(out)


def letters(text: str) -> int:
    """Number of skeleton letters.

    :param text: Any text.
    :return: ``len(skeleton(text))``.
    """
    return len(skeleton(text))


def script_of(text: str) -> str:
    """Script family by letter majority.

    :param text: Any text.
    :return: ``hebrew``, ``arabic`` or ``none``.
    """
    sk = skeleton(text)
    if not sk:
        return "none"
    hebrew = sum(1 for ch in sk if _is_hebrew(ch))
    return "arabic" if len(sk) - hebrew > hebrew else "hebrew"


def ja_band(script: str, text: str) -> str:
    """Coarse band of a target (the rule of orchestrator/alignment_profiles.py).

    :param script: ``hebrew`` / ``arabic`` / other.
    :param text: Target text.
    :return: ``ar``; ``ja`` (Hebrew script, >= 3 distinct JA markers and >= 40 marker tokens per 1,000 words); ``heb``;
        else the script itself.
    """
    if script == "arabic":
        return "ar"
    words = re.findall(r"[א-ת]+", text)
    hits = [w for w in words if w in JA_MARKERS]
    if script == "hebrew" and len(set(hits)) >= 3 and words and 1000.0 * len(hits) / len(words) >= 40:
        return "ja"
    return "heb" if script == "hebrew" else script


def word_tokens(text: str) -> List[str]:
    """Words as letter skeletons, words of fewer than two letters dropped.

    :param text: Any text.
    :return: Skeleton tokens in order.
    """
    return [t for t in (skeleton(w) for w in text.split()) if len(t) >= 2]


def shingles(tokens: Sequence[str], n: int = 3) -> Set[Tuple[str, ...]]:
    """Word n-gram shingles.

    :param tokens: Tokens in order.
    :param n: Shingle length.
    :return: Distinct shingles.
    """
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def genre_of(tokens: Sequence[str], script: str) -> str:
    """Genre family by formula marker tokens.

    :param tokens: Skeleton tokens of a target.
    :param script: ``hebrew`` or ``arabic``.
    :return: The family with the most marker hits (at least two, and more than any other family), else ``other``.
    """
    hits = {name: sum(1 for t in tokens if t in markers)
            for name, markers in GENRE_MARKERS.get(script, {}).items()}
    if not hits:
        return "other"
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])
    if ranked[0][1] < 2 or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return "other"
    return ranked[0][0]


def parse_ranges(spec: str) -> List[int]:
    """Line numbers named in a range string (``"1-5, 7; 9-10"``); words are ignored.

    :param spec: Range text.
    :return: Sorted distinct numbers.
    """
    out: Set[int] = set()
    for lo, hi in re.findall(r"(\d+)\s*(?:[-–]\s*(\d+))?", spec or ""):
        a, b = int(lo), int(hi) if hi else int(lo)
        out.update(range(min(a, b), max(a, b) + 1))
    return sorted(out)


def compress_ranges(idx: Iterable[int], one_based: bool = True) -> str:
    """Render indices as compact ranges.

    :param idx: Line indices (0-based).
    :param one_based: Print as 1-based line numbers.
    :return: E.g. ``"1-3,6"``; ``"-"`` when empty.
    """
    vals = sorted(set(int(i) + (1 if one_based else 0) for i in idx))
    if not vals:
        return "-"
    parts: List[str] = []
    start = prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        start = prev = v
    parts.append(f"{start}-{prev}" if prev > start else str(start))
    return ",".join(parts)


# --- I/O ---------------------------------------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON-lines file.

    :param path: File path.
    :return: Records (blank lines skipped).
    :raises SystemExit: When the file cannot be read or parsed.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Stream a JSON-lines file record by record.

    :param path: File path.
    :return: Iterator over records.
    :raises SystemExit: When the file cannot be opened.
    """
    try:
        f = open(path, encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    with f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write records as JSON lines (UTF-8, not ASCII-escaped).

    :param path: Output file; the parent directory is created.
    :param records: Records.
    :return: Number written.
    :raises SystemExit: When the file cannot be written.
    """
    n = 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    except OSError as exc:
        raise SystemExit(f"cannot write {path}: {exc}") from exc
    return n


def write_text(path: Path, text: str) -> None:
    """Write a text file.

    :param path: Output file; the parent directory is created.
    :param text: Content.
    :raises SystemExit: When the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot write {path}: {exc}") from exc


# --- fixture construction helpers ----------------------------------------------------------------------------------

def remedy_class(judgment: Mapping[str, Any], paired_text_id: str) -> str:
    """Remedy class of one audited pair (the rule of orchestrator/audit_remedies.py behind REMEDIES.md).

    :param judgment: One judge's verdict on the item.
    :param paired_text_id: Text id (``T1``...) the builder had paired with the photograph.
    :return: ``keep`` / ``re_pair`` / ``trim`` / ``crop`` / ``re_pair_crop`` / ``trim_and_crop`` / ``drop`` /
        ``unsure``.
    """
    paired = next((t for t in judgment.get("texts", []) if t.get("text_id") == paired_text_id), {})
    on = paired.get("on_photo")
    verdict = judgment.get("verdict")
    if verdict == "complete_pair":
        return "keep" if judgment.get("best_text_id") == paired_text_id else "re_pair"
    if verdict == "text_has_extra":
        return "trim" if on in ("part", "most") else "re_pair"
    if verdict == "photo_has_extra":
        return "crop" if on in ("all", "most") else "re_pair_crop"
    if verdict == "both_partial":
        return "trim_and_crop"
    if verdict == "wrong_text":
        return "drop"
    return "unsure"


def judge_line_map(text: str, lines: Sequence[str]) -> Dict[int, Optional[int]]:
    """Map the judges' line numbers (1-based over ``text.split("\\n")``, as the judges saw the text) to indices of the
    target's non-blank ``lines`` (the gate's index space).

    :param text: Target text.
    :param lines: Target lines.
    :return: Judge number -> 0-based line index, or None for a line the target does not keep (a blank line).
    :raises ValueError: When ``lines`` is not the non-blank subsequence of the text's lines.
    """
    kept = [ln for ln in lines if ln.strip()]
    out: Dict[int, Optional[int]] = {}
    j = 0
    for i, raw in enumerate(text.split("\n"), start=1):
        if j < len(kept) and raw == kept[j]:
            out[i] = j
            j += 1
        else:
            out[i] = None
    if j != len(kept):
        raise ValueError("target lines are not the non-blank lines of its text")
    return out


def line_truth(mapping: Mapping[int, Optional[int]], off: Iterable[int] = (), part: Iterable[int] = (),
               unknown: Iterable[int] = ()) -> List[str]:
    """Per-line truth of a target on one photograph, in the gate's index space.

    :param mapping: Judge number -> line index (:func:`judge_line_map`).
    :param off: Judge numbers of lines not on the photograph.
    :param part: Judge numbers of lines partly on it.
    :param unknown: Judge numbers the judges could not place.
    :return: One of :data:`LINE_STATES` per line (default ``on``).
    :raises ValueError: For a judge number outside the text.
    """
    n = sum(1 for v in mapping.values() if v is not None)
    truth = ["on"] * n
    for state, nums in (("off", off), ("part", part), ("unknown", unknown)):
        for k in nums:
            if k not in mapping:
                raise ValueError(f"line {k} is outside the text ({len(mapping)} lines)")
            idx = mapping[k]
            if idx is not None:
                truth[idx] = state
    return truth


def hamming(a: Optional[str], b: Optional[str]) -> Optional[int]:
    """Hamming distance of two hex pHashes.

    :param a: Hex hash or None.
    :param b: Hex hash or None.
    :return: Distance, or None when a hash is missing or malformed.
    """
    if not a or not b or not _HEX_RE.fullmatch(a) or not _HEX_RE.fullmatch(b):
        return None
    return bin(int(a, 16) ^ int(b, 16)).count("1")


# --- gate record parsing ------------------------------------------------------------------------------------------

def _as_list(value: Any) -> List[Any]:
    """A list or tuple as a list; anything else as an empty list.

    :param value: Any value.
    :return: The list.
    """
    return list(value) if isinstance(value, (list, tuple)) else []


def remedies_of(rec: Mapping[str, Any]) -> List[str]:
    """Remedies of a pair record.

    :param rec: Pair record.
    :return: Remedy codes.
    """
    return [str(r) for r in _as_list(rec.get("remedies"))]


def record_side_keys(rec: Mapping[str, Any]) -> List[str]:
    """Side keys a record covers (``side_key`` plus ``side_keys`` of a merged record).

    :param rec: Pair record.
    :return: Side keys, ``side_key`` first.
    """
    keys = [str(k) for k in _as_list(rec.get("side_keys"))]
    first = rec.get("side_key")
    first = "" if first is None else str(first)
    if first not in keys:
        keys.insert(0, first)
    return keys


def kept_lines_for(rec: Mapping[str, Any], side_key: str) -> Optional[List[int]]:
    """Kept line indices of one side in a record, in output order.

    ``line_refs`` (``[[side_key, index], ...]``) is used when present; else ``line_indices`` of a one-side record.

    :param rec: Pair record.
    :param side_key: The side.
    :return: Indices, or None when the record does not say.
    """
    refs = _as_list(rec.get("line_refs"))
    if refs and all(isinstance(r, (list, tuple)) and len(r) == 2 for r in refs):
        return [int(i) for sk, i in refs if str(sk) == side_key]
    idx = rec.get("line_indices")
    if idx is None or len(record_side_keys(rec)) > 1:
        return None
    return [int(i) for i in _as_list(idx)]


def trimmed_lines_for(rec: Mapping[str, Any], side_key: str) -> List[int]:
    """Trimmed line indices of one side (``trimmed_lines`` as a list, or as ``{side id: [indices]}``).

    :param rec: Pair record.
    :param side_key: The side.
    :return: Sorted indices.
    """
    tl = rec.get("trimmed_lines")
    if isinstance(tl, dict):
        sid = f"{rec.get('row_id')}|{side_key}"
        vals = [i for k, v in tl.items() if str(k) in (side_key, sid) for i in _as_list(v)]
        if not vals and len(tl) == 1 and len(record_side_keys(rec)) == 1:
            vals = _as_list(next(iter(tl.values())))
        return sorted({int(i) for i in vals})
    return sorted({int(i) for i in _as_list(tl)})


def moved_lines_of(rec: Mapping[str, Any]) -> Dict[str, List[int]]:
    """Lines a record says belong to another face: ``{face: [indices]}`` (values may also be ``[side, index]`` pairs
    or ``{side: [indices]}``).

    :param rec: Pair record.
    :return: Face -> sorted indices (empty faces dropped).
    """
    moved = rec.get("moved_to")
    if not isinstance(moved, dict):
        return {}
    out: Dict[str, List[int]] = {}
    for face, value in moved.items():
        items = [i for v in value.values() for i in _as_list(v)] if isinstance(value, dict) else _as_list(value)
        vals = {int(i[1]) if isinstance(i, (list, tuple)) else int(i) for i in items}
        if vals:
            out[str(face)] = sorted(vals)
    return out


def has_crop(rec: Mapping[str, Any]) -> bool:
    """Whether a record crops its photograph.

    :param rec: Pair record.
    :return: True for a crop box or ``crop`` in remedies.
    """
    return rec.get("crop") not in (None, [], {}) or "crop" in remedies_of(rec)


def has_trim(rec: Mapping[str, Any], side_key: str, off_lines: Iterable[int] = (),
             n_lines: Optional[int] = None) -> bool:
    """Whether a record trims this side's text.

    A one-side record: ``trim`` in remedies, trimmed or moved lines. A merged record (several ``side_keys``): this
    side's own trimmed lines, or fewer kept lines of this side than it has (``n_lines``); the record-level remedy only
    when the record does not say which lines of this side it keeps. Either way, dropping a line known not to be on
    the photograph counts as a trim even when the record does not call it one (e.g. removed by cleaning).

    :param rec: Pair record.
    :param side_key: The side.
    :param off_lines: Lines known not to be on the photograph.
    :param n_lines: Lines of the side, if known.
    :return: True when trimmed.
    """
    kept = kept_lines_for(rec, side_key)
    merged = len(record_side_keys(rec)) > 1
    if trimmed_lines_for(rec, side_key):
        return True
    if (not merged or kept is None) and ("trim" in remedies_of(rec) or moved_lines_of(rec)):
        return True
    if kept is None:
        return False
    kept_set = set(kept)
    if any(i not in kept_set for i in off_lines):
        return True
    return merged and n_lines is not None and len(kept_set) < n_lines


def classify_record(rec: Mapping[str, Any], side_key: str, off_lines: Iterable[int] = (),
                    n_lines: Optional[int] = None) -> str:
    """Outcome class of one record for one side.

    :param rec: Pair record.
    :param side_key: The side.
    :param off_lines: Lines known not to be on the photograph (see :func:`has_trim`).
    :param n_lines: Lines of the side, if known.
    :return: One of :data:`ACCEPT_OUTCOMES`, ``review`` or ``reject``.
    """
    decision = rec.get("decision")
    if decision == "accept":
        trim, crop = has_trim(rec, side_key, off_lines, n_lines), has_crop(rec)
        return {(False, False): "accept_as_is", (True, False): "accept_trim", (False, True): "accept_crop",
                (True, True): "accept_trim_crop"}[(trim, crop)]
    return "review" if decision == "review" else "reject"


def photo_match(rec: Mapping[str, Any], photos: Set[str]) -> Optional[str]:
    """How a record sits on a set of same-face photographs.

    :param rec: Pair record.
    :param photos: sha256 of one face.
    :return: ``main`` (its photograph), ``duplicate`` (listed in ``duplicate_photos``) or None.
    """
    if rec.get("image_sha256") in photos:
        return "main"
    if {str(s) for s in _as_list(rec.get("duplicate_photos"))} & photos:
        return "duplicate"
    return None


def index_records(records: Iterable[Mapping[str, Any]]
                  ) -> Tuple[Dict[Tuple[str, str], List[Mapping[str, Any]]], Dict[str, List[Mapping[str, Any]]]]:
    """Index pair records by (row_id, side_key) and by photograph.

    :param records: Pair records.
    :return: ``(by_side, by_photo)``; merged records are listed under every side they cover and duplicate
        photographs under every listed sha256.
    """
    by_side: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    by_photo: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        for sk in record_side_keys(rec):
            by_side[(str(rec.get("row_id")), sk)].append(rec)
        shas = {rec.get("image_sha256")} | {str(s) for s in _as_list(rec.get("duplicate_photos"))}
        for sha in shas - {None}:
            by_photo[str(sha)].append(rec)
    return by_side, by_photo


def outcome_on_photo(recs: Sequence[Mapping[str, Any]], photos: Set[str], side_key: str,
                     off_lines: Iterable[int] = (), n_lines: Optional[int] = None
                     ) -> Tuple[str, Optional[Mapping[str, Any]], Optional[str]]:
    """The gate's outcome for one side on one face.

    :param recs: The side's records.
    :param photos: sha256 of the face.
    :param side_key: The side.
    :param off_lines: Lines known not to be on the face.
    :param n_lines: Lines of the side, if known.
    :return: ``(outcome, record, match)`` with outcome in :data:`OUTCOMES`; match is ``main`` / ``duplicate`` / None.
    """
    if not recs:
        return "missing", None, None
    here = [(r, photo_match(r, photos)) for r in recs]
    here = [(r, m) for r, m in here if m]
    if here:
        here.sort(key=lambda rm: (DECISION_RANK.get(str(rm[0].get("decision")), 0), rm[1] == "main"), reverse=True)
        rec, match = here[0]
        return classify_record(rec, side_key, off_lines, n_lines), rec, match
    placed = [r for r in recs if r.get("decision") in ("accept", "review") and r.get("image_sha256")]
    if placed:
        placed.sort(key=lambda r: DECISION_RANK.get(str(r.get("decision")), 0), reverse=True)
        return "elsewhere", placed[0], None
    return "reject", recs[0], None


# --- fixture scoring ------------------------------------------------------------------------------------------------

def is_acceptable(outcome: str, rec: Optional[Mapping[str, Any]], acceptable: Sequence[str],
                  merge_ok: bool = True) -> bool:
    """Whether an outcome is one a fixture accepts (``"accept_as_is+merge"`` also requires the ``merge`` remedy and a
    merge with allowed partner lines only).

    :param outcome: Outcome class.
    :param rec: The record (for remedy requirements).
    :param acceptable: Acceptable outcome specs.
    :param merge_ok: Every other side merged into the record is an allowed partner with allowed lines.
    :return: True when acceptable.
    """
    rem = set(remedies_of(rec)) if rec is not None else set()
    for spec in acceptable:
        name, _, need = spec.partition("+")
        if outcome == name and (not need or (need in rem and (need != "merge" or merge_ok))):
            return True
    return False


def fixture_verdict(remedy: str, outcome: str, rec: Optional[Mapping[str, Any]], expected: Mapping[str, Any],
                    moved_elsewhere: bool, photo_other_text: bool, merge_ok: bool = True) -> Tuple[str, str]:
    """Verdict of one fixture under the §8.1 mapping.

    :param remedy: The fixture's remedy class.
    :param outcome: Gate outcome on the audited photo.
    :param rec: The record behind the outcome.
    :param expected: The fixture's ``expected`` block.
    :param moved_elsewhere: The side is accepted on a photograph of another face.
    :param photo_other_text: The audited photo is accepted with another text of the document.
    :param merge_ok: Sides merged into the record are allowed partners (see :func:`is_acceptable`).
    :return: ``(verdict, kind)``; verdict is ``pass`` / ``harm`` / ``loss`` / ``mismatch`` / ``not_repaired`` /
        ``missing`` / ``unscored``.
    """
    if remedy == "unsure":
        return "unscored", "unsure"
    if outcome == "missing":
        return "missing", "no record for this side"
    accepted = outcome in ACCEPT_OUTCOMES
    if remedy in USABLE_REMEDIES:
        acceptable = expected.get("acceptable") or [PASS_OUTCOME[remedy]]
        if accepted and is_acceptable(outcome, rec, acceptable, merge_ok):
            return "pass", outcome
        if not accepted:
            return "loss", "moved_away" if outcome == "elsewhere" else outcome
        if remedy in ("trim", "trim_and_crop") and outcome in ("accept_as_is", "accept_crop"):
            return "harm", "accepted_untrimmed"
        return "mismatch", MISMATCH_KINDS.get((remedy, outcome), "other")
    if remedy == "drop":
        return ("harm", "wrong_text_accepted") if accepted else ("pass", outcome)
    if accepted:
        return "harm", "accepted_on_wrong_photo"
    if moved_elsewhere or photo_other_text:
        kinds = [k for k, v in (("text_moved", moved_elsewhere), ("photo_got_other_text", photo_other_text)) if v]
        return "pass", "+".join(kinds)
    return "not_repaired", outcome


def line_check(fixture: Mapping[str, Any], rec: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Kept lines of an accepted record against the fixture's per-line truth.

    :param fixture: Fixture record.
    :param rec: The accepted record on the audited photo, or None.
    :return: ``{status, kept, kept_off, lost_on, kept_part, dropped_part, why}``; status is ``ok`` / ``kept_off`` /
        ``lost_on`` / ``kept_off+lost_on`` / ``index_mismatch`` / ``n/a``.
    """
    if rec is None or rec.get("decision") != "accept":
        return {"status": "n/a", "why": "not accepted"}
    if not fixture["expected"].get("line_check", True):
        return {"status": "n/a", "why": "fixture has no line-level truth"}
    if "reline" in remedies_of(rec):
        return {"status": "n/a", "why": "relined"}
    kept = kept_lines_for(rec, fixture["side_key"])
    if kept is None:
        return {"status": "n/a", "why": "record gives no line indices"}
    truth = fixture["line_truth"]
    if any(i < 0 or i >= len(truth) for i in kept):
        return {"status": "index_mismatch", "kept": kept, "why": f"index outside 0..{len(truth) - 1}"}
    ks = set(kept)
    kept_off = [i for i in sorted(ks) if truth[i] == "off"]
    lost_on = [i for i in range(len(truth)) if truth[i] == "on" and i not in ks]
    status = {(False, False): "ok", (True, False): "kept_off", (False, True): "lost_on",
              (True, True): "kept_off+lost_on"}[(bool(kept_off), bool(lost_on))]
    return {"status": status, "kept": sorted(ks), "kept_off": kept_off, "lost_on": lost_on,
            "kept_part": [i for i in sorted(ks) if truth[i] == "part"],
            "dropped_part": [i for i in range(len(truth)) if truth[i] == "part" and i not in ks], "why": ""}


def score_fixture(fixture: Mapping[str, Any], by_side: Mapping[Tuple[str, str], List[Mapping[str, Any]]],
                  by_photo: Mapping[str, List[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Score one fixture against indexed gate records.

    :param fixture: Fixture record (see the module docstring).
    :param by_side: Records by (row_id, side_key).
    :param by_photo: Records by photograph.
    :return: Scored item: outcome, verdict, kind, record summary, line check, sibling and rotation checks.
    """
    exp = fixture["expected"]
    row, side = fixture["row_id"], fixture["side_key"]
    photos = {fixture["image_sha256"], *exp.get("same_face_photos", [])}
    off = [i for i, s in enumerate(fixture["line_truth"]) if s == "off"]
    recs = by_side.get((row, side), [])
    outcome, rec, match = outcome_on_photo(recs, photos, side, off, len(fixture["line_truth"]))
    on_face = {id(r): r for sha in photos for r in by_photo.get(sha, [])}.values()
    other_texts = [r for r in on_face if r.get("decision") == "accept" and side not in
                   (record_side_keys(r) if str(r.get("row_id")) == row else [])
                   and (str(r.get("row_id")) == row or r.get("document_group") == fixture.get("document_group"))]
    moved = any(r.get("decision") == "accept" and not photo_match(r, photos) and r.get("image_sha256") for r in recs)
    merged_wrong: List[str] = []
    merged_duplicate: List[str] = []
    merged_unverified: List[str] = []
    merge_ok = True
    if rec is not None and outcome in ACCEPT_OUTCOMES:
        partners = exp.get("merge_partners") or {}
        for sk in record_side_keys(rec):
            kept = kept_lines_for(rec, sk)
            if sk == side or kept == []:
                continue
            if sk in exp.get("siblings_not_on_photo", []):
                merged_wrong.append(sk)
            elif sk in exp.get("duplicate_editions", []):
                merged_duplicate.append(sk)
            elif sk not in partners:
                merged_unverified.append(sk)
            if sk not in partners or kept is None or not set(kept) <= set(partners[sk]):
                merge_ok = False
    want = exp.get("photo_should_get")
    got_expected = None
    if want:
        got_expected = any(str(r.get("row_id")) == want["row_id"] and want["side_key"] in record_side_keys(r)
                           for r in other_texts)
    photo_other = got_expected if want else bool(other_texts)
    verdict, kind = fixture_verdict(fixture["remedy"], outcome, rec, exp, moved, bool(photo_other), merge_ok)
    if verdict not in ("harm", "unscored") and (merged_wrong or merged_duplicate):
        verdict, kind = "harm", "merged_text_not_on_photo" if merged_wrong else "merged_duplicate_edition"
    sibling_accepts = []
    for sk in exp.get("siblings_not_on_photo", []):
        for r in by_side.get((row, sk), []):
            kept = kept_lines_for(r, sk)
            if r.get("decision") == "accept" and photo_match(r, photos) and kept != []:
                sibling_accepts.append({"side_key": sk, "pair_id": r.get("pair_id"),
                                        "kept_lines": None if kept is None else len(kept)})
    rotation = None
    if exp.get("rotation") in (90, 180, 270) and outcome in ACCEPT_OUTCOMES and rec is not None:
        got = int(rec.get("rotation") or 0)
        rotation = {"wanted": exp["rotation"], "got": got,
                    "ok": got == exp["rotation"] or (exp["rotation"] == 180 and "rotate180" in remedies_of(rec))}
    lc = line_check(fixture, rec) if outcome in ACCEPT_OUTCOMES else {"status": "n/a", "why": "not accepted"}
    strict = verdict == "pass" and lc["status"] in ("ok", "n/a")
    return {
        "item_id": fixture["item_id"], "stratum": fixture["stratum"], "row_id": row, "side_key": side,
        "remedy": fixture["remedy"], "remedy_judgeB": fixture.get("remedy_judgeB"),
        "confidence": fixture.get("confidence"), "outcome": outcome, "verdict": verdict, "kind": kind,
        "strict_pass": strict,
        "match": match, "pair_id": rec.get("pair_id") if rec else None,
        "decision": rec.get("decision") if rec else None, "remedies": remedies_of(rec) if rec else [],
        "reasons": [str(x) for x in _as_list(rec.get("reasons"))] if rec else [],
        "record_photo": rec.get("image_sha256") if rec else None, "line_check": lc,
        "photo_other_texts": sorted({str(r.get("pair_id")) for r in other_texts}),
        "photo_got_expected_text": got_expected, "sibling_accepts": sibling_accepts,
        "merged_text_not_on_photo": merged_wrong, "merged_duplicate_edition": merged_duplicate,
        "merged_unverified": merged_unverified, "rotation": rotation,
        "expected_text": exp.get("text", ""),
    }


def score_fixtures(fixtures: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
                   ) -> List[Dict[str, Any]]:
    """Score every fixture.

    :param fixtures: Fixture records.
    :param records: Gate pair records.
    :return: Scored items in fixture order.
    """
    by_side, by_photo = index_records(records)
    return [score_fixture(fx, by_side, by_photo) for fx in fixtures]


def summarize_fixtures(scored: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Agreement matrix, verdict counts, harm and loss.

    :param scored: Output of :func:`score_fixtures`.
    :return: ``{matrix, verdicts, harm, harm_spec, loss, loss_spec, mismatch_kinds, line_status, sibling_harm,
        rotation_missed, n}``; item lists hold item ids.
    """
    matrix = {rem: Counter() for rem in REMEDY_CLASSES}
    verdicts = {rem: Counter() for rem in REMEDY_CLASSES}
    for s in scored:
        matrix.setdefault(s["remedy"], Counter())[s["outcome"]] += 1
        verdicts.setdefault(s["remedy"], Counter())[s["verdict"]] += 1
    harm = [s["item_id"] for s in scored if s["verdict"] == "harm"]
    loss = [s["item_id"] for s in scored if s["verdict"] == "loss"]
    merge_kinds = ("merged_text_not_on_photo", "merged_duplicate_edition")
    return {
        "n": len(scored), "matrix": {k: dict(v) for k, v in matrix.items()},
        "verdicts": {k: dict(v) for k, v in verdicts.items()},
        "harm": harm, "harm_spec": [s["item_id"] for s in scored if s["verdict"] == "harm"
                                    and s["remedy"] in SPEC_HARM_REMEDIES and s["kind"] not in merge_kinds],
        "harm_merged": [s["item_id"] for s in scored if s["verdict"] == "harm" and s["kind"] in merge_kinds],
        "loss": loss, "loss_spec": [s["item_id"] for s in scored
                                    if s["verdict"] == "loss" and s["remedy"] in SPEC_LOSS_REMEDIES],
        "mismatch_kinds": dict(Counter(s["kind"] for s in scored if s["verdict"] == "mismatch")),
        "strict_pass": [s["item_id"] for s in scored if s.get("strict_pass")],
        "pass_kept_off": [s["item_id"] for s in scored if s["verdict"] == "pass"
                          and "kept_off" in s["line_check"]["status"]],
        "pass_lost_on": [s["item_id"] for s in scored if s["verdict"] == "pass"
                         and "lost_on" in s["line_check"]["status"]],
        "line_status": dict(Counter(s["line_check"]["status"] for s in scored)),
        "sibling_harm": [s["item_id"] for s in scored if s["sibling_accepts"]],
        "merged_unverified": [s["item_id"] for s in scored if s.get("merged_unverified")],
        "rotation_missed": [s["item_id"] for s in scored if s["rotation"] and not s["rotation"]["ok"]],
        "missing": [s["item_id"] for s in scored if s["verdict"] == "missing"],
    }


# --- planted scoring ------------------------------------------------------------------------------------------------

def has_inversion(seq: Sequence[int]) -> bool:
    """Whether a sequence is not strictly increasing.

    :param seq: Values in output order.
    :return: True when some value is not larger than the one before it.
    """
    return any(b <= a for a, b in zip(seq, seq[1:]))


def score_planted_record(entry: Mapping[str, Any], recs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Score the gate's records of one planted candidate.

    :param entry: Manifest entry (``row_id, class, side_key, n_lines, line_photos, permutation``).
    :param recs: The gate's records for the planted row.
    :return: ``{row_id, class, outcome, false_accept, accepts: [...], n_records}``; outcome is ``missing`` /
        ``false_accept`` / ``unverifiable`` / ``trimmed_ok`` / ``restored`` / ``subset_accept`` / ``accept_empty`` /
        ``review`` / ``reject``.
    """
    cls, n = entry["class"], int(entry["n_lines"])
    truth = [set(x) for x in entry["line_photos"]]
    perm = entry.get("permutation")
    base = {"row_id": entry["row_id"], "class": cls, "n_records": len(recs), "accepts": []}
    if not recs:
        return {**base, "outcome": "missing", "false_accept": False}
    false_accept = unverifiable = restored = False
    kept_any = False
    for rec in (r for r in recs if r.get("decision") == "accept"):
        kept = kept_lines_for(rec, str(entry.get("side_key", "")))
        assumed = kept is None
        if kept is None:
            kept = list(range(n))
        photo = rec.get("image_sha256")
        outside = [i for i in kept if not 0 <= i < n]
        inside = [i for i in kept if 0 <= i < n]
        wrong = [i for i in inside if photo not in truth[i]]
        inverted = perm is not None and has_inversion([perm[i] for i in inside])
        relined = "reline" in remedies_of(rec)
        if wrong or inverted:
            false_accept = True
        elif outside or relined:
            unverifiable = True
        if perm is not None and not inverted and not outside and len(set(inside)) == n:
            restored = True
        kept_any = kept_any or bool(inside)
        base["accepts"].append({"pair_id": rec.get("pair_id"), "photo": photo, "kept": inside,
                                "assumed_all": assumed, "wrong_lines": wrong, "order_inverted": inverted,
                                "outside": outside, "relined": relined, "remedies": remedies_of(rec)})
    if false_accept:
        outcome = "false_accept"
    elif unverifiable:
        outcome = "unverifiable"
    elif base["accepts"]:
        if not kept_any:
            outcome = "accept_empty"
        elif cls == "shuffled":
            outcome = "restored" if restored else "subset_accept"
        else:
            outcome = "trimmed_ok"
    elif any(r.get("decision") == "review" for r in recs):
        outcome = "review"
    else:
        outcome = "reject"
    return {**base, "outcome": outcome, "false_accept": false_accept}


def score_planted(manifest: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
                  ) -> List[Dict[str, Any]]:
    """Score every planted candidate.

    :param manifest: Manifest entries.
    :param records: Gate pair records of the planted run.
    :return: Scored entries in manifest order.
    """
    by_row: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        by_row[str(rec.get("row_id"))].append(rec)
    return [score_planted_record(m, by_row.get(m["row_id"], [])) for m in manifest]


def planted_rates(scored: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """False-accept rate and outcome counts per planted class.

    :param scored: Output of :func:`score_planted`.
    :return: Class -> ``{n, scored, false_accept, fa_rate, fa_rate_upper, outcomes}``; ``fa_rate_upper`` also counts
        unverifiable accepts.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for cls in PLANTED_CLASSES + tuple(sorted({s["class"] for s in scored} - set(PLANTED_CLASSES))):
        rows = [s for s in scored if s["class"] == cls]
        if not rows:
            continue
        outcomes = Counter(s["outcome"] for s in rows)
        n_scored = len(rows) - outcomes.get("missing", 0)
        fa = outcomes.get("false_accept", 0)
        upper = fa + outcomes.get("unverifiable", 0)
        out[cls] = {"n": len(rows), "scored": n_scored, "false_accept": fa,
                    "fa_rate": round(fa / n_scored, 4) if n_scored else None,
                    "fa_rate_upper": round(upper / n_scored, 4) if n_scored else None, "outcomes": dict(outcomes)}
    return out


# --- planting -------------------------------------------------------------------------------------------------------

@dataclass
class Donor:
    """A stage-2 ``accepted`` side usable as planting material.

    :ivar row_id: Candidate id.
    :ivar side_key: Side key.
    :ivar document_group: Document group.
    :ivar source: Candidate source.
    :ivar collection: Collection prefix (:func:`collection_prefix`).
    :ivar script: ``hebrew`` / ``arabic`` (of the target).
    :ivar band: ``ja`` / ``heb`` / ``ar``.
    :ivar photo: The stage-2 photograph (sha256).
    :ivar face: The photograph plus same-face copies of it in the candidate (pHash <= :data:`SAME_FACE_PHASH`).
    :ivar lines: Target lines.
    :ivar letters: Target letters.
    :ivar evidence_lines: Lines with at least five letters.
    :ivar tokens: Skeleton word tokens.
    :ivar canonical_ids: The candidate's canonical ids.
    """

    row_id: str
    side_key: str
    document_group: str
    source: str
    collection: str
    script: str
    band: str
    photo: str
    face: List[str]
    lines: List[str]
    letters: int
    evidence_lines: int
    tokens: List[str]
    canonical_ids: List[str]


def is_rnl(candidate: Mapping[str, Any]) -> bool:
    """Whether a candidate names Russian National Library holdings (training-only material, kept out of planting).

    :param candidate: Stage-1 record.
    :return: True for RNL material.
    """
    if candidate.get("rnl"):
        return True
    names = " ".join([str(candidate.get("shelfmark") or "")] + [str(c) for c in candidate.get("canonical_ids") or []])
    return bool(RNL_RE.search(names.replace("_", " ")))


def collection_prefix(candidate: Mapping[str, Any]) -> str:
    """Collection prefix of a candidate: the first three tokens of its first non-APD canonical id
    (``Cambridge_CUL_T``, ``New_York_JTS``, ``Manchester_JRL_B``...), else the head of its shelf-mark
    (``P.Vind``, ``P.Louvre``).

    :param candidate: Stage-1 record.
    :return: Prefix string.
    """
    for cid in candidate.get("canonical_ids") or []:
        if cid and not str(cid).startswith("apd:"):
            return "_".join(str(cid).split("_")[:3])
    shelf = re.sub(r"\s+", " ", str(candidate.get("shelfmark") or "")).strip()
    head = re.split(r"[\s\d,:]", shelf, maxsplit=1)[0] if shelf else ""
    head = re.sub(r"\.?inv\.?$", "", head)
    return "shelfmark:" + (head or "unknown")


def image_on_disk(image: Mapping[str, Any], exists: Callable[[str], bool] = os.path.exists) -> bool:
    """Whether an image entry has its file on disk.

    :param image: Image entry of a stage-1 record.
    :param exists: Path check (injected for tests).
    :return: True for an on-disk status, a sha256 and an existing path.
    """
    return (image.get("status") in ON_DISK_STATUS and bool(image.get("sha256")) and bool(image.get("path"))
            and exists(str(image["path"])))


def read_shas(reads: Iterable[Mapping[str, Any]], min_letters: int = 10) -> Set[str]:
    """Photographs with at least one usable read.

    :param reads: Read records (``reads.jsonl``).
    :param min_letters: Minimum letters of the read.
    :return: sha256 set.
    """
    out: Set[str] = set()
    for r in reads:
        n_letters = int(r.get("letters_hebrew") or 0) + int(r.get("letters_arabic") or 0)
        if r.get("finish") == "ok" and n_letters >= min_letters:
            out.add(str(r["sha256"]))
    return out


def same_face(candidate: Mapping[str, Any], sha: str, exists: Callable[[str], bool] = os.path.exists) -> List[str]:
    """A photograph plus its same-face copies among a candidate's on-disk images.

    :param candidate: Stage-1 record.
    :param sha: The photograph.
    :param exists: Path check.
    :return: sha256 list, ``sha`` first.
    """
    ims = [im for im in candidate.get("images") or [] if image_on_disk(im, exists)]
    ref = next((im for im in ims if im["sha256"] == sha), None)
    out = [sha]
    for im in ims:
        d = hamming(ref.get("phash") if ref else None, im.get("phash"))
        if im["sha256"] not in out and d is not None and d <= SAME_FACE_PHASH:
            out.append(im["sha256"])
    return out


def side_material(candidate: Mapping[str, Any], side: Mapping[str, Any], with_reads: Set[str],
                  exists: Callable[[str], bool] = os.path.exists) -> Optional[Donor]:
    """A clean stage-2 ``accepted`` side as planting material, without size limits.

    :param candidate: Stage-1 record of the side.
    :param side: Its stage-2 decision.
    :param with_reads: Photographs with usable reads.
    :param exists: Path check.
    :return: The side, or None when it is not accepted cleanly (stage-2 reasons), its photograph has no read or file,
        its candidate is RNL material, or its target is not Hebrew or Arabic script.
    """
    if side.get("outcome") != "accepted" or side.get("reasons") or side.get("image_sha256") not in with_reads:
        return None
    if is_rnl(candidate):
        return None
    t = next((t for t in candidate.get("targets") or [] if t.get("side_key") == side["side_key"]), None)
    photo = next((im for im in candidate.get("images") or [] if im.get("sha256") == side["image_sha256"]
                  and image_on_disk(im, exists)), None)
    if t is None or photo is None:
        return None
    lines = [ln for ln in t.get("lines") or [] if ln.strip()]
    if not lines:
        return None
    line_letters = [letters(ln) for ln in lines]
    script = script_of("\n".join(lines))
    if script not in ("hebrew", "arabic"):
        return None
    return Donor(row_id=candidate["row_id"], side_key=side["side_key"], document_group=candidate["document_group"],
                 source=candidate["source"], collection=collection_prefix(candidate), script=script,
                 band=ja_band(script, "\n".join(lines)), photo=side["image_sha256"],
                 face=same_face(candidate, side["image_sha256"], exists), lines=lines, letters=sum(line_letters),
                 evidence_lines=sum(1 for n in line_letters if n >= 5), tokens=word_tokens("\n".join(lines)),
                 canonical_ids=list(candidate.get("canonical_ids") or []))


def is_flattened(d: Donor, max_line_letters: int = 120) -> bool:
    """Whether a side has a line longer than a manuscript line (a flattened text).

    :param d: The side.
    :param max_line_letters: Longest line allowed.
    :return: True when some line is longer.
    """
    return any(letters(ln) > max_line_letters for ln in d.lines)


def build_donors(candidates: Mapping[str, Mapping[str, Any]], sides: Iterable[Mapping[str, Any]],
                 with_reads: Set[str], exclude_groups: Set[str],
                 exists: Callable[[str], bool] = os.path.exists) -> List[Donor]:
    """Planting material: clean stage-2 ``accepted`` sides whose photograph is on disk and has a read.

    Kept (:func:`side_material`, then): not in an excluded group (the audited fixtures); a target of >= 60 letters with
    >= 3 lines of >= 5 letters and no line over 120 letters (no flattened texts).

    :param candidates: Stage-1 records by row_id.
    :param sides: Stage-2 side decisions.
    :param with_reads: Photographs with usable reads.
    :param exclude_groups: Document groups never used.
    :param exists: Path check.
    :return: Donors sorted by (row_id, side_key).
    """
    out: List[Donor] = []
    for s in sides:
        c = candidates.get(s["row_id"])
        if c is None or c["document_group"] in exclude_groups:
            continue
        d = side_material(c, s, with_reads, exists)
        if d is None or is_flattened(d) or d.letters < 60 or d.evidence_lines < 3:
            continue
        out.append(d)
    out.sort(key=lambda d: (d.row_id, d.side_key))
    return out


#: Word n-gram length of the formula shingles per script (Arabic-script formulae vary too much in spelling for 3-grams).
SHINGLE_N = {"hebrew": 3, "arabic": 2}


@dataclass
class ShingleStats:
    """Document frequency of word shingles per script (counted per document group; :data:`SHINGLE_N` words).

    :ivar df: Script -> shingle -> number of groups holding it.
    :ivar groups: Script -> number of groups.
    :ivar by_donor: (row_id, side_key) -> the donor's shingles.
    """

    df: Dict[str, Counter]
    groups: Dict[str, int]
    by_donor: Dict[Tuple[str, str], Set[Tuple[str, ...]]]

    def formula_min(self, script: str) -> int:
        """Lowest document frequency of a formula shingle.

        :param script: ``hebrew`` / ``arabic``.
        :return: 5 for Hebrew script, 3 for Arabic script (smaller pool).
        """
        return 5 if script == "hebrew" else 3

    def stop_max(self, script: str) -> float:
        """Document frequency above which a formula is too common to mark a genre (e.g. the basmala).

        :param script: ``hebrew`` / ``arabic``.
        :return: 25 % of the script's groups.
        """
        return 0.25 * max(1, self.groups.get(script, 1))


def shingle_stats(donors: Sequence[Donor]) -> ShingleStats:
    """Shingle document frequencies over the donor pool.

    :param donors: Donors.
    :return: The statistics.
    """
    df: Dict[str, Counter] = {"hebrew": Counter(), "arabic": Counter()}
    per_group: Dict[Tuple[str, str], Set[Tuple[str, ...]]] = defaultdict(set)
    by_donor: Dict[Tuple[str, str], Set[Tuple[str, ...]]] = {}
    for d in donors:
        sh = shingles(d.tokens, SHINGLE_N.get(d.script, 3))
        by_donor[(d.row_id, d.side_key)] = sh
        per_group[(d.script, d.document_group)] |= sh
    groups: Counter = Counter()
    for (script, _g), sh in per_group.items():
        df[script].update(sh)
        groups[script] += 1
    return ShingleStats(df=df, groups=dict(groups), by_donor=by_donor)


def pair_overlap(a: Donor, b: Donor, stats: ShingleStats) -> Dict[str, Any]:
    """Shared shingles of two donors, split into exclusive, content and formula shingles.

    :param a: First donor.
    :param b: Second donor.
    :param stats: Shingle statistics.
    :return: ``{exclusive, content, shared_fraction, formula, formula_score, formula_shingles}``: exclusive = held by
        these two groups only; content = rarer than a formula; shared_fraction = shared shingles / the smaller set;
        formula = at least :meth:`ShingleStats.formula_min` groups and not a stop formula; the score sums
        ``log(groups / df)`` over formula shingles.
    """
    df = stats.df[a.script]
    sa, sb = stats.by_donor[(a.row_id, a.side_key)], stats.by_donor[(b.row_id, b.side_key)]
    shared = sa & sb
    fmin, stop = stats.formula_min(a.script), stats.stop_max(a.script)
    formula = sorted((s for s in shared if fmin <= df[s] <= stop), key=lambda s: (df[s], s))
    n_groups = max(1, stats.groups.get(a.script, 1))
    return {"exclusive": sum(1 for s in shared if df[s] <= 2), "content": sum(1 for s in shared if df[s] < fmin),
            "shared_fraction": round(len(shared) / max(1, min(len(sa), len(sb))), 4),
            "formula": len(formula), "formula_score": round(math.fsum(math.log(n_groups / df[s]) for s in formula), 3),
            "formula_shingles": [" ".join(s) for s in formula[:8]]}


def same_text_risk(overlap: Mapping[str, Any]) -> bool:
    """Whether two texts may be editions of one text (then they are never planted as a wrong pair).

    :param overlap: Output of :func:`pair_overlap`.
    :return: True for more than two shingles held by the pair alone, more than six rarer-than-formula shingles, or a
        tenth or more of the smaller text's shingles shared.
    """
    return overlap["exclusive"] > 2 or overlap["content"] > 6 or overlap["shared_fraction"] >= 0.1


def genre_need(script: str, genre: str) -> int:
    """Shared formula shingles a same-genre swap needs.

    :param script: ``hebrew`` / ``arabic``.
    :param genre: Genre family of both texts (:func:`genre_of`).
    :return: Hebrew script 2 for a named family (letter, deed), else 3; Arabic script (word pairs) 3, else 4.
    """
    base = 2 if script == "hebrew" else 3
    return base if genre != "other" else base + 1


def _class_rng(seed: int, cls: str) -> random.Random:
    """Deterministic generator per planted class.

    :param seed: Base seed.
    :param cls: Class name.
    :return: Seeded generator.
    """
    return random.Random(seed * 1000003 + zlib.crc32(cls.encode("utf-8")))


def _distinct(a: Donor, b: Donor) -> bool:
    """Whether two donors are different documents from different collections.

    :param a: First donor.
    :param b: Second donor.
    :return: True for another group, no shared canonical id and another collection prefix.
    """
    return (a.document_group != b.document_group and not set(a.canonical_ids) & set(b.canonical_ids)
            and a.collection != b.collection)


def no_shuffle_adjacency(n: int, rng: random.Random, tries: int = 500) -> List[int]:
    """A permutation of ``range(n)`` in which no two neighbours keep their original succession.

    :param n: Length (>= 4 for a result without kept successions).
    :param rng: Generator.
    :param tries: Rejection-sampling budget before a fixed fallback.
    :return: ``perm`` with ``perm[k]`` = original index of the line placed at position k.
    """
    base = list(range(n))
    for _ in range(tries):
        perm = base[:]
        rng.shuffle(perm)
        if all(b != a + 1 for a, b in zip(perm, perm[1:])):
            return perm
    return base[1::2][::-1] + base[0::2][::-1]


def _images_for(candidate: Mapping[str, Any], shas: Set[str], exists: Callable[[str], bool]) -> List[Dict[str, Any]]:
    """Copies of a candidate's on-disk image entries whose sha256 is in a set.

    :param candidate: Stage-1 record.
    :param shas: Wanted photographs.
    :param exists: Path check.
    :return: Image entries (duplicates kept as listed).
    """
    return [copy.deepcopy(dict(im)) for im in candidate.get("images") or []
            if image_on_disk(im, exists) and im["sha256"] in shas]


def other_faces(donor: Donor, accepted: Mapping[str, Sequence[Tuple[str, str]]],
                candidate: Mapping[str, Any], exists: Callable[[str], bool]) -> List[str]:
    """Photographs of the donor's document that hold another accepted side on a clearly different face.

    :param donor: The donor.
    :param accepted: row_id -> [(side_key, sha256)] of every stage-2 accepted side.
    :param candidate: The donor's stage-1 record.
    :param exists: Path check.
    :return: sha256 of those faces (with their same-face copies).
    """
    ims = {im["sha256"]: im for im in candidate.get("images") or [] if image_on_disk(im, exists)}
    ref = ims.get(donor.photo, {})
    out: List[str] = []
    for side_key, sha in accepted.get(donor.row_id, []):
        if side_key == donor.side_key or sha in donor.face or sha not in ims:
            continue
        d = hamming(ref.get("phash"), ims[sha].get("phash"))
        if d is not None and d > OTHER_FACE_PHASH:
            out.extend(s for s in same_face(candidate, sha, exists) if s not in out and s not in donor.face)
    return out


def planted_candidate(cls: str, n: int, text_donor: Mapping[str, Any], side_key: str, lines: Sequence[str],
                      images: Sequence[Mapping[str, Any]], planted_notes: Mapping[str, Any]) -> Dict[str, Any]:
    """A planted stage-1 candidate (same fields as ``stage1_images.jsonl``).

    :param cls: Planted class.
    :param n: Number within the class (1-based).
    :param text_donor: Stage-1 record of the text donor (source, script, unit).
    :param side_key: Side key of the planted target.
    :param lines: Planted lines.
    :param images: Image entries.
    :param planted_notes: Provenance for ``notes.planted``.
    :return: The record; row_id ``planted_<class>:<n>``, document_group ``planted:<class>:<n>``, verdict ``planted``.
    """
    text = "\n".join(lines)
    sk = skeleton(text)
    heb = sum(1 for ch in sk if _is_hebrew(ch))
    return {
        "row_id": f"planted_{cls}:{n:03d}", "source": text_donor.get("source"),
        "document_group": f"planted:{cls}:{n:03d}", "canonical_ids": [], "pgpids": [], "ktiv_id": None,
        "shelfmark": f"planted {cls} {n:03d}", "script": text_donor.get("script"),
        "languages_stated": text_donor.get("languages_stated") or [], "verdict": "planted", "witnesses": [],
        "unit": text_donor.get("unit"), "n_letters": len(sk),
        "targets": [{"side_key": side_key, "text": text, "raw_text": text, "lines": list(lines),
                     "pointer": {"planted": cls, "n": n, "manifest": MANIFEST_NAME}, "side_marker": None,
                     "letters_hebrew": heb, "letters_arabic": len(sk) - heb}],
        "images": [dict(im) for im in images], "rnl": False,
        "notes": {"planted": dict(planted_notes), "edition_source": "planted negative (acceptance test only)"},
    }


def _donor_ref(d: Donor, role: str) -> Dict[str, Any]:
    """Manifest description of a donor.

    :param d: The donor.
    :param role: Its role (``text`` / ``photos`` / ``appended``).
    :return: Description.
    """
    return {"role": role, "row_id": d.row_id, "side_key": d.side_key, "document_group": d.document_group,
            "collection": d.collection, "source": d.source, "photo": d.photo, "letters": d.letters,
            "n_lines": len(d.lines), "band": d.band}


@dataclass
class ClassUsage:
    """Document groups used within one planted class.

    :ivar reuse_roles: When True (``same_genre``) a group may serve once per role (text donor, photo donor) and a
        pair is never planted in both directions; otherwise a group serves once in the class.
    :ivar by_role: Role (``text`` / ``photos`` / ``appended``) -> groups.
    :ivar pairs: (text group, photo group) pairs planted.
    """

    reuse_roles: bool = False
    by_role: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    pairs: Set[Tuple[str, str]] = field(default_factory=set)

    def groups(self) -> Set[str]:
        """Every group used in the class.

        :return: Group ids.
        """
        return set().union(*self.by_role.values()) if self.by_role else set()

    def free(self, group: str, role: str) -> bool:
        """Whether a group may take a role.

        :param group: Document group.
        :param role: Role.
        :return: True when allowed.
        """
        return group not in (self.by_role[role] if self.reuse_roles else self.groups())

    def add(self, group: str, role: str) -> None:
        """Record a use.

        :param group: Document group.
        :param role: Role.
        """
        self.by_role[role].add(group)


class Planter:
    """Builds the planted classes from one donor pool with one seed.

    :ivar candidates: Stage-1 records by row_id.
    :ivar donors: Donor pool (:func:`build_donors`).
    :ivar by_script: The pool by script.
    :ivar stats: Shingle statistics of the pool.
    :ivar sides_by_row: row_id -> stage-2 ``accepted`` side decisions (clean or not).
    :ivar accepted: row_id -> [(side_key, sha256)] of those decisions.
    :ivar with_reads: Photographs with usable reads.
    :ivar seed: Base seed.
    :ivar exists: Path check.
    :ivar used: Document groups used by earlier classes (avoided while the pool allows).
    """

    def __init__(self, candidates: Mapping[str, Mapping[str, Any]], donors: Sequence[Donor], stats: ShingleStats,
                 sides_by_row: Mapping[str, Sequence[Mapping[str, Any]]], with_reads: Set[str], seed: int,
                 exists: Callable[[str], bool] = os.path.exists) -> None:
        """Keep the inputs.

        :param candidates: Stage-1 records by row_id.
        :param donors: Donor pool.
        :param stats: Shingle statistics.
        :param sides_by_row: Accepted stage-2 decisions per row.
        :param with_reads: Photographs with usable reads.
        :param seed: Base seed.
        :param exists: Path check.
        """
        self.candidates = candidates
        self.donors = sorted(donors, key=lambda d: (d.row_id, d.side_key))
        self.by_script: Dict[str, List[Donor]] = defaultdict(list)
        for d in self.donors:
            self.by_script[d.script].append(d)
        self.stats = stats
        self.sides_by_row = sides_by_row
        self.accepted = {row: [(s["side_key"], s["image_sha256"]) for s in ss if s.get("image_sha256")]
                         for row, ss in sides_by_row.items()}
        self.with_reads = with_reads
        self.seed = seed
        self.exists = exists
        self.used: Set[str] = set()
        self._sg_cache: Dict[Tuple[str, str], List[Tuple[float, Donor, Dict[str, Any]]]] = {}

    def _free(self, d: Donor, usage: ClassUsage, role: str, strict: bool) -> bool:
        """Whether a donor's group may still take a role.

        :param d: Donor.
        :param usage: Usage in this class.
        :param role: ``text`` / ``photos`` / ``appended``.
        :param strict: Also avoid groups used by earlier classes.
        :return: True when free.
        """
        return usage.free(d.document_group, role) and not (strict and d.document_group in self.used)

    def _same_genre_partners(self, a: Donor) -> List[Tuple[float, Donor, Dict[str, Any]]]:
        """Every possible same-genre photo donor for A, ignoring group usage (cached).

        Same script, another group and collection prefix, B's text 0.67-1.5x A's letters, no same-text risk
        (:func:`same_text_risk`), the same genre family (:func:`genre_of`) and at least :func:`genre_need` shared
        formula shingles.

        :param a: Text donor.
        :return: ``(formula score, B, overlap)`` list.
        """
        key = (a.row_id, a.side_key)
        if key not in self._sg_cache:
            genre = genre_of(a.tokens, a.script)
            out: List[Tuple[float, Donor, Dict[str, Any]]] = []
            for b in self.by_script[a.script]:
                if not _distinct(a, b) or not 0.67 <= a.letters / max(1, b.letters) <= 1.5:
                    continue
                ov = pair_overlap(a, b, self.stats)
                if same_text_risk(ov) or ov["formula"] < genre_need(a.script, genre):
                    continue
                if genre_of(b.tokens, b.script) == genre:
                    out.append((ov["formula_score"], b, {**ov, "genre": genre}))
            self._sg_cache[key] = out
        return self._sg_cache[key]

    def _swap(self, cls: str, a: Donor, rng: random.Random, usage: ClassUsage, strict: bool
              ) -> Optional[Tuple[Donor, Dict[str, Any]]]:
        """Photo donor B for a cross-document or same-genre swap of A's text.

        ``cross_doc``: same script, another group and collection prefix, B's text 0.8-1.25x A's letters, no same-text
        risk and no shared formula shingle; B drawn at random. ``same_genre``: the free partner
        (:meth:`_same_genre_partners`) with the highest formula score.

        :param cls: ``cross_doc`` or ``same_genre``.
        :param a: Text donor.
        :param rng: Generator.
        :param usage: Usage in this class.
        :param strict: Avoid groups used by earlier classes.
        :return: ``(B, overlap)`` or None.
        """
        if cls == "same_genre":
            free = [x for x in self._same_genre_partners(a) if self._free(x[1], usage, "photos", strict)
                    and (x[1].document_group, a.document_group) not in usage.pairs]
            if not free:
                return None
            _, b, ov = max(free, key=lambda x: (x[0], x[1].row_id, x[1].side_key))
            return b, ov
        found: List[Tuple[Donor, Dict[str, Any]]] = []
        for b in self.by_script[a.script]:
            if not _distinct(a, b) or not self._free(b, usage, "photos", strict):
                continue
            if not 0.8 <= a.letters / max(1, b.letters) <= 1.25:
                continue
            ov = pair_overlap(a, b, self.stats)
            if not same_text_risk(ov) and not ov["formula"]:
                found.append((b, ov))
        return found[rng.randrange(len(found))] if found else None

    def _other_face_side(self, a: Donor) -> Optional[Donor]:
        """Another cleanly accepted side of A's candidate on a clearly different face (pHash distance > 10), with at
        least 30 letters on two lines of five letters and no flattened line.

        :param a: The side.
        :return: That side, or None.
        """
        c = self.candidates[a.row_id]
        phash = {im.get("sha256"): im.get("phash") for im in c.get("images") or []}
        out: List[Donor] = []
        for s in self.sides_by_row.get(a.row_id, []):
            if s["side_key"] == a.side_key:
                continue
            b = side_material(c, s, self.with_reads, self.exists)
            if b is None or set(b.face) & set(a.face) or is_flattened(b) or b.letters < 30 or b.evidence_lines < 2:
                continue
            d = hamming(phash.get(a.photo), phash.get(b.photo))
            if d is not None and d > OTHER_FACE_PHASH:
                out.append(b)
        return sorted(out, key=lambda d: d.side_key)[0] if out else None

    def _photos_with_reads(self, row_id: str) -> Set[str]:
        """On-disk photographs of a candidate that have a read.

        :param row_id: Candidate id.
        :return: sha256 set.
        """
        return {im["sha256"] for im in self.candidates[row_id].get("images") or []
                if image_on_disk(im, self.exists) and im["sha256"] in self.with_reads}

    def _all_photos(self, row_id: str) -> Set[str]:
        """Every listed photograph of a candidate (for the null exclusion).

        :param row_id: Candidate id.
        :return: sha256 set.
        """
        return {im["sha256"] for im in self.candidates[row_id].get("images") or [] if im.get("sha256")}

    def plant_one(self, cls: str, a: Donor, n: int, rng: random.Random, usage: ClassUsage, strict: bool
                  ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], List[Tuple[str, str]]]]:
        """Try to plant one record of a class from text donor A.

        :param cls: Planted class.
        :param a: Text donor (the side whose lines lead the planted target).
        :param n: Number within the class.
        :param rng: Generator.
        :param usage: Usage in this class.
        :param strict: Avoid groups used by earlier classes.
        :return: ``(candidate, manifest entry, [(group, role)])`` or None when A does not fit the class.
        """
        ca = self.candidates[a.row_id]
        donors = [_donor_ref(a, "text")]
        selection: Dict[str, Any] = {}
        perm: Optional[List[int]] = None
        na = len(a.lines)
        if cls in ("cross_doc", "same_genre"):
            found = self._swap(cls, a, rng, usage, strict)
            if found is None:
                return None
            b, ov = found
            shas = self._photos_with_reads(b.row_id)
            images = _images_for(self.candidates[b.row_id], shas, self.exists)
            lines = list(a.lines)
            line_photos: List[List[str]] = [[] for _ in lines]
            donors.append(_donor_ref(b, "photos"))
            segments = [{"from": 0, "to": na - 1, "origin": "text of another document", "photos": []}]
            selection = {"letters_ratio": round(a.letters / max(1, b.letters), 3), **ov}
            expected = {"outcome": "reject", "text": "no accept on any photo (every line belongs to another document)"}
            roles = {s: "photos of the other document" for s in sorted(shas)}
        elif cls == "other_face_append":
            b = self._other_face_side(a)
            if b is None:
                return None
            lines = list(a.lines) + list(b.lines)
            line_photos = [list(a.face)] * na + [list(b.face)] * len(b.lines)
            images = _images_for(ca, set(a.face) | set(b.face), self.exists)
            donors.append(_donor_ref(b, "appended"))
            segments = [{"from": 0, "to": na - 1, "origin": "side", "photos": list(a.face)},
                        {"from": na, "to": len(lines) - 1, "origin": "other face (appended)", "photos": list(b.face)}]
            expected = {"outcome": "trim_or_move",
                        "text": f"accept lines {compress_ranges(range(na))} (1-based) on photo {a.photo[:12]}; lines "
                                f"{compress_ranges(range(na, len(lines)))} trimmed there or moved to {b.photo[:12]} "
                                f"(the other face) only"}
            roles = {**{s: "face of the side" for s in a.face}, **{s: "other face" for s in b.face}}
        elif cls == "concat_docs":
            pool = [c for c in self.by_script[a.script] if _distinct(a, c) and self._free(c, usage, "appended", strict)
                    and 0.3 <= c.letters / max(1, a.letters) <= 1.2]
            ok = [c for c in pool if not same_text_risk(pair_overlap(a, c, self.stats))]
            if not ok:
                return None
            c_don = ok[rng.randrange(len(ok))]
            extra = other_faces(a, self.accepted, ca, self.exists)
            lines = list(a.lines) + list(c_don.lines)
            line_photos = [list(a.face)] * na + [[] for _ in c_don.lines]
            images = _images_for(ca, set(a.face) | set(extra), self.exists)
            donors.append(_donor_ref(c_don, "appended"))
            segments = [{"from": 0, "to": na - 1, "origin": "side", "photos": list(a.face)},
                        {"from": na, "to": len(lines) - 1, "origin": "unrelated document (appended)", "photos": []}]
            selection = {"letters_ratio_appended": round(c_don.letters / max(1, a.letters), 3)}
            expected = {"outcome": "trim",
                        "text": f"accept lines {compress_ranges(range(na))} (1-based) on photo {a.photo[:12]}; lines "
                                f"{compress_ranges(range(na, len(lines)))} trimmed (they are on no photo)"}
            roles = {**{s: "face of the side" for s in a.face}, **{s: "other face of the document" for s in extra}}
        elif cls == "shuffled":
            if na < 6 or a.evidence_lines < 5:
                return None
            perm = no_shuffle_adjacency(na, rng)
            lines = [a.lines[k] for k in perm]
            extra = other_faces(a, self.accepted, ca, self.exists)
            line_photos = [list(a.face)] * na
            images = _images_for(ca, set(a.face) | set(extra), self.exists)
            segments = [{"from": 0, "to": na - 1, "origin": "side, line order permuted", "photos": list(a.face)}]
            expected = {"outcome": "reject_or_review",
                        "text": f"no accept with the planted order; an accept on {a.photo[:12]} must keep lines in "
                                f"their true order (permutation[k] = true index of planted line k)"}
            roles = {**{s: "face of the side" for s in a.face}, **{s: "other face of the document" for s in extra}}
        else:
            raise ValueError(f"unknown planted class {cls}")
        if not images:
            return None
        groups = [(d["document_group"], d["role"]) for d in donors]
        null_exclude = sorted(set().union(*(self._all_photos(d["row_id"]) for d in donors)))
        notes = {"class": cls, "n": n, "seed": self.seed, "never_build": True, "manifest": MANIFEST_NAME,
                 "donors": donors, "null_exclude_sha256": null_exclude}
        cand = planted_candidate(cls, n, ca, a.side_key, lines, images, notes)
        entry = {"row_id": cand["row_id"], "document_group": cand["document_group"], "class": cls, "n": n,
                 "file": f"{cls}.jsonl", "side_key": a.side_key, "script": a.script, "band": a.band,
                 "n_lines": len(lines), "letters": cand["n_letters"], "donors": donors, "segments": segments,
                 "line_photos": line_photos, "permutation": perm,
                 "photos": sorted({im["sha256"] for im in images}), "photo_roles": roles, "expected": expected,
                 "selection": selection, "null_exclude_sha256": null_exclude, "disjoint_from_other_classes": strict}
        return cand, entry, groups

    def plant_class(self, cls: str, per_class: int, quotas: Mapping[str, int]
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Plant one class. Bands are filled to their quotas, first from groups no earlier class used, then from any
        group; a remaining shortfall is filled from Judaeo-Arabic, then Hebrew, then Arabic sides. Within a class a
        document group is used once (``same_genre``: once as text donor and once as photo donor, see
        :class:`ClassUsage`).

        :param cls: Planted class.
        :param per_class: Records wanted.
        :param quotas: Band -> records wanted (``ar``, ``ja``, ``heb``).
        :return: ``(candidates, manifest entries)``.
        """
        rng = _class_rng(self.seed, cls)
        order = list(self.donors)
        rng.shuffle(order)
        if cls == "same_genre":
            # scarcest text donors first, so the matching uses rare partners before common ones
            n_partners = {(d.row_id, d.side_key): len(self._same_genre_partners(d)) for d in order}
            order = sorted((d for d in order if n_partners[(d.row_id, d.side_key)]),
                           key=lambda d: n_partners[(d.row_id, d.side_key)])
        usage = ClassUsage(reuse_roles=cls == "same_genre")
        cands: List[Dict[str, Any]] = []
        entries: List[Dict[str, Any]] = []
        got: Counter = Counter()

        def fill(band: str, need: int, strict: bool) -> None:
            """Plant up to ``need`` records of one band.

            :param band: Band.
            :param need: Records wanted.
            :param strict: Avoid groups used by earlier classes.
            """
            k = 0
            for a in order:
                if k >= need or len(cands) >= per_class:
                    return
                if a.band != band or not self._free(a, usage, "text", strict):
                    continue
                res = self.plant_one(cls, a, len(cands) + 1, rng, usage, strict)
                if res is None:
                    continue
                cand, entry, groups = res
                cands.append(cand)
                entries.append(entry)
                for group, role in groups:
                    usage.add(group, role)
                if len(groups) > 1:
                    usage.pairs.add((groups[0][0], groups[1][0]))
                got[band] += 1
                k += 1

        for strict in (True, False):
            for band, q in quotas.items():
                fill(band, q - got[band], strict)
        for strict in (True, False):
            for band in ("ja", "heb", "ar"):
                fill(band, per_class - len(cands), strict)
        if usage.reuse_roles:
            for e in entries:
                text_group = e["donors"][0]["document_group"]
                photo_group = e["donors"][1]["document_group"] if len(e["donors"]) > 1 else None
                e["group_reused_in_class"] = (text_group in usage.by_role["photos"]
                                              or photo_group in usage.by_role["text"])
        self.used |= usage.groups()
        return cands, entries


def default_quotas(per_class: int) -> Dict[str, int]:
    """Band quotas per class: a quarter Arabic script, the rest Hebrew script with Judaeo-Arabic first.

    :param per_class: Records per class.
    :return: ``{"ar": ..., "ja": ..., "heb": ...}``.
    """
    ar = per_class // 4
    ja = (per_class - ar + 1) // 2 + (per_class - ar) // 10
    return {"ar": ar, "ja": ja, "heb": per_class - ar - ja}


def plant_from_work_dir(work_dir: Path, exclude_groups: Set[str], per_class: int, seed: int,
                        classes: Sequence[str] = PLANTING_ORDER) -> Tuple[Dict[str, List[Dict[str, Any]]],
                                                                            List[Dict[str, Any]]]:
    """Plant every class from a frozen novel_build work dir (read-only).

    :param work_dir: Work dir with ``stages/stage1_images.jsonl``, ``stages/stage2_sides.jsonl``, ``reads/reads.jsonl``.
    :param exclude_groups: Document groups never used (the audited fixtures).
    :param per_class: Records per class.
    :param seed: Base seed.
    :param classes: Classes to plant, in planting order.
    :return: ``(class -> candidates, manifest)``.
    """
    candidates = {c["row_id"]: c for c in iter_jsonl(work_dir / "stages" / "stage1_images.jsonl")}
    sides = list(iter_jsonl(work_dir / "stages" / "stage2_sides.jsonl"))
    with_reads = read_shas(iter_jsonl(work_dir / "reads" / "reads.jsonl"))
    sides_by_row: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sides:
        if s.get("outcome") == "accepted" and s.get("image_sha256"):
            sides_by_row[s["row_id"]].append(s)
    donors = build_donors(candidates, sides, with_reads, exclude_groups)
    planter = Planter(candidates, donors, shingle_stats(donors), sides_by_row, with_reads, seed)
    out: Dict[str, List[Dict[str, Any]]] = {}
    manifest: List[Dict[str, Any]] = []
    for cls in classes:
        cands, entries = planter.plant_class(cls, per_class, default_quotas(per_class))
        out[cls] = cands
        manifest.extend(entries)
    return out, manifest


def write_planted(out_dir: Path, planted: Mapping[str, Sequence[Mapping[str, Any]]],
                  manifest: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Write one candidate file per class plus the manifest, only into a ``pool_planted`` directory.

    :param out_dir: Target directory (its name must be ``pool_planted``).
    :param planted: Class -> candidates.
    :param manifest: Manifest entries.
    :return: File name -> records written.
    :raises ValueError: For any other directory (planted records must never land in a dataset pool).
    """
    if out_dir.name != "pool_planted":
        raise ValueError(f"refusing to write planted records outside a pool_planted directory: {out_dir}")
    counts = {f"{cls}.jsonl": write_jsonl(out_dir / f"{cls}.jsonl", recs) for cls, recs in planted.items()}
    counts[MANIFEST_NAME] = write_jsonl(out_dir / MANIFEST_NAME, manifest)
    return counts


# --- report ---------------------------------------------------------------------------------------------------------

def _cell(text: Any) -> str:
    """A markdown table cell (pipes and newlines escaped).

    :param text: Any value.
    :return: Cell text.
    """
    return str(text if text is not None else "").replace("|", "/").replace("\n", " ")


def render_report(fixture_scores: Sequence[Mapping[str, Any]], planted_scores: Sequence[Mapping[str, Any]],
                  meta: Mapping[str, Any]) -> str:
    """The acceptance report as markdown.

    :param fixture_scores: Scored fixtures (may be empty).
    :param planted_scores: Scored planted records (may be empty).
    :param meta: Input description (paths, counts, warnings).
    :return: Markdown text.
    """
    out = ["# Gate v2 acceptance test", "", "INTERNAL / PRIVATE (GATE_V2_SPEC.md §8).", ""]
    for k, v in meta.items():
        out.append(f"- {k}: {v}")
    if fixture_scores:
        summ = summarize_fixtures(fixture_scores)
        n_scored = sum(1 for s in fixture_scores if s["verdict"] not in ("unscored", "missing"))
        out += ["", "## Fixtures: judged remedy x gate outcome (audited photo)", "",
                "| remedy | n | " + " | ".join(OUTCOMES) + " |", "|---|---:|" + "---:|" * len(OUTCOMES)]
        for rem in REMEDY_CLASSES:
            row = summ["matrix"].get(rem, {})
            out.append(f"| {rem} | {sum(row.values())} | " + " | ".join(str(row.get(o, 0)) for o in OUTCOMES) + " |")
        verdict_cols = ("pass", "harm", "loss", "mismatch", "not_repaired", "missing", "unscored")
        out += ["", "| remedy | " + " | ".join(verdict_cols) + " |", "|---|" + "---:|" * len(verdict_cols)]
        for rem in REMEDY_CLASSES:
            row = summ["verdicts"].get(rem, {})
            out.append(f"| {rem} | " + " | ".join(str(row.get(v, 0)) for v in verdict_cols) + " |")
        n_pass = sum(1 for s in fixture_scores if s["verdict"] == "pass")
        n_unscored = summ["verdicts"]["unsure"].get("unscored", 0)
        out += ["", f"- scored items: {n_scored} (missing {len(summ['missing'])}, unscored {n_unscored}); "
                f"pass {n_pass}; strict pass (kept lines also match the line truth where the fixture has one) "
                f"{len(summ['strict_pass'])}",
                f"- pass by class but off-photo lines kept (partial harm): {summ['pass_kept_off'] or '-'}",
                f"- pass by class but on-photo lines dropped (partial loss): {summ['pass_lost_on'] or '-'}",
                f"- **harm** (§8.1: accept where drop/re_pair, untrimmed accept where trim): {len(summ['harm_spec'])} "
                f"{summ['harm_spec']}",
                f"- **harm from merges** (record merges in a text not on the photo, or a second edition of the same "
                f"text): {len(summ['harm_merged'])} {summ['harm_merged']}",
                f"- harm, every class (incl. trim_and_crop and merges): {len(summ['harm'])} {summ['harm']}",
                f"- **loss** (§8.1: reject/review/elsewhere where keep/trim/crop): {len(summ['loss_spec'])} "
                f"{summ['loss_spec']}; with trim_and_crop: {len(summ['loss'])} {summ['loss']}",
                f"- mismatch kinds: {summ['mismatch_kinds'] or '-'}",
                f"- kept lines vs line truth (accepts on the audited photo): {summ['line_status']}",
                f"- other texts the judges saw were NOT on the photo, accepted there: {summ['sibling_harm'] or '-'}",
                f"- merges with a text whose place the fixture does not know: {summ['merged_unverified'] or '-'}",
                f"- rotation asked by the judges and not applied: {summ['rotation_missed'] or '-'}",
                f"- missing (no record for the side): {summ['missing'] or '-'}",
                "", "## Fixtures: every item", "",
                "| item | stratum | remedy | conf | gate outcome | verdict | kind | remedies | lines (1-based) | "
                "other text on photo | expected |", "|---|---|---|---:|---|---|---|---|---|---|---|"]
        for s in fixture_scores:
            lc = s["line_check"]
            if lc["status"] in ("n/a",):
                lines_txt = f"n/a ({lc.get('why', '')})"
            elif lc["status"] == "index_mismatch":
                lines_txt = f"index mismatch {lc.get('kept')}"
            else:
                lines_txt = (f"{lc['status']}: kept {compress_ranges(lc['kept'])}"
                             + (f"; kept off {compress_ranges(lc['kept_off'])}" if lc["kept_off"] else "")
                             + (f"; lost on {compress_ranges(lc['lost_on'])}" if lc["lost_on"] else ""))
            other = "; ".join(s["photo_other_texts"])[:80]
            if s["photo_got_expected_text"] is not None:
                other = f"expected text {'yes' if s['photo_got_expected_text'] else 'no'}; {other}"
            if s["sibling_accepts"]:
                other += " SIBLING ACCEPTED " + ", ".join(
                    f"{a['side_key']} ({a['kept_lines'] if a['kept_lines'] is not None else '?'} lines)"
                    for a in s["sibling_accepts"])
            rem = s["remedy"] + (f" / B {s['remedy_judgeB']}" if s.get("remedy_judgeB") else "")
            rot = f" rot {s['rotation']['got']}/{s['rotation']['wanted']}" if s["rotation"] else ""
            out.append(f"| {s['item_id']} | {s['stratum']} | {rem} | {s['confidence']} | {s['outcome']}"
                       f"{' (dup)' if s['match'] == 'duplicate' else ''} | {s['verdict']} | {_cell(s['kind'])} | "
                       f"{','.join(s['remedies']) or '-'}{rot} | {_cell(lines_txt)} | {_cell(other)} | "
                       f"{_cell(s['expected_text'])[:160]} |")
    if planted_scores:
        rates = planted_rates(planted_scores)
        cols = ("false_accept", "unverifiable", "review", "reject", "trimmed_ok", "restored", "subset_accept",
                "accept_empty", "missing")
        out += ["", "## Planted negatives: false accepts per class", "",
                "| class | n | scored | false accepts | FA rate | FA rate incl. unverifiable | "
                + " | ".join(cols) + " |",
                "|---|---:|---:|---:|---:|---:|" + "---:|" * len(cols)]
        for cls, r in rates.items():
            out.append(f"| {cls} | {r['n']} | {r['scored']} | {r['false_accept']} | {r['fa_rate']} | "
                       f"{r['fa_rate_upper']} | " + " | ".join(str(r["outcomes"].get(c, 0)) for c in cols) + " |")
        out += ["", "## Planted negatives: accepted records", "",
                "| planted row | class | outcome | accepted photo | kept (1-based) | wrong lines | order inverted |",
                "|---|---|---|---|---|---|---|"]
        for s in planted_scores:
            for a in s["accepts"]:
                out.append(f"| {s['row_id']} | {s['class']} | {s['outcome']} | {str(a['photo'])[:12]} | "
                           f"{compress_ranges(a['kept'])}{' (assumed all)' if a['assumed_all'] else ''} | "
                           f"{compress_ranges(a['wrong_lines'])} | {a['order_inverted']} |")
    return "\n".join(out) + "\n"


# --- CLI ------------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Command-line parser.

    :return: The parser.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=Path, help="gate pairs.jsonl of the real pool (scored against the fixtures)")
    ap.add_argument("--planted-pairs", type=Path, help="gate pairs.jsonl of the planted candidates")
    ap.add_argument("--out", type=Path, help="markdown report")
    ap.add_argument("--json", type=Path, help="optional JSON dump of every scored item")
    ap.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_PLANTED_DIR / MANIFEST_NAME)
    ap.add_argument("--plant", action="store_true", help="build the planted candidates instead of scoring")
    ap.add_argument("--work-dir", type=Path, default=W1, help="frozen novel_build work dir (read-only; --plant)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_PLANTED_DIR, help="pool_planted directory (--plant)")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Score a gate run, or plant the negatives.

    :param argv: Arguments (default: ``sys.argv``).
    :return: Process exit code.
    """
    args = build_parser().parse_args(argv)
    if args.plant:
        exclude = {fx["document_group"] for fx in load_jsonl(args.fixtures)} if args.fixtures.exists() else set()
        planted, manifest = plant_from_work_dir(args.work_dir, exclude, args.per_class, args.seed)
        counts = write_planted(args.out_dir, planted, manifest)
        print(json.dumps({"out_dir": str(args.out_dir), "files": counts,
                          "by_class_band": dict(Counter(f"{m['class']}:{m['band']}" for m in manifest))}, indent=1))
        return 0
    if not args.out or not (args.pairs or args.planted_pairs):
        raise SystemExit("--out and at least one of --pairs / --planted-pairs are required (or --plant)")
    meta: Dict[str, Any] = {}
    fixture_scores: List[Dict[str, Any]] = []
    planted_scores: List[Dict[str, Any]] = []
    if args.pairs:
        records = load_jsonl(args.pairs)
        fixtures = load_jsonl(args.fixtures)
        leaked = sorted({str(r.get("row_id")) for r in records if str(r.get("row_id", "")).startswith("planted_")})
        meta["pairs"] = f"{args.pairs} ({len(records)} records)"
        meta["fixtures"] = f"{args.fixtures} ({len(fixtures)} items)"
        if leaked:
            meta["WARNING planted rows in the real pairs file"] = f"{len(leaked)} (e.g. {leaked[:3]})"
        fixture_scores = score_fixtures(fixtures, records)
    if args.planted_pairs:
        records = load_jsonl(args.planted_pairs)
        manifest = load_jsonl(args.manifest)
        meta["planted pairs"] = f"{args.planted_pairs} ({len(records)} records)"
        meta["manifest"] = f"{args.manifest} ({len(manifest)} planted candidates)"
        planted_scores = score_planted(manifest, records)
    write_text(args.out, render_report(fixture_scores, planted_scores, meta))
    if args.json:
        payload = {"meta": meta, "fixtures": fixture_scores,
                   "fixture_summary": summarize_fixtures(fixture_scores) if fixture_scores else None,
                   "planted": planted_scores,
                   "planted_rates": planted_rates(planted_scores) if planted_scores else None}
        write_text(args.json, json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
