# File name: canon_diff.py
# Date: 10/25/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Canon-comparison families: KTIV human transcriptions aligned to Sefaria canonical texts.

Two families, both built from the HUMAN diplomatic transcriptions of the KTIV viewer (API-shape bundles with word
boxes, reading order rebuilt by :mod:`src.finetuning.qwen_hebrew.ktiv_layout`) and the Sefaria cache of
:mod:`sefaria`. Every target is computed deterministically by alignment; no model writes any part of a row
(no-self-labels rule).

* ``canon_identify`` -- input: one page column of a fragment (≤ :data:`MAX_CHUNK_WORDS` words); target: a JSON
  object naming the canonical work, title, reference range and the matched canonical span.
* ``canon_diff`` -- input: the same chunk plus the aligned canonical passage in a ``Canonical text`` block (as a
  retrieval step would supply it at inference); target: a JSON variant apparatus ``{canonical_ref,
  agreement_ratio, lacuna_words, variants: [{type, fragment, canon, pos}], summary}``.

Pipeline (:func:`build`): catalogue → corpus (the KTIV scholarly frame / title names the work, e.g. Talmud Bavli,
Mishneh Torah, Mishnah); anchor search (rare word trigrams of consonant skeletons, voted by diagonal) over the whole
corpus, preferring the catalogue's tractate when it scores within :data:`PRIOR_MARGIN`; word alignment
(:class:`difflib.SequenceMatcher` on skeleton keys) inside a padded window; variant typing with the skeleton rule
(:func:`skeleton`: final forms folded, internal ו/י dropped, word-final א≡ה, abbreviation and acronym expansion,
word division) -- ``orthographic`` when the consonant skeletons agree, else ``substitution`` / ``omission`` /
``addition`` / ``order``. Canonical words facing a lacuna (``[...]``) or a word with illegible letters (dots) are
counted as lacuna words, never as variants. A chunk becomes rows only when its similarity (share of the fragment's
legible words matched in order) reaches the floor (:data:`DEFAULT_FLOOR`, chosen from the 30-page audit).

Splits: a manuscript in the hold-out set -- KTIV system numbers of the religious-140 and sealed benchmarks and the
benchmark registries, :class:`holdouts.Holdouts` by pgpid / canonical id / shelfmark key (both tiers), or a page whose
25-letter shingles overlap a benchmark ground truth by ≥ :data:`GT_OVERLAP_HOLD` -- is ``heldout``; the rest by
``sha1(sys_num)`` 80/10/10 → ``train`` / ``dev`` / ``test_indist``.

Rows carry ``meta.condition`` (``edition`` now; a ``read`` twin with our VLM reads as input can reuse the same target
and :func:`diff_messages` / :func:`identify_messages` with ``condition="read"``).

The second part (:func:`build_pd`) assesses the archive.org text layers of the public-domain editions registered in
``logs/text_finetune_27b/pd_editions.json`` (local) and, where Sefaria carries the same edition in Hebrew and
English, emits ``translate_en`` pairs verified against the book's English text layer.

CLI::

    python -m src.finetuning.text27b.canon_diff build [--floor 0.5] [--limit-mss N] [--corpus bavli ...]
    python -m src.finetuning.text27b.canon_diff audit [--n 30]
    python -m src.finetuning.text27b.canon_diff pd
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import difflib
import hashlib
import json
import logging
import os
import random
import re
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import orjson

from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as R
from src.finetuning.text27b import sefaria as S
from src.finetuning.text27b import textnorm as TN

LOG = logging.getLogger(__name__)

FAMILY_IDENTIFY = "canon_identify"
FAMILY_DIFF = "canon_diff"
FAMILY_PD = "pd_editions"
VARIANT_TYPES: Tuple[str, ...] = ("omission", "addition", "substitution", "orthographic", "order")
#: Similarity floor (share of the fragment's legible words matched in order); see the audit in the report.
DEFAULT_FLOOR = 0.5
#: A chunk shorter than this many legible words is not aligned (too little to locate).
MIN_LEGIBLE_WORDS = 12
#: Chunks are cut on line boundaries to at most this many words.
MAX_CHUNK_WORDS = 320
#: Rendered-row token cap (build brief).
MAX_ROW_TOKENS = 6000
#: An omission / addition longer than this many words marks a layout jump, an unmarked lost margin or an abbreviated
#: citation the alignment cannot tell apart: such chunks are not turned into rows.
MAX_BLOCK_WORDS = 40
#: Canonical words outside lacunae per legible fragment word above which the span overran the fragment.
MAX_SPAN_RATIO = 2.0
#: Trigrams occurring more often than this in a corpus are not anchors.
MAX_ANCHOR_DF = 40
#: The catalogue's tractate wins when its similarity is within this of the best candidate's.
PRIOR_MARGIN = 0.05
#: Candidate windows aligned per chunk.
TOP_K = 4
#: Canonical words shown before / after the matched span in the canon_diff context block.
CONTEXT_PAD = 12
#: Page share of 25-letter shingles found in a benchmark GT above which a page is held out.
GT_OVERLAP_HOLD = 0.5
GT_SHINGLE = 25
#: Repository-relative default of the KTIV scrape directory (``paths.json`` entry ``ktiv_dir`` overrides it). The
#: directory also holds unrelated files: only ``*_transcription*.json`` is ever globbed.
DEFAULT_KTIV_DIR = "src/datasets/raw_data/cairo_genizah/ktiv"
#: Repository-relative benchmark files carrying KTIV system numbers (``paths.json`` entry ``ktiv_benchmarks``).
DEFAULT_KTIV_BENCHMARKS: Tuple[str, ...] = (
    "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1/genizah_religious_v1.json",
    "src/datasets/raw_data/cairo_genizah/evaluations/sealed_v1/literary/sealed_v1_literary.json",
)
SPLIT_BUCKETS: Tuple[Tuple[int, str], ...] = ((80, "train"), (90, "dev"), (100, "test_indist"))

# ---------------------------------------------------------------------------------------- prompts
NOTE_EDITION = ("The text below is a diplomatic transcription of one manuscript page or column made by a scholar. "
                "[...] marks a lacuna and dots mark illegible letters; abbreviations and spellings are the "
                "scribe's.")
NOTE: Dict[str, str] = {"edition": NOTE_EDITION}
NOTE.update({c: R.READ_NOTE_V2 for c in R.READ_LIKE_CONDITIONS})
INSTRUCTION_IDENTIFY = ("Identify the canonical text this Genizah fragment ({script}) preserves. Return only a JSON "
                        "object with the keys work, title, he_title, ref, he_ref and canonical_text: the work and "
                        "tractate or book, the reference range in the standard printed text (Sefaria-style), and the "
                        "printed text from the first to the last word the fragment preserves.")
INSTRUCTION_DIFF = ("Compare this Genizah fragment ({script}) with the canonical text given after it and list its "
                    "variant readings. Return only a JSON object with the keys canonical_ref, agreement_ratio, "
                    "lacuna_words, variants and summary. Each variant is {{\"type\", \"fragment\", \"canon\", "
                    "\"pos\"}}: type is omission (canonical words missing from the fragment), addition (words only "
                    "in the fragment), substitution, orthographic (same consonants: plene or defective spelling, "
                    "final letters, abbreviations, word division) or order (transposed words); pos is line:word of "
                    "the fragment where the variant starts (for an omission, the word before which text is "
                    "missing). The canonical passage may run a little beyond the fragment: compare only the part "
                    "the fragment preserves, and do not count canonical words that fall in its lacunae.")


# ---------------------------------------------------------------------------------------- tokens
_FINALS = str.maketrans("ךםןףץ", "כמנפצ")
_ABBR = set("'׳\"״")
_GERSHAYIM = set("\"״")
_HEB = re.compile(r"[א-ת]")
_SPLIT = re.compile(r"[\s־׀]+")  # whitespace, maqaf, paseq


def norm_word(token: str) -> str:
    """Letters-only normal form of a Hebrew-script token: NFKD, combining marks dropped, final forms folded.

    :param token: One whitespace token.
    :returns: Hebrew letters only (``""`` when none).
    """
    t = unicodedata.normalize("NFKD", token)
    t = "".join(ch for ch in t if "א" <= ch <= "ת")
    return t.translate(_FINALS)


def skeleton(norm: str) -> str:
    """Consonant skeleton for the orthographic rule: internal ו / י dropped, word-final א read as ה.

    The first and last letters are kept (a prefixed ו "and" or a suffixed י is morphology, not spelling). In words
    of four letters or more an internal א is a vowel letter too (``מיכאן`` ≡ ``מיכן``, ``דאמי`` ≡ ``דמי``); in
    three-letter words it is kept (``שאל`` ≠ ``של``).

    :param norm: Output of :func:`norm_word`.
    :returns: The skeleton.
    """
    if len(norm) <= 2:
        return norm.replace("א", "ה") if norm.endswith("א") and len(norm) == 2 else norm
    vowels = "[ויא]" if len(norm) >= 4 else "[וי]"
    body = norm[0] + re.sub(vowels, "", norm[1:-1]) + norm[-1]
    return body[:-1] + "ה" if body.endswith("א") else body


def strip_marks_he(token: str) -> str:
    """Display form of a canonical token: nikkud and cantillation removed (punctuation kept).

    :param token: Token.
    :returns: Token without combining marks.
    """
    t = unicodedata.normalize("NFKD", token)
    return unicodedata.normalize("NFC", "".join(ch for ch in t if not unicodedata.combining(ch)))


def is_abbrev(raw: str) -> bool:
    """Whether a token carries an abbreviation mark (geresh / gershayim, ASCII or Hebrew).

    :param raw: Raw token.
    :returns: True when marked.
    """
    return any(ch in _ABBR for ch in raw)


@dataclass
class FTok:
    """A fragment token.

    :ivar raw: Token as transcribed.
    :ivar norm: :func:`norm_word` of it.
    :ivar line: 1-based line in the chunk.
    :ivar word: 1-based word in the line (gaps counted).
    :ivar gap: A lacuna token (``[...]``).
    :ivar uncertain: Contains illegible letters (dots).
    :ivar edge: Next to a lacuna (possibly a partial word): it may match, but a difference there is not a variant.
    """

    raw: str
    norm: str
    line: int
    word: int
    gap: bool = False
    uncertain: bool = False
    edge: bool = False

    @property
    def legible(self) -> bool:
        """Neither a lacuna nor a word with illegible letters.

        :returns: True when legible.
        """
        return not self.gap and not self.uncertain

    @property
    def pos(self) -> str:
        """``line:word`` position.

        :returns: Position string.
        """
        return f"{self.line}:{self.word}"


def tokenize_fragment(lines: Sequence[str]) -> List[FTok]:
    """Tokenise chunk lines (KTIV transcription conventions: ``[...]`` lacunae, dots for illegible letters).

    Tokens without any Hebrew letter (``:``, sigla) are dropped, except lacunae.

    :param lines: Chunk lines.
    :returns: Tokens in reading order.
    """
    out: List[FTok] = []
    for li, line in enumerate(lines, 1):
        w = 0
        for raw in _SPLIT.split(line.strip()):
            if not raw:
                continue
            gap = bool(re.fullmatch(r"\[[.…\s]*\]?|[.…]{3,}", raw))
            n = norm_word(raw)
            if not gap and not n:
                continue
            w += 1
            unc = (not gap) and ("." in raw.strip(".:,") or raw.startswith(".") or "?" in raw or "[" in raw)
            out.append(FTok(raw=raw, norm=n, line=li, word=w, gap=gap, uncertain=unc))
    for k, t in enumerate(out):
        if t.legible and ((k > 0 and out[k - 1].gap) or (k + 1 < len(out) and out[k + 1].gap)):
            t.edge = True
    return out


def tokenize_canon(text: str) -> List[str]:
    """Display tokens of a canonical segment (whitespace / maqaf split, marks removed, letterless tokens dropped).

    :param text: Cleaned segment text.
    :returns: Tokens.
    """
    out = []
    for raw in _SPLIT.split(text):
        t = strip_marks_he(raw)
        if t and _HEB.search(t):
            out.append(t)
    return out


# ---------------------------------------------------------------------------------------- abbreviations
def abbrev_expands(abbr: str, words: Sequence[str]) -> bool:
    """Whether an abbreviated token stands for ``words`` (``אמ'``→``אמר``, ``ר'``→``רבי``, ``א"ר``→``אמר רבי``).

    A token with a final geresh abbreviates one word that starts with its letters; a token with gershayim is an
    acronym of the initials of 2-4 words (one leading prefix letter ו / ד / ש / ב / ל / מ / ה / כ may be glued to the
    first initial). A token with a geresh after several letters (``א'ע'פ'``) is an acronym as well.

    :param abbr: Abbreviated raw token.
    :param words: Normalised words of the other side.
    :returns: True when the expansion fits.
    """
    if not is_abbrev(abbr) or not words:
        return False
    stem = norm_word(abbr)
    if not stem:
        return False
    if (any(ch in _GERSHAYIM for ch in abbr) or sum(ch in _ABBR for ch in abbr) >= 2) and len(words) >= 2:
        initials = "".join(w[:1] for w in words)
        if stem == initials:
            return True
        return len(stem) == len(words) + 1 and stem[0] in "ודשבלמהכ" and words[0][:2] == stem[:2] and \
            "".join(w[:1] for w in words[1:]) == stem[2:]
    if len(words) == 1:
        w = words[0]
        return len(w) > len(stem) and w.startswith(stem)
    return False


def ortho_equal(frag_raw: Sequence[str], canon_raw: Sequence[str]) -> bool:
    """Whether two word groups differ only orthographically.

    Same skeleton word by word, same skeleton when joined (word division; not across abbreviation marks), or one
    side is an abbreviation / acronym of the other.

    :param frag_raw: Fragment raw tokens.
    :param canon_raw: Canonical raw tokens.
    :returns: True when the difference is orthographic.
    """
    fn = [norm_word(t) for t in frag_raw]
    cn = [norm_word(t) for t in canon_raw]
    if not fn or not cn:
        return False
    if len(fn) == len(cn) and all(skeleton(a) == skeleton(b) for a, b in zip(fn, cn)):
        return True
    marked = any(is_abbrev(t) for t in frag_raw) or any(is_abbrev(t) for t in canon_raw)
    if not marked and skeleton("".join(fn)) == skeleton("".join(cn)):  # word division (never across abbreviations)
        return True
    if len(frag_raw) == 1 and len(canon_raw) == 1 and is_abbrev(frag_raw[0]) and is_abbrev(canon_raw[0]):
        # two abbreviations: same kind of mark (geresh vs gershayim) and one stem extending the other
        kind = (any(ch in _GERSHAYIM for ch in frag_raw[0]), any(ch in _GERSHAYIM for ch in canon_raw[0]))
        a, b = fn[0], cn[0]
        return kind[0] == kind[1] and (a.startswith(b) or b.startswith(a))
    if len(frag_raw) == 1 and abbrev_expands(frag_raw[0], cn):
        return True
    if len(canon_raw) == 1 and abbrev_expands(canon_raw[0], fn):
        return True
    return False


# ---------------------------------------------------------------------------------------- catalogue
@dataclass
class CatRecord:
    """KTIV catalogue facts of one manuscript (from the merged shelfmark file).

    :ivar sys_num: NLI system number.
    :ivar canonical_id: Merged canonical id.
    :ivar shelfmark: KTIV shelfmark.
    :ivar title: ``basic_catalog.title``.
    :ivar frames: Scholarly frames (``title``, ``frame``, ``domain``).
    :ivar notes: ``full_catalog.notes``.
    :ivar pgpids: PGP ids of the merged record.
    """

    sys_num: str
    canonical_id: Optional[str] = None
    shelfmark: Optional[str] = None
    title: Optional[str] = None
    frames: List[Dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None
    pgpids: List[str] = field(default_factory=list)


def _flat_strings(value: Any) -> List[str]:
    """All strings inside a (possibly nested) list value.

    :param value: String, list or None.
    :returns: Strings.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for v in value for s in _flat_strings(v)]
    return []


def load_catalogue(merged_path: Union[str, Path], sys_nums: Optional[Set[str]] = None) -> Dict[str, CatRecord]:
    """Read the KTIV catalogue blocks of the merged shelfmark file.

    :param merged_path: ``merged_shelfmarks.jsonl``.
    :param sys_nums: Keep only these manuscripts (None = all with a KTIV block).
    :returns: sys_num → record.
    """
    out: Dict[str, CatRecord] = {}
    with open(merged_path, "rb") as fh:
        for line in fh:
            if b'"ktiv"' not in line:
                continue
            r = orjson.loads(line)
            k = (r.get("sources") or {}).get("ktiv")
            if not isinstance(k, dict):
                continue
            sn = str(k.get("sys_num") or "")
            if not sn or (sys_nums is not None and sn not in sys_nums):
                continue
            frames = []
            for e in k.get("scholarly_entries") or []:
                wc = ((e.get("subsections") or {}).get("writing_characteristics") or {}) if isinstance(e, dict) else {}
                if wc:
                    frames.append({kk: wc.get(kk) for kk in ("title", "frame", "domain")})
            bc = k.get("basic_catalog") or {}
            fc = k.get("full_catalog") or {}
            pg = r.get("pgpids") or []
            out[sn] = CatRecord(sys_num=sn, canonical_id=r.get("canonical_id"),
                                shelfmark=k.get("shelf_mark") or r.get("shelfmark_display"), title=bc.get("title"),
                                frames=frames, notes=fc.get("notes") if isinstance(fc.get("notes"), str) else None,
                                pgpids=[str(x) for x in pg] if isinstance(pg, list) else [])
    return out


#: Frame-title / Hebrew-title cues → corpus key (first match over frames, then the catalogue title).
_WORK_CUES: Tuple[Tuple[str, str], ...] = (
    ("talmud bavli", "bavli"), ("mishneh torah", "mishneh_torah"), ("perek kinyan torah", "mishnah"),
    ("mishnah", "mishnah"), ("halakhot gedolot", "halakhot_gedolot"), ("ben sira", "ben_sira"),
    ("תלמוד בבלי", "bavli"), ("משנה תורה", "mishneh_torah"), ("הלכות גדולות", "halakhot_gedolot"),
    ("פרק קנין תורה", "mishnah"), ("משנה", "mishnah"))
#: Frame cues naming Maimonides: his Hebrew halakhic text on KTIV is the Mishneh Torah even when the frame's work
#: label reads "Mishnah" (checked before :data:`_WORK_CUES`).
_RAMBAM_CUES: Tuple[str, ...] = ("moses b. maimon", "rambam", "משה בן מימון", "רמב\"ם")
#: Frame-title cues that name another (non-canonical) work and stop the cue search.
_OTHER_WORKS: Tuple[str, ...] = ("piyyut", "פיוט", "commentary", "פירוש", "rashi")


def work_of(rec: CatRecord) -> Optional[str]:
    """Corpus a manuscript belongs to, from its scholarly frames (first that names a corpus) or its title.

    :param rec: Catalogue record.
    :returns: Corpus key or None.
    """
    cands: List[str] = []
    for f in rec.frames:
        cands += _flat_strings(f.get("title")) + _flat_strings(f.get("frame"))
    cands += _flat_strings(rec.title)
    frames = [s.lower() for f in rec.frames for s in _flat_strings(f.get("frame"))]
    if any(cue in s for s in frames for cue in _RAMBAM_CUES) and not any("perush" in s or "commentary" in s
                                                                          for s in frames):
        return "mishneh_torah"
    for c in cands:
        low = c.lower()
        if any(o in low for o in _OTHER_WORKS) and "talmud bavli [text]" not in low:
            continue
        for cue, key in _WORK_CUES:
            if cue in low:
                return key
    return None


def _latin_key(name: str) -> str:
    """Spelling-insensitive key of a transliterated tractate name (``Mo'ed Qatan`` ≡ ``Moed Katan``).

    :param name: Name.
    :returns: Consonant key.
    """
    s = re.sub(r"[^a-z]", "", name.lower())
    for a, b in (("ch", "h"), ("kh", "k"), ("tz", "z"), ("ts", "z"), ("q", "k"), ("sh", "s")):
        s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s[:1] + re.sub(r"[aeiouy]", "", s[1:])


@dataclass
class CatRef:
    """The catalogue's own reference for a manuscript (a prior, checked against the alignment).

    :ivar corpus: Corpus key.
    :ivar title_hint: Transliterated tractate / book named by the frame (``Betzah``) or None.
    :ivar text: The raw frame / title string used.
    :ivar daf_from: First daf index (Talmud) or None.
    :ivar daf_to: Last daf index or None.
    """

    corpus: Optional[str]
    title_hint: Optional[str] = None
    text: Optional[str] = None
    daf_from: Optional[int] = None
    daf_to: Optional[int] = None


_FRAME_BAVLI = re.compile(r"\[Talmud Bavli\]:\s*([A-Za-z' \-]+?)\s+(\d+)\s*([ab])?(?:\s*[–-]\s*(\d+)?\s*([ab])?)?\s*$")
_FRAME_OTHER = re.compile(r"\[(Mishnah|Tosefta)\]:\s*([A-Za-z' \-]+?)\s*(?:\d.*)?$")


def catalogue_ref(rec: CatRecord) -> CatRef:
    """Parse the catalogue's corpus and (for the Bavli) tractate and daf range.

    :param rec: Catalogue record.
    :returns: The prior.
    """
    corpus = work_of(rec)
    ref = CatRef(corpus=corpus)
    for f in rec.frames:
        for s in _flat_strings(f.get("frame")):
            m = _FRAME_BAVLI.search(s.strip())
            if m:
                ref.title_hint, ref.text = m.group(1).strip(), s.strip()
                a = int(m.group(2)) * 2 - 2 + (1 if m.group(3) == "b" else 0)
                b_daf = int(m.group(4)) if m.group(4) else int(m.group(2))
                b = b_daf * 2 - 2 + (1 if (m.group(5) or m.group(3)) == "b" else 0)
                ref.daf_from, ref.daf_to = a, max(a, b)
                return ref
            m2 = _FRAME_OTHER.search(s.strip())
            if m2 and ref.title_hint is None:
                ref.title_hint, ref.text = m2.group(2).strip(), s.strip()
            if s and ref.text is None and corpus and "]:" in s:
                ref.text = s.strip()
    return ref


_GEMATRIA = {ch: v for v, ch in zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400],
                                     "אבגדהוזחטיכלמנסעפצקרשת")}


def _looks_numeral(token: str) -> bool:
    """Whether a short token is a Hebrew numeral (letter values non-increasing: ``יז``, ``קטו``, ``ב``; ``טו`` / ``טז``).

    :param token: Token (quotes and ranges allowed).
    :returns: True for a numeral-shaped token.
    """
    t = re.sub(r"['\"׳״:\-–]", "", token)
    if not t or len(t) > 4 or any(ch not in _GEMATRIA for ch in t):
        return False
    if t in ("טו", "טז"):
        return True
    vals = [_GEMATRIA[ch] for ch in t]
    return all(a >= b for a, b in zip(vals, vals[1:]))


def hebrew_title_prior(title: Optional[str], texts: Sequence[Any], corpus: str) -> Optional[str]:
    """Sefaria title named by a KTIV Hebrew catalogue title.

    Mishneh Torah: the words after ``הלכות`` (first word equal, second word equal when both have one); Talmud /
    Mishnah: the tractate's Hebrew name right after an opening parenthesis or ``מסכת``. The longest match wins.

    :param title: KTIV ``basic_catalog.title``.
    :param texts: Corpus texts (``title`` / ``he_title``).
    :param corpus: Corpus key.
    :returns: Sefaria title or None.
    """
    if not title:
        return None
    best: Optional[Tuple[int, str]] = None
    if corpus == "mishneh_torah":
        m = re.search(r"הלכות\s+([^,()]+)", title)
        if not m:
            return None
        kw = m.group(1).split()
        while len(kw) > 1 and _looks_numeral(kw[-1]):
            kw.pop()
        for t in texts:
            he = str(getattr(t, "he_title", "") or "")
            m2 = re.search(r"הלכות\s+(.+)$", he)
            if not m2:
                continue
            sw = m2.group(1).split()
            if not sw or not kw or sw[0] != kw[0]:
                continue
            n = 1 + sum(1 for a, b in zip(sw[1:2], kw[1:2]) if a == b)
            if len(sw) > 1 and len(kw) > 1 and sw[1] != kw[1]:
                continue
            if best is None or n > best[0]:
                best = (n, t.title)
        return best[1] if best else None
    for t in texts:
        he = re.sub(r"^משנה\s+", "", str(getattr(t, "he_title", "") or ""))
        if len(he) < 2:
            continue
        if re.search(r"(?:\(|מסכת\s)" + re.escape(he) + r"(?=[\s)]|$)", title):
            if best is None or len(he) > best[0]:
                best = (len(he), t.title)
    return best[1] if best else None


def match_title(hint: Optional[str], titles: Sequence[str]) -> Optional[str]:
    """Sefaria title a transliterated catalogue name refers to.

    :param hint: Catalogue name (``Betzah``).
    :param titles: Sefaria titles of the corpus.
    :returns: The matching title or None.
    """
    if not hint:
        return None
    k = _latin_key(hint)
    for t in titles:
        if k in (_latin_key(t), _latin_key(re.sub(r"^(Mishnah|Tosefta|Mishneh Torah,)\s+", "", t))):
            return t
    return None


# ---------------------------------------------------------------------------------------- KTIV pages
@dataclass
class Chunk:
    """One aligned unit: a page column (or a line range of a long column).

    :ivar sys_num: Manuscript.
    :ivar fl: Page image id.
    :ivar page: 1-based page index in the bundle.
    :ivar column: 1-based column.
    :ivar part: 1-based part of a split column.
    :ivar n_parts: Parts the column was split into.
    :ivar lines: Text lines.
    :ivar mode: Layout mode of the page reconstruction.
    """

    sys_num: str
    fl: str
    page: int
    column: int
    part: int
    n_parts: int
    lines: List[str]
    mode: str = "geometry"

    @property
    def key(self) -> str:
        """Page key used in row ids.

        :returns: ``sys:fl:c<col>[.<part>]``.
        """
        return f"{self.sys_num}:{self.fl}:c{self.column}" + (f".{self.part}" if self.n_parts > 1 else "")

    @property
    def text(self) -> str:
        """Chunk text.

        :returns: Lines joined by newlines.
        """
        return "\n".join(self.lines)


def ktiv_dir(p: _paths.Paths) -> Path:
    """KTIV scrape directory (``paths.json`` ``ktiv_dir`` or :data:`DEFAULT_KTIV_DIR`).

    :param p: Build paths.
    :returns: Directory.
    """
    raw = p.get("ktiv_dir")
    q = Path(os.path.expanduser(str(raw))) if raw else Path(DEFAULT_KTIV_DIR)
    return q if q.is_absolute() else p.repo / q


def load_api_bundles(directory: Path) -> Dict[str, Dict[str, Any]]:
    """API-shape KTIV bundles, one per manuscript (the shared selection rule of ``ktiv_bundles``).

    :param directory: KTIV scrape directory.
    :returns: sys_num → bundle (``bundle_file`` attached).
    """
    from src.finetuning.qwen_hebrew.ktiv_bundles import bundle_shape, bundle_sys_num, select_bundles

    out = {}
    for sys_num, (path, doc) in sorted(select_bundles(directory).items()):
        if bundle_shape(doc) != "api":
            continue
        doc["bundle_file"] = path.name
        out[bundle_sys_num(doc, path) or sys_num] = doc
    return out


def split_lines(lines: Sequence[str], max_words: int = MAX_CHUNK_WORDS) -> List[List[str]]:
    """Cut a column into near-equal line runs of at most ``max_words`` words.

    :param lines: Column lines.
    :param max_words: Word cap per part.
    :returns: Parts (one part when the column fits).
    """
    words = [len(ln.split()) for ln in lines]
    total = sum(words)
    if total <= max_words:
        return [list(lines)]
    n = -(-total // max_words)
    target = total / n
    parts: List[List[str]] = [[]]
    acc = 0
    for ln, w in zip(lines, words):
        if parts[-1] and acc + w > target * len(parts) and len(parts) < n:
            parts.append([])
        parts[-1].append(ln)
        acc += w
    return parts


def page_chunks(bundle: Mapping[str, Any], sys_num: str, max_words: int = MAX_CHUNK_WORDS) -> List[Chunk]:
    """Chunks of every transcribed page of a bundle (columns from the KTIV layout reconstruction).

    :param bundle: API-shape bundle.
    :param sys_num: Manuscript.
    :param max_words: Word cap per chunk.
    :returns: Chunks.
    """
    from src.finetuning.qwen_hebrew.ktiv_layout import reconstruct_page

    out: List[Chunk] = []
    for pi, page in enumerate(bundle.get("pages") or [], 1):
        items = (page.get("annotation_page") or {}).get("items") or []
        if not items:
            continue
        rec = reconstruct_page(items)
        fl = str(page.get("fl") or f"p{pi}")
        for ci, col in enumerate(rec.get("columns") or [], 1):
            lines = [ln["text"] for ln in col.get("lines") or [] if ln.get("text", "").strip()]
            if not lines:
                continue
            parts = split_lines(lines, max_words)
            for k, part in enumerate(parts, 1):
                out.append(Chunk(sys_num, fl, pi, ci, k, len(parts), part, str(rec.get("mode"))))
    return out


# ---------------------------------------------------------------------------------------- canon index
class CanonIndex:
    """Token arrays and a sorted trigram index over every text of one corpus.

    :ivar corpus: Corpus key.
    :ivar texts: The loaded :class:`sefaria.CanonText` objects.
    :ivar raw: Display tokens (global positions).
    :ivar skel_id: Skeleton vocabulary id per token.
    :ivar seg_of: Global segment number per token.
    :ivar segments: Global segment list.
    :ivar seg_text: Text index of each global segment.
    :ivar title_start: Global token start of each text (plus the end sentinel).
    """

    _BITS = 21

    def __init__(self, corpus: str, texts: Sequence[S.CanonText]) -> None:
        """Build the index.

        :param corpus: Corpus key.
        :param texts: Loaded texts.
        :returns: None.
        """
        self.corpus = corpus
        self.texts = list(texts)
        self.raw: List[str] = []
        seg_of: List[int] = []
        skel: List[int] = []
        self.segments: List[S.Segment] = []
        seg_text: List[int] = []
        self.vocab: Dict[str, int] = {}
        starts = []
        for ti, t in enumerate(self.texts):
            starts.append(len(self.raw))
            for seg in t.segments:
                g = len(self.segments)
                self.segments.append(seg)
                seg_text.append(ti)
                for tok in tokenize_canon(seg.text):
                    self.raw.append(tok)
                    seg_of.append(g)
                    sk = skeleton(norm_word(tok))
                    skel.append(self.vocab.setdefault(sk, len(self.vocab)))
        starts.append(len(self.raw))
        if len(self.vocab) >= 1 << self._BITS:
            raise ValueError("skeleton vocabulary too large for the trigram key")
        self.title_start = np.asarray(starts, dtype=np.int64)
        self.seg_of = np.asarray(seg_of, dtype=np.int64)
        self.seg_text = np.asarray(seg_text, dtype=np.int64)
        self.skel_id = np.asarray(skel, dtype=np.int64)
        keys = self._tri_keys(self.skel_id)
        # a trigram must not straddle two texts
        valid = np.ones(len(keys), dtype=bool)
        for s in self.title_start[1:-1]:
            for d in (1, 2):
                if 0 <= s - d < len(valid):
                    valid[s - d] = False
        pos = np.nonzero(valid)[0]
        keys = keys[pos]
        order = np.argsort(keys, kind="stable")
        self.tri_keys = keys[order]
        self.tri_pos = pos[order]

    @classmethod
    def _tri_keys(cls, ids: np.ndarray) -> np.ndarray:
        """Trigram keys of consecutive ids.

        :param ids: Vocabulary ids.
        :returns: ``len(ids) - 2`` int64 keys.
        """
        if len(ids) < 3:
            return np.zeros(0, dtype=np.int64)
        b = cls._BITS
        return (ids[:-2] << (2 * b)) | (ids[1:-1] << b) | ids[2:]

    def key_of(self, a: str, b: str, c: str) -> Optional[int]:
        """Trigram key of three skeletons (None when a word is not in the corpus vocabulary).

        :param a: First skeleton.
        :param b: Second skeleton.
        :param c: Third skeleton.
        :returns: Key or None.
        """
        ids = [self.vocab.get(x) for x in (a, b, c)]
        if any(i is None for i in ids):
            return None
        return (ids[0] << (2 * self._BITS)) | (ids[1] << self._BITS) | ids[2]

    def postings(self, key: int) -> np.ndarray:
        """Global start positions of a trigram.

        :param key: Trigram key.
        :returns: Positions.
        """
        lo = int(np.searchsorted(self.tri_keys, key, side="left"))
        hi = int(np.searchsorted(self.tri_keys, key, side="right"))
        return self.tri_pos[lo:hi]

    def text_of_pos(self, pos: int) -> int:
        """Index of the text containing a global position.

        :param pos: Global token position.
        :returns: Text index.
        """
        pos = min(max(0, int(pos)), int(self.title_start[-1]) - 1)
        return int(np.searchsorted(self.title_start, pos, side="right") - 1)

    def bounds(self, ti: int) -> Tuple[int, int]:
        """Global token range of a text.

        :param ti: Text index.
        :returns: ``(start, end)``.
        """
        return int(self.title_start[ti]), int(self.title_start[ti + 1])

    def norm(self, pos: int) -> str:
        """Normalised word at a global position.

        :param pos: Position.
        :returns: :func:`norm_word` of the token.
        """
        return norm_word(self.raw[pos])

    def ref_range(self, a: int, b: int) -> Tuple[str, str]:
        """English and Hebrew references of the canonical tokens ``[a, b)``.

        :param a: First global position.
        :param b: End position (exclusive).
        :returns: ``(ref, he_ref)``.
        """
        s0 = self.segments[int(self.seg_of[a])]
        s1 = self.segments[int(self.seg_of[max(a, b - 1)])]
        if s0.section == s1.section:
            ref = f"{s0.title} {s0.section}:{s0.seg}" + (f"-{s1.seg}" if s1.seg != s0.seg else "")
        else:
            ref = f"{s0.title} {s0.section}:{s0.seg}-{s1.section}:{s1.seg}"
        he0, he1 = s0.he_section, s1.he_section
        he = f"{s0.he_title} {he0}" + (f"–{he1}" if he1 != he0 else "")
        return ref, he


# ---------------------------------------------------------------------------------------- search
@dataclass
class Candidate:
    """An anchor-search window.

    :ivar text_index: Text (tractate / book) index in the corpus.
    :ivar diag: Canonical position of fragment token 0 (weighted median of the anchors).
    :ivar score: Sum of anchor weights (1/df).
    :ivar anchors: Number of anchors in the window.
    """

    text_index: int
    diag: int
    score: float
    anchors: int


def anchor_candidates(toks: Sequence[FTok], index: CanonIndex, top_k: int = TOP_K,
                      max_df: int = MAX_ANCHOR_DF) -> List[Candidate]:
    """Rare-trigram voting: candidate canonical windows for a fragment chunk.

    :param toks: Fragment tokens.
    :param index: Corpus index.
    :param top_k: Windows returned (non-overlapping, best first).
    :param max_df: Trigrams more frequent than this are ignored.
    :returns: Candidates.
    """
    diags: List[int] = []
    weights: List[float] = []
    for i in range(len(toks) - 2):
        a, b, c = toks[i], toks[i + 1], toks[i + 2]
        if not (a.legible and b.legible and c.legible):
            continue
        key = index.key_of(skeleton(a.norm), skeleton(b.norm), skeleton(c.norm))
        if key is None:
            continue
        post = index.postings(key)
        if len(post) == 0 or len(post) > max_df:
            continue
        w = 1.0 / len(post)
        for g in post.tolist():
            diags.append(g - i)
            weights.append(w)
    if not diags:
        return []
    order = np.argsort(diags, kind="stable")
    d = np.asarray(diags, dtype=np.int64)[order]
    wt = np.asarray(weights, dtype=np.float64)[order]
    width = max(30, len(toks) // 4)
    csum = np.concatenate([[0.0], np.cumsum(wt)])
    hi = np.searchsorted(d, d + width, side="right")
    scores = csum[hi] - csum[np.arange(len(d))]
    out: List[Candidate] = []
    used = np.zeros(len(d), dtype=bool)
    for start in np.argsort(-scores, kind="stable"):
        if len(out) >= top_k:
            break
        if used[start]:
            continue
        lo_i, hi_i = int(start), int(hi[start])
        if used[lo_i:hi_i].any():
            continue
        used[lo_i:hi_i] = True
        seg_w = wt[lo_i:hi_i]
        seg_d = d[lo_i:hi_i]
        cw = np.cumsum(seg_w)
        med = int(seg_d[int(np.searchsorted(cw, cw[-1] / 2.0))])
        ti = index.text_of_pos(max(0, med + len(toks) // 2))
        out.append(Candidate(ti, med, float(scores[start]), hi_i - lo_i))
    return out


# ---------------------------------------------------------------------------------------- alignment
def trim_punct(text: str) -> str:
    """Drop sentence punctuation (``. : , ;`` and sof pasuq) around and between the words of a variant string.

    :param text: Space-joined tokens.
    :returns: Cleaned string.
    """
    words = [w.strip(".:,;\u05c3") for w in text.split()]
    return " ".join(w for w in words if w)


@dataclass
class Variant:
    """One typed difference.

    :ivar type: One of :data:`VARIANT_TYPES`.
    :ivar fragment: Fragment words (raw).
    :ivar canon: Canonical words.
    :ivar pos: ``line:word`` in the fragment.
    :ivar f_range: Fragment token range.
    :ivar c_range: Canonical token range (window-relative).
    """

    type: str
    fragment: str
    canon: str
    pos: str
    f_range: Tuple[int, int] = (0, 0)
    c_range: Tuple[int, int] = (0, 0)

    def as_json(self) -> Dict[str, str]:
        """Target form (sentence punctuation trimmed from the word strings).

        :returns: ``{type, fragment, canon, pos}``.
        """
        return {"type": self.type, "fragment": trim_punct(self.fragment), "canon": trim_punct(self.canon),
                "pos": self.pos}


@dataclass
class Alignment:
    """Result of aligning one chunk to one canonical window.

    :ivar start: Global canonical position of the first matched word.
    :ivar end: Global canonical end (exclusive) of the last matched word.
    :ivar similarity: Legible fragment words matched (equal skeleton) / legible fragment words (an unmatched word
        next to a lacuna -- possibly partial -- is left out of the denominator).
    :ivar agreement_ratio: Canonical words reproduced exactly or orthographically / canonical words outside lacunae.
    :ivar lacuna_words: Canonical words facing lacunae or illegible words.
    :ivar variants: Typed variants.
    :ivar matched: Fragment words matched.
    :ivar legible: Legible fragment words.
    :ivar unaligned: Legible fragment words outside the matched span.
    :ivar canon_words: Canonical words in the span.
    :ivar text_index: Text index.
    :ivar torn_margin: Line-break omissions were re-classed as the lost margin of a torn column.
    :ivar f_start: First fragment token of the matched span.
    :ivar f_end: Fragment end (exclusive) of the matched span.
    """

    start: int
    end: int
    similarity: float
    agreement_ratio: float
    lacuna_words: int
    variants: List[Variant]
    matched: int
    legible: int
    unaligned: int
    canon_words: int
    text_index: int
    torn_margin: bool = False
    f_start: int = 0
    f_end: int = 0

    def counts(self) -> Dict[str, int]:
        """Variant counts by type.

        :returns: type → count (every type present).
        """
        c = collections.Counter(v.type for v in self.variants)
        return {t: int(c.get(t, 0)) for t in VARIANT_TYPES}


def _join(toks: Sequence[str]) -> str:
    """Space-joined tokens.

    :param toks: Tokens.
    :returns: Text.
    """
    return " ".join(toks)


#: Word-group sizes tried when peeling orthographic pairs off a block (acronyms first, then one-to-one).
_PEEL_SHAPES: Tuple[Tuple[int, int], ...] = ((2, 1), (1, 2), (3, 1), (1, 3), (1, 1))


def _peel(fr: Sequence[str], cr: Sequence[str], front: bool) -> Optional[Tuple[int, int]]:
    """Sizes of an orthographic word-group pair at the front (or back) of a block, if any.

    :param fr: Fragment raw tokens of the remaining block.
    :param cr: Canonical tokens of the remaining block.
    :param front: Peel from the front (else from the back).
    :returns: ``(fragment words, canonical words)`` or None.
    """
    for a, b in _PEEL_SHAPES:
        if a > len(fr) or b > len(cr):
            continue
        fa = fr[:a] if front else fr[len(fr) - a:]
        cb = cr[:b] if front else cr[len(cr) - b:]
        if (a, b) != (1, 1):
            joined = not any(is_abbrev(t) for t in list(fa) + list(cb)) and \
                skeleton("".join(norm_word(t) for t in fa)) == skeleton("".join(norm_word(t) for t in cb))
            if not (joined or (is_abbrev(fa[0]) if a == 1 else is_abbrev(cb[0]))):
                continue  # many-to-one only through word division or an abbreviation / acronym
        if ortho_equal(fa, cb):
            return a, b
    return None


def _classify_block(ftoks: Sequence[FTok], craw: Sequence[str], f0: int, c0: int, pos_at: str) -> List[Variant]:
    """Typed variants of one non-equal block with legible fragment words on both sides.

    :param ftoks: Fragment tokens of the block.
    :param craw: Canonical tokens of the block.
    :param f0: Fragment index of the block start.
    :param c0: Canonical (window) index of the block start.
    :param pos_at: Position string for an empty fragment side.
    :returns: Variants.
    """
    fr = [t.raw for t in ftoks]
    if not ftoks:
        return [Variant("omission", "", _join(craw), pos_at, (f0, f0), (c0, c0 + len(craw)))]
    if not craw:
        return [Variant("addition", _join(fr), "", ftoks[0].pos, (f0, f0 + len(fr)), (c0, c0))]
    if ortho_equal(fr, craw):
        return [Variant("orthographic", _join(fr), _join(craw), ftoks[0].pos, (f0, f0 + len(fr)), (c0, c0 + len(craw)))]
    fs = sorted(skeleton(t.norm) for t in ftoks)
    cs = sorted(skeleton(norm_word(t)) for t in craw)
    if len(fr) >= 2 and fs == cs:
        return [Variant("order", _join(fr), _join(craw), ftoks[0].pos, (f0, f0 + len(fr)), (c0, c0 + len(craw)))]
    # peel orthographic / abbreviation pairs off both ends, then pair word by word when the lengths agree
    out_head: List[Variant] = []
    out_tail: List[Variant] = []
    i, j = 0, 0
    fi, cj = len(fr), len(craw)
    while i < fi and j < cj:
        step = _peel(fr[i:fi], craw[j:cj], front=True)
        if step is None:
            break
        a, b = step
        out_head.append(Variant("orthographic", _join(fr[i:i + a]), _join(craw[j:j + b]), ftoks[i].pos,
                                (f0 + i, f0 + i + a), (c0 + j, c0 + j + b)))
        i, j = i + a, j + b
    while fi > i and cj > j:
        step = _peel(fr[i:fi], craw[j:cj], front=False)
        if step is None:
            break
        a, b = step
        out_tail.insert(0, Variant("orthographic", _join(fr[fi - a:fi]), _join(craw[cj - b:cj]), ftoks[fi - a].pos,
                                   (f0 + fi - a, f0 + fi), (c0 + cj - b, c0 + cj)))
        fi, cj = fi - a, cj - b
    mid_f, mid_c = ftoks[i:fi], craw[j:cj]
    mid: List[Variant] = []
    if mid_f and mid_c and len(mid_f) == len(mid_c):
        run: List[int] = []
        for k in range(len(mid_f) + 1):
            same = k < len(mid_f) and ortho_equal([mid_f[k].raw], [mid_c[k]])
            if k < len(mid_f) and not same:
                run.append(k)
                continue
            if run:
                a, b = run[0], run[-1] + 1
                mid.append(Variant("substitution", _join(t.raw for t in mid_f[a:b]), _join(mid_c[a:b]), mid_f[a].pos,
                                   (f0 + i + a, f0 + i + b), (c0 + j + a, c0 + j + b)))
                run = []
            if k < len(mid_f):
                mid.append(Variant("orthographic", mid_f[k].raw, mid_c[k], mid_f[k].pos,
                                   (f0 + i + k, f0 + i + k + 1), (c0 + j + k, c0 + j + k + 1)))
    elif mid_f and mid_c:
        mid.append(Variant("substitution", _join(t.raw for t in mid_f), _join(mid_c), mid_f[0].pos,
                           (f0 + i, f0 + fi), (c0 + j, c0 + cj)))
    elif mid_f:
        mid.append(Variant("addition", _join(t.raw for t in mid_f), "", mid_f[0].pos, (f0 + i, f0 + fi), (c0 + j, c0 + j)))
    elif mid_c:
        at = ftoks[i].pos if i < len(ftoks) else pos_at
        mid.append(Variant("omission", "", _join(mid_c), at, (f0 + i, f0 + i), (c0 + j, c0 + cj)))
    return out_head + mid + out_tail


def merge_order(variants: List[Variant], ftoks: Sequence[FTok], craw: Sequence[str], window: int = 8) -> List[Variant]:
    """Merge two nearby variants that exchange the same words into one ``order`` variant.

    An addition + an omission of the same words, or two substitutions that mirror each other (``X→Y`` then
    ``Y→X``), within ``window`` fragment words are a transposition. The merged variant spans both, with the words
    between them, on each side.

    :param variants: Variants.
    :param ftoks: Fragment tokens (whole chunk).
    :param craw: Canonical window tokens.
    :param window: Maximum distance (fragment words) between the two.
    :returns: Variants with transpositions merged, in fragment order.
    """
    movable = ("addition", "omission", "substitution")

    def skels(v: Variant) -> Tuple[List[str], List[str]]:
        """Skeletons of a variant's fragment and canonical words.

        :param v: Variant.
        :returns: ``(fragment skeletons, canonical skeletons)``.
        """
        return ([skeleton(ftoks[k].norm) for k in range(*v.f_range)],
                [skeleton(norm_word(t)) for t in craw[v.c_range[0]:v.c_range[1]]])

    out = sorted(variants, key=lambda v: (v.f_range[0], v.c_range[0]))
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(out):
            if a.type not in movable:
                continue
            for j in range(i + 1, len(out)):
                b = out[j]
                if b.f_range[0] - a.f_range[1] > window:
                    break
                if b.type not in movable:
                    continue
                pair = {a.type, b.type}
                if not (pair == {"addition", "omission"} or pair == {"substitution"}):
                    continue
                (fa, ca), (fb, cb) = skels(a), skels(b)
                if not fa + fb or sorted(fa + fb) != sorted(ca + cb) or (pair == {"substitution"} and fa == ca):
                    continue
                f_lo, f_hi = min(a.f_range[0], b.f_range[0]), max(a.f_range[1], b.f_range[1])
                c_lo, c_hi = min(a.c_range[0], b.c_range[0]), max(a.c_range[1], b.c_range[1])
                merged = Variant("order", _join(t.raw for t in ftoks[f_lo:f_hi]), _join(craw[c_lo:c_hi]),
                                 ftoks[f_lo].pos if f_lo < len(ftoks) else a.pos, (f_lo, f_hi), (c_lo, c_hi))
                out = sorted([v for k, v in enumerate(out) if k not in (i, j)] + [merged],
                             key=lambda v: (v.f_range[0], v.c_range[0]))
                changed = True
                break
            if changed:
                break
    return out


def trim_edges(eq: List[Tuple[str, int, int, int, int]], min_block: int = 3,
               max_gap: int = 4) -> List[Tuple[str, int, int, int, int]]:
    """Drop short equal blocks at either end that sit apart from the rest (a stray common word).

    :param eq: ``equal`` opcodes in order.
    :param min_block: Blocks shorter than this can be dropped.
    :param max_gap: A block is stray when more than this many words separate it from its neighbour block.
    :returns: The kept blocks.
    """
    eq = list(eq)
    while len(eq) > 1:
        a, b = eq[0], eq[1]
        if a[2] - a[1] < min_block and max(b[1] - a[2], b[3] - a[4]) > max_gap:
            eq.pop(0)
        else:
            break
    while len(eq) > 1:
        a, b = eq[-2], eq[-1]
        if b[2] - b[1] < min_block and max(b[1] - a[2], b[3] - a[4]) > max_gap:
            eq.pop()
        else:
            break
    return eq


def extend_edges(toks: Sequence[FTok], craw: Sequence[str], fa: int, ca: int, fb: int, cb: int,
                 limit: int = 6) -> Tuple[int, int, int, int]:
    """Grow the matched span over orthographic word pairs just outside it (``אמ' ר'`` before ``יוחנן``).

    :param toks: Fragment tokens.
    :param craw: Canonical window tokens.
    :param fa: Span start in the fragment.
    :param ca: Span start in the window.
    :param fb: Span end in the fragment (exclusive).
    :param cb: Span end in the window (exclusive).
    :param limit: Word pairs added at most on each side.
    :returns: The new ``(fa, ca, fb, cb)``.
    """
    for _ in range(limit):
        if fa > 0 and ca > 0 and toks[fa - 1].legible and ortho_equal([toks[fa - 1].raw], [craw[ca - 1]]):
            fa, ca = fa - 1, ca - 1
        else:
            break
    for _ in range(limit):
        if fb < len(toks) and cb < len(craw) and toks[fb].legible and ortho_equal([toks[fb].raw], [craw[cb]]):
            fb, cb = fb + 1, cb + 1
        else:
            break
    return fa, ca, fb, cb


def torn_margin(variants: Sequence[Variant], toks: Sequence[FTok], fa: int, fb: int,
                share: float = 0.5, min_boundaries: int = 3) -> List[Variant]:
    """Variants that sit at the line breaks of a torn column (each line cut by the tear, not by the scribe).

    A variant is "at a break" when it loses canonical words there: an omission before the first word of a line, or
    a substitution with more canonical than fragment words that ends a line or starts one (a partial word at the
    tear). When at least ``share`` of the line breaks inside the matched span carry such a variant, they are the lost
    margin and are returned (the caller counts their canonical words as lacuna words).

    :param variants: Variants of the alignment.
    :param toks: Fragment tokens.
    :param fa: Span start (fragment index).
    :param fb: Span end (exclusive).
    :param share: Share of line breaks that marks a torn column.
    :param min_boundaries: Fewer line breaks than this never count as torn.
    :returns: The variants to re-class (empty when the column is not torn).
    """
    lines = sorted({t.line for t in toks[fa:fb]})
    boundaries = len(lines) - 1
    if boundaries < min_boundaries:
        return []

    def line_end(k: int) -> bool:
        """Whether token ``k`` is the last of its line.

        :param k: Token index.
        :returns: True at a line end.
        """
        return k == len(toks) - 1 or toks[k + 1].line != toks[k].line

    at_break: List[Variant] = []
    broken: Set[int] = set()
    for v in variants:
        i1, i2 = v.f_range
        lost = (v.c_range[1] - v.c_range[0]) - (i2 - i1)
        if v.type == "omission" and fa < i1 < fb and toks[i1].word == 1:
            at_break.append(v)
            broken.add(toks[i1].line)
        elif v.type == "substitution" and lost >= 1 and i2 > i1:
            if line_end(i2 - 1) and i2 < fb:
                at_break.append(v)
                broken.add(toks[i2].line)
            elif toks[i1].word == 1 and i1 > fa:
                at_break.append(v)
                broken.add(toks[i1].line)
    if len(broken) < share * boundaries:
        return []
    return at_break


def align_window(toks: Sequence[FTok], index: CanonIndex, lo: int, hi: int) -> Optional[Alignment]:
    """Align a chunk to canonical tokens ``[lo, hi)`` and type the differences.

    :param toks: Fragment tokens.
    :param index: Corpus index.
    :param lo: Window start (global).
    :param hi: Window end (global, exclusive).
    :returns: The alignment, or None when nothing matches.
    """
    craw = index.raw[lo:hi]
    ckeys = [skeleton(norm_word(t)) for t in craw]
    fkeys = [("\x00gap%d" % i) if t.gap else ("\x00unc" + t.norm if t.uncertain else skeleton(t.norm))
             for i, t in enumerate(toks)]
    sm = difflib.SequenceMatcher(None, fkeys, ckeys, autojunk=False)
    ops = sm.get_opcodes()
    eq = trim_edges([op for op in ops if op[0] == "equal"])
    if not eq:
        return None
    hit = np.zeros(len(toks), dtype=bool)
    for op in eq:
        hit[op[1]:op[2]] = True
    legible = sum(1 for k, t in enumerate(toks) if t.legible and (hit[k] or not t.edge))
    if not legible:
        return None
    fa, ca = eq[0][1], eq[0][3]
    fb, cb = eq[-1][2], eq[-1][4]
    fa, ca, fb, cb = extend_edges(toks, craw, fa, ca, fb, cb)
    matched = sum(op[2] - op[1] for op in eq)
    unaligned = sum(1 for k, t in enumerate(toks) if t.legible and not t.edge and (k < fa or k >= fb))
    variants: List[Variant] = []
    lacuna = 0
    ok = 0
    for tag, i1, i2, j1, j2 in ops:
        if i2 <= fa and j2 <= ca or i1 >= fb and j1 >= cb:
            continue
        i1, i2 = max(i1, fa), min(i2, fb)
        j1, j2 = max(j1, ca), min(j2, cb)
        if tag == "equal":
            ok += j2 - j1
            for k in range(i2 - i1):
                if toks[i1 + k].norm != norm_word(craw[j1 + k]):
                    variants.append(Variant("orthographic", toks[i1 + k].raw, craw[j1 + k], toks[i1 + k].pos,
                                            (i1 + k, i1 + k + 1), (j1 + k, j1 + k + 1)))
            continue
        block = list(toks[i1:i2])
        near = list(toks[max(0, i1 - 1):min(len(toks), i2 + 1)])
        if any(not t.legible or t.edge for t in block) or (tag == "insert" and any(not t.legible for t in near)):
            lacuna += j2 - j1
            continue
        pos_at = toks[i1].pos if i1 < len(toks) else (toks[-1].pos if toks else "1:1")
        vs = _classify_block(block, craw[j1:j2], i1, j1, pos_at)
        ok += sum(v.c_range[1] - v.c_range[0] for v in vs if v.type == "orthographic")
        variants.extend(vs)
    torn = torn_margin(variants, toks, fa, fb)
    if torn:
        lacuna += sum(v.c_range[1] - v.c_range[0] for v in torn)
        variants = [v for v in variants if not any(v is t for t in torn)]
    variants = merge_order(variants, toks, craw)
    ok += sum(v.c_range[1] - v.c_range[0] for v in variants if v.type == "order")
    span = cb - ca
    denom = max(1, span - lacuna)
    return Alignment(start=lo + ca, end=lo + cb, similarity=round(matched / legible, 4),
                     agreement_ratio=round(min(1.0, ok / denom), 3), lacuna_words=int(lacuna), variants=variants,
                     matched=int(matched), legible=int(legible), unaligned=int(unaligned), canon_words=int(span),
                     text_index=index.text_of_pos(lo + ca), torn_margin=bool(torn), f_start=int(fa), f_end=int(fb))


def locate(toks: Sequence[FTok], index: CanonIndex, prior_text: Optional[int] = None,
           top_k: int = TOP_K) -> Tuple[Optional[Alignment], List[Candidate]]:
    """Anchor search + alignment of one chunk; the catalogue's text wins within :data:`PRIOR_MARGIN`.

    :param toks: Fragment tokens.
    :param index: Corpus index.
    :param prior_text: Text index named by the catalogue (or None).
    :param top_k: Candidate windows to align.
    :returns: ``(best alignment or None, candidates)``.
    """
    cands = anchor_candidates(toks, index, top_k)
    n = len(toks)
    pad = max(20, int(0.3 * n))
    best: List[Alignment] = []
    for c in cands:
        t0, t1 = index.bounds(c.text_index)
        lo, hi = max(t0, c.diag - pad), min(t1, c.diag + n + pad)
        if hi - lo < 3:
            continue
        al = align_window(toks, index, lo, hi)
        for _ in range(3):  # grow the window while fragment words stay unmatched at an end and growing helps
            if al is None or al.unaligned == 0:
                break
            grow_lo = al.f_start >= 3 and lo > t0
            grow_hi = n - al.f_end >= 3 and hi < t1
            if not (grow_lo or grow_hi):
                break
            lo2, hi2 = (max(t0, lo - pad) if grow_lo else lo), (min(t1, hi + pad) if grow_hi else hi)
            al2 = align_window(toks, index, lo2, hi2)
            if al2 is None or al2.similarity <= al.similarity:
                break
            al, lo, hi = al2, lo2, hi2
        if al is not None:
            best.append(al)
    if not best:
        return None, cands
    top = max(best, key=lambda a: (a.similarity, a.matched))
    if prior_text is not None:
        pri = [a for a in best if a.text_index == prior_text and a.similarity >= top.similarity - PRIOR_MARGIN]
        if pri:
            top = max(pri, key=lambda a: (a.similarity, a.matched))
    return top, cands


# ---------------------------------------------------------------------------------------- hold-outs
def ktiv_benchmark_sysnums(p: _paths.Paths) -> Tuple[Dict[str, str], List[str]]:
    """KTIV system numbers held out by the benchmarks, and their ground-truth texts.

    Reads the religious-140 and sealed-literary files (``paths.json`` ``ktiv_benchmarks`` or
    :data:`DEFAULT_KTIV_BENCHMARKS`) and the registries' ``sys_nums``
    (:func:`benchmark_registry.registered_benchmark_manuscripts`).

    :param p: Build paths.
    :returns: ``({sys_num: reason}, [ground-truth texts])``.
    """
    from src.datasets.evaluations import benchmark_registry as _reg

    raw = p.get("ktiv_benchmarks") or list(DEFAULT_KTIV_BENCHMARKS)
    out: Dict[str, str] = {}
    gts: List[str] = []
    for x in ([raw] if isinstance(raw, (str, Path)) else raw):
        q = Path(os.path.expanduser(str(x)))
        q = q if q.is_absolute() else p.repo / q
        if not q.exists():
            LOG.warning("KTIV benchmark file missing: %s", q)
            continue
        data = json.loads(q.read_text(encoding="utf-8"))
        name = str(data.get("benchmark") or q.stem)
        for d in data.get("docs") or []:
            sn = str(d.get("sys_num") or "")
            if sn:
                out.setdefault(sn, f"ktiv_benchmark:{name}")
            if isinstance(d.get("gt"), str):
                gts.append(d["gt"])
    for sn in sorted(_reg.registered_benchmark_manuscripts()):
        out.setdefault(str(sn), "ktiv_registry")
    return out, gts


def _letters(text: str) -> str:
    """Hebrew letters of a text, finals folded (shingle alphabet).

    :param text: Text.
    :returns: Letter string.
    """
    return "".join(ch for ch in text if "א" <= ch <= "ת").translate(_FINALS)


def shingle_set(texts: Iterable[str], n: int = GT_SHINGLE) -> Set[str]:
    """Letter n-gram shingles of texts.

    :param texts: Texts.
    :param n: Shingle length.
    :returns: Shingles.
    """
    out: Set[str] = set()
    for t in texts:
        s = _letters(t)
        out.update(s[i:i + n] for i in range(0, max(0, len(s) - n + 1)))
    return out


def shingle_share(text: str, shingles: Set[str], n: int = GT_SHINGLE) -> float:
    """Share of a text's shingles present in a shingle set.

    :param text: Text.
    :param shingles: Reference shingles.
    :param n: Shingle length.
    :returns: Share in [0, 1] (0 for texts shorter than ``n``).
    """
    s = _letters(text)
    mine = [s[i:i + n] for i in range(0, max(0, len(s) - n + 1))]
    return sum(1 for x in mine if x in shingles) / len(mine) if mine else 0.0


def split_of(sys_num: str) -> str:
    """Deterministic 80/10/10 split of a manuscript.

    :param sys_num: System number.
    :returns: ``train`` / ``dev`` / ``test_indist``.
    """
    b = int(hashlib.sha1(str(sys_num).encode("utf-8")).hexdigest(), 16) % 100
    return next(name for cut, name in SPLIT_BUCKETS if b < cut)


def heldout_reason(rec: CatRecord, ktiv_held: Mapping[str, str], holdouts: Any) -> Optional[str]:
    """Why a manuscript is held out (KTIV benchmark sys_num, or H by pgpid / canonical id / shelfmark key).

    :param rec: Catalogue record (only ``sys_num`` needed when the others are absent).
    :param ktiv_held: sys_num → reason from :func:`ktiv_benchmark_sysnums`.
    :param holdouts: A :class:`holdouts.Holdouts` (or None).
    :returns: Reason or None.
    """
    if rec.sys_num in ktiv_held:
        return ktiv_held[rec.sys_num]
    if holdouts is None:
        return None
    return holdouts.match_any(pgpids=rec.pgpids, canonicals=[c for c in [rec.canonical_id] if c],
                              shelfmarks=[s for s in [rec.shelfmark] if s], include_eval=True)


# ---------------------------------------------------------------------------------------- rows
def _script() -> str:
    """Script name used in the prompts (KTIV literary fragments are Hebrew script).

    :returns: ``Hebrew script``.
    """
    return R.SCRIPT_NAME["hebrew"]


def identify_messages(fragment: str, target: Mapping[str, Any], condition: str = "edition") -> List[R.Message]:
    """Messages of a ``canon_identify`` row.

    :param fragment: Chunk text (transcription or read).
    :param target: Target object (work, title, he_title, ref, he_ref, canonical_text).
    :param condition: Input condition (selects the note).
    :returns: Messages.
    """
    user = (f"{INSTRUCTION_IDENTIFY.format(script=_script())}\n\n{NOTE[condition]}\n\n"
            f"<document>\n{fragment}\n</document>")
    return R.make_messages(user, json.dumps(target, ensure_ascii=False))


def diff_messages(fragment: str, canon_ref: str, canon_text: str, target: Mapping[str, Any],
                  condition: str = "edition") -> List[R.Message]:
    """Messages of a ``canon_diff`` row.

    :param fragment: Chunk text.
    :param canon_ref: Reference of the context passage.
    :param canon_text: Context passage (the matched span plus :data:`CONTEXT_PAD` words each side).
    :param target: Target object.
    :param condition: Input condition.
    :returns: Messages.
    """
    user = (f"{INSTRUCTION_DIFF.format(script=_script())}\n\n{NOTE[condition]}\n\n"
            f"<document>\n{fragment}\n</document>\n\nCanonical text ({canon_ref}):\n{canon_text}")
    return R.make_messages(user, json.dumps(target, ensure_ascii=False))


def summary_sentence(ref: str, al: Alignment) -> str:
    """Templated one-sentence summary of an apparatus.

    :param ref: Canonical reference.
    :param al: Alignment.
    :returns: Sentence.
    """
    c = al.counts()
    names = (("orthographic", "orthographic", "orthographic"), ("substitution", "substitution", "substitutions"),
             ("omission", "omission", "omissions"), ("addition", "addition", "additions"),
             ("order", "transposition", "transpositions"))
    parts = [f"{c[k]} {one if c[k] == 1 else many}" for k, one, many in names if c[k]]
    n = sum(c.values())
    if not parts:
        body = "with no variants"
    else:
        body = f"with {n} variant{'s' if n != 1 else ''} (" + (", ".join(parts[:-1]) + " and " + parts[-1]
                                                              if len(parts) > 1 else parts[0]) + ")"
    lac = (f"; {al.lacuna_words} canonical word{'s fall' if al.lacuna_words != 1 else ' falls'} in lacunae"
           if al.lacuna_words else "")
    return f"Against {ref} the fragment reproduces {round(100 * al.agreement_ratio)}% of the canonical words " \
           f"outside its lacunae, {body}{lac}."


def diff_target(ref: str, al: Alignment) -> Dict[str, Any]:
    """The ``canon_diff`` target object.

    :param ref: Canonical reference of the matched span.
    :param al: Alignment.
    :returns: ``{canonical_ref, agreement_ratio, lacuna_words, variants, summary}``.
    """
    return {"canonical_ref": ref, "agreement_ratio": al.agreement_ratio, "lacuna_words": al.lacuna_words,
            "variants": [v.as_json() for v in al.variants], "summary": summary_sentence(ref, al)}


def _base_meta(family: str, chunk: Chunk, split: str, condition: str, **fields: Any) -> Dict[str, Any]:
    """Meta block with the v0_1 fields filled for a canon row.

    :param family: Family.
    :param chunk: Chunk.
    :param split: Split.
    :param condition: Condition.
    :param fields: Further fields.
    :returns: Meta.
    """
    meta = R.new_meta(family, condition, split, chunk.key)
    meta.update({"lang_class": "HE", "pgpid": None, "shelfmark": None, "shelfmark_group": None,
                 "held_out_vision_benchmark": False, "input_cer": None, "input_source": "ktiv_transcription",
                 "input_truncated": False, "source": "ktiv+sefaria", "target_origin": "deterministic_alignment",
                 "self_label": False, "target_raw": None, "strip_ops": [], "reader_exposed": None,
                 "exposure_level": None})
    meta.update(fields)
    return meta


def _finish(row: Dict[str, Any], tokenizer: Optional[Path], template: Optional[Path]) -> Dict[str, Any]:
    """Fill the token fields of a row.

    :param row: Row.
    :param tokenizer: ``tokenizer.json`` (None = configured).
    :param template: Chat template (None = configured).
    :returns: The row.
    """
    tgt = row["messages"][-1]["content"]
    row["meta"]["target_chars"] = len(tgt)
    row["meta"]["target_tokens"] = R.count_tokens(tgt, tokenizer)
    row["meta"]["tokens"] = R.tokens_of_row(row, tokenizer, template)
    return row


# ---------------------------------------------------------------------------------------- build
@dataclass
class BuildConfig:
    """Build options.

    :ivar floor: Similarity floor.
    :ivar corpora: Corpora to build (None = every corpus a catalogue names).
    :ivar limit_mss: Manuscripts per corpus (0 = all; development).
    :ivar max_words: Chunk word cap.
    :ivar seed: Seed for the audit sample.
    :ivar max_block: Longest omission / addition (words) a row may carry.
    :ivar max_span_ratio: Largest (canonical span − lacuna words) / legible fragment words.
    """

    floor: float = DEFAULT_FLOOR
    corpora: Optional[Tuple[str, ...]] = None
    limit_mss: int = 0
    max_words: int = MAX_CHUNK_WORDS
    seed: int = 3407
    max_block: int = MAX_BLOCK_WORDS
    max_span_ratio: float = MAX_SPAN_RATIO


def _hist(values: Sequence[float], edges: Sequence[float]) -> Dict[str, int]:
    """Histogram with labelled bins.

    :param values: Values.
    :param edges: Bin edges (ascending; the last bin is closed).
    :returns: ``"a-b"`` → count.
    """
    out = {}
    for a, b in zip(edges[:-1], edges[1:]):
        last = b == edges[-1]
        out[f"{a:.2f}-{b:.2f}"] = sum(1 for v in values if a <= v < b or (last and v == b))
    return out


def align_corpus(chunks: Sequence[Tuple[Chunk, CatRecord, CatRef]], index: CanonIndex) -> List[Dict[str, Any]]:
    """Align every chunk of one corpus.

    :param chunks: ``(chunk, catalogue record, catalogue prior)`` triples.
    :param index: Corpus index.
    :returns: One record per chunk: chunk, record, prior, tokens, alignment (or None), candidates.
    """
    titles = [t.title for t in index.texts]
    out = []
    for n, (ch, rec, prior) in enumerate(chunks):
        toks = tokenize_fragment(ch.lines)
        legible = sum(1 for t in toks if t.legible)
        prior_title = match_title(prior.title_hint, titles) or hebrew_title_prior(rec.title, index.texts, index.corpus)
        prior_ti = titles.index(prior_title) if prior_title else None
        al, cands = (locate(toks, index, prior_ti) if legible >= MIN_LEGIBLE_WORDS else (None, []))
        out.append({"chunk": ch, "rec": rec, "prior": prior, "prior_title": prior_title, "toks": toks,
                    "legible": legible, "al": al, "n_candidates": len(cands),
                    "section_agrees": section_agrees(prior, al, index) if al is not None else None})
        if n and n % 500 == 0:
            LOG.info("%s: aligned %d / %d chunks", index.corpus, n, len(chunks))
    return out


def longest_block(al: Alignment) -> int:
    """Words of the longest omission (canonical side) or addition (fragment side) of an alignment.

    :param al: Alignment.
    :returns: Word count (0 without such variants).
    """
    sizes = [v.c_range[1] - v.c_range[0] for v in al.variants if v.type == "omission"]
    sizes += [v.f_range[1] - v.f_range[0] for v in al.variants if v.type == "addition"]
    return max(sizes, default=0)


def shape_gate(al: Alignment, max_block: int = MAX_BLOCK_WORDS, max_span_ratio: float = MAX_SPAN_RATIO) -> Optional[str]:
    """Reject reason for an alignment whose shape betrays a layout jump (or None when it passes).

    :param al: Alignment.
    :param max_block: Longest omission / addition allowed.
    :param max_span_ratio: Largest (span − lacuna words) / legible fragment words.
    :returns: ``long_block`` / ``span_overrun`` or None.
    """
    if longest_block(al) > max_block:
        return "long_block"
    if (al.canon_words - al.lacuna_words) / max(1, al.legible) > max_span_ratio:
        return "span_overrun"
    return None


def section_agrees(prior: CatRef, al: Alignment, index: CanonIndex, slack: int = 2) -> Optional[bool]:
    """Whether the aligned daf range overlaps the catalogue's (Talmud frames only; ± ``slack`` pages).

    :param prior: Catalogue prior.
    :param al: Alignment.
    :param index: Corpus index.
    :param slack: Tolerance in daf sides.
    :returns: True / False, or None when the catalogue gives no daf range or the text is not a Talmud tractate.
    """
    if prior.daf_from is None or prior.daf_to is None:
        return None
    try:
        a = S.daf_index(index.segments[int(index.seg_of[al.start])].section)
        b = S.daf_index(index.segments[int(index.seg_of[max(al.start, al.end - 1)])].section)
    except ValueError:
        return None
    return a <= prior.daf_to + slack and b >= prior.daf_from - slack


def _context(index: CanonIndex, al: Alignment, pad: int = CONTEXT_PAD) -> Tuple[str, str]:
    """Context passage for a canon_diff row: the matched span plus ``pad`` words each side, inside its text.

    :param index: Corpus index.
    :param al: Alignment.
    :param pad: Words each side.
    :returns: ``(ref, text)``.
    """
    t0, t1 = index.bounds(al.text_index)
    lo, hi = max(t0, al.start - pad), min(t1, al.end + pad)
    return index.ref_range(lo, hi)[0], " ".join(index.raw[lo:hi])


def make_rows(res: Mapping[str, Any], index: CanonIndex, split: str, held: Optional[str], licence: str,
              lic_class: str, version: str, tokenizer: Optional[Path] = None, template: Optional[Path] = None,
              condition: str = "edition") -> List[Dict[str, Any]]:
    """The two rows of one accepted chunk.

    :param res: Alignment record from :func:`align_corpus`.
    :param index: Corpus index.
    :param split: Split.
    :param held: Hold-out reason (or None).
    :param licence: Licence string of the canonical version.
    :param lic_class: Its licence class.
    :param version: Sefaria version title.
    :param tokenizer: ``tokenizer.json``.
    :param template: Chat template.
    :param condition: Input condition.
    :returns: ``[identify_row, diff_row]`` (rows over :data:`MAX_ROW_TOKENS` are dropped).
    """
    ch: Chunk = res["chunk"]
    rec: CatRecord = res["rec"]
    al: Alignment = res["al"]
    text = index.texts[al.text_index]
    ref, he_ref = index.ref_range(al.start, al.end)
    spec = S.CORPORA.get(index.corpus)
    ident = {"work": spec.work if spec else index.corpus, "title": text.title, "he_title": text.he_title,
             "ref": ref, "he_ref": he_ref, "canonical_text": " ".join(index.raw[al.start:al.end])}
    ctx_ref, ctx_text = _context(index, al)
    dtarget = diff_target(ref, al)
    prior: CatRef = res["prior"]
    common = dict(shelfmark=rec.shelfmark, pgpid=(rec.pgpids[0] if rec.pgpids else None),
                  held_out_vision_benchmark=bool(held and held.startswith("ktiv_benchmark")),
                  licence=f"KTIV (NLI) transcription, terms: credit; canonical text: Sefaria '{version}' ({licence})",
                  licence_class=lic_class, work=spec.work if spec else index.corpus, page_range=ref,
                  sys_num=ch.sys_num, fl=ch.fl, column=ch.column, part=ch.part, canonical_ref=ref,
                  catalogue_ref=prior.text, catalogue_title=res.get("prior_title"),
                  catalogue_agrees=(None if not res.get("prior_title") else res["prior_title"] == text.title),
                  catalogue_section_agrees=res.get("section_agrees"),
                  similarity=al.similarity, agreement_ratio=al.agreement_ratio, variant_counts=al.counts(),
                  lacuna_words=al.lacuna_words, torn_margin=al.torn_margin, heldout_reason=held,
                  canonical_version=version,
                  provenance={"ktiv_sys_num": ch.sys_num, "ktiv_fl": ch.fl, "column": ch.column, "part": ch.part,
                              "layout_mode": ch.mode, "sefaria_title": text.title, "sefaria_version": version,
                              "canonical_span": [al.start - index.bounds(al.text_index)[0],
                                                 al.end - index.bounds(al.text_index)[0]],
                              "legible_words": al.legible, "matched_words": al.matched,
                              "unaligned_words": al.unaligned})
    rows = []
    for fam, msgs in ((FAMILY_IDENTIFY, identify_messages(ch.text, ident, condition)),
                      (FAMILY_DIFF, diff_messages(ch.text, ctx_ref, ctx_text, dtarget, condition))):
        row = {"messages": msgs, "meta": _base_meta(fam, ch, split, condition, **common)}
        _finish(row, tokenizer, template)
        if row["meta"]["tokens"] <= MAX_ROW_TOKENS:
            rows.append(row)
    return rows


def _counts(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Rows and tokens per family × work × split.

    :param rows: Rows.
    :returns: Nested counts.
    """
    out: Dict[str, Any] = {}
    for r in rows:
        m = r["meta"]
        d = out.setdefault(m["family"], {}).setdefault(str(m.get("work")), {}).setdefault(m["split"],
                                                                                          {"rows": 0, "tokens": 0})
        d["rows"] += 1
        d["tokens"] += int(m.get("tokens") or 0)
    return out


def _atomic_json(path: Path, obj: Any) -> None:
    """Write JSON atomically.

    :param path: Destination.
    :param obj: Object.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def build(cfg: BuildConfig, p: Optional[_paths.Paths] = None) -> Dict[str, Any]:
    """Build ``families/canon_identify.jsonl``, ``families/canon_diff.jsonl`` and ``canon_manifest.json``.

    :param cfg: Options.
    :param p: Build paths.
    :returns: The manifest.
    """
    from src.finetuning.text27b.holdouts import Holdouts

    p = p or _paths.load_paths()
    root = S.sefaria_dir(p)
    sman = S.load_manifest(root)
    bundles = load_api_bundles(ktiv_dir(p))
    cat = load_catalogue(p.merged_shelfmarks, set(bundles))
    ktiv_held, gts = ktiv_benchmark_sysnums(p)
    gt_sh = shingle_set(gts)
    try:
        H = Holdouts.load(p=p)
    except FileNotFoundError:
        LOG.warning("holdouts.json missing: only the KTIV benchmark sys_nums hold manuscripts out")
        H = None
    by_corpus: Dict[str, List[Tuple[Chunk, CatRecord, CatRef]]] = collections.defaultdict(list)
    ms_stats = collections.Counter()
    held_ms: Dict[str, str] = {}
    for sn, bundle in bundles.items():
        rec = cat.get(sn) or CatRecord(sys_num=sn)
        prior = catalogue_ref(rec)
        reason = heldout_reason(rec, ktiv_held, H)
        if reason:
            held_ms[sn] = reason
        if prior.corpus is None:
            ms_stats["no_canonical_work"] += 1
            continue
        if cfg.corpora and prior.corpus not in cfg.corpora:
            continue
        if prior.corpus not in {t.get("corpus") for t in sman.get("texts") or []}:
            ms_stats[f"corpus_not_cached:{prior.corpus}"] += 1
            continue
        if cfg.limit_mss and sum(1 for _ in {c[0].sys_num for c in by_corpus[prior.corpus]}) >= cfg.limit_mss:
            continue
        ms_stats[f"manuscripts:{prior.corpus}"] += 1
        for ch in page_chunks(bundle, sn, cfg.max_words):
            by_corpus[prior.corpus].append((ch, rec, prior))
    tok_path = p.tokenizer_json
    tpl_path = p.chat_template
    rows_identify: List[Dict[str, Any]] = []
    rows_diff: List[Dict[str, Any]] = []
    sims: Dict[str, List[float]] = collections.defaultdict(list)
    audit_pool: List[Dict[str, Any]] = []
    reject = collections.Counter()
    variant_tot = collections.Counter()
    catalogue_agree = collections.Counter()
    held_counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    licences: Dict[str, Dict[str, int]] = {}
    for corpus, chunks in sorted(by_corpus.items()):
        texts = S.load_corpus(corpus, root, sman)
        LOG.info("%s: %d texts, %d chunks; indexing", corpus, len(texts), len(chunks))
        index = CanonIndex(corpus, texts)
        LOG.info("%s: %d canonical tokens, vocab %d", corpus, len(index.raw), len(index.vocab))
        for res in align_corpus(chunks, index):
            ch, al = res["chunk"], res["al"]
            sims[corpus].append(al.similarity if al else 0.0)
            text = index.texts[al.text_index] if al else None
            rec = _audit_record(res, index)
            audit_pool.append(rec)
            if res["legible"] < MIN_LEGIBLE_WORDS:
                drop = "too_short"
            elif al is None:
                drop = "no_anchor"
            elif al.similarity < cfg.floor:
                drop = "below_floor"
            else:
                drop = shape_gate(al, cfg.max_block, cfg.max_span_ratio)
            if drop:
                reject[drop] += 1
                rec["decision"] = drop
                continue
            reason = held_ms.get(ch.sys_num)
            if reason is None and shingle_share(ch.text, gt_sh) >= GT_OVERLAP_HOLD:
                reason = "gt_overlap"
            split = "heldout" if reason else split_of(ch.sys_num)
            if reason:
                held_counts[corpus][reason] += 1
            pt = res.get("prior_title")
            catalogue_agree["no_prior" if not pt else ("agrees" if pt == text.title else "differs")] += 1
            if res.get("section_agrees") is not None:
                catalogue_agree["daf_" + ("agrees" if res["section_agrees"] else "differs")] += 1
            lic = text.licence
            licences.setdefault(corpus, {}).setdefault(lic, 0)
            licences[corpus][lic] += 1
            made = make_rows(res, index, split, reason, lic, S.licence_class(lic), text.version, tok_path, tpl_path)
            rec["decision"] = "accepted" if len(made) == 2 else "over_token_cap"
            rec["split"] = split
            if len(made) < 2:
                reject["over_token_cap"] += 1
            for r in made:
                (rows_identify if r["meta"]["family"] == FAMILY_IDENTIFY else rows_diff).append(r)
            variant_tot.update(al.counts())
        del index
    fam_dir = p.out("families")
    fam_dir.mkdir(parents=True, exist_ok=True)
    n1 = R.write_jsonl(fam_dir / f"{FAMILY_IDENTIFY}.jsonl", rows_identify)
    n2 = R.write_jsonl(fam_dir / f"{FAMILY_DIFF}.jsonl", rows_diff)
    audit_path = p.out("audit", "canon_audit_pool.jsonl")
    R.write_jsonl(audit_path, audit_pool)
    all_rows = rows_identify + rows_diff
    held_ms_in_rows = {r["meta"]["sys_num"] for r in rows_diff if r["meta"]["split"] == "heldout"}
    bench = {k: v for k, v in ktiv_held.items() if k in bundles}
    manifest = {
        "build": R.BUILD, "families": {FAMILY_IDENTIFY: n1, FAMILY_DIFF: n2},
        "counts": _counts(all_rows),
        "self_labels": "none: every target is computed deterministically from the KTIV human transcription and "
                       "the Sefaria text by word alignment; no model output is used as input or target",
        "condition": {"built": ["edition"], "planned": ["read (same targets, VLM read as input)"]},
        "manuscripts": dict(ms_stats), "chunks_by_corpus": {k: len(v) for k, v in by_corpus.items()},
        "rejected_chunks": dict(reject),
        "similarity": {k: {"n": len(v), "hist": _hist(v, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]),
                           "median": float(np.median(v)) if v else None} for k, v in sims.items()},
        "floor": cfg.floor, "variant_counts": dict(variant_tot), "catalogue_agreement": dict(catalogue_agree),
        "holdouts": {"ktiv_benchmark_sysnums_total": len(ktiv_held),
                     "ktiv_benchmark_sysnums_with_api_bundle": len(bench),
                     "ktiv_benchmark_by_reason": dict(collections.Counter(bench.values())),
                     "manuscripts_held_any_reason": len([s for s in held_ms if s in bundles]),
                     "manuscripts_held_by_reason": dict(collections.Counter(held_ms[s] for s in held_ms
                                                                            if s in bundles)),
                     "heldout_manuscripts_with_rows": len(held_ms_in_rows),
                     "benchmark_sysnums_in_rows": len({s for s in held_ms_in_rows if s in ktiv_held}),
                     "benchmark_sysnums_in_non_heldout_rows": sorted({r["meta"]["sys_num"] for r in all_rows
                                                                       if r["meta"]["sys_num"] in ktiv_held
                                                                       and r["meta"]["split"] != "heldout"}),
                     "heldout_chunks_by_corpus_reason": {k: dict(v) for k, v in held_counts.items()},
                     "holdouts_json": H.meta.get("built_at") if H is not None else None},
        "licences": licences,
        "torn_margin_rows": sum(1 for r in rows_diff if r["meta"].get("torn_margin")),
        "tokens_total": {FAMILY_IDENTIFY: sum(r["meta"]["tokens"] for r in rows_identify),
                         FAMILY_DIFF: sum(r["meta"]["tokens"] for r in rows_diff)},
        "audit": audit_summary(p.out("audit", AUDIT_FILE), cfg.floor),
        "sefaria": {k: sman.get(k) for k in ("repo_commit", "books_json_generated_at", "books_json_sha256",
                                              "bucket_last_export", "version_preferences")},
        "build_args": dataclasses.asdict(cfg),
    }
    _atomic_json(p.out("canon_manifest.json"), manifest)
    return manifest


# ---------------------------------------------------------------------------------------- audit
#: The hand-checked audit sample (under ``out_dir/audit``).
AUDIT_FILE = "canon_audit_30.jsonl"


def audit_summary(path: Path, floor: float) -> Optional[Dict[str, Any]]:
    """Hand-check verdicts of the audit sample by similarity band (the evidence for the floor).

    :param path: Audit jsonl (records with a ``hand_check`` block).
    :param floor: Floor in use.
    :returns: Summary, or None when the file is absent.
    """
    if not path.exists():
        return None
    recs = R.load_jsonl(path)
    bands = (("<0.35", 0.0, 0.35), ("0.35-floor", 0.35, floor), (">=floor", floor, 1.01))
    out: Dict[str, Any] = {"file": f"audit/{path.name}", "n": len(recs), "floor": floor, "bands": {}}
    for name, lo, hi in bands:
        xs = [r for r in recs if lo <= float(r.get("similarity") or 0) < hi and r.get("hand_check")]
        out["bands"][name] = {
            "n": len(xs),
            "identification": dict(collections.Counter(r["hand_check"]["identification"] for r in xs)),
            "apparatus": dict(collections.Counter(r["hand_check"]["apparatus"] for r in xs))}
    fin = [r for r in recs if (r.get("final_build") or {}).get("accepted") and r.get("hand_check")]
    out["accepted_in_final_build"] = {"n": len(fin), "identification": dict(collections.Counter(
        r["hand_check"].get("final_build_identification", r["hand_check"]["identification"]) for r in fin))}
    out["checked_on"] = sorted({str(r["hand_check"].get("checked_on")) for r in recs if r.get("hand_check")})
    return out


def _audit_record(res: Mapping[str, Any], index: CanonIndex) -> Dict[str, Any]:
    """Compact record of one chunk's alignment for the hand audit.

    :param res: Alignment record.
    :param index: Corpus index.
    :returns: Record.
    """
    ch: Chunk = res["chunk"]
    al: Optional[Alignment] = res["al"]
    out = {"key": ch.key, "corpus": index.corpus, "sys_num": ch.sys_num, "catalogue_ref": res["prior"].text,
           "catalogue_title": res.get("prior_title"), "catalogue_label": res["rec"].title,
           "section_agrees": res.get("section_agrees"), "legible": res["legible"], "fragment": ch.text}
    if al is not None:
        ref, _ = index.ref_range(al.start, al.end)
        out.update({"ref": ref, "similarity": al.similarity, "agreement_ratio": al.agreement_ratio,
                    "lacuna_words": al.lacuna_words, "variant_counts": al.counts(),
                    "canonical_span": " ".join(index.raw[al.start:al.end]),
                    "variants": [v.as_json() for v in al.variants]})
    return out


def audit_sample(pool: Sequence[Mapping[str, Any]], n: int = 30, seed: int = 3407) -> List[Mapping[str, Any]]:
    """Stratified audit sample: across corpora and similarity bands (low, borderline, high).

    :param pool: Audit records.
    :param n: Sample size.
    :param seed: Seed.
    :returns: Sampled records (sorted by corpus, similarity).
    """
    rng = random.Random(seed)
    bands = ((0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 1.01))
    strata: Dict[Tuple[str, int], List[Mapping[str, Any]]] = collections.defaultdict(list)
    for r in pool:
        if "similarity" not in r:
            continue
        b = next(i for i, (a, c) in enumerate(bands) if a <= r["similarity"] < c)
        strata[(r["corpus"], b)].append(r)
    keys = sorted(strata)
    out: List[Mapping[str, Any]] = []
    i = 0
    while len(out) < n and any(strata[k] for k in keys):
        k = keys[i % len(keys)]
        if strata[k]:
            out.append(strata[k].pop(rng.randrange(len(strata[k]))))
        i += 1
    return sorted(out, key=lambda r: (r["corpus"], r["similarity"]))


# ---------------------------------------------------------------------------------------- PD editions
def djvu_pages(xml_path: Union[str, Path]) -> List[str]:
    """Page texts of an archive.org ``_djvu.xml`` text layer (one string per OBJECT, lines joined by newlines).

    :param xml_path: File.
    :returns: Page texts in scan order.
    """
    pages: List[str] = []
    for _, el in ET.iterparse(str(xml_path), events=("end",)):
        if el.tag == "OBJECT":
            lines = []
            for line in el.iter("LINE"):
                words = [w.text or "" for w in line.iter("WORD")]
                if words:
                    lines.append(" ".join(words))
            pages.append("\n".join(lines))
            el.clear()
    return pages


def assess_pages(pages: Sequence[str], n: int = 10, seed: int = 3407) -> Dict[str, Any]:
    """Text-layer quality on a sample of pages: Hebrew letters, Latin letters, English dictionary-word share.

    :param pages: Page texts.
    :param n: Pages sampled (non-empty pages only).
    :param seed: Seed.
    :returns: Per-page stats and totals.
    """
    rng = random.Random(seed)
    idx = [i for i, t in enumerate(pages) if len(t.strip()) > 200]
    pick = sorted(rng.sample(idx, min(n, len(idx))))
    words = TN.common_words()
    per = []
    for i in pick:
        t = pages[i]
        sc = TN.script_counts(t)
        lat = re.findall(r"[A-Za-z]{3,}", t)
        dict_share = (sum(1 for w in lat if w.lower() in words) / len(lat)) if (lat and words) else None
        per.append({"page_index": i, "hebrew_letters": sc["hebrew"], "latin_letters": sc["latin"],
                    "latin_words": len(lat), "dictionary_share": None if dict_share is None else round(dict_share, 3),
                    "sample": re.sub(r"\s+", " ", t)[:160]})
    he = sum(x["hebrew_letters"] for x in per)
    return {"pages": len(pages), "sampled": per, "hebrew_letters_sampled": he,
            "hebrew_usable": he > 500, "english_dictionary_share_median":
                float(np.median([x["dictionary_share"] for x in per if x["dictionary_share"] is not None]))
                if any(x["dictionary_share"] is not None for x in per) else None}


def _fold_en(text: str) -> str:
    """Lower-case alphanumeric folding for OCR containment checks (hyphenated line breaks joined).

    :param text: Text.
    :returns: Folded text (single spaces).
    """
    t = re.sub(r"-\s*\n\s*", "", text)
    t = re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", t).lower())
    return re.sub(r"\s+", " ", t).strip()


def ocr_containment(target: str, ocr_folded: str, n: int = 4) -> float:
    """Share of a target's word 4-grams found in the folded OCR text (verbatim check robust to OCR noise).

    :param target: English target.
    :param ocr_folded: :func:`_fold_en` of the book's text layer.
    :param n: Shingle size in words.
    :returns: Share in [0, 1] (1.0 for targets shorter than ``n`` words found verbatim, else 0.0).
    """
    w = _fold_en(target).split()
    if len(w) < n:
        return 1.0 if " ".join(w) in ocr_folded else 0.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return sum(1 for g in grams if g in ocr_folded) / len(grams)


def pd_pairs(he: S.CanonText, en: S.CanonText, ocr_text: str, slug: str, licence: str,
             tokenizer: Optional[Path] = None, template: Optional[Path] = None, min_containment: float = 0.6,
             band: Tuple[float, float] = (0.5, 2.5), group: int = 6, lic_class: Optional[str] = None,
             pair_type: str = "same_edition") -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """``translate_en`` rows from one edition's Hebrew and English versions (same verse numbering).

    Consecutive verses of a chapter are grouped (up to ``group``) so a row is a passage; a pair is kept when the
    English is verified in the book's own text layer (word 4-gram containment ≥ ``min_containment``) and the
    length ratio (normalised English / Hebrew characters) is inside ``band``.

    :param he: Hebrew version.
    :param en: English version.
    :param ocr_text: The book's text layer (archive.org).
    :param slug: Literature directory slug (``source = book:<slug>``).
    :param licence: Licence string.
    :param tokenizer: ``tokenizer.json``.
    :param template: Chat template.
    :param min_containment: Verification threshold.
    :param band: Length-ratio band.
    :param group: Verses per row.
    :param lic_class: Licence class (default: derived from ``licence``).
    :param pair_type: ``same_edition`` or ``cross_edition`` (Hebrew of another edition, recorded in provenance).
    :returns: ``(rows, drop counts)``.
    """
    folded = _fold_en(ocr_text)
    en_by = {(s.section, s.seg): s.text for s in en.segments}
    chapters: Dict[str, List[S.Segment]] = collections.defaultdict(list)
    for s in he.segments:
        if (s.section, s.seg) in en_by:
            chapters[s.section].append(s)
    rows: List[Dict[str, Any]] = []
    drops = collections.Counter()
    for sec in sorted(chapters, key=lambda x: [int(y) if y.isdigit() else y for y in x.split(":")]):
        segs = chapters[sec]
        for k in range(0, len(segs), group):
            g = segs[k:k + group]
            src = "\n".join(f"({s.seg}) {s.text}" for s in g)
            tgt = "\n".join(f"({s.seg}) {en_by[(s.section, s.seg)]}" for s in g)
            cont = ocr_containment(tgt, folded)
            ratio = len(TN.normalize_for_cer(tgt)) / max(1, len(TN.normalize_for_cer(src)))
            if cont < min_containment:
                drops["unverified_in_text_layer"] += 1
                continue
            if not band[0] <= ratio <= band[1]:
                drops["length_band"] += 1
                continue
            ref = f"{he.title} {sec}:{g[0].seg}" + (f"-{g[-1].seg}" if len(g) > 1 else "")
            user = R.user_prompt("translate_en", "edition", "hebrew", src)
            key = f"book:{slug}:{ref}"
            meta = R.new_meta("translate_en", "edition", "train", key)
            meta.update({"lang_class": "HE", "pgpid": None, "shelfmark": None, "shelfmark_group": None,
                         "held_out_vision_benchmark": False, "input_cer": None, "input_source": "sefaria_edition",
                         "input_truncated": False, "licence": licence,
                         "licence_class": lic_class or S.licence_class(licence),
                         "source": f"book:{slug}", "work": slug, "page_range": ref,
                         "provenance": {"sefaria_title": he.title, "he_version": he.version, "en_version": en.version,
                                        "pair_type": pair_type,
                                        "text_layer_containment": round(cont, 3), "length_ratio": round(ratio, 3)},
                         "target_origin": "edition_translation", "self_label": False, "target_raw": None,
                         "strip_ops": [], "reader_exposed": None, "exposure_level": None, "family_file": FAMILY_PD})
            row = {"messages": R.make_messages(user, tgt), "meta": meta}
            _finish(row, tokenizer, template)
            if row["meta"]["tokens"] > MAX_ROW_TOKENS:
                drops["over_token_cap"] += 1
                continue
            rows.append(row)
    return rows, dict(drops)


def load_pd_registry(p: _paths.Paths) -> List[Dict[str, Any]]:
    """The local registry of public-domain editions (``logs/text_finetune_27b/pd_editions.json``, git-ignored).

    Each entry: ``slug``, ``files`` (list of ``{pdf, djvu_xml}`` relative to the literature dir), optional
    ``licence`` / ``licence_class`` and ``sefaria`` ``{he_corpus, en_corpus, pair_type}`` (corpus keys of
    :data:`sefaria.CORPORA` holding the Hebrew and the edition's English; ``pair_type`` = ``cross_edition`` when the
    Hebrew comes from another edition).

    :param p: Build paths.
    :returns: Entries (empty when the file is absent).
    """
    q = p.get("pd_editions") or (p.repo / "logs/text_finetune_27b/pd_editions.json")
    q = Path(q)
    if not q.exists():
        return []
    return json.loads(q.read_text(encoding="utf-8")).get("editions") or []


def build_pd(p: Optional[_paths.Paths] = None, seed: int = 3407) -> Dict[str, Any]:
    """Assess the registered PD editions and write ``families/pd_editions.jsonl`` where pairs can be made.

    :param p: Build paths.
    :param seed: Page-sample seed.
    :returns: Report (also stored under ``pd_editions`` in ``canon_manifest.json`` when that file exists).
    """
    p = p or _paths.load_paths()
    lit = p.academic_literature
    report: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    root = S.sefaria_dir(p)
    for ed in load_pd_registry(p):
        slug = ed["slug"]
        ent: Dict[str, Any] = {"files": []}
        ocr_all = []
        for f in ed.get("files") or []:
            pages = djvu_pages(lit / slug / f["djvu_xml"])
            ocr_all.append("\n".join(pages))
            ent["files"].append({"pdf": f.get("pdf"), "assessment": assess_pages(pages, seed=seed)})
        ent["hebrew_text_layer_usable"] = any(x["assessment"]["hebrew_usable"] for x in ent["files"])
        sf = ed.get("sefaria")
        if sf:
            he_l = S.load_corpus(sf["he_corpus"], root)
            en_l = S.load_corpus(sf["en_corpus"], root)
            he_t, en_t = (he_l[0] if he_l else None), (en_l[0] if en_l else None)
            if he_t is not None and en_t is not None:
                made, drops = pd_pairs(he_t, en_t, "\n".join(ocr_all), slug,
                                       ed.get("licence") or "public domain", p.tokenizer_json, p.chat_template,
                                       lic_class=ed.get("licence_class"), pair_type=sf.get("pair_type", "same_edition"))
                for r in made:  # split by chapter (all verse groups of a chapter share a split)
                    chapter = r["meta"]["page_range"].rsplit(" ", 1)[-1].split(":")[0]
                    r["meta"]["split"] = split_of(f"{slug}:{chapter}")
                rows.extend(made)
                ent["pairs"] = {"rows": len(made), "tokens": sum(r["meta"]["tokens"] for r in made), "dropped": drops,
                                "splits": dict(collections.Counter(r["meta"]["split"] for r in made))}
            else:
                ent["pairs"] = {"rows": 0, "reason": "Sefaria Hebrew / English version not in the cache"}
        report[slug] = ent
    if rows:
        R.write_jsonl(p.out("families", f"{FAMILY_PD}.jsonl"), rows)
    man_path = p.out("canon_manifest.json")
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["pd_editions"] = report
        man.setdefault("families", {})[FAMILY_PD] = len(rows)
        man.setdefault("counts", {})[FAMILY_PD] = _counts(rows).get("translate_en", {})
        _atomic_json(man_path, man)
    return report


# ---------------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: Exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="align KTIV transcriptions to Sefaria and write the two families")
    b.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    b.add_argument("--corpus", action="append")
    b.add_argument("--limit-mss", type=int, default=0)
    b.add_argument("--max-words", type=int, default=MAX_CHUNK_WORDS)
    a = sub.add_parser("audit", help="draw the stratified hand-audit sample from the last build's pool")
    a.add_argument("--n", type=int, default=30)
    a.add_argument("--seed", type=int, default=3407)
    a.add_argument("--out", help="output jsonl (default audit/canon_audit_sample_new.jsonl; the hand-checked "
                                 "audit/canon_audit_30.jsonl is never overwritten by default)")
    sub.add_parser("pd", help="assess the public-domain editions and build their pairs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = _paths.load_paths()
    if args.cmd == "build":
        man = build(BuildConfig(floor=args.floor, corpora=tuple(args.corpus) if args.corpus else None,
                                limit_mss=args.limit_mss, max_words=args.max_words), p)
        print(json.dumps({k: man[k] for k in ("families", "manuscripts", "rejected_chunks", "similarity",
                                               "variant_counts", "holdouts")}, ensure_ascii=False, indent=1))
    elif args.cmd == "audit":
        pool = R.load_jsonl(p.out("audit", "canon_audit_pool.jsonl"))
        sample = audit_sample(pool, args.n, args.seed)
        out = Path(args.out) if args.out else p.out("audit", "canon_audit_sample_new.jsonl")
        R.write_jsonl(out, sample)
        print(f"{len(sample)} audit records -> {out}")
    else:
        print(json.dumps(build_pd(p), ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
