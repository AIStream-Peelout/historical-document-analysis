"""The single public module for shelf-mark normalisation (facade over the two engines).

Every module outside the facade, the two engines and ``merging/merge_shelfmarks.py`` imports shelf-mark helpers
from here, never from ``document_models/genizah_normalizer.py`` (:class:`ShelfmarkNormalizer`) or
``merging/institution_tokens.py`` directly. The facade is thin: it calls those engines (and the join splitter of
``merging/merge_shelfmarks.py``) and does not copy their code. Since PR-6 it owns the printed-mark helpers that
used to live in ``finetuning/text27b/lit_segment.py`` (:func:`split_joins`, :func:`split_piece_lists`,
:func:`legacy_variants`, ``_clean_sm``, :func:`alt_classmark`) and ``finetuning/text27b/shelfmarks.py``
(:func:`prep_shelfmark`), and :func:`cudl_manifest`; those modules re-import them. The lint test
``tests/test_shelfmark_single_source.py`` enforces this rule over ``src/``.

Tracking key
    The ES canonical id (:func:`canonical_id`, e.g. ``Cambridge_CUL_T_S_AS_18_170``) is the tracking key
    of every ledger, hold-out, manifest and registry. The only fuzzy join key is the institution-qualified
    :func:`loose_key`. The institution-free :func:`core_key` collides across institutions (175 key groups,
    402 ES ids, measured 2026-11-02: ``Berlin_JCB_A2`` vs ``Manchester_JRL_A_2``) and is never persisted.

PR-1 behaviour (kept)
    Every function reproduces the reference columns of the shelf-mark probe matrix byte for byte
    (``tests/fixtures/shelfmark_probes.json``) except for the departures listed in the golden test's
    ``INTENDED_CHANGES`` table. The id-input rule (PR-1): when the input already is an ES id (a registry token
    prefix followed by a core, recognised by :func:`split_id`, with no whitespace), :func:`core_id` returns its
    core and :func:`canonical_id` returns it unchanged when no institution is given or the institution's token is
    the id's own token; only a conflicting institution is left to the engine, so it is not silently ignored. The
    rule covers registry-token ids only: an ES id without a registry token (slug-prefixed institutions such as
    ``Chabad_Lubavitch_Library_…``, ``Unknown_…`` ids, bare NLI numbers; 256 ids on 2026-11-03) is read like a
    raw mark.

PR-2 behaviour (implemented; :data:`NORMALISER_VERSION` ``2026.11.03-3``, kept in ``2026.11.03-4``)
    Facade-only fixes; the engines and the merge recipe are unchanged, so no ES id changes.

    * :func:`side_suffix` and one id per leaf: ``T-S 12.301 r/recto/verso/ע"א/ע"ב``, ``T-S 10J5.17 1r`` and
      ``AIU XII.87v`` give the side-less ES id; the side is a separate value. A folio other than 1 after a
      box-and-item mark is a child leaf and stays in the id (``T-S Misc. 28.79, f. 14r`` ->
      ``Cambridge_CUL_T_S_Misc_28_79_14``). Folio ranges
      (``…9r-10r``), piece letters (``12a``) and roman numerals (``Plate IV``) are never stripped.
    * :func:`canonical_id` pre-pass: a glued prefix (``T-S12.301``) gets its boundary back; a pre-fix Oxford id
      (``Oxford_Bodleian_MS_heb_c_13_25``) is re-parsed to the current id (also in :func:`core_id`); a spaced
      institution head (``Cambridge University Library T-S AS 181.163``) is stripped.
    * Series folds (``Taylor-Schechter`` -> ``T-S``, ``T-S Arabic [Box]`` -> ``T-S Ar.``, ``ULC Or`` -> ``CUL Or``,
      ``Lewis-Gibson X`` / ``Westminster College X`` -> ``L-G X``), used by :func:`canonical_id`;
      ``BnF Hebr`` -> ``hebr`` is a :func:`variants` candidate only (no ES id uses either BnF form for a probe).
    * Institution inference when ``institution`` is not given: the head inside the mark (``resolve_token``),
      then :data:`institution_tokens.SERIES_TOKENS` (a bare ``Or. <n>`` stays ``None``), then an engine-parsed
      Bodleian mark (``Heb. e. 98``), then an institution needle in the digit-free prefix (``Geneva 119``).
      ``MS heb`` alone and a mark with ``°`` never give Oxford; a Lewis-Gibson / Mosseri series word under a
      Cambridge University Library KTIV head gives its own token.
    * ``None`` from :func:`canonical_id` (and ``""`` from :func:`core_id` / :func:`core_key` /
      :func:`loose_key`) for join strings (use :func:`split` first), bare NLI system numbers, garbage, and marks
      whose institution cannot be determined. Garbage is a placeholder (``N/A``, ``see above``) with any
      institution, or a digit-free mark naming nothing when no institution is given.
    * Every ES id keeps a non-empty :func:`loose_key` of its own (garbage ids, the bare NLI-number id and
      ``Mittwoch_8`` aside), and :func:`canonical_id` is idempotent on its own outputs: a slug id
      (:func:`_is_slug_id`) comes back unchanged (an ``Unknown_…`` id gives ``None`` but is keyed on itself).
      Exception: a slug id whose holder is one word (``Mittwoch_8``, ``Krupp_2188`` from the institution
      ``Krupp``) is not told apart from a bare core (``Mingana_12``, ``DK_12``) and is read like a raw mark
      (``None`` without an institution; 60 of 84,131 distinct merged-corpus outputs, 1 ES id).
    * Known cost of one id per leaf (R3): three ES ids keyed on one side (``Cambridge_CUL_T_S_H3_104r``,
      ``Cambridge_CUL_T_S_NS_J401r``, the Cairo ENL ``…_404r``) are not what their raw marks now give; the
      re-key is listed in section 4 of the migration plan and ``resolve()`` (PR-6) falls back to the side-ful id.
    * :func:`split`, :func:`variants` and :func:`parent`.

PR-4 behaviour (this version, :data:`NORMALISER_VERSION` ``2026.11.03-4``)
    * A colon between two digits with no space (``TS 12:301``, ``T-S NS 327:140``, ``T-S Ar.41:18``) is a
      class-mark separator, not the end of a catalogue head: a facade pre-pass (:func:`_class_colon`) rewrites it
      to ``.`` before head detection, in :func:`canonical_id`, :func:`core_id` (so :func:`core_key` and
      :func:`loose_key`), :func:`side_suffix` and :func:`variants`. Before, the engine read everything up to the
      colon as a head (``core_key("T-S NS 327:140") == "140"``, ``loose_key("TS 12:301") ==
      loose_key("TS 5:301")``). A colon followed by a space stays a head (``Cambridge CUL: T-S 12.301``). Every
      probe of the golden fixture keeps its value; no ES id changes.

PR-6 behaviour (no existing output changes; :data:`NORMALISER_VERSION` stays ``2026.11.03-4``)
    * :class:`ShelfmarkIndex`: the ES ids of the merged corpus (``merged/merged_shelfmarks.jsonl``, read-only) with
      their alias ids (PGP historic spellings and every source spelling of a record, :func:`record_marks`),
      :func:`loose_key` and :func:`core_key` maps, the PGP ids and the parent lookup (:func:`parent` with the index's
      ids). Built from a path or from records; cached on disk only under a caller-given directory, keyed by
      :data:`NORMALISER_VERSION`, the index format and the merged file's size and modification time.
    * :func:`resolve` (``x`` = raw mark, ES id or ``pgpid:<n>``) -> :class:`Resolution` ``(cids, how)``: :func:`split`
      -> per part :func:`side_suffix` -> per :func:`variants` spelling (``lossy=False``): exact ES id -> alias ->
      :func:`loose_key` -> :func:`core_key` (only when its ids name one institution); then the side-ful ES id (R3:
      three ES ids keep a side) and :func:`parent` (a multi-fragment parent only, never a box / volume id). It never
      returns an id that is not in the index. An optional ``institution`` is tried before the inferred one. A
      numeric range (``T-S 12.6-7``, ``ff. 9-10``) gets ``note="range"`` and is ``unmatched`` unless the ES holds
      a record of that very range: what a range names is open question Q11 of the migration plan, so it never
      collapses to its first leaf or to a parent.
    * :func:`variants` / :func:`legacy_variants` gain ``lossy`` (default True = unchanged output): False leaves out
      the rewrites that drop a glued note number or a piece letter (they can name another leaf).
    * The printed-mark helpers of ``text27b/lit_segment.py`` and ``text27b.shelfmarks.prep_shelfmark`` moved here
      verbatim; :func:`cudl_manifest` is implemented here (a verbatim copy of the scrape-queue function; the
      queue's own copy is removed when that file's pending work lands).

:data:`NORMALISER_VERSION` is bumped on any change to engine or facade output and belongs in every key
cache signature.
"""

from __future__ import annotations

import collections
import csv
import functools
import hashlib
import json
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import (AbstractSet, Any, Dict, FrozenSet, Iterable, Iterator, List, Mapping, NamedTuple, Optional,
                    Pattern, Set, Tuple, Union)

from src.datasets.document_models.genizah_normalizer import OxfordShelfmark, ShelfmarkNormalizer
from src.datasets.merging import institution_tokens as _engine

#: The Bodleian registry token (``Oxford_Bodleian``), re-exported from the engine.
OXFORD_TOKEN: str = _engine.OXFORD_TOKEN

#: Version of the normalisation output. Bump on any change to an engine's or this facade's output, and
#: include it in every cache keyed by shelf-mark keys.
NORMALISER_VERSION: str = "2026.11.03-4"

# ------------------------------------------------------------------------------------ patterns (facade-owned)

#: A letter in any script (a word character that is not a digit or an underscore): word boundaries for series words.
_LETTER = r"[^\W\d_]"

#: Legacy series spellings the normaliser does not accept -> the spelling it does (R4). 0 ES ids use the old forms.
_SERIES_FOLDS: Tuple[Tuple[Pattern[str], str], ...] = (
    (re.compile(r"\bTaylor[-\s]+Schechter\b"), "T-S"),
    (re.compile(r"\bT-S\s+Arabic\b\.?\s*(?:Box\b\s*)?"), "T-S Ar. "),
    (re.compile(r"\bULC\s+Or\b"), "CUL Or"),
    (re.compile(r"\bLewis[-\s]+Gibson[\s,]+(?!L-G\b)(?=[^\s:])"), "L-G "),
    (re.compile(r"\bWestminster\s+College[\s,]+(?!L-G\b)(?=[^\s:])"), "L-G "),
)
#: ``BnF Hebr.`` -> ``hebr.``: a :func:`variants` candidate only, not applied by :func:`canonical_id`.
_BNF_FOLD: Tuple[Pattern[str], str] = (re.compile(r"\bBnF\s+Hebr\b\.?"), "hebr.")

#: The safe spelling rewrites of the unseen-data ``rekey.clean_shelfmark`` key cleaner. Its side-suffix and
#: ``r``/``v``-after-digit rules are NOT here (they turn ``Plate IV`` into ``Plate I`` and ``10J5.17 1r`` into
#: ``10J5.17 1``; :func:`side_suffix` replaces them), and neither is its ``f.``/``fol.`` deletion (it would also
#: delete the Bodleian size letter of ``MS Heb. f. 56``).
_SAFE_SPELLINGS: Tuple[Tuple[Pattern[str], str], ...] = (
    (re.compile(r"\bArabic Box\b", re.I), "Ar."),
    (re.compile(r"\bBox\b\s*", re.I), ""),
    (re.compile(r"\bT\.-S\.|\bT\.S\.|\bTS\b(?=[\s\d])", re.I), "T-S"),
    (re.compile(r"\((?:secondary use|not listed)[^)]*\)", re.I), ""),
)

#: A Taylor-Schechter prefix glued to the number that follows (R2: ``T-S12.301``).
_GLUED_TS = re.compile(r"(?<![^\W\d_])(T-S|TS)(?=\d)")
#: KTIV head: ``<institution>, <city>, <country> Ms. <core>`` (or ``… Cod. <core>``, e.g. ``Austrian National
#: Library, Vienna, Austria Cod. hebr. 65``; the engine keeps such a head in the core, as the ES ids do).
_KTIV_HEAD = re.compile(r"^(?P<head>.*,.*?)\s(?:Ms|Cod)\.\s+(?P<rest>.*)$", re.I | re.S)
#: Taylor-Schechter spellings the normaliser reads as ``T-S`` but that are not a ``SERIES_TOKENS`` word
#: (``T.-S.``, ``T.S.``, ``T. S.``, ``TS``): rewritten for institution inference only (the core is the engine's).
_TS_SPELLING = re.compile(r"\bT\.\s*-?\s*S\.|\bTS\b(?=[\s\d])", re.I)
#: A colon between two digits with no space (``TS 12:301``): a class-mark separator, never a catalogue head.
_CLASS_COLON = re.compile(r"(?<=\d):(?=\d)")
#: A bare NLI system number (``990000412230205171``): a catalogue record number, not a shelf mark.
_NLI_SYSTEM_NUMBER = re.compile(r"\d{12,}")
#: Trailing join punctuation a splitter can leave on a part (``T-S 16.22 +``, ``T-S 12.613;``).
_JOIN_TAIL = " \t+;&,-/"
#: Placeholders that name no mark, whatever institution comes with them (compared after :func:`_placeholder_key`).
_PLACEHOLDERS = frozenset({"", "n/a", "na", "unknown", "shelfmark unknown", "see above", "none", "null", "nan",
                           "item page", "כתב יד"})

#: Side words / markers -> normalised side.
_SIDE_WORDS = {"recto": "r", "verso": "v", "א": "r", "ב": "v"}
_SIDE_WORD = re.compile(r"^(?P<base>.*?\S)[\s,]+(?P<side>recto|verso)\.?$", re.I | re.S)
_SIDE_HEBREW = re.compile(r"^(?P<base>.*?\S)[\s,]+ע\s*[\"״'׳]?\s*(?P<side>[אב])$", re.S)
_SIDE_RANGE = re.compile(r"\d[rv]\s*[-–_]\s*\d+[rv]\.?$")
_SIDE_FOLIO = re.compile(
    r"^(?P<base>.*[./]\d+[A-Za-z]?)[\s,]+(?:(?:fol|ff?)\.\s*)?(?P<folio>\d+)\s*(?P<side>[rv])\.?$", re.S)
#: A base that ends in a join (``T-S 12.301 recto and`` of ``… recto and verso``), or in a join and a number
#: (``T-S 12.301 f. 1r and 2`` of ``… 1r and 2v``): no side is stripped then. A ``/`` glued to a number
#: (``577/3``) is part of the mark, not a join.
_JOINED_BASE = re.compile(r"(?:\band|\bund|[+&/;]|\b(?:recto|verso))(?:\s+\d+)?$", re.I)
_SIDE_LETTER = re.compile(r"^(?P<base>.*\d[A-Za-z]?)[\s,]+(?P<side>[rv])\.?$", re.S)
_SIDE_GLUED = re.compile(r"^(?P<base>.*\d)(?P<side>[rv])\.?$", re.S)


def _class_colon(mark: str) -> str:
    """*mark* with every colon between two digits (no space) rewritten to ``.`` (``TS 12:301`` -> ``TS 12.301``).

    Such a colon separates the parts of a class mark; read as the colon of a ``<head>: <mark>`` catalogue head it
    would drop everything before it. A colon followed by a space (``Cambridge CUL: T-S 12.301``) is left alone.

    :param mark: Raw mark.
    :type mark: str
    :return: The rewritten mark (the mark itself when it has no such colon).
    :rtype: str
    """
    return _CLASS_COLON.sub(".", mark) if mark else mark


def _squash(text: str) -> str:
    """*text* lower-cased with every non-alphanumeric character removed (``T-S`` / ``T_S`` -> ``ts``).

    :param text: Any string.
    :type text: str
    :return: The squashed string.
    :rtype: str
    """
    return re.sub(r"[\W_]+", "", text).lower()


#: Squashed series words (:data:`institution_tokens.SERIES_TOKENS`: ``T-S``, ``ENA``, ``DK``, ``Or.``, ``Add.``,
#: ``L-G``, ``MS heb``, ``Bodl``, ``Yevr`` …) and the acronym parts of registry tokens (``CUL``, ``JTS``, ``MTA``,
#: ``JRL`` …): a mark whose head is one of these is a shelf mark, not a slug id (:func:`_is_slug_id`).
_SERIES_HEADS = frozenset(
    {_squash(word) for word in _engine.SERIES_TOKENS}
    | {_squash(part) for tok in _engine._REGISTRY_TOKENS for part in tok.split("_")[1:] if part.isupper()})


def _series_patterns() -> List[Tuple[Pattern[str], str]]:
    """Whole-word, case-insensitive patterns of the :data:`institution_tokens.SERIES_TOKENS` words.

    A space inside a series word matches any run of whitespace, including none (``Or.1080`` = ``Or. 1080``).

    :return: ``(pattern, series word)`` pairs, longest word first.
    :rtype: List[Tuple[Pattern[str], str]]
    """
    out: List[Tuple[Pattern[str], str]] = []
    for word in sorted(_engine.SERIES_TOKENS, key=lambda w: (-len(w), w)):
        body = r"\s*".join(re.escape(part) for part in word.split(" "))
        left = f"(?<!{_LETTER})" if word[:1].isalpha() else ""
        right = f"(?!{_LETTER})" if word[-1:].isalpha() else ""
        out.append((re.compile(left + body + right, re.I), word))
    return out


_SERIES_PATTERNS: List[Tuple[Pattern[str], str]] = _series_patterns()


# ------------------------------------------------------------------------------------ ids and cores

def split_id(cid: str) -> Tuple[Optional[str], str]:
    """Split an ES id into its registry token (longest prefix) and its core.

    :param cid: Canonical (ES) id, e.g. ``Cambridge_CUL_T_S_12_301``.
    :type cid: str
    :return: ``(token, core)``; ``(None, cid)`` when no registry token prefixes the id.
    :rtype: Tuple[Optional[str], str]
    """
    return _engine.split_id(cid)


def _id_input_core(mark: str) -> Optional[str]:
    """The core of *mark* when *mark* already is an ES id (the PR-1 id-input rule).

    An ES id is a registry token followed by ``_<core>``, with no whitespace. One exception has an empty core:
    a bare multi-word registry token (``Cambridge_Lewis_Gibson``, an ES id whose shelf mark is the collection
    name alone) is an ES id with core ``""``. A bare one-word token (``Geneva``, ``Frankfurt``) is not taken
    as an ES id, because it is indistinguishable from a plain word.

    :param mark: Raw mark or ES id.
    :type mark: str
    :return: The core after the registry token (``""`` for a bare multi-word token), or ``None`` when *mark*
        is not an ES id (no registry token prefix, a bare one-word token, or any whitespace).
    :rtype: Optional[str]
    """
    if not mark or any(ch.isspace() for ch in mark):
        return None
    token, core = _engine.split_id(mark)
    if token is None or (not core and "_" not in token):
        return None
    return core


def _reparse_old_oxford(mark: str) -> Optional[str]:
    """The current ES id of a pre-fix Oxford id (R11), e.g. ``Oxford_Bodleian_MS_heb_c_13_25``.

    Before the 2026-10 Oxford fix the Bodleian core had no ``Bodl_`` series word; such ids survive in old
    registries. The core is re-read by the engine's Bodleian parser (:meth:`ShelfmarkNormalizer.parse_oxford`)
    with its underscores as spaces. The ES holds 0 ids of the old form, so no ES id is re-parsed.

    :param mark: Raw mark or id.
    :type mark: str
    :return: The current id (``Oxford_Bodleian_Bodl_MS_heb_c_13_25``), or ``None`` when *mark* is not an
        old-form Oxford id.
    :rtype: Optional[str]
    """
    if _id_input_core(mark) is None:
        return None
    token, core = _engine.split_id(mark)
    if token != _engine.OXFORD_TOKEN or not core.startswith("MS_"):
        return None
    parsed = ShelfmarkNormalizer.parse_oxford(core.replace("_", " "))
    return _engine.combine(token, parsed.core) if parsed is not None else None


def _is_nli_system_number(mark: str) -> bool:
    """Whether *mark* is a bare NLI catalogue system number (``990000412230205171``), not a shelf mark.

    :param mark: Raw mark.
    :type mark: str
    :return: True for 12 or more digits and nothing else.
    :rtype: bool
    """
    return bool(_NLI_SYSTEM_NUMBER.fullmatch((mark or "").strip()))


def _series_hit(text: str) -> Optional[Tuple[int, str]]:
    """The first :data:`institution_tokens.SERIES_TOKENS` word in *text* (longest word at that position).

    :param text: Raw mark.
    :type text: str
    :return: ``(start offset, series word)``, or ``None`` when *text* has no series word.
    :rtype: Optional[Tuple[int, str]]
    """
    best: Optional[Tuple[int, int, str]] = None
    for pattern, word in _SERIES_PATTERNS:
        m = pattern.search(text)
        if m and (best is None or (m.start(), -len(word)) < (best[0], -best[1])):
            best = (m.start(), len(word), word)
    return (best[0], best[2]) if best else None


def _placeholder_key(mark: str) -> str:
    """*mark* lower-cased with brackets, dashes, dots, underscores and question marks blanked, spaces collapsed.

    :param mark: Raw mark.
    :type mark: str
    :return: The comparison key for :data:`_PLACEHOLDERS`.
    :rtype: str
    """
    return re.sub(r"\s+", " ", re.sub(r"[()\[\]{}\-–—_.?!*]+", " ", mark or "")).strip().lower()


def _is_garbage(mark: str, institution: Optional[str] = None) -> bool:
    """Whether *mark* names nothing.

    A placeholder (:data:`_PLACEHOLDERS`: ``N/A``, ``see above``, ``(shelfmark unknown)``, ``None``, a dash, empty)
    is garbage whatever the institution. Otherwise a mark is garbage only when no institution is given (``None`` or
    blank) and it has no digit, no institution needle and no series word (``Plate IV``). With an institution a
    digit-free mark names a holding (``MS Meunier`` of the Ben Zvi Institute, the KTIV private-holder marks such as
    ``Benayahu, Meir, Jerusalem, Israel``, which the ES holds as ids) and is read as in the reference.

    Narrower than "no digit, or ``classify_shelfmark == 'reject'``": the validator rejects about 50 real probe
    marks (KTIV heads, ``Geneva 119``, ``PER H 22``, ``F 1908.44`` …), and a digit-free mark that names a
    holder or a volume (``Mosseri, Jacques, Paris, France``; ``Freer Genizah XVI``) is kept as in the reference.

    :param mark: Raw mark.
    :type mark: str
    :param institution: Institution spelling given with the mark (blank counts as none).
    :type institution: Optional[str]
    :return: True for garbage.
    :rtype: bool
    """
    text = mark or ""
    if _placeholder_key(text) in _PLACEHOLDERS:
        return True
    if (institution and institution.strip()) or any(ch.isdigit() for ch in text):
        return False
    return _engine.resolve_token(text) is None and _series_hit(text) is None


def _is_slug_id(mark: str) -> bool:
    """Whether *mark* is shaped like an ES id that carries no registry token.

    No whitespace, no ``.`` and no ``-``, no registry token (:func:`split_id`), not a bare NLI system number and not
    a placeholder (``Item_page``); the first two ``_`` segments are alphabetic (a slug names its holder before any
    number: ``Chabad_Lubavitch_Library_…``, ``Private_Collection_…``) and no head of one to three segments is a
    series word or a registry-token acronym (:data:`_SERIES_HEADS`: ``T_S_12_301``, ``MS_heb_c_13``, ``JRL_A_2``). A
    bare core or a malformed mark (``T_S_12_301``, ``ENA_2727_5``, ``A_2``, ``DK_12``, ``12_301``, ``T-S_12.301``) is
    read like a raw mark. Accepted:
    slug-token ids (``Russian_State_Library_Guenzburg_1181``, ``BNU_Strasbourg_4038``, ``Ben_Zvi_Institute_MS_Meunier``:
    the slug prefix names the holder) and ``Unknown_…`` ids (``Unknown_DK_123``) other than garbage ones (an
    ``Unknown_`` id whose rest is garbage, :func:`_is_garbage`: ``Unknown_N_A``, ``Unknown_see_above``). 256 ES ids
    on 2026-11-03 have no registry token; 251 have this shape. The other five are read like raw marks: the bare
    NLI-number id, ``Item_page``, two garbage ``Unknown_`` ids, and ``Mittwoch_8`` (a one-word holder, not told
    apart from a bare core such as ``Mingana_12``). 0 of the 80,709 merged ``shelfmark_display`` marks has this
    shape, so no raw mark is affected.

    :param mark: Raw mark or id.
    :type mark: str
    :return: True for a no-registry-token id shape.
    :rtype: bool
    """
    if not mark or any(ch.isspace() or ch in ".-" for ch in mark):
        return False
    if _engine.split_id(mark)[0] is not None or _is_nli_system_number(mark) or _placeholder_key(mark) in _PLACEHOLDERS:
        return False
    segments = mark.split("_")
    if len(segments) < 2 or not (segments[0].isalpha() and segments[1].isalpha()):
        return False
    if any(_squash("_".join(segments[:n])) in _SERIES_HEADS for n in (1, 2, 3)):
        return False
    return not (mark.startswith("Unknown_") and _is_garbage(mark[len("Unknown_"):].replace("_", " ")))


def core_id(mark: str) -> str:
    """The institution-free core of a shelf mark, case preserved.

    :meth:`ShelfmarkNormalizer.to_canonical_id` of the mark as written (``T-S 12.301`` -> ``T_S_12_301``; a side
    suffix stays in the core: use :func:`side_suffix` or :func:`canonical_id` for the leaf). An ES id passed in
    gives its own core (``Cambridge_CUL_T_S_12_301`` -> ``T_S_12_301``, the token stripped by :func:`split_id`;
    a pre-fix Oxford id gives its current core, ``Oxford_Bodleian_MS_heb_c_13_25`` -> ``Bodl_MS_heb_c_13_25``);
    the bare-token id ``Cambridge_Lewis_Gibson`` gives ``""`` (see :func:`_id_input_core`). Returns ``""`` for a
    join string (call :func:`split` first), a bare NLI system number and garbage (:func:`_is_garbage`). A colon
    between two digits is read as ``.`` (:func:`_class_colon`: ``T-S NS 327:140`` -> ``T_S_NS_327_140``).

    :param mark: Raw shelf mark or ES id.
    :type mark: str
    :return: The core, or ``""``.
    :rtype: str
    """
    if _id_input_core(mark) is None:
        mark = _class_colon(mark)
        if _is_join(mark) or _is_nli_system_number(mark) or _is_garbage(mark):
            return ""
    return _core(mark)


def _core(mark: str) -> str:
    """The core of *mark* with no ``""`` rule: the id-input core, the re-parsed pre-fix Oxford core, else the engine's.

    :param mark: Raw shelf mark or ES id.
    :type mark: str
    :return: The core.
    :rtype: str
    """
    old = _reparse_old_oxford(mark)
    if old is not None:
        return _engine.split_id(old)[1]
    core = _id_input_core(mark)
    return core if core is not None else ShelfmarkNormalizer.to_canonical_id(mark)


def _institution_token(institution: str) -> str:
    """The registry token of an institution spelling, a series word written as an institution, else a slug.

    ``institution_token`` of the engine, except that an institution spelling with no registry needle that is
    itself a series word (``Westminster College`` -> ``Cambridge_Lewis_Gibson``) takes the series word's token.

    :param institution: Institution / collection spelling.
    :type institution: str
    :return: The token.
    :rtype: str
    """
    return (_engine.resolve_token(institution) or _engine.SERIES_TOKENS.get(institution.strip())
            or _engine.institution_token(institution))


def _head(mark: str) -> Tuple[str, str, bool]:
    """The institution head written inside a mark, the rest, and whether the engine leaves the head in the core.

    Three shapes: a legacy-store head before the last ``:``; a KTIV head before `` Ms. `` (with a comma before it); and a
    spaced head before the first series word (``Cambridge University Library T-S AS 181.163``, R5), which the
    engine does not strip. A head counts only if it names a registry institution (``resolve_token``).

    :param mark: Raw mark.
    :type mark: str
    :return: ``(head, rest, spaced)``; ``("", mark, False)`` when the mark carries no recognised head.
    :rtype: Tuple[str, str, bool]
    """
    if ":" in mark:
        head, rest = mark.rsplit(":", 1)
        spaced = False
    else:
        m = _KTIV_HEAD.match(mark)
        if m:
            head, rest, spaced = m.group("head"), m.group("rest"), False
        else:
            hit = _series_hit(mark)
            if hit is None or hit[0] == 0:
                return "", mark, False
            head, rest, spaced = mark[:hit[0]], mark[hit[0]:], True
    head = head.strip(" ,;:")
    if not head or any(ch.isdigit() for ch in head) or _engine.resolve_token(head) is None:
        return "", mark, False
    return head, rest.strip(), spaced


def _series_fold(mark: str, bnf: bool = False) -> str:
    """Rewrite legacy series spellings into the forms the normaliser accepts (R4).

    :param mark: Raw mark.
    :type mark: str
    :param bnf: Also fold ``BnF Hebr.`` to ``hebr.`` (a :func:`variants` candidate; not used by
        :func:`canonical_id`).
    :type bnf: bool
    :return: The folded mark (the mark itself when nothing applies).
    :rtype: str
    """
    out = mark
    for pattern, repl in _SERIES_FOLDS + ((_BNF_FOLD,) if bnf else ()):
        out = pattern.sub(repl, out)
    return re.sub(r"\s+", " ", out).strip() if out != mark else mark


#: The ``MS heb`` series word in any spelling (``Ms. Heb.``, ``MS heb``): on its own it does not name Oxford.
_MS_HEB = re.compile(r"\bms\.?\s*heb\b", re.I)
#: Cambridge sub-collection tokens a series word gives under a Cambridge University Library KTIV head.
_CUL_SUBCOLLECTIONS = frozenset({"Cambridge_Lewis_Gibson", "Cambridge_Mosseri"})


def _infer_token(text: str, base: str) -> Optional[str]:
    """Infer the registry token of a mark given without an institution.

    Order: the head inside the mark (:func:`_head`, ``resolve_token``: legacy-store ``Inst:``, KTIV ``… Ms.`` /
    ``… Cod.`` and spaced heads), then the first series word (:data:`institution_tokens.SERIES_TOKENS`, with the
    ``T.-S.`` / ``T.S.`` / ``T. S.`` / ``TS`` spellings read as ``T-S``; an ambiguous bare ``Or.`` gives ``None``),
    then a mark the engine parses as a Bodleian shelf mark (``Heb. e. 98``), then an institution needle in the
    digit-free prefix of the mark (``Manchester L 155``, ``Geneva 119``, ``Strasbourg 4038``, ``NLI 4°9463.10``,
    ``Dropsie 472``). Under a Cambridge University Library head a Lewis-Gibson or Mosseri series word names the
    token (``…, England Ms. L-G Add. 3`` -> ``Cambridge_Lewis_Gibson``, as the ES ids do). Without a head, Oxford
    is inferred only when the engine parses the mark as a Bodleian shelf mark (and it has no ``°``) or the mark
    names Oxford by a word other than ``MS heb`` (``The Bodleian Libraries, …``); a mark with ``°`` never infers
    it. ``MS heb`` alone does not name Oxford: ``Ms. Heb. 24°10285``, ``Ms. heb. 28 (JTS)`` (NLI / JTS marks)
    give ``None``.

    :param text: The mark after the pre-pass (glued prefix and series folds).
    :type text: str
    :param base: The mark without its side suffix and spaced head.
    :type base: str
    :return: The token, or ``None`` when no institution can be determined.
    :rtype: Optional[str]
    """
    head, rest, _ = _head(text)
    hit = _series_hit(_TS_SPELLING.sub("T-S", rest))
    series = _engine.SERIES_TOKENS[hit[1]] if hit is not None else None
    if head:
        token = _engine.resolve_token(head)
        return series if token == "Cambridge_CUL" and series in _CUL_SUBCOLLECTIONS else token
    oxford = "°" not in base and ShelfmarkNormalizer.parse_oxford(base) is not None
    if hit is not None:
        token = series
    elif oxford:
        return _engine.OXFORD_TOKEN
    else:
        prefix = re.split(r"\d", base, maxsplit=1)[0].strip(" ,.;:")
        token = _engine.resolve_token(prefix) if prefix else None
    if token == _engine.OXFORD_TOKEN and not oxford:
        # ``MS heb`` alone (a series word or the engine's fallback needle) does not name Oxford
        named = "°" not in base and _engine.resolve_token(_MS_HEB.sub(" ", text)) == _engine.OXFORD_TOKEN
        return token if named else None
    return token


def canonical_id(mark: str, institution: Optional[str] = None) -> Optional[str]:
    """The ES canonical id of a shelf mark: ``<institution token>_<core>``, one id per leaf.

    Same recipe as the primary ES index (``institution_tokens.canonical_id(institution, mark)``) with a facade
    pre-pass: a colon between two digits is read as ``.`` (:func:`_class_colon`: ``TS 12:301`` ->
    ``Cambridge_CUL_T_S_12_301``), a glued ``T-S12.301`` gets its boundary back, the legacy series spellings are folded
    (:func:`_series_fold`), a spaced institution head is stripped (:func:`_head`) and one terminal side marker is
    removed (:func:`side_suffix`). The token comes from *institution* when one is given (blank, or a spelling whose
    token is ``Unknown…`` such as ``Unknown Library``, counts as none); otherwise it is inferred
    (:func:`_infer_token`).

    Returns ``None`` (never an ``Unknown_…`` id) for a join string (call :func:`split` first), a bare NLI system
    number, garbage (:func:`_is_garbage`: a placeholder always, a digit-free mark naming nothing only when no
    institution is given), an empty core, and a mark whose institution cannot be determined (e.g. a bare
    ``Or. 5566``). Trailing join punctuation (``ENA NS 1.1 ;``) is trimmed first.

    One id per leaf has a known cost (R3, migration plan section 4): three ES ids are keyed on a single side
    (``Cambridge_CUL_T_S_H3_104r``, ``Cambridge_CUL_T_S_NS_J401r``, the Cairo ENL ``…_inv_no_404r``), so their raw
    marks now give the side-less ids, which the ES does not hold; ``resolve()`` (PR-6) falls back to the side-ful
    form until the ES re-key. ``T-S NS 225.25r`` / ``…25v`` give ``Cambridge_CUL_T_S_NS_225_25``, which the ES
    also holds (a Tier-1 duplicate). The ES ids themselves come back unchanged (id-input rule).

    The id-input rule: an ES id passed in is returned unchanged when *institution* is not given or
    ``institution_token(institution)`` is the id's own token (a pre-fix ``Oxford_Bodleian_MS_heb_…`` id comes
    back re-parsed to the current id). The engine does not always round-trip a matching institution: it
    re-normalises the core, so an id with a ``__sys`` suffix (306 registry-token ``__sys`` ids on 2026-11-03,
    e.g. ``London_BL_Add_27002__sys990000659290205171``) would lose an underscore. Only a conflicting
    institution is left to the engine, so it is not silently ignored
    (``canonical_id("Cambridge_CUL_T_S_12_301", "Jewish Theological Seminary")`` gives the engine's
    ``New_York_JTS_Cambridge_CUL_T_S_12_301``). The bare-token id ``Cambridge_Lewis_Gibson`` (empty core) is an
    ES id too and comes back unchanged. An ES id that ends in a side (``Cambridge_CUL_T_S_NS_225_25r``, one of
    11) is the ES's own key and also comes back unchanged. A slug id (an ES id without a registry token whose
    holder is named by at least two words, :func:`_is_slug_id`: 251 of the 256 such ids on 2026-11-03) comes back
    unchanged when no institution is given, except an ``Unknown_…`` id, which names no institution and gives
    ``None`` (R7; :func:`loose_key` still keys it). So ``canonical_id`` is idempotent on its own outputs, except a
    one-word-holder slug output (``Krupp_2188``, ``Mittwoch_8``), which is read like a raw mark. A bare core
    written as an id (``T_S_12_301``, ``A_2``, ``12_301``) is never echoed: it is read like a raw mark.

    The merge keeps its own per-source recipes (``merge_shelfmarks``).

    :param mark: Raw shelf mark or ES id.
    :type mark: str
    :param institution: Institution / collection spelling of any source (PGP ``library_abbrev``, a
        legacy-store collection, a KTIV holding library).
    :type institution: Optional[str]
    :return: The ES id, or ``None``.
    :rtype: Optional[str]
    """
    inst = institution if institution and institution.strip() else None
    if inst is not None and _institution_token(inst).split("_")[0].lower() == "unknown":
        inst = None  # ``Unknown Library`` names no institution (R7): infer one, as for a blank institution
    if _id_input_core(mark) is not None:
        own = _engine.split_id(mark)[0]
        if inst is None or _institution_token(inst) == own:
            return _reparse_old_oxford(mark) or mark
        return _engine.canonical_id(institution, mark)
    if inst is None and _is_slug_id(mark):
        return None if mark.startswith("Unknown_") else mark
    mark = _class_colon(mark)
    if _is_join(mark) or _is_nli_system_number(mark) or _is_garbage(mark, inst):
        return None
    text = _series_fold(_GLUED_TS.sub(r"\1 ", mark.strip().rstrip(_JOIN_TAIL)))
    head, rest, spaced = _head(text)
    base = side_suffix(rest if spaced else text)[0]
    core = _core(base)
    if not core:
        return None
    token = _institution_token(inst) if inst is not None else _infer_token(text, base)
    return _engine.combine(token, core) if token else None


def loose_key(mark_or_id: str, institution: Optional[str] = None) -> str:
    """The one fuzzy join key: institution-qualified lower-case alphanumerics.

    ``ShelfmarkNormalizer.loose_key(canonical_id(mark_or_id, institution))``: ``|`` marks a boundary
    between two digit runs, so ``T-S AS 181.163`` and ``T-S AS 18.1163`` stay apart. A raw mark (institution
    given or inferred), a side of it and its ES id give the same key (``T-S 12.301 recto`` =
    ``Cambridge_CUL_T_S_12_301`` -> ``cambridgecults12|301``). Same format as the decontamination ``clean_v2``
    keys.

    Every ES id keeps a non-empty key of its own: an ``Unknown_…`` id, for which :func:`canonical_id` gives
    ``None``, is keyed on itself (``Unknown_DK_123`` -> ``unknowndk123``, apart from ``Budapest_MTA_DK_123``).
    Only garbage ids (``Unknown_N_A``), the bare NLI system number id and the one-word-holder slug id
    ``Mittwoch_8`` (:func:`_is_slug_id`) give ``""``.

    :param mark_or_id: Raw shelf mark or ES id.
    :type mark_or_id: str
    :param institution: Institution spelling (see :func:`canonical_id`).
    :type institution: Optional[str]
    :return: The key, or ``""`` when :func:`canonical_id` gives ``None`` and the input is not an id.
    :rtype: str
    """
    cid = canonical_id(mark_or_id, institution)
    if cid is None and _is_slug_id(mark_or_id):
        cid = mark_or_id
    return ShelfmarkNormalizer.loose_key(cid) if cid else ""


def core_key(mark: str) -> str:
    """Institution-free loose key: ``ShelfmarkNormalizer.loose_key(core_id(mark))``.

    HAZARD: this key collides across institutions (175 key groups / 402 ES ids, e.g.
    ``Birmingham_Mingana_12`` / ``Budapest_MTA_12`` / ``Cincinnati_HUC_12`` all give ``12``). Use it only
    as a resolver fallback when the match is unique; never persist it as a tracking key.

    :param mark: Raw shelf mark or ES id.
    :type mark: str
    :return: The key (``""`` when :func:`core_id` is ``""``).
    :rtype: str
    """
    return ShelfmarkNormalizer.loose_key(core_id(mark))


# ------------------------------------------------------------------------------------ sides, joins, variants

def side_suffix(mark: str) -> Tuple[str, Optional[str]]:
    """Strip ONE terminal side marker: ``(base, side)``.

    The side is normalised to ``r`` / ``v`` (``recto`` / ``verso`` / ``ע"א`` / ``ע"ב`` / a ``r`` or ``v`` glued
    to the last number) or to ``<folio><r|v>`` (``T-S 10J5.17 1r``, ``T-S 10J5.17 1 r`` and
    ``T-S 10J5.17, fol. 1r`` -> ``1r``; ``T-S 10J5.17, 1v`` -> ``1v``). The folio form is taken only after a
    dotted number (a box-and-item mark). Folio 1 is the item's own leaf, so it leaves the base
    (``T-S 10J5.17, 1v`` -> ``("T-S 10J5.17", "1v")``, the ES id ``Cambridge_CUL_T_S_10J5_17``); any other folio
    number is a child leaf of the item, so only the side letter is stripped and the number stays in the base
    (``T-S Misc. 28.79, f. 14r`` -> ``("T-S Misc. 28.79, f. 14", "14r")``, the ES id
    ``Cambridge_CUL_T_S_Misc_28_79_14``, as ``T-S Misc. 28.79.14`` gives). ``ENA 2727, 5r`` gives
    ``("ENA 2727, 5", "r")``
    and ``MS Heb. d 66, fol. 9r`` gives ``("MS Heb. d 66, fol. 9", "r")``, because a Bodleian folio is a leaf of
    its own (``Oxford_Bodleian_Bodl_MS_heb_c_13_25``). Never stripped: a folio range (``…9r-10r``, ``…_9r_10r``),
    a piece letter (``12a``), an upper-case or roman letter (``Plate IV``, ``F 1908.44A``), a join string
    (:func:`_is_join`: ``T-S 16.22v / T-S 10J5.17r``) and a side whose base would end in a join, a join and a
    number, or another side (``T-S 12.301 recto and verso``, ``… recto + verso``, ``T-S 12.301 f. 1r and 2v``).

    A colon between two digits is read as ``.`` first (:func:`_class_colon`), so ``T-S 12:301 1r`` gives
    ``("T-S 12.301", "1r")``; a mark with no side comes back unchanged.

    :param mark: Raw shelf mark.
    :type mark: str
    :return: ``(base, side)``; ``(mark, None)`` when the mark ends in no side marker.
    :rtype: Tuple[str, Optional[str]]
    """
    text = _class_colon((mark or "").rstrip())
    if _is_join(text):
        return mark, None
    for pattern in (_SIDE_WORD, _SIDE_HEBREW):
        m = pattern.match(text)
        if m:
            base = m.group("base").rstrip(" ,")
            return (mark, None) if _JOINED_BASE.search(base) else (base, _SIDE_WORDS[m.group("side").lower()])
    if _SIDE_RANGE.search(text):
        return mark, None
    for pattern in (_SIDE_FOLIO, _SIDE_LETTER, _SIDE_GLUED):
        m = pattern.match(text)
        if m:
            base = m.group("base").rstrip(" ,")
            if _JOINED_BASE.search(base):
                return mark, None
            folio = m.groupdict().get("folio") or ""
            if folio and int(folio) != 1:
                base = text[:m.end("folio")]  # a child leaf: strip the side letter only
            return base, folio + m.group("side")
    return mark, None


def _merge_split(mark: str) -> List[str]:
    """The merge's legacy-store join splitter (``merge_shelfmarks.split_shelfmarks``), digit-free parts re-glued.

    The merge splits on every collection anchor, so a mark that starts with an institution word next to its
    series (``CAJS Halper 472``, ``Kaufmann DK 231.1``) comes back as ``["CAJS", "Halper 472"]``. A part with
    no digit is not a shelf mark: it is glued back onto the part that follows it (or onto the previous one when
    it is last).

    :param mark: Raw mark.
    :type mark: str
    :return: The parts (one part for a single mark).
    :rtype: List[str]
    """
    # Lazy import: the merge module pulls the image-correction and source loaders, which callers of the facade
    # do not need.
    from src.datasets.merging.merge_shelfmarks import split_shelfmarks

    out: List[str] = []
    pending = ""
    for part in split_shelfmarks(mark):
        part = part.strip()
        if not any(ch.isdigit() for ch in part):
            pending = f"{pending} {part}".strip()
            continue
        out.append(f"{pending} {part}".strip() if pending else part)
        pending = ""
    if pending:
        if out:
            out[-1] = f"{out[-1]} {pending}"
        else:
            out.append(pending)
    return out


def split(mark: str) -> List[str]:
    """Split a join string into single marks.

    The union of the two existing splitters: ``merge_shelfmarks.split_shelfmarks`` (legacy-store ``Inst:``
    prefixes carried forward, `` - `` joins) and, on each of its parts, :func:`split_joins` (``+``,
    ``;``, ``and``, ``&``, prefix inheritance for a bare second number, piece lists such as ``8J5.2 a, b, c``).
    Trailing join punctuation is trimmed (``T-S 16.22 +`` -> ``T-S 16.22``). A part with no digit is not a
    mark: a merge part such as the ``CAJS`` of ``CAJS Halper 472`` is glued back (:func:`_merge_split`), and a
    :func:`split_joins` result with a digit-free piece (the ``and`` of ``The National and University Library of
    Strasbourg, … Ms. 4038``) is not taken as a join. A single mark returns ``[mark]`` unchanged; a blank
    string returns ``[]``.

    :param mark: Raw mark, possibly a join.
    :type mark: str
    :return: The single marks.
    :rtype: List[str]
    """
    if not (mark or "").strip():
        return []
    parts: List[str] = []
    for part in _merge_split(mark):
        pieces = [p.strip().rstrip(_JOIN_TAIL).strip() for p in split_joins(part)]
        pieces = [p for p in pieces if p]
        if any(not any(ch.isdigit() for ch in p) for p in pieces):
            # an "and" inside an institution name (``The National and University Library of …``) is not a join
            pieces = [part.strip().rstrip(_JOIN_TAIL).strip()]
        parts.extend(pieces)
    return parts if len(parts) > 1 else [mark]


def _is_join(mark: str) -> bool:
    """Whether *mark* joins several marks (R1 / R10), not just lists the pieces of one (R12, kept as one id).

    :param mark: Raw mark.
    :type mark: str
    :return: True when :func:`split` gives more than one part and the parts are not merely the piece list of
        the mark (``T-S 8J5.2 a, b, c``).
    :rtype: bool
    """
    parts = split(mark)
    return len(parts) > 1 and parts != split_piece_lists(re.sub(r"\s+", " ", mark.strip()))


def variants(mark: str, lossy: bool = True) -> List[str]:
    """Legacy or printed spellings of a mark, rewritten into forms the normaliser accepts; the input first.

    In order: the input; the series folds (:func:`_series_fold`, ``BnF Hebr.`` -> ``hebr.`` included) of the
    input with a colon between two digits read as ``.`` (:func:`_class_colon`);
    :func:`legacy_variants` of the folded mark (and of the input when the folds changed it; its
    last-resort candidates come last within its list); the :func:`prep_shelfmark` cleaning; the safe
    spelling rewrites of the unseen-data key cleaner (:data:`_SAFE_SPELLINGS`); ``_clean_sm``
    (old-number parentheses, glued note digits) and :func:`alt_classmark` (a second classmark printed
    in parentheses). Duplicates and empty strings are dropped. Candidates are not checked against the ES:
    :func:`resolve` does that.

    With ``lossy=False`` (what :func:`resolve` uses) the lossy candidates are left out: the lossy last-resort
    variants of :func:`legacy_variants` and ``_clean_sm`` (a trailing note number dropped), each of which can name
    another leaf (``T-S AS 18.1163`` -> ``T-S AS 18.11``, ``TS 12 301`` -> ``TS 12``).

    :param mark: Raw mark (one part; see :func:`split`).
    :type mark: str
    :param lossy: Include the lossy candidates (default True: the PR-2 behaviour).
    :type lossy: bool
    :return: Distinct candidate spellings, the input first.
    :rtype: List[str]
    """
    folded = _series_fold(_class_colon(mark), bnf=True)
    candidates: List[Optional[str]] = [mark, folded]
    candidates += legacy_variants(folded, lossy=lossy)[1:]
    if folded != mark:
        candidates += legacy_variants(mark, lossy=lossy)[1:]
    safe = mark
    for pattern, repl in _SAFE_SPELLINGS:
        safe = pattern.sub(repl, safe)
    candidates += [prep_shelfmark(mark), re.sub(r"\s+", " ", safe).strip(" ,;."),
                   _clean_sm(mark) if lossy else None,
                   alt_classmark(mark)]
    out: List[str] = []
    for cand in candidates:
        if cand and cand.strip() and cand not in out:
            out.append(cand)
    return out


def parent(mark: str, ids: Optional[AbstractSet[str]] = None, institution: Optional[str] = None) -> Optional[str]:
    """The multi-fragment parent of a mark: its ES id with the last numeric component dropped, if that id exists.

    ``T-S Misc.28.79.14`` -> ``Cambridge_CUL_T_S_Misc_28_79`` when *ids* holds it. The parent is returned only
    when it is in *ids*: with ``ids=None`` the result is always ``None`` (``resolve()``, PR-6, supplies the ES id
    set). A numeric range (``T-S 10J5.17-18``) has no parent: its ``-18`` is a range end, not a child number.

    :param mark: Raw shelf mark or ES id.
    :type mark: str
    :param ids: The ES ids the parent must be one of (e.g. the merged corpus' canonical ids).
    :type ids: Optional[AbstractSet[str]]
    :param institution: Institution spelling (see :func:`canonical_id`).
    :type institution: Optional[str]
    :return: The parent ES id, or ``None``.
    :rtype: Optional[str]
    """
    if ids is None or _is_range(mark):
        return None
    cid = canonical_id(mark, institution)
    if cid is None:
        return None
    token, core = _engine.split_id(cid)
    parts = core.split("_")
    if token is None or len(parts) < 2 or not parts[-1].isdigit():
        return None
    candidate = _engine.combine(token, "_".join(parts[:-1]))
    return candidate if candidate in ids else None


# ------------------------------------------------------------------------------------ printed-mark helpers
# Moved verbatim from ``finetuning/text27b/lit_segment.py`` in PR-6 (``_SIDE_WORDS`` renamed
# ``_LIT_SIDE_WORDS``: the facade's own ``_SIDE_WORDS`` maps side words to sides). ``lit_segment``
# re-imports them; :func:`split` and :func:`variants` call them.

_LANG_PAREN = re.compile(r"\s*\((?:[^()]*\b(?:Arabic|Hebrew|Aramaic|Persian|Greek|Ladino|Cat\.|\*)[^()]*|\*)\)\s*")


_LIT_SIDE_WORDS = re.compile(r"\s*\((?:and\s+)?(?:recto|verso|not listed)[^()]*\)|\s+(?:recto|verso)\b\.?", re.I)


_NOS_LIST = re.compile(r"^(?P<base>.*?\d[a-z]?)\s*,?\s*nos\.?\s*(?P<list>\d+[a-z]?(?:\s*,\s*\d+[a-z]?)+)[\s,]*"
                       r"(?P<tail>\(\*\))?\s*$")
_LETTER_LIST = re.compile(r"^(?P<base>.*\d)\s*(?P<list>[a-z](?:\s*,\s*[a-z])+)[\s,]*(?P<tail>\(\*\))?\s*$")


def split_piece_lists(part: str) -> List[str]:
    """Split one shelfmark that lists several numbered or lettered pieces into one shelfmark per piece.

    ``TS NS J 185, nos. 5, 6, 2`` → ``TS NS J 185, no. 5`` / ``no. 6`` / ``no. 2``; ``TS NS J 67 a, b (*)`` →
    ``TS NS J 67a`` / ``TS NS J 67b``. A list of side letters only (``r, v``) is left alone.

    :param part: One shelfmark (one part of a join).
    :returns: The pieces (the part alone when it lists no pieces).
    """
    m = _NOS_LIST.match(part)
    if m:
        return [f"{m.group('base').rstrip(' ,')}, no. {n}" for n in re.findall(r"\d+[a-z]?", m.group("list"))]
    m = _LETTER_LIST.match(part)
    if m:
        lets = re.findall(r"[a-z]", m.group("list"))
        if not set(lets) <= {"r", "v"}:
            return [f"{m.group('base')}{x}" for x in lets]
    return [part]


def split_joins(raw: str) -> List[str]:
    """Split a printed join (``A and B``, ``A + B``, ``A; B``) into its shelfmarks; a bare second number inherits
    the first mark's collection prefix (``TS NS J 110 and 152`` → ``TS NS J 110``, ``TS NS J 152``; ``..., f. 141 and
    f. 10`` → ``..., f. 141``, ``..., f. 10``). A part that lists several pieces (``nos. 5, 6, 2``, ``67 a, b``) is
    split into one shelfmark per piece (:func:`split_piece_lists`).

    :param raw: Printed shelfmark.
    :returns: Parts (the raw string alone when it is not a join and lists no pieces).
    """
    s = re.sub(r"\s+", " ", (raw or "").strip())
    bits = [b.strip(" ,;") for b in re.split(r"\s+(?:and|\+|&)\s+|\s*;\s*", s) if b.strip(" ,;")]
    if len(bits) < 2:
        return split_piece_lists(s) if s else []
    out = [bits[0]]
    for b in bits[1:]:
        if re.match(r"^(?:ff?\.|fols?\.|nos?\.)\s*\d", b):
            base = re.sub(r",?\s*(?:ff?\.|fols?\.|nos?\.)\s*[\d\w ,]+$", "", bits[0]).strip(" ,")
            out.append(f"{base}, {b}")
        elif re.fullmatch(r"\d+[a-z]?", b):
            base = re.sub(r"\d+[a-z]?\s*$", "", bits[0]).rstrip()
            out.append(f"{base} {b}" if base else b)
        else:
            out.append(b)
    return [p for b in out for p in split_piece_lists(b)]


def legacy_variants(raw: str, neubauer: Optional[Mapping[str, str]] = None, lossy: bool = True) -> List[str]:
    """Rewrite an old-style shelfmark into the spellings the corpus normaliser matches.

    Covers the forms of 1920s–1970s scholarship: ``TS`` for ``T-S``, ``Box`` volumes, ``f.`` / ``fol.`` / ``no.`` /
    ``no`` folio markers (the first of a list of folios / numbers / part letters), ``TS 8 J 34`` for ``T-S 8J34``,
    ``TS Arabic (Box)`` for ``T-S Ar.``, a part letter in parentheses or after a space (``J 318(a)``, ``J 67 a``),
    a volume in parentheses (``Ar. Box 18 (1).33`` → ``Ar.18(1).33``), ``ULC Or Box 5, f. 13`` for
    ``CUL Or.1080 5.13``, ``Westminster College, Misc. 34`` / ``West. Coll., Frag. Cairens., f. 34`` for
    ``L-G Misc. 34`` (and ``Arabica I, f. N`` for ``L-G Ar. I.N``), ``Bodl. MS Heb. c 28, fol. 33`` /
    ``Bodleian MS Heb. d 66.5`` for ``Bodl. MS Heb. c 28/33`` / ``d 66/5``, Neubauer catalogue numbers
    (``Oxford (Neubauer) 2878/4``) when a crosswalk to MS Heb. volumes is supplied, side words (``Verso``,
    ``(and Verso)``) and a language / ``(Cat. N)`` / ``(*)`` parenthesis after the mark. Last-resort variants (marked
    by their position in the list): a footnote number glued to the folio (``37.1141`` → ``37.11``; ``15.60v74`` →
    ``15.60``), the folio without a part letter, a Dropsie / Penn ``E 16.510`` as ``UPenn E 16510`` and
    ``JTS Krengel 5.123`` as ``JTS Krengel 123``.

    :param raw: Shelfmark as printed (one part of a join; see :func:`split_joins`).
    :param neubauer: Optional Neubauer catalogue number -> ``MS Heb. <letter> <number>`` crosswalk.
    :param lossy: Also add the lossy last-resort variants (glued footnote number, side letter + note and part letter
        dropped; each can name another leaf, e.g. ``T-S AS 18.1163`` -> ``T-S AS 18.11``). Default True (the
        behaviour before PR-6); :func:`resolve` passes False. The Penn / Krengel rewrites are always added.
    :returns: Distinct candidate spellings, the raw (stripped) form first.
    """
    s0 = re.sub(r"\s+", " ", (raw or "").strip())
    out: List[str] = [s0] if s0 else []
    s = _LANG_PAREN.sub(" ", s0)
    s = _LIT_SIDE_WORDS.sub(" ", s).strip(" ,;")
    s = re.sub(r"(?<=\d)\s*\(([a-z])\)", r"\1", s)
    s = re.sub(r"(?<=\d)\s*\((\d)\)(?=\s*[.,])", r"(\1)", s)
    s, n_or = re.subn(r"^(?:ULC|CUL|T\.?-?\s?S\.?)\s+Or\.?\s*(?:1080\s*)?Box\s+(\d+)", r"CUL Or.1080 \1", s)
    s = re.sub(r"^(?:Westminster\s+College|West\.?\s*Coll\.?)\s*(?:\(Westminster College\))?\s*,?\s*"
               r"(?:Misc\.?|Frag\.?\s*Cairens\.?)\s*,?\s*(?:(?:ff?\.|fols?\.)\s*)?(\d+)", r"L-G Misc. \1", s)
    s = re.sub(r"^(?:Westminster\s+College|West\.?\s*Coll\.?)\s*(?:\(Westminster College\))?\s*,?\s*"
               r"(?:Ar\.?|Arabica)\s*(?:\(Arabica\)\s*)?(I{1,2})\s*,?\s*(?:ff?\.|fols?\.)\s*(\d+)", r"L-G Ar. \1.\2", s)
    if not n_or:
        s = re.sub(r"^(?:ULC|CUL)\s+(?=Or)", "T-S ", s)
    s = re.sub(r"^T\.?\s?S\.?\b\.?", "T-S", s)
    s = re.sub(r"^T-S\s+Arabic\s*(?:Box\s*)?", "T-S Ar.", s)
    s = re.sub(r"^T-S\s+(NS\s+J|NS|Misc\.?|AS|K|Box\s+K)\s*Box\s+", lambda m: f"T-S {m.group(1)} ", s)
    s = re.sub(r"^T-S\s+Box\s+", "T-S ", s)
    s = re.sub(r"^T-S\s+Misc\w*\.?\s+(?:Box\s+)?", "T-S Misc.", s)
    s = re.sub(r"^T-S\s+(\d+)\s*J\s*(\d+)", r"T-S \1J\2", s)
    s = re.sub(r"^T-S\s+NS\s+J\s*(\d+)", r"T-S NS J\1", s)
    s = re.sub(r"^T-S\s+K\s+(\d+)", r"T-S K\1", s)
    s = re.sub(r"^Bodleian\b\.?", "Bodl.", s)
    s = re.sub(r"(MS\.?\s+Heb\.?\s+[a-f])\.?\s*(\d+)\s*[.,]\s*(?:ff?\.|fols?\.)?\s*(\d+[a-z]?)", r"\1 \2/\3", s)
    m = re.match(r"^(?:Oxford|Bodl\.?)\s*(?:\(Neubauer\)|Neubauer)?\s*(?:Cat\.?\s*)?(\d{4})\s*/\s*(\d+[a-z]?)", s)
    if m and neubauer and m.group(1) in neubauer:
        s = f"Bodl. {neubauer[m.group(1)]}/{m.group(2)}"
    s = re.sub(r",?\s*(?:ff?\.|fols?\.)\s*(\d+)\s*([a-d])(?:\s*,\s*[a-d])*\b", r".\1\2", s)
    s = re.sub(r",?\s*(?:ff?\.|fols?\.)\s*(\d+[a-zA-Z]?)(?:\s*[-–]\s*\d+)?", r".\1", s)
    s = re.sub(r",?\s*\bnos?(?:\.\s*|\s+)(\d+)(?:\s*,\s*\d+)*,?", r".\1", s)
    s = re.sub(r"^(ENA\s+\d+[A-Za-z]?)\s*\.\s*(\d+)([A-Z])\b",
               lambda mm: f"{mm.group(1)}.{mm.group(2)}{mm.group(3).lower()}", s)
    s = re.sub(r"(?<=\d)\s+([a-z])(?:\s*,\s*[a-z])*\s*$", r"\1", s)
    s = re.sub(r"\s+\.", ".", s).strip(" ,;.")
    if s and s not in out:
        out.append(s)
    # last resort: glued footnote numbers, side letter + note, part letter, Penn E numbers, Krengel box numbers
    g = re.match(r"^(.*\.\d{1,3})(\d{2})$", s)
    if lossy and g and len(re.sub(r"\D", "", s.split(".")[-1])) >= 4:
        out.append(g.group(1))
    g = re.match(r"^(.*\.\d{1,3})[rv]\d{0,3}$", s)
    if lossy and g:
        out.append(g.group(1))
    g = re.match(r"^(.*\.\d{1,3})[a-d]$", s)
    if lossy and g and g.group(1) not in out:
        out.append(g.group(1))
    g = re.match(r"^E\s?(\d{1,2})\.(\d{3})$", s)
    if g:
        out.append(f"UPenn E {g.group(1)}{g.group(2)}")
    g = re.match(r"^(JTS\s+Krengel)\s+\d{1,2}\.(\d{1,3})$", s)
    if g:
        out.append(f"{g.group(1)} {g.group(2)}")
    return out


def _clean_sm(sm: str) -> str:
    """Shelfmark of an anthology line without a ``(formerly ...)`` / old-number parenthesis or glued note digits.

    :param sm: Text before the language parenthesis.
    :returns: Cleaned shelfmark.
    """
    sm = re.sub(r"(?<=\d)\s*\((\d)\)(?=\.\d)", r"(\1)", sm)
    sm = re.sub(r"\s*\((?!\d\)\.)(?:formerly|olim|old|=)?[^()]*\)\s*[\d¹²³⁴-⁹]*", "", sm).strip()
    sm = re.sub(r"[¹²³⁴-⁹]+$", "", sm)
    sm = re.sub(r"\s+\d{1,3}$", "", sm) if re.search(r"\d[\d.]*\s+\d{1,3}$", sm) else sm
    return sm.strip(" ·,;")


def alt_classmark(sm: str) -> Optional[str]:
    """A second classmark printed in parentheses after a shelfmark (``Mosseri L 129.1 (II 127.1)``,
    ``Mosseri L 291 (Ia, 21)``, ``X (formerly Y)``), completed with the mark's collection word when it starts with a
    volume number (``Mosseri II 127.1``).

    :param sm: Shelfmark text as printed (before :func:`_clean_sm`).
    :returns: The alternate classmark, or None.
    """
    for m in re.finditer(r"\(([^()]{1,40})\)", sm or ""):
        inner = m.group(1).strip()
        if re.fullmatch(r"\d|[a-z]|\*", inner) or not re.search(r"\d", inner):
            continue
        if re.search(r"Arabic|Hebrew|Aramaic|Persian|Greek|Ladino|Cat\.", inner):
            continue
        inner = re.sub(r"^(?:formerly|olim|old|now|=)\s*:?\s*", "", inner).strip(" ,;")
        if re.match(r"^(?:[IVX]+[a-z]?|\d)", inner):
            first = (sm.split() or [""])[0]
            inner = f"{first} {inner}" if first and not first[0].isdigit() else inner
        return inner
    return None


# Moved verbatim from ``finetuning/text27b/shelfmarks.py`` in PR-6 (that module re-imports it).
def prep_shelfmark(raw: str) -> str:
    """Clean a raw shelfmark key before normalising (underscore keys, stray prefixes, folio / side tails).

    :param raw: Raw key (from a page's ``shelf_marks_mentioned``, a PGP record, prose).
    :returns: Cleaned shelfmark string.
    """
    s = str(raw).strip()
    s = re.sub(r"^(?:Loc|Ref|Book|Doc_ref|Ms|MS_ref)_(?=[A-Z])", "", s)
    s = re.sub(r"^MS\.?\s+(?=ENA|T-S|TS\b|JTS)", "", s)
    s = re.sub(r"^(?:[uxv](?=_?[A-Z]{2}|_?T_S)|_+)", "", s).lstrip("_")
    if "_" in s and " " not in s.strip("_ "):
        s = s.replace("_", " ")
    s = re.sub(r"\s*\([^()]*\)?\s*$", "", s)
    # Folio designators become ", f. N" -- but neither a Bodleian size letter ("MS heb. f 56/12")
    # nor the Taylor-Schechter F series ("T-S F99.22"): the 2026-10-18 prototype matched "f"
    # case-insensitively, which dropped those letters and broke their keys.
    s = re.sub(r"(?<!heb\.)(?<!heb)(?<!Heb\.)(?<!Heb)(?<!arab\.)(?<!Arab\.)(?<!arab)(?<!Arab)"
               r"[\s,]+(?:f|fol|fols|folio|ff|Fol|Fols|Folio|FOL)\.?\s*(?=\d)", ", f. ", s)
    s = re.sub(r"(?<=\d)\s?[rv]$", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .,;:()[]{}")
    return s


# ------------------------------------------------------------------------------------ frozen legacy keys

def legacy_gate_key(mark: str) -> str:
    """FROZEN 2026-11: the pre-PR-4 decontamination-gate key; new code must use :func:`loose_key`.

    ``ShelfmarkNormalizer.loose_key`` of ``ShelfmarkNormalizer._strip_institution(mark)`` (of *mark* itself when
    the stripped core is empty): institution-free, with no facade pre-pass. It is the base of
    ``build_pgp_editions.fragment_key``, the key format of the sealed benchmark's exclusion inventory, which must
    not drift (the PR-4 membership gate found that the facade key would change that inventory's membership).
    Do not use it for anything else.

    :param mark: Raw shelf mark, ES id or loose key.
    :type mark: str
    :return: The key (``""`` for an empty mark).
    :rtype: str
    """
    if not mark:
        return ""
    return ShelfmarkNormalizer.loose_key(ShelfmarkNormalizer._strip_institution(mark) or mark)


# ------------------------------------------------------------------------------------ re-exports

def institution_token(institution: Optional[str]) -> str:
    """The registry token of an institution spelling, else a slug of it (thin re-export of the engine).

    ``institution_tokens.institution_token``: the registry token whose needle occurs in *institution*
    (``John Rylands Library, University of Manchester`` -> ``Manchester_JRL``), else an underscore slug of the text
    (``Unknown`` for ``None`` / empty), never empty. Unlike the token :func:`canonical_id` takes from its
    ``institution`` argument, a series word written as an institution (``Westminster College``) is not mapped to
    its series' token here.

    :param institution: Institution / collection spelling of any source.
    :type institution: Optional[str]
    :return: The token.
    :rtype: str
    """
    return _engine.institution_token(institution)


def parse_oxford(mark: Optional[str]) -> Optional[OxfordShelfmark]:
    """The engine's parse of a Bodleian shelf mark (thin re-export of :meth:`ShelfmarkNormalizer.parse_oxford`).

    :param mark: Raw mark (``MS Heb. d 66, fol. 9r``, ``Bodl. MS heb. c 13/25``).
    :type mark: Optional[str]
    :return: The engine's parsed Bodleian mark (``letter``, ``volume``, ``leaf``, ``core`` …), or ``None`` when
        *mark* is not a Bodleian shelf mark.
    :rtype: Optional[OxfordShelfmark]
    """
    return ShelfmarkNormalizer.parse_oxford(mark)


def classify_shelfmark(mark: str) -> str:
    """Validator class of a string (thin re-export of :meth:`ShelfmarkNormalizer.classify_shelfmark`).

    :param mark: Candidate string.
    :type mark: str
    :return: ``"standard"``, ``"berlin"``, ``"bibliographic"`` (a Davidson *Thesaurus* citation) or ``"reject"``.
    :rtype: str
    """
    return ShelfmarkNormalizer.classify_shelfmark(mark)


def is_shelfmark(mark: str) -> bool:
    """Whether a string is a manuscript shelf mark (validator for parsed / LLM-emitted marks).

    Wraps :meth:`ShelfmarkNormalizer.classify_shelfmark`: ``"standard"`` and ``"berlin"`` are shelf
    marks; ``"bibliographic"`` (a Davidson *Thesaurus* citation) and ``"reject"`` are not.

    :param mark: Candidate string.
    :type mark: str
    :return: True for a shelf mark.
    :rtype: bool
    """
    return ShelfmarkNormalizer.classify_shelfmark(mark) in ("standard", "berlin")


def institution_info(mark: str) -> Dict[str, Optional[str]]:
    """Full-name institution taxonomy of a shelf mark (the ES ``institution`` / ``collection`` fields).

    Re-export of :meth:`ShelfmarkNormalizer.get_institution_info`. This taxonomy is separate from the
    registry tokens of :func:`canonical_id` (open question Q12).

    :param mark: Raw shelf mark.
    :type mark: str
    :return: The engine's institution dictionary.
    :rtype: Dict[str, Optional[str]]
    """
    return ShelfmarkNormalizer.get_institution_info(mark)


#: CUDL IIIF manifest base (copied verbatim from the Arabic scrape queue with :func:`cudl_manifest`).
CUDL_BASE = "https://cudl.lib.cam.ac.uk/iiif/"
# Cambridge shelfmark prefixes -> CUDL manifest id prefix (learned from the
# export rows that carry both a shelfmark and a CUDL manifest URL).
_CUDL_PREFIXES = [("T-S ", "MS-TS"), ("CUL Or.", "MS-OR"), ("CUL Add.", "MS-ADD"),
                  ("Moss. ", "MS-MOSSERI")]
#: The PGP export the default fragment list of :func:`cudl_manifest` is read from (read-only, on first use).
PGP_EXPORT = Path(__file__).resolve().parent / "raw_data/cairo_genizah/pgp_raw/data"


@functools.lru_cache(maxsize=None)
def _fragment_shelfmarks(fragments_csv: Path) -> FrozenSet[str]:
    """Shelfmarks listed in one ``fragments.csv``, read once per path.

    :param fragments_csv: PGP export ``fragments.csv``.
    :type fragments_csv: Path
    :return: Shelfmarks (empty when the file does not exist).
    :rtype: FrozenSet[str]
    """
    if not fragments_csv.exists():
        return frozenset()
    with open(fragments_csv, encoding="utf-8-sig") as fh:
        return frozenset(r["shelfmark"] for r in csv.DictReader(fh))


def known_fragments(pgp_dir: Optional[Path] = None) -> FrozenSet[str]:
    """Shelfmarks of the physical fragments of a PGP export, read on first use (not at import).

    :param pgp_dir: PGP export directory holding ``fragments.csv``; default :data:`PGP_EXPORT`.
    :type pgp_dir: Optional[Path]
    :return: Fragment shelfmarks (empty when the export has no ``fragments.csv``).
    :rtype: FrozenSet[str]
    """
    return _fragment_shelfmarks((pgp_dir if pgp_dir is not None else PGP_EXPORT) / "fragments.csv")


def cudl_manifest(shelfmark: str, known: Optional[AbstractSet[str]] = None) -> Optional[str]:
    """Construct the CUDL IIIF manifest URL for a single Cambridge shelfmark.

    Owned by the facade since PR-6: a verbatim copy of ``indexing/arabic_scrape_queue.cudl_manifest`` (with
    :data:`CUDL_BASE`, the prefix table and :func:`known_fragments`). The queue keeps its own copy, untouched, while
    that file carries unrelated pending work; the copy is removed (the queue then re-exports this one) when that
    work lands. ``tests/test_shelfmarks_resolve.py`` pins that both give identical URLs on the probe set.

    Tokens after the collection prefix become ``-``-joined id parts: numbers
    zero-padded to five digits, letters upper-cased — e.g. ``T-S 13J1.10`` ->
    ``MS-TS-00013-J-00001-00010``, ``T-S Ar.18(1).135`` ->
    ``MS-TS-AR-00018-00001-00135``, ``CUL Or.1080 J10`` -> ``MS-OR-01080-J-00010``.

    NOT an identity key: a document-level sub-number of a known fragment is dropped, so a sub-numbered child
    returns its **parent's** manifest (``T-S Misc.28.79.14`` -> ``MS-TS-MISC-00028-00079``).

    :param shelfmark: Single (non-joined) shelfmark.
    :type shelfmark: str
    :param known: Fragment shelfmarks of the PGP export in use (:func:`known_fragments`);
        default: those of the export under :data:`PGP_EXPORT`, read only when needed.
    :type known: Optional[AbstractSet[str]]
    :return: Manifest URL, or ``None`` for non-Cambridge / joined shelfmarks.
    :rtype: Optional[str]
    """
    if " + " in shelfmark:
        return None
    # Manifests are per physical fragment: a document-level sub-number
    # (``T-S 13J4.15.1`` is the first document on fragment ``T-S 13J4.15``)
    # is dropped when the base shelfmark is a known fragment.
    base = re.sub(r"\.[0-9]+$", "", shelfmark)
    if base != shelfmark and base in (known if known is not None else known_fragments()):
        shelfmark = base
    for prefix, ms in _CUDL_PREFIXES:
        if shelfmark.startswith(prefix):
            rest = shelfmark[len(prefix):]
            tokens = re.findall(r"[A-Za-z]+|[0-9]+", rest)
            if not tokens:
                return None
            parts = [t.zfill(5) if t.isdigit() else t.upper() for t in tokens]
            return CUDL_BASE + "-".join([ms] + parts)
    return None


# ------------------------------------------------------------------------------------ resolver (PR-6)

#: The merged corpus the default :class:`ShelfmarkIndex` is built from (read-only; never written, never a cache).
MERGED_SHELFMARKS = Path(__file__).resolve().parent / "raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
#: Layout / build-rule version of a pickled :class:`ShelfmarkIndex` (part of the cache signature with
#: :data:`NORMALISER_VERSION`).
INDEX_FORMAT = 4
#: Every ``how`` of a :class:`Resolution`, strongest first; a join takes the ``how`` of its weakest part.
HOW_ORDER: Tuple[str, ...] = ("exact", "alias", "loose", "core", "parent", "pgpid", "unmatched")
_PGPID_PREFIX = "pgpid:"
#: A bare page / folio / line reference (``p. 23``, ``f. 12r``, ``fols. 3-4``), never a shelf mark on its own (the
#: whole string: ``F 1908.44BB`` is a Freer mark).
_PAGE_REF = re.compile(r"^\s*(?:pp?|ff?|fols?|folios?|pages?|ll?|lines?|nos?)\.?\s*\d+[rvab]?"
                       r"(?:\s*[-\u2013]\s*\d+[rvab]?)?\s*$", re.IGNORECASE)
#: A number, an optional side / piece letter, a dash and a number (candidate numeric range, see :func:`_is_range`).
_RANGE = re.compile(r"(?<!\d)(\d+)([^\W\d_]?)\s*[-\N{EN DASH}\N{EM DASH}]\s*(\d+)([^\W\d_]?)")
#: An ES id that ends in a numbered child component (``Cambridge_CUL_T_S_Misc_28_79_14``), group 1 = the parent id.
_CHILD_ID = re.compile(r"^(.+)_\d+[^\W\d_]?$")


def _is_range(mark: str) -> bool:
    """Whether a raw mark names a numeric range (``T-S 10J5.17-18``, ``T-S 12.6-7``, ``ff. 9-10``, ``1v-2r``).

    What a range names is open (migration plan Q11). A range is ``<n1>-<n2>`` (hyphen, en or em dash) where the
    end is larger than the start (``17-18``), or equal with a side / piece letter (``21r-21v``, ``2-2a``), or a
    one- or two-digit abbreviation of a larger end (``161-63``, ``6416-18``). A zero-padded end is a number
    component, not a range end: CUDL ids (``MS-TS-00012-00301``) and box numbers (``FR 9-004/20``) are no ranges,
    nor is an accession number (``AM 2005-125``).

    :param mark: Raw mark (one part of :func:`split`).
    :type mark: str
    :return: True when the mark holds a numeric range.
    :rtype: bool
    """
    for m in _RANGE.finditer(mark or ""):
        start, start_letter, end, end_letter = m.groups()
        if len(end) > 1 and end.startswith("0"):
            continue
        if len(end) < len(start) and len(end) <= 2:
            end = start[:len(start) - len(end)] + end
        if int(end) > int(start) or (int(end) == int(start) and (start_letter or end_letter)):
            return True
    return False


class Resolution(NamedTuple):
    """The ES ids a shelf mark, an ES id or a ``pgpid:<n>`` resolves to, and how.

    :param cids: ES ids (sorted; every one is in the index), empty when unmatched.
    :param how: One of :data:`HOW_ORDER`.
    :param ambiguous: True when an ``alias`` / ``loose`` / ``core`` step of a single mark (of any part of a join)
        matched more than one ES id: the ids are an over-match (several leaves, or a leaf and its duplicate record),
        not one leaf. ``exact``, ``parent`` and ``pgpid`` results are never ambiguous (a PGP id naming several
        records is a join, not an ambiguity).
    :param note: ``"range"`` when the mark (or a part of a join) is a numeric range (``T-S 12.6-7``): migration
        plan Q11 is open, so a range is ``unmatched`` (never its first leaf or a parent) unless the ES holds a
        record of that very range (``exact``). ``""`` otherwise.
    """

    cids: List[str]
    how: str
    ambiguous: bool = False
    note: str = ""


def record_marks(record: Mapping[str, Any]) -> List[str]:
    """Every single-mark spelling a merged record is known by, its canonical id first.

    Lifted from ``merging/legacy_store_inventory.record_marks`` and made source-generic: the canonical id, the
    display form, the PGP fragment mark and its historic marks (also those of ``_extra_fragments``), the
    ``shelf_mark`` of every other source (list-valued sources: each entry's), the KTIV catalogue mark, and the
    ``aliases`` / ``source_marks`` lists of a slim record (the test fixture's form). A join string (more than one
    part under :func:`split`) is skipped: its partners have records of their own.

    :param record: One ``merged_shelfmarks.jsonl`` line (full or slim form).
    :type record: Mapping[str, Any]
    :return: Distinct non-empty spellings.
    :rtype: List[str]
    """
    marks: List[str] = [record.get("canonical_id") or "", record.get("shelfmark_display") or ""]
    marks += list(record.get("source_marks") or []) + list(record.get("aliases") or [])
    sources = record.get("sources") or {}
    pgp = sources.get("pgp") or {}
    for fragment in [pgp.get("fragment") or {}] + list(pgp.get("_extra_fragments") or []):
        marks.append(fragment.get("shelfmark") or "")
        marks += (fragment.get("shelfmarks_historic") or "").split(";")
    for name, block in sources.items():
        if name == "pgp":
            continue
        for entry in (block if isinstance(block, list) else [block]):
            if not isinstance(entry, dict):
                continue
            marks.append(entry.get("shelf_mark") or "")
            listed = entry.get("shelfmarks")
            if isinstance(listed, dict) and isinstance(listed.get("shelfmark"), dict):
                marks.append(listed["shelfmark"].get("value") or "")
    out: List[str] = []
    for i, mark in enumerate(m.strip() for m in marks if isinstance(m, str)):
        if mark and mark not in out and (i == 0 or len(split(mark)) == 1):
            out.append(mark)
    return out


#: An upper-case Roman numeral token (``AIU VII C``, ``Reinach III``, ``Paris_AIU_VII_C``).
_ROMAN = re.compile(r"(?<![A-Za-z])[IVX]+(?![a-z])(?=$|[^A-Za-z]|[A-Z](?![A-Za-z]))")


def _has_digit(text: str) -> bool:
    """Whether a spelling, id or key carries a number: a digit, or (in a spelling or id) an upper-case Roman
    numeral token. A shelf mark names a numbered item; ``BL``, ``Bodl.``, ``JRL`` or ``AIU`` do not.

    :param text: Spelling, id or key.
    :type text: str
    :return: True when *text* has a digit or a Roman numeral token.
    :rtype: bool
    """
    return any(ch.isdigit() for ch in text) or bool(_ROMAN.search(text))


#: A siglum-like token: an all-capitals word of 2+ letters (``PER``, ``BL``), a dotted abbreviation (``Heid.``,
#: ``inv.``) or letters glued to digits (``1900b``, ``44BB``, ``F3``); a prose word (``Genizah``, ``Letter``) is none.
_SIGLUM = re.compile(r"(?<![^\W\d_])(?:[A-Z]{2,}(?![a-z])|[^\W\d_]+\.|\d+[^\W\d_]+|[^\W\d_]+\d+)")
#: Dotted prose / reference abbreviations that are no siglum (``Fig. 12``, ``Vol. 2``, ``Doc. 32``, ``p. 23``).
_PROSE_ABBREVIATIONS = frozenset({"art", "cf", "ch", "chap", "doc", "docs", "ed", "eds", "f", "ff", "fig", "figs",
                                  "fol", "fols", "ibid", "l", "ll", "n", "nn", "no", "nos", "p", "pl", "pp", "pt",
                                  "sec", "tab", "vol", "vols"})


def _has_siglum(spelling: str) -> bool:
    """Whether a spelling carries a siglum-like token (:data:`_SIGLUM`) other than a prose abbreviation.

    :param spelling: The spelling.
    :type spelling: str
    :return: True for ``PER H 34``, ``p. Heid. Hebr. 17``, ``Lévi 1900b``; False for ``Genizah 5``, ``Fig. 12``.
    :rtype: bool
    """
    for m in _SIGLUM.finditer(spelling):
        token = m.group(0)
        if not (token.endswith(".") and token[:-1].lower() in _PROSE_ABBREVIATIONS):
            return True
    return False


def _core_key_usable(key: str, spelling: str) -> bool:
    """Whether an institution-free :func:`core_key` is strong enough for the unique-core step of :func:`resolve`.

    The spelling must carry a number (:func:`_has_digit`) and either pass :func:`is_shelfmark` (a series or holder
    word) or name an institution (:func:`canonical_id` is not ``None``); or the key has at least 4 characters with
    a letter and a digit (the text27b ``match_shelfmark`` junk rule), the spelling is not a bare page / folio /
    line reference, and it has at least two digit groups or a siglum-like token (:func:`_has_siglum`: ``PER``,
    ``Heid.``, ``1900b``). ``5``, ``152``, ``a5``, ``p. 23``, ``f. 12r`` and prose-like mentions (``Genizah 5``,
    ``Letter 12``, ``Fig. 12``) are not usable; ``F 1908.44BB`` (Freer), ``PER H 34`` and ``p. Heid. Hebr. 17``
    are.

    :param key: The core key.
    :type key: str
    :param spelling: The spelling the key came from.
    :type spelling: str
    :return: True when the core step may use the key.
    :rtype: bool
    """
    if not key or not _has_digit(spelling):
        return False
    if is_shelfmark(spelling) or canonical_id(spelling) is not None:
        return True
    if not (len(key) >= 4 and any(ch.isalpha() for ch in key) and any(ch.isdigit() for ch in key)
            and not _PAGE_REF.match(spelling)):
        return False
    return len(re.findall(r"\d+", spelling)) >= 2 or _has_siglum(spelling)


#: One segment of a bare-number core: digits with at most one glued piece letter (``5``, ``132``, ``5a``) or an
#: upper-case Roman numeral (``VII``). A core made only of such segments names no collection.
_BARE_NUMBER_SEGMENT = re.compile(r"\d+[^\W\d_]?|[IVX]+")


def _is_bare_number_core(spelling: str) -> bool:
    """Whether the institution-free core of a spelling is only a number: every segment of its case-kept
    :func:`core_id` is digits with at most one piece letter, or an upper-case Roman numeral
    (:data:`_BARE_NUMBER_SEGMENT`).

    ``JTS 5`` (``5``), ``CUL 132`` (``132``), ``JTS 12.301`` (``12_301``), ``JTS 5a`` and ``JTS VII`` are bare; ``ENA
    5`` (``ENA_5``), ``Mingana 5`` (``Mingana_5``), ``CUL VII C 5`` and ``Sassoon 526`` are not: their core keeps a
    series or holder word.

    :param spelling: The spelling the core key came from.
    :type spelling: str
    :return: True when the core has at least one segment and all of them are bare numbers.
    :rtype: bool
    """
    segments = [seg for seg in core_id(spelling).split("_") if seg]
    return bool(segments) and all(_BARE_NUMBER_SEGMENT.fullmatch(seg) for seg in segments)


def _named_tokens(spelling: str, institution: Optional[str]) -> Set[str]:
    """The registry tokens a spelling names: the token of its own :func:`canonical_id` (the institution inferred from
    the mark: ``JTS 5`` -> ``New_York_JTS``) and, when an institution is given with it, that institution's registry
    token (``resolve_token`` or a series word; a slug of an unregistered holder name counts as none).

    :param spelling: The spelling.
    :type spelling: str
    :param institution: Institution spelling given with the mark, or ``None``.
    :type institution: Optional[str]
    :return: The tokens (empty when neither names a registry institution).
    :rtype: Set[str]
    """
    tokens: Set[str] = set()
    cid = canonical_id(spelling)
    if cid is not None and _engine.split_id(cid)[0]:
        tokens.add(_engine.split_id(cid)[0])
    if institution and institution.strip():
        given = _engine.resolve_token(institution) or _engine.SERIES_TOKENS.get(institution.strip())
        if given:
            tokens.add(given)
    return tokens


def _core_hits_of_named_institution(ids: AbstractSet[str], spelling: str, institution: Optional[str]) -> Set[str]:
    """Drop the unique-core hits of a bare-number core (:func:`_is_bare_number_core`) whose institution token is not
    one the spelling names (:func:`_named_tokens`).

    A bare number carries no collection: ``JTS 5``, ``CUL 5`` and ``CUL 132`` give the core keys ``5`` / ``132``,
    which can be unique in the index to ``Birmingham_Mingana_5`` or ``Budapest_MTA_132``; the mark names another
    institution, so the hit is a coincidence of numbers, not the leaf. A core that keeps a series or holder word
    (``Sassoon 526`` -> ``Various_Sassoon_526``, a legacy-store ``New York JTS: T-S AS 62.262`` ->
    ``Cambridge_CUL_T_S_AS_62_262``) is left alone: its word carries the collection. A spelling that names no
    registry institution is left alone too.

    :param ids: The ids of :meth:`ShelfmarkIndex.core_unique` for the spelling's core key.
    :type ids: AbstractSet[str]
    :param spelling: The spelling the core key came from.
    :type spelling: str
    :param institution: Institution spelling given with the mark, or ``None``.
    :type institution: Optional[str]
    :return: The ids kept (all of *ids* unless the core is a bare number and the spelling names an institution).
    :rtype: Set[str]
    """
    if not ids or not _is_bare_number_core(spelling):
        return set(ids)
    named = _named_tokens(spelling, institution)
    if not named:
        return set(ids)
    return {cid for cid in ids if _engine.split_id(cid)[0] in named}


def _narrow(ids: AbstractSet[str], spelling: str, how: str) -> "Resolution":
    """A multi-id ``alias`` / ``loose`` / ``core`` hit narrowed to the one id whose core equals the spelling's
    case-exact :func:`core_id` (``p. Heid. Hebr. 17`` -> ``Heidelberg_p_Heid_Hebr_17``, not its case-variant
    duplicate ``Heidelberg_P_Heid_Hebr_17``); otherwise every id, ``ambiguous`` when there are several.

    :param ids: The ids the step matched.
    :type ids: AbstractSet[str]
    :param spelling: The spelling resolved.
    :type spelling: str
    :param how: The step.
    :type how: str
    :return: The resolution.
    :rtype: Resolution
    """
    if len(ids) > 1:
        core = core_id(spelling)
        same = [cid for cid in ids if core and _engine.split_id(cid)[1] == core]
        if len(same) == 1:
            return Resolution(same, how)
    return Resolution(sorted(ids), how, len(ids) > 1)


def _verbatim_key(spelling: str) -> str:
    """The key of a spelling in the ``verbatim`` table of :class:`ShelfmarkIndex`: :func:`prep_shelfmark`, white
    space collapsed, case-folded.

    :param spelling: A record's own spelling, or a raw part.
    :type spelling: str
    :return: The key (``""`` for an empty spelling).
    :rtype: str
    """
    return re.sub(r"\s+", " ", prep_shelfmark(spelling)).strip().casefold()


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Records of a JSONL file, one per non-blank line.

    :param path: JSONL file.
    :type path: Path
    :return: Iterator of records.
    :rtype: Iterator[Dict[str, Any]]
    """
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


class ShelfmarkIndex:
    """The ES ids of the merged corpus and the lookups :func:`resolve` walks.

    Lifted from ``text27b.shelfmarks`` (``Corpus`` / ``_load_merged`` / ``compact_index``) and
    ``legacy_store_inventory.record_marks`` (:func:`record_marks`). Tables:

    * ``ids``: every ``canonical_id``;
    * ``alias``: alias id -> ES ids. An alias id is the record's own registry token + :func:`core_id` of another
      spelling of the record (PGP historic marks, source marks, the display form; side stripped by
      :func:`side_suffix`) when that differs from the record's id. A spelling whose core has no number
      (:func:`_has_digit`: ``BL``, ``Bodl.``, ``JRL``) gives no alias id and no key: it names no item;
    * ``loose``: :func:`loose_key` of each id and alias id -> ES ids (ids and spellings with a number only);
    * ``core``: :func:`core_key` of each id and spelling -> ES ids (institution-free, ids and spellings with a
      number only;
      :func:`resolve` uses a core key only when its ids carry one institution token, see :meth:`core_unique`, and
      the key is strong enough, see :func:`_core_key_usable`);
    * ``by_pgpid``: PGP id -> ES ids, and ``pgpids``: ES id -> PGP ids; from the records' ``pgpids`` and from the
      optional ``fragments`` rows: a fragment mark that resolves to ONE id by an ``exact`` / ``alias`` / ``loose``
      step links that id to the row's PGP ids (merged records miss the PGP ids of many multi-document fragments);
    * ``fragment_marks``: PGP id -> fragment shelf marks (optional ``fragments`` rows, e.g. PGP ``fragments.csv``
      or ``text27b`` ``Corpus.fragments``), resolved like raw marks for a ``pgpid:<n>`` that still has no id;
    * ``verbatim``: :func:`_verbatim_key` of each spelling of a record (:func:`record_marks`, with a number) -> ES
      ids, plus the historic marks (``historic`` / ``shelfmarks_historic``) of a ``fragments`` row whose mark
      resolves to one id by an ``exact`` / ``alias`` / ``loose`` step. A spelling other than the display form is
      left out when its institution-free :func:`core_key` names another record (:meth:`_add_verbatim`).
      :func:`resolve` uses the table as a last resort (``alias``) when it names exactly one id: a record's own
      display form reaches the record even when its id has no registry token (``Bar-Ilan University Library, Ramat
      Gan, Israel Ms. 1039``), and a historic mark reaches its fragment although the core step may refuse it as a
      weak mention.

    The parent lookup is :func:`parent` with ``ids``; :meth:`numbered_parents` (ids with numbered child ids) gates
    the parent fallback of :func:`resolve`.

    :param source: Path of a ``merged_shelfmarks.jsonl`` file, or an iterable of records (full or slim form:
        ``canonical_id``, ``shelfmark_display``, ``institution``, ``pgpids``, ``aliases``, ``source_marks``).
    :type source: Union[str, os.PathLike, Iterable[Mapping[str, Any]]]
    :param cache_dir: Directory for the pickled index when *source* is a path; keyed by
        :data:`NORMALISER_VERSION`, :data:`INDEX_FORMAT` and the file's size and modification time. Never under
        ``raw_data`` (``ValueError``). ``None``: no cache.
    :type cache_dir: Optional[Union[str, os.PathLike]]
    :param fragments: Rows with ``shelfmark`` and ``pgpids`` (a list, or a ``;`` / ``,`` separated string).
    :type fragments: Optional[Iterable[Mapping[str, Any]]]
    """

    def __init__(self, source: Union[str, "os.PathLike[str]", Iterable[Mapping[str, Any]]], *,
                 cache_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
                 fragments: Optional[Iterable[Mapping[str, Any]]] = None) -> None:
        """Build the index, or load it from the cache.

        :param source: See the class docstring.
        :type source: Union[str, os.PathLike, Iterable[Mapping[str, Any]]]
        :param cache_dir: See the class docstring.
        :type cache_dir: Optional[Union[str, os.PathLike]]
        :param fragments: See the class docstring.
        :type fragments: Optional[Iterable[Mapping[str, Any]]]
        :return: None.
        :rtype: None
        """
        self.ids: Set[str] = set()
        self.alias: Dict[str, Set[str]] = collections.defaultdict(set)
        self.loose: Dict[str, Set[str]] = collections.defaultdict(set)
        self.core: Dict[str, Set[str]] = collections.defaultdict(set)
        self.by_pgpid: Dict[str, Set[str]] = collections.defaultdict(set)
        self.pgpids: Dict[str, List[str]] = {}
        self.fragment_marks: Dict[str, List[str]] = collections.defaultdict(list)
        self.verbatim: Dict[str, Set[str]] = collections.defaultdict(set)
        self._numbered_parents: Optional[Set[str]] = None
        if isinstance(source, (str, os.PathLike)):
            path = Path(source)
            cache = self._cache_file(path, cache_dir) if cache_dir is not None else None
            sig = (NORMALISER_VERSION, INDEX_FORMAT, path.stat().st_size, path.stat().st_mtime_ns)
            if cache is not None and cache.exists():
                with open(cache, "rb") as fh:
                    blob = pickle.load(fh)
                if blob.get("sig") == sig:
                    self.__dict__.update(blob["tables"])
                    self._add_fragments(fragments)
                    return
            self._build(_read_jsonl(path))
            if cache is not None:
                self._write_cache(cache, sig)
        else:
            self._build(source)
        self._add_fragments(fragments)

    @staticmethod
    def _cache_file(path: Path, cache_dir: Union[str, "os.PathLike[str]"]) -> Path:
        """The pickle file of the index of *path* under *cache_dir* (never under ``raw_data``).

        :param path: Merged corpus file.
        :type path: Path
        :param cache_dir: Caller-given cache directory.
        :type cache_dir: Union[str, os.PathLike]
        :return: The cache file path (named by the file's stem and a hash of its full path, so two corpora with one
            stem do not share a pickle).
        :rtype: Path
        :raises ValueError: When *cache_dir* lies under a ``raw_data`` directory.
        """
        directory = Path(cache_dir).resolve()
        if "raw_data" in directory.parts:
            raise ValueError(f"the shelf-mark index cache must not live under raw_data: {directory}")
        tag = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
        return directory / f"shelfmark_index_{path.stem}_{tag}.pkl"

    def _write_cache(self, cache: Path, sig: Tuple[Any, ...]) -> None:
        """Pickle the tables atomically (temporary file + rename).

        :param cache: Cache file.
        :type cache: Path
        :param sig: Cache signature.
        :type sig: Tuple[Any, ...]
        :return: None.
        :rtype: None
        """
        cache.parent.mkdir(parents=True, exist_ok=True)
        tables = {k: v for k, v in self.__dict__.items() if k not in ("fragment_marks", "_numbered_parents")}
        fd, tmp = tempfile.mkstemp(prefix=".shelfmark_index.", suffix=".tmp", dir=str(cache.parent))
        with os.fdopen(fd, "wb") as fh:
            pickle.dump({"sig": sig, "tables": tables}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)

    def _build(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Fill the tables from merged records.

        :param records: Merged records (full or slim form).
        :type records: Iterable[Mapping[str, Any]]
        :return: None.
        :rtype: None
        """
        pending: List[Tuple[str, str, str]] = []
        for record in records:
            cid = record.get("canonical_id") or ""
            if not cid:
                continue
            self.ids.add(cid)
            pgpids = [str(p) for p in record.get("pgpids") or []]
            self.pgpids[cid] = pgpids
            for pgpid in pgpids:
                self.by_pgpid[pgpid].add(cid)
            token = _engine.split_id(cid)[0]
            # an id or spelling without a number (``London_BL``, ``Bodl.``) gives no key and no alias id
            loose = {loose_key(cid)} if _has_digit(cid) else set()
            cores = {core_key(cid)} if _has_digit(cid) else set()
            display = (record.get("shelfmark_display") or "").strip()
            for mark in record_marks(record)[1:]:
                if _has_digit(mark):
                    (self.verbatim[_verbatim_key(mark)].add(cid) if mark == display
                     else pending.append((_verbatim_key(mark), cid, core_key(mark))))
                core = core_id(side_suffix(mark)[0])
                if not core or not _has_digit(core):
                    continue
                cores.add(ShelfmarkNormalizer.loose_key(core))
                if token:
                    alias = _engine.combine(token, core)
                    loose.add(ShelfmarkNormalizer.loose_key(alias))
                    if alias != cid:
                        self.alias[alias].add(cid)
            for key in loose - {""}:
                self.loose[key].add(cid)
            for key in cores - {""}:
                self.core[key].add(cid)
        self._add_verbatim(pending)
        self.verbatim.pop("", None)
        for table in (self.alias, self.loose, self.core, self.by_pgpid, self.verbatim):
            table.default_factory = None

    def _add_verbatim(self, spellings: Iterable[Tuple[str, str, str]]) -> None:
        """Add a record's other spellings (historic and source marks; not its display form) to ``verbatim``, unless
        the spelling's institution-free :func:`core_key` names another record: ``L 124`` is the historic mark of a
        Mosseri fragment and also the core of ``Manchester_JRL_L_124``, so it names neither for sure.

        :param spellings: ``(verbatim key, ES id, core key)`` triples (the full ``core`` table must be built).
        :type spellings: Iterable[Tuple[str, str, str]]
        :return: None.
        :rtype: None
        """
        for key, cid, core in spellings:
            if key and not (self.core.get(core) or set()) - {cid}:
                self.verbatim.setdefault(key, set()).add(cid)

    def _add_fragments(self, fragments: Optional[Iterable[Mapping[str, Any]]]) -> None:
        """Record the fragment shelf marks of PGP ids, and link a fragment mark's PGP ids to the one ES id it names.

        A mark that resolves to exactly one id by an ``exact`` / ``alias`` / ``loose`` step (never ``core`` or
        ``parent``, never ambiguous) adds the row's PGP ids to ``pgpids`` / ``by_pgpid`` of that id, and the row's
        historic marks (``historic`` or ``shelfmarks_historic``, ``;`` separated) to ``verbatim`` (added after every
        row is resolved, so the order of the rows does not matter). Every mark is also kept in ``fragment_marks``
        for :func:`resolve` of a ``pgpid:<n>``.

        :param fragments: Rows with ``shelfmark``, ``pgpids`` and optional historic marks.
        :type fragments: Optional[Iterable[Mapping[str, Any]]]
        :return: None.
        :rtype: None
        """
        historic: List[Tuple[str, str, str]] = []
        for row in fragments or ():
            mark = (row.get("shelfmark") or "").strip()
            raw = row.get("pgpids") or []
            items = raw if isinstance(raw, (list, tuple, set)) else re.split(r"[;,]", str(raw))
            pgpids = [p for p in (str(x).strip() for x in items) if p]
            if not mark or not pgpids:
                continue
            for pgpid in pgpids:
                if mark not in self.fragment_marks[pgpid]:
                    self.fragment_marks[pgpid].append(mark)
            res = resolve(mark, index=self)
            if len(res.cids) != 1 or res.ambiguous or res.how not in ("exact", "alias", "loose"):
                continue
            cid = res.cids[0]
            known = self.pgpids.setdefault(cid, [])
            for pgpid in pgpids:
                if pgpid not in known:
                    known.append(pgpid)
                self.by_pgpid.setdefault(pgpid, set()).add(cid)
            for old in str(row.get("historic") or row.get("shelfmarks_historic") or "").split(";"):
                if _has_digit(old):
                    historic.append((_verbatim_key(old), cid, core_key(old)))
        self._add_verbatim(historic)

    def numbered_parents(self) -> Set[str]:
        """The ES ids that have numbered child ids in the index (``…_Misc_28_79`` with ``…_Misc_28_79_14``): the ids
        that are evidently multi-fragment parents (computed once, on first use).

        :return: The parent ids.
        :rtype: Set[str]
        """
        if self._numbered_parents is None:
            found = (_CHILD_ID.match(cid) for cid in self.ids)
            self._numbered_parents = {m.group(1) for m in found if m and m.group(1) in self.ids}
        return self._numbered_parents

    def core_unique(self, key: str) -> Set[str]:
        """The ES ids of a :func:`core_key` when they carry one institution token, else nothing.

        :param key: Institution-free core key.
        :type key: str
        :return: The ids (empty when the key is unknown or names several institutions).
        :rtype: Set[str]
        """
        ids = self.core.get(key) or set()
        tokens = {_engine.split_id(cid)[0] or cid for cid in ids}
        return set(ids) if len(tokens) == 1 else set()

    def institutions(self, key: str) -> int:
        """The number of institutions (registry tokens; an id without one counts as its own) behind a core key.

        :param key: Institution-free core key.
        :type key: str
        :return: The count (0 for an unknown key).
        :rtype: int
        """
        return len({_engine.split_id(cid)[0] or cid for cid in self.core.get(key) or ()})

    def __len__(self) -> int:
        """The number of ES ids.

        :return: ``len(ids)``.
        :rtype: int
        """
        return len(self.ids)


def _resolve_spelling(spelling: str, index: ShelfmarkIndex, institution: Optional[str]) -> Optional[Resolution]:
    """The exact -> alias -> loose -> unique-core chain for one spelling.

    The unique-core step never crosses institutions on a bare number: when the core is only a number (``JTS 5``,
    ``CUL 132``) and the spelling names an institution, a hit of another institution's token is dropped
    (:func:`_core_hits_of_named_institution`).

    :param spelling: One candidate spelling (a :func:`variants` entry).
    :type spelling: str
    :param index: The ES index.
    :type index: ShelfmarkIndex
    :param institution: Institution spelling given with the mark (see :func:`canonical_id`), or ``None``; when
        given, each step tries it first and then the institution inferred from the mark.
    :type institution: Optional[str]
    :return: The first step that matches (``ambiguous`` when an alias / loose / core step names several ids and
        :func:`_narrow` cannot pick the case-exact one), or ``None``. A spelling without a number (no digit, no
        Roman numeral token) matches only as an ES id itself: it names no item.
    :rtype: Optional[Resolution]
    """
    if spelling in index.ids:
        return Resolution([spelling], "exact")
    if not _has_digit(spelling):
        return None
    # the given institution first, then the one inferred from the mark (a holder name can disagree with the
    # token the ES id carries, e.g. ``Strasbourg 4038`` given with the library's full name)
    institutions = [institution, None] if institution else [None]
    cids = [cid for cid in (canonical_id(spelling, inst) for inst in institutions) if cid is not None]
    for cid in cids:
        if cid in index.ids:
            return Resolution([cid], "exact")
    for cid in cids:
        if cid in index.alias:
            return _narrow(index.alias[cid], spelling, "alias")
    for inst in institutions:
        key = loose_key(spelling, inst)
        if key and key in index.loose:
            return _narrow(index.loose[key], spelling, "loose")
    key = core_key(spelling)
    ids = index.core_unique(key) if _core_key_usable(key, spelling) else set()
    ids = _core_hits_of_named_institution(ids, spelling, institution)
    return _narrow(ids, spelling, "core") if ids else None


def _resolve_part(part: str, index: ShelfmarkIndex, institution: Optional[str]) -> Resolution:
    """Resolve one single mark (no join): per :func:`variants` spelling of its side-less base the exact -> alias ->
    loose -> unique-core chain, then a record's own spelling naming one id (``verbatim``, ``alias``), then the
    side-ful ES id (R3), then :func:`parent` when the parent is a multi-fragment parent
    (:func:`_is_fragment_parent`). The spellings are ``variants(base, lossy=False)``.

    :param part: One part of :func:`split`.
    :type part: str
    :param index: The ES index.
    :type index: ShelfmarkIndex
    :param institution: Institution spelling given with the mark, or ``None``.
    :type institution: Optional[str]
    :return: The resolution (``unmatched`` with no ids when nothing matches).
    :rtype: Resolution
    """
    if part.strip() in index.ids:
        return Resolution([part.strip()], "exact")
    if _is_range(part):
        # only the ES's own record of that very range (``ENA 1846.3-4`` -> ``New_York_JTS_ENA_1846_3_4``)
        for inst in ([institution, None] if institution else [None]):
            cid = canonical_id(part, inst)
            if cid in index.ids:
                return Resolution([cid], "exact", False, "range")
        return Resolution([], "unmatched", False, "range")
    base, side = side_suffix(part)
    spellings = variants(base, lossy=False)
    for spelling in spellings:
        hit = _resolve_spelling(spelling, index, institution)
        if hit is not None:
            return hit
    if _has_digit(part):
        # last resort: a record's own spelling (display form, historic / source mark) naming exactly one id
        own = index.verbatim.get(_verbatim_key(part)) or set()
        if len(own) == 1:
            return Resolution(sorted(own), "alias")
    if side is not None:
        cid = canonical_id(base, institution)
        token = _engine.split_id(cid)[0] if cid else None
        core = _core(_class_colon(part.strip()))
        sideful = _engine.combine(token, core) if token and core else None
        if sideful in index.ids:
            return Resolution([sideful], "exact")
    for spelling in spellings:
        up = parent(spelling, ids=index.ids, institution=institution)
        if up is not None and _is_fragment_parent(up, index):
            return Resolution([up], "parent")
    return Resolution([], "unmatched")


def _is_fragment_parent(cid: str, index: ShelfmarkIndex) -> bool:
    """Whether an ES id is a multi-fragment parent: its core keeps at least two segments with a digit AND the index
    holds numbered children of it (:meth:`ShelfmarkIndex.numbered_parents`).

    ``T-S Misc.28.79.14`` -> ``Cambridge_CUL_T_S_Misc_28_79`` qualifies when the ES also holds
    ``…_Misc_28_79_1`` … (a fragment whose documents are numbered under it). An ordinary two-segment leaf
    (``Cambridge_CUL_T_S_10J5_17``) with no numbered children does not, nor does a box or volume id
    (``Cambridge_CUL_T_S_NS_322`` for ``T-S NS 322.64``, ``Strasbourg_4038`` for ``… Ms. 4038.10``): the leaf is not
    part of a multi-fragment document there, and a box id would join every leaf of the box.

    :param cid: Candidate parent ES id.
    :type cid: str
    :param index: The ES index.
    :type index: ShelfmarkIndex
    :return: True when the parent fallback may use it.
    :rtype: bool
    """
    core = _engine.split_id(cid)[1]
    if sum(any(ch.isdigit() for ch in seg) for seg in core.split("_")) < 2:
        return False
    return cid in index.numbered_parents()


def resolve(x: str, *, index: ShelfmarkIndex, institution: Optional[str] = None) -> Resolution:
    """Resolve a raw shelf mark, an ES id or ``pgpid:<n>`` to the ES ids of *index*.

    Chain: :func:`split` -> per part :func:`side_suffix` -> per :func:`variants` spelling of the base (the input
    first): an ES id itself or :func:`canonical_id` in the index (``exact``) -> an alias id (``alias``) ->
    :func:`loose_key` (``loose``) -> :func:`core_key` when its ids carry one institution token (``core``; a
    bare-number core such as ``JTS 5`` never reaches another institution's record,
    :func:`_core_hits_of_named_institution`); when no spelling matches, a record's own spelling (display form, PGP
    historic mark; ``verbatim`` table of the index) when it names exactly one id (``alias``), then the side-ful ES
    id (``exact``: three ES ids keep a side, R3) and then :func:`parent` of the first spelling that has one in the
    index, when it is a multi-fragment parent (two or more numbered segments left and numbered children in the
    index, never a bare box or volume id; ``parent``). The spellings are ``variants(base, lossy=False)``: the lossy
    last-resort rewrites (a glued note number or a piece letter dropped) could name another leaf. ``pgpid:<n>``
    gives the records carrying that PGP id (``pgpid``), else the resolution of its fragment shelf marks
    (``fragments`` of the index; ``how`` = ``pgpid``).

    A part that is a numeric range (:func:`_is_range`: ``T-S 10J5.17-18``, ``T-S 12.6-7``, ``ff. 9-10``) gets
    ``note="range"`` and only the ES's own record of that very range (an ES id itself, or :func:`canonical_id` in
    the index: ``ENA 1846.3-4`` -> ``New_York_JTS_ENA_1846_3_4``, ``exact``); otherwise it is ``unmatched`` with
    ``ambiguous=False``. No variant, alias, loose, core or parent step runs: migration plan Q11 (one id per range,
    or one per leaf?) is open, so a range never collapses to its first leaf or to a parent.

    A join returns the union of its parts' ids with the ``how`` of its weakest part (:data:`HOW_ORDER`; one
    unmatched part makes the join ``unmatched``, with the other parts' ids). It never returns an id that is not in
    the index. Multi-id contract: an ``alias`` / ``loose`` / ``core`` step that names several ids returns the one
    whose core is the input's case-exact core when there is exactly one (:func:`_narrow`), else all of them with
    ``ambiguous=True`` (an over-match: several sys-number records of one manuscript, or a leaf and its duplicate
    record); a caller that needs one leaf must treat an ambiguous result as unresolved. A spelling without a number
    (``BL``, ``Bodl.``) matches only as an ES id itself, and the institution-free core step needs a strong key
    (:func:`_core_key_usable`), so garbage stays ``unmatched``.

    :param x: Raw mark, ES id or ``pgpid:<n>``.
    :type x: str
    :param index: The ES index (:class:`ShelfmarkIndex`).
    :type index: ShelfmarkIndex
    :param institution: Optional institution spelling given with the mark (PGP ``library``, a KTIV holding
        library, a catalogue's collection), used as by :func:`canonical_id` / :func:`loose_key`; ``None`` infers it
        from the mark. Ignored for ``pgpid:<n>``.
    :type institution: Optional[str]
    :return: The ids and how they were found.
    :rtype: Resolution
    """
    text = (x or "").strip()
    if not text:
        return Resolution([], "unmatched")
    if text.lower().startswith(_PGPID_PREFIX):
        pgpid = text[len(_PGPID_PREFIX):].strip()
        ids = set(index.by_pgpid.get(pgpid) or ())
        if not ids:
            for mark in index.fragment_marks.get(pgpid) or ():
                ids |= set(resolve(mark, index=index).cids)
        return Resolution(sorted(ids & index.ids), "pgpid" if ids else "unmatched")
    parts = [_resolve_part(part, index, institution) for part in split(text)]
    if not parts:
        return Resolution([], "unmatched")
    how = max((p.how for p in parts), key=HOW_ORDER.index)
    note = "range" if any(p.note == "range" for p in parts) else ""
    return Resolution(sorted({c for p in parts for c in p.cids} & index.ids), how, any(p.ambiguous for p in parts),
                      note)
