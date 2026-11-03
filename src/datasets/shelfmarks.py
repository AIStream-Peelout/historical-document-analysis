"""The single public module for shelf-mark normalisation (facade over the two engines).

Every module outside the facade, the two engines and ``merging/merge_shelfmarks.py`` imports shelf-mark helpers
from here, never from ``document_models/genizah_normalizer.py`` (:class:`ShelfmarkNormalizer`) or
``merging/institution_tokens.py`` directly. The facade is thin: it calls those engines (and the join / legacy
spelling helpers of ``merging/merge_shelfmarks.py``, ``finetuning/text27b/lit_segment.py`` and
``finetuning/text27b/shelfmarks.py``) and does not copy their code. The lint test
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

PR-2 behaviour (implemented; this version, :data:`NORMALISER_VERSION` ``2026.11.03-3``)
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

Planned in PR-6 (NOT implemented here)
    ``resolve(x, *, index=None)`` and ``ShelfmarkIndex``: the ES-backed resolver chain (exact id, merge aliases,
    :func:`loose_key`, unique :func:`core_key`, :func:`parent`). It supplies the ES id set that :func:`parent`
    takes as ``ids``; the ``lit_segment`` split / variant helpers and :func:`cudl_manifest` move into the facade
    then.

:data:`NORMALISER_VERSION` is bumped on any change to engine or facade output and belongs in every key
cache signature.
"""

from __future__ import annotations

import re
from typing import AbstractSet, Dict, List, Optional, Pattern, Tuple

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer
from src.datasets.merging import institution_tokens as _engine
from src.finetuning.text27b import lit_segment as _lit

#: Version of the normalisation output. Bump on any change to an engine's or this facade's output, and
#: include it in every cache keyed by shelf-mark keys.
NORMALISER_VERSION: str = "2026.11.03-3"

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
    join string (call :func:`split` first), a bare NLI system number and garbage (:func:`_is_garbage`).

    :param mark: Raw shelf mark or ES id.
    :type mark: str
    :return: The core, or ``""``.
    :rtype: str
    """
    if _id_input_core(mark) is None and (_is_join(mark) or _is_nli_system_number(mark) or _is_garbage(mark)):
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
    pre-pass: a glued ``T-S12.301`` gets its boundary back, the legacy series spellings are folded
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

    :param mark: Raw shelf mark.
    :type mark: str
    :return: ``(base, side)``; ``(mark, None)`` when the mark ends in no side marker.
    :rtype: Tuple[str, Optional[str]]
    """
    text = (mark or "").rstrip()
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
    prefixes carried forward, `` - `` joins) and, on each of its parts, ``lit_segment.split_joins`` (``+``,
    ``;``, ``and``, ``&``, prefix inheritance for a bare second number, piece lists such as ``8J5.2 a, b, c``).
    Trailing join punctuation is trimmed (``T-S 16.22 +`` -> ``T-S 16.22``). A part with no digit is not a
    mark: a merge part such as the ``CAJS`` of ``CAJS Halper 472`` is glued back (:func:`_merge_split`), and a
    ``split_joins`` result with a digit-free piece (the ``and`` of ``The National and University Library of
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
        pieces = [p.strip().rstrip(_JOIN_TAIL).strip() for p in _lit.split_joins(part)]
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
    return len(parts) > 1 and parts != _lit.split_piece_lists(re.sub(r"\s+", " ", mark.strip()))


def variants(mark: str) -> List[str]:
    """Legacy or printed spellings of a mark, rewritten into forms the normaliser accepts; the input first.

    In order: the input; the series folds (:func:`_series_fold`, ``BnF Hebr.`` -> ``hebr.`` included);
    ``lit_segment.legacy_variants`` of the folded mark (and of the input when the folds changed it; its
    last-resort candidates come last within its list); the text27b ``prep_shelfmark`` cleaning; the safe
    spelling rewrites of the unseen-data key cleaner (:data:`_SAFE_SPELLINGS`); ``lit_segment._clean_sm``
    (old-number parentheses, glued note digits) and ``lit_segment.alt_classmark`` (a second classmark printed
    in parentheses). Duplicates and empty strings are dropped. Candidates are not checked against the ES:
    ``resolve()`` (PR-6) does that.

    :param mark: Raw mark (one part; see :func:`split`).
    :type mark: str
    :return: Distinct candidate spellings, the input first.
    :rtype: List[str]
    """
    # Lazy import: the text27b key module loads its corpus paths and is itself a facade caller-to-be (PR-6).
    from src.finetuning.text27b.shelfmarks import prep_shelfmark

    folded = _series_fold(mark, bnf=True)
    candidates: List[Optional[str]] = [mark, folded]
    candidates += _lit.legacy_variants(folded)[1:]
    if folded != mark:
        candidates += _lit.legacy_variants(mark)[1:]
    safe = mark
    for pattern, repl in _SAFE_SPELLINGS:
        safe = pattern.sub(repl, safe)
    candidates += [prep_shelfmark(mark), re.sub(r"\s+", " ", safe).strip(" ,;."), _lit._clean_sm(mark),
                   _lit.alt_classmark(mark)]
    out: List[str] = []
    for cand in candidates:
        if cand and cand.strip() and cand not in out:
            out.append(cand)
    return out


def parent(mark: str, ids: Optional[AbstractSet[str]] = None, institution: Optional[str] = None) -> Optional[str]:
    """The multi-fragment parent of a mark: its ES id with the last numeric component dropped, if that id exists.

    ``T-S Misc.28.79.14`` -> ``Cambridge_CUL_T_S_Misc_28_79`` when *ids* holds it. The parent is returned only
    when it is in *ids*: with ``ids=None`` the result is always ``None`` (``resolve()``, PR-6, supplies the ES id
    set).

    :param mark: Raw shelf mark or ES id.
    :type mark: str
    :param ids: The ES ids the parent must be one of (e.g. the merged corpus' canonical ids).
    :type ids: Optional[AbstractSet[str]]
    :param institution: Institution spelling (see :func:`canonical_id`).
    :type institution: Optional[str]
    :return: The parent ES id, or ``None``.
    :rtype: Optional[str]
    """
    if ids is None:
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


# ------------------------------------------------------------------------------------ re-exports

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


def cudl_manifest(shelfmark: str, known: Optional[AbstractSet[str]] = None) -> Optional[str]:
    """Construct the CUDL IIIF manifest URL for a single Cambridge shelfmark (thin re-export).

    Calls ``src.datasets.indexing.arabic_scrape_queue.cudl_manifest`` unchanged; the implementation moves into
    the facade in PR-6. Tokens after the collection prefix become ``-``-joined id parts: numbers zero-padded to
    five digits, letters upper-cased, e.g. ``T-S 13J1.10`` -> ``MS-TS-00013-J-00001-00010``.

    NOT an identity key: a document-level sub-number of a known fragment is dropped, so a sub-numbered child
    returns its **parent's** manifest (``T-S Misc.28.79.14`` -> ``MS-TS-MISC-00028-00079``).

    :param shelfmark: Single (non-joined) PGP-style shelfmark.
    :type shelfmark: str
    :param known: Fragment shelfmarks of the PGP export in use; default: those of the PGP export the queue
        module reads, read only when needed.
    :type known: Optional[AbstractSet[str]]
    :return: Manifest URL, or ``None`` for non-Cambridge / joined shelfmarks.
    :rtype: Optional[str]
    """
    # Lazy import: the queue module is an indexing script (not loaded on every facade import) and will itself
    # import this facade once PR-4 migrates it.
    from src.datasets.indexing.arabic_scrape_queue import cudl_manifest as _queue_cudl_manifest

    return _queue_cudl_manifest(shelfmark, known)
