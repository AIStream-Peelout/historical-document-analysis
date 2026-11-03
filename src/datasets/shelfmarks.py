"""The single public module for shelf-mark normalisation (facade over the two engines).

Every module outside the facade, the two engines and ``merging/merge_shelfmarks.py`` imports shelf-mark helpers
from here, never from ``document_models/genizah_normalizer.py`` (:class:`ShelfmarkNormalizer`) or
``merging/institution_tokens.py`` directly. The facade is thin: it calls those engines and does not copy
their code. The lint test ``tests/test_shelfmark_single_source.py`` enforces this rule over ``src/``.

Tracking key
    The ES canonical id (:func:`canonical_id`, e.g. ``Cambridge_CUL_T_S_AS_18_170``) is the tracking key
    of every ledger, hold-out, manifest and registry. The only fuzzy join key is the institution-qualified
    :func:`loose_key`. The institution-free :func:`core_key` collides across institutions (175 key groups,
    402 ES ids, measured 2026-11-02: ``Berlin_JCB_A2`` vs ``Manchester_JRL_A_2``) and is never persisted.

PR-1 behaviour (this version, :data:`NORMALISER_VERSION` ``2026.11.03-1``)
    Every function reproduces the reference columns of the shelf-mark probe matrix byte for byte
    (``tests/fixtures/shelfmark_probes.json``). The only departure is the id-input rule: when the input
    already is an ES id (a registry token prefix followed by a core, recognised by :func:`split_id`, with
    no whitespace), :func:`core_id` returns its core and :func:`canonical_id` returns it unchanged when no
    institution is given or the institution's token is the id's own token; only a conflicting institution
    is left to the engine, so it is not silently ignored. It ships in PR-1 because the T3 round trip
    (``core_id(es_id) == split_id(es_id)[1]``, ``loose_key(mark) == loose_key(canonical_id(mark)) ==
    loose_key(es_id)``) needs it. The rule covers registry-token ids only: the 256 ES ids without a registry
    token (2026-11-03: slug-prefixed institutions such as ``Chabad_Lubavitch_Library_…``, ``Unknown_…`` and
    ``Unknown_Library__sys…`` ids, bare NLI numbers) go to the engine like any raw mark, as in the reference,
    so :func:`canonical_id` gives an ``Unknown_…`` id for 239 of them (``Unknown_<id>`` for 216).

Planned in PR-2 (NOT implemented here; documented so callers know what will change)
    * ``side_suffix`` and one id per leaf: ``T-S 12.301 r/recto/verso`` and ``AIU XII.87v`` give the
      side-less id plus a separate side.
    * Glued prefixes (``T-S12.301``) and pre-fix Oxford ids (``Oxford_Bodleian_MS_heb_…``) re-parsed.
    * Series folds in ``variants`` (``Taylor-Schechter``, ``T-S Arabic [Box]``, ``ULC Or``,
      ``Lewis-Gibson``, ``Westminster College``, ``BnF Hebr``).
    * ``None`` from :func:`canonical_id` (and ``""`` from :func:`core_id`) for garbage, join strings and
      bare NLI system numbers; ``split`` for joins.
    * Stripping a spaced KTIV head (``Cambridge University Library T-S …``).
    * Institution inference when ``institution`` is not given (KTIV / legacy-store head, then
      :data:`institution_tokens.SERIES_TOKENS`).

    ``split``, ``side_suffix``, ``parent``, ``variants`` (PR-2) and ``resolve`` (PR-6) are not part of
    this version.

:data:`NORMALISER_VERSION` is bumped on any change to engine or facade output and belongs in every key
cache signature.
"""

from __future__ import annotations

from typing import AbstractSet, Dict, Optional, Tuple

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer
from src.datasets.merging import institution_tokens as _engine

#: Version of the normalisation output. Bump on any change to an engine's or this facade's output, and
#: include it in every cache keyed by shelf-mark keys.
NORMALISER_VERSION: str = "2026.11.03-1"


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


def core_id(mark: str) -> str:
    """The institution-free core of a shelf mark, case preserved.

    :meth:`ShelfmarkNormalizer.to_canonical_id` of the mark (``T-S 12.301`` -> ``T_S_12_301``). An ES id
    passed in gives its own core (``Cambridge_CUL_T_S_12_301`` -> ``T_S_12_301``, the token stripped by
    :func:`split_id`); the bare-token id ``Cambridge_Lewis_Gibson`` gives ``""`` (see :func:`_id_input_core`).
    In PR-1 a join string, garbage or an NLI system number still yields the engine's core; returning ``""``
    for them is PR-2.

    :param mark: Raw shelf mark or ES id.
    :type mark: str
    :return: The core.
    :rtype: str
    """
    core = _id_input_core(mark)
    return core if core is not None else ShelfmarkNormalizer.to_canonical_id(mark)


def canonical_id(mark: str, institution: Optional[str] = None) -> Optional[str]:
    """The ES canonical id of a shelf mark: ``<institution token>_<core>``.

    Same recipe as the primary ES index (``institution_tokens.canonical_id(institution, mark)``). The
    id-input rule: an ES id passed in is returned unchanged when *institution* is ``None`` (the engine
    would give ``Unknown_<id>``) or when ``institution_token(institution)`` is the id's own token. The
    engine does not always round-trip a matching institution: it re-normalises the core, so an id with a
    ``__sys`` suffix (the 306 registry-token ``__sys`` ids on 2026-11-03, e.g.
    ``London_BL_Add_27002__sys990000659290205171``; 312 ``__sys`` ids in all, the other 6 are
    ``Unknown_Library__sys…`` ids with no registry token) would lose an underscore. Only a conflicting
    institution is left to the engine, so it is not silently ignored
    (``canonical_id("Cambridge_CUL_T_S_12_301", "Jewish Theological Seminary")`` gives the engine's
    ``New_York_JTS_Cambridge_CUL_T_S_12_301``). The bare-token id ``Cambridge_Lewis_Gibson`` (empty core)
    is an ES id too and comes back unchanged. In PR-1 the token otherwise comes only from *institution*
    (``Unknown`` when it is not given), and the result is never ``None``; institution inference and
    ``None`` for garbage / joins are PR-2. The merge keeps its own per-source recipes
    (``merge_shelfmarks``).

    The id-input rule covers registry-token ids only (recognised by :func:`split_id`). The 256 ES ids without
    a registry token (2026-11-03: slug-prefixed institutions such as ``Chabad_Lubavitch_Library_…``,
    ``Unknown_…`` ids, bare NLI numbers) go to the engine like a raw mark, as in the reference: for 239 of
    them ``canonical_id(es_id)`` gives an ``Unknown_…`` id (``Unknown_<id>`` for 216; the 23
    ``Unknown_Library__sys…``-style ones also lose an underscore), and the 17 ids already prefixed
    ``Unknown_`` come back unchanged.

    :param mark: Raw shelf mark or ES id.
    :type mark: str
    :param institution: Institution / collection spelling of any source (PGP ``library_abbrev``, a
        legacy-store collection, a KTIV holding library).
    :type institution: Optional[str]
    :return: The ES id.
    :rtype: Optional[str]
    """
    if _id_input_core(mark) is not None and (
            institution is None or _engine.institution_token(institution) == _engine.split_id(mark)[0]):
        return mark
    return _engine.canonical_id(institution, mark)


def loose_key(mark_or_id: str, institution: Optional[str] = None) -> str:
    """The one fuzzy join key: institution-qualified lower-case alphanumerics.

    ``ShelfmarkNormalizer.loose_key(canonical_id(mark_or_id, institution))``: ``|`` marks a boundary
    between two digit runs, so ``T-S AS 181.163`` and ``T-S AS 18.1163`` stay apart. A raw mark with its
    institution and its ES id give the same key (``Cambridge_CUL_T_S_12_301`` -> ``cambridgecults12|301``).
    Same format as the decontamination ``clean_v2`` keys.

    :param mark_or_id: Raw shelf mark or ES id.
    :type mark_or_id: str
    :param institution: Institution spelling (see :func:`canonical_id`).
    :type institution: Optional[str]
    :return: The key, or ``""`` when :func:`canonical_id` gives ``None``.
    :rtype: str
    """
    cid = canonical_id(mark_or_id, institution)
    return ShelfmarkNormalizer.loose_key(cid) if cid else ""


def core_key(mark: str) -> str:
    """Institution-free loose key: ``ShelfmarkNormalizer.loose_key(core_id(mark))``.

    HAZARD: this key collides across institutions (175 key groups / 402 ES ids, e.g.
    ``Birmingham_Mingana_12`` / ``Budapest_MTA_12`` / ``Cincinnati_HUC_12`` all give ``12``). Use it only
    as a resolver fallback when the match is unique; never persist it as a tracking key.

    :param mark: Raw shelf mark or ES id.
    :type mark: str
    :return: The key.
    :rtype: str
    """
    return ShelfmarkNormalizer.loose_key(core_id(mark))


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
