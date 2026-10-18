# File name: text_representation.py
# Date: 10/7/26 (semantic_v4 added 10/9/26)
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Versioned embedding text: the one function the indexers, the verifier and any training use.

Text versions (recorded per index in ``_meta.text_representation_version`` and per document in
``text_representation_version``):

* ``v8``: the production text that built the indices up to ``genizah_merged_v8`` and ``bibliography_text_only_0.7``,
  i.e. ``doc.create_text_representation()`` of :class:`GenizahDocument` / :class:`BibliographyDocument`, unchanged.
* ``semantic_v3``: the text the fine-tuned embedder was trained and evaluated on (index v9). Fragment records go
  through :func:`src.embeddings.semantic_text.semantic_text`; bibliography pages through
  :func:`src.embeddings.semantic_text.bibliography_semantic_text` (only the ``Source PDF:`` line is dropped).
* ``semantic_v4`` (index v10): fragment records get the layout :func:`src.embeddings.semantic_text.semantic_v4_text`
  over their ``semantic_v3`` text, and their eligibility is decided on that ``semantic_v3`` text
  (:func:`src.embeddings.semantic_text.semantic_v4_ineligible_reason`), so it cannot be re-derived from the embedded
  text: use :func:`embed_text_for_version` / :func:`embedding_text`, never :func:`is_text_eligible`. Bibliography
  pages keep the ``semantic_v3`` page text.

:data:`DEFAULT_TEXT_VERSION` stays ``semantic_v3`` until the commit that pins the v4-trained model.

Documents declare which variant applies through a class attribute ``embedding_text_kind`` (``"fragment"`` or
``"bibliography"``). Note that ``embedding_models.NomicsEmbedding.create_text_representation`` is the legacy ColQwen
text, not ``v8``.
"""

from typing import Any, NamedTuple

from src.embeddings.semantic_text import (
    bibliography_semantic_text,
    is_semantic_eligible,
    semantic_text,
    semantic_v4_ineligible_reason,
    semantic_v4_text,
    text_sha256,
)

TEXT_VERSION_V8 = "v8"
TEXT_VERSION_SEMANTIC_V3 = "semantic_v3"
TEXT_VERSION_SEMANTIC_V4 = "semantic_v4"
TEXT_VERSIONS = (TEXT_VERSION_V8, TEXT_VERSION_SEMANTIC_V3, TEXT_VERSION_SEMANTIC_V4)
# flips to semantic_v4 in the same commit that pins the v4-trained model (FINETUNED_MODEL_*), never before
DEFAULT_TEXT_VERSION = TEXT_VERSION_SEMANTIC_V4   # flipped 2026-10-11 with the v10 model pin (v10-lora-s0211)

FRAGMENT = "fragment"
BIBLIOGRAPHY = "bibliography"


class EmbeddingText(NamedTuple):
    """The text a record is embedded from, whether it gets a vector at all, and the text's hash.

    :param text: Versioned text (may be empty).
    :param eligible: False when the record must be indexed without a vector.
    :param text_sha256: :func:`src.embeddings.semantic_text.text_sha256` of ``text`` (also for an empty text): the
        hash of the exact bytes handed to the embedder.
    """

    text: str
    eligible: bool
    text_sha256: str


class DocumentCanary(NamedTuple):
    """One real indexed record whose text (and, in ``_meta``, vector) proves the text function and model.

    :param doc_id: The record's index ``_id`` (the canonical id of a fragment, the page uuid of a bibliography page).
    :param text: Its versioned text.
    """

    doc_id: str
    text: str


def check_text_version(version: str) -> str:
    """Validate a text version name.

    :param version: Version name.
    :returns: The same name.
    :rtype: str
    :raises ValueError: For an unknown version.
    """
    if version not in TEXT_VERSIONS:
        raise ValueError(f"unknown text representation version {version!r}; expected one of {TEXT_VERSIONS}")
    return version


def text_kind(doc: Any) -> str:
    """Which text variant a document takes.

    :param doc: A document with an ``embedding_text_kind`` class attribute.
    :returns: :data:`FRAGMENT` or :data:`BIBLIOGRAPHY`.
    :rtype: str
    :raises TypeError: When the document type declares no (or an unknown) kind.
    """
    kind = getattr(doc, "embedding_text_kind", None)
    if kind not in (FRAGMENT, BIBLIOGRAPHY):
        raise TypeError(f"{type(doc).__name__} declares no embedding_text_kind ({FRAGMENT!r} or {BIBLIOGRAPHY!r})")
    return kind


def _production_text(doc: Any, include_scholarly_frames: bool = True) -> str:
    """A document's production (``v8``) text.

    :param doc: A document with a ``create_text_representation()`` method.
    :param include_scholarly_frames: False renders the text without the KTIV scholarly frames
        (``GenizahDocument.create_text_representation(include_scholarly_frames=False)``, for the frames ablation).
        True calls ``create_text_representation()`` without the keyword, so document types that lack it (e.g.
        :class:`BibliographyDocument`) work on the default path.
    :returns: The production text.
    :rtype: str
    """
    if include_scholarly_frames:
        return doc.create_text_representation()
    return doc.create_text_representation(include_scholarly_frames=False)


def text_for_version(v8_text: str, version: str, kind: str = FRAGMENT) -> str:
    """Turn an already-built ``v8`` text into the given version.

    The verifier uses this directly on a bibliography page's stored ``full_text_content`` (which is its ``v8`` text).
    For ``semantic_v4`` fragments the text alone does not tell whether it is eligible (see
    :func:`embed_text_for_version`).

    :param v8_text: The document's ``create_text_representation()`` output.
    :param version: Target text version.
    :param kind: :data:`FRAGMENT` or :data:`BIBLIOGRAPHY`.
    :returns: Versioned text.
    :rtype: str
    """
    check_text_version(version)
    if version == TEXT_VERSION_V8:
        return v8_text
    if kind == BIBLIOGRAPHY:
        return bibliography_semantic_text(v8_text)
    # The own shelf mark is read from the v8 text's "Shelf Mark:" line, i.e. doc.shelf_mark whenever the production
    # text names it: exactly how the training text was made.
    if version == TEXT_VERSION_SEMANTIC_V4:
        return semantic_v4_text(semantic_text(v8_text))
    return semantic_text(v8_text)


def embed_text_for_version(v8_text: str, version: str, kind: str = FRAGMENT) -> EmbeddingText:
    """Versioned text, eligibility and text hash from an already-built ``v8`` text.

    * ``v8``: the text itself, eligible when non-blank.
    * ``semantic_v3`` fragments: :func:`src.embeddings.semantic_text.semantic_text`, eligible by
      :func:`src.embeddings.semantic_text.is_semantic_eligible`.
    * ``semantic_v4`` fragments: ``semantic_v4_text(v3)`` with ``v3 = semantic_text(v8_text)``, eligible when
      ``semantic_v4_ineligible_reason(v3) == "eligible"`` (decided on the v3 text, which the v4 layout has merged).
    * Bibliography pages, both semantic versions: :func:`src.embeddings.semantic_text.bibliography_semantic_text`,
      eligible when non-blank. The v4 layout is not applied to them: it would fold a page's unlabelled prose lines
      into the field line before them.

    :param v8_text: The document's ``create_text_representation()`` output.
    :param version: Target text version.
    :param kind: :data:`FRAGMENT` or :data:`BIBLIOGRAPHY`.
    :returns: ``(text, eligible, text_sha256)``.
    :rtype: EmbeddingText
    """
    check_text_version(version)
    if version == TEXT_VERSION_SEMANTIC_V4 and kind == FRAGMENT:
        v3 = semantic_text(v8_text)
        text, eligible = semantic_v4_text(v3), semantic_v4_ineligible_reason(v3) == "eligible"
    else:
        text = text_for_version(v8_text, version, kind)
        eligible = is_text_eligible(text, version, kind)
    return EmbeddingText(text, eligible, text_sha256(text))


def create_text_representation(doc: Any, version: str = DEFAULT_TEXT_VERSION,
                               include_scholarly_frames: bool = True) -> str:
    """The text a document is embedded from, for a given text version.

    :param doc: A :class:`GenizahDocument` or :class:`BibliographyDocument` (``v8`` accepts any document with a
        ``create_text_representation()`` method).
    :param version: :data:`TEXT_VERSION_V8` (today's text, unchanged), :data:`TEXT_VERSION_SEMANTIC_V3` or
        :data:`TEXT_VERSION_SEMANTIC_V4`.
    :param include_scholarly_frames: False builds the ``v8`` text without the KTIV scholarly frames (see
        :func:`_production_text`; only document types whose ``create_text_representation`` takes the keyword).
    :returns: Versioned text (may be empty for the semantic versions).
    :rtype: str
    """
    check_text_version(version)
    production = _production_text(doc, include_scholarly_frames)
    if version == TEXT_VERSION_V8:
        return production
    return text_for_version(production, version, text_kind(doc))


def is_text_eligible(text: str, version: str, kind: str = FRAGMENT) -> bool:
    """Whether a versioned text may be embedded (an empty string never is).

    :param text: Output of :func:`create_text_representation` / :func:`text_for_version`.
    :param version: Its text version.
    :param kind: :data:`FRAGMENT` or :data:`BIBLIOGRAPHY`.
    :returns: For ``semantic_v3`` fragments, :func:`src.embeddings.semantic_text.is_semantic_eligible`; otherwise
        whether the text is non-blank.
    :rtype: bool
    :raises ValueError: For ``semantic_v4`` fragments: their eligibility is decided on the ``semantic_v3`` text
        (alternate-title parts, ``Subjects:`` lines), which the v4 layout has merged into lines that cannot be checked
        part-wise; use :func:`embed_text_for_version` (or :func:`embedding_text`) instead.
    """
    check_text_version(version)
    if version == TEXT_VERSION_SEMANTIC_V4 and kind == FRAGMENT:
        raise ValueError("semantic_v4 eligibility is decided on the semantic_v3 text and cannot be checked on the "
                         "embedded text; use embed_text_for_version(v8_text, 'semantic_v4') or embedding_text(doc, "
                         "'semantic_v4')")
    if version == TEXT_VERSION_SEMANTIC_V3 and kind == FRAGMENT:
        return is_semantic_eligible(text)
    return bool(text.strip())


def embedding_text(doc: Any, version: str = DEFAULT_TEXT_VERSION,
                   include_scholarly_frames: bool = True) -> EmbeddingText:
    """Versioned text, eligibility and text hash for one document (:func:`embed_text_for_version` of its ``v8`` text).

    :param doc: Document (see :func:`create_text_representation`).
    :param version: Text version.
    :param include_scholarly_frames: False builds the ``v8`` text without the KTIV scholarly frames (see
        :func:`_production_text`).
    :returns: ``(text, eligible, text_sha256)``.
    :rtype: EmbeddingText
    """
    check_text_version(version)
    production = _production_text(doc, include_scholarly_frames)
    kind = FRAGMENT if version == TEXT_VERSION_V8 else text_kind(doc)
    return embed_text_for_version(production, version, kind)
