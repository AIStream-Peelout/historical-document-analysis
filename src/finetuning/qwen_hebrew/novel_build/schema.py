"""Record types shared by every stage of the novel-rows build, with JSON-lines helpers.

Rules the types encode: a side is placed by reads, never by a file name or a canvas order (``ImageRef.label`` is
evidence for review only); reads never become targets; every exclusion carries reasons; script is measured from the
target while the source's language statement is kept apart.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Type, TypeVar

T = TypeVar("T")

OUTCOMES = ("accepted", "needs_review", "unassigned", "dropped")
SPLITS = ("train", "val", "excluded")
SCRIPTS = ("hebrew", "arabic", "mixed", "none")


@dataclass
class ImageRef:
    """One photograph known for a candidate.

    :param image_id: Stable id inside the candidate (``<origin>:<index>``) until the sha256 is known.
    :param origin: Where the photograph comes from, e.g. ``fjp_local``, ``fjp_bucket``, ``iiif_cudl``, ``iiif_pgp``,
        ``iiif_nli``, ``bodleian``, ``ktiv_zip``, ``cudl_local``, ``apd_local``, ``aodeh_local``, ``pgpid_mirror``.
    :param url: Remote URL, if any.
    :param path: Local file once present or fetched.
    :param sha256: Hex digest of the file bytes.
    :param phash: 64-bit perceptual hash as hex.
    :param width: Pixel width.
    :param height: Pixel height.
    :param canvas_index: Position in the library manifest, if any.
    :param label: Library canvas label (``1r``) or slip side; review evidence only, never an assignment basis.
    :param slip_outcome: Result of a shelf-mark slip read, if one exists.
    :param status: ``listed`` | ``present`` | ``fetched`` | ``missing`` | ``too_small`` | ``head_404`` | ``duplicate``.
    :param note: Free text.
    """

    image_id: str
    origin: str
    url: Optional[str] = None
    path: Optional[str] = None
    sha256: Optional[str] = None
    phash: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    canvas_index: Optional[int] = None
    label: Optional[str] = None
    slip_outcome: Optional[str] = None
    status: str = "listed"
    note: str = ""


@dataclass
class TargetSide:
    """The edition text of one side (or one unlabelled block) of a candidate, with provenance.

    :param side_key: ``recto`` | ``verso`` | ``""`` (unlabelled) | ``page`` | another edition label.
    :param text: Target text after the source's sigla rules (what a row would train on).
    :param raw_text: Text as extracted, before sigla rules.
    :param lines: ``text`` split into lines.
    :param pointer: Provenance: file, section/block/page ids, line range, as the source gives them.
    :param side_marker: The edition's own side marker, if any (``ע"א``, ``Recto``, ...).
    :param letters_hebrew: Hebrew letter count of ``text``.
    :param letters_arabic: Arabic letter count of ``text``.
    """

    side_key: str
    text: str
    raw_text: str
    lines: List[str]
    pointer: Dict[str, Any]
    side_marker: Optional[str] = None
    letters_hebrew: int = 0
    letters_arabic: int = 0


@dataclass
class Candidate:
    """One ledger candidate normalised across sources.

    :param row_id: The ledger's ``row_id``.
    :param source: Ledger source name (``pgp``, ``fjp``, ``fjp_scrape``, ``ktiv``, ``literature``, ``lit_run2``,
        ``apd_v3``, ``aodeh_v2_1``).
    :param document_group: The ledger's document group (sides and editions of one fragment share it).
    :param canonical_ids: ES canonical ids (``cids_es``).
    :param pgpids: PGP ids, if any.
    :param ktiv_id: NLI system number, if any.
    :param shelfmark: The source's shelf-mark string.
    :param script: Ledger script (measured on the whole candidate; re-measured per side later).
    :param languages_stated: The source's language statement, verbatim.
    :param verdict: Ledger verdict (``novel`` for everything this build consumes).
    :param witnesses: Ledger witnesses (empty for novel).
    :param unit: ``document`` | ``side`` | ``page``.
    :param n_letters: Ledger letter count.
    :param targets: Sides with text (filled by :mod:`.targets`).
    :param images: Known photographs (filled by :mod:`.images`).
    :param rnl: True when any id or mark names the Russian National Library holdings.
    :param notes: Free-form provenance and flags.
    """

    row_id: str
    source: str
    document_group: str
    canonical_ids: List[str]
    pgpids: List[str]
    ktiv_id: Optional[str]
    shelfmark: str
    script: str
    languages_stated: Any
    verdict: str
    witnesses: List[str]
    unit: str
    n_letters: int
    targets: List[TargetSide] = field(default_factory=list)
    images: List[ImageRef] = field(default_factory=list)
    rnl: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadRecord:
    """One OCR read of one photograph, evidence for side placement only.

    :param sha256: The photograph's sha256.
    :param reader: ``kraken:<model>`` | ``vlm:<model key>`` | ``gemini:<model>`` (legacy cache only).
    :param text: Read text.
    :param lines: Read lines.
    :param n_lines: Number of lines.
    :param letters_hebrew: Hebrew letters in ``text``.
    :param letters_arabic: Arabic letters in ``text``.
    :param finish: ``ok`` | ``length`` | ``blank`` | ``error`` | ``looping``.
    :param seconds: Wall-clock seconds of the request.
    :param read_at: ISO timestamp.
    :param meta: Reader-specific extras (confidences, prepared size, request ids).
    """

    sha256: str
    reader: str
    text: str
    lines: List[str]
    n_lines: int
    letters_hebrew: int
    letters_arabic: int
    finish: str
    seconds: float
    read_at: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SideDecision:
    """Placement of one target side on one photograph (or its refusal).

    :param row_id: Candidate id.
    :param side_key: Target side.
    :param outcome: One of :data:`OUTCOMES`.
    :param reasons: Reason codes (empty for a clean acceptance).
    :param image_sha256: The chosen photograph, when accepted.
    :param basis: ``p99`` | ``p95_with_prior`` | ``single_image_p99`` | ...; the rule that accepted it.
    :param reader: The read used as evidence.
    :param recall: Clipped n-gram recall of the target in the read.
    :param null_p95: Null-distribution p95 for this side's length bin.
    :param null_p99: Null-distribution p99.
    :param margin: Recall margin over the runner-up photograph, if any.
    :param competitors: Per-photograph scores considered, for review.
    """

    row_id: str
    side_key: str
    outcome: str
    reasons: List[str]
    image_sha256: Optional[str] = None
    basis: Optional[str] = None
    reader: Optional[str] = None
    recall: Optional[float] = None
    null_p95: Optional[float] = None
    null_p99: Optional[float] = None
    margin: Optional[float] = None
    competitors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class QualityResult:
    """Quality gate result for one accepted side.

    :param row_id: Candidate id.
    :param side_key: Target side.
    :param ok: True when every gate passed.
    :param reasons: Failing gates, reason codes.
    :param script: Script measured from the final text (one of :data:`SCRIPTS`).
    :param letters_hebrew: Hebrew letters of the final text.
    :param letters_arabic: Arabic letters of the final text.
    :param tokens: Tokenised length of question + answer, if measured.
    :param text: Final target text after repairs (harakat stripped, finals repaired).
    """

    row_id: str
    side_key: str
    ok: bool
    reasons: List[str]
    script: str
    letters_hebrew: int
    letters_arabic: int
    tokens: Optional[int]
    text: str


@dataclass
class HoldoutResult:
    """Hold-out and split decision for one side.

    :param row_id: Candidate id.
    :param side_key: Target side.
    :param held_out: True when any registry, benchmark key or text shingle matched.
    :param reasons: ``benchmark:<name>:<id>`` / ``text:<set>:<id>`` / ``exclusion:<name>`` codes.
    :param split: One of :data:`SPLITS`.
    :param rnl: Russian National Library holder flag.
    :param flags: ``boilerplate:<set>:<id>`` for every text match the boilerplate exception discounted (the strict
        text rule would have held the side out on it); empty under the strict rule.
    """

    row_id: str
    side_key: str
    held_out: bool
    reasons: List[str]
    split: str
    rnl: bool
    flags: List[str] = field(default_factory=list)


@dataclass
class RowRecord:
    """Manifest record of one written training or validation row.

    :param stem: Row stem (unique).
    :param row_id: Candidate id.
    :param side_key: Target side.
    :param source: Ledger source.
    :param document_group: Ledger document group.
    :param canonical_ids: ES canonical ids.
    :param pgpids: PGP ids.
    :param image_sha256: The photograph.
    :param image_path: Local path of the photograph used.
    :param image_url: Remote URL, if any.
    :param image_origin: Image origin.
    :param target_pointer: Provenance of the target text.
    :param side_basis: The accepted :class:`SideDecision` as a dict (reader, recall, null, margin, basis).
    :param script: Measured script.
    :param languages_stated: The source's language statement.
    :param verdict: Ledger verdict.
    :param witnesses: Ledger witnesses.
    :param quality_flags: Non-fatal quality notes.
    :param holdout_reasons: Reasons when excluded (empty for rows).
    :param split: ``train`` | ``val``.
    :param letters: Letters in the target.
    :param target_tokens: Tokenised target length.
    """

    stem: str
    row_id: str
    side_key: str
    source: str
    document_group: str
    canonical_ids: List[str]
    pgpids: List[str]
    image_sha256: str
    image_path: str
    image_url: Optional[str]
    image_origin: str
    target_pointer: Dict[str, Any]
    side_basis: Dict[str, Any]
    script: str
    languages_stated: Any
    verdict: str
    witnesses: List[str]
    quality_flags: List[str]
    holdout_reasons: List[str]
    split: str
    letters: int
    target_tokens: int


def to_dict(record: Any) -> Dict[str, Any]:
    """Serialise a dataclass record (nested dataclasses included) to a plain dict.

    :param record: A dataclass instance.
    :return: JSON-serialisable dict.
    """
    return dataclasses.asdict(record)


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Build a record from a dict, rebuilding nested ``targets`` and ``images`` lists.

    :param cls: The record class.
    :param data: A dict as written by :func:`to_dict`.
    :return: The record.
    """
    payload = dict(data)
    if cls is Candidate:
        payload["targets"] = [TargetSide(**t) for t in payload.get("targets", [])]
        payload["images"] = [ImageRef(**i) for i in payload.get("images", [])]
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in names})  # type: ignore[arg-type]


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    """Write records (dataclasses or dicts) as JSON lines, creating parent dirs.

    :param path: Output file.
    :param records: Records to write.
    :return: Number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            payload = to_dict(rec) if dataclasses.is_dataclass(rec) else rec
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path, cls: Optional[Type[T]] = None) -> Iterator[Any]:
    """Iterate JSON lines, optionally as records of ``cls``.

    :param path: Input file.
    :param cls: Record class, or None for plain dicts.
    :return: Iterator of records.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            yield from_dict(cls, data) if cls is not None else data
