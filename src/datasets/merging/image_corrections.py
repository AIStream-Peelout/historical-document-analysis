"""Durable corrections for the merged corpus: legacy-store images and transcriptions filed under the wrong record.

Some photographs in the legacy image store were saved under another fragment's name, and some legacy-store
transcriptions sit under a record whose photographs show another fragment. When the shelf-mark slip in a photograph
(or an eye check) shows where an image or a transcription belongs, the finding is recorded here, one JSON line per
correction, and the merge (:func:`src.datasets.merging.merge_shelfmarks.apply_image_corrections` and
:func:`src.datasets.merging.merge_shelfmarks.apply_transcription_corrections`) applies it every time it is
re-created, so the next index cut serves the corrected records.

Every line carries ``kind``: ``"image"`` (the default when the field is absent, as in the first lines of the file)
or ``"transcription"``.

Image correction (one per image file):

* ``image`` -- the file name as it appears in a record's ``images.fjp`` (a bare name, or the basename of an
  absolute URL entry).
* ``url`` -- the file's full public URL, or ``null``.
* ``sha256`` -- the file's sha256 (64 lowercase hex characters), or ``null``.
* ``from_canonical_id`` -- the record the file is filed under now.
* ``to_canonical_id`` -- the record the photograph shows, or ``null`` to detach it (the photograph shows a fragment
  the corpus has no record for).
* ``to_shelfmark_text`` -- the shelf mark the slip names, as read (required when ``to_canonical_id`` is ``null``).
* ``side`` -- the side the slip names (``"r"``, ``"v"`` or an index such as ``"2"``), or ``null``.
* ``basis`` -- ``"slip_read"`` (an automatic read of the slip) or ``"eye"`` (checked by a person).
* ``evidence`` -- the slip text, at most :data:`EVIDENCE_MAX_CHARS` characters.
* ``reviewed`` -- whether a person has confirmed the correction.
* ``recorded_at`` -- ISO date the correction was recorded (``YYYY-MM-DD``).
* ``source`` -- the run that produced it (:data:`SOURCE_SLIP_RELABEL`, :data:`SOURCE_LABEL_PILOT`, ...).
* ``note`` -- optional free text.

Image kinds, by the two canonical ids: move (``to_canonical_id`` names another record); detach
(``to_canonical_id`` is ``null``); side only (``to_canonical_id == from_canonical_id`` and ``side`` is set); revoke
(``to_canonical_id == from_canonical_id`` and ``side`` is ``null`` -- the file is where it belongs; a revoke appended
after a wrong correction cancels it: the file is append-only, later lines win per image).

Transcription correction (``kind: "transcription"``; one or more sections of one legacy-store record):

* ``from_canonical_id`` -- the merged record whose ``sources.fjp`` holds the transcription now.
* ``to_canonical_id`` -- the record the text belongs to, or ``null`` to detach it.
* ``fjp_doc_id`` -- the ``doc_id`` of the legacy-store record (the ``sources.fjp`` entry) that holds the sections.
* ``sections`` -- non-empty list of ``{"key": str, "sha256": str | null}``: the section key (the key of a
  dict-shaped ``transcriptions``, or the 1-based position in a list-shaped one, as a string) and the
  :func:`section_digest` of its value when recorded. A digest that no longer matches stops the correction (the
  merge counts it as ``section_changed``), so a re-scraped or re-ordered source is never moved blindly.
* ``to_shelfmark_text`` -- optional: the shelf mark the text belongs to, as read, or ``null``.
* ``basis``, ``evidence``, ``reviewed``, ``recorded_at``, ``source``, ``note`` -- as for images.

Transcription kinds: move (``to_canonical_id`` names another record), detach (``null``), revoke
(``to_canonical_id == from_canonical_id``: cancels an earlier correction of the same sections). Later lines win per
``(from_canonical_id, fjp_doc_id, section key)``. Use :func:`build_transcription_correction` to make one from a
merged record (it fills the digests).
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_IMAGE_CORRECTIONS = os.path.join(
    _REPO_ROOT, "src", "datasets", "raw_data", "cairo_genizah", "merged", "image_corrections.jsonl"
)

KIND_IMAGE = "image"
KIND_TRANSCRIPTION = "transcription"
KINDS = (KIND_IMAGE, KIND_TRANSCRIPTION)

BASIS_SLIP_READ = "slip_read"
BASIS_EYE = "eye"
BASIS_VALUES = (BASIS_SLIP_READ, BASIS_EYE)
SOURCE_SLIP_RELABEL = "legacy_store_slip_relabel_2026-10-31"
SOURCE_LABEL_PILOT = "label_pilot_2026-10-30"
EVIDENCE_MAX_CHARS = 60

REQUIRED_FIELDS = (
    "image", "url", "sha256", "from_canonical_id", "to_canonical_id", "to_shelfmark_text", "side", "basis",
    "evidence", "reviewed", "recorded_at", "source",
)
OPTIONAL_FIELDS = ("note", "kind")
TRANSCRIPTION_REQUIRED_FIELDS = (
    "kind", "from_canonical_id", "to_canonical_id", "fjp_doc_id", "sections", "basis", "evidence", "reviewed",
    "recorded_at", "source",
)
TRANSCRIPTION_OPTIONAL_FIELDS = ("to_shelfmark_text", "note")

# Relabel-run outcomes: these two become corrections, ``needs_eye`` goes to the review file, the rest change nothing.
OUTCOME_KNOWN = "relabel_to_known_document"
OUTCOME_UNKNOWN = "relabel_to_unknown_document"
OUTCOME_NEEDS_EYE = "needs_eye"
CORRECTION_OUTCOMES = (OUTCOME_KNOWN, OUTCOME_UNKNOWN)
NO_CHANGE_OUTCOMES = ("confirmed", "no_slip", "unreadable")
RELABEL_OUTCOMES = CORRECTION_OUTCOMES + (OUTCOME_NEEDS_EYE,) + NO_CHANGE_OUTCOMES

# Review reasons written by :func:`split_relabels` and :func:`hold_for_reviewed`.
REVIEW_IMAGE_NOT_IN_MANIFEST = "image_not_in_manifest"
REVIEW_FILE_NAME_MISMATCH = "file_name_differs_from_manifest_url"
REVIEW_RESOLVED_TO_OWN_RECORD = "resolved_to_own_record"
REVIEW_KNOWN_WITHOUT_RESOLVED_ID = "known_document_without_resolved_id"
REVIEW_DETACH_WITHOUT_SLIP_TEXT = "detach_without_slip_text"
REVIEW_CONFLICTING = "conflicting_corrections_for_image"
REVIEW_HAS_REVIEWED_CORRECTION = "image_has_reviewed_correction"

# Relabel-row field names, first match wins (the run's own names first).
_RELABEL_FIELDS: Dict[str, Tuple[str, ...]] = {
    "sha256": ("sha256",),
    "from": ("old_canonical_id", "canonical_id", "from_canonical_id"),
    "file_name": ("file_name", "image"),
    "slip_text": ("slip_text", "slip", "label_text"),
    "parsed": ("parsed_shelf_mark", "parsed_shelfmark", "slip_shelfmark"),
    "resolved": ("resolved_canonical_id", "to_canonical_id"),
    "side": ("slip_side", "side"),
    "outcome": ("outcome",),
}

# Label pilot: the eye verdict that marks a misfiled photograph (another fragment, not a join partner).
PILOT_MISFILED_VERDICT = "wrong fragment"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def image_name(entry: str) -> str:
    """Return the file name an ``images.fjp`` entry (bare name or absolute URL) points at.

    :param entry: A bare file name (``"X_1_1r.jpg"``) or an absolute URL ending in one.
    :returns: The file name, the part after the last ``/``.
    """
    return str(entry).rstrip("/").rsplit("/", 1)[-1]


def short_evidence(text: Optional[str], limit: int = EVIDENCE_MAX_CHARS) -> str:
    """Collapse whitespace in a slip reading and cut it to *limit* characters (``"..."`` marks a cut).

    :param text: The slip text, or ``None``.
    :param limit: Maximum length of the result.
    :returns: The shortened text (``""`` for ``None``).
    """
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def record_kind(record: Mapping[str, Any]) -> str:
    """Return a correction line's ``kind`` (``"image"`` when the field is absent).

    :param record: A correction record.
    :returns: :data:`KIND_IMAGE` or :data:`KIND_TRANSCRIPTION` (or whatever unknown value the line holds).
    """
    kind = record.get("kind")
    return KIND_IMAGE if kind is None else kind


def correction_kind(record: Mapping[str, Any]) -> str:
    """Classify a validated correction as ``move``, ``detach``, ``side_only`` or ``revoke``.

    A transcription correction is never ``side_only``: target = source is a revoke.

    :param record: A correction record (image or transcription).
    :returns: The kind.
    """
    to_id = record.get("to_canonical_id")
    if to_id is None:
        return "detach"
    if to_id != record.get("from_canonical_id"):
        return "move"
    if record_kind(record) == KIND_TRANSCRIPTION:
        return "revoke"
    return "side_only" if record.get("side") is not None else "revoke"


def section_digest(value: Any) -> str:
    """The sha256 of one transcription section's value (stable across re-merges of the same source data).

    A string is hashed as its UTF-8 bytes; anything else (a section object of a list-shaped ``transcriptions``) as
    its canonical JSON (sorted keys, no ASCII escaping).

    :param value: The section's value.
    :returns: 64 lowercase hex characters.
    """
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transcription_sections(transcriptions: Any) -> List[Tuple[str, Any]]:
    """The sections of a legacy-store ``transcriptions`` value, keyed as corrections name them.

    A dict gives its own keys (as strings), a list the 1-based position of each entry (``"1"``, ``"2"``, ...).
    Anything else (``None``, an empty value) has no sections.

    :param transcriptions: ``sources.fjp[i]["transcriptions"]`` of a merged record.
    :returns: ``[(section key, value), ...]`` in stored order.
    """
    if isinstance(transcriptions, dict):
        return [(str(key), value) for key, value in transcriptions.items()]
    if isinstance(transcriptions, list):
        return [(str(position), value) for position, value in enumerate(transcriptions, 1)]
    return []


def _require_str(record: Mapping[str, Any], key: str, nullable: bool) -> Optional[str]:
    """Return ``record[key]`` when it is a non-empty string (or ``None`` when *nullable*), else raise.

    :param record: The row being validated.
    :param key: Field name.
    :param nullable: Whether ``None`` is allowed.
    :returns: The value.
    :raises ValueError: On a wrong type or an empty string.
    """
    value = record.get(key)
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string{' or null' if nullable else ''}, got {value!r}")
    return value


def _validate_common(record: Mapping[str, Any]) -> None:
    """Check the fields every kind shares: ``basis``, ``evidence``, ``reviewed``, ``recorded_at``, ``source``,
    ``note``.

    :param record: The parsed JSON object.
    :raises ValueError: On a wrong value.
    """
    if record.get("basis") not in BASIS_VALUES:
        raise ValueError(f"basis must be one of {BASIS_VALUES}, got {record.get('basis')!r}")
    evidence = record.get("evidence")
    if not isinstance(evidence, str) or len(evidence) > EVIDENCE_MAX_CHARS:
        raise ValueError(f"evidence must be a string of at most {EVIDENCE_MAX_CHARS} characters, got {evidence!r}")
    if not isinstance(record.get("reviewed"), bool):
        raise ValueError(f"reviewed must be true or false, got {record.get('reviewed')!r}")
    recorded_at = record.get("recorded_at")
    if not isinstance(recorded_at, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", recorded_at):
        raise ValueError(f"recorded_at must be an ISO date YYYY-MM-DD, got {recorded_at!r}")
    datetime.date.fromisoformat(recorded_at)  # raises ValueError on an impossible date
    _require_str(record, "source", nullable=False)
    note = record.get("note")
    if note is not None and not isinstance(note, str):
        raise ValueError(f"note must be a string, got {note!r}")


def _validate_image_correction(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Check one image correction and return a normalised copy (an integer ``side`` becomes a string).

    :param record: The parsed JSON object (``kind`` absent or ``"image"``).
    :returns: A new dict with exactly the schema's fields (``note`` / ``kind`` only when present).
    :raises ValueError: When a field is missing, unknown, of the wrong type or inconsistent with the others.
    """
    missing = [k for k in REQUIRED_FIELDS if k not in record]
    unknown = [k for k in record if k not in REQUIRED_FIELDS and k not in OPTIONAL_FIELDS]
    if missing or unknown:
        raise ValueError(f"missing fields {missing}, unknown fields {unknown}")

    image = _require_str(record, "image", nullable=False)
    if "/" in image:
        raise ValueError(f"image must be a file name, not a path or URL: {image!r}")
    url = _require_str(record, "url", nullable=True)
    if url is not None and image_name(url) != image:
        raise ValueError(f"url {url!r} does not end in image {image!r}")
    sha256 = record.get("sha256")
    if sha256 is not None and not (isinstance(sha256, str) and _SHA256_RE.match(sha256)):
        raise ValueError(f"sha256 must be 64 lowercase hex characters or null, got {sha256!r}")
    from_id = _require_str(record, "from_canonical_id", nullable=False)
    to_id = _require_str(record, "to_canonical_id", nullable=True)
    to_text = _require_str(record, "to_shelfmark_text", nullable=True)
    if to_id is None and to_text is None:
        raise ValueError("a detach (to_canonical_id null) needs to_shelfmark_text")
    side = record.get("side")
    if isinstance(side, int) and not isinstance(side, bool):
        side = str(side)
    if side is not None and not (isinstance(side, str) and side.strip()):
        raise ValueError(f"side must be a non-empty string, an integer or null, got {side!r}")
    _validate_common(record)

    out = {key: record[key] for key in REQUIRED_FIELDS}
    out.update(image=image, url=url, from_canonical_id=from_id, to_canonical_id=to_id, to_shelfmark_text=to_text,
               side=side)
    for key in OPTIONAL_FIELDS:
        if key in record:
            out[key] = record[key]
    return out


def _validate_transcription_correction(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Check one transcription correction and return a normalised copy.

    :param record: The parsed JSON object (``kind == "transcription"``).
    :returns: A new dict with the schema's fields (``to_shelfmark_text`` always present, ``note`` when given).
    :raises ValueError: When a field is missing, unknown, of the wrong type or inconsistent with the others.
    """
    missing = [k for k in TRANSCRIPTION_REQUIRED_FIELDS if k not in record]
    unknown = [k for k in record
               if k not in TRANSCRIPTION_REQUIRED_FIELDS and k not in TRANSCRIPTION_OPTIONAL_FIELDS]
    if missing or unknown:
        raise ValueError(f"missing fields {missing}, unknown fields {unknown}")
    from_id = _require_str(record, "from_canonical_id", nullable=False)
    to_id = _require_str(record, "to_canonical_id", nullable=True)
    to_text = _require_str(record, "to_shelfmark_text", nullable=True)
    doc_id = record.get("fjp_doc_id")
    if isinstance(doc_id, int) and not isinstance(doc_id, bool):
        doc_id = str(doc_id)
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"fjp_doc_id must be a non-empty string, got {doc_id!r}")
    sections = record.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError(f"sections must be a non-empty list, got {sections!r}")
    normalised: List[Dict[str, Optional[str]]] = []
    for section in sections:
        if not isinstance(section, Mapping) or set(section) - {"key", "sha256"} or "key" not in section:
            raise ValueError(f"each section must be {{'key': str, 'sha256': str | null}}, got {section!r}")
        key = section["key"]
        if isinstance(key, int) and not isinstance(key, bool):
            key = str(key)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"section key must be a non-empty string, got {key!r}")
        digest = section.get("sha256")
        if digest is not None and not (isinstance(digest, str) and _SHA256_RE.match(digest)):
            raise ValueError(f"section sha256 must be 64 lowercase hex characters or null, got {digest!r}")
        normalised.append({"key": key, "sha256": digest})
    keys = [section["key"] for section in normalised]
    if len(set(keys)) != len(keys):
        raise ValueError(f"section keys repeat: {keys}")
    _validate_common(record)

    out: Dict[str, Any] = {
        "kind": KIND_TRANSCRIPTION, "from_canonical_id": from_id, "to_canonical_id": to_id,
        "to_shelfmark_text": to_text, "fjp_doc_id": doc_id, "sections": normalised,
    }
    out.update({key: record[key] for key in ("basis", "evidence", "reviewed", "recorded_at", "source")})
    if "note" in record:
        out["note"] = record["note"]
    return out


def validate_correction(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Check one correction record (image or transcription) against the schema and return a normalised copy.

    :param record: The parsed JSON object.
    :returns: A new dict with exactly the schema's fields for its kind.
    :raises ValueError: When the kind is unknown, or a field is missing, unknown, of the wrong type or inconsistent
        with the others.
    """
    if not isinstance(record, Mapping):
        raise ValueError(f"a correction must be a JSON object, got {type(record).__name__}")
    kind = record_kind(record)
    if kind == KIND_IMAGE:
        return _validate_image_correction(record)
    if kind == KIND_TRANSCRIPTION:
        return _validate_transcription_correction(record)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def _read_jsonl(path: str) -> Iterable[Tuple[int, Any]]:
    """Yield ``(line_number, parsed_object)`` for each non-blank line of a JSONL file.

    :param path: File path.
    :returns: Iterator of 1-based line numbers and parsed values.
    :raises ValueError: On a line that is not valid JSON (the message names the file and line).
    """
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not valid JSON ({exc.msg})") from exc


def iter_corrections(path: str) -> Iterable[Dict[str, Any]]:
    """Yield every validated correction of a corrections file in file order, both kinds (no override applied).

    :param path: Path to ``image_corrections.jsonl``.
    :returns: Iterator of normalised correction records.
    :raises ValueError: On a malformed line (the message names the file and line).
    """
    for number, row in _read_jsonl(path):
        try:
            yield validate_correction(row)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc


def iter_image_corrections(path: str) -> Iterable[Dict[str, Any]]:
    """Yield every validated image correction of a corrections file in file order (no override applied).

    Transcription lines are validated too but not yielded.

    :param path: Path to ``image_corrections.jsonl``.
    :returns: Iterator of normalised image correction records.
    :raises ValueError: On a malformed line (the message names the file and line).
    """
    for record in iter_corrections(path):
        if record_kind(record) == KIND_IMAGE:
            yield record


def load_image_corrections(path: str = DEFAULT_IMAGE_CORRECTIONS) -> Dict[str, Dict[str, Any]]:
    """Load the image corrections into ``{image file name: correction}``; a later line overrides an earlier one.

    :param path: Path to ``image_corrections.jsonl``.
    :returns: The effective correction per image file (revokes included; the merge skips them).
    :raises ValueError: On a malformed line.
    :raises FileNotFoundError: When *path* does not exist.
    """
    corrections: Dict[str, Dict[str, Any]] = {}
    for record in iter_image_corrections(path):
        corrections[record["image"]] = record
    return corrections


TranscriptionKey = Tuple[str, str, str]


def load_transcription_corrections(path: str = DEFAULT_IMAGE_CORRECTIONS) -> Dict[TranscriptionKey, Dict[str, Any]]:
    """Load the transcription corrections, one entry per section; a later line overrides an earlier one.

    :param path: Path to ``image_corrections.jsonl``.
    :returns: ``{(from_canonical_id, fjp_doc_id, section key): correction}``; the correction is the whole line,
        whose ``sections`` may name more sections than the key (revokes included; the merge skips them).
    :raises ValueError: On a malformed line.
    :raises FileNotFoundError: When *path* does not exist.
    """
    corrections: Dict[TranscriptionKey, Dict[str, Any]] = {}
    for record in iter_corrections(path):
        if record_kind(record) != KIND_TRANSCRIPTION:
            continue
        for section in record["sections"]:
            corrections[(record["from_canonical_id"], record["fjp_doc_id"], section["key"])] = record
    return corrections


def _dedupe_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Key used to skip a correction already in the file.

    Image: kind, image, target record and side (the side is in the key so that a later side fix, or a revoke after a
    side-only correction, is not dropped). Transcription: kind, source record, legacy-store record, section keys and
    target record.

    :param record: A validated correction.
    :returns: The key.
    """
    if record_kind(record) == KIND_TRANSCRIPTION:
        return (KIND_TRANSCRIPTION, record["from_canonical_id"], record["fjp_doc_id"],
                tuple(sorted(section["key"] for section in record["sections"])), record["to_canonical_id"])
    return KIND_IMAGE, record["image"], record["to_canonical_id"], record["side"]


def new_corrections(path: str, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The records :func:`append_image_corrections` would append: validated, minus those already in *path* or
    repeated earlier in *records*.

    :param path: Path to ``image_corrections.jsonl`` (may be absent).
    :param records: Correction records.
    :returns: The fresh validated records, in input order.
    :raises ValueError: When a record (or an existing line) is malformed.
    """
    validated = [validate_correction(record) for record in records]
    seen = {_dedupe_key(r) for r in iter_corrections(path)} if os.path.exists(path) else set()
    fresh: List[Dict[str, Any]] = []
    for record in validated:
        key = _dedupe_key(record)
        if key in seen:
            continue
        seen.add(key)
        fresh.append(record)
    return fresh


def append_image_corrections(path: str, records: Sequence[Mapping[str, Any]]) -> int:
    """Append corrections (either kind) to *path* (created when absent), skipping ones already recorded.

    The file is append-only: existing lines are never rewritten. A record whose dedupe key (:func:`_dedupe_key`)
    is already in the file, or earlier in *records*, is skipped. Every record is validated before anything is
    written, so a bad record leaves the file untouched.

    :param path: Path to ``image_corrections.jsonl``.
    :param records: Correction records to add.
    :returns: Number of lines appended.
    :raises ValueError: When a record (or an existing line) is malformed.
    """
    fresh = new_corrections(path, records)
    if fresh:
        with open(path, "a", encoding="utf-8") as fh:
            for record in fresh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(fresh)


def build_transcription_correction(
    merged_record: Mapping[str, Any],
    fjp_doc_id: str,
    section_keys: Sequence[str],
    to_canonical_id: Optional[str],
    *,
    basis: str,
    evidence: str,
    reviewed: bool,
    source: str,
    to_shelfmark_text: Optional[str] = None,
    recorded_at: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Make a transcription correction for sections of one legacy-store record of a merged record, with digests.

    :param merged_record: The merged record the sections are filed under now (its ``canonical_id`` is the source).
    :param fjp_doc_id: ``doc_id`` of the ``sources.fjp`` entry that holds them.
    :param section_keys: Section keys (:func:`transcription_sections`).
    :param to_canonical_id: The record the text belongs to, or ``None`` to detach it.
    :param basis: :data:`BASIS_EYE` or :data:`BASIS_SLIP_READ`.
    :param evidence: Why (at most :data:`EVIDENCE_MAX_CHARS` characters; cut with :func:`short_evidence`).
    :param reviewed: Whether a person confirmed it.
    :param source: The run or review that produced it.
    :param to_shelfmark_text: The shelf mark the text belongs to, as read, or ``None``.
    :param recorded_at: ISO date (default: today).
    :param note: Optional free text.
    :returns: A validated correction record.
    :raises KeyError: When the record has no such legacy-store entry, or the entry no such section.
    :raises ValueError: When the result does not validate.
    """
    entries = (merged_record.get("sources") or {}).get("fjp") or []
    entry = next((e for e in entries if str(e.get("doc_id")) == str(fjp_doc_id)), None)
    if entry is None:
        raise KeyError(f"{merged_record.get('canonical_id')} has no legacy-store record {fjp_doc_id!r}")
    sections = dict(transcription_sections(entry.get("transcriptions")))
    missing = [key for key in section_keys if str(key) not in sections]
    if missing:
        raise KeyError(f"legacy-store record {fjp_doc_id!r} has no sections {missing} (has {sorted(sections)})")
    record: Dict[str, Any] = {
        "kind": KIND_TRANSCRIPTION,
        "from_canonical_id": merged_record["canonical_id"],
        "to_canonical_id": to_canonical_id,
        "to_shelfmark_text": to_shelfmark_text,
        "fjp_doc_id": str(fjp_doc_id),
        "sections": [{"key": str(key), "sha256": section_digest(sections[str(key)])} for key in section_keys],
        "basis": basis,
        "evidence": short_evidence(evidence),
        "reviewed": reviewed,
        "recorded_at": recorded_at or datetime.date.today().isoformat(),
        "source": source,
    }
    if note is not None:
        record["note"] = note
    return validate_correction(record)


def _pick(row: Mapping[str, Any], field: str, required: bool = True) -> Any:
    """Read a relabel-row field under any of its accepted names (:data:`_RELABEL_FIELDS`).

    :param row: One relabel row.
    :param field: Logical field (``"sha256"``, ``"from"``, ``"file_name"``, ``"slip_text"``, ``"parsed"``,
        ``"resolved"``, ``"side"``, ``"outcome"``).
    :param required: Raise when none of the names is present.
    :returns: The value, or ``None`` when absent and not required.
    :raises KeyError: When a required field is absent under every accepted name.
    """
    for name in _RELABEL_FIELDS[field]:
        if name in row:
            return row[name]
    if required:
        raise KeyError(f"relabel row lacks {field!r} (looked for {_RELABEL_FIELDS[field]})")
    return None


class _ManifestIndex:
    """An image manifest indexed for the relabel join.

    Rows are keyed by the record the file was filed under when the run read it: ``old_canonical_id`` (the v2
    manifest rewrites ``canonical_id`` to the new record on moved rows and to ``None`` on detached rows), else
    ``canonical_id``.
    """

    def __init__(self, manifest_path: str) -> None:
        """Read and index the manifest.

        :param manifest_path: JSONL with ``canonical_id`` (and, in v2, ``old_canonical_id``), ``url``, ``sha256``
            per image file.
        """
        self.by_name: Dict[Tuple[Optional[str], str, str], dict] = {}
        self.by_pair: Dict[Tuple[Optional[str], str], List[dict]] = collections.defaultdict(list)
        self.by_sha: Dict[str, List[dict]] = collections.defaultdict(list)
        for _, row in _read_jsonl(manifest_path):
            filed_under = row.get("old_canonical_id") or row.get("canonical_id")
            sha = row["sha256"]
            if row.get("url"):
                self.by_name[(filed_under, sha, image_name(row["url"]))] = row
            self.by_pair[(filed_under, sha)].append(row)
            self.by_sha[sha].append(row)

    def find(self, from_id: str, sha: str, file_name: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
        """Find the manifest row of one relabel row.

        With a ``file_name``: the row filed under *from_id* with this sha256 whose url ends in the name; failing
        that, a single row with this record and sha256 (or this sha256 alone) is reported as a name mismatch.
        Without one: the single row with this record and sha256, else the single row with this sha256.

        :param from_id: The record the image was filed under.
        :param sha: The image's sha256.
        :param file_name: The run's ``file_name``, or ``None``.
        :returns: ``(manifest row or None, review reason or None)``.
        """
        if file_name:
            row = self.by_name.get((from_id, sha, file_name))
            if row is not None:
                return row, None
        candidates = self.by_pair.get((from_id, sha)) or []
        if not candidates:
            candidates = self.by_sha.get(sha) or []
        if len(candidates) != 1:
            return None, REVIEW_IMAGE_NOT_IN_MANIFEST
        row = candidates[0]
        if file_name and image_name(row.get("url") or "") != file_name:
            return row, REVIEW_FILE_NAME_MISMATCH
        return row, None


def split_relabels(
    relabels_path: str,
    manifest_path: str,
    recorded_at: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Turn a slip-relabel run into image corrections plus a review list.

    ``relabel_to_known_document`` and ``relabel_to_unknown_document`` become corrections with ``to_canonical_id`` =
    the resolved canonical id when there is one (a move: the slip names one of the documents of the set, or a
    corpus record outside it) and ``null`` otherwise (a detach: the slip names a fragment the corpus has no record
    for; ``to_shelfmark_text`` = the parsed shelf mark, else the slip text). ``confirmed``, ``no_slip`` and
    ``unreadable`` become nothing.

    These go to the review list instead of the corrections: ``needs_eye`` rows; rows whose image is not in the
    manifest, or whose ``file_name`` is not the basename of the manifest url; rows resolved to the record they are
    filed under; ``relabel_to_known_document`` rows without a resolved id (the run's own contract says they have
    one); detaches without any slip text; and every row of an image that gets more than one distinct correction.

    :param relabels_path: The run's ``image_relabels.jsonl`` (sha256, ``file_name``, old canonical id, slip text,
        parsed shelf mark, resolved canonical id, slip side, outcome per image).
    :param manifest_path: The image manifest the run wrote or read (``canonical_id`` / ``old_canonical_id``,
        ``url``, ``sha256`` per image file); the correction's ``url`` is the manifest url.
    :param recorded_at: ISO date for ``recorded_at`` (default: today).
    :returns: ``(corrections, review_rows)``; each review row is the relabel row plus ``review_reason`` and, when
        known, ``image`` / ``url``.
    :raises ValueError: On an outcome outside :data:`RELABEL_OUTCOMES`.
    """
    recorded_at = recorded_at or datetime.date.today().isoformat()
    manifest = _ManifestIndex(manifest_path)
    candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    review: List[Dict[str, Any]] = []
    for number, row in _read_jsonl(relabels_path):
        outcome = _pick(row, "outcome")
        if outcome not in RELABEL_OUTCOMES:
            raise ValueError(f"{relabels_path}:{number}: unknown outcome {outcome!r}")
        if outcome in NO_CHANGE_OUTCOMES:
            continue
        sha, from_id = _pick(row, "sha256"), _pick(row, "from")
        file_name = _pick(row, "file_name", required=False) or None
        manifest_row, problem = manifest.find(from_id, sha, file_name)
        url = (manifest_row or {}).get("url") or None
        image = file_name or (image_name(url) if url else None)
        located = {key: value for key, value in (("image", image), ("url", url)) if value}
        if outcome == OUTCOME_NEEDS_EYE:
            review.append({**row, **located, "review_reason": OUTCOME_NEEDS_EYE})
            continue
        if problem is not None or not image:
            review.append({**row, **located, "review_reason": problem or REVIEW_IMAGE_NOT_IN_MANIFEST})
            continue
        resolved = _pick(row, "resolved", required=False) or None
        if resolved == from_id:
            review.append({**row, **located, "review_reason": REVIEW_RESOLVED_TO_OWN_RECORD})
            continue
        if outcome == OUTCOME_KNOWN and resolved is None:
            review.append({**row, **located, "review_reason": REVIEW_KNOWN_WITHOUT_RESOLVED_ID})
            continue
        slip_text = _pick(row, "slip_text", required=False)
        parsed = _pick(row, "parsed", required=False)
        if resolved is None and not short_evidence(parsed or slip_text):
            review.append({**row, **located, "review_reason": REVIEW_DETACH_WITHOUT_SLIP_TEXT})
            continue
        side = _pick(row, "side", required=False)
        record = validate_correction({
            "image": image,
            "url": url,
            "sha256": sha,
            "from_canonical_id": from_id,
            "to_canonical_id": resolved,
            "to_shelfmark_text": short_evidence(parsed or slip_text, 120) or None,
            "side": side if side not in ("", None) else None,
            "basis": BASIS_SLIP_READ,
            "evidence": short_evidence(slip_text),
            "reviewed": False,
            "recorded_at": recorded_at,
            "source": SOURCE_SLIP_RELABEL,
        })
        candidates.append((record, {**row, **located}))

    # One correction per image file: an image with two different corrections (two source records, or two targets)
    # is left for a person to settle.
    by_image: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = collections.defaultdict(list)
    for record, row in candidates:
        by_image[record["image"]].append((record, row))
    corrections: List[Dict[str, Any]] = []
    for image in sorted(by_image):
        group = by_image[image]
        distinct = {(r["from_canonical_id"], r["to_canonical_id"], r["side"]) for r, _ in group}
        if len(distinct) == 1:
            corrections.append(group[0][0])
        else:
            review.extend({**row, "review_reason": REVIEW_CONFLICTING} for _, row in group)
    return corrections, review


def write_review(review_path: str, review: Sequence[Mapping[str, Any]]) -> None:
    """Write review rows as JSONL (overwrites *review_path*).

    :param review_path: Destination.
    :param review: Rows.
    """
    with open(review_path, "w", encoding="utf-8") as fh:
        for row in review:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def from_relabels(
    relabels_path: str,
    manifest_path: str,
    review_path: Optional[str] = None,
    recorded_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert a slip-relabel run into correction records (see :func:`split_relabels`).

    Rows that need a person (``needs_eye`` and the other review reasons) never become corrections; they are written
    to *review_path* (overwritten) when one is given.

    :param relabels_path: The run's ``image_relabels.jsonl``.
    :param manifest_path: The image manifest the run read.
    :param review_path: Where to write the review rows (JSONL), or ``None`` to skip writing.
    :param recorded_at: ISO date for ``recorded_at`` (default: today).
    :returns: The correction records, one per image file, sorted by image.
    """
    corrections, review = split_relabels(relabels_path, manifest_path, recorded_at=recorded_at)
    if review_path is not None:
        write_review(review_path, review)
    return corrections


def hold_for_reviewed(
    candidates: Sequence[Mapping[str, Any]],
    established: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Keep automatic image corrections from overriding a reviewed one.

    An unreviewed candidate for an image whose established correction is reviewed is dropped when it says the same
    (same source, target and side) and sent to review otherwise, since appending it would make it the later line
    that wins. Reviewed candidates and candidates for other images are kept.

    :param candidates: Image corrections about to be appended (e.g. :func:`split_relabels` output).
    :param established: ``{image: correction}`` already recorded or appended in the same run (e.g.
        :func:`load_image_corrections` updated with the run's reviewed records).
    :returns: ``(kept, review_rows, agreeing)``: the candidates to append, the held ones as review rows (the
        candidate plus ``review_reason`` and ``reviewed_correction``), and the number dropped as already recorded.
    """
    kept: List[Dict[str, Any]] = []
    held: List[Dict[str, Any]] = []
    agreeing = 0
    for candidate in candidates:
        current = established.get(candidate["image"])
        if candidate.get("reviewed") or not current or not current.get("reviewed"):
            kept.append(dict(candidate))
            continue
        same = all(candidate.get(k) == current.get(k) for k in ("from_canonical_id", "to_canonical_id", "side"))
        if same:
            agreeing += 1
            continue
        held.append({**candidate, "review_reason": REVIEW_HAS_REVIEWED_CORRECTION,
                     "reviewed_correction": {k: current.get(k) for k in (
                         "from_canonical_id", "to_canonical_id", "side", "basis", "source", "recorded_at")}})
    return kept, held, agreeing


def from_label_pilot(
    eye_review_path: str,
    sample_path: str,
    targets: Mapping[str, Mapping[str, Any]],
    recorded_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert the label pilot's eye-confirmed misfilings into corrections (``basis = "eye"``, reviewed).

    Only rows whose eye verdict is :data:`PILOT_MISFILED_VERDICT` count; join partners are not misfilings. The pilot
    records the slip text but not the record it names, so the caller supplies that per pilot key in *targets*.

    :param eye_review_path: The pilot's ``eye_review.jsonl`` (``key``, ``canonical_id``, ``file``, ``label_read``,
        ``eye``, ``note`` per image).
    :param sample_path: The pilot's ``sample.jsonl`` (``key``, ``image_url``, ``sha256`` per image).
    :param targets: ``{pilot key: {"to_canonical_id": str | None, "to_shelfmark_text": str, "side": str | None}}``
        for every misfiled row.
    :param recorded_at: ISO date for ``recorded_at`` (default: today).
    :returns: One correction per misfiled image, in pilot-key order.
    :raises KeyError: When a misfiled row has no entry in *targets*, or a *targets* key is not a misfiled row.
    """
    recorded_at = recorded_at or datetime.date.today().isoformat()
    samples = {row["key"]: row for _, row in _read_jsonl(sample_path)}
    misfiled = [row for _, row in _read_jsonl(eye_review_path) if row.get("eye") == PILOT_MISFILED_VERDICT]
    keys = {row["key"] for row in misfiled}
    if keys != set(targets):
        raise KeyError(f"targets must cover exactly the misfiled rows: missing {sorted(keys - set(targets))}, "
                       f"extra {sorted(set(targets) - keys)}")
    records: List[Dict[str, Any]] = []
    for row in sorted(misfiled, key=lambda r: r["key"]):
        sample, target = samples[row["key"]], targets[row["key"]]
        url = sample.get("image_url")
        records.append(validate_correction({
            "image": image_name(url or row["file"]),
            "url": url,
            "sha256": sample.get("sha256"),
            "from_canonical_id": row["canonical_id"],
            "to_canonical_id": target.get("to_canonical_id"),
            "to_shelfmark_text": target.get("to_shelfmark_text"),
            "side": target.get("side"),
            "basis": BASIS_EYE,
            "evidence": short_evidence(row.get("label_read")),
            "reviewed": True,
            "recorded_at": recorded_at,
            "source": SOURCE_LABEL_PILOT,
        }))
    return records

