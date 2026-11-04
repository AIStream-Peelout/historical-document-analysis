"""Stage 6 of the novel-rows build: assemble page rows from the stage files, check them, write the dataset.

A row is one manuscript side that passed every stage: a target extracted from its edition (stage0), a photograph
found (stage1), the side placed on that photograph by reads (stage2), the quality gates (stage3) and the hold-outs
(stage5). The build adds its own checks on top (the ledger still says ``novel``; the accepted photograph is a found
stage1 image with a local file; the read that placed the side exists; the text is not empty) and never re-decides
novelty, sides or quality.

A last gate makes the plan's human review executable (:func:`review_gate`; the private reviews file comes from
``--reviews`` or the env var ``NOVEL_BUILD_REVIEWS``). A passed quality stage only means the automated checks passed:
a side of a scanned-edition source (``literature``, ``lit_run2``) counts once its work's OCR eye sheet is listed under
``approved_works``, and a rule-recovered placement (decision reasons ``spill_ratio_ok`` / ``conflict_ratio_ok``) once
an ``approved_rows`` entry says ``ok`` for it, when the review policy of that class is ``require``. Under ``flag`` the
row is admitted with a ``review_pending_*`` entry in its ``quality_flags``. A verdict other than ``ok`` excludes the
row under either policy. With no reviews file the defaults apply: scanned-edition works ``require``, rule-recovered
placements ``flag``. The works named in the reviews file's ``required_works`` are ``require`` whatever the policy of
the rest says (the typewritten theses need their eye sheet; printed editions can be flagged);
``python -m src.finetuning.qwen_hebrew.novel_build.build --list-works`` prints the work keys of stage0 to choose from.

Rows are in the KTIV ``FEATURES`` schema every page builder writes (``image`` = the local file, which
``save_dataset`` embeds), ``question = SCRIPT_NEUTRAL_PROMPT`` on every row, section ``novel_page``.

Outputs of :func:`write_dataset`: the saved DatasetDict (splits ``train_page`` and ``val``) with ``README.md``,
``PRIVATE.md`` (when private) and ``stats.json``; ``manifest.jsonl`` (one :class:`~.schema.RowRecord` per row, plus
the RNL holder flag and the shelf mark); ``decisions.jsonl`` (one record per (candidate, side): the last stage it
reached and the reasons it stopped there); the images-once export next to it. :func:`main` also writes the stage6
manifest and decisions into the work dir and regenerates ``FUNNEL.md`` there.

:func:`acceptance_checks` runs before anything is written; any violation stops the build. A dataset holding a
legacy-store photograph or transcription, or a Russian National Library document, is always private, and
``--public`` refuses it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image as PILImage

from src.finetuning.qwen_hebrew.build_pgp_arabic_editions import TASK
from src.finetuning.qwen_hebrew.build_pgp_editions import save_dataset
from src.finetuning.qwen_hebrew.images_once import export_images_once
from src.finetuning.qwen_hebrew.novel_build import funnel, paths
from src.finetuning.qwen_hebrew.novel_build.schema import (Candidate, ImageRef, QualityResult, RowRecord,
                                                           SideDecision, read_jsonl, to_dict, write_jsonl)
from src.finetuning.qwen_hebrew.prompts import SCRIPT_NEUTRAL_PROMPT

SECTION = "novel_page"
PROMPT_NAME = "SCRIPT_NEUTRAL_PROMPT"
#: Ledger source -> ``label_source`` of its rows.
LABEL_SOURCES = {"pgp": "pgp_edition", "fjp": "legacy_store_diplomatic", "fjp_scrape": "legacy_store_scrape",
                 "ktiv": "ktiv_nli", "literature": "scanned_edition", "lit_run2": "run2_edition",
                 "apd_v3": "apd_edition", "aodeh_v2_1": "book_edition"}
#: Ledger sources whose transcriptions come from the legacy store.
LEGACY_STORE_SOURCES = ("fjp", "fjp_scrape")
#: Image origins starting with this come from the legacy store.
LEGACY_STORE_ORIGIN_PREFIX = "fjp"
#: Hold-out split -> dataset split name.
SPLIT_NAMES = {"train": "train_page", "val": "val"}
#: Characters a training answer may not hold (the notebooks' label-hygiene set).
HYGIENE_CHARS = "␣&#$_{}<>\\"
#: Question + answer tokens a row may use: context 12,288 minus the image budget 6,835 minus 200 of headroom.
TOKEN_BUDGET = 12288 - 6835 - 200
#: The reads file inside the work dir; the stage-file location is the fallback.
READS_FILE = "reads/reads.jsonl"
#: Output directory names this build may write.
OUT_NAME = re.compile(r"^genizah_novel_v\d+")
#: What a README may not carry: the legacy store's source key (only its public-safe label) and any host or storage path.
FORBIDDEN_README = (LEGACY_STORE_ORIGIN_PREFIX, "/volumes", "studio_offload", "localhost", "http://", "https://")
PRIVATE_BANNER = ("**PRIVATE. Never publish, push or release this dataset, its images-once export or anything derived "
                  "from them.**")
#: Ledger sources whose targets are OCR of scanned (typewritten) editions: each of their works needs an eye sheet.
TYPEWRITTEN_SOURCES = ("literature", "lit_run2")
#: Environment variable naming the private reviews file (``--reviews`` wins over it).
REVIEWS_ENV = "NOVEL_BUILD_REVIEWS"
#: How ``stats.json`` names the reviews file in its input digests (never its path).
REVIEWS_INPUT_NAME = "reviews.json"
#: A reviewer's verdicts on one (candidate, side); only ``ok`` approves.
REVIEW_VERDICTS = ("ok", "wrong_side", "wrong_image", "partial", "bad_ocr")
#: What a review policy does with a class of rows that has no approval: refuse them or admit them with a flag.
REVIEW_POLICY_VALUES = ("require", "flag")
#: Review classes and their default policy (no reviews file: scanned-edition works need approval, rule-recovered
#: placements are flagged).
DEFAULT_REVIEW_POLICY = {"typewritten_works": "require", "rule_recovered": "flag"}
#: Top-level keys of the reviews file (``required_works``: work keys that are ``require`` whatever the policy says).
REVIEW_FILE_KEYS = ("approved_works", "approved_rows", "required_works", "review_policy")
#: Fields of an ``approved_works`` entry (all required; ``note`` is optional) and of an ``approved_rows`` entry.
WORK_ENTRY_FIELDS = ("work", "reviewed_by", "date", "sheet")
ROW_ENTRY_FIELDS = ("row_id", "side_key", "verdict")
#: Entry fields that are matched by exact string: surrounding whitespace is refused (it would never match).
KEY_FIELDS = ("work", "row_id", "side_key")
EXIF_ORIENTATION = 0x0112
ROTATED_ORIENTATIONS = (5, 6, 7, 8)
#: ``images_once`` manifest keys copied into ``stats.json`` (counts only, no paths).
IMAGES_ONCE_KEYS = ("n_rows", "n_images", "image_bytes", "row_image_bytes", "dedupe_ratio", "splits")
#: Manifest columns, documented in the README.
PROVENANCE_COLUMNS = (
    ("stem", "row id (also the row's `stem`)"), ("row_id", "the ledger candidate"), ("side_key", "the target side"),
    ("source", "ledger source"), ("document_group", "ledger document group (sides and editions of one fragment)"),
    ("canonical_ids", "canonical shelf-mark ids"), ("pgpids", "PGP ids"),
    ("image_sha256", "sha256 of the photograph bytes"), ("image_path", "local file the row embeds"),
    ("image_url", "where the photograph was fetched from, if anywhere"), ("image_origin", "photograph origin"),
    ("target_pointer", "where the target text comes from (file, section / block / page, lines)"),
    ("side_basis", "the accepted side placement: rule, reader, recall, null p95/p99, margin, the read used"),
    ("script", "script measured on the target"), ("languages_stated", "the source's language statement"),
    ("verdict", "ledger novelty verdict (always `novel`)"), ("witnesses", "ledger witnesses (empty for novel)"),
    ("quality_flags", "non-fatal quality notes"), ("holdout_reasons", "hold-out reasons (empty for rows)"),
    ("split", "`train` or `val`"), ("letters", "letters in the target"), ("target_tokens", "measured tokens (0: not measured)"),
    ("rnl", "Russian National Library holder flag"), ("shelfmark", "the source's shelf mark"))


# ------------------------------------------------------------------------------------------------------------- helpers

def slug(text: str) -> str:
    """ASCII form of a key for stems: runs of other characters become ``_``; a short digest is appended when the key
    holds non-ASCII characters (which the slug drops).

    :param text: Key.
    :return: Slug.
    """
    out = re.sub(r"\W+", "_", text, flags=re.ASCII).strip("_")
    if not text.isascii():
        out = f"{out}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}".strip("_")
    return out


def row_stem(source: str, row_id: str, side_key: str) -> str:
    """Stem of a row: ``novel_<source>_<row id>_<side or page>_page``.

    :param source: Ledger source.
    :param row_id: Candidate id.
    :param side_key: Target side.
    :return: Stem.
    """
    return f"novel_{source}_{slug(row_id)}_{slug(side_key) or 'page'}_page"


def image_size(path: Path) -> Tuple[int, int]:
    """Upright pixel size of a photograph (EXIF orientation applied, as the image feature decodes it).

    :param path: Image file.
    :return: ``(width, height)``.
    """
    with PILImage.open(path) as image:
        width, height = image.size
        orientation = image.getexif().get(EXIF_ORIENTATION, 1)
    return (height, width) if orientation in ROTATED_ORIENTATIONS else (width, height)


def count_records(path: Path) -> int:
    """Non-empty lines of a JSON-lines file, without parsing them.

    :param path: File.
    :return: Record count.
    """
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def find_reads_file(work_dir: Path) -> Optional[Path]:
    """The reads file of a work dir.

    :param work_dir: The build's work dir.
    :return: ``reads/reads.jsonl``, else the stage2 reads file, else None.
    """
    for path in (work_dir / READS_FILE, paths.stage_path(work_dir, "reads")):
        if path.is_file():
            return path
    return None


def load_reads_index(path: Optional[Path]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Summaries of the reads, by photograph and reader (a later read of the same pair wins).

    :param path: Reads JSON lines (:class:`~.schema.ReadRecord`), or None.
    :return: ``sha256 -> reader -> {reader, finish, n_lines, letters_hebrew, letters_arabic, read_at, seconds}``.
    """
    index: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    if path is None:
        return index
    for rec in read_jsonl(path):
        index[rec["sha256"]][rec["reader"]] = {
            "reader": rec["reader"], "finish": rec.get("finish"), "n_lines": rec.get("n_lines"),
            "letters_hebrew": rec.get("letters_hebrew"), "letters_arabic": rec.get("letters_arabic"),
            "read_at": rec.get("read_at"), "seconds": rec.get("seconds")}
    return index


def find_read(index: Mapping[str, Mapping[str, Dict[str, Any]]], sha256: Optional[str],
              reader: Optional[str]) -> Optional[Dict[str, Any]]:
    """The read a side decision names.

    :param index: :func:`load_reads_index` result.
    :param sha256: The photograph.
    :param reader: The decision's reader: a full reader name, or a reader kind (``kraken``) that names exactly one
        reader of the photograph.
    :return: The read summary, or None.
    """
    by_reader = index.get(sha256 or "", {})
    if not reader:
        return None
    if reader in by_reader:
        return by_reader[reader]
    if ":" not in reader:
        hits = [read for name, read in by_reader.items() if name.split(":", 1)[0] == reader]
        if len(hits) == 1:
            return hits[0]
    return None


def pick_image(cand: Optional[Candidate], sha256: Optional[str],
               is_file: Callable[[Path], bool]) -> Optional[ImageRef]:
    """The found stage1 photograph with a given sha256 (one with a local file when several share it).

    :param cand: The stage1 candidate, or None.
    :param sha256: The accepted photograph.
    :param is_file: File-existence check.
    :return: The image, or None when no found image has that sha256.
    """
    hits = [image for image in funnel.found_images(cand) if sha256 and image.sha256 == sha256]
    with_file = [image for image in hits if image.path and is_file(Path(image.path))]
    return (with_file or hits or [None])[0]


def side_basis(decision: SideDecision, read: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The accepted side placement as the manifest's ``side_basis``.

    :param decision: The accepted decision.
    :param read: The read it used (:func:`find_read`).
    :return: Rule, reader, recall, null p95/p99, margin, reasons, coverage, competitor count and the read.
    """
    coverage = next((c.get("coverage") for c in decision.competitors if isinstance(c, dict) and c.get("coverage")),
                    None)
    return {"outcome": decision.outcome, "basis": decision.basis, "reader": decision.reader,
            "recall": decision.recall, "null_p95": decision.null_p95, "null_p99": decision.null_p99,
            "margin": decision.margin, "reasons": list(decision.reasons), "image_sha256": decision.image_sha256,
            "n_competitors": len(decision.competitors), "coverage": coverage, "read": read}


def build_gates(rec: Mapping[str, Any], image: Optional[ImageRef], read: Optional[Dict[str, Any]],
                result: QualityResult, is_file: Callable[[Path], bool]) -> List[str]:
    """The build's own checks on a side that cleared the hold-outs.

    :param rec: Its :func:`.funnel.trace` record.
    :param image: The accepted photograph (:func:`pick_image`).
    :param read: The read that placed the side (:func:`find_read`).
    :param result: Its quality result.
    :param is_file: File-existence check.
    :return: Reason codes (empty: the side becomes a row).
    """
    reasons: List[str] = []
    if rec["verdict"] != "novel":
        reasons.append("not_in_ledger" if rec["verdict"] == "not_in_ledger" else f"ledger_verdict_{rec['verdict']}")
    if rec["source"] not in LABEL_SOURCES:
        reasons.append("unknown_source")
    if image is None:
        reasons.append("accepted_image_not_in_stage1")
    elif not image.path:
        reasons.append("image_no_local_path")
    elif not is_file(Path(image.path)):
        reasons.append("image_file_missing")
    if read is None:
        reasons.append("side_read_missing")
    if not (result.text or "").strip():
        reasons.append("empty_text")
    return reasons


# ------------------------------------------------------------------------------------------------------- review gate

@dataclass(frozen=True)
class Reviews:
    """The human-review state of a build: what reviewers approved or rejected, and how strictly each class is gated.

    :param approved_works: ``<source>:<work>`` (see :func:`work_key`) -> its approval entry (``work``,
        ``reviewed_by``, ``date``, ``sheet``, optional ``note``): scanned-edition works whose OCR eye sheet was done.
    :param approved_rows: ``(row_id, side_key)`` -> its entry (``verdict`` one of :data:`REVIEW_VERDICTS`, optional
        ``note``).
    :param policy: ``typewritten_works`` and ``rule_recovered`` -> ``require`` (a row without an approval is excluded)
        or ``flag`` (admitted with a ``review_pending_*`` entry in its ``quality_flags``).
    :param required_works: ``<source>:<work>`` keys that are ``require`` whatever ``policy["typewritten_works"]``
        says (that policy then applies to the other scanned-edition works): the typewritten theses need their eye
        sheet, printed editions with a good OCR can be flagged.
    """

    approved_works: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    approved_rows: Mapping[Tuple[str, str], Mapping[str, str]] = field(default_factory=dict)
    policy: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_REVIEW_POLICY))
    required_works: FrozenSet[str] = field(default_factory=frozenset)

    def work_policy(self, key: str) -> str:
        """The policy that applies to a scanned-edition work.

        :param key: The work key (:func:`work_key`).
        :return: ``require`` for a work in ``required_works``, else the ``typewritten_works`` policy.
        """
        return "require" if key in self.required_works else self.policy["typewritten_works"]


def _work_key_problem(key: str, where: str) -> Optional[str]:
    """What is wrong with a work key as the reviews file spells it.

    :param key: The key (``<source>:<work>``).
    :param where: Its position, for the message.
    :return: A message, or None when the key reads ``<source>:<work>`` with a source of :data:`TYPEWRITTEN_SOURCES`
        and no surrounding whitespace.
    """
    source, _, name = key.partition(":")
    if key != key.strip():
        return f"{where}: {key!r} has leading or trailing whitespace (it would never match)"
    if source not in TYPEWRITTEN_SOURCES or not name:
        return f"{where}: {key!r} must read <source>:<work> with source one of {', '.join(TYPEWRITTEN_SOURCES)}"
    return None


def _entry_problems(entry: Mapping[str, Any], where: str, required: Sequence[str], optional: Sequence[str] = (),
                    may_be_empty: Sequence[str] = ()) -> List[str]:
    """Problems of one reviews-file entry: unknown or missing fields, non-string or blank values.

    :param entry: The entry.
    :param where: Its position, for the messages (``approved_rows[3]``).
    :param required: Fields it must hold.
    :param optional: Fields it may hold.
    :param may_be_empty: String fields that may be the empty string (an unlabelled side's ``side_key``).
    :return: One message per problem.
    """
    problems = [f"{where}: unknown field {name!r}" for name in entry if name not in (*required, *optional)]
    for name in (*required, *optional):
        if name not in entry:
            if name in required:
                problems.append(f"{where}: missing field {name!r}")
            continue
        value = entry[name]
        if not isinstance(value, str):
            problems.append(f"{where}.{name}: must be a string")
        elif not value.strip() and name not in may_be_empty:
            problems.append(f"{where}.{name}: must not be blank")
        elif name in KEY_FIELDS and value != value.strip():
            problems.append(f"{where}.{name}: {value!r} has leading or trailing whitespace (it would never match)")
    return problems


def parse_reviews(raw: Any) -> Reviews:
    """Validate and index the content of a reviews file.

    The file is one JSON object with four optional keys: ``approved_works`` (a list of ``{work, reviewed_by, date,
    sheet}`` objects, ``note`` optional; ``work`` reads ``<source>:<work>`` with a source of
    :data:`TYPEWRITTEN_SOURCES`, see :func:`work_key`), ``approved_rows`` (a list of ``{row_id, side_key, verdict}``
    objects, ``note`` optional; ``verdict`` one of :data:`REVIEW_VERDICTS`), ``required_works`` (a list of work keys
    that are ``require`` whatever ``typewritten_works`` says) and ``review_policy`` (``typewritten_works`` and
    ``rule_recovered``, each ``require`` or ``flag``; defaults :data:`DEFAULT_REVIEW_POLICY`).

    :param raw: The decoded JSON.
    :return: The reviews.
    :raises ValueError: Listing every problem found. A typo in a key, a verdict or a policy must not quietly turn a
        rejection into nothing, so unknown keys, unknown verdicts and duplicate entries are errors.
    """
    if not isinstance(raw, dict):
        raise ValueError("invalid reviews file: it must hold one JSON object")
    problems = [f"unknown top-level key {key!r} (known: {', '.join(REVIEW_FILE_KEYS)})"
                for key in raw if key not in REVIEW_FILE_KEYS]
    policy = dict(DEFAULT_REVIEW_POLICY)
    given = raw.get("review_policy", {})
    if not isinstance(given, dict):
        problems.append("review_policy: must be an object")
        given = {}
    for name, value in given.items():
        if name not in DEFAULT_REVIEW_POLICY:
            problems.append(f"review_policy: unknown class {name!r} (known: {', '.join(DEFAULT_REVIEW_POLICY)})")
        elif value not in REVIEW_POLICY_VALUES:
            problems.append(f"review_policy.{name}: {value!r} is not one of {', '.join(REVIEW_POLICY_VALUES)}")
        else:
            policy[name] = value
    listed: Dict[str, List[Any]] = {}
    for name in ("approved_works", "approved_rows", "required_works"):
        listed[name] = raw.get(name, [])
        if not isinstance(listed[name], list):
            problems.append(f"{name}: must be a list")
            listed[name] = []
    works: Dict[str, Dict[str, str]] = {}
    for i, entry in enumerate(listed["approved_works"]):
        where = f"approved_works[{i}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: must be an object with {', '.join(WORK_ENTRY_FIELDS)}")
            continue
        found = _entry_problems(entry, where, WORK_ENTRY_FIELDS, ("note",))
        problems += found
        if found:
            continue
        problem = _work_key_problem(entry["work"], f"{where}.work")
        if problem is not None:
            problems.append(problem)
        elif entry["work"] in works:
            problems.append(f"{where}.work: {entry['work']!r} is listed twice")
        else:
            works[entry["work"]] = dict(entry)
    required: List[str] = []
    for i, key in enumerate(listed["required_works"]):
        where = f"required_works[{i}]"
        problem = (_work_key_problem(key, where) if isinstance(key, str)
                   else f"{where}: must be a string <source>:<work>")
        if problem is not None:
            problems.append(problem)
        elif key in required:
            problems.append(f"{where}: {key!r} is listed twice")
        else:
            required.append(key)
    rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    for i, entry in enumerate(listed["approved_rows"]):
        where = f"approved_rows[{i}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: must be an object with {', '.join(ROW_ENTRY_FIELDS)}")
            continue
        found = _entry_problems(entry, where, ROW_ENTRY_FIELDS, ("note",), may_be_empty=("side_key", "note"))
        problems += found
        if found:
            continue
        key = (entry["row_id"], entry["side_key"])
        if entry["verdict"] not in REVIEW_VERDICTS:
            problems.append(f"{where}.verdict: {entry['verdict']!r} is not one of {', '.join(REVIEW_VERDICTS)}")
        elif key in rows:
            problems.append(f"{where}: ({key[0]!r}, {key[1]!r}) is listed twice")
        else:
            rows[key] = dict(entry)
    if problems:
        raise ValueError("invalid reviews file:\n  - " + "\n  - ".join(problems))
    return Reviews(approved_works=works, approved_rows=rows, policy=policy, required_works=frozenset(required))


def reviews_path(arg: Optional[Path]) -> Optional[Path]:
    """The reviews file to use: the ``--reviews`` argument, else the env var :data:`REVIEWS_ENV`, else none.

    :param arg: The ``--reviews`` argument.
    :return: The path, or None.
    """
    if arg is not None:
        return arg
    return Path(os.environ[REVIEWS_ENV]) if os.environ.get(REVIEWS_ENV) else None


def load_reviews(path: Optional[Path]) -> Reviews:
    """Read a reviews file.

    :param path: The file (see :func:`parse_reviews`), or None for no reviews: the default policy, nothing approved.
    :return: The reviews.
    :raises ValueError: When the file is missing or invalid (a configured file that cannot be used is an error, never
        silently "no approvals").
    """
    if path is None:
        return Reviews()
    if not path.is_file():
        raise ValueError(f"reviews file not found: {path}")
    return parse_reviews(json.loads(path.read_text(encoding="utf-8")))


def work_key(source: str, pointer: Mapping[str, Any]) -> str:
    """The work a side of a scanned-edition source belongs to, spelled as an ``approved_works`` entry spells it.

    ``literature``: ``literature:<work>/<part>`` from the target pointer's ``work`` and ``part`` fields (``work`` is
    the scanned-edition folder and ``part`` the file inside it, e.g. ``dissertations/<one thesis>``; ``work`` alone
    would merge several theses). ``lit_run2``: ``lit_run2:<doc_id without its last ':' segment>`` from the pointer's
    ``doc_id`` (``lit:sample_1970:29,30`` gives ``lit_run2:lit:sample_1970``; the last segment is the item inside the
    work). A pointer without those fields gives ``<source>:unknown``, which no sensible approval names.

    :param source: Ledger source.
    :param pointer: The side's target pointer.
    :return: The work key.
    """
    if source == "literature":
        name = "/".join(str(pointer[part]) for part in ("work", "part") if pointer.get(part))
    else:
        doc_id = str(pointer.get("doc_id") or "")
        name = doc_id.rsplit(":", 1)[0] if ":" in doc_id else doc_id
    return f"{source}:{name or 'unknown'}"


def review_gate(source: str, row_id: str, side_key: str, pointer: Mapping[str, Any],
                decision_reasons: Sequence[str], reviews: Reviews) -> Tuple[List[str], List[str]]:
    """The human-review gate for one side that passed every other check.

    * A verdict other than ``ok`` in ``approved_rows`` excludes the side with ``review_rejected:<verdict>``, whatever
      its class and the policy.
    * A side of a source in :data:`TYPEWRITTEN_SOURCES` whose work (:func:`work_key`) is not in ``approved_works`` is
      ``review_pending_work:<source>:<work>``.
    * A rule-recovered side (decision reasons ``spill_ratio_ok`` / ``conflict_ratio_ok``) with no ``approved_rows``
      entry at all is ``review_pending_row``. A row-level ``ok`` does not approve an unapproved work.

    A pending code excludes the side when the policy of its class is ``require`` and is returned as a flag (kept in
    ``quality_flags`` of the admitted row) when it is ``flag``. The policy of a scanned-edition work is
    :meth:`Reviews.work_policy`: ``require`` for the works in ``required_works``, else ``typewritten_works``.

    :param source: Ledger source.
    :param row_id: Candidate id.
    :param side_key: Target side.
    :param pointer: The side's target pointer (:func:`work_key`).
    :param decision_reasons: Reasons of the side's stage2 decision.
    :param reviews: The reviews.
    :return: ``(exclusion reasons, flags)``; empty exclusion reasons mean the side is admitted.
    """
    reasons: List[str] = []
    flags: List[str] = []
    entry = reviews.approved_rows.get((row_id, side_key))
    verdict = entry["verdict"] if entry is not None else None
    if verdict is not None and verdict != "ok":
        reasons.append(f"{funnel.REVIEW_REJECTED}:{verdict}")
    if source in TYPEWRITTEN_SOURCES:
        key = work_key(source, pointer)
        if key not in reviews.approved_works:
            (reasons if reviews.work_policy(key) == "require" else flags).append(f"{funnel.REVIEW_PENDING_WORK}:{key}")
    if verdict is None and funnel.is_rule_recovered(decision_reasons):
        (reasons if reviews.policy["rule_recovered"] == "require" else flags).append(funnel.REVIEW_PENDING_ROW)
    return reasons, flags


def review_violations(manifest: Sequence[RowRecord], reviews: Reviews) -> List[str]:
    """The review gate's acceptance rules, checked on the admitted rows themselves (not on the gate's code path).

    No admitted row of a class under policy ``require`` lacks its approval (a scanned-edition row: its work in
    ``approved_works``, where a work in ``required_works`` is ``require`` whatever ``typewritten_works`` says; a
    rule-recovered row: an ``ok`` entry in ``approved_rows``); no admitted row of a class under ``flag`` lacks its
    ``review_pending_*`` entry in ``quality_flags``; no admitted row has a verdict other than ``ok``.

    :param manifest: Row manifest.
    :param reviews: The reviews.
    :return: Violation messages (empty: acceptable).
    """
    unapproved_work: List[str] = []
    unapproved_row: List[str] = []
    rejected: List[str] = []
    unflagged: List[str] = []
    for m in manifest:
        entry = reviews.approved_rows.get((m.row_id, m.side_key))
        verdict = entry["verdict"] if entry is not None else None
        flags = set(m.quality_flags)
        if verdict is not None and verdict != "ok":
            rejected.append(m.stem)
        key = work_key(m.source, m.target_pointer) if m.source in TYPEWRITTEN_SOURCES else None
        if key is not None and key not in reviews.approved_works:
            if reviews.work_policy(key) == "require":
                unapproved_work.append(m.stem)
            elif f"{funnel.REVIEW_PENDING_WORK}:{key}" not in flags:
                unflagged.append(m.stem)
        if verdict is None and funnel.is_rule_recovered((m.side_basis or {}).get("reasons") or []):
            if reviews.policy["rule_recovered"] == "require":
                unapproved_row.append(m.stem)
            elif funnel.REVIEW_PENDING_ROW not in flags:
                unflagged.append(m.stem)
    return (_examples("scanned-edition row whose work is not in approved_works (policy require)", unapproved_work)
            + _examples("rule-recovered row without an ok entry in approved_rows (policy require)", unapproved_row)
            + _examples("row with a review verdict other than ok", rejected)
            + _examples("unreviewed row without its review_pending flag in quality_flags (policy flag)",
                        list(dict.fromkeys(unflagged))))


def unmatched_rejections(reviews: Reviews, data: funnel.StageData) -> List[str]:
    """Rejections that name no (candidate, side) of stage0: a mistyped key would leave the row in the dataset.

    A verdict other than ``ok`` only works when its ``(row_id, side_key)`` is exact, so one that matches nothing is a
    violation (an ``ok`` that matches nothing merely leaves its row pending).

    :param reviews: The reviews.
    :param data: Loaded stage data (all stage0 candidates, whatever ``--limit`` says).
    :return: ``[]`` or one violation message.
    """
    known = {(cand.row_id, target.side_key) for cand in data.candidates for target in cand.targets}
    missing = [f"{row_id}#{side_key}" for (row_id, side_key), entry in reviews.approved_rows.items()
               if entry["verdict"] != "ok" and (row_id, side_key) not in known]
    return ([f"review verdict names no (candidate, side) of stage0: {len(missing)}, e.g. {', '.join(missing[:3])}"]
            if missing else [])


def list_works(cands: Iterable[Candidate]) -> Dict[str, int]:
    """Sides per scanned-edition work of the stage0 candidates, spelled as ``approved_works`` / ``required_works`` do.

    :param cands: Stage0 candidates (:func:`work_key` reads their sides' pointers; other sources have no work).
    :return: ``work key -> sides``, most sides first, ties by key.
    """
    counts: Counter = Counter()
    for cand in cands:
        if cand.source in TYPEWRITTEN_SOURCES:
            for target in cand.targets:
                counts[work_key(cand.source, target.pointer)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def works_lines(counts: Mapping[str, int]) -> List[str]:
    """What ``--list-works`` prints: a header with the totals per source, then one line per work.

    :param counts: :func:`list_works` result.
    :return: Lines (``<sides>  <work key>``).
    """
    per_source: Dict[str, List[int]] = {source: [0, 0] for source in TYPEWRITTEN_SOURCES}
    for key, sides in counts.items():
        totals = per_source.setdefault(key.split(":", 1)[0], [0, 0])
        totals[0] += 1
        totals[1] += sides
    head = ", ".join(f"{source} {works} works / {sides} sides" for source, (works, sides) in per_source.items())
    return [f"scanned-edition works in stage0: {len(counts)} works, {sum(counts.values())} sides ({head})",
            "sides  work key (the string approved_works and required_works use)"
            ] + [f"{sides:>5}  {key}" for key, sides in counts.items()]


def unmatched_required_works(reviews: Reviews, data: funnel.StageData) -> List[str]:
    """Required works that name no work of stage0: a mistyped key would fall back to the looser policy.

    A work in ``required_works`` is ``require``; one that matches nothing silently leaves the intended work under
    ``typewritten_works`` (possibly ``flag``), so it is a violation.

    :param reviews: The reviews.
    :param data: Loaded stage data (all stage0 candidates, whatever ``--limit`` says).
    :return: ``[]`` or one violation message.
    """
    missing = sorted(reviews.required_works - set(list_works(data.candidates)))
    return ([f"required_works names no work of stage0: {len(missing)}, e.g. {', '.join(missing[:3])}"]
            if missing else [])


def review_stats(reviews: Reviews, manifest: Sequence[RowRecord]) -> Dict[str, Any]:
    """The review gate's part of ``stats.json``: the policy, what was approved or ruled on, and the unreviewed rows.

    :param reviews: The reviews.
    :param manifest: Row manifest.
    :return: ``{policy, required_works, approved_works, row_verdicts, rows_admitted_with_flags}`` (counts only, no
        keys or paths).
    """
    flagged = sum(1 for m in manifest if any(funnel.review_kind(flag) in ("pending_work", "pending_row")
                                             for flag in m.quality_flags))
    return {"policy": dict(reviews.policy), "required_works": len(reviews.required_works),
            "approved_works": len(reviews.approved_works),
            "row_verdicts": dict(Counter(entry["verdict"] for entry in reviews.approved_rows.values())),
            "rows_admitted_with_flags": flagged}


# ----------------------------------------------------------------------------------------------------------- assembly

def assemble_rows(work_dir: Path, *, pool: Path, data: Optional[funnel.StageData] = None,
                  reads_path: Optional[Path] = None, limit: Optional[int] = None,
                  is_file: Callable[[Path], bool] = Path.is_file, reviews: Optional[Reviews] = None
                  ) -> Tuple[List[Dict[str, Any]], List[RowRecord], List[Dict[str, Any]]]:
    """Join the stage files into rows, their manifest and one decision per (candidate, side).

    A side that cleared the hold-outs and the build's own checks meets the review gate last (:func:`review_gate`):
    its review exclusions are appended to the decision's ``reasons`` (the side stays at stage ``holdout_cleared``),
    and a row admitted under policy ``flag`` carries the review flags in ``quality_flags``.

    :param work_dir: The build's work dir.
    :param pool: The novelty ledger.
    :param data: Stage data already loaded (default: load from ``work_dir`` and ``pool``).
    :param reads_path: Reads file (default: :func:`find_reads_file`).
    :param limit: Only the first N stage0 candidates.
    :param is_file: File-existence check (tests stub it).
    :param reviews: Approvals and policy (default: no reviews file, i.e. the default policy and nothing approved).
    :return: ``(rows, manifest, decisions)``: rows in the KTIV ``FEATURES`` schema, one :class:`RowRecord` per row
        in the same order, and the decision records (the :func:`.funnel.trace` fields plus ``stem``,
        ``label_source``, ``image_origin`` and ``shelfmark``; stage ``row`` for rows).
    :raises ValueError: When two rows would get the same stem.
    """
    data = data if data is not None else funnel.load_stage_data(work_dir, pool, build_outputs=False)
    reviews = reviews if reviews is not None else Reviews()
    reads = load_reads_index(reads_path if reads_path is not None else find_reads_file(work_dir))
    cands = {cand.row_id: cand for cand in data.candidates}
    stage1 = {cand.row_id: cand for cand in data.images}
    sides = {(d.row_id, d.side_key): d for d in data.sides}
    quality = {(q.row_id, q.side_key): q for q in data.quality}
    rows: List[Dict[str, Any]] = []
    manifest: List[RowRecord] = []
    decisions: List[Dict[str, Any]] = []
    for rec in funnel.trace(data, limit=limit):
        cand = cands[rec["row_id"]]
        decision = {**rec, "stem": None, "label_source": LABEL_SOURCES.get(rec["source"]), "image_origin": None,
                    "shelfmark": cand.shelfmark}
        decisions.append(decision)
        if rec["stage"] != "holdout_cleared":
            continue
        key = (rec["row_id"], rec["side_key"])
        side, result = sides[key], quality[key]
        image = pick_image(stage1.get(rec["row_id"]), side.image_sha256, is_file)
        read = find_read(reads, side.image_sha256, side.reader)
        decision["image_origin"] = image.origin if image is not None else None
        gates = build_gates(rec, image, read, result, is_file)
        target = next(t for t in cand.targets if t.side_key == rec["side_key"])
        review_reasons: List[str] = []
        review_flags: List[str] = []
        if not gates:         # the review gate is the last one: only sides that would be rows wait on a reviewer
            review_reasons, review_flags = review_gate(rec["source"], rec["row_id"], rec["side_key"], target.pointer,
                                                       side.reasons, reviews)
        if gates or review_reasons:
            decision["reasons"] = gates + review_reasons
            continue
        stem = row_stem(rec["source"], rec["row_id"], rec["side_key"])
        text = result.text.strip()
        width, height = image_size(Path(image.path))
        tokens = result.tokens or 0
        rows.append({"image": image.path, "question": SCRIPT_NEUTRAL_PROMPT, "answer": text, "task": TASK,
                     "section": SECTION, "stem": stem, "label_source": LABEL_SOURCES[rec["source"]],
                     "target_chars": len(text), "target_tokens": tokens, "image_width": width,
                     "image_height": height})
        manifest.append(RowRecord(
            stem=stem, row_id=rec["row_id"], side_key=rec["side_key"], source=rec["source"],
            document_group=rec["document_group"], canonical_ids=list(cand.canonical_ids), pgpids=list(cand.pgpids),
            image_sha256=side.image_sha256, image_path=image.path, image_url=image.url, image_origin=image.origin,
            target_pointer=dict(target.pointer), side_basis=side_basis(side, read), script=result.script,
            languages_stated=cand.languages_stated, verdict=rec["verdict"], witnesses=list(cand.witnesses),
            quality_flags=list(dict.fromkeys([*result.reasons, *review_flags])),
            holdout_reasons=list(rec["holdout_reasons"]), split=rec["split"],
            letters=result.letters_hebrew + result.letters_arabic, target_tokens=tokens))
        decision.update(stage="row", stem=stem)
    duplicates = sorted(stem for stem, n in Counter(row["stem"] for row in rows).items() if n > 1)
    if duplicates:
        raise ValueError(f"{len(duplicates)} stem(s) would be written twice, e.g. {duplicates[:3]}")
    return rows, manifest, decisions


def rows_by_split(rows: Sequence[Dict[str, Any]], manifest: Sequence[RowRecord]) -> Dict[str, List[Dict[str, Any]]]:
    """Rows grouped into the dataset's splits.

    :param rows: Rows.
    :param manifest: Their manifest records (same order).
    :return: ``{"train_page": [...], "val": [...]}``.
    """
    out: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SPLIT_NAMES.values()}
    for row, record in zip(rows, manifest):
        out[SPLIT_NAMES[record.split]].append(row)
    return out


def row_decisions(decisions: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    """Decision records of the rows, by stem.

    :param decisions: Decision records.
    :return: ``stem -> record``.
    """
    return {d["stem"]: d for d in decisions if d.get("stage") == "row" and d.get("stem")}


def privacy_reasons(manifest: Sequence[RowRecord], decisions: Sequence[Mapping[str, Any]]) -> List[str]:
    """Why the dataset must stay private: legacy-store photographs or transcriptions, RNL documents.

    :param manifest: Row manifest.
    :param decisions: Decision records (they carry the RNL holder flag).
    :return: One sentence per reason (empty: nothing forces privacy).
    """
    by_stem = row_decisions(decisions)
    store_images = sum(1 for m in manifest if (m.image_origin or "").startswith(LEGACY_STORE_ORIGIN_PREFIX))
    store_text = sum(1 for m in manifest if m.source in LEGACY_STORE_SOURCES)
    rnl = sum(1 for m in manifest if (by_stem.get(m.stem) or {}).get("rnl"))
    reasons = []
    if store_images:
        reasons.append(f"{store_images} row(s) use photographs from the project's legacy image store")
    if store_text:
        reasons.append(f"{store_text} row(s) carry transcriptions from the project's legacy store")
    if rnl:
        reasons.append(f"{rnl} row(s) hold Russian National Library documents (training only, never public)")
    return reasons


def manifest_lines(manifest: Sequence[RowRecord], decisions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Manifest records as written: the :class:`RowRecord` fields plus ``rnl`` and ``shelfmark``.

    :param manifest: Row manifest.
    :param decisions: Decision records (source of the two extra fields).
    :return: Dicts, one per row.
    """
    by_stem = row_decisions(decisions)
    return [{**to_dict(m), "rnl": bool((by_stem.get(m.stem) or {}).get("rnl")),
             "shelfmark": (by_stem.get(m.stem) or {}).get("shelfmark")} for m in manifest]


# --------------------------------------------------------------------------------------------------------- acceptance

def _examples(label: str, stems: Sequence[str]) -> List[str]:
    """One violation message for a list of offending rows.

    :param label: What is wrong.
    :param stems: Offending stems.
    :return: ``[]`` or one message with the count and up to three stems.
    """
    return [f"{label}: {len(stems)} row(s), e.g. {', '.join(stems[:3])}"] if stems else []


def acceptance_checks(rows: Sequence[Dict[str, Any]], manifest: Sequence[RowRecord],
                      decisions: Sequence[Mapping[str, Any]], counts: Mapping[str, Any],
                      reviews: Optional[Reviews] = None) -> List[str]:
    """The acceptance rules the build enforces; the build writes nothing while any is violated.

    Every row has an image sha256, a side basis (an accepted decision with its rule, reader and the read of the same
    photograph) and a hold-out result (split train or val) without reasons; the ledger verdict is ``novel``; the
    question is ``SCRIPT_NEUTRAL_PROMPT``; no answer holds a hygiene character (:data:`HYGIENE_CHARS`); no measured
    token count exceeds :data:`TOKEN_BUDGET`; stems are unique; validation rows share no document group with training
    rows; every (candidate, side) that is not a row carries a reason; the stage counts are monotone; and the review
    gate holds (:func:`review_violations`): where the review policy requires a class, no admitted row of that class
    lacks its approval.

    :param rows: Rows.
    :param manifest: Their manifest (same order).
    :param decisions: Decision records.
    :param counts: :func:`.funnel.stage_counts` of the stage data with this manifest as rows.
    :param reviews: Approvals and policy (default: no reviews file, i.e. the default policy and nothing approved).
    :return: Violation messages (empty: acceptable).
    """
    out: List[str] = []
    if not rows:
        out.append("no rows assembled")
    if [row["stem"] for row in rows] != [m.stem for m in manifest]:
        out.append("rows and manifest do not list the same stems in the same order")
    by_stem = row_decisions(decisions)
    out += _examples("no image sha256", [m.stem for m in manifest if not m.image_sha256])
    out += _examples("no side basis (accepted decision with rule, reader and a read of the row's photograph)",
                     [m.stem for m in manifest if not _has_side_basis(m)])
    out += _examples("no hold-out result (split train or val)", [m.stem for m in manifest
                                                                 if m.split not in funnel.ROW_SPLITS])
    out += _examples("hold-out reasons on a row", [m.stem for m in manifest if m.holdout_reasons])
    out += _examples("ledger verdict is not novel", [m.stem for m in manifest if m.verdict != "novel"])
    out += _examples("no decision record", [m.stem for m in manifest if m.stem not in by_stem])
    out += _examples("question is not SCRIPT_NEUTRAL_PROMPT",
                     [row["stem"] for row in rows if row.get("question") != SCRIPT_NEUTRAL_PROMPT])
    out += _examples(f"answer holds a hygiene character ({HYGIENE_CHARS})",
                     [row["stem"] for row in rows if any(ch in HYGIENE_CHARS for ch in row.get("answer") or "")])
    out += _examples(f"measured tokens over the budget {TOKEN_BUDGET}",
                     sorted({row["stem"] for row in rows if (row.get("target_tokens") or 0) > TOKEN_BUDGET}
                            | {m.stem for m in manifest if (m.target_tokens or 0) > TOKEN_BUDGET}))
    duplicates = sorted({stem for stem, n in Counter(row["stem"] for row in rows).items() if n > 1}
                        | {stem for stem, n in Counter(m.stem for m in manifest).items() if n > 1})
    out += _examples("stem not unique", duplicates)
    train_groups = {m.document_group for m in manifest if m.split == "train"}
    out += _examples("validation row shares a document group with training rows",
                     [m.stem for m in manifest if m.split == "val" and m.document_group in train_groups])
    silent = [f"{d.get('row_id')}#{d.get('side_key')}" for d in decisions
              if d.get("stage") != "row" and not d.get("reasons")]
    if silent:
        out.append(f"excluded without a reason: {len(silent)} (candidate, side), e.g. {', '.join(silent[:3])}")
    out += [f"stage counts not monotone: {v}" for v in funnel.monotone_violations(counts)]
    out += review_violations(manifest, reviews if reviews is not None else Reviews())
    return out


def _has_side_basis(record: RowRecord) -> bool:
    """True when a manifest record carries a complete side basis.

    :param record: Manifest record.
    :return: Accepted, with a rule, a reader, a read and the row's own photograph.
    """
    basis = record.side_basis or {}
    return (basis.get("outcome") == "accepted" and bool(basis.get("basis")) and bool(basis.get("reader"))
            and bool(basis.get("read")) and basis.get("image_sha256") == record.image_sha256)


# ----------------------------------------------------------------------------------------------- stats, README, writing

def build_stats(name: str, manifest: Sequence[RowRecord], decisions: Sequence[Mapping[str, Any]],
                counts: Mapping[str, Any], inputs: Sequence[Mapping[str, Any]], private_reasons: Sequence[str], *,
                limit: Optional[int] = None, built: Optional[str] = None,
                reviews: Optional[Reviews] = None) -> Dict[str, Any]:
    """``stats.json`` of a build (source names shown with their public-safe labels; no paths).

    :param name: Dataset directory name.
    :param manifest: Row manifest.
    :param decisions: Decision records.
    :param counts: :func:`.funnel.stage_counts` with the manifest as rows.
    :param inputs: Input digest records (names relative).
    :param private_reasons: :func:`privacy_reasons`.
    :param limit: The candidate limit of the run, if any.
    :param built: Build time (default: now, UTC).
    :param reviews: Approvals and policy (default: no reviews file, i.e. the default policy and nothing approved).
    :return: Statistics.
    """
    per: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(Counter))
    groups: Dict[str, set] = defaultdict(set)
    for m in manifest:
        label = funnel.source_label(m.source)
        per[label][m.script][SPLIT_NAMES.get(m.split, m.split)] += 1
        groups[label].add(m.document_group)
    reasons = Counter(funnel.reason_code(reason) for d in decisions if d.get("stage") != "row"
                      for reason in dict.fromkeys(d.get("reasons") or []))
    return {"name": name, "built": built or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "task": TASK, "section": SECTION, "prompt_name": PROMPT_NAME, "token_budget": TOKEN_BUDGET,
            "rows": dict(Counter(SPLIT_NAMES.get(m.split, m.split) for m in manifest)),
            "row_groups": len({m.document_group for m in manifest}),
            "row_letters": sum(m.letters for m in manifest),
            "rows_by_source_script_split": {src: {scr: dict(c) for scr, c in by_scr.items()} for src, by_scr in per.items()},
            "groups_by_source": {src: len(g) for src, g in groups.items()},
            "label_sources": dict(Counter(LABEL_SOURCES.get(m.source, "unknown") for m in manifest)),
            "decisions_by_stage": dict(Counter(d.get("stage") for d in decisions)),
            "exclusion_reasons": dict(reasons.most_common()),
            "funnel": {"total": counts["total"],
                       "by_source": {funnel.source_label(src): cells for src, cells in counts["by_source"].items()}},
            "review": review_stats(reviews if reviews is not None else Reviews(), manifest),
            "inputs": [dict(i) for i in inputs], "limit": limit, "private": bool(private_reasons),
            "private_reasons": list(private_reasons), "acceptance": {"violations": []}}


def check_public_text(text: str) -> None:
    """Refuse text naming the legacy store's other names, hosts or storage paths.

    :param text: README or notice text.
    :raises ValueError: When a forbidden word or location appears.
    """
    hits = [word for word in FORBIDDEN_README if word in text.lower()]
    if hits:
        raise ValueError(f"README text would name {hits}")


def readme_text(stats: Mapping[str, Any]) -> str:
    """``README.md`` of the dataset (PRIVATE banner first when private).

    :param stats: :func:`build_stats` result.
    :return: Markdown.
    :raises ValueError: When the text would name something :func:`check_public_text` refuses.
    """
    rows = stats["rows"]
    review = stats["review"]
    lines = [f"# {stats['name']}", ""]
    if stats.get("private_reasons"):
        lines += [f"{PRIVATE_BANNER} {'; '.join(stats['private_reasons'])}.", ""]
    lines += [f"Page-transcription rows of manuscript sides whose documents no earlier fine-tune trained on (the novelty "
              f"ledger's `novel` verdict; no replay of earlier data). Built by "
              f"`src/finetuning/qwen_hebrew/novel_build/build.py` on {stats['built']}. Every row: question "
              f"`{stats['prompt_name']}` (script neutral), task `{stats['task']}`, section `{stats['section']}`, "
              "`label_source` naming the edition the target comes from.", "",
              f"- rows: train_page {rows.get('train_page', 0)}, val {rows.get('val', 0)} ({stats['row_groups']} "
              f"document groups, {stats['row_letters']} letters)",
              "- each side was placed on its photograph by reads (never by a file name or a canvas order), passed the "
              "quality gates and the benchmark hold-outs; validation rows share no document group with training rows",
              f"- human review gate: scanned-edition works `{review['policy']['typewritten_works']}`, rule-recovered "
              f"placements `{review['policy']['rule_recovered']}` (`require`: no row without a recorded approval; "
              f"`flag`: admitted with a `review_pending_*` entry in `quality_flags`)"
              + (f"; {review['required_works']} named work(s) are always `require`" if review["required_works"] else "")
              + f"; {review['rows_admitted_with_flags']} row(s) carry such a flag",
              "", "## Rows per source, script and split", ""]
    table = [[src, scr, by_split.get("train_page", 0), by_split.get("val", 0)]
             for src, by_scr in sorted(stats["rows_by_source_script_split"].items())
             for scr, by_split in sorted(by_scr.items())]
    lines += ["| source | script | train_page | val |", "|---|---|---:|---:|"]
    lines += [f"| {src} | {scr} | {train} | {val} |" for src, scr, train, val in table]
    lines += ["", "Document groups per source: " + ", ".join(f"{src} {n}" for src, n in
                                                             sorted(stats["groups_by_source"].items())), "",
              "## Funnel", "",
              "`FUNNEL.md` in the build's work dir (regenerated after every build; "
              "`python -m src.finetuning.qwen_hebrew.novel_build.funnel`) counts candidates, sides and document groups "
              "per source and script at every stage, from the novelty ledger through the stage files to these rows, "
              "with the drop reasons per stage and the review queue. Totals (sides from side verified on; "
              "document groups in parentheses):", "",
              "| stage | count |", "|---|---:|"]
    total = stats["funnel"]["total"]
    for stage in funnel.STAGES:
        if stage.rendered:
            unit = "candidates" if stage.level == "ledger" or stage.level == "target" else "sides"
            lines.append(f"| {stage.label} | {total[stage.key][unit]} ({total[stage.key]['groups']}) |")
    lines += ["", "## Provenance", "", "`manifest.jsonl`, one record per row:", ""]
    lines += [f"- `{name}`: {text}" for name, text in PROVENANCE_COLUMNS]
    lines += ["", "`decisions.jsonl`: one record per (candidate, side) with the last stage it reached (`row` for rows) "
                  "and the reasons it stopped there. `stats.json`: every count above, input file digests, exclusion "
                  "reasons."]
    text = "\n".join(lines) + "\n"
    check_public_text(text)
    return text


def private_text(reasons: Sequence[str]) -> str:
    """``PRIVATE.md`` notice.

    :param reasons: :func:`privacy_reasons` (may be empty when the build was made private by choice).
    :return: Markdown.
    """
    text = PRIVATE_BANNER + "\n\n" + ("\n".join(f"- {reason}" for reason in reasons) + "\n" if reasons else "")
    check_public_text(text)
    return text


def ensure_banner(readme: str, reasons: Sequence[str]) -> str:
    """Add the PRIVATE banner under the README title when it is missing.

    :param readme: README text.
    :param reasons: Privacy reasons.
    :return: README with the banner.
    """
    if PRIVATE_BANNER in readme:
        return readme
    title, _, rest = readme.partition("\n")
    return f"{title}\n\n{PRIVATE_BANNER} {'; '.join(reasons)}\n{rest}"


def write_dataset(rows_by_split: Mapping[str, List[Dict[str, Any]]], manifest: Sequence[RowRecord],
                  decisions: Sequence[Mapping[str, Any]], out_dir: Path, *, export: bool, private: bool,
                  stats: Dict[str, Any], readme: str) -> Dict[str, Any]:
    """Write the dataset, its sidecars, manifest and decisions, and (with ``export``) the images-once copy.

    ``build_pgp_editions.save_dataset`` writes the splits (each row's ``image`` path is embedded by the image feature)
    with ``README.md``, ``PRIVATE.md`` (when private) and ``stats.json`` as sidecars, replacing an earlier build
    atomically; ``manifest.jsonl`` and ``decisions.jsonl`` follow. A legacy-store or RNL row forces ``private``.
    The export goes to ``<out_dir>_images_once`` (an earlier export there is removed first) and gets the README and
    the PRIVATE notice too; ``stats.json`` then records the export's counts.

    :param rows_by_split: Dataset split name -> rows.
    :param manifest: Row manifest.
    :param decisions: Decision records.
    :param out_dir: Dataset directory.
    :param export: Write the images-once export.
    :param private: Mark the dataset private (forced when :func:`privacy_reasons` finds a reason).
    :param stats: :func:`build_stats` result.
    :param readme: :func:`readme_text` result.
    :return: The statistics as written.
    """
    reasons = privacy_reasons(manifest, decisions)
    private = private or bool(reasons)
    stats = {**stats, "private": private, "private_reasons": reasons}
    readme = ensure_banner(readme, reasons) if private else readme
    check_public_text(readme)
    sidecars = {"README.md": readme, "stats.json": json.dumps(stats, ensure_ascii=False, indent=1)}
    if private:
        sidecars["PRIVATE.md"] = private_text(reasons)
    save_dataset({name: list(rows) for name, rows in rows_by_split.items()}, out_dir, sidecars)
    write_jsonl(out_dir / "manifest.jsonl", manifest_lines(manifest, decisions))
    write_jsonl(out_dir / "decisions.jsonl", decisions)
    if export and any(rows_by_split.values()):
        export_dir = out_dir.with_name(out_dir.name + "_images_once")
        if export_dir.exists():
            shutil.rmtree(export_dir)       # a stale export would keep images of rows an earlier build exported
        exported = export_images_once(out_dir, export_dir)
        for name in ("README.md", "PRIVATE.md"):
            if name in sidecars:
                (export_dir / name).write_text(sidecars[name], encoding="utf-8")
        stats["images_once"] = {key: exported[key] for key in IMAGES_ONCE_KEYS if key in exported}
        text = json.dumps(stats, ensure_ascii=False, indent=1)
        (out_dir / "stats.json").write_text(text, encoding="utf-8")
        (export_dir / "stats.json").write_text(text, encoding="utf-8")
    return stats


def out_dir_problems(out: Path, work_dir: Path) -> List[str]:
    """Why an output directory may not be written: it overlaps the work dir, or it is not ``genizah_novel_v<N>``.

    :param out: Dataset directory.
    :param work_dir: The build's work dir.
    :return: Problems (empty: writable).
    """
    o, w = out.resolve(), work_dir.resolve()
    problems = []
    if o == w or w in o.parents or o in w.parents:
        problems.append(f"output {out} overlaps the work dir {work_dir}")
    if not OUT_NAME.match(out.name):
        problems.append(f"output {out.name} is not a genizah_novel_v<N> directory")
    return problems


def check_out_dir(out: Path, work_dir: Path) -> None:
    """Refuse an output that overlaps the work dir or is not a ``genizah_novel_v<N>`` directory.

    :param out: Dataset directory.
    :param work_dir: The build's work dir.
    :raises ValueError: With :func:`out_dir_problems`.
    """
    problems = out_dir_problems(out, work_dir)
    if problems:
        raise ValueError("; ".join(problems))


def summary_lines(rows: Sequence[Dict[str, Any]], manifest: Sequence[RowRecord],
                  decisions: Sequence[Mapping[str, Any]], counts: Mapping[str, Any],
                  private_reasons: Sequence[str], reviews: Optional[Reviews] = None) -> List[str]:
    """What a run prints: rows, decisions by stage, funnel totals, the review gate, privacy.

    :param rows: Rows.
    :param manifest: Row manifest.
    :param decisions: Decision records.
    :param counts: Stage counts.
    :param private_reasons: :func:`privacy_reasons`.
    :param reviews: Approvals and policy (default: no reviews file, i.e. the default policy and nothing approved).
    :return: Lines.
    """
    splits = Counter(SPLIT_NAMES.get(m.split, m.split) for m in manifest)
    total = counts["total"]
    chain = " -> ".join(
        f"{stage.label} {total[stage.key]['candidates' if stage.level in ('ledger', 'target') else 'sides']}"
        f" ({total[stage.key]['groups']})" for stage in funnel.STAGES if stage.rendered)
    review = funnel.review_counts(decisions, manifest)
    reviews = reviews if reviews is not None else Reviews()
    policy, gate, works = reviews.policy, review["total"], review["pending_works"]
    named = f", {len(reviews.required_works)} required works" if reviews.required_works else ""
    return [f"rows: {len(rows)} (train_page {splits.get('train_page', 0)}, val {splits.get('val', 0)}; "
            f"{len({m.document_group for m in manifest})} document groups)",
            f"decisions by stage: {dict(Counter(d.get('stage') for d in decisions))}",
            f"funnel: {chain}",
            f"review (typewritten works {policy['typewritten_works']}{named}, rule-recovered "
            f"{policy['rule_recovered']}): "
            f"pending work {gate['pending_work']} sides ({len(works)} works), pending row {gate['pending_row']}, "
            f"rejected {gate['rejected']}, admitted with flags {gate['admitted_flagged']}; rule-recovered admitted "
            f"{gate['rule_admitted']}, excluded {gate['rule_excluded']} ({gate['rule_excluded_by_review']} by review)",
            "private: " + ("; ".join(private_reasons) if private_reasons else "nothing forces privacy")]


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: assemble, check and write the dataset (or only print, with ``--dry-run``; or list the scanned-edition work
    keys of stage0, with ``--list-works``).

    Exits with status 1 when any acceptance check fails (nothing is written then).

    :param argv: Arguments (default: ``sys.argv``).
    :raises ValueError: When ``--list-works`` finds no stage0 file, or the reviews file is missing or invalid.
    """
    parser = argparse.ArgumentParser(description="Assemble the novel-rows dataset from the stage files, check the "
                                                 "acceptance rules and write it.")
    parser.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR, help="The build's work dir.")
    parser.add_argument("--out", type=Path, default=paths.DEFAULT_OUT_DIR, help="Dataset directory to write.")
    parser.add_argument("--pool", type=Path, default=paths.DEFAULT_POOL, help="The novelty ledger.")
    parser.add_argument("--no-export", action="store_true", help="Skip the images-once export.")
    parser.add_argument("--dry-run", action="store_true", help="Print the counts and the violations; write nothing.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N stage0 candidates.")
    parser.add_argument("--public", action="store_true",
                        help="A build meant for release: refuse when any row would use a legacy-store photograph or "
                             "transcription or hold a Russian National Library document.")
    parser.add_argument("--reviews", type=Path, default=None,
                        help="Private JSON file with the human-review state: approved_works, approved_rows, "
                             "required_works, review_policy "
                             f"(env {REVIEWS_ENV}; default none: scanned-edition works need approval "
                             "and nothing is approved). Never commit it.")
    parser.add_argument("--list-works", action="store_true",
                        help="Print every scanned-edition work key of the work dir's stage0 candidates with its side "
                             "count (the keys approved_works and required_works use), then exit.")
    args = parser.parse_args(argv)
    if args.list_works:
        stage0 = paths.stage_path(args.work_dir, "candidates")
        if not stage0.is_file():
            raise ValueError(f"stage0 candidates not found: {stage0}")
        for line in works_lines(list_works(read_jsonl(stage0, Candidate))):
            print(line)
        return
    reviews_file = reviews_path(args.reviews)
    reviews = load_reviews(reviews_file)
    data = funnel.load_stage_data(args.work_dir, args.pool, build_outputs=False)
    reads_path = find_reads_file(args.work_dir)
    rows, manifest, decisions = assemble_rows(args.work_dir, pool=args.pool, data=data, reads_path=reads_path,
                                              limit=args.limit, reviews=reviews)
    counts = funnel.stage_counts(replace(data, rows=list(manifest)))
    violations = (acceptance_checks(rows, manifest, decisions, counts, reviews) + unmatched_rejections(reviews, data)
                  + unmatched_required_works(reviews, data) + out_dir_problems(args.out, args.work_dir))
    private = privacy_reasons(manifest, decisions)
    if args.public:
        violations += [f"public build refused: {reason}" for reason in private]
    for line in summary_lines(rows, manifest, decisions, counts, private, reviews):
        print(line)
    print("violations: " + ("none" if not violations else str(len(violations))))
    for violation in violations:
        print(f"  - {violation}")
    if args.dry_run:
        if violations:
            raise SystemExit(1)
        return
    if violations:
        print("refusing to write: fix the violations above")
        raise SystemExit(1)
    check_out_dir(args.out, args.work_dir)
    inputs = list(data.inputs)
    if reads_path is not None:
        inputs.append(funnel.input_record(reads_path, reads_path.relative_to(args.work_dir).as_posix(),
                                          "reads (side-placement evidence)", count_records(reads_path)))
    if reviews_file is not None:
        inputs.append(funnel.input_record(reviews_file, REVIEWS_INPUT_NAME,
                                          "private reviews file (approved and required works, row verdicts, policy)",
                                          len(reviews.approved_works) + len(reviews.approved_rows)
                                          + len(reviews.required_works)))
    stats = build_stats(args.out.name, manifest, decisions, counts, inputs, private, limit=args.limit,
                        reviews=reviews)
    stats = write_dataset(rows_by_split(rows, manifest), manifest, decisions, args.out, export=not args.no_export,
                          private=bool(private), stats=stats, readme=readme_text(stats))
    write_jsonl(paths.stage_path(args.work_dir, "rows"), manifest_lines(manifest, decisions))
    write_jsonl(args.work_dir / funnel.DECISIONS_FILE, decisions)
    funnel_md = args.work_dir / funnel.FUNNEL_FILE
    funnel_md.write_text(funnel.render(funnel.compute_funnel(args.work_dir, args.pool)), encoding="utf-8")
    print(f"wrote {args.out} ({sum(stats['rows'].values())} rows, private={stats['private']}), the stage6 files and "
          f"{funnel_md}")


if __name__ == "__main__":
    main()
