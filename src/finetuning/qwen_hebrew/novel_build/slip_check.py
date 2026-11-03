"""Shelf-mark slip check of legacy-store photographs filed under a shelf-mark-named file.

Legacy-store photographs whose file is named by a shelf mark (the first ``_``-separated part of the file name is not
all digits) were misfiled at about 8 % in an earlier pilot, while photographs named by a PGP id were not. For every
such photograph in stage1 without a slip outcome, a VLM on LM Studio reads the shelf-mark label or slip in the
photograph (:data:`SLIP_PROMPT`, through :func:`src.models.ocr.lms_transcriber.transcribe_with_lm_studio_meta`,
which holds the LM Studio lock and the yield gate) and the answer is compared with the shelf mark and the canonical
ids the photograph is filed under, through the shelf-mark facade (:func:`~src.datasets.shelfmarks.loose_key`
equality, or the answer being one of the filed mark's :func:`~src.datasets.shelfmarks.variants`).

Outcomes: ``match`` | ``mismatch`` | ``no_label`` | ``unreadable`` | ``error`` (the LM Studio call failed; retried
on the next run). Results go to ``<work dir>/slip_checks.jsonl`` (one record per photograph sha256, appended as they
come, resumable); every ``mismatch`` and ``unreadable`` goes to the eye queue (``slip_eye_queue.jsonl`` and
``slip_eye_queue.md``). Nothing is dropped, moved or relabelled here. Run it alone: one LM Studio consumer at a time.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse

from src.datasets import shelfmarks as shelf
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, ImageRef, read_jsonl, write_jsonl
from src.models.ocr import lms_transcriber

SLIP_PROMPT = ("Read the shelf-mark label or slip visible in this photograph, if any. Answer with the shelf mark "
               "exactly as written, or NONE.")
PROMPT_NAME = "slip_label_v1"
PROMPT_SHA1 = hashlib.sha1(SLIP_PROMPT.encode("utf-8")).hexdigest()
#: Image origins of the legacy store.
LEGACY_STORE_ORIGINS = ("fjp_local", "fjp_bucket")
OUTCOMES = ("match", "mismatch", "no_label", "unreadable", "error")
#: Outcomes listed in the eye queue.
EYE_OUTCOMES = ("mismatch", "unreadable")
CHECKS_FILE = "slip_checks.jsonl"
EYE_QUEUE_JSONL = "slip_eye_queue.jsonl"
EYE_QUEUE_MD = "slip_eye_queue.md"
STOP_FILE = "slip_check.stop"
MAX_CONSECUTIVE_ERRORS = 5
DEFAULT_MAX_TOKENS = 256
PROGRESS_EVERY = 25
_ANSWER_PREFIX = re.compile(r"^(?:the\s+)?(?:shelf[\s-]?mark|label|slip|answer)(?:\s+(?:is|reads))?\s*[:：]?\s*", re.I)
_NO_LABEL = re.compile(r"^(?:none|nothing|n/a|no)\b", re.I)
_QUOTES = "`*\"'“”‘’ \t"

#: ``transcribe(model_key, image_path, prompt)`` -> LM Studio completion meta (``text``, ``finish_reason``, ...) or
#: None when the call failed.
Transcribe = Callable[[str, str, str], Optional[Dict[str, Any]]]


@dataclass
class SlipItem:
    """One photograph to check, with every filing it appears under.

    :param sha256: Photograph sha256.
    :param path: Local file.
    :param url: Remote URL, if any.
    :param origin: Image origin.
    :param file_name: The legacy store's file name.
    :param filings: ``{row_id, shelfmark, canonical_ids, keys, spellings}`` per candidate filing the photograph.
    """

    sha256: str
    path: str
    url: Optional[str]
    origin: str
    file_name: str
    filings: List[Dict[str, Any]] = field(default_factory=list)


def lms_transcribe(model_key: str, image_path: str, prompt: str, *,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> Optional[Dict[str, Any]]:
    """Send one photograph and a prompt to LM Studio (lock and yield gate included).

    :param model_key: LM Studio model key.
    :param image_path: Image file.
    :param prompt: Prompt.
    :param max_tokens: Output token budget.
    :return: Completion meta (``text``, ``finish_reason``, token counts), or None when the call failed.
    """
    return asyncio.run(lms_transcriber.transcribe_with_lm_studio_meta(model_key, image_path, prompt,
                                                                      max_tokens=max_tokens))


# ----------------------------------------------------------------------------------------------------------- selection

def store_file_name(image: ImageRef) -> str:
    """The legacy store's file name of a photograph: the URL's last part, else the local file name.

    :param image: Stage1 image.
    :return: File name (``""`` when neither is known).
    """
    if image.url:
        return Path(unquote(urlparse(image.url).path)).name
    return Path(image.path).name if image.path else ""


def shelfmark_named(file_name: str) -> bool:
    """True when a file is named by a shelf mark: its first ``_``-separated part is not all digits.

    :param file_name: File name.
    :return: Shelf-mark-named (a PGP-id-named file such as ``12345_1.jpg`` gives False).
    """
    head = Path(file_name).stem.split("_")[0]
    return bool(head) and not head.isdigit()


def institutions(canonical_ids: Iterable[str]) -> List[str]:
    """Institution tokens of canonical ids (used to read a label that names no institution).

    :param canonical_ids: Canonical ids.
    :return: Distinct tokens, in order.
    """
    out: List[str] = []
    for cid in canonical_ids:
        token = shelf.split_id(cid)[0]
        if token and token not in out:
            out.append(token)
    return out


def mark_keys(text: str, institution_tokens: Sequence[str]) -> Set[str]:
    """Facade keys of a mark: ``loose_key`` of every non-lossy variant of every part of a join; a spelling that
    names no institution is keyed under each given institution. A spelling without a digit names no shelf mark
    (with an institution given, the facade would key any words, e.g. ``illegible``).

    :param text: Shelf mark as written (or read).
    :param institution_tokens: Institutions to try for a spelling without one.
    :return: Non-empty keys.
    """
    keys: Set[str] = set()
    for part in shelf.split(text):
        for spelling in shelf.variants(part, lossy=False):
            if not any(ch.isdigit() for ch in spelling):
                continue
            key = shelf.loose_key(spelling)
            if key:
                keys.add(key)
                continue
            keys.update(k for k in (shelf.loose_key(spelling, token) for token in institution_tokens) if k)
    return keys


def _norm(text: str) -> str:
    """Case- and whitespace-folded spelling.

    :param text: Text.
    :return: Folded text.
    """
    return " ".join(text.split()).casefold()


def filing(cand: Candidate) -> Dict[str, Any]:
    """What a photograph filed under a candidate should be labelled.

    :param cand: Stage1 candidate.
    :return: ``{row_id, shelfmark, canonical_ids, keys, spellings}`` (keys: facade keys of the mark and of the ids;
        spellings: folded variants of the mark).
    """
    tokens = institutions(cand.canonical_ids)
    keys = mark_keys(cand.shelfmark, tokens) if cand.shelfmark else set()
    keys.update(k for k in (shelf.loose_key(cid) for cid in cand.canonical_ids) if k)
    spellings = {_norm(v) for part in shelf.split(cand.shelfmark or "") for v in shelf.variants(part, lossy=False)}
    return {"row_id": cand.row_id, "shelfmark": cand.shelfmark, "canonical_ids": list(cand.canonical_ids),
            "keys": sorted(keys), "spellings": sorted(spellings)}


def select_images(cands: Sequence[Candidate], done: Set[str],
                  is_file: Callable[[Path], bool] = Path.is_file) -> Tuple[List[SlipItem], Counter]:
    """Legacy-store photographs to check: no slip outcome yet, shelf-mark-named, a local file, not done.

    :param cands: Stage1 candidates.
    :param done: sha256 already checked (any outcome but ``error``).
    :param is_file: File-existence check.
    :return: Items (one per sha256, in first-seen order) and the skip counts by reason.
    """
    items: Dict[str, SlipItem] = {}
    skipped: Counter = Counter()
    for cand in cands:
        for image in cand.images:
            if image.origin not in LEGACY_STORE_ORIGINS:
                continue
            name = store_file_name(image)
            reason = ("has_slip_outcome" if image.slip_outcome else "not_shelfmark_named" if not shelfmark_named(name)
                      else "no_sha256" if not image.sha256 else "already_checked" if image.sha256 in done
                      else "no_local_file" if not image.path or not is_file(Path(image.path)) else None)
            if reason:
                skipped[reason] += 1
                continue
            item = items.setdefault(image.sha256, SlipItem(sha256=image.sha256, path=image.path, url=image.url,
                                                           origin=image.origin, file_name=name))
            if all(f["row_id"] != cand.row_id for f in item.filings):
                item.filings.append(filing(cand))
    return list(items.values()), skipped


# --------------------------------------------------------------------------------------------------------- comparison

def clean_answer(text: str) -> str:
    """The label in a model answer: the first line left after quotes, markup, a ``Shelf mark:`` prefix and full stops
    at either end.

    :param text: Raw answer.
    :return: The label (``""`` when nothing is left).
    """
    for raw in text.splitlines():
        line = raw.strip().strip(_QUOTES)
        if not line or line.startswith("```"):
            continue
        line = _ANSWER_PREFIX.sub("", line).strip().strip(_QUOTES + ".")
        if line:
            return line
    return ""


def is_no_label(label: str) -> bool:
    """True when the answer says there is no label (``NONE``, ``No label visible``, ``N/A``; never with a digit).

    :param label: Cleaned answer.
    :return: No label.
    """
    return not any(ch.isdigit() for ch in label) and bool(_NO_LABEL.match(label))


def compare(label: str, filings: Sequence[Mapping[str, Any]]) -> Tuple[str, str, List[str], List[str]]:
    """Compare a read label with the filings of a photograph.

    :param label: Cleaned, non-empty label that is not a no-label answer.
    :param filings: :func:`filing` records.
    :return: ``(outcome, reason, read keys, matched row ids)``: ``match`` when a key of the label is a key of a
        filing or the label is one of the filed mark's variants; ``unreadable`` when the label gives no key;
        else ``mismatch``.
    """
    tokens = institutions(cid for f in filings for cid in f["canonical_ids"])
    keys = mark_keys(label, tokens)
    matched = [f["row_id"] for f in filings if keys & set(f["keys"]) or _norm(label) in set(f["spellings"])]
    if matched:
        return "match", "label_matches_filing", sorted(keys), matched
    if not keys:
        return "unreadable", "no_shelfmark_parsed", [], []
    return "mismatch", "label_names_another_mark", sorted(keys), []


def check_image(item: SlipItem, *, model: str, transcribe: Transcribe,
                clock: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """Read the label of one photograph and decide the outcome.

    The LM Studio call is the one place an exception is caught: any failure is recorded as outcome ``error``.

    :param item: Photograph to check.
    :param model: LM Studio model key.
    :param transcribe: The LM Studio call (:data:`Transcribe`).
    :param clock: Seconds clock.
    :return: The check record.
    """
    record: Dict[str, Any] = {
        "sha256": item.sha256, "file_name": item.file_name, "origin": item.origin, "image_path": item.path,
        "url": item.url, "filings": [{k: f[k] for k in ("row_id", "shelfmark", "canonical_ids", "keys")}
                                     for f in item.filings],
        "model": model, "prompt_name": PROMPT_NAME, "prompt_sha1": PROMPT_SHA1, "read": None, "finish_reason": None,
        "label": None, "read_keys": [], "matched_row_ids": [], "outcome": "error", "reason": "", "seconds": None,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    started = clock()
    try:
        meta = transcribe(model, item.path, SLIP_PROMPT)
    except Exception as exc:  # noqa: BLE001 - a failed LM Studio call is recorded as an ``error`` outcome
        record.update(reason=f"{type(exc).__name__}: {exc}"[:300], seconds=round(clock() - started, 2))
        return record
    record["seconds"] = round(clock() - started, 2)
    if meta is None:
        record["reason"] = "no_response"
        return record
    text = meta.get("text") or ""
    record.update(read=text, finish_reason=meta.get("finish_reason"))
    label = clean_answer(text)
    record["label"] = label
    if meta.get("finish_reason") == "length":
        record.update(outcome="unreadable", reason="answer_cut_by_token_budget")
    elif not label:
        record.update(outcome="unreadable", reason="empty_answer")
    elif is_no_label(label):
        record.update(outcome="no_label", reason="model_saw_no_label")
    else:
        outcome, reason, keys, matched = compare(label, item.filings)
        record.update(outcome=outcome, reason=reason, read_keys=keys, matched_row_ids=matched)
    return record


# ----------------------------------------------------------------------------------------------- results, eye queue

def load_checks(path: Path) -> Dict[str, Dict[str, Any]]:
    """Latest check record per photograph.

    :param path: ``slip_checks.jsonl`` (may not exist).
    :return: ``sha256 -> record``.
    """
    return {rec["sha256"]: rec for rec in read_jsonl(path)} if path.is_file() else {}


def eye_entry(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Eye-queue entry of a check record.

    :param record: Check record.
    :return: Outcome, reason, the read and the label, the expected marks and ids, row ids and the image.
    """
    filings = record.get("filings") or []
    return {"sha256": record["sha256"], "outcome": record["outcome"], "reason": record.get("reason"),
            "read": record.get("read"), "label": record.get("label"),
            "expected": [f.get("shelfmark") for f in filings], "canonical_ids": [c for f in filings
                                                                                for c in f.get("canonical_ids") or []],
            "row_ids": [f.get("row_id") for f in filings], "image_path": record.get("image_path"),
            "file_name": record.get("file_name"), "url": record.get("url")}


def _md(value: Any) -> str:
    """A value as one Markdown table cell.

    :param value: Any value (None -> empty; lists joined with ``; ``).
    :return: Text with ``|`` escaped and whitespace folded.
    """
    if value is None:
        return ""
    text = "; ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    return " ".join(text.split()).replace("|", "\\|")


def render_eye_queue(queue: Sequence[Mapping[str, Any]], checked: int, generated_at: str) -> str:
    """``slip_eye_queue.md``.

    :param queue: Eye entries.
    :param checked: Photographs with a check record.
    :param generated_at: Generation time.
    :return: Markdown.
    """
    lines = ["# Slip-check eye queue", "",
             f"Generated {generated_at} from `{CHECKS_FILE}` ({checked} photographs checked, {len(queue)} here). Each "
             "legacy-store photograph below is filed under a shelf-mark-named file; the label read in the photograph "
             "names another mark (`mismatch`) or could not be read (`unreadable`). Nothing was dropped, moved or "
             "relabelled: decide by eye.", "",
             "| # | outcome | reason | read | expected shelf mark | canonical ids | row ids | image |",
             "|---:|---|---|---|---|---|---|---|"]
    for n, entry in enumerate(queue, 1):
        lines.append("| " + " | ".join([str(n)] + [_md(entry.get(k)) for k in (
            "outcome", "reason", "read", "expected", "canonical_ids", "row_ids", "image_path")]) + " |")
    return "\n".join(lines) + "\n"


def write_eye_queue(work_dir: Path, checks: Mapping[str, Mapping[str, Any]],
                    generated_at: Optional[str] = None) -> int:
    """Rewrite the eye queue from every check record (all runs).

    :param work_dir: The build's work dir.
    :param checks: :func:`load_checks` result.
    :param generated_at: Generation time (default: now, UTC).
    :return: Entries written.
    """
    queue = [eye_entry(rec) for rec in checks.values() if rec.get("outcome") in EYE_OUTCOMES]
    queue.sort(key=lambda e: (EYE_OUTCOMES.index(e["outcome"]), e.get("file_name") or "", e["sha256"]))
    write_jsonl(work_dir / EYE_QUEUE_JSONL, queue)
    when = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (work_dir / EYE_QUEUE_MD).write_text(render_eye_queue(queue, len(checks), when), encoding="utf-8")
    return len(queue)


def run_slip_checks(work_dir: Path, *, model: str, transcribe: Transcribe = lms_transcribe,
                    limit: Optional[int] = None, stop_file: Optional[Path] = None,
                    max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
                    is_file: Callable[[Path], bool] = Path.is_file) -> Dict[str, int]:
    """Check every eligible photograph of stage1 not checked yet; append each record at once; rewrite the eye queue.

    :param work_dir: The build's work dir (reads the stage1 file, writes the check and eye-queue files).
    :param model: LM Studio model key.
    :param transcribe: The LM Studio call (tests stub it).
    :param limit: At most N photographs this run.
    :param stop_file: Stop before the next photograph when this file exists.
    :param max_consecutive_errors: Stop after this many ``error`` outcomes in a row.
    :param is_file: File-existence check.
    :return: Counts: one per outcome checked this run, ``skipped_<reason>``, ``eye_queue``, and ``stopped_by_file``
        or ``stopped_after_errors`` when the run stopped early.
    """
    cands = list(read_jsonl(paths.stage_path(work_dir, "images"), Candidate))
    checks_path = work_dir / CHECKS_FILE
    checks = load_checks(checks_path)
    done = {sha for sha, rec in checks.items() if rec.get("outcome") != "error"}
    items, skipped = select_images(cands, done, is_file)
    counts: Counter = Counter({f"skipped_{reason}": n for reason, n in skipped.items()})
    todo = items if limit is None else items[:limit]
    print(f"slip check: {len(items)} photograph(s) to check, {len(todo)} this run; skipped {dict(skipped)}")
    errors_in_row = 0
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    with checks_path.open("a", encoding="utf-8") as fh:
        for n, item in enumerate(todo, 1):
            if stop_file is not None and stop_file.exists():
                counts["stopped_by_file"] = 1
                break
            record = check_image(item, model=model, transcribe=transcribe)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            checks[item.sha256] = record
            counts[record["outcome"]] += 1
            errors_in_row = errors_in_row + 1 if record["outcome"] == "error" else 0
            if n % PROGRESS_EVERY == 0:
                print(f"  {n}/{len(todo)}: {dict((k, counts[k]) for k in OUTCOMES if counts[k])}")
            if errors_in_row >= max_consecutive_errors:
                counts["stopped_after_errors"] = 1
                print(f"stopping: {errors_in_row} errors in a row (last: {record['reason']})")
                break
    counts["eye_queue"] = write_eye_queue(work_dir, checks)
    return dict(counts)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: run the slip check over stage1 and write the eye queue.

    :param argv: Arguments (default: ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Read the shelf-mark label in legacy-store photographs named by "
                                                 "shelf mark and compare it with their filing (eye queue; nothing "
                                                 "is moved).")
    parser.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR, help="The build's work dir.")
    parser.add_argument("--model", required=True, help="LM Studio model key of the VLM that reads the label.")
    parser.add_argument("--limit", type=int, default=None, help="At most N photographs this run.")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help=f"Stop before the next photograph when this file exists (default: <work dir>/{STOP_FILE}).")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Output token budget per read.")
    args = parser.parse_args(argv)
    stop_file = args.stop_file if args.stop_file is not None else args.work_dir / STOP_FILE

    def transcribe(model_key: str, image_path: str, prompt: str) -> Optional[Dict[str, Any]]:
        """:func:`lms_transcribe` with the CLI's token budget.

        :param model_key: LM Studio model key.
        :param image_path: Image file.
        :param prompt: Prompt.
        :return: Completion meta or None.
        """
        return lms_transcribe(model_key, image_path, prompt, max_tokens=args.max_tokens)

    counts = run_slip_checks(args.work_dir, model=args.model, transcribe=transcribe, limit=args.limit,
                             stop_file=stop_file)
    print(f"slip check done: {counts}")


if __name__ == "__main__":
    main()
