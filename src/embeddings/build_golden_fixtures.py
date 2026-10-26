# File name: build_golden_fixtures.py
# Date: 10/9/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Build the golden fixtures that pin the ``semantic_v3`` and ``semantic_v4`` text functions on real records.

Plan B7 (genizah_search ``docs/handoffs/v10_embedder/V10_EMBEDDER_PLAN.md``): regex fingerprints alone did not catch
the 2026-10-09 drift of 28k texts, so both repos keep about 200 stratified real records with the hash of the text
and the eligibility reason their copy of the functions must reproduce. This script streams
``merged_shelfmarks.jsonl`` (read-only), builds each record's production (``v8``) text with
``GenizahDocument.create_text_representation()``, draws a seeded stratified sample (:data:`STRATA`, then a random
remainder up to ``--total``) and writes, for the same records in file order:

* ``tests/data/semantic_v4_golden.jsonl``: ``{"doc_id", "stratum", "v8_text", "v4_text", "v4_sha256", "eligible",
  "reason"}`` with ``(v4_text, reason) = semantic_v4_record(v8_text)[:2]``, ``v4_sha256 = text_sha256(v4_text)`` and
  ``eligible = reason == "eligible"`` (the row shape the genizah_search ``test_semantic_v4.py`` reads too);
* ``tests/data/semantic_v3_golden.jsonl``: ``{"doc_id", "stratum", "v8_text", "v3_sha256", "reason"}`` with
  ``v3_sha256 = text_sha256(semantic_text(v8_text))`` and
  ``reason = semantic_ineligible_reason(semantic_text(v8_text))``.

``stratum`` names the :data:`STRATA` entry whose draw took the record (:data:`RANDOM_STRATUM` for the remainder); a
record often belongs to further strata as well.

The fixtures store the ``v8`` text itself, so they pin the text functions, not the document model: a later
document-model change can change which records a rebuild samples, never the expected values of a stored row.
``tests/test_semantic_v4.py`` recomputes every row with the live functions. Records whose text names an excluded
source (:data:`EXCLUDED_TEXT_RE`) are never sampled: the fixtures live in a public repo.

For the same reason no edition text is stored. :func:`mask_edition_text` replaces every letter of the
``Transcription:`` and ``Translation:`` values with a letter of the same script and case, chosen from the record id
and the position. Length, spacing, punctuation, brackets and the editor credits the text functions strip stay as
they are. The rows are computed from the masked text, so they still pin the text functions on the real record shapes.
The mask depends only on the position, so it is idempotent: ``tests/test_build_golden_fixtures.py`` checks that
masking a committed fixture's text leaves it unchanged.

Run (about 5 minutes for the whole file; ``--limit`` for a smoke run that must not overwrite the fixtures;
``--mask-existing`` re-masks the committed fixtures in place without reading the merged file)::

    .venv/bin/python -m src.embeddings.build_golden_fixtures
    .venv/bin/python -m src.embeddings.build_golden_fixtures --limit 2000 --out-dir /tmp/golden_smoke
    .venv/bin/python -m src.embeddings.build_golden_fixtures --mask-existing
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from src.datasets.indexing.index_merged_genizah import DEFAULT_MERGED, iter_merged_documents
from src.embeddings.semantic_text import (
    ALT_PREFIX,
    DESC_PREFIX,
    EDITOR_RE,
    MASK,
    MAX_ALT_CHARS,
    PGP_EDITOR_RE,
    V4_DROP_PREFIXES,
    fold_field_lines,
    merge_title_parts,
    semantic_ineligible_reason,
    semantic_text,
    semantic_v4_ineligible_reason,
    semantic_v4_record,
    text_sha256,
    title_key,
    title_parts,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = _REPO_ROOT / "tests" / "data"
V4_FIXTURE_NAME = "semantic_v4_golden.jsonl"
V3_FIXTURE_NAME = "semantic_v3_golden.jsonl"
DEFAULT_SEED = 13
DEFAULT_TOTAL = 200
#: ``stratum`` of the records drawn for the random remainder.
RANDOM_STRATUM = "random"
# Sampled in this order, each from the records no earlier stratum took (a record can belong to several strata, so a
# stratum usually ends up with more members in the sample than its quota). The first six are plan B7's list; the
# others cover the remaining branches of the v4 functions.
STRATA: Tuple[Tuple[str, int], ...] = (
    ("ktiv_only_alt_titles", 20),     # KTIV-only records with an Alternate titles line
    ("bracketed_frame", 20),          # the Alternate titles line holds a "[" (a KTIV scholarly frame)
    ("multiline_description", 20),    # the Description value spans lines: unlabelled continuation lines
    ("admitted_short", 20),           # v3 "short", admitted by the v4 festival / topic patterns
    ("boilerplate_only", 20),         # boilerplate_only under v3 or v4
    ("empty", 20),                    # nothing left after the v3 text function
    ("v4_stricter", 10),              # v3 eligible, v4 not (a Subjects line or a bracketed label lifted it under v3)
    ("short_not_admitted", 10),       # v4 "short"
    ("long_alt_titles", 10),          # Alternate titles longer than MAX_ALT_CHARS: the cap can apply
    ("alt_titles_without_description", 10),  # the Alternate titles line becomes the Description line
    ("alt_part_in_description", 10),  # an alternate-title part the whole-word de-duplication drops
    ("alt_dedupe_across_parts", 5),   # a kept part whose key spans two earlier keys (dedupe_spans_parts)
)
# Never sampled: a record whose production text names one of these sources (the fixtures are published).
EXCLUDED_TEXT_RE = re.compile(r"friedberg|fjms|fjp|fgp|genizah\.org", re.IGNORECASE)
#: Field labels whose value is edition text (a transcription, or a translation of one): :func:`mask_edition_text`.
EDITION_FIELD_PREFIXES: Tuple[str, ...] = ("Transcription:", "Translation:")
# Letter classes and the letters that replace them: same script and case, so the text functions see the same shape.
_MASK_ALPHABETS: Tuple[Tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"[א-ת]"), "אבגדהוזחטיכלמנסעפצקרשת"),
    (re.compile(r"[ء-غف-يٱ-ۓ]"), "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"),
    (re.compile(r"[a-z]"), "abcdefghijklmnopqrstuvwxyz"),
    (re.compile(r"[A-Z]"), "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
)
# Line breaks that str.splitlines() (and some editors) honour but json.dumps(ensure_ascii=False) leaves raw inside a
# string; written as JSON escapes so that one row is always one physical line.
_RAW_LINE_BREAK_ESCAPES = {0x85: "\\u0085", 0x2028: "\\u2028", 0x2029: "\\u2029"}


class ScannedRecord(NamedTuple):
    """One record of the merged file as the sampler sees it.

    :param doc_id: Canonical id (the index ``_id``).
    :param v8_text: Production text.
    :param strata: Names of the :data:`STRATA` it belongs to.
    :param v3_reason: ``semantic_ineligible_reason`` of its v3 text.
    :param v4_reason: ``semantic_v4_ineligible_reason`` of its v3 text.
    """

    doc_id: str
    v8_text: str
    strata: FrozenSet[str]
    v3_reason: str
    v4_reason: str


def _mask_letter(ch: str, doc_id: str, pos: int) -> str:
    """The letter that replaces ``ch`` at a position of a record's text (non-letters are returned unchanged).

    :param ch: One character.
    :param doc_id: Record id (part of the seed, so records do not share a pattern).
    :param pos: Character offset in the record's text.
    :returns: A letter of the same script and case as ``ch``, or ``ch`` itself.
    :rtype: str
    """
    for pattern, alphabet in _MASK_ALPHABETS:
        if pattern.fullmatch(ch):
            digest = hashlib.blake2b(f"{doc_id}:{pos}".encode("utf-8"), digest_size=4).digest()
            return alphabet[int.from_bytes(digest, "big") % len(alphabet)]
    return ch


def mask_edition_text(v8_text: str, doc_id: str) -> str:
    """Replace the letters of a text's edition fields (:data:`EDITION_FIELD_PREFIXES`), keeping their shape.

    Only the value of a line that starts with an edition label is changed; the label, every non-letter character and
    the editor credits the text functions strip (``EDITOR_RE``, ``PGP_EDITOR_RE``) are kept. The replacement depends
    on the record id and the position only, so masking a masked text returns it unchanged.

    :param v8_text: Production text of one record.
    :param doc_id: The record's id.
    :returns: The text with its transcription and translation letters replaced.
    :rtype: str
    """
    out: List[str] = []
    offset = 0
    for line in v8_text.split("\n"):
        prefix = next((p for p in EDITION_FIELD_PREFIXES if line.startswith(p)), None)
        if prefix is not None:
            kept = [m.span() for rx in (EDITOR_RE, PGP_EDITOR_RE) for m in rx.finditer(line)]
            chars = list(line)
            for i in range(len(prefix), len(line)):
                if not any(start <= i < end for start, end in kept):
                    chars[i] = _mask_letter(line[i], doc_id, offset + i)
            line = "".join(chars)
        out.append(line)
        offset += len(line) + 1
    return "\n".join(out)


def field_value(text: str, prefix: str) -> Optional[str]:
    """The value of a text's first line with a given field label.

    :param text: Record text.
    :param prefix: Field label, e.g. ``"Description:"``.
    :returns: The stripped value, or ``None`` when no line carries the label.
    :rtype: Optional[str]
    """
    return next((ln[len(prefix):].strip() for ln in text.split("\n") if ln.startswith(prefix)), None)


def dedupe_spans_parts(v3_text: str) -> bool:
    """Whether the title fold keeps an alternate-title part whose key occurs, as whole words, across two earlier keys.

    :func:`src.embeddings.semantic_text.merge_title_parts` compares a part's key with the description's key and with
    each earlier kept part's key on its own, so ``4:2`` is kept after ``3:4; 2:5 – 11``. A lookup in those keys joined
    by single spaces (the draft code of the text-pipeline audit) would leave it out, so these records tell the two
    apart and pin the reference behaviour.

    :param v3_text: ``semantic_text`` of a production text.
    :returns: True when some kept part's key occurs only across the joined earlier keys.
    :rtype: bool
    """
    lines = [ln for ln in fold_field_lines(v3_text).split("\n") if not ln.startswith(V4_DROP_PREFIXES)]
    parts = title_parts(lines)
    if not parts:
        return False
    desc = next((ln[len(DESC_PREFIX):].strip() for ln in lines if ln.startswith(DESC_PREFIX)), "")
    joined = f" {title_key(desc)} " if desc else " "
    for part in merge_title_parts(desc, parts)[0]:
        key = title_key(part)
        if f" {key} " in joined:
            return True
        joined += f"{key} "
    return False


def record_strata(sources_present: Sequence[str], description: str, v3_text: str, v3_reason: str,
                  v4_reason: str) -> FrozenSet[str]:
    """The :data:`STRATA` a record belongs to.

    :param sources_present: The merged record's ``sources_present``.
    :param description: The document's description (its ``Description:`` value before the text functions).
    :param v3_text: ``semantic_text`` of its production text.
    :param v3_reason: ``semantic_ineligible_reason(v3_text)``.
    :param v4_reason: ``semantic_v4_ineligible_reason(v3_text)``.
    :returns: Stratum names.
    :rtype: FrozenSet[str]
    """
    alt = field_value(v3_text, ALT_PREFIX)
    desc = field_value(v3_text, DESC_PREFIX)
    names = set()
    if alt is not None:
        if list(sources_present) == ["ktiv"]:
            names.add("ktiv_only_alt_titles")
        if "[" in alt.replace(MASK, ""):
            names.add("bracketed_frame")
        if len(alt) > MAX_ALT_CHARS:
            names.add("long_alt_titles")
        if desc is None:
            names.add("alt_titles_without_description")
        elif any(title_key(p) and f" {title_key(p)} " in f" {title_key(desc)} " for p in alt.split(";")):
            names.add("alt_part_in_description")
        if dedupe_spans_parts(v3_text):
            names.add("alt_dedupe_across_parts")
    if "\n" in description.strip() and len(fold_field_lines(v3_text).split("\n")) < len(v3_text.split("\n")):
        names.add("multiline_description")
    if v3_reason == "short" and v4_reason == "eligible":
        names.add("admitted_short")
    if "boilerplate_only" in (v3_reason, v4_reason):
        names.add("boilerplate_only")
    if v3_reason == "empty":
        names.add("empty")
    if v3_reason == "eligible" and v4_reason != "eligible":
        names.add("v4_stricter")
    if v4_reason == "short":
        names.add("short_not_admitted")
    return frozenset(names)


def scan_records(docs: Iterable, excluded: Counter) -> List[ScannedRecord]:
    """Build every document's production text and classify it.

    :param docs: ``GenizahDocument`` instances (e.g. :func:`iter_merged_documents`).
    :param excluded: Counter incremented under ``"excluded_source_name"`` for each record left out
        (:data:`EXCLUDED_TEXT_RE`).
    :returns: The records that may be sampled, in file order.
    :rtype: List[ScannedRecord]
    """
    out: List[ScannedRecord] = []
    for doc in docs:
        v8 = doc.create_text_representation()
        if EXCLUDED_TEXT_RE.search(v8):
            excluded["excluded_source_name"] += 1
            continue
        v3 = semantic_text(v8)
        v3_reason, v4_reason = semantic_ineligible_reason(v3), semantic_v4_ineligible_reason(v3)
        sources = (doc.full_metadata or {}).get("sources_present") or []
        strata = record_strata(sources, doc.description or "", v3, v3_reason, v4_reason)
        out.append(ScannedRecord(doc.doc_id or "", v8, strata, v3_reason, v4_reason))
    return out


def draw_sample(records: Sequence[ScannedRecord], seed: int = DEFAULT_SEED,
                total: int = DEFAULT_TOTAL) -> Dict[int, str]:
    """Seeded stratified sample: each stratum's quota in :data:`STRATA` order, then a random remainder.

    :param records: Scanned records in file order.
    :param seed: Random seed.
    :param total: Target sample size (more when the strata alone exceed it).
    :returns: ``{position in records: stratum whose draw took it}`` (:data:`RANDOM_STRATUM` for the remainder),
        positions ascending (file order).
    :rtype: Dict[int, str]
    """
    rng = random.Random(seed)
    drawn: Dict[int, str] = {}
    for name, quota in STRATA:
        pool = [i for i, rec in enumerate(records) if name in rec.strata and i not in drawn]
        drawn.update(dict.fromkeys(rng.sample(pool, min(quota, len(pool))), name))
    rest = [i for i in range(len(records)) if i not in drawn]
    drawn.update(dict.fromkeys(rng.sample(rest, min(max(total - len(drawn), 0), len(rest))), RANDOM_STRATUM))
    return dict(sorted(drawn.items()))


def fixture_rows(v8_text: str, doc_id: str, stratum: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    """The v4 and v3 fixture rows of one record.

    :param v8_text: Production text.
    :param doc_id: Canonical id.
    :param stratum: The stratum whose draw took the record (:func:`draw_sample`).
    :returns: ``(v4 row, v3 row)``.
    :rtype: Tuple[Dict[str, object], Dict[str, object]]
    """
    v4_text, reason, v3 = semantic_v4_record(v8_text)
    v4_row = {"doc_id": doc_id, "stratum": stratum, "v8_text": v8_text, "v4_text": v4_text,
              "v4_sha256": text_sha256(v4_text), "eligible": reason == "eligible", "reason": reason}
    v3_row = {"doc_id": doc_id, "stratum": stratum, "v8_text": v8_text, "v3_sha256": text_sha256(v3),
              "reason": semantic_ineligible_reason(v3)}
    return v4_row, v3_row


def jsonl_line(row: Dict[str, object]) -> str:
    """One row as a single physical JSON line: non-ASCII kept as is, except the raw line breaks it would carry.

    :param row: Row.
    :returns: The JSON text, without the trailing newline.
    :rtype: str
    """
    return json.dumps(row, ensure_ascii=False).translate(_RAW_LINE_BREAK_ESCAPES)


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    """Write rows as UTF-8 JSON lines (:func:`jsonl_line`).

    :param path: Output file (parent directories are created).
    :param rows: Rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(jsonl_line(row) + "\n")


def summary(records: Sequence[ScannedRecord], drawn: Dict[int, str], excluded: Counter) -> Dict[str, object]:
    """Counts to print: per stratum (available / in the sample / drawn by it), reasons and v3 → v4 transitions.

    :param records: Scanned records.
    :param drawn: The sample (:func:`draw_sample`).
    :param excluded: Exclusion counter from :func:`scan_records`.
    :returns: Summary dict.
    :rtype: Dict[str, object]
    """
    sample = [records[i] for i in drawn]
    by_draw = Counter(drawn.values())
    return {
        "records_scanned": len(records) + sum(excluded.values()),
        **dict(excluded),
        "sample_size": len(sample),
        "strata": {name: {"available": sum(name in r.strata for r in records),
                          "in_sample": sum(name in r.strata for r in sample), "drawn": by_draw[name]}
                   for name, _ in STRATA},
        "drawn_at_random": by_draw[RANDOM_STRATUM],
        "sample_v4_reasons": dict(Counter(r.v4_reason for r in sample)),
        "sample_v3_reasons": dict(Counter(r.v3_reason for r in sample)),
        "corpus_v3_to_v4": {f"{a}->{b}": n for (a, b), n in
                            sorted(Counter((r.v3_reason, r.v4_reason) for r in records).items())},
    }


def mask_existing(out_dir: Path) -> Dict[str, int]:
    """Mask the edition text of fixtures already written (:func:`mask_edition_text`) and recompute their rows.

    The v4 fixture supplies each row's id, stratum and text; the v3 fixture must hold the same records in the same
    order. Both files are rewritten.

    :param out_dir: Directory holding both fixtures.
    :returns: ``{"rows", "masked", "v4_reason_changed", "v3_reason_changed"}`` counts.
    :rtype: Dict[str, int]
    :raises ValueError: When the two fixtures do not list the same records in the same order.
    """
    with open(out_dir / V4_FIXTURE_NAME, encoding="utf-8") as fh:
        old_v4 = [json.loads(line) for line in fh if line.strip()]
    with open(out_dir / V3_FIXTURE_NAME, encoding="utf-8") as fh:
        old_v3 = [json.loads(line) for line in fh if line.strip()]
    if [r["doc_id"] for r in old_v4] != [r["doc_id"] for r in old_v3]:
        raise ValueError("the v3 and v4 fixtures do not hold the same records in the same order")
    counts = Counter(rows=len(old_v4))
    rows = []
    for v4, v3 in zip(old_v4, old_v3):
        masked = mask_edition_text(v4["v8_text"], v4["doc_id"])
        counts["masked"] += masked != v4["v8_text"]
        new_v4, new_v3 = fixture_rows(masked, v4["doc_id"], v4["stratum"])
        counts["v4_reason_changed"] += new_v4["reason"] != v4["reason"]
        counts["v3_reason_changed"] += new_v3["reason"] != v3["reason"]
        rows.append((new_v4, new_v3))
    write_jsonl(out_dir / V4_FIXTURE_NAME, (v4 for v4, _ in rows))
    write_jsonl(out_dir / V3_FIXTURE_NAME, (v3 for _, v3 in rows))
    return dict(counts)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Scan the merged file, sample, and write both fixtures (or re-mask the existing ones).

    :param argv: Command-line arguments (``None``: ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged", default=DEFAULT_MERGED, help="merged_shelfmarks.jsonl (read-only)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="where the two fixtures are written")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="sample size (strata first)")
    parser.add_argument("--limit", type=int, default=None, help="scan only the first N records (smoke runs)")
    parser.add_argument("--mask-existing", action="store_true",
                        help="mask the edition text of the fixtures in --out-dir and recompute them; no merged file")
    args = parser.parse_args(argv)
    if args.mask_existing:
        print(json.dumps(mask_existing(args.out_dir), indent=1))
        return
    if args.limit is not None and args.out_dir.resolve() == DEFAULT_OUT_DIR.resolve():
        parser.error("--limit samples a prefix of the file: pass --out-dir so the committed fixtures are kept")
    excluded: Counter = Counter()
    records = scan_records(iter_merged_documents(args.merged, limit=args.limit), excluded)
    drawn = draw_sample(records, args.seed, args.total)
    rows = [fixture_rows(mask_edition_text(records[i].v8_text, records[i].doc_id), records[i].doc_id, stratum)
            for i, stratum in drawn.items()]
    write_jsonl(args.out_dir / V4_FIXTURE_NAME, (v4 for v4, _ in rows))
    write_jsonl(args.out_dir / V3_FIXTURE_NAME, (v3 for _, v3 in rows))
    print(json.dumps(summary(records, drawn, excluded), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
