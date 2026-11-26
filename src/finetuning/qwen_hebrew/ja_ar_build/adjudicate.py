# File name: adjudicate.py
# Date: 11/22/26
# Author: Isaac Godfried. Coded by Claude.
"""Blind-audit verdicts on photograph/text pairs: scoring the judges' records and deciding each pair (ja_ar_v1).

The logic of ``logs/next_round/ja_ar_v1/novel1119/audit/score.py`` and ``apply_audit.py`` lives here so it is under
version control and under test (those scripts are thin command lines around it).

Why it was rewritten (integrity audit of 2026-11-22). The scorer took the worse of two verdicts but filled the
record's other fields with ``(judge A or judge B)``, so judge B's "the photograph has more main text" could be paired
with judge A's "it is only a marginal note" and the pair was accepted as marginalia (three training rows). The
applier accepted every ``complete_pair`` at once, although the judges' prompt allowed "minor" uncovered writing in a
complete pair and said order and restoration observations did not change the verdict: 140 of 320 rows approved as
complete carried such a warning. Lines were removed from a text on one judge's list even when the other judge had
found the pair complete.

Rules now.

* Every judge's own record is kept whole (:func:`score_item`, field ``records``); nothing is merged across judges
  except the verdict, which is the worse of the two (:func:`worse`).
* A record approves nothing unless it is whole: every field its contract asks for must be there with an allowed
  value (:func:`record_problems`). A missing answer is never read as "no concern".
* A pair is approved as complete only when EVERY judge who saw it gives ``complete_pair`` and none of them reports a
  concern (:func:`concerns`): writing the text leaves out (of any size or kind), text lines not on the photograph,
  restored words, a clear order problem, garbled lines, or a confidence under :data:`MIN_CONFIDENCE`; and only
  when at least :data:`JUDGES_FOR_APPROVAL` judges saw it (``audit_hold:one_judge_complete`` otherwise: the pair
  waits for a second judge). Measured on the 282 items of the 2026-11-19 audits that two judges saw: where one
  judge found a pair complete without any concern (48 items), the other agreed in 24.
* A text is changed on the judges' word (lines removed because they are not on the photograph, unresolved lines put
  back because they are) only when at least two judges saw the pair and name exactly the same lines, and neither
  has any other concern. One judge's word changes no text; such a pair waits for a second judge
  (``audit_hold:one_judge_edit``). Line numbers a judge gives must be numbers of the pair's own lines: when the
  text rules changed the number of lines the judge was shown, no text is changed on those numbers.
* Unresolved lines beside a pair's text (``open_lines`` of :mod:`.assemble`) must be settled by the judges: found
  absent by every judge, or put back by two. A judge of the first contract was not shown them; that judge settles
  them as absent only by reporting no uncovered writing at all.
* Every audit of a pair counts (:func:`load_audits` keeps them all). Verdicts of one contract on the same version
  of a pair are read together, as so many judges (:func:`merge_verdicts`). A verdict of the second contract takes
  the place of first-contract verdicts, but it does not silence them: when an earlier judge saw writing the text
  leaves out and the later judges approve the text without putting lines back, or an earlier judge saw text that is
  not on the photograph and the later ones remove nothing, the pair is held
  (``audit_hold:earlier_judges_differ``).
* A verdict applies only to the version of the pair that was judged (:func:`verdict_applies`). A pair that changed
  after an adverse verdict is not approved by the assembly's rule either (``earlier_verdict_on_other_version``).
  One change does not make another version (2026-11-22, the re-ingest that orders a line's characters by their
  drawn position): a text that holds the same letters, line by line, on the same photograph, crop and rotation,
  with the same unresolved lines beside it, is the text the judges compared with the photograph; only spaces,
  punctuation and gap marks moved (:func:`verdict_match` then says ``same_letters``, the pair's audit record and
  its approval say so, and the approval is bound to the text as it is now). A letter added, removed or moved makes
  another version.
* Marginal notes and other hands are in scope (:data:`approval.SCOPE`): no exception accepts a pair whose
  photograph shows them and whose text leaves them out.
* Every accepted pair's approval is bound to its final answer (:func:`build_rows.stamp_approval`). An approval
  that cannot be bound (the text rules changed the letters the judges saw; lines are still unresolved) is no
  approval, and the pair goes back to review as the assembly made it: a change the judges prescribed is kept only
  together with its approval.

Contracts. ``v1`` is the prompt of ``build/audit3`` with ``uncovered_kind`` (audits up to 2026-11-19): the judge saw
the pair's assembled lines. ``v2`` (:mod:`.audit_contract`) shows the final answer and the unresolved lines, and its
key carries the pair's fingerprint and a digest of the unresolved lines, so a verdict applies only to exactly the
pair that was judged. A ``v1`` verdict applies only to a pair with the judged lines AND the judged photograph,
rotation and crop, which the audit's hidden key records (:func:`image_of`); without the key a ``v1`` verdict applies
to nothing.
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from src.finetuning.qwen_hebrew.ja_ar_build import audit_contract, build_rows

SEVERITY = {"complete_pair": 0, "photo_has_extra": 1, "text_has_extra": 2, "both_partial": 3, "wrong_text": 4,
            "cannot_judge": -1}
VERDICTS = ["complete_pair", "photo_has_extra", "text_has_extra", "both_partial", "wrong_text", "cannot_judge"]
#: Largest share of a pair's lines that judges may remove as not on the photograph.
MAX_DROP_SHARE = 0.4
#: A verdict with a lower confidence approves nothing (1 = "little more than a guess").
MIN_CONFIDENCE = 3
#: Judges whose agreement a change of the text needs.
JUDGES_FOR_EDIT = 2
#: Judges who must find a pair complete, without a concern, before it is approved (see the module docstring).
JUDGES_FOR_APPROVAL = 2
FINAL_LETTERS = str.maketrans("", "", "ךםןףץ")
#: Hebrew and Arabic letters: what a judge's comparison of a text with its photograph is about.
LETTER_RE = re.compile(r"[\u05d0-\u05ea\u0621-\u064a\u066e-\u06d3]")
#: Allowed answers per contract; a record with another value, or none, in one of these fields is not whole.
ALLOWED = {"v1": {"uncovered_writing": ("none", "minor", "substantial"),
                  "supplied_text": ("none", "minor", "substantial", "suspected"),
                  "order_problem": ("none", "suspected", "clear")},
           "v2": {"uncovered_writing": ("none", "some"), "supplied_text": ("none", "some"),
                  "order_problem": ("none", "suspected", "clear"),
                  "uncovered_kind": ("none", "main_text_lines", "marginal_or_other_hand", "other_page_or_text")}}
#: Fields that must be lists of text line numbers.
LINE_LISTS = {"v1": ("lines_not_on_photo",), "v2": ("lines_not_on_photo", "garbled_lines")}
#: Concerns that belong to a verdict of ``text_has_extra`` itself and so do not stand against removing the lines.
OWN_TO_TEXT_EXTRA = ("lines_not_on_photo", "share_not_all")
#: Earlier judges' concerns that later judges of the second contract answer by their own, explicit observation.
ANSWERED_BY_TWO = ("supplied_text", "order_clear", "low_confidence")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file.

    :param path: File path.
    :return: The records.
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_judgments(folder: Path, name: str, judge: str) -> Dict[str, Dict[str, Any]]:
    """One judge's own records of an audit, by item id.

    :param folder: Folder of the ``judge{A,B}_<name>_batchNN.json`` files.
    :param name: Audit name.
    :param judge: ``A`` or ``B``.
    :return: Mapping item id -> the judge's record.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(folder.glob(f"judge{judge}_{name}_batch*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data if isinstance(data, list) else data.get("judgments", []):
            if isinstance(record, dict) and "item_id" in record:
                out[record["item_id"]] = record
    return out


def worse(va: str, vb: str) -> str:
    """The conservative verdict of two judges.

    :param va: Judge A verdict (may be empty).
    :param vb: Judge B verdict (may be empty).
    :return: The more severe one; ``cannot_judge`` only when no judge could judge.
    """
    if not va:
        return vb
    return vb if vb and SEVERITY.get(vb, -1) > SEVERITY.get(va, -1) else va


def record_problems(record: Mapping[str, Any], contract: str = "v1") -> List[str]:
    """What keeps one judge's record from being whole: fields its contract asks for that are missing or invalid.

    :param record: The judge's record.
    :param contract: ``v1`` or ``v2``.
    :return: ``record:<field>`` per such field (empty: the record answers everything it was asked).
    """
    out: List[str] = []
    if record.get("verdict") not in VERDICTS:
        out.append("record:verdict")
    confidence = record.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 1 <= confidence <= 5:
        out.append("record:confidence")
    for name, values in ALLOWED[contract].items():
        if record.get(name) not in values:
            out.append(f"record:{name}")
    for name in LINE_LISTS[contract]:
        value = record.get(name)
        if not isinstance(value, list) or any(isinstance(n, bool) or not isinstance(n, int) for n in value):
            out.append(f"record:{name}")
    if not isinstance(record.get("share_on_photo"), str):
        out.append("record:share_on_photo")
    return out


def concerns(record: Mapping[str, Any], contract: str = "v1") -> List[str]:
    """What in one judge's record stands against approving the pair's text as it is.

    :param record: The judge's record.
    :param contract: ``v1`` or ``v2``.
    :return: Concern names (empty: the judge answered everything and saw nothing against it). ``record:<field>``
        for an answer that is missing or invalid (:func:`record_problems`); ``uncovered:<kind>`` for any writing
        the text leaves out, whatever its size (a judge's "minor" was an unread first or last line in 11 of 16
        sampled cases, 2026-11-22), also when only the kind says so; ``lines_not_on_photo``; ``share_not_all``;
        ``supplied_text`` (words where the photograph has no ink); ``order_clear``; ``garbled_lines``;
        ``low_confidence``.
    """
    out = record_problems(record, contract)
    kind = record.get("uncovered_kind")
    if record.get("uncovered_writing") != "none" or kind not in (None, "", "none"):
        out.append(f"uncovered:{kind or 'kind_unknown'}")
    if record.get("lines_not_on_photo"):
        out.append("lines_not_on_photo")
    share = record.get("share_on_photo")
    if isinstance(share, str) and not share.strip().lower().startswith("all"):
        out.append("share_not_all")
    if record.get("supplied_text") != "none":
        out.append("supplied_text")
    if record.get("order_problem") == "clear":
        out.append("order_clear")
    if record.get("garbled_lines"):
        out.append("garbled_lines")
    confidence = record.get("confidence")
    if isinstance(confidence, int) and not isinstance(confidence, bool) and confidence < MIN_CONFIDENCE:
        out.append("low_confidence")
    return sorted(set(out))


def score_item(key: Mapping[str, Any], ja: Optional[Mapping[str, Any]], jb: Optional[Mapping[str, Any]]
               ) -> Dict[str, Any]:
    """The result record of one judged item: the conservative verdict and each judge's own record, whole.

    :param key: The item's hidden key (``item_id``, ``pair_id``, ``stratum``, ``decision``, ``letters``; for the
        second contract also ``contract``, ``fingerprint``, ``open_shown``, ``open_digest`` and ``lines_match``).
    :param ja: Judge A's record, or None.
    :param jb: Judge B's record, or None.
    :return: The record. No field mixes two judges' observations.
    """
    va, vb = (ja or {}).get("verdict", ""), (jb or {}).get("verdict", "")
    records = {name: dict(rec) for name, rec in (("A", ja), ("B", jb)) if rec}
    contract = "v2" if key.get("contract") == "v2" else "v1"
    out = {"item_id": key["item_id"], "pair_id": key["pair_id"], "stratum": key.get("stratum"),
           "decision": key.get("decision"), "letters": key.get("letters"), "judge_a": va, "judge_b": vb,
           "final": worse(va, vb), "confidence_a": (ja or {}).get("confidence"),
           "confidence_b": (jb or {}).get("confidence"), "notes_a": (ja or {}).get("notes"),
           "notes_b": (jb or {}).get("notes"), "records": records,
           "concerns": {name: concerns(rec, contract) for name, rec in records.items()}}
    for field in ("contract", "fingerprint", "open_shown", "open_digest", "lines_match"):
        if key.get(field) is not None:
            out[field] = key[field]
    return out


def judge_records(verdict: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Each judge's own record of a result.

    :param verdict: Result record (:func:`score_item`, :func:`load_audit` or :func:`merge_verdicts`).
    :return: ``{judge: record}`` for the judges who gave a verdict.
    """
    return {name: rec for name, rec in (verdict.get("records") or {}).items() if rec and rec.get("verdict")}


def line_key(line: str) -> str:
    """A line without its final-form letters and spaces: equal for a line before and after its finals were restored.

    :param line: A target line.
    :return: The key.
    """
    return "".join(line.translate(FINAL_LETTERS).split())


def same_lines(judged: Optional[Sequence[str]], current: Sequence[str]) -> bool:
    """Whether a judged item's lines are the pair's lines (final letters set aside).

    :param judged: Lines of the judged item (None when the item is unknown).
    :param current: Lines of the run's pair.
    :return: True when both lists have the same length and equal keys line by line.
    """
    return judged is not None and len(judged) == len(current) and all(
        line_key(a) == line_key(b) for a, b in zip(judged, current))


def letters_of(line: str, finals: bool = True) -> str:
    """The Hebrew and Arabic letters of a line, in order: no spaces, points, punctuation or gap marks.

    :param line: A text line.
    :param finals: Keep final-form letters (False: left out, for texts judged before the finals were restored).
    :return: The letters.
    """
    text = "".join(LETTER_RE.findall(line))
    return text if finals else text.translate(FINAL_LETTERS)


def same_letters(judged: Optional[Sequence[str]], current: Sequence[str], finals: bool = True) -> bool:
    """Whether a judged text and a pair's text hold the same letters, line by line.

    :param judged: Lines of the judged item (None when the item is unknown).
    :param current: Lines of the text now.
    :param finals: Compare final-form letters too.
    :return: True when both have the same number of lines and every line the same letters in the same order.
    """
    return judged is not None and len(judged) == len(current) and all(
        letters_of(a, finals) == letters_of(b, finals) for a, b in zip(judged, current))


def _open_answers(record: Mapping[str, Any], entries: int) -> Optional[List[Tuple[str, int]]]:
    """One judge's answers on the unresolved lines shown, per entry.

    :param record: The judge's record (``open_lines`` = ``[{"entry": 1-based number, "on_photo": "none" | "all" |
        "some" | "cannot_tell", "count": lines found, counted from the pair's text}]``).
    :param entries: Number of entries the item showed.
    :return: ``[(on_photo, count)]`` per entry, or None when an entry has no usable answer.
    """
    given = {a.get("entry"): a for a in record.get("open_lines") or [] if isinstance(a, dict)}
    out: List[Tuple[str, int]] = []
    for k in range(1, entries + 1):
        answer = given.get(k)
        if answer is None or answer.get("on_photo") not in ("none", "all", "some", "cannot_tell"):
            return None
        count = answer.get("count")
        out.append((answer["on_photo"], count if isinstance(count, int) and not isinstance(count, bool) else 0))
    return out


def _settle_open(pair: Mapping[str, Any], verdict: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]
                 ) -> Tuple[Optional[str], List[Tuple[int, int]]]:
    """What the judges say about the unresolved lines beside a pair's text.

    :param pair: Pair record.
    :param verdict: Its result record (``contract``, ``open_shown`` = lines shown per entry).
    :param records: The judges' own records, none of which carries a concern.
    :return: ``(hold reason or None, [(entry index, lines to put back)])``. With no hold reason every entry not
        listed for putting back was found absent by every judge.
    """
    entries = [k for k, e in enumerate(pair.get("open_lines") or []) if not e.get("settled")]
    if not entries:
        return None, []
    if verdict.get("contract") != "v2":
        return None, []                 # no concern means no uncovered writing at all: the lines are not on it
    shown = list(verdict.get("open_shown") or [])
    if len(shown) != len(entries):
        return "audit_hold:open_lines_changed", []
    answers = {name: _open_answers(rec, len(entries)) for name, rec in records.items()}
    if any(a is None for a in answers.values()):
        return "audit_hold:open_lines_unanswered", []
    if any(found == "cannot_tell" for a in answers.values() for found, _ in a):
        return "audit_hold:open_lines_uncertain", []
    restore: List[Tuple[int, int]] = []
    for pos, k in enumerate(entries):
        entry = pair["open_lines"][k]
        given = {a[pos] for a in answers.values()}
        founds = {found for found, _ in given}
        if founds == {"none"}:
            continue
        if len(records) < JUDGES_FOR_EDIT:
            return "audit_hold:one_judge_edit", []
        counts = {shown[pos] if found == "all" else count for found, count in given}
        if len(counts) != 1 or "none" in founds:
            return "audit_hold:open_lines_disagree", []
        count = counts.pop()
        inside = entry.get("after_line") is not None and entry.get("before_line") is not None
        if count <= 0 or count > shown[pos] or (inside and count != len(entry["lines"])):
            return "audit_hold:open_lines_disagree", []
        if count == shown[pos] and shown[pos] < len(entry["lines"]):
            return "audit_hold:open_lines_longer_than_shown", []       # the text goes on beyond what was shown
        restore.append((k, count))
    return None, restore


def _put_back(pair: MutableMapping[str, Any], restore: Sequence[Tuple[int, int]]) -> None:
    """Put unresolved lines that the judges found on the photograph back into a pair's text.

    The pair's side keys follow its lines: a line of another side of the edition brings that side into the pair,
    so the novelty and hold-out verdicts of that side and its source text count for the pair too.

    :param pair: Pair record (changed in place: ``lines``, ``line_refs``, ``line_indices``, ``side_keys``,
        ``side_key``, ``open_lines``).
    :param restore: ``[(entry index, lines to put back)]``; the lines nearest to the pair's text are taken.
    """
    jobs = []
    for k, count in restore:
        entry = pair["open_lines"][k]
        if entry.get("after_line") is not None:
            at, picked = int(entry["after_line"]), list(range(count))
        else:
            at, picked = int(entry["before_line"]) - 1, list(range(len(entry["lines"]) - count, len(entry["lines"])))
        jobs.append((at, k, picked))
    for at, k, picked in sorted(jobs, reverse=True):                    # from the end, so positions stay valid
        entry = pair["open_lines"][k]
        pair["lines"][at:at] = [entry["lines"][i] for i in picked]
        pair["line_refs"][at:at] = [list(entry["refs"][i]) for i in picked]
        pair["line_indices"][at:at] = [entry["refs"][i][1] for i in picked]
        entry["settled"] = f"restored:{len(picked)}"
    sides: List[str] = []
    for side, _ in pair["line_refs"]:
        if side not in sides:
            sides.append(side)
    pair["side_keys"], pair["side_key"] = sides, sides[0]
    pair.pop("unread_lines", None)                                      # its numbers no longer fit the text


def _parallel(pair: Mapping[str, Any]) -> bool:
    """Whether a pair's line references run parallel to its lines, so a line can be removed or inserted.

    :param pair: Pair record.
    :return: True when ``line_refs`` and ``line_indices`` are lists as long as ``lines``.
    """
    return all(isinstance(pair.get(key), list) and len(pair[key]) == len(pair["lines"])
               for key in ("line_refs", "line_indices"))


def decide(pair: MutableMapping[str, Any], verdict: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    """Decision of a judged pair (see the module docstring for the rules).

    :param pair: Pair record. Changed in place when two judges agree to remove lines that are not on the photograph
        (they are kept in ``lines_removed_by_judges``) or to put back unresolved lines that are; its settled
        ``open_lines`` entries are marked. :func:`apply_verdicts` therefore decides on a copy.
    :param verdict: Its result record with each judge's own record (``records``).
    :return: ``(decision, flag or hold reason)``: ``accept`` with ``audit_complete``, ``audit_lines_dropped`` or
        ``audit_lines_restored``; otherwise ``review`` with ``audit_hold:<why>`` (``one_judge_complete`` and
        ``one_judge_edit`` name the pairs a second judge would settle).
    """
    records = judge_records(verdict)
    if not records:
        return "review", "audit_hold:no_judge_record"
    contract = "v2" if verdict.get("contract") == "v2" else "v1"
    verdicts = {rec["verdict"] for rec in records.values()}
    final = max(verdicts, key=lambda value: SEVERITY.get(value, -1))
    found = {name: concerns(rec, contract) for name, rec in records.items()}
    # a judge's line numbers are numbers of the text the judge was shown: the pair's own lines, unless the text
    # rules changed the number of lines of a second-contract item
    numbered = contract == "v1" or bool(verdict.get("lines_match"))
    if verdicts == {"complete_pair"}:
        if any(found.values()):
            return "review", "audit_hold:complete_with_concerns"
        if len(records) < JUDGES_FOR_APPROVAL:
            return "review", "audit_hold:one_judge_complete"
        why, restore = _settle_open(pair, verdict, records)
        if why:
            return "review", why
        if restore:
            if not numbered:
                return "review", "audit_hold:line_numbers_shifted"
            if not _parallel(pair):
                return "review", "audit_hold:open_lines_changed"
            _put_back(pair, restore)
        for entry in pair.get("open_lines") or []:
            entry.setdefault("settled", "absent")
        return "accept", "audit_lines_restored" if restore else "audit_complete"
    if verdicts == {"text_has_extra"}:
        n_lines = len(pair["lines"])
        lists = [rec.get("lines_not_on_photo") for rec in records.values()]
        valid = all(isinstance(numbers, list) and numbers and all(
            isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= n_lines for n in numbers)
                    for numbers in lists)
        other = any(c not in OWN_TO_TEXT_EXTRA for cs in found.values() for c in cs)
        if not valid or other or len(set(lists[0])) > MAX_DROP_SHARE * n_lines:
            return "review", "audit_hold:text_has_extra"
        if len(records) < JUDGES_FOR_EDIT:
            return "review", "audit_hold:one_judge_edit"
        if any(sorted(set(numbers)) != sorted(set(lists[0])) for numbers in lists):
            return "review", "audit_hold:text_has_extra"
        if not numbered:
            return "review", "audit_hold:line_numbers_shifted"
        if not _parallel(pair):
            return "review", "audit_hold:text_has_extra"
        why, restore = _settle_open(pair, verdict, records)
        if why or restore:
            return "review", why or "audit_hold:text_has_extra"
        gone = sorted({n - 1 for n in lists[0]})
        pair.setdefault("lines_removed_by_judges", []).append(
            {"audit": verdict.get("audit"), "numbers": [k + 1 for k in gone],
             "refs": [list(pair["line_refs"][k]) for k in gone], "lines": [pair["lines"][k] for k in gone]})
        for key in ("lines", "line_refs", "line_indices"):
            pair[key] = [v for k, v in enumerate(pair[key]) if k not in set(gone)]
        pair.pop("unread_lines", None)                                  # its numbers no longer fit the text
        for entry in pair.get("open_lines") or []:
            entry.setdefault("settled", "absent")
        return "accept", "audit_lines_dropped"
    if final == "complete_pair":                                        # one judge found it complete, one could not judge
        final = "cannot_judge"
    return "review", f"audit_hold:{final}"


def image_of(record: Mapping[str, Any]) -> List[Any]:
    """The photograph as a pair shows it.

    :param record: A pair record or an audit key record (``image_sha256``, ``rotation``, ``crop``).
    :return: ``[digest, rotation, crop box or None]``.
    """
    crop = [int(v) for v in record["crop"]] if record.get("crop") else None
    return [record.get("image_sha256"), int(record.get("rotation") or 0), crop]


def load_audit(folder: Path, name: str, key_dirs: Sequence[Path] = ()) -> Dict[str, Dict[str, Any]]:
    """One audit's results by pair id, each with the judged lines, the judged image and each judge's own record.

    The judges' records are read from their own files when those are in the folder (the source of every result);
    a result's stored ``records`` are used otherwise. Merged fields of older result files are never read.

    :param folder: Folder of ``judge_items_<name>.jsonl``, ``audit_<name>_results.jsonl`` and the judges' files.
    :param name: Audit name.
    :param key_dirs: Folders searched for the audit's hidden key ``<name>_key.jsonl`` (the first that has it).
    :return: ``{pair id: result record + "lines" + "records" + "audit" + "judged_image" + "open_lines_shown"}``;
        ``lines`` and ``open_lines_shown`` are what the judges were shown (the item's text and its unresolved
        lines), ``judged_image`` is :func:`image_of` the key record, or None when no key was found. The key's
        second-contract fields (fingerprint, digest of the unresolved lines, lines shown) take the place of the
        result's.
    """
    items = {r["item_id"]: r for r in load_jsonl(folder / f"judge_items_{name}.jsonl")}
    own = {judge: load_judgments(folder, name, judge) for judge in ("A", "B")}
    key_path = next((d / f"{name}_key.jsonl" for d in key_dirs if (d / f"{name}_key.jsonl").exists()), None)
    keys = {k["item_id"]: k for k in load_jsonl(key_path)} if key_path else {}
    out: Dict[str, Dict[str, Any]] = {}
    for result in load_jsonl(folder / f"audit_{name}_results.jsonl"):
        item_id = result["item_id"]
        records = {judge: own[judge][item_id] for judge in ("A", "B") if item_id in own[judge]}
        key = keys.get(item_id) or {}
        bound = {f: key[f] for f in ("contract", "fingerprint", "open_shown", "open_digest", "lines_match")
                 if key.get(f) is not None}
        out[result["pair_id"]] = {**result, **bound, "records": records or result.get("records") or {},
                                  "lines": (items.get(item_id) or {}).get("lines"), "audit": name,
                                  "open_lines_shown": (items.get(item_id) or {}).get("open_lines"),
                                  "judged_image": image_of(key) if key else None}
    return out


def load_audits(folder: Path, names: Sequence[str], key_dirs: Sequence[Path] = ()
                ) -> Dict[str, List[Dict[str, Any]]]:
    """Results of several audits by pair id: every audit of a pair, oldest first. None replaces another.

    :param folder: The audits' folder.
    :param names: Audit names, oldest first.
    :param key_dirs: Folders searched for each audit's hidden key.
    :return: ``{pair id: [result, ...]}``.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name in names:
        for pair_id, result in load_audit(folder, name, key_dirs).items():
            out.setdefault(pair_id, []).append(result)
    return out


def merge_verdicts(verdicts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Several audits' verdicts of one contract on one version of a pair, read as one: every judge counts.

    :param verdicts: Applicable results of the same contract, oldest first (at least one).
    :return: The latest result with the records of every judge of every audit (keys ``<audit>:<judge>``), the
        worst verdict as ``final`` and the audits' names as ``audits``. A single result is returned as it is.
    """
    if len(verdicts) == 1:
        return dict(verdicts[0])
    records = {f"{v['audit']}:{name}": rec for v in verdicts for name, rec in judge_records(v).items()}
    final = max((rec["verdict"] for rec in records.values()), key=lambda value: SEVERITY.get(value, -1),
                default="cannot_judge")
    return {**verdicts[-1], "records": records, "final": final, "audits": [v["audit"] for v in verdicts]}


def _same_open_lines(shown: Optional[Sequence[Mapping[str, Any]]], pair: Mapping[str, Any]) -> bool:
    """Whether the unresolved lines a judge was shown are the ones beside the pair's text now, letter for letter.

    :param shown: The judged item's ``open_lines`` (None when the item is unknown).
    :param pair: Pair record.
    :return: True when the entries stand at the same places, list as many lines beyond the ones shown and hold the
        same letters line by line.
    """
    now = audit_contract.judge_open_lines(pair)
    return shown is not None and len(shown) == len(now) and all(
        a.get("position") == b["position"] and a.get("not_shown") == b["not_shown"]
        and same_letters(a.get("lines"), b["lines"]) for a, b in zip(shown, now))


def verdict_match(pair: Mapping[str, Any], verdict: Mapping[str, Any], cands: Mapping[str, Mapping[str, Any]],
                  models: Mapping[str, Any], mirrored: Set[str]) -> Optional[str]:
    """How a verdict fits a pair as it is now.

    :param pair: The run's pair record.
    :param verdict: A result of :func:`load_audit`.
    :param cands: Candidate records by row id.
    :param models: clean_v2 3-gram models.
    :param mirrored: clean_v2 mirrored editions.
    :return: ``exact``: for the second contract the judged fingerprint is the pair's (photograph, rotation, crop,
        source text and final answer all equal) and the unresolved lines beside its text are the ones the judges
        were shown; for the first, the judged lines are the pair's lines, final letters aside, on the judged
        photograph, rotation and crop. ``same_letters``: the photograph, rotation and crop are the judged ones and
        the text (second contract: the final answer and the unresolved lines shown; first: the pair's lines, final
        letters aside) holds the judged letters line by line, so only spaces, punctuation or gap marks differ.
        None: another version, or a verdict whose key is lost.
    """
    if verdict.get("contract") == "v2":
        answer = build_rows.final_answer(pair, cands, models, mirrored)
        if answer.fails:
            return None
        if (verdict.get("fingerprint") == build_rows.pair_fingerprint(pair, answer, cands)
                and verdict.get("open_digest") == audit_contract.open_digest(pair)):
            return "exact"
        if (verdict.get("judged_image") == image_of(pair) and same_letters(verdict.get("lines"), answer.lines)
                # the judges' line numbers must still be numbers of the pair's own lines (see :func:`decide`)
                and (not verdict.get("lines_match") or len(answer.lines) == len(pair["lines"]))
                and _same_open_lines(verdict.get("open_lines_shown"), pair)):
            return "same_letters"
        return None
    if verdict.get("judged_image") != image_of(pair):
        return None
    if same_lines(verdict.get("lines"), pair["lines"]):
        return "exact"
    return "same_letters" if same_letters(verdict.get("lines"), pair["lines"], finals=False) else None


def verdict_applies(pair: Mapping[str, Any], verdict: Mapping[str, Any], cands: Mapping[str, Mapping[str, Any]],
                    models: Mapping[str, Any], mirrored: Set[str]) -> bool:
    """Whether a verdict was given on this pair's text (:func:`verdict_match`: exactly, or letter for letter).

    :param pair: The run's pair record.
    :param verdict: A result of :func:`load_audit`.
    :param cands: Candidate records by row id.
    :param models: clean_v2 3-gram models.
    :param mirrored: clean_v2 mirrored editions.
    :return: True when the verdict applies.
    """
    return verdict_match(pair, verdict, cands, models, mirrored) is not None


def _adverse(verdict: Mapping[str, Any]) -> bool:
    """Whether a result holds anything against the pair it judged.

    :param verdict: A result record.
    :return: True unless every judge found the pair complete without a concern.
    """
    contract = "v2" if verdict.get("contract") == "v2" else "v1"
    records = judge_records(verdict)
    return not records or any(rec["verdict"] != "complete_pair" or concerns(rec, contract)
                              for rec in records.values())


def conflict_with_earlier(earlier: Mapping[str, Any], note: Optional[str], judges: int) -> bool:
    """Whether an approval by second-contract judges leaves an earlier judge's observation unanswered.

    :param earlier: The merged first-contract verdict on the same pair.
    :param note: The flag of the later approval (``audit_complete``, ``audit_lines_restored``,
        ``audit_lines_dropped``).
    :param judges: Number of second-contract judges behind it.
    :return: True when an earlier judge saw writing the text leaves out and no line was put back, saw text that is
        not on the photograph and no line was removed, called the text wrong, or noted restored words or a clear
        order problem that fewer than :data:`JUDGES_FOR_EDIT` later judges looked at.
    """
    for record in judge_records(earlier).values():
        found = concerns(record, "v1")
        verdict = record["verdict"]
        if verdict == "wrong_text":
            return True
        if (verdict in ("photo_has_extra", "both_partial") or any(c.startswith("uncovered") for c in found)) \
                and note != "audit_lines_restored":
            return True
        if (verdict in ("text_has_extra", "both_partial") or "lines_not_on_photo" in found) \
                and note != "audit_lines_dropped":
            return True
        if any(c in ANSWERED_BY_TWO or c.startswith("record:") for c in found) and judges < JUDGES_FOR_EDIT:
            return True
    return False


def _hold(pair: MutableMapping[str, Any], reason: str, counts: Counter) -> None:
    """Send an accepted pair back to review.

    :param pair: Pair record (changed in place).
    :param reason: Why.
    :param counts: Counter of what happened.
    """
    pair["decision"] = "review"
    pair["reasons"] = sorted(set(pair.get("reasons") or []) | {reason})
    pair.pop("approval", None)
    counts[f"accept->review:{reason}"] += 1


def apply_verdicts(pairs: Sequence[MutableMapping[str, Any]], verdicts: Mapping[str, Sequence[Mapping[str, Any]]],
                   cands: Mapping[str, Mapping[str, Any]], models: Mapping[str, Any], mirrored: Set[str]
                   ) -> Counter:
    """Apply audit verdicts to a run's pairs and bind the approval of every accepted pair.

    A judged pair takes the decision of :func:`decide` on the verdicts that apply to it (second contract before
    first, each contract's audits read together). A change of its text that the judges prescribed is kept only
    when the approval of the changed text can be bound; otherwise the pair goes to review as the assembly made it.
    Afterwards every pair that stands as accepted by the run's own rule gets its approval bound too, unless an
    adverse verdict exists on another version of it; a pair whose approval cannot be bound
    (:func:`build_rows.stamp_approval`) goes to review with the reason. No line may end in two accepted pairs of
    one candidate.

    :param pairs: The run's pair records (changed in place).
    :param verdicts: Every result per pair id, oldest first (:func:`load_audits`).
    :param cands: Candidate records by row id.
    :param models: clean_v2 3-gram models.
    :param mirrored: clean_v2 mirrored editions.
    :return: Counts of what happened.
    """
    counts: Counter = Counter()
    for pair in pairs:
        before = pair["decision"]
        mine = list(verdicts.get(pair["pair_id"]) or [])
        matches = [(v, verdict_match(pair, v, cands, models, mirrored)) for v in mine]
        applicable = [v for v, how in matches if how]
        carried = any(how == "same_letters" for _, how in matches)
        if not mine:
            counts[f"unjudged_{before}"] += 1
        elif not applicable:
            counts["judged_but_pair_changed"] += 1
            if before == "accept" and any(_adverse(v) for v in mine):
                _hold(pair, "earlier_verdict_on_other_version", counts)
        else:
            second = [v for v in applicable if v.get("contract") == "v2"]
            first = [v for v in applicable if v.get("contract") != "v2"]
            verdict = merge_verdicts(second or first)
            if verdict.get("contract") != "v2" and verdict.get("lines") != pair["lines"]:
                counts["judged_before_finals_restored"] += 1
            if carried:
                counts["judged_same_letters"] += 1
            records = judge_records(verdict)
            contract = "v2" if verdict.get("contract") == "v2" else "v1"
            trial = copy.deepcopy(pair)                                 # a prescribed change is kept only if approved
            decision, note = decide(trial, verdict)
            if decision == "accept" and second and first and conflict_with_earlier(merge_verdicts(first), note,
                                                                                    len(records)):
                decision, note = "review", "audit_hold:earlier_judges_differ"
            why = None
            if decision == "accept":
                trial["decision"], trial["reasons"] = "accept", []
                trial["flags"] = sorted(set(trial.get("flags") or []) | {note})
                why = build_rows.stamp_approval(trial, cands, models, mirrored, "judges", contract=contract,
                                                audit=verdict["audit"], item_id=verdict["item_id"],
                                                judges=sorted(records), text_judged="same_letters" if carried else "exact",
                                                answer_seen=contract == "v2" and note == "audit_complete")
                if why is None:
                    pair.clear()
                    pair.update(trial)
            if decision != "accept" or why:
                pair["decision"] = "review"
                pair["reasons"] = sorted(set(pair.get("reasons") or []) | {why or note})
            pair["audit"] = {"name": verdict["audit"], "item_id": verdict["item_id"], "final": verdict.get("final"),
                             "was": before, "note": note, "contract": contract,
                             "match": "same_letters" if carried else "exact",
                             "audits": [v["audit"] for v in applicable],
                             "judges": {name: rec["verdict"] for name, rec in records.items()},
                             "concerns": {name: concerns(rec, contract) for name, rec in records.items()}}
            counts[f"{before}->{decision}:{note}"] += 1
            if why:
                counts[f"accept->review:{why}"] += 1
            elif decision == "accept":
                counts["approved_by_judges"] += 1
            continue
        if pair["decision"] != "accept":
            continue
        why = build_rows.stamp_approval(pair, cands, models, mirrored, "rule")
        if why:
            _hold(pair, why, counts)
        else:
            counts["approved_by_rule"] += 1
    holders: Dict[Tuple[str, str, int], List[MutableMapping[str, Any]]] = {}
    for pair in pairs:                                                  # a line put back into two texts is in neither
        if pair["decision"] == "accept" and not (pair.get("dup_rank") or 0):
            for side, index in pair.get("line_refs") or []:
                holders.setdefault((pair["row_id"], side, index), []).append(pair)
    for held in holders.values():
        if len(held) > 1:
            for pair in held:
                if pair["decision"] == "accept":
                    _hold(pair, "line_in_two_pairs", counts)
    return counts


def strict_rate(results: Sequence[Mapping[str, Any]]) -> Tuple[int, int]:
    """Judged items and those every judge found complete without any concern.

    :param results: Result records (:func:`score_item`).
    :return: ``(items at least one judge could judge, items complete under the rules of :func:`decide`)``. Items
        no judge could judge are left out of both; a release audit counts them against the rate
        (:mod:`.release_audit`).
    """
    judged = complete = 0
    for result in results:
        records = judge_records(result)
        verdicts = {rec["verdict"] for rec in records.values()}
        if not records or verdicts == {"cannot_judge"}:
            continue
        judged += 1
        if not _adverse(result):
            complete += 1
    return judged, complete
