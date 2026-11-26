# File name: audit_contract.py
# Date: 11/22/26
# Author: Isaac Godfried. Coded by Claude.
"""The second contract of the blind audit of photograph/text pairs (round ja_ar_v1): what a judge is asked.

The first contract (``logs/next_round/ja_ar_v1/build/audit3/make_prompts.py``, audits up to 2026-11-19) let a judge
call a pair complete with "minor" writing left out, said that restored words and order problems "do not change the
verdict", and showed the judge the pair's assembled lines, which the row builder changed afterwards. The integrity
audit of 2026-11-22 traced the omissions of the model trained on those rows to exactly these openings. This contract
closes them:

* the judge sees the final answer of the row (:func:`build_rows.final_answer`) on the image the row would show;
* complete means complete: every line the document's scribes wrote on the photograph, in reading order, and nothing
  else (:data:`approval.SCOPE`); any uncovered in-scope writing, restored word, clear order problem or garbled line
  makes a pair not complete;
* the judge is shown the unresolved lines next to the text (``open_lines`` of :mod:`.assemble`) and says, for each
  stretch, whether they stand on the photograph;
* marginal notes and other hands are in scope; modern library matter and a neighbouring leaf's edge are not.

:mod:`.adjudicate` reads the answers. The template's fields are ``{lines} {items} {py} {work} {out}``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional

#: Name of this contract in keys, results and approvals.
CONTRACT = "v2"
#: Unresolved lines shown per stretch: the ones nearest to the pair's text.
OPEN_SHOWN = 8

TEMPLATE = """You are auditing photograph/text pairs for a handwriting-recognition training set of Cairo Genizah documents (medieval Hebrew-script and Arabic-script manuscripts and papyri). This is a careful reading task, not a coding task.

Each item is ONE photograph of a manuscript page and ONE transcription that a pairing system proposes as the training target for exactly this photograph. A training row tells the model "this photograph reads as this text". Text that is not on the photograph teaches the model to invent; writing on the photograph that the text leaves out teaches it to skip. An earlier round of this audit let pairs with "minor" omissions pass, and the model trained on them learned to leave out lines at the ends of pages. So the standard here is exact: the text must hold every line that the scribes of this document wrote on this photograph, in the order they are read, and nothing that is not written there.

Your items: lines {lines} (1-based line numbers) of {items}. Each line is JSON: {{item_id, image_path, lines: [...], open_lines: [...]}}. Read only those lines (for example with a short Python snippet printing them). Do not read any other file in that directory or elsewhere about these items, and do not search the repository for information about them: the audit is blind.

Facts you need
- Texts are in Hebrew script (Hebrew, Aramaic, or Judaeo-Arabic, that is Arabic language in Hebrew letters) or in Arabic script; a text can mix both. "[...]" marks a place the editor could not read; it stands for ink, or for a hole, that is on the photograph.
- Edition lines usually correspond to manuscript lines. Letters often continue in the margins (written sideways or upside-down); editions place margin lines after the main text, often as a run of short lines.
- A photograph shows one face of a leaf or one page; it may have been cut down to one page of an opening. Photographs include scale bars and shelf-mark labels; a strip of a neighbouring leaf may be visible at an edge.
- In scope is everything the scribes of this document wrote on this photograph: the main text, and also marginal notes, words added between the lines, addresses, dockets and writing in another hand. Out of scope, and allowed on the photograph without being in the text: modern library matter (shelf-mark labels, stamps, pencilled folio numbers), scale bars and colour charts, and letters of a neighbouring leaf at an edge.
- open_lines: the edition this text was taken from has further lines next to this text that the pairing system could not place on any photograph. Each entry has a number (entry), says where the lines would stand (position: after a text line, before a text line, or between two text lines) and lists them in the edition's order (lines); not_shown says how many more lines follow beyond the ones listed. These lines may be on this photograph (lines a machine reader could not read), on another photograph, or not manuscript text at all (an editor's note). Nothing is known about which: you decide by looking.

Method for every item (do every step)
1. View the photograph with your image-reading tool. If the writing is too small to read reliably, crop bands (top third, middle, bottom third, each margin) at native resolution with PIL ({py}) into {work}/ and view those; rotate crops for sideways or upside-down writing.
2. BEFORE comparing with the text, read for yourself the first two and the last two lines of the main text block, and one or two lines in the middle.
3. Look for those lines in the text. Decide how much of the text is on the photograph: all / most (at least 80 %) / part / none. List the text line numbers (1-based) that are NOT on the photograph and say what they seem to be (other face, other fragment, cannot find).
4. Count the written lines on the photograph (main block; margin lines separately) and compare with the text's line count. If the photograph has more lines than the text, find out where they are: above the first text line, below the last, between two text lines, in a margin.
5. For every open_lines entry, look at the place its position names and say whether its lines are written there (on_photo): none (the place is blank, torn away, or holds other writing) / all (every listed line is there) / some (give count: how many of the listed lines are there, counting from the text outward, that is the first lines of an entry that follows a text line and the last lines of an entry that precedes one) / cannot_tell (there is writing at that place and you cannot read enough of it to say). Do not answer "none" because you could not read the place: that is cannot_tell.
6. uncovered_writing: once the text lines and the open lines you found are accounted for, does the photograph still hold in-scope writing that neither covers? none / some. "none" means none at all: one further line, a marginal note of three words, an address or a docket are all "some". Describe it (uncovered_description) and give its kind (uncovered_kind): main_text_lines (further lines of the same text block, above, below or between the transcribed lines) / marginal_or_other_hand (notes in a margin, a later or different hand, pen trials, an address, a docket) / other_page_or_text (a further page of an opening, or a separate text block on the sheet) / none. If several apply, give the one that holds the most writing.
7. supplied_text: does the text contain words where the photograph has no ink (a hole, a tear, a lost or rubbed-away area), that is, words an editor restored? none / some. Give the text line numbers (supplied_lines).
8. order_problem: within lines you compared closely, are words or letters in a different order in the text than on the photograph, or are whole lines in a different order than they are read on the page? none / suspected / clear. Give the text line numbers (order_lines).
9. garbled_lines: text line numbers whose letters are not what is written at that place although a line is written there (not a letter or two: most of the line differs, or its letters are scrambled).
10. Verdict, for the text TOGETHER WITH the open lines you found on the photograph:
   complete_pair: it holds every in-scope line of the photograph, in reading order, and nothing that is not there. It is NOT complete when step 3 lists a line, step 6 says some, step 7 says some, step 8 says clear, step 9 lists a line, or step 5 has a cannot_tell.
   text_has_extra: text lines are not on this photograph (step 3) and nothing in scope is left uncovered.
   photo_has_extra: in-scope writing is left uncovered (step 6) and every text line is on the photograph.
   both_partial: both of these.
   wrong_text: the text is not what is on the photograph.
   cannot_judge: illegible or too uncertain.
   Confidence 1-5 (5 = first, middle and last lines matched word for word and the line count agrees; 1 = little more than a guess). Quote two or three words you actually read on the photograph and the text line where they occur.
Genizah cursive is hard. "cannot_judge" or "cannot_tell" with a reason is better than a guess, and a low confidence is information we need. Nobody is helped by a pair passed as complete that is not.

Rules: use nothing but viewing and cropping the photographs and reading your item lines. No OCR services, no model or API calls, no network. Do not modify anything outside {work}/ and your output file. The photographs are private; do not copy them anywhere else.

Write your judgments as a JSON array to {out} (one object per item: item_id, share_on_photo, lines_not_on_photo, photo_lines_main, photo_lines_margin, open_lines (an array with one object per entry: entry, on_photo, count, note), uncovered_writing, uncovered_kind, uncovered_description, supplied_text, supplied_lines, order_problem, order_lines, garbled_lines, verdict, confidence, matched_evidence, notes). Field formats, which a program checks (a record with a missing or differently worded field approves nothing): for share_on_photo (all / most / part / none), on_photo, uncovered_writing, uncovered_kind, supplied_text, order_problem and verdict write exactly one of the words given above, in lower case, never null and never empty (write "none" where you found nothing); lines_not_on_photo, supplied_lines, order_lines and garbled_lines are JSON arrays of text line numbers ([] when there are none); photo_lines_main, photo_lines_margin, confidence and count are integers (count is 0 unless on_photo is "some"); open_lines holds one object for every entry of the item, in the item's order ([] when the item has no entry). Check that the file parses as JSON. Then return as your final message only one line per item, "<item_id> <verdict> <confidence>", nothing else.
"""


def position_text(entry: Mapping[str, Any]) -> str:
    """Where an entry of unresolved lines would stand, in the judge's words.

    :param entry: An ``open_lines`` entry of a pair (``after_line``, ``before_line``: 1-based text line numbers).
    :return: The position.
    """
    after, before = entry.get("after_line"), entry.get("before_line")
    if after is not None and before is not None:
        return f"between text lines {after} and {before}"
    if after is not None:
        return f"after text line {after}"
    return f"before text line {before}"


def judge_open_lines(pair: Mapping[str, Any], shown: int = OPEN_SHOWN) -> List[Dict[str, Any]]:
    """The unresolved lines beside a pair's text as a judge is shown them.

    Entries that were settled already are left out. Of a long stretch only the lines nearest to the pair's text
    are listed: its first lines when it follows a text line, its last lines when it only precedes one.

    :param pair: Pair record.
    :param shown: Most lines listed per entry.
    :return: ``[{"entry": 1-based number, "position": text, "lines": [...], "not_shown": count}]``.
    """
    out: List[Dict[str, Any]] = []
    for entry in pair.get("open_lines") or []:
        if entry.get("settled"):
            continue
        lines = list(entry["lines"])
        listed = lines[:shown] if entry.get("after_line") is not None else lines[-shown:]
        out.append({"entry": len(out) + 1, "position": position_text(entry), "lines": listed,
                    "not_shown": len(lines) - len(listed)})
    return out


def open_digest(pair: Mapping[str, Any]) -> Optional[str]:
    """Digest of the unresolved lines beside a pair's text: which lines, of which side, at which place.

    A verdict on unresolved lines applies only while they are the lines the judges were shown
    (:func:`adjudicate.verdict_applies`): an assembly run that leaves the answer as it was but changes the lines
    beside it must not inherit answers given about other lines.

    :param pair: Pair record.
    :return: Hex SHA-256 over every unsettled entry (position, references, lines), or None when there is none.
    """
    entries = [[e.get("after_line"), e.get("before_line"), [list(r) for r in e.get("refs") or []], list(e["lines"])]
               for e in pair.get("open_lines") or [] if not e.get("settled")]
    if not entries:
        return None
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False).encode("utf-8")).hexdigest()


def open_shown(pair: Mapping[str, Any], shown: int = OPEN_SHOWN) -> Optional[List[int]]:
    """Lines listed per unsettled entry, for the hidden key (:func:`adjudicate.decide` checks answers against it).

    :param pair: Pair record.
    :param shown: Most lines listed per entry.
    :return: Counts in entry order, or None when the pair has no unsettled entry.
    """
    counts = [len(e["lines"]) for e in judge_open_lines(pair, shown)]
    return counts or None
