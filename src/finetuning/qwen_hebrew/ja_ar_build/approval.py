# File name: approval.py
# Date: 11/22/26
# Author: Isaac Godfried. Coded by Claude.
"""What an approval of a photograph/text pair covers, and the fingerprint that binds it (round ja_ar_v1).

Why. The integrity audit of 2026-11-22 found that a pair approved by the judges could change afterwards: the row
builder removed further lines from the approved text, and the approval stayed. An approval is therefore bound to
everything a row is made of. :func:`fingerprint` is the digest of

* the declared output scope (:data:`SCOPE`),
* the photograph as the row shows it: its digest, its rotation and its crop,
* the source text the answer was taken from (:func:`source_digest`: the full lines of every side the pair draws on,
  so a re-ingest of the transcription voids the approval),
* the final answer, character for character.

:mod:`.adjudicate` stamps the fingerprint when a pair is approved (by the assembly's rule or by judges);
:mod:`.build_rows` computes it again for the answer it is about to write and builds no row when the two differ.
Nothing may change a pair between its approval and its row.

Scope ``page_text_v1``. A row's answer is every line that the document's scribes wrote on the photograph as the row
shows it (after crop and rotation), each in the reading order of its page, and nothing that is not written there:
no editor's note, label, translation or restored word. Out of scope, and so allowed on the photograph without a
line in the answer: modern library matter (shelf-mark labels, stamps, pencilled folio numbers), scale bars, colour
charts, and the stray edge of a neighbouring leaf. Marginal notes and other hands are in scope: a pair whose
photograph shows them and whose text leaves them out is held until the text covers them or a crop leaves them out
(the evaluation ground truths transcribe margins, and a row that skips them teaches skipping).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, List, Mapping, Optional, Sequence

#: The declared output scope of a row (see the module docstring). Changing what a row's answer must cover means a
#: new scope name, which voids every earlier approval.
SCOPE = "page_text_v1"


def side_lines(cand: Mapping[str, Any], side_key: str) -> List[str]:
    """The non-empty lines of one target side of a candidate, as the gate and the row builder index them.

    :param cand: Candidate record.
    :param side_key: Side key.
    :return: The side's lines.
    :raises KeyError: When the candidate has no such side.
    """
    for target in cand.get("targets") or []:
        if (target.get("side_key") or "") == side_key:
            return [line for line in target.get("lines") or [] if line.strip()]
    raise KeyError(f"candidate {cand.get('row_id')} has no side {side_key!r}")


def source_digest(cand: Mapping[str, Any], side_keys: Sequence[str]) -> str:
    """Digest of the source text a pair draws on: the full lines of each of its sides.

    :param cand: Candidate record.
    :param side_keys: The pair's side keys, in the pair's order.
    :return: Hex SHA-256.
    """
    payload = [str(cand.get("row_id")), str(cand.get("source")), [[key, side_lines(cand, key)] for key in side_keys]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def fingerprint(pair: Mapping[str, Any], answer: str, source: str, scope: str = SCOPE) -> str:
    """Digest of everything an approval of a pair covers.

    :param pair: Pair record (``image_sha256``, ``rotation``, ``crop``).
    :param answer: The final answer text of the row.
    :param source: :func:`source_digest` of the pair.
    :param scope: The declared output scope.
    :return: Hex SHA-256.
    """
    crop = [int(v) for v in pair["crop"]] if pair.get("crop") else None
    payload = [scope, pair.get("image_sha256"), int(pair.get("rotation") or 0), crop, source, answer]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def edition_review_key(pair: Mapping[str, Any], current: str) -> str:
    """Bind a source inspection to the exact PDF, transformed photograph, and printed lines.

    :param pair: Edition pair with its source evidence and open line inventory.
    :param current: Final-answer fingerprint from the ordinary builder.
    :return: Review key before or after the reviewer settles an open line.
    """
    evidence = pair["edition"]
    issues = [{k: v for k, v in issue.items() if k != "settled"} for issue in pair["open_lines"]]
    payload = [current, evidence["pdf_sha256"], evidence["view_sha256"], evidence["line_ids"], issues]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def unsettled_open_lines(pair: Mapping[str, Any]) -> int:
    """Unresolved lines next to a pair's text that nobody has settled yet.

    :param pair: Pair record (``open_lines`` of :mod:`.assemble`; an entry a judge settled carries ``settled``).
    :return: Number of such lines.
    """
    return sum(len(entry.get("lines") or []) for entry in pair.get("open_lines") or [] if not entry.get("settled"))


def approval_state(pair: Mapping[str, Any], current: str) -> Optional[str]:
    """Why a pair's approval does not cover the answer about to be written, or None when it does.

    :param pair: Pair record.
    :param current: Fingerprint of the pair as it would be built now.
    :return: ``no_approval``, ``approval_other_scope``, ``approval_void`` (the pair changed since its approval),
        ``unresolved_lines`` (approved although lines next to its text are unsettled) or None.
    """
    approval = pair.get("approval")
    if not approval or not approval.get("fingerprint"):
        return "no_approval"
    if approval.get("scope") != SCOPE:
        return "approval_other_scope"
    if approval["fingerprint"] != current:
        return "approval_void"
    if approval.get("by") == "source_review" and (
            not pair.get("edition") or approval.get("review_key") != edition_review_key(pair, current)):
        return "approval_void"
    if unsettled_open_lines(pair):
        return "unresolved_lines"
    return None
