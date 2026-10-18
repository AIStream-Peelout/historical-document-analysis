# File name: build_readings.py
# Date: 10/13/26
# Author: Isaac Godfried. Coded originally by Claude Sonnet 5.5.
"""Per-page "readings" sidecars: every reader's lines for a page, for the website's reader-comparison view.

:mod:`src.datasets.consensus.two_reader_lines` keeps, per page, the VLM's raw replies and Kraken's fragments in the
raw cache (``ai_reads/raw/<stem>.json``) and one merged record per page in ``ai_reads_<model>.jsonl``. This module
reshapes the two into ONE JSON file per page, ``ai_reads/readings/<stem>.json`` (schema ``readings-v1``, described in
``docs/readings_sidecar.md``): the whole-page VLM read, each decode-policy unit read (column crops, rescue halves, the
halves of a failed column), Kraken's fragments, the policy trace and the final merged lines, all in the page's 0-1000
frame. It is a reshaping: no model, Kraken, Elasticsearch or network call, and it never writes to the records or the
raw cache.

Reply to unit mapping. A page read under ``--decode-policy`` stores ``vlm_raw`` as a list parallel to
``decode_policy["requests"]`` (see ``PolicyResult.vlm_raw``; a single request stores the bare string): reply ``i``
answers request ``i``, whose ``unit`` / ``role`` / ``crop`` the trace records. The whole-page reply is NOT always first
(a column page that needed a whole-page read gets it last), so replies are attached by the request's unit name, never by
position alone. Each attribution is verified against the request (the reply has the length the trace recorded); when
the list and the trace disagree, only a verified leading whole-page reply is attributed and every other reader carries
``reply_index: null`` and ``lines: null`` with a note (``provenance.reply_mapping`` = ``partial``). No such page exists
in the 2026-10-13 data. A parse that differs from the one the trace recorded (the parser changed since) is noted, not
treated as a mismatch.

Where each reader's lines come from: a crop reply is parsed with ``two_reader_lines.parse_grounded`` and its boxes are
mapped to the page frame with ``reread.to_page_box``; the whole-page read of a page without a policy trace (and the
cached whole-page read of a ``--revisit``) is the parse STORED when the page was read, because that parse is what the
final lines were built from (today's parser recovers more lines from 280 of the 37,000 old replies); Kraken's lines are the
fragments of the record's HTR cache key.

Usage (repo root, CPU only):
    .venv/bin/python -m src.datasets.consensus.build_readings --records ai_reads_A.jsonl ai_reads_B.jsonl
    .venv/bin/python -m src.datasets.consensus.build_readings --records ai_reads_A.jsonl --since 2026-10-13T20:00:00Z
    .venv/bin/python -m src.datasets.consensus.build_readings --records ai_reads_A.jsonl --force --limit 100

A sidecar is skipped when it is current: same schema and builder version, and built from this record or a newer one
(``decoded_at``). A page whose record was rewritten since (``--revisit``, ``--rematch``, ``--rekraken``) is rebuilt, a
page whose raw entry is missing, unreadable or belongs to another image / policy state is counted and skipped, and a
second run with nothing new writes nothing, so a loop can call it every half hour with ``--since``.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from src.datasets.consensus import pipeline_decode_policy as pdp
from src.datasets.consensus import two_reader_lines as trl
from src.datasets.consensus.reread import to_page_box

SCHEMA_VERSION = "readings-v1"
BUILDER_VERSION = "1.0.0"
BUILDER = f"build_readings {BUILDER_VERSION}"
PAGE_UNIT = pdp.PAGE_UNIT                       # request unit (and role) of a whole-page read
SOURCE_LEGACY = "legacy"                        # final.source of a page read without the decode policy
MAPPING_SINGLE = "single"                       # one reply, one page reader (a page without a policy trace)
MAPPING_BY_REQUEST = "by_request"               # every reply verified against its traced request
MAPPING_PARTIAL = "partial"                     # the list and the trace disagree: some readers carry no lines
DEFAULT_RAW_DIR = trl.RAW_DIR
DEFAULT_OUT_DIR = trl.OUT_DIR / "readings"
MANIFEST_NAME = "manifest.jsonl"                # default manifest, inside the sidecar directory (sidecars are *.json)
PROGRESS_EVERY = 5000

Json = Dict[str, Any]


# ---------------------------------------------------------------------------
# Boxes and lines
# ---------------------------------------------------------------------------

def int_box(box: Sequence[float]) -> List[int]:
    """A 0-1000 box as ints inside the frame with ordered corners (the rule of ``two_reader_lines._clamp_box``).

    :param box: ``[x1, y1, x2, y2]``.
    :return: The box clamped to 0-1000 and rounded.
    """
    x1, y1, x2, y2 = (int(round(max(0.0, min(1000.0, float(v))))) for v in box)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def reader_lines(lines: Sequence[Mapping[str, Any]], crop: Optional[Sequence[int]] = None, width: int = 0,
                 height: int = 0) -> List[Json]:
    """A reply's ``{text, box}`` lines as sidecar lines with int boxes in the page's 0-1000 frame.

    :param lines: Lines of one reply in reply order (``parse_grounded``, or the stored parse).
    :param crop: Pixel crop box ``(x1, y1, x2, y2)`` the reply was read from; its boxes are then in the crop's 0-1000
        frame and are mapped to the page frame (:func:`~src.datasets.consensus.reread.to_page_box`). None for a
        whole-page reply.
    :param width: Page width in pixels (only used with ``crop``).
    :param height: Page height in pixels (only used with ``crop``).
    :return: ``{index, text, box}`` per line; ``index`` is the position in the reply.
    """
    out = []
    for i, line in enumerate(lines):
        box = line["box"] if crop is None else to_page_box(line["box"], crop, width, height)
        out.append({"index": i, "text": line["text"], "box": int_box(box)})
    return out


def kraken_lines(frags: Sequence[Mapping[str, Any]]) -> List[Json]:
    """Kraken's fragments as sidecar lines.

    :param frags: Raw-cache fragments ``{text, conf, box}`` (0-1000 page frame).
    :return: ``{index, text, box, conf}`` per fragment; ``index`` is the position in ``frags``.
    """
    return [{"index": i, "text": f["text"], "box": int_box(f["box"]), "conf": f.get("conf")}
            for i, f in enumerate(frags)]


def crop_box_1000(crop: Optional[Sequence[int]], width: int, height: int) -> Optional[List[int]]:
    """A pixel crop box in the page's 0-1000 frame.

    :param crop: Pixel box ``(x1, y1, x2, y2)``, or None (the whole page).
    :param width: Page width in pixels.
    :param height: Page height in pixels.
    :return: The int box, or None without a crop.
    """
    return None if not crop else int_box(pdp.crop_box_to_page(crop, width, height))


# ---------------------------------------------------------------------------
# Attributing the stored replies to the traced requests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Attributed:
    """One traced request with the reply the builder attributed to it.

    :param request: The trace's request record (``unit``, ``role``, ``crop``, ``parsed``, ``n_lines``, ...).
    :param index: Position of the reply in the raw entry's ``vlm_raw`` (None: not attributable).
    :param parsed: Whether the reply parsed as a grounded JSON array (None when not attributable).
    :param lines: The reply's sidecar lines, page frame (None when not attributable).
    """

    request: Mapping[str, Any]
    index: Optional[int]
    parsed: Optional[bool]
    lines: Optional[List[Json]]

    @property
    def unit(self) -> str:
        """Unit name of the request.

        :return: ``page``, ``col0``, ``col0.top``, ``top``, ...
        """
        return str(self.request.get("unit"))


def reply_list(vlm_raw: Any) -> List[str]:
    """The raw entry's ``vlm_raw`` as a list of replies.

    :param vlm_raw: A reply string, a list of replies (a page that took several requests) or None.
    :return: The replies in storage order (empty for None).
    """
    if vlm_raw is None:
        return []
    return [vlm_raw] if isinstance(vlm_raw, str) else list(vlm_raw)


def parse_reply(request: Mapping[str, Any], raw: str, cached_page: Optional[Mapping[str, Any]], width: int,
                height: int) -> Tuple[bool, List[Json]]:
    """The lines of one stored reply as the pipeline used them.

    The cached whole-page read of a ``--revisit`` (request ``cached``) keeps the parse stored with it
    (``decode_policy["cached_whole_page"]``), the parse the policy judged and the old record was built from; every
    other reply is parsed with ``parse_grounded`` and a crop reply's boxes are mapped to the page frame.

    :param request: The reply's traced request.
    :param raw: The reply text.
    :param cached_page: The trace's ``cached_whole_page`` (``vlm_lines``, ``parsed``), if any.
    :param width: Page width in pixels.
    :param height: Page height in pixels.
    :return: ``(parsed, lines)`` with sidecar lines.
    """
    if request.get("cached") and cached_page is not None:
        return bool(cached_page.get("parsed")), reader_lines(cached_page.get("vlm_lines") or [])
    parsed, lines = trl.parse_grounded(raw)
    return parsed, reader_lines(lines, request.get("crop"), width, height)


def reply_matches_request(request: Mapping[str, Any], raw: str) -> bool:
    """Whether a stored reply is the one a traced request describes: it has the recorded length.

    The trace records each reply's length (``reply_chars``); no parser change can alter it, so it is what identifies
    the reply at a position of ``vlm_raw``.

    :param request: The traced request.
    :param raw: The stored reply at the request's position.
    :return: False only when the trace records a length and it differs.
    """
    return "reply_chars" not in request or request["reply_chars"] == len(raw)


def parse_drift_note(request: Mapping[str, Any], parsed: bool, n_lines: int) -> Optional[str]:
    """A note when today's parse of a reply differs from the parse the trace recorded.

    The cached whole-page read of a revisit is exempt: its parse is the stored one.

    :param request: The traced request (``parsed``, ``n_lines`` as parsed when the page was read).
    :param parsed: Whether the reply parses as a grounded JSON array now.
    :param n_lines: Lines parsed from it now.
    :return: The note, or None when the parse agrees with the trace.
    """
    if request.get("cached") or (request.get("parsed", parsed) == parsed and request.get("n_lines", n_lines) == n_lines):
        return None
    return (f"reply of {request.get('unit')}: parsed {n_lines} lines (parsed={parsed}) now, the trace recorded "
            f"{request.get('n_lines')} (parsed={request.get('parsed')}); the parser changed since the page was read")


@dataclass(frozen=True)
class Attribution:
    """The stored replies of a policy page attached to its traced requests.

    :param items: One :class:`Attributed` per traced request, in request order.
    :param exact: Every reply is verified against its request (list and trace agree).
    :param notes: Why the mapping is not exact, and any parse that differs from the trace.
    """

    items: List[Attributed]
    exact: bool
    notes: List[str]


def attribute_replies(entry: Mapping[str, Any], trace: Mapping[str, Any], width: int, height: int) -> Attribution:
    """Attach each stored reply of a policy page to its traced request.

    ``vlm_raw`` is parallel to ``trace["requests"]``. The mapping is taken as exact only when the two have the same
    length, the unit names are distinct and every reply has the length its request recorded
    (:func:`reply_matches_request`). Otherwise only a verified leading whole-page reply is attributed and a note says
    why: the rest is left unattributed rather than guessed. A parse that differs from the one the trace recorded
    (a parser change since the page was read) does not unattribute a reply; it is noted.

    :param entry: Raw-cache entry of the page.
    :param trace: Its ``decode_policy`` trace.
    :param width: Page width in pixels.
    :param height: Page height in pixels.
    :return: The attribution.
    """
    requests = list(trace.get("requests") or [])
    replies = reply_list(entry.get("vlm_raw"))
    cached_page = trace.get("cached_whole_page")
    reads: List[Optional[Tuple[bool, List[Json], bool]]] = []
    for i, request in enumerate(requests):
        if i >= len(replies):
            reads.append(None)
            continue
        parsed, lines = parse_reply(request, replies[i], cached_page, width, height)
        reads.append((parsed, lines, reply_matches_request(request, replies[i])))
    units = [r.get("unit") for r in requests]
    exact = (bool(requests) and len(requests) == len(replies) and len(set(units)) == len(units)
             and all(r is not None and r[2] for r in reads))
    items: List[Attributed] = []
    notes: List[str] = []
    for i, request in enumerate(requests):
        read = reads[i]
        leading_page = i == 0 and request.get("unit") == PAGE_UNIT and read is not None and read[2]
        if read is not None and (exact or leading_page):
            items.append(Attributed(request, i, read[0], read[1]))
            drift = parse_drift_note(request, read[0], len(read[1]))
            if drift:
                notes.append(drift)
        else:
            items.append(Attributed(request, None, None, None))
    if not exact:
        notes.insert(0, f"reply mapping ambiguous: vlm_raw holds {len(replies)} replies for {len(requests)} traced "
                        f"requests, or a reply does not have the length its request recorded; only a verified "
                        f"leading whole-page reply is attributed, the other readers have reply_index null and "
                        f"lines null")
    return Attribution(items, exact, notes)


# ---------------------------------------------------------------------------
# Readers and the policy section
# ---------------------------------------------------------------------------

def health_labels(trace: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """Health label of each traced request, by unit name.

    :param trace: The page's ``decode_policy`` trace.
    :return: ``page`` -> the whole-page read's verdict (``plan.whole_page``); a plan unit -> its crop attempt's
        verdict; a half of a failed column -> its entry in the unit's ``halves`` attempt.
    """
    labels: Dict[str, Optional[str]] = {}
    whole = (trace.get("plan") or {}).get("whole_page")
    if whole:
        labels[PAGE_UNIT] = whole.get("label")
    for unit in trace.get("units") or []:
        for attempt in unit.get("attempts") or []:
            if attempt.get("source") == "crop":
                labels[unit["name"]] = (attempt.get("health") or {}).get("label")
            elif attempt.get("source") == "halves":
                for half in attempt.get("halves") or []:
                    labels[half["unit"]] = half.get("label")
    return labels


def vlm_reader(model: str, unit: str, role: str, crop: Optional[Sequence[int]], reply_index: Optional[int],
               parsed: Optional[bool], health_label: Optional[str], lines: Optional[List[Json]], width: int,
               height: int, note: Optional[str] = None) -> Json:
    """One VLM reader of the sidecar (the page read, or one crop read of a unit).

    :param model: LM Studio key of the VLM.
    :param unit: ``page`` or the unit name (``col0``, ``col0.top``, ``top``, ...).
    :param role: ``page``, ``column`` or ``half``.
    :param crop: Pixel crop box of a unit read (None for the page).
    :param reply_index: Position of the reply in the raw entry's ``vlm_raw`` (None: no reply text / not attributable).
    :param parsed: Whether the reply parsed.
    :param health_label: The policy's verdict on the read (None when the page was read without the policy).
    :param lines: Sidecar lines, or None when the reply could not be attributed.
    :param width: Page width in pixels.
    :param height: Page height in pixels.
    :param note: Why ``reply_index`` / ``lines`` are null, if they are.
    :return: The reader record.
    """
    is_page = unit == PAGE_UNIT
    reader: Json = {
        "id": f"vlm:{model}:page" if is_page else f"vlm:{model}:{unit}:crop", "kind": "vlm",
        "scope": "page" if is_page else "unit", "unit": None if is_page else unit, "role": role,
        "reply_index": reply_index, "parsed": parsed, "health_label": health_label,
        "crop_box_px": list(crop) if crop else None, "crop_box_1000": crop_box_1000(crop, width, height),
        "lines": lines}
    if note:
        reader["note"] = note
    return reader


def policy_vlm_readers(model: str, attributions: Sequence[Attributed], labels: Mapping[str, Optional[str]],
                       width: int, height: int) -> List[Json]:
    """The VLM readers of a policy page: the whole-page read first (when there was one), then the unit reads.

    :param model: LM Studio key of the VLM.
    :param attributions: Output of :func:`attribute_replies`.
    :param labels: Output of :func:`health_labels`.
    :param width: Page width in pixels.
    :param height: Page height in pixels.
    :return: One reader per traced request, in that order.
    """
    ordered = sorted(attributions, key=lambda a: a.unit != PAGE_UNIT)        # stable: units keep request order
    return [vlm_reader(model, a.unit, str(a.request.get("role")), a.request.get("crop"), a.index, a.parsed,
                       labels.get(a.unit), a.lines, width, height,
                       None if a.index is not None else "reply could not be attributed, see provenance.notes")
            for a in ordered]


def legacy_page_reader(model: str, entry: Mapping[str, Any], width: int, height: int) -> Json:
    """The page read of a page without a policy trace: the parse stored with the raw entry.

    :param model: LM Studio key of the VLM.
    :param entry: Raw-cache entry (``vlm_raw`` a string, or None for an entry rebuilt from a record).
    :param width: Page width in pixels.
    :param height: Page height in pixels.
    :return: The page reader; ``reply_index`` is 0 with a stored reply and null (with a note) without one.
    """
    has_reply = isinstance(entry.get("vlm_raw"), str)
    return vlm_reader(model, PAGE_UNIT, PAGE_UNIT, None, 0 if has_reply else None, bool(entry.get("parsed")), None,
                      reader_lines(entry.get("vlm_lines") or []), width, height,
                      None if has_reply else "no raw reply cached; lines are the parse stored with the entry")


def kraken_reader(htr_model: str, cache_key: str, frags: Sequence[Mapping[str, Any]]) -> Json:
    """The Kraken reader of the sidecar.

    :param htr_model: Name of the recognition model.
    :param cache_key: HTR cache key of the fragments (the model name when no service key was set).
    :param frags: The fragments of that key.
    :return: The reader record.
    """
    return {"id": f"kraken:{cache_key}", "kind": "htr", "scope": "page", "unit": None, "htr_model": htr_model,
            "cache_key": cache_key, "lines": kraken_lines(frags)}


def attempt_section(attempt: Mapping[str, Any], unit_name: str, by_unit: Mapping[str, Attributed]) -> Json:
    """One candidate reading of a unit, with the replies it came from.

    :param attempt: The trace's attempt (``source`` crop / halves / share, ``letters``, ``health`` ...).
    :param unit_name: The unit it belongs to.
    :param by_unit: Attributions by request unit name.
    :return: ``source``, ``label``, ``flags``, ``letters``, ``n_lines``, ``reply_index`` and, as the source needs,
        ``reply_indices`` (a halves attempt joins two replies) and a ``note``.
    """
    source = attempt.get("source")
    health = attempt.get("health") or {}
    out: Json = {"source": source, "label": health.get("label"), "flags": list(health.get("flags") or []),
                 "letters": attempt.get("letters"), "n_lines": attempt.get("n_lines"), "reply_index": None}
    if source == "crop":
        reply = by_unit.get(unit_name)
        out["reply_index"] = reply.index if reply else None
    elif source == "halves":
        names = [h.get("unit") for h in attempt.get("halves") or []]
        out["reply_indices"] = [by_unit[n].index if n in by_unit else None for n in names]
        out["note"] = "joined from the replies in reply_indices (top half, bottom half)"
    elif source == "share":
        page = by_unit.get(PAGE_UNIT)
        out["reply_index"] = page.index if page else None
        out["note"] = "lines of the whole-page read (reply_index) whose box centre lies in the unit's crop"
    else:
        out["note"] = f"unknown source {source!r}: reply not attributed"
    return out


def unit_section(unit: Mapping[str, Any], by_unit: Mapping[str, Attributed], width: int, height: int) -> Json:
    """One plan unit of the policy section.

    :param unit: The trace's unit record (``name``, ``role``, ``crop``, ``attempts``, ``used``, ``resolved`` ...).
    :param by_unit: Attributions by request unit name.
    :param width: Page width in pixels.
    :param height: Page height in pixels.
    :return: ``name``, ``role``, ``crop_box_px``, ``crop_box_1000``, ``resolved_by`` (``crop`` / ``halves`` /
        ``share`` / ``unresolved``), ``resolved``, ``health_label`` and ``attempts``. ``health_label`` is the verdict
        on the reading that resolved the unit, or on the last reading tried when none did.
    """
    attempts = [attempt_section(a, unit["name"], by_unit) for a in unit.get("attempts") or []]
    used = unit.get("used")
    chosen = next((a for a in attempts if a["source"] == used), attempts[-1] if attempts else None)
    crop = unit.get("crop")
    return {"name": unit["name"], "role": unit.get("role"), "crop_box_px": list(crop) if crop else None,
            "crop_box_1000": crop_box_1000(crop, width, height), "resolved_by": used,
            "resolved": unit.get("resolved"), "health_label": chosen["label"] if chosen else None,
            "attempts": attempts}


def policy_section(trace: Mapping[str, Any], attributions: Sequence[Attributed], width: int, height: int) -> Json:
    """The sidecar's ``policy`` block from a decode-policy trace.

    :param trace: The raw entry's ``decode_policy`` trace.
    :param attributions: Output of :func:`attribute_replies`.
    :param width: Page width in pixels.
    :param height: Page height in pixels.
    :return: ``version``, ``action``, ``reason``, ``final_source``, ``calls``, ``fallback`` (the page fallback, null
        when the assembly stood), ``mode`` (``live`` / ``revisit``), ``decided_at`` and ``units``.
    """
    by_unit = {a.unit: a for a in attributions}
    return {"version": trace.get("version"), "action": trace.get("action"), "reason": trace.get("reason"),
            "final_source": trace.get("final_source"), "calls": trace.get("calls"),
            "fallback": trace.get("page_fallback"), "mode": trace.get("mode"), "decided_at": trace.get("decided_at"),
            "units": [unit_section(u, by_unit, width, height) for u in trace.get("units") or []]}


# ---------------------------------------------------------------------------
# One page
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """The current UTC time as ISO-8601 to the second.

    :return: E.g. ``2026-10-13T22:30:00+00:00``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def readings_for_page(record: Mapping[str, Any], raw_entry: Mapping[str, Any], raw_relpath: str,
                      built_at: Optional[str] = None) -> Json:
    """The ``readings-v1`` sidecar of one page.

    Pure: no I/O, no clock unless ``built_at`` is omitted. A page whose raw entry has a ``decode_policy`` trace gets a
    ``policy`` block, its replies attributed by request (:func:`attribute_replies`) and ``final.source`` taken from
    the trace; any other page is ``legacy``: the page read, Kraken, ``policy`` null.

    :param record: The page's record from ``ai_reads_<model>.jsonl``.
    :param raw_entry: Its raw-cache entry (the caller has checked :func:`entry_mismatch`).
    :param raw_relpath: Path of the raw-cache file relative to the ``ai_reads`` directory, stored in ``provenance``.
    :param built_at: Build time (ISO-8601); None = now.
    :return: The sidecar as a JSON-ready dict.
    """
    ai = record["ai_read"]
    model = ai["vlm_model"]
    width = int(raw_entry.get("width") or record["image_width"])
    height = int(raw_entry.get("height") or record["image_height"])
    trace = raw_entry.get("decode_policy") or None
    notes: List[str] = []
    if trace is None:
        readers = [legacy_page_reader(model, raw_entry, width, height)]
        policy, final_source, mapping = None, SOURCE_LEGACY, MAPPING_SINGLE
    else:
        attribution = attribute_replies(raw_entry, trace, width, height)
        notes = list(attribution.notes)
        readers = policy_vlm_readers(model, attribution.items, health_labels(trace), width, height)
        policy = policy_section(trace, attribution.items, width, height)
        final_source = trace.get("final_source") or (ai.get("decode_policy") or {}).get("final_source")
        mapping = MAPPING_BY_REQUEST if attribution.exact else MAPPING_PARTIAL
    htr_model = ai.get("htr_model") or trl.HTR_MODEL_NAME
    cache_key = ai.get("htr_cache_key") or htr_model
    frags = trl.cached_frags(raw_entry, cache_key)
    if frags is None:
        notes.append(f"no Kraken fragments cached for key {cache_key}: no htr reader")
    else:
        readers.append(kraken_reader(htr_model, cache_key, frags))
    return {
        "schema_version": SCHEMA_VERSION, "doc_id": record["doc_id"], "image_index": record["image_index"],
        "image_url": record["image_url"], "image_sha256": record["image_sha256"],
        "image_width": record["image_width"], "image_height": record["image_height"],
        "source_index": record["source_index"], "decoded_at": ai["decoded_at"], "rule_version": ai["rule_version"],
        "vlm_model": model, "vlm_revision": ai.get("vlm_revision"), "policy": policy, "readers": readers,
        "final": {"source": final_source, "lines": ai["lines"]},
        "provenance": {"raw_cache": raw_relpath, "built_at": built_at or utc_now_iso(), "builder": BUILDER,
                       "reply_mapping": mapping, "notes": notes}}


def entry_mismatch(record: Mapping[str, Any], entry: Mapping[str, Any]) -> Optional[str]:
    """Why a raw-cache entry cannot be this record's page, or None when it can.

    The raw cache is keyed by ``(doc_id, image_index, model)``, so an entry can belong to another image (an image
    list that changed between index versions) or be ahead of / behind its record (a ``--revisit`` or ``--rematch``
    in progress); a sidecar built from such a pair would show readers that did not produce the final lines.

    :param record: The page's record.
    :param entry: The raw entry found for it.
    :return: ``image_url``, ``vlm_model``, ``sha256``, ``size``, ``policy_stamp`` (the entry's trace and the record's
        ``decode_policy`` stamp disagree) or ``htr_key`` (no fragments cached for the record's HTR key); None when
        the entry fits.
    """
    ai = record["ai_read"]
    if entry.get("image_url") != record["image_url"]:
        return "image_url"
    if entry.get("vlm_model") != ai["vlm_model"]:
        return "vlm_model"
    if entry.get("sha256") and record.get("image_sha256") and entry["sha256"] != record["image_sha256"]:
        return "sha256"
    if (entry.get("width"), entry.get("height")) != (record["image_width"], record["image_height"]):
        return "size"
    trace, stamp = entry.get("decode_policy"), ai.get("decode_policy")
    if bool(trace) != bool(stamp):
        return "policy_stamp"
    if trace and any(trace.get(k) != stamp.get(k) for k in ("version", "action", "final_source", "calls")):
        return "policy_stamp"
    if trl.cached_frags(entry, ai.get("htr_cache_key") or ai.get("htr_model") or trl.HTR_MODEL_NAME) is None:
        return "htr_key"
    return None


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def parse_time(text: str) -> datetime:
    """An ISO-8601 time as an aware datetime.

    :param text: E.g. ``2026-10-13T20:00:00Z``, ``2026-10-13T20:00:00+00:00`` or a bare date (taken as UTC).
    :return: The time in its own zone, UTC when none was given.
    """
    value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def load_json(path: Path) -> Optional[Json]:
    """A JSON file, or None when it is absent or not valid JSON.

    The one place a read error is swallowed: a file another process is replacing, or cut short by a crash, is retried
    on the next run instead of stopping the whole walk.

    :param path: File to read.
    :return: The parsed object, or None (also when the file holds JSON that is not an object).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json_atomic(path: Path, obj: Any) -> None:
    """Replace ``path`` with ``obj`` as UTF-8 JSON via a per-process temp file and a rename (readers never see half a file).

    :param path: Destination.
    :param obj: JSON-serialisable value.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def append_manifest(path: Path, row: Mapping[str, Any]) -> None:
    """Append one line to the manifest JSONL.

    :param path: Manifest file (created on first use).
    :param row: The line's fields.
    """
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_records(path: Path, counts: Counter) -> Iterator[Json]:
    """The records of a JSONL file that is still being appended to.

    :param path: Records file.
    :param counts: Gains ``partial_tail`` when the last line has no newline yet (a record being written is not one yet).
    :return: The records in file order.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n"):
                counts["partial_tail"] += 1
            elif line.strip():
                yield json.loads(line)


def sidecar_status(path: Path, record: Mapping[str, Any]) -> str:
    """Whether the sidecar at ``path`` is current for ``record``.

    Current = this page's sidecar (same ``doc_id``, ``image_index`` and ``vlm_model``: on a case-insensitive
    filesystem two doc ids that differ only by case, like ``..._J24B`` and ``..._J24b``, share a file name), same
    schema and builder version, built from this record or a newer one (``decoded_at``). The "newer" half keeps a page
    recorded twice (the later record wins) from being rebuilt back and forth on every run.

    :param path: The sidecar file.
    :param record: The page's record.
    :return: ``missing`` (no usable file), ``stale`` or ``current``.
    """
    existing = load_json(path)
    if existing is None:
        return "missing"
    built_by = (existing.get("provenance") or {}).get("builder")
    page = (record["doc_id"], record["image_index"], record["ai_read"]["vlm_model"])
    if (existing.get("doc_id"), existing.get("image_index"), existing.get("vlm_model")) != page:
        return "stale"
    if existing.get("schema_version") != SCHEMA_VERSION or built_by != BUILDER:
        return "stale"
    built_from = parse_time(existing.get("decoded_at") or "1970-01-01")
    return "current" if built_from >= parse_time(record["ai_read"]["decoded_at"]) else "stale"


def build_readings(records: Sequence[Path], raw_dir: Path, out_dir: Path, manifest: Optional[Path] = None,
                   since: Optional[datetime] = None, limit: int = 0, force: bool = False) -> Counter:
    """Write the sidecar of every record that needs one.

    Records are walked in file order. A record is skipped when ``decoded_at`` is before ``since``, or when its
    sidecar is current (unless ``force``). Otherwise its raw entry is found with
    :func:`~src.datasets.consensus.two_reader_lines.raw_cache_path` (so an older entry without the model suffix is
    found as the pipeline finds it), checked (:func:`entry_mismatch`), reshaped and written atomically to
    ``<out_dir>/<raw cache stem>.json``, and a line is appended to ``manifest``.

    :param records: Records files (``ai_reads_<model>.jsonl``).
    :param raw_dir: Raw-cache directory.
    :param out_dir: Sidecar directory (created).
    :param manifest: Manifest JSONL to append to (None: no manifest).
    :param since: Only records decoded at or after this time.
    :param limit: Stop after this many sidecars written (0 = no limit).
    :param force: Rewrite sidecars that are current.
    :return: Counts: ``records`` seen, ``before_since``, ``up_to_date``, ``built`` (``built_new`` + ``rebuilt``),
        ``built_policy`` / ``built_legacy``, ``policy_<action>``, ``ambiguous`` (reply mapping ``partial``),
        ``raw_missing``, ``raw_unreadable``, ``raw_mismatch`` (+ ``raw_mismatch:<reason>``), ``readers_vlm`` /
        ``readers_htr`` of the sidecars written, ``bytes_written`` and ``partial_tail``.
    """
    counts: Counter = Counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for records_path in records:
        for record in iter_records(records_path, counts):
            counts["records"] += 1
            if counts["records"] % PROGRESS_EVERY == 0:
                print(f"  {counts['records']} records walked, {counts['built']} sidecars written "
                      f"({time.time() - started:.0f} s)", flush=True)
            ai = record["ai_read"]
            if since is not None and parse_time(ai["decoded_at"]) < since:
                counts["before_since"] += 1
                continue
            raw_path = trl.raw_cache_path(record, ai["vlm_model"], raw_dir=raw_dir)
            sidecar_path = out_dir / f"{raw_path.stem}.json"
            status = sidecar_status(sidecar_path, record)
            if status == "current" and not force:
                counts["up_to_date"] += 1
                continue
            if not raw_path.exists():
                counts["raw_missing"] += 1
                continue
            entry = load_json(raw_path)
            if entry is None:
                counts["raw_unreadable"] += 1
                continue
            reason = entry_mismatch(record, entry)
            if reason:
                counts["raw_mismatch"] += 1
                counts[f"raw_mismatch:{reason}"] += 1
                continue
            built_at = utc_now_iso()
            relpath = Path(os.path.relpath(raw_path, out_dir.parent)).as_posix()
            doc = readings_for_page(record, entry, relpath, built_at)
            write_json_atomic(sidecar_path, doc)
            count_built(counts, doc, status, sidecar_path)
            if manifest is not None:
                append_manifest(manifest, {
                    "stem": raw_path.stem, "doc_id": doc["doc_id"], "image_index": doc["image_index"],
                    "built_at": built_at, "readers": len(doc["readers"]),
                    "policy_action": (doc["policy"] or {}).get("action")})
            if limit and counts["built"] >= limit:
                return counts
    return counts


def count_built(counts: Counter, doc: Mapping[str, Any], status: str, path: Path) -> None:
    """Add one written sidecar to the run's counts.

    :param counts: The run's counts (updated in place).
    :param doc: The sidecar written.
    :param status: Its :func:`sidecar_status` before the write.
    :param path: Where it was written (for its size).
    """
    counts["built"] += 1
    counts["built_new" if status == "missing" else "rebuilt"] += 1
    counts["built_policy" if doc["policy"] else "built_legacy"] += 1
    if doc["policy"]:
        counts[f"policy_{doc['policy']['action']}"] += 1
    if doc["provenance"]["reply_mapping"] == MAPPING_PARTIAL:
        counts["ambiguous"] += 1
    for reader in doc["readers"]:
        counts[f"readers_{reader['kind']}"] += 1
    counts["bytes_written"] += path.stat().st_size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The command-line parser.

    :return: Parser for ``python -m src.datasets.consensus.build_readings``.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="+", required=True, type=Path, help="ai_reads_<model>.jsonl file(s)")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="raw-cache directory (default ai_reads/raw)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="sidecar directory (default ai_reads/readings)")
    ap.add_argument("--since", type=parse_time, default=None, metavar="ISO",
                    help="only records whose ai_read.decoded_at is at or after this time (e.g. 2026-10-13T20:00:00Z)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many sidecars written (0 = no limit)")
    ap.add_argument("--force", action="store_true", help="rewrite sidecars that are current")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest JSONL appended with one line per sidecar written (default <out-dir>/manifest.jsonl)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the builder.

    :param argv: Command-line arguments (None = ``sys.argv``).
    :return: Exit status (0).
    """
    ap = build_parser()
    args = ap.parse_args(argv)
    missing = [str(p) for p in args.records if not p.exists()]
    if missing:
        ap.error(f"--records not found: {', '.join(missing)}")
    manifest = args.manifest or args.out_dir / MANIFEST_NAME
    started = time.time()
    counts = build_readings(args.records, args.raw_dir, args.out_dir, manifest, since=args.since, limit=args.limit,
                            force=args.force)
    summary = {**dict(sorted(counts.items())), "elapsed_s": round(time.time() - started, 1)}
    print(f"build_readings {BUILDER_VERSION}: {counts['built']} sidecars written ({counts['built_new']} new, "
          f"{counts['rebuilt']} rebuilt), {counts['up_to_date']} up to date, {counts['raw_missing']} without a raw "
          f"entry -> {args.out_dir}", flush=True)
    print("summary: " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
