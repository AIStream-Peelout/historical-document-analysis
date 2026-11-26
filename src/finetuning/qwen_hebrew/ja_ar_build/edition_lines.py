"""Preserve printed edition baselines and propose their locations on manuscript photographs."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import fitz
from rapidfuzz import fuzz, process

from src.datasets.indexing.legacy_store import pdf_transcriptions as pdf
from src.finetuning.qwen_hebrew.ja_ar_build import textnorm
from src.finetuning.qwen_hebrew.ja_ar_build.align import ReadStream

DOUBLE_ANGLE = re.compile(r">>(.+?)<<|<<(.+?)>>")
DOTTED = re.compile(r"\.(?:\s*\.)+|…")
EDGE_LETTERS = re.compile(r"[א-תء-ي][א-תء-ي '\u05f3\u05f4]*")
# A full-line match is placement evidence. A weak read never edits the edition text.
MIN_ANCHOR = 12
MIN_SCORE = 85
MIN_LEAD = 9


def digest(text: str) -> str:
    """Hash a UTF-8 text witness.

    :param text: Text to bind to the record.
    :return: Hexadecimal SHA-256.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def edge_reading(body: str) -> str | None:
    """Keep an alternative for an erased row with a single letter run between two dotted losses.

    :param body: Diplomatic baseline, excluding an explicit printed number.
    :return: Source-only alternative for visual inspection, if the layout fits.
    """
    matches = list(EDGE_LETTERS.finditer(body))
    if len(matches) != 1 or len(textnorm.skeleton(matches[0][0])) < 2:
        return None
    ink = matches[0]
    edges = (body[:ink.start()], body[ink.end():])
    if all(DOTTED.search(edge) and not edge.strip(" \t.[]…&*") for edge in edges):
        return "[...]" + ink[0].strip() + "[...]"
    return None


def propose(body: str, staged: dict[str, set[str]]) -> tuple[str | None, list[str]]:
    """Propose a reading from the edition and retain uncertainty about its notation.

    :param body: Printed baseline after explicit editorial annotation and numbering.
    :param staged: Previously cleaned lines keyed by letter skeleton for this edition.
    :return: Provisional source text and reasons for checking it against the manuscript.
    """
    skel = textnorm.skeleton(body)
    warnings = []
    if any(mark in body for mark in "[](){}?") or DOTTED.search(body):
        warnings.append("damage_or_editorial_notation")
    if pdf.EDITOR_NOTE_RE.search(body) or re.search(r"[A-Za-z]", body):
        warnings.append("possible_apparatus")
    if re.search(r"[א-ת][0-9A-Za-z]|[0-9A-Za-z][א-ת]", body):
        warnings.append("mixed_glyphs")
    if not skel:
        return None, warnings + ["check_gap_or_mark"]
    # The old line may share letters but differ in punctuation, bracket scope or word spacing.
    # Use its spelling only when the *cleaned* baseline also agrees byte for byte.
    result = pdf.clean_folio([body], None)
    lines = result["cleaned_lines"]
    if len(lines) != 1:
        return None, warnings + ["source_letters_without_single_reading"]
    line = lines[0]
    if line not in staged.get(textnorm.skeleton(line), set()):
        warnings.append("check_standalone_cleanup")
    if textnorm.skeleton(line) != skel:
        warnings.append("letters_changed_in_cleanup")
    return line, warnings


def printed_lines(path: Path, edition: dict) -> tuple[str, list[dict]]:
    """Read each physical PDF baseline once, before folio cleanup and photo assignment.

    :param path: Original edition PDF.
    :param edition: Staged edition record with source sides and cleaned lines.
    :return: Edition digest and ordered baseline records, including layout notes and gaps.
    """
    data = path.read_bytes()
    pdf_sha = hashlib.sha256(data).hexdigest()
    suffix = re.search(r"__([0-9a-f]{12,64})\.pdf$", path.name)
    if suffix and not pdf_sha.startswith(suffix[1]):
        raise ValueError(f"Edition digest differs from filename: {path.name}")
    staged = defaultdict(set)
    for side in edition["sides"]:
        for line in side["lines"]:
            staged[textnorm.skeleton(line)].add(line)
    with fitz.open(stream=data, filetype="pdf") as doc:
        visual = pdf.document_lines(doc, layout_notes=True, glyphs=pdf.pdf_glyphs.GlyphDecoder(doc),
                                    glyph_finals=True, keep_geometry=True)
    header = pdf.parse_header([v.text for v in visual if v.page_index == 0])
    header_size = int(header["header_lines"]) if header else 0
    page, ordinal, in_notes = -1, 0, False
    folio = None
    records = []
    for index, v in enumerate(visual):
        if v.page_index != page:
            page, ordinal, in_notes = v.page_index, 0, False
        ordinal += 1
        raw = v.text
        marker = pdf.match_folio_marker(raw)
        if marker:
            in_notes = False
            folio = marker.label if marker.kind != "edition_ref" else None
        if pdf.NOTES_START_RE.match(raw):
            in_notes = True
        body = raw.strip()
        # Keep the title and footnotes addressable. An explicit binding can recover quoted manuscript text.
        kind = v.layout_note
        if not kind and page == 0 and index < header_size:
            kind = "edition_header"
        if not kind and in_notes:
            kind = "edition_notes"
        if not kind and marker:
            if marker.kind in pdf.TEXT_AFTER_HEADING_KINDS and marker.rest.strip():
                body = marker.rest.strip()
            else:
                kind = "folio_heading"
        inline = []
        for m in DOUBLE_ANGLE.finditer(body):
            content = (m[1] or m[2]).strip()
            side_heading = re.fullmatch(r'[אב\sע"״]{1,8}', content)
            if (m.group() == body and side_heading) or re.match(r'קריאתי\s*:', content):
                inline.append(m.group())
        for annotation in inline:
            body = body.replace(annotation, " ", 1)
        if not body.strip() and not kind:
            kind = "edition_annotation"
        number = pdf.LINE_NUMBER_RE.match(body)
        if number:
            body = body[number.end():].lstrip()
        if body.strip().isdigit() and not kind:
            kind = "numeric_label_or_mark"
        source, flags = (None, []) if kind else propose(body, staged)
        if inline and not kind:
            flags.append("explicit_inline_apparatus_removed")
        if ("<<" in body or ">>" in body) and not kind:
            flags.append("unresolved_inline_annotation")
        runs = [{"text": span[1], "x": bounds} for span, bounds in zip(v.spans, v.span_extents, strict=True)]
        occupied = sorted(run["x"] for run in runs if textnorm.skeleton(run["text"]))
        if any(b[0]-a[1] > max(30, 3*v.size) for a, b in zip(occupied, occupied[1:])):
            flags.append("check_column_order")
        alternative = edge_reading(body) if not kind and not source else None
        if alternative:
            flags.append("verify_surviving_ink_between_gaps")
        records.append({"line_id": f"{edition['row_id']}|{pdf_sha}|p{page+1}.{ordinal}",
                        "pdf_page": page+1, "ordinal": ordinal, "baseline_pt": v.y, "x_extent": [v.x0, v.x1],
                        "runs": runs, "edition_folio": folio,
                        "raw": raw, "raw_sha256": digest(raw), "printed_number": int(number[1]) if number else None,
                        "kind": kind or "source_or_gap", "inline_apparatus": inline, "text": source,
                        "flags": flags, "alternative_edge_reading": alternative, "placement": None})
    return pdf_sha, records


def apply_bindings(lines: list[dict], bindings: list[dict], pdf_sha: str, photos: dict[str, dict]) -> None:
    """Apply source-inspected placements after checking edition, photo and exact printed locations.

    :param lines: Parsed edition baselines.
    :param bindings: Explicit source inspections for this edition.
    :param pdf_sha: Actual edition digest.
    :param photos: The manuscript's photos keyed by digest.
    """
    by_location = {(l["pdf_page"], l["ordinal"]): l for l in lines}
    for binding in bindings:
        if binding["pdf_sha256"] != pdf_sha or binding["image_sha256"] not in photos:
            raise ValueError("Source placement refers to a different edition or photograph")
        p = photos[binding["image_sha256"]]
        crop = binding.get("crop")
        rotation = binding["rotation"]
        if rotation not in (0, 90, 180, 270) or (crop is not None and not
                (len(crop) == 4 and all(type(x) is int for x in crop)
                 and 0 <= crop[0] < crop[2] <= p["width"] and 0 <= crop[1] < crop[3] <= p["height"])):
            raise ValueError("Invalid inspected view")
        for item in binding["lines"]:
            line = by_location[item["pdf_page"], item["ordinal"]]
            if line["raw_sha256"] != item["raw_sha256"] or line["placement"] is not None:
                raise ValueError(f"Changed or multiply assigned printed baseline: {line['line_id']}")
            if "text" in item:
                supplied = textnorm.skeleton(item["text"] or "")
                if Counter(supplied) - Counter(textnorm.skeleton(line["raw"])):
                    raise ValueError(f"Reading supplies letters absent from the source: {line['line_id']}")
                line["text"] = item["text"]
            line["flags"] = sorted(set(line["flags"] + binding.get("flags", []) + item.get("flags", [])))
            line["placement"] = {"method": "source_inspection", "image_sha256": p["sha256"], "rotation": rotation,
                                 "crop": crop, "observation": binding["observation"]}


def anchor_lines(streams: list[ReadStream]) -> tuple[list[str], list[dict]]:
    """Index positive photo evidence with its original stream and read-line location.

    :param streams: Stored usable reads of this manuscript's photos.
    :return: Searchable line skeletons and matching view records.
    """
    values, locations = [], []
    for st in streams:
        start = 0
        for i, length in enumerate(st.line_letters):
            if length >= MIN_ANCHOR-2:
                values.append(st.stream[start:start+length])
                box = st.bboxes[i] if st.bboxes else None
                locations.append({"image_sha256": st.sha256, "rotation": st.rotation, "crop": st.crop,
                                  "stream_key": st.key, "read_index": i,
                                  "stored_image_bbox": box if box and all(map(math.isfinite, box)) else None})
            start += length
    return values, locations


def match_photo(line: dict, values: list[str], locations: list[dict]) -> dict | None:
    """Look for a unique full-line match, with the runner-up from a competing view.

    :param line: Printed source baseline and its provisional reading.
    :param values: Cached read-line skeletons.
    :param locations: Locations parallel to values.
    :return: A tentative placement, or None when evidence is ambiguous.
    """
    query = textnorm.skeleton(line["text"] or line["raw"])
    if len(query) < MIN_ANCHOR or not values:
        return None
    hits = process.extract(query, values, scorer=fuzz.ratio, limit=None, score_cutoff=MIN_SCORE-MIN_LEAD)
    if not hits or hits[0][1] < MIN_SCORE:
        return None
    _, best, idx = hits[0]
    winner = locations[idx]
    for _, score, other in hits[1:]:
        if best-score >= MIN_LEAD:
            break
        competitor = locations[other]
        if any(winner[k] != competitor[k] for k in ("image_sha256", "rotation", "crop")):
            return None
        if winner["stream_key"] == competitor["stream_key"] and winner["read_index"] != competitor["read_index"]:
            return None
    return dict(winner, method="cached_line_match", score=round(best, 2))


def place_lines(lines: list[dict], streams: list[ReadStream]) -> None:
    """Attach direct evidence and propose internal spans bracketed by ordered anchors in the same read.

    :param lines: Ordered baselines of one edition with any explicit inspections applied.
    :param streams: Cached photo reads. Unplaced source lines always remain in the ledger.
    """
    values, locations = anchor_lines(streams)
    previous = None
    for i, line in enumerate(lines):
        if (line["placement"] or line["kind"] != "source_or_gap"
                or {"possible_apparatus", "check_column_order", "unresolved_inline_annotation"}.intersection(line["flags"])):
            previous = None
            continue
        line["placement"] = match_photo(line, values, locations)
        where = line["placement"]
        if not where:
            continue
        if previous is not None:
            first = lines[previous]
            before = first["placement"]
            steps = where["read_index"] - before["read_index"]
            left, right = before.get("stored_image_bbox"), where.get("stored_image_bbox")
            common_column = not (left and right) or min(left[2], right[2]) > max(left[0], right[0])
            if (first["pdf_page"] == line["pdf_page"] and where["stream_key"] == before["stream_key"]
                    and first["edition_folio"] == line["edition_folio"] and common_column
                    and 1 < i-previous <= 6 and 1 < steps <= 8):
                for mid in lines[previous+1:i]:
                    mid["placement"] = {k: where[k] for k in ("image_sha256", "rotation", "crop")}
                    mid["placement"].update(method="between_ordered_anchors",
                                            anchor_line_ids=[first["line_id"], line["line_id"]])
                    mid["flags"].append("verify_intermediate_placement")
        previous = i


def place_labelled_folios(lines: list[dict], photos: dict[str, dict], edition: dict) -> None:
    """Propose a unique matching catalogue face only for an explicit manuscript folio label.

    :param lines: Edition baselines, including stronger placements and headings.
    :param photos: Manuscript photographs keyed by digest.
    :param edition: Source sides with ingest flags.
    """
    ambiguous = {"heading_other_shelfmark", "duplicate_folio_label", "folio_order_break", "shelfmark_header_file_mismatch"}
    if any(ambiguous.intersection(s["pointer"].get("flags", [])) for s in edition["sides"]):
        return
    by_label = defaultdict(list)
    for sha, p in photos.items():
        by_label[str(p.get("label") or "").strip().lower()].append(sha)
    sections = defaultdict(list)
    for line in lines:
        if line["edition_folio"] and line["kind"] == "source_or_gap":
            sections[line["edition_folio"]].append(line)
    for label, section in sections.items():
        if len(by_label[label]) != 1:
            continue
        sha, = by_label[label]
        anchored = [l["placement"] for l in section if l["placement"]]
        for line in section:
            line["catalogue_face_hint"] = sha
        direct = [p for p in anchored if p["method"] in {"source_inspection", "cached_line_match"}]
        if not direct:
            for line in section:
                if not line["placement"]:
                    line["flags"].append("uncorroborated_folio_label")
            continue
        rotations = {p["rotation"] for p in anchored}
        if any(p["image_sha256"] != sha or p["crop"] for p in anchored) or len(rotations) > 1:
            for line in section:
                line["flags"].append("folio_view_conflict")
            continue
        for line in section:
            if not line["placement"]:
                line["placement"] = {"method": "matching_folio_label", "image_sha256": sha,
                                     "rotation": next(iter(rotations), 0), "crop": None, "folio": label}
                line["flags"].append("verify_catalogue_face")


def previous_locations(lines: list[dict], pairs: list[dict]) -> None:
    """Use unique edition text as a location witness, preserving disagreements and skipped source rows.

    :param lines: Physical edition baselines after direct photo matching and source inspection.
    :param pairs: Former assembly proposals of this edition, including their view transforms.
    """
    by_text = defaultdict(list)
    for pair in pairs:
        if pair["decision"] not in {"accept", "review"} or pair.get("dup_rank") or not pair.get("image_sha256"):
            continue
        for i, text in enumerate(pair["lines"]):
            by_text[textnorm.skeleton(text)].append((pair, i))
    keys = [textnorm.skeleton(line["text"] or "") for line in lines]
    counts = Counter(keys)
    last = None
    for i, (line, key) in enumerate(zip(lines, keys)):
        if line["kind"] != "source_or_gap" or {"possible_apparatus",
                                                "unresolved_inline_annotation"}.intersection(line["flags"]):
            last = None
            continue
        split_layout = "check_column_order" in line["flags"]
        if split_layout:
            last = None
        hits = by_text.get(key, [])
        if len(key) < 4 or counts[key] != 1 or len(hits) != 1:
            if line["placement"]:
                last = None
            continue
        pair, position = hits[0]
        view = {k: pair[k] for k in ("image_sha256", "rotation", "crop")}
        current = line["placement"]
        if current and any(current[k] != v for k, v in view.items()):
            line["flags"].append("previous_view_differs")
            line["other_view"] = dict(view, previous_pair_id=pair["pair_id"], previous_line=position)
            last = None
            continue
        if not current:
            line["placement"] = dict(view, method="previous_pair_line", previous_pair_id=pair["pair_id"],
                                     previous_line=position)
        if last is not None:
            start, prior_pair, old_position = last
            if (pair["pair_id"] == prior_pair and lines[start]["pdf_page"] == line["pdf_page"]
                    and lines[start]["edition_folio"] == line["edition_folio"]
                    and 1 < i-start <= 8 and 0 < position-old_position <= 8):
                for missing in lines[start+1:i]:
                    if not missing["placement"]:
                        missing["placement"] = dict(view, method="between_previous_pair_lines",
                                                    anchor_line_ids=[lines[start]["line_id"], line["line_id"]])
                        missing["flags"].append("verify_intermediate_placement")
        last = None if split_layout else (i, pair["pair_id"], position)
    place_joined_predecessors(lines, by_text)


def place_joined_predecessors(lines: list[dict], prior: dict[str, list[tuple[dict, int]]]) -> None:
    """Locate an old fused line only when it uniquely equals whole consecutive printed baselines.

    :param lines: Source baselines in order. An unresolved mark or a new PDF page interrupts the search.
    :param prior: Previous locations by letter sequence.
    """
    parts, starts, ends = [], {}, {}
    size = 0
    section = None
    for i, line in enumerate(lines):
        key = (line["pdf_page"], line["edition_folio"])
        if key != section:
            parts.append("|")
            size += 1
            section = key
        admissible = (line["kind"] == "source_or_gap"
                      and not {"possible_apparatus", "unresolved_inline_annotation"}.intersection(line["flags"]))
        part = textnorm.skeleton(line["text"] or "") if admissible else ""
        starts[size] = i
        parts.append(part or "|")
        size += len(parts[-1])
        ends[size] = i
    full = "".join(parts)
    for old, hits in prior.items():
        if len(old) < MIN_ANCHOR or len(hits) != 1:
            continue
        offset = full.find(old)
        if offset not in starts or offset+len(old) not in ends or full.find(old, offset+1) >= 0:
            continue
        first, last = starts[offset], ends[offset+len(old)]
        if first >= last:
            continue
        pair, old_i = hits[0]
        view = {k: pair[k] for k in ("image_sha256", "rotation", "crop")}
        span = lines[first:last+1]
        if any(l["placement"] and any(l["placement"][k] != v for k, v in view.items()) for l in span):
            for line in span:
                line["flags"].append("prior_merged_line_view_conflict")
            continue
        for line in span:
            if not line["placement"]:
                line["placement"] = dict(view, method="previous_merged_line", previous_pair_id=pair["pair_id"],
                                         previous_line=old_i)


def place_block_edges(lines: list[dict]) -> None:
    """Recover short source runs at the edges of a continuous printed block as provisional placements.

    :param lines: Edition baselines with existing source or photo evidence. All uncertainty survives.
    """
    blocks: list[list[dict]] = []
    block: list[dict] = []
    for line in lines:
        barrier = (line["kind"] != "source_or_gap" or
                   {"check_column_order", "possible_apparatus", "unresolved_inline_annotation",
                    "previous_view_differs"}.intersection(line["flags"]))
        if block and (barrier or (block[-1]["pdf_page"], block[-1]["edition_folio"]) !=
                      (line["pdf_page"], line["edition_folio"]) or
                      line["baseline_pt"] - block[-1]["baseline_pt"] > 38):
            blocks.append(block)
            block = []
        if not barrier:
            block.append(line)
    if block:
        blocks.append(block)
    for block in blocks:
        placed = [l["placement"] for l in block if l["placement"]]
        views = {(p["image_sha256"], p["rotation"], tuple(p["crop"] or ())) for p in placed}
        if len(views) != 1:
            continue
        photo, rotation, crop = views.pop()
        if crop:
            continue
        anchors = [i for i, l in enumerate(block) if l["placement"] and l["placement"]["method"]
                   in {"source_inspection", "cached_line_match", "previous_pair_line"}]
        if len(anchors) < 2 or len(anchors) * 3 < len(block):
            continue
        for i, line in enumerate(block):
            if line["placement"] or not (i < anchors[0] or i > anchors[-1]):
                continue
            nearest = anchors[:2] if i < anchors[0] else anchors[-2:]
            if min(abs(i-j) for j in nearest) <= 4:
                line["placement"] = {"image_sha256": photo, "rotation": rotation, "crop": None,
                                     "method": "edition_block_edge", "anchor_line_ids": [block[j]["line_id"] for j in nearest]}
                line["flags"].append("verify_block_edge_on_photo")
