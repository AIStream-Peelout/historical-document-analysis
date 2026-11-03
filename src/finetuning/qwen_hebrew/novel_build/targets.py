"""Target texts of the novel-rows candidates: one :class:`~.schema.TargetSide` per side, with provenance.

The edition text of a candidate comes from its record in the candidates directory (same ``row_id``) unless said
otherwise. Per source:

* ``pgp``: the edition is cut into labelled blocks by :func:`build_pgp_editions.parse_edition` (Recto / Verso /
  ``ע"א`` / ``ע"ב`` / margin / address labels; blocks before the first side label belong to the recto). One side per
  side label in edition order, assembled as that builder assembles a page (:func:`build_pgp_editions.assemble_answer`:
  the side's main blocks, then its margin, address and other blocks, each in edition order); an edition without side
  labels is one side ``""``. That builder cleans every line while it parses (:func:`build_pgp_editions.clean_line`:
  notation normalisation, then ``clean_diplomatic``, lost text as ``[...]``), so ``raw_text`` is the side's block
  lines joined and ``text`` is the same after the builder's row normalisation (NFC, stripped). Arabic-script
  editions (ledger script ``arabic``, or ``mixed`` with at least as many Arabic as Hebrew letters) are cut by the
  Arabic-script builder instead (:func:`build_pgp_arabic_editions.best_sections`, which runs
  ``arabic_script.split_sections`` with the ``[...]`` gap token, and :func:`build_pgp_arabic_editions.sides_of`).
* ``fjp``: the legacy-store cleaner's side record (``cleaned_sections.jsonl``), joined on ``(canonical_id,
  section_index)``; a ``#r<k>`` row id names the record whose ``record_index`` (its position among the document's
  records, file order) is ``k``. The text is already cleaned.
* ``fjp_scrape``: the raw site-scrape text through the legacy-store sigla rules
  (:func:`legacy_store_sigla.clean_section`; one side per side the rules find, the record's own side when there is
  one).
* ``ktiv``: the library's page reconstruction as one side ``page``.
* ``literature``: the scanned-edition blocks (``lit_blocks.jsonl``) of the candidate's work, part, pages and raw shelf
  keys, in page order, checked against the candidate's text block by block.
* ``lit_run2``: the run-2 edition record with ``doc_id`` = ``row_id``, one side per line side (``unknown`` = ``""``).
* ``apd_v3`` / ``aodeh_v2_1``: pass-through of an earlier build's row (its answer text as the candidate extractor took
  it, the manifest side, the manifest image).

Every side: ``lines`` = the non-empty lines of ``text``, letters by :func:`count_letters`. A candidate left without a
side gets reason codes in ``notes["target_reasons"]``; non-fatal flags per side key go to ``notes["target_flags"]``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.datasets.cleaning import legacy_store_sigla as sigla
from src.finetuning.qwen_hebrew import build_pgp_arabic_editions as pae
from src.finetuning.qwen_hebrew import build_pgp_editions as bpe
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, ImageRef, TargetSide, read_jsonl, write_jsonl

HEBREW_LETTER_RE = re.compile("[א-ת]")
ARABIC_LETTER_RE = re.compile("[ء-يٱ-ۓ]")
#: A text is ``mixed`` when both scripts have at least this many letters ...
MIXED_MIN_LETTERS = 20
#: ... and the minority script holds at least this share of all letters.
MIXED_MIN_SHARE = 0.10

#: The legacy-store cleaner's side records.
DEFAULT_SECTIONS = Path(os.environ.get(
    "NOVEL_BUILD_SECTIONS", str(paths.DATASETS_ROOT / "legacy_store_diplomatic" / "cleaned_sections.jsonl")))
#: The scanned-literature transcription blocks of the ledger.
DEFAULT_LIT_BLOCKS = Path(os.environ.get(
    "NOVEL_BUILD_LIT_BLOCKS", str(paths.REPO_ROOT / "logs" / "next_round" / "unseen_v1" / "lit_blocks.jsonl")))
_RUN2_DIR = paths.REPO_ROOT / "logs" / "run2_data_2026-10-27"
#: The run-2 edition files (``os.pathsep``-separated in the environment variable).
DEFAULT_RUN2: Tuple[Path, ...] = tuple(Path(p) for p in os.environ.get(
    "NOVEL_BUILD_RUN2", os.pathsep.join(str(_RUN2_DIR / name) for name in ("vlm_pages_hebrew_ja.jsonl",
                                                                            "vlm_pages_arabic.jsonl"))
).split(os.pathsep) if p)
#: Pass-through sources: dataset directory under the datasets root and the origin of its images.
PASS_THROUGH: Dict[str, Tuple[Path, str]] = {
    "apd_v3": (Path("arabic_external") / "apd" / "apd_arabic_pages_v3", "apd_local"),
    "aodeh_v2_1": (Path("book_edition_aodeh_1992_v2_1"), "aodeh_local"),
}

_LEGACY_ROW_ID_RE = re.compile(r"^(?P<cid>.+):(?P<section>\d+)(?:#r(?P<k>\d+))?$")
_LIT_SOURCE_RE = re.compile(r"^literature:(?P<work>[^/]+)/(?P<part>.+) pages (?P<pages>\d+(?:,\d+)*)$")
_PAGE_FILE_RE = re.compile(r"page_(\d+)_structured")
_KEY_LINE_RE = re.compile(r"[\s_](?:line|l)(\d+)$")
_KTIV_SOURCE_PREFIX = "nli_ktiv_transcription_"
_UNLABELLED_RUN2_SIDES = ("", "unknown")


@dataclass
class Built:
    """The sides built for one candidate, or why none could be built.

    :param sides: Target sides in edition order.
    :param reasons: Reason codes when no side could be built.
    :param flags: Non-fatal flags per side key.
    :param images: Photographs that replace the candidate's list (pass-through sources), or None to keep it.
    """

    sides: List[TargetSide] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    flags: Dict[str, List[str]] = field(default_factory=dict)
    images: Optional[List[ImageRef]] = None


def path_exists(path: str) -> bool:
    """Whether a local file exists (tests replace this function).

    :param path: File path.
    :return: True when the path exists.
    """
    return Path(path).exists()


def count_letters(text: str) -> Tuple[int, int]:
    """Hebrew and Arabic letters of a text.

    :param text: Any text.
    :return: ``(Hebrew letters U+05D0-05EA, Arabic letters U+0621-064A and U+0671-06D3)``.
    """
    return len(HEBREW_LETTER_RE.findall(text)), len(ARABIC_LETTER_RE.findall(text))


def measure_script(text: str) -> str:
    """The script of a text, measured from its letters.

    :param text: Any text.
    :return: ``mixed`` when both scripts have at least :data:`MIXED_MIN_LETTERS` letters and the minority holds at
        least :data:`MIXED_MIN_SHARE` of the letters, else ``hebrew`` or ``arabic`` by majority (``hebrew`` on a tie),
        ``none`` without letters.
    """
    hebrew, arabic = count_letters(text)
    if not hebrew + arabic:
        return "none"
    minority = min(hebrew, arabic)
    if minority >= MIXED_MIN_LETTERS and minority >= MIXED_MIN_SHARE * (hebrew + arabic):
        return "mixed"
    return "hebrew" if hebrew >= arabic else "arabic"


def make_side(side_key: str, text: str, raw_text: str, pointer: Dict[str, Any],
              side_marker: Optional[str] = None) -> TargetSide:
    """A target side with its lines and letter counts.

    :param side_key: Side key.
    :param text: Target text after the source's sigla rules.
    :param raw_text: Text as extracted.
    :param pointer: Provenance.
    :param side_marker: The edition's own side marker, if any.
    :return: The side.
    """
    hebrew, arabic = count_letters(text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return TargetSide(side_key=side_key, text=text, raw_text=raw_text, lines=lines, pointer=pointer,
                      side_marker=side_marker, letters_hebrew=hebrew, letters_arabic=arabic)


def unique(items: Sequence[Any]) -> List[Any]:
    """Distinct items in first-seen order.

    :param items: Items.
    :return: The items without repeats.
    """
    return list(dict.fromkeys(items))


def unique_keys(keys: Sequence[str]) -> List[str]:
    """Make side keys unique inside one candidate (a repeated key gets ``~<n>``).

    :param keys: Side keys in order.
    :return: The keys, the second and later copies of a key suffixed ``~2``, ``~3``, ...
    """
    seen: Counter = Counter()
    out: List[str] = []
    for key in keys:
        seen[key] += 1
        out.append(key if seen[key] == 1 else f"{key}~{seen[key]}")
    return out


def note_field(notes: str, key: str) -> Optional[str]:
    """One ``key=value`` field of a ``;``-separated notes string.

    :param notes: Notes string.
    :param key: Field name.
    :return: The value, stripped, or None when absent.
    """
    match = re.search(rf"(?:^|;)\s*{re.escape(key)}=([^;]*)", notes or "")
    return match.group(1).strip() if match else None


def leading_int(value: Optional[str]) -> Optional[int]:
    """The integer a value starts with (``"0/1"`` -> 0).

    :param value: Text or None.
    :return: The integer, or None.
    """
    match = re.match(r"\d+", value or "")
    return int(match.group(0)) if match else None


def source_records(candidates_dir: Path, source: str, row_ids: Optional[Set[str]] = None,
                   drop: Sequence[str] = ("text_shingles",)) -> Dict[str, Dict[str, Any]]:
    """The records of one source's candidate file, by ``row_id``.

    :param candidates_dir: Directory of ``<source>.jsonl`` files.
    :param source: Ledger source name.
    :param row_ids: Keep only these ids (all when None).
    :param drop: Fields left out of the returned records.
    :return: ``{row_id: record}``; empty when the source has no candidate file.
    """
    path = candidates_dir / f"{source}.jsonl"
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for record in read_jsonl(path):
        row_id = str(record.get("row_id"))
        if row_ids is None or row_id in row_ids:
            out[row_id] = {k: v for k, v in record.items() if k not in drop}
    return out


# ----------------------------------------------------------------------------- pgp


def use_arabic_splitter(text: str, ledger_script: str) -> bool:
    """Whether a PGP edition is cut by the Arabic-script builder.

    :param text: Edition text.
    :param ledger_script: The ledger's script of the candidate.
    :return: True for ``arabic``, and for ``mixed`` with at least as many Arabic as Hebrew letters.
    """
    hebrew, arabic = count_letters(text)
    return ledger_script == "arabic" or (ledger_script == "mixed" and arabic >= hebrew)


def pgp_hebrew_sides(text: str, pgpid: str) -> Built:
    """Sides of a PGP edition by its side labels (:func:`build_pgp_editions.parse_edition`).

    :param text: Edition text.
    :param pgpid: PGP id (pointer).
    :return: One side per side label (``""`` for an edition without side labels), its marker the label line that
        named the side (``Recto``, ``ע"א``); flags: ``complex_side_labels``, ``flattened_line_breaks``,
        ``unrendered_lines`` and ``notation_<flag>`` for notation the cleaner left (the builder drops such pages).
    """
    edition = bpe.parse_edition(text)
    edition_flags = (["complex_side_labels"] if edition.complex_sides else []) + (
        ["flattened_line_breaks"] if edition.flattened else [])
    kept = [True] * len(edition.blocks)
    order: List[Optional[str]] = unique([b.side for b in edition.blocks]) if edition.labelled else [None]
    built = Built()
    for side in order:
        answer = bpe.assemble_answer(edition, kept, side)
        raw = answer.text
        clean = unicodedata.normalize("NFC", raw).strip()
        if not clean:
            continue
        key = side or ""
        marker = next((b.label for b in edition.blocks
                       if side is not None and b.side == side and b.explicit_side), None)
        pointer = {"source": "pgp_export_footnotes", "pgpid": pgpid, "parser": "parse_edition",
                   "block_indices": list(answer.blocks), "kinds": [edition.blocks[i].kind for i in answer.blocks]}
        built.sides.append(make_side(key, clean, raw, pointer, marker))
        flags = edition_flags + [f"notation_{f}" for f in sorted({f for line in answer.flags for f in line})]
        if any(edition.blocks[i].n_dropped for i in answer.blocks):
            flags.append("unrendered_lines")
        if flags:
            built.flags[key] = flags
    if not built.sides:
        built.reasons.append("edition_no_lines")
    return built


def pgp_arabic_sides(text: str, pgpid: str) -> Built:
    """Sides of an Arabic-script PGP edition (:func:`build_pgp_arabic_editions.best_sections`, ``sides_of``).

    :param text: Edition text.
    :param pgpid: PGP id (pointer).
    :return: One side per side label (``""`` for text outside the labels, flagged ``text_outside_side_labels`` when
        labelled sides exist too); the marker is the side the label named (the splitter keeps no label line).
    """
    sections = pae.best_sections([text])
    by_side = pae.sides_of(sections)
    labelled = any(by_side)
    built = Built()
    for side, lines in by_side.items():
        raw = "\n".join(lines)
        clean = raw.strip()
        if not clean:
            continue
        indices = [i for i, (s, _) in enumerate(sections) if s == side]
        pointer = {"source": "pgp_export_footnotes", "pgpid": pgpid, "parser": "split_sections",
                   "block_indices": indices, "kinds": ["section"] * len(indices)}
        built.sides.append(make_side(side, clean, raw, pointer, side or None))
        if not side and labelled:
            built.flags[side] = ["text_outside_side_labels"]
    if not built.sides:
        built.reasons.append("edition_no_lines")
    return built


# ----------------------------------------------------------------------------- legacy store


def load_sections(path: Path) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """The legacy-store cleaner's side records by ``(canonical_id, section_index)``.

    :param path: ``cleaned_sections.jsonl``.
    :return: Records in file order, each with ``record_index`` = its position among its document's records.
    """
    seen: Counter = Counter()
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(path):
        cid = str(record["canonical_id"])
        groups[(cid, int(record["section_index"]))].append(dict(record, record_index=seen[cid]))
        seen[cid] += 1
    return dict(groups)


def find_section(row_id: str,
                 sections: Dict[Tuple[str, int], List[Dict[str, Any]]]) -> Tuple[Optional[Dict[str, Any]], str]:
    """The side record of a legacy-store candidate.

    ``row_id`` = ``<canonical_id>:<section_index>``, plus ``#r<k>`` when the section holds several records; ``k`` is
    the record's ``record_index`` (the candidate extractor writes it so).

    :param row_id: Ledger row id.
    :param sections: :func:`load_sections` result.
    :return: ``(record, "")`` or ``(None, reason)``.
    """
    match = _LEGACY_ROW_ID_RE.match(row_id)
    if match is None:
        return None, "row_id_unparsed"
    group = sections.get((match["cid"], int(match["section"])), [])
    if not group:
        return None, "legacy_section_not_found"
    if match["k"] is None:
        return (group[0], "") if len(group) == 1 else (None, "legacy_section_ambiguous")
    hits = [r for r in group if r["record_index"] == int(match["k"])]
    return (hits[0], "") if hits else (None, "legacy_record_not_found")


def legacy_section_sides(row_id: str, sections: Dict[Tuple[str, int], List[Dict[str, Any]]], file_name: str,
                         edition_text: Optional[str] = None) -> Built:
    """The side of a legacy-store candidate from the cleaner's record (text already cleaned).

    :param row_id: Ledger row id.
    :param sections: :func:`load_sections` result.
    :param file_name: Name of the sections file (pointer).
    :param edition_text: The candidate record's text, to cross-check the join (flag
        ``section_text_differs_from_candidate``).
    :return: One side keyed by the record's side (``""`` without one), its marker that side when the cleaner's side
        decision is ``marker``; flags = the cleaner's flags.
    """
    record, reason = find_section(row_id, sections)
    if record is None:
        return Built(reasons=[reason])
    raw = str(record.get("text") or "")
    text = raw.strip()
    if not text:
        return Built(reasons=["section_text_empty"])
    key = str(record.get("side") or "")
    pointer = {"source": "legacy_store_sections", "file": file_name, "canonical_id": record.get("canonical_id"),
               "doc_id": record.get("doc_id"), "section_index": record.get("section_index"),
               "record_index": record["record_index"], "column": record.get("column"), "family": record.get("family"),
               "side_source": record.get("side_source"), "side_decision": record.get("side_decision")}
    marker = (key or None) if record.get("side_decision") == "marker" else None
    flags = sorted(k for k, v in (record.get("flags") or {}).items() if v)
    if edition_text is not None and edition_text != raw:
        flags.append("section_text_differs_from_candidate")
    return Built(sides=[make_side(key, text, raw, pointer, marker)], flags={key: flags} if flags else {})


def scrape_sides(raw: str, side: Optional[str], canonical_id: Optional[str], notes: str) -> Built:
    """Sides of a legacy-store site-scrape part through the sigla rules (:func:`legacy_store_sigla.clean_section`).

    The rules run without a lexicon (fragment-order and uncertain-wrap repairs then change nothing).

    :param raw: Raw scrape text.
    :param side: The record's side, if any (used when the rules find one side).
    :param canonical_id: Document id (pointer).
    :param notes: The record's notes (``entry=``, ``block=``, ``part=`` for the pointer).
    :return: One side per side the rules find; flags = the rules' flags.
    """
    cleaned = sigla.clean_section(raw, sigla.finals_mapping_for(raw), family=sigla.classify_family(raw))
    if not cleaned:
        return Built(reasons=["no_text_after_sigla"])
    single = len(cleaned) == 1
    keys = unique_keys([(side if single and side else None) or s.side or "" for s in cleaned])
    base = {"source": "legacy_store_site_scrape", "canonical_id": canonical_id,
            "entry": leading_int(note_field(notes, "entry")), "block": leading_int(note_field(notes, "block")),
            "part": note_field(notes, "part")}
    built = Built()
    for i, (key, section) in enumerate(zip(keys, cleaned)):
        pointer = dict(base, sigla_side=i, sigla_sides=len(cleaned), family=section.family)
        marker = section.side if section.side_source == "marker" else (side if single and side else None)
        built.sides.append(make_side(key, section.text.strip(), raw, pointer, marker))
        flags = sorted(k for k, v in section.flags.items() if v)
        if flags:
            built.flags[key] = flags
    return built


# ----------------------------------------------------------------------------- ktiv


def ktiv_sides(text: str, ktiv_id: Optional[str], row_id: str, notes: str, edition_source: str) -> Built:
    """The library page reconstruction as one side ``page``.

    :param text: Edition text.
    :param ktiv_id: Library system number.
    :param row_id: Ledger row id (``<sys_num>:<fl>``).
    :param notes: The record's notes (``bundle=``).
    :param edition_source: The record's edition source (``..._api`` / ``..._dom``: the pointer source).
    :return: One side.
    """
    shape = edition_source[len(_KTIV_SOURCE_PREFIX):] if edition_source.startswith(_KTIV_SOURCE_PREFIX) else "api"
    pointer: Dict[str, Any] = {"source": f"ktiv_{shape}", "sys_num": ktiv_id,
                               "fl": row_id.split(":", 1)[1] if ":" in row_id else row_id,
                               "bundle": note_field(notes, "bundle")}
    losing = note_field(notes, "from_losing_bundle")
    if losing:
        pointer["from_losing_bundle"] = losing
    return Built(sides=[make_side("page", text.strip(), text, pointer)])


# ----------------------------------------------------------------------------- literature


def page_number(page_file: str) -> Optional[int]:
    """Page number of a structured page file (``page_013_structured.json`` -> 13).

    :param page_file: Page file path.
    :return: The number, or None.
    """
    match = _PAGE_FILE_RE.search(page_file or "")
    return int(match.group(1)) if match else None


def block_order(block: Dict[str, Any]) -> Tuple[int, int, int]:
    """Reading order of a literature block: page, line number in its key (``..._line11``), file order.

    :param block: Block record with ``file_order``.
    :return: Sort key.
    """
    line = _KEY_LINE_RE.search(str(block.get("shelf_key") or ""))
    return page_number(str(block.get("page_file") or "")) or 0, int(line.group(1)) if line else 0, block["file_order"]


def load_lit_blocks(path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Literature blocks by ``(work, part)``.

    :param path: ``lit_blocks.jsonl``.
    :return: Blocks with ``file_order`` added.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for i, block in enumerate(read_jsonl(path)):
        groups[(str(block.get("work")), str(block.get("part")))].append(dict(block, file_order=i))
    return dict(groups)


def match_blocks(blocks: Sequence[Dict[str, Any]], text: str) -> Optional[List[Dict[str, Any]]]:
    """The blocks whose texts, joined by newlines in order, are exactly the candidate's text.

    The candidate extractor joined its blocks' texts with newlines; a block of another candidate (same key and page,
    another side) is skipped when the text does not continue with it.

    :param blocks: Candidate blocks in reading order.
    :param text: The candidate's edition text.
    :return: The matching blocks, or None when they do not rebuild the text.
    """
    if blocks and "\n".join(str(b.get("text") or "") for b in blocks) == text:
        return list(blocks)
    pos, chosen = 0, []
    for block in blocks:
        part = str(block.get("text") or "")
        end = pos + len(part)
        if part and text.startswith(part, pos) and (end == len(text) or text[end] == "\n"):
            chosen.append(block)
            pos = end + 1
    return chosen if chosen and pos >= len(text) else None


def literature_sides(edition_source: str, raw_keys: Sequence[str], side: Optional[str], edition_text: str,
                     blocks: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> Built:
    """The side of a scanned-edition candidate from its literature blocks.

    The candidate extractor groups blocks by side, so the blocks of one candidate share its side: one side keyed by
    the candidate's side (``""`` without one).

    :param edition_source: ``literature:<work>/<part> pages <n>,<m>``.
    :param raw_keys: The candidate's raw shelf keys (``resolution.raw_keys``).
    :param side: The candidate's side, if any.
    :param edition_text: The candidate's text (the blocks must rebuild it).
    :param blocks: :func:`load_lit_blocks` result.
    :return: One side, or reasons ``edition_source_unparsed`` / ``lit_blocks_not_found`` / ``lit_blocks_mismatch``.
    """
    match = _LIT_SOURCE_RE.match(edition_source or "")
    if match is None:
        return Built(reasons=["edition_source_unparsed"])
    pages = {int(p) for p in match["pages"].split(",")}
    keys = set(raw_keys)
    pool = sorted((b for b in blocks.get((match["work"], match["part"]), [])
                   if page_number(str(b.get("page_file") or "")) in pages and b.get("shelf_key") in keys),
                  key=block_order)
    if not pool:
        return Built(reasons=["lit_blocks_not_found"])
    chosen = match_blocks(pool, edition_text)
    if chosen is None:
        return Built(reasons=["lit_blocks_mismatch"])
    raw = "\n".join(str(b.get("text") or "") for b in chosen)
    shelf_keys = unique([str(b.get("shelf_key")) for b in chosen])
    pointer: Dict[str, Any] = {"source": "scanned_edition", "work": match["work"], "part": match["part"],
                               "page_files": unique([str(b.get("page_file")) for b in chosen]),
                               "shelf_key": shelf_keys[0]}
    if len(shelf_keys) > 1:
        pointer["shelf_keys"] = shelf_keys
    return Built(sides=[make_side(side or "", raw.strip(), raw, pointer, side or None)])


# ----------------------------------------------------------------------------- run 2


def load_run2(run2_path: Union[Path, Sequence[Path]]) -> Dict[str, Tuple[Dict[str, Any], str]]:
    """Run-2 edition records by ``doc_id``.

    :param run2_path: One file or several (the first record of an id wins).
    :return: ``{doc_id: (record, file name)}``.
    """
    files = [Path(run2_path)] if isinstance(run2_path, (str, Path)) else [Path(p) for p in run2_path]
    out: Dict[str, Tuple[Dict[str, Any], str]] = {}
    for path in files:
        for record in read_jsonl(path):
            out.setdefault(str(record.get("doc_id")), (record, path.name))
    return out


def run2_sides(record: Dict[str, Any], file_name: str) -> Built:
    """Sides of a run-2 edition record: its lines grouped by their side.

    :param record: Run-2 record (``transcription_lines`` = ``[{n, side, part, text}]``).
    :param file_name: Name of the file it came from (pointer).
    :return: One side per line side in order of appearance (``unknown`` -> ``""``).
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for line in record.get("transcription_lines") or []:
        side = str(line.get("side") or "")
        groups.setdefault("" if side in _UNLABELLED_RUN2_SIDES else side, []).append(line)
    built = Built()
    for key, lines in groups.items():
        raw = "\n".join(str(line.get("text") or "") for line in lines)
        if not raw.strip():
            continue
        pointer = {"source": "run2_edition", "file": file_name, "doc_id": record.get("doc_id"),
                   "line_numbers": [line.get("n") for line in lines]}
        built.sides.append(make_side(key, raw.strip(), raw, pointer, key if key in ("recto", "verso") else None))
    if not built.sides:
        built.reasons.append("run2_no_lines")
    return built


# ----------------------------------------------------------------------------- pass-through


def load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    """A built dataset's manifest by ``stem``.

    :param path: ``manifest.jsonl``.
    :return: ``{stem: row}``.
    """
    return {str(row["stem"]): row for row in read_jsonl(path)}


def pass_through_image(row: Dict[str, Any], dataset_dir: Path, stem: str, origin: str) -> ImageRef:
    """The photograph of a built row: the manifest's file, else ``<dataset>_images/<stem>.jpg``.

    :param row: Manifest row.
    :param dataset_dir: The dataset directory.
    :param stem: Row stem.
    :param origin: Image origin (``apd_local`` / ``aodeh_local``).
    :return: The image (``present`` when the file exists, else ``missing``).
    """
    fallback = dataset_dir.with_name(dataset_dir.name + "_images") / f"{stem}.jpg"
    options = [str(p) for p in (row.get("image_path"), fallback) if p]
    path = next((p for p in options if path_exists(p)), None)
    canvas = row.get("canvas_index", row.get("canvas"))
    return ImageRef(image_id=f"{origin}:0", origin=origin, url=row.get("image_url") or None, path=path or options[0],
                    sha256=row.get("image_sha256") or None, canvas_index=int(canvas) if canvas is not None else None,
                    label=row.get("canvas_label") or None, status="present" if path else "missing",
                    note="" if path else "image file not found")


def pass_through_sides(cand: Candidate, record: Optional[Dict[str, Any]], datasets_root: Path,
                       manifests: Dict[str, Dict[str, Dict[str, Any]]]) -> Built:
    """The side and photograph of a pass-through candidate (an earlier build's row).

    :param cand: Candidate (``apd_v3`` / ``aodeh_v2_1``).
    :param record: Its candidate record (``edition_text`` = the row's answer).
    :param datasets_root: Datasets root.
    :param manifests: Manifest cache by source (filled here).
    :return: One side keyed by the manifest side, and the image that replaces the candidate's list.
    """
    rel, origin = PASS_THROUGH[cand.source]
    dataset_dir = datasets_root / rel
    if cand.source not in manifests:
        manifests[cand.source] = load_manifest(dataset_dir / "manifest.jsonl")
    stem = str(cand.notes.get("stem") or cand.row_id.split(":", 1)[-1])
    row = manifests[cand.source].get(stem)
    if row is None:
        return Built(reasons=["manifest_row_not_found"])
    images = [pass_through_image(row, dataset_dir, stem, origin)]
    text = (record or {}).get("edition_text")
    if not isinstance(text, str) or not text.strip():
        return Built(reasons=["no_candidate_record" if record is None else "edition_text_empty"], images=images)
    pointer = {"source": dataset_dir.name, "stem": stem}
    return Built(sides=[make_side(str(row.get("side") or ""), text.strip(), text, pointer)], images=images)


# ----------------------------------------------------------------------------- dispatch


@dataclass
class TargetInputs:
    """Per-run inputs of :func:`build_sides`.

    :param sections: :func:`load_sections` result.
    :param sections_name: Name of the sections file (pointer).
    :param blocks: :func:`load_lit_blocks` result.
    :param run2: :func:`load_run2` result.
    :param datasets_root: Datasets root (pass-through sources).
    :param manifests: Manifest cache of the pass-through sources.
    """

    sections: Dict[Tuple[str, int], List[Dict[str, Any]]]
    sections_name: str
    blocks: Dict[Tuple[str, str], List[Dict[str, Any]]]
    run2: Dict[str, Tuple[Dict[str, Any], str]]
    datasets_root: Path
    manifests: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)


def build_sides(cand: Candidate, record: Optional[Dict[str, Any]], inputs: TargetInputs) -> Built:
    """The target sides of one candidate, by source.

    :param cand: Candidate.
    :param record: Its record in the candidates directory, or None.
    :param inputs: Per-run inputs.
    :return: Sides, or the reasons there are none.
    """
    source = cand.source
    if source in PASS_THROUGH:
        return pass_through_sides(cand, record, inputs.datasets_root, inputs.manifests)
    if source == "fjp":
        return legacy_section_sides(cand.row_id, inputs.sections, inputs.sections_name,
                                    (record or {}).get("edition_text"))
    if source == "lit_run2":
        hit = inputs.run2.get(cand.row_id)
        return run2_sides(*hit) if hit else Built(reasons=["run2_record_not_found"])
    if record is None:
        return Built(reasons=["no_candidate_record"])
    text = record.get("edition_text")
    if not isinstance(text, str) or not text.strip():
        return Built(reasons=["edition_text_empty"])
    if source == "pgp":
        pgpid = cand.pgpids[0] if cand.pgpids else cand.row_id.split(":", 1)[-1]
        if use_arabic_splitter(text, cand.script):
            return pgp_arabic_sides(text, pgpid)
        return pgp_hebrew_sides(text, pgpid)
    if source == "fjp_scrape":
        canonical_id = record.get("canonical_id") or (cand.canonical_ids[0] if cand.canonical_ids else None)
        return scrape_sides(text, record.get("side") or None, canonical_id, str(record.get("notes") or ""))
    if source == "ktiv":
        return ktiv_sides(text, cand.ktiv_id, cand.row_id, str(record.get("notes") or ""),
                          str(record.get("edition_source") or ""))
    if source == "literature":
        resolution = cand.notes.get("resolution") or {}
        return literature_sides(str(cand.notes.get("edition_source") or ""), resolution.get("raw_keys") or [],
                                cand.notes.get("side"), text, inputs.blocks)
    return Built(reasons=["unknown_source"])


def attach_targets(cands: List[Candidate], *, candidates_dir: Path, sections_path: Path, lit_blocks_path: Path,
                   run2_path: Union[Path, Sequence[Path]], datasets_root: Path) -> None:
    """Fill ``cand.targets`` of every candidate (in place; earlier targets and target notes are replaced).

    Inputs are read only for the sources present. Pass-through sources also get their manifest image as
    ``cand.images``.

    :param cands: Candidates.
    :param candidates_dir: Directory of the candidate files (edition texts).
    :param sections_path: The legacy-store cleaner's side records.
    :param lit_blocks_path: The literature blocks.
    :param run2_path: The run-2 edition file, or several.
    :param datasets_root: Datasets root (pass-through sources).
    """
    wanted: Dict[str, Set[str]] = defaultdict(set)
    for cand in cands:
        wanted[cand.source].add(cand.row_id)
    records = {source: source_records(candidates_dir, source, ids) for source, ids in wanted.items()}
    inputs = TargetInputs(sections=load_sections(sections_path) if "fjp" in wanted else {},
                          sections_name=sections_path.name,
                          blocks=load_lit_blocks(lit_blocks_path) if "literature" in wanted else {},
                          run2=load_run2(run2_path) if "lit_run2" in wanted else {}, datasets_root=datasets_root)
    for cand in cands:
        built = build_sides(cand, records[cand.source].get(cand.row_id), inputs)
        cand.targets = built.sides
        if built.images is not None:
            cand.images = built.images
        cand.notes.pop("target_reasons", None)
        cand.notes.pop("target_flags", None)
        if not built.sides:
            cand.notes["target_reasons"] = built.reasons or ["no_target"]
        if built.flags:
            cand.notes["target_flags"] = built.flags


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (re-attach targets to an existing stage-0 file).

    :return: The parser.
    """
    ap = argparse.ArgumentParser(description="Re-attach target texts to the stage-0 candidates file, in place.")
    ap.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR, help="Build work dir.")
    ap.add_argument("--candidates-dir", type=Path, default=paths.DEFAULT_CANDIDATES_DIR,
                    help="Directory of the per-source candidate files.")
    ap.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS,
                    help="Legacy-store cleaned side records (env NOVEL_BUILD_SECTIONS).")
    ap.add_argument("--lit-blocks", type=Path, default=DEFAULT_LIT_BLOCKS,
                    help="Literature blocks (env NOVEL_BUILD_LIT_BLOCKS).")
    ap.add_argument("--run2", type=Path, nargs="+", default=list(DEFAULT_RUN2),
                    help="Run-2 edition files (env NOVEL_BUILD_RUN2).")
    ap.add_argument("--datasets-root", type=Path, default=paths.DATASETS_ROOT,
                    help="Datasets root of the pass-through sources.")
    ap.add_argument("--sources", default="", help="Comma-separated sources (default: all).")
    ap.add_argument("--limit", type=int, default=None, help="At most N candidates per source.")
    ap.add_argument("--missing-only", action="store_true", help="Only candidates without targets (resume).")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    """Re-attach targets to the stage-0 file of a work dir and print per-source counts.

    :param argv: Arguments (default: ``sys.argv``).
    """
    args = build_parser().parse_args(argv)
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    stage = paths.stage_path(args.work_dir, "candidates")
    cands: List[Candidate] = list(read_jsonl(stage, Candidate))
    taken: Counter = Counter()
    selected: List[Candidate] = []
    for cand in cands:
        if (sources and cand.source not in sources) or (args.missing_only and cand.targets):
            continue
        if args.limit is not None and taken[cand.source] >= args.limit:
            continue
        taken[cand.source] += 1
        selected.append(cand)
    attach_targets(selected, candidates_dir=args.candidates_dir, sections_path=args.sections,
                   lit_blocks_path=args.lit_blocks, run2_path=args.run2, datasets_root=args.datasets_root)
    write_jsonl(stage, cands)
    counts = {source: {"candidates": n, "with_targets": sum(1 for c in selected if c.source == source and c.targets)}
              for source, n in sorted(taken.items())}
    print(json.dumps({"stage_file": str(stage), "re_attached": len(selected), "by_source": counts}, indent=1))


if __name__ == "__main__":
    main()
