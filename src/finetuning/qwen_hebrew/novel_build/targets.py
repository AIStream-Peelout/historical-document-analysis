"""Target texts of the novel-rows candidates: one :class:`~.schema.TargetSide` per side, with provenance.

The edition text of a candidate comes from its record in the candidates directory (same ``row_id``) unless said
otherwise. Per source:

* ``pgp``: the edition is cut into labelled blocks by :func:`build_pgp_editions.parse_edition` (Recto / Verso /
  ``ע"א`` / ``ע"ב`` / margin / address labels; blocks before the first side label belong to the recto). One side per
  side label in edition order, assembled as that builder assembles a page (:func:`build_pgp_editions.assemble_answer`:
  the side's main blocks, then its margin, address and other blocks, each in edition order); an edition without side
  labels is one side ``""``. An edition with complex side labels (the parser's ``complex_sides``: page, folio or
  fragment labels such as ``Fol. 2r``, ``Recto (page 3)`` or ``page b``, or a side opened twice by a bare label) is
  cut by its sections instead (:func:`section_groups`): one side per section label, keyed by that label, so the text
  of two folios that share a side name is never joined; blocks before the first section label form the side ``""``.
  That builder cleans every line while it parses (:func:`build_pgp_editions.clean_line`: notation normalisation,
  then ``clean_diplomatic``, lost text as ``[...]``), so ``raw_text`` is the side's block lines joined and ``text``
  is the same after the builder's row normalisation (NFC, stripped). Letters supplied by the editor but stored with
  reversed brackets (``>ה<``) are first rewritten as the builders expect them (``<ה>``), so they are dropped like
  every supplied span (:func:`unreverse_supplied`). Arabic-script editions (ledger script ``arabic``, or ``mixed``
  with at least as many Arabic as Hebrew letters) are cut by the Arabic-script builder instead
  (:func:`build_pgp_arabic_editions.best_sections`, which runs ``arabic_script.split_sections`` with the ``[...]``
  gap token, and :func:`build_pgp_arabic_editions.sides_of`).
* ``fjp``: the legacy-store cleaner's side record (``cleaned_sections.jsonl``), joined on ``(canonical_id,
  section_index)``; a ``#r<k>`` row id names the record whose ``record_index`` (its position among the document's
  records, file order) is ``k``. The text is already cleaned.
* ``fjp_scrape``: the raw site-scrape text through the legacy-store sigla rules
  (:func:`legacy_store_sigla.clean_section`; one side per side the rules find, the record's own side when there is
  one), with a lexicon of known words for the rules' word-order repairs (by default the words of the cleaned
  legacy-store sections, :func:`load_lexicon`).
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
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

from src.datasets.cleaning import legacy_store_sigla as sigla
from src.finetuning.qwen_hebrew import build_pgp_arabic_editions as pae
from src.finetuning.qwen_hebrew import build_pgp_editions as bpe
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, ImageRef, TargetSide, read_jsonl, write_jsonl

HEBREW_LETTER_RE = re.compile("[א-ת]")
HEBREW_RUN_RE = re.compile("[א-ת]+")
ARABIC_LETTER_RE = re.compile("[ء-يٱ-ۓ]")
#: A text is ``mixed`` when both scripts have at least this many letters ...
MIXED_MIN_LETTERS = 20
#: ... and the minority script holds at least this share of all letters.
MIXED_MIN_SHARE = 0.10

#: The legacy-store cleaner's side records.
DEFAULT_SECTIONS = Path(os.environ.get(
    "NOVEL_BUILD_SECTIONS", str(paths.DATASETS_ROOT / "legacy_store_diplomatic" / "cleaned_sections.jsonl")))
#: ``lexicon_path`` value of :func:`attach_targets`: the lexicon is built from its ``sections_path``.
LEXICON_FROM_SECTIONS = "sections"
#: Shortest Hebrew letter run that goes into a lexicon (the legacy-store build's rule).
LEXICON_MIN_LETTERS = 2
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
#: Letters supplied by the editor stored with reversed brackets (``>ה<`` for ``<ה>``), within the length the builder's
#: ``<x>`` rule takes; doubled brackets (``>>x<<``) are struck text, not supplied letters.
REVERSED_SUPPLIED_RE = re.compile(r"(?<![<>])>([^<>\n]{1,20})<(?![<>])")
#: ``page``/``sheet`` used for a place on the current page (``top of the page``, ``halfway down the page``).
_ON_PAGE_RE = re.compile(r"\b(?:top|bottom|middle|foot|head|edge|corner|side|back|backe)\s+of\s+(?:the\s+)?"
                         r"(?:page|sheet)\b|\bdown\s+the\s+(?:page|sheet)\b", re.I)
#: A label that only enumerates (``II``, ``b``): parts of a side when a section that names the side is in force.
_ENUMERATOR_RE = re.compile(r"\W*(?:[a-f]|[ivx]{1,4})\W*", re.I)


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


def unreverse_supplied(line: str) -> Tuple[str, int]:
    """Rewrite supplied-letter spans stored with reversed brackets (``>ה<``) the way editions write them (``<ה>``).

    Both edition builders drop ``<x>`` (letters the editor supplied are not ink) but not a reversed span: the
    Hebrew-script cleaner keeps it, brackets and letters, as unresolved notation, and the Arabic-script cleaner cuts
    the line at the stray brackets. A ``>`` that closes an earlier ``<`` of the line opens no span, so the ink between
    two supplied spans (``א<ל>גאני אב<ו>``) stays.

    :param line: One edition line.
    :return: ``(line, spans rewritten)``.
    """
    out: List[str] = []
    pos = n = 0
    for match in REVERSED_SUPPLIED_RE.finditer(line):
        before = line[:match.start()]
        if before.count("<") != before.count(">"):
            continue
        out += [line[pos:match.start()], f"<{match.group(1)}>"]
        pos, n = match.end(), n + 1
    return "".join(out) + line[pos:], n


def edition_lines(text: str) -> Tuple[List[str], List[int]]:
    """An edition's lines as :func:`build_pgp_editions.parse_edition` splits them, reversed supplied spans rewritten.

    :param text: Edition text.
    :return: ``(lines, reversed spans rewritten per line)``.
    """
    norm = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    pairs = [unreverse_supplied(line) for line in norm.split("\n")]
    return [line for line, _ in pairs], [n for _, n in pairs]


@dataclass
class WalkBlock:
    """One edition block as :func:`walk_edition` sees it.

    :param section: Index of the section label in force (None before the first one).
    :param lines: Rendered lines.
    :param n_dropped: Ink lines the edition does not render.
    :param reversed_spans: Reversed supplied spans rewritten on its rendered lines.
    """

    section: Optional[int]
    lines: List[str] = field(default_factory=list)
    n_dropped: int = 0
    reversed_spans: int = 0


@dataclass
class EditionWalk:
    """The section labels of an edition and its blocks (the parser's blocks, in the same order).

    :param labels: Section label lines (stripped), in edition order.
    :param blocks: Blocks that hold rendered or unrendered lines.
    """

    labels: List[str]
    blocks: List[WalkBlock]


def names_page(label: bpe.Label, line: str) -> bool:
    """Whether a label names a page, folio or fragment (``Fol. 2r``, ``page b``, ``II``, ``Second fragment``).

    The parser's paging test, except that a place on the current page (``top of the page``) names no other page.

    :param label: The parsed label line.
    :param line: The label line.
    :return: True for a page label.
    """
    if not label.paging:
        return False
    if not _ON_PAGE_RE.search(line):
        return True
    rest = bpe.parse_label(_ON_PAGE_RE.sub(" ", line))
    return bool(rest and rest.paging)


def opens_section(label: bpe.Label, line: str, in_force: bool, side_in_force: Optional[str]) -> bool:
    """Whether a label line opens a new section of an edition with complex side labels.

    A main label opens one when it names a side or a page (a bare side repeated, ``Folio 2, recto``, ``page b``),
    except an enumerator (``II``, ``b``) under a section that names a side, which numbers parts of that side. A
    region label (margin, address, other) stays in the section in force unless it names another side, and so does a
    main label that names neither (``bottom corner``, ``column 2``, ``top of the page``).

    :param label: The parsed label line.
    :param line: The label line.
    :param in_force: A section is in force.
    :param side_in_force: The side the section in force names, if any.
    :return: True when the label opens a section.
    """
    if label.kind != "main":
        return label.side is not None and not (in_force and label.side == side_in_force)
    if label.side is not None:
        return True
    if not names_page(label, line):
        return False
    return not (side_in_force is not None and _ENUMERATOR_RE.fullmatch(line.strip()))


def walk_edition(lines: Sequence[str], reversed_spans: Sequence[int]) -> EditionWalk:
    """Walk an edition the way :func:`build_pgp_editions.parse_edition` does, recording the section of each block.

    The walk uses the parser's own steps (line-number detection, rule lines, label test, apparatus stop, line
    cleaner), so its blocks are the parser's blocks; it also keeps the label of a block the parser drops because the
    label line is followed by a blank line, which is how most editions write a label.

    :param lines: Edition lines (:func:`edition_lines`).
    :param reversed_spans: Reversed supplied spans rewritten per line.
    :return: Section labels and blocks.
    """
    numbered = bpe.detect_line_numbering(lines)
    labels: List[str] = []
    label_sides: List[Optional[str]] = []
    blocks: List[WalkBlock] = []
    cur: Optional[WalkBlock] = None
    section: Optional[int] = None
    for raw, n_reversed in zip(lines, reversed_spans):
        stripped = raw.strip()
        if not stripped or bpe._RULE_RE.fullmatch(stripped):
            cur = None
            continue
        label = bpe.parse_label(raw)
        if label is not None:
            if label.kind == "stop":
                break
            if opens_section(label, raw, section is not None, None if section is None else label_sides[section]):
                labels.append(stripped)
                label_sides.append(label.side)
                section = len(labels) - 1
            cur = WalkBlock(section)
            blocks.append(cur)
            continue
        cleaned = bpe.clean_line(raw, numbered)
        if cleaned.kind == "commentary":
            continue
        if cur is None:
            cur = WalkBlock(section)
            blocks.append(cur)
        if cleaned.kind == "dropped":
            cur.n_dropped += 1
            continue
        cur.lines.append(cleaned.text)
        cur.reversed_spans += n_reversed
    return EditionWalk(labels, [b for b in blocks if b.lines or b.n_dropped])


def walk_matches(walk: EditionWalk, edition: bpe.ParsedEdition) -> bool:
    """Whether a walk found the parser's blocks (same count, lines and unrendered lines, in order).

    :param walk: :func:`walk_edition` result.
    :param edition: :func:`build_pgp_editions.parse_edition` result of the same lines.
    :return: True when they agree.
    """
    return len(walk.blocks) == len(edition.blocks) and all(
        w.lines == b.lines and w.n_dropped == b.n_dropped for w, b in zip(walk.blocks, edition.blocks))


@dataclass
class BlockGroup:
    """Edition blocks that make one target side.

    :param key: Side key.
    :param marker: The edition's label that named the side, if any.
    :param blocks: Block indices in edition order.
    :param flags: Flags the grouping adds.
    """

    key: str
    marker: Optional[str]
    blocks: List[int]
    flags: List[str] = field(default_factory=list)


def side_groups(edition: bpe.ParsedEdition) -> List[BlockGroup]:
    """Blocks by side label (``recto`` / ``verso``) in edition order; one group ``""`` without side labels.

    :param edition: Parsed edition.
    :return: The groups, each marked by the first label line of a block that named its side.
    """
    if not edition.labelled:
        return [BlockGroup("", None, list(range(len(edition.blocks))))]
    groups: List[BlockGroup] = []
    for side in unique([b.side for b in edition.blocks]):
        marker = next((b.label for b in edition.blocks if b.side == side and b.explicit_side), None)
        groups.append(BlockGroup(side or "", marker, [i for i, b in enumerate(edition.blocks) if b.side == side]))
    return groups


def section_groups(walk: EditionWalk) -> List[BlockGroup]:
    """Blocks by section of an edition with complex side labels (:func:`opens_section`).

    :param walk: :func:`walk_edition` result.
    :return: One group per section label in edition order, keyed and marked by the label (whitespace collapsed), the
        blocks before the first section label first as the group ``""`` (flag
        ``unlabelled_block_in_complex_edition``). A label that repeats keeps its own group.
    """
    members: Dict[Optional[int], List[int]] = {}
    for i, block in enumerate(walk.blocks):
        members.setdefault(block.section, []).append(i)
    groups: List[BlockGroup] = []
    for section, blocks in members.items():
        if section is None:
            groups.append(BlockGroup("", None, blocks, ["unlabelled_block_in_complex_edition"]))
        else:
            label = " ".join(walk.labels[section].split())
            groups.append(BlockGroup(label, label, blocks))
    return groups


def pgp_hebrew_sides(text: str, pgpid: str) -> Built:
    """Sides of a PGP edition by its side labels (:func:`build_pgp_editions.parse_edition`).

    Reversed supplied spans are rewritten first (:func:`unreverse_supplied`). An edition with complex side labels is
    cut by its sections (:func:`section_groups`), any other by its side labels (:func:`side_groups`); each side is
    assembled as the builder assembles a page (main blocks, then margin, address and other blocks).

    :param text: Edition text.
    :param pgpid: PGP id (pointer).
    :return: One side per group with text, keys made unique (:func:`unique_keys`); its marker the label line that
        named the side (``Recto``, ``ע"א``, ``Fol. 2r``); flags: ``complex_side_labels`` (every side of such an
        edition), ``unlabelled_block_in_complex_edition``, ``flattened_line_breaks``, ``notation_<flag>`` for
        notation the cleaner left (the builder drops such pages), ``unrendered_lines`` and ``supplied_reversed``.
        Reason ``edition_walk_mismatch`` when the section walk does not find the parser's blocks.
    """
    lines, reversed_spans = edition_lines(text)
    edition = bpe.parse_edition("\n".join(lines))
    walk = walk_edition(lines, reversed_spans)
    if not walk_matches(walk, edition):
        return Built(reasons=["edition_walk_mismatch"])
    edition_flags = (["complex_side_labels"] if edition.complex_sides else []) + (
        ["flattened_line_breaks"] if edition.flattened else [])
    groups = section_groups(walk) if edition.complex_sides else side_groups(edition)
    made: List[Tuple[BlockGroup, bpe.Answer, str]] = []
    for group in groups:
        members = set(group.blocks)
        answer = bpe.assemble_answer(edition, [i in members for i in range(len(edition.blocks))], None)
        clean = unicodedata.normalize("NFC", answer.text).strip()
        if clean:
            made.append((group, answer, clean))
    built = Built()
    for key, (group, answer, clean) in zip(unique_keys([g.key for g, _, _ in made]), made):
        pointer = {"source": "pgp_export_footnotes", "pgpid": pgpid, "parser": "parse_edition",
                   "block_indices": list(answer.blocks), "kinds": [edition.blocks[i].kind for i in answer.blocks]}
        built.sides.append(make_side(key, clean, answer.text, pointer, group.marker))
        notation = sorted({f for line in answer.flags for f in line})
        flags = edition_flags + group.flags + [f"notation_{f}" for f in notation]
        if any(edition.blocks[i].n_dropped for i in answer.blocks):
            flags.append("unrendered_lines")
        if any(walk.blocks[i].reversed_spans for i in answer.blocks):
            flags.append("supplied_reversed")
        if flags:
            built.flags[key] = flags
    if not built.sides:
        built.reasons.append("edition_no_lines")
    return built


def pgp_arabic_sides(text: str, pgpid: str) -> Built:
    """Sides of an Arabic-script PGP edition (:func:`build_pgp_arabic_editions.best_sections`, ``sides_of``).

    Reversed supplied spans are rewritten first (:func:`unreverse_supplied`); a side whose lines the rewrite changes
    is flagged ``supplied_reversed``.

    :param text: Edition text.
    :param pgpid: PGP id (pointer).
    :return: One side per side label (``""`` for text outside the labels, flagged ``text_outside_side_labels`` when
        labelled sides exist too); the marker is the side the label named (the splitter keeps no label line).
    """
    pairs = [unreverse_supplied(line) for line in text.splitlines()]
    rewritten = any(n for _, n in pairs)
    sections = pae.best_sections(["\n".join(line for line, _ in pairs) if rewritten else text])
    by_side = pae.sides_of(sections)
    unrewritten = pae.sides_of(pae.best_sections([text])) if rewritten else by_side
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
        flags = ["text_outside_side_labels"] if not side and labelled else []
        if unrewritten.get(side) != lines:
            flags.append("supplied_reversed")
        if flags:
            built.flags[side] = flags
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


def load_lexicon(path: Path, min_letters: int = LEXICON_MIN_LETTERS) -> FrozenSet[str]:
    """Known words for the sigla rules' word-order repairs, from a file of cleaned texts.

    The legacy-store build gave :func:`legacy_store_sigla.clean_section` the Hebrew letter runs of at least two
    letters of clean target texts and of library word annotations; the same rule over the cleaned legacy-store
    sections (their ``text``) gives a word list of the same kind from the data at hand.

    :param path: JSON lines with a ``text`` field (the cleaner's side records).
    :param min_letters: Shortest word kept.
    :return: The words.
    """
    words: Set[str] = set()
    for record in read_jsonl(path):
        words.update(w for w in HEBREW_RUN_RE.findall(str(record.get("text") or "")) if len(w) >= min_letters)
    return frozenset(words)


def scrape_sides(raw: str, side: Optional[str], canonical_id: Optional[str], notes: str,
                 lexicon: Optional[FrozenSet[str]] = None) -> Built:
    """Sides of a legacy-store site-scrape part through the sigla rules (:func:`legacy_store_sigla.clean_section`).

    Without a lexicon the rules' word-order repairs (a lone final letter stored before its word, ``ן שאי``; uncertain
    wraps stored in visual order, ``ש?ול?ם``) change nothing.

    :param raw: Raw scrape text.
    :param side: The record's side, if any (used when the rules find one side).
    :param canonical_id: Document id (pointer).
    :param notes: The record's notes (``entry=``, ``block=``, ``part=`` for the pointer).
    :param lexicon: Known words for the word-order repairs (:func:`load_lexicon`), or None.
    :return: One side per side the rules find; flags = the rules' flags.
    """
    cleaned = sigla.clean_section(raw, sigla.finals_mapping_for(raw), family=sigla.classify_family(raw),
                                  lexicon=lexicon)
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
    :param lexicon: Known words for the legacy-store scrape's word-order repairs, or None.
    """

    sections: Dict[Tuple[str, int], List[Dict[str, Any]]]
    sections_name: str
    blocks: Dict[Tuple[str, str], List[Dict[str, Any]]]
    run2: Dict[str, Tuple[Dict[str, Any], str]]
    datasets_root: Path
    manifests: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    lexicon: Optional[FrozenSet[str]] = None


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
        return scrape_sides(text, record.get("side") or None, canonical_id, str(record.get("notes") or ""),
                            inputs.lexicon)
    if source == "ktiv":
        return ktiv_sides(text, cand.ktiv_id, cand.row_id, str(record.get("notes") or ""),
                          str(record.get("edition_source") or ""))
    if source == "literature":
        resolution = cand.notes.get("resolution") or {}
        return literature_sides(str(cand.notes.get("edition_source") or ""), resolution.get("raw_keys") or [],
                                cand.notes.get("side"), text, inputs.blocks)
    return Built(reasons=["unknown_source"])


def attach_targets(cands: List[Candidate], *, candidates_dir: Path, sections_path: Path, lit_blocks_path: Path,
                   run2_path: Union[Path, Sequence[Path]], datasets_root: Path,
                   lexicon_path: Union[Path, str, None] = LEXICON_FROM_SECTIONS) -> None:
    """Fill ``cand.targets`` of every candidate (in place; earlier targets and target notes are replaced).

    Inputs are read only for the sources present. Pass-through sources also get their manifest image as
    ``cand.images``.

    :param cands: Candidates.
    :param candidates_dir: Directory of the candidate files (edition texts).
    :param sections_path: The legacy-store cleaner's side records.
    :param lit_blocks_path: The literature blocks.
    :param run2_path: The run-2 edition file, or several.
    :param datasets_root: Datasets root (pass-through sources).
    :param lexicon_path: Source of the legacy-store scrape's lexicon (:func:`load_lexicon`):
        :data:`LEXICON_FROM_SECTIONS` (default) for ``sections_path``, another file of cleaned texts, or None for no
        lexicon.
    """
    wanted: Dict[str, Set[str]] = defaultdict(set)
    for cand in cands:
        wanted[cand.source].add(cand.row_id)
    records = {source: source_records(candidates_dir, source, ids) for source, ids in wanted.items()}
    lexicon = None
    if "fjp_scrape" in wanted and lexicon_path is not None:
        lexicon = load_lexicon(sections_path if lexicon_path == LEXICON_FROM_SECTIONS else Path(lexicon_path))
    inputs = TargetInputs(sections=load_sections(sections_path) if "fjp" in wanted else {},
                          sections_name=sections_path.name,
                          blocks=load_lit_blocks(lit_blocks_path) if "literature" in wanted else {},
                          run2=load_run2(run2_path) if "lit_run2" in wanted else {}, datasets_root=datasets_root,
                          lexicon=lexicon)
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
    ap.add_argument("--lexicon", type=Path, default=None,
                    help="Cleaned texts (JSON lines with `text`) whose words drive the legacy-store scrape's "
                         "word-order repairs (default: the --sections file).")
    ap.add_argument("--no-lexicon", action="store_true", help="Clean the legacy-store scrape without a lexicon.")
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
    lexicon_path = None if args.no_lexicon else (args.lexicon or LEXICON_FROM_SECTIONS)
    attach_targets(selected, candidates_dir=args.candidates_dir, sections_path=args.sections,
                   lit_blocks_path=args.lit_blocks, run2_path=args.run2, datasets_root=args.datasets_root,
                   lexicon_path=lexicon_path)
    write_jsonl(stage, cands)
    counts = {source: {"candidates": n, "with_targets": sum(1 for c in selected if c.source == source and c.targets)}
              for source, n in sorted(taken.items())}
    print(json.dumps({"stage_file": str(stage), "re_attached": len(selected), "by_source": counts}, indent=1))


if __name__ == "__main__":
    main()
