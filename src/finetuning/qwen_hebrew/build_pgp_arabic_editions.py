# File name: build_pgp_arabic_editions.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Arabic-script documentary pages for training, from Princeton Geniza Project editions.

The fine-tuning mixtures hold Arabic *language* in Hebrew letters (Judaeo-Arabic) but next to no
Arabic *script*. This builder makes page-transcription rows from the PGP editions of documents
written in Arabic script that no benchmark holds out.

Documents: rows of ``merged/arabic_scrape_priority_queue.csv`` with a permitted image (tiers 1 to
3), a single fragment, a library IIIF manifest, and an edition that is at least 90 % Arabic
letters. Held out: every registered benchmark document (``benchmark_registry``, which includes the
Arabic-script benchmark) and every fragment of the Hebrew-script benchmarks
(``build_documentary_grounding.decontam_reason``).

Pages: a library image is used only when the record says which text is on it:

* the manifest has one image and the edition one side;
* or the image's label (``1r`` / ``1v``) names the side of an edition section (``Recto`` /
  ``Verso``) and exactly one image carries that label;
* or the edition has no side labels, PGP's ``side`` field names one side, and exactly one image
  carries that label.

Everything else is skipped with a reason. Images of the image store carry no side label, so
only library images are used. A page is also skipped when its text is too dense for the image
(more than 1,000 letters per megapixel: Cambridge serves 2,000 pixels on the long side, and a long
scroll at that size has letters a few pixels wide; such a row would teach the model to invent).
An image gets one row: when PGP has two records of the same text on it, the fuller edition is
kept; when two different documents share the side, the image is dropped (neither edition is the
whole page).

Side assignments. A document skipped for a *side* reason (an unlabelled edition and several
images, image labels that are missing, duplicated or do not name sides such as figgy's
``00000001.tif``, or text outside the side labels) can be written to ``--emit-unassigned`` (one
:func:`unassigned_record` per document) for a reader that looks at the images and decides which
text is on which image. Its decisions come back through ``--side-assignments`` (one object per
document: ``pgpid``, ``edition_fingerprint``, ``decision`` accept / drop, ``reader``, ``pages`` of
``{canvas_index, image_url, assign}`` with ``assign`` in ``all`` / ``recto`` / ``verso``, free-form
``evidence``) and are applied by :func:`apply_side_assignment` only where the labels say nothing:
a decision made for another edition text (:func:`edition_fingerprint`; a drop too) or other
images is stale and the document is offered again, a dropped document stays out, ``all`` places
only an edition without side labels, and an unlabelled edition is never split across images.
Every manifest row says how its side was found (``side_source``: ``single`` when the manifest
has one image, ``label`` from image labels, ``side_match:<reader>`` from an assignment).

Rows (KTIV ``FEATURES`` schema, ``label_source="pgp_edition"``): one ``fragment_transcribe`` row
per page, asking with :data:`ARABIC_FRAGMENT_PROMPT` for the visible text of that side, one line
per line, with ``[...]`` where the editor restored text or marked a loss
(``arabic_script.split_sections`` with a gap token). Splits: ``train_page`` and ``val``; a
document is in one split, chosen by a hash of its PGP id, and a document whose fragment is
already in another training set is never put in ``val``.

Inputs: the PGP export (``--pgp-dir``: edition text from ``<pgp-dir>/footnotes.csv``) and the
queue built from it (``--queue-csv``, ``arabic_scrape_queue.py --pgp-dir ... --out-dir ...``);
defaults: the export and queue under ``raw_data``. ``--pgp-dir`` needs ``--queue-csv``, and when
either side is a pinned export (``COMMIT``) the queue's summary must record the same commit
(:func:`check_queue_export`); ``stats.json`` records both.

Output (default ``pgp_arabic_editions_v2`` on the NAS; the v1 set and its image directories are
refused as outputs, as is any directory that contains one of them, in any spelling the
filesystem treats as the same: :func:`check_outputs`)::

    <out>/            saved DatasetDict + manifest.jsonl + stats.json
    <out>_images/     the page images (<pgpid>_<n>.jpg; ``--image-cache`` puts them elsewhere)

``--seed-cache-from IMAGES_DIR MANIFEST`` copies an image from an earlier build (its images
directory and ``manifest.jsonl``) when the image URL is identical instead of downloading it again.

Export it with ``images_once`` before adding it to a mixture.

Usage::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.build_pgp_arabic_editions [--out DIR] [--limit N]
        [--pgp-dir DIR] [--queue-csv FILE] [--image-cache DIR] [--seed-cache-from IMAGES_DIR MANIFEST]
        [--side-assignments FILE] [--emit-unassigned FILE]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image as PILImage

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.benchmark_registry import registered_benchmark_documents
from src.datasets.evaluations.helper_eval_scripts import build_arabic_benchmark as bab
from src.datasets.indexing import arabic_scrape_queue as asq
from src.finetuning.qwen_hebrew import build_documentary_grounding as bdg
from src.finetuning.qwen_hebrew.build_pgp_editions import save_dataset
from src.finetuning.qwen_hebrew.prompts import FRAGMENT_TRANSCRIBE_PROMPT

V1_OUT = bab.NAS_DATASETS / "pgp_arabic_editions_v1"
DEFAULT_OUT = bab.NAS_DATASETS / "pgp_arabic_editions_v2"
PROTECTED_OUTPUTS = (V1_OUT, V1_OUT.with_name(V1_OUT.name + "_images"), V1_OUT.with_name(V1_OUT.name + "_images_once"))
GAP = "[...]"
MIN_PAGE_LETTERS = 40
MIN_ARABIC_SHARE = 0.9
# The training notebook's per-row hygiene gate (derive_v24a data cell): an Arabic-script answer needs at least
# 40 letters in U+0620-064A and at least 9x as many of them as Hebrew letters. Applied per page here so a mixed
# page of a mainly-Arabic document never reaches the mixture.
NOTEBOOK_ARABIC_RE = re.compile(r"[ؠ-ي]")
NOTEBOOK_HEBREW_RE = re.compile(r"[א-ת]")
NOTEBOOK_MIN_ARABIC = 40
NOTEBOOK_ARABIC_OVER_HEBREW = 9
VAL_SHARE = 0.05
SAME_TEXT_F1 = 0.5              # two editions with this much letter 5-gram overlap are editions of one text
TASK = "fragment_transcribe"
SECTION = "pgp_arabic_page"
LABEL_SOURCE = "pgp_edition"
_HEBREW_SCRIPT_CLAUSE = "handwritten Hebrew script (the language may be Hebrew, Judeo-Arabic, or Aramaic)"
ARABIC_FRAGMENT_PROMPT = FRAGMENT_TRANSCRIBE_PROMPT.replace(_HEBREW_SCRIPT_CLAUSE, "handwritten Arabic script")
assert ARABIC_FRAGMENT_PROMPT != FRAGMENT_TRANSCRIBE_PROMPT, "the fragment prompt no longer names its script the way this builder expects"
ASSIGN_VALUES = ("all", "recto", "verso")
DECISIONS = ("accept", "drop")
STALE = "side assignment stale"
DROPPED = "side match dropped"
INVALID = "side assignment invalid"
_SIDE_PROBLEM = re.compile(r"edition without side labels and several images|edition has text outside its side labels"
                           r"|\d+ images labelled (recto|verso)(; \d+ images labelled (recto|verso))*")


def candidate_documents(queue_csv: Path, held_ids: Set[str], held_pgpids: Set[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """PGP documents of the Arabic queue that may become training pages.

    :param queue_csv: ``arabic_scrape_priority_queue.csv``.
    :type queue_csv: Path
    :param held_ids: Canonical ids held out by a registered benchmark.
    :type held_ids: Set[str]
    :param held_pgpids: PGP ids held out by a registered benchmark.
    :type held_pgpids: Set[str]
    :return: ``({pgpid: {canonical_id, shelfmark, library, tier, pgp_side, manifests}}, {pgpid: reason
        it was left out})`` in queue order.
    :rtype: Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]
    """
    docs: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}
    with open(queue_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pgpid = row["pgpid"]
            manifests = [u for u in re.split(r"[;|\s]+", (row["iiif_urls"] or row["suggested_iiif"] or "").strip()) if u.startswith("http")]
            if pgpid in held_pgpids or row["canonical_id"] in held_ids:
                skipped[pgpid] = "held out by a benchmark"
            elif row["tier"] not in ("1", "2", "3"):
                skipped[pgpid] = "no permitted image"
            elif row["single_fragment"] != "True":
                skipped[pgpid] = "join: the edition covers several fragments"
            elif not manifests:
                skipped[pgpid] = "no library manifest"
            else:
                docs[pgpid] = {"canonical_id": row["canonical_id"], "shelfmark": row["shelfmark"], "library": row["library"],
                               "tier": row["tier"], "pgp_side": row["side"], "manifests": manifests}
    return docs, skipped


def best_sections(editions: List[str]) -> List[Tuple[str, List[str]]]:
    """The fullest edition of a document, cut by side, with losses marked.

    :param editions: Edition texts of one PGP document.
    :type editions: List[str]
    :return: ``(side, lines)`` of the edition with the most visible Arabic letters (empty when
        there is no edition text).
    :rtype: List[Tuple[str, List[str]]]
    """
    best: Tuple[int, List[Tuple[str, List[str]]]] = (0, [])
    for content in editions:
        sections = ar.split_sections(content, GAP)
        letters = len(ar.arabic_letters("\n".join(line for _, lines in sections for line in lines)))
        if letters > best[0]:
            best = (letters, sections)
    return best[1]


def single_side(pgp_side: str) -> str:
    """The one side PGP's ``side`` field names, if it names exactly one.

    :param pgp_side: Field value such as ``"verso"``, ``"recto and verso"`` or ``"verso ; verso"``.
    :type pgp_side: str
    :return: ``"recto"``, ``"verso"`` or ``""``.
    :rtype: str
    """
    named = {side for side in ("recto", "verso") if side in pgp_side.lower()}
    return named.pop() if len(named) == 1 else ""


def sides_of(sections: List[Tuple[str, List[str]]]) -> Dict[str, List[str]]:
    """Edition lines by side, a side cut in several sections joined in edition order.

    :param sections: ``(side, lines)`` of the edition, ``side`` in ``{"", "recto", "verso"}``.
    :type sections: List[Tuple[str, List[str]]]
    :return: ``{side: lines}`` in order of first appearance (``""``: text outside the side labels).
    :rtype: Dict[str, List[str]]
    """
    by_side: Dict[str, List[str]] = {}
    for side, lines in sections:
        by_side.setdefault(side, []).extend(lines)
    return by_side


def assign_pages(sections: List[Tuple[str, List[str]]], canvas_sides: List[str], pgp_side: str = "") -> Tuple[Dict[int, List[str]], str]:
    """Decide which image holds which lines of an edition, only where the record says so.

    :param sections: ``(side, lines)`` of the edition, ``side`` in ``{"", "recto", "verso"}``.
    :type sections: List[Tuple[str, List[str]]]
    :param canvas_sides: Side named by each image's label (``""`` when the label does not say).
    :type canvas_sides: List[str]
    :param pgp_side: PGP's ``side`` field of the document.
    :type pgp_side: str
    :return: ``({image index: lines}, why nothing or not everything was assigned)``.
    :rtype: Tuple[Dict[int, List[str]], str]
    """
    by_side = sides_of(sections)
    labelled = [side for side in by_side if side]
    if not by_side or not canvas_sides:
        return {}, "no edition text" if not by_side else "no image"
    if "" in by_side and labelled:
        return {}, "edition has text outside its side labels"
    if not labelled:
        side = single_side(pgp_side)
        if len(canvas_sides) == 1:
            return {0: by_side[""]}, ""
        if side and canvas_sides.count(side) == 1:
            return {canvas_sides.index(side): by_side[""]}, ""
        return {}, "edition without side labels and several images"
    if len(labelled) == 1 and len(canvas_sides) == 1 and canvas_sides[0] == "":
        return {0: by_side[labelled[0]]}, ""
    pages = {canvas_sides.index(side): by_side[side] for side in labelled if canvas_sides.count(side) == 1}
    missing = [side for side in labelled if canvas_sides.count(side) != 1]
    return pages, "; ".join(f"{canvas_sides.count(side)} images labelled {side}" for side in missing)


def is_side_problem(problem: str) -> bool:
    """Whether :func:`assign_pages` failed because the record does not say which image holds which text.

    :param problem: Second value returned by :func:`assign_pages`.
    :type problem: str
    :return: True for an unlabelled edition with several images, image labels missing or repeated
        for the edition's sides, and text outside the side labels; False for no text or no image.
    :rtype: bool
    """
    return bool(_SIDE_PROBLEM.fullmatch(problem))


def edition_fingerprint(sides: Dict[str, List[str]]) -> str:
    """Identity of the edition text the builder pairs with images.

    The text hashed is the cleaned edition as :func:`best_sections` returns it (PGP edition content
    from ``footnotes.csv``, the fullest of a document's editions, reduced by
    ``arabic_script.split_sections`` with the ``[...]`` gap token), grouped by :func:`sides_of`: for
    each side in order a header line ``@<side>`` (``@`` alone for text outside the side labels)
    followed by its lines, everything joined by newlines and encoded as UTF-8. It changes when the
    export's text, the cleaning or the choice among editions changes. It can be recomputed from an
    :func:`unassigned_record` as ``edition_fingerprint(record["sections"])``.

    :param sides: ``{side: lines}`` from :func:`sides_of`.
    :type sides: Dict[str, List[str]]
    :return: sha1 hex digest.
    :rtype: str
    """
    text = "\n".join("\n".join([f"@{side}"] + list(lines)) for side, lines in sides.items())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def unassigned_record(pgpid: str, doc: Dict[str, Any], sides: Dict[str, List[str]], canvases: List[Dict[str, Any]],
                      reason: str) -> Dict[str, Any]:
    """One line of ``unassigned.jsonl``: a document whose text the record cannot place on its images.

    :param pgpid: PGP document id.
    :type pgpid: str
    :param doc: Entry of :func:`candidate_documents`.
    :type doc: Dict[str, Any]
    :param sides: ``{side: lines}`` of the edition used (:func:`sides_of`).
    :type sides: Dict[str, List[str]]
    :param canvases: Images of the document (``bab.planned_images``: ``url``, ``label``).
    :type canvases: List[Dict[str, Any]]
    :param reason: Why the builder could not pair text and images.
    :type reason: str
    :return: ``{pgpid, shelfmark, canonical_id, library, edition_fingerprint, sections: {side: lines},
        canvases: [{index, label, image_url}], reason}``.
    :rtype: Dict[str, Any]
    """
    return {"pgpid": pgpid, "shelfmark": doc["shelfmark"], "canonical_id": doc["canonical_id"], "library": doc["library"],
            "edition_fingerprint": edition_fingerprint(sides), "sections": {side: list(lines) for side, lines in sides.items()},
            "canvases": [{"index": n, "label": canvas["label"], "image_url": canvas["url"]} for n, canvas in enumerate(canvases)],
            "reason": reason}


def load_side_assignments(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read a side matcher's ``side_assignments.jsonl``.

    :param path: One JSON object per line: ``pgpid``, ``edition_fingerprint``, ``decision``
        (``accept`` / ``drop``), ``reader``, ``pages`` (``[{canvas_index, image_url, assign}]``),
        ``evidence``.
    :type path: Path
    :return: ``{pgpid: assignment}``; a later line for the same document replaces an earlier one.
    :rtype: Dict[str, Dict[str, Any]]
    :raises ValueError: when a line lacks a field, names an unknown decision or has malformed pages.
    """
    assignments: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [key for key in ("pgpid", "edition_fingerprint", "decision", "reader", "pages") if key not in record]
            if missing:
                raise ValueError(f"{path}:{number}: missing {', '.join(missing)}")
            if record["decision"] not in DECISIONS:
                raise ValueError(f"{path}:{number}: decision {record['decision']!r} is not one of {DECISIONS}")
            if not isinstance(record["pages"], list) or not all(
                    isinstance(page, dict) and {"canvas_index", "image_url", "assign"} <= set(page) for page in record["pages"]):
                raise ValueError(f"{path}:{number}: pages must be a list of {{canvas_index, image_url, assign}}")
            assignments[str(record["pgpid"])] = record
    return assignments


def apply_side_assignment(sides: Dict[str, List[str]], canvases: List[Dict[str, Any]],
                          assignment: Dict[str, Any]) -> Tuple[Dict[int, List[str]], str]:
    """Pages of a document from a side matcher's decision, checked against the current record.

    * a decision made for another edition text (fingerprint): no page (``side assignment stale``),
      whatever the decision, so that a document dropped under an earlier text is offered again.
    * ``drop``: no page (``side match dropped``).
    * other images at the assigned indices: no page (``side assignment stale (images changed)``).
    * ``all``: the whole edition on one image; only for an edition without side labels (text
      under a side label is placed with ``recto`` / ``verso``, where the image labels are checked;
      unlabelled text next to a labelled side is an implicit other side, and one photograph of a
      fragment does not show both), and it must be the only page (an unlabelled edition is never
      split across images).
    * ``recto`` / ``verso``: that side's lines on that image; every line of the edition must be
      under a side label, a side goes to one image only, and an image whose label names a side
      may only receive that side. A side of the edition left unassigned gets no page (reported).

    :param sides: ``{side: lines}`` of the current edition (:func:`sides_of`).
    :type sides: Dict[str, List[str]]
    :param canvases: Current images of the document (``bab.planned_images``: ``url``, ``side``).
    :type canvases: List[Dict[str, Any]]
    :param assignment: One entry of :func:`load_side_assignments`.
    :type assignment: Dict[str, Any]
    :return: ``({image index: lines}, problem)``; the problem says why there is no page, or which
        side got none, and is ``""`` otherwise.
    :rtype: Tuple[Dict[int, List[str]], str]
    """
    if assignment["edition_fingerprint"] != edition_fingerprint(sides):
        return {}, STALE
    if assignment["decision"] == "drop":
        return {}, DROPPED
    pages = assignment["pages"]
    if not pages:
        return {}, f"{INVALID}: no page"
    if any(not isinstance(page["canvas_index"], int) or not 0 <= page["canvas_index"] < len(canvases)
           or canvases[page["canvas_index"]]["url"] != page["image_url"] for page in pages):
        return {}, f"{STALE} (images changed)"
    indices = [page["canvas_index"] for page in pages]
    values = [page["assign"] for page in pages]
    if len(set(indices)) != len(indices):
        return {}, f"{INVALID}: an image is assigned twice"
    if any(value not in ASSIGN_VALUES for value in values):
        return {}, f"{INVALID}: assign must be one of {', '.join(ASSIGN_VALUES)}"
    if "all" in values:
        if len(pages) != 1:
            return {}, f"{INVALID}: 'all' must be the only page"
        if set(sides) != {""}:
            return {}, f"{INVALID}: 'all' is only for an edition without side labels"
        return {indices[0]: sides[""]}, ""
    if "" in sides:
        return {}, f"{INVALID}: text outside the side labels can only be assigned with 'all'"
    if len(set(values)) != len(values):
        return {}, f"{INVALID}: a side is assigned to several images"
    if any(value not in sides for value in values):
        return {}, f"{INVALID}: the edition has no {', '.join(value for value in values if value not in sides)}"
    if any(canvases[index]["side"] not in ("", value) for index, value in zip(indices, values)):
        return {}, f"{INVALID}: an image label names the other side"
    unassigned = [side for side in sides if side not in values]
    return {index: sides[value] for index, value in zip(indices, values)}, "; ".join(f"side match left {side} unassigned"
                                                                                       for side in unassigned)


def seed_cache_index(images_dir: Path, manifest: Path) -> Dict[str, Path]:
    """Images an earlier build downloaded, by URL.

    :param images_dir: That build's image directory.
    :type images_dir: Path
    :param manifest: That build's ``manifest.jsonl`` (``image_url``, ``image_path``); the file is
        looked up by name in ``images_dir``, so a moved directory still works.
    :type manifest: Path
    :return: ``{image_url: file}`` for every image still on disk and not empty.
    :rtype: Dict[str, Path]
    """
    index: Dict[str, Path] = {}
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            path = images_dir / Path(row["image_path"]).name
            if row.get("image_url") and path.is_file() and path.stat().st_size > 0:
                index.setdefault(row["image_url"], path)
    return index


def fetch_image(url: str, dst: Path, seed: Optional[Dict[str, Path]] = None) -> str:
    """Put the image at ``url`` in ``dst``.

    :param url: Image URL.
    :type url: str
    :param dst: Destination file.
    :type dst: Path
    :param seed: :func:`seed_cache_index` of an earlier build (read only, never written).
    :type seed: Optional[Dict[str, Path]]
    :return: ``"cache"`` (``dst`` was already there), ``"seed"`` (copied from the earlier build,
        same URL) or ``"download"``.
    :rtype: str
    :raises OSError: when the download is refused or fails.
    """
    if dst.exists() and dst.stat().st_size > 0:
        return "cache"
    if seed and url in seed:
        tmp = dst.with_suffix(".part")
        shutil.copyfile(seed[url], tmp)
        os.replace(tmp, dst)
        return "seed"
    bab.save_image(url, dst)
    return "download"


def path_key(path: Path) -> Tuple[str, ...]:
    """Components of a resolved path as the filesystems here compare them.

    APFS and the NAS share (SMB) ignore case and Unicode normalisation, so
    ``PGP_Arabic_Editions_v1`` is the directory ``pgp_arabic_editions_v1``.

    :param path: Any path (it need not exist).
    :type path: Path
    :return: NFC-normalised, case-folded components of ``path.resolve()``.
    :rtype: Tuple[str, ...]
    """
    return tuple(unicodedata.normalize("NFC", part).casefold() for part in Path(path).resolve().parts)


def file_identities(path: Path) -> List[Tuple[int, int]]:
    """``(st_dev, st_ino)`` of a path and of each of its ancestors that exists, nearest first.

    :param path: A resolved path.
    :type path: Path
    :return: Identities; the first is the path's own when it exists.
    :rtype: List[Tuple[int, int]]
    """
    return [(st.st_dev, st.st_ino) for st in (p.stat() for p in (path, *path.parents) if p.exists())]


def check_outputs(paths: Iterable[Optional[Path]], protected: Optional[Iterable[Path]] = None) -> None:
    """Refuse to write into, or over, an earlier, published build.

    An output may not be a protected directory, lie inside one, or contain one (``save_dataset``
    replaces its output directory, so an ancestor of v1 would take v1 with it). Paths are compared
    after resolving, case folding and Unicode normalisation (:func:`path_key`), and paths that
    exist are also compared by file identity, so a spelling the filesystem treats as the same
    directory is refused too.

    :param paths: Output files and directories of this run (``None`` entries are ignored).
    :type paths: Iterable[Optional[Path]]
    :param protected: Directories that must never be written; default :data:`PROTECTED_OUTPUTS`
        (the v1 set, its image cache and its images-once export), read when called.
    :type protected: Optional[Iterable[Path]]
    :raises ValueError: when an output is a protected directory, inside one or an ancestor of one.
    """
    guarded = [(directory, path_key(directory), file_identities(directory), directory.exists())
               for directory in (Path(p).resolve() for p in (PROTECTED_OUTPUTS if protected is None else protected))]
    for path in paths:
        if path is None:
            continue
        resolved = Path(path).resolve()
        key, chain = path_key(resolved), file_identities(resolved)
        own = chain[0] if resolved.exists() else None
        for directory, guarded_key, guarded_chain, exists in guarded:
            shared = min(len(key), len(guarded_key))
            if (key[:shared] == guarded_key[:shared]                 # same spelling: one is the other or inside it
                    or (exists and guarded_chain[0] in chain)        # same file: the output is it or inside it
                    or (own is not None and own in guarded_chain)):  # same file: the output is it or contains it
                raise ValueError(f"refusing to write {path}: {directory} is a published build and must not be overwritten")


def check_queue_export(queue_csv: Path, pgp_dir: Path) -> Dict[str, str]:
    """Check that the queue and the edition text come from the same PGP export.

    The queue fixes the candidate documents, their PGP ``side`` field, canonical ids and IIIF URLs;
    the edition text comes from ``<pgp_dir>/footnotes.csv``. When either records a pinned export
    (the snapshot's ``COMMIT`` file; the queue's summary, :func:`asq.queue_provenance`), both must
    record the same one.

    :param queue_csv: Arabic queue CSV.
    :type queue_csv: Path
    :param pgp_dir: PGP export directory of the edition text.
    :type pgp_dir: Path
    :return: ``{pgp_export_commit, queue_pgp_export_commit, queue_pgp_dir}`` for ``stats.json``.
    :rtype: Dict[str, str]
    :raises ValueError: when the two exports differ, or only one of them is pinned.
    """
    commit, queue = asq.pgp_export_commit(pgp_dir), asq.queue_provenance(queue_csv)
    if commit != queue["pgp_export_commit"]:
        raise ValueError(f"the queue {queue_csv} was built from PGP export {queue['pgp_export_commit'] or '(unrecorded)'} "
                         f"({queue['pgp_dir'] or 'no summary'}), the edition text comes from {commit or '(unpinned)'} ({pgp_dir}): "
                         f"rebuild the queue with arabic_scrape_queue --pgp-dir {pgp_dir} --out-dir DIR and pass --queue-csv")
    return {"pgp_export_commit": commit, "queue_pgp_export_commit": queue["pgp_export_commit"], "queue_pgp_dir": queue["pgp_dir"]}


def resolve_shared_images(pages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter]:
    """Keep at most one page per image.

    :param pages: Planned pages; each has ``url`` (the image) and ``text`` (its target).
    :type pages: List[Dict[str, Any]]
    :return: ``(pages kept, in their original order; counts of pages dropped by reason)``. Several
        editions of one text on an image: the one with the most Arabic letters stays. Different
        texts on one image: none stays.
    :rtype: Tuple[List[Dict[str, Any]], Counter]
    """
    by_url: Dict[str, List[Dict[str, Any]]] = {}
    for page in pages:
        by_url.setdefault(page["url"], []).append(page)
    dropped: Counter = Counter()
    keep = set()
    for group in by_url.values():
        letters = [ar.arabic_letters(page["text"]) for page in group]
        fullest = max(range(len(group)), key=lambda i: len(letters[i]))
        if all(ar.ngram_overlap(letters[i], letters[fullest])["f1"] >= SAME_TEXT_F1 for i in range(len(group)) if i != fullest):
            keep.add(id(group[fullest]))
            dropped["another edition of the same page is fuller"] += len(group) - 1
        else:
            dropped["several documents on one image"] += len(group)
    return [page for page in pages if id(page) in keep], dropped


def split_of(pgpid: str, val_share: float) -> str:
    """Deterministic split of a document.

    :param pgpid: PGP document id.
    :type pgpid: str
    :param val_share: Share of documents that go to validation.
    :type val_share: float
    :return: ``"val"`` or ``"train_page"``.
    :rtype: str
    """
    bucket = int(hashlib.sha1(pgpid.encode("utf-8")).hexdigest(), 16) % 10_000
    return "val" if bucket < round(val_share * 10_000) else "train_page"


def mainly_arabic_script(text: str) -> bool:
    """Whether a page target passes the training notebook's Arabic-script hygiene gate.

    :param text: Page target text (the row's answer).
    :type text: str
    :return: True when the text has at least :data:`NOTEBOOK_MIN_ARABIC` Arabic letters and at least
        :data:`NOTEBOOK_ARABIC_OVER_HEBREW` times as many Arabic as Hebrew letters.
    :rtype: bool
    """
    n_ar, n_he = len(NOTEBOOK_ARABIC_RE.findall(text)), len(NOTEBOOK_HEBREW_RE.findall(text))
    return n_ar >= NOTEBOOK_MIN_ARABIC and n_ar >= NOTEBOOK_ARABIC_OVER_HEBREW * n_he


def page_row(image_path: Path, text: str, stem: str, width: int, height: int) -> Dict[str, Any]:
    """One page-transcription row in the KTIV ``FEATURES`` schema.

    :param image_path: Page image on disk.
    :type image_path: Path
    :param text: Target: the visible text of the page, one line per line.
    :type text: str
    :param stem: Row id.
    :type stem: str
    :param width: Image width in pixels.
    :type width: int
    :param height: Image height in pixels.
    :type height: int
    :return: Feature dict.
    :rtype: Dict[str, Any]
    """
    return {"image": str(image_path), "question": ARABIC_FRAGMENT_PROMPT, "answer": text, "task": TASK, "section": SECTION,
            "stem": stem, "label_source": LABEL_SOURCE, "target_chars": len(text), "target_tokens": 0,
            "image_width": width, "image_height": height}


def build(out: Path, limit: Optional[int] = None, delay_s: float = 0.5, max_side: int = 4000, max_canvases: int = 8,
          val_share: float = VAL_SHARE, held_out: Optional[Tuple[Set[str], Set[str]]] = None,
          benchmark_index: Optional[bdg.BenchmarkIndex] = None, in_training: Optional[Dict[str, str]] = None,
          queue_csv: Optional[Path] = None, footnotes_csv: Optional[Path] = None, image_cache: Optional[Path] = None,
          seed_cache: Optional[Dict[str, Path]] = None, side_assignments: Optional[Dict[str, Dict[str, Any]]] = None,
          emit_unassigned: Optional[Path] = None) -> Dict[str, Any]:
    """Build the dataset directory.

    :param out: Output directory of the saved DatasetDict.
    :type out: Path
    :param limit: Only the first N candidate documents (dry run).
    :type limit: Optional[int]
    :param delay_s: Pause between requests to library servers.
    :type delay_s: float
    :param max_side: Long-side cap of the requested images.
    :type max_side: int
    :param max_canvases: A manifest with more images is a whole volume and is skipped.
    :type max_canvases: int
    :param val_share: Share of documents for validation.
    :type val_share: float
    :param held_out: ``(canonical ids, PGP ids)`` held out by benchmarks; default: the registry.
    :type held_out: Optional[Tuple[Set[str], Set[str]]]
    :param benchmark_index: Hebrew-script benchmark index; default: the real one.
    :type benchmark_index: Optional[bdg.BenchmarkIndex]
    :param in_training: ``build_arabic_benchmark.training_documents`` result; default: the real sets.
    :type in_training: Optional[Dict[str, str]]
    :param queue_csv: Arabic queue (``arabic_scrape_queue.py``); default ``bab.QUEUE_CSV``.
    :type queue_csv: Optional[Path]
    :param footnotes_csv: PGP export ``footnotes.csv`` with the edition text; default ``bab.FOOTNOTES_CSV``.
    :type footnotes_csv: Optional[Path]
    :param image_cache: Directory of the page images; default ``<out>_images``.
    :type image_cache: Optional[Path]
    :param seed_cache: :func:`seed_cache_index` of an earlier build: images with an identical URL
        are copied from it instead of downloaded.
    :type seed_cache: Optional[Dict[str, Path]]
    :param side_assignments: :func:`load_side_assignments` result, used for documents whose
        records do not say which image holds which text.
    :type side_assignments: Optional[Dict[str, Dict[str, Any]]]
    :param emit_unassigned: Where to write ``unassigned.jsonl`` (:func:`unassigned_record` of every
        document left out for a side reason and without a usable assignment); not written when None.
    :type emit_unassigned: Optional[Path]
    :return: The statistics (also written to ``stats.json``).
    :rtype: Dict[str, Any]
    :raises ValueError: when an output would land in or over the v1 build (:func:`check_outputs`), or
        the queue was built from another PGP export than the edition text (:func:`check_queue_export`);
        both are checked before anything is read from the network or written.
    """
    queue_csv = queue_csv if queue_csv is not None else bab.QUEUE_CSV
    footnotes_csv = footnotes_csv if footnotes_csv is not None else bab.FOOTNOTES_CSV
    images_dir = image_cache if image_cache is not None else out.with_name(out.name + "_images")
    check_outputs([out, images_dir, emit_unassigned])
    exports = check_queue_export(queue_csv, footnotes_csv.parent)
    assignments = side_assignments or {}
    held_ids, held_pgpids = held_out if held_out is not None else registered_benchmark_documents()
    index = benchmark_index if benchmark_index is not None else bdg.load_benchmark_index()
    trained = in_training if in_training is not None else bab.training_documents(bab.TRAINING_MANIFESTS, bab.TRAINING_ID_LISTS)
    docs, skipped = candidate_documents(queue_csv, held_ids, held_pgpids)
    if limit:
        docs = dict(list(docs.items())[:limit])
    editions = bab.load_editions(footnotes_csv, docs)
    images_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, List[Dict[str, Any]]] = {"train_page": [], "val": []}
    manifest: List[Dict[str, Any]] = []
    planned: List[Dict[str, Any]] = []
    unassigned: List[Dict[str, Any]] = []
    page_skips: Counter = Counter()
    assignment_outcomes: Counter = Counter()
    image_sources: Counter = Counter()
    for pgpid, doc in docs.items():
        reason = bdg.decontam_reason(doc["canonical_id"] or doc["shelfmark"], (), bdg.DocRefs(shelfmarks={doc["shelfmark"]}), index)
        if reason:
            skipped[pgpid] = f"fragment of a benchmark ({reason})"
            continue
        sections = best_sections(editions.get(pgpid, []))
        text = "\n".join(line for _, lines in sections for line in lines)
        arabic, hebrew, share = ar.script_share(text)
        if arabic < MIN_PAGE_LETTERS:
            skipped[pgpid] = f"fewer than {MIN_PAGE_LETTERS} visible Arabic letters"
            continue
        if share < MIN_ARABIC_SHARE:
            skipped[pgpid] = "mixed script"
            continue
        canvases, note = bab.planned_images(pgpid, doc, [], True, max_side, max_canvases, delay_s)
        pages, problem = assign_pages(sections, [c["side"] for c in canvases], doc["pgp_side"])
        side_source = "single" if len(canvases) == 1 else "label"
        assigned_sides: Dict[int, str] = {}
        if not pages and canvases and is_side_problem(problem):          # the record does not say which image: a side matcher may
            sides = sides_of(sections)
            assignment = assignments.get(pgpid)
            outcome = ""
            if assignment is not None:
                pages, outcome = apply_side_assignment(sides, canvases, assignment)
                assignment_outcomes["accepted" if pages else "dropped" if outcome == DROPPED
                                    else "stale" if outcome.startswith(STALE) else "invalid"] += 1
            if emit_unassigned is not None and not note and (assignment is None or outcome.startswith(STALE)):
                unassigned.append(unassigned_record(pgpid, doc, sides, canvases, f"{problem} ({outcome})" if outcome else problem))
            if assignment is not None:
                if not pages:
                    skipped[pgpid] = outcome
                    continue
                side_source = f"side_match:{assignment['reader']}"
                assigned_sides = {page["canvas_index"]: page["assign"] for page in assignment["pages"]
                                  if page["assign"] in ("recto", "verso")}
                problem = outcome
        if not pages:
            skipped[pgpid] = re.sub(r"https?://\S+: ", "", note) or problem
            continue
        if problem:
            page_skips[problem] += 1
        split = "train_page" if bab.training_set_of(doc["canonical_id"], [pgpid], trained) else split_of(pgpid, val_share)
        for canvas_index, lines in sorted(pages.items()):
            page_text = "\n".join(lines)
            letters = len(ar.arabic_letters(page_text))
            if letters < MIN_PAGE_LETTERS:
                page_skips[f"page under {MIN_PAGE_LETTERS} Arabic letters"] += 1
                continue
            if not mainly_arabic_script(page_text):
                page_skips["page not mainly Arabic script (notebook gate)"] += 1
                continue
            planned.append({"pgpid": pgpid, "doc": doc, "split": split, "image_index": canvas_index, "text": page_text,
                            "lines": len(lines), "letters": letters, "url": canvases[canvas_index]["url"],
                            "label": canvases[canvas_index]["label"],
                            "side": assigned_sides.get(canvas_index, canvases[canvas_index]["side"]), "side_source": side_source})
    planned, shared = resolve_shared_images(planned)
    page_skips.update(shared)
    for page in planned:
        pgpid, doc = page["pgpid"], page["doc"]
        path = images_dir / f"{pgpid}_{page['image_index']}.jpg"
        try:
            image_sources[fetch_image(page["url"], path, seed_cache)] += 1
        except OSError:                                                  # refused or missing at the library
            page_skips["image download failed"] += 1
            continue
        with PILImage.open(path) as image:
            width, height = image.size
        if page["letters"] / (width * height / 1e6) > bab.MAX_LETTERS_PER_MEGAPIXEL:
            page_skips["text too dense for the image"] += 1
            continue
        stem = f"pgpar_{pgpid}_{page['image_index']}_page"
        splits[page["split"]].append(page_row(path, page["text"], stem, width, height))
        manifest.append({"stem": stem, "pgpid": pgpid, "canonical_id": doc["canonical_id"], "shelfmark": doc["shelfmark"],
                         "tier": doc["tier"], "split": page["split"], "image_index": page["image_index"], "label": page["label"],
                         "side": page["side"], "side_source": page["side_source"], "image_url": page["url"], "image_path": str(path),
                         "lines": page["lines"], "arabic_letters": page["letters"]})
    stats = {"candidate_documents": len(docs), "documents": len({m["pgpid"] for m in manifest}), "pages": len(manifest),
             "rows": {name: len(rows) for name, rows in splits.items()},
             "arabic_letters": sum(m["arabic_letters"] for m in manifest),
             "documents_already_in_another_training_set": len({m["pgpid"] for m in manifest
                                                               if bab.training_set_of(m["canonical_id"], [m["pgpid"]], trained)}),
             "by_tier": dict(Counter(m["tier"] for m in manifest)), "skipped": dict(Counter(skipped.values())),
             "page_skips": dict(page_skips),
             "side_sources": dict(Counter("side_match" if m["side_source"].startswith("side_match:") else m["side_source"]
                                          for m in manifest)),
             "side_assignments": {"loaded": len(assignments), **{key: assignment_outcomes[key]
                                                                 for key in ("accepted", "dropped", "stale", "invalid")},
                                  "unused": len(assignments) - sum(assignment_outcomes.values())},
             "unassigned_emitted": len(unassigned) if emit_unassigned is not None else None,
             "images": dict(image_sources),
             "inputs": {"queue_csv": str(queue_csv), "footnotes_csv": str(footnotes_csv), **exports, "image_cache": str(images_dir)},
             "skipped_documents": skipped, "prompt": ARABIC_FRAGMENT_PROMPT}
    save_dataset(splits, out, {"manifest.jsonl": "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
                               "stats.json": json.dumps(stats, ensure_ascii=False, indent=1)})
    if emit_unassigned is not None:
        emit_unassigned.parent.mkdir(parents=True, exist_ok=True)
        emit_unassigned.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in unassigned), encoding="utf-8")
    return stats


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface.

    :return: Parser; ``--pgp-dir``, ``--queue-csv`` and ``--image-cache`` default to ``None``,
        meaning the export and queue under ``raw_data`` and ``<out>_images``.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"dataset directory (default: {DEFAULT_OUT}; v1 is refused)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N candidate documents")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between requests to library servers")
    parser.add_argument("--pgp-dir", type=Path, default=None,
                        help=f"PGP export directory; edition text from <pgp-dir>/footnotes.csv (default: {bab.FOOTNOTES_CSV.parent}); "
                             f"needs --queue-csv")
    parser.add_argument("--queue-csv", type=Path, default=None,
                        help=f"Arabic queue built from that export by arabic_scrape_queue.py (default: {bab.QUEUE_CSV}); "
                             f"its summary must record the same export commit")
    parser.add_argument("--image-cache", type=Path, default=None, help="directory of the page images (default: <out>_images)")
    parser.add_argument("--seed-cache-from", type=Path, nargs=2, metavar=("IMAGES_DIR", "MANIFEST"), default=None,
                        help="an earlier build's images directory and manifest.jsonl: images with an identical URL are copied, "
                             "not downloaded (the earlier build is only read)")
    parser.add_argument("--side-assignments", type=Path, default=None,
                        help="side_assignments.jsonl of a side matcher, for documents whose labels do not place the text")
    parser.add_argument("--emit-unassigned", type=Path, default=None,
                        help="write unassigned.jsonl: documents left out for a side reason and without a usable assignment")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pgp_dir is not None and args.queue_csv is None:
        parser.error(f"--pgp-dir needs --queue-csv: the default queue ({bab.QUEUE_CSV}) was built from another export; "
                     f"build one with arabic_scrape_queue --pgp-dir {args.pgp_dir} --out-dir DIR")
    seed = seed_cache_index(*args.seed_cache_from) if args.seed_cache_from else None
    stats = build(args.out, limit=args.limit, delay_s=args.delay, queue_csv=args.queue_csv,
                  footnotes_csv=args.pgp_dir / "footnotes.csv" if args.pgp_dir is not None else None,
                  image_cache=args.image_cache, seed_cache=seed,
                  side_assignments=load_side_assignments(args.side_assignments) if args.side_assignments else None,
                  emit_unassigned=args.emit_unassigned)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("skipped_documents", "prompt")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
