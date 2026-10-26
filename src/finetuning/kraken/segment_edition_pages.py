# File name: segment_edition_pages.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Segment and read the Genizah Arabic edition pages with Kraken, keeping every line's geometry.

Input: the PGP Arabic-edition manifest (``build_pgp_arabic_editions``), one row per page image with
``stem, pgpid, canonical_id, image_path, lines`` (the edition's line count) among other fields.
For each selected page this runs one segmenter and one recogniser (the options of
``run_arabic_benchmark_kraken``) and writes ``<out-dir>/<tag>/<stem>.json``::

    {"stem", "pgpid", "canonical_id", "image_path", "image_size": [w, h], "expected_lines",
     "n_lines", "n_dropped", "secs", "source",
     "lines": [{"index", "baseline": [[x, y], ...], "boundary": [[x, y], ...], "text", "mean_confidence"}]}

``image_path`` is the manifest's path (the host path, whatever the image was read from).
``expected_lines`` is the manifest's ``lines``. ``n_lines`` counts every line the segmenter found
and ``n_dropped`` the lines without a usable boundary polygon, which are not in ``lines``.
``lines`` keeps the segmenter's order, including lines read as empty text. Coordinates are pixels of
the full-resolution image file. ``mean_confidence`` is null for a line with no characters.

After each page a row ``{stem, expected_lines, n_lines, n_dropped, ratio, secs}`` is appended to
``<out-dir>/<tag>/summary.jsonl``, where ``ratio = n_lines / expected_lines``. At the end of every
run the summary is rebuilt from the page JSONs in the run directory (one row per page JSON, in
manifest order), so a page whose summary row was lost or duplicated by an interrupted run is listed
exactly once. The outputs feed the forced-alignment step (edition lines onto Kraken lines) and the
segmenter comparison (one tag per segmenter).

The run is resumable: a page whose JSON exists is skipped. A JSON is written to a temporary file
and renamed, so an interrupted page is redone. A page that raises is logged, gets no JSON and is
retried on the next run. Pages whose image is missing are listed and skipped.

Inside the ``kraken-train:k711`` image this module imports the reader helpers from
``src/datasets/evaluations/helper_eval_scripts/run_arabic_benchmark_kraken.py``, so the repository
must be mounted too (read-only is enough), e.g. at ``/repo``::

    cd /repo && python -m src.finetuning.kraken.segment_edition_pages \\
        --manifest /data/pgp_arabic_editions_v2/manifest.jsonl --image-root /data/pgp_arabic_editions_v2_images \\
        --out-dir /out --tag blla2026_ppocrv6m --segmenter /segmodels/blla_2026/blla.mlmodel \\
        --recogniser /ar/ppocrv6_medium.safetensors [--ids stems.txt] [--limit 5]

``--image-root`` replaces the directory of every manifest ``image_path`` (the images sit flat in
one directory). Without it the manifest paths are used as they are, which works on the host.

Re-read mode: ``--geometry-from DIR`` (an earlier run directory, ``<out-dir>/<tag>``) skips the
segmenter. For each selected page the lines of ``DIR/<stem>.json`` (baseline and boundary, in that
order) are handed to the recogniser alone, and ``<out-dir>/<tag>/<stem>.json`` gets the same shape
with new ``text`` / ``mean_confidence``, the same geometry and indices, ``n_lines`` / ``n_dropped``
copied from the source JSON, and ``source`` = ``geometry from <DIR name> + <recogniser>, rec <input>,
kraken <version>``. ``--segmenter`` is then optional (and ignored). Pages without a source JSON are
listed and skipped; a page whose image size differs from the source JSON's ``image_size`` fails and
is retried on the next run. Resuming and the summary rebuild work as above::

    cd /repo && python -m src.finetuning.kraken.segment_edition_pages \\
        --manifest /data/pgp_arabic_editions_v2/manifest.jsonl --image-root /data/pgp_arabic_editions_v2_images \\
        --out-dir /out --tag blla2026_arfinetuned --geometry-from /out/blla2026_ppocrv6m \\
        --recogniser /ar/arabic_finetuned.safetensors
"""
import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:                     # also runnable as a plain script path
    sys.path.insert(0, str(_REPO))

from src.datasets.evaluations.helper_eval_scripts.run_arabic_benchmark_kraken import (  # noqa: E402
    MIN_BOUNDARY_POINTS,
    TEXT_DIRECTION,
    KrakenReader,
    LineRead,
    PageRead,
    add_reader_arguments,
    check_tag,
    emit,
    kraken_version,
    mean_confidence,
    quiet_kraken,
    read_jsonl,
    read_page,
    reader_from_args,
    source_from_args,
    timed_read,
)

DEFAULT_MANIFEST = Path("/Volumes/home/studio_offload/datasets/pgp_arabic_editions_v2/manifest.jsonl")
SUMMARY_FILE = "summary.jsonl"
ID_FIELDS = ("stem", "pgpid", "canonical_id")

PageJob = Tuple[Dict[str, Any], Path]
PageReadFn = Callable[[Dict[str, Any], Path], PageRead]


def read_ids(path: Optional[Path]) -> Optional[Set[str]]:
    """Ids from a text file: one per line, blank lines and ``#`` comments ignored.

    :param path: Ids file, or None.
    :type path: Optional[Path]
    :return: The ids, or None when no file is given.
    :rtype: Optional[Set[str]]
    """
    if path is None:
        return None
    with open(path, encoding="utf-8") as fh:
        return {line.split("#", 1)[0].strip() for line in fh} - {""}


def select_pages(rows: Iterable[Dict[str, Any]], ids: Optional[Set[str]] = None, limit: int = 0) -> List[Dict[str, Any]]:
    """Manifest rows to process, in manifest order.

    :param rows: Manifest rows.
    :type rows: Iterable[Dict[str, Any]]
    :param ids: Keep a row when its ``stem``, ``pgpid`` or ``canonical_id`` is in this set; None keeps all.
    :type ids: Optional[Set[str]]
    :param limit: Keep only the first N selected rows; 0 keeps all.
    :type limit: int
    :return: Selected rows.
    :rtype: List[Dict[str, Any]]
    """
    selected = [row for row in rows if ids is None or any(str(row.get(name)) in ids for name in ID_FIELDS)]
    return selected[:limit] if limit else selected


def tag_dir(out_dir: Path, tag: str) -> Path:
    """The directory of one run.

    :param out_dir: Output root.
    :type out_dir: Path
    :param tag: Run tag.
    :type tag: str
    :return: ``<out_dir>/<tag>``.
    :rtype: Path
    """
    return out_dir / check_tag(tag)


def page_json_path(run_dir: Path, stem: str) -> Path:
    """Where one page's JSON goes.

    :param run_dir: Run directory.
    :type run_dir: Path
    :param stem: Manifest stem.
    :type stem: str
    :return: ``<run_dir>/<stem>.json``.
    :rtype: Path
    :raises ValueError: When the stem is not a plain file name.
    """
    if not stem or "/" in stem or "\\" in stem or stem.startswith("."):
        raise ValueError(f"stem must be a plain file name, got {stem!r}")
    return run_dir / f"{stem}.json"


def pending_pages(rows: Iterable[Dict[str, Any]], run_dir: Path) -> List[Dict[str, Any]]:
    """Rows without a page JSON yet, first occurrence per stem.

    :param rows: Selected manifest rows.
    :type rows: Iterable[Dict[str, Any]]
    :param run_dir: Run directory.
    :type run_dir: Path
    :return: Rows still to process.
    :rtype: List[Dict[str, Any]]
    """
    seen: Set[str] = set()
    pending: List[Dict[str, Any]] = []
    for row in rows:
        stem = row["stem"]
        if stem not in seen and not page_json_path(run_dir, stem).exists():
            pending.append(row)
        seen.add(stem)
    return pending


def resolve_image(row: Dict[str, Any], image_root: Optional[Path] = None) -> Path:
    """The image file to read for a manifest row.

    :param row: Manifest row with ``image_path``.
    :type row: Dict[str, Any]
    :param image_root: Directory replacing the manifest path's directory (container mounts), or None.
    :type image_root: Optional[Path]
    :return: ``image_root / basename`` when a root is given, else the manifest path.
    :rtype: Path
    """
    manifest_path = Path(row["image_path"])
    return image_root / manifest_path.name if image_root is not None else manifest_path


def split_missing(rows: Iterable[Dict[str, Any]], image_root: Optional[Path] = None) -> Tuple[List[PageJob], List[Dict[str, Any]]]:
    """Separate rows whose image exists from rows whose image is missing.

    :param rows: Pending manifest rows.
    :type rows: Iterable[Dict[str, Any]]
    :param image_root: See :func:`resolve_image`.
    :type image_root: Optional[Path]
    :return: ``([(row, image path)], [rows with a missing image])``.
    :rtype: Tuple[List[PageJob], List[Dict[str, Any]]]
    """
    jobs: List[PageJob] = []
    missing: List[Dict[str, Any]] = []
    for row in rows:
        path = resolve_image(row, image_root)
        if path.is_file():
            jobs.append((row, path))
        else:
            missing.append(row)
    return jobs, missing


def split_no_geometry(rows: Iterable[Dict[str, Any]], geometry_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate rows that have a source page JSON in ``geometry_dir`` from rows that do not (re-read mode).

    :param rows: Pending manifest rows.
    :type rows: Iterable[Dict[str, Any]]
    :param geometry_dir: Earlier run directory holding ``<stem>.json`` files.
    :type geometry_dir: Path
    :return: ``([rows with a source JSON], [rows without one])``.
    :rtype: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
    """
    found: List[Dict[str, Any]] = []
    absent: List[Dict[str, Any]] = []
    for row in rows:
        (found if page_json_path(geometry_dir, row["stem"]).is_file() else absent).append(row)
    return found, absent


def json_points(points: Iterable[Sequence[float]]) -> List[Tuple[int, int]]:
    """``[[x, y], ...]`` from a page JSON as kraken's integer ``(x, y)`` tuples.

    :param points: Points as stored in a page JSON.
    :type points: Iterable[Sequence[float]]
    :return: Rounded integer tuples.
    :rtype: List[Tuple[int, int]]
    """
    return [(int(round(float(point[0]))), int(round(float(point[1])))) for point in points]


def segmentation_from_lines(lines: Sequence[Dict[str, Any]], imagename: str = "") -> Any:
    """A kraken baseline ``Segmentation`` rebuilt from the ``lines`` of a page JSON, in their order.

    Each line becomes ``BaselineLine(id="line_<index>", type="baselines", baseline, boundary)``; the
    segmentation is ``type="baselines"``, ``text_direction`` as the segmenter's config, no script
    detection. ``RecognitionTaskModel.predict(image, segmentation, config)`` then yields one record
    per line in this order.

    :param lines: Page-JSON lines (``index``, ``baseline``, ``boundary``; other keys are ignored).
    :type lines: Sequence[Dict[str, Any]]
    :param imagename: Image path recorded in the segmentation.
    :type imagename: str
    :return: ``kraken.containers.Segmentation``.
    :rtype: Any
    :raises ValueError: When a line's boundary has fewer than three points (it cannot be cropped).
    """
    from kraken.containers import BaselineLine, Segmentation

    baseline_lines = []
    for line in lines:
        boundary = json_points(line["boundary"])
        if len(boundary) < MIN_BOUNDARY_POINTS:
            raise ValueError(f"line {line['index']}: boundary has {len(boundary)} points, need {MIN_BOUNDARY_POINTS}")
        baseline_lines.append(BaselineLine(id=f"line_{line['index']}", type="baselines",
                                           baseline=json_points(line["baseline"]), boundary=boundary))
    return Segmentation(type="baselines", imagename=imagename, text_direction=TEXT_DIRECTION, script_detection=False,
                        lines=baseline_lines)


def page_from_geometry(geometry: Dict[str, Any], records: Sequence[Any], image_size: Tuple[int, int]) -> PageRead:
    """A page read that keeps a source page JSON's geometry and takes text from new recognition records.

    :param geometry: Source page JSON (``lines``, ``n_lines``, ``n_dropped``).
    :type geometry: Dict[str, Any]
    :param records: Recognition records (``.prediction``, ``.confidences``), one per source line, in order.
    :type records: Sequence[Any]
    :param image_size: ``(width, height)`` of the image read.
    :type image_size: Tuple[int, int]
    :return: Lines with the source baseline / boundary and the new text and confidence.
    :rtype: PageRead
    :raises ValueError: When the record count differs from the line count, or the source indices are not 0..n-1.
    """
    lines = geometry["lines"]
    if [line["index"] for line in lines] != list(range(len(lines))):
        raise ValueError("source line indices are not 0..n-1 in order")
    if len(records) != len(lines):
        raise ValueError(f"{len(records)} recognition records for {len(lines)} lines")
    return PageRead(lines=[LineRead(baseline=line["baseline"], boundary=line["boundary"], text=record.prediction,
                                    mean_confidence=mean_confidence(record.confidences))
                           for line, record in zip(lines, records)],
                    n_lines=geometry["n_lines"], n_dropped=geometry["n_dropped"], image_size=image_size)


def reread_page(reader: KrakenReader, geometry: Dict[str, Any], image_path: Path) -> PageRead:
    """Recognise one page on the lines of an earlier run's page JSON (no segmenter).

    :param reader: Reader with a recogniser (its segmenter is not used).
    :type reader: KrakenReader
    :param geometry: Source page JSON of the same page.
    :type geometry: Dict[str, Any]
    :param image_path: Page image.
    :type image_path: Path
    :return: The source geometry with new text.
    :rtype: PageRead
    :raises ValueError: When the image size differs from the source JSON's ``image_size``.
    """
    from PIL import Image
    from kraken import binarization

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    if list(rgb.size) != list(geometry["image_size"]):
        raise ValueError(f"image is {list(rgb.size)} but the geometry was found on {geometry['image_size']}")
    segmentation = segmentation_from_lines(geometry["lines"], str(image_path))
    records: List[Any] = []
    if segmentation.lines:
        records = list(reader.recogniser.predict(binarization.nlbin(rgb) if reader.rec_input == "binarized" else rgb,
                                                 segmentation, reader.rec_config))
    return page_from_geometry(geometry, records, rgb.size)


def segment_row(reader: KrakenReader, row: Dict[str, Any], image_path: Path) -> PageRead:
    """Segment and read one manifest row's page (the default mode's :data:`PageReadFn`).

    :param reader: Loaded segmenter + recogniser.
    :type reader: KrakenReader
    :param row: Manifest row (unused; part of the :data:`PageReadFn` signature).
    :type row: Dict[str, Any]
    :param image_path: Page image.
    :type image_path: Path
    :return: The page read.
    :rtype: PageRead
    """
    return read_page(reader, image_path)


def reread_row(reader: KrakenReader, geometry_dir: Path, row: Dict[str, Any], image_path: Path) -> PageRead:
    """Re-read one manifest row's page on the geometry of ``<geometry_dir>/<stem>.json`` (re-read mode's :data:`PageReadFn`).

    :param reader: Reader with a recogniser.
    :type reader: KrakenReader
    :param geometry_dir: Earlier run directory.
    :type geometry_dir: Path
    :param row: Manifest row.
    :type row: Dict[str, Any]
    :param image_path: Page image.
    :type image_path: Path
    :return: The page read.
    :rtype: PageRead
    """
    geometry = json.loads(page_json_path(geometry_dir, row["stem"]).read_text(encoding="utf-8"))
    return reread_page(reader, geometry, image_path)


def load_recogniser(recogniser: str, rec_input: str = "rgb", threads: int = 4) -> KrakenReader:
    """Load a recogniser alone for CPU inference (re-read mode: no segmenter).

    :param recogniser: Recognition model path (e.g. the PP-OCRv6 ``.safetensors``).
    :type recogniser: str
    :param rec_input: ``rgb`` or ``binarized``.
    :type rec_input: str
    :param threads: torch intra-op threads.
    :type threads: int
    :return: A reader whose segmenter and segmenter config are None.
    :rtype: KrakenReader
    """
    import torch
    from kraken.models import load_models
    from kraken.tasks.recognition import RecognitionInferenceConfig, RecognitionTaskModel

    torch.set_num_threads(threads)
    return KrakenReader(segmenter=None, seg_config=None,
                        recogniser=RecognitionTaskModel(load_models(recogniser, tasks=["recognition"])),
                        rec_config=RecognitionInferenceConfig(), rec_input=rec_input)


def recogniser_from_args(args: argparse.Namespace) -> KrakenReader:
    """Load the recogniser the parsed options describe (re-read mode).

    :param args: Parsed options.
    :type args: argparse.Namespace
    :return: The loaded reader.
    :rtype: KrakenReader
    """
    return load_recogniser(args.recogniser, args.rec_input, args.threads)


def geometry_source(args: argparse.Namespace, version: Optional[str] = None) -> str:
    """Provenance string of re-read mode: ``--source`` when given, else built from the geometry run and recogniser.

    :param args: Parsed options (``geometry_from``, ``recogniser``, ``rec_input``, ``source``).
    :type args: argparse.Namespace
    :param version: kraken version (default: the installed one).
    :type version: Optional[str]
    :return: e.g. ``geometry from blla2026_ppocrv6m + ar.safetensors, rec rgb, kraken 7.1.1``.
    :rtype: str
    """
    if args.source:
        return args.source
    return (f"geometry from {args.geometry_from.resolve().name} + {Path(args.recogniser).name}, rec {args.rec_input}, "
            f"kraken {version or kraken_version()}")


def line_json(index: int, line: LineRead) -> Dict[str, Any]:
    """One line of a page JSON.

    :param index: Position in the segmenter's order (after dropping boundary-less lines).
    :type index: int
    :param line: The recognised line.
    :type line: LineRead
    :return: ``{index, baseline, boundary, text, mean_confidence}``.
    :rtype: Dict[str, Any]
    """
    return {"index": index, "baseline": line.baseline, "boundary": line.boundary, "text": line.text,
            "mean_confidence": line.mean_confidence}


def page_json(row: Dict[str, Any], page: PageRead, secs: float, source: str) -> Dict[str, Any]:
    """The JSON written for one page.

    :param row: Manifest row.
    :type row: Dict[str, Any]
    :param page: The page's read.
    :type page: PageRead
    :param secs: Wall time in seconds.
    :type secs: float
    :param source: Provenance string.
    :type source: str
    :return: See the module docstring.
    :rtype: Dict[str, Any]
    """
    return {"stem": row["stem"], "pgpid": row.get("pgpid"), "canonical_id": row.get("canonical_id"),
            "image_path": row.get("image_path"), "image_size": list(page.image_size),
            "expected_lines": row.get("lines"), "n_lines": page.n_lines, "n_dropped": page.n_dropped,
            "secs": secs, "source": source,
            "lines": [line_json(index, line) for index, line in enumerate(page.lines)]}


def summary_row(row: Dict[str, Any], page: PageRead, secs: float) -> Dict[str, Any]:
    """One ``summary.jsonl`` row.

    :param row: Manifest row.
    :type row: Dict[str, Any]
    :param page: The page's read.
    :type page: PageRead
    :param secs: Wall time in seconds.
    :type secs: float
    :return: ``{stem, expected_lines, n_lines, n_dropped, ratio, secs}``; ``ratio`` is None without an expected count.
    :rtype: Dict[str, Any]
    """
    return page_summary({"stem": row["stem"], "expected_lines": row.get("lines"), "n_lines": page.n_lines,
                         "n_dropped": page.n_dropped, "secs": secs})


def page_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The ``summary.jsonl`` row of a page JSON (:func:`page_json`).

    :param payload: Page JSON, or any dict with ``stem, expected_lines, n_lines, n_dropped, secs``.
    :type payload: Dict[str, Any]
    :return: ``{stem, expected_lines, n_lines, n_dropped, ratio, secs}``; ``ratio`` is None without an expected count.
    :rtype: Dict[str, Any]
    """
    expected = payload.get("expected_lines")
    ratio = round(payload["n_lines"] / expected, 3) if expected else None
    return {"stem": payload["stem"], "expected_lines": expected, "n_lines": payload["n_lines"],
            "n_dropped": payload["n_dropped"], "ratio": ratio, "secs": payload["secs"]}


def rebuild_summary(run_dir: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Rewrite ``summary.jsonl`` from the page JSONs of a run directory (temporary file + rename).

    One row per ``<stem>.json``: manifest stems first, in manifest order, then page JSONs whose stem is
    not in the manifest, by name.

    :param run_dir: Run directory.
    :type run_dir: Path
    :param rows: Manifest rows (their order sets the summary's order).
    :type rows: Iterable[Dict[str, Any]]
    :return: Rows written.
    :rtype: int
    """
    pages = {path.stem: path for path in run_dir.glob("*.json")}
    manifest_stems = [stem for stem in dict.fromkeys(row["stem"] for row in rows) if stem in pages]
    ordered = manifest_stems + sorted(set(pages) - set(manifest_stems))
    tmp = run_dir / f"{SUMMARY_FILE}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for stem in ordered:
            payload = json.loads(pages[stem].read_text(encoding="utf-8"))
            fh.write(json.dumps(page_summary(payload), ensure_ascii=False) + "\n")
    os.replace(tmp, run_dir / SUMMARY_FILE)
    return len(ordered)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON through a temporary file and a rename, so a half-written page never looks done.

    :param path: Target file.
    :type path: Path
    :param payload: JSON-serialisable object.
    :type payload: Dict[str, Any]
    :return: None.
    :rtype: None
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, path)


def run_pages(jobs: List[PageJob], read_fn: PageReadFn, run_dir: Path, source: str,
              log: Callable[[str], None] = emit) -> Dict[str, int]:
    """Read every page, writing its JSON and appending its summary row.

    :param jobs: ``(manifest row, image path)`` pairs.
    :type jobs: List[PageJob]
    :param read_fn: ``(manifest row, image path) -> PageRead`` (:func:`segment_row`, :func:`reread_row`; injectable for tests).
    :type read_fn: PageReadFn
    :param run_dir: Run directory (created when missing).
    :type run_dir: Path
    :param source: Provenance string.
    :type source: str
    :param log: Progress printer.
    :type log: Callable[[str], None]
    :return: ``{"written": n, "failed": n}``.
    :rtype: Dict[str, int]
    """
    stats = {"written": 0, "failed": 0}
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / SUMMARY_FILE, "a", encoding="utf-8") as summary:
        for n, (row, image_path) in enumerate(jobs, 1):
            page, error, secs = timed_read(partial(read_fn, row), image_path)
            if page is None:
                stats["failed"] += 1
                log(f"  {n:4d}/{len(jobs)} {row['stem']}: FAILED {error} ({secs}s)")
                continue
            write_json_atomic(page_json_path(run_dir, row["stem"]), page_json(row, page, secs, source))
            info = summary_row(row, page, secs)
            summary.write(json.dumps(info, ensure_ascii=False) + "\n")
            summary.flush()
            stats["written"] += 1
            log(f"  {n:4d}/{len(jobs)} {row['stem']}: {page.n_lines} lines ({page.n_dropped} dropped) "
                f"vs {info['expected_lines']} expected, {secs}s")
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: segment and read the selected edition pages, or re-read them on an earlier run's geometry.

    :param argv: Arguments (default: ``sys.argv``).
    :type argv: Optional[List[str]]
    :return: None.
    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     conflict_handler="resolve")  # lets --segmenter below replace the shared required one
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="edition manifest jsonl (default: the NAS copy)")
    parser.add_argument("--out-dir", type=Path, required=True, help="output root; pages go to <out-dir>/<tag>/")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="directory holding the page images (default: the manifest's image_path as is)")
    parser.add_argument("--ids", type=Path, default=None,
                        help="text file of stems / pgpids / canonical ids to keep, one per line")
    add_reader_arguments(parser)
    parser.add_argument("--segmenter", default=None,
                        help="baseline segmentation model path, or 'default' for kraken's bundled segmenter "
                             "(required unless --geometry-from is given)")
    parser.add_argument("--geometry-from", type=Path, default=None,
                        help="earlier run directory (<out-dir>/<tag>) whose page JSONs give the lines: "
                             "only the recogniser runs")
    args = parser.parse_args(argv)
    geometry_dir: Optional[Path] = args.geometry_from
    if geometry_dir is None and args.segmenter is None:
        parser.error("--segmenter is required unless --geometry-from is given")
    run_dir = tag_dir(args.out_dir, args.tag)
    if geometry_dir is not None and not geometry_dir.is_dir():
        parser.error(f"--geometry-from {geometry_dir} is not a directory")
    if geometry_dir is not None and geometry_dir.resolve() == run_dir.resolve():
        parser.error("--geometry-from must be another run's directory, not this run's")
    quiet_kraken()

    manifest_rows = read_jsonl(args.manifest)
    rows = select_pages(manifest_rows, read_ids(args.ids), args.limit)
    pending = pending_pages(rows, run_dir)
    readable, no_geometry = (pending, []) if geometry_dir is None else split_no_geometry(pending, geometry_dir)
    jobs, missing = split_missing(readable, args.image_root)
    emit(f"{args.tag}: {len(rows)} pages selected, {len(rows) - len(pending)} done before, "
         f"{len(jobs)} to read, {len(missing)} without an image"
         + ("" if geometry_dir is None else f", {len(no_geometry)} without a geometry JSON in {geometry_dir}"))
    for row in missing[:10]:
        emit(f"  missing image: {row['stem']} -> {resolve_image(row, args.image_root)}")
    for row in no_geometry[:10]:
        emit(f"  no geometry: {row['stem']} -> {page_json_path(geometry_dir, row['stem'])}")
    if geometry_dir is not None and args.segmenter is not None:
        emit(f"  note: --segmenter {args.segmenter} is ignored with --geometry-from")
    if jobs:
        started = time.perf_counter()
        if geometry_dir is None:
            stats = run_pages(jobs, partial(segment_row, reader_from_args(args)), run_dir, source_from_args(args))
        else:
            stats = run_pages(jobs, partial(reread_row, recogniser_from_args(args), geometry_dir), run_dir,
                              geometry_source(args))
        emit(f"done: {stats} in {(time.perf_counter() - started) / 60:.1f} min -> {run_dir}")
    if run_dir.is_dir():
        emit(f"summary: {rebuild_summary(run_dir, manifest_rows)} pages -> {run_dir / SUMMARY_FILE}")


if __name__ == "__main__":
    main()
