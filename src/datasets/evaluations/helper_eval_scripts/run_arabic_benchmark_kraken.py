# File name: run_arabic_benchmark_kraken.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Read the Arabic-script benchmark with Kraken 7.1 (one segmenter + one recogniser) for scoring.

Writes ``<bench>/outputs_other/<tag>.jsonl``, one row per benchmark image::

    {"doc_id", "image_index", "text", "secs", "source", "n_lines", "n_dropped"}

``text`` holds the recognised lines in the segmenter's order, joined by newlines (blank predictions
are left out). ``n_lines`` is the number of lines the segmenter found and ``n_dropped`` the number
of those without a usable boundary polygon (missing, or fewer than three points), which cannot be
recognised. Score the rows with::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.score_arabic_benchmark --outputs outputs_other

The run is resumable: an image that already has a text answer is skipped. A page that raises is
logged and written with ``"text": null`` plus an ``"error"`` field. The scorer ignores such rows,
the next run retries the image, and the last row written for an image wins. A last row torn by a
kill (it does not decode) is ignored with a warning and cut off before new rows are appended, so its
image is read again; an undecodable row anywhere else stops the run.

The script needs only the standard library, Pillow, torch and kraken (>= 7.1 for the PP-OCRv6
safetensors recognisers). torch and kraken are imported inside the functions that use them, so the
pure helpers can be imported and tested without either. It does not import anything else from this
repository, so the file can be mounted on its own into the ``kraken-train:k711`` image. Every path
comes from the arguments::

    python run_arabic_benchmark_kraken.py --bench /bench --tag ppocrv6m_zs_blla2026 \\
        --segmenter /segmodels/blla_2026/blla.mlmodel --recogniser /ar/ppocrv6_medium.safetensors \\
        --ids-from /bench/outputs_other/kraken_service_midrash_gen01.jsonl

``--segmenter default`` loads kraken's bundled baseline segmenter. ``--seg-input binarized`` and
``--rec-input binarized`` pass kraken's ``nlbin`` binarisation of the page to the segmenter or the
recogniser instead of the RGB page.

``segment_edition_pages`` (in ``src/finetuning/kraken``) reuses the reader helpers here and writes
per-line geometry instead of page text.
"""
import argparse
import importlib.metadata
import json
import logging
import time
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

DEFAULT_BENCHMARK = Path("/Volumes/home/studio_offload/datasets/arabic_script_benchmark_v0")
OUTPUTS_SUBDIR = "outputs_other"
INPUT_MODES = ("rgb", "binarized")
TEXT_DIRECTION = "horizontal-rl"
MIN_BOUNDARY_POINTS = 3


@dataclass
class LineRead:
    """One recognised line, in pixel coordinates of the page image as loaded.

    :param baseline: Baseline polyline ``[[x, y], ...]``.
    :param boundary: Boundary polygon ``[[x, y], ...]``.
    :param text: Recognised text (logical order).
    :param mean_confidence: Mean per-character confidence; ``None`` when the line has no characters.
    """

    baseline: List[List[int]]
    boundary: List[List[int]]
    text: str
    mean_confidence: Optional[float]


@dataclass
class PageRead:
    """The read of one page image.

    :param lines: Recognised lines in the segmenter's order.
    :param n_lines: Lines the segmenter found, including dropped ones.
    :param n_dropped: Lines dropped before recognition for lack of a usable boundary polygon.
    :param image_size: ``(width, height)`` of the image the coordinates refer to.
    """

    lines: List[LineRead]
    n_lines: int
    n_dropped: int
    image_size: Tuple[int, int]


@dataclass
class KrakenReader:
    """A loaded segmenter + recogniser pair with its inference settings.

    :param segmenter: ``kraken.tasks.segmentation.SegmentationTaskModel``.
    :param seg_config: ``kraken.configs.SegmentationInferenceConfig``.
    :param recogniser: ``kraken.tasks.recognition.RecognitionTaskModel``.
    :param rec_config: ``kraken.tasks.recognition.RecognitionInferenceConfig``.
    :param seg_input: ``rgb`` or ``binarized``: the image the segmenter sees.
    :param rec_input: ``rgb`` or ``binarized``: the image the recogniser sees.
    """

    segmenter: Any
    seg_config: Any
    recogniser: Any
    rec_config: Any
    seg_input: str = "rgb"
    rec_input: str = "rgb"


class ImageJob(NamedTuple):
    """One benchmark image to read.

    :param doc_id: Benchmark document id.
    :param image_index: Image index within the document.
    :param file: Image file name under ``<bench>/images``.
    """

    doc_id: str
    image_index: int
    file: str


ReadFn = Callable[[Path], PageRead]


def emit(message: str) -> None:
    """Print a progress line immediately (the runs are watched through container logs).

    :param message: Line to print.
    :type message: str
    :return: None.
    :rtype: None
    """
    print(message, flush=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """All rows of a JSON-lines file, skipping blank lines.

    :param path: JSON-lines file.
    :type path: Path
    :return: Rows in file order.
    :rtype: List[Dict[str, Any]]
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_records(bench: Path) -> Dict[str, Dict[str, Any]]:
    """Benchmark records by document id, in file order.

    :param bench: Benchmark directory holding ``benchmark.jsonl``.
    :type bench: Path
    :return: ``{doc_id: record}``.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return {record["id"]: record for record in read_jsonl(bench / "benchmark.jsonl")}


def doc_ids_from(path: Optional[Path]) -> Optional[List[str]]:
    """Unique document ids of an answers file, in first-seen order.

    :param path: Answers jsonl (rows with ``doc_id``), or None.
    :type path: Optional[Path]
    :return: The ids, or None when no file is given. An empty file gives an empty list.
    :rtype: Optional[List[str]]
    """
    if path is None:
        return None
    return list(dict.fromkeys(row["doc_id"] for row in read_jsonl(path)))


def select_ids(records: Dict[str, Dict[str, Any]], requested: Optional[Sequence[str]],
               limit: int = 0) -> Tuple[List[str], List[str]]:
    """The documents to read.

    :param records: Benchmark records by document id.
    :type records: Dict[str, Dict[str, Any]]
    :param requested: Ids to read in this order (``--ids-from``), or None for every record in file order.
    :type requested: Optional[Sequence[str]]
    :param limit: Keep only the first N known ids; 0 keeps all.
    :type limit: int
    :return: ``(ids to read, requested ids that are not in the benchmark)``.
    :rtype: Tuple[List[str], List[str]]
    """
    wanted = list(records) if requested is None else list(requested)
    known = [doc_id for doc_id in wanted if doc_id in records]
    unknown = [doc_id for doc_id in wanted if doc_id not in records]
    return (known[:limit] if limit else known), unknown


def read_jsonl_torn_tail(path: Path, log: Callable[[str], None] = emit) -> List[Dict[str, Any]]:
    """Rows of an append-only JSON-lines file whose last row may have been torn by a kill.

    A final non-blank line that does not decode (cut JSON, or a UTF-8 character cut in half) is
    skipped with a warning; an undecodable line anywhere else raises.

    :param path: JSON-lines file.
    :type path: Path
    :param log: Warning printer.
    :type log: Callable[[str], None]
    :return: Rows in file order.
    :rtype: List[Dict[str, Any]]
    :raises ValueError: On an undecodable line other than the last.
    """
    with open(path, "rb") as fh:
        lines = [line for line in fh if line.strip()]
    rows = [json.loads(line) for line in lines[:-1]]
    if lines:
        try:
            rows.append(json.loads(lines[-1]))
        except ValueError:  # JSONDecodeError or UnicodeDecodeError: the last write was cut off
            log(f"warning: {path}: ignoring a torn last row ({len(lines[-1])} bytes); its image is read again")
    return rows


def done_pairs(out_path: Path, log: Callable[[str], None] = emit) -> Set[Tuple[str, int]]:
    """``(doc_id, image_index)`` pairs that already have a text answer.

    Rows whose ``text`` is not a string (pages that failed) do not count, so they are retried; nor
    does a torn last row (:func:`read_jsonl_torn_tail`).

    :param out_path: Output jsonl (may not exist yet).
    :type out_path: Path
    :param log: Warning printer.
    :type log: Callable[[str], None]
    :return: Answered pairs.
    :rtype: Set[Tuple[str, int]]
    """
    if not out_path.exists():
        return set()
    return {(row["doc_id"], int(row["image_index"])) for row in read_jsonl_torn_tail(out_path, log)
            if isinstance(row.get("text"), str)}


def end_on_line_boundary(path: Path, log: Callable[[str], None] = emit) -> None:
    """Make an append-only JSON-lines file end with a newline, so the next row cannot glue onto its last one.

    A final line without a newline that decodes gets its newline; one that does not (a row torn by a
    kill, which :func:`done_pairs` did not count) is cut off.

    :param path: JSON-lines file (may not exist yet).
    :type path: Path
    :param log: Warning printer.
    :type log: Callable[[str], None]
    :return: None.
    :rtype: None
    """
    if not path.exists():
        return
    with open(path, "rb+") as fh:
        data = fh.read()
        if not data or data.endswith(b"\n"):
            return
        start = data.rfind(b"\n") + 1
        try:
            json.loads(data[start:])
        except ValueError:  # torn row: drop it, its image is read again
            fh.truncate(start)
            log(f"warning: {path}: cut off a torn last row ({len(data) - start} bytes)")
            return
        fh.write(b"\n")


def build_jobs(records: Dict[str, Dict[str, Any]], ids: Iterable[str], done: Set[Tuple[str, int]]) -> List[ImageJob]:
    """Images still to read, document by document in ``ids`` order, images in record order.

    :param records: Benchmark records by document id.
    :type records: Dict[str, Dict[str, Any]]
    :param ids: Documents to read; ids missing from ``records`` are ignored.
    :type ids: Iterable[str]
    :param done: Pairs that already have an answer.
    :type done: Set[Tuple[str, int]]
    :return: Pending jobs.
    :rtype: List[ImageJob]
    """
    jobs: List[ImageJob] = []
    for doc_id in ids:
        for image in records.get(doc_id, {}).get("images", []):
            job = ImageJob(doc_id, int(image["image_index"]), image["file"])
            if (job.doc_id, job.image_index) not in done:
                jobs.append(job)
    return jobs


def check_tag(tag: str) -> str:
    """Validate a run tag, which becomes a file or directory name.

    :param tag: Run tag, e.g. ``ppocrv6m_zs_blla2026``.
    :type tag: str
    :return: The tag, unchanged.
    :rtype: str
    :raises ValueError: When the tag is empty, contains a path separator or starts with a dot.
    """
    if not tag or "/" in tag or "\\" in tag or tag.startswith("."):
        raise ValueError(f"tag must be a plain file name, got {tag!r}")
    return tag


def output_path(bench: Path, tag: str) -> Path:
    """Where a run's answers go.

    :param bench: Benchmark directory.
    :type bench: Path
    :param tag: Run tag.
    :type tag: str
    :return: ``<bench>/outputs_other/<tag>.jsonl``.
    :rtype: Path
    """
    return bench / OUTPUTS_SUBDIR / f"{check_tag(tag)}.jsonl"


def default_source(segmenter: str, recogniser: str, seg_input: str, rec_input: str, version: str) -> str:
    """Provenance string stored with every output row.

    :param segmenter: Segmenter path or ``default``.
    :type segmenter: str
    :param recogniser: Recogniser path.
    :type recogniser: str
    :param seg_input: Segmenter input mode.
    :type seg_input: str
    :param rec_input: Recogniser input mode.
    :type rec_input: str
    :param version: Installed kraken version.
    :type version: str
    :return: e.g. ``blla.mlmodel + ppocrv6_medium.safetensors, seg rgb, rec rgb, kraken 7.1.1``.
    :rtype: str
    """
    return f"{Path(segmenter).name} + {Path(recogniser).name}, seg {seg_input}, rec {rec_input}, kraken {version}"


def points_to_json(points: Optional[Iterable[Sequence[float]]]) -> List[List[int]]:
    """Integer ``[[x, y], ...]`` from kraken's point lists (tuples, lists or numpy rows).

    :param points: Points, or None.
    :type points: Optional[Iterable[Sequence[float]]]
    :return: Rounded integer pairs; empty when ``points`` is None.
    :rtype: List[List[int]]
    """
    if points is None:
        return []
    return [[int(round(float(point[0]))), int(round(float(point[1])))] for point in points]


def mean_confidence(confidences: Optional[Sequence[float]]) -> Optional[float]:
    """Mean of a line's per-character confidences.

    :param confidences: Confidences, possibly empty or None.
    :type confidences: Optional[Sequence[float]]
    :return: The mean rounded to 4 places, or None when there are none.
    :rtype: Optional[float]
    """
    values = [float(value) for value in (confidences if confidences is not None else [])]
    return round(sum(values) / len(values), 4) if values else None


def line_read(record: Any) -> LineRead:
    """Convert one kraken recognition record (``.prediction``, ``.confidences``, ``.baseline``, ``.boundary``).

    :param record: A record yielded by ``RecognitionTaskModel.predict``.
    :type record: Any
    :return: The line in plain Python types.
    :rtype: LineRead
    """
    return LineRead(baseline=points_to_json(record.baseline), boundary=points_to_json(record.boundary),
                    text=record.prediction, mean_confidence=mean_confidence(record.confidences))


def keep_bounded_lines(lines: Sequence[Any]) -> List[Any]:
    """Segmented lines that can be recognised; their boundaries are cast to integer tuples in place.

    A line without a boundary, or with fewer than three boundary points, cannot be cropped and is
    dropped (kraken 7.1's segmenter returns ``boundary=None`` for some lines).

    :param lines: ``segmentation.lines`` (objects with a mutable ``boundary`` attribute).
    :type lines: Sequence[Any]
    :return: The kept lines in their original order.
    :rtype: List[Any]
    """
    kept: List[Any] = []
    for line in lines:
        if line.boundary is None:
            continue
        boundary = [tuple(int(round(float(value))) for value in point) for point in line.boundary]
        if len(boundary) >= MIN_BOUNDARY_POINTS:
            line.boundary = boundary
            kept.append(line)
    return kept


def page_text(page: PageRead) -> str:
    """A page's answer text: non-blank line predictions joined by newlines.

    :param page: Page read.
    :type page: PageRead
    :return: Text in the segmenter's line order.
    :rtype: str
    """
    return "\n".join(line.text for line in page.lines if line.text.strip())


def answer_row(job: ImageJob, page: Optional[PageRead], secs: float, source: str,
               error: Optional[str] = None) -> Dict[str, Any]:
    """One output row for the benchmark scorer.

    :param job: The image read.
    :type job: ImageJob
    :param page: Its read, or None when it failed.
    :type page: Optional[PageRead]
    :param secs: Wall time in seconds.
    :type secs: float
    :param source: Provenance string.
    :type source: str
    :param error: Failure description, when the page failed.
    :type error: Optional[str]
    :return: ``{doc_id, image_index, text, secs, source, n_lines, n_dropped}`` plus ``error`` on failure.
    :rtype: Dict[str, Any]
    """
    row: Dict[str, Any] = {"doc_id": job.doc_id, "image_index": job.image_index,
                           "text": page_text(page) if page is not None else None, "secs": secs, "source": source,
                           "n_lines": page.n_lines if page is not None else 0,
                           "n_dropped": page.n_dropped if page is not None else 0}
    if error is not None:
        row["error"] = error
    return row


def timed_read(read_fn: ReadFn, image_path: Path) -> Tuple[Optional[PageRead], Optional[str], float]:
    """Read one page, timing it; a page that raises is reported instead of ending the run.

    :param read_fn: ``image path -> PageRead``.
    :type read_fn: ReadFn
    :param image_path: Page image.
    :type image_path: Path
    :return: ``(page or None, error or None, seconds)``.
    :rtype: Tuple[Optional[PageRead], Optional[str], float]
    """
    start = time.perf_counter()
    try:
        page = read_fn(image_path)
    except Exception as exc:  # noqa: BLE001 -- one unreadable page must not end an hours-long run
        return None, repr(exc), round(time.perf_counter() - start, 1)
    return page, None, round(time.perf_counter() - start, 1)


def run_jobs(jobs: Sequence[ImageJob], read_fn: ReadFn, image_dir: Path, out_path: Path, source: str,
             log: Callable[[str], None] = emit) -> Dict[str, int]:
    """Read every job and append its row to the output file (flushed per row, so a kill loses at most one page).

    A torn last row left by an earlier kill is cut off first (:func:`end_on_line_boundary`).

    :param jobs: Pending images.
    :type jobs: Sequence[ImageJob]
    :param read_fn: ``image path -> PageRead`` (injectable for tests).
    :type read_fn: ReadFn
    :param image_dir: ``<bench>/images``.
    :type image_dir: Path
    :param out_path: Output jsonl (appended to).
    :type out_path: Path
    :param source: Provenance string.
    :type source: str
    :param log: Progress printer.
    :type log: Callable[[str], None]
    :return: ``{"answered": n, "failed": n}``.
    :rtype: Dict[str, int]
    """
    stats = {"answered": 0, "failed": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    end_on_line_boundary(out_path, log)
    with open(out_path, "a", encoding="utf-8") as fh:
        for n, job in enumerate(jobs, 1):
            page, error, secs = timed_read(read_fn, image_dir / job.file)
            row = answer_row(job, page, secs, source, error)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if page is None:
                stats["failed"] += 1
                log(f"  {n:4d}/{len(jobs)} {job.doc_id}#{job.image_index}: FAILED {error} ({secs}s)")
            else:
                stats["answered"] += 1
                log(f"  {n:4d}/{len(jobs)} {job.doc_id}#{job.image_index}: {page.n_lines} lines "
                    f"({page.n_dropped} dropped), {len(row['text'])} chars, {secs}s")
    return stats


def kraken_version() -> str:
    """The installed kraken version (no kraken import needed).

    :return: e.g. ``7.1.1``.
    :rtype: str
    """
    return importlib.metadata.version("kraken")


def quiet_kraken() -> None:
    """Silence kraken's and torch's per-page warnings and info logging for a batch run.

    :return: None.
    :rtype: None
    """
    warnings.filterwarnings("ignore")
    logging.getLogger("kraken").setLevel(logging.ERROR)


def load_segmenter(spec: str) -> Any:
    """Load a baseline segmenter by path, or kraken's bundled default.

    :param spec: ``default`` or a model path (``.mlmodel`` / ``.safetensors``).
    :type spec: str
    :return: ``SegmentationTaskModel``.
    :rtype: Any
    """
    from kraken.models import load_models
    from kraken.tasks.segmentation import SegmentationTaskModel

    if spec == "default":
        return SegmentationTaskModel.load_model()
    return SegmentationTaskModel(load_models(spec, tasks=["segmentation"]))


def load_reader(segmenter: str, recogniser: str, seg_input: str = "rgb", rec_input: str = "rgb",
                threads: int = 4) -> KrakenReader:
    """Load a segmenter + recogniser pair for CPU inference.

    :param segmenter: Segmenter path or ``default``.
    :type segmenter: str
    :param recogniser: Recognition model path (e.g. the PP-OCRv6 ``.safetensors``).
    :type recogniser: str
    :param seg_input: ``rgb`` or ``binarized``.
    :type seg_input: str
    :param rec_input: ``rgb`` or ``binarized``.
    :type rec_input: str
    :param threads: torch intra-op threads.
    :type threads: int
    :return: The loaded reader.
    :rtype: KrakenReader
    :raises ValueError: On an unknown input mode.
    """
    for mode in (seg_input, rec_input):
        if mode not in INPUT_MODES:
            raise ValueError(f"input mode must be one of {INPUT_MODES}, got {mode!r}")
    import torch
    from kraken.configs import SegmentationInferenceConfig
    from kraken.models import load_models
    from kraken.tasks.recognition import RecognitionInferenceConfig, RecognitionTaskModel

    torch.set_num_threads(threads)
    seg_config = SegmentationInferenceConfig(text_direction=TEXT_DIRECTION, accelerator="cpu", device=1,
                                             precision="32-true")
    rec_model = RecognitionTaskModel(load_models(recogniser, tasks=["recognition"]))
    return KrakenReader(segmenter=load_segmenter(segmenter), seg_config=seg_config, recogniser=rec_model,
                        rec_config=RecognitionInferenceConfig(), seg_input=seg_input, rec_input=rec_input)


def read_page(reader: KrakenReader, image_path: Path) -> PageRead:
    """Segment and recognise one page image.

    :param reader: Loaded segmenter + recogniser.
    :type reader: KrakenReader
    :param image_path: Page image.
    :type image_path: Path
    :return: Lines in the segmenter's order with page-pixel geometry.
    :rtype: PageRead
    """
    from PIL import Image
    from kraken import binarization

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    binarized = binarization.nlbin(rgb) if "binarized" in (reader.seg_input, reader.rec_input) else None
    segmentation = reader.segmenter.predict(binarized if reader.seg_input == "binarized" else rgb, reader.seg_config)
    n_lines = len(segmentation.lines)
    segmentation.lines = keep_bounded_lines(segmentation.lines)
    records: List[Any] = []
    if segmentation.lines:
        records = list(reader.recogniser.predict(binarized if reader.rec_input == "binarized" else rgb,
                                                 segmentation, reader.rec_config))
    return PageRead(lines=[line_read(record) for record in records], n_lines=n_lines,
                    n_dropped=n_lines - len(segmentation.lines), image_size=rgb.size)


def add_reader_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options shared by the Kraken readers (models, inputs, threads, tag, limit, source).

    :param parser: Parser to extend.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    parser.add_argument("--tag", required=True, help="run name: output file / directory name")
    parser.add_argument("--segmenter", required=True,
                        help="baseline segmentation model path, or 'default' for kraken's bundled segmenter")
    parser.add_argument("--recogniser", required=True, help="recognition model path (.safetensors or .mlmodel)")
    parser.add_argument("--seg-input", choices=INPUT_MODES, default="rgb", help="image given to the segmenter")
    parser.add_argument("--rec-input", choices=INPUT_MODES, default="rgb", help="image given to the recogniser")
    parser.add_argument("--limit", type=int, default=0, help="only the first N documents / pages (0 = all)")
    parser.add_argument("--threads", type=int, default=4, help="torch intra-op threads")
    parser.add_argument("--source", default="",
                        help="provenance string stored with every output (default: built from models and inputs)")


def reader_from_args(args: argparse.Namespace) -> KrakenReader:
    """Load the reader the parsed options describe.

    :param args: Parsed options from :func:`add_reader_arguments`.
    :type args: argparse.Namespace
    :return: The loaded reader.
    :rtype: KrakenReader
    """
    return load_reader(args.segmenter, args.recogniser, args.seg_input, args.rec_input, args.threads)


def source_from_args(args: argparse.Namespace, version: Optional[str] = None) -> str:
    """The provenance string: ``--source`` when given, else :func:`default_source`.

    :param args: Parsed options from :func:`add_reader_arguments`.
    :type args: argparse.Namespace
    :param version: kraken version (default: the installed one).
    :type version: Optional[str]
    :return: Provenance string.
    :rtype: str
    """
    if args.source:
        return args.source
    return default_source(args.segmenter, args.recogniser, args.seg_input, args.rec_input, version or kraken_version())


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: read the selected benchmark images and append the answers.

    :param argv: Arguments (default: ``sys.argv``).
    :type argv: Optional[List[str]]
    :return: None.
    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCHMARK,
                        help="benchmark directory with benchmark.jsonl and images/ (default: the NAS copy)")
    parser.add_argument("--ids-from", type=Path, default=None,
                        help="answers jsonl whose doc_ids, in first-seen order, select the documents")
    add_reader_arguments(parser)
    args = parser.parse_args(argv)
    quiet_kraken()

    records = load_records(args.bench)
    ids, unknown = select_ids(records, doc_ids_from(args.ids_from), args.limit)
    if unknown:
        emit(f"warning: {len(unknown)} requested ids are not in the benchmark, skipped (e.g. {unknown[0]})")
    out_path = output_path(args.bench, args.tag)
    done = done_pairs(out_path)
    jobs = build_jobs(records, ids, done)
    emit(f"{args.tag}: {len(ids)} docs, {len(jobs)} images to read ({len(done)} answered before)")
    if not jobs:
        return
    reader = reader_from_args(args)
    started = time.perf_counter()
    stats = run_jobs(jobs, partial(read_page, reader), args.bench / "images", out_path, source_from_args(args))
    emit(f"done: {stats} in {(time.perf_counter() - started) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
