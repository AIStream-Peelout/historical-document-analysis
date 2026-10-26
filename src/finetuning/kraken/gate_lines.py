# File name: gate_lines.py
# Date: 10/19/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Pre-training gate for compiled Kraken line datasets: look at what ``ketos train`` will read.

``ketos compile`` turns PAGE XML (or pre-cropped line images) into an Arrow IPC file of extracted line
images.  Kraken 7.1.1 writes this layout (checked on an arrow compiled from Agapet PAGE XML)::

    lines: struct<text: string, im: binary (PNG bytes), language: list<string>>
    train: bool    validation: bool    test: bool
    schema metadata b"lines": JSON {type: kraken_recognition_baseline | kraken_recognition_bbox,
                                     alphabet: {char: count}, counts: {all, train, validation, test}, ...}

Three commands:

* ``sample`` writes ``--n`` seeded-random rows of each arrow to ``<out-root>/<arrow stem>/`` as
  ``row_<index>.png`` with ``index.tsv`` (row, text, width, height, file) and an ``index.html`` contact
  sheet (image above its right-to-left text), so a human can check images against their text before
  training.
* ``count`` prints, per arrow, the dataset type, the row count, every alphabet symbol the TEXT RULES of
  ``build_arabic_lines.py`` forbid (tatweel, harakat U+064B to U+0652, bidi controls, Latin letters,
  editorial brackets, whitespace other than U+0020) as ``TEXT RULE violation`` and every other symbol
  outside the Arabic blocks.  It fails with status 4 on a violation (a builder regression) and with
  status 3 when baseline and bbox arrows are mixed (``ketos train`` refuses such a run).
* ``early-check`` prints the lines of the installed kraken's ``ketos/recognition.py`` and
  ``train/ppocr.py`` that decide when training stops and how the learning rate is scheduled.  It fails
  with status 5 when ``train/ppocr.py`` instantiates no ``EarlyStopping`` (``ketos train -q early`` sets
  ``max_epochs=-1``, so such a run never ends) and with status 6 when it reads
  ``trainer.estimated_stepping_batches`` (Lightning returns ``inf`` for ``max_epochs=-1``).

When an arrow's columns do not match the layout above the schema is printed and the exit status is 2.
Runs inside the kraken-train:k711 container and needs only pyarrow, Pillow and the standard library::

    python /work/gate_lines.py sample --out-root /data/gate --n 30 --seed 0 /data/arrow/agapet_train.arrow
    python /work/gate_lines.py count /data/arrow/*.arrow
    python /work/gate_lines.py early-check
"""
import argparse
import bisect
import html
import importlib.util
import io
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
from PIL import Image

EXIT_SCHEMA = 2
EXIT_MIXED = 3
EXIT_TEXT_RULE = 4
EXIT_NO_EARLY_STOPPING = 5
EXIT_INFINITE_STEPS = 6
IMAGE_FIELDS = ("im", "image")
TEXT_FIELDS = ("text", "transcription")
# Arabic, Arabic Supplement, Arabic Extended-A, Presentation Forms-A and -B.
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
# Symbols the TEXT RULES remove (tatweel, harakat, bidi controls) or whose lines they drop (Latin letters,
# editorial brackets).  Mirrors _STRIP, _BIDI, _LATIN and _EDITORIAL of build_arabic_lines.py by hand: the
# container has neither lxml nor numpy, so that module cannot be imported here.  Several of these lie inside
# ARABIC_RANGES (U+0640, U+064B to U+0652, U+061C, U+FEFF), so they are tested first.
FORBIDDEN_RANGES = (
    (0x0640, 0x0640), (0x064B, 0x0652),  # tatweel; tanwin, short vowels, shadda, sukun
    (0x061C, 0x061C), (0x200E, 0x200F), (0x202A, 0x202E), (0x2066, 0x2069), (0xFEFF, 0xFEFF),  # bidi controls
    (0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00D6), (0x00D8, 0x00F6), (0x00F8, 0x024F),  # Latin letters
    (0x1E00, 0x1EFF),
)
EDITORIAL_MARKS = frozenset("[]<>{}〈〉⟨⟩〈〉")
ROW_GLOB = "row_*.png"
PPOCR_FILE = "train/ppocr.py"
TRAINER_FILES = ("ketos/recognition.py", PPOCR_FILE)
TRAINER_PATTERN = re.compile(r"max_epochs|EarlyStopping|configure_callbacks|estimated_stepping_batches|cos_t_max")


@dataclass(frozen=True)
class LineLayout:
    """Where a compiled arrow keeps each line's image and text.

    :param column: Top-level struct column holding both fields, or ``""`` when they are top-level columns.
    :param image_field: Name of the image-bytes field.
    :param text_field: Name of the transcription field.
    """

    column: str
    image_field: str
    text_field: str


@dataclass(frozen=True)
class SampleRow:
    """One dumped line.

    :param row: Row index in the arrow file.
    :param text: Transcription as stored.
    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :param file: PNG file name inside the sample directory.
    """

    row: int
    text: str
    width: int
    height: int
    file: str


@dataclass(frozen=True)
class ArrowCount:
    """Size and type of one compiled arrow.

    :param path: Arrow file.
    :param kind: ``type`` from the kraken metadata (``?`` when absent).
    :param rows: Rows actually stored.
    :param meta_all: ``counts.all`` from the kraken metadata (-1 when absent).
    :param alphabet: Symbol counts from the kraken metadata.
    """

    path: str
    kind: str
    rows: int
    meta_all: int
    alphabet: Dict[str, int]


def _pick(names: Sequence[str], wanted: Sequence[str]) -> Optional[str]:
    """First name of ``wanted`` present in ``names``.

    :param names: Available field names.
    :type names: Sequence[str]
    :param wanted: Candidate names in order of preference.
    :type wanted: Sequence[str]
    :return: The first match, or None.
    :rtype: Optional[str]
    """
    return next((w for w in wanted if w in names), None)


def find_layout(schema: pa.Schema) -> Optional[LineLayout]:
    """Locate the image and text fields of a compiled line dataset.

    A struct column with image and text children (kraken's ``lines``) wins over top-level columns.

    :param schema: Arrow schema of the file.
    :type schema: pa.Schema
    :return: The layout, or None when no image/text pair is found.
    :rtype: Optional[LineLayout]
    """
    for field in schema:
        if pa.types.is_struct(field.type):
            children = [field.type.field(i).name for i in range(field.type.num_fields)]
            im, text = _pick(children, IMAGE_FIELDS), _pick(children, TEXT_FIELDS)
            if im and text:
                return LineLayout(field.name, im, text)
    im, text = _pick(schema.names, IMAGE_FIELDS), _pick(schema.names, TEXT_FIELDS)
    return LineLayout("", im, text) if im and text else None


def kraken_metadata(schema: pa.Schema) -> Dict[str, Any]:
    """Kraken's dataset metadata record (``b"lines"`` schema metadata), decoded.

    :param schema: Arrow schema of the file.
    :type schema: pa.Schema
    :return: The metadata dict, or ``{}`` when the file carries none.
    :rtype: Dict[str, Any]
    """
    raw = (schema.metadata or {}).get(b"lines")
    return json.loads(raw) if raw else {}


def sample_indices(total: int, n: int, seed: int) -> List[int]:
    """Sorted seeded-random row indices.

    :param total: Rows in the file.
    :type total: int
    :param n: Rows wanted (all rows when the file is smaller).
    :type n: int
    :param seed: Random seed.
    :type seed: int
    :return: ``min(n, total)`` distinct indices in ascending order.
    :rtype: List[int]
    """
    return sorted(random.Random(seed).sample(range(total), min(max(n, 0), total)))


def locate_rows(batch_sizes: Sequence[int], rows: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Map global row indices to ``(row, batch index, offset in batch)``.

    :param batch_sizes: Row count of every record batch, in file order.
    :type batch_sizes: Sequence[int]
    :param rows: Global row indices, each below ``sum(batch_sizes)``.
    :type rows: Sequence[int]
    :return: One triple per requested row, in the order given.
    :rtype: List[Tuple[int, int, int]]
    """
    starts: List[int] = []
    total = 0
    for size in batch_sizes:
        starts.append(total)
        total += size
    out = []
    for row in rows:
        if not 0 <= row < total:
            raise IndexError(f"row {row} outside 0..{total - 1}")
        batch = bisect.bisect_right(starts, row) - 1  # last batch starting at or before row: skips empty ones
        out.append((row, batch, row - starts[batch]))
    return out


def tsv_field(text: str) -> str:
    """Escape a value for one TSV cell (backslash, tab, CR and LF become ``\\\\``, ``\\t``, ``\\r``, ``\\n``).

    :param text: Raw value.
    :type text: str
    :return: Single-line value without tabs.
    :rtype: str
    """
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def as_png(data: bytes) -> Tuple[bytes, int, int]:
    """PNG bytes and size of an encoded image (re-encoded only when it is not already PNG).

    :param data: Encoded image bytes from the arrow.
    :type data: bytes
    :return: ``(png bytes, width, height)``.
    :rtype: Tuple[bytes, int, int]
    """
    with Image.open(io.BytesIO(data)) as im:
        width, height = im.size
        if im.format == "PNG":
            return data, width, height
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue(), width, height


def is_arabic(char: str) -> bool:
    """Whether a symbol lies in one of the Arabic Unicode blocks.

    :param char: One code point.
    :type char: str
    :return: True for Arabic-block code points.
    :rtype: bool
    """
    return len(char) == 1 and any(lo <= ord(char) <= hi for lo, hi in ARABIC_RANGES)


def is_forbidden(char: str) -> bool:
    """Whether a symbol breaks the TEXT RULES of ``build_arabic_lines.py``.

    Forbidden are tatweel, U+064B to U+0652, bidi controls, Latin letters, editorial brackets and any
    whitespace other than U+0020 (the rules collapse every whitespace run to one space).  A multi-character
    key is forbidden when any of its characters is.

    :param char: One alphabet key (normally one code point).
    :type char: str
    :return: True when the symbol should never reach a compiled arrow.
    :rtype: bool
    """
    if len(char) != 1:
        return any(is_forbidden(c) for c in char)
    code = ord(char)
    return (
        char in EDITORIAL_MARKS
        or (char.isspace() and char != " ")
        or any(lo <= code <= hi for lo, hi in FORBIDDEN_RANGES)
    )


def symbol_label(char: str) -> str:
    """Readable label of an alphabet key.

    :param char: One alphabet key.
    :type char: str
    :return: ``U+XXXX NAME 'c'`` (one code and name per character).
    :rtype: str
    """
    names = " ".join(unicodedata.name(c, "?") for c in char)
    codes = " ".join(f"U+{ord(c):04X}" for c in char)
    return f"{codes} {names} {char!r}"


def _by_count(alphabet: Dict[str, int]) -> List[Tuple[str, int]]:
    """Alphabet items, most frequent first, ties by symbol.

    :param alphabet: Symbol counts from the kraken metadata.
    :type alphabet: Dict[str, int]
    :return: ``(symbol, count)`` pairs.
    :rtype: List[Tuple[str, int]]
    """
    return sorted(alphabet.items(), key=lambda kv: (-kv[1], kv[0]))


def text_rule_violations(alphabet: Dict[str, int]) -> List[Tuple[str, int]]:
    """Alphabet symbols the TEXT RULES forbid (see :func:`is_forbidden`), most frequent first.

    :param alphabet: Symbol counts from the kraken metadata.
    :type alphabet: Dict[str, int]
    :return: ``(label, count)`` pairs; labels read ``U+XXXX NAME 'c'``.
    :rtype: List[Tuple[str, int]]
    """
    return [(symbol_label(c), n) for c, n in _by_count(alphabet) if is_forbidden(c)]


def alphabet_outliers(alphabet: Dict[str, int]) -> List[Tuple[str, int]]:
    """Allowed alphabet symbols outside the Arabic blocks (digits, punctuation ...), most frequent first.

    Space and the symbols reported by :func:`text_rule_violations` are excluded.

    :param alphabet: Symbol counts from the kraken metadata.
    :type alphabet: Dict[str, int]
    :return: ``(label, count)`` pairs; labels read ``U+XXXX NAME 'c'``.
    :rtype: List[Tuple[str, int]]
    """
    return [
        (symbol_label(c), n) for c, n in _by_count(alphabet)
        if not (c == " " or is_forbidden(c) or is_arabic(c))
    ]


def html_page(title: str, rows: Sequence[SampleRow]) -> str:
    """Contact sheet: each line image above its right-to-left transcription.

    :param title: Page heading (the arrow name).
    :type title: str
    :param rows: Dumped lines.
    :type rows: Sequence[SampleRow]
    :return: A standalone HTML document.
    :rtype: str
    """
    cards = "\n".join(
        f'<figure><img src="{html.escape(r.file)}" alt="row {r.row}">'
        f'<figcaption><span class="meta">row {r.row} &middot; {r.width}&times;{r.height}</span>'
        f'<div class="text" dir="rtl" lang="ar">{html.escape(r.text)}</div></figcaption></figure>'
        for r in rows
    )
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:sans-serif;margin:16px;background:#fafafa;color:#222}"
        "figure{margin:0 0 20px;padding:8px;background:#fff;border:1px solid #ddd}"
        "img{display:block;max-width:100%;height:auto;margin-left:auto}"
        ".meta{font-size:12px;color:#777}.text{font-size:22px;margin-top:4px;white-space:pre-wrap}"
        f"</style></head><body><h1>{html.escape(title)}</h1>\n{cards}\n</body></html>\n"
    )


def read_lines(path: Path, layout: LineLayout, rows: Sequence[int]) -> List[Tuple[int, Optional[bytes], str]]:
    """Read the image bytes and text of selected rows, touching only the batches that hold them.

    :param path: Arrow IPC file.
    :type path: Path
    :param layout: Where the fields live (from :func:`find_layout`).
    :type layout: LineLayout
    :param rows: Global row indices.
    :type rows: Sequence[int]
    :return: ``(row, image bytes or None, text)`` per requested row, in the order given.
    :rtype: List[Tuple[int, Optional[bytes], str]]
    """
    out = []
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        sizes = [reader.get_batch(i).num_rows for i in range(reader.num_record_batches)]
        for row, b, offset in locate_rows(sizes, rows):
            batch = reader.get_batch(b)
            if layout.column:
                record = batch.column(layout.column)[offset].as_py() or {}
                image, text = record.get(layout.image_field), record.get(layout.text_field)
            else:
                image = batch.column(layout.image_field)[offset].as_py()
                text = batch.column(layout.text_field)[offset].as_py()
            out.append((row, image, text or ""))
    return out


def count_rows(path: Path) -> int:
    """Rows stored in an arrow IPC file.

    :param path: Arrow IPC file.
    :type path: Path
    :return: Sum of the record-batch row counts.
    :rtype: int
    """
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))


def read_schema(path: Path) -> pa.Schema:
    """Schema (with metadata) of an arrow IPC file.

    :param path: Arrow IPC file.
    :type path: Path
    :return: The file schema.
    :rtype: pa.Schema
    """
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).schema


def write_sample(out_dir: Path, title: str, lines: Sequence[Tuple[int, Optional[bytes], str]]) -> List[SampleRow]:
    """Write PNGs, ``index.tsv`` and ``index.html`` for sampled lines, replacing an earlier dump.

    :param out_dir: Sample directory (created; earlier ``row_*.png`` files are removed).
    :type out_dir: Path
    :param title: Heading for the contact sheet.
    :type title: str
    :param lines: ``(row, image bytes or None, text)`` triples; rows without an image are skipped.
    :type lines: Sequence[Tuple[int, Optional[bytes], str]]
    :return: The rows written.
    :rtype: List[SampleRow]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(ROW_GLOB):
        old.unlink()
    written: List[SampleRow] = []
    for row, image, text in lines:
        if not image:
            print(f"  row {row}: no image bytes, skipped", file=sys.stderr)
            continue
        png, width, height = as_png(image)
        name = f"row_{row:07d}.png"
        (out_dir / name).write_bytes(png)
        written.append(SampleRow(row, text, width, height, name))
    tsv = ["row\ttext\twidth\theight\tfile"]
    tsv += [f"{r.row}\t{tsv_field(r.text)}\t{r.width}\t{r.height}\t{r.file}" for r in written]
    (out_dir / "index.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    (out_dir / "index.html").write_text(html_page(title, written), encoding="utf-8")
    return written


def count_arrow(path: Path) -> ArrowCount:
    """Type, size and alphabet of one compiled arrow.

    :param path: Arrow IPC file.
    :type path: Path
    :return: The summary.
    :rtype: ArrowCount
    """
    meta = kraken_metadata(read_schema(path))
    return ArrowCount(
        path=str(path),
        kind=str(meta.get("type", "?")),
        rows=count_rows(path),
        meta_all=int(meta.get("counts", {}).get("all", -1)),
        alphabet=dict(meta.get("alphabet", {})),
    )


def mixed_kinds(counts: Sequence[ArrowCount]) -> bool:
    """Whether the arrows mix dataset types (baseline and bbox cannot share a training run).

    :param counts: Per-arrow summaries.
    :type counts: Sequence[ArrowCount]
    :return: True when more than one type is present.
    :rtype: bool
    """
    return len({c.kind for c in counts}) > 1


def print_schema_mismatch(path: Path, schema: pa.Schema) -> int:
    """Report an arrow whose columns are not the expected image/text pair.

    :param path: Arrow IPC file.
    :type path: Path
    :param schema: Its schema.
    :type schema: pa.Schema
    :return: :data:`EXIT_SCHEMA`.
    :rtype: int
    """
    print(f"{path}: no image/text fields (looked for {IMAGE_FIELDS} / {TEXT_FIELDS}); schema:", file=sys.stderr)
    print(schema.remove_metadata(), file=sys.stderr)
    return EXIT_SCHEMA


def cmd_sample(paths: Sequence[Path], out_root: Path, n: int, seed: int) -> int:
    """``sample`` command: dump ``n`` random lines per arrow.

    :param paths: Arrow files.
    :type paths: Sequence[Path]
    :param out_root: Root of the per-arrow sample directories.
    :type out_root: Path
    :param n: Lines per arrow.
    :type n: int
    :param seed: Random seed (the same seed gives the same rows).
    :type seed: int
    :return: Exit status.
    :rtype: int
    """
    for path in paths:
        schema = read_schema(path)
        layout = find_layout(schema)
        if layout is None:
            return print_schema_mismatch(path, schema)
        total = count_rows(path)
        lines = read_lines(path, layout, sample_indices(total, n, seed))
        out_dir = out_root / path.stem
        written = write_sample(out_dir, path.name, lines)
        kind = kraken_metadata(schema).get("type", "?")
        print(f"{path}: {kind}, {len(written)} of {total} lines -> {out_dir}/index.html")
    return 0


def cmd_count(paths: Sequence[Path]) -> int:
    """``count`` command: rows, type, TEXT RULE violations and other non-Arabic symbols per arrow.

    Both errors are reported when both occur; the status then names the TEXT RULE violation.

    :param paths: Arrow files.
    :type paths: Sequence[Path]
    :return: Exit status: :data:`EXIT_TEXT_RULE` when an alphabet holds a forbidden symbol, else
        :data:`EXIT_MIXED` when baseline and bbox arrows are mixed, else 0.
    :rtype: int
    """
    counts = [count_arrow(p) for p in paths]
    violated: List[str] = []
    print("arrow\ttype\trows\tmeta_all\talphabet")
    for c in counts:
        print(f"{c.path}\t{c.kind}\t{c.rows}\t{c.meta_all}\t{len(c.alphabet)}")
        violations = text_rule_violations(c.alphabet)
        for label, num in violations:
            print(f"    TEXT RULE violation {label}: {num}")
        for label, num in alphabet_outliers(c.alphabet):
            print(f"    non-Arabic symbol {label}: {num}")
        if violations:
            violated.append(c.path)
    print(f"total\t-\t{sum(c.rows for c in counts)}\t-\t-")
    mixed = mixed_kinds(counts)
    if violated:
        print(f"ERROR: TEXT RULE violations in {violated}: build_arabic_lines.py let forbidden symbols through", file=sys.stderr)
    if mixed:
        print(f"ERROR: mixed dataset types {sorted({c.kind for c in counts})}; ketos train refuses them", file=sys.stderr)
    if violated:
        return EXIT_TEXT_RULE
    return EXIT_MIXED if mixed else 0


def kraken_root() -> Path:
    """Directory of the installed kraken package, found without importing it (an import pulls in torch).

    :return: The package directory.
    :rtype: Path
    """
    spec = importlib.util.find_spec("kraken")
    if spec is None or spec.origin is None:
        raise SystemExit("kraken is not installed in this interpreter")
    return Path(spec.origin).parent


def trainer_hooks(root: Path, files: Sequence[str] = TRAINER_FILES) -> List[Tuple[str, int, str]]:
    """Lines of kraken's training code that decide when a run stops and how the LR is scheduled.

    :param root: kraken package directory.
    :type root: Path
    :param files: Files to scan, relative to ``root``.
    :type files: Sequence[str]
    :return: ``(file, line number, stripped line)`` for every line matching :data:`TRAINER_PATTERN`.
    :rtype: List[Tuple[str, int, str]]
    """
    out = []
    for rel in files:
        for num, line in enumerate((root / rel).read_text(encoding="utf-8").splitlines(), 1):
            if TRAINER_PATTERN.search(line):
                out.append((rel, num, line.strip()))
    return out


def early_stop_status(hooks: Sequence[Tuple[str, int, str]]) -> int:
    """Whether ``ketos train -q early`` can stop a PP-OCRv6 run cleanly, judged from :func:`trainer_hooks`.

    :param hooks: ``(file, line number, line)`` triples.
    :type hooks: Sequence[Tuple[str, int, str]]
    :return: :data:`EXIT_NO_EARLY_STOPPING` when ``train/ppocr.py`` instantiates no ``EarlyStopping``,
        else :data:`EXIT_INFINITE_STEPS` when it reads ``estimated_stepping_batches``, else 0.
    :rtype: int
    """
    ppocr = [text for rel, _, text in hooks if rel == PPOCR_FILE]
    if not any("EarlyStopping(" in text for text in ppocr):
        return EXIT_NO_EARLY_STOPPING
    if any("estimated_stepping_batches" in text for text in ppocr):
        return EXIT_INFINITE_STEPS
    return 0


def cmd_early_check(root: Optional[Path]) -> int:
    """``early-check`` command: print kraken's stop/schedule lines and judge ``-q early`` for PP-OCRv6.

    :param root: kraken package directory (default: the installed package).
    :type root: Optional[Path]
    :return: Exit status from :func:`early_stop_status`.
    :rtype: int
    """
    root = root or kraken_root()
    hooks = trainer_hooks(root)
    for rel, num, text in hooks:
        print(f"{rel}:{num}:{text}")
    status = early_stop_status(hooks)
    if status == EXIT_NO_EARLY_STOPPING:
        print(f"ERROR: {root / PPOCR_FILE} instantiates no EarlyStopping; -q early sets max_epochs=-1, "
              "so a PP-OCRv6 run would never stop", file=sys.stderr)
    elif status == EXIT_INFINITE_STEPS:
        print(f"ERROR: {root / PPOCR_FILE} reads trainer.estimated_stepping_batches, which is inf under -q early "
              "(max_epochs=-1); check how its LR schedule uses it", file=sys.stderr)
    return status


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Command-line arguments.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    :return: Parsed namespace.
    :rtype: argparse.Namespace
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("sample", help="dump random compiled lines (PNG + text) per arrow")
    sp.add_argument("arrows", nargs="+", type=Path)
    sp.add_argument("--out-root", type=Path, required=True, help="per-arrow dirs <out-root>/<arrow stem>/")
    sp.add_argument("--n", type=int, default=30, help="lines per arrow")
    sp.add_argument("--seed", type=int, default=0)
    cp = sub.add_parser("count", help="rows, dataset type, TEXT RULE violations and non-Arabic symbols per arrow")
    cp.add_argument("arrows", nargs="+", type=Path)
    ep = sub.add_parser("early-check", help="can ketos train -q early stop a PP-OCRv6 run (kraken source lines)")
    ep.add_argument("--kraken-root", type=Path, default=None, help="kraken package dir (default: the installed one)")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    :return: Exit status.
    :rtype: int
    """
    args = parse_args(argv)
    if args.command == "sample":
        return cmd_sample(args.arrows, args.out_root, args.n, args.seed)
    if args.command == "early-check":
        return cmd_early_check(args.kraken_root)
    return cmd_count(args.arrows)


if __name__ == "__main__":
    sys.exit(main())
