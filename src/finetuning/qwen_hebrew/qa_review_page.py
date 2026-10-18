# File name: qa_review_page.py
# Date: 9/24/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Build the ``pgp_qa_v1`` review page: page image, answer-line box and transcription per QA row.

One self-contained HTML page for reviewing the extractive QA rows that
:mod:`src.finetuning.qwen_hebrew.build_pgp_qa` wrote: ``qa_review_template.html`` (next to this
module) with every row inlined as JSON. A row joins its QA record (``pgp_qa_v1/manifest.jsonl``)
with the edition page it was drawn from (``pgp_editions_v1/manifest.jsonl``, keyed by
``(canonical_id, image_index)``) and carries one highlight box per answer line, taken from the
page's two-reader evidence in the consensus raw cache
(``ai_reads/raw/<canonical_id>__<image_index>__<vlm_model>.json``):

* a Kraken row first: the page's Kraken fragments grouped into rows (:func:`rows_from_frags`);
  the row most similar to the edition line wins when its letters-only similarity
  (:func:`src.datasets.consensus.line_rule.similarity`) is >= 0.5;
* else the most similar VLM line box, when its similarity is >= 0.6;
* else no box (the transcription pane still highlights the line).

Boxes are in the readers' 0-1000 page units. Verdicts given on the page (good / wrong / unsure)
stay in the browser's localStorage; "Copy verdicts" exports them as JSON. The ``qa-review``
entry of ``.claude/launch.json`` serves the NAS folder on port 8766.

Usage (repo root, after ``build_pgp_qa``)::

    .venv/bin/python -m src.finetuning.qwen_hebrew.qa_review_page [--data-json rows.json]
"""
import argparse
import collections
import functools
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.datasets.consensus.line_rule import similarity

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
TEMPLATE_PATH = _HERE / "qa_review_template.html"
DATA_PLACEHOLDER = "__DATA__"
NAS_DATASETS = Path("/Volumes/home/studio_offload/datasets")
DEFAULT_QA_MANIFEST = NAS_DATASETS / "pgp_qa_v1/manifest.jsonl"
DEFAULT_EDITIONS_MANIFEST = NAS_DATASETS / "pgp_editions_v1/manifest.jsonl"
DEFAULT_RAW_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/ai_reads/raw"
DEFAULT_VLM_MODEL = "qwen3-vl-8b-heb-v21b-step1200"
DEFAULT_OUT = NAS_DATASETS / "pgp_qa_v1/qa_review.html"
KRAKEN_MIN_SIM = 0.5   # a Kraken row at least this similar to the edition line gives the box
VLM_MIN_SIM = 0.6      # otherwise a VLM line at least this similar does
_SCRIPT_CLOSE = re.compile(r"</(?=script)", re.IGNORECASE)

Box = List[float]
KrakenRow = Tuple[str, Box]
Evidence = Tuple[List[Dict[str, Any]], List[KrakenRow]]   # (VLM lines, Kraken rows) of one page
EvidenceLookup = Callable[[str, int], Optional[Evidence]]


# ----------------------------------------------------------------------------- inputs

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file.

    :param path: File with one JSON object per line (blank lines are skipped).
    :returns: The records in file order.
    """
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_pages(pages: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Key edition-manifest pages by ``(canonical_id, image_index)``.

    :param pages: ``pgp_editions_v1`` manifest records.
    :returns: ``(canonical_id, image_index) -> record`` (a later duplicate replaces an earlier one).
    """
    return {(m["canonical_id"], m["image_index"]): m for m in pages}


def _y_centre(frag: Dict[str, Any]) -> float:
    """Vertical centre of a fragment box.

    :param frag: Fragment with ``box = [x1, y1, x2, y2]``.
    :returns: ``(y1 + y2) / 2``.
    """
    return (frag["box"][1] + frag["box"][3]) / 2


def rows_from_frags(frags: Sequence[Dict[str, Any]]) -> List[KrakenRow]:
    """Group Kraken fragments into text rows.

    Fragments are visited in order of vertical centre; each joins the first row whose vertical
    band holds its centre (the band then grows to cover the fragment), else it opens a new row.
    A row's text is its fragments concatenated right to left (by left edge, the Hebrew reading
    order) and its box is their union.

    :param frags: Kraken fragments ``{"text": str, "box": [x1, y1, x2, y2]}`` (0-1000 units).
    :returns: ``(text, box)`` per row, in the order the rows were opened (top down).
    """
    rows: List[Dict[str, Any]] = []
    for frag in sorted(frags, key=_y_centre):
        yc = _y_centre(frag)
        for row in rows:
            if row["y1"] <= yc <= row["y2"]:
                row["frags"].append(frag)
                row["y1"] = min(row["y1"], frag["box"][1])
                row["y2"] = max(row["y2"], frag["box"][3])
                break
        else:
            rows.append({"y1": frag["box"][1], "y2": frag["box"][3], "frags": [frag]})
    out: List[KrakenRow] = []
    for row in rows:
        fs = sorted(row["frags"], key=lambda f: -f["box"][0])
        box = [min(f["box"][0] for f in fs), row["y1"], max(f["box"][2] for f in fs), row["y2"]]
        out.append(("".join(f["text"] for f in fs), box))
    return out


def load_evidence(raw_dir: Path, canonical_id: str, image_index: int, vlm_model: str) -> Optional[Evidence]:
    """Read one page's reader evidence from the consensus raw cache.

    :param raw_dir: The ``ai_reads/raw`` directory.
    :param canonical_id: Page document id.
    :param image_index: Page image index.
    :param vlm_model: VLM model name in the cache file name.
    :returns: ``(vlm_lines, kraken_rows)``, or None when the page has no cache file.
    """
    path = raw_dir / f"{canonical_id}__{image_index}__{vlm_model}.json"
    if not path.exists():
        return None
    cache = json.loads(path.read_text(encoding="utf-8"))
    return cache.get("vlm_lines") or [], rows_from_frags(cache.get("frags") or [])


def evidence_loader(raw_dir: Path, vlm_model: str) -> EvidenceLookup:
    """A memoised :func:`load_evidence` (QA rows share pages, so each cache file is read once).

    :param raw_dir: The ``ai_reads/raw`` directory.
    :param vlm_model: VLM model name in the cache file names.
    :returns: ``(canonical_id, image_index) -> evidence or None``.
    """
    @functools.lru_cache(maxsize=None)
    def lookup(canonical_id: str, image_index: int) -> Optional[Evidence]:
        """Cached :func:`load_evidence` for one page.

        :param canonical_id: Page document id.
        :param image_index: Page image index.
        :returns: The page's evidence, or None.
        """
        return load_evidence(raw_dir, canonical_id, image_index, vlm_model)
    return lookup


# ----------------------------------------------------------------------------- boxes

def best_match(line_text: str, candidates: Iterable[Tuple[str, Box]]) -> Tuple[float, Optional[Box]]:
    """The candidate reading most similar to an edition line.

    :param line_text: The edition line.
    :param candidates: ``(text, box)`` readings.
    :returns: ``(similarity, box)`` of the best one (the first on ties); ``(0.0, None)`` if none.
    """
    return max(((similarity(line_text, text), box) for text, box in candidates),
               default=(0.0, None), key=lambda x: x[0])


def pick_box(line_text: str, evidence: Optional[Evidence]) -> Optional[Dict[str, Any]]:
    """Choose the highlight box of one edition line: a Kraken row, else a VLM line, else none.

    The most similar Kraken row wins when its similarity is >= :data:`KRAKEN_MIN_SIM`, even if a
    VLM line is more similar; otherwise the most similar VLM line when >= :data:`VLM_MIN_SIM`.

    :param line_text: The edition line the answer quotes.
    :param evidence: The page's ``(vlm_lines, kraken_rows)``, or None when it has no reads.
    :returns: ``{"src": "kraken" | "vlm", "box": [x1, y1, x2, y2] (ints), "sim": similarity to
        2 places}``, or None.
    """
    if evidence is None:
        return None
    vlm_lines, kraken_rows = evidence
    sim, box = best_match(line_text, kraken_rows)
    if sim >= KRAKEN_MIN_SIM:
        return {"src": "kraken", "box": [round(v) for v in box], "sim": round(sim, 2)}
    sim, box = best_match(line_text, ((v["text"], v["box"]) for v in vlm_lines))
    if sim >= VLM_MIN_SIM:
        return {"src": "vlm", "box": [round(v) for v in box], "sim": round(sim, 2)}
    return None


# ----------------------------------------------------------------------------- rows

def parse_answer(answer: str) -> Any:
    """Decode a QA answer as stored in the manifest.

    :param answer: The manifest ``answer`` string.
    :returns: The parsed JSON when it starts with ``{`` or ``[``, else ``{"answer": answer}``.
    """
    return json.loads(answer) if answer.startswith(("{", "[")) else {"answer": answer}


def target_lines(answer: Any) -> List[Any]:
    """The one-based page lines an answer quotes.

    :param answer: A parsed answer: ``{"line": N, "text": ...}``, a list of those, or an answer
        without lines (e.g. ``{"answer": "not stated"}``).
    :returns: One ``line`` value per quoted item (None for a list item without one), in order.
    """
    if isinstance(answer, list):
        return [a.get("line") for a in answer if isinstance(a, dict)]
    if isinstance(answer, dict) and "line" in answer:
        return [answer["line"]]
    return []


def review_row(qa: Dict[str, Any], page: Dict[str, Any], answer: Any,
               boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One page row: the fields ``qa_review_template.html`` reads, in a fixed order.

    :param qa: QA manifest record.
    :param page: Its edition manifest record.
    :param answer: The parsed answer.
    :param boxes: Highlight boxes, one per answer line that has one.
    :returns: The row.
    """
    return {"id": qa["stem"], "family": qa["family"], "section": qa.get("section"),
            "doc": qa["canonical_id"], "pgpid": qa["pgpid"], "img": qa["image_url"],
            "w": page["image_width"], "h": page["image_height"], "split": qa["split"],
            "question": qa["question"], "answer": answer, "line": qa.get("line"),
            "lines": page["lines"], "evidence": qa.get("evidence"), "boxes": boxes,
            "side": page.get("side"), "source": page.get("edition_source")}


def build_rows(qa_records: Iterable[Dict[str, Any]], pages: Dict[Tuple[str, int], Dict[str, Any]],
               evidence_for: EvidenceLookup) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """Build the review rows and count where their boxes came from.

    QA records whose page is not in ``pages`` are dropped; answer lines outside the page are
    skipped and not counted.

    :param qa_records: ``pgp_qa_v1`` manifest records.
    :param pages: Edition pages from :func:`index_pages`.
    :param evidence_for: ``(canonical_id, image_index) -> evidence or None``.
    :returns: ``(rows, box sources)``; the counter has ``kraken`` / ``vlm`` per box and ``none``
        per answer line without one.
    """
    rows: List[Dict[str, Any]] = []
    sources: collections.Counter = collections.Counter()
    for qa in qa_records:
        page = pages.get((qa["canonical_id"], qa["image_index"]))
        if not page:
            continue
        lines = page["lines"]
        answer = parse_answer(qa["answer"])
        evidence = evidence_for(qa["canonical_id"], qa["image_index"])
        boxes: List[Dict[str, Any]] = []
        for n in target_lines(answer):
            if not n or n < 1 or n > len(lines):
                continue
            choice = pick_box(lines[n - 1], evidence)
            sources[choice["src"] if choice else "none"] += 1
            if choice:
                boxes.append({"line": n, **choice})
        rows.append(review_row(qa, page, answer, boxes))
    return rows, sources


def build_review_rows(qa_manifest: Path, editions_manifest: Path, raw_dir: Path,
                      vlm_model: str) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """Read both manifests and build the review rows.

    :param qa_manifest: ``pgp_qa_v1/manifest.jsonl``.
    :param editions_manifest: ``pgp_editions_v1/manifest.jsonl``.
    :param raw_dir: Consensus raw cache (``ai_reads/raw``).
    :param vlm_model: VLM model name in the cache file names.
    :returns: ``(rows, box sources)`` as from :func:`build_rows`.
    """
    pages = index_pages(read_jsonl(editions_manifest))
    return build_rows(read_jsonl(qa_manifest), pages, evidence_loader(raw_dir, vlm_model))


# ----------------------------------------------------------------------------- page

def rows_json(rows: Sequence[Dict[str, Any]]) -> str:
    """Serialise the rows as the page's data (UTF-8 text, not ASCII-escaped).

    :param rows: Review rows.
    :returns: JSON text.
    """
    return json.dumps(rows, ensure_ascii=False)


def escape_script(text: str) -> str:
    """Make JSON text safe to inline in a ``<script>`` element.

    Every ``</script`` (any case) becomes ``<\\/script``, so the data cannot close the element;
    ``\\/`` is the JSON escape of ``/``, so the text still parses to the same value.

    :param text: JSON text.
    :returns: The escaped text.
    """
    return _SCRIPT_CLOSE.sub(r"<\\/", text)


def load_template(path: Path = TEMPLATE_PATH) -> str:
    """Read the page template.

    :param path: Template HTML holding one :data:`DATA_PLACEHOLDER`.
    :returns: The template text.
    """
    return path.read_text(encoding="utf-8")


def render_page(rows: Sequence[Dict[str, Any]], template: str) -> str:
    """Inline the rows into the page template.

    :param rows: Review rows.
    :param template: Template HTML holding :data:`DATA_PLACEHOLDER` once, inside
        ``<script id="data" type="application/json">``.
    :returns: The standalone page.
    :raises ValueError: If the template does not hold the placeholder exactly once.
    """
    if template.count(DATA_PLACEHOLDER) != 1:
        raise ValueError(f"the template must hold {DATA_PLACEHOLDER} exactly once")
    return template.replace(DATA_PLACEHOLDER, escape_script(rows_json(rows)))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: build the rows, write the page and optionally the rows as JSON.

    :param argv: Command-line arguments (default: ``sys.argv[1:]``).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa-manifest", type=Path, default=DEFAULT_QA_MANIFEST, help="pgp_qa_v1 manifest.jsonl")
    ap.add_argument("--editions-manifest", type=Path, default=DEFAULT_EDITIONS_MANIFEST,
                    help="pgp_editions_v1 manifest.jsonl")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="consensus raw cache (ai_reads/raw)")
    ap.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL, help="VLM model name in the raw cache file names")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="review page to write")
    ap.add_argument("--data-json", type=Path, default=None, help="also write the rows as JSON here")
    args = ap.parse_args(argv)
    if not args.raw_dir.is_dir():
        ap.error(f"--raw-dir {args.raw_dir} is not a directory (every answer line would have no box)")
    rows, sources = build_review_rows(args.qa_manifest, args.editions_manifest, args.raw_dir, args.vlm_model)
    if args.data_json:
        args.data_json.parent.mkdir(parents=True, exist_ok=True)
        args.data_json.write_text(rows_json(rows), encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_page(rows, load_template()), encoding="utf-8")
    logger.info("rows: %d | box sources: %s | families: %s", len(rows), dict(sources),
                dict(collections.Counter(r["family"] for r in rows)))
    logger.info("wrote %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
