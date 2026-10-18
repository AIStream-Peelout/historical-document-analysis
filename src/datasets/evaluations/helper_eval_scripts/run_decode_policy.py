# File name: run_decode_policy.py
# Date: 10/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Run the decode policy (``src/datasets/consensus/decode_policy.py``) on PGP-131 (``verified``) or religious-140.

Each page is planned from two things already on disk: the cached Kraken lines (``kraken_lines_k7_ktivseg_e10_rgb_mv.json``
in the page's output folder) and the existing whole-page read (``--fallback-stem``, which is never re-decoded). Then:

* ``keep``: the whole-page read is copied to the new stem, with no model call.
* ``columns``: each column crop is read with the benchmark prompt (``build_fragment_prompt``).
* ``rescue``: the single column's top and bottom halves are read.

A crop read that fails health is read again as bands within the crop (``linewise.build_bands`` / ``_read_band`` /
``_dedup_join``, given the crop's Kraken lines only), and :func:`~src.datasets.consensus.decode_policy.assemble`
picks the text. Each page produces:

* ``<outputs>/<doc_id>/<stem>.txt``, which ``score_genizah_offline.py --benchmark verified`` and
  ``run_religious_benchmark.py --score-only`` pick up automatically;
* a trace at ``transcription_results/decode_policy_details/<benchmark>/<stem>/<doc_id>.json``;
* a timing line on stdout.

The run is resumable: pages whose ``<stem>.txt`` exists are skipped. ``--force`` re-decodes the ``--doc-ids``
pages anyway and overwrites their output and trace (it refuses to run without ``--doc-ids``). Traces keep each
candidate read's letter counts and verdict but not its text, so a page cannot be re-assembled offline after a
policy change: re-decode it with ``--force``. The run is one sequential LM Studio consumer, with crops read one
at a time.

``--dry-run`` plans every page without calling the model and writes a JSON report: actions, whole-page read
health and estimated crop reads. For religious-140 it also gives the column detector's precision and recall
against the spec's ``n_columns``. That field is ground truth, read only by this report and never by the policy.

Usage (repo root)::

    PYTHONPATH=. .venv/bin/python src/datasets/evaluations/helper_eval_scripts/run_decode_policy.py \\
        --benchmark religious --vlm-model qwen3-vl-8b-heb-v21b-step1200 \\
        --stem qwen3_vl_8b_heb_v21b_decode_policy --fallback-stem qwen3_vl_8b_heb_v21b_step1200 [--dry-run]
        [--doc-ids ID1,ID2 --force]
"""
import argparse
import asyncio
import collections
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.datasets.consensus import decode_policy as dp  # noqa: E402
from src.datasets.consensus.consensus_gate import build_fragment_prompt  # noqa: E402
from src.datasets.consensus.linewise import _dedup_join, _read_band, build_bands  # noqa: E402
from src.datasets.consensus.read_health import letter_count  # noqa: E402
from src.models.ocr.lms_transcriber import check_lm_studio_health, transcribe_with_lm_studio  # noqa: E402

KRAKEN_LINES_NAME = "kraken_lines_k7_ktivseg_e10_rgb_mv.json"
DETAILS_DIR = _REPO / "src/datasets/evaluations/transcription_results/decode_policy_details"
CROP_JPEG_QUALITY = 92                       # linewise's crop quality
_STEM_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SCORER_SKIPPED = ("ground_truth", "consensus")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Where one benchmark's files live and how its pages are read.

    :param name: ``verified`` (PGP-131) or ``religious`` (religious-140).
    :param spec: Benchmark JSON (``{"docs": [...]}``).
    :param outputs: Per-page output folders (``<outputs>/<doc_id>/<stem>.txt`` and the Kraken cache).
    :param images: Image folder; None means the spec's ``image`` field holds each page's path.
    :param token_cap: ``max_tokens`` of the benchmark's reads (whole page and crops).
    """

    name: str
    spec: Path
    outputs: Path
    images: Optional[Path]
    token_cap: int


_PGP = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
_REL = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1"
BENCHMARKS: Dict[str, BenchmarkConfig] = {
    "verified": BenchmarkConfig("verified", _PGP / "genizah_test_v1_verified.json",
                                _REPO / "src/datasets/evaluations/transcription_raw_outputs", _PGP / "images", 8192),
    "religious": BenchmarkConfig("religious", _REL / "genizah_religious_v1.json", _REL / "raw_outputs", None, 2500),
}


@dataclass
class Page:
    """One page's inputs as read from disk.

    :param doc_id: Benchmark document id.
    :param kraken: Cached Kraken ``/transcribe_lines`` response.
    :param whole: The existing whole-page read (None when the fallback stem is missing).
    :param image: The page in Kraken's view frame, RGB (None when the image file is missing).
    :param ink_gap: :func:`~src.datasets.consensus.decode_policy.central_ink_gap` of ``image`` (None: no gap / no image).
    """

    doc_id: str
    kraken: Dict[str, Any]
    whole: Optional[str]
    image: Optional[Image.Image]
    ink_gap: Optional[int]


# ----------------------------------------------------------------------------- inputs


def benchmark_config(name: str, spec: Optional[Path] = None, outputs: Optional[Path] = None,
                     images: Optional[Path] = None) -> BenchmarkConfig:
    """A benchmark's configuration, with optional path overrides (tests point these at temporary copies).

    :param name: ``verified`` or ``religious``.
    :type name: str
    :param spec: Override for the benchmark JSON.
    :type spec: Optional[Path]
    :param outputs: Override for the output folder root.
    :type outputs: Optional[Path]
    :param images: Override for the image folder (religious: images are looked up by their file name).
    :type images: Optional[Path]
    :return: The configuration.
    :rtype: BenchmarkConfig
    :raises KeyError: For an unknown benchmark.
    """
    base = BENCHMARKS[name]
    return BenchmarkConfig(base.name, spec or base.spec, outputs or base.outputs, images or base.images, base.token_cap)


def image_path(cfg: BenchmarkConfig, doc: Dict[str, Any]) -> Path:
    """Path of a page's image.

    :param cfg: Benchmark configuration.
    :type cfg: BenchmarkConfig
    :param doc: Spec entry.
    :type doc: Dict[str, Any]
    :return: The spec's ``image`` path (religious), re-rooted under ``cfg.images`` when that is set, or
        ``<images>/<doc_id>.jpg`` (PGP-131).
    :rtype: Path
    """
    if doc.get("image"):
        path = Path(doc["image"])
        return cfg.images / path.name if cfg.images else path
    return cfg.images / f"{doc['doc_id']}.jpg"


def select_docs(docs: Sequence[Dict[str, Any]], doc_ids: str = "", limit: int = 0) -> List[Dict[str, Any]]:
    """The spec entries to process.

    :param docs: All spec entries, in spec order.
    :type docs: Sequence[Dict[str, Any]]
    :param doc_ids: Comma-separated ids to keep (empty = all).
    :type doc_ids: str
    :param limit: Keep at most this many (0 = all).
    :type limit: int
    :return: Selected entries, in spec order.
    :rtype: List[Dict[str, Any]]
    :raises ValueError: When a requested id is not in the spec.
    """
    wanted = [d.strip() for d in doc_ids.split(",") if d.strip()]
    missing = sorted(set(wanted) - {d["doc_id"] for d in docs})
    if missing:
        raise ValueError(f"doc ids not in the benchmark: {missing}")
    chosen = [d for d in docs if not wanted or d["doc_id"] in wanted]
    return chosen[:limit] if limit else chosen


def check_stems(stem: str, fallback_stem: str) -> None:
    """Refuse output stems the scorers would ignore or that would clobber the fallback read.

    :param stem: Output stem.
    :type stem: str
    :param fallback_stem: Existing whole-page stem.
    :type fallback_stem: str
    :raises ValueError: For a stem with path characters, one the scorers skip (``kraken_*``, ``*_flat``,
        ``ground_truth``, ``consensus``), or one equal to the fallback stem.
    """
    for s in (stem, fallback_stem):
        if not _STEM_RE.match(s):
            raise ValueError(f"bad stem {s!r}: letters, digits, '_', '.', '-' only")
    if stem.startswith("kraken_") or stem.endswith("_flat") or stem in _SCORER_SKIPPED:
        raise ValueError(f"stem {stem!r} would be skipped by the scorers")
    if stem == fallback_stem:
        raise ValueError("--stem must differ from --fallback-stem (the fallback read is never overwritten)")


def load_page(cfg: BenchmarkConfig, doc: Dict[str, Any], fallback_stem: str) -> Optional[Page]:
    """Read one page's cached inputs (Kraken lines, whole-page read, image in the view frame, ink gap).

    :param cfg: Benchmark configuration.
    :type cfg: BenchmarkConfig
    :param doc: Spec entry.
    :type doc: Dict[str, Any]
    :param fallback_stem: Stem of the existing whole-page read.
    :type fallback_stem: str
    :return: The page, or None when its Kraken cache is missing.
    :rtype: Optional[Page]
    """
    out_dir = cfg.outputs / doc["doc_id"]
    kraken_file = out_dir / KRAKEN_LINES_NAME
    if not kraken_file.exists():
        return None
    kraken = json.loads(kraken_file.read_text(encoding="utf-8"))
    whole_file = out_dir / f"{fallback_stem}.txt"
    whole = whole_file.read_text(encoding="utf-8", errors="replace") if whole_file.exists() else None
    path = image_path(cfg, doc)
    image = None
    if path.exists():
        with Image.open(path) as im:
            image = dp.view_image(im.convert("RGB"), dp.kraken_rotation(kraken))
    return Page(doc["doc_id"], kraken, whole, image, dp.central_ink_gap(image) if image is not None else None)


# ----------------------------------------------------------------------------- dry run


def bands_per_unit(plan: dp.PagePlan) -> List[int]:
    """Band reads each unit would cost if its crop read failed (``build_bands`` over its Kraken lines, no bisection).

    :param plan: Page plan.
    :type plan: dp.PagePlan
    :return: Band count per unit, in unit order.
    :rtype: List[int]
    """
    width, height = plan.image_size
    return [len(build_bands(u.boxes, width, height)) for u in plan.units]


def plan_record(page: Page, plan: dp.PagePlan) -> Dict[str, Any]:
    """Dry-run record of one page's plan.

    :param page: The page.
    :type page: Page
    :param plan: Its plan.
    :type plan: dp.PagePlan
    :return: Action, reason, detector facts, whole-page health and the read estimate.
    :rtype: Dict[str, Any]
    """
    bands = bands_per_unit(plan)
    whole = plan.whole_page
    return {"doc_id": page.doc_id, "action": plan.action, "reason": plan.reason,
            "kraken_columns": plan.kraken_columns, "n_columns": len(plan.columns), "ink_gap_x": plan.ink_gap_x,
            "ink_gap_in_gutter": plan.ink_gap_in_gutter, "rotation": plan.rotation,
            "image_missing": page.image is None, "kraken_letters": plan.kraken_letters,
            "whole_page_label": whole.label if whole else "missing",
            "whole_page_flags": list(whole.flags) if whole else [],
            "whole_page_letter_ratio": whole.facts.letter_ratio if whole else None,
            "units": [{**u.to_dict(), "n_bands": n} for u, n in zip(plan.units, bands)],
            "reads_base": len(plan.units), "reads_with_bands": len(plan.units) + sum(bands)}


def detector_scores(records: Sequence[Dict[str, Any]], truth: Dict[str, int], key: str) -> Dict[str, Any]:
    """Precision and recall of a two-column call against the spec's column counts.

    :param records: Dry-run page records.
    :type records: Sequence[Dict[str, Any]]
    :param truth: ``doc_id -> n_columns`` from the spec (ground truth; report only).
    :type truth: Dict[str, int]
    :param key: Record field holding the predicted column count (``n_columns`` = the policy's gated call,
        ``kraken_columns`` = Kraken's split alone).
    :type key: str
    :return: ``tp``, ``fp``, ``fn``, ``tn``, ``precision``, ``recall`` and the ids of the errors.
    :rtype: Dict[str, Any]
    """
    rows = [(r["doc_id"], r[key] >= 2, truth[r["doc_id"]] >= 2) for r in records if r["doc_id"] in truth]
    tp = [d for d, p, t in rows if p and t]
    fp = [d for d, p, t in rows if p and not t]
    fn = [d for d, p, t in rows if t and not p]
    tn = sum(1 for _, p, t in rows if not p and not t)
    return {"tp": len(tp), "fp": len(fp), "fn": len(fn), "tn": tn,
            "precision": round(len(tp) / (len(tp) + len(fp)), 3) if tp or fp else None,
            "recall": round(len(tp) / (len(tp) + len(fn)), 3) if tp or fn else None,
            "false_positives": fp, "false_negatives": fn}


def summarize(records: Sequence[Dict[str, Any]], docs: Sequence[Dict[str, Any]], cfg: BenchmarkConfig,
              args: argparse.Namespace, th: dp.PolicyThresholds, skipped: Sequence[str]) -> Dict[str, Any]:
    """The dry-run report.

    :param records: Page records.
    :type records: Sequence[Dict[str, Any]]
    :param docs: Spec entries of the run (the source of ``n_columns`` for the detector scores).
    :type docs: Sequence[Dict[str, Any]]
    :param cfg: Benchmark configuration.
    :type cfg: BenchmarkConfig
    :param args: CLI arguments.
    :type args: argparse.Namespace
    :param th: Policy thresholds.
    :type th: dp.PolicyThresholds
    :param skipped: Ids of pages without a Kraken cache.
    :type skipped: Sequence[str]
    :return: Report dict (JSON-ready).
    :rtype: Dict[str, Any]
    """
    report: Dict[str, Any] = {
        "benchmark": cfg.name, "vlm_model": args.vlm_model, "stem": args.stem, "fallback_stem": args.fallback_stem,
        "thresholds": th.to_dict(), "n_pages": len(records), "skipped_no_kraken": list(skipped),
        "actions": dict(collections.Counter(r["action"] for r in records)),
        "reasons": dict(collections.Counter(r["reason"] for r in records)),
        "whole_page_labels": dict(collections.Counter(r["whole_page_label"] for r in records)),
        "whole_page_flags": dict(collections.Counter(f for r in records for f in r["whole_page_flags"])),
        "whole_page_labels_on_column_pages": dict(collections.Counter(
            r["whole_page_label"] for r in records if r["action"] == dp.ACTION_COLUMNS)),
        "rotated_pages": [r["doc_id"] for r in records if r["rotation"]],
        "images_missing": [r["doc_id"] for r in records if r["image_missing"]],
        "reads": {"base": sum(r["reads_base"] for r in records),
                  "worst_case_with_bands": sum(r["reads_with_bands"] for r in records),
                  "note": "base = one read per planned crop; worst case adds every crop's bands "
                          "(band bisection can add 2 reads per suspect band on top)"},
    }
    truth = {d["doc_id"]: int(d["n_columns"]) for d in docs if "n_columns" in d}
    if truth:
        report["detector"] = {"policy": detector_scores(records, truth, "n_columns"),
                              "kraken_only": detector_scores(records, truth, "kraken_columns")}
    report["pages"] = list(records)
    return report


def format_summary(report: Dict[str, Any]) -> str:
    """Human-readable summary of a dry-run report.

    :param report: Report from :func:`summarize`.
    :type report: Dict[str, Any]
    :return: A few lines of text.
    :rtype: str
    """
    lines = [f"{report['benchmark']}: {report['n_pages']} pages, fallback {report['fallback_stem']}",
             f"  actions: {report['actions']}",
             f"  reasons: {report['reasons']}",
             f"  whole-page labels: {report['whole_page_labels']}; flags: {report['whole_page_flags']}",
             f"  whole-page labels on column pages: {report['whole_page_labels_on_column_pages']}",
             f"  crop reads: base {report['reads']['base']}, worst case with bands "
             f"{report['reads']['worst_case_with_bands']}"]
    for name, det in (report.get("detector") or {}).items():
        lines.append(f"  detector {name}: TP {det['tp']} FP {det['fp']} FN {det['fn']} TN {det['tn']} "
                     f"precision {det['precision']} recall {det['recall']}")
    if report["rotated_pages"]:
        lines.append(f"  rotated Kraken views (planned in the view frame): {len(report['rotated_pages'])}")
    if report["skipped_no_kraken"] or report["images_missing"]:
        lines.append(f"  no Kraken cache: {report['skipped_no_kraken']}; no image: {report['images_missing']}")
    return "\n".join(lines)


def dry_run(docs: Sequence[Dict[str, Any]], cfg: BenchmarkConfig, args: argparse.Namespace,
            th: dp.PolicyThresholds) -> Dict[str, Any]:
    """Plan every page without any model call and write the report.

    :param docs: Spec entries to plan.
    :type docs: Sequence[Dict[str, Any]]
    :param cfg: Benchmark configuration.
    :type cfg: BenchmarkConfig
    :param args: CLI arguments (``report`` overrides the report path).
    :type args: argparse.Namespace
    :param th: Policy thresholds.
    :type th: dp.PolicyThresholds
    :return: The report.
    :rtype: Dict[str, Any]
    """
    records, skipped = [], []
    for doc in docs:
        page = load_page(cfg, doc, args.fallback_stem)
        if page is None:
            skipped.append(doc["doc_id"])
            continue
        records.append(plan_record(page, dp.plan_page(page.kraken, page.whole, th, ink_gap_x=page.ink_gap)))
    report = summarize(records, docs, cfg, args, th, skipped)
    suffix = "" if th.require_ink_gap else "__no_ink_gap"
    path = args.report or args.details_dir / cfg.name / args.stem / f"dry_run__{args.fallback_stem}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(format_summary(report), flush=True)
    print(f"report: {path}", flush=True)
    return report


# ----------------------------------------------------------------------------- real run


async def read_bands(image: Image.Image, unit: dp.ReadUnit, doc_id: str, vlm_model: str,
                     tmp: str) -> Tuple[str, List[Dict[str, Any]], int]:
    """Read a unit as overlapping bands (``linewise._read_band``), given only the unit's own Kraken lines.

    :param image: View-frame page.
    :type image: Image.Image
    :param unit: The unit whose crop read failed.
    :type unit: dp.ReadUnit
    :param doc_id: Document id (prompt).
    :type doc_id: str
    :param vlm_model: LM Studio model id.
    :type vlm_model: str
    :param tmp: Folder for the band crops.
    :type tmp: str
    :return: ``(joined text, per-band records, model calls made)``.
    :rtype: Tuple[str, List[Dict[str, Any]], int]
    """
    texts, details, calls = [], [], 0
    for band in build_bands(unit.boxes, image.width, image.height):
        text, flags = await _read_band(image, band, doc_id, vlm_model, tmp, list(unit.seg_lines))
        texts.append(text)
        details.append({"bbox": [int(v) for v in band], "letters": letter_count(text), "flags": list(flags)})
        calls += 1 + 2 * flags.count("bisect")
    return _dedup_join(texts), details, calls


async def decode_page(page: Page, plan: dp.PagePlan, cfg: BenchmarkConfig, vlm_model: str,
                      th: dp.PolicyThresholds) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """Read a planned page's units (bands for a failed crop) and assemble the page text.

    A crop read fails here exactly when :func:`~src.datasets.consensus.decode_policy.assemble` would reject it,
    the page's weak-Kraken over-generation limit included.

    :param page: The page (with its view-frame image).
    :type page: Page
    :param plan: Its plan.
    :type plan: dp.PagePlan
    :param cfg: Benchmark configuration (``token_cap`` for crop reads).
    :type cfg: BenchmarkConfig
    :param vlm_model: LM Studio model id.
    :type vlm_model: str
    :param th: Policy thresholds.
    :type th: dp.PolicyThresholds
    :return: ``(page text, assemble trace, call counts)``; the counts are ``calls`` (all requests),
        ``crop_calls`` and ``crop_none`` (crop requests that returned nothing).
    :rtype: Tuple[str, Dict[str, Any], Dict[str, int]]
    """
    reads: Dict[str, dp.UnitReads] = {}
    stats = {"calls": 0, "crop_calls": 0, "crop_none": 0}
    runaway = dp.weak_kraken_limit(plan, page.whole, th)        # the over-generation limit assemble() will use
    if plan.action != dp.ACTION_KEEP:
        prompt = build_fragment_prompt(page.doc_id)
        with tempfile.TemporaryDirectory(prefix="decode_policy_") as tmp:
            for unit in plan.units:
                unit_dir = Path(tmp) / unit.name
                unit_dir.mkdir()
                crop_path = unit_dir / "crop.jpg"
                dp.crop_unit(page.image, unit).save(crop_path, "JPEG", quality=CROP_JPEG_QUALITY)
                reply = await transcribe_with_lm_studio(vlm_model, str(crop_path), prompt, max_tokens=cfg.token_cap)
                stats["calls"] += 1
                stats["crop_calls"] += 1
                stats["crop_none"] += reply is None
                unit_reads = dp.UnitReads(crop=(reply or "").strip())
                if dp.judge_read(unit_reads.crop, unit.kraken_texts, th.crop_token_cap, th, runaway).failed:
                    unit_reads.bands, unit_reads.band_details, n = await read_bands(
                        page.image, unit, page.doc_id, vlm_model, str(unit_dir))
                    stats["calls"] += n
                reads[unit.name] = unit_reads
    text, trace = dp.assemble(plan, reads, page.whole, th)
    return text, trace, stats


async def ensure_model_served(vlm_model: str) -> None:
    """Fail unless LM Studio answers and serves the model.

    :param vlm_model: LM Studio model id.
    :type vlm_model: str
    :raises RuntimeError: When LM Studio is unreachable or does not list the model.
    """
    served = await check_lm_studio_health()
    if vlm_model not in served:
        raise RuntimeError(f"{vlm_model} not served by LM Studio: {served}")


async def run_real(docs: Sequence[Dict[str, Any]], cfg: BenchmarkConfig, args: argparse.Namespace,
                   th: dp.PolicyThresholds) -> Dict[str, Any]:
    """Decode every page that has no output yet (every selected page with ``--force``), one request at a time.

    :param docs: Spec entries to process.
    :type docs: Sequence[Dict[str, Any]]
    :param cfg: Benchmark configuration.
    :type cfg: BenchmarkConfig
    :param args: CLI arguments (``force`` re-decodes pages whose output exists, overwriting output and trace).
    :type args: argparse.Namespace
    :param th: Policy thresholds.
    :type th: dp.PolicyThresholds
    :return: Run summary: pages written, skipped, actions, final sources, calls, seconds.
    :rtype: Dict[str, Any]
    :raises RuntimeError: When LM Studio is not serving the model, at the start, or after a page whose every
        crop read came back empty while LM Studio has stopped answering.
    """
    name = f"{args.stem}.txt"
    done = {d["doc_id"] for d in docs if (cfg.outputs / d["doc_id"] / name).exists()}
    todo = list(docs) if args.force else [d for d in docs if d["doc_id"] not in done]
    forced = f" (--force: {len(done)} existing outputs overwritten)" if args.force and done else ""
    print(f"{cfg.name}: {len(docs)} pages, {len(todo)} to decode -> {args.stem}{forced}", flush=True)
    summary: Dict[str, Any] = {"written": 0, "skipped": [], "actions": collections.Counter(),
                               "final_sources": collections.Counter(), "calls": 0, "seconds": 0.0}
    if not todo:
        return summary
    await ensure_model_served(args.vlm_model)
    details = args.details_dir / cfg.name / args.stem
    details.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for i, doc in enumerate(todo, 1):
        t0 = time.monotonic()
        page = load_page(cfg, doc, args.fallback_stem)
        if page is None:
            print(f"[{i}/{len(todo)}] {doc['doc_id']} SKIP: no Kraken cache {KRAKEN_LINES_NAME}", flush=True)
            summary["skipped"].append(doc["doc_id"])
            continue
        plan = dp.plan_page(page.kraken, page.whole, th, ink_gap_x=page.ink_gap)
        if plan.action != dp.ACTION_KEEP and page.image is None:
            print(f"[{i}/{len(todo)}] {doc['doc_id']} SKIP: no image for a {plan.action} plan", flush=True)
            summary["skipped"].append(doc["doc_id"])
            continue
        text, trace, stats = await decode_page(page, plan, cfg, args.vlm_model, th)
        if stats["crop_calls"] and stats["crop_none"] == stats["crop_calls"]:
            await ensure_model_served(args.vlm_model)     # every crop came back empty: is LM Studio still there?
        seconds = round(time.monotonic() - t0, 1)
        trace.update(doc_id=doc["doc_id"], benchmark=cfg.name, vlm_model=args.vlm_model, stem=args.stem,
                     fallback_stem=args.fallback_stem, plan=plan.to_dict(), thresholds=th.to_dict(),
                     calls=stats, seconds=seconds)
        (details / f"{doc['doc_id']}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
        out = cfg.outputs / doc["doc_id"] / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")                # written last: its existence marks the page done
        summary["written"] += 1
        summary["actions"][plan.action] += 1
        summary["final_sources"][trace["final_source"]] += 1
        summary["calls"] += stats["calls"]
        notes = "".join([" weak-kraken" if trace["weak_kraken_relaxation"] else "",
                         f" fallback={trace['page_fallback']}" if trace["page_fallback"] else ""])
        print(f"[{i}/{len(todo)}] {doc['doc_id']} {plan.action}/{plan.reason} calls={stats['calls']} "
              f"{seconds:.1f}s letters {trace['whole_page_letters']}->{trace['final_letters']} "
              f"({trace['final_source']}{notes})", flush=True)
    summary["seconds"] = round(time.monotonic() - started, 1)
    print(f"done: {summary['written']} written, {len(summary['skipped'])} skipped, "
          f"actions {dict(summary['actions'])}, sources {dict(summary['final_sources'])}, "
          f"{summary['calls']} model calls, {summary['seconds']:.0f}s", flush=True)
    return summary


# ----------------------------------------------------------------------------- CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (None = ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    :return: Parsed arguments.
    :rtype: argparse.Namespace
    :raises SystemExit: For ``--force`` without ``--doc-ids`` (argparse's usage error).
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    ap.add_argument("--vlm-model", required=True, help="LM Studio model id for the crop reads")
    ap.add_argument("--stem", required=True, help="output stem (<outputs>/<doc_id>/<stem>.txt)")
    ap.add_argument("--fallback-stem", required=True, help="stem of the existing whole-page read")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--doc-ids", default="", help="comma-separated doc ids (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="re-decode the --doc-ids pages even when <stem>.txt exists, overwriting output and trace")
    ap.add_argument("--dry-run", action="store_true", help="plan only: no model calls, write a JSON report")
    ap.add_argument("--require-ink-gap", action=argparse.BooleanOptionalAction, default=True,
                    help="a Kraken column split also needs a central ink gap in the image (default: on)")
    ap.add_argument("--report", type=Path, default=None, help="dry-run report path (default: under --details-dir)")
    ap.add_argument("--spec", type=Path, default=None, help="override the benchmark JSON")
    ap.add_argument("--outputs-dir", type=Path, default=None, help="override the per-page output root")
    ap.add_argument("--images-dir", type=Path, default=None, help="override the image folder")
    ap.add_argument("--details-dir", type=Path, default=DETAILS_DIR, help="trace / report root")
    args = ap.parse_args(argv)
    if args.force and not args.doc_ids.strip():
        ap.error("--force re-decodes listed pages only: pass --doc-ids")
    return args


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI entry point.

    :param argv: Arguments (None = ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    :return: The dry-run report, or the real run's summary.
    :rtype: Dict[str, Any]
    """
    args = parse_args(argv)
    check_stems(args.stem, args.fallback_stem)
    cfg = benchmark_config(args.benchmark, args.spec, args.outputs_dir, args.images_dir)
    docs = select_docs(json.loads(cfg.spec.read_text(encoding="utf-8"))["docs"], args.doc_ids, args.limit)
    th = dp.thresholds_for(cfg.token_cap, args.require_ink_gap)
    if args.dry_run:
        return dry_run(docs, cfg, args, th)
    return asyncio.run(run_real(docs, cfg, args, th))


if __name__ == "__main__":
    main()
