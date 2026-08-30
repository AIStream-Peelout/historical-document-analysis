"""Score models on the Genizah religious-text benchmark (genizah_religious_v1).

Runs each configured model on the 140 pages, saves raw outputs, and scores
them with the SAME machinery as the PGP offline scorer (failure mode / aligned
F1 / order-independent n-gram precision / CER-on-substantive) — so the numbers
are methodologically comparable across the two benchmarks, while living in a
separate namespace.  Results are broken out by ``is_talmud`` and by column
count (single vs multi), because the multi-column pages are the hard class the
T-S NS 219.40 Megillah failure exposed.

Models: v19a-step1300 (flagship VLM, LM Studio) + kraken_raw (docker :8002,
flat text in segmentation order) + kraken_seg (same Kraken recognition, lines
reordered into reading order from their geometry via
``ktiv_layout.reorder_ocr_lines`` — the same reconstructor the GT uses, so
kraken_seg isolates recognition quality from reading order).
Sequential, resumable (skip pages already transcribed).  Per-page Kraken line
geometry is cached in ``raw_outputs/<doc>/kraken_lines.json`` so the reorder
can be re-derived offline.

Honesty metrics (this scorer only; the PGP scorer is untouched): each row
carries ``bag_overlap`` (Dice multiset char overlap vs GT — NOTE: ~0.8+ is
chance level for same-length Hebrew text, so high values alone prove
nothing), ``line_ngram_p`` (median per-line 5-gram precision over the
output's own lines with >= 8 letters — near-chance-free evidence the lines
are really this page's text), and ``order_scramble``
(``line_ngram_p >= 0.5 and aligned_f1 < 0.5``: content verified, sequence
wrong).  The aggregate replaces the VLM-centric ``halluc`` column with
``readOK`` (median line_ngram_p) + ``orderScr`` (share of scrambled pages),
so a line-OCR model with no language prior is never labelled "hallucinated".

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/run_religious_benchmark.py \\
        [--limit N] [--score-only]
"""

import argparse
import asyncio
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.evaluations.metrics import (  # noqa: E402
    cer_pair,
    genizah_visible_ink_gt,
    normalize_ink_hypothesis,
)
from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import (  # noqa: E402
    letters_only,
)
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import (  # noqa: E402
    aligned_prf,
    asserted_letters,
    classify,
    ngram_precision,
)
from src.datasets.consensus.consensus_gate import build_fragment_prompt  # noqa: E402
from src.models.ocr.lms_transcriber import (  # noqa: E402
    check_lm_studio_health,
    transcribe_with_lm_studio,
)
from src.models.ocr.kraken_transcriber import (  # noqa: E402
    preload_kraken_model,
    transcribe_with_kraken,
    transcribe_with_kraken_lines,
)
from src.finetuning.qwen_hebrew.ktiv_layout import reorder_ocr_lines  # noqa: E402

_BENCH = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1"
_OUT = _BENCH / "raw_outputs"
_KRAKEN_MODEL = str(_REPO / "src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel")
VLM_MODEL = "qwen3-vl-8b-heb-v19a-step1300"
KRAKEN_KEY = "kraken_raw"
KRAKEN_SEG_KEY = "kraken_seg"
_LINES_NAME = "kraken_lines.json"
LINE_NGRAM_MIN_LETTERS = 8      # shorter output lines carry no 5-gram evidence
ORDER_SCRAMBLE_LINE_P = 0.5     # content verified at line level ...
ORDER_SCRAMBLE_MAX_F1 = 0.5     # ... but the page sequence does not align


async def transcribe_all(docs: list, vlm_model: str = VLM_MODEL) -> None:
    """Run all engines over every page, saving raw outputs (resumable).

    LM Studio is only touched when a VLM output is actually missing, so
    kraken-only passes never contend with the shared production server.

    :param docs: Benchmark doc entries.
    :type docs: list
    :param vlm_model: LM Studio model id to transcribe with; outputs are
        saved under the id (``-`` replaced by ``_``), so different models
        accumulate side by side and are all picked up by :func:`score`.
    :type vlm_model: str
    """
    need_vlm = [d for d in docs
                if not (_OUT / d["doc_id"] / f"{vlm_model.replace('-', '_')}.txt").exists()]
    if need_vlm:
        models = await check_lm_studio_health()
        if vlm_model not in models:
            raise RuntimeError(f"{vlm_model} not served by LM Studio: {models}")
    preload_kraken_model(_KRAKEN_MODEL)
    for i, d in enumerate(docs, 1):
        outdir = _OUT / d["doc_id"]
        outdir.mkdir(parents=True, exist_ok=True)
        vlm_f = outdir / f"{vlm_model.replace('-', '_')}.txt"
        krk_f = outdir / f"{KRAKEN_KEY}.txt"
        lines_f = outdir / _LINES_NAME
        seg_f = outdir / f"{KRAKEN_SEG_KEY}.txt"
        if not krk_f.exists():
            txt = await transcribe_with_kraken(_KRAKEN_MODEL, d["image"], timeout=180.0)
            krk_f.write_text(txt or "", encoding="utf-8")
        if not lines_f.exists():
            res = await transcribe_with_kraken_lines(_KRAKEN_MODEL, d["image"], timeout=300.0)
            if res is not None:
                lines_f.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        if not seg_f.exists() and lines_f.exists():
            lines = json.loads(lines_f.read_text()).get("lines", [])
            seg_f.write_text(reorder_ocr_lines(lines), encoding="utf-8")
        if not vlm_f.exists():
            txt = await transcribe_with_lm_studio(
                vlm_model, d["image"], build_fragment_prompt(d["doc_id"]), max_tokens=2500)
            vlm_f.write_text(txt or "", encoding="utf-8")
        if i % 20 == 0:
            print(f"  transcribed {i}/{len(docs)}", flush=True)


def bag_overlap(hyp_letters: str, gt_letters: str) -> float:
    """Dice multiset character overlap between hypothesis and GT letters.

    Order-independent by construction.  Interpret with care: two UNRELATED
    Hebrew texts of similar length overlap ~0.8+ at the unigram level, so
    only low values are informative (the output is not even letter-plausible).

    :param hyp_letters: Hypothesis reduced to letters.
    :type hyp_letters: str
    :param gt_letters: Ground truth reduced to letters.
    :type gt_letters: str
    :return: Dice overlap in [0, 1].
    :rtype: float
    """
    if not hyp_letters or not gt_letters:
        return 0.0
    h, g = collections.Counter(hyp_letters), collections.Counter(gt_letters)
    return 2 * sum((h & g).values()) / (len(hyp_letters) + len(gt_letters))


def line_ngram_precision(raw: str, gt_letters: str, n: int = 5) -> float:
    """Median per-line n-gram precision of an output's own lines vs the GT.

    Judges each output line in isolation (no cross-line n-grams), so it
    measures whether the LINES are really this page's text independent of
    the order they were emitted in — the honest "did it read the ink"
    signal for a line-OCR model.

    :param raw: Raw model output (newline-separated lines).
    :type raw: str
    :param gt_letters: Ground truth reduced to letters.
    :type gt_letters: str
    :param n: N-gram size.
    :type n: int
    :return: Median over lines with >= :data:`LINE_NGRAM_MIN_LETTERS`
        letters; 0.0 when no line is long enough to carry evidence.
    :rtype: float
    """
    gt_grams = {gt_letters[i:i + n] for i in range(len(gt_letters) - n + 1)}
    vals = []
    for line in raw.splitlines():
        letters = letters_only(normalize_ink_hypothesis(line))
        if len(letters) < LINE_NGRAM_MIN_LETTERS:
            continue
        grams = [letters[i:i + n] for i in range(len(letters) - n + 1)]
        vals.append(sum(g in gt_grams for g in grams) / len(grams))
    return statistics.median(vals) if vals else 0.0


def discover_model_keys(docs: list) -> list:
    """Find every model with saved raw outputs for these docs.

    VLM outputs are named after their LM Studio id (``-`` → ``_``), so any
    model transcribed with ``--vlm-model`` shows up here without a code
    change; the shared ``qwen3_vl_8b_heb_`` prefix is stripped from labels
    (``v19a_step1300`` etc.) to match previously published score rows.

    :param docs: Benchmark doc entries.
    :type docs: list
    :return: Sorted ``(file_key, label)`` pairs.
    :rtype: list
    """
    keys = set()
    for d in docs:
        outdir = _OUT / d["doc_id"]
        if not outdir.is_dir():
            continue
        for f in outdir.glob("*.txt"):
            if f.stem not in (KRAKEN_KEY, KRAKEN_SEG_KEY):
                keys.add(f.stem)
    vlm = [(k, k.removeprefix("qwen3_vl_8b_heb_")) for k in sorted(keys)]
    return vlm + [(KRAKEN_KEY, KRAKEN_KEY), (KRAKEN_SEG_KEY, KRAKEN_SEG_KEY)]


def score(docs: list) -> list:
    """Score every model's saved output against each page's GT.

    :param docs: Benchmark doc entries.
    :type docs: list
    :return: Per-(doc, model) score rows.
    :rtype: list
    """
    model_keys = discover_model_keys(docs)
    rows = []
    for d in docs:
        gt_ink = genizah_visible_ink_gt(d["gt"])
        gt_letters = letters_only(gt_ink)
        if len(gt_letters) < 50:
            continue
        outdir = _OUT / d["doc_id"]
        for key, label in model_keys:
            f = outdir / f"{key}.txt"
            if not f.exists():
                continue
            raw = f.read_text(errors="replace")
            hyp = normalize_ink_hypothesis(raw)
            hyp_letters = letters_only(hyp)
            ngram_p = ngram_precision(hyp_letters, gt_letters)
            mode = classify(raw, asserted_letters(hyp), gt_letters, ngram_p)
            _, _, f1 = aligned_prf(hyp, gt_ink)
            cer_s, _ = cer_pair(hyp, gt_ink)
            line_p = line_ngram_precision(raw, gt_letters)
            rows.append(dict(
                doc_id=d["doc_id"], model=label,
                is_talmud=d["is_talmud"], columns=("multi" if d["n_columns"] >= 2 else "single"),
                genre=d["genre"], failure_mode=mode,
                ngram_precision=round(ngram_p, 4), aligned_f1=round(f1, 4),
                cer=round(cer_s, 4),
                bag_overlap=round(bag_overlap(hyp_letters, gt_letters), 4),
                line_ngram_p=round(line_p, 4),
                order_scramble=int(line_p >= ORDER_SCRAMBLE_LINE_P
                                   and f1 < ORDER_SCRAMBLE_MAX_F1),
                gt_letters=len(gt_letters), hyp_letters=len(hyp_letters),
            ))
    return rows


def _agg(rows: list, label: str) -> dict:
    """Aggregate a group of score rows into headline metrics.

    :param rows: Rows for one model (and optionally one slice).
    :type rows: list
    :param label: Group label.
    :type label: str
    :return: Summary dict.
    :rtype: dict
    """
    modes = collections.Counter(r["failure_mode"] for r in rows)
    subst = [r for r in rows if r["failure_mode"] == "substantive"]
    return dict(
        slice=label, n=len(rows),
        subst=round(modes["substantive"] / len(rows), 3) if rows else 0,
        readOK=round(statistics.median(r["line_ngram_p"] for r in rows), 4) if rows else 0,
        orderScr=round(sum(r["order_scramble"] for r in rows) / len(rows), 3) if rows else 0,
        ngramP=round(statistics.median(r["ngram_precision"] for r in rows), 4) if rows else 0,
        F1=round(statistics.median(r["aligned_f1"] for r in rows), 4) if rows else 0,
        CERsub=round(statistics.median(r["cer"] for r in subst), 4) if subst else None,
    )


def report(rows: list) -> None:
    """Print the comparison table (overall + Talmud/column slices) and save CSV.

    :param rows: All score rows.
    :type rows: list
    """
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_BENCH / "religious_scores_long.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    print(f"\n{'model':16s} {'slice':14s} {'n':>4s} {'subst':>6s} {'readOK':>7s} "
          f"{'ordScr':>6s} {'ngramP':>7s} {'F1':>6s} {'CER*':>6s}")
    summary = []
    for model, rs in sorted(by_model.items()):
        for label, sub in [("ALL", rs),
                            ("talmud", [r for r in rs if r["is_talmud"]]),
                            ("multi-col", [r for r in rs if r["columns"] == "multi"]),
                            ("single-col", [r for r in rs if r["columns"] == "single"])]:
            if not sub:
                continue
            a = _agg(sub, label)
            a["model"] = model
            summary.append(a)
            cer = f"{a['CERsub']:.3f}" if a["CERsub"] is not None else "  —"
            print(f"{model:16s} {label:14s} {a['n']:4d} {a['subst']:6.0%} {a['readOK']:7.3f} "
                  f"{a['orderScr']:6.0%} {a['ngramP']:7.3f} {a['F1']:6.3f} {cer:>6s}")
    with open(_BENCH / "religious_scores_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "slice", "n", "subst", "readOK",
                                           "orderScr", "ngramP", "F1", "CERsub"])
        w.writeheader()
        w.writerows([{k: s.get(k) for k in w.fieldnames} for s in summary])
    print("\n* CER over substantive attempts only. Multi-col = the hard two-column class.")
    print("readOK = median per-line 5-gram precision (order-free line-content check);")
    print("ordScr = share of pages whose lines verify (>=0.5) but page F1 < 0.5 —")
    print("content right, reading order wrong.")


def main() -> None:
    """Transcribe (unless --score-only) then score and report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--vlm-model", default=VLM_MODEL,
                        help="LM Studio model id to transcribe with "
                             f"(default: {VLM_MODEL}); scoring always covers "
                             "every model found in raw_outputs")
    args = parser.parse_args()
    docs = json.load(open(_BENCH / "genizah_religious_v1.json"))["docs"]
    if args.limit:
        docs = docs[:args.limit]
    if not args.score_only:
        asyncio.run(transcribe_all(docs, args.vlm_model))
    rows = score(docs)
    if not rows:
        print("no scored rows")
        return
    report(rows)


if __name__ == "__main__":
    main()
