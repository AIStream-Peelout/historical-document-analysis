"""Score models on the Genizah religious-text benchmark (genizah_religious_v1).

Runs each configured model on the 140 pages, saves raw outputs, and scores
them with the SAME machinery as the PGP offline scorer (failure mode / aligned
F1 / order-independent n-gram precision / CER-on-substantive) — so the numbers
are methodologically comparable across the two benchmarks, while living in a
separate namespace.  Results are broken out by ``is_talmud`` and by column
count (single vs multi), because the multi-column pages are the hard class the
T-S NS 219.40 Megillah failure exposed.

Models: v19a-step1300 (flagship VLM, LM Studio) + kraken_raw (docker :8002).
Sequential, resumable (skip pages already transcribed).

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
)

_BENCH = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1"
_OUT = _BENCH / "raw_outputs"
_KRAKEN_MODEL = str(_REPO / "src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel")
VLM_MODEL = "qwen3-vl-8b-heb-v19a-step1300"
KRAKEN_KEY = "kraken_raw"


async def transcribe_all(docs: list) -> None:
    """Run both engines over every page, saving raw outputs (resumable).

    :param docs: Benchmark doc entries.
    :type docs: list
    """
    models = await check_lm_studio_health()
    if VLM_MODEL not in models:
        raise RuntimeError(f"{VLM_MODEL} not served by LM Studio: {models}")
    preload_kraken_model(_KRAKEN_MODEL)
    for i, d in enumerate(docs, 1):
        outdir = _OUT / d["doc_id"]
        outdir.mkdir(parents=True, exist_ok=True)
        vlm_f = outdir / f"{VLM_MODEL.replace('-', '_')}.txt"
        krk_f = outdir / f"{KRAKEN_KEY}.txt"
        if not krk_f.exists():
            txt = await transcribe_with_kraken(_KRAKEN_MODEL, d["image"], timeout=180.0)
            krk_f.write_text(txt or "", encoding="utf-8")
        if not vlm_f.exists():
            txt = await transcribe_with_lm_studio(
                VLM_MODEL, d["image"], build_fragment_prompt(d["doc_id"]), max_tokens=2500)
            vlm_f.write_text(txt or "", encoding="utf-8")
        if i % 20 == 0:
            print(f"  transcribed {i}/{len(docs)}", flush=True)


def score(docs: list) -> list:
    """Score every model's saved output against each page's GT.

    :param docs: Benchmark doc entries.
    :type docs: list
    :return: Per-(doc, model) score rows.
    :rtype: list
    """
    rows = []
    for d in docs:
        gt_ink = genizah_visible_ink_gt(d["gt"])
        gt_letters = letters_only(gt_ink)
        if len(gt_letters) < 50:
            continue
        outdir = _OUT / d["doc_id"]
        for key in (VLM_MODEL.replace("-", "_"), KRAKEN_KEY):
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
            rows.append(dict(
                doc_id=d["doc_id"], model=("v19a_step1300" if key != KRAKEN_KEY else key),
                is_talmud=d["is_talmud"], columns=("multi" if d["n_columns"] >= 2 else "single"),
                genre=d["genre"], failure_mode=mode,
                ngram_precision=round(ngram_p, 4), aligned_f1=round(f1, 4),
                cer=round(cer_s, 4), gt_letters=len(gt_letters), hyp_letters=len(hyp_letters),
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
        halluc=round((modes["hallucinated"] + modes["loop_collapse"]) / len(rows), 3) if rows else 0,
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
    print(f"\n{'model':16s} {'slice':14s} {'n':>4s} {'subst':>6s} {'halluc':>6s} "
          f"{'ngramP':>7s} {'F1':>6s} {'CER*':>6s}")
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
            print(f"{model:16s} {label:14s} {a['n']:4d} {a['subst']:6.0%} {a['halluc']:6.0%} "
                  f"{a['ngramP']:7.3f} {a['F1']:6.3f} {cer:>6s}")
    with open(_BENCH / "religious_scores_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "slice", "n", "subst", "halluc",
                                           "ngramP", "F1", "CERsub"])
        w.writeheader()
        w.writerows([{k: s.get(k) for k in w.fieldnames} for s in summary])
    print("\n* CER over substantive attempts only. Multi-col = the hard two-column class.")


def main() -> None:
    """Transcribe (unless --score-only) then score and report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    docs = json.load(open(_BENCH / "genizah_religious_v1.json"))["docs"]
    if args.limit:
        docs = docs[:args.limit]
    if not args.score_only:
        asyncio.run(transcribe_all(docs))
    rows = score(docs)
    if not rows:
        print("no scored rows")
        return
    report(rows)


if __name__ == "__main__":
    main()
