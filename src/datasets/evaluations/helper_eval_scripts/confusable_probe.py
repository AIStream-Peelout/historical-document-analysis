"""Letter-confusion probe: which glyph pairs does a model swap, on which script?

Motivation (v1.9, 2026-08-18): the user observes ה/ח swaps on Rashi-type
script even with v18b.  Before choosing a fix (font bank vs vision capacity)
that impression must become a number, split by script, and be re-runnable
after training.  This is the FREE half of the probe: it scores outputs the
benchmarks already produced — no model calls.

Method: for each (document, model) the hypothesis and ground truth are reduced
to their Hebrew-letter streams (spaces/punctuation dropped so alignment sees
glyph identity only) and aligned with Levenshtein edit operations.  Every
substitution op contributes one (gt_letter -> hyp_letter) event.  Rates are
reported per pair as P(hyp = b | gt = a) over the GT letter count of ``a``,
and pooled per canonical confusable pair (both directions).  Documents whose
hypothesis is not a substantive read (CER > CER_MAX) are excluded, so the
counts reflect misreads rather than hallucination alignments.

Corpora / buckets:
* Genizah verified 131 (bucket = script sidecar: judaeo_arabic / hebrew /
  aramaic / untagged) — models qwen3_vl_8b_heb_v18b_step700, v17_step800,
  kraken_raw.
* Talmud 65 pages (bucket = section: gemara = square print, rashi / tosafot =
  Rashi type) — same VLM models.

Outputs (transcription_results/confusable_probe/): confusable_pairs.csv,
confusable_top.csv, console summary.

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/confusable_probe.py
"""

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import Levenshtein

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.evaluations.metrics import (  # noqa: E402
    cer_pair,
    genizah_visible_ink_gt,
    normalize_ink_hypothesis,
)
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import (  # noqa: E402
    _OUTPUTS,
    load_benchmark,
)
from src.datasets.document_models.talmud_gt_parser import load_gt_directory  # noqa: E402

_BENCH_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
_TALMUD_GT = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/talmud_sample/texts"
_HEB = re.compile(r"[א-ת]")
CER_MAX = 0.6
MODELS = ["qwen3_vl_8b_heb_v18b_step700", "qwen3_vl_8b_heb_v17_step800", "kraken_raw"]
TALMUD_SECTIONS = {"gemara": "גמרא", "rashi": "רשי", "tosafot": "תוספות"}
CANONICAL_PAIRS = [("ה", "ח"), ("ד", "ר"), ("ו", "ן"), ("ב", "כ"), ("ג", "נ"),
                   ("ט", "ס"), ("ם", "ס"), ("ע", "צ"), ("י", "ו"), ("ז", "ן"),
                   ("כ", "פ"), ("ת", "ח"), ("ה", "ת"), ("ם", "ט"), ("ן", "ר"),
                   ("צ", "ץ"), ("מ", "ם"), ("נ", "ן"), ("כ", "ך"), ("פ", "ף")]


def letters(text: str) -> str:
    """Reduce text to its Hebrew letter stream.

    :param text: Any text.
    :type text: str
    :return: Concatenated Hebrew letters.
    :rtype: str
    """
    return "".join(_HEB.findall(text))


def confusion_events(hyp: str, gt: str) -> collections.Counter:
    """Count substitution events (gt_letter, hyp_letter) from an alignment.

    :param hyp: Hypothesis letter stream.
    :type hyp: str
    :param gt: Ground-truth letter stream.
    :type gt: str
    :return: Counter of (gt_letter, hyp_letter) pairs.
    :rtype: collections.Counter
    """
    events = collections.Counter()
    for op, i, j in Levenshtein.editops(gt, hyp):
        if op == "replace":
            events[(gt[i], hyp[j])] += 1
    return events


def genizah_docs(models: list) -> list:
    """Yield (bucket, model, hyp_text, gt_text) tuples for the verified 131.

    :param models: Model keys to score.
    :type models: list
    :return: List of tuples.
    :rtype: list
    """
    tags_path = _BENCH_DIR / "genizah_test_v1_script_tags.json"
    tags = {k: v.get("bucket", "untagged") for k, v in json.load(open(tags_path)).items()} \
        if tags_path.exists() else {}
    out = []
    for d in load_benchmark("verified"):
        gt = genizah_visible_ink_gt(d["gt"])
        for m in models:
            p = _OUTPUTS / d["doc_id"] / f"{m}.txt"
            if p.exists():
                out.append((f"genizah:{tags.get(d['doc_id'], 'untagged')}", m,
                            normalize_ink_hypothesis(p.read_text(errors="replace")), gt))
    return out


def talmud_docs(models: list) -> list:
    """Yield (bucket, model, hyp_text, gt_text) tuples for the 65 Talmud pages.

    :param models: Model keys to score (VLMs; kraken has no per-section files).
    :type models: list
    :return: List of tuples.
    :rtype: list
    """
    gts = load_gt_directory(_TALMUD_GT)
    out = []
    for stem, sections in gts.items():
        page_dir = _OUTPUTS / f"{stem}_page_001"
        if not page_dir.is_dir():
            continue
        for sec_key, sec_gt_key in TALMUD_SECTIONS.items():
            gt = sections.get(sec_key) or sections.get(sec_gt_key) or ""
            if len(letters(gt)) < 50:
                continue
            for m in models:
                p = page_dir / f"{m}_{sec_key}.txt"
                if p.exists():
                    out.append((f"talmud:{sec_key}", m, p.read_text(errors="replace"), gt))
    return out


def main() -> None:
    """Score all available outputs and write the confusion tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path,
                        default=Path("transcription_results/confusable_probe"))
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    docs = genizah_docs(models) + talmud_docs([m for m in models if "kraken" not in m])
    events = collections.defaultdict(collections.Counter)      # (bucket, model) -> pair counter
    gt_letter_counts = collections.defaultdict(collections.Counter)
    n_docs = collections.Counter()
    n_skipped = collections.Counter()
    for bucket, model, hyp, gt in docs:
        h, g = letters(hyp), letters(gt)
        if len(h) < 20 or len(g) < 20:
            n_skipped[(bucket, model)] += 1
            continue
        cer, _ = cer_pair(h, g)
        if cer > CER_MAX:
            n_skipped[(bucket, model)] += 1
            continue
        events[(bucket, model)].update(confusion_events(h, g))
        gt_letter_counts[(bucket, model)].update(g)
        n_docs[(bucket, model)] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (bucket, model), ev in sorted(events.items()):
        glc = gt_letter_counts[(bucket, model)]
        for a, b in CANONICAL_PAIRS:
            ab, ba = ev[(a, b)], ev[(b, a)]
            na, nb = glc[a], glc[b]
            rows.append(dict(bucket=bucket, model=model, pair=f"{a}/{b}",
                             gt_a=na, gt_b=nb, a_to_b=ab, b_to_a=ba,
                             rate_a_to_b=round(ab / na, 4) if na else None,
                             rate_b_to_a=round(ba / nb, 4) if nb else None,
                             pooled_rate=round((ab + ba) / (na + nb), 4) if (na + nb) else None,
                             n_docs=n_docs[(bucket, model)]))
    with open(args.out_dir / "confusable_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Top confusions per bucket/model (data-driven, not just the canonical list).
    top_rows = []
    for (bucket, model), ev in sorted(events.items()):
        glc = gt_letter_counts[(bucket, model)]
        total_subs = sum(ev.values())
        for (a, b), c in ev.most_common(8):
            top_rows.append(dict(bucket=bucket, model=model, gt=a, hyp=b, count=c,
                                 share_of_subs=round(c / total_subs, 3) if total_subs else None,
                                 rate_given_gt=round(c / glc[a], 4) if glc[a] else None))
    with open(args.out_dir / "confusable_top.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(top_rows[0].keys()))
        w.writeheader()
        w.writerows(top_rows)

    print(f"{'bucket':22s} {'model':30s} {'docs':>4s} {'skip':>4s} "
          f"{'ה/ח':>7s} {'ד/ר':>7s} {'ו/ן':>7s} {'ב/כ':>7s} {'ג/נ':>7s} {'ט/ס':>7s} {'ם/ס':>7s} {'ע/צ':>7s} {'י/ו':>7s}")
    key_pairs = ["ה/ח", "ד/ר", "ו/ן", "ב/כ", "ג/נ", "ט/ס", "ם/ס", "ע/צ", "י/ו"]
    by_key = {(r["bucket"], r["model"], r["pair"]): r for r in rows}
    for (bucket, model) in sorted(events):
        cells = []
        for pr in key_pairs:
            r = by_key.get((bucket, model, pr))
            cells.append(f"{r['pooled_rate']*100:6.2f}%" if r and r["pooled_rate"] is not None else "     —")
        print(f"{bucket:22s} {model:30s} {n_docs[(bucket, model)]:4d} {n_skipped[(bucket, model)]:4d} " + " ".join(cells))
    print("\n(pooled swap rate = (a→b + b→a) / (GT count of a + b); substantive reads only, CER ≤ "
          f"{CER_MAX}; letters-only alignment)")
    print(f"Wrote {args.out_dir}/confusable_pairs.csv and confusable_top.csv")


if __name__ == "__main__":
    main()
