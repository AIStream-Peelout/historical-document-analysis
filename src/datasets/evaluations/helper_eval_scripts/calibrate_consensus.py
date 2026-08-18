"""Consensus-gate calibration: cross-family agreement vs transcription accuracy.

Production question (cairogenizah.ai rollout): if Kraken (specialist HTR) and
the fine-tuned VLM independently transcribe a fragment and we AUTO-ACCEPT the
VLM transcript whenever the two engines' outputs agree above a threshold, what
fraction of the corpus clears the gate, and how accurate is what clears?

The two engines share no architecture, no training data and no language prior,
so their agreement is evidence about the ink itself rather than about a shared
bias — the same convergence logic the offline scorer's Tier A uses, repurposed
as a per-document confidence gate.  This script computes the calibration curve
entirely from saved benchmark outputs; no model is run.

Agreement metric: directional 5-gram containment between the two hypotheses'
letter streams (same n-gram machinery as the offline scorer), combined as a
harmonic mean ("agreement F1").  The v18b→kraken direction is the operative
hallucination check: the share of what the VLM wrote that the order-blind HTR
also saw.

Outputs (transcription_results/consensus_calibration/):
* consensus_per_doc.csv    — per-fragment agreement + accuracy metrics
* consensus_curve.csv      — acceptance/accuracy sweep over thresholds
* consensus_calibration.png — three-panel calibration figure

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/calibrate_consensus.py
"""

import argparse
import collections
import csv
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
    _OUTPUTS,
    aligned_prf,
    asserted_letters,
    classify,
    load_benchmark,
    ngram_precision,
)

VLM_MODEL = "qwen3_vl_8b_heb_v18b_step700"
HTR_MODEL = "kraken_raw"
THRESHOLDS = [round(x * 0.05, 2) for x in range(19)]  # 0.00 .. 0.90

# Figure palette (dataviz reference instance).
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_RED = "#d03b3b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"


def harmonic(a: float, b: float) -> float:
    """Harmonic mean of two rates.

    :param a: First rate in [0, 1].
    :type a: float
    :param b: Second rate in [0, 1].
    :type b: float
    :return: Harmonic mean, 0.0 when either rate is 0.
    :rtype: float
    """
    return (2 * a * b / (a + b)) if (a + b) else 0.0


def load_tier_buckets(long_csv: Path) -> dict:
    """Read tier and script bucket per fragment from the offline long table.

    :param long_csv: Path to genizah_offline_long.csv.
    :type long_csv: Path
    :return: Mapping fragment_id -> (tier, script_bucket).
    :rtype: dict
    """
    out = {}
    with open(long_csv) as fh:
        for r in csv.DictReader(fh):
            out[r["fragment_id"]] = (r["tier"], r["script_bucket"])
    return out


def side_metrics(raw: str, gt_ink: str, gt_letters: str) -> dict:
    """Score one model output against the ground truth (scorer-identical).

    :param raw: Raw saved model output.
    :type raw: str
    :param gt_ink: Visible-ink ground truth.
    :type gt_ink: str
    :param gt_letters: Ground truth reduced to letters.
    :type gt_letters: str
    :return: Dict with letters, mode, ngram precision, aligned F1 and CER.
    :rtype: dict
    """
    hyp = normalize_ink_hypothesis(raw)
    hyp_letters = letters_only(hyp)
    ngram_p = ngram_precision(hyp_letters, gt_letters)
    mode = classify(raw, asserted_letters(hyp), gt_letters, ngram_p)
    _, _, f1 = aligned_prf(hyp, gt_ink)
    cer_s, _ = cer_pair(hyp, gt_ink)
    return dict(letters=hyp_letters, mode=mode, ngram_p=ngram_p, f1=f1, cer=cer_s)


def collect_rows(bench: str, tiers: dict) -> list:
    """Build the per-fragment agreement/accuracy table.

    :param bench: Benchmark selector passed to :func:`load_benchmark`.
    :type bench: str
    :param tiers: fragment_id -> (tier, script_bucket) mapping.
    :type tiers: dict
    :return: One dict per fragment that has both engines' outputs.
    :rtype: list
    """
    rows = []
    missing = collections.Counter()
    for d in load_benchmark(bench):
        doc_id = d["doc_id"]
        outdir = _OUTPUTS / doc_id
        vlm_f = outdir / f"{VLM_MODEL}.txt"
        htr_f = outdir / f"{HTR_MODEL}.txt"
        if not vlm_f.exists() or not htr_f.exists():
            missing[("vlm" if not vlm_f.exists() else "htr")] += 1
            continue
        gt_ink = genizah_visible_ink_gt(d["gt"])
        gt_letters = letters_only(gt_ink)
        if len(gt_letters) < 50:
            missing["gt_short"] += 1
            continue

        vlm = side_metrics(vlm_f.read_text(errors="replace"), gt_ink, gt_letters)
        htr = side_metrics(htr_f.read_text(errors="replace"), gt_ink, gt_letters)
        agree_v_in_k = ngram_precision(vlm["letters"], htr["letters"])
        agree_k_in_v = ngram_precision(htr["letters"], vlm["letters"])
        tier, bucket = tiers.get(doc_id, ("?", "untagged"))
        rows.append(dict(
            fragment_id=doc_id, tier=tier, script_bucket=bucket,
            gt_letters=len(gt_letters),
            agree_f1=round(harmonic(agree_v_in_k, agree_k_in_v), 4),
            agree_v_in_k=round(agree_v_in_k, 4),
            agree_k_in_v=round(agree_k_in_v, 4),
            vlm_mode=vlm["mode"], vlm_ngram_gt=round(vlm["ngram_p"], 4),
            vlm_f1_gt=round(vlm["f1"], 4), vlm_cer=round(vlm["cer"], 4),
            htr_mode=htr["mode"], htr_ngram_gt=round(htr["ngram_p"], 4),
            htr_f1_gt=round(htr["f1"], 4), htr_cer=round(htr["cer"], 4),
        ))
    if missing:
        print(f"skipped fragments: {dict(missing)}")
    return rows


def sweep(rows: list) -> list:
    """Sweep the acceptance threshold and aggregate accepted/rejected pools.

    :param rows: Per-fragment records from :func:`collect_rows`.
    :type rows: list
    :return: One dict per threshold with acceptance and accuracy stats.
    :rtype: list
    """
    total_chars = sum(r["gt_letters"] for r in rows)
    curve = []
    for x in THRESHOLDS:
        acc = [r for r in rows if r["agree_f1"] >= x]
        rej = [r for r in rows if r["agree_f1"] < x]
        acc_a = [r for r in acc if r["tier"] == "A"]
        rej_a = [r for r in rej if r["tier"] == "A"]
        bad = [r for r in acc if r["vlm_mode"] in ("hallucinated", "loop_collapse")]
        # Compound production gate: the loop screen needs no GT (loop_ratio on
        # the hypothesis alone), so it can run before the agreement gate.
        acc_s = [r for r in acc if r["vlm_mode"] != "loop_collapse"]
        acc_s_a = [r for r in acc_s if r["tier"] == "A"]
        bad_s = [r for r in acc_s if r["vlm_mode"] == "hallucinated"]
        curve.append(dict(
            threshold=x,
            n_accept=len(acc),
            pct_docs=round(len(acc) / len(rows), 3),
            pct_chars=round(sum(r["gt_letters"] for r in acc) / total_chars, 3),
            n_bad_accepted=len(bad),
            pct_docs_screened=round(len(acc_s) / len(rows), 3),
            n_bad_screened=len(bad_s),
            vlm_cer_med_accept_A_screened=round(statistics.median(
                [r["vlm_cer"] for r in acc_s_a]), 4) if acc_s_a else None,
            n_accept_tierA=len(acc_a),
            vlm_cer_med_accept_A=round(statistics.median(
                [r["vlm_cer"] for r in acc_a]), 4) if acc_a else None,
            htr_cer_med_accept_A=round(statistics.median(
                [r["htr_cer"] for r in acc_a]), 4) if acc_a else None,
            vlm_ngram_gt_med_accept=round(statistics.median(
                [r["vlm_ngram_gt"] for r in acc]), 4) if acc else None,
            vlm_f1_med_accept=round(statistics.median(
                [r["vlm_f1_gt"] for r in acc]), 4) if acc else None,
            vlm_cer_med_reject_A=round(statistics.median(
                [r["vlm_cer"] for r in rej_a]), 4) if rej_a else None,
        ))
    return curve


def make_figure(rows: list, curve: list, out_png: Path) -> None:
    """Render the three-panel calibration figure.

    :param rows: Per-fragment records.
    :type rows: list
    :param curve: Threshold sweep records.
    :type curve: list
    :param out_png: Destination PNG path.
    :type out_png: Path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), facecolor=_SURFACE)
    for ax in axes:
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_MUTED)
        ax.tick_params(colors=_MUTED, labelsize=9)

    # Panel A: per-fragment agreement vs CER (Tier A), hallucinations flagged.
    ax = axes[0]
    tier_a = [r for r in rows if r["tier"] == "A"]
    bad = [r for r in rows
           if r["vlm_mode"] in ("hallucinated", "loop_collapse")]
    ax.scatter([r["agree_f1"] for r in tier_a],
               [min(r["vlm_cer"], 1.2) for r in tier_a],
               s=28, color=_BLUE, alpha=0.75, linewidths=0, zorder=3)
    ax.scatter([r["agree_f1"] for r in bad],
               [min(r["vlm_cer"], 1.2) for r in bad],
               s=46, color=_RED, marker="x", linewidths=1.6, zorder=4,
               label="v18b hallucination / loop")
    ax.set_xlabel("cross-family agreement (5-gram F1)", color=_INK, fontsize=10)
    ax.set_ylabel("v18b CER vs GT (Tier A, clipped 1.2)", color=_INK, fontsize=10)
    ax.set_title("Agreement predicts accuracy", color=_INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK)

    # Panel B: acceptance vs threshold (raw gate vs loop-screened gate).
    ax = axes[1]
    xs = [c["threshold"] for c in curve]
    ax.plot(xs, [c["pct_docs"] * 100 for c in curve], color=_BLUE, linewidth=2)
    ax.plot(xs, [c["pct_docs_screened"] * 100 for c in curve], color=_AQUA,
            linewidth=2)
    ax.annotate("agreement only", (xs[4], curve[4]["pct_docs"] * 100),
                textcoords="offset points", xytext=(8, 8),
                color=_BLUE, fontsize=9)
    ax.annotate("+ loop screen (GT-free)",
                (xs[4], curve[4]["pct_docs_screened"] * 100),
                textcoords="offset points", xytext=(8, -16),
                color=_AQUA, fontsize=9)
    ax.set_xlabel("auto-accept threshold X", color=_INK, fontsize=10)
    ax.set_ylabel("share auto-accepted (%)", color=_INK, fontsize=10)
    ax.set_title("Coverage of the gate", color=_INK, fontsize=11, loc="left")

    # Panel C: accepted-pool accuracy vs threshold.
    ax = axes[2]
    ax.plot(xs, [c["vlm_cer_med_accept_A"] for c in curve],
            color=_BLUE, linewidth=2)
    ax.plot(xs, [c["htr_cer_med_accept_A"] for c in curve],
            color=_ORANGE, linewidth=2, linestyle="--")
    ax.annotate("v18b (published)", (xs[8], curve[8]["vlm_cer_med_accept_A"]),
                textcoords="offset points", xytext=(6, 8),
                color=_BLUE, fontsize=9)
    ax.annotate("kraken (reference)", (xs[8], curve[8]["htr_cer_med_accept_A"]),
                textcoords="offset points", xytext=(6, -14),
                color=_ORANGE, fontsize=9)
    for c in curve[::2]:
        if c["n_bad_accepted"]:
            ax.annotate(str(c["n_bad_accepted"]),
                        (c["threshold"], 0.02), color=_RED, fontsize=8,
                        ha="center")
    ax.text(0.02, 0.02, "red = hallucinated docs still accepted",
            transform=ax.transAxes, color=_RED, fontsize=8)
    ax.set_xlabel("auto-accept threshold X", color=_INK, fontsize=10)
    ax.set_ylabel("median CER of accepted (Tier A)", color=_INK, fontsize=10)
    ax.set_title("Accuracy of what clears", color=_INK, fontsize=11, loc="left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=_SURFACE)
    plt.close(fig)


def main() -> None:
    """Run the calibration study end to end."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("frozen", "verified"),
                        default="verified")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("transcription_results/consensus_calibration"))
    parser.add_argument("--long-csv", type=Path,
                        default=Path("transcription_results/paper_table/"
                                     "genizah_offline_long.csv"))
    args = parser.parse_args()

    tiers = load_tier_buckets(args.long_csv)
    rows = collect_rows(args.benchmark, tiers)
    print(f"fragments with both engines: {len(rows)} "
          f"(Tier A: {sum(1 for r in rows if r['tier'] == 'A')})")

    curve = sweep(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_doc_csv = args.out_dir / "consensus_per_doc.csv"
    with open(per_doc_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    curve_csv = args.out_dir / "consensus_curve.csv"
    with open(curve_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)

    def fmt(value: float, width: int) -> str:
        """Format an optional metric for the console table.

        :param value: Metric value or None.
        :type value: float
        :param width: Column width.
        :type width: int
        :return: Right-aligned fixed-width cell.
        :rtype: str
        """
        return (f"{value:.3f}" if value is not None else "—").rjust(width)

    print(f"\n{'X':>5s} {'acc%':>6s} {'chars%':>7s} {'bad':>4s} "
          f"{'scr%':>6s} {'sbad':>5s} {'CER_A(scr)':>11s} "
          f"{'CER_A(krkn)':>12s} {'ngramGT':>8s} {'rejCER_A':>9s}")
    for c in curve:
        print(f"{c['threshold']:5.2f} {c['pct_docs']:6.1%} {c['pct_chars']:7.1%} "
              f"{c['n_bad_accepted']:4d} {c['pct_docs_screened']:6.1%} "
              f"{c['n_bad_screened']:5d} "
              f"{fmt(c['vlm_cer_med_accept_A_screened'], 11)} "
              f"{fmt(c['htr_cer_med_accept_A'], 12)} "
              f"{fmt(c['vlm_ngram_gt_med_accept'], 8)} "
              f"{fmt(c['vlm_cer_med_reject_A'], 9)}")

    # Operating point: the compound gate (loop screen is GT-free, so it is
    # always available in production) at the lowest hallucination-free X.
    clean = [c for c in curve if c["n_bad_screened"] == 0 and c["threshold"] > 0]
    if clean:
        best = min(clean, key=lambda c: c["threshold"])
        print(f"\ncompound gate (loop screen + agreement ≥ X): lowest "
              f"hallucination-free X={best['threshold']:.2f} → accepts "
              f"{best['pct_docs_screened']:.0%} of fragments at Tier-A median "
              f"CER {best['vlm_cer_med_accept_A_screened']}")

    # Agreement-bin evidence (Tier A only).
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]
    print("\nTier-A median v18b CER by agreement bin:")
    for lo, hi in bins:
        sel = [r["vlm_cer"] for r in rows
               if r["tier"] == "A" and lo <= r["agree_f1"] < hi]
        med = f"{statistics.median(sel):.3f}" if sel else "—"
        print(f"  [{lo:.1f},{hi:.1f}): n={len(sel):3d}  median CER {med}")

    make_figure(rows, curve, args.out_dir / "consensus_calibration.png")
    print(f"\nWrote {per_doc_csv}, {curve_csv}, consensus_calibration.png")
    print("NOTE: benchmark fragments are curated (single image, ≥200 chars, "
          "low gap density) — corpus-wide acceptance FRACTIONS will run lower; "
          "the CER-given-agreement relationship is the transferable part.")


if __name__ == "__main__":
    main()
