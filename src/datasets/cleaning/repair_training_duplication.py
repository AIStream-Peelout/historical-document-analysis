# File name: repair_training_duplication.py
# Date: 8/9/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""Repair self-duplicated transcriptions in the Genizah TRAINING manifest.

45 of the 1,145 ``train_clean_manifest.jsonl`` docs concatenate two copies of
the same edition transcription (the defect found and repaired in the frozen
benchmark by ``audit_genizah_benchmark.py``), so the model was trained to
transcribe the page and then transcribe it again — the prime suspect for
v1.7's over-generation. This adapter applies the audit's detection to the
manifest schema and repairs with a **training-specific tie-break**: where the
benchmark repair keeps the copy with more letters (the fuller edition), we
keep the copy with more :data:`GAP` tokens. The gap-marked copy is the
image-faithful one — the fuller copy's extra letters are editorial
reconstruction of ink that is *not on the page*, which trains hallucination,
and discarding the gap-marked copy would also delete most of the doc's
gap-token supervision (163 of 298 ``[...]`` occurrences under the benchmark
policy).

``visible_chars`` is recomputed exactly. ``recon_frac`` cannot be recomputed
exactly (the bracketed originals were destroyed by cleaning), so it is
re-estimated by attributing the doc's reconstruction chars to each copy in
proportion to its GAP count; docs whose estimate crosses the original
``is_training_clean`` gate (0.10) are flagged for review, never silently
dropped.

Usage::

    python -m src.datasets.cleaning.repair_training_duplication \
        [--manifest PATH] [--out PATH] [--report PATH] [--threshold 0.55]
"""
import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from src.datasets.cleaning.clean_genizah_transcriptions import GAP, SEMITIC_RE  # noqa: E402
from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import (  # noqa: E402
    find_self_duplication,
    letters_only,
)

_CLEANUP_DIR = _REPO / "src/datasets/raw_data/genizah_cleanup"
DEFAULT_MANIFEST = _CLEANUP_DIR / "train_clean_manifest.jsonl"
DEFAULT_OUT = _CLEANUP_DIR / "train_clean_manifest_repaired.jsonl"
DEFAULT_REPORT = _CLEANUP_DIR / "duplication_repair_report.json"
RECON_GATE = 0.10


def split_duplication_gap_first(
    text: str, threshold: float = 0.55
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Split a self-duplicated text and keep the gap-marked copy.

    Scans candidate split points around the midpoint (same scan as the
    benchmark's ``repair_duplication``) for the division whose two sides are
    most letter-similar, then keeps the side with more :data:`GAP` tokens
    (ties broken toward more letters, matching the benchmark policy).

    :param text: Manifest ``cleaned_text`` suspected of self-duplication.
    :param threshold: Minimum half-similarity for a split to be applied.
    :returns: Tuple of (repaired text, info dict) where info is ``None`` when
        no qualifying split exists; info records the similarity, cut index,
        which side was kept and each side's gap/letter counts.
    """
    words = text.split()
    n = len(words)
    best: Optional[Tuple[float, int, str, str]] = None
    for cut in range(int(n * 0.35), int(n * 0.65)):
        a = " ".join(words[:cut])
        b = " ".join(words[cut:])
        la, lb = letters_only(a), letters_only(b)
        if not la or not lb:
            continue
        sim = difflib.SequenceMatcher(None, la, lb, autojunk=False).ratio()
        if best is None or sim > best[0]:
            best = (sim, cut, a, b)
    if best is None or best[0] < threshold:
        return text, None
    sim, cut, a, b = best
    gaps_a, gaps_b = a.count(GAP), b.count(GAP)
    letters_a, letters_b = len(letters_only(a)), len(letters_only(b))
    if gaps_a != gaps_b:
        keep_a = gaps_a > gaps_b
    else:
        keep_a = letters_a >= letters_b
    kept, dropped = (a, b) if keep_a else (b, a)
    info = {
        "sim": round(sim, 4),
        "cut_word": cut,
        "kept_side": "first" if keep_a else "second",
        "gaps_kept": kept.count(GAP),
        "gaps_dropped": dropped.count(GAP),
        "letters_kept": len(letters_only(kept)),
        "letters_dropped": len(letters_only(dropped)),
    }
    return kept, info


def estimate_recon_frac(
    old_recon_frac: float, old_visible: int, new_visible: int,
    gaps_kept: int, gaps_dropped: int,
) -> float:
    """Estimate the repaired doc's reconstruction fraction.

    The cleaner counted reconstruction letters across *both* copies; after
    repair only one copy remains and the originals are unrecoverable, so the
    doc's reconstruction chars are attributed to each copy in proportion to
    its GAP-token count (uniform split when neither copy has gaps).

    :param old_recon_frac: Manifest ``recon_frac`` before repair.
    :param old_visible: Manifest ``visible_chars`` before repair.
    :param new_visible: Visible chars of the kept copy.
    :param gaps_kept: GAP tokens in the kept copy.
    :param gaps_dropped: GAP tokens in the dropped copy.
    :returns: Estimated reconstruction fraction of the kept copy.
    """
    if old_recon_frac > 0 and old_visible > 0:
        recon_total = old_recon_frac * old_visible / (1 - old_recon_frac)
    else:
        recon_total = 0.0
    total_gaps = gaps_kept + gaps_dropped
    share = gaps_kept / total_gaps if total_gaps else 0.5
    recon_kept = recon_total * share
    denom = recon_kept + new_visible
    return recon_kept / denom if denom else 0.0


def repair_manifest(
    manifest_path: Path, out_path: Path, report_path: Path, threshold: float
) -> Dict[str, Any]:
    """Detect and repair self-duplicated docs, writing a new manifest.

    :param manifest_path: Input training manifest JSONL.
    :param out_path: Output manifest JSONL (all docs; repaired where flagged).
    :param report_path: JSON report of every repair for review.
    :param threshold: Duplication-similarity threshold (audit default 0.55).
    :returns: The report dict (also written to ``report_path``).
    """
    repairs: List[Dict[str, Any]] = []
    kept_rows: List[str] = []
    n_docs = 0
    for line in open(manifest_path):
        rec = json.loads(line)
        n_docs += 1
        sim = find_self_duplication(rec["cleaned_text"], threshold)
        if sim > threshold:
            repaired, info = split_duplication_gap_first(rec["cleaned_text"], threshold)
            if info is not None:
                new_visible = len(SEMITIC_RE.findall(repaired))
                recon_est = estimate_recon_frac(
                    rec["recon_frac"], rec["visible_chars"], new_visible,
                    info["gaps_kept"], info["gaps_dropped"],
                )
                entry = {
                    "doc_id": rec["doc_id"],
                    "shelf_mark": rec["shelf_mark"],
                    "detect_sim": round(sim, 4),
                    **info,
                    "visible_chars_old": rec["visible_chars"],
                    "visible_chars_new": new_visible,
                    "recon_frac_old": rec["recon_frac"],
                    "recon_frac_est": round(recon_est, 4),
                    "recon_gate_flag": recon_est >= RECON_GATE,
                }
                repairs.append(entry)
                rec["cleaned_text"] = repaired
                rec["visible_chars"] = new_visible
                rec["recon_frac"] = round(recon_est, 4)
                rec["_repaired_duplication"] = True
        kept_rows.append(json.dumps(rec, ensure_ascii=False))
    out_path.write_text("\n".join(kept_rows) + "\n")
    report = {
        "manifest": str(manifest_path),
        "out": str(out_path),
        "threshold": threshold,
        "n_docs": n_docs,
        "n_repaired": len(repairs),
        "tie_break": "keep copy with more GAP tokens (image-faithful), "
                     "ties toward more letters",
        "gaps_kept_total": sum(r["gaps_kept"] for r in repairs),
        "gaps_dropped_total": sum(r["gaps_dropped"] for r in repairs),
        "recon_gate_flags": [r["doc_id"] for r in repairs if r["recon_gate_flag"]],
        "repairs": sorted(repairs, key=lambda r: -r["detect_sim"]),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()
    rep = repair_manifest(args.manifest, args.out, args.report, args.threshold)
    print(f"repaired {rep['n_repaired']}/{rep['n_docs']} docs; "
          f"gaps kept {rep['gaps_kept_total']} vs dropped {rep['gaps_dropped_total']}; "
          f"recon-gate flags: {len(rep['recon_gate_flags'])}")
    for r in rep["repairs"][:10]:
        print(f"  {r['doc_id']:44s} sim={r['detect_sim']:.3f} kept={r['kept_side']:6s} "
              f"gaps {r['gaps_kept']}/{r['gaps_dropped']} letters "
              f"{r['letters_kept']}/{r['letters_dropped']}")
