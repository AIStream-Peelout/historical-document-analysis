# File name: line_agreement_probe.py
# Date: 9/8/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Two-reader line agreement probe: Kraken lines vs VLM grounded lines vs KTIV GT.

Question: when Kraken (line HTR) and the grounded VLM read the same line the
same way, how often is that line actually right — and how many right lines
would such a rule miss?  Runs on the 24 grounding-eval pages, which have KTIV
ground-truth line text + geometry.

Stage 1 (``--kraken``): call the Kraken microservice ``/transcribe_lines`` on
each page image; cache ``preds/kraken_lines/<page>.json``.
Stage 2 (``--report``): for every VLM grounded line, gather the Kraken
fragments inside its vertical band (Kraken segments these hands into word
fragments, so a VLM line usually collects several), concatenate them right to
left, and compute letters-only similarity between the two readers.  Ground
truth is assigned GEOMETRICALLY (the GT line with the largest vertical
overlap), independent of either reader's text.

Usage (repo root):
    .venv/bin/python src/datasets/evaluations/grounding_eval/line_agreement_probe.py --kraken
    .venv/bin/python src/datasets/evaluations/grounding_eval/line_agreement_probe.py --report
"""
import argparse
import asyncio
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import Levenshtein
from PIL import Image

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")
sys.path.insert(0, str(REPO))
Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
QUERIES = HERE / "grounding_eval.jsonl"
PREDS = HERE / "preds"
KRAKEN_DIR = PREDS / "kraken_lines"
KRAKEN_MODEL = str(REPO / "src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel")
MODELS = {"v2.0a": "qwen3-vl-8b-heb-v20a-step1800", "v1.9a": "qwen3-vl-8b-heb-v19a-step1300"}
_HEB = re.compile(r"[א-ת]")
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)


from src.datasets.consensus.line_rule import (  # noqa: E402
    assign_fragments, letters, overlap_frac, similarity as sim)


def cer(hyp: str, ref: str) -> Optional[float]:
    """Letters-only CER of ``hyp`` against ``ref``; None if ``ref`` has no letters."""
    h, r = letters(hyp), letters(ref)
    if not r:
        return None
    return Levenshtein.distance(h, r) / len(r)


def v_iou(a: List[float], b: List[float]) -> float:
    """Vertical (1-D) IoU of two boxes."""
    ov = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    un = (a[3] - a[1]) + (b[3] - b[1]) - ov
    return ov / un if un > 0 else 0.0


def load_pages() -> List[Dict[str, Any]]:
    """Grounded queries: one per page with GT lines (0-1000 boxes)."""
    rows = [json.loads(l) for l in QUERIES.read_text().splitlines() if l.strip()]
    return [r for r in rows if r["mode"] == "grounded"]


async def run_kraken(pages: List[Dict[str, Any]]) -> None:
    """Stage 1: cache Kraken line reads for every page."""
    from src.models.ocr.kraken_transcriber import transcribe_with_kraken_lines
    KRAKEN_DIR.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(pages, 1):
        out = KRAKEN_DIR / (Path(p["image"]).stem + ".json")
        if out.exists():
            continue
        res = await transcribe_with_kraken_lines(KRAKEN_MODEL, p["image"], timeout=300.0)
        if res is None:
            print(f"  {i}/{len(pages)} {out.stem}: KRAKEN FAILED", flush=True)
            continue
        with Image.open(p["image"]) as im:
            res["pil_size"] = list(im.size)
        out.write_text(json.dumps(res, ensure_ascii=False))
        print(f"  {i}/{len(pages)} {out.stem}: {len(res.get('lines', []))} fragments, "
              f"binarize={res.get('used_binarization')}, image_size={res.get('image_size')}", flush=True)


def kraken_fragments(page: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Kraken fragments for a page with boxes normalized to 0-1000."""
    f = KRAKEN_DIR / (Path(page["image"]).stem + ".json")
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    W, H = d["pil_size"]
    out = []
    for ln in d.get("lines", []):
        x0, y0, x1, y1 = ln["bbox"]
        out.append(dict(text=ln["text"], conf=ln.get("confidence"),
                        box=[1000 * x0 / W, 1000 * y0 / H, 1000 * x1 / W, 1000 * y1 / H]))
    return out


def geometric_gt(box: List[float], gt_lines: List[Dict[str, Any]]) -> Optional[int]:
    """GT line with the largest vertical IoU that overlaps horizontally; None if no overlap."""
    best, best_j = 0.0, None
    for j, g in enumerate(gt_lines):
        if overlap_frac(box, g["box"], 0) < 0.5:
            continue
        v = v_iou(box, g["box"])
        if v > best:
            best, best_j = v, j
    return best_j if best > 0.2 else None


def probe_model(pages: List[Dict[str, Any]], model_key: str) -> Dict[str, Any]:
    """Stage 2 for one VLM: per-line records + rule sweep."""
    dump = json.loads((PREDS / f"grounded_{model_key}.json").read_text())
    preds = {p["doc_id"]: p for p in dump["preds"]}
    recs: List[Dict[str, Any]] = []
    pages_ok, pages_nokraken, gt_total, gt_no_frag = 0, 0, 0, 0
    per_page = []
    for page in pages:
        frags = kraken_fragments(page)
        pr = preds[page["doc_id"]]
        gt = page["gt_lines"]
        gt_total += len(gt)
        if frags is None:
            pages_nokraken += 1
            continue
        # weak-Kraken GT lines: no fragment inside the GT band
        for g in gt:
            if not any(overlap_frac(g["box"], f["box"], 1) >= 0.5 and overlap_frac(g["box"], f["box"], 0) >= 0.5 for f in frags):
                gt_no_frag += 1
        if not pr.get("ok"):
            per_page.append(dict(page=Path(page["image"]).stem, parsed=False, n_vlm=0, n_gt=len(gt)))
            continue
        pages_ok += 1
        vl = pr["pred_lines"]
        groups = assign_fragments(vl, frags)
        n_page = 0
        for i, ln in enumerate(vl):
            ktext = " ".join(frags[j]["text"] for j in groups[i])
            gj = geometric_gt(ln["box"], gt)
            tj = ln["gt_idx"] if ln.get("matched") else None
            rec = dict(page=Path(page["image"]).stem, i=i, vlm=ln["text"], kraken=ktext, n_frag=len(groups[i]),
                       agree=sim(ln["text"], ktext), gt_geo=gj, gt_text=tj,
                       vlm_cer=cer(ln["text"], gt[gj]["text"]) if gj is not None else None,
                       kraken_cer=cer(ktext, gt[gj]["text"]) if gj is not None else None,
                       vlm_cer_textalign=cer(ln["text"], gt[tj]["text"]) if tj is not None else None)
            recs.append(rec)
            n_page += 1
        per_page.append(dict(page=Path(page["image"]).stem, parsed=True, n_vlm=n_page, n_gt=len(gt)))
    # ---- sweep
    scored = [r for r in recs if r["vlm_cer"] is not None]
    good = lambda r: r["vlm_cer"] <= 0.10
    sweep = []
    for tau in TAUS:
        acc = [r for r in scored if r["agree"] >= tau]
        rej = [r for r in scored if r["agree"] < tau]
        sweep.append(dict(
            tau=tau, accepted=len(acc), accepted_share=round(len(acc) / len(scored), 3) if scored else None,
            p_cer_le_010=round(sum(good(r) for r in acc) / len(acc), 3) if acc else None,
            p_cer_le_020=round(sum(r["vlm_cer"] <= 0.20 for r in acc) / len(acc), 3) if acc else None,
            p_cer_gt_050=round(sum(r["vlm_cer"] > 0.50 for r in acc) / len(acc), 3) if acc else None,
            recall_of_good=round(sum(good(r) for r in acc) / max(1, sum(good(r) for r in scored)), 3),
            median_cer_accepted=round(statistics.median(r["vlm_cer"] for r in acc), 3) if acc else None,
            median_cer_rejected=round(statistics.median(r["vlm_cer"] for r in rej), 3) if rej else None,
            both_wrong_among_accepted=round(sum((r["vlm_cer"] > 0.10 and (r["kraken_cer"] or 9) > 0.10) for r in acc) / len(acc), 3) if acc else None,
        ))
    summary = dict(
        model=model_key, pages=len(pages), pages_parsed=pages_ok, pages_without_kraken=pages_nokraken,
        gt_lines=gt_total, gt_lines_without_kraken_fragment=gt_no_frag,
        vlm_lines=len(recs), vlm_lines_with_geometric_gt=len(scored),
        vlm_lines_no_geometric_gt=len(recs) - len(scored),
        vlm_lines_zero_fragments=sum(r["n_frag"] == 0 for r in recs),
        box_on_wrong_line=sum(1 for r in scored if r["gt_text"] is not None and r["gt_text"] != r["gt_geo"]),
        vlm_cer_median=round(statistics.median(r["vlm_cer"] for r in scored), 3) if scored else None,
        vlm_share_cer_le_010=round(sum(good(r) for r in scored) / len(scored), 3) if scored else None,
        vlm_share_cer_gt_050=round(sum(r["vlm_cer"] > 0.5 for r in scored) / len(scored), 3) if scored else None,
        kraken_cer_median=round(statistics.median(r["kraken_cer"] for r in scored if r["kraken_cer"] is not None), 3) if scored else None,
        kraken_share_cer_le_010=round(sum((r["kraken_cer"] or 9) <= 0.10 for r in scored) / len(scored), 3) if scored else None,
        agree_median=round(statistics.median(r["agree"] for r in scored), 3) if scored else None,
        sweep=sweep, per_page=per_page,
    )
    return dict(summary=summary, lines=recs)


def report(pages: List[Dict[str, Any]]) -> None:
    """Stage 2 for every model with a grounded dump; write JSON + print tables."""
    out: Dict[str, Any] = {}
    for label, key in MODELS.items():
        if not (PREDS / f"grounded_{key}.json").exists():
            continue
        res = probe_model(pages, key)
        out[label] = res
        s = res["summary"]
        print(f"\n=== {label} ({key})")
        for k, v in s.items():
            if k not in ("sweep", "per_page"):
                print(f"  {k}: {v}")
        print("  tau  acc  share  P(CER<=.10)  P(CER<=.20)  P(CER>.5)  recall(good)  medCER acc/rej  both-wrong")
        for w in s["sweep"]:
            print(f"  {w['tau']:.1f}  {w['accepted']:4d}  {w['accepted_share']}  {w['p_cer_le_010']}  {w['p_cer_le_020']}  "
                  f"{w['p_cer_gt_050']}  {w['recall_of_good']}  {w['median_cer_accepted']}/{w['median_cer_rejected']}  {w['both_wrong_among_accepted']}")
    (PREDS / "line_agreement_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nwrote {PREDS / 'line_agreement_probe.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kraken", action="store_true", help="stage 1: run Kraken /transcribe_lines (cached)")
    ap.add_argument("--report", action="store_true", help="stage 2: agreement vs GT report")
    a = ap.parse_args()
    pages = load_pages()
    if a.kraken:
        asyncio.run(run_kraken(pages))
    if a.report:
        report(pages)
