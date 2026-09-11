# File name: box_quality.py
# Date: 9/9/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Box-quality metrics for grounded page reads (the v2.1 target metrics).

Text quality is measured elsewhere; this scores the GEOMETRY of the line
boxes a model emits, because v2.0a's boxes turned out to be a layout prior
(one x-range repeated down the page, drifting off the line) rather than a
measurement.  Two inputs are supported:

* ``preds/grounded_<model>.json`` dumps on the 24 grounding-eval pages (KTIV
  ground-truth line geometry available) → template rate, box-on-right-line
  rate, vertical IoU, width ratio and vertical drift against ground truth.
* ``ai_reads`` records or raw-cache entries (site pages, no ground truth) →
  template rate and width ratio against the Kraken row extent.

Usage (repo root):
    .venv/bin/python src/datasets/evaluations/grounding_eval/box_quality.py --preds qwen3-vl-8b-heb-v20a-step1800 [...]
    .venv/bin/python src/datasets/evaluations/grounding_eval/box_quality.py --records path/to/ai_reads.jsonl
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")
sys.path.insert(0, str(REPO))
from src.datasets.consensus.line_rule import overlap_frac  # noqa: E402
from src.datasets.evaluations.grounding_eval.line_agreement_probe import (  # noqa: E402
    PREDS, geometric_gt, load_pages, v_iou)


def template_rate(boxes: List[List[float]]) -> Optional[float]:
    """Share of a page's lines whose x-range equals the page's modal x-range (rounded to 5‰)."""
    if len(boxes) < 4:
        return None
    key = collections.Counter((round(b[0] / 5), round(b[2] / 5)) for b in boxes)
    return key.most_common(1)[0][1] / len(boxes)


def _med(vals: List[float], d: int = 3) -> Optional[float]:
    return round(statistics.median(vals), d) if vals else None


def score_preds(model: str) -> Dict[str, Any]:
    """Geometry metrics for one model's grounded dump on the eval pages.

    :param model: LM Studio key (dump ``preds/grounded_<model>.json``).
    :returns: Summary dict.
    """
    pages = {p["doc_id"]: p for p in load_pages()}
    dump = json.loads((PREDS / f"grounded_{model}.json").read_text())
    tmpl, right, viou, wratio, drift, n_lines, parsed = [], [], [], [], [], 0, 0
    for pr in dump["preds"]:
        if not pr.get("ok"):
            continue
        parsed += 1
        gt = pages[pr["doc_id"]]["gt_lines"]
        boxes = [ln["box"] for ln in pr["pred_lines"]]
        t = template_rate(boxes)
        if t is not None:
            tmpl.append(t)
        for ln in pr["pred_lines"]:
            n_lines += 1
            if not ln.get("matched"):
                continue
            g = gt[ln["gt_idx"]]["box"]
            right.append(geometric_gt(ln["box"], gt) == ln["gt_idx"])
            viou.append(v_iou(ln["box"], g))
            wratio.append((ln["box"][2] - ln["box"][0]) / max(1.0, g[2] - g[0]))
            drift.append(((ln["box"][1] + ln["box"][3]) - (g[1] + g[3])) / 2)
    return dict(model=model, pages_parsed=parsed, lines=n_lines, lines_with_text_gt=len(right),
                template_rate_mean=round(statistics.mean(tmpl), 3) if tmpl else None,
                pages_template_ge_50=sum(t >= 0.5 for t in tmpl),
                box_on_right_line=round(sum(right) / len(right), 3) if right else None,
                vertical_iou_median=_med(viou), width_ratio_median=_med(wratio, 2),
                vertical_drift_median=_med(drift, 1), vertical_drift_abs_median=_med([abs(d) for d in drift], 1))


def score_records(path: Path) -> Dict[str, Any]:
    """Geometry metrics for ai_reads records (no GT): template rate, width vs Kraken row.

    :param path: JSONL of loader records.
    :returns: Summary dict.
    """
    tmpl, wratio, pages = [], [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ls = r["ai_read"]["lines"]
        pages += 1
        t = template_rate([l["bbox"] for l in ls])
        if t is not None:
            tmpl.append(t)
        for l in ls:
            fr = l.get("htr_fragments") or []
            if len(fr) >= 2:
                row_w = max(f[2] for f in fr) - min(f[0] for f in fr)
                if row_w > 0:
                    wratio.append((l["bbox"][2] - l["bbox"][0]) / row_w)
    return dict(file=path.name, pages=pages, template_rate_mean=round(statistics.mean(tmpl), 3) if tmpl else None,
                pages_template_ge_50=sum(t >= 0.5 for t in tmpl), width_ratio_vs_kraken_row_median=_med(wratio, 2),
                note="bbox is the evidence box in rule-v2 records; use raw-cache vlm_lines for the raw VLM box")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="*", default=[], help="model keys with preds/grounded_<key>.json")
    ap.add_argument("--records", nargs="*", default=[], help="ai_reads JSONL files")
    a = ap.parse_args()
    for m in a.preds:
        print(json.dumps(score_preds(m), ensure_ascii=False))
    for f in a.records:
        print(json.dumps(score_records(Path(f)), ensure_ascii=False))
