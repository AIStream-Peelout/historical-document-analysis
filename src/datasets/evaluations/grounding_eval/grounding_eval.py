"""Quantitative grounding eval for the current model (v19a) on KTIV geometry.

Three modes, matching the three v20 grounding-task candidates:

* ``locate``  — phrase -> bbox.  Target = union of the phrase's word boxes
  (proper phrase-level target; the Aug-28 probe scored against full-line
  boxes, understating IoU).  Phrase must be unique on the page.
* ``read_box`` — line bbox -> text.  CER vs the line's GT text.
* ``grounded`` — whole-page transcription as JSON lines with bbox_2d each
  ("pull boxes from what it transcribes"): parse rate + per-line text CER
  and box IoU after text-similarity alignment to GT lines.

All coordinates are 0-1000 normalized to the ORIGINAL scan frame (KTIV boxes
live there; normalization makes them valid for the resized benchmark image
too, since aspect is preserved).  Eval pages: clean (non-mash) religious
benchmark pages with geometry mode == "geometry" — decontaminated by
construction and images already local.

Usage:
  --build            write grounding_eval.jsonl (CPU only)
  --run MODE         run v19a on one mode (locate|read_box|grounded)
  --limit N          cap queries
"""
import argparse
import asyncio
import difflib
import io
import json
import random
import re
import statistics
import sys
import zipfile
from pathlib import Path

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")
sys.path.insert(0, str(REPO))
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from src.finetuning.qwen_hebrew.ktiv_bundles import pick_best_bundle  # noqa: E402
from src.finetuning.qwen_hebrew.ktiv_layout import (  # noqa: E402
    extract_words, reconstruct_page, hebrew_letters)
from src.datasets.evaluations.metrics import cer_pair  # noqa: E402

BENCH = REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1"
KT = REPO / "src/datasets/raw_data/cairo_genizah/ktiv"
OUT = Path(__file__).parent / "grounding_eval.jsonl"
MODEL = "qwen3-vl-8b-heb-v19a-step1300"
_HEB = re.compile(r"[א-ת]")
_JSON_BOX = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

P_LOCATE = ('Locate the exact Hebrew phrase "{phrase}" on this manuscript page. '
            'Respond with ONLY a JSON object {{"bbox_2d": [x1, y1, x2, y2]}} '
            'giving the phrase\'s bounding box. No other text.')
P_READBOX = ('Transcribe ONLY the Hebrew text inside the region bbox_2d = '
             '[{x0}, {y0}, {x1}, {y1}] (coordinates normalized 0-1000) on this '
             'manuscript page. Return the text alone, no commentary.')
P_GROUNDED = ('Transcribe this manuscript page line by line. Respond with ONLY '
              'a JSON array; each element {"text": "...", "bbox_2d": [x1, y1, '
              'x2, y2]} gives one line\'s transcription and its bounding box. '
              'Preserve reading order.')


def norm_box(box, W, H):
    """Normalize a pixel box to 0-1000 ints."""
    x0, y0, x1, y1 = box
    return [round(1000 * x0 / W), round(1000 * y0 / H),
            round(1000 * x1 / W), round(1000 * y1 / H)]


def iou(a, b):
    """IoU of two [x0,y0,x1,y1] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ar = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ar if ar > 0 else 0.0


def original_dims(sys_num, fl):
    """(W, H) of the original scan from the image zip, or None."""
    for zpath in KT.glob(f"*{sys_num}*_images*.zip"):
        try:
            with zipfile.ZipFile(zpath) as zf:
                member = next((n for n in zf.namelist() if fl in n), None)
                if member:
                    with Image.open(io.BytesIO(zf.read(member))) as im:
                        return im.width, im.height
        except (zipfile.BadZipFile, OSError):
            continue
    return None


def build(n_pages=24, seed=20260830):
    """Assemble eval queries from clean geometry-mode benchmark pages."""
    rng = random.Random(seed)
    docs = json.load(open(BENCH / "genizah_religious_v1.json"))["docs"]
    clean = [d for d in docs if not d.get("gt_mash_v1") in (True, "True")
             and Path(d["image"]).exists()]
    rng.shuffle(clean)
    rows, pages_used = [], 0
    for d in clean:
        if pages_used >= n_pages:
            break
        win = pick_best_bundle(KT.glob(f"*{d['sys_num']}*_transcription*.json"))
        if not win:
            continue
        page = next((p for p in win[1].get("pages", [])
                     if p.get("fl") == d["fl"] or p.get("image_id") == d["fl"]), None)
        if page is None:
            continue
        items = (page.get("annotation_page") or {}).get("items") or []
        rec = reconstruct_page(items)
        if rec["mode"] != "geometry" or len(rec["lines"]) < 4:
            continue
        dims = original_dims(d["sys_num"], d["fl"])
        if not dims:
            continue
        W, H = dims
        words = [w for w in extract_words(items)
                 if not w["gap"] and len(_HEB.findall(w["text"])) >= 2]
        page_letters = "".join(_HEB.findall(rec["text"]))
        # --- locate: 3 unique phrases of 2-3 consecutive words on one line
        lines_w = []
        from src.finetuning.qwen_hebrew.ktiv_layout import split_columns, cluster_lines, _median_height
        lh = _median_height(words)
        for col in split_columns(words, lh):
            lines_w.extend(cluster_lines(col, lh))
        cands = []
        for lw in lines_w:
            ws = [w for w in lw if not w["gap"] and len(_HEB.findall(w["text"])) >= 2]
            for i in range(len(ws) - 1):
                for span in (3, 2):
                    seg = ws[i:i + span]
                    if len(seg) < span:
                        continue
                    phrase = " ".join(w["text"] for w in seg)
                    pl = "".join(_HEB.findall(phrase))
                    if len(pl) < 8 or page_letters.count(pl) != 1:
                        continue
                    box = [min(w["box"][0] for w in seg), min(w["box"][1] for w in seg),
                           max(w["box"][2] for w in seg), max(w["box"][3] for w in seg)]
                    cands.append((phrase, box))
                    break
        rng.shuffle(cands)
        for phrase, box in cands[:3]:
            rows.append(dict(mode="locate", doc_id=d["doc_id"], image=d["image"],
                             phrase=phrase, gt_box=norm_box(box, W, H)))
        # --- read_box: 2 mid-page lines with >= 15 letters
        good_lines = [ln for ln in rec["lines"]
                      if hebrew_letters(ln["text"]) >= 15 and "[...]" not in ln["text"]]
        rng.shuffle(good_lines)
        for ln in good_lines[:2]:
            rows.append(dict(mode="read_box", doc_id=d["doc_id"], image=d["image"],
                             gt_text=ln["text"], gt_box=norm_box(ln["box"], W, H)))
        # --- grounded: the page itself (lines + boxes)
        rows.append(dict(mode="grounded", doc_id=d["doc_id"], image=d["image"],
                         gt_lines=[dict(text=ln["text"], box=norm_box(ln["box"], W, H))
                                   for ln in rec["lines"]]))
        pages_used += 1
    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in rows:
        by[r["mode"]] = by.get(r["mode"], 0) + 1
    print(f"built {len(rows)} queries over {pages_used} pages: {by}")


async def run(mode, limit):
    """Run v19a on one mode and score."""
    from src.models.ocr.lms_transcriber import transcribe_with_lm_studio
    rows = [json.loads(l) for l in open(OUT)]
    rows = [r for r in rows if r["mode"] == mode][:limit or None]
    print(f"{mode}: {len(rows)} queries vs {MODEL}")
    res = []
    for i, r in enumerate(rows, 1):
        if mode == "locate":
            prompt = P_LOCATE.format(phrase=r["phrase"])
        elif mode == "read_box":
            b = r["gt_box"]
            prompt = P_READBOX.format(x0=b[0], y0=b[1], x1=b[2], y1=b[3])
        else:
            prompt = P_GROUNDED
        out = await transcribe_with_lm_studio(MODEL, r["image"], prompt,
                                              max_tokens=3500 if mode == "grounded" else 300)
        if not out or not out.strip():
            res.append(dict(ok=False))
            continue
        if mode == "locate":
            m = _JSON_BOX.search(out)
            if not m:
                res.append(dict(ok=False)); continue
            pb = [int(v) for v in m.groups()]
            g = r["gt_box"]
            cx, cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
            res.append(dict(ok=True, iou=iou(pb, g),
                            hit=int(g[0] <= cx <= g[2] and g[1] <= cy <= g[3])))
        elif mode == "read_box":
            cer, _ = cer_pair(out, r["gt_text"])
            # coverage: is the answer roughly this line and not the whole page?
            res.append(dict(ok=True, cer=cer,
                            len_ratio=len(_HEB.findall(out)) / max(1, len(_HEB.findall(r["gt_text"])))))
        else:
            try:
                arr = json.loads(re.search(r"\[.*\]", out, re.S).group(0))
                assert isinstance(arr, list)
            except Exception:
                res.append(dict(ok=False)); continue
            pairs = []
            for pl in arr:
                t, b = str(pl.get("text", "")), pl.get("bbox_2d")
                if not t or not (isinstance(b, list) and len(b) == 4):
                    continue
                best = max(r["gt_lines"], key=lambda g:
                           difflib.SequenceMatcher(a=t, b=g["text"]).ratio())
                sim = difflib.SequenceMatcher(a=t, b=best["text"]).ratio()
                if sim >= 0.5:
                    pairs.append((iou([float(v) for v in b], best["box"]),
                                  cer_pair(t, best["text"])[0]))
            res.append(dict(ok=True, n_lines=len(arr), matched=len(pairs),
                            gt_n=len(r["gt_lines"]),
                            miou=statistics.median([p[0] for p in pairs]) if pairs else 0.0,
                            mcer=statistics.median([p[1] for p in pairs]) if pairs else None))
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    ok = [r for r in res if r.get("ok")]
    print(f"\n== {mode}: parse-ok {len(ok)}/{len(res)}")
    if mode == "locate" and ok:
        print(f"   center-hit {sum(r['hit'] for r in ok)}/{len(ok)}  "
              f"median IoU {statistics.median(r['iou'] for r in ok):.3f}  "
              f"IoU>=0.5: {sum(r['iou'] >= 0.5 for r in ok)}/{len(ok)}")
    if mode == "read_box" and ok:
        print(f"   median CER {statistics.median(r['cer'] for r in ok):.3f}  "
              f"median len-ratio {statistics.median(r['len_ratio'] for r in ok):.2f}")
    if mode == "grounded" and ok:
        print(f"   lines matched {sum(r['matched'] for r in ok)}/{sum(r['gt_n'] for r in ok)}  "
              f"median line-IoU {statistics.median(r['miou'] for r in ok):.3f}  "
              f"median line-CER {statistics.median(r['mcer'] for r in ok if r['mcer'] is not None):.3f}")
    Path(__file__).with_name(f"grounding_eval_{mode}_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", choices=("locate", "read_box", "grounded"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.build:
        build()
    if args.run:
        asyncio.run(run(args.run, args.limit))
