# File name: build_viewer.py
# Date: 9/8/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Build the self-contained grounding viewer (predicted vs ground-truth boxes).

Joins the eval queries (``grounding_eval.jsonl``) with every per-query
prediction dump in ``preds/`` (written by ``grounding_eval.py --run``) and
injects the result, plus the eval page images as downscaled JPEG data URIs,
into ``viewer.template.html``. The output is ONE html file that works from
``file://`` with no server and can be published as an Artifact when it is
under 16 MB (use ``--max-side 1300`` for that build).

Usage (from the repo root):
    .venv/bin/python src/datasets/evaluations/grounding_eval/viewer/build_viewer.py
        [--max-side 1800] [--quality 82] [--out viewer/grounding_viewer.html]
        [--preds-dir grounding_eval/preds]
"""
import argparse
import base64
import io
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
REPO = EVAL_DIR.parents[3]
QUERIES = EVAL_DIR / "grounding_eval.jsonl"
BENCH_JSON = (REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1"
              / "genizah_religious_v1.json")
TEMPLATE = HERE / "viewer.template.html"
DEFAULT_MODEL = "qwen3-vl-8b-heb-v19a-step1300"   # the eval's suffix-less default
MODES = ("locate", "read_box", "grounded")
_MODEL_RE = re.compile(r"-v(\d)(\d)([a-z]?)-step(\d+)$")


def model_meta(key: str) -> Dict[str, Any]:
    """Human label and sort key for an LM Studio model key.

    :param key: e.g. ``qwen3-vl-8b-heb-v20a-step1800``.
    :returns: ``{key, label, ver, step}`` (label like ``v2.0a · step 1800``).
    """
    m = _MODEL_RE.search(key)
    if not m:
        return {"key": key, "label": key, "ver": key, "step": 0}
    ver = f"v{m.group(1)}.{m.group(2)}{m.group(3)}"
    return {"key": key, "label": f"{ver} · step {m.group(4)}", "ver": ver, "step": int(m.group(4))}


def results_suffix(model: str) -> str:
    """Suffix the eval uses for a model's canonical ``*_results.json``.

    :param model: LM Studio model key.
    :returns: ``""`` for the eval's default model, else ``_<ver>_<step>``.
    """
    if model == DEFAULT_MODEL:
        return ""
    return f"_{model.rsplit('-', 2)[-2]}_{model.rsplit('-', 1)[-1]}"


def _median(vals: List[float]) -> Optional[float]:
    """Median or None for an empty list."""
    return round(statistics.median(vals), 4) if vals else None


def summarize(mode: str, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-query scores the way ``grounding_eval.py`` prints them.

    :param mode: Eval mode.
    :param entries: Per-query dicts with ``ok`` and the mode's score fields.
    :returns: Summary dict (``n``, ``ok`` and mode-specific medians/rates).
    """
    ok = [e for e in entries if e.get("ok")]
    s: Dict[str, Any] = {"n": len(entries), "ok": len(ok)}
    if mode == "locate":
        s.update(hit=sum(e["hit"] for e in ok), median_iou=_median([e["iou"] for e in ok]),
                 iou50=sum(e["iou"] >= 0.5 for e in ok))
    elif mode == "read_box":
        s.update(median_cer=_median([e["cer"] for e in ok]),
                 median_len_ratio=_median([e["len_ratio"] for e in ok]))
    else:
        s.update(matched=sum(e["matched"] for e in ok), gt_n=sum(e["gt_n"] for e in ok),
                 median_line_iou=_median([e["miou"] for e in ok]),
                 median_line_cer=_median([e["mcer"] for e in ok if e.get("mcer") is not None]))
    return s


def encode_image(path: Path, max_side: int, quality: int) -> Tuple[str, int, int]:
    """Downscale a page image and return it as a JPEG data URI.

    :param path: Source image.
    :param max_side: Longest side of the embedded copy, in pixels.
    :param quality: JPEG quality.
    :returns: ``(data_uri, width, height)`` of the embedded copy.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1.0:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        return ("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
                im.width, im.height)


def load_preds(preds_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Read every ``preds/<mode>_<model>.json`` dump.

    :param preds_dir: Directory of dumps.
    :returns: ``runs[model][mode] = dump`` (mode-level dict incl. ``preds``).
    """
    runs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for f in sorted(preds_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if not {"model", "mode", "preds"} <= set(d):
            continue
        runs.setdefault(d["model"], {})[d["mode"]] = d
    return runs


def build(max_side: int, quality: int, out: Path, preds_dir: Path) -> None:
    """Assemble the viewer's DATA blob and write the html.

    :param max_side: Longest side of embedded page images.
    :param quality: JPEG quality of embedded page images.
    :param out: Output html path.
    :param preds_dir: Directory of prediction dumps.
    """
    rows = [json.loads(l) for l in QUERIES.read_text().splitlines() if l.strip()]
    bench = {d["doc_id"]: d for d in json.load(open(BENCH_JSON))["docs"]}

    pages: List[Dict[str, Any]] = []
    page_ix: Dict[str, int] = {}
    queries: Dict[str, List[Dict[str, Any]]] = {m: [] for m in MODES}
    for r in rows:
        pid = Path(r["image"]).stem
        if pid not in page_ix:
            page_ix[pid] = len(pages)
            meta = bench.get(r["doc_id"], {})
            pages.append({"id": pid, "doc_id": r["doc_id"], "image": r["image"],
                          "shelf_mark": meta.get("shelf_mark", ""), "genre": meta.get("genre", ""),
                          "n_columns": meta.get("n_columns", ""),
                          "q": {m: [] for m in MODES}})
        mode = r["mode"]
        q = {k: v for k, v in r.items() if k not in ("mode", "image", "doc_id")}
        q["idx"] = len(queries[mode])
        q["page"] = pid
        queries[mode].append(q)
        pages[page_ix[pid]]["q"][mode].append(q["idx"])

    print(f"embedding {len(pages)} page images at max side {max_side}px q{quality} ...")
    for p in pages:
        src = Path(p.pop("image"))
        p["src"], p["w"], p["h"] = encode_image(src, max_side, quality)
        p["thumb"], _, _ = encode_image(src, 160, 70)
        with Image.open(src) as im:
            p["native_w"], p["native_h"] = im.size

    runs_raw = load_preds(preds_dir)
    models = sorted((model_meta(k) for k in runs_raw), key=lambda m: (m["ver"], m["step"]), reverse=True)
    runs: Dict[str, Dict[str, Any]] = {}
    for m in models:
        key = m["key"]
        runs[key] = {}
        for mode, dump in runs_raw[key].items():
            preds: List[Optional[Dict[str, Any]]] = [None] * len(queries[mode])
            for p in dump["preds"]:
                q = queries[mode][p["idx"]]
                assert p["doc_id"] == pages[page_ix[q["page"]]]["doc_id"], (key, mode, p["idx"])
                keep = {k: v for k, v in p.items()
                        if k not in ("doc_id", "image", "phrase", "gt_box", "gt_text", "gt_lines")}
                preds[p["idx"]] = keep
            entry: Dict[str, Any] = {"decoded_at": dump.get("decoded_at"),
                                     "temperature": dump.get("temperature"),
                                     "summary": summarize(mode, [p for p in preds if p]),
                                     "preds": preds}
            canon = EVAL_DIR / f"grounding_eval_{mode}{results_suffix(key)}_results.json"
            if canon.exists():
                entry["canonical"] = summarize(mode, json.loads(canon.read_text()))
                entry["canonical"]["file"] = canon.name
            runs[key][mode] = entry
            print(f"  {key:34s} {mode:9s} {entry['summary']}")

    data = {"built_at": datetime.now().isoformat(timespec="seconds"), "image_max_side": max_side,
            "pages": pages, "queries": queries, "models": models, "runs": runs}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text()
    assert "/*__DATA__*/" in html, "template lacks the /*__DATA__*/ injection marker"
    out.write_text(html.replace("/*__DATA__*/", blob, 1))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB; {len(models)} models)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-side", type=int, default=1800)
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--out", type=Path, default=HERE / "grounding_viewer.html")
    ap.add_argument("--preds-dir", type=Path, default=EVAL_DIR / "preds")
    a = ap.parse_args()
    build(a.max_side, a.quality, a.out, a.preds_dir)
