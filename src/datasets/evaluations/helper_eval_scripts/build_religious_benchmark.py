"""Build the Genizah religious-text benchmark (separate from PGP + printed Talmud).

A THIRD benchmark namespace in the project's taxonomy:
  1. printed Talmud (Vilna)  2. PGP documentary (letters/ketubbot/legal)
  3. THIS: Genizah religious text (Talmud MSS + whatever Bible/liturgy has GT)

Scored entirely separately from the 131-fragment PGP benchmark — never merged,
so the already-reported documentary numbers stand untouched.

Sources (all decontaminated via DecontamGate — provably disjoint from the 906
v1.9a-trained KTIV sys_nums, genizah_clean_v2, and the PGP benchmark):
* API-shape KTIV bundles → CLEAN whole-word GT via ktiv_layout.reconstruct_page
  (geometry-based reading order + fragment-word merge — the training-builder
  logic). Preferred.
* non-Talmud docs are included regardless of shape ("some is better than none"
  — Bible/liturgy Genizah is almost entirely un-transcribed on NLI), tagged so
  they can be attended to despite tiny n.

Every page is tagged: genre, n_columns (single vs multi — the hard class the
T-S NS 219.40 Megillah failure exposed), gt_shape (api_clean / dom_flat),
gt_words. Images pulled from the NLI IIIF server (not Cloudflare-gated) at
~2800px wide (≈ the model's 6.5MP policy) via the per-page ``fl`` id.

Usage (from repo root):
    PYTHONPATH=. python -m src.datasets.evaluations.helper_eval_scripts.build_religious_benchmark \\
        [--limit N] [--talmud-cap 130]
"""

import argparse
import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None   # KTIV native scans can exceed PIL's bomb limit

from src.datasets.evaluations.helper_eval_scripts.decontam_gate import DecontamGate
from src.finetuning.qwen_hebrew.ktiv_layout import reconstruct_page, hebrew_letters

_REPO = Path(__file__).resolve().parents[4]
KT = _REPO / "src/datasets/raw_data/cairo_genizah/ktiv"
POOL = _REPO / "src/datasets/raw_data/cairo_genizah/religious_eval/viable_pool.json"
OUT = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1"
IIIF = "https://iiif.nli.org.il/IIIFv21/{fl}/full/2800,/0/default.jpg"
MIN_LETTERS_TALMUD = 120
MIN_LETTERS_OTHER = 40      # non-Talmud is rare/precious — keep shorter fragments
MAX_PAGES_PER_DOC = 2       # spread the budget across manuscripts (hand diversity)
TALMUD_PAGE_BUDGET = 140    # all non-Talmud pages kept; Talmud capped here


def _bundle_for(sys_num: str) -> dict:
    """Load the transcription bundle for a sys_num (richest if duplicated).

    :param sys_num: NLI system number.
    :type sys_num: str
    :return: Parsed bundle dict, or {} if none.
    :rtype: dict
    """
    best, best_n = {}, -1
    for f in KT.glob(f"*{sys_num}*_transcription*.json"):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        n = sum(len((p.get("text") or "")) for p in d.get("pages") or []) or \
            sum(len((p.get("annotation_page") or {}).get("items") or []) for p in d.get("pages") or [])
        if n > best_n:
            best, best_n = d, n
    return best


def _meta_for(sys_num: str) -> dict:
    """Load a metadata record for a sys_num (for shelfmark + genre).

    :param sys_num: NLI system number.
    :type sys_num: str
    :return: Metadata dict, or {}.
    :rtype: dict
    """
    for f in KT.glob(f"*{sys_num}*.json"):
        if f.name.endswith("_transcription.json") or "_transcription(" in f.name:
            continue
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        s = d.get("sys_num") or (d.get("shelfmarks") or {}).get("system_no")
        if str(s) == sys_num:
            return d
    return {}


def _dom_page_text(page: dict) -> str:
    """Flattened text for a DOM-shape page (fallback GT).

    :param page: Bundle page dict.
    :type page: str
    :return: Text.
    :rtype: str
    """
    if page.get("text"):
        return page["text"]
    return "\n".join(page.get("lines") or [])


_TARGET_W = 2800   # ~6.5MP at these aspect ratios; matches the model resolution policy


def _save_scaled(im: Image.Image, dest: Path) -> bool:
    """Save an image downscaled to ~_TARGET_W wide (never upscale).

    :param im: Source PIL image.
    :type im: Image.Image
    :param dest: Destination path.
    :type dest: Path
    :return: True on success.
    :rtype: bool
    """
    im = im.convert("RGB")
    if im.width > _TARGET_W:
        im = im.resize((_TARGET_W, round(im.height * _TARGET_W / im.width)), Image.LANCZOS)
    im.save(dest, "JPEG", quality=90)
    return True


def _image_from_zip(sys_num: str, fl: str, dest: Path) -> bool:
    """Extract the page image for ``fl`` from a local KTIV image zip.

    The scrape already downloaded ``*_images.zip`` per manuscript, with members
    named ``NNNN_<FL>.jpg`` — so no network is needed. Preferred over IIIF.

    :param sys_num: NLI system number.
    :type sys_num: str
    :param fl: NLI FL identifier.
    :type fl: str
    :param dest: Destination path.
    :type dest: Path
    :return: True when extracted and saved.
    :rtype: bool
    """
    for zpath in KT.glob(f"*{sys_num}*_images*.zip"):
        try:
            with zipfile.ZipFile(zpath) as zf:
                member = next((n for n in zf.namelist()
                              if fl in n and n.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))), None)
                if not member:
                    continue
                return _save_scaled(Image.open(io.BytesIO(zf.read(member))), dest)
        except (zipfile.BadZipFile, OSError):
            continue
    return False


def _get_image(sys_num: str, fl: str, dest: Path) -> str:
    """Get the page image: local zip first, IIIF fallback.

    :param sys_num: NLI system number.
    :type sys_num: str
    :param fl: NLI FL identifier.
    :type fl: str
    :param dest: Destination path.
    :type dest: Path
    :return: Source used ("zip" / "iiif" / "" on failure).
    :rtype: str
    """
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    if _image_from_zip(sys_num, fl, dest):
        return "zip"
    try:
        req = urllib.request.Request(IIIF.format(fl=fl), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            _save_scaled(Image.open(io.BytesIO(r.read())), dest)
        return "iiif"
    except Exception as exc:
        print(f"    image {fl} failed (no zip + IIIF {exc})")
        return ""


def build(limit: int, talmud_cap: int) -> None:
    """Assemble the benchmark: catalog + images + summary.

    :param limit: Max docs to process (0 = all).
    :type limit: int
    :param talmud_cap: Max Talmud docs (non-Talmud always fully included).
    :type talmud_cap: int
    """
    gate = DecontamGate()
    pool = json.load(open(POOL))
    # non-Talmud first (rare, always keep), then Talmud up to the cap, clean GT first
    non_t = [p for p in pool if p["genre"] != "talmud/halakha"]
    talmud = sorted([p for p in pool if p["genre"] == "talmud/halakha"],
                    key=lambda p: (p["shape"] != "nli_ktiv_viewer", -p["words"]))
    chosen = non_t + talmud[:talmud_cap]
    if limit:
        chosen = chosen[:limit]
    print(f"pool {len(pool)}: non-Talmud {len(non_t)}, Talmud (capped) {len(talmud[:talmud_cap])}"
          f" → processing {len(chosen)} docs")

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    entries, stats = [], {"pages": 0, "img_fail": 0, "too_short": 0, "recontam": 0,
                          "img_zip": 0, "img_iiif": 0}
    talmud_pages = 0
    for i, p in enumerate(chosen, 1):
        sys_num = p["sys_num"]
        is_talmud = p["genre"] == "talmud/halakha"
        if is_talmud and talmud_pages >= TALMUD_PAGE_BUDGET:
            continue
        meta = _meta_for(sys_num)
        sm = meta.get("shelf_mark", "")
        clean, reason = gate.check(sys_num, sm)
        if not clean:
            stats["recontam"] += 1
            continue
        bundle = _bundle_for(sys_num)
        min_letters = MIN_LETTERS_TALMUD if is_talmud else MIN_LETTERS_OTHER
        # reconstruct all pages, then prefer multi-column (the hard class) up to the per-doc cap
        cand = []
        for page in bundle.get("pages") or []:
            fl = page.get("fl")
            if not fl:
                continue
            items = (page.get("annotation_page") or {}).get("items") or []
            if items:
                rec = reconstruct_page(items)
                gt, ncol, shape = rec["text"], len(rec["columns"]), "api_clean"
            else:
                gt = _dom_page_text(page)
                ncol, shape = (2 if "\n\n" in gt else 1), "dom_flat"
            if hebrew_letters(gt) < min_letters:
                stats["too_short"] += 1
                continue
            cand.append((fl, gt, ncol, shape, page.get("image_name")))
        cand.sort(key=lambda c: (-c[2], -hebrew_letters(c[1])))   # multi-column first, then longer
        for fl, gt, ncol, shape, iname in cand[:MAX_PAGES_PER_DOC]:
            img = OUT / "images" / f"{sys_num}_{fl}.jpg"
            src = _get_image(sys_num, fl, img)
            if not src:
                stats["img_fail"] += 1
                continue
            if src == "zip":
                stats["img_zip"] += 1
            elif src == "iiif":
                stats["img_iiif"] += 1
            entries.append({
                "doc_id": f"KTIV_{sys_num}_{fl}", "sys_num": sys_num, "fl": fl,
                "shelf_mark": sm, "genre": p["genre"], "is_talmud": is_talmud,
                "n_columns": ncol, "gt_shape": shape, "gt": gt,
                "gt_letters": hebrew_letters(gt), "image": str(img), "image_name": iname,
            })
            stats["pages"] += 1
            if is_talmud:
                talmud_pages += 1
        if i % 25 == 0:
            print(f"  {i}/{len(chosen)} docs, {stats['pages']} pages "
                  f"({talmud_pages} talmud)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"benchmark": "genizah_religious_v1", "docs": entries},
              open(OUT / "genizah_religious_v1.json", "w"), ensure_ascii=False, indent=1)
    # summary
    import collections
    by_genre = collections.Counter(e["genre"] for e in entries)
    by_col = collections.Counter(("multi" if e["n_columns"] >= 2 else "single") for e in entries)
    by_shape = collections.Counter(e["gt_shape"] for e in entries)
    non_talmud_pages = [e for e in entries if not e["is_talmud"]]
    summary = dict(total_pages=len(entries), docs=len({e["sys_num"] for e in entries}),
                   by_genre=dict(by_genre), by_columns=dict(by_col), by_gt_shape=dict(by_shape),
                   non_talmud_pages=len(non_talmud_pages), **stats)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print("\n=== BENCHMARK BUILT ===")
    print(json.dumps(summary, indent=1))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--talmud-cap", type=int, default=130)
    args = ap.parse_args()
    build(args.limit, args.talmud_cap)


if __name__ == "__main__":
    main()
