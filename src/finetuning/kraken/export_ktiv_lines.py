"""Export KTIV human-transcribed lines as Kraken recognition ground truth.

Reads the API-shape KTIV bundles and page zips exactly as the VLM builder
does (same id-level benchmark exclusion and shingle decontamination, see
:mod:`src.finetuning.qwen_hebrew.build_ktiv_dataset`), keeps only fully
transcribed lines (no gap token, no dotted or bracketed words, no ``@``
sigla), crops each line from the page image at native resolution using an
ink-centred vertical band (KTIV word boxes are vertically noisy, so the band
is re-centred on the darkest window near the box centre, sized by the
column's line pitch), rescales it to at most ``--height`` px and
writes ``<stem>.jpg`` + ``<stem>.gt.txt`` pairs in Kraken's ``path`` format,
plus ``train.txt`` / ``val.txt`` manifests split by manuscript.

Text is NFKD-normalised (the MiDRASH convention) and checked against the
recogniser's codec; lines with out-of-codec characters are dropped and the
characters counted in ``stats.json``.  Human transcriptions only — VLM
output is never used, so the two readers' errors stay uncorrelated.

Usage (from repo root):
    PYTHONPATH=. python -m src.finetuning.kraken.export_ktiv_lines \\
        --out /Volumes/home/studio_offload/datasets/kraken_ktiv_lines \\
        --codec midrash_codec.json [--limit 0] [--workers 6] [--height 160]
"""

import argparse
import io
import json
import logging
import random
import re
import unicodedata
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image as PILImage

from src.finetuning.qwen_hebrew import build_ktiv_dataset as B
from src.finetuning.qwen_hebrew.ktiv_layout import (
    GAP_TOKEN,
    hebrew_letters,
    reconstruct_page,
)

LINE_HEIGHT = 160          # stored line height; Kraken rescales to its 120-px input
MIN_LINE_LETTERS = 3
MIN_BOX_H = 12
MIN_BOX_W = 40
CROP_HALF = 0.55           # crop half-height, in line pitches
CORE = 0.6                 # ink-search window height, in line pitches
SEARCH = 0.45              # max shift of the ink window from the box centre, in pitches
PITCH_FROM_BOX = 0.8       # pitch fallback for single-line columns, in median box heights
X_MARGIN = 0.01            # horizontal margin, fraction of the line width
VAL_FRACTION = 0.04
SPLIT_SEED = 20260914
JPEG_QUALITY = 92
_BAD = re.compile(r"[.\[\]@]")
_WS = re.compile(r"\s+")
log = logging.getLogger("export_ktiv_lines")


def clean_text(text: str) -> Optional[str]:
    """Normalise a line transcription or reject it.

    :param text: Line text in logical order, as reconstructed from KTIV words.
    :type text: str
    :return: NFKD-normalised, whitespace-collapsed text, or ``None`` when the
        line holds a gap token, a dotted/bracketed (uncertain) word or fewer
        than :data:`MIN_LINE_LETTERS` Hebrew letters.
    :rtype: Optional[str]
    """
    toks = text.split()
    if not toks or any(t == GAP_TOKEN or _BAD.search(t) for t in toks):
        return None
    norm = _WS.sub(" ", unicodedata.normalize("NFKD", " ".join(toks))).strip()
    if hebrew_letters(norm) < MIN_LINE_LETTERS:
        return None
    return norm


def out_of_codec(text: str, codec: FrozenSet[str]) -> List[str]:
    """Characters of ``text`` missing from the recogniser codec.

    :param text: Normalised line text.
    :type text: str
    :param codec: Characters the model can emit.
    :type codec: FrozenSet[str]
    :return: Sorted distinct offending characters (empty when all are known).
    :rtype: List[str]
    """
    return sorted({c for c in text if c not in codec})


def column_pitch(lines: Sequence[Dict], h_med: float) -> float:
    """Line pitch (centre-to-centre spacing) of one column.

    :param lines: Column lines, each with a ``box``; any order.
    :type lines: Sequence[Dict]
    :param h_med: Median line-box height on the page, px (fallback basis).
    :type h_med: float
    :return: Median spacing between consecutive line centres, or
        :data:`PITCH_FROM_BOX` × ``h_med`` when the column has a single line.
    :rtype: float
    """
    centres = sorted((ln["box"][1] + ln["box"][3]) / 2 for ln in lines)
    gaps = [b - a for a, b in zip(centres, centres[1:]) if b - a > 1]
    if not gaps:
        return PITCH_FROM_BOX * h_med
    return float(sorted(gaps)[len(gaps) // 2])


def ink_centre(page: np.ndarray, x0: int, x1: int, centre: float, pitch: float) -> float:
    """Re-centre a line on its ink.

    Builds the row darkness profile of the page strip ``[x0, x1)`` around
    ``centre``, subtracts the background level and picks the window of height
    :data:`CORE` × ``pitch`` with the most ink among windows whose centre lies
    within ±:data:`SEARCH` × ``pitch`` of the box centre (so a neighbouring
    line cannot win).

    :param page: Grayscale page as a 2-D uint8 array.
    :type page: np.ndarray
    :param x0: Left edge of the line, px.
    :type x0: int
    :param x1: Right edge of the line, px.
    :type x1: int
    :param centre: Vertical centre of the KTIV line box, px.
    :type centre: float
    :param pitch: Column line pitch, px.
    :type pitch: float
    :return: Vertical centre of the ink window, px.
    :rtype: float
    """
    height = page.shape[0]
    r0 = max(0, int(centre - (SEARCH + CORE / 2) * pitch))
    r1 = min(height, int(centre + (SEARCH + CORE / 2) * pitch) + 1)
    core = max(2, int(CORE * pitch))
    if r1 - r0 <= core or x1 - x0 < 2:
        return centre
    dark = 255.0 - page[r0:r1, max(0, x0):x1].mean(axis=1)
    dark = np.clip(dark - np.percentile(dark, 30), 0, None)
    k = max(3, int(pitch // 6))
    dark = np.convolve(dark, np.ones(k) / k, mode="same")
    csum = np.concatenate([[0.0], np.cumsum(dark)])
    sums = csum[core:] - csum[:-core]              # window sums, start index = row offset
    starts = np.arange(len(sums))
    shift = np.abs(r0 + starts + core / 2 - centre) / max(1.0, SEARCH * pitch)
    best = int(np.argmax(sums * (1.0 - 0.3 * np.clip(shift, 0, 1))))
    win = dark[best:best + core]
    if win.sum() <= 0:
        return centre
    rows = np.arange(best, best + core)
    return float(r0 + (rows * win).sum() / win.sum())


def export_page(job: Dict) -> Dict:
    """Crop and write every clean line of one page (pool worker).

    :param job: ``sys_num``, ``fl``, ``zip``, ``member``, ``columns`` (from
        :func:`reconstruct_page`), ``out``, ``height`` and ``codec`` (list).
    :type job: Dict
    :return: ``{"stats": Counter-like dict, "oov": {char: count},
        "written": [relative image paths], "index": [{"path", "w", "h", "letters"}]}``.
    :rtype: Dict
    """
    stats: Counter = Counter()
    oov: Counter = Counter()
    written: List[str] = []
    index: List[Dict] = []
    im = B.load_page_image(Path(job["zip"]), job["member"]).convert("L")
    width, height = im.size
    heights = sorted(ln["box"][3] - ln["box"][1]
                     for col in job["columns"] for ln in col["lines"])
    h_med = heights[len(heights) // 2] if heights else 0
    if h_med < MIN_BOX_H:
        return {"stats": {"page_degenerate": 1}, "oov": {}, "written": [], "index": []}
    codec = frozenset(job["codec"])
    page = np.asarray(im)
    ms_dir = Path(job["out"]) / "lines" / job["sys_num"]
    ms_dir.mkdir(parents=True, exist_ok=True)
    for ci, col in enumerate(job["columns"]):
        pitch = column_pitch(col["lines"], h_med)
        for i, ln in enumerate(col["lines"]):
            stats["lines"] += 1
            text = clean_text(ln["text"])
            if text is None:
                stats["lines_damaged_or_short"] += 1
                continue
            bad = out_of_codec(text, codec) if codec else []
            if bad:
                stats["lines_out_of_codec"] += 1
                oov.update(bad)
                continue
            x0, y0, x1, y1 = ln["box"]
            c = ink_centre(page, int(x0), int(x1), (y0 + y1) / 2, pitch)
            top = max(0, int(round(c - CROP_HALF * pitch)))
            bottom = min(height, int(round(c + CROP_HALF * pitch)))
            xm = X_MARGIN * (x1 - x0)
            box = (max(0, int(x0 - xm)), top, min(width, int(x1 + xm)), bottom)
            if box[3] - box[1] < MIN_BOX_H or box[2] - box[0] < MIN_BOX_W:
                stats["lines_tiny_box"] += 1
                continue
            crop = im.crop(box)
            if crop.height > job["height"]:
                new_w = max(1, round(crop.width * job["height"] / crop.height))
                crop = crop.resize((new_w, job["height"]), PILImage.LANCZOS)
            stem = f"{job['sys_num']}_{job['fl']}_c{ci}_l{i:03d}"
            crop.save(ms_dir / f"{stem}.jpg", "JPEG", quality=JPEG_QUALITY)
            (ms_dir / f"{stem}.gt.txt").write_text(text, encoding="utf-8")
            rel = f"lines/{job['sys_num']}/{stem}.jpg"
            written.append(rel)
            index.append({"path": rel, "w": crop.width, "h": crop.height, "letters": hebrew_letters(text)})
            stats["lines_written"] += 1
            stats["letters"] += hebrew_letters(text)
    return {"stats": dict(stats), "oov": dict(oov), "written": written, "index": index}


def collect_jobs(bundles: List[dict], ktiv_dir: Path, shingles: set, out: Path,
                 height: int, codec: Sequence[str], limit: int = 0) -> Tuple[List[Dict], Counter]:
    """Gate manuscripts and pages the way the VLM builder does and build page jobs.

    :param bundles: API-shape bundles after id-level benchmark exclusion.
    :type bundles: List[dict]
    :param ktiv_dir: KTIV directory holding the image zips.
    :type ktiv_dir: Path
    :param shingles: Benchmark letter shingles for manuscript decontamination.
    :type shingles: set
    :param out: Export root.
    :type out: Path
    :param height: Stored line height, px.
    :type height: int
    :param codec: Recogniser codec characters (empty = no codec filtering).
    :type codec: Sequence[str]
    :param limit: Max manuscripts (0 = all).
    :type limit: int
    :return: (page jobs, manuscript/page counters).
    :rtype: Tuple[List[Dict], Counter]
    """
    stats: Counter = Counter()
    jobs: List[Dict] = []
    for n, doc in enumerate(bundles):
        if limit and n >= limit:
            break
        sys_num = doc["sys_num"]
        pages = []
        for p in doc.get("pages") or []:
            items = ((p.get("annotation_page") or {}).get("items")) or []
            page = reconstruct_page(items)
            if page["lines"]:
                pages.append((p.get("fl") or "", page))
        if not pages:
            continue
        if shingles and any(B.shingle_hits(pg["text"], shingles) >= B.DECONTAM_MIN_HITS
                            for _, pg in pages):
            stats["ms_contaminated"] += 1
            continue
        stats["ms_kept"] += 1
        for fl, page in pages:
            found = B.find_zip_member(ktiv_dir, sys_num, fl)
            if not found:
                stats["page_no_image"] += 1
                continue
            zpath, member = found
            with zipfile.ZipFile(zpath) as zf, zf.open(member) as fh:
                width, height_px = PILImage.open(io.BytesIO(fh.read())).size
            max_x = max(ln["box"][2] for ln in page["lines"])
            max_y = max(ln["box"][3] for ln in page["lines"])
            if (max_x > width * 1.02 or max_y > height_px * 1.02
                    or min(width, height_px) < B.MIN_IMAGE_SIDE_PX):
                stats["page_out_of_frame"] += 1
                continue
            stats["pages"] += 1
            jobs.append({"sys_num": sys_num, "fl": fl, "zip": str(zpath), "member": member,
                         "columns": page["columns"], "out": str(out), "height": height,
                         "codec": list(codec)})
    return jobs, stats


def split_manuscripts(sys_nums: Sequence[str], val_fraction: float, seed: int) -> Set[str]:
    """Pick the validation manuscripts with a seeded shuffle.

    :param sys_nums: Manuscript ids (duplicates allowed).
    :type sys_nums: Sequence[str]
    :param val_fraction: Share of manuscripts held out.
    :type val_fraction: float
    :param seed: RNG seed.
    :type seed: int
    :return: Validation manuscript ids.
    :rtype: Set[str]
    """
    ms = sorted(set(sys_nums))
    random.Random(seed).shuffle(ms)
    return set(ms[:max(1, int(round(len(ms) * val_fraction)))])


def export(out: Path, codec: Sequence[str], limit: int, workers: int, height: int,
           val_fraction: float, seed: int) -> Dict:
    """Run the full export and write manifests plus ``stats.json``.

    :param out: Export root (created).
    :type out: Path
    :param codec: Recogniser codec characters (empty = no filtering).
    :type codec: Sequence[str]
    :param limit: Max manuscripts (0 = all).
    :type limit: int
    :param workers: Pool size.
    :type workers: int
    :param height: Stored line height, px.
    :type height: int
    :param val_fraction: Share of manuscripts held out for validation.
    :type val_fraction: float
    :param seed: Split seed.
    :type seed: int
    :return: The stats dict that was written.
    :rtype: Dict
    """
    out.mkdir(parents=True, exist_ok=True)
    bundles = B.load_bundles(B.KTIV_DIR)
    bundles, excl = B.exclude_benchmark_manuscripts(bundles, B.KTIV_DIR)
    shingles = B.benchmark_shingles(B.BENCH_PATH) | B.benchmark_shingles(B.RELIGIOUS_BENCH_PATH)
    jobs, stats = collect_jobs(bundles, B.KTIV_DIR, shingles, out, height, codec, limit)
    stats.update({f"excluded_{k}": v for k, v in excl.items()})
    log.info("%d pages from %d manuscripts to export", len(jobs), stats["ms_kept"])
    oov: Counter = Counter()
    written_by_ms: Dict[str, List[str]] = {}
    index_fh = (out / "index.jsonl").open("w", encoding="utf-8")
    with Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(export_page, jobs, chunksize=4), 1):
            stats.update(res["stats"])
            oov.update(res["oov"])
            for rel in res["written"]:
                written_by_ms.setdefault(rel.split("/")[1], []).append(rel)
            for rec in res.get("index", []):
                index_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                log.info("%d/%d pages, %d lines written", k, len(jobs), stats["lines_written"])
    index_fh.close()
    val_ms = split_manuscripts(list(written_by_ms), val_fraction, seed)
    train = sorted(p for ms, ps in written_by_ms.items() if ms not in val_ms for p in ps)
    val = sorted(p for ms, ps in written_by_ms.items() if ms in val_ms for p in ps)
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    summary = {"stats": dict(stats), "train_lines": len(train), "val_lines": len(val),
               "train_manuscripts": len(written_by_ms) - len(val_ms), "val_manuscripts": len(val_ms),
               "out_of_codec_chars": {c: n for c, n in oov.most_common(60)},
               "height": height, "val_fraction": val_fraction, "seed": seed}
    (out / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("done: train %d lines / val %d lines; stats -> %s", len(train), len(val), out / "stats.json")
    return summary


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="export root (NAS)")
    ap.add_argument("--codec", type=Path, default=None, help="JSON list of codec characters")
    ap.add_argument("--limit", type=int, default=0, help="max manuscripts (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--height", type=int, default=LINE_HEIGHT)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    codec = json.loads(args.codec.read_text(encoding="utf-8")) if args.codec else []
    export(args.out, codec, args.limit, args.workers, args.height, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
