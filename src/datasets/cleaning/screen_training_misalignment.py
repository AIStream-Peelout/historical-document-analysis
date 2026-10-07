# File name: screen_training_misalignment.py
# Date: 8/9/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""Screen Genizah TRAINING docs for image/transcription misalignment.

The verified benchmark lost 19/150 fragments to pairings where the image is
not (or is only part of) the page the edition transcribed. The training set
inherits the same source defect. This screen combines two evidence layers:

1. **PGP metadata flags** (joined by shelfmark): records whose PGP shelfmark
   is a join (``+``), whose side is "recto and verso", or which carry multiple
   IIIF manifests describe an object with more surfaces than our single
   training image.
2. **Kraken empirical check** (MiDRASH model via the Docker microservice):
   for each training image, transcribe and compute

   * ``containment`` — fraction of Kraken's character 5-grams present in the
     training answer (the benchmark's misalignment detector: near-zero means
     the image is a *different page* than the transcription), and
   * ``gt_ratio`` — GT letters / Kraken letters (a large ratio with a normal
     Kraken read means the transcription covers *more* document than the
     pictured side — e.g. T-S NS 184.58, a ~7-line scrap carrying a
     1,951-letter GT — which containment alone cannot catch).

Kraken here is a screening instrument only; its output never enters training
data. Drop decisions are conservative: a doc is proposed for DROP only when
metadata and Kraken agree; Kraken-only anomalies go to a review list.

Stage 2 appends one JSON line per doc to ``--results`` and skips already
present doc_ids, so the long job can be killed and resumed freely.

Usage::

    python -m src.datasets.cleaning.screen_training_misalignment \
        [--stage metadata|kraken|report|all] [--manifest PATH]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer  # noqa: E402
from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import (  # noqa: E402
    letters_only,
    overlap_with_gt,
)
from src.finetuning.qwen_hebrew.build_genizah_dataset import render_answer  # noqa: E402

_CLEANUP_DIR = _REPO / "src/datasets/raw_data/genizah_cleanup"
_PGP_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/pgp_raw/data"
DEFAULT_MANIFEST = _CLEANUP_DIR / "train_clean_manifest_repaired.jsonl"
DEFAULT_IMAGES_DIR = _REPO / "genizah_images/merged_v1/clean"
DEFAULT_FLAGS = _CLEANUP_DIR / "misalignment_metadata_flags.json"
DEFAULT_RESULTS = _CLEANUP_DIR / "misalignment_kraken_results.jsonl"
DEFAULT_REPORT = _CLEANUP_DIR / "misalignment_screen_report.json"

KRAKEN_URL = "http://localhost:8002"
KRAKEN_MODEL = str(_REPO / "src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel")

CONTAINMENT_DROP = 0.12   # benchmark audit threshold
GT_RATIO_FLAG = 2.5       # GT letters vs Kraken letters
MIN_KRAKEN_LETTERS = 150  # below this, Kraken read too little to be evidence
# Strong-evidence tier: at >=500 letters read, "Kraken produced garbage" is no
# longer a plausible explanation for near-zero containment on a Hebrew-script
# page — the image is simply not the transcribed page. Verified on the JTS ENA
# family that the CONVERSE (tiny reads) is instrument blindness (graph-paper
# backing defeats binarization), so weak reads are never drop evidence.
STRONG_KRAKEN_LETTERS = 500
STRONG_CONTAINMENT = 0.05
WEAK_SUPERSET_RATIO = 4.0
MANUAL_DROPS = _CLEANUP_DIR / "misalignment_manual_drops.json"


def _norm(s: str) -> str:
    """Digit-boundary-safe shelfmark join key.

    :param s: Shelfmark string.
    :returns: Normalized key (see :meth:`ShelfmarkNormalizer.loose_key`).
    """
    return ShelfmarkNormalizer.loose_key(s)


def _variants(shelf_mark: str) -> Set[str]:
    """Generate join-key variants for a training shelfmark.

    :param shelf_mark: Manifest ``shelf_mark`` value.
    :returns: Set of normalized candidate keys.
    """
    n = _norm(shelf_mark)
    out = {n}
    for pref in ("oxford", "bodl", "cambridgecul", "cambridge"):
        if n.startswith(pref):
            out.add(n[len(pref):])
    return out


def metadata_flags(manifest_path: Path, flags_path: Path) -> Dict[str, Dict[str, Any]]:
    """Join training docs to PGP records and persist misalignment signals.

    :param manifest_path: Training manifest JSONL.
    :param flags_path: Output JSON path (doc_id -> signal dict).
    :returns: The flags mapping.
    """
    import pandas as pd

    recs = [json.loads(l) for l in open(manifest_path)]
    frags = pd.read_csv(_PGP_DIR / "fragments.csv", dtype=str, keep_default_na=False)
    docs = pd.read_csv(_PGP_DIR / "documents.csv", dtype=str, keep_default_na=False)
    doc_by_id = {r["pgpid"].strip(): r for _, r in docs.iterrows()}

    frag_idx: Dict[str, List[Any]] = {}
    for _, f in frags.iterrows():
        for key in _variants(f["shelfmark"]):
            frag_idx.setdefault(key, []).append(f)
        if f["shelfmarks_historic"]:
            for h in f["shelfmarks_historic"].split(";"):
                frag_idx.setdefault(_norm(h), []).append(f)

    flags: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        hit = None
        for key in _variants(r["shelf_mark"]):
            if key in frag_idx:
                hit = frag_idx[key]
                break
        if hit is None:
            flags[r["doc_id"]] = {"matched": False}
            continue
        pgpids = {p.strip() for f in hit for p in str(f["pgpids"]).split(";") if p.strip()}
        sig = {"matched": True, "join": False, "side_both": False,
               "multi_iiif": False, "n_pgp_docs": len(pgpids), "languages": []}
        langs: Set[str] = set()
        for pid in pgpids:
            d = doc_by_id.get(pid)
            if d is None:
                continue
            sig["join"] = sig["join"] or "+" in d["shelfmark"]
            sig["side_both"] = sig["side_both"] or "recto and verso" in d["side"]
            sig["multi_iiif"] = sig["multi_iiif"] or ";" in d["iiif_urls"]
            if d["languages_primary"]:
                langs.update(p.strip() for p in d["languages_primary"].split(";"))
        sig["languages"] = sorted(langs)
        sig["strong"] = sig["join"] or sig["multi_iiif"]
        sig["at_risk"] = sig["strong"] or sig["side_both"]
        flags[r["doc_id"]] = sig
    flags_path.write_text(json.dumps(flags, indent=1))
    n_risk = sum(1 for s in flags.values() if s.get("at_risk"))
    n_strong = sum(1 for s in flags.values() if s.get("strong"))
    print(f"metadata flags: {len(flags)} docs, at_risk={n_risk}, strong={n_strong}")
    return flags


def kraken_transcribe(image_path: Path, timeout: float = 300.0) -> Optional[str]:
    """Transcribe one image through the Kraken microservice.

    :param image_path: Local image file.
    :param timeout: Request timeout in seconds.
    :returns: Raw line-per-row text, or ``None`` on service failure.
    """
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{KRAKEN_URL}/transcribe",
            data={"model_path": KRAKEN_MODEL},
            files={"image": (image_path.name, f)},
            timeout=timeout,
        )
    if resp.status_code != 200:
        return None
    return resp.json().get("text", "")


def kraken_screen(manifest_path: Path, images_dir: Path, results_path: Path,
                  limit: Optional[int] = None) -> None:
    """Run the resumable Kraken pass over all training images.

    :param manifest_path: Training manifest JSONL (repaired text preferred).
    :param images_dir: Directory of ``<doc_id>.jpg`` training images.
    :param results_path: JSONL results file (append; doc_ids present are skipped).
    :param limit: Optional cap on new docs this invocation (smoke tests).
    """
    done: Set[str] = set()
    if results_path.exists():
        for line in open(results_path):
            done.add(json.loads(line)["doc_id"])
    requests.post(f"{KRAKEN_URL}/preload", json={"model_path": KRAKEN_MODEL}, timeout=120)

    todo = []
    for line in open(manifest_path):
        rec = json.loads(line)
        if rec["doc_id"] in done:
            continue
        matches = list(images_dir.glob(rec["doc_id"] + ".*"))
        if matches:
            todo.append((rec, matches[0]))
    if limit:
        todo = todo[:limit]
    print(f"kraken screen: {len(done)} done, {len(todo)} to run", flush=True)

    with open(results_path, "a") as out:
        t0 = time.time()
        for i, (rec, img) in enumerate(todo, 1):
            gt = render_answer(rec["cleaned_text"])
            gt_letters = letters_only(gt)
            text = kraken_transcribe(img)
            row: Dict[str, Any] = {"doc_id": rec["doc_id"]}
            if text is None:
                row["error"] = "service_failure"
            else:
                k_letters = letters_only(text)
                row["kraken_letters"] = len(k_letters)
                row["gt_letters"] = len(gt_letters)
                row["containment"] = round(overlap_with_gt(text, gt_letters), 4)
                row["gt_ratio"] = round(len(gt_letters) / max(len(k_letters), 1), 2)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)} ({rate:.1f}s/doc, "
                      f"~{rate * (len(todo) - i) / 3600:.1f}h left)", flush=True)


def build_report(flags_path: Path, results_path: Path, report_path: Path) -> Dict[str, Any]:
    """Combine metadata flags and Kraken results into drop/review lists.

    :param flags_path: Stage-1 flags JSON.
    :param results_path: Stage-2 results JSONL.
    :param report_path: Output report JSON.
    :returns: The report dict.
    """
    flags = json.loads(flags_path.read_text())
    rows = [json.loads(l) for l in open(results_path)]
    by_id = {r["doc_id"]: r for r in rows}

    manual = json.loads(MANUAL_DROPS.read_text()) if MANUAL_DROPS.exists() else {}
    drop, review_wrong_page, review_superset = [], [], []
    review_weak_superset, weak_read = [], []
    for doc_id, r in by_id.items():
        if "error" in r:
            weak_read.append(doc_id)
            continue
        if doc_id in manual:
            drop.append({"doc_id": doc_id, **r, "mode": "manual",
                         "reason": manual[doc_id]})
            continue
        sig = flags.get(doc_id, {})
        # Kraken is a Hebrew-script instrument: on Arabic-script pages it
        # force-decodes cursive into voluminous Hebrew-alphabet garbage, so
        # containment evidence is void there (verified: T-S Ar. docs read
        # 800+ letters at containment 0 while correctly paired). Any PGP
        # language that is Arabic-but-not-Judaeo-Arabic voids kraken drops.
        arabic_possible = any(
            "arabic" in l.lower() and "judaeo" not in l.lower()
            and "judeo" not in l.lower()
            for l in sig.get("languages", []))
        low = (not arabic_possible
               and r["containment"] < CONTAINMENT_DROP
               and r["kraken_letters"] >= MIN_KRAKEN_LETTERS)
        strong = (not arabic_possible
                  and r["containment"] < STRONG_CONTAINMENT
                  and r["kraken_letters"] >= STRONG_KRAKEN_LETTERS)
        superset = (not arabic_possible
                    and r["gt_ratio"] >= GT_RATIO_FLAG
                    and r["kraken_letters"] >= MIN_KRAKEN_LETTERS)
        if strong:
            drop.append({"doc_id": doc_id, **r, "signals": sig,
                         "mode": "wrong_page_strong"})
        elif low and sig.get("at_risk"):
            drop.append({"doc_id": doc_id, **r, "signals": sig})
        elif low:
            review_wrong_page.append({"doc_id": doc_id, **r})
        elif superset and (sig.get("at_risk") or sig.get("side_both")):
            drop.append({"doc_id": doc_id, **r, "signals": sig, "mode": "superset"})
        elif superset:
            review_superset.append({"doc_id": doc_id, **r})
        elif r["kraken_letters"] < MIN_KRAKEN_LETTERS:
            if r["gt_ratio"] >= WEAK_SUPERSET_RATIO:
                review_weak_superset.append({"doc_id": doc_id, **r})
            weak_read.append(doc_id)

    report = {
        "containment_drop_threshold": CONTAINMENT_DROP,
        "gt_ratio_flag_threshold": GT_RATIO_FLAG,
        "min_kraken_letters": MIN_KRAKEN_LETTERS,
        "strong_kraken_letters": STRONG_KRAKEN_LETTERS,
        "strong_containment": STRONG_CONTAINMENT,
        "weak_superset_ratio": WEAK_SUPERSET_RATIO,
        "n_scored": len(by_id),
        "n_drop": len(drop),
        "n_review_wrong_page": len(review_wrong_page),
        "n_review_superset": len(review_superset),
        "n_review_weak_superset": len(review_weak_superset),
        "n_weak_read": len(weak_read),
        "drop": sorted(drop, key=lambda r: r.get("containment", 0)),
        "review_wrong_page": sorted(review_wrong_page, key=lambda r: r["containment"]),
        "review_superset": sorted(review_superset, key=lambda r: -r["gt_ratio"]),
        "review_weak_superset": sorted(review_weak_superset,
                                       key=lambda r: -r["gt_ratio"]),
        "weak_read": sorted(weak_read),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"report: scored={report['n_scored']} drop={report['n_drop']} "
          f"review_wrong_page={report['n_review_wrong_page']} "
          f"review_superset={report['n_review_superset']} "
          f"weak_superset={report['n_review_weak_superset']} "
          f"weak={report['n_weak_read']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["metadata", "kraken", "report", "all"],
                        default="all")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--flags", type=Path, default=DEFAULT_FLAGS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap new kraken docs this run (smoke test)")
    args = parser.parse_args()
    if args.stage in ("metadata", "all"):
        metadata_flags(args.manifest, args.flags)
    if args.stage in ("kraken", "all"):
        kraken_screen(args.manifest, args.images_dir, args.results, args.limit)
    if args.stage in ("report", "all"):
        build_report(args.flags, args.results, args.report)
