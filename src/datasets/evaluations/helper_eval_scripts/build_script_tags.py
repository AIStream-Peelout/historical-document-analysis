# File name: build_script_tags.py
# Date: 8/9/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""Build the EVAL-ONLY language/script sidecar for the Genizah benchmark.

Joins each benchmark fragment to PGP metadata by shelfmark and records its
``languages_primary`` / ``languages_secondary`` plus a coarse script bucket,
so ``score_genizah_offline.py`` can stratify metrics per script (the
cursive/Arabic weakness axis). The frozen benchmark files are NEVER touched:
tags live in a sidecar JSON next to them, and nothing from this file enters
training data, prompts, or model inputs at training or inference time.

Buckets (priority order when multiple primaries):

* ``arabic_script`` — Arabic-language records (Arabic script on the page)
* ``judaeo_arabic`` — Arabic language in Hebrew script
* ``aramaic``       — Aramaic (legal formulae, usually with Hebrew)
* ``hebrew``        — Hebrew
* ``untagged``      — no PGP match or empty language fields

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/build_script_tags.py
"""
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))

from src.datasets import shelfmarks  # noqa: E402

_BENCH_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
_PGP_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/pgp_raw/data"
SIDECAR = _BENCH_DIR / "genizah_test_v1_script_tags.json"


def _variants(shelf_mark: str) -> List[str]:
    """Candidate join keys for one benchmark shelfmark, most exact first.

    :func:`src.datasets.shelfmarks.loose_key` (institution-qualified, the institution inferred from the mark) of
    every :func:`src.datasets.shelfmarks.variants` spelling, in that order (the mark itself first); spellings that
    cannot be keyed are dropped.

    :param shelf_mark: Raw shelfmark.
    :type shelf_mark: str
    :returns: Distinct loose keys (empty when no spelling can be keyed).
    :rtype: List[str]
    """
    out: List[str] = []
    for key in (shelfmarks.loose_key(v) for v in shelfmarks.variants(shelf_mark)):
        if key and key not in out:
            out.append(key)
    return out


def bucket_for(primary: str, secondary: str) -> str:
    """Map PGP language fields to a coarse script bucket.

    :param primary: ``languages_primary`` cell (semicolon-separated).
    :param secondary: ``languages_secondary`` cell.
    :returns: Bucket name.
    """
    langs = primary.lower()
    if "judaeo-arabic" in langs or "judeo-arabic" in langs:
        return "judaeo_arabic"
    if "arabic" in langs:
        return "arabic_script"
    if "aramaic" in langs:
        return "aramaic"
    if "hebrew" in langs:
        return "hebrew"
    # Fall back to secondary before giving up.
    langs2 = secondary.lower()
    for name, b in (("judaeo-arabic", "judaeo_arabic"), ("arabic", "arabic_script"),
                    ("aramaic", "aramaic"), ("hebrew", "hebrew")):
        if name in langs2:
            return b
    return "untagged"


def main() -> None:
    """Join benchmark fragments to PGP languages and write the sidecar."""
    import pandas as pd

    frags = pd.read_csv(_PGP_DIR / "fragments.csv", dtype=str, keep_default_na=False)
    docs = pd.read_csv(_PGP_DIR / "documents.csv", dtype=str, keep_default_na=False)
    doc_by_id = {r["pgpid"].strip(): r for _, r in docs.iterrows()}

    frag_idx: Dict[str, List[Any]] = {}
    for _, f in frags.iterrows():
        marks = [f["shelfmark"]] + (f["shelfmarks_historic"].split(";") if f["shelfmarks_historic"] else [])
        # a historic spelling often keys to the current mark: index each fragment once per key
        for key in {shelfmarks.loose_key(m) for m in marks} - {""}:
            frag_idx.setdefault(key, []).append(f)

    bench_docs: Dict[str, str] = {}
    for name in ("genizah_test_v1.json", "genizah_test_v1_verified.json"):
        spec = json.loads((_BENCH_DIR / name).read_text())
        for d in spec["docs"]:
            bench_docs[d["doc_id"]] = d["shelf_mark"]

    tags: Dict[str, Dict[str, Any]] = {}
    for doc_id, shelf_mark in sorted(bench_docs.items()):
        hit = None
        for key in _variants(shelf_mark):
            if key in frag_idx:
                hit = frag_idx[key]
                break
        if hit is None:
            tags[doc_id] = {"bucket": "untagged", "matched": False}
            continue
        prim: List[str] = []
        sec: List[str] = []
        for f in hit:
            for pid in str(f["pgpids"]).split(";"):
                d = doc_by_id.get(pid.strip())
                if d is None:
                    continue
                prim.append(d["languages_primary"])
                sec.append(d["languages_secondary"])
        primary = ";".join(p for p in prim if p)
        secondary = ";".join(s for s in sec if s)
        tags[doc_id] = {
            "bucket": bucket_for(primary, secondary),
            "languages_primary": primary,
            "languages_secondary": secondary,
            "matched": True,
        }

    SIDECAR.write_text(json.dumps(tags, ensure_ascii=False, indent=1))
    dist = collections.Counter(t["bucket"] for t in tags.values())
    print(f"wrote {SIDECAR.name}: {len(tags)} docs; buckets {dict(dist.most_common())}")


if __name__ == "__main__":
    main()
