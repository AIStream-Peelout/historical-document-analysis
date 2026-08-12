"""Build the consensus-pilot inventory: untranscribed fragments with images.

Selects records from ``merged_shelfmarks.jsonl`` that (a) carry at least one
FJP image filename, (b) share no image file with the FJP transcription corpus
(``genizah_img_trans.jsonl``), and (c) have no PGP id with substantial edition
content in ``footnotes.csv`` — i.e. fragments nobody has transcribed.  A
seeded sample is then verified against GCS (HEAD request; the bucket holds
only part of the FJP image set) until the requested count of fetchable
documents is reached, so the pilot also measures image availability.

Identifier note: exclusion matches on IMAGE FILENAMES and PGP ids, not on
canonical ids — the corpora use different id schemes and shelf-mark
normalisation is exactly the hard problem the merge project exists to solve.

Usage (from repo root):
    PYTHONPATH=. python -m src.datasets.consensus.build_pilot_inventory \\
        --count 500 --seed 20260811
"""

import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
RAW = _REPO / "src/datasets/raw_data"
MERGED_JSONL = RAW / "cairo_genizah/merged/merged_shelfmarks.jsonl"
IMG_TRANS_JSONL = RAW / "genizah_cleanup/genizah_img_trans.jsonl"
FOOTNOTES_CSV = RAW / "cairo_genizah/pgp_raw/data/footnotes.csv"
DEFAULT_OUT_DIR = RAW / "cairo_genizah/consensus_pilot"

GCS_IMAGES_BASE = "https://storage.googleapis.com/cairo-genizah-es-json/images"
PGP_CONTENT_MIN_CHARS = 100


def load_transcribed_image_files() -> set:
    """Return lowercased image basenames of the FJP transcription corpus.

    :return: Set of image filenames belonging to transcribed documents.
    :rtype: set
    """
    files = set()
    with open(IMG_TRANS_JSONL) as fh:
        for line in fh:
            rec = json.loads(line)
            for url in rec.get("image_urls") or []:
                files.add(url.rsplit("/", 1)[-1].lower())
    return files


def load_pgp_transcribed_ids() -> set:
    """Return PGP document ids that carry substantial edition content.

    :return: Set of PGP document_id strings with content > 100 chars.
    :rtype: set
    """
    csv.field_size_limit(sys.maxsize)
    ids = set()
    with open(FOOTNOTES_CSV, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            content = row.get("content") or ""
            if len(content) > PGP_CONTENT_MIN_CHARS:
                ids.add(row["document_id"].strip())
    return ids


def build_pool() -> list:
    """Collect untranscribed merged records that have FJP images.

    :return: Candidate records (dicts from the merged corpus).
    :rtype: list
    """
    trans_files = load_transcribed_image_files()
    pgp_trans = load_pgp_transcribed_ids()
    pool = []
    with open(MERGED_JSONL) as fh:
        for line in fh:
            rec = json.loads(line)
            fjp = (rec.get("images") or {}).get("fjp") or []
            if not fjp:
                continue
            if any(fn.lower() in trans_files for fn in fjp):
                continue
            if any(p in pgp_trans for p in (rec.get("pgpids") or [])):
                continue
            pool.append(rec)
    return pool


def resolve_image_url(filename: str, timeout: float = 10.0) -> str:
    """Return the fetchable GCS URL for an FJP image filename, or ``""``.

    Tries the filename as stored, then lowercased (the bucket mixes both).

    :param filename: FJP image basename from the merged corpus.
    :type filename: str
    :param timeout: Per-request timeout in seconds.
    :type timeout: float
    :return: Working URL or empty string when the object is absent.
    :rtype: str
    """
    for name in (filename, filename.lower()):
        url = f"{GCS_IMAGES_BASE}/{name}"
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return url
        except Exception:
            continue
    return ""


def sample_available(pool: list, count: int, seed: int) -> tuple:
    """Seeded-sample the pool, keeping records whose first image is in GCS.

    :param pool: Candidate records.
    :type pool: list
    :param count: Number of fetchable documents wanted.
    :type count: int
    :param seed: RNG seed (pilot reproducibility).
    :type seed: int
    :return: (inventory rows, number of candidates probed).
    :rtype: tuple
    """
    rng = random.Random(seed)
    order = rng.sample(range(len(pool)), len(pool))
    rows, probed = [], 0
    for idx in order:
        if len(rows) >= count:
            break
        rec = pool[idx]
        fjp = rec["images"]["fjp"]
        probed += 1
        url = resolve_image_url(fjp[0])
        if not url:
            continue
        rows.append(dict(
            canonical_id=rec["canonical_id"],
            shelfmark_display=rec.get("shelfmark_display"),
            institution=rec.get("institution"),
            pgpids=rec.get("pgpids") or [],
            sources_present=rec.get("sources_present") or [],
            image_file=fjp[0],
            image_url=url,
            n_images=len(fjp),
        ))
        if probed % 50 == 0:
            print(f"  probed {probed}, kept {len(rows)}", flush=True)
    return rows, probed


def main() -> None:
    """Build and write the pilot inventory plus its stats file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    pool = build_pool()
    print(f"untranscribed pool with FJP images: {len(pool)}")
    rows, probed = sample_available(pool, args.count, args.seed)
    availability = len(rows) / probed if probed else 0.0
    print(f"kept {len(rows)} of {probed} probed "
          f"(GCS availability {availability:.0%})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inv_path = args.out_dir / "pilot_inventory.jsonl"
    with open(inv_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = dict(
        pool_size=len(pool), requested=args.count, kept=len(rows),
        probed=probed, gcs_availability=round(availability, 4),
        seed=args.seed,
    )
    with open(args.out_dir / "pilot_inventory_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"wrote {inv_path}")


if __name__ == "__main__":
    main()
