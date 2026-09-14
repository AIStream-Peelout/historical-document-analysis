#!/usr/bin/env python3
"""Upload the Bodleian direct-scrape masters to GCS.

Copies every downloaded Bodleian image (``bodleian/images/<canonical_id>/<stem>.jpg``)
up to ``gs://cairo-genizah-es-json/BODLEIAN/<canonical_id>/<stem>.jpg``. These
are exactly the bucket-relative paths recorded in the merged records'
``images.bodleian`` block: both sides resolve files via
:func:`bodleian_images.bodleian_image_sources`, so a pointer the merge writes is
always a file this uploader produces. Idempotent: existing objects are skipped
unless ``--overwrite``.

Run (needs Google credentials, e.g. ``GOOGLE_APPLICATION_CREDENTIALS``)::

    python -m src.datasets.merging.upload_bodleian_images            # upload
    python -m src.datasets.merging.upload_bodleian_images --dry-run  # list only
    python -m src.datasets.merging.upload_bodleian_images --overwrite

This is the only step that touches the bucket; the merge itself just records
the (deterministic) pointers, so the JSONL is valid before the upload
completes. Run this BEFORE a re-index goes live in the web app, or records
flagged ``populated`` will point at objects that do not exist yet.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
from typing import Dict

import dotenv

from src.datasets.merging.bodleian_images import (
    DEFAULT_BODLEIAN_DIR,
    DEFAULT_RECORDS_GLOB,
    GCS_BUCKET,
    BodleianImages,
    bodleian_image_sources,
    load_bodleian_records,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _content_type(object_path: str) -> str:
    """Return the MIME type to store for an image object.

    :param object_path: Bucket-relative object path (its extension decides).
    :returns: A MIME type string (defaults to ``image/jpeg``).
    """
    guessed, _ = mimetypes.guess_type(object_path)
    return guessed or "image/jpeg"


def _upload_source(bucket, source: BodleianImages, overwrite: bool) -> Dict[str, int]:
    """Upload one fragment's image files.

    :param bucket: A ``google.cloud.storage.Bucket``.
    :param source: The fragment's resolved image files.
    :param overwrite: Re-upload objects that already exist.
    :returns: ``{"uploaded": n, "skipped": n}`` for this fragment.
    """
    uploaded = skipped = 0
    for object_path, local_file in source.entries:
        blob = bucket.blob(object_path)
        if not overwrite and blob.exists():
            skipped += 1
            continue
        blob.upload_from_filename(local_file, content_type=_content_type(object_path))
        uploaded += 1
    return {"uploaded": uploaded, "skipped": skipped}


def upload(
    records_glob: str = DEFAULT_RECORDS_GLOB,
    bodleian_dir: str = DEFAULT_BODLEIAN_DIR,
    bucket_name: str = GCS_BUCKET,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    """Upload Bodleian images to the GCS bucket.

    :param records_glob: Glob matching the scraper's ``records/*.json`` files.
    :param bodleian_dir: Root of the scrape tree (``local_path`` is relative to it).
    :param bucket_name: Destination GCS bucket.
    :param overwrite: Re-upload objects that already exist.
    :param dry_run: Log intended uploads without contacting GCS.
    :returns: Counts dict ``{uploaded, skipped, fragments}``.
    """
    sources = bodleian_image_sources(load_bodleian_records(records_glob), bodleian_dir)
    bucket = None
    if not dry_run:
        # Load the repo-root .env (works from any cwd) so GOOGLE_APPLICATION_CREDENTIALS
        # is available to google.auth.default() / storage.Client().
        dotenv.load_dotenv(os.path.join(_REPO_ROOT, ".env"))
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS is not set. Add it to the repo-root "
                ".env (path to a service-account JSON) or run "
                "`gcloud auth application-default login`."
            )
        from google.cloud import storage  # imported lazily so --dry-run needs no creds
        bucket = storage.Client().bucket(bucket_name)

    uploaded = skipped = 0
    for cid, source in sorted(sources.items()):
        if dry_run:
            for object_path, local_file in source.entries:
                print(f"[dry-run] {object_path}  <- {local_file}")
                uploaded += 1
            continue
        counts = _upload_source(bucket, source, overwrite)
        uploaded += counts["uploaded"]
        skipped += counts["skipped"]
        print(f"{cid}: done ({uploaded} uploaded, {skipped} skipped so far)")

    result = {"uploaded": uploaded, "skipped": skipped, "fragments": len(sources)}
    print(f"Complete: {result}")
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-glob", default=DEFAULT_RECORDS_GLOB)
    parser.add_argument("--bodleian-dir", default=DEFAULT_BODLEIAN_DIR)
    parser.add_argument("--bucket", default=GCS_BUCKET)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    upload(args.records_glob, args.bodleian_dir, args.bucket, args.overwrite, args.dry_run)


if __name__ == "__main__":
    main()
