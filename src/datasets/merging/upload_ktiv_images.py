#!/usr/bin/env python3
"""Upload KTIV scan images from the local zip archives / image folders to GCS.

Streams each fragment image out of the downloaded KTIV ``*.zip`` archives — or,
for manuscripts whose zip download failed, copies the loose files the scraper's
IIIF fallback saved into ``*_images/`` folders — up to
``gs://cairo-genizah-es-json/KTIV/<sys_num>/<member>``. These are exactly the
bucket-relative paths recorded in the merged records' ``images.ktiv`` block:
both sides resolve sources via :func:`ktiv_images.ktiv_image_sources`, so a
pointer the merge writes is always a path this uploader produces. No on-disk
extraction; ``header.pdf`` covers are skipped. Idempotent: existing objects are
skipped unless ``--overwrite``.

Run (needs Google credentials, e.g. ``GOOGLE_APPLICATION_CREDENTIALS``)::

    python -m src.datasets.merging.upload_ktiv_images            # upload
    python -m src.datasets.merging.upload_ktiv_images --dry-run  # list only
    python -m src.datasets.merging.upload_ktiv_images --overwrite

This is the only step that touches the bucket; the merge itself just records
the (deterministic) pointers, so the JSONL is valid before the upload
completes. Run this BEFORE re-running the merge goes live in the web app, or
records flagged ``populated`` will point at objects that do not exist yet.
"""

from __future__ import annotations

import argparse
import glob
import mimetypes
import os
import zipfile
from typing import Dict

import dotenv

from src.datasets.merging.ktiv_images import (
    GCS_BUCKET,
    ManuscriptImages,
    image_folders,
    ktiv_image_sources,
)

# Repo-anchored default location of the KTIV archives and image folders.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_KTIV_DIR = os.path.join(
    _REPO_ROOT, "src", "datasets", "raw_data", "cairo_genizah", "ktiv"
)
DEFAULT_ZIP_GLOB = os.path.join(DEFAULT_KTIV_DIR, "*.zip")


def _content_type(object_path: str) -> str:
    """Return the MIME type to store for an image object.

    :param object_path: Bucket-relative object path (its extension decides).
    :returns: A MIME type string (defaults to ``image/jpeg``).
    """
    guessed, _ = mimetypes.guess_type(object_path)
    return guessed or "image/jpeg"


def _upload_source(bucket, source: ManuscriptImages, overwrite: bool) -> Dict[str, int]:
    """Upload one manuscript's images from its zip or folder source.

    :param bucket: A ``google.cloud.storage.Bucket``.
    :param source: The manuscript's resolved image source.
    :param overwrite: Re-upload objects that already exist.
    :returns: ``{"uploaded": n, "skipped": n}`` for this manuscript.
    """
    uploaded = skipped = 0
    zf = zipfile.ZipFile(source.container) if source.kind == "zip" else None
    try:
        for object_path, source_name in source.entries:
            blob = bucket.blob(object_path)
            if not overwrite and blob.exists():
                skipped += 1
                continue
            content_type = _content_type(object_path)
            if zf is not None:
                with zf.open(source_name) as fh:
                    blob.upload_from_file(fh, content_type=content_type)
            else:
                blob.upload_from_filename(
                    os.path.join(source.container, source_name),
                    content_type=content_type,
                )
            uploaded += 1
    finally:
        if zf is not None:
            zf.close()
    return {"uploaded": uploaded, "skipped": skipped}


def upload(
    zip_glob: str = DEFAULT_ZIP_GLOB,
    ktiv_dir: str = DEFAULT_KTIV_DIR,
    bucket_name: str = GCS_BUCKET,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    """Upload KTIV images (zip members and folder files) to the GCS bucket.

    :param zip_glob: Glob matching the KTIV ``*.zip`` archives.
    :param ktiv_dir: Directory whose subdirectories are candidate image folders.
    :param bucket_name: Destination GCS bucket.
    :param overwrite: Re-upload objects that already exist.
    :param dry_run: Log intended uploads without contacting GCS.
    :returns: Counts dict ``{uploaded, skipped, manuscripts, from_zip, from_folder}``.
    """
    sources = ktiv_image_sources(glob.glob(zip_glob), image_folders(ktiv_dir))
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
    kinds = {"zip": 0, "folder": 0}
    for sys_num, source in sorted(sources.items()):
        kinds[source.kind] += 1
        if dry_run:
            for object_path, _ in source.entries:
                print(f"[dry-run] {object_path}  <- {source.kind}")
                uploaded += 1
            continue
        counts = _upload_source(bucket, source, overwrite)
        uploaded += counts["uploaded"]
        skipped += counts["skipped"]
        print(f"{sys_num} ({source.kind}): done "
              f"({uploaded} uploaded, {skipped} skipped so far)")

    result = {
        "uploaded": uploaded,
        "skipped": skipped,
        "manuscripts": len(sources),
        "from_zip": kinds["zip"],
        "from_folder": kinds["folder"],
    }
    print(f"Complete: {result}")
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-glob", default=DEFAULT_ZIP_GLOB)
    parser.add_argument("--ktiv-dir", default=DEFAULT_KTIV_DIR)
    parser.add_argument("--bucket", default=GCS_BUCKET)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    upload(args.zip_glob, args.ktiv_dir, args.bucket, args.overwrite, args.dry_run)


if __name__ == "__main__":
    main()
