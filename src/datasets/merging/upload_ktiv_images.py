#!/usr/bin/env python3
"""Upload KTIV scan images from the local zip archives to GCS.

Streams each fragment image out of the downloaded KTIV ``*.zip`` archives
straight to ``gs://cairo-genizah-es-json/KTIV/<sys_num>/<member>`` — the same
bucket-relative paths recorded in the merged records' ``images.ktiv`` block (see
:mod:`ktiv_images`). No on-disk extraction; the ``header.pdf`` covers are
skipped. Idempotent: existing objects are skipped unless ``--overwrite``.

Run (needs Google credentials, e.g. ``GOOGLE_APPLICATION_CREDENTIALS``)::

    python -m src.datasets.merging.upload_ktiv_images            # upload
    python -m src.datasets.merging.upload_ktiv_images --dry-run  # list only
    python -m src.datasets.merging.upload_ktiv_images --overwrite

This is the only step that touches the bucket; the merge itself just records the
(deterministic) pointers, so the JSONL is valid before the upload completes. .
"""

from __future__ import annotations

import argparse
import glob
import os
import zipfile
from typing import Optional

import dotenv

from src.datasets.merging.ktiv_images import (
    GCS_BUCKET,
    _is_image,
    _pick_zip_per_sysnum,
    gcs_object_path,
)

# Repo-anchored default location of the KTIV archives.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_ZIP_GLOB = os.path.join(
    _REPO_ROOT, "src", "datasets", "raw_data", "cairo_genizah", "ktiv", "*.zip"
)


def upload(
    zip_glob: str = DEFAULT_ZIP_GLOB,
    bucket_name: str = GCS_BUCKET,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    """Stream KTIV images from local zip files to the GCS bucket..

    :param zip_glob: Glob matching the KTIV ``*.zip`` archives.
    :param bucket_name: Destination GCS bucket.
    :param overwrite: Re-upload objects that already exist.
    :param dry_run: Log intended uploads without contacting GCS.
    :returns: Counts dict ``{uploaded, skipped, manuscripts}``.
    """
    chosen = _pick_zip_per_sysnum(glob.glob(zip_glob))
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
    for sys_num, zpath in sorted(chosen.items()):
        with zipfile.ZipFile(zpath) as zf:
            for member in sorted(n for n in zf.namelist() if _is_image(n)):
                object_path = gcs_object_path(sys_num, member)
                if dry_run:
                    print(f"[dry-run] {object_path}")
                    uploaded += 1
                    continue
                blob = bucket.blob(object_path)
                if not overwrite and blob.exists():
                    skipped += 1
                    continue
                with zf.open(member) as fh:
                    blob.upload_from_file(fh, content_type="image/jpeg")
                uploaded += 1
        print(f"{sys_num}: done ({uploaded} uploaded, {skipped} skipped so far)")

    result = {"uploaded": uploaded, "skipped": skipped, "manuscripts": len(chosen)}
    print(f"Complete: {result}")
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-glob", default=DEFAULT_ZIP_GLOB)
    parser.add_argument("--bucket", default=GCS_BUCKET)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    upload(args.zip_glob, args.bucket, args.overwrite, args.dry_run)


if __name__ == "__main__":
    main()
