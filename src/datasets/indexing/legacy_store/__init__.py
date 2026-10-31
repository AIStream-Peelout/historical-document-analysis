# File name: __init__.py
# Date: 10/30/26
# Author: Isaac Godfried. Coded by Claude.
"""Legacy image store material downloaded by hand: transcription PDFs and page images.

Modules:

* ``pdf_transcriptions`` -- structured ingest of the transcription PDFs in both styles (one
  record per folio of a header-and-folio PDF, one per side of an old one-side-per-file PDF: raw
  lines, cleaned lines, font findings, provenance and quality signals, page-row exclusions).
* ``raw_package`` -- one manifest over every raw root (old scrape folders, new downloads, the
  bucket snapshot): images deduplicated by SHA-256 and dated against the label-race fix, PDFs by
  text digest, catalogue / joins / bibliography attached, KTIV status and image policy per shelf
  mark.

Everything here is internal research material; nothing is republished.
"""
