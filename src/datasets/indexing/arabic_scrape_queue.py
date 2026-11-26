"""Build the Arabic-script scraping priority queue.

The Genizah benchmarks are Hebrew-script by construction (the single
"arabic_script" bucket entry is a PGP language-tag artifact whose text is
Hebrew). This queue ranks every PGP document whose *primary language is
Arabic* (Arabic script on the page, as opposed to Judaeo-Arabic in Hebrew
letters) by how close it is to being usable as evaluation or training data.

Ground truth comes only from PGP's own data export: ``footnotes.csv`` rows
whose ``doc_relation`` is an edition carry the transcription text in
``content``. Image-store transcriptions present in the merged corpus are
reported for information but never count as usable ground truth (no
permission to redistribute or publish against them). "Permitted" images are
library images (PGP / institution IIIF manifests, NLI-Ktiv images) and, since
the 2026-11-01 rule change, image-store photographs (``image_source``
``fjp_only``): those are permitted training inputs (a dataset holding them is
private and never released) but never benchmark-grade, so such a row is tier
2 at best. For Cambridge shelfmarks that lack an IIIF link in the export, the
CUDL manifest URL is constructed from the shelfmark (the export shows the
pattern is regular); with ``--verify-cudl`` each constructed URL is fetched
and only resolving manifests count as library images (a document with
image-store photographs is still checked, and a resolving manifest replaces
them as its image source).

The merged record of a document is found by PGP id; when several records link
the same id the first is kept unless it has no images and a later one with
images answers to the document's shelf mark (:func:`pick_merged_record`).
A document no record links falls back to the records without PGP ids that
answer to its shelf mark, when exactly one does (:func:`join_merged`). The
queue lists each document's image-store photographs (``fjp_images``) and how
its merged record was found (``merged_join``).

Tiers (each maps to one scraping action):

1. benchmark-ready — PGP transcription with >= 200 Arabic letters, library
   image, not in the v1.9/v2.0 training set: download the image.
2. PGP transcription but not benchmark-grade (short, the document is in the
   training set, or its only images are image-store photographs): download
   the image; training/analysis use.
3. PGP reports a transcription that is missing from the export: scrape the
   transcription from the site plus the image.
4. image-store transcription only (internal analysis only): download the image.
5. no transcription anywhere, permitted image: image-only material for
   pseudo-labelling or manual transcription.
6. PGP transcription but no permitted image (unconstructed or unverified
   manifest, no image-store photographs): source the image from the holding
   library.

Documents with neither transcription nor permitted image are counted in the
summary, not listed.

Inputs: the PGP export directory (``documents.csv``, ``footnotes.csv``, ``fragments.csv``;
``--pgp-dir``, default the export under ``raw_data``), the merged corpus, the ``decontam``
training inventories and the local image directories. Outputs: the queue CSV, the image-only
backlog CSV and the JSON summary, written to ``--out-dir`` (default the merged directory under
``raw_data``, the files the training and benchmark builders read). A queue from another export
than the default needs an ``--out-dir`` other than that directory. The summary records the export
it was built from (``pgp_dir``, ``pgp_export_commit`` from the export's ``COMMIT`` file), which
the training builder checks against its edition text (:func:`queue_provenance`).

Usage (from the repo root):
    PYTHONPATH=. .venv/bin/python -m src.datasets.indexing.arabic_scrape_queue [--verify-cudl]
        [--pgp-dir DIR] [--out-dir DIR]
"""
import argparse
import ast
import collections
import csv
import functools
import json
import re
import time
from pathlib import Path
from typing import AbstractSet, Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer

REPO = Path(__file__).resolve().parents[3]
PGP = REPO / "src/datasets/raw_data/cairo_genizah/pgp_raw/data"
MERGED = REPO / "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
CLEAN_V2_KEYS = REPO / "src/datasets/raw_data/cairo_genizah/decontam/clean_v2_keys.json"
CLEAN_V2_IDS = REPO / "src/datasets/raw_data/cairo_genizah/decontam/clean_v2_ids.json"
LOCAL_IMAGE_DIRS = [REPO / "genizah_images/merged_v1/clean", REPO / "genizah_images/merged_v1/sample100",
                    REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1/images",
                    REPO / "src/datasets/raw_data/cairo_genizah/ktiv"]
OUT_BACKLOG = REPO / "src/datasets/raw_data/cairo_genizah/merged/arabic_image_only_backlog.csv"
OUT_CSV = REPO / "src/datasets/raw_data/cairo_genizah/merged/arabic_scrape_priority_queue.csv"
OUT_SUMMARY = REPO / "src/datasets/raw_data/cairo_genizah/merged/arabic_scrape_priority_summary.json"

MIN_BENCH_LETTERS = 200
MIN_TRANSCRIPTION_LETTERS = 20
# Library images (benchmark-grade). Image-store photographs (``fjp_only``) are permitted training inputs too since
# the 2026-11-01 rule change, but never benchmark-grade (:func:`tier_for`).
LIBRARY_IMAGE_SOURCES = frozenset({"iiif_pgp", "ktiv", "iiif_pgp+ktiv", "iiif_cudl_constructed"})
STORE_IMAGE_SOURCE = "fjp_only"
PERMITTED_IMAGE_SOURCES = set(LIBRARY_IMAGE_SOURCES) | {STORE_IMAGE_SOURCE}
CUDL_BASE = "https://cudl.lib.cam.ac.uk/iiif/"
_ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")
_HEBREW = re.compile(r"[֐-׿]")
# Cambridge shelfmark prefixes -> CUDL manifest id prefix (learned from the
# export rows that carry both a shelfmark and a CUDL manifest URL).
_CUDL_PREFIXES = [("T-S ", "MS-TS"), ("CUL Or.", "MS-OR"), ("CUL Add.", "MS-ADD"),
                  ("Moss. ", "MS-MOSSERI")]
# Image-only material is ordered by genre: page-like Arabic-script genres
# first, tabular lists last.
_TYPE_PRIORITY = {"State document": 0, "Letter": 1, "Legal document": 2,
                  "Legal query or responsum": 3, "Literary text": 4,
                  "Paraliterary text": 5, "Credit instrument or private receipt": 6,
                  "List or table": 7}

TIERS = {
    1: ("benchmark-ready: PGP transcription >=200 Arabic letters, library image, not in training",
        "download_image"),
    2: ("PGP transcription but not benchmark-grade (short, in training set, or image-store photographs only)",
        "download_image"),
    3: ("PGP lists a transcription missing from the export", "scrape_transcription+image"),
    4: ("image-store transcription only (internal analysis only)", "download_image"),
    5: ("no transcription; permitted image only", "image_only"),
    6: ("PGP transcription but no permitted image: source it from the holding library",
        "source_image_from_library"),
}
COLUMNS = ["rank", "tier", "tier_reason", "action", "pgpid", "pgp_url", "shelfmark",
           "single_fragment", "canonical_id", "library", "collection", "doc_type",
           "languages_primary", "side", "multifragment", "doc_date", "gt_source",
           "pgp_transcription_letters", "pgp_hebrew_letters", "fjp_transcription_letters",
           "in_training_clean_v2", "local_image", "image_source", "iiif_urls", "suggested_iiif", "cudl_verified",
           "fjp_images", "merged_join"]


def _as_list(value: Any) -> List[str]:
    """Coerce a list-ish field (list, literal string, scalar) to strings.

    :param value: Raw field value.
    :return: List of stripped strings.
    """
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    if isinstance(value, str) and value.startswith("["):
        try:
            return [str(v).strip() for v in ast.literal_eval(value)]
        except (ValueError, SyntaxError):
            pass
    return [s.strip() for s in str(value).split(";") if s.strip()]


def _strings(obj: Any) -> Iterable[str]:
    """Yield every string nested anywhere inside a JSON-like object.

    :param obj: Nested dict/list/str structure.
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def loose_key(shelfmark: str) -> str:
    """Loose join key for a shelfmark, tolerant of normaliser failures.

    :param shelfmark: Raw shelfmark string.
    :return: Loose key.
    """
    try:
        return ShelfmarkNormalizer.loose_key(ShelfmarkNormalizer.to_canonical_id(shelfmark))
    except Exception:  # noqa: BLE001 — odd shelfmarks fall back to the plain key
        return ShelfmarkNormalizer.loose_key(shelfmark)


def arabic_primary(row: Dict[str, str]) -> bool:
    """Whether a PGP document row is Arabic-language (Arabic script), not Judaeo-Arabic.

    :param row: documents.csv row.
    :return: True for Arabic-primary documents.
    """
    lp = row["languages_primary"].lower()
    return "arabic" in lp and "judaeo-arabic" not in lp and "judeo-arabic" not in lp


@functools.lru_cache(maxsize=None)
def _fragment_shelfmarks(fragments_csv: Path) -> FrozenSet[str]:
    """Shelfmarks listed in one ``fragments.csv``, read once per path.

    :param fragments_csv: PGP export ``fragments.csv``.
    :type fragments_csv: Path
    :return: Shelfmarks (empty when the file does not exist).
    :rtype: FrozenSet[str]
    """
    if not fragments_csv.exists():
        return frozenset()
    with open(fragments_csv, encoding="utf-8-sig") as fh:
        return frozenset(r["shelfmark"] for r in csv.DictReader(fh))


def known_fragments(pgp_dir: Optional[Path] = None) -> FrozenSet[str]:
    """Shelfmarks of the physical fragments of a PGP export, read on first use (not at import).

    :param pgp_dir: PGP export directory holding ``fragments.csv``; default :data:`PGP`.
    :type pgp_dir: Optional[Path]
    :return: Fragment shelfmarks (empty when the export has no ``fragments.csv``).
    :rtype: FrozenSet[str]
    """
    return _fragment_shelfmarks((pgp_dir if pgp_dir is not None else PGP) / "fragments.csv")


def cudl_manifest(shelfmark: str, known: Optional[AbstractSet[str]] = None) -> Optional[str]:
    """Construct the CUDL IIIF manifest URL for a single Cambridge shelfmark.

    Tokens after the collection prefix become ``-``-joined id parts: numbers
    zero-padded to five digits, letters upper-cased — e.g. ``T-S 13J1.10`` ->
    ``MS-TS-00013-J-00001-00010``, ``T-S Ar.18(1).135`` ->
    ``MS-TS-AR-00018-00001-00135``, ``CUL Or.1080 J10`` -> ``MS-OR-01080-J-00010``.

    :param shelfmark: Single (non-joined) shelfmark.
    :type shelfmark: str
    :param known: Fragment shelfmarks of the PGP export in use (:func:`known_fragments`);
        default: those of the export under :data:`PGP`, read only when needed.
    :type known: Optional[AbstractSet[str]]
    :return: Manifest URL, or ``None`` for non-Cambridge / joined shelfmarks.
    :rtype: Optional[str]
    """
    if " + " in shelfmark:
        return None
    # Manifests are per physical fragment: a document-level sub-number
    # (``T-S 13J4.15.1`` is the first document on fragment ``T-S 13J4.15``)
    # is dropped when the base shelfmark is a known fragment.
    base = re.sub(r"\.[0-9]+$", "", shelfmark)
    if base != shelfmark and base in (known if known is not None else known_fragments()):
        shelfmark = base
    for prefix, ms in _CUDL_PREFIXES:
        if shelfmark.startswith(prefix):
            rest = shelfmark[len(prefix):]
            tokens = re.findall(r"[A-Za-z]+|[0-9]+", rest)
            if not tokens:
                return None
            parts = [t.zfill(5) if t.isdigit() else t.upper() for t in tokens]
            return CUDL_BASE + "-".join([ms] + parts)
    return None


def verify_urls(urls: List[str], delay_s: float = 0.25) -> Dict[str, bool]:
    """Check which constructed manifest URLs resolve (HTTP 200).

    :param urls: Candidate manifest URLs.
    :param delay_s: Pause between requests (politeness).
    :return: url -> resolved.
    """
    import requests  # local import: only needed with --verify-cudl

    out: Dict[str, bool] = {}
    session = requests.Session()
    for i, u in enumerate(urls, 1):
        try:
            r = session.get(u, timeout=15, stream=True)
            out[u] = r.status_code == 200
            r.close()
        except requests.RequestException:
            out[u] = False
        if i % 50 == 0:
            print(f"  verified {i}/{len(urls)}: {sum(out.values())} resolve", flush=True)
        time.sleep(delay_s)
    return out


def local_image_index() -> Dict[str, str]:
    """Index every locally stored page image by id stem.

    Stems are file basenames without extension; KTIV files also index under
    their CUDL-style fragment id (page suffix ``-000-NNNNN`` stripped).

    :return: id stem -> path.
    """
    idx: Dict[str, str] = {}
    for d in LOCAL_IMAGE_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                stem = f.stem
                idx.setdefault(stem, str(f))
                idx.setdefault(re.sub(r"-000-\d+$", "", stem), str(f))
    return idx


def pgp_export_commit(pgp_dir: Optional[Path] = None) -> str:
    """The PGP export commit a pinned snapshot directory records.

    :param pgp_dir: PGP export directory; default :data:`PGP`.
    :type pgp_dir: Optional[Path]
    :return: Content of its ``COMMIT`` file, or ``""`` when there is none (an unpinned export).
    :rtype: str
    """
    path = (pgp_dir if pgp_dir is not None else PGP) / "COMMIT"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def queue_provenance(queue_csv: Path) -> Dict[str, str]:
    """The PGP export a queue was built from, as the summary written next to it records.

    :param queue_csv: A queue CSV written by :func:`main` (its summary is the
        :data:`OUT_SUMMARY` file name in the same directory).
    :type queue_csv: Path
    :return: ``{"pgp_dir", "pgp_export_commit"}``; empty strings when the summary is missing or
        predates these fields.
    :rtype: Dict[str, str]
    """
    path = queue_csv.with_name(OUT_SUMMARY.name)
    summary = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {"pgp_dir": summary.get("pgp_dir", ""), "pgp_export_commit": summary.get("pgp_export_commit", "")}


def load_pgp_editions(pgpids: set, pgp_dir: Optional[Path] = None) -> Dict[str, str]:
    """Concatenate PGP edition content per document from footnotes.csv.

    :param pgpids: Documents of interest.
    :type pgpids: set
    :param pgp_dir: PGP export directory; default :data:`PGP`.
    :type pgp_dir: Optional[Path]
    :return: pgpid -> transcription text (all edition footnotes joined).
    :rtype: Dict[str, str]
    """
    text: Dict[str, str] = collections.defaultdict(str)
    with open((pgp_dir if pgp_dir is not None else PGP) / "footnotes.csv", encoding="utf-8-sig") as fh:
        for f in csv.DictReader(fh):
            pid = f["document_id"].strip()
            if pid in pgpids and "Edition" in f["doc_relation"] and f["content"].strip():
                text[pid] += "\n" + f["content"]
    return text


def _merged_record(d: Dict[str, Any]) -> Dict[str, Any]:
    """The queue's view of one merged-corpus record.

    :param d: One parsed line of the merged corpus.
    :type d: Dict[str, Any]
    :return: ``{canonical_id, shelfmark_display, fjp_arabic, fjp, ktiv, fjp_images}``: ``fjp_arabic`` counts the
        Arabic letters of the image store's transcription text, ``fjp`` / ``ktiv`` say whether the record lists
        image-store photographs / Ktiv images, and ``fjp_images`` lists the image-store photographs exactly as
        stored (bare file names or URLs).
    :rtype: Dict[str, Any]
    """
    src = d.get("sources") or {}
    fjp_text = " ".join(s for s in _strings(src.get("fjp")) if len(s) > 20)
    images = d.get("images") if isinstance(d.get("images"), dict) else {}
    raw = images.get("fjp")
    fjp_images = [str(v) for v in raw if v] if isinstance(raw, list) else _as_list(raw)
    return {"canonical_id": str(d.get("canonical_id") or ""),
            "shelfmark_display": str(d.get("shelfmark_display") or ""),
            "fjp_arabic": len(_ARABIC.findall(fjp_text)),
            "fjp": bool(fjp_images), "ktiv": bool(images.get("ktiv")), "fjp_images": fjp_images}


def record_keys(record: Dict[str, Any]) -> FrozenSet[str]:
    """The loose shelf-mark keys a merged record answers to.

    :param record: Queue view of a merged record (:func:`_merged_record`).
    :type record: Dict[str, Any]
    :return: :func:`loose_key` of its display shelf mark and ``ShelfmarkNormalizer.loose_key`` of its canonical
        id, without empty keys.
    :rtype: FrozenSet[str]
    """
    display = record.get("shelfmark_display") or ""
    keys = {loose_key(display) if display else "", ShelfmarkNormalizer.loose_key(record.get("canonical_id") or "")}
    return frozenset(k for k in keys if k)


def _has_images(record: Dict[str, Any]) -> bool:
    """Whether a merged record lists image-store photographs or Ktiv images.

    :param record: Queue view of a merged record (:func:`_merged_record`).
    :type record: Dict[str, Any]
    :return: True when it lists either.
    :rtype: bool
    """
    return bool(record.get("fjp") or record.get("ktiv"))


def load_merged_index(merged_path: Optional[Path] = None
                      ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Read the merged corpus once into the two indexes the queue joins against.

    :param merged_path: Merged corpus JSONL; default :data:`MERGED`.
    :type merged_path: Optional[Path]
    :return: ``(linked, unlinked)``: ``linked`` maps a PGP id to every record listing it, in file order;
        ``unlinked`` maps each :func:`record_keys` key of a record listing no PGP id to those records, in file
        order. Records are :func:`_merged_record` views.
    :rtype: Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]
    """
    linked: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    unlinked: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    with open(merged_path if merged_path is not None else MERGED, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rec = _merged_record(d)
            pgpids = _as_list(d.get("pgpids"))
            for pid in dict.fromkeys(pgpids):
                linked[pid].append(rec)
            if not pgpids:
                for key in record_keys(rec):
                    unlinked[key].append(rec)
    return dict(linked), dict(unlinked)


def load_merged_by_pgpid() -> Dict[str, Dict[str, Any]]:
    """Index the merged corpus by PGP id: the first record listing each id.

    :return: pgpid -> {canonical_id, shelfmark_display, fjp_arabic, fjp, ktiv, fjp_images} (the image-store
        transcription size, the image flags and the image-store photographs as stored); every linked record per
        id is in :func:`load_merged_index`.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return {pid: records[0] for pid, records in load_merged_index()[0].items()}


def load_merged_unlinked_by_key() -> Dict[str, List[Dict[str, Any]]]:
    """Index the merged records that list no PGP id by their loose shelf-mark keys (:func:`record_keys`).

    :return: loose key -> records, in file order.
    :rtype: Dict[str, List[Dict[str, Any]]]
    """
    return load_merged_index()[1]


def pick_merged_record(records: List[Dict[str, Any]], shelf: str) -> Dict[str, Any]:
    """Choose the merged record of a document among the records linking its PGP id.

    The first record wins, as it always has, unless it lists no images (neither image-store photographs nor Ktiv
    images) and a later record lists images and answers to the document's shelf mark (one of its
    :func:`record_keys` equals :func:`loose_key` of ``shelf``): then the first such later record wins.

    :param records: The records linking the document's PGP id, in file order (may be empty).
    :type records: List[Dict[str, Any]]
    :param shelf: The document's PGP shelf mark.
    :type shelf: str
    :return: The chosen record, or ``{}`` when there is none.
    :rtype: Dict[str, Any]
    """
    if not records:
        return {}
    if _has_images(records[0]):
        return records[0]
    key = loose_key(shelf)
    for rec in records[1:]:
        if _has_images(rec) and key in record_keys(rec):
            return rec
    return records[0]


def join_merged(pgpid: str, shelf: str, linked: Dict[str, List[Dict[str, Any]]],
                unlinked: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, Any], str, str]:
    """Find a document's merged record: by PGP id, else by shelf mark among the records listing no PGP id.

    :param pgpid: The document's PGP id.
    :type pgpid: str
    :param shelf: The document's PGP shelf mark.
    :type shelf: str
    :param linked: PGP id -> linked records (:func:`load_merged_index`).
    :type linked: Dict[str, List[Dict[str, Any]]]
    :param unlinked: Loose key -> records listing no PGP id (:func:`load_merged_index`).
    :type unlinked: Dict[str, List[Dict[str, Any]]]
    :return: ``(record, merged_join, event)``: the record (``{}`` when none), ``"pgpid"`` / ``"shelfmark"`` /
        ``""`` for how it was found, and the summary event: ``"multilink_resolved_by_shelfmark"`` when
        :func:`pick_merged_record` passed over the first linked record, ``"shelfmark_fallback_used"`` when exactly
        one unlinked record answers to the shelf mark, ``"shelfmark_fallback_ambiguous"`` when several do (no
        record is then used), else ``""``.
    :rtype: Tuple[Dict[str, Any], str, str]
    """
    records = linked.get(pgpid, [])
    if records:
        rec = pick_merged_record(records, shelf)
        return rec, "pgpid", "" if rec is records[0] else "multilink_resolved_by_shelfmark"
    key = loose_key(shelf)
    candidates = unlinked.get(key, []) if key else []
    if len(candidates) == 1:
        return candidates[0], "shelfmark", "shelfmark_fallback_used"
    if candidates:
        return {}, "", "shelfmark_fallback_ambiguous"
    return {}, "", ""


def image_source(iiif: str, merged: Dict[str, Any]) -> str:
    """Classify the best available image source for a document.

    :param iiif: PGP/fragment IIIF manifest URL(s), possibly empty.
    :type iiif: str
    :param merged: Merged-corpus record (may be empty).
    :type merged: Dict[str, Any]
    :return: One of iiif_pgp+ktiv / iiif_pgp / ktiv / fjp_only (image-store photographs only) / none.
    :rtype: str
    """
    ktiv = bool(merged.get("ktiv"))
    if iiif and ktiv:
        return "iiif_pgp+ktiv"
    if iiif:
        return "iiif_pgp"
    if ktiv:
        return "ktiv"
    return STORE_IMAGE_SOURCE if merged.get("fjp") else "none"


def tier_for(row: Dict[str, Any]) -> int:
    """Assign a tier to one queue row (0 = not listed).

    Image-store photographs (``fjp_only``) are permitted training inputs since the 2026-11-01 rule change but
    never benchmark-grade: such a row is never tier 1 and otherwise follows the cascade (2, 3, 4, 5).

    :param row: Row with transcription/image/training facts filled in.
    :type row: Dict[str, Any]
    :return: Tier number, or 0 when there is nothing actionable.
    :rtype: int
    """
    pgp_letters = row["pgp_transcription_letters"]
    has_transcription = pgp_letters >= MIN_TRANSCRIPTION_LETTERS or row["gt_source"] == "pgp_site"
    if row["image_source"] not in PERMITTED_IMAGE_SOURCES:
        return 6 if has_transcription else 0
    if (pgp_letters >= MIN_BENCH_LETTERS and not row["in_training_clean_v2"]
            and row["image_source"] != STORE_IMAGE_SOURCE):
        return 1
    if pgp_letters >= MIN_TRANSCRIPTION_LETTERS:
        return 2
    if row["gt_source"] == "pgp_site":
        return 3
    if row["fjp_transcription_letters"] >= MIN_TRANSCRIPTION_LETTERS:
        return 4
    return 5


def build(verify_cudl: bool = False, pgp_dir: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the ranked queue.

    :param verify_cudl: Fetch constructed CUDL manifests and promote only the
        ones that resolve; otherwise constructed URLs stay suggestions.
    :type verify_cudl: bool
    :param pgp_dir: PGP export directory (``documents.csv``, ``footnotes.csv``,
        ``fragments.csv``); default :data:`PGP`.
    :type pgp_dir: Optional[Path]
    :return: (rows, summary) — rows are tiers 1-6 sorted by rank. The summary counts, besides the tiers,
        ``fjp_only_listed`` (listed rows whose only images are image-store photographs) and, over all
        Arabic-primary documents, the merged-record joins of :func:`join_merged`: ``shelfmark_fallback_used``,
        ``multilink_resolved_by_shelfmark`` and ``shelfmark_fallback_ambiguous``.
    :rtype: Tuple[List[Dict[str, Any]], Dict[str, Any]]
    """
    pgp_dir = pgp_dir if pgp_dir is not None else PGP
    with open(pgp_dir / "documents.csv", encoding="utf-8-sig") as fh:
        docs = [r for r in csv.DictReader(fh) if arabic_primary(r)]
    editions = load_pgp_editions({d["pgpid"] for d in docs}, pgp_dir)
    linked, unlinked = load_merged_index()
    keys = json.load(open(CLEAN_V2_KEYS))
    train_keys = set(keys if isinstance(keys, list) else keys.keys())
    ids = json.load(open(CLEAN_V2_IDS)) if CLEAN_V2_IDS.exists() else []
    train_ids = set(map(str, ids if isinstance(ids, list) else ids.keys()))
    local = local_image_index()
    with open(pgp_dir / "fragments.csv", encoding="utf-8-sig") as fh:
        frags = {r["shelfmark"]: r for r in csv.DictReader(fh)}
    known = known_fragments(pgp_dir)

    rows: List[Dict[str, Any]] = []
    joins: collections.Counter = collections.Counter()
    for d in docs:
        shelf = d["shelfmark"]
        frag = frags.get(shelf.split(" + ")[0], {})
        m, merged_join, event = join_merged(d["pgpid"], shelf, linked, unlinked)
        joins[event] += 1
        iiif = d["iiif_urls"].strip() or frag.get("iiif_url", "").strip()
        pgp_text = editions.get(d["pgpid"], "")
        pgp_letters = len(_ARABIC.findall(pgp_text))
        has_pgp_flag = d["has_transcription"].strip().upper() == "Y"
        if pgp_letters >= MIN_TRANSCRIPTION_LETTERS:
            gt_source = "pgp_export"
        elif has_pgp_flag:
            gt_source = "pgp_site"
        elif m.get("fjp_arabic", 0) >= MIN_TRANSCRIPTION_LETTERS:
            gt_source = "fjp_only"
        else:
            gt_source = "none"
        rows.append({
            "pgpid": d["pgpid"], "pgp_url": d["url"], "shelfmark": shelf,
            "single_fragment": " + " not in shelf and d["multifragment"].strip() == "",
            "canonical_id": m.get("canonical_id", ""),
            "library": frag.get("library", ""), "collection": frag.get("collection", ""),
            "doc_type": d["type"], "languages_primary": d["languages_primary"],
            "side": d["side"], "multifragment": d["multifragment"],
            "doc_date": d["doc_date_standard"], "gt_source": gt_source,
            "pgp_transcription_letters": pgp_letters,
            "pgp_hebrew_letters": len(_HEBREW.findall(pgp_text)),
            "fjp_transcription_letters": m.get("fjp_arabic", 0),
            "in_training_clean_v2": loose_key(shelf) in train_keys
                                    or (m.get("canonical_id", "") in train_ids),
            "local_image": (local.get(m.get("canonical_id", "")) or
                            local.get((cudl_manifest(shelf, known) or "").rsplit("/", 1)[-1], "")),
            "image_source": image_source(iiif, m), "iiif_urls": iiif,
            "suggested_iiif": "" if iiif else (cudl_manifest(shelf, known) or ""),
            "cudl_verified": "", "fjp_images": ";".join(m.get("fjp_images", [])), "merged_join": merged_join,
        })

    # Constructed CUDL manifests: verify (optional) and promote the ones that resolve (a library image replaces
    # image-store photographs as the image source).
    candidates = [r for r in rows if r["suggested_iiif"]
                  and r["image_source"] not in LIBRARY_IMAGE_SOURCES
                  and (r["pgp_transcription_letters"] >= MIN_TRANSCRIPTION_LETTERS
                       or r["gt_source"] == "pgp_site")]
    if verify_cudl and candidates:
        print(f"verifying {len(candidates)} constructed CUDL manifests ...", flush=True)
        ok = verify_urls([r["suggested_iiif"] for r in candidates])
        for r in candidates:
            r["cudl_verified"] = ok[r["suggested_iiif"]]
            if ok[r["suggested_iiif"]]:
                r["iiif_urls"] = r["suggested_iiif"]
                r["image_source"] = "iiif_cudl_constructed"

    listed: List[Dict[str, Any]] = []
    skipped = collections.Counter()
    for r in rows:
        tier = tier_for(r)
        if tier == 0:
            skipped[r["gt_source"]] += 1
            continue
        r.update({"tier": tier, "tier_reason": TIERS[tier][0], "action": TIERS[tier][1]})
        listed.append(r)

    def sort_key(r: Dict[str, Any]) -> Tuple:
        return (r["tier"], -r["pgp_transcription_letters"], -r["fjp_transcription_letters"],
                _TYPE_PRIORITY.get(r["doc_type"], 9),
                0 if r["single_fragment"] else 1,
                0 if r["doc_date"] else 1, int(r["pgpid"]))
    listed.sort(key=sort_key)
    for i, r in enumerate(listed, 1):
        r["rank"] = i

    def count(tier: int, key: str, top: int = 8) -> Dict[str, int]:
        return dict(collections.Counter(r[key] or "(blank)" for r in listed
                                        if r["tier"] == tier).most_common(top))
    t1 = [r for r in listed if r["tier"] == 1]
    summary = {
        "pgp_dir": str(pgp_dir), "pgp_export_commit": pgp_export_commit(pgp_dir),
        "arabic_primary_docs": len(docs), "listed": len(listed),
        "by_tier": {t: sum(r["tier"] == t for r in listed) for t in TIERS},
        "skipped_nothing_actionable_by_gt_source": dict(skipped),
        "cudl_constructed": len(candidates),
        "cudl_verified_resolving": sum(1 for r in candidates if r["cudl_verified"] is True),
        "tier1_by_library": count(1, "library"), "tier1_by_type": count(1, "doc_type"),
        "tier1_single_fragment": sum(r["single_fragment"] for r in t1),
        "tier1_letters_median": (sorted(r["pgp_transcription_letters"] for r in t1)[len(t1) // 2]
                                 if t1 else 0),
        "tier6_by_library": count(6, "library"),
        "tier5_by_type": count(5, "doc_type"),
        "fjp_only_listed": sum(1 for r in listed if r["image_source"] == STORE_IMAGE_SOURCE),
        "shelfmark_fallback_used": joins["shelfmark_fallback_used"],
        "multilink_resolved_by_shelfmark": joins["multilink_resolved_by_shelfmark"],
        "shelfmark_fallback_ambiguous": joins["shelfmark_fallback_ambiguous"],
    }
    return listed, summary


def output_paths(out_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """Where the queue, the backlog and the summary are written.

    :param out_dir: Output directory; default: :data:`OUT_CSV`, :data:`OUT_BACKLOG` and
        :data:`OUT_SUMMARY` themselves (the merged directory under ``raw_data``).
    :type out_dir: Optional[Path]
    :return: ``(queue CSV, backlog CSV, summary JSON)``, with the default file names.
    :rtype: Tuple[Path, Path, Path]
    """
    if out_dir is None:
        return OUT_CSV, OUT_BACKLOG, OUT_SUMMARY
    return out_dir / OUT_CSV.name, out_dir / OUT_BACKLOG.name, out_dir / OUT_SUMMARY.name


def _same_path(a: Path, b: Path) -> bool:
    """Whether two paths name the same place, ignoring case (APFS and the NAS share ignore it).

    :param a: First path.
    :type a: Path
    :param b: Second path.
    :type b: Path
    :return: True when the resolved paths are equal after case folding.
    :rtype: bool
    """
    return str(a.resolve()).casefold() == str(b.resolve()).casefold()


def out_dir_problem(pgp_dir: Optional[Path], out_dir: Optional[Path]) -> str:
    """Why a run may not write where it would: a queue from another export must not replace the default one.

    :param pgp_dir: ``--pgp-dir`` (``None``: the default export :data:`PGP`).
    :type pgp_dir: Optional[Path]
    :param out_dir: ``--out-dir`` (``None``: the default files under ``raw_data``).
    :type out_dir: Optional[Path]
    :return: The problem, or ``""`` when the run may write its outputs.
    :rtype: str
    """
    if pgp_dir is None or _same_path(pgp_dir, PGP):
        return ""
    if out_dir is None:
        return (f"--pgp-dir {pgp_dir} is another export than {PGP}: give --out-dir, the default outputs under raw_data "
                f"are the queue the training and benchmark builders read")
    if _same_path(output_paths(out_dir)[0], OUT_CSV):
        return f"--out-dir {out_dir} is the default queue directory under raw_data: a queue from {pgp_dir} must not replace it"
    return ""


def _display_path(path: Path) -> Path:
    """A path relative to the repo when it is inside it, else the path itself.

    :param path: Output file.
    :type path: Path
    :return: Path for the closing message.
    :rtype: Path
    """
    return path.relative_to(REPO) if path.is_relative_to(REPO) else path


def main(argv: Optional[List[str]] = None) -> None:
    """Write the queue CSV and a JSON summary, printing the summary.

    :param argv: Command-line arguments (default: ``sys.argv[1:]``).
    :type argv: Optional[List[str]]
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify-cudl", action="store_true",
                    help="fetch constructed CUDL manifests; promote only the ones that resolve")
    ap.add_argument("--pgp-dir", type=Path, default=None,
                    help=f"PGP export directory with documents.csv, footnotes.csv, fragments.csv (default: {PGP})")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"directory for the queue, backlog and summary files (default: {OUT_CSV.parent})")
    args = ap.parse_args(argv)
    problem = out_dir_problem(args.pgp_dir, args.out_dir)
    if problem:
        ap.error(problem)
    out_csv, out_backlog, out_summary = output_paths(args.out_dir)
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = build(verify_cudl=args.verify_cudl, pgp_dir=args.pgp_dir)
    order = {1: 0, 2: 1, 3: 2, 6: 3}
    main_rows = sorted((r for r in rows if r["tier"] in order), key=lambda r: (order[r["tier"]], r["rank"]))
    for i, r in enumerate(main_rows, 1):
        r["rank"] = i
    backlog = [r for r in rows if r["tier"] not in order]
    for path, subset in ((out_csv, main_rows), (out_backlog, backlog)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in COLUMNS} for r in subset)
    summary["main_queue_rows"] = len(main_rows)
    summary["backlog_rows"] = len(backlog)
    summary["main_queue_with_local_image"] = sum(1 for r in main_rows if r["local_image"])
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"wrote {_display_path(out_csv)} ({len(main_rows)} transcription-bearing rows) "
          f"+ {out_backlog.name} ({len(backlog)} image-only/store-text-only rows)")


if __name__ == "__main__":
    main()
