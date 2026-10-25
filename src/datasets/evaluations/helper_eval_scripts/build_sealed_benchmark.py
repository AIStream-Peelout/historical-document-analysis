# File name: build_sealed_benchmark.py
# Date: 10/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Build ``sealed_v1``: a confirmation benchmark that no model was trained on and no decision was tuned on.

PGP-131 (documentary) and religious-140 (literary) have both been used to pick checkpoints, blends,
thresholds and voters, so a number measured on them is no longer an unbiased estimate. ``sealed_v1``
holds pages in the SAME formats, so the existing readers and scorers work on it through their
``--spec`` / ``--images-dir`` / ``--outputs-dir`` flags. A page is admitted only when

* no training source of the VLM fine-tunes (genizah_clean v1/v2, the PGP edition, documentary
  grounding, QA, Arabic-edition and VQA-parse sets, the KTIV v1-v4 page lists, and the v2.2 /
  v2.3 mixtures built from them; every split) and no training list of the production Kraken
  models (KTIV recognition lines, KTIV segmentation pages) contains its document or manuscript;
* no benchmark contains its document or manuscript, and it is not a join of one (shared PGP id,
  shared canonical id, sub-part of a shelfmark, the same fragment under another name, a merged
  record shared with a KTIV manuscript);
* no ground-truth-scored analysis (checkpoint comparisons, re-read experiments, probes,
  evaluation output folders) has used it.

:func:`build_inventory` reads every source and records its role and what it contributed; the
build report lists them, with the sources that could not be read. A required source that is
missing stops the build: a held-out set must not be drawn without knowing what is excluded.

Documentary pages: PGP documents with a digital edition (``footnotes.csv``, relation "Digital
Edition", >= 50 characters), mapped to one merged record (joins, multi-fragment records and
fragments shared with another document are left out, as in the edition builder). The edition is
parsed and cleaned by the edition builder (:func:`build_pgp_editions.parse_edition`: labels,
sides, notation, ``[...]`` gaps). Which image shows which side is decided from the reads the
consensus pipeline already cached (``ai_reads/raw``; no model is run): a side is on the image when
the share of its letter 4-grams found in the two readers' lines exceeds the largest share on the
edition builder's 50 frozen null pages (with >= 3 hits) and its present blocks hold
>= :data:`SIDE_MIN_COVERAGE` of the side's letters; a page is kept only when exactly one side is on
it. This is the edition builder's presence test without its line anchor and with a lower coverage
bar: every page that passed the strict version has already been used for training. The page's
ground truth is the whole side (main text first, then margins), and it must pass the builder's
notation gate, carry >= :data:`DOC_MIN_LETTERS` Hebrew letters on >= 3 lines, be mostly Hebrew
script (the scorer counts Hebrew letters only), and not repeat itself. One page per document.

Literary pages: KTIV transcriptions in API shape (word boxes), ground truth and column count from
:func:`ktiv_layout.reconstruct_page` exactly as ``build_religious_benchmark.py`` derives them,
letter floors 120 (Talmud/halakha) / 40 (other genres), no GT mash, an image in the local or NAS
zip store (or the bucket), at most two pages per manuscript, at least
:data:`LIT_MIN_MULTI` two-column pages.

Sampling is stratified (documentary: primary language x PGP type; literary: column class x genre),
proportional to the eligible pool, seeded with :data:`SEED`. Nothing is tuned on the result: no
threshold above was chosen from model scores, and the README says the set is read once per
confirmed candidate.

Outputs (``evaluations/sealed_v1``)::

    documentary/sealed_v1_documentary.json            verified-JSON format of PGP-131 (+ pgpid, language, type, source)
    documentary/sealed_v1_documentary_script_tags.json  per-script sidecar for score_genizah_offline.py
    documentary/images/<doc_id>.jpg
    literary/sealed_v1_literary.json                  religious-140 format (absolute image paths)
    literary/images/<sys_num>_<FL>.jpg
    sealed_v1.json                                    held-out registry (benchmark_registry.REGISTRY_LOCATIONS)
    build_report.json                                 sources, pools, reasons, strata, selection

Usage (repo root)::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.build_sealed_benchmark [--dry-run] [--report PATH]
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from PIL import Image as PILImage
from PIL import ImageOps

from src.datasets.cleaning.clean_genizah_transcriptions import clean_diplomatic
from src.datasets.evaluations import benchmark_registry as reg
from src.datasets.evaluations.helper_eval_scripts.audit_genizah_benchmark import find_self_duplication, letters_only
from src.datasets.evaluations.helper_eval_scripts.build_religious_benchmark import _ktiv_meta_index
from src.datasets.evaluations.helper_eval_scripts.build_script_tags import bucket_for
from src.datasets.evaluations.helper_eval_scripts.decontam_gate import DecontamGate
from src.datasets.evaluations.metrics import dedup_repeated_tail
from src.datasets.tools.ktiv_zip_mover import resolve_zips
from src.finetuning.qwen_hebrew import build_pgp_editions as bpe
from src.finetuning.qwen_hebrew.build_ktiv_dataset import benchmark_shingles, shingle_hits
from src.finetuning.qwen_hebrew.ktiv_bundles import bundle_shape, select_bundles
from src.finetuning.qwen_hebrew.ktiv_layout import hebrew_letters, reconstruct_page

PILImage.MAX_IMAGE_PIXELS = None
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[4]
CG = REPO / "src/datasets/raw_data/cairo_genizah"
NAS = Path("/Volumes/home/studio_offload/datasets")
HF_KTIV_V2 = Path("/Volumes/home/studio_offload/hf_home_merge/hub/datasets--isaacmg--genizah_ktiv_v2/snapshots")
NAME = "sealed_v1"
SEED = 20261012
OUT_DIR = CG / "evaluations" / NAME
DOC_SPEC_NAME = f"{NAME}_documentary.json"
DOC_TAGS_NAME = f"{NAME}_documentary_script_tags.json"
LIT_SPEC_NAME = f"{NAME}_literary.json"

PGP131_DIR = CG / "evaluations/genizah_test_v1"
RELIGIOUS_SPEC = CG / "evaluations/genizah_religious_v1/genizah_religious_v1.json"
FOOTNOTES_CSV = CG / "pgp_raw/data/footnotes.csv"
RAW_DIR = CG / "ai_reads/raw"
NULL_POOL = NAS / "pgp_editions_v1_inputs/null_pool.json"
KTIV_DIR = CG / "ktiv"

DOC_TARGET = 300
LIT_TARGET = 150
LIT_MIN_MULTI = 50
LIT_MAX_PAGES_PER_MS = 2
LIT_MIN_LETTERS_TALMUD = 120    # build_religious_benchmark.MIN_LETTERS_TALMUD
LIT_MIN_LETTERS_OTHER = 40      # build_religious_benchmark.MIN_LETTERS_OTHER
MASH_RUN = 35                   # refresh_gt_v1_1.MASH_RUN: a longer space-free letter run is GT mash
DOC_MIN_EDITION_CHARS = 50
DOC_MIN_LETTERS = bpe.MIN_ANSWER_LETTERS
DOC_MIN_LINES = bpe.MIN_ANSWER_LINES
SIDE_MIN_COVERAGE = 0.7
HEBREW_MIN_SHARE = 0.5
SELF_DUPLICATION_MAX = 0.55     # score_genizah_offline.load_benchmark repairs a GT above this
EVIDENCE_MODELS = ("qwen3-vl-8b-heb-v21b-step1200", "qwen3-vl-8b-heb-v22i-a50")
GCS_HOST = "storage.googleapis.com"
KTIV_TARGET_WIDTH = 2800        # build_religious_benchmark._TARGET_W
DOWNLOAD_RETRIES = 3

_SYS_RE = re.compile(r"(?<!\d)(99\d{16})(?!\d)")
_STEM_PGPID_RE = re.compile(r"^(?:vqa_pgpqa|pgpqa|pgpar|vqa|pgp)_(\d+)_\d+_")
_STEM_DG_RE = re.compile(r"^dg_(.+?)__\d+_")
_FL_RE = re.compile(r"(FL\d+)")
_HEB_RUN_RE = re.compile(r"[א-ת]+")

LANGUAGE_OF_BUCKET = {"judaeo_arabic": "Judaeo-Arabic", "arabic_script": "Arabic", "hebrew": "Hebrew"}
TYPE_OF_PGP = {"Letter": "letter", "Legal document": "legal", "List or table": "list"}
# KTIV catalogue subjects (basic_catalog.title, ';'-separated) -> genre, first matching subject wins
GENRE_KEYWORDS = (
    ("talmud/halakha", ("תלמוד", "משנה", "תוספתא", "הלכה", "הלכות", "הלכתית", "רי\"ף", "המנהיג", "שאלות ותשובות",
                        "שו\"ת", "פסקים", "Talmud", "Mishna", "Halakh")),
    ("midrash/aggadah", ("מדרש", "אגדה", "Midrash")),
    ("liturgy/piyyut", ("פיוט", "תפילה", "תפלה", "סידור", "מחזור", "סליחות", "קינות", "קינה", "Piyyut", "Liturg",
                        "Prayer")),
    ("bible/masorah", ("תנ\"ך", "מקרא", "מסורה", "תרגום", "תורה", "נביאים", "כתובים", "תהלים", "Bible", "Masor",
                       "Targum")),
)


# ----------------------------------------------------------------------------- inventory


@dataclass
class Inventory:
    """Keys a sealed page must avoid, each mapped to the first source that put it there.

    :param ids: Canonical (merged) ids of documents.
    :param pgpids: PGP document ids.
    :param urls: Image URLs.
    :param sys_nums: KTIV manuscripts (NLI system numbers).
    :param keys: Institution-free fragment keys (:func:`build_pgp_editions.fragment_key`).
    :param sources: One audit row per source: name, role, path, status, counts, note.
    """

    ids: Dict[str, str] = field(default_factory=dict)
    pgpids: Dict[str, str] = field(default_factory=dict)
    urls: Dict[str, str] = field(default_factory=dict)
    sys_nums: Dict[str, str] = field(default_factory=dict)
    keys: Dict[str, str] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, source: str, role: str, path: Any, ids: Iterable[str] = (), pgpids: Iterable[str] = (),
            urls: Iterable[str] = (), sys_nums: Iterable[str] = (), note: str = "") -> None:
        """Record one source and the keys it holds (an earlier source keeps its claim on a key).

        :param source: Source name (``<role>:<dataset>``).
        :type source: str
        :param role: ``training``, ``kraken_training``, ``benchmark`` or ``decision``.
        :type role: str
        :param path: File or directory read.
        :type path: Any
        :param ids: Canonical ids.
        :type ids: Iterable[str]
        :param pgpids: PGP document ids.
        :type pgpids: Iterable[str]
        :param urls: Image URLs.
        :type urls: Iterable[str]
        :param sys_nums: KTIV system numbers.
        :type sys_nums: Iterable[str]
        :param note: What the source is.
        :type note: str
        """
        row: Dict[str, Any] = {"source": source, "role": role, "path": str(path), "status": "read", "note": note}
        for name, table, values in (("ids", self.ids, ids), ("pgpids", self.pgpids, pgpids),
                                    ("urls", self.urls, urls), ("sys_nums", self.sys_nums, sys_nums)):
            vals = {str(v).strip() for v in values if v and str(v).strip()}
            for v in vals:
                table.setdefault(v, source)
            row[name] = len(vals)
        self.sources.append(row)

    def missing(self, source: str, role: str, path: Any, note: str = "") -> None:
        """Record a source that could not be read.

        :param source: Source name.
        :type source: str
        :param role: Source role.
        :type role: str
        :param path: Expected location.
        :type path: Any
        :param note: Why it is missing / what it would have held.
        :type note: str
        """
        self.sources.append({"source": source, "role": role, "path": str(path), "status": "missing", "note": note})


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Objects of a JSONL file (blank lines skipped).

    :param path: JSONL file.
    :type path: Path
    :return: Iterator of objects.
    :rtype: Iterator[Dict[str, Any]]
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def record_keys(records: Iterable[Dict[str, Any]]) -> Tuple[Set[str], Set[str], Set[str]]:
    """Document keys of manifest / job / analysis records.

    :param records: Objects with any of ``canonical_id`` / ``doc_id`` / ``cid``, ``pgpid`` / ``pid``,
        ``image_url``.
    :type records: Iterable[Dict[str, Any]]
    :return: ``(canonical ids, pgpids, image urls)``.
    :rtype: Tuple[Set[str], Set[str], Set[str]]
    """
    ids: Set[str] = set()
    pgpids: Set[str] = set()
    urls: Set[str] = set()
    for r in records:
        for k in ("canonical_id", "doc_id", "cid"):
            if r.get(k):
                ids.add(str(r[k]))
        for k in ("pgpid", "pid"):
            if r.get(k):
                pgpids.add(str(r[k]))
        if r.get("image_url"):
            urls.add(str(r["image_url"]))
    return ids, pgpids, urls


def parse_stem(stem: str) -> Tuple[str, str]:
    """What a training-row stem identifies.

    :param stem: Row stem (``pgp_446_0_page``, ``pgpqa_499_0_qa_date_0``, ``vqa_pgpqa_41322_1_...``,
        ``dg_<canonical id>__0_loc0``, ``ktiv_<sys_num>_<FL>...``).
    :type stem: str
    :return: ``("pgpid", id)``, ``("canonical_id", id)``, ``("sys_num", id)`` or ``("", "")`` for rows of
        sources without Geniza document ids (Talmud print, synthetic, outside Arabic sets).
    :rtype: Tuple[str, str]
    """
    if stem.startswith(("ktiv_", "KTIV_")):
        m = _SYS_RE.search(stem)
        return ("sys_num", m.group(1)) if m else ("", "")
    m = _STEM_PGPID_RE.match(stem)
    if m:
        return "pgpid", m.group(1)
    m = _STEM_DG_RE.match(stem)
    if m:
        return "canonical_id", m.group(1)
    return "", ""


def stem_keys(stems: Iterable[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """Document keys of a list of row stems.

    :param stems: Row stems.
    :type stems: Iterable[str]
    :return: ``(canonical ids, pgpids, sys_nums)``.
    :rtype: Tuple[Set[str], Set[str], Set[str]]
    """
    out: Dict[str, Set[str]] = {"canonical_id": set(), "pgpid": set(), "sys_num": set()}
    for s in stems:
        kind, key = parse_stem(s)
        if kind:
            out[kind].add(key)
    return out["canonical_id"], out["pgpid"], out["sys_num"]


def parquet_stems(paths: Iterable[Path]) -> List[str]:
    """The ``stem`` column of row parquet files (no other column is read).

    :param paths: Parquet files.
    :type paths: Iterable[Path]
    :return: All stems.
    :rtype: List[str]
    """
    stems: List[str] = []
    for p in paths:
        stems.extend(pq.read_table(p, columns=["stem"]).column("stem").to_pylist())
    return stems


def arrow_stems(paths: Iterable[Path]) -> List[str]:
    """The ``stem`` column of ``save_to_disk`` arrow shards, read through a memory map (image bytes stay unread).

    :param paths: Arrow stream files.
    :type paths: Iterable[Path]
    :return: All stems.
    :rtype: List[str]
    """
    stems: List[str] = []
    for p in paths:
        with pa.memory_map(str(p), "r") as src:
            reader = pa.ipc.open_stream(src)
            idx = reader.schema.get_field_index("stem")
            for batch in reader:
                stems.extend(batch.column(idx).to_pylist())
    return stems


def dataset_stems(path: Path) -> List[str]:
    """Stems of every split of a local ``save_to_disk`` dataset.

    :param path: Dataset directory.
    :type path: Path
    :return: Stems.
    :rtype: List[str]
    """
    from datasets import load_from_disk
    dd = load_from_disk(str(path))
    return [s for split in dd for s in dd[split].select_columns(["stem"])["stem"]]


def sys_nums_in(names: Iterable[str]) -> Set[str]:
    """KTIV system numbers found in names (file names, stems, URLs).

    :param names: Strings.
    :type names: Iterable[str]
    :return: System numbers.
    :rtype: Set[str]
    """
    return {m.group(1) for n in names for m in [_SYS_RE.search(str(n))] if m}


def _require(path: Path, what: str) -> Path:
    """Stop when a required source is missing.

    :param path: Expected file or directory.
    :type path: Path
    :param what: What it is.
    :type what: str
    :return: The path.
    :rtype: Path
    :raises FileNotFoundError: When it does not exist (e.g. the NAS is not mounted).
    """
    if not path.exists():
        raise FileNotFoundError(f"required exclusion source missing: {what} ({path}); is the NAS mounted?")
    return path


def add_documentary_training(inv: Inventory, nas: Path = NAS, repo: Path = REPO) -> None:
    """Documentary training sources of the VLM fine-tunes (every split).

    :param inv: Inventory to fill.
    :type inv: Inventory
    :param nas: NAS datasets root.
    :type nas: Path
    :param repo: Repository root.
    :type repo: Path
    """
    for name, path in (("genizah_clean_v1", repo / "src/datasets/processed/qwen_hebrew_genizah"),
                       ("genizah_clean_v2", repo / "src/datasets/processed/qwen_hebrew_genizah_v2")):
        if path.exists():
            inv.add(f"training:{name}", "training", path, ids=dataset_stems(path),
                    note="v1.7 / v1.8 documentary set; stems are canonical ids")
        else:
            inv.missing(f"training:{name}", "training", path, "local dataset copy absent")
    ids_json = _require(repo / "src/datasets/raw_data/cairo_genizah/decontam/clean_v2_ids.json", "clean_v2 ids")
    inv.add("training:genizah_clean_v2_ids", "training", ids_json, ids=json.loads(ids_json.read_text()),
            note="decontam inventory of genizah_clean_v2 (all splits)")
    for ds, note in (("pgp_editions_v1", "v2.2 edition pages (train + val)"),
                     ("documentary_grounding_v1", "v2.2 documentary grounding rows (train + val)"),
                     ("pgp_qa_v1", "v2.2a QA rows"), ("pgp_qa_v2", "v2.2b QA rows"),
                     ("pgp_arabic_editions_v1", "v2.3a Arabic edition pages"),
                     ("pgp_vqa_parse_v1", "v2.3q VQA parse rows (built; trained or not, held out)")):
        path = _require(nas / ds / "manifest.jsonl", f"{ds} manifest")
        ids, pgpids, urls = record_keys(iter_jsonl(path))
        inv.add(f"training:{ds}", "training", path, ids=ids, pgpids=pgpids, urls=urls, note=note)
    for ds in ("pgp_editions_v1_images_once", "pgp_edition_pages_v1_images_once", "documentary_grounding_v1_images_once",
               "pgp_qa_v1_images_once", "pgp_qa_v2_images_once", "pgp_arabic_editions_v1_images_once",
               "pgp_vqa_parse_v1_images_once"):
        rows = _require(nas / ds / "rows", f"{ds} rows")
        ids, pgpids, sys_nums = stem_keys(parquet_stems(sorted(rows.glob("*.parquet"))))
        inv.add(f"training:{ds}", "training", rows, ids=ids, pgpids=pgpids, sys_nums=sys_nums,
                note="images-once export the mixtures draw from")
    for mix, note in (("genizah_v22_pilot", "v2.2a mixture"), ("genizah_v22_full", "v2.2b mixture (rev2)"),
                      ("genizah_v23a_arabic", "v2.3a mixture (run + control rows)"),
                      ("genizah_v23q_vqa", "v2.3q mixture (built)")):
        rows = _require(nas / mix / "rows", f"{mix} rows")
        files = sorted(rows.glob("*.parquet"))
        if mix == "genizah_v22_full" and (nas / "genizah_v22_full_rows_train_rev1.parquet").exists():
            files.append(nas / "genizah_v22_full_rows_train_rev1.parquet")
        ids, pgpids, sys_nums = stem_keys(parquet_stems(files))
        inv.add(f"training:{mix}", "training", rows, ids=ids, pgpids=pgpids, sys_nums=sys_nums, note=note)


def add_ktiv_training(inv: Inventory, nas: Path = NAS, repo: Path = REPO) -> None:
    """KTIV page lists of the VLM fine-tunes and the training lists of the production Kraken models.

    :param inv: Inventory to fill.
    :type inv: Inventory
    :param nas: NAS datasets root.
    :type nas: Path
    :param repo: Repository root.
    :type repo: Path
    """
    v19a = _require(repo / "src/datasets/raw_data/cairo_genizah/decontam/ktiv_v19a_train_sysnums.json", "v1.9a list")
    inv.add("training:genizah_ktiv_v1_train", "training", v19a, sys_nums=json.loads(v19a.read_text()),
            note="v1.9a KTIV train manuscripts (verified against dataset stems)")
    for version, sub in ((1, "ktiv_dataset_v1/images"), (2, "ktiv_dataset_v2_images"), (3, "ktiv_dataset_v3_images"),
                         (4, "ktiv_dataset_v4_images")):
        path = _require(nas / sub, f"KTIV v{version} page images")
        inv.add(f"training:genizah_ktiv_v{version}_pages", "training", path, sys_nums=sys_nums_in(os.listdir(path)),
                note=f"per-manuscript page-image folders the KTIV v{version} build wrote (train + val)")
    for version in (1, 2, 3):
        val = _require(nas / f"genizah_ktiv_v{version}" / "val", f"KTIV v{version} val split")
        _, _, sys_nums = stem_keys(arrow_stems(sorted(val.glob("data-*.arrow"))))
        inv.add(f"training:genizah_ktiv_v{version}_val", "training", val, sys_nums=sys_nums,
                note=f"KTIV v{version} validation manuscripts (eval loss)")
    known = _require(nas / "genizah_ktiv_v4/val_manuscripts.json", "KTIV v4 split manifest")
    inv.add("training:genizah_ktiv_v4_manuscripts", "training", known,
            sys_nums=json.loads(known.read_text())["known_manuscripts"], note="every v4 manuscript (train + val)")
    rows = _require(nas / "genizah_ktiv_v4_images_once/rows", "KTIV v4 rows")
    _, _, sys_nums = stem_keys(parquet_stems(sorted(rows.glob("*.parquet"))))
    inv.add("training:genizah_ktiv_v4_images_once", "training", rows, sys_nums=sys_nums,
            note="rows the v2.2/v2.3 mixtures sample KTIV from")
    hf_files = sorted(HF_KTIV_V2.glob("*/data/*.parquet")) if HF_KTIV_V2.exists() else []
    if hf_files:
        _, _, sys_nums = stem_keys(parquet_stems(hf_files))
        inv.add("training:hf_cache_genizah_ktiv_v2", "training", HF_KTIV_V2, sys_nums=sys_nums,
                note="local Hub cache of isaacmg/genizah_ktiv_v2 (train split)")
    else:
        inv.missing("training:hf_cache_genizah_ktiv_v2", "training", HF_KTIV_V2, "no local Hub cache")
    lines = nas / "kraken_ktiv_lines"
    names: List[str] = []
    for f in ("train.txt", "val.txt", "train.all.txt", "val.all.txt"):
        if (lines / f).exists():
            names.extend((lines / f).read_text(encoding="utf-8").splitlines())
    if names:
        inv.add("kraken_training:kraken_ktiv_lines", "kraken_training", lines, sys_nums=sys_nums_in(names),
                note="KTIV recognition fine-tune (ktivseg e10) line lists")
    else:
        inv.missing("kraken_training:kraken_ktiv_lines", "kraken_training", lines, "line lists absent")
    pagexml = nas / "kraken_segmenter/ktiv_pagexml_v1/pagexml"
    if pagexml.exists():
        inv.add("kraken_training:kraken_segmenter_pagexml", "kraken_training", pagexml,
                sys_nums=sys_nums_in(os.listdir(pagexml)), note="KTIV segmentation training pages")
    else:
        inv.missing("kraken_training:kraken_segmenter_pagexml", "kraken_training", pagexml, "absent")


def add_benchmarks(inv: Inventory, repo: Path = REPO, exclude_registry: str = NAME) -> None:
    """Every benchmark and every registered held-out set.

    :param inv: Inventory to fill.
    :type inv: Inventory
    :param repo: Repository root.
    :type repo: Path
    :param exclude_registry: The registry this build writes (never read as an exclusion).
    :type exclude_registry: str
    """
    cg = repo / "src/datasets/raw_data/cairo_genizah"
    bench = cg / "evaluations/genizah_test_v1"
    for f, note in (("genizah_test_v1_verified.json", "PGP-131 (+ replaced documents)"),
                    ("genizah_test_v1.json", "frozen 150 the 131 were verified from")):
        spec = json.loads(_require(bench / f, f).read_text(encoding="utf-8"))
        ids, pgpids, urls = record_keys(spec["docs"])
        ids |= {x for r in spec.get("replacements", []) for x in (r.get("removed"), r.get("added")) if x}
        inv.add(f"benchmark:{f}", "benchmark", bench / f, ids=ids, pgpids=pgpids, urls=urls, note=note)
    catalog = json.loads(_require(bench / "genizah_test_v1_catalog.json", "PGP catalog").read_text(encoding="utf-8"))
    inv.add("benchmark:genizah_test_v1_catalog.json", "benchmark", bench / "genizah_test_v1_catalog.json",
            ids=catalog.keys(), urls=[f"https://{GCS_HOST}/cairo-genizah-es-json/images/{img}"
                                      for v in catalog.values() for img in (v.get("images") or [])],
            note="catalog of the frozen benchmark")
    ids_json = _require(cg / "decontam/benchmark_ids.json", "benchmark ids")
    inv.add("benchmark:benchmark_ids", "benchmark", ids_json, ids=json.loads(ids_json.read_text()))
    rel = json.loads(_require(cg / "evaluations/genizah_religious_v1/genizah_religious_v1.json", "religious-140")
                     .read_text(encoding="utf-8"))["docs"]
    inv.add("benchmark:religious_140", "benchmark", RELIGIOUS_SPEC, ids=[d["doc_id"] for d in rel],
            sys_nums=[d["sys_num"] for d in rel], note="whole manuscripts (sys_num)")
    ground = repo / "src/datasets/evaluations/grounding_eval/grounding_eval.jsonl"
    if ground.exists():
        inv.add("benchmark:grounding_eval", "benchmark", ground,
                sys_nums=sys_nums_in(r["doc_id"] for r in iter_jsonl(ground)), note="religious-140 pages")
    for name in reg.BENCHMARK_REGISTRIES:
        if name == exclude_registry:
            continue
        ids, pgpids = reg.registered_benchmark_documents([name])
        inv.add(f"benchmark:registry:{name}", "benchmark", reg.registry_path(name), ids=ids, pgpids=pgpids,
                sys_nums=reg.registered_benchmark_manuscripts([name]))


def add_decision_sets(inv: Inventory, repo: Path = REPO, nas: Path = NAS) -> None:
    """Pages scored against ground truth in analyses that informed decisions, plus probe and output folders.

    :param inv: Inventory to fill.
    :type inv: Inventory
    :param repo: Repository root.
    :type repo: Path
    :param nas: NAS datasets root.
    :type nas: Path
    """
    logs = repo / "logs"
    unseen = logs / "v22b_diagnosis/unseen_editions_compare.json"
    if unseen.exists():
        ids, pgpids, _ = record_keys(json.loads(unseen.read_text(encoding="utf-8")))
        inv.add("decision:unseen_editions_compare", "decision", unseen, ids=ids, pgpids=pgpids,
                note="v2.2b diagnosis: checkpoints scored against editions of untrained documents")
    else:
        inv.missing("decision:unseen_editions_compare", "decision", unseen)
    double = logs / "v22b_diagnosis/double_editions.json"
    if double.exists():
        inv.add("decision:double_editions", "decision", double,
                pgpids=[str(r["pid"]) for r in json.loads(double.read_text(encoding="utf-8"))],
                note="edition-vs-edition floor analysis")
    else:
        inv.missing("decision:double_editions", "decision", double)
    for name, path, note in (
            ("offbench_sample_jobs", logs / "v22b_diagnosis/offbench/jobs.jsonl", "off-benchmark sample reads (v2.2b diagnosis)"),
            ("consensus_pilot", repo / "src/datasets/raw_data/cairo_genizah/consensus_pilot/pilot_inventory.jsonl",
             "v1.9 consensus-gate pilot (no GT; held out conservatively)"),
            ("vectorless_pilot200", logs / "next_round/consensus_pool/jobs_vectorless_pilot200.jsonl",
             "vector-less description pilot (no GT; held out conservatively)")):
        if path.exists():
            ids, pgpids, urls = record_keys(iter_jsonl(path))
            inv.add(f"decision:{name}", "decision", path, ids=ids, pgpids=pgpids, urls=urls, note=note)
        else:
            inv.missing(f"decision:{name}", "decision", path)
    failed = sorted((logs / "next_round/failed_reads").glob("*.jsonl"))
    if failed:
        recs = [r for f in failed for r in iter_jsonl(f)]
        ids, pgpids, urls = record_keys(recs)
        inv.add("decision:failed_reads", "decision", logs / "next_round/failed_reads", ids=ids, pgpids=pgpids,
                urls=urls, note="re-read experiments scored against held-out GT")
    samples = sorted((repo / "src/datasets/raw_data/cairo_genizah/ai_reads").glob("jobs_*sample10*.jsonl"))
    if samples:
        ids, pgpids, urls = record_keys(r for f in samples for r in iter_jsonl(f))
        inv.add("decision:checkpoint_sample10", "decision", samples[0].parent, ids=ids, urls=urls,
                note="10-image checkpoint comparison sets")
    probe = nas / "kraken_segmenter/agreement_probe_images"
    if probe.exists():
        inv.add("decision:kraken_agreement_probe", "decision", probe,
                ids=[n.split("__")[0] for n in os.listdir(probe) if "__" in n], note="Kraken cut-over probe")
    else:
        inv.missing("decision:kraken_agreement_probe", "decision", probe)
    for sub in ("transcription_raw_outputs", "transcription_raw_outputs_archive_20260726",
                "transcription_raw_outputs_probe_columns", "genizah_images"):
        path = repo / "src/datasets/evaluations" / sub
        if path.exists():
            names = [n for n in os.listdir(path) if not n.startswith(".")]
            inv.add(f"decision:{sub}", "decision", path, ids=names, sys_nums=sys_nums_in(names),
                    note="documents with evaluation outputs")


def build_inventory(nas: Path = NAS, repo: Path = REPO, exclude_registry: str = NAME) -> Inventory:
    """Read every exclusion source.

    :param nas: NAS datasets root.
    :type nas: Path
    :param repo: Repository root.
    :type repo: Path
    :param exclude_registry: The registry this build writes.
    :type exclude_registry: str
    :return: The inventory (not yet expanded across id spaces, see :func:`expand_inventory`).
    :rtype: Inventory
    """
    inv = Inventory()
    add_benchmarks(inv, repo, exclude_registry)          # the order decides which source a key is attributed to
    add_ktiv_training(inv, nas, repo)
    add_documentary_training(inv, nas, repo)
    add_decision_sets(inv, repo, nas)
    return inv


# ----------------------------------------------------------------------------- merged index


@dataclass
class MergedIndex:
    """What the sealed build needs from ``merged_shelfmarks.jsonl``.

    :param pgpids_of: ``canonical id -> pgpids``.
    :param canonicals_of: ``pgpid -> canonical ids``.
    :param sys_of: ``canonical id -> KTIV sys_num`` (records with a KTIV block).
    :param cids_of_sys: ``sys_num -> canonical ids``.
    :param records: Full records of the wanted documents and manuscripts.
    """

    pgpids_of: Dict[str, List[str]]
    canonicals_of: Dict[str, List[str]]
    sys_of: Dict[str, str]
    cids_of_sys: Dict[str, List[str]]
    records: Dict[str, Dict[str, Any]]


def scan_merged(path: Path, wanted_pgpids: Set[str], wanted_sys: Set[str]) -> MergedIndex:
    """One pass over the merged index (regex pre-parse; full JSON only for wanted records).

    :param path: ``merged_shelfmarks.jsonl``.
    :type path: Path
    :param wanted_pgpids: PGP ids whose records are parsed fully.
    :type wanted_pgpids: Set[str]
    :param wanted_sys: KTIV sys_nums whose records are parsed fully.
    :type wanted_sys: Set[str]
    :return: The index.
    :rtype: MergedIndex
    """
    rx_id = re.compile(r'^\{"canonical_id": "((?:[^"\\]|\\.)*)"')
    rx_p = re.compile(r'"pgpids": \[([^\]]*)\]')
    rx_sys = re.compile(r'"sys_num": "(\d{18})"')
    pgpids_of: Dict[str, List[str]] = {}
    canon_of: Dict[str, List[str]] = collections.defaultdict(list)
    sys_of: Dict[str, str] = {}
    cids_of_sys: Dict[str, List[str]] = collections.defaultdict(list)
    records: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = rx_id.match(line)
            if not m:
                continue
            cid = json.loads(f'"{m.group(1)}"')
            mp = rx_p.search(line)
            pids = [x.strip().strip('"') for x in mp.group(1).split(",") if x.strip()] if mp else []
            pgpids_of[cid] = pids
            for p in pids:
                canon_of[p].append(cid)
            ms = rx_sys.search(line)
            if ms:
                sys_of[cid] = ms.group(1)
                cids_of_sys[ms.group(1)].append(cid)
            if any(p in wanted_pgpids for p in pids) or (ms and ms.group(1) in wanted_sys):
                rec = json.loads(line)
                records[cid] = {k: rec.get(k) for k in ("canonical_id", "institution", "shelfmark_display", "images")}
    return MergedIndex(pgpids_of, dict(canon_of), sys_of, dict(cids_of_sys), records)


def expand_inventory(inv: Inventory, merged: MergedIndex, pgp_docs: Dict[str, Dict[str, str]],
                     p2f: Dict[str, Set[str]], ktiv_meta: Dict[str, Dict[str, Any]], gate: DecontamGate) -> None:
    """Carry every exclusion across id spaces, in place.

    Excluded manuscripts reach their merged records; PGP ids and canonical ids are then closed over
    each other (a PGP id reaches every canonical id it appears under and back, so the parts of a
    joined document go together; a merged record holds at most one PGP id, so the closure stays
    within join groups); canonical ids with a KTIV block reach their manuscript; fragment keys are
    taken from every excluded canonical id, PGP shelfmark and fragment, KTIV shelfmark and the two
    decontamination inventories (benchmark and genizah_clean_v2 keys).

    :param inv: Inventory (updated in place).
    :type inv: Inventory
    :param merged: Merged index.
    :type merged: MergedIndex
    :param pgp_docs: ``documents.csv`` by pgpid.
    :type pgp_docs: Dict[str, Dict[str, str]]
    :param p2f: ``pgpid -> fragment shelfmarks``.
    :type p2f: Dict[str, Set[str]]
    :param ktiv_meta: KTIV metadata by sys_num.
    :type ktiv_meta: Dict[str, Dict[str, Any]]
    :param gate: Decontamination gate (loose-key normaliser and its inventories).
    :type gate: DecontamGate
    """
    for sys_num, src in list(inv.sys_nums.items()):
        for cid in merged.cids_of_sys.get(sys_num, []):
            inv.ids.setdefault(cid, src)
    grew = True
    while grew:
        n_before = len(inv.ids) + len(inv.pgpids)
        for pid, src in list(inv.pgpids.items()):
            for cid in merged.canonicals_of.get(pid, []):
                inv.ids.setdefault(cid, src)
        for cid, src in list(inv.ids.items()):
            for pid in merged.pgpids_of.get(cid, []):
                inv.pgpids.setdefault(pid, src)
        grew = len(inv.ids) + len(inv.pgpids) > n_before
    for cid, src in list(inv.ids.items()):
        if cid in merged.sys_of:
            inv.sys_nums.setdefault(merged.sys_of[cid], src)
    names: List[Tuple[str, str]] = [(cid, src) for cid, src in inv.ids.items()]
    for pid, src in inv.pgpids.items():
        shelf = (pgp_docs.get(pid) or {}).get("shelfmark") or ""
        names += [(x.strip(), src) for x in shelf.split("+") if x.strip()]
        names += [(f, src) for f in p2f.get(pid, set())]
    for sys_num, src in inv.sys_nums.items():
        shelf = (ktiv_meta.get(sys_num) or {}).get("shelf_mark") or ""
        if shelf:
            names.append((shelf, src))
    names += [(k, "benchmark:decontam_benchmark_keys") for k in sorted(gate.bench_keys)]
    names += [(k, "training:decontam_clean_v2_keys") for k in sorted(gate.clean_keys)]
    for name, src in names:
        key = bpe.fragment_key(name, gate)
        if key:
            inv.keys.setdefault(key, src)


# ----------------------------------------------------------------------------- strata, sampling


def language_stratum(primary: str, secondary: str = "") -> str:
    """The documentary language stratum of a PGP document.

    :param primary: ``languages_primary``.
    :type primary: str
    :param secondary: ``languages_secondary``.
    :type secondary: str
    :return: ``Judaeo-Arabic``, ``Arabic``, ``Hebrew`` or ``other`` (the script-tag bucket priority:
        Judaeo-Arabic before Arabic before Aramaic before Hebrew; Aramaic and untagged are ``other``).
    :rtype: str
    """
    return LANGUAGE_OF_BUCKET.get(bucket_for(primary or "", secondary or ""), "other")


def type_stratum(pgp_type: str) -> str:
    """The documentary type stratum of a PGP document.

    :param pgp_type: PGP ``type``.
    :type pgp_type: str
    :return: ``letter``, ``legal``, ``list`` or ``other``.
    :rtype: str
    """
    return TYPE_OF_PGP.get((pgp_type or "").strip(), "other")


def genre_of(title: str, extra: str = "") -> str:
    """Genre of a KTIV manuscript from its catalogue subjects.

    :param title: ``basic_catalog.title`` (subjects separated by ``;``).
    :type title: str
    :param extra: More catalogue text (writing-characteristics titles), used when no subject matches.
    :type extra: str
    :return: ``talmud/halakha``, ``midrash/aggadah``, ``liturgy/piyyut``, ``bible/masorah`` or ``other``.
    :rtype: str
    """
    for part in [p.strip() for p in (title or "").split(";") if p.strip()] + [extra or ""]:
        for genre, words in GENRE_KEYWORDS:
            if any(w in part for w in words):
                return genre
    return "other"


def allocate(pool_counts: Dict[str, int], n: int) -> Dict[str, int]:
    """Proportional allocation with largest remainders, capped by each stratum's pool.

    :param pool_counts: ``stratum -> eligible items``.
    :type pool_counts: Dict[str, int]
    :param n: Items to allocate.
    :type n: int
    :return: ``stratum -> items to draw`` (sums to ``min(n, total)``).
    :rtype: Dict[str, int]
    """
    total = sum(pool_counts.values())
    if total <= n:
        return dict(pool_counts)
    quota = {k: n * v / total for k, v in pool_counts.items()}
    out = {k: min(int(q), pool_counts[k]) for k, q in quota.items()}
    order = sorted(pool_counts, key=lambda k: (-(quota[k] - int(quota[k])), k))
    while sum(out.values()) < n:
        grew = False
        for k in order:
            if sum(out.values()) >= n:
                break
            if out[k] < pool_counts[k]:
                out[k] += 1
                grew = True
        if not grew:
            break
    return out


def seeded_order(items: Sequence[Any], key: str, seed: int = SEED) -> List[Any]:
    """Items in a seeded pseudo-random order that does not depend on the input order.

    :param items: Items.
    :type items: Sequence[Any]
    :param key: Attribute or dict key holding each item's stable id.
    :type key: str
    :param seed: Seed.
    :type seed: int
    :return: Reordered items.
    :rtype: List[Any]
    """
    def ident(x: Any) -> str:
        """Stable id of an item.

        :param x: Item.
        :type x: Any
        :return: Its id.
        :rtype: str
        """
        return str(x[key] if isinstance(x, dict) else getattr(x, key))
    return sorted(items, key=lambda x: hashlib.sha1(f"{seed}:{ident(x)}".encode("utf-8")).hexdigest())


# ----------------------------------------------------------------------------- documentary


@dataclass
class DocCandidate:
    """A PGP document that passed the document-level gates.

    :param pgpid: PGP id.
    :param canonical_id: Its only canonical id.
    :param urls: Served image URLs (index = image_index).
    :param content: Digital edition text.
    :param parsed: Parsed edition.
    :param meta: ``documents.csv`` row.
    :param institution: Merged-record institution.
    :param language: Language stratum.
    :param doc_type: Type stratum.
    """

    pgpid: str
    canonical_id: str
    urls: List[str]
    content: str
    parsed: Any
    meta: Dict[str, str]
    institution: str
    language: str
    doc_type: str


@dataclass
class SideEvidence:
    """Presence of one edition side on one page.

    :param side: ``recto`` / ``verso``, or None for an unlabelled edition.
    :param share: Share of the side's 4-grams found in the readers.
    :param null_max: Largest share on the null pages.
    :param hits: Matched 4-grams.
    :param present: ``share > null_max`` and ``hits >= MIN_HITS``.
    :param coverage: Letter share of the side's blocks that are present on their own.
    """

    side: Optional[str]
    share: float
    null_max: float
    hits: int
    present: bool
    coverage: float


@dataclass
class DocPage:
    """An eligible documentary page.

    :param cand: Its document.
    :param image_index: Index in ``cand.urls``.
    :param side: Assigned side (None = whole unlabelled edition).
    :param answer: Ground truth.
    :param evidence: Side evidence that decided it.
    :param model: Evidence reader.
    :param sha256: Hash of the image bytes the readers saw.
    """

    cand: DocCandidate
    image_index: int
    side: Optional[str]
    answer: Any
    evidence: SideEvidence
    model: str
    sha256: str

    @property
    def doc_id(self) -> str:
        """Benchmark id of the page (one page per document).

        :return: The canonical id.
        :rtype: str
        """
        return self.cand.canonical_id


def load_digital_editions(path: Path = FOOTNOTES_CSV, min_chars: int = DOC_MIN_EDITION_CHARS) -> Dict[str, Dict[str, str]]:
    """Longest digital edition per PGP document.

    :param path: ``footnotes.csv``.
    :type path: Path
    :param min_chars: Minimum stripped length of the content cell.
    :type min_chars: int
    :return: ``pgpid -> footnotes row`` (relation contains "Digital Edition").
    :rtype: Dict[str, Dict[str, str]]
    """
    eds: Dict[str, Dict[str, str]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            content = (r.get("content") or "").strip()
            if "Digital Edition" not in (r.get("doc_relation") or "") or len(content) < min_chars:
                continue
            pid = (r.get("document_id") or "").strip()
            if pid and (pid not in eds or len(content) > len(eds[pid]["content"].strip())):
                eds[pid] = r
    return eds


def merged_image_urls(images: Optional[Dict[str, Any]]) -> List[str]:
    """Absolute image URLs of a merged record in the index's ``image_urls`` order.

    Mirrors ``genizah_document._merged_image_urls``: the preferred source first, then KTIV,
    Bodleian and the legacy store; legacy-store entries are bare file names under the bucket.

    :param images: The record's ``images`` block.
    :type images: Optional[Dict[str, Any]]
    :return: URLs; position = ``image_index``.
    :rtype: List[str]
    """
    images = images or {}
    base = f"https://{GCS_HOST}/cairo-genizah-es-json/images"
    by = {"fjp": [i if str(i).startswith("http") else f"{base}/{i}" for i in (images.get("fjp") or [])],
          "ktiv": list((images.get("ktiv") or {}).get("image_urls") or []),
          "bodleian": list((images.get("bodleian") or {}).get("image_urls") or [])}
    order = ["ktiv", "bodleian", "fjp"]
    if images.get("preferred_source") in order:
        order.remove(images["preferred_source"])
        order.insert(0, images["preferred_source"])
    return [u for s in order for u in by[s]]


def is_gcs(url: str) -> bool:
    """Whether a URL is on the public bucket host (the only host this build downloads from).

    :param url: URL.
    :type url: str
    :return: True for ``https://storage.googleapis.com/...``.
    :rtype: bool
    """
    p = urlparse(url)
    return p.scheme == "https" and p.netloc == GCS_HOST


def doc_exclusion(pid: str, cid: str, names: Iterable[str], urls: Iterable[str], inv: Inventory,
                  merged: MergedIndex, gate: DecontamGate, parents: Dict[str, str]) -> Optional[str]:
    """The exclusion source that rules a documentary candidate out, if any.

    :param pid: PGP id.
    :type pid: str
    :param cid: Canonical id.
    :type cid: str
    :param names: Shelfmarks and fragment names of the document.
    :type names: Iterable[str]
    :param urls: Its image URLs.
    :type urls: Iterable[str]
    :param inv: Expanded inventory.
    :type inv: Inventory
    :param merged: Merged index.
    :type merged: MergedIndex
    :param gate: Decontamination gate.
    :type gate: DecontamGate
    :param parents: ``parent id -> source`` for excluded ids that carry a ``_<n>`` part suffix.
    :type parents: Dict[str, str]
    :return: ``<source> (<layer>)``, or None when the document is clean.
    :rtype: Optional[str]
    """
    if pid in inv.pgpids:
        return f"{inv.pgpids[pid]} (pgpid)"
    if cid in inv.ids:
        return f"{inv.ids[cid]} (canonical id)"
    parent = bpe.strip_suffix(cid)
    if parent != cid and parent in inv.ids:
        return f"{inv.ids[parent]} (sub-part of an excluded id)"
    if cid in parents:
        return f"{parents[cid]} (an excluded id is a sub-part)"
    if parent != cid and parent in parents and merged.pgpids_of.get(parent):
        return f"{parents[parent]} (sibling sub-part)"
    if merged.sys_of.get(cid) in inv.sys_nums:
        return f"{inv.sys_nums[merged.sys_of[cid]]} (KTIV manuscript of the record)"
    for name in names:
        key = bpe.fragment_key(name, gate)
        if key and key in inv.keys:
            return f"{inv.keys[key]} (fragment key)"
    for u in urls:
        if u in inv.urls:
            return f"{inv.urls[u]} (image url)"
    return None


def documentary_candidates(editions: Dict[str, Dict[str, str]], merged: MergedIndex, pgp_docs: Dict[str, Dict[str, str]],
                           fragments: Tuple[Dict[str, List[str]], Dict[str, Set[str]]], inv: Inventory,
                           gate: DecontamGate, bench: Any) -> Tuple[List[DocCandidate], collections.Counter, collections.Counter]:
    """Documents with a digital edition that pass the document gates and the exclusion inventory.

    :param editions: ``pgpid -> footnotes row``.
    :type editions: Dict[str, Dict[str, str]]
    :param merged: Merged index.
    :type merged: MergedIndex
    :param pgp_docs: ``documents.csv`` by pgpid.
    :type pgp_docs: Dict[str, Dict[str, str]]
    :param fragments: ``(fragment -> pgpids, pgpid -> fragments)``.
    :type fragments: Tuple[Dict[str, List[str]], Dict[str, Set[str]]]
    :param inv: Expanded inventory.
    :type inv: Inventory
    :param gate: Decontamination gate.
    :type gate: DecontamGate
    :param bench: PGP-131 :class:`build_pgp_editions.Benchmark` (text-duplicate layer).
    :type bench: Any
    :return: ``(candidates, drop reasons, exclusions by source)``.
    :rtype: Tuple[List[DocCandidate], collections.Counter, collections.Counter]
    """
    f2p, p2f = fragments
    parents = {bpe.strip_suffix(c): s for c, s in inv.ids.items() if bpe.strip_suffix(c) != c}
    reasons: collections.Counter = collections.Counter()
    by_source: collections.Counter = collections.Counter()
    out: List[DocCandidate] = []
    for pid in sorted(editions, key=lambda p: (len(p), p)):
        meta = pgp_docs.get(pid, {})
        shelf = meta.get("shelfmark") or ""
        frags = p2f.get(pid, set())
        cids = merged.canonicals_of.get(pid, [])
        if not cids:
            reasons["not_in_merged_index"] += 1
            continue
        if len(cids) > 1 or (meta.get("multifragment") or "").strip() or "+" in shelf or len(frags) > 1:
            reasons["join_or_multifragment"] += 1
            continue
        cid = cids[0]
        urls = merged_image_urls((merged.records.get(cid) or {}).get("images"))
        names = {cid} | {x.strip() for x in shelf.split("+") if x.strip()} | set(frags)
        why = doc_exclusion(pid, cid, names, urls, inv, merged, gate, parents)
        if why:
            reasons["excluded"] += 1
            by_source[why] += 1
            continue
        _, containment, _ = bpe.benchmark_overlap(editions[pid]["content"], bench)
        if containment >= bpe.SHINGLE_CONTAINMENT:
            reasons["benchmark_text_duplicate"] += 1
            continue
        if {q for f in frags for q in f2p.get(f, [])} - {pid}:
            reasons["fragment_shared_with_other_document"] += 1
            continue
        if not urls:
            reasons["no_image"] += 1
            continue
        if not all(is_gcs(u) for u in urls):
            reasons["image_not_on_bucket"] += 1
            continue
        parsed = bpe.parse_edition(editions[pid]["content"])
        if parsed.flattened:
            reasons["edition_flattened"] += 1
            continue
        if parsed.complex_sides:
            reasons["complex_side_labels"] += 1
            continue
        if not any(b.lines for b in parsed.blocks):
            reasons["edition_empty_after_cleaning"] += 1
            continue
        reasons["candidate"] += 1
        out.append(DocCandidate(pid, cid, urls, editions[pid]["content"], parsed, meta,
                                (merged.records.get(cid) or {}).get("institution") or "",
                                language_stratum(meta.get("languages_primary", ""), meta.get("languages_secondary", "")),
                                type_stratum(meta.get("type", ""))))
    return out, reasons, by_source


def ngram_presence(lines: Sequence[str], reader: Any, nulls: Sequence[Any]) -> Tuple[float, float, int, bool]:
    """The edition builder's n-gram presence test without the line anchor.

    :param lines: Edition lines.
    :type lines: Sequence[str]
    :param reader: The page's :class:`build_pgp_editions.Reader`.
    :type reader: Any
    :param nulls: Null-page readers.
    :type nulls: Sequence[Any]
    :return: ``(share, null max, hits, present)``.
    :rtype: Tuple[float, float, int, bool]
    """
    g = bpe.grams(lines)
    if not g:
        return 0.0, 0.0, 0, False
    hits = len(g & reader.grams)
    share = hits / len(g)
    null_max = max((len(g & n.grams) / len(g) for n in nulls), default=0.0)
    return share, null_max, hits, share > null_max and hits >= bpe.MIN_HITS


def edition_sides(ed: Any) -> List[Optional[str]]:
    """Sides of a parsed edition that hold lines.

    :param ed: Parsed edition.
    :type ed: Any
    :return: ``[None]`` for an unlabelled edition, else ``recto`` / ``verso`` in that order.
    :rtype: List[Optional[str]]
    """
    if not ed.labelled:
        return [None]
    return [s for s in ("recto", "verso") if any(b.side == s and b.lines for b in ed.blocks)]


def side_evidence(ed: Any, reader: Any, nulls: Sequence[Any]) -> List[SideEvidence]:
    """Presence of each side of an edition on one page.

    :param ed: Parsed edition.
    :type ed: Any
    :param reader: The page's readers.
    :type reader: Any
    :param nulls: Null-page readers.
    :type nulls: Sequence[Any]
    :return: One entry per side.
    :rtype: List[SideEvidence]
    """
    out = []
    for side in edition_sides(ed):
        blocks = [b for b in ed.blocks if b.lines and (side is None or b.side == side)]
        share, null_max, hits, present = ngram_presence([ln for b in blocks for ln in b.lines], reader, nulls)
        letters = sum(b.hebrew for b in blocks)
        covered = sum(b.hebrew for b in blocks if ngram_presence(b.lines, reader, nulls)[3])
        out.append(SideEvidence(side, round(share, 4), round(null_max, 4), hits, present,
                                round(covered / letters, 4) if letters else 0.0))
    return out


def assign_side(evidence: Sequence[SideEvidence], min_coverage: float = SIDE_MIN_COVERAGE) -> Tuple[Optional[SideEvidence], str]:
    """The side a page shows, when exactly one side is on it.

    :param evidence: :func:`side_evidence` of the page.
    :type evidence: Sequence[SideEvidence]
    :param min_coverage: Required letter share of the side's present blocks.
    :type min_coverage: float
    :return: ``(evidence of the assigned side, "")`` or ``(None, reason)``.
    :rtype: Tuple[Optional[SideEvidence], str]
    """
    present = [e for e in evidence if e.present]
    if not present:
        return None, "no_side_present"
    if len(present) > 1:
        return None, "several_sides_present"
    if present[0].coverage < min_coverage:
        return None, "side_coverage_low"
    return present[0], ""


def side_answer(ed: Any, side: Optional[str]) -> Any:
    """Ground truth of one side: all its blocks, main text first (:func:`build_pgp_editions.assemble_answer`).

    :param ed: Parsed edition.
    :type ed: Any
    :param side: Side (None = the whole unlabelled edition).
    :type side: Optional[str]
    :return: The :class:`build_pgp_editions.Answer`.
    :rtype: Any
    """
    return bpe.assemble_answer(ed, [True] * len(ed.blocks), side)


def gt_gate(text: str, flags: Iterable[Iterable[str]], n_lines: int) -> Optional[str]:
    """Reject a documentary ground truth the scorer could not use as is.

    :param text: Ground-truth text (lines joined by newlines).
    :type text: str
    :param flags: Per line, unresolved notation flags (the edition builder's).
    :type flags: Iterable[Iterable[str]]
    :param n_lines: Number of lines.
    :type n_lines: int
    :return: Drop reason, or None.
    :rtype: Optional[str]
    """
    found = sorted({f for fl in flags for f in fl})
    if found:
        return f"notation_{found[0]}"
    heb = bpe.n_hebrew(text)
    if heb < HEBREW_MIN_SHARE * max(1, bpe.n_semitic(text)):
        return "mostly_arabic_script"
    if heb < DOC_MIN_LETTERS:
        return "too_few_letters"
    if n_lines < DOC_MIN_LINES:
        return "too_few_lines"
    if find_self_duplication(text) > SELF_DUPLICATION_MAX:
        return "self_duplication"
    if letters_only(dedup_repeated_tail(text)) != letters_only(text):
        return "repeated_tail"
    return None


def load_evidence(raw_dir: Path, cid: str, index: int, url: str, models: Sequence[str] = EVIDENCE_MODELS) -> Tuple[Optional[Dict[str, Any]], str]:
    """The cached two-reader evidence of one page image, if a read of exactly this URL exists.

    :param raw_dir: ``ai_reads/raw``.
    :type raw_dir: Path
    :param cid: Canonical id.
    :type cid: str
    :param index: Image index.
    :type index: int
    :param url: The image URL the page must have been read from.
    :type url: str
    :param models: Readers in order of preference (the null pool was read by the first).
    :type models: Sequence[str]
    :return: ``(raw evidence, model)`` or ``(None, "")``.
    :rtype: Tuple[Optional[Dict[str, Any]], str]
    """
    for model in models:
        path = raw_dir / f"{cid}__{index}__{model}.json"
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:   # the running pipeline may be writing this file
            continue
        if raw.get("image_url") == url:
            return raw, model
    return None, ""


def documentary_pages(cands: Sequence[DocCandidate], raw_dir: Path, pool_names: Sequence[str],
                      pool: Dict[str, Any]) -> Tuple[List[DocPage], collections.Counter]:
    """Eligible pages of the candidates (one per document at most).

    :param cands: Documentary candidates.
    :type cands: Sequence[DocCandidate]
    :param raw_dir: ``ai_reads/raw``.
    :type raw_dir: Path
    :param pool_names: Frozen null pool (evidence file names).
    :type pool_names: Sequence[str]
    :param pool: Null-pool readers.
    :type pool: Dict[str, Any]
    :return: ``(pages, outcome counts per page image)``.
    :rtype: Tuple[List[DocPage], collections.Counter]
    """
    outcomes: collections.Counter = collections.Counter()
    pages: List[DocPage] = []
    for cand in cands:
        doc_pages: List[DocPage] = []
        for idx, url in enumerate(cand.urls):
            raw, model = load_evidence(raw_dir, cand.canonical_id, idx, url)
            if raw is None:
                outcomes["no_evidence"] += 1
                continue
            reader = bpe.reader_from_raw(raw)
            null_keys, _ = bpe.null_sample(bpe.page_key(cand.canonical_id, idx), pool_names)
            nulls = [pool[k] for k in null_keys]
            ev, reason = assign_side(side_evidence(cand.parsed, reader, nulls))
            if ev is None:
                outcomes[reason] += 1
                continue
            ans = side_answer(cand.parsed, ev.side)
            reason = gt_gate(ans.text, ans.flags, len(ans.lines))
            if reason:
                outcomes[reason] += 1
                continue
            outcomes["eligible"] += 1
            doc_pages.append(DocPage(cand, idx, ev.side, ans, ev, model, raw.get("sha256") or ""))
        if doc_pages:
            pages.append(seeded_order(doc_pages, "image_index", seed=bpe.stable_hash(cand.canonical_id) % 10 ** 9)[0])
    return pages, outcomes


def sample_documentary(pages: Sequence[DocPage], n: int = DOC_TARGET, seed: int = SEED) -> Tuple[List[DocPage], Dict[str, Any]]:
    """Stratified sample by language x type, proportional to the pool; single-image documents first.

    :param pages: Eligible pages (one per document).
    :type pages: Sequence[DocPage]
    :param n: Target size.
    :type n: int
    :param seed: Seed.
    :type seed: int
    :return: ``(selected pages, strata table)``.
    :rtype: Tuple[List[DocPage], Dict[str, Any]]
    """
    strata: Dict[str, List[DocPage]] = collections.defaultdict(list)
    for p in pages:
        strata[f"{p.cand.language}|{p.cand.doc_type}"].append(p)
    quota = allocate({k: len(v) for k, v in strata.items()}, n)
    selected: List[DocPage] = []
    for k in sorted(strata):
        ordered = seeded_order(strata[k], "doc_id", seed)
        ordered.sort(key=lambda p: len(p.cand.urls) > 1)       # stable: single-image documents first
        selected.extend(ordered[:quota[k]])
    table = {k: {"pool": len(strata[k]), "selected": quota[k]} for k in sorted(strata)}
    return sorted(selected, key=lambda p: p.doc_id), table


def documentary_doc(page: DocPage) -> Dict[str, Any]:
    """One spec entry in the verified-JSON format (+ sealed fields).

    :param page: Selected page.
    :type page: DocPage
    :return: The entry.
    :rtype: Dict[str, Any]
    """
    gt = page.answer.text
    cleaned, recon, visible = clean_diplomatic(gt)
    meta = page.cand.meta
    return {
        "doc_id": page.doc_id, "shelf_mark": meta.get("shelfmark", ""), "institution": page.cand.institution,
        "image_url": page.cand.urls[page.image_index], "gt": gt, "visible_chars": visible,
        "gaps": gt.count(bpe.GAP_TOKEN), "recon_frac": round(recon / max(1, recon + visible), 4),
        "_repaired_duplication": False,
        "pgpid": page.cand.pgpid, "language": page.cand.language, "type": page.cand.doc_type, "source": NAME,
        "image_index": page.image_index, "n_images": len(page.cand.urls), "side": page.side,
        "languages_primary": meta.get("languages_primary", ""), "pgp_type": meta.get("type", ""),
        "image_sha256": page.sha256,
        "alignment": {"reader": page.model, **{k: v for k, v in asdict(page.evidence).items() if k != "side"}},
    }


# ----------------------------------------------------------------------------- literary


@dataclass
class LitPage:
    """An eligible KTIV page.

    :param sys_num: Manuscript.
    :param fl: NLI page id.
    :param shelf_mark: KTIV shelfmark.
    :param genre: Genre.
    :param n_columns: Columns of the reconstructed page.
    :param gt_shape: ``api_clean`` or ``api_served``.
    :param gt: Ground truth.
    :param image_source: ``("zip", "<zip path>|<member>")`` or ``("url", <bucket URL>)``.
    :param title: KTIV catalogue title.
    :param canonical_overlap_train: >= 2 shared 25-letter shingles with KTIV training page text.
    :param benchmark_text_overlap: >= 2 shared shingles with religious-140 / PGP-131 GT.
    """

    sys_num: str
    fl: str
    shelf_mark: str
    genre: str
    n_columns: int
    gt_shape: str
    gt: str
    image_source: Tuple[str, str]
    title: str
    canonical_overlap_train: bool
    benchmark_text_overlap: bool

    @property
    def doc_id(self) -> str:
        """Religious-140 style id.

        :return: ``KTIV_<sys_num>_<FL>``.
        :rtype: str
        """
        return f"KTIV_{self.sys_num}_{self.fl}"

    @property
    def multi(self) -> bool:
        """Two or more columns.

        :return: ``n_columns >= 2``.
        :rtype: bool
        """
        return self.n_columns >= 2


def longest_run(text: str) -> int:
    """Longest space-free Hebrew letter run (the GT-mash symptom).

    :param text: GT text.
    :type text: str
    :return: Run length.
    :rtype: int
    """
    return max((len(m) for m in _HEB_RUN_RE.findall(text)), default=0)


def zip_members(sys_num: str) -> List[Tuple[str, str]]:
    """Image members of a manuscript's zips (local scrape folder and NAS store) and unzipped folders.

    :param sys_num: Manuscript.
    :type sys_num: str
    :return: ``(container, member)`` pairs; container is a zip path or a folder path.
    :rtype: List[Tuple[str, str]]
    """
    out: List[Tuple[str, str]] = []
    for z in resolve_zips(sys_num):
        try:
            with zipfile.ZipFile(z) as zf:
                out += [(str(z), n) for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
        except (zipfile.BadZipFile, OSError):    # truncated scrape downloads are known (logs/ktiv_move)
            continue
    folder = KTIV_DIR / f"ktiv_PNX_MANUSCRIPTS{sys_num}-1_images"
    if folder.is_dir():
        out += [(str(folder), n) for n in sorted(os.listdir(folder)) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    return out


def page_image_source(fl: str, members: Sequence[Tuple[str, str]], urls: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Where a KTIV page image can be read from.

    :param fl: NLI page id.
    :type fl: str
    :param members: :func:`zip_members` of the manuscript.
    :type members: Sequence[Tuple[str, str]]
    :param urls: Bucket URLs of the manuscript's merged records.
    :type urls: Sequence[str]
    :return: ``("zip", "<container>|<member>")``, ``("url", url)`` or None.
    :rtype: Optional[Tuple[str, str]]
    """
    for container, member in members:
        if _FL_RE.search(member) and _FL_RE.search(member).group(1) == fl:
            return "zip", f"{container}|{member}"
    for u in urls:
        m = _FL_RE.search(u.rsplit("/", 1)[-1])
        if m and m.group(1) == fl and is_gcs(u):
            return "url", u
    return None


def literary_pages(bundles: Dict[str, Tuple[Path, Dict[str, Any]]], inv: Inventory, merged: MergedIndex,
                   ktiv_meta: Dict[str, Dict[str, Any]], gate: DecontamGate, train_shingles: Set[str],
                   bench_shingles: Set[str]) -> Tuple[List[LitPage], collections.Counter, collections.Counter]:
    """Eligible pages of the KTIV manuscripts no exclusion source holds.

    :param bundles: :func:`ktiv_bundles.select_bundles` result.
    :type bundles: Dict[str, Tuple[Path, Dict[str, Any]]]
    :param inv: Expanded inventory.
    :type inv: Inventory
    :param merged: Merged index.
    :type merged: MergedIndex
    :param ktiv_meta: KTIV metadata by sys_num.
    :type ktiv_meta: Dict[str, Dict[str, Any]]
    :param gate: Decontamination gate.
    :type gate: DecontamGate
    :param train_shingles: Shingles of KTIV training page text.
    :type train_shingles: Set[str]
    :param bench_shingles: Shingles of religious-140 and PGP-131 GT.
    :type bench_shingles: Set[str]
    :return: ``(pages, page outcomes, manuscript exclusions by source)``.
    :rtype: Tuple[List[LitPage], collections.Counter, collections.Counter]
    """
    outcomes: collections.Counter = collections.Counter()
    by_source: collections.Counter = collections.Counter()
    pages: List[LitPage] = []
    for sys_num, (_, doc) in sorted(bundles.items()):
        if bundle_shape(doc) != "api":
            outcomes["ms_dom_shape"] += 1
            continue
        meta = ktiv_meta.get(sys_num, {})
        shelf = meta.get("shelf_mark") or ""
        why = inv.sys_nums.get(sys_num)
        if why is None:
            why = next((f"{inv.ids[c]} (merged record of the manuscript)" for c in merged.cids_of_sys.get(sys_num, [])
                        if c in inv.ids), None)
        if why is None and shelf:
            key = bpe.fragment_key(shelf, gate)
            why = f"{inv.keys[key]} (fragment key)" if key and key in inv.keys else None
        if why:
            outcomes["ms_excluded"] += 1
            by_source[why] += 1
            continue
        title = (meta.get("basic_catalog") or {}).get("title") or ""
        extra = " ".join(((e.get("subsections") or {}).get("writing_characteristics") or {}).get("title", "")
                         for e in (meta.get("scholarly_entries") or []))
        genre = genre_of(title, extra)
        min_letters = LIT_MIN_LETTERS_TALMUD if genre == "talmud/halakha" else LIT_MIN_LETTERS_OTHER
        urls = [u for c in merged.cids_of_sys.get(sys_num, [])
                for u in ((((merged.records.get(c) or {}).get("images") or {}).get("ktiv") or {}).get("image_urls") or [])]
        members: Optional[List[Tuple[str, str]]] = None
        for page in doc.get("pages") or []:
            fl = page.get("fl")
            items = (page.get("annotation_page") or {}).get("items") or []
            if not fl or not items:
                outcomes["page_no_word_boxes"] += 1
                continue
            rec = reconstruct_page(items)
            if hebrew_letters(rec["text"]) < min_letters:
                outcomes["page_too_few_letters"] += 1
                continue
            if longest_run(rec["text"]) >= MASH_RUN:
                outcomes["page_gt_mash"] += 1
                continue
            if members is None:
                members = zip_members(sys_num)
            src = page_image_source(fl, members, urls)
            if src is None:
                outcomes["page_no_image_local_or_bucket"] += 1
                continue
            outcomes["eligible"] += 1
            pages.append(LitPage(sys_num, fl, shelf, genre, max(len(rec["columns"]), 1),
                                 "api_served" if rec.get("mode") == "served_breaklines" else "api_clean",
                                 rec["text"], src, title, shingle_hits(rec["text"], train_shingles) >= 2,
                                 shingle_hits(rec["text"], bench_shingles) >= 2))
    return pages, outcomes, by_source


def sample_literary(pages: Sequence[LitPage], n: int = LIT_TARGET, min_multi: int = LIT_MIN_MULTI,
                    max_per_ms: int = LIT_MAX_PAGES_PER_MS, seed: int = SEED) -> Tuple[List[LitPage], Dict[str, Any]]:
    """Stratified literary sample: column class x genre, >= ``min_multi`` multi-column pages, <= ``max_per_ms`` per manuscript.

    The multi-column share is the pool's, raised to ``min_multi``; within a column class genres are
    allocated proportionally. Pages are drawn in seeded order; a page whose manuscript already has
    ``max_per_ms`` pages is skipped and its stratum's quota is refilled from the class's other strata.

    :param pages: Eligible pages.
    :type pages: Sequence[LitPage]
    :param n: Target size.
    :type n: int
    :param min_multi: Minimum multi-column pages.
    :type min_multi: int
    :param max_per_ms: Page cap per manuscript.
    :type max_per_ms: int
    :param seed: Seed.
    :type seed: int
    :return: ``(selected pages, strata table)``.
    :rtype: Tuple[List[LitPage], Dict[str, Any]]
    """
    by_class: Dict[str, List[LitPage]] = {"multi": [p for p in pages if p.multi], "single": [p for p in pages if not p.multi]}
    n_multi = min(len(by_class["multi"]), max(min_multi, round(n * len(by_class["multi"]) / max(1, len(pages)))))
    class_quota = {"multi": n_multi, "single": n - n_multi}
    per_ms: collections.Counter = collections.Counter()
    selected: List[LitPage] = []
    table: Dict[str, Any] = {}
    for cls in ("multi", "single"):
        strata: Dict[str, List[LitPage]] = collections.defaultdict(list)
        for p in by_class[cls]:
            strata[p.genre].append(p)
        quota = allocate({g: len(v) for g, v in strata.items()}, class_quota[cls])
        taken: Dict[str, int] = collections.Counter()
        leftovers: List[LitPage] = []
        for g in sorted(strata):
            for p in seeded_order(strata[g], "doc_id", seed):
                if taken[g] < quota[g] and per_ms[p.sys_num] < max_per_ms:
                    selected.append(p)
                    per_ms[p.sys_num] += 1
                    taken[g] += 1
                else:
                    leftovers.append(p)
        for p in seeded_order(leftovers, "doc_id", seed):       # refill quota lost to the manuscript cap
            if sum(taken.values()) >= class_quota[cls]:
                break
            if p not in selected and per_ms[p.sys_num] < max_per_ms:
                selected.append(p)
                per_ms[p.sys_num] += 1
                taken[p.genre] += 1
        for g in sorted(strata):
            table[f"{cls}|{g}"] = {"pool": len(strata[g]), "quota": quota.get(g, 0), "selected": taken[g]}
    chosen = {p.doc_id for p in selected}
    for p in seeded_order(pages, "doc_id", seed):        # a class the manuscript cap starved: top up from the other
        if len(selected) >= n:
            break
        if p.doc_id not in chosen and per_ms[p.sys_num] < max_per_ms:
            selected.append(p)
            chosen.add(p.doc_id)
            per_ms[p.sys_num] += 1
            table.setdefault("top_up", {"selected": 0})["selected"] += 1
    return sorted(selected, key=lambda p: p.doc_id), table


def disjoint_from_documentary(pages: Sequence[LitPage], doc_pages: Sequence[DocPage],
                              merged: MergedIndex) -> Tuple[List[LitPage], int]:
    """Literary pages whose manuscript is not the fragment of a documentary page (the halves never share a leaf).

    :param pages: Eligible literary pages.
    :type pages: Sequence[LitPage]
    :param doc_pages: Selected documentary pages.
    :type doc_pages: Sequence[DocPage]
    :param merged: Merged index.
    :type merged: MergedIndex
    :return: ``(kept pages, dropped count)``.
    :rtype: Tuple[List[LitPage], int]
    """
    doc_ids = {p.doc_id for p in doc_pages}
    doc_sys = {merged.sys_of[i] for i in doc_ids if i in merged.sys_of}
    kept = [p for p in pages if p.sys_num not in doc_sys
            and not any(c in doc_ids for c in merged.cids_of_sys.get(p.sys_num, []))]
    return kept, len(pages) - len(kept)


def literary_doc(page: LitPage, images_dir: Path) -> Dict[str, Any]:
    """One spec entry in the religious-140 format (+ sealed fields).

    :param page: Selected page.
    :type page: LitPage
    :param images_dir: Absolute image folder.
    :type images_dir: Path
    :return: The entry.
    :rtype: Dict[str, Any]
    """
    return {
        "doc_id": page.doc_id, "sys_num": page.sys_num, "fl": page.fl, "shelf_mark": page.shelf_mark,
        "genre": page.genre, "is_talmud": page.genre == "talmud/halakha", "n_columns": page.n_columns,
        "gt_shape": page.gt_shape, "gt": page.gt, "gt_letters": hebrew_letters(page.gt),
        "image": str(images_dir / f"{page.sys_num}_{page.fl}.jpg"), "image_name": page.image_source[1].split("|")[-1].rsplit("/", 1)[-1],
        "gt_mash": False, "canonical_overlap_train": page.canonical_overlap_train,
        "benchmark_text_overlap": page.benchmark_text_overlap, "title": page.title, "source": NAME,
    }


# ----------------------------------------------------------------------------- images


def fetch_bucket_image(url: str, dest: Path, session: requests.Session, sha256: str = "") -> Dict[str, Any]:
    """Download a bucket image; keep the original bytes unless EXIF says it must be rotated upright.

    :param url: ``https://storage.googleapis.com/...`` URL (anything else is refused).
    :type url: str
    :param dest: Target JPEG.
    :type dest: Path
    :param session: HTTP session.
    :type session: requests.Session
    :param sha256: Expected hash of the bytes (the image the readers saw); "" skips the check.
    :type sha256: str
    :return: ``{"ok", "reason", "sha256", "width", "height", "reencoded"}``.
    :rtype: Dict[str, Any]
    :raises ValueError: For a URL outside the bucket host.
    """
    if not is_gcs(url):
        raise ValueError(f"refusing to download from outside {GCS_HOST}: {url}")
    data = b""
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
            data = resp.content
            break
        except requests.RequestException as exc:   # transient network failures on a long download run
            logger.warning("download failed (%d/%d) %s: %s", attempt + 1, DOWNLOAD_RETRIES, url, exc)
            time.sleep(2 * (attempt + 1))
    if not data:
        return {"ok": False, "reason": "download_failed"}
    digest = hashlib.sha256(data).hexdigest()
    if sha256 and digest != sha256:
        return {"ok": False, "reason": "image_changed_since_read", "sha256": digest}
    im = PILImage.open(io.BytesIO(data))
    orientation = im.getexif().get(0x0112, 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if im.format == "JPEG" and orientation in (0, 1) and im.mode == "RGB":
        dest.write_bytes(data)
        return {"ok": True, "reason": "", "sha256": digest, "width": im.width, "height": im.height, "reencoded": False}
    up = ImageOps.exif_transpose(im).convert("RGB")
    up.save(dest, "JPEG", quality=95)
    return {"ok": True, "reason": "", "sha256": digest, "width": up.width, "height": up.height, "reencoded": True}


def save_scaled(im: PILImage.Image, dest: Path, width: int = KTIV_TARGET_WIDTH) -> Tuple[int, int]:
    """Save a page image downscaled to ``width`` (never upscaled), as religious-140 stores its pages.

    :param im: Image.
    :type im: PILImage.Image
    :param dest: Target JPEG.
    :type dest: Path
    :param width: Target width.
    :type width: int
    :return: Saved ``(width, height)``.
    :rtype: Tuple[int, int]
    """
    im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), PILImage.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=90)
    return im.width, im.height


def materialise_ktiv_image(source: Tuple[str, str], dest: Path, session: requests.Session) -> Dict[str, Any]:
    """Write a KTIV page image from its zip / folder or the bucket.

    :param source: :func:`page_image_source` result.
    :type source: Tuple[str, str]
    :param dest: Target JPEG.
    :type dest: Path
    :param session: HTTP session (bucket fallback).
    :type session: requests.Session
    :return: ``{"ok", "reason", "from", "width", "height"}``.
    :rtype: Dict[str, Any]
    """
    kind, where = source
    if kind == "zip":
        container, member = where.split("|", 1)
        if container.endswith(".zip"):
            with zipfile.ZipFile(container) as zf:
                im = PILImage.open(io.BytesIO(zf.read(member)))
                w, h = save_scaled(im, dest)
        else:
            w, h = save_scaled(PILImage.open(Path(container) / member), dest)
        return {"ok": True, "reason": "", "from": where, "width": w, "height": h}
    tmp = dest.with_suffix(".download")
    meta = fetch_bucket_image(where, tmp, session)
    if not meta["ok"]:
        return {**meta, "from": where}
    w, h = save_scaled(PILImage.open(tmp), dest)
    tmp.unlink()
    return {"ok": True, "reason": "", "from": where, "width": w, "height": h}


# ----------------------------------------------------------------------------- build


def registry_entries(doc_pages: Sequence[DocPage], lit_pages: Sequence[LitPage], merged: MergedIndex) -> Tuple[Set[str], Set[str], Set[str]]:
    """What the held-out registry must list so that every builder leaves the sealed pages out.

    :param doc_pages: Selected documentary pages.
    :type doc_pages: Sequence[DocPage]
    :param lit_pages: Selected literary pages.
    :type lit_pages: Sequence[LitPage]
    :param merged: Merged index.
    :type merged: MergedIndex
    :return: ``(ids, pgpids, sys_nums)``: documentary doc ids and every canonical id of their PGP ids,
        literary doc ids and the merged records of their manuscripts; PGP ids; literary manuscripts and
        the KTIV manuscripts of the documentary records.
    :rtype: Tuple[Set[str], Set[str], Set[str]]
    """
    ids: Set[str] = set()
    pgpids: Set[str] = set()
    sys_nums: Set[str] = set()
    for p in doc_pages:
        pgpids.add(p.cand.pgpid)
        ids.add(p.doc_id)
        ids.update(merged.canonicals_of.get(p.cand.pgpid, []))
        if p.doc_id in merged.sys_of:
            sys_nums.add(merged.sys_of[p.doc_id])
    for p in lit_pages:
        ids.add(p.doc_id)
        sys_nums.add(p.sys_num)
        ids.update(merged.cids_of_sys.get(p.sys_num, []))
        pgpids.update(pid for c in merged.cids_of_sys.get(p.sys_num, []) for pid in merged.pgpids_of.get(c, []))
    return ids, pgpids, sys_nums


def training_ktiv_shingles(nas: Path = NAS) -> Set[str]:
    """25-letter shingles of every KTIV v4 page-transcription target (train + val).

    :param nas: NAS datasets root.
    :type nas: Path
    :return: Shingles (informational overlap flag of literary pages).
    :rtype: Set[str]
    """
    out: Set[str] = set()
    for f in sorted((nas / "genizah_ktiv_v4_images_once/rows").glob("*.parquet")):
        t = pq.read_table(f, columns=["task", "answer"])
        for task, ans in zip(t.column("task").to_pylist(), t.column("answer").to_pylist()):
            if task in ("fragment_transcribe", "page_short"):
                s = "".join(re.findall(r"[א-ת]", ans or ""))
                out.update(s[i:i + 25] for i in range(len(s) - 24))
    return out


def build(out_dir: Path = OUT_DIR, dry_run: bool = False, report_path: Optional[Path] = None,
          seed: int = SEED, doc_target: int = DOC_TARGET, lit_target: int = LIT_TARGET) -> Dict[str, Any]:
    """Build (or dry-run) the sealed benchmark.

    :param out_dir: Output directory (created; never anything else under ``raw_data``).
    :type out_dir: Path
    :param dry_run: Select only: no images, no spec, no registry.
    :type dry_run: bool
    :param report_path: Where to write the build report (default ``out_dir/build_report.json``; a dry run
        writes nothing unless given one).
    :type report_path: Optional[Path]
    :param seed: Sampling seed.
    :type seed: int
    :param doc_target: Documentary pages wanted.
    :type doc_target: int
    :param lit_target: Literary pages wanted.
    :type lit_target: int
    :return: The build report.
    :rtype: Dict[str, Any]
    """
    t0 = time.time()
    gate = DecontamGate()
    inv = build_inventory()
    logger.info("inventory: %d sources (%.0fs)", len(inv.sources), time.time() - t0)
    editions = load_digital_editions()
    bundles = select_bundles(KTIV_DIR)
    api_sys = {s for s, (_, d) in bundles.items() if bundle_shape(d) == "api"}
    merged = scan_merged(bpe.MERGED_JSONL, set(editions), api_sys)
    pgp_docs = bpe.load_csv_by(bpe.DOCUMENTS_CSV, "pgpid")
    fragments = bpe.load_fragment_pgpids(bpe.FRAGMENTS_CSV)
    ktiv_meta = _ktiv_meta_index()
    expand_inventory(inv, merged, pgp_docs, fragments[1], ktiv_meta, gate)
    logger.info("expanded: %d ids, %d pgpids, %d sys_nums, %d keys", len(inv.ids), len(inv.pgpids), len(inv.sys_nums),
                len(inv.keys))

    others = [n for n in reg.BENCHMARK_REGISTRIES if n != NAME]
    bench = bpe.load_benchmark(gate, extra_ids=reg.registered_benchmark_documents(others)[0])
    cands, doc_reasons, doc_excl = documentary_candidates(editions, merged, pgp_docs, fragments, inv, gate, bench)
    pool_names = json.loads(_require(NULL_POOL, "edition-builder null pool").read_text(encoding="utf-8"))["pages"]
    pool = {n: bpe.reader_from_raw(json.loads((RAW_DIR / n).read_text(encoding="utf-8"))) for n in pool_names}
    doc_eligible, doc_outcomes = documentary_pages(cands, RAW_DIR, pool_names, pool)
    doc_selected, doc_strata = sample_documentary(doc_eligible, doc_target, seed)

    bench_sh = benchmark_shingles(RELIGIOUS_SPEC) | benchmark_shingles(bpe.BENCH_GT_JSON)
    lit_eligible, lit_outcomes, lit_excl = literary_pages(bundles, inv, merged, ktiv_meta, gate,
                                                          training_ktiv_shingles(), bench_sh)
    lit_eligible, lit_shared = disjoint_from_documentary(lit_eligible, doc_selected, merged)
    lit_selected, lit_strata = sample_literary(lit_eligible, lit_target, seed=seed)

    report: Dict[str, Any] = {
        "name": NAME, "built": date.today().isoformat(), "seed": seed, "dry_run": dry_run,
        "sources": inv.sources,
        "expanded_inventory": {"ids": len(inv.ids), "pgpids": len(inv.pgpids), "urls": len(inv.urls),
                               "sys_nums": len(inv.sys_nums), "fragment_keys": len(inv.keys)},
        "documentary": {
            "digital_editions": len(editions), "document_reasons": dict(doc_reasons),
            "excluded_by_source": dict(doc_excl.most_common()), "page_outcomes": dict(doc_outcomes),
            "eligible_pages": len(doc_eligible), "target": doc_target, "selected": len(doc_selected),
            "strata": doc_strata,
            "selected_multi_image_documents": sum(len(p.cand.urls) > 1 for p in doc_selected),
            "selected_sides": dict(collections.Counter(str(p.side) for p in doc_selected)),
            "selected_readers": dict(collections.Counter(p.model for p in doc_selected)),
            "rules": {"side_min_coverage": SIDE_MIN_COVERAGE, "min_letters": DOC_MIN_LETTERS,
                      "min_lines": DOC_MIN_LINES, "hebrew_min_share": HEBREW_MIN_SHARE,
                      "evidence_readers": list(EVIDENCE_MODELS), "null_pool": str(NULL_POOL)}},
        "literary": {
            "manuscripts_with_bundles": len(bundles), "api_manuscripts": len(api_sys),
            "page_outcomes": dict(lit_outcomes), "excluded_by_source": dict(lit_excl.most_common()),
            "dropped_shared_with_documentary": lit_shared,
            "eligible_pages": len(lit_eligible),
            "eligible_manuscripts": len({p.sys_num for p in lit_eligible}),
            "eligible_multi_column": sum(p.multi for p in lit_eligible),
            "target": lit_target, "selected": len(lit_selected),
            "selected_multi_column": sum(p.multi for p in lit_selected),
            "selected_manuscripts": len({p.sys_num for p in lit_selected}), "strata": lit_strata,
            "eligible_image_sources": dict(collections.Counter(p.image_source[0] for p in lit_eligible)),
            "selected_image_sources": dict(collections.Counter(p.image_source[0] for p in lit_selected)),
            "selected_genres": dict(collections.Counter(p.genre for p in lit_selected)),
            "selected_canonical_overlap_train": sum(p.canonical_overlap_train for p in lit_selected),
            "selected_benchmark_text_overlap": sum(p.benchmark_text_overlap for p in lit_selected)},
    }
    if not dry_run:
        report["images"] = write_outputs(out_dir, doc_selected, lit_selected, seed)
        doc_failed = set(report["images"]["documentary_failed"])
        lit_failed = set(report["images"]["literary_failed"])
        ids, pgpids, sys_nums = registry_entries([p for p in doc_selected if p.doc_id not in doc_failed],
                                                 [p for p in lit_selected if p.doc_id not in lit_failed], merged)
        reg.write_registry(out_dir / f"{NAME}.json", NAME, ids, pgpids, report["built"], sys_nums=sys_nums)
        report["registry"] = {"path": str(out_dir / f"{NAME}.json"), "ids": len(ids), "pgpids": len(pgpids),
                              "sys_nums": len(sys_nums)}
    report["seconds"] = round(time.time() - t0, 1)
    path = report_path or (None if dry_run else out_dir / "build_report.json")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def write_outputs(out_dir: Path, doc_pages: Sequence[DocPage], lit_pages: Sequence[LitPage], seed: int) -> Dict[str, Any]:
    """Download / extract the images and write both specs and the script-tag sidecar.

    A page whose image cannot be obtained (or no longer hashes to what the readers saw) is dropped
    from its spec and listed in the report.

    :param out_dir: Output directory.
    :type out_dir: Path
    :param doc_pages: Selected documentary pages.
    :type doc_pages: Sequence[DocPage]
    :param lit_pages: Selected literary pages.
    :type lit_pages: Sequence[LitPage]
    :param seed: Sampling seed (recorded).
    :type seed: int
    :return: Image counts and failures.
    :rtype: Dict[str, Any]
    """
    session = requests.Session()
    doc_dir, lit_dir = out_dir / "documentary", out_dir / "literary"
    doc_docs, doc_failed = [], {}
    for p in doc_pages:
        dest = doc_dir / "images" / f"{p.doc_id}.jpg"
        meta = fetch_bucket_image(p.cand.urls[p.image_index], dest, session, p.sha256)
        if not meta["ok"]:
            doc_failed[p.doc_id] = meta["reason"]
            continue
        entry = documentary_doc(p)
        entry.update(image_width=meta["width"], image_height=meta["height"], image_reencoded=meta["reencoded"])
        doc_docs.append(entry)
    criteria = ("PGP digital edition (>= 50 chars); one merged record, no join / multi-fragment / shared fragment; "
                "outside every training, Kraken-training, benchmark and GT-scored analysis set and their joins "
                "(build_report.json); side on the image by the edition builder's 4-gram presence test without "
                f"line anchor (> null max of 50 frozen null pages, >= 3 hits, present-block coverage >= {SIDE_MIN_COVERAGE}, "
                f"exactly one side); GT = that side's edition blocks, builder notation gate, >= {DOC_MIN_LETTERS} "
                f"Hebrew letters, >= {DOC_MIN_LINES} lines, Hebrew script; one page per document; stratified by "
                f"language x type, seed {seed}")
    spec = {"version": f"{NAME}_documentary", "frozen_date": date.today().isoformat(), "criteria": criteria,
            "gap_rendering": bpe.GAP_TOKEN, "n": len(doc_docs), "docs": doc_docs, "replacements": [],
            "derived_from": "pgp_raw/data/footnotes.csv (Digital Edition)", "source": NAME, "seed": seed,
            "rule": "SEALED: never tune, select or debug on this set (README.md)"}
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / DOC_SPEC_NAME).write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    tags = {d["doc_id"]: {"bucket": bucket_for(d["languages_primary"], ""), "languages_primary": d["languages_primary"],
                          "languages_secondary": "", "matched": True} for d in doc_docs}
    (doc_dir / DOC_TAGS_NAME).write_text(json.dumps(tags, ensure_ascii=False, indent=1), encoding="utf-8")

    lit_docs, lit_failed = [], {}
    images = (lit_dir / "images").resolve()
    for p in lit_pages:
        meta = materialise_ktiv_image(p.image_source, images / f"{p.sys_num}_{p.fl}.jpg", session)
        if not meta["ok"]:
            lit_failed[p.doc_id] = meta["reason"]
            continue
        lit_docs.append(literary_doc(p, images))
    lit_spec = {"benchmark": f"{NAME}_literary", "docs": lit_docs, "gt_rev": "1.1",
                "gt_rev_note": "GT from the repaired ktiv_layout.reconstruct_page (as religious-140 rev 1.1)",
                "source": NAME, "seed": seed, "rule": "SEALED: never tune, select or debug on this set (README.md)"}
    (lit_dir / LIT_SPEC_NAME).write_text(json.dumps(lit_spec, ensure_ascii=False, indent=1), encoding="utf-8")
    stale = prune_stale_images(doc_dir / "images", {f"{d['doc_id']}.jpg" for d in doc_docs})
    stale += prune_stale_images(images, {Path(d["image"]).name for d in lit_docs})
    return {"documentary_written": len(doc_docs), "documentary_failed": doc_failed,
            "literary_written": len(lit_docs), "literary_failed": lit_failed, "stale_images_removed": stale}


def prune_stale_images(images_dir: Path, keep: Set[str]) -> List[str]:
    """Remove images an earlier build of this directory wrote that the current spec no longer lists.

    :param images_dir: The builder's own image folder.
    :type images_dir: Path
    :param keep: File names the spec references.
    :type keep: Set[str]
    :return: Removed file names.
    :rtype: List[str]
    """
    removed = []
    for p in sorted(images_dir.glob("*.jpg")) if images_dir.is_dir() else []:
        if p.name not in keep:
            p.unlink()
            removed.append(p.name)
    return removed


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="select only: no images, specs or registry")
    ap.add_argument("--report", type=Path, default=None, help="build report path (default out-dir/build_report.json)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--doc-target", type=int, default=DOC_TARGET)
    ap.add_argument("--lit-target", type=int, default=LIT_TARGET)
    a = ap.parse_args()
    report = build(a.out_dir, a.dry_run, a.report, a.seed, a.doc_target, a.lit_target)
    summary = {k: report[k] for k in ("documentary", "literary")}
    for half in summary.values():
        half.pop("strata", None)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
