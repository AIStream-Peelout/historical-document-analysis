# File name: exposure.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Reader-training exposure of every machine read (text27b, fix B2 of the v0_2 plan).

A read row's input is a page transcription written by one of our VLM readers. When that reader was
trained on the same page (or the same document, or the same fragment), the read is easier than a
fresh read and any eval built on it is optimistic. This module generalises the 2026-10-18 probe
script (``a50_training_overlap.py``, which covered the 129 probe documents and the ``a50`` reader)
to every read of the parallel corpus and of the v0_1 read rows, for both readers.

Training exposure, from local copies only (nothing is downloaded):

* ``v22a_pilot`` -- the v22a fine-tune's rows (``genizah_v22_pilot`` at the pinned revision
  :data:`PILOT_REVISION`, local rows parquet). Row stems name a PGP document and image index
  (``pgp_<pgpid>_<page>_...``, ``pgpqa_...``), a canonical id and image index (``dg_<id>__<page>_...``)
  or a KTIV manuscript page (``ktiv_<sys_num>_FL<id>_...``). The source manifests map stems to image
  URLs, image sha256 and canonical ids.
* ``clean_v2`` / ``clean_v1`` -- the documentary sets of v21b and of its warm-start lineage
  (``save_to_disk`` datasets; stems are canonical ids, one embedded image per document whose sha256
  is compared with the read page's ``image_sha256``).
* ``ktiv_v1`` ... ``ktiv_v4`` -- the KTIV page directories of each KTIV dataset build
  (``<sys_num>/FL<id>*.jpg``): train and val manuscripts, i.e. a superset of what was trained.

Which mixtures each reader saw is :data:`READER_MIXTURES` (a50 = blend of v21b-1200 and v22a-1300,
and v22a warm-starts from v21b-1200, so a50 inherits all of v21b's exposure).

Levels (strongest first, :data:`LEVELS` = ``rows.EXPOSURE_LEVELS``):

* ``page`` -- the read page itself was a training image (same image URL, same image sha256, or the
  same KTIV ``(sys_num, FL id)`` page).
* ``pgpid`` -- the same PGP document had training rows (on another page).
* ``fragment`` -- the same canonical id (shelfmark) had training rows, possibly for another PGP
  document on it, or its KTIV manuscript (system number) is in a KTIV build.
* ``none``.

Outputs (``paths.out_dir``): ``exposure.jsonl`` -- one record per read page
``{pgpid, read_model, page_key, image_url, level, read_level, routes, read_source}`` (``level`` is
the page's own level, ``read_level`` the strongest over the read) -- and ``exposure_report.json``
(exposed counts per level on the v0_1 read rows, CER distributions exposed vs unexposed: the proxy
test that memorised pages read better, a within-document paired comparison of the two readers, and
agreement with the 10-18 probe result).

Public API:

* :data:`LEVELS`, :data:`LEVEL_RANK`, :data:`READ_CONDITIONS`, :data:`READER_MIXTURES`,
  :data:`PILOT_REVISION`.
* :class:`ExposureSources` (``from_paths``), :class:`TrainingIndex`, :func:`build_training_index`
  and the per-source loaders :func:`add_pilot`, :func:`add_clean`, :func:`add_ktiv_dirs`.
* :class:`Read`, :func:`load_read_pages`, :func:`collect_reads`, :func:`merged_sys_nums`.
* :func:`classify_read`, :func:`build_exposure`.
* :func:`exposure_lookup`, :func:`tag_row`, :func:`tag_rows`, :func:`near_duplicate`.
* :func:`cer_summary`, :func:`report_rows`, :func:`report_reads`, :func:`report_paired`,
  :func:`compare_prototype`.
* CLI ``python -m src.finetuning.text27b.exposure`` (see :func:`main`).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from src.finetuning.text27b import paths as _paths
from src.finetuning.text27b import rows as _rows

LOG = logging.getLogger(__name__)

#: Exposure levels, strongest first (``page``, ``pgpid``, ``fragment``, ``none``).
LEVELS: Tuple[str, ...] = _rows.EXPOSURE_LEVELS
#: Level → rank (``none`` = 0, ``page`` = 3).
LEVEL_RANK: Dict[str, int] = {lvl: len(LEVELS) - 1 - i for i, lvl in enumerate(LEVELS)}
#: Row conditions whose input is a machine read (tagged by :func:`tag_rows`).
READ_CONDITIONS: Tuple[str, ...] = ("read", "read_hard", "read_partial")
#: Hub revision of the v22a pilot rows (pinned in the v22a notebook); recorded in the report.
PILOT_REVISION = "c8c15251"
#: KTIV build mixtures in build order.
KTIV_MIXTURES: Tuple[str, ...] = ("ktiv_v1", "ktiv_v2", "ktiv_v3", "ktiv_v4")
#: Training mixtures each reader saw, directly or through its warm-start lineage.
READER_MIXTURES: Dict[str, Tuple[str, ...]] = {
    "a50": ("v22a_pilot", "clean_v2", "clean_v1") + KTIV_MIXTURES,
    "v21b": ("clean_v2", "clean_v1") + KTIV_MIXTURES[:3],
}
#: What each mixture stands for (copied into the report).
MIXTURE_ROLES: Dict[str, str] = {
    "v22a_pilot": "direct: v22a-1300 trained on these rows (a50 only)",
    "clean_v2": "direct: v21b-1200 documentary set",
    "clean_v1": "lineage: v1.7 warm-start chain documentary set",
    "ktiv_v1": "KTIV build pages (train and val manuscripts: superset)",
    "ktiv_v2": "KTIV build pages (train and val manuscripts: superset)",
    "ktiv_v3": "KTIV build pages (train and val manuscripts: superset; v21b's KTIV set)",
    "ktiv_v4": "KTIV build pages (train and val manuscripts: superset; source of the pilot's KTIV rows, a50 only)",
}

#: Reader-training sources relative to the NAS datasets root (``paths.json`` key ``nas_datasets``,
#: default: the parent directory of ``book_editions_nas``).
NAS_SOURCES: Dict[str, str] = {
    "pilot_train": "genizah_v22_pilot/rows/train.parquet",
    "pilot_val": "genizah_v22_pilot/rows/val.parquet",
    "manifest:pgp_editions": "pgp_editions_v1/manifest.jsonl",
    "manifest:pgp_qa": "pgp_qa_v1/manifest.jsonl",
    "manifest:documentary_grounding": "documentary_grounding_v1/manifest.jsonl",
    "ktiv:ktiv_v1": "ktiv_dataset_v1/images",
    "ktiv:ktiv_v2": "ktiv_dataset_v2_images",
    "ktiv:ktiv_v3": "ktiv_dataset_v3_images",
    "ktiv:ktiv_v4": "ktiv_dataset_v4_images",
}
#: Repository-relative ``save_to_disk`` datasets.
REPO_SOURCES: Dict[str, str] = {
    "clean:clean_v2": "src/datasets/processed/qwen_hebrew_genizah_v2",
    "clean:clean_v1": "src/datasets/processed/qwen_hebrew_genizah",
}
#: Reads files per reader, relative to ``paths.ai_reads``.
READ_FILES: Dict[str, str] = {
    "a50": "ai_reads_qwen3-vl-8b-heb-v22i-a50.jsonl",
    "v21b": "ai_reads_qwen3-vl-8b-heb-v21b-step1200.jsonl",
}

_PGP_STEM = re.compile(r"^(?:vqa_)?(?:pgp|pgpqa)_(\d+)_(\d+)_")
_DG_STEM = re.compile(r"^dg_(.+?)__(\d+)_")
_KTIV_STEM = re.compile(r"^ktiv_(99\d{10,16})_(FL\d+)")
_SYS = re.compile(r"(99\d{10,16})")
_KTIV_URL = re.compile(r"/KTIV/(99\d{10,16})/(?:[^/]*?_)?(FL\d+)\.[A-Za-z]+$")
_FL_FILE = re.compile(r"^(FL\d+)(?:[._]|$)")
_CID_HEAD = re.compile(rb'"canonical_id":\s*"([^"]+)"')


# ------------------------------------------------------------------------------------ helpers
def id_key(cid: str) -> str:
    """Normalised canonical-id key (lower-case alphanumerics, as the 10-18 probe and decontam use).

    :param cid: Canonical id or dataset stem.
    :returns: The key (``"Cambridge_CUL_T_S_13J2_5"`` → ``"cambridgecults13j25"``).
    """
    return re.sub(r"[^0-9a-z]", "", (cid or "").lower())


def parse_pilot_stem(stem: str) -> Optional[Tuple[str, str, Optional[int]]]:
    """Classify a v22a pilot row stem.

    :param stem: Row stem (``pgp_446_0_page``, ``pgpqa_499_0_qa_date_0``, ``dg_<cid>__0_loc2``,
        ``ktiv_<sys_num>_FL<id>_sec0``).
    :returns: ``("pgpid", pgpid, page)``, ``("cid", canonical_id, page)``, ``("ktiv", "<sys>/<FL>", None)``
        or None for an unknown stem.
    """
    m = _PGP_STEM.match(stem)
    if m:
        return "pgpid", m.group(1), int(m.group(2))
    m = _DG_STEM.match(stem)
    if m:
        return "cid", m.group(1), int(m.group(2))
    m = _KTIV_STEM.match(stem)
    if m:
        return "ktiv", f"{m.group(1)}/{m.group(2)}", None
    return None


def ktiv_page_of_url(url: Optional[str]) -> Optional[Tuple[str, str]]:
    """KTIV ``(sys_num, FL id)`` of a KTIV-hosted page image URL.

    :param url: Image URL (``.../KTIV/<sys_num>/0002_FL168307758.jpg``).
    :returns: ``(sys_num, "FL168307758")`` or None when the URL is not a KTIV page.
    """
    if not url:
        return None
    m = _KTIV_URL.search(url)
    return (m.group(1), m.group(2)) if m else None


def page_key(doc_id: str, image_index: int) -> str:
    """Key of one read page.

    :param doc_id: Canonical id of the imaged fragment.
    :param image_index: Image index within the record.
    :returns: ``"<doc_id>:<image_index>"``.
    """
    return f"{doc_id}:{int(image_index)}"


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Objects of a JSONL file (blank lines skipped).

    :param path: File.
    :returns: Iterator of dicts.
    """
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                yield orjson.loads(line)


def _write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (temporary file in the same directory, then rename).

    :param path: Destination.
    :param obj: JSON-serialisable object.
    :returns: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, sort_keys=False)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------ sources
@dataclass
class ExposureSources:
    """Locations of the reader-training sources, the reads and the corpus maps."""

    pilot_train: Optional[Path]
    pilot_val: Optional[Path]
    manifests: Dict[str, Path]
    clean: Dict[str, Path]
    ktiv_dirs: Dict[str, Path]
    read_files: Dict[str, Path]
    pgp_parallel: Path
    merged_pgpid_map: Optional[Path]
    merged_shelfmarks: Optional[Path]

    @classmethod
    def from_paths(cls, p: Optional[_paths.Paths] = None,
                   overrides: Optional[Mapping[str, Union[str, Path]]] = None) -> "ExposureSources":
        """Resolve the default sources from the build configuration.

        NAS-relative entries resolve against ``paths.get("nas_datasets")`` or, when that key is
        absent, the parent directory of ``paths.book_editions_nas``. A ``paths.json`` key
        ``exposure_sources`` (a dict with the keys of :data:`NAS_SOURCES` / :data:`REPO_SOURCES`, or
        ``read:<model>``) and ``overrides`` replace single entries with absolute paths.

        :param p: Build paths (default :func:`paths.load_paths`).
        :param overrides: Extra replacements (same keys as ``exposure_sources``).
        :returns: The resolved sources.
        """
        p = p or _paths.load_paths()
        nas = p.get("nas_datasets") or p.book_editions_nas.parent
        nas = Path(nas)
        table: Dict[str, Path] = {k: nas / v for k, v in NAS_SOURCES.items()}
        table.update({k: p.repo / v for k, v in REPO_SOURCES.items()})
        table.update({f"read:{m}": p.ai_reads / f for m, f in READ_FILES.items()})
        for src in (p.get("exposure_sources") or {}, overrides or {}):
            for k, v in dict(src).items():
                table[k] = Path(os.path.expanduser(str(v)))
        return cls(pilot_train=table.get("pilot_train"), pilot_val=table.get("pilot_val"),
                   manifests={k.split(":", 1)[1]: v for k, v in table.items() if k.startswith("manifest:")},
                   clean={k.split(":", 1)[1]: v for k, v in table.items() if k.startswith("clean:")},
                   ktiv_dirs={k.split(":", 1)[1]: v for k, v in table.items() if k.startswith("ktiv:")},
                   read_files={k.split(":", 1)[1]: v for k, v in table.items() if k.startswith("read:")},
                   pgp_parallel=p.pgp_parallel, merged_pgpid_map=p.merged_pgpid_map,
                   merged_shelfmarks=p.merged_shelfmarks)

    def missing(self) -> List[str]:
        """Names of configured sources that do not exist on disk.

        :returns: Source names (``pilot_train``, ``manifest:pgp_qa``, ``ktiv:ktiv_v4`` ...).
        """
        named: List[Tuple[str, Optional[Path]]] = [("pilot_train", self.pilot_train), ("pilot_val", self.pilot_val),
                                                   ("pgp_parallel", self.pgp_parallel),
                                                   ("merged_pgpid_map", self.merged_pgpid_map),
                                                   ("merged_shelfmarks", self.merged_shelfmarks)]
        for prefix, table in (("manifest", self.manifests), ("clean", self.clean), ("ktiv", self.ktiv_dirs),
                              ("read", self.read_files)):
            named += [(f"{prefix}:{k}", v) for k, v in table.items()]
        return [name for name, path in named if path is None or not Path(path).exists()]

    def describe(self) -> Dict[str, Any]:
        """Source names, locations and existence (for the report).

        :returns: ``{name: {"path": str, "exists": bool}}``.
        """
        out: Dict[str, Any] = {}
        missing = set(self.missing())
        for name, path in [("pilot_train", self.pilot_train), ("pilot_val", self.pilot_val),
                           ("pgp_parallel", self.pgp_parallel), ("merged_pgpid_map", self.merged_pgpid_map),
                           ("merged_shelfmarks", self.merged_shelfmarks)] + \
                [(f"manifest:{k}", v) for k, v in self.manifests.items()] + \
                [(f"clean:{k}", v) for k, v in self.clean.items()] + \
                [(f"ktiv:{k}", v) for k, v in self.ktiv_dirs.items()] + \
                [(f"read:{k}", v) for k, v in self.read_files.items()]:
            out[name] = {"path": str(path) if path else None, "exists": name not in missing}
        return out


# ----------------------------------------------------------------------------- training index
_KINDS: Tuple[str, ...] = ("url", "sha256", "pgpid", "fragment", "ktiv_sys", "ktiv_page")


@dataclass
class TrainingIndex:
    """What the reader mixtures trained on, by match kind, each key mapped to the mixtures that hold it.

    Kinds: ``url`` (page image URL), ``sha256`` (page image bytes), ``ktiv_page`` (``"<sys>/<FL>"``)
    give page-level evidence; ``pgpid`` gives document-level; ``fragment`` (:func:`id_key` of a
    canonical id) and ``ktiv_sys`` give fragment-level evidence.
    """

    url: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    sha256: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    pgpid: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    fragment: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    ktiv_sys: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    ktiv_page: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    val_pgpids: Set[str] = field(default_factory=set)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add(self, kind: str, key: Optional[str], mixture: str) -> None:
        """Record that ``mixture`` trained on ``key`` of ``kind``.

        :param kind: One of ``url``, ``sha256``, ``pgpid``, ``fragment``, ``ktiv_sys``, ``ktiv_page``.
        :param key: The key (ignored when empty).
        :param mixture: Mixture name (``v22a_pilot``, ``clean_v2`` ...).
        :returns: None.
        :raises KeyError: For an unknown kind.
        """
        if kind not in _KINDS:
            raise KeyError(kind)
        if key:
            getattr(self, kind)[str(key)].add(mixture)

    def hits(self, kind: str, key: Optional[str], mixtures: Iterable[str]) -> Set[str]:
        """Mixtures among ``mixtures`` that trained on ``key`` of ``kind``.

        :param kind: Match kind (see :meth:`add`).
        :param key: The key.
        :param mixtures: The reader's mixtures (:data:`READER_MIXTURES`).
        :returns: The matching mixture names (empty when none).
        """
        if not key:
            return set()
        table = getattr(self, kind)
        found = table.get(str(key)) if isinstance(table, dict) else None
        return set(found or ()) & set(mixtures)

    def sizes(self) -> Dict[str, Dict[str, int]]:
        """Number of keys per kind and mixture.

        :returns: ``{kind: {mixture: n_keys}}``.
        """
        out: Dict[str, Dict[str, int]] = {}
        for kind in _KINDS:
            c: collections.Counter = collections.Counter()
            for mixes in getattr(self, kind).values():
                c.update(mixes)
            out[kind] = dict(sorted(c.items()))
        return out


def _manifest_maps(manifests: Mapping[str, Path]) -> Dict[str, Dict[Any, Any]]:
    """Stem / page → image URL, sha256 and canonical id from the pilot's source manifests.

    :param manifests: ``{component: manifest.jsonl}``; missing files are skipped.
    :returns: ``{"url_of_stem", "sha_of_stem", "url_of_page", "sha_of_page", "cid_of_pgp"}`` where page
        keys are ``("pgpid", pgpid, index)`` or ``("cid", canonical_id, index)`` and ``cid_of_pgp`` maps
        ``(pgpid, index)`` to the canonical id.
    """
    out: Dict[str, Dict[Any, Any]] = {k: {} for k in ("url_of_stem", "sha_of_stem", "url_of_page", "sha_of_page",
                                                      "cid_of_pgp")}
    for comp, path in manifests.items():
        if not Path(path).exists():
            LOG.warning("manifest %s missing: %s", comp, path)
            continue
        for rec in _iter_jsonl(Path(path)):
            url, sha = rec.get("image_url"), rec.get("image_sha256")
            idx = int(rec.get("image_index") or 0)
            stems = ([rec["stem"]] if rec.get("stem") else []) + list(rec.get("rows") or [])
            for st in stems:
                if url:
                    out["url_of_stem"][st] = url
                if sha:
                    out["sha_of_stem"][st] = sha
            cid = rec.get("canonical_id") or rec.get("doc_id")
            keys = []
            if rec.get("pgpid") is not None:
                keys.append(("pgpid", str(rec["pgpid"]), idx))
                if rec.get("canonical_id"):
                    out["cid_of_pgp"][(str(rec["pgpid"]), idx)] = rec["canonical_id"]
            if cid:
                keys.append(("cid", cid, idx))
            for k in keys:
                if url:
                    out["url_of_page"].setdefault(k, url)
                if sha:
                    out["sha_of_page"].setdefault(k, sha)
    return out


def add_pilot(index: TrainingIndex, train_parquet: Path, manifests: Mapping[str, Path],
              mixture: str = "v22a_pilot") -> Dict[str, Any]:
    """Add the v22a pilot's training rows to the index (port of ``pilot_exposure``).

    PGP stems give ``pgpid`` keys, their page URL / sha256 (manifests) and the fragment's canonical
    id; ``dg_`` stems give ``fragment`` keys and page URLs; ``ktiv_`` stems give ``ktiv_page`` and
    ``ktiv_sys`` keys.

    :param index: Index to fill.
    :param train_parquet: Pilot ``rows/train.parquet`` (columns ``stem``, ``source``).
    :param manifests: Source manifests (see :func:`_manifest_maps`).
    :param mixture: Mixture name recorded on every key.
    :returns: Stats ``{rows, rows_by_source, pgpids, page_urls, page_sha256, unmapped_pages, unknown_stems}``.
    """
    maps = _manifest_maps(manifests)
    tab = pq.read_table(train_parquet, columns=["stem", "source"]).to_pylist()
    by_source: collections.Counter = collections.Counter()
    unmapped = unknown = 0
    pgpids: Set[str] = set()
    urls: Set[str] = set()
    shas: Set[str] = set()
    for row in tab:
        stem, src = row["stem"], row.get("source") or ""
        by_source[src] += 1
        parsed = parse_pilot_stem(stem)
        if parsed is None:
            m = _SYS.search(stem)
            if src.startswith("ktiv") and m:
                index.add("ktiv_sys", m.group(1), mixture)
            else:
                unknown += 1
            continue
        kind, ident, page = parsed
        if kind == "ktiv":
            index.add("ktiv_page", ident, mixture)
            index.add("ktiv_sys", ident.split("/", 1)[0], mixture)
            continue
        if kind == "pgpid":
            index.add("pgpid", ident, mixture)
            pgpids.add(ident)
            cid = maps["cid_of_pgp"].get((ident, page))
            if cid:
                index.add("fragment", id_key(cid), mixture)
        else:
            index.add("fragment", id_key(ident), mixture)
        url = maps["url_of_stem"].get(stem) or maps["url_of_page"].get((kind, ident, page))
        sha = maps["sha_of_stem"].get(stem) or maps["sha_of_page"].get((kind, ident, page))
        if url:
            index.add("url", url, mixture)
            urls.add(url)
        if sha:
            index.add("sha256", sha, mixture)
            shas.add(sha)
        if not url and not sha:
            unmapped += 1
    return {"rows": len(tab), "rows_by_source": dict(by_source), "pgpids": len(pgpids), "page_urls": len(urls),
            "page_sha256": len(shas), "unmapped_pages": unmapped, "unknown_stems": unknown}


def pilot_val_pgpids(val_parquet: Optional[Path]) -> Set[str]:
    """PGP ids of the pilot's validation rows (seen only as eval loss, not trained).

    :param val_parquet: Pilot ``rows/val.parquet`` or None.
    :returns: pgpids (empty when the file is absent).
    """
    if not val_parquet or not Path(val_parquet).exists():
        return set()
    stems = pq.read_table(val_parquet, columns=["stem"]).column("stem").to_pylist()
    return {m.group(1) for s in stems for m in [_PGP_STEM.match(s)] if m}


def add_clean(index: TrainingIndex, path: Path, mixture: str) -> Dict[str, Any]:
    """Add a ``genizah_clean`` dataset's train split (stems = canonical ids, one embedded image each).

    :param index: Index to fill.
    :param path: ``save_to_disk`` directory with ``train/data-*.arrow``.
    :param mixture: Mixture name (``clean_v2`` / ``clean_v1``).
    :returns: Stats ``{train_docs, train_images}``.
    """
    docs = imgs = 0
    for f in sorted(glob.glob(str(Path(path) / "train" / "data-*.arrow"))):
        with pa.memory_map(f, "r") as src:
            reader = pa.ipc.open_stream(src)
            i_stem = reader.schema.get_field_index("stem")
            i_img = reader.schema.get_field_index("image")
            for batch in reader:
                stems = batch.column(i_stem).to_pylist()
                images = batch.column(i_img).to_pylist() if i_img >= 0 else [None] * len(stems)
                for stem, img in zip(stems, images):
                    index.add("fragment", id_key(stem), mixture)
                    docs += 1
                    if img and img.get("bytes"):
                        index.add("sha256", hashlib.sha256(img["bytes"]).hexdigest(), mixture)
                        imgs += 1
    return {"train_docs": docs, "train_images": imgs}


def add_ktiv_dirs(index: TrainingIndex, dirs: Mapping[str, Path], page_sys: Optional[Set[str]] = None
                  ) -> Dict[str, Any]:
    """Add KTIV build page directories (``<dir>/<sys_num>/FL<id>*.jpg``).

    Every manuscript directory gives a ``ktiv_sys`` key. Page files are listed only for the system
    numbers in ``page_sys`` (the ones some read page is hosted under), which keeps the NAS listing
    small; ``None`` lists every directory.

    :param index: Index to fill.
    :param dirs: ``{mixture: directory}``; missing directories are skipped.
    :param page_sys: System numbers whose page files to list, or None for all.
    :returns: Stats ``{mixture: {"manuscripts": n, "listed": n, "pages": n}}``.
    """
    out: Dict[str, Any] = {}
    for mixture, d in dirs.items():
        d = Path(d)
        if not d.exists():
            LOG.warning("KTIV dir %s missing: %s", mixture, d)
            out[mixture] = {"manuscripts": 0, "listed": 0, "pages": 0, "missing": True}
            continue
        n_ms = n_listed = n_pages = 0
        for entry in os.scandir(d):
            m = _SYS.fullmatch(entry.name)
            if not m:
                continue
            n_ms += 1
            index.add("ktiv_sys", entry.name, mixture)
            if page_sys is not None and entry.name not in page_sys:
                continue
            n_listed += 1
            fls = {mm.group(1) for name in os.listdir(entry.path) for mm in [_FL_FILE.match(name)] if mm}
            for fl in fls:
                index.add("ktiv_page", f"{entry.name}/{fl}", mixture)
            n_pages += len(fls)
        out[mixture] = {"manuscripts": n_ms, "listed": n_listed, "pages": n_pages}
    return out


def build_training_index(sources: ExposureSources, page_sys: Optional[Set[str]] = None,
                         ktiv_pages: bool = True) -> TrainingIndex:
    """Build the index of everything the reader mixtures trained on.

    :param sources: Source locations; absent sources are skipped with a warning (and named in
        ``index.stats["missing_sources"]``).
    :param page_sys: KTIV system numbers whose page files to list (see :func:`add_ktiv_dirs`).
    :param ktiv_pages: List KTIV page files at all (False → manuscript level only).
    :returns: The filled :class:`TrainingIndex`.
    """
    index = TrainingIndex()
    t0 = time.time()
    if sources.pilot_train and Path(sources.pilot_train).exists():
        index.stats["v22a_pilot"] = add_pilot(index, Path(sources.pilot_train), sources.manifests)
    else:
        LOG.warning("pilot rows missing: %s", sources.pilot_train)
    index.val_pgpids = pilot_val_pgpids(sources.pilot_val)
    for name, path in sources.clean.items():
        if Path(path).exists():
            index.stats[name] = add_clean(index, Path(path), name)
        else:
            LOG.warning("clean dataset %s missing: %s", name, path)
    index.stats["ktiv"] = add_ktiv_dirs(index, sources.ktiv_dirs,
                                        page_sys=(page_sys if ktiv_pages else set()))
    index.stats["missing_sources"] = [m for m in sources.missing() if not m.startswith("read:")]
    index.stats["seconds"] = round(time.time() - t0, 1)
    LOG.info("training index built in %.1fs: %s", time.time() - t0, index.sizes())
    return index


# ----------------------------------------------------------------------------------------- reads
@dataclass
class Read:
    """One reader's read of one PGP document (the pages it read and the CER the corpus records)."""

    pgpid: str
    model: str
    pages: List[Dict[str, Any]]
    cids: List[str]
    cer: Optional[float] = None
    coverage: Optional[float] = None
    source: str = "pgp_parallel"


def load_read_pages(path: Path, known_ids: Optional[Set[str]] = None,
                    url_to_id: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Page records of one reader's reads file (url, sha256 per canonical id and image index).

    Records whose ``doc_id`` is not in ``known_ids`` are attached to ``url_to_id[image_url]`` when
    that resolves (ids renamed between index versions, as the v0 builder does) and dropped otherwise.

    :param path: Reads JSONL (``doc_id``, ``image_index``, ``image_url``, ``image_sha256`` per line).
    :param known_ids: Canonical ids to keep (None keeps every record).
    :param url_to_id: Image URL → canonical id fallback.
    :returns: ``{"by_page": {(doc_id, idx): {url, sha256}}, "by_cid": {doc_id: [idx, ...]},
        "sha_of_url": {url: sha256}}``.
    """
    by_page: Dict[Tuple[str, int], Dict[str, Optional[str]]] = {}
    sha_of_url: Dict[str, str] = {}
    for rec in _iter_jsonl(Path(path)):
        doc_id, url, sha = rec.get("doc_id"), rec.get("image_url"), rec.get("image_sha256")
        if url and sha:
            sha_of_url[url] = sha
        if known_ids is not None and doc_id not in known_ids:
            doc_id = (url_to_id or {}).get(url or "")
            if not doc_id:
                continue
        idx = int(rec.get("image_index") or 0)
        by_page[(doc_id, idx)] = {"url": url, "sha256": sha}
    by_cid: Dict[str, List[int]] = collections.defaultdict(list)
    for doc_id, idx in sorted(by_page):
        by_cid[doc_id].append(idx)
    return {"by_page": by_page, "by_cid": dict(by_cid), "sha_of_url": sha_of_url}


def _merged_maps(path: Optional[Path]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """pgpid → canonical ids and image URL → canonical id from ``merged_pgpid_map.json``.

    :param path: The map file (``{cid: {"pgpids": [...], "image_urls": [...]}}``) or None.
    :returns: ``(cids_of_pgpid, url_to_id)`` (empty when the file is absent).
    """
    cids_of: Dict[str, List[str]] = collections.defaultdict(list)
    url_to_id: Dict[str, str] = {}
    if not path or not Path(path).exists():
        return {}, {}
    with open(path, "rb") as fh:
        mp = orjson.loads(fh.read())
    for cid, info in mp.items():
        for p in info.get("pgpids") or []:
            cids_of[str(p)].append(cid)
        for url in info.get("image_urls") or []:
            url_to_id.setdefault(url, cid)
    return dict(cids_of), url_to_id


def collect_reads(pgp_parallel: Path, read_pages: Mapping[str, Mapping[str, Any]],
                  cids_of_pgpid: Mapping[str, Sequence[str]],
                  wanted: Optional[Set[Tuple[str, str]]] = None, sample_docs: Optional[int] = None) -> List[Read]:
    """Every read of the parallel corpus plus the wanted ``(pgpid, model)`` reads it lacks.

    A parallel-corpus read keeps its own pages (``doc_id``, ``image_index``, ``image_url``; sha256
    filled from the reads file). A wanted read the corpus lacks (the v0 builder's read-only pool) is
    rebuilt from the reads file: every page of every canonical id of the document, deduplicated by
    sha256, as the builder's ``read_only_pool`` does.

    :param pgp_parallel: ``pgp_parallel`` JSONL (``pgpid``, ``merged_doc_ids``, ``reads``).
    :param read_pages: ``{model: load_read_pages(...)}``.
    :param cids_of_pgpid: pgpid → canonical ids (merged map).
    :param wanted: ``(pgpid, model)`` pairs that must be present (e.g. from v0_1 read rows).
    :param sample_docs: Keep only the first N parallel-corpus documents (development runs).
    :returns: The reads.
    """
    out: List[Read] = []
    have: Set[Tuple[str, str]] = set()
    for n, row in enumerate(_iter_jsonl(Path(pgp_parallel))):
        if sample_docs is not None and n >= sample_docs:
            break
        pid = str(row["pgpid"])
        cids = sorted(set(row.get("merged_doc_ids") or []) | set(cids_of_pgpid.get(pid, [])))
        for rd in row.get("reads") or []:
            model = rd["model"]
            rp = read_pages.get(model, {})
            pages = []
            for pg in rd.get("pages") or []:
                idx = int(pg.get("image_index") or 0)
                url = pg.get("image_url")
                sha = (rp.get("by_page", {}).get((pg.get("doc_id"), idx)) or {}).get("sha256") \
                    or rp.get("sha_of_url", {}).get(url or "")
                pages.append({"doc_id": pg.get("doc_id"), "image_index": idx, "image_url": url, "image_sha256": sha})
            out.append(Read(pid, model, pages, cids, rd.get("cer_vs_edition"), rd.get("coverage"), "pgp_parallel"))
            have.add((pid, model))
    for pid, model in sorted(wanted or set()):
        if (pid, model) in have:
            continue
        rp = read_pages.get(model, {})
        cids = sorted(set(cids_of_pgpid.get(pid, [])))
        pages, seen = [], set()
        for cid in cids:
            for idx in rp.get("by_cid", {}).get(cid, []):
                rec = rp["by_page"][(cid, idx)]
                if rec.get("sha256") and rec["sha256"] in seen:
                    continue
                seen.add(rec.get("sha256"))
                pages.append({"doc_id": cid, "image_index": idx, "image_url": rec.get("url"),
                              "image_sha256": rec.get("sha256")})
        out.append(Read(pid, model, pages, cids, None, None, "reads_file"))
    return out


def merged_sys_nums(path: Optional[Path], cids: Set[str]) -> Dict[str, Set[str]]:
    """KTIV system numbers of merged records (one pass over the merged corpus).

    :param path: ``merged_shelfmarks.jsonl`` (``images.ktiv.sys_num`` / ``sources.ktiv.sys_num``), or None.
    :param cids: Canonical ids wanted.
    :returns: ``{canonical_id: {sys_num, ...}}`` for the wanted ids that have one.
    """
    out: Dict[str, Set[str]] = {}
    if not path or not Path(path).exists():
        return out
    with open(path, "rb") as fh:
        for line in fh:
            m = _CID_HEAD.search(line[:400])
            if not m or m.group(1).decode("utf-8") not in cids or b'"ktiv"' not in line:
                continue
            rec = orjson.loads(line)
            nums: Set[str] = set()
            for node in ((rec.get("images") or {}).get("ktiv"), (rec.get("sources") or {}).get("ktiv")):
                if isinstance(node, dict) and node.get("sys_num"):
                    nums.add(str(node["sys_num"]))
            if nums:
                out[rec["canonical_id"]] = nums
    return out


# -------------------------------------------------------------------------------- classification
def _max_level(levels: Iterable[str]) -> str:
    """Strongest of several levels.

    :param levels: Level names.
    :returns: The strongest (``none`` for an empty input).
    """
    best = "none"
    for lvl in levels:
        if LEVEL_RANK[lvl] > LEVEL_RANK[best]:
            best = lvl
    return best


def classify_read(read: Read, index: TrainingIndex, sys_of_cid: Mapping[str, Set[str]],
                  mixtures: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Exposure records of one read: one per page (one with ``page_key=None`` for a page-less read).

    :param read: The read.
    :param index: Training index.
    :param sys_of_cid: Canonical id → KTIV system numbers (:func:`merged_sys_nums`).
    :param mixtures: The reader's mixtures (default ``READER_MIXTURES[read.model]``).
    :returns: Records ``{pgpid, read_model, page_key, image_url, level, read_level, routes, read_source}``;
        ``routes`` are ``"<mixture>:<kind>"`` strings behind the record's own level.
    :raises KeyError: For a reader without an entry in :data:`READER_MIXTURES` when ``mixtures`` is None.
    """
    mix = tuple(mixtures) if mixtures is not None else READER_MIXTURES[read.model]
    doc_routes: Set[str] = {f"{m}:pgpid" for m in index.hits("pgpid", read.pgpid, mix)}
    doc_level = "pgpid" if doc_routes else "none"
    frag_routes: Set[str] = set()
    sys_nums: Set[str] = set()
    for cid in read.cids:
        frag_routes |= {f"{m}:fragment" for m in index.hits("fragment", id_key(cid), mix)}
        sys_nums |= set(sys_of_cid.get(cid, ()))
    for pg in read.pages:
        kp = ktiv_page_of_url(pg.get("image_url"))
        if kp:
            sys_nums.add(kp[0])
    for s in sys_nums:
        frag_routes |= {f"{m}:ktiv_sys" for m in index.hits("ktiv_sys", s, mix)}
    if doc_level == "none" and frag_routes:
        doc_level = "fragment"
    doc_routes |= frag_routes if doc_level == "fragment" else set()
    page_hits: List[Set[str]] = []
    for pg in read.pages:
        hits = {f"{m}:url" for m in index.hits("url", pg.get("image_url"), mix)}
        hits |= {f"{m}:sha256" for m in index.hits("sha256", pg.get("image_sha256"), mix)}
        kp = ktiv_page_of_url(pg.get("image_url"))
        if kp:
            hits |= {f"{m}:ktiv_page" for m in index.hits("ktiv_page", f"{kp[0]}/{kp[1]}", mix)}
        page_hits.append(hits)
    read_level = "page" if any(page_hits) else doc_level
    if not read.pages:
        return [{"pgpid": read.pgpid, "read_model": read.model, "page_key": None, "image_url": None,
                 "level": doc_level, "read_level": doc_level, "routes": sorted(doc_routes),
                 "read_source": read.source}]
    out = []
    for pg, hits in zip(read.pages, page_hits):
        level = "page" if hits else doc_level
        out.append({"pgpid": read.pgpid, "read_model": read.model,
                    "page_key": page_key(pg.get("doc_id") or "", pg.get("image_index") or 0),
                    "image_url": pg.get("image_url"), "level": level, "read_level": read_level,
                    "routes": sorted(hits if hits else doc_routes), "read_source": read.source})
    return out


def build_exposure(reads: Sequence[Read], index: TrainingIndex, sys_of_cid: Mapping[str, Set[str]]
                   ) -> List[Dict[str, Any]]:
    """Exposure records of every read whose reader has known mixtures.

    :param reads: Reads (:func:`collect_reads`).
    :param index: Training index.
    :param sys_of_cid: Canonical id → KTIV system numbers.
    :returns: Records (see :func:`classify_read`); reads of unknown readers are skipped with a warning.
    """
    out: List[Dict[str, Any]] = []
    skipped: collections.Counter = collections.Counter()
    for rd in reads:
        if rd.model not in READER_MIXTURES:
            skipped[rd.model] += 1
            continue
        out.extend(classify_read(rd, index, sys_of_cid))
    if skipped:
        LOG.warning("reads of readers without a mixture list skipped: %s", dict(skipped))
    return out


# ------------------------------------------------------------------------------------- tagging
def exposure_lookup(exposure: Union[str, Path, Iterable[Mapping[str, Any]], Mapping[Tuple[str, str], str]]
                    ) -> Dict[Tuple[str, str], str]:
    """``(pgpid, read_model)`` → read-level exposure.

    :param exposure: ``exposure.jsonl`` path, an iterable of records, or an existing lookup.
    :returns: The lookup (strongest ``read_level`` per read).
    """
    if isinstance(exposure, Mapping):
        return {(str(k[0]), str(k[1])): v for k, v in exposure.items()}
    records = _iter_jsonl(Path(exposure)) if isinstance(exposure, (str, Path)) else exposure
    out: Dict[Tuple[str, str], str] = {}
    for rec in records:
        k = (str(rec["pgpid"]), str(rec["read_model"]))
        lvl = rec.get("read_level") or rec.get("level") or "none"
        out[k] = _max_level([out.get(k, "none"), lvl])
    return out


def tag_row(row: Dict[str, Any], lookup: Mapping[Tuple[str, str], str]) -> Optional[str]:
    """Set ``meta.reader_exposed`` and ``meta.exposure_level`` on one row (in place).

    Read-like rows (:data:`READ_CONDITIONS`) get the level of their ``(pgpid, read_model)`` read and
    ``reader_exposed = level != "none"``; a read missing from the lookup gets ``None`` for both.
    Rows whose input is not a machine read (edition, noised edition, re-scripted edition,
    literature) get ``reader_exposed = False`` and ``exposure_level = None`` (not applicable).

    :param row: Row with ``meta``.
    :param lookup: :func:`exposure_lookup` result.
    :returns: The level set, ``"n/a"`` for a non-read row, or None for a read missing from the lookup.
    """
    meta = row.setdefault("meta", {})
    if meta.get("condition") not in READ_CONDITIONS:
        meta["reader_exposed"] = False
        meta["exposure_level"] = None
        return "n/a"
    level = lookup.get((str(meta.get("pgpid")), str(meta.get("read_model"))))
    meta["exposure_level"] = level
    meta["reader_exposed"] = None if level is None else level != "none"
    return level


def tag_rows(rows: Iterable[Dict[str, Any]],
             exposure: Union[str, Path, Iterable[Mapping[str, Any]], Mapping[Tuple[str, str], str]],
             stats: Optional[collections.Counter] = None) -> List[Dict[str, Any]]:
    """Tag rows with reader exposure (in place; see :func:`tag_row`).

    :param rows: Rows (consumed once).
    :param exposure: ``exposure.jsonl`` path, records or a lookup.
    :param stats: Optional counter updated with ``level:<level>`` / ``level:n/a`` / ``level:missing``.
    :returns: The rows, as a list.
    """
    lookup = exposure_lookup(exposure)
    out = []
    for row in rows:
        lvl = tag_row(row, lookup)
        if stats is not None:
            stats[f"level:{lvl if lvl is not None else 'missing'}"] += 1
        out.append(row)
    return out


def near_duplicate(row: Mapping[str, Any], cer_max: float = 0.05) -> bool:
    """Exposed read row whose input is a near copy of the edition (plan B2: drop these from train).

    :param row: A tagged row.
    :param cer_max: CER below which an exposed read counts as a near duplicate.
    :returns: True when ``reader_exposed`` and ``input_cer < cer_max``.
    """
    meta = row.get("meta") or {}
    cer = meta.get("input_cer")
    return bool(meta.get("reader_exposed")) and cer is not None and float(cer) < cer_max


# ---------------------------------------------------------------------------------------- report
def cer_summary(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Distribution of CER values (None values are ignored).

    :param values: CERs.
    :returns: ``{n, mean, p10, p25, p50, p75, p90, share_lt_0_05, share_lt_0_10}`` (only ``n`` when empty).
    """
    arr = np.array([float(v) for v in values if v is not None], dtype=float)
    if not arr.size:
        return {"n": 0}
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    return {"n": int(arr.size), "mean": round(float(arr.mean()), 4),
            **{f"p{p}": round(float(v), 4) for p, v in zip((10, 25, 50, 75, 90), q)},
            "share_lt_0_05": round(float((arr < 0.05).mean()), 4), "share_lt_0_10": round(float((arr < 0.10).mean()), 4)}


def report_rows(metas: Iterable[Mapping[str, Any]], lookup: Mapping[Tuple[str, str], str],
                near_dup_cer: float = 0.05) -> Dict[str, Any]:
    """Exposure of read rows per split and reader: row and document counts per level, CER exposed vs not.

    :param metas: Row ``meta`` dicts (non-read rows are ignored).
    :param lookup: :func:`exposure_lookup` result.
    :param near_dup_cer: CER bound of :func:`near_duplicate`.
    :returns: ``{split: {model|"all": {rows, rows_by_level, exposed_rows, docs, exposed_docs, missing,
        near_duplicates, cer_exposed, cer_unexposed, cer_by_level}}}``.
    """
    groups: Dict[Tuple[str, str], List[Tuple[str, Optional[str], Optional[float]]]] = collections.defaultdict(list)
    for m in metas:
        if m.get("condition") not in READ_CONDITIONS:
            continue
        lvl = lookup.get((str(m.get("pgpid")), str(m.get("read_model"))))
        item = (str(m.get("pgpid")), lvl, m.get("input_cer"))
        for model in (str(m.get("read_model")), "all"):
            groups[(m.get("split") or "?", model)].append(item)
    out: Dict[str, Any] = collections.defaultdict(dict)
    for (split, model), items in sorted(groups.items()):
        by_level = collections.Counter(lvl or "missing" for _, lvl, _ in items)
        exposed = [it for it in items if it[1] not in (None, "none")]
        unexposed = [it for it in items if it[1] == "none"]
        out[split][model] = {
            "rows": len(items), "rows_by_level": dict(sorted(by_level.items())), "exposed_rows": len(exposed),
            "docs": len({p for p, _, _ in items}), "exposed_docs": len({p for p, _, _ in exposed}),
            "missing": by_level.get("missing", 0),
            "near_duplicates": sum(1 for _, _, c in exposed if c is not None and c < near_dup_cer),
            "cer_exposed": cer_summary([c for _, _, c in exposed]),
            "cer_unexposed": cer_summary([c for _, _, c in unexposed]),
            "cer_by_level": {lvl: cer_summary([c for _, lv, c in items if lv == lvl]) for lvl in LEVELS},
        }
    return dict(out)


def _in_gate(rd: Read, lo: float = 0.6, hi: float = 1.5) -> bool:
    """Whether a read's coverage is inside the read gate (so CER measures letters, not missing text).

    :param rd: The read.
    :param lo: Lower coverage bound.
    :param hi: Upper coverage bound.
    :returns: True inside the gate.
    """
    return rd.coverage is not None and lo <= rd.coverage <= hi


def report_reads(reads: Sequence[Read], lookup: Mapping[Tuple[str, str], str]) -> Dict[str, Any]:
    """Exposure and CER of every parallel-corpus read, per reader and level (all reads and in-gate only).

    :param reads: Reads with ``cer`` / ``coverage`` (reads-file reads without a CER count only).
    :param lookup: :func:`exposure_lookup` result.
    :returns: ``{model: {reads, docs, reads_by_level, exposed_docs, cer_all: {...}, cer_in_gate: {...}}}``.
    """
    out: Dict[str, Any] = {}
    by_model: Dict[str, List[Read]] = collections.defaultdict(list)
    for rd in reads:
        by_model[rd.model].append(rd)
    for model, rds in sorted(by_model.items()):
        lv = {id(r): lookup.get((r.pgpid, r.model), "missing") for r in rds}
        by_level = collections.Counter(lv.values())
        entry: Dict[str, Any] = {"reads": len(rds), "docs": len({r.pgpid for r in rds}),
                                 "reads_by_level": dict(sorted(by_level.items())),
                                 "exposed_docs": len({r.pgpid for r in rds if lv[id(r)] not in ("none", "missing")})}
        for name, keep in (("cer_all", lambda r: True), ("cer_in_gate", _in_gate)):
            sub = [r for r in rds if keep(r) and r.cer is not None]
            entry[name] = {"exposed": cer_summary([r.cer for r in sub if lv[id(r)] not in ("none", "missing")]),
                           "unexposed": cer_summary([r.cer for r in sub if lv[id(r)] == "none"]),
                           **{lvl: cer_summary([r.cer for r in sub if lv[id(r)] == lvl]) for lvl in LEVELS}}
        out[model] = entry
    return out


def bootstrap_mean_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 3407
                        ) -> Dict[str, Any]:
    """Difference of means ``mean(a) - mean(b)`` with a percentile-bootstrap 95 % interval.

    The two samples are resampled independently (they hold different documents).

    :param a: First sample.
    :param b: Second sample.
    :param n_boot: Bootstrap replicates.
    :param seed: RNG seed.
    :returns: ``{n_a, n_b, mean_diff, median_diff, ci95: [lo, hi]}`` (``mean_diff`` None when a sample is empty).
    """
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not xa.size or not xb.size:
        return {"n_a": int(xa.size), "n_b": int(xb.size), "mean_diff": None, "median_diff": None, "ci95": None}
    rng = np.random.default_rng(seed)
    boots = (rng.choice(xa, size=(n_boot, xa.size)).mean(axis=1) - rng.choice(xb, size=(n_boot, xb.size)).mean(axis=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_a": int(xa.size), "n_b": int(xb.size), "mean_diff": round(float(xa.mean() - xb.mean()), 4),
            "median_diff": round(float(np.median(xa) - np.median(xb)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def report_paired(reads: Sequence[Read], lookup: Mapping[Tuple[str, str], str], model_a: str = "a50",
                  model_b: str = "v21b") -> Dict[str, Any]:
    """Within-document comparison of two readers' CER, grouped by which of them is exposed.

    Reader skill differs, so the informative number is the difference between groups: if exposure
    helps, ``cer_a - cer_b`` is lower where only ``model_a`` is exposed than where neither is
    (``did_vs_neither``: difference-in-differences with a bootstrap interval).

    :param reads: Reads with CER.
    :param lookup: :func:`exposure_lookup` result.
    :param model_a: First reader.
    :param model_b: Second reader.
    :returns: ``{"all": {group: cer_summary(diff), "did_vs_neither": {group: bootstrap_mean_diff}},
        "in_gate": {...}}`` with groups ``neither``, ``<a>_only``, ``<b>_only``, ``both``; differences
        are ``cer_a - cer_b`` (signed, so the ``share_lt_*`` fields are not meaningful here).
    """
    by_doc: Dict[str, Dict[str, Read]] = collections.defaultdict(dict)
    for rd in reads:
        if rd.cer is not None and rd.model in (model_a, model_b):
            by_doc[rd.pgpid][rd.model] = rd
    out: Dict[str, Any] = {}
    for name, keep in (("all", lambda a, b: True), ("in_gate", lambda a, b: _in_gate(a) and _in_gate(b))):
        groups: Dict[str, List[float]] = collections.defaultdict(list)
        for pid, pair in by_doc.items():
            if model_a not in pair or model_b not in pair:
                continue
            a, b = pair[model_a], pair[model_b]
            if not keep(a, b):
                continue
            ea = lookup.get((pid, model_a), "none") != "none"
            eb = lookup.get((pid, model_b), "none") != "none"
            g = "both" if ea and eb else f"{model_a}_only" if ea else f"{model_b}_only" if eb else "neither"
            groups[g].append(float(a.cer) - float(b.cer))
        out[name] = {g: cer_summary(v) for g, v in sorted(groups.items())}
        if groups.get("neither"):
            out[name]["did_vs_neither"] = {g: bootstrap_mean_diff(v, groups["neither"])
                                           for g, v in sorted(groups.items()) if g != "neither"}
    return out


def compare_prototype(lookup: Mapping[Tuple[str, str], str], overlap: Mapping[str, Any],
                      model: str = "a50") -> Dict[str, Any]:
    """Agreement with the 10-18 probe result (``a50_training_overlap.json``) on its documents.

    :param lookup: :func:`exposure_lookup` result.
    :param overlap: The parsed probe JSON (``{"docs": {pgpid: evidence}}``).
    :param model: Reader the probe covered.
    :returns: ``{docs, compared, exposed_agree, exposed_only_here, exposed_only_probe, page_agree,
        page_only_here, page_only_probe}`` (the ``only`` entries list pgpids).
    """
    docs = overlap.get("docs") or {}
    res: Dict[str, Any] = {"docs": len(docs), "compared": 0, "exposed_agree": 0, "exposed_only_here": [],
                           "exposed_only_probe": [], "page_agree": 0, "page_only_here": [], "page_only_probe": []}
    for pid, ev in sorted(docs.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
        if ev.get("read_model") != model:
            continue
        lvl = lookup.get((str(pid), model))
        if lvl is None:
            continue
        res["compared"] += 1
        here, there = lvl != "none", bool(ev.get("a50_trained"))
        if here == there:
            res["exposed_agree"] += 1
        else:
            res["exposed_only_here" if here else "exposed_only_probe"].append(str(pid))
        page_here = lvl == "page"
        page_there = bool(ev.get("v22a_page_url_hits") or ev.get("genizah_clean_v2_train_image_sha256_hits")
                          or ev.get("genizah_clean_v1_train_image_sha256_hits"))
        if page_here == page_there:
            res["page_agree"] += 1
        else:
            res["page_only_here" if page_here else "page_only_probe"].append(str(pid))
    return res


# ------------------------------------------------------------------------------------------- CLI
def _read_metas(files: Sequence[Path]) -> List[Dict[str, Any]]:
    """``meta`` dicts of the read-like rows of row files (split taken from the meta or the file name).

    :param files: Row JSONL files.
    :returns: Metas of rows whose condition is in :data:`READ_CONDITIONS`.
    """
    out = []
    for f in files:
        for row in _iter_jsonl(Path(f)):
            m = row.get("meta") or {}
            if m.get("condition") in READ_CONDITIONS:
                m.setdefault("split", Path(f).stem)
                out.append(m)
    return out


def run(sources: ExposureSources, row_files: Sequence[Path], out_dir: Path,
        sample_docs: Optional[int] = None, ktiv_pages: bool = True,
        prototype: Optional[Path] = None) -> Dict[str, Any]:
    """Build ``exposure.jsonl`` and ``exposure_report.json`` for every read.

    :param sources: Source locations.
    :param row_files: Row JSONL files whose read rows must be covered and are reported on.
    :param out_dir: Output directory.
    :param sample_docs: Limit the parallel corpus to its first N documents (development).
    :param ktiv_pages: List KTIV page files for page-level KTIV matches.
    :param prototype: ``a50_training_overlap.json`` to compare with, or None.
    :returns: The report dict (also written).
    """
    t0 = time.time()
    metas = _read_metas(row_files)
    wanted = {(str(m["pgpid"]), str(m["read_model"])) for m in metas if m.get("pgpid") and m.get("read_model")}
    cids_of, url_to_id = _merged_maps(sources.merged_pgpid_map)
    known = set(url_to_id.values()) | {c for cs in cids_of.values() for c in cs}
    read_pages = {m: load_read_pages(p, known_ids=known or None, url_to_id=url_to_id)
                  for m, p in sources.read_files.items() if Path(p).exists()}
    reads = collect_reads(sources.pgp_parallel, read_pages, cids_of, wanted=wanted, sample_docs=sample_docs)
    LOG.info("%d reads (%d wanted by rows) in %.1fs", len(reads), len(wanted), time.time() - t0)
    page_sys = {kp[0] for rd in reads for pg in rd.pages for kp in [ktiv_page_of_url(pg.get("image_url"))] if kp}
    index = build_training_index(sources, page_sys=page_sys, ktiv_pages=ktiv_pages)
    all_cids = {c for rd in reads for c in rd.cids}
    sys_of = merged_sys_nums(sources.merged_shelfmarks, all_cids)
    records = build_exposure(reads, index, sys_of)
    out_dir.mkdir(parents=True, exist_ok=True)
    _rows.write_jsonl(out_dir / "exposure.jsonl", records)
    os.chmod(out_dir / "exposure.jsonl", 0o644)
    lookup = exposure_lookup(records)
    report: Dict[str, Any] = {
        "build": _rows.BUILD,
        "build_args": {"row_files": [str(f) for f in row_files], "sample_docs": sample_docs, "ktiv_pages": ktiv_pages,
                       "pilot_revision": PILOT_REVISION, "reader_mixtures": {k: list(v) for k, v in READER_MIXTURES.items()},
                       "mixture_roles": MIXTURE_ROLES},
        "sources": sources.describe(),
        "training_index": {"stats": index.stats, "keys": index.sizes(), "pilot_val_pgpids": len(index.val_pgpids)},
        "reads": {"total": len(reads), "by_source": dict(collections.Counter(r.source for r in reads)),
                  "pages": sum(len(r.pages) for r in reads), "page_records": len(records),
                  "pages_with_sha256": sum(1 for r in reads for p in r.pages if p.get("image_sha256")),
                  "ktiv_hosted_pages": sum(1 for r in reads for p in r.pages if ktiv_page_of_url(p.get("image_url"))),
                  "reads_without_pages": sum(1 for r in reads if not r.pages),
                  "docs_with_ktiv_sys": len({r.pgpid for r in reads if any(c in sys_of for c in r.cids)})},
        "routes_by_level": {},
        "rows": report_rows(metas, lookup),
        "parallel_reads": report_reads([r for r in reads if r.source == "pgp_parallel"], lookup),
        "paired_a50_minus_v21b": report_paired(reads, lookup),
        "pilot_val_only_reads": sum(1 for (p, m), lvl in lookup.items() if lvl == "none" and m == "a50"
                                    and p in index.val_pgpids),
    }
    routes: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    seen_reads: Set[Tuple[str, str]] = set()
    for rec in records:
        k = (rec["pgpid"], rec["read_model"])
        if rec["level"] == rec["read_level"] and k not in seen_reads:
            seen_reads.add(k)
            routes[rec["read_level"]].update({r.split(":", 1)[0] for r in rec["routes"]} or {"-"})
    report["routes_by_level"] = {lvl: dict(c.most_common()) for lvl, c in routes.items()}
    docs_any = collections.defaultdict(bool)
    for (p, _m), lvl in lookup.items():
        docs_any[p] |= lvl != "none"
    report["docs"] = {"with_reads": len(docs_any), "exposed_any_reader": sum(docs_any.values())}
    if prototype and Path(prototype).exists():
        with open(prototype, "rb") as fh:
            report["prototype_check"] = compare_prototype(lookup, orjson.loads(fh.read()))
    report["seconds"] = round(time.time() - t0, 1)
    _write_json(out_dir / "exposure_report.json", report)
    return report


def _print_summary(report: Mapping[str, Any]) -> None:
    """Print the headline numbers of a report.

    :param report: :func:`run` result.
    :returns: None.
    """
    print(f"reads {report['reads']['total']:,} ({report['reads']['by_source']}), "
          f"page records {report['reads']['page_records']:,}; docs exposed (any reader) "
          f"{report['docs']['exposed_any_reader']:,}/{report['docs']['with_reads']:,}")
    for split, by_model in report["rows"].items():
        for model, e in by_model.items():
            print(f"  rows {split:5s} {model:5s}: {e['exposed_rows']:,}/{e['rows']:,} exposed {e['rows_by_level']} "
                  f"near-dup {e['near_duplicates']}; CER median exposed {e['cer_exposed'].get('p50')} "
                  f"vs unexposed {e['cer_unexposed'].get('p50')}")
    for model, e in report["parallel_reads"].items():
        ig = e["cer_in_gate"]
        print(f"  parallel {model}: {e['reads_by_level']}; in-gate CER median exposed {ig['exposed'].get('p50')} "
              f"(n {ig['exposed']['n']}) vs unexposed {ig['unexposed'].get('p50')} (n {ig['unexposed']['n']})")
    for name, groups in report["paired_a50_minus_v21b"].items():
        print(f"  paired a50-v21b {name}: " + ", ".join(f"{g} n={s['n']} median={s.get('p50')}"
                                                       for g, s in groups.items() if g != "did_vs_neither"))
        for g, d in (groups.get("did_vs_neither") or {}).items():
            print(f"    {name} {g} minus neither: mean {d['mean_diff']} CI95 {d['ci95']} (median {d['median_diff']})")
    if "prototype_check" in report:
        pc = report["prototype_check"]
        print(f"  prototype check: {pc['exposed_agree']}/{pc['compared']} exposure agree, "
              f"{pc['page_agree']}/{pc['compared']} page-level agree")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: build ``exposure.jsonl`` + ``exposure_report.json`` under ``paths.out_dir``.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :returns: Exit code (0 on success, 2 when a required source is missing).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--paths", type=Path, default=None, help="paths.json (default: $TEXT27B_PATHS or the repo default)")
    ap.add_argument("--rows", type=Path, nargs="*", default=None,
                    help="row JSONL files to cover and report on (default: v0_1 train/dev/test)")
    ap.add_argument("--out-dir", type=Path, default=None, help="output directory (default: paths.out_dir)")
    ap.add_argument("--sample-docs", type=int, default=None, help="first N parallel-corpus documents only (dev)")
    ap.add_argument("--no-ktiv-pages", action="store_true", help="skip listing KTIV page files (manuscript level only)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="run although some sources are absent (e.g. NAS not mounted); exposure is then under-counted")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = _paths.load_paths(args.paths)
    sources = ExposureSources.from_paths(p)
    missing = sources.missing()
    if "pgp_parallel" in missing or (missing and not args.allow_missing):
        LOG.error("sources missing: %s (pass --allow-missing to under-count knowingly; pgp_parallel is always required)",
                  missing)
        return 2
    row_files = args.rows if args.rows is not None else [p.text_ft_v0_1 / f"{s}.jsonl" for s in ("train", "dev", "test")]
    row_files = [f for f in row_files if Path(f).exists()]
    report = run(sources, row_files, args.out_dir or p.out_dir, sample_docs=args.sample_docs,
                 ktiv_pages=not args.no_ktiv_pages, prototype=p.a50_overlap)
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
