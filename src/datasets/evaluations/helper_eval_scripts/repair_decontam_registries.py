"""Repair the decontamination registries: pre-fix Oxford ids -> current ES ids (migration plan PR-5, F5 / R11).

The registries under ``src/datasets/raw_data/cairo_genizah/decontam/`` (git-ignored data) were written before the
2026-10 Oxford fix, so some of their ids still have the pre-fix form (``Oxford_Bodleian_MS_heb_a_2_4``, or the
case variant ``Oxford_Bodleian_Bodl_MS_Heb_d_66_13``) that the ES no longer holds (0 old-form ids in
``merged_shelfmarks.jsonl``). Exact-id hold-outs miss those documents, and the stored keys of the two key
registries are the keys of the old ids.

What the tool does (re-runnable; ``--dry-run`` is the default, ``--apply`` writes):

1. Reads every id registry (:data:`ID_REGISTRIES`) and the sealed registry (:data:`SEALED_REGISTRY`, read only) and
   classifies every id: a current ES id, a KTIV page id (``KTIV_<system number>_FL<file>``, the sealed literary
   pages, not an ES namespace), a pre-fix Oxford id that maps to a current ES id, or unmapped (reported, never
   changed).
2. Maps a pre-fix Oxford id with :func:`oxford_reparse`: the id's core is re-read by the Bodleian parser
   (:func:`src.datasets.shelfmarks.parse_oxford`, underscores as spaces) under the Oxford token. This is the
   recipe of the facade's old-Oxford re-parse (``canonical_id`` on an old-form id), extended to the
   ``Bodl_MS_Heb`` case variant, which ``canonical_id`` echoes unchanged because its core already starts with
   ``Bodl_``. ``oxford_id_map.recompute_id`` is not called: it recomputes an id from a merged record's source
   blocks, and a registry holds a bare id whose record is no longer under that id. Its recorded output, the
   id map CSV written by ``oxford_id_map`` (:data:`OXFORD_MAP_CSV`, old -> new over the pre-fix merge), is the
   check: a mapping is applied only when the mapped id is an ES id and, when the CSV lists the old id, the CSV
   names the same new id.
3. Regenerates the two key registries (:data:`KEY_REGISTRIES`) with :func:`src.datasets.shelfmarks.loose_key`
   over the repaired ids. A key that is not the facade key of a registry id and starts with the Oxford word is
   stale (a key of a pre-fix Oxford id) and is dropped. Every other key, including the institution-free keys
   that an older engine wrote into ``benchmark_keys.json`` (not reproducible by the facade, read by the
   sealed build's FROZEN fragment key), is kept byte-identical. A change to any key outside the Oxford word
   stops the tool.
4. Keeps each file's key order and JSON formatting: the serialisation options are detected from the file's own
   bytes (:func:`detect_format`) and must reproduce them exactly before anything is written. Ids are replaced in
   place (registry ``ids`` stay aligned with their ``pgpids``); a key registry stays sorted.
5. ``--apply`` first copies each file it will change, byte for byte, into :data:`BACKUP_DIR` and records path,
   sha256 before, sha256 after and the entries changed in ``backup_manifest.json``. An existing backup that
   differs from the file is never overwritten. Files are rewritten in place (never moved or renamed).

The report (per file: ids, ids not in ES before, ids mapped, ids still not in ES after, key diffs) is printed and,
with ``--report``, written as Markdown (``--apply`` writes :data:`REPORT_MD` by default).

Run::

    python -m src.datasets.evaluations.helper_eval_scripts.repair_decontam_registries            # dry run
    python -m src.datasets.evaluations.helper_eval_scripts.repair_decontam_registries --apply
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.datasets import shelfmarks  # noqa: E402
from src.datasets.merging import oxford_id_map  # noqa: E402

#: Repo-relative locations (resolved against the repository root at run time).
DECONTAM_DIR = "src/datasets/raw_data/cairo_genizah/decontam"
SEALED_REGISTRY = "src/datasets/raw_data/cairo_genizah/evaluations/sealed_v1/sealed_v1.json"
MERGED_JSONL = "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"
OXFORD_MAP_CSV = "src/datasets/raw_data/cairo_genizah/merged/" + oxford_id_map.DRYRUN_CSV
AUDIT_DIR = "logs/next_round/unseen_v1/shelfmark_audit"
BACKUP_DIR = AUDIT_DIR + "/decontam_backup_2026-11-03"
BACKUP_MANIFEST = "backup_manifest.json"
REPORT_MD = AUDIT_DIR + "/pr5_registry_repair.md"

#: Registries of ES ids under :data:`DECONTAM_DIR`: a JSON list, or ``{"ids": [...], ...}``.
ID_REGISTRIES: Tuple[str, ...] = (
    "benchmark_ids.json", "clean_v2_ids.json", "arabic_script_benchmark_v0.json",
    "held_out_edition_pages_v0.json", "offbench_sample_v0.json", "two_edition_documents_v0.json",
)
#: Key registry -> the id registry it keys (sorted JSON lists of loose keys).
KEY_REGISTRIES: Dict[str, str] = {
    "benchmark_keys.json": "benchmark_ids.json",
    "clean_v2_keys.json": "clean_v2_ids.json",
}
#: Files under :data:`DECONTAM_DIR` that hold no ES ids (glob -> what they hold); read by no step of the repair.
NON_ID_FILES: Dict[str, str] = {
    "ktiv_v19a_train_sysnums.json": "KTIV system numbers",
    "v20_ktiv_candidate_pool*.json": "KTIV candidate pools (system numbers + raw KTIV marks)",
}
#: A sealed literary page id (``KTIV_<18-digit system number>_FL<file id>``): its own namespace, not an ES id.
KTIV_PAGE_ID = re.compile(r"KTIV_(\d{18})_FL\d+")
#: First word of the Oxford token, lower-cased: every Oxford loose key starts with it.
OXFORD_WORD = shelfmarks.OXFORD_TOKEN.split("_")[0].lower()
_CANONICAL_ID = re.compile(rb'"canonical_id":\s*"((?:[^"\\]|\\.)*)"')

CURRENT = "current"
MAPPED = "mapped"
KTIV_PAGE = "ktiv_page_id"
UNMAPPED = "unmapped"


@dataclass(frozen=True)
class Resolution:
    """How one registry id resolves against the ES ids.

    :ivar old: The registry id.
    :ivar new: The id to store (``old`` itself unless mapped); ``None`` when unmapped.
    :ivar status: :data:`CURRENT`, :data:`MAPPED`, :data:`KTIV_PAGE` or :data:`UNMAPPED`.
    :ivar note: Why (the id-map check for a mapped id; the reason for an unmapped one).
    """

    old: str
    new: Optional[str]
    status: str
    note: str = ""


@dataclass(frozen=True)
class JsonFormat:
    """Serialisation options that reproduce a registry file byte for byte.

    :ivar indent: ``json.dumps`` indent (``None`` = one line).
    :ivar ensure_ascii: ``json.dumps`` ``ensure_ascii``.
    :ivar trailing_newline: Whether the file ends with ``\\n``.
    """

    indent: Optional[int]
    ensure_ascii: bool
    trailing_newline: bool


@dataclass
class FilePlan:
    """The planned change of one registry file.

    :ivar path: Repo-relative path.
    :ivar kind: ``ids``, ``keys`` or ``sealed`` (read only).
    :ivar entries: Number of ids (or keys) in the file.
    :ivar not_in_es_before: Ids that are not ES ids before the repair.
    :ivar mapped: ``(old, new)`` pairs applied.
    :ivar unmapped_after: Resolutions of ids still not ES ids after the repair (KTIV page ids included).
    :ivar keys_removed: Stale keys dropped (key registries).
    :ivar keys_added: Facade keys added (key registries).
    :ivar old_text: The file as read.
    :ivar new_text: The file as it would be written (equal to ``old_text`` when nothing changes).
    :ivar id_map_confirmed: Mapped ids that the id map CSV lists with the same new id.
    """

    path: str
    kind: str
    entries: int
    not_in_es_before: List[str] = field(default_factory=list)
    mapped: List[Tuple[str, str]] = field(default_factory=list)
    unmapped_after: List[Resolution] = field(default_factory=list)
    keys_removed: List[str] = field(default_factory=list)
    keys_added: List[str] = field(default_factory=list)
    old_text: str = ""
    new_text: str = ""
    id_map_confirmed: int = 0

    @property
    def changed(self) -> bool:
        """Whether the file would be rewritten.

        :return: ``new_text != old_text``.
        :rtype: bool
        """
        return self.new_text != self.old_text

    @property
    def entries_changed(self) -> int:
        """Number of entries the repair changes (mapped ids, or removed + added keys).

        :return: The count.
        :rtype: int
        """
        return len(self.mapped) + len(self.keys_removed) + len(self.keys_added)


# ------------------------------------------------------------------------------------------------ loading


def load_es_ids(merged_path: Path) -> Set[str]:
    """The canonical ids of the primary ES index source (one per merged record).

    :param merged_path: ``merged_shelfmarks.jsonl`` (read only).
    :type merged_path: Path
    :return: The id set.
    :rtype: Set[str]
    """
    ids: Set[str] = set()
    with open(merged_path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            m = _CANONICAL_ID.search(line[:600])
            ids.add(json.loads(b'"' + m.group(1) + b'"') if m else json.loads(line)["canonical_id"])
    return ids


def load_oxford_map(csv_path: Path) -> Dict[str, Set[str]]:
    """Old -> new ids recorded by ``oxford_id_map`` over the pre-fix merge (rows whose id changes).

    :param csv_path: The id map CSV (:data:`OXFORD_MAP_CSV`); absent -> empty map.
    :type csv_path: Path
    :return: Old id -> every new id the CSV gives it.
    :rtype: Dict[str, Set[str]]
    """
    out: Dict[str, Set[str]] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            old, new = row["old_canonical_id"], row["new_canonical_id"]
            if old and new and old != new:
                out.setdefault(old, set()).add(new)
    return out


def sha256_text(text: str) -> str:
    """SHA-256 of a text's UTF-8 bytes.

    :param text: Text.
    :type text: str
    :return: Hex digest.
    :rtype: str
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ mapping


def oxford_reparse(cid: str) -> Optional[str]:
    """The current id of an Oxford id: its core re-read by the Bodleian parser under the Oxford token.

    :param cid: A registry id (``Oxford_Bodleian_MS_heb_a_2_4``, ``Oxford_Bodleian_Bodl_MS_Heb_d_66_13``).
    :type cid: str
    :return: The re-parsed id (``Oxford_Bodleian_Bodl_MS_heb_a_2_4``), or ``None`` when *cid* is not under the
        Oxford token or its core is not a Bodleian mark.
    :rtype: Optional[str]
    """
    token, core = shelfmarks.split_id(cid)
    if token != shelfmarks.OXFORD_TOKEN or not core:
        return None
    parsed = shelfmarks.parse_oxford(core.replace("_", " "))
    return f"{token}_{parsed.core}" if parsed is not None else None


def map_registry_id(cid: str, es_ids: Set[str], oxford_map: Dict[str, Set[str]]) -> Resolution:
    """Resolve one registry id against the ES ids.

    :param cid: Registry id.
    :type cid: str
    :param es_ids: Current ES ids.
    :type es_ids: Set[str]
    :param oxford_map: :func:`load_oxford_map` result (may be empty).
    :type oxford_map: Dict[str, Set[str]]
    :return: The resolution. Only an Oxford id is ever mapped; any other id that is not an ES id is
        :data:`UNMAPPED` (or :data:`KTIV_PAGE`) and stays as it is.
    :rtype: Resolution
    """
    if cid in es_ids:
        return Resolution(cid, cid, CURRENT)
    if KTIV_PAGE_ID.fullmatch(cid):
        return Resolution(cid, None, KTIV_PAGE, "sealed KTIV page id (system number + file), not an ES namespace")
    new = oxford_reparse(cid)
    if new is None:
        return Resolution(cid, None, UNMAPPED, "not an ES id and not an Oxford id")
    if new not in es_ids:
        return Resolution(cid, None, UNMAPPED, f"Oxford re-parse gives {new}, not an ES id")
    recorded = oxford_map.get(cid)
    if recorded is not None and recorded != {new}:
        return Resolution(cid, None, UNMAPPED, f"id map gives {sorted(recorded)}, re-parse gives {new}")
    return Resolution(cid, new, MAPPED, "id map agrees" if recorded else "not listed in the id map")


# ------------------------------------------------------------------------------------------------ JSON format


_FORMATS: Tuple[JsonFormat, ...] = tuple(
    JsonFormat(indent, ascii_, nl)
    for indent in (None, 0, 1, 2, 4) for ascii_ in (True, False) for nl in (False, True)
)


def dump_json(obj: Any, fmt: JsonFormat) -> str:
    """Serialise *obj* with a detected format.

    :param obj: JSON value.
    :type obj: Any
    :param fmt: Format.
    :type fmt: JsonFormat
    :return: The text.
    :rtype: str
    """
    return json.dumps(obj, indent=fmt.indent, ensure_ascii=fmt.ensure_ascii) + ("\n" if fmt.trailing_newline else "")


def detect_format(text: str, obj: Any) -> JsonFormat:
    """Find the serialisation options that reproduce *text* exactly from *obj*.

    :param text: The file's text.
    :type text: str
    :param obj: ``json.loads(text)``.
    :type obj: Any
    :return: The first matching format.
    :rtype: JsonFormat
    :raises ValueError: When no candidate format reproduces the file (it is then not rewritten).
    """
    for fmt in _FORMATS:
        if dump_json(obj, fmt) == text:
            return fmt
    raise ValueError("no json.dumps format reproduces the file byte for byte")


def ids_of(obj: Any) -> List[str]:
    """The id list of a registry object.

    :param obj: A JSON list, or a dict with an ``ids`` list.
    :type obj: Any
    :return: The ids.
    :rtype: List[str]
    """
    return list(obj["ids"] if isinstance(obj, dict) else obj)


def with_ids(obj: Any, ids: List[str]) -> Any:
    """A copy of a registry object with its ids replaced (dict key order kept).

    :param obj: A JSON list, or a dict with an ``ids`` list.
    :type obj: Any
    :param ids: New ids.
    :type ids: List[str]
    :return: The new object.
    :rtype: Any
    """
    if isinstance(obj, dict):
        return {k: (ids if k == "ids" else v) for k, v in obj.items()}
    return ids


# ------------------------------------------------------------------------------------------------ planning


def plan_id_file(rel: str, text: str, es_ids: Set[str], oxford_map: Dict[str, Set[str]],
                 kind: str = "ids") -> FilePlan:
    """Plan the repair of one id registry.

    :param rel: Repo-relative path (for the report).
    :type rel: str
    :param text: File text.
    :type text: str
    :param es_ids: Current ES ids.
    :type es_ids: Set[str]
    :param oxford_map: :func:`load_oxford_map` result.
    :type oxford_map: Dict[str, Set[str]]
    :param kind: ``ids`` (written on apply) or ``sealed`` (reported only: ``new_text`` stays ``old_text``).
    :type kind: str
    :return: The plan.
    :rtype: FilePlan
    :raises ValueError: When a mapped id would duplicate an id already in the file.
    """
    obj = json.loads(text)
    fmt = detect_format(text, obj)
    ids = ids_of(obj)
    plan = FilePlan(path=rel, kind=kind, entries=len(ids), old_text=text, new_text=text)
    new_ids: List[str] = []
    for cid in ids:
        res = map_registry_id(cid, es_ids, oxford_map)
        if res.status != CURRENT:
            plan.not_in_es_before.append(cid)
        if res.status == MAPPED and kind == "ids":
            plan.mapped.append((cid, res.new))
            plan.id_map_confirmed += res.note == "id map agrees"
            new_ids.append(res.new)
        else:
            if res.status == MAPPED:
                res = Resolution(cid, None, UNMAPPED, f"maps to {res.new}; this registry is reported, not written")
            if res.status != CURRENT:
                plan.unmapped_after.append(res)
            new_ids.append(cid)
    if len(set(new_ids)) != len(new_ids) and len(set(ids)) == len(ids):
        raise ValueError(f"{rel}: a mapped id duplicates an id already in the file")
    if plan.mapped:
        plan.new_text = dump_json(with_ids(obj, new_ids), fmt)
    return plan


def regenerate_keys(old_keys: Sequence[str], ids: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    """Regenerate a key registry from its (repaired) ids with the facade ``loose_key``.

    :param old_keys: The file's keys (sorted).
    :type old_keys: Sequence[str]
    :param ids: The repaired ids of the registry the file keys.
    :type ids: Iterable[str]
    :return: ``(new sorted keys, removed, added)``. Removed = keys starting with the Oxford word that are not the
        facade key of any id (keys of pre-fix Oxford ids); added = facade keys missing from the file.
    :rtype: Tuple[List[str], List[str], List[str]]
    :raises ValueError: When a key outside the Oxford word would change.
    """
    facade = {k for k in (shelfmarks.loose_key(i) for i in ids) if k}
    old = set(old_keys)
    removed = sorted(k for k in old if k not in facade and k.startswith(OXFORD_WORD))
    added = sorted(facade - old)
    outside = [k for k in added if not k.startswith(OXFORD_WORD)]
    if outside:
        raise ValueError(f"facade keys outside the Oxford word would be added: {outside[:5]}")
    return sorted((old - set(removed)) | facade), removed, added


def plan_key_file(rel: str, text: str, ids_after: Sequence[str]) -> FilePlan:
    """Plan the regeneration of one key registry.

    :param rel: Repo-relative path.
    :type rel: str
    :param text: File text (a sorted JSON list of keys).
    :type text: str
    :param ids_after: The repaired ids of the registry it keys.
    :type ids_after: Sequence[str]
    :return: The plan.
    :rtype: FilePlan
    :raises ValueError: When the file is not a sorted list.
    """
    keys = json.loads(text)
    fmt = detect_format(text, keys)
    if keys != sorted(keys):
        raise ValueError(f"{rel}: key registry is not sorted")
    new_keys, removed, added = regenerate_keys(keys, ids_after)
    plan = FilePlan(path=rel, kind="keys", entries=len(keys), keys_removed=removed, keys_added=added,
                    old_text=text, new_text=text)
    if removed or added:
        plan.new_text = dump_json(new_keys, fmt)
    return plan


def ids_after(plan: FilePlan) -> List[str]:
    """The ids an id-registry plan would leave in its file.

    :param plan: An id plan.
    :type plan: FilePlan
    :return: The ids.
    :rtype: List[str]
    """
    return ids_of(json.loads(plan.new_text))


def build_plan(repo: Path, es_ids: Set[str], oxford_map: Dict[str, Set[str]]) -> List[FilePlan]:
    """Plan the repair of every registry.

    :param repo: Repository root.
    :type repo: Path
    :param es_ids: Current ES ids.
    :type es_ids: Set[str]
    :param oxford_map: :func:`load_oxford_map` result.
    :type oxford_map: Dict[str, Set[str]]
    :return: One plan per id registry, key registry and the sealed registry (when present), in that order.
    :rtype: List[FilePlan]
    """
    plans: Dict[str, FilePlan] = {}
    for name in ID_REGISTRIES:
        rel = f"{DECONTAM_DIR}/{name}"
        plans[name] = plan_id_file(rel, (repo / rel).read_text(encoding="utf-8"), es_ids, oxford_map)
    out = list(plans.values())
    for name, id_name in KEY_REGISTRIES.items():
        rel = f"{DECONTAM_DIR}/{name}"
        out.append(plan_key_file(rel, (repo / rel).read_text(encoding="utf-8"), ids_after(plans[id_name])))
    if (repo / SEALED_REGISTRY).exists():
        out.append(plan_id_file(SEALED_REGISTRY, (repo / SEALED_REGISTRY).read_text(encoding="utf-8"),
                                es_ids, oxford_map, kind="sealed"))
    return out


def unclassified_files(repo: Path) -> List[str]:
    """JSON files under the decontam directory that no part of the repair accounts for.

    :param repo: Repository root.
    :type repo: Path
    :return: Their names (should be empty).
    :rtype: List[str]
    """
    known = set(ID_REGISTRIES) | set(KEY_REGISTRIES)
    out = []
    for p in sorted((repo / DECONTAM_DIR).glob("*.json")):
        if p.name not in known and not any(p.match(g) for g in NON_ID_FILES):
            out.append(p.name)
    return out


# ------------------------------------------------------------------------------------------------ apply


def _display_path(path: Path, repo: Path) -> str:
    """A path for the manifest: repo-relative when inside the repository, else its file name.

    :param path: Path.
    :type path: Path
    :param repo: Repository root.
    :type repo: Path
    :return: The display path.
    :rtype: str
    """
    return str(path.relative_to(repo)) if path.is_relative_to(repo) else path.name


def apply_plan(plans: Sequence[FilePlan], repo: Path, backup_dir: Path) -> List[Dict[str, Any]]:
    """Back up and rewrite (in place) every file whose plan changes it.

    :param plans: :func:`build_plan` result.
    :type plans: Sequence[FilePlan]
    :param repo: Repository root.
    :type repo: Path
    :param backup_dir: Backup directory (created when missing).
    :type backup_dir: Path
    :return: The manifest entries written for this run.
    :rtype: List[Dict[str, Any]]
    :raises FileExistsError: When a backup of a file exists and differs from the file (never overwritten).
    """
    todo = [p for p in plans if p.changed and p.kind in ("ids", "keys")]
    if not todo:
        return []
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in todo:
        src = repo / p.path
        dst = backup_dir / src.name
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise FileExistsError(f"backup {dst.name} exists and differs from {p.path}; not overwritten")
        if not dst.exists():
            shutil.copyfile(src, dst)
        if hashlib.sha256(dst.read_bytes()).hexdigest() != hashlib.sha256(src.read_bytes()).hexdigest():
            raise OSError(f"backup of {p.path} is not byte-identical")
    manifest_path = backup_dir / BACKUP_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"files": []}
    entries: List[Dict[str, Any]] = []
    for p in todo:
        with open(repo / p.path, "w", encoding="utf-8", newline="") as fh:
            fh.write(p.new_text)
        written = (repo / p.path).read_text(encoding="utf-8")
        if written != p.new_text:
            raise OSError(f"{p.path}: written text differs from the plan")
        entries.append({
            "path": p.path,
            "backup": _display_path(backup_dir / Path(p.path).name, repo),
            "sha256_before": sha256_text(p.old_text),
            "sha256_after": sha256_text(p.new_text),
            "entries_changed": {"ids_mapped": len(p.mapped), "keys_removed": len(p.keys_removed),
                                "keys_added": len(p.keys_added)},
            "normaliser_version": shelfmarks.NORMALISER_VERSION,
            "applied_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    done = {e["path"] for e in entries}
    manifest["files"] = [e for e in manifest["files"] if e["path"] not in done] + entries
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return entries


# ------------------------------------------------------------------------------------------------ report


def render_report(plans: Sequence[FilePlan], mode: str, es_count: int, oxford_map_rows: int,
                  unclassified: Sequence[str]) -> str:
    """Markdown report of a run.

    :param plans: :func:`build_plan` result.
    :type plans: Sequence[FilePlan]
    :param mode: ``dry-run`` or ``apply``.
    :type mode: str
    :param es_count: Number of ES ids.
    :type es_count: int
    :param oxford_map_rows: Number of old ids in the id map CSV (0 when absent).
    :type oxford_map_rows: int
    :param unclassified: :func:`unclassified_files` result.
    :type unclassified: Sequence[str]
    :return: The report.
    :rtype: str
    """
    mapped_total = sum(len(p.mapped) for p in plans)
    confirmed = sum(p.id_map_confirmed for p in plans)
    lines = [
        f"# Decontamination registry repair ({mode})",
        "",
        f"* ES ids: {es_count:,}; normaliser {shelfmarks.NORMALISER_VERSION}; id map CSV old ids: {oxford_map_rows:,}.",
        f"* Ids mapped: {mapped_total} ({confirmed} confirmed by the id map CSV); files changed: "
        f"{sum(p.changed for p in plans if p.kind != 'sealed')}.",
        f"* Unclassified decontam files: {list(unclassified) or 'none'}.",
        "",
        "| file | kind | entries | not in ES before | mapped | not in ES after | keys removed | keys added | changes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in plans:
        lines.append(
            f"| `{Path(p.path).name}` | {p.kind} | {p.entries} | {len(p.not_in_es_before)} | {len(p.mapped)} | "
            f"{len(p.unmapped_after)} | {len(p.keys_removed)} | {len(p.keys_added)} | {p.entries_changed} |")
    for p in plans:
        if p.mapped:
            lines += ["", f"## `{Path(p.path).name}`: ids mapped ({len(p.mapped)})", "",
                      "| old id | current ES id |", "|---|---|"]
            lines += [f"| `{o}` | `{n}` |" for o, n in p.mapped]
        if p.keys_removed or p.keys_added:
            lines += ["", f"## `{Path(p.path).name}`: key diff (-{len(p.keys_removed)} / +{len(p.keys_added)})", "",
                      "| removed (pre-fix Oxford) | added (facade `loose_key`) |", "|---|---|"]
            n = max(len(p.keys_removed), len(p.keys_added))
            rem, add = p.keys_removed + [""] * n, p.keys_added + [""] * n
            lines += [f"| {f'`{rem[i]}`' if rem[i] else ''} | {f'`{add[i]}`' if add[i] else ''} |" for i in range(n)]
        if p.unmapped_after:
            reasons: Dict[str, int] = {}
            for r in p.unmapped_after:
                reasons[r.status if r.status == KTIV_PAGE else r.note] = reasons.get(
                    r.status if r.status == KTIV_PAGE else r.note, 0) + 1
            lines += ["", f"## `{Path(p.path).name}`: ids not in ES after ({len(p.unmapped_after)})", ""]
            lines += [f"* {reason}: {count}" for reason, count in sorted(reasons.items())]
            others = [r for r in p.unmapped_after if r.status != KTIV_PAGE]
            lines += [f"  * `{r.old}`: {r.note}" for r in others]
    lines += ["", "Files holding no ES ids (not read): "
              + "; ".join(f"`{g}` ({what})" for g, what in NON_ID_FILES.items()) + "."]
    return "\n".join(lines) + "\n"


def run(repo: Path, apply: bool, backup_dir: Optional[Path] = None,
        report_path: Optional[Path] = None) -> Tuple[List[FilePlan], List[Dict[str, Any]], str]:
    """Plan (and with *apply* carry out) the repair.

    :param repo: Repository root.
    :type repo: Path
    :param apply: Write the changes (after backing up).
    :type apply: bool
    :param backup_dir: Backup directory (default :data:`BACKUP_DIR` under *repo*).
    :type backup_dir: Optional[Path]
    :param report_path: Where to write the Markdown report (``None``: not written).
    :type report_path: Optional[Path]
    :return: ``(plans, manifest entries written, report text)``.
    :rtype: Tuple[List[FilePlan], List[Dict[str, Any]], str]
    """
    es_ids = load_es_ids(repo / MERGED_JSONL)
    oxford_map = load_oxford_map(repo / OXFORD_MAP_CSV)
    plans = build_plan(repo, es_ids, oxford_map)
    entries = apply_plan(plans, repo, backup_dir or repo / BACKUP_DIR) if apply else []
    report = render_report(plans, "apply" if apply else "dry-run", len(es_ids), len(oxford_map),
                           unclassified_files(repo))
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    return plans, entries, report


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :type argv: Optional[Sequence[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="plan and report only (default)")
    mode.add_argument("--apply", action="store_true", help="back up, then rewrite the registries in place")
    parser.add_argument("--report", default=None,
                        help=f"Markdown report path, repo-relative (default with --apply: {REPORT_MD})")
    args = parser.parse_args(argv)
    report = args.report or (REPORT_MD if args.apply else None)
    plans, entries, text = run(_REPO, args.apply, report_path=_REPO / report if report else None)
    print(text)
    if args.apply:
        print(f"applied: {len(entries)} file(s) rewritten; backups + manifest in {BACKUP_DIR}")
    else:
        print(f"dry run: {sum(p.changed for p in plans if p.kind != 'sealed')} file(s) would change")


if __name__ == "__main__":
    main()
