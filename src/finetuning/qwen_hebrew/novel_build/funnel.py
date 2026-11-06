"""Funnel of the novel-rows build: candidates, sides and document groups at every stage, per source and script.

Every number comes from the novelty ledger (all verdicts) and the stage files; nothing is typed (acceptance rule 8).
Stages, in order:

inventoried -> held out (ledger) -> previously trained -> uncertain history -> novel (the ledger's verdicts) ->
target extracted (stage0 candidate with at least one target side) -> image found (stage1: at least one photograph
present or fetched) -> side verified (stage2 accepted) -> quality passed (stage3 ok) -> held out by the builder
(stage5 excluded) -> validation rows -> newly usable training rows (stage6 manifest).

Each stage is counted from its own file (a side counts as "quality passed" when stage3 says ok, whatever the earlier
files say), so a stale or inconsistent stage file shows up as a count that exceeds the stage before it
(:func:`monotone_violations`). The drop reasons come from :func:`trace`, which follows every (candidate, side) of
stage0 down the stages and stops at the first stage it does not pass.

Script: the ledger's script up to "side verified"; from "quality passed" on, the script measured on the final target.
Document groups are the ledger's ``document_group`` (sides and editions of one fragment share one group).

The build's human-review gate (approved works, approved rows, review policy) is reported apart, in the review table
(:func:`review_counts`): per source the sides pending work or row review, the sides rejected by review, the rows
admitted with review flags, and the rule-recovered placements admitted and excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build.schema import (SCRIPTS, Candidate, HoldoutResult, ImageRef, QualityResult,
                                                           RowRecord, SideDecision, read_jsonl)

#: Image statuses that count as "image found".
FOUND_STATUSES = ("present", "fetched")
#: Hold-out splits that make a row.
ROW_SPLITS = ("train", "val")
#: The build's per-(candidate, side) decisions, inside the work dir (written by :mod:`.build`).
DECISIONS_FILE = "stages/stage6_decisions.jsonl"
#: Default output name inside the work dir.
FUNNEL_FILE = "FUNNEL.md"
#: Ledger verdict -> funnel stage.
VERDICT_STAGES = {"held_out": "held_out_ledger", "previously_trained": "previously_trained",
                  "uncertain_history": "uncertain_history", "novel": "novel"}
#: Ledger witness lists named as evidence when a non-novel record lists no witness dataset.
EVIDENCE_KEYS = ("image", "document", "image_uncertain", "document_uncertain")
#: Display names of the legacy-store sources (reports never print the ledger's source keys for them).
SOURCE_LABELS = {"fjp": "legacy_store", "fjp_scrape": "legacy_store_scrape"}
#: Stage-2 reason codes that mark a rule-recovered placement on an accepted side (``side_assign.FLAG_ONLY``; a test
#: pins the two lists together).
RULE_RECOVERED_REASONS = ("spill_ratio_ok", "conflict_ratio_ok")
#: Review codes the build writes. Each is an exclusion reason when its class is under review policy ``require`` and a
#: ``quality_flags`` entry on the admitted row when it is under ``flag``; ``review_pending_work`` and
#: ``review_rejected`` are followed by ``:<source>:<work>`` and ``:<verdict>``.
REVIEW_PENDING_WORK = "review_pending_work"
REVIEW_PENDING_ROW = "review_pending_row"
REVIEW_REJECTED = "review_rejected"
#: Columns of the review table: key, display label.
REVIEW_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("pending_work", "pending work review"), ("pending_row", "pending row review"),
    ("rejected", "rejected by review"), ("admitted_flagged", "admitted with flags"),
    ("rule_admitted", "rule-recovered admitted"), ("rule_excluded", "rule-recovered excluded"),
    ("rule_excluded_by_review", "of which by review"))


@dataclass(frozen=True)
class Stage:
    """One funnel stage.

    :param key: Stage key.
    :param label: Display label.
    :param level: ``ledger`` (candidates), ``target`` (candidates and their target sides) or ``side`` (sides).
    :param rendered: Shown in ``FUNNEL.md`` (internal stages only feed the consistency check).
    """

    key: str
    label: str
    level: str
    rendered: bool = True


STAGES: Tuple[Stage, ...] = (
    Stage("inventoried", "inventoried", "ledger"),
    Stage("held_out_ledger", "held out (ledger)", "ledger"),
    Stage("previously_trained", "previously trained", "ledger"),
    Stage("uncertain_history", "uncertain history", "ledger"),
    Stage("novel", "novel", "ledger"),
    Stage("target_extracted", "target extracted", "target"),
    Stage("image_found", "image found", "target"),
    Stage("side_verified", "side verified", "side"),
    Stage("quality_passed", "quality passed", "side"),
    Stage("held_out_builder", "held out by the builder", "side"),
    Stage("holdout_cleared", "hold-outs cleared", "side", rendered=False),
    Stage("val_rows", "validation rows", "side"),
    Stage("train_rows", "newly usable training rows", "side"),
    Stage("rows", "rows (training + validation)", "side", rendered=False),
)
STAGE_KEYS = tuple(stage.key for stage in STAGES)
STAGE_LABELS = {stage.key: stage.label for stage in STAGES}
#: Stages whose counts may never exceed the stage before them (in candidates, sides and document groups).
MONOTONE_CHAIN = ("novel", "target_extracted", "image_found", "side_verified", "quality_passed", "holdout_cleared",
                  "rows")
#: Stage a traced record reached -> the stage it failed to reach (where its reasons are counted).
NEXT_STAGE = {"novel": "target_extracted", "target_extracted": "image_found", "image_found": "side_verified",
              "side_verified": "quality_passed", "quality_passed": "held_out_builder", "holdout_cleared": "rows"}
#: Stages of the reasons table, in order, with display labels.
REASON_STAGES = (("held_out_ledger", "held out (ledger)"), ("previously_trained", "previously trained"),
                 ("uncertain_history", "uncertain history"), ("target_extracted", "target extracted"),
                 ("image_found", "image found"), ("side_verified", "side verified"),
                 ("quality_passed", "quality passed"), ("held_out_builder", "held out by the builder"),
                 ("rows", "built as a row"), ("val_rows", "validation rows"))


@dataclass
class LedgerEntry:
    """The fields of one ledger record the funnel and the build use.

    :param row_id: Ledger ``row_id``.
    :param source: Ledger source.
    :param script: Ledger script.
    :param document_group: Ledger document group.
    :param verdict: ``flags.verdict``.
    :param witnesses: ``flags.witnesses`` (training datasets that saw the document).
    :param held_out: ``flags.held_out`` (why the ledger holds the record out), if any.
    :param evidence: Names of the non-empty witness lists among :data:`EVIDENCE_KEYS`.
    """

    row_id: str
    source: str
    script: str
    document_group: str
    verdict: str
    witnesses: List[str] = field(default_factory=list)
    held_out: Optional[str] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class StageData:
    """Everything the funnel (and the build) reads, loaded once.

    :param ledger: Ledger entries (every verdict).
    :param candidates: Stage0 candidates (targets).
    :param images: Stage1 candidates (completed image lists).
    :param sides: Stage2 side decisions.
    :param quality: Stage3 quality results.
    :param holdout: Stage5 hold-out results.
    :param rows: Stage6 row manifest.
    :param decisions: Stage6 per-(candidate, side) decisions (plain dicts).
    :param inputs: One record per input file: ``name`` (relative to the work dir; the ledger by file name),
        ``role``, ``present``, ``sha256``, ``records``.
    """

    ledger: List[LedgerEntry]
    candidates: List[Candidate]
    images: List[Candidate]
    sides: List[SideDecision]
    quality: List[QualityResult]
    holdout: List[HoldoutResult]
    rows: List[RowRecord] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Member:
    """One candidate (``side_key`` None) or one side at one stage.

    :param source: Ledger source.
    :param script: Ledger script or measured script, by stage.
    :param row_id: Candidate id.
    :param side_key: Target side, None for candidate-level membership.
    :param group: Document group.
    """

    source: str
    script: str
    row_id: str
    side_key: Optional[str]
    group: str


# ------------------------------------------------------------------------------------------------------------- loading

def source_label(source: str) -> str:
    """Display name of a ledger source (the legacy-store sources get their public-safe names).

    :param source: Ledger source key.
    :return: Display name.
    """
    return SOURCE_LABELS.get(source, source)


def file_sha256(path: Path) -> str:
    """sha256 of a file, streamed.

    :param path: File.
    :return: Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_record(path: Path, name: str, role: str, records: int) -> Dict[str, Any]:
    """Digest record of one input file.

    :param path: The file.
    :param name: Name shown in reports (relative; never an absolute path).
    :param role: What the file is.
    :param records: Records read from it.
    :return: ``{name, role, present, sha256, records}`` (sha256 and records None when the file is missing).
    """
    present = path.is_file()
    return {"name": name, "role": role, "present": present, "sha256": file_sha256(path) if present else None,
            "records": records if present else None}


def ledger_entry(record: Mapping[str, Any]) -> LedgerEntry:
    """Project one ledger record onto the fields the funnel uses.

    :param record: Ledger record (``unseen_pool.jsonl`` line).
    :return: The entry; missing values become ``unknown`` / ``missing`` / a per-row group.
    """
    flags = record.get("flags") or {}
    row_id = str(record["row_id"])
    held_out = flags.get("held_out")
    return LedgerEntry(row_id=row_id, source=str(record.get("source") or "unknown"),
                       script=str(record.get("script") or "unknown"),
                       document_group=str(record.get("document_group") or f"row:{row_id}"),
                       verdict=str(flags.get("verdict") or "missing"),
                       witnesses=[str(w) for w in flags.get("witnesses") or []],
                       held_out=None if held_out is None else str(held_out),
                       evidence=[key for key in EVIDENCE_KEYS if flags.get(key)])


def load_ledger(pool: Path) -> List[LedgerEntry]:
    """Read the novelty ledger (every verdict).

    :param pool: Ledger JSON lines.
    :return: Entries in file order.
    """
    return [ledger_entry(record) for record in read_jsonl(pool)]


def load_stage_data(work_dir: Path, pool: Path, *, build_outputs: bool = True) -> StageData:
    """Load the ledger and every stage file that exists (a missing stage file reads as empty).

    :param work_dir: The build's work dir.
    :param pool: The novelty ledger.
    :param build_outputs: Also read the stage6 manifest and decisions (False for the build itself, which makes them).
    :return: The loaded data, with a digest record per input file.
    """
    ledger = load_ledger(pool)
    inputs = [input_record(pool, pool.name, "novelty ledger (every verdict)", len(ledger))]
    specs = (("candidates", Candidate, "stage0 candidates and targets"), ("images", Candidate, "stage1 photographs"),
             ("sides", SideDecision, "stage2 side decisions"), ("quality", QualityResult, "stage3 quality results"),
             ("holdout", HoldoutResult, "stage5 hold-outs and split"))
    if build_outputs:
        specs += (("rows", RowRecord, "stage6 row manifest"),)
    loaded: Dict[str, List[Any]] = {}
    for key, cls, role in specs:
        path = paths.stage_path(work_dir, key)
        loaded[key] = list(read_jsonl(path, cls)) if path.is_file() else []
        inputs.append(input_record(path, paths.STAGE_FILES[key], role, len(loaded[key])))
    decisions: List[Dict[str, Any]] = []
    if build_outputs:
        path = work_dir / DECISIONS_FILE
        decisions = list(read_jsonl(path)) if path.is_file() else []
        inputs.append(input_record(path, DECISIONS_FILE, "stage6 decisions", len(decisions)))
    return StageData(ledger=ledger, candidates=loaded["candidates"], images=loaded["images"], sides=loaded["sides"],
                     quality=loaded["quality"], holdout=loaded["holdout"], rows=loaded.get("rows", []),
                     decisions=decisions, inputs=inputs)


# ------------------------------------------------------------------------------------------------------------ identity

class Lookup:
    """Source, ledger script and document group of a candidate id: the ledger first, then the stage0 candidate."""

    def __init__(self, data: StageData) -> None:
        """Index the ledger and the stage0 candidates.

        :param data: Loaded stage data.
        """
        self.ledger = {entry.row_id: entry for entry in data.ledger}
        self.candidates = {cand.row_id: cand for cand in data.candidates}

    def ident(self, row_id: str) -> Tuple[str, str, str]:
        """``(source, ledger script, document group)`` of a candidate.

        :param row_id: Candidate id.
        :return: The triple; ``unknown`` and a per-row group when neither the ledger nor stage0 knows the id.
        """
        entry = self.ledger.get(row_id)
        if entry is not None:
            return entry.source, entry.script, entry.document_group
        cand = self.candidates.get(row_id)
        if cand is not None:
            return cand.source, cand.script or "unknown", cand.document_group
        return "unknown", "unknown", f"row:{row_id}"


def found_images(cand: Optional[Candidate]) -> List[ImageRef]:
    """Photographs of a stage1 candidate that count as found.

    :param cand: Stage1 candidate, or None.
    :return: Its images with a found status.
    """
    return [image for image in cand.images if image.status in FOUND_STATUSES] if cand is not None else []


def cleared(result: HoldoutResult) -> bool:
    """True when a hold-out result lets the side become a row.

    :param result: Stage5 result.
    :return: Not held out and split train or val.
    """
    return not result.held_out and result.split in ROW_SPLITS


def is_rule_recovered(reasons: Iterable[str]) -> bool:
    """True when the reasons of an accepted side decision include a rule-recovery code.

    :param reasons: A stage2 decision's reasons.
    :return: Whether any is in :data:`RULE_RECOVERED_REASONS` (``spill_ratio_ok`` / ``conflict_ratio_ok``).
    """
    return any(reason in RULE_RECOVERED_REASONS for reason in reasons or [])


def review_kind(code: str) -> Optional[str]:
    """Which review class a reason or flag code belongs to.

    :param code: A reason or a ``quality_flags`` entry.
    :return: ``pending_work`` (``review_pending_work:<source>:<work>``), ``pending_row`` (``review_pending_row``),
        ``rejected`` (``review_rejected:<verdict>``), or None for any other code.
    """
    if code == REVIEW_PENDING_ROW:
        return "pending_row"
    if code.startswith(REVIEW_PENDING_WORK + ":"):
        return "pending_work"
    if code.startswith(REVIEW_REJECTED + ":"):
        return "rejected"
    return None


# ---------------------------------------------------------------------------------------------------------- counting

def stage_members(data: StageData) -> Dict[str, List[Member]]:
    """Members of every stage, each stage taken from its own file (see the module docstring).

    :param data: Loaded stage data (``rows`` may be an in-memory manifest).
    :return: ``stage key -> members``.
    """
    lookup = Lookup(data)
    members: Dict[str, List[Member]] = {key: [] for key in STAGE_KEYS}
    for entry in data.ledger:
        member = Member(entry.source, entry.script, entry.row_id, None, entry.document_group)
        members["inventoried"].append(member)
        if entry.verdict in VERDICT_STAGES:
            members[VERDICT_STAGES[entry.verdict]].append(member)
    targets = {cand.row_id: [target.side_key for target in cand.targets] for cand in data.candidates}
    for cand in data.candidates:
        source, script, group = lookup.ident(cand.row_id)
        members["target_extracted"] += [Member(source, script, cand.row_id, side, group) for side in targets[cand.row_id]]
    for cand in data.images:
        if found_images(cand):
            source, script, group = lookup.ident(cand.row_id)
            members["image_found"] += [Member(source, script, cand.row_id, side, group)
                                       for side in targets.get(cand.row_id, [])]
    for decision in data.sides:
        if decision.outcome == "accepted":
            source, script, group = lookup.ident(decision.row_id)
            members["side_verified"].append(Member(source, script, decision.row_id, decision.side_key, group))
    measured = {(result.row_id, result.side_key): result.script for result in data.quality}
    for result in data.quality:
        if result.ok:
            source, _, group = lookup.ident(result.row_id)
            members["quality_passed"].append(Member(source, result.script, result.row_id, result.side_key, group))
    for result in data.holdout:
        source, script, group = lookup.ident(result.row_id)
        script = measured.get((result.row_id, result.side_key), script)
        key = "holdout_cleared" if cleared(result) else "held_out_builder"
        members[key].append(Member(source, script, result.row_id, result.side_key, group))
    for row in data.rows:
        member = Member(row.source, row.script, row.row_id, row.side_key, row.document_group)
        members["rows"].append(member)
        members["val_rows" if row.split == "val" else "train_rows"].append(member)
    return members


def count_cell(members: Iterable[Member], with_sides: bool) -> Dict[str, Optional[int]]:
    """Distinct candidates, sides and document groups of some members.

    :param members: Members of one stage (one slice).
    :param with_sides: Count sides (False for ledger stages, where sides are unknown).
    :return: ``{candidates, sides, groups}``; ``sides`` None when not counted.
    """
    items = list(members)
    return {"candidates": len({m.row_id for m in items}),
            "sides": len({(m.row_id, m.side_key) for m in items if m.side_key is not None}) if with_sides else None,
            "groups": len({m.group for m in items})}


def ordered_scripts(found: Iterable[str]) -> List[str]:
    """The schema's scripts first, then any other script value seen.

    :param found: Script values present.
    :return: Ordered list.
    """
    present = set(found)
    return [script for script in SCRIPTS if script in present] + sorted(present - set(SCRIPTS))


def stage_counts(data: StageData) -> Dict[str, Any]:
    """Stage counts at every aggregation level.

    :param data: Loaded stage data (``rows`` may be an in-memory manifest).
    :return: ``{total, by_source, by_script, by_source_script, sources, scripts}``; each cell is
        :func:`count_cell` of one stage.
    """
    members = stage_members(data)
    levels = {stage.key: stage.level != "ledger" for stage in STAGES}
    sources = sorted({m.source for items in members.values() for m in items})
    scripts = ordered_scripts(m.script for items in members.values() for m in items)
    sliced: Dict[str, Dict[Tuple[str, str], List[Member]]] = {}
    for key, items in members.items():
        buckets: Dict[Tuple[str, str], List[Member]] = defaultdict(list)
        for m in items:
            buckets[(m.source, m.script)].append(m)
        sliced[key] = buckets

    def select(key: str, source: Optional[str], script: Optional[str]) -> Dict[str, Optional[int]]:
        """Count one stage over the members of a source and/or script (None = any).

        :param key: Stage key.
        :param source: Source filter.
        :param script: Script filter.
        :return: The cell.
        """
        chosen = [m for (src, scr), items in sliced[key].items() if source in (None, src) and script in (None, scr)
                  for m in items]
        return count_cell(chosen, levels[key])

    return {"total": {key: select(key, None, None) for key in STAGE_KEYS},
            "by_source": {src: {key: select(key, src, None) for key in STAGE_KEYS} for src in sources},
            "by_script": {scr: {key: select(key, None, scr) for key in STAGE_KEYS} for scr in scripts},
            "by_source_script": {src: {scr: {key: select(key, src, scr) for key in STAGE_KEYS} for scr in scripts}
                                 for src in sources},
            "sources": sources, "scripts": scripts}


def chain_violations(cells: Mapping[str, Mapping[str, Optional[int]]], label: str) -> List[str]:
    """Stages of :data:`MONOTONE_CHAIN` whose count exceeds the stage before them.

    :param cells: ``stage key -> cell`` of one slice.
    :param label: Slice name for the messages.
    :return: One message per violation.
    """
    out: List[str] = []
    for unit in ("candidates", "sides", "groups"):
        previous: Optional[Tuple[str, int]] = None
        for key in MONOTONE_CHAIN:
            value = (cells.get(key) or {}).get(unit)
            if value is None:
                continue
            if previous is not None and value > previous[1]:
                out.append(f"{label}: {unit} at {STAGE_LABELS[key]} ({value}) exceed {STAGE_LABELS[previous[0]]} "
                           f"({previous[1]})")
            previous = (key, value)
    return out


def monotone_violations(counts: Mapping[str, Any]) -> List[str]:
    """Every stage count that exceeds the stage before it, over all sources and per source.

    :param counts: :func:`stage_counts` result.
    :return: Messages (empty when every chain is monotone).
    """
    out = chain_violations(counts["total"], "all sources")
    for source, cells in counts["by_source"].items():
        out += chain_violations(cells, f"source {source_label(source)}")
    return out


# ------------------------------------------------------------------------------------------------------------- trace

def image_reasons(cand: Optional[Candidate]) -> List[str]:
    """Why a candidate has no photograph found.

    :param cand: Its stage1 record, or None.
    :return: ``not_in_stage1`` | ``no_image_listed`` | one ``image_<status>`` per distinct status.
    """
    if cand is None:
        return ["not_in_stage1"]
    if not cand.images:
        return ["no_image_listed"]
    return sorted({f"image_{image.status}" for image in cand.images})


def trace(data: StageData, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Follow every (candidate, side) of stage0 down the stages; stop at the first stage it does not pass.

    A candidate without a target gives one record with ``side_key`` ``""`` at stage ``novel``. The stage reached is
    one of ``novel``, ``target_extracted``, ``image_found``, ``side_verified``, ``quality_passed``,
    ``holdout_cleared``; ``reasons`` say why the record stopped there (empty at ``holdout_cleared``, where the build's
    own checks take over). A key listed twice in a stage file counts with its last record.

    :param data: Loaded stage data.
    :param limit: Only the first N stage0 candidates.
    :return: Records ``{row_id, side_key, source, document_group, ledger_script, verdict, stage, reasons,
        side_outcome, image_sha256, measured_script, quality_flags, split, holdout_reasons, rnl, rule_recovered}``
        (``rule_recovered``: the side was accepted by the spill or conflict ratio rule, :func:`is_rule_recovered`).
    """
    lookup = Lookup(data)
    stage1 = {cand.row_id: cand for cand in data.images}
    sides = {(d.row_id, d.side_key): d for d in data.sides}
    quality = {(q.row_id, q.side_key): q for q in data.quality}
    holdout = {(h.row_id, h.side_key): h for h in data.holdout}
    out: List[Dict[str, Any]] = []
    candidates = data.candidates if limit is None else data.candidates[:limit]
    for cand in candidates:
        source, script, group = lookup.ident(cand.row_id)
        entry = lookup.ledger.get(cand.row_id)
        base = {"row_id": cand.row_id, "source": source, "document_group": group, "ledger_script": script,
                "verdict": entry.verdict if entry is not None else "not_in_ledger", "side_outcome": None,
                "image_sha256": None, "measured_script": None, "quality_flags": [], "split": None,
                "holdout_reasons": [], "rnl": bool(cand.rnl), "rule_recovered": False}
        if not cand.targets:
            out.append({**base, "side_key": "", "stage": "novel", "reasons": ["no_target"]})
            continue
        found = found_images(stage1.get(cand.row_id))
        for target in cand.targets:
            key = (cand.row_id, target.side_key)
            out.append(_trace_side({**base, "side_key": target.side_key}, found, stage1.get(cand.row_id),
                                   sides.get(key), quality.get(key), holdout.get(key)))
    return out


def _trace_side(rec: Dict[str, Any], found: Sequence[ImageRef], stage1: Optional[Candidate],
                decision: Optional[SideDecision], result: Optional[QualityResult],
                hold: Optional[HoldoutResult]) -> Dict[str, Any]:
    """Advance one side record through stages 1-5 (see :func:`trace`).

    :param rec: The record so far (``side_key`` set).
    :param found: Found photographs of the candidate.
    :param stage1: Its stage1 record, or None.
    :param decision: Its stage2 decision, or None.
    :param result: Its stage3 result, or None.
    :param hold: Its stage5 result, or None.
    :return: The finished record.
    """
    rec.update(stage="target_extracted", reasons=[])
    if not found:
        rec["reasons"] = image_reasons(stage1)
        return rec
    rec["stage"] = "image_found"
    if decision is None:
        rec["reasons"] = ["no_side_decision"]
        return rec
    rec["side_outcome"] = decision.outcome
    if decision.outcome != "accepted":
        rec["reasons"] = list(decision.reasons) or [f"side_{decision.outcome}"]
        return rec
    rec.update(stage="side_verified", image_sha256=decision.image_sha256,
               rule_recovered=is_rule_recovered(decision.reasons))
    if result is None:
        rec["reasons"] = ["no_quality_record"]
        return rec
    rec["measured_script"] = result.script
    if not result.ok:
        rec["reasons"] = list(result.reasons) or ["quality_failed"]
        return rec
    rec.update(stage="quality_passed", quality_flags=list(result.reasons))
    if hold is None:
        rec["reasons"] = ["no_holdout_record"]
        return rec
    rec.update(split=hold.split, rnl=rec["rnl"] or bool(hold.rnl))
    if not cleared(hold):
        rec["reasons"] = list(hold.reasons) or ["holdout_excluded"]
        return rec
    rec.update(stage="holdout_cleared", holdout_reasons=list(hold.reasons))
    return rec


# ----------------------------------------------------------------------------------------------------------- reasons

def reason_code(reason: str) -> str:
    """A hold-out reason without its document id (``benchmark:pgp131:<id>`` -> ``benchmark:pgp131``).

    :param reason: Reason code.
    :return: The first two ``:`` parts when there are three or more, else the reason.
    """
    parts = reason.split(":", 2)
    return ":".join(parts[:2]) if len(parts) == 3 else reason


def ledger_reasons(entry: LedgerEntry) -> List[str]:
    """Reason codes of a ledger exclusion: the held-out reason, or the witness datasets.

    :param entry: A held-out, previously trained or uncertain-history entry.
    :return: Codes (whitespace folded to ``_``); when no witness dataset is listed, the non-empty evidence lists, else
        ``text_share`` (``text_share_uncertain`` for an uncertain-history entry).
    """
    if entry.verdict == "held_out":
        return ["_".join(entry.held_out.split())] if entry.held_out else ["held_out_unspecified"]
    if entry.witnesses:
        return list(dict.fromkeys(entry.witnesses))
    fallback = "text_share_uncertain" if entry.verdict == "uncertain_history" else "text_share"
    return list(entry.evidence) or [fallback]


def nest(counter: Mapping[Tuple[str, str], int]) -> Dict[str, Dict[str, int]]:
    """``{(reason, source): n}`` -> ``{reason: {source: n}}``.

    :param counter: Flat counts.
    :return: Nested counts.
    """
    out: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (reason, source), n in counter.items():
        out[reason][source] = n
    return dict(out)


def drop_reasons(data: StageData, traced: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Reason counts per stage and source: why candidates or sides did not reach each stage.

    Ledger stages count candidates once per reason; builder stages count sides once per reason (a side with two
    reasons counts in both). Hold-out reasons are counted without their document ids. Sides that cleared the
    hold-outs but are not rows take the build's reasons from the stage6 decisions, else ``not_built``; the build's
    review reasons are counted without their work (``review_pending_work:literature``) and verdict
    (``review_rejected:wrong_side``), which :func:`review_counts` lists.

    :param data: Loaded stage data.
    :param traced: :func:`trace` records.
    :return: ``stage key -> reason -> source -> count``.
    """
    table: Dict[str, Counter] = defaultdict(Counter)
    for entry in data.ledger:
        stage = VERDICT_STAGES.get(entry.verdict)
        if stage in ("held_out_ledger", "previously_trained", "uncertain_history"):
            for reason in ledger_reasons(entry):
                table[stage][(reason, entry.source)] += 1
    in_stage0 = {cand.row_id for cand in data.candidates}
    for entry in data.ledger:
        if entry.verdict == "novel" and entry.row_id not in in_stage0:
            table["target_extracted"][("not_in_stage0", entry.source)] += 1
    rows = {(row.row_id, row.side_key) for row in data.rows}
    built = {(d.get("row_id"), d.get("side_key")): d for d in data.decisions}
    for rec in traced:
        if rec["stage"] == "holdout_cleared":
            key = (rec["row_id"], rec["side_key"])
            if key in rows:
                continue
            prior = built.get(key) or {}
            reasons = list(prior.get("reasons") or []) if prior.get("stage") != "row" else []
            target, reasons = "rows", reasons or ["not_built"]
        else:
            target, reasons = NEXT_STAGE[rec["stage"]], list(rec["reasons"])
        if target in ("held_out_builder", "rows"):
            reasons = [reason_code(reason) for reason in reasons]
        for reason in dict.fromkeys(reasons):
            table[target][(reason, rec["source"])] += 1
    for row in data.rows:
        if row.split == "val":
            table["val_rows"][("validation_split", row.source)] += 1
    return {stage: nest(counter) for stage, counter in table.items()}


def review_queue(data: StageData) -> Dict[str, Dict[str, int]]:
    """Side placements sent to review (stage2 ``needs_review``), by reason and source.

    :param data: Loaded stage data.
    :return: ``reason -> source -> sides``.
    """
    lookup = Lookup(data)
    counter: Counter = Counter()
    for decision in data.sides:
        if decision.outcome == "needs_review":
            source = lookup.ident(decision.row_id)[0]
            for reason in dict.fromkeys(decision.reasons or ["unspecified"]):
                counter[(reason, source)] += 1
    return nest(counter)


def review_counts(decisions: Sequence[Mapping[str, Any]], rows: Sequence[RowRecord]) -> Dict[str, Any]:
    """The human-review gate in numbers: what waits on a reviewer, what a reviewer rejected, what got in unreviewed.

    Counted from the build's own outputs: its per-(candidate, side) decisions (:func:`trace` records plus the build's
    reasons) and its row manifest. A side is *pending work review* / *pending row review* / *rejected by review* when
    it cleared every stage and the build's other checks and carries the matching review reason (a side with two review
    reasons counts in each column); *admitted with flags* counts rows whose ``quality_flags`` hold a
    ``review_pending_*`` code (policy ``flag``). Rule-recovered sides (:func:`is_rule_recovered`) are split into
    those that became rows and those that did not, with the number of the latter that the review gate stopped.

    :param decisions: Stage6 decision records (empty before the first build: every count is then 0).
    :param rows: Stage6 row manifest.
    :return: ``{sources, by_source, total, pending_works}``: ``by_source`` and ``total`` map each key of
        :data:`REVIEW_COLUMNS` to a count (sides, or rows for ``admitted_flagged``); ``pending_works`` maps each
        pending ``<source>:<work>`` key to the sides waiting on it.
    """
    cells: Dict[str, Counter] = defaultdict(Counter)
    works: Counter = Counter()
    for row in rows:
        cells[row.source]["admitted_flagged"] += int(any(review_kind(flag) in ("pending_work", "pending_row")
                                                         for flag in row.quality_flags))
    for decision in decisions:
        cell = cells[str(decision.get("source") or "unknown")]
        built = decision.get("stage") == "row"
        kinds = set()
        if decision.get("stage") == "holdout_cleared":
            for reason in dict.fromkeys(decision.get("reasons") or []):
                kind = review_kind(reason)
                if kind is not None:
                    kinds.add(kind)
                if kind == "pending_work":
                    works[reason.split(":", 1)[1]] += 1
        for kind in kinds:
            cell[kind] += 1
        if decision.get("rule_recovered"):
            if built:
                cell["rule_admitted"] += 1
            else:
                cell["rule_excluded"] += 1
                cell["rule_excluded_by_review"] += int(bool(kinds))
    sources = sorted(cells)
    keys = [key for key, _ in REVIEW_COLUMNS]
    by_source = {src: {key: cells[src][key] for key in keys} for src in sources}
    return {"sources": sources, "by_source": by_source,
            "total": {key: sum(by_source[src][key] for src in sources) for key in keys},
            "pending_works": dict(sorted(works.items()))}


def duplicate_keys(data: StageData) -> Dict[str, int]:
    """Keys listed more than once per stage file (the last record counts).

    :param data: Loaded stage data.
    :return: ``stage file key -> number of extra records``.
    """
    def extra(keys: Iterable[Any]) -> int:
        """Records beyond the first per key.

        :param keys: One key per record.
        :return: Extra records.
        """
        return sum(n - 1 for n in Counter(keys).values())

    return {"candidates": extra(c.row_id for c in data.candidates), "images": extra(c.row_id for c in data.images),
            "sides": extra((d.row_id, d.side_key) for d in data.sides),
            "quality": extra((q.row_id, q.side_key) for q in data.quality),
            "holdout": extra((h.row_id, h.side_key) for h in data.holdout),
            "rows": extra(row.stem for row in data.rows)}


# -------------------------------------------------------------------------------------------------------- the funnel

def boilerplate_counts(holdout: Iterable[HoldoutResult]) -> Dict[str, Any]:
    """What the hold-out's boilerplate exception did, from the results' ``flags``.

    :param holdout: Stage5 results.
    :return: ``{admitted, flagged_sides, matches_discounted: {"boilerplate:<set>": n}}``: sides admitted that the
        strict text rule holds out, sides carrying any flag, and the discounted matches per GT set.
    """
    results = list(holdout)
    flagged = [r for r in results if r.flags]
    matches = Counter(":".join(flag.split(":")[:2]) for r in results for flag in r.flags)
    return {"admitted": sum(1 for r in flagged if not r.held_out), "flagged_sides": len(flagged),
            "matches_discounted": dict(sorted(matches.items()))}


def load_text_rule(work_dir: Path) -> Optional[Dict[str, Any]]:
    """The text rule stage 5 recorded (``paths.TEXT_RULE_FILE``), or None when the sidecar is absent.

    :param work_dir: The build's work dir.
    :return: The sidecar's record.
    """
    path = work_dir / paths.TEXT_RULE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def funnel_from_data(data: StageData, *, now: Optional[datetime] = None,
                     text_rule: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The funnel of loaded stage data.

    :param data: Loaded stage data.
    :param now: Generation time (default: now, UTC).
    :param text_rule: The text rule record stage 5 wrote (:func:`load_text_rule`), if any.
    :return: ``{generated_at, inputs, stages, counts, reasons, review_queue, review, boilerplate, text_rule,
        other_verdicts, duplicates, monotone_violations}`` (JSON-serialisable; ``review`` is :func:`review_counts`,
        ``boilerplate`` is :func:`boilerplate_counts`).
    """
    counts = stage_counts(data)
    when = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"generated_at": when, "inputs": data.inputs,
            "stages": [{"key": s.key, "label": s.label, "level": s.level, "rendered": s.rendered} for s in STAGES],
            "counts": counts, "reasons": drop_reasons(data, trace(data)), "review_queue": review_queue(data),
            "review": review_counts(data.decisions, data.rows),
            "boilerplate": boilerplate_counts(data.holdout), "text_rule": dict(text_rule) if text_rule else None,
            "other_verdicts": dict(Counter(e.verdict for e in data.ledger if e.verdict not in VERDICT_STAGES)),
            "duplicates": duplicate_keys(data), "monotone_violations": monotone_violations(counts)}


def compute_funnel(work_dir: Path, pool: Path, *, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Load the ledger, the stage files and the text-rule sidecar and compute the funnel.

    :param work_dir: The build's work dir.
    :param pool: The novelty ledger.
    :param now: Generation time (default: now, UTC).
    :return: See :func:`funnel_from_data`.
    """
    return funnel_from_data(load_stage_data(work_dir, pool), now=now, text_rule=load_text_rule(work_dir))


def text_rule_lines(record: Optional[Mapping[str, Any]], boilerplate: Optional[Mapping[str, Any]]) -> List[str]:
    """Markdown lines describing the text hold-out rule and its boilerplate exception.

    :param record: The sidecar record (:func:`load_text_rule`), or None.
    :param boilerplate: :func:`boilerplate_counts`, or None.
    :return: Bullet lines.
    """
    admitted = (boilerplate or {}).get("admitted", 0)
    discounted = (boilerplate or {}).get("matches_discounted") or {}
    if record is None:
        rule = ["- rule not recorded (stage 5 ran without the sidecar)" + (f"; {admitted} side(s) carry boilerplate "
                "flags and are admitted" if admitted else "")]
    elif record.get("name") == "strict":
        rule = ["- rule `strict`: every shingle shared with a benchmark document counts (no boilerplate exception)."]
    else:
        rule = [f"- rule `{record.get('name')}`: a shared shingle held by at least {record.get('common_df')} "
                f"independent document groups of the pool ({record.get('pool_groups')} groups counted) is "
                "boilerplate and does not count towards the shared-shingle thresholds; a side still holds out when "
                f"its longest shared run spans at least {record.get('long_run_letters')} letters or at least "
                f"{record.get('share_threshold')} of its shingles are shared with one benchmark set. Identity "
                "hold-outs are unchanged.",
                f"- sides admitted that the strict rule holds out: {admitted}"
                + ("; matches discounted: " + ", ".join(f"{k.split(':', 1)[1]} {n}" for k, n in discounted.items())
                   if discounted else "")]
    return ["", "## Text hold-out rule", ""] + rule


# ---------------------------------------------------------------------------------------------------------- rendering

def _md(value: Any) -> str:
    """A value as one Markdown table cell.

    :param value: Any value (None -> empty).
    :return: Text with ``|`` escaped and whitespace folded.
    """
    return "" if value is None else " ".join(str(value).split()).replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], numeric_from: int) -> List[str]:
    """A Markdown table.

    :param header: Column names.
    :param rows: Cell values.
    :param numeric_from: Index of the first right-aligned column.
    :return: Lines.
    """
    align = ["---" if i < numeric_from else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(_md(h) for h in header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(_md(cell) for cell in row) + " |" for row in rows]
    return lines


def stage_lines(cells_for: Mapping[str, Mapping[str, Mapping[str, Optional[int]]]],
                columns: Sequence[str]) -> List[List[Any]]:
    """Rows of a stage table: one per rendered stage (two for the target stages: candidates and sides).

    :param cells_for: ``column -> stage key -> cell``.
    :param columns: Column order.
    :return: ``[stage label, unit, "n (g)" per column]``.
    """
    lines: List[List[Any]] = []
    for stage in STAGES:
        if not stage.rendered:
            continue
        units = {"ledger": ("candidates",), "target": ("candidates", "sides"), "side": ("sides",)}[stage.level]
        for unit in units:
            shown = "rows" if stage.key in ("val_rows", "train_rows") else unit
            lines.append([stage.label, shown] + [f"{cells_for[col][stage.key][unit]} "
                                                f"({cells_for[col][stage.key]['groups']})" for col in columns])
    return lines


def review_lines(review: Mapping[str, Any]) -> List[str]:
    """The human-review gate section of ``FUNNEL.md``.

    :param review: :func:`review_counts` result.
    :return: Markdown lines (a heading, an explanation, the per-source table and the pending works).
    """
    out = ["", "## Human review gate", "",
           "Sides that cleared every stage and the build's own checks, then met the review gate (a private reviews "
           "file: approved works, approved rows, review policy). Counts are sides, except *admitted with flags*, "
           "which counts rows; a side with two review reasons counts in each column. *Pending*: no approval yet "
           "(excluded under policy `require`, which the reviews file's `required_works` imposes work by work; "
           "admitted with a `review_pending_*` entry in `quality_flags` under `flag`); *rejected*: a reviewer's "
           "verdict other than `ok`. *Rule-recovered*: accepted at stage 2 by the spill or conflict ratio rule; "
           "*excluded* counts those that did not become rows at any stage.", ""]
    keys = [key for key, _ in REVIEW_COLUMNS]
    rows = [[source_label(src)] + [review["by_source"][src][key] for key in keys] for src in review["sources"]]
    rows.append(["all"] + [review["total"][key] for key in keys])
    out += _table(["source"] + [label for _, label in REVIEW_COLUMNS], rows, 1)
    works = review["pending_works"]
    out += ["", "Pending work review by work (the `approved_works` entry that clears it), sides:", ""]
    out += _table(["work", "sides"], [[work, works[work]] for work in sorted(works, key=lambda w: (-works[w], w))],
                  1) if works else ["(no work awaits review)"]
    return out


def render(funnel: Mapping[str, Any]) -> str:
    """``FUNNEL.md`` of a computed funnel.

    :param funnel: :func:`compute_funnel` result.
    :return: Markdown.
    """
    counts = funnel["counts"]
    scripts, sources = counts["scripts"], counts["sources"]
    out = ["# Novel-rows funnel", "",
           f"INTERNAL. Generated {funnel['generated_at']} by `python -m src.finetuning.qwen_hebrew.novel_build.funnel` "
           "from the files below; every number is computed from them, none is typed.", "",
           "## Inputs", ""]
    out += _table(["file", "role", "sha256", "records"],
                  [[i["name"], i["role"], i["sha256"] or "missing", i["records"] if i["present"] else "missing"]
                   for i in funnel["inputs"]], 3)
    out += ["", "Cells are `n (g)`: n candidates (ledger stages; the candidate lines of target extracted and image "
                "found) or sides (the side lines; side verified onward; validation and training rows are sides), "
                "g distinct document groups. Script: the ledger's script up to side verified, the script measured on "
                "the final target from quality passed on.", "",
            "## All sources", ""]
    columns = scripts + ["all"]
    cells = {**{scr: counts["by_script"][scr] for scr in scripts}, "all": counts["total"]}
    out += _table(["stage", "unit"] + columns, stage_lines(cells, columns), 2)
    for source in sources:
        out += ["", f"## {source_label(source)}", ""]
        cells = {**counts["by_source_script"][source], "all": counts["by_source"][source]}
        out += _table(["stage", "unit"] + columns, stage_lines(cells, columns), 2)
    out += ["", "## Drop reasons per stage", "",
            "Why candidates (ledger stages) or sides (builder stages) did not reach a stage; one count per reason, so "
            "a side with two reasons counts twice. Hold-out reasons without their document ids.", ""]
    reason_sources = sorted({src for by_reason in funnel["reasons"].values() for by_src in by_reason.values()
                             for src in by_src})
    rows = []
    for key, label in REASON_STAGES:
        by_reason = funnel["reasons"].get(key, {})
        for reason in sorted(by_reason, key=lambda r: (-sum(by_reason[r].values()), r)):
            rows.append([label, reason] + [by_reason[reason].get(src, 0) for src in reason_sources]
                        + [sum(by_reason[reason].values())])
    out += _table(["stage", "reason"] + [source_label(s) for s in reason_sources] + ["all"], rows, 2) if rows \
        else ["(no reasons)"]
    out += text_rule_lines(funnel.get("text_rule"), funnel.get("boilerplate"))
    out += ["", "## Review queue", "", "Side placements sent to review (stage2 `needs_review`), sides by reason.", ""]
    queue = funnel["review_queue"]
    queue_sources = sorted({src for by_src in queue.values() for src in by_src})
    rows = [[reason] + [queue[reason].get(src, 0) for src in queue_sources] + [sum(queue[reason].values())]
            for reason in sorted(queue, key=lambda r: (-sum(queue[r].values()), r))]
    out += _table(["reason"] + [source_label(s) for s in queue_sources] + ["all"], rows, 1) if rows \
        else ["(empty)"]
    uncertain = counts["total"]["uncertain_history"]
    out += ["", f"Ledger review queue: {uncertain['candidates']} uncertain-history candidates "
                f"({uncertain['groups']} document groups), neither counted as novel nor discarded."]
    out += review_lines(funnel["review"])
    out += ["", "## Consistency", ""]
    violations = funnel["monotone_violations"]
    out += ([f"- NOT MONOTONE: {v}" for v in violations] if violations else
            ["- Every stage count is at most the count of the stage before it (candidates, sides and document "
             "groups; all sources and per source)."])
    duplicates = {k: v for k, v in funnel["duplicates"].items() if v}
    out.append(f"- Keys listed more than once (the last record counts): {duplicates}" if duplicates
               else "- No key is listed twice in any stage file.")
    if funnel["other_verdicts"]:
        out.append(f"- Ledger records with another verdict (counted in inventoried only): {funnel['other_verdicts']}")
    missing = [i["name"] for i in funnel["inputs"] if not i["present"]]
    if missing:
        out.append(f"- Missing input files (read as empty): {', '.join(missing)}")
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: write ``FUNNEL.md`` from the ledger and the stage files.

    :param argv: Arguments (default: ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Write the novel-rows funnel (FUNNEL.md) from the novelty ledger "
                                                 "and the stage files.")
    parser.add_argument("--work-dir", type=Path, default=paths.DEFAULT_WORK_DIR, help="The build's work dir.")
    parser.add_argument("--pool", type=Path, default=paths.DEFAULT_POOL, help="The novelty ledger (every verdict).")
    parser.add_argument("--out", type=Path, default=None, help="Output Markdown (default: <work dir>/FUNNEL.md).")
    args = parser.parse_args(argv)
    funnel = compute_funnel(args.work_dir, args.pool)
    out = args.out if args.out is not None else args.work_dir / FUNNEL_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(funnel), encoding="utf-8")
    total = funnel["counts"]["total"]
    print(f"wrote {out}: novel {total['novel']['candidates']} candidates, rows {total['rows']['sides']} "
          f"({total['rows']['groups']} groups); {len(funnel['monotone_violations'])} monotone violation(s)")


if __name__ == "__main__":
    main()
