#!/usr/bin/env python3
"""
Merge duplicate Institution / Place nodes and remove non-holding edges (dry run by default).

Import step 4b: run after ``merged_shelfmarks_import.py`` and before
``geocode_places.py``. The rules live in ``institution_merges.py``; the edge property
merge policy is genizah_search's ``docs/kg_consolidation/census_2026-10-12/edge_merge_policy.md``
(the same rules as its ``scripts/kg/consolidate_scholars.py``).

* **Dry run** (default, read-only): reads the rule nodes and their edges in one read
  transaction, plans every operation in order (edge removals, institution merges, place
  merges, edge moves, then deletion of the Institution nodes the removals left without
  edges) and writes to ``--out``::

      summary.json   counts and the plan's sha256
      plan.json      the operations --apply re-checks and executes
      edges.tsv      one row per edge removed, moved or folded and per node deleted
      review.tsv     rules that could not be planned (missing target, NULL-vs-list data_sources)

* **Apply** (``--apply --plan DIR``): re-plans from the live graph and refuses unless the
  result equals ``DIR/plan.json`` (the graph has not changed since the dry run). Then runs one
  write transaction per operation, appending the full before-state of every node and edge it
  touches to ``DIR/changelog_<UTC>.jsonl`` first, and checks afterwards that the node and
  relationship totals moved as planned and that a new plan is empty.
* **Revert** (``--revert DIR/changelog_<UTC>.jsonl``): undoes the committed operations of that
  log, newest first, and appends a ``reverted`` record per operation, so an interrupted revert
  can be resumed.

The pass is idempotent: on a graph it has already cleaned it plans nothing. After a failed or
partial apply, run a new dry run and apply that plan.

Usage (from the repo root)::

    PYTHONPATH=. python src/datasets/indexing/neo4j/consolidate_institutions.py [--out DIR]
    PYTHONPATH=. python src/datasets/indexing/neo4j/consolidate_institutions.py --apply --plan DIR
    PYTHONPATH=. python src/datasets/indexing/neo4j/consolidate_institutions.py --revert DIR/changelog_<UTC>.jsonl

Connection settings come from ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD`` /
``NEO4J_DATABASE`` (environment or the repo ``.env``; ``--uri`` / ``--database`` override) and
are never printed. Keep dry-run directories (default ``docs/kg_consolidation/institutions_<UTC>/``):
their change logs are the only way to revert, so ``--apply`` refuses a plan directory under a
temporary directory unless ``--allow-temp-plan`` is given. Never ``--apply`` against production
without Isaac's approval of the dry run.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.datasets.indexing.neo4j.institution_merges import (  # noqa: E402
    EDGE_MOVES, EDGE_REMOVALS, INSTITUTION_MERGES, KEEP_AFFILIATIONS, PLACE_MERGES, EdgeMove, EdgeRemoval,
    InstitutionMerge)

#: plan.json format; --apply refuses any other.
PLAN_FORMAT = 1
#: Labels whose nodes are matched by ``canonical_shelfmark`` instead of ``name``.
SHELFMARK_LABELS = ("Fragment",)
#: Label shown for a node in reports, in order of preference.
DISPLAY_LABELS = ("Fragment", "Person", "Scholar", "BiblicalPerson", "Institution", "Place", "BookArticle")
#: Relationship property per merged node label that records the names merged into it.
ALIAS_PROPS = {"Institution": "aliases", "Place": "name_variants"}
#: Properties that are part of an edge's identity (different values = different facts).
IDENTITY_PROPS = ("pgpid", "role", "pages")
SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: Placeholder id of an edge created by an earlier action of the plan: ``new:<op index>:<n>``.
PLACEHOLDER = re.compile(r"^new:\d+:\d+$")
#: Directories whose contents the OS may delete: --apply refuses a plan directory under one.
TEMP_ROOTS = ("/tmp", "/private/tmp", "/var/tmp", "/var/folders", "/private/var/folders")
DEFAULT_OUT_ROOT = project_root / "docs" / "kg_consolidation"

# Edge property merge policy (edge_merge_policy.md section 3). Everything not listed keeps the primary's value.
UNION_PROPS = ("data_sources", "source_books")
ALL_VALUES_LISTS = {"book_uuid": "book_uuids", "page_uuid": "page_uuids", "sub_collection": "sub_collections"}
ALT_LISTS = {"evidence": "evidence_alt", "raw_citation": "raw_citation_alt", "location": "location_alt",
             "notes": "notes_alt"}
RANKED_PROPS = {"confidence": ("low", "medium", "high"),
                "transcription_extent": ("Unknown", "Partial", "Full"),
                "translation_extent": ("Unknown", "Partial", "Full")}
OR_FLAGS = ("certain", "deceased", "has_transcription", "has_translation", "has_discussion", "primary")
AND_FLAGS = ("uncertain",)
#: Properties that describe where the primary edge was extracted; they stay together, never filled from another edge.
#: Any other property only a non-primary edge carries (PGP's HELD_AT ``sub_collection``) is kept, not dropped.
PROVENANCE_PROPS = ("book_uuid", "page_uuid", "evidence", "evidence_page", "source")


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------
@dataclass
class Node:
    """A node of the snapshot.

    :param eid: elementId.
    :param labels: Node labels.
    :param props: Node properties (only the key properties for nodes outside the rules).
    :param unloaded: Number of incident edges that were not read into the snapshot.
    :param stub: True for a node outside the rules (an edge endpoint); the plan never changes it.
    """

    eid: str
    labels: List[str]
    props: Dict[str, Any]
    unloaded: int = 0
    stub: bool = False

    @property
    def label(self) -> str:
        """The label used in reports and business keys."""
        return next((lab for lab in DISPLAY_LABELS if lab in self.labels), self.labels[0] if self.labels else "")

    @property
    def key(self) -> str:
        """Business key: ``canonical_shelfmark`` for a Fragment, else ``name``."""
        prop = "canonical_shelfmark" if self.label in SHELFMARK_LABELS else "name"
        return str(self.props.get(prop) or "")


@dataclass
class Edge:
    """A relationship of the snapshot.

    :param eid: elementId (or a ``new:`` placeholder for one the plan creates).
    :param type: Relationship type.
    :param start: Start node elementId.
    :param end: End node elementId.
    :param props: Relationship properties.
    """

    eid: str
    type: str
    start: str
    end: str
    props: Dict[str, Any]


class MemoryGraph:
    """The part of the graph the rules touch, held in memory.

    The planner applies each planned action to it at once, so later operations see the effect of
    earlier ones; the tests use it as a stand-in for Neo4j.

    :param nodes: Nodes of the snapshot.
    :param edges: Relationships of the snapshot.
    """

    def __init__(self, nodes: Iterable[Node], edges: Iterable[Edge]) -> None:
        self.nodes: Dict[str, Node] = {n.eid: n for n in nodes}
        self.edges: Dict[str, Edge] = {}
        self._by_node: Dict[str, Set[str]] = {eid: set() for eid in self.nodes}
        self._created = 0
        for edge in edges:
            self._add(edge)

    def _add(self, edge: Edge) -> None:
        """Index one edge.

        :param edge: The relationship.
        """
        self.edges[edge.eid] = edge
        self._by_node.setdefault(edge.start, set()).add(edge.eid)
        self._by_node.setdefault(edge.end, set()).add(edge.eid)

    def find(self, label: str, name: str) -> Optional[Node]:
        """Return the non-stub node with ``label`` and ``name``.

        :param label: Node label.
        :param name: Value of ``name``.
        :return: The node, or None.
        """
        return next((n for n in self.nodes.values()
                     if not n.stub and label in n.labels and n.props.get("name") == name), None)

    def incident(self, eid: str) -> List[Edge]:
        """Loaded edges touching a node, in a stable order.

        :param eid: Node elementId.
        :return: The edges.
        """
        return sorted((self.edges[e] for e in self._by_node.get(eid, ())), key=lambda e: eid_sort_key(e.eid))

    def degree(self, eid: str) -> int:
        """Total number of edges on a node, loaded or not.

        :param eid: Node elementId.
        :return: The degree.
        """
        return len(self._by_node.get(eid, ())) + self.nodes[eid].unloaded

    def apply(self, action: Mapping[str, Any]) -> Optional[str]:
        """Apply one action with the same meaning :func:`run_action` gives it in Neo4j.

        :param action: An action dict.
        :return: elementId of a created edge (its ``ref`` placeholder when it has one) or node.
        """
        do = action["do"]
        if do == "create_edge":
            self._created += 1
            eid = action.get("ref") or f"mem:{self._created}"
            self._add(Edge(eid, action["type"], action["start"], action["end"], dict(action["props"])))
            return eid
        if do == "create_node":
            self._created += 1
            eid = f"mem:{self._created}"
            self.nodes[eid] = Node(eid, list(action["labels"]), dict(action["props"]))
            self._by_node[eid] = set()
            return eid
        if do == "replace_node":
            self.nodes[action["eid"]].props = dict(action["props"])
        elif do == "set_edge":
            self.edges[action["eid"]].props = dict(action["props"])
        elif do == "delete_edge":
            edge = self.edges.pop(action["eid"])
            self._by_node[edge.start].discard(edge.eid)
            self._by_node[edge.end].discard(edge.eid)
        elif do == "set_node":
            self.nodes[action["eid"]].props.update(action["props"])
        elif do == "delete_node":
            if self.degree(action["eid"]):
                raise ValueError(f"node {action['eid']} still has edges")
            del self.nodes[action["eid"]]
            self._by_node.pop(action["eid"], None)
        else:
            raise ValueError(f"unknown action {do!r}")
        return None


@dataclass
class Op:
    """One operation: a list of actions applied in one write transaction.

    :param index: Position in the plan.
    :param kind: remove | merge | rename | move | delete_orphan.
    :param title: One-line description.
    :param note: The rule's note.
    :param actions: Actions in execution order.
    :param rows: Report rows for edges.tsv.
    """

    index: int
    kind: str
    title: str
    note: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        """Serialise for plan.json.

        :return: JSON-compatible dict.
        """
        return {"index": self.index, "kind": self.kind, "title": self.title, "note": self.note,
                "actions": self.actions}


@dataclass
class Plan:
    """The planned operations and the rules that could not be planned.

    :param ops: Operations in order.
    :param review: ``{"rule", "problem"}`` items.
    :param absent: Rules whose nodes do not exist (already applied, or not imported).
    """

    ops: List[Op] = field(default_factory=list)
    review: List[Dict[str, str]] = field(default_factory=list)
    absent: int = 0


# ---------------------------------------------------------------------------
# Edge property merge policy
# ---------------------------------------------------------------------------
def eid_sort_key(eid: str) -> Tuple[Tuple[int, Any], ...]:
    """Sort key for an elementId or placeholder, numeric parts compared as numbers.

    :param eid: elementId such as ``5:<uuid>:42`` or ``new:3:0``.
    :return: The key.
    """
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in eid.split(":"))


def as_list(value: Any) -> List[Any]:
    """Wrap a scalar in a list; None becomes [].

    :param value: A property value.
    :return: The value as a list.
    """
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def union_lists(values: Iterable[Any]) -> List[Any]:
    """Order-preserving union of several lists (or scalars), dropping None.

    :param values: Lists or scalars.
    :return: The union.
    """
    out: List[Any] = []
    for value in values:
        for item in as_list(value):
            if item is not None and item not in out:
                out.append(item)
    return out


def version_key(version: str) -> Tuple[Tuple[int, ...], str]:
    """Sort key for ``pipeline_version`` values such as ``v3``.

    :param version: The version string.
    :return: The key.
    """
    return tuple(int(n) for n in re.findall(r"\d+", str(version))), str(version)


def primary_rank(edge: Edge, on_survivor: Set[str]) -> Tuple[Any, ...]:
    """Sort key choosing the primary edge: has page_uuid > confidence high > on the survivor > lowest id.

    :param edge: A candidate edge.
    :param on_survivor: elementIds of the edges already on the surviving node.
    :return: The key (smallest is primary).
    """
    return (edge.props.get("page_uuid") is None, edge.props.get("confidence") != "high",
            edge.eid not in on_survivor, eid_sort_key(edge.eid))


def merge_edge_props(edges: Sequence[Edge], on_survivor: Set[str] = frozenset()) -> Dict[str, Any]:
    """Merge the properties of edges that are the same fact (edge_merge_policy.md section 3).

    One addition to the policy: a property that only non-primary edges carry and that is not part
    of the primary's provenance (:data:`PROVENANCE_PROPS`) is kept, so folding an enriched edge into
    a PGP edge does not drop PGP's ``sub_collection``.

    :param edges: Two or more edges with the same identity.
    :param on_survivor: elementIds of those already on the surviving node.
    :return: The merged properties.
    """
    ordered = sorted(edges, key=lambda e: primary_rank(e, on_survivor))
    merged = dict(ordered[0].props)
    for edge in ordered[1:]:
        for key, value in edge.props.items():
            if value is not None and merged.get(key) is None and key not in PROVENANCE_PROPS:
                merged[key] = value
    for key in UNION_PROPS:
        values = union_lists(e.props.get(key) for e in ordered)
        if values:
            merged[key] = values
    for key, list_key in ALL_VALUES_LISTS.items():
        values = union_lists(as_list(e.props.get(key)) + as_list(e.props.get(list_key)) for e in ordered)
        if values and values != [merged.get(key)]:
            merged[list_key] = values
    for key, alt_key in ALT_LISTS.items():
        alts = [v for v in union_lists(as_list(e.props.get(key)) + as_list(e.props.get(alt_key)) for e in ordered)
                if v != merged.get(key)]
        if alts:
            merged[alt_key] = alts
    for key, order in RANKED_PROPS.items():
        values = [e.props[key] for e in ordered if e.props.get(key) is not None]
        if values:
            merged[key] = max(values, key=lambda v: order.index(v) if v in order else -1)
    for keys, combine in ((OR_FLAGS, any), (AND_FLAGS, all)):
        for key in keys:
            values = [bool(e.props[key]) for e in ordered if e.props.get(key) is not None]
            if values:
                merged[key] = combine(values)
    versions = [e.props["pipeline_version"] for e in ordered if e.props.get("pipeline_version") is not None]
    if versions:
        merged["pipeline_version"] = max(versions, key=version_key)
    return merged


def mixes_null_sources(edges: Sequence[Edge]) -> bool:
    """True when some edges have ``data_sources`` NULL (the PGP tier on the map) and others a list.

    :param edges: Edges about to be folded into one.
    :return: Whether folding them would change the map tier.
    """
    sources = [e.props.get("data_sources") is None for e in edges]
    return any(sources) and not all(sources)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def identity_value(value: Any) -> Any:
    """Hashable, whitespace-normalised form of an identity property value.

    :param value: A property value.
    :return: The comparable form.
    """
    if isinstance(value, str):
        return " ".join(value.split())
    return tuple(value) if isinstance(value, list) else value


def edge_identity(edge: Edge, anchor: str) -> Tuple[Any, ...]:
    """Identity of an edge seen from the node it is (or will be) attached to.

    :param edge: The relationship, with ``anchor`` as one endpoint.
    :param anchor: elementId of the node the edge hangs on.
    :return: ``(type, direction, other endpoint, pgpid, role, pages)``.
    """
    outgoing = edge.start == anchor
    other = edge.end if outgoing else edge.start
    return ((edge.type, "out" if outgoing else "in", other)
            + tuple(identity_value(edge.props.get(k)) for k in IDENTITY_PROPS))


def evidence_of(edge: Edge) -> str:
    """Short evidence text for reports.

    :param edge: The relationship.
    :return: At most 160 characters of ``evidence``.
    """
    return " ".join(str(edge.props.get("evidence") or "").split())[:160]


class Planner:
    """Builds the operation list, applying each action to the in-memory graph as it goes.

    :param graph: The snapshot.
    """

    def __init__(self, graph: MemoryGraph) -> None:
        self.graph = graph
        self.plan = Plan()

    def _new_op(self, kind: str, title: str, note: str) -> Op:
        """Start an operation.

        :param kind: Operation kind.
        :param title: One-line description.
        :param note: The rule's note.
        :return: The operation (not yet in the plan).
        """
        return Op(index=len(self.plan.ops), kind=kind, title=title, note=note)

    def _commit(self, op: Op) -> None:
        """Apply an operation's actions to the graph and add it to the plan if it does anything.

        :param op: The operation.
        """
        if not op.actions:
            return
        for action in op.actions:
            self.graph.apply(action)
        self.plan.ops.append(op)

    def _row(self, op: Op, action: str, edge: Edge, src: str, dst: str) -> None:
        """Add an edges.tsv row.

        :param op: The operation.
        :param action: remove | move | fold | drop_loop.
        :param edge: The edge as it is before the operation.
        :param src: Name of the node the edge leaves (or is removed from).
        :param dst: Name of the node it moves to ('' for a removal).
        """
        start = self.graph.nodes[edge.start]
        op.rows.append([op.index, op.kind, action, edge.type, start.label, start.key, src, dst,
                        json.dumps(edge.props.get("data_sources"), ensure_ascii=False),
                        json.dumps(edge.props.get("source_books"), ensure_ascii=False), evidence_of(edge)])

    def redirect(self, op: Op, edges: Sequence[Edge], old: Node, targets: Mapping[str, Node]) -> Optional[str]:
        """Plan moving ``edges`` off ``old``: each edge's ``old`` endpoint becomes ``targets[edge.eid]``.

        An edge whose identity already exists on the target is folded into that edge with the merge
        policy; one that would become a self-loop is dropped. Nothing is planned when a fold would mix
        NULL and list ``data_sources``.

        :param op: The operation to add actions to.
        :param edges: Edges incident to ``old``.
        :param old: The node the edges leave.
        :param targets: Edge elementId -> node the edge moves to.
        :return: A problem description, or None.
        """
        groups: Dict[Tuple[str, Tuple[Any, ...]], List[Edge]] = {}
        moved: List[Tuple[Edge, Edge]] = []
        for edge in edges:
            target = targets[edge.eid]
            start = target.eid if edge.start == old.eid else edge.start
            end = target.eid if edge.end == old.eid else edge.end
            new = Edge(edge.eid, edge.type, start, end, edge.props)
            moved.append((edge, new))
            groups.setdefault((target.eid, edge_identity(new, target.eid)), []).append(new)
        creates, sets, deletes = [], [], []
        counters = {"n": 0}
        for (target_eid, identity), members in groups.items():
            target = self.graph.nodes[target_eid]
            existing = sorted((e for e in self.graph.incident(target_eid)
                               if e.eid not in self.graph._by_node.get(old.eid, set())
                               and edge_identity(e, target_eid) == identity),
                              key=lambda e: primary_rank(e, {e.eid}))[:1]
            originals = {m.eid: next(o for o, n in moved if n.eid == m.eid) for m in members}
            if mixes_null_sources(existing + members):
                return f"NULL vs list data_sources when folding {members[0].type} into {target.props.get('name')}"
            if members[0].start == members[0].end:
                for m in members:
                    self._row(op, "drop_loop", originals[m.eid], old.props["name"], target.props["name"])
                    deletes.append({"do": "delete_edge", "eid": m.eid})
                continue
            if existing:
                keep = existing[0]
                props = merge_edge_props([keep] + members, on_survivor={keep.eid})
                sets.append({"do": "set_edge", "eid": keep.eid, "props": props})
                for m in members:
                    self._row(op, "fold", originals[m.eid], old.props["name"], target.props["name"])
            else:
                props = merge_edge_props(members) if len(members) > 1 else dict(members[0].props)
                ref = f"new:{op.index}:{counters['n']}"
                counters["n"] += 1
                first = members[0]
                creates.append({"do": "create_edge", "ref": ref, "type": first.type, "start": first.start,
                                "end": first.end, "props": props, "from_edges": [m.eid for m in members]})
                for m in members:
                    self._row(op, "move" if len(members) == 1 else "fold", originals[m.eid],
                              old.props["name"], target.props["name"])
            deletes.extend({"do": "delete_edge", "eid": m.eid} for m in members)
        op.actions.extend(creates + sets + deletes)
        return None

    def remove_edges(self, rule: EdgeRemoval) -> None:
        """Plan one :class:`EdgeRemoval`.

        :param rule: The rule.
        """
        end = self.graph.find(rule.end_label, rule.end_name)
        if end is None:
            self.plan.absent += 1
            return
        who = rule.start_key or "*"
        op = self._new_op("remove", f"remove {rule.rel_type} {who} -> {rule.end_name}", rule.note)
        for edge in self.graph.incident(end.eid):
            start = self.graph.nodes[edge.start]
            if (edge.type == rule.rel_type and edge.end == end.eid
                    and (not rule.start_key or start.key == rule.start_key)
                    and (not rule.source_book or rule.source_book in as_list(edge.props.get("source_books")))):
                self._row(op, "remove", edge, rule.end_name, "")
                op.actions.append({"do": "delete_edge", "eid": edge.eid})
        self._commit(op)

    def merge_node(self, label: str, name: str, rule: InstitutionMerge) -> None:
        """Plan merging the node ``name`` into ``rule.canonical`` (or renaming it when that is absent).

        :param label: Institution or Place.
        :param name: The duplicate's name.
        :param rule: The merge rule.
        """
        dup = self.graph.find(label, name)
        if dup is None:
            self.plan.absent += 1
            return
        canonical = self.graph.find(label, rule.canonical)
        kind = "merge" if canonical is not None else "rename"
        op = self._new_op(kind, f"{kind} {label} '{name}' -> '{rule.canonical}'", rule.note)
        targets: Dict[str, Node] = {}
        to_move: List[Edge] = []
        for edge in self.graph.incident(dup.eid):
            target_name = rule.canonical
            if edge.type == "AFFILIATED_WITH" and edge.end == dup.eid:
                target_name = rule.affiliation_target
            if target_name == KEEP_AFFILIATIONS or (canonical is None and target_name == rule.canonical):
                continue
            target = self.graph.find(label, target_name)
            if target is None:
                self.plan.review.append({"rule": op.title, "problem": f"{label} '{target_name}' is not in the graph"})
                return
            targets[edge.eid] = target
            to_move.append(edge)
        problem = self.redirect(op, to_move, dup, targets)
        if problem:
            self.plan.review.append({"rule": op.title, "problem": problem})
            return
        remaining = self.graph.degree(dup.eid) - len(to_move)
        if canonical is not None and remaining:
            self._commit(op)  # kept edges (KEEP_AFFILIATIONS): the duplicate stays, so it is no alias yet
            return
        survivor = canonical or dup
        update = survivor_update(survivor, dup, name, rule.canonical, ALIAS_PROPS[label])
        if update:
            op.actions.append({"do": "set_node", "eid": survivor.eid, "props": update})
        if canonical is not None:
            op.actions.append({"do": "delete_node", "eid": dup.eid})
            op.rows.append([op.index, op.kind, "delete_node", "", label, name, name, rule.canonical, "", "", ""])
        self._commit(op)

    def move_edges(self, rule: EdgeMove) -> None:
        """Plan one :class:`EdgeMove`.

        :param rule: The rule.
        """
        old = self.graph.find(rule.end_label, rule.from_name)
        if old is None:
            self.plan.absent += 1
            return
        op = self._new_op("move", f"move {rule.rel_type} {rule.start_key} -> '{rule.from_name}' to "
                                  f"'{rule.to_name}'", rule.note)
        edges = [e for e in self.graph.incident(old.eid)
                 if e.type == rule.rel_type and e.end == old.eid and self.graph.nodes[e.start].key == rule.start_key]
        if not edges:
            self.plan.absent += 1
            return
        target = self.graph.find(rule.end_label, rule.to_name)
        if target is None:
            self.plan.review.append({"rule": op.title, "problem": f"'{rule.to_name}' is not in the graph"})
            return
        problem = self.redirect(op, edges, old, {e.eid: target for e in edges})
        if problem:
            self.plan.review.append({"rule": op.title, "problem": problem})
            return
        self._commit(op)

    def delete_orphans(self, names: Iterable[str]) -> None:
        """Plan deleting the named Institution nodes that have no edges left.

        :param names: Institution names the removals touched.
        """
        for name in names:
            node = self.graph.find("Institution", name)
            if node is None or self.graph.degree(node.eid):
                continue
            op = self._new_op("delete_orphan", f"delete Institution '{name}' (no edges left)",
                              "Every edge it had was removed as not a holding.")
            op.actions.append({"do": "delete_node", "eid": node.eid})
            op.rows.append([op.index, op.kind, "delete_node", "", "Institution", name, name, "", "", "", ""])
            self._commit(op)


def survivor_update(survivor: Node, dup: Node, name: str, canonical: str, alias_prop: str) -> Dict[str, Any]:
    """Properties to set on the node that absorbs ``dup``.

    The survivor keeps its own name, pin and other properties; ``data_sources`` and
    ``source_books`` become the union, and the duplicate's name (and its own aliases) are added
    to ``alias_prop`` (an ``aliases`` list, or the comma-separated ``name_variants`` of a Place).
    When the survivor is the duplicate itself (no canonical node yet), it is renamed.

    :param survivor: The canonical node, or ``dup`` when it is being renamed.
    :param dup: The duplicate.
    :param name: The duplicate's name.
    :param canonical: The canonical name.
    :param alias_prop: ``aliases`` or ``name_variants``.
    :return: Changed properties only.
    """
    update: Dict[str, Any] = {}
    if survivor is dup:
        update["name"] = canonical
    else:
        for key in UNION_PROPS:
            values = union_lists([survivor.props.get(key), dup.props.get(key)])
            if values and values != as_list(survivor.props.get(key)):
                update[key] = values
    current = split_aliases(survivor.props.get(alias_prop))
    names = [n for n in union_lists([current, [name], split_aliases(dup.props.get(alias_prop))]) if n != canonical]
    if names != current:
        update[alias_prop] = names if alias_prop == "aliases" else ", ".join(names)
    return update


def split_aliases(value: Any) -> List[str]:
    """Alias names from an ``aliases`` list or a comma-separated ``name_variants`` string.

    :param value: The property value.
    :return: The names.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(v) for v in value]


def build_plan(graph: MemoryGraph) -> Plan:
    """Plan every rule against the snapshot, in order: removals, merges, moves, orphan deletion.

    :param graph: The snapshot (it is modified: it ends in the planned state).
    :return: The plan.
    """
    planner = Planner(graph)
    for rule in EDGE_REMOVALS:
        planner.remove_edges(rule)
    for name, rule in INSTITUTION_MERGES.items():
        planner.merge_node("Institution", name, rule)
    for name, rule in PLACE_MERGES.items():
        planner.merge_node("Place", name, rule)
    for rule in EDGE_MOVES:
        planner.move_edges(rule)
    planner.delete_orphans(dict.fromkeys(r.end_name for r in EDGE_REMOVALS if r.end_label == "Institution"))
    return planner.plan


def rule_names() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Node names the rules read, split into source nodes (fully read) and target nodes.

    :return: ``(sources, targets)``, each label -> names.
    """
    sources: Dict[str, Set[str]] = {"Institution": set(), "Place": set()}
    targets: Dict[str, Set[str]] = {"Institution": set(), "Place": set()}
    for label, rules in (("Institution", INSTITUTION_MERGES), ("Place", PLACE_MERGES)):
        for name, rule in rules.items():
            sources[label].add(name)
            targets[label].add(rule.canonical)
            if rule.affiliation_target != KEEP_AFFILIATIONS:
                targets[label].add(rule.affiliation_target)
    for removal in EDGE_REMOVALS:
        sources[removal.end_label].add(removal.end_name)
    for move in EDGE_MOVES:
        sources[move.end_label].add(move.from_name)
        targets[move.end_label].add(move.to_name)
    return sources, targets


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def canonical_dumps(obj: Any) -> str:
    """Deterministic JSON used for the plan hash and the --apply comparison.

    :param obj: JSON-compatible object.
    :return: The JSON text.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def plan_json(plan: Plan, target: Mapping[str, str]) -> Dict[str, Any]:
    """The plan.json document.

    :param plan: The plan.
    :param target: ``{uri, database, store}`` the plan was made against.
    :return: JSON-compatible dict.
    """
    return {"format": PLAN_FORMAT, "target": dict(target), "ops": [op.to_json() for op in plan.ops],
            "review": plan.review}


def summarize(plan: Plan) -> Dict[str, Any]:
    """Counts by operation kind and action.

    :param plan: The plan.
    :return: The summary.
    """
    actions = Counter(a["do"] for op in plan.ops for a in op.actions)
    rows = Counter(row[2] for op in plan.ops for row in op.rows)
    return {"operations": dict(Counter(op.kind for op in plan.ops)), "actions": dict(actions),
            "edges": {k: v for k, v in rows.items() if k != "delete_node"},
            "nodes_deleted": actions.get("delete_node", 0),
            "relationship_delta": actions.get("create_edge", 0) - actions.get("delete_edge", 0),
            "review": len(plan.review), "rules_without_nodes": plan.absent,
            "plan_sha256": hashlib.sha256(canonical_dumps([op.to_json() for op in plan.ops]).encode()).hexdigest()}


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """Write a tab-separated file.

    :param path: Output path.
    :param header: Column names.
    :param rows: Rows.
    """
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_reports(plan: Plan, out: Path, target: Mapping[str, str]) -> Dict[str, Any]:
    """Write summary.json, plan.json, edges.tsv and review.tsv.

    :param plan: The plan.
    :param out: Output directory (created).
    :param target: ``{uri, database, store}``.
    :return: The summary.
    """
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(plan)
    summary["target"] = dict(target)
    summary["generated"] = utc_stamp()
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "plan.json").write_text(json.dumps(plan_json(plan, target), ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
    write_tsv(out / "edges.tsv", ["op", "kind", "action", "rel_type", "start_label", "start_key", "from", "to",
                                  "data_sources", "source_books", "evidence"],
              (row for op in plan.ops for row in op.rows))
    write_tsv(out / "review.tsv", ["rule", "problem"], ([r["rule"], r["problem"]] for r in plan.review))
    return summary


def print_plan(plan: Plan, summary: Mapping[str, Any]) -> None:
    """Print the operations and the summary.

    :param plan: The plan.
    :param summary: Its summary.
    """
    for op in plan.ops:
        counts = Counter(row[2] for row in op.rows)
        detail = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
        print(f"  [{op.index:3d}] {op.title}  ({detail})")
    for item in plan.review:
        print(f"  REVIEW  {item['rule']}: {item['problem']}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("target", "generated")},
                     ensure_ascii=False, indent=2))


def utc_stamp() -> str:
    """Current UTC time as ``YYYYMMDDTHHMMSSZ``.

    :return: The stamp.
    """
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Neo4j access
# ---------------------------------------------------------------------------
NODE_QUERY = """
MATCH (n:{label}) WHERE n.name IN $names
RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props, COUNT {{ (n)--() }} AS degree
"""
EDGE_FIELDS = """
RETURN DISTINCT elementId(r) AS eid, type(r) AS type, elementId(startNode(r)) AS start,
       elementId(endNode(r)) AS end, properties(r) AS props, elementId(m) AS other, labels(m) AS other_labels,
       m.name AS other_name, m.canonical_shelfmark AS other_csm
"""
SOURCE_EDGES = "MATCH (n)-[r]-(m) WHERE elementId(n) IN $eids" + EDGE_FIELDS
TARGET_EDGES = "MATCH (n)-[r]-(m) WHERE elementId(n) IN $eids AND elementId(m) IN $neighbors" + EDGE_FIELDS


def read_snapshot(tx: Any) -> MemoryGraph:
    """Read the rule nodes, all edges of the source nodes and the target nodes' edges to the same neighbours.

    :param tx: A read transaction.
    :return: The snapshot.
    """
    sources, targets = rule_names()
    nodes: Dict[str, Node] = {}
    degrees: Dict[str, int] = {}
    source_eids: List[str] = []
    for label in ("Institution", "Place"):
        names = sorted(sources[label] | targets[label])
        for rec in tx.run(NODE_QUERY.format(label=label), names=names):
            nodes[rec["eid"]] = Node(rec["eid"], list(rec["labels"]), dict(rec["props"]))
            degrees[rec["eid"]] = rec["degree"]
            if rec["props"]["name"] in sources[label]:
                source_eids.append(rec["eid"])
    edges: Dict[str, Edge] = {}

    def take(records: Iterable[Any]) -> None:
        for rec in records:
            edges[rec["eid"]] = Edge(rec["eid"], rec["type"], rec["start"], rec["end"], dict(rec["props"]))
            if rec["other"] not in nodes:
                props = {k: rec[f] for k, f in (("name", "other_name"), ("canonical_shelfmark", "other_csm"))
                         if rec[f] is not None}
                nodes[rec["other"]] = Node(rec["other"], list(rec["other_labels"]), props, stub=True)

    take(tx.run(SOURCE_EDGES, eids=source_eids))
    neighbors = sorted({e.start for e in edges.values()} | {e.end for e in edges.values()})
    target_eids = sorted(set(degrees) - set(source_eids))
    take(tx.run(TARGET_EDGES, eids=target_eids, neighbors=neighbors))
    graph = MemoryGraph(nodes.values(), edges.values())
    for eid, degree in degrees.items():
        graph.nodes[eid].unloaded = max(0, degree - len(graph._by_node.get(eid, ())))
    return graph


def store_id(eid: str) -> str:
    """The database store part of an elementId (``4:<store>:<n>``).

    :param eid: An elementId.
    :return: The store id.
    """
    parts = eid.split(":")
    return parts[1] if len(parts) == 3 else ""


def current_target(session: Any, uri: str, database: str) -> Dict[str, str]:
    """``{uri, database, store}`` of the connected database.

    :param session: A session.
    :param uri: Bolt URI.
    :param database: Database name.
    :return: The target.
    """
    eid = session.execute_read(lambda tx: tx.run("MATCH (n) RETURN elementId(n) AS eid LIMIT 1").single()["eid"])
    return {"uri": uri, "database": database, "store": store_id(eid)}


def graph_totals(session: Any) -> Dict[str, int]:
    """Node and relationship totals.

    :param session: A session.
    :return: ``{nodes, relationships}``.
    """
    def work(tx: Any) -> Dict[str, int]:
        nodes = tx.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = tx.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": nodes, "relationships": rels}
    return session.execute_read(work)


def safe_name(name: str) -> str:
    """Return ``name`` if it can be spliced into Cypher as a label or relationship type.

    :param name: Label or type.
    :return: The same name.
    """
    if not SAFE_NAME.match(name):
        raise ValueError(f"unsafe label or type {name!r}")
    return name


def run_expect(tx: Any, query: str, expected: int, **params: Any) -> List[Any]:
    """Run a query that must return ``expected`` rows.

    :param tx: Transaction.
    :param query: Cypher.
    :param expected: Expected row count.
    :param params: Query parameters.
    :return: The records.
    """
    records = list(tx.run(query, **params))
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} row(s), got {len(records)}: {query.split(chr(10))[0]} {params}")
    return records


def resolve(eid: str, refs: Mapping[str, str]) -> str:
    """Map a ``new:`` placeholder to the elementId it was created with.

    :param eid: elementId or placeholder.
    :param refs: Placeholder -> elementId.
    :return: The elementId.
    """
    return refs[eid] if PLACEHOLDER.match(eid) else eid


def run_action(tx: Any, action: Mapping[str, Any]) -> Optional[str]:
    """Execute one resolved action.

    Actions: ``create_edge`` (type, start, end, props; returns the new elementId), ``set_edge``
    (eid, props: replaces all properties), ``delete_edge`` (eid), ``set_node`` (eid, props: sets
    the given properties), ``delete_node`` (eid; refuses a node that still has edges), and for
    revert only ``create_node`` (labels, props; returns the new elementId) and ``replace_node``
    (eid, props: replaces all properties).

    :param tx: Write transaction.
    :param action: The action, with placeholders already resolved.
    :return: elementId of a created edge or node, else None.
    """
    do = action["do"]
    if do == "create_edge":
        rec = run_expect(tx, f"""
            MATCH (a) WHERE elementId(a) = $start
            MATCH (b) WHERE elementId(b) = $end
            CREATE (a)-[r:{safe_name(action['type'])}]->(b) SET r = $props
            RETURN elementId(r) AS eid""", 1, start=action["start"], end=action["end"], props=action["props"])
        return rec[0]["eid"]
    if do == "set_edge":
        run_expect(tx, "MATCH ()-[r]->() WHERE elementId(r) = $eid SET r = $props RETURN 1", 1,
                   eid=action["eid"], props=action["props"])
    elif do == "delete_edge":
        run_expect(tx, "MATCH ()-[r]->() WHERE elementId(r) = $eid DELETE r RETURN 1", 1, eid=action["eid"])
    elif do == "set_node":
        run_expect(tx, "MATCH (n) WHERE elementId(n) = $eid SET n += $props RETURN 1", 1,
                   eid=action["eid"], props=action["props"])
    elif do == "replace_node":
        run_expect(tx, "MATCH (n) WHERE elementId(n) = $eid SET n = $props RETURN 1", 1,
                   eid=action["eid"], props=action["props"])
    elif do == "delete_node":
        run_expect(tx, "MATCH (n) WHERE elementId(n) = $eid AND NOT (n)--() DELETE n RETURN 1", 1, eid=action["eid"])
    elif do == "create_node":
        labels = ":".join(safe_name(lab) for lab in action["labels"])
        rec = run_expect(tx, f"CREATE (n:{labels}) SET n = $props RETURN elementId(n) AS eid", 1, props=action["props"])
        return rec[0]["eid"]
    else:
        raise ValueError(f"unknown action {do!r}")
    return None


def read_before(tx: Any, actions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Full current state of every node and edge the (resolved) actions change or delete.

    :param tx: The operation's write transaction.
    :param actions: Resolved actions.
    :return: ``{"nodes": {eid: {labels, props}}, "edges": {eid: {type, start, end, props, keys}}}``.
    """
    node_eids = sorted({a["eid"] for a in actions if a["do"] in ("set_node", "delete_node")})
    edge_eids = sorted({a["eid"] for a in actions if a["do"] in ("set_edge", "delete_edge")})
    nodes = {r["eid"]: {"labels": r["labels"], "props": r["props"]} for r in run_expect(tx, """
        MATCH (n) WHERE elementId(n) IN $eids RETURN elementId(n) AS eid, labels(n) AS labels,
        properties(n) AS props""", len(node_eids), eids=node_eids)}
    edges = {r["eid"]: {"type": r["type"], "start": r["start"], "end": r["end"], "props": r["props"],
                        "start_key": r["start_key"], "end_key": r["end_key"]} for r in run_expect(tx, """
        MATCH (a)-[r]->(b) WHERE elementId(r) IN $eids
        RETURN elementId(r) AS eid, type(r) AS type, elementId(a) AS start, elementId(b) AS end,
               properties(r) AS props, coalesce(a.canonical_shelfmark, a.name) AS start_key,
               coalesce(b.canonical_shelfmark, b.name) AS end_key""", len(edge_eids), eids=edge_eids)}
    return {"nodes": nodes, "edges": edges}


def append_log(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON record to a change log and flush it to disk.

    :param path: Change log path.
    :param record: The record.
    """
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_log(path: Path) -> List[Dict[str, Any]]:
    """Read a change log.

    :param path: Change log path.
    :return: Its records.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_totals(ops: Sequence[Mapping[str, Any]], before: Mapping[str, int]) -> Dict[str, int]:
    """Node and relationship totals after the operations.

    :param ops: plan.json operations.
    :param before: Totals before.
    :return: Expected totals.
    """
    actions = Counter(a["do"] for op in ops for a in op["actions"])
    return {"nodes": before["nodes"] - actions["delete_node"],
            "relationships": before["relationships"] + actions["create_edge"] - actions["delete_edge"]}


def under_temp_dir(path: Path) -> bool:
    """True when ``path`` is inside a directory the OS may clean.

    :param path: A directory.
    :return: Whether it is temporary.
    """
    resolved = str(path.resolve())
    return any(resolved == root or resolved.startswith(root + "/") for root in TEMP_ROOTS)


def run_dry(session: Any, target: Mapping[str, str], out: Path) -> Dict[str, Any]:
    """Plan against the live graph and write the reports.

    :param session: A READ session.
    :param target: ``{uri, database, store}``.
    :param out: Output directory.
    :return: The summary.
    """
    plan = build_plan(session.execute_read(read_snapshot))
    summary = write_reports(plan, out, target)
    print_plan(plan, summary)
    print(f"\nDry run written to {out}")
    return summary


def run_apply(driver: Any, database: str, target: Mapping[str, str], plan_dir: Path) -> Dict[str, Any]:
    """Re-plan, check the plan equals the reviewed one, and execute it with a change log.

    :param driver: Neo4j driver.
    :param database: Database name.
    :param target: ``{uri, database, store}`` of the connected database.
    :param plan_dir: Dry-run directory.
    :return: Verification result.
    """
    saved = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    if saved.get("format") != PLAN_FORMAT:
        raise SystemExit(f"{plan_dir}/plan.json has format {saved.get('format')}, expected {PLAN_FORMAT}")
    if saved["target"]["store"] != target["store"]:
        raise SystemExit(f"plan was made against store {saved['target']['store']}, connected to {target['store']}")
    with driver.session(database=database, default_access_mode="READ") as session:
        plan = build_plan(session.execute_read(read_snapshot))
        before = graph_totals(session)
    fresh = [op.to_json() for op in plan.ops]
    if canonical_dumps(fresh) != canonical_dumps(saved["ops"]):
        raise SystemExit("The graph changed since the dry run (the new plan differs from plan.json). "
                         "Run a new dry run and review it.")
    if not fresh:
        print("Nothing to apply.")
        return {"applied": 0}
    log = plan_dir / f"changelog_{utc_stamp()}.jsonl"
    append_log(log, {"event": "target", **target, "plan_sha256": summarize(plan)["plan_sha256"]})
    refs: Dict[str, str] = {}
    with driver.session(database=database) as session:
        for op in fresh:
            actions = [{**a, **{k: resolve(a[k], refs) for k in ("eid", "start", "end") if k in a}}
                       for a in op["actions"]]
            tx = session.begin_transaction()
            try:
                append_log(log, {"event": "before", "op": op["index"], "title": op["title"], "actions": actions,
                                 **read_before(tx, actions)})
                created = {}
                for action in actions:
                    new_eid = run_action(tx, action)
                    if action["do"] == "create_edge":
                        created[action["ref"]] = new_eid
                append_log(log, {"event": "applied", "op": op["index"], "created": created})
                tx.commit()
            except Exception as error:
                tx.close()
                append_log(log, {"event": "failed", "op": op["index"], "error": f"{type(error).__name__}: {error}"})
                raise
            refs.update(created)
            append_log(log, {"event": "committed", "op": op["index"]})
            print(f"  applied [{op['index']:3d}] {op['title']}")
    with driver.session(database=database, default_access_mode="READ") as session:
        after = graph_totals(session)
        again = build_plan(session.execute_read(read_snapshot))
    result = {"applied": len(fresh), "changelog": str(log), "totals_before": before, "totals_after": after,
              "totals_expected": expected_totals(fresh, before), "second_plan_ops": len(again.ops)}
    result["ok"] = result["totals_after"] == result["totals_expected"] and not again.ops
    print(json.dumps(result, indent=2))
    return result


def revert_actions(before: Mapping[str, Any], created: Mapping[str, str], node_map: Dict[str, str],
                   edge_map: Dict[str, str]) -> List[Tuple[Dict[str, Any], Optional[Tuple[Dict[str, str], str]]]]:
    """Actions that undo one committed operation.

    Order: delete the edges it created, re-create the nodes it deleted, re-create the edges it
    deleted, restore the properties of the edges and nodes it changed.

    :param before: The operation's ``before`` record.
    :param created: Its ``created`` map (placeholder -> elementId).
    :param node_map: Old -> new elementId of nodes re-created by earlier reverts (updated by the caller).
    :param edge_map: Old -> new elementId of edges re-created by earlier reverts.
    :return: ``(action, (map, old eid) or None)`` pairs; the map entry is filled with the action's result.
    """
    node = lambda eid: node_map.get(eid, eid)  # noqa: E731
    edge = lambda eid: edge_map.get(eid, eid)  # noqa: E731
    out: List[Tuple[Dict[str, Any], Optional[Tuple[Dict[str, str], str]]]] = []
    for eid in created.values():
        out.append(({"do": "delete_edge", "eid": edge(eid)}, None))
    actions = before["actions"]
    for a in actions:
        if a["do"] == "delete_node":
            state = before["nodes"][a["eid"]]
            out.append(({"do": "create_node", "labels": state["labels"], "props": state["props"]}, (node_map, a["eid"])))
    for a in actions:
        if a["do"] == "delete_edge" and a["eid"] in before["edges"]:
            state = before["edges"][a["eid"]]
            out.append(({"do": "create_edge", "type": state["type"], "start": state["start"], "end": state["end"],
                         "props": state["props"]}, (edge_map, a["eid"])))
    for a in actions:
        if a["do"] == "set_edge":
            out.append(({"do": "set_edge", "eid": edge(a["eid"]), "props": before["edges"][a["eid"]]["props"]}, None))
    for a in actions:
        if a["do"] == "set_node":
            out.append(({"do": "replace_node", "eid": node(a["eid"]), "props": before["nodes"][a["eid"]]["props"]}, None))
    return out


def run_revert(driver: Any, database: str, target: Mapping[str, str], log: Path) -> Dict[str, Any]:
    """Undo the committed, not yet reverted operations of a change log, newest first.

    :param driver: Neo4j driver.
    :param database: Database name.
    :param target: ``{uri, database, store}`` of the connected database.
    :param log: Change log path.
    :return: ``{"reverted": n}``.
    """
    records = read_log(log)
    if not records or records[0].get("event") != "target":
        raise SystemExit(f"{log} does not start with a target record")
    if records[0]["store"] != target["store"]:
        raise SystemExit(f"{log} was written for store {records[0]['store']}, connected to {target['store']}")
    by_op: Dict[int, Dict[str, Any]] = {}
    node_map: Dict[str, str] = {}
    edge_map: Dict[str, str] = {}
    for rec in records[1:]:
        if rec["event"] == "reverted":
            node_map.update(rec["node_map"])
            edge_map.update(rec["edge_map"])
        by_op.setdefault(rec["op"], {})[rec["event"]] = rec
    unclear = [i for i, ev in by_op.items() if "applied" in ev and not ({"committed", "failed"} & set(ev))]
    if unclear:
        raise SystemExit(f"operations {unclear} were written but their commit was not logged; inspect the graph")
    pending = sorted((i for i, ev in by_op.items() if "committed" in ev and "reverted" not in ev), reverse=True)
    with driver.session(database=database) as session:
        for index in pending:
            events = by_op[index]
            new_nodes: Dict[str, str] = {}
            new_edges: Dict[str, str] = {}
            with session.begin_transaction() as tx:
                for action, slot in revert_actions(events["before"], events["applied"]["created"], node_map, edge_map):
                    if action["do"] == "create_edge":
                        action = {**action, "start": node_map.get(action["start"], action["start"]),
                                  "end": node_map.get(action["end"], action["end"])}
                    result = run_action(tx, action)
                    if slot is not None:
                        slot[0][slot[1]] = result
                        (new_nodes if slot[0] is node_map else new_edges)[slot[1]] = result
                tx.commit()
            append_log(log, {"event": "reverted", "op": index, "node_map": new_nodes, "edge_map": new_edges})
            print(f"  reverted [{index:3d}] {events['before']['title']}")
    return {"reverted": len(pending)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--uri", help="Bolt URI (default $NEO4J_URI)")
    parser.add_argument("--database", help="database (default $NEO4J_DATABASE or genizah-prod)")
    parser.add_argument("--out", type=Path, help="dry-run directory (default docs/kg_consolidation/institutions_<UTC>)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="execute the plan in --plan")
    mode.add_argument("--revert", type=Path, metavar="CHANGELOG", help="undo a change log")
    parser.add_argument("--plan", type=Path, help="dry-run directory to apply")
    parser.add_argument("--allow-temp-plan", action="store_true", help="allow --plan under a temporary directory")
    args = parser.parse_args(argv)
    if args.apply and not args.plan:
        parser.error("--apply needs --plan DIR (the reviewed dry run)")
    if args.apply and under_temp_dir(args.plan) and not args.allow_temp_plan:
        parser.error(f"{args.plan} is under a temporary directory; its change log could be lost "
                     "(use --allow-temp-plan for a rehearsal)")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run a dry run, an apply or a revert.

    :param argv: Arguments (default ``sys.argv[1:]``).
    :return: Exit status.
    """
    import dotenv
    from neo4j import GraphDatabase, READ_ACCESS

    args = parse_args(argv)
    dotenv.load_dotenv(project_root / ".env")
    uri = args.uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    database = args.database or os.getenv("NEO4J_DATABASE", "genizah-prod")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        print("NEO4J_PASSWORD not set", file=sys.stderr)
        return 2
    driver = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER", "neo4j"), password))
    try:
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            target = current_target(session, uri, database)
            print(f"Target: {uri} database={database} store={target['store']}")
            if not (args.apply or args.revert):
                run_dry(session, target, args.out or DEFAULT_OUT_ROOT / f"institutions_{utc_stamp()}")
                return 0
        if args.apply:
            return 0 if run_apply(driver, database, target, args.plan).get("ok", True) else 1
        run_revert(driver, database, target, args.revert)
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
