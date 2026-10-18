"""Tests for the post-import Institution / Place consolidation pass.

The pass merges duplicate Institution nodes ("Cambridge, University Library" ->
"Cambridge University Library"), removes edges that are not holdings ("from the
Cairo Genizah") and fixes Place duplicates. These tests pin the planner against an
in-memory graph: edge routing and folding, the property merge policy, idempotence,
revert, and the agreement between the merge rules and InstitutionNormalizer (which
stops the importers re-creating the duplicates).
"""

import copy
from collections import Counter
from typing import Any, Dict, List, Tuple

import pytest

from src.datasets.document_models.institution_normalizer import InstitutionNormalizer, _EXACT_ALIASES
from src.datasets.indexing.neo4j import consolidate_institutions as ci
from src.datasets.indexing.neo4j.consolidate_institutions import (
    Edge, MemoryGraph, Node, Planner, build_plan, merge_edge_props, revert_actions)
from src.datasets.indexing.neo4j.institution_merges import (
    EDGE_MOVES, EDGE_REMOVALS, INSTITUTION_MERGES, KEEP_AFFILIATIONS, PLACE_MERGES,
    InstitutionMerge)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def inst(eid: str, name: str, **props: Any) -> Node:
    """An Institution node."""
    return Node(eid, ["Institution"], {"name": name, **props})


def place(eid: str, name: str, **props: Any) -> Node:
    """A Place node."""
    return Node(eid, ["Place"], {"name": name, **props})


def frag(eid: str, csm: str) -> Node:
    """A Fragment endpoint (outside the rules)."""
    return Node(eid, ["Fragment"], {"canonical_shelfmark": csm}, stub=True)


def scholar(eid: str, name: str) -> Node:
    """A Scholar endpoint (outside the rules)."""
    return Node(eid, ["Scholar"], {"name": name}, stub=True)


def edge(eid: str, rel: str, start: str, end: str, **props: Any) -> Edge:
    """A relationship."""
    return Edge(eid, rel, start, end, props)


def state(graph: MemoryGraph) -> Tuple[Counter, Counter]:
    """The graph by business keys (elementIds change when a revert re-creates a node)."""
    nodes = Counter((n.label, n.key, repr(sorted(n.props.items()))) for n in graph.nodes.values() if not n.stub)
    edges = Counter((e.type, graph.nodes[e.start].key, graph.nodes[e.end].key, repr(sorted(e.props.items())))
                    for e in graph.edges.values())
    return nodes, edges


def edges_between(graph: MemoryGraph, start_key: str, end_name: str) -> List[Edge]:
    """Edges from the node with key ``start_key`` to the node named ``end_name``."""
    return [e for e in graph.edges.values()
            if graph.nodes[e.start].key == start_key and graph.nodes[e.end].props.get("name") == end_name]


@pytest.fixture
def cambridge() -> MemoryGraph:
    """CUL, the university, and the bare-city duplicate 'Cambridge'."""
    return MemoryGraph(
        [inst("i:cul", "Cambridge University Library", data_sources=["pgp"], source_books=["a"]),
         inst("i:uni", "University of Cambridge", data_sources=["enriched"]),
         inst("i:cam", "Cambridge", data_sources=["enriched"], source_books=["b"]),
         frag("f:1", "T_S_12_244"), frag("f:2", "T_S_K5"), scholar("s:1", "Solomon Schechter")],
        [edge("r:1", "HELD_AT", "f:1", "i:cul", data_sources=["pgp"], sub_collection="Old Series"),
         edge("r:2", "HELD_AT", "f:1", "i:cam", data_sources=["enriched"], evidence="MSS Cambridge, T-S 12.244",
              page_uuid="p1", book_uuid="b1", confidence="high", source_books=["jewish_proselytism"]),
         edge("r:3", "HELD_AT", "f:2", "i:cam", data_sources=["enriched"], evidence="Cambridge T-S K5"),
         edge("r:4", "AFFILIATED_WITH", "s:1", "i:cam", data_sources=["enriched"], evidence="Schechter in Cambridge")])


# --------------------------------------------------------------------------
# Rules and normalizer stay in step
# --------------------------------------------------------------------------
def test_no_merge_chains():
    """A canonical or affiliation target is never itself merged away (one hop per name)."""
    for rules in (INSTITUTION_MERGES, PLACE_MERGES):
        for name, rule in rules.items():
            assert rule.canonical not in rules, name
            assert rule.affiliation_target not in rules, name
            assert rule.canonical != name


@pytest.mark.parametrize("name,rule", sorted(INSTITUTION_MERGES.items()))
def test_normalizer_matches_merge_rules(name: str, rule: InstitutionMerge):
    expected = rule.canonical if rule.normalize else name
    assert InstitutionNormalizer.normalize(name) == expected


def test_exact_aliases_are_exactly_the_normalize_rules():
    normalized = {n: r.canonical for n, r in INSTITUTION_MERGES.items() if r.normalize}
    assert _EXACT_ALIASES == normalized


def test_canonicals_and_removal_targets_keep_their_names():
    """The pass must still find these nodes after an import ran them through the normalizer."""
    names = {r.canonical for r in INSTITUTION_MERGES.values()}
    names |= {r.affiliation_target for r in INSTITUTION_MERGES.values()} - {KEEP_AFFILIATIONS}
    names |= {r.end_name for r in EDGE_REMOVALS if r.end_label == "Institution"}
    for name in sorted(names):
        assert InstitutionNormalizer.normalize(name) == name


@pytest.mark.parametrize("raw,canonical", [
    ("ספרייה הבריטית", "British Library, London"),          # Hebrew: the substring scan folds it to ''
    ("  the hebrew university,  JERUSALEM ", "Hebrew University of Jerusalem"),
    ("Taylor-Schechter Collection", "Cambridge University Library"),   # substring keys still work
    ("Cambridge", "Cambridge"),                                # bare city names are left to the pass
])
def test_exact_alias_lookup(raw: str, canonical: str):
    assert InstitutionNormalizer.normalize(raw) == canonical


def test_rule_types_are_safe_cypher_names():
    for rule in EDGE_REMOVALS + EDGE_MOVES:
        assert ci.safe_name(rule.rel_type) == rule.rel_type


# --------------------------------------------------------------------------
# Edge property merge policy
# --------------------------------------------------------------------------
def test_merge_props_primary_has_page_uuid():
    pgp = edge("r:1", "HELD_AT", "f", "i", data_sources=["pgp"], sub_collection="Old Series")
    enriched = edge("r:2", "HELD_AT", "f", "i", data_sources=["enriched"], evidence="ev2", page_uuid="p2",
                    book_uuid="b2", confidence="high", source_books=["x"])
    other = edge("r:3", "HELD_AT", "f", "i", data_sources=["enriched"], evidence="ev3", confidence="medium",
                 book_uuid="b3", source_books=["y"])
    merged = merge_edge_props([pgp, enriched, other], on_survivor={"r:1"})
    assert merged["evidence"] == "ev2" and merged["page_uuid"] == "p2"
    assert merged["evidence_alt"] == ["ev3"]
    assert merged["book_uuids"] == ["b2", "b3"]
    assert merged["data_sources"] == ["enriched", "pgp"]
    assert merged["source_books"] == ["x", "y"]
    assert merged["confidence"] == "high"
    # PGP's sub_collection survives although the enriched edge is primary.
    assert merged["sub_collection"] == "Old Series"
    assert "sub_collections" not in merged


def test_merge_props_never_borrows_provenance():
    primary = edge("r:1", "HELD_AT", "f", "i", data_sources=["enriched"], page_uuid="p1", evidence="ev1")
    other = edge("r:2", "HELD_AT", "f", "i", data_sources=["enriched"], evidence="ev2", evidence_page=7)
    merged = merge_edge_props([primary, other])
    assert "evidence_page" not in merged  # page 7 belongs to ev2, not to the primary's evidence


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------
def test_merge_routes_folds_and_deletes(cambridge: MemoryGraph):
    planner = Planner(cambridge)
    planner.merge_node("Institution", "Cambridge", INSTITUTION_MERGES["Cambridge"])
    [op] = planner.plan.ops
    assert Counter(row[2] for row in op.rows) == {"fold": 1, "move": 2, "delete_node": 1}
    assert cambridge.find("Institution", "Cambridge") is None
    # T-S 12.244 already had a PGP HELD_AT to CUL: one edge, enriched evidence, PGP sub_collection kept.
    [held] = edges_between(cambridge, "T_S_12_244", "Cambridge University Library")
    assert held.eid == "r:1"
    assert held.props["evidence"] == "MSS Cambridge, T-S 12.244"
    assert held.props["sub_collection"] == "Old Series"
    assert held.props["data_sources"] == ["enriched", "pgp"]
    assert len(edges_between(cambridge, "T_S_K5", "Cambridge University Library")) == 1
    # The bare city's affiliation goes to the university, not to the library.
    assert len(edges_between(cambridge, "Solomon Schechter", "University of Cambridge")) == 1
    assert not edges_between(cambridge, "Solomon Schechter", "Cambridge University Library")
    cul = cambridge.find("Institution", "Cambridge University Library")
    assert cul.props["aliases"] == ["Cambridge"]
    assert cul.props["data_sources"] == ["pgp", "enriched"]
    assert cul.props["source_books"] == ["a", "b"]


def test_keep_affiliations_leaves_duplicate_with_its_affiliations():
    graph = MemoryGraph(
        [inst("i:sbb", "Staatsbibliothek zu Berlin"), inst("i:ber", "Berlin"),
         frag("f:1", "Berlin_SBB_Or_Oct_352"), scholar("s:1", "Someone")],
        [edge("r:1", "HELD_AT", "f:1", "i:ber", data_sources=["enriched"]),
         edge("r:2", "AFFILIATED_WITH", "s:1", "i:ber", data_sources=["enriched"])])
    planner = Planner(graph)
    planner.merge_node("Institution", "Berlin", INSTITUTION_MERGES["Berlin"])
    assert INSTITUTION_MERGES["Berlin"].affiliation_target == KEEP_AFFILIATIONS
    assert graph.find("Institution", "Berlin") is not None
    assert len(edges_between(graph, "Someone", "Berlin")) == 1
    assert len(edges_between(graph, "Berlin_SBB_Or_Oct_352", "Staatsbibliothek zu Berlin")) == 1
    assert "aliases" not in graph.find("Institution", "Staatsbibliothek zu Berlin").props


def test_rename_when_canonical_absent_then_next_duplicate_merges():
    graph = MemoryGraph(
        [inst("i:a", "Tel-Aviv University"), inst("i:b", "University of Tel Aviv"),
         scholar("s:1", "A"), scholar("s:2", "B")],
        [edge("r:1", "AFFILIATED_WITH", "s:1", "i:a", data_sources=["enriched"]),
         edge("r:2", "AFFILIATED_WITH", "s:2", "i:b", data_sources=["enriched"])])
    planner = Planner(graph)
    for name in ("Tel-Aviv University", "University of Tel Aviv"):
        planner.merge_node("Institution", name, INSTITUTION_MERGES[name])
    assert [op.kind for op in planner.plan.ops] == ["rename", "merge"]
    tau = graph.find("Institution", "Tel Aviv University")
    assert tau.eid == "i:a"
    assert tau.props["aliases"] == ["Tel-Aviv University", "University of Tel Aviv"]
    assert {graph.nodes[e.start].key for e in graph.incident("i:a")} == {"A", "B"}


def test_removals_then_orphan_deletion():
    graph = MemoryGraph(
        [inst("i:cg", "Cairo Genizah"), inst("i:d", "אוסף Dropsie"), inst("i:keep", "Paris Nat. Bibl."),
         frag("f:1", "T_S_20_85"), frag("f:2", "T_S_16_134"), frag("f:3", "ENA_3012"), frag("f:4", "Other")],
        [edge("r:1", "HELD_AT", "f:1", "i:cg", data_sources=["enriched"]),
         edge("r:2", "HELD_AT", "f:2", "i:d", data_sources=["enriched"], source_books=["malk-ish-1-155"]),
         edge("r:3", "HELD_AT", "f:4", "i:d", data_sources=["enriched"], source_books=["another_book"]),
         edge("r:4", "HELD_AT", "f:3", "i:keep", data_sources=["enriched"]),
         edge("r:5", "HELD_AT", "f:4", "i:keep", data_sources=["enriched"])])
    planner = Planner(graph)
    for rule in EDGE_REMOVALS:
        planner.remove_edges(rule)
    planner.delete_orphans(["Cairo Genizah", "אוסף Dropsie", "Paris Nat. Bibl."])
    assert set(graph.edges) == {"r:3", "r:5"}            # other books / other fragments untouched
    assert graph.find("Institution", "Cairo Genizah") is None
    assert graph.find("Institution", "אוסף Dropsie") is not None
    assert graph.find("Institution", "Paris Nat. Bibl.") is not None


def test_orphan_with_unloaded_edges_is_kept():
    node = inst("i:cg", "Cairo Genizah")
    node.unloaded = 1
    graph = MemoryGraph([node], [])
    planner = Planner(graph)
    planner.delete_orphans(["Cairo Genizah"])
    assert not planner.plan.ops


def test_null_vs_list_sources_go_to_review():
    graph = MemoryGraph(
        [inst("i:c", "Bodleian Library, Oxford"), inst("i:d", "Bodl."), frag("f:1", "Bodl_X")],
        [edge("r:1", "HELD_AT", "f:1", "i:c"),                              # data_sources NULL (PGP tier)
         edge("r:2", "HELD_AT", "f:1", "i:d", data_sources=["enriched"])])
    planner = Planner(graph)
    planner.merge_node("Institution", "Bodl.", INSTITUTION_MERGES["Bodl."])
    assert not planner.plan.ops
    assert "NULL vs list" in planner.plan.review[0]["problem"]
    assert set(graph.edges) == {"r:1", "r:2"}


def test_missing_affiliation_target_goes_to_review():
    graph = MemoryGraph(
        [inst("i:cul", "Cambridge University Library"), inst("i:cam", "Cambridge"), scholar("s:1", "X")],
        [edge("r:1", "AFFILIATED_WITH", "s:1", "i:cam", data_sources=["enriched"])])
    planner = Planner(graph)
    planner.merge_node("Institution", "Cambridge", INSTITUTION_MERGES["Cambridge"])
    assert not planner.plan.ops
    assert "University of Cambridge" in planner.plan.review[0]["problem"]


def test_place_merge_records_name_variant_and_move():
    graph = MemoryGraph(
        [place("p:t", "Tarūja", name_variants="تروجة"), place("p:u", "Turūja", name_variants="تروجة"),
         place("p:r", "Raḥba"), place("p:ar", "al-Raḥba"),
         frag("f:1", "V_A_69"), Node("x:1", ["Person"], {"name": "Sittūna bt. Ḥayyīm b. Solomon"}, stub=True)],
        [edge("r:1", "MENTIONS_PLACE", "f:1", "p:u"),
         edge("r:2", "ORIGINATED_FROM", "x:1", "p:r", data_sources=["enriched"])])
    planner = Planner(graph)
    planner.merge_node("Place", "Turūja", PLACE_MERGES["Turūja"])
    planner.move_edges(EDGE_MOVES[0])
    assert graph.find("Place", "Tarūja").props["name_variants"] == "تروجة, Turūja"
    assert graph.find("Place", "Turūja") is None
    assert len(edges_between(graph, "V_A_69", "Tarūja")) == 1
    assert len(edges_between(graph, "Sittūna bt. Ḥayyīm b. Solomon", "al-Raḥba")) == 1
    assert not graph.incident("p:r")


# --------------------------------------------------------------------------
# Idempotence and revert
# --------------------------------------------------------------------------
@pytest.fixture
def mixed() -> MemoryGraph:
    """A graph that exercises removals, merges with folds, a place merge and orphans."""
    nodes = [inst("i:cul", "Cambridge University Library", data_sources=["pgp"]),
             inst("i:uni", "University of Cambridge"),
             inst("i:cam", "Cambridge"), inst("i:cucl", "Cambridge, University Library"),
             inst("i:cg", "Cairo Genizah"), inst("i:sie", "Cathedral of Siena"),
             place("p:t", "Tarūja"), place("p:u", "Turūja"),
             frag("f:1", "T_S_12_244"), frag("f:2", "T_S_K5"), frag("f:3", "Maestà_altarpiece"),
             frag("f:4", "V_A_69"), scholar("s:1", "Solomon Schechter"), scholar("s:2", "Isaacs")]
    edges = [edge("r:1", "HELD_AT", "f:1", "i:cul", data_sources=["pgp"], sub_collection="Old Series"),
             edge("r:2", "HELD_AT", "f:1", "i:cam", data_sources=["enriched"], evidence="e2", page_uuid="p2"),
             edge("r:3", "HELD_AT", "f:1", "i:cucl", data_sources=["enriched"], evidence="e3"),
             edge("r:4", "HELD_AT", "f:2", "i:cucl", data_sources=["enriched"], evidence="e4"),
             edge("r:5", "AFFILIATED_WITH", "s:1", "i:cam", data_sources=["enriched"]),
             edge("r:6", "AFFILIATED_WITH", "s:2", "i:cucl", data_sources=["enriched"]),
             edge("r:7", "HELD_AT", "f:2", "i:cg", data_sources=["enriched"]),
             edge("r:8", "HELD_AT", "f:3", "i:sie", data_sources=["enriched"]),
             edge("r:9", "MENTIONS_PLACE", "f:4", "p:u")]
    return MemoryGraph(nodes, edges)


def test_second_plan_is_empty(mixed: MemoryGraph):
    first = build_plan(mixed)
    assert first.ops and not first.review
    assert not build_plan(mixed).ops


def test_actions_replayed_on_a_copy_give_the_planned_state(mixed: MemoryGraph):
    """The plan's actions alone (placeholders resolved) reproduce the simulated end state."""
    replay = copy.deepcopy(mixed)
    plan = build_plan(mixed)
    refs: Dict[str, str] = {}
    for op in plan.ops:
        for action in op.to_json()["actions"]:
            action = {**action, **{k: ci.resolve(action[k], refs) for k in ("eid", "start", "end") if k in action}}
            created = replay.apply({**action, "ref": None} if action["do"] == "create_edge" else action)
            if action["do"] == "create_edge":
                refs[action["ref"]] = created
    assert state(replay) == state(mixed)


def memory_before(graph: MemoryGraph, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What :func:`ci.read_before` records, read from the in-memory graph."""
    nodes = {a["eid"]: {"labels": graph.nodes[a["eid"]].labels, "props": dict(graph.nodes[a["eid"]].props)}
             for a in actions if a["do"] in ("set_node", "delete_node")}
    edges = {}
    for a in actions:
        if a["do"] in ("set_edge", "delete_edge"):
            e = graph.edges[a["eid"]]
            edges[e.eid] = {"type": e.type, "start": e.start, "end": e.end, "props": dict(e.props)}
    return {"actions": actions, "nodes": nodes, "edges": edges}


def test_revert_restores_the_original_graph(mixed: MemoryGraph):
    original = copy.deepcopy(mixed)
    work = copy.deepcopy(mixed)
    plan = build_plan(mixed)
    log: List[Tuple[Dict[str, Any], Dict[str, str]]] = []
    refs: Dict[str, str] = {}
    for op in plan.ops:
        actions = [{**a, **{k: ci.resolve(a[k], refs) for k in ("eid", "start", "end") if k in a}}
                   for a in op.to_json()["actions"]]
        before = memory_before(work, actions)
        created = {}
        for action in actions:
            new = work.apply({**action, "ref": None} if action["do"] == "create_edge" else action)
            if action["do"] == "create_edge":
                created[action["ref"]] = new
        refs.update(created)
        log.append((before, created))
    assert state(work) == state(mixed)
    assert state(work) != state(original)
    node_map: Dict[str, str] = {}
    edge_map: Dict[str, str] = {}
    for before, created in reversed(log):
        for action, slot in revert_actions(before, created, node_map, edge_map):
            if action["do"] == "create_edge":
                action = {**action, "start": node_map.get(action["start"], action["start"]),
                          "end": node_map.get(action["end"], action["end"])}
            result = work.apply(action)
            if slot is not None:
                slot[0][slot[1]] = result
    assert state(work) == state(original)


# --------------------------------------------------------------------------
# CLI guards
# --------------------------------------------------------------------------
def test_apply_needs_a_reviewed_plan():
    with pytest.raises(SystemExit):
        ci.parse_args(["--apply"])


def test_apply_refuses_a_temporary_plan_dir():
    with pytest.raises(SystemExit):
        ci.parse_args(["--apply", "--plan", "/tmp/somewhere"])
    assert ci.parse_args(["--apply", "--plan", "/tmp/somewhere", "--allow-temp-plan"]).apply
