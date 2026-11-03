"""Tests of stage 5 (hold-out and split) of the novel-rows build on synthetic registries, ids and texts.

Shelf marks, ids and PGP ids are invented (box numbers no collection uses); GT texts are random words over the Hebrew
and Arabic alphabets. No real registry, index or GT file is read: resources are built from ``tmp_path`` files and
in-memory sets, and ``main`` runs with ``load_resources`` monkeypatched.
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.datasets import shelfmarks as shelf
from src.finetuning.qwen_hebrew.novel_build import holdout as ho
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build.schema import (Candidate, HoldoutResult, QualityResult, read_jsonl,
                                                           write_jsonl)

HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"                  # 22 Hebrew letters (an alphabet, not a text)
ARABIC = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"              # 28 Arabic letters (an alphabet, not a text)
REGISTRY = "test_registry_v0"


def words(alphabet: str, letters: int, seed: int) -> str:
    """Random words over an alphabet.

    :param alphabet: Letters.
    :param letters: Approximate letter count.
    :param seed: Random seed.
    :return: The text.
    """
    rng = random.Random(seed)
    out: List[str] = []
    while sum(len(w) for w in out) < letters:
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(2, 6))))
    return " ".join(out)


def cand(row_id: str, *, group: Optional[str] = None, ids: Sequence[str] = (), pgpids: Sequence[str] = (),
         shelfmark: str = "", source: str = "pgp", ktiv_id: Optional[str] = None, rnl: bool = False,
         notes: Optional[Dict[str, Any]] = None) -> Candidate:
    """A candidate.

    :param row_id: Row id.
    :param group: Document group (default from the row id).
    :param ids: Canonical ids.
    :param pgpids: PGP ids.
    :param shelfmark: Shelf mark.
    :param source: Source.
    :param ktiv_id: KTIV system number.
    :param rnl: RNL flag.
    :param notes: Notes.
    :return: The candidate.
    """
    return Candidate(row_id=row_id, source=source, document_group=group or f"group:{row_id}", canonical_ids=list(ids),
                     pgpids=list(pgpids), ktiv_id=ktiv_id, shelfmark=shelfmark, script="hebrew", languages_stated=None,
                     verdict="novel", witnesses=[], unit="document", n_letters=0, rnl=rnl, notes=notes or {})


def passed(row_id: str, side_key: str = "recto", text: Optional[str] = None) -> QualityResult:
    """A quality-passed side.

    :param row_id: Row id.
    :param side_key: Side key.
    :param text: Final text (default: random words seeded by the row id).
    :return: The result.
    """
    final = text if text is not None else words(HEBREW, 120, seed=sum(map(ord, row_id + side_key)))
    return QualityResult(row_id=row_id, side_key=side_key, ok=True, reasons=[], script="hebrew", letters_hebrew=120,
                         letters_arabic=0, tokens=None, text=final)


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """A registry directory with one synthetic registry.

    :param tmp_path: Pytest temporary directory.
    :return: The directory.
    """
    folder = tmp_path / "decontam"
    folder.mkdir()
    (folder / f"{REGISTRY}.json").write_text(json.dumps({
        "benchmark": REGISTRY, "ids": ["Cambridge_CUL_T_S_98_7", "Cambridge_CUL_T_S_99_123"], "pgpids": ["999001"],
        "sys_nums": ["990000000000000077"]}), encoding="utf-8")
    return folder


GT_DOC = words(HEBREW, 300, seed=99)
GT_ARABIC = words(ARABIC, 300, seed=98)


def resources(registry_dir: Path, *, apd: Optional[Dict[str, str]] = None,
              named: Optional[Dict[str, Dict[str, List[str]]]] = None,
              index: Optional[shelf.ShelfmarkIndex] = None) -> ho.HoldoutResources:
    """Resources over the synthetic registry, frozen and religious lists and GT sets.

    :param registry_dir: Registry directory.
    :param apd: APD exclusion identifiers.
    :param named: Named exclusions (default: one synthetic name).
    :param index: Merged-corpus index.
    :return: The resources.
    """
    frozen_ids = registry_dir / "frozen_ids.json"
    frozen_ids.write_text(json.dumps(["Cambridge_CUL_T_S_95_3"]), encoding="utf-8")
    frozen_keys = registry_dir / "frozen_keys.json"
    frozen_keys.write_text(json.dumps(["cambridgecults95|3", "olderkey|1"]), encoding="utf-8")
    religious = registry_dir / "religious.json"
    religious.write_text(json.dumps({"docs": [{"doc_id": "KTIV_990000000000000088_FL5", "sys_num": "990000000000000088",
                                               "shelf_mark": "Cambridge University Library, Cambridge, England Ms. T-S AS 999.2"}]}),
                         encoding="utf-8")
    benchmarks = [ho.registry_set(REGISTRY, registry_dir, {"999001": "T-S 93.1"}, index),
                  ho.frozen_set(frozen_ids, frozen_keys, index), ho.religious_set(religious)]
    gt = {"genizah_test_v1": [("gt_doc_1", GT_DOC), ("gt_doc_2", words(HEBREW, 300, seed=97))],
          "arabic_script_benchmark_v0": [("ar_doc_1", GT_ARABIC)]}
    return ho.make_resources(benchmarks, gt, apd=apd, index=index,
                             named=named if named is not None else {"named_doc": {"pgpids": ["999003"],
                                                                                 "marks": ["T-S 96.1"]}})


def decide(cands: Sequence[Candidate], quality: Sequence[QualityResult], res: ho.HoldoutResources) -> Dict[tuple, HoldoutResult]:
    """Decisions keyed by (row id, side key).

    :param cands: Candidates.
    :param quality: Quality results.
    :param res: Resources.
    :return: ``{(row_id, side_key): result}``.
    """
    out = ho.decide_holdout({c.row_id: c for c in cands}, quality, res)
    return {(r.row_id, r.side_key): r for r in out}


def test_benchmark_id_pgpid_and_manuscript_hits(registry_dir: Path) -> None:
    """Registry ids, PGP ids and KTIV manuscripts hold a candidate out with the registry's name.

    :param registry_dir: Directory of the synthetic registry.
    """
    res = resources(registry_dir)
    cands = [cand("pgp:1", ids=["Cambridge_CUL_T_S_98_7"]), cand("pgp:2", pgpids=["999001"]),
             cand("990000000000000077:FL9", source="ktiv", ktiv_id="990000000000000077"), cand("pgp:3", ids=["Cambridge_CUL_T_S_90_1"])]
    out = decide(cands, [passed(c.row_id) for c in cands], res)
    assert out[("pgp:1", "recto")].reasons == [f"benchmark:{REGISTRY}:Cambridge_CUL_T_S_98_7"]
    assert out[("pgp:2", "recto")].reasons == [f"benchmark:{REGISTRY}:pgpid:999001"]
    assert out[("990000000000000077:FL9", "recto")].reasons == [f"benchmark:{REGISTRY}:sys:990000000000000077"]
    for key in [("pgp:1", "recto"), ("pgp:2", "recto")]:
        assert out[key].held_out and out[key].split == "excluded"
    assert not out[("pgp:3", "recto")].held_out and out[("pgp:3", "recto")].split in ("train", "val")


def test_loose_key_hits(registry_dir: Path) -> None:
    """Loose keys of shelf-mark parts match registry ids, registered PGP marks, the frozen list and the religious pages.

    :param registry_dir: Directory of the synthetic registry.
    """
    res = resources(registry_dir)
    cands = [cand("lit:1", shelfmark="T-S 99.123", source="literature"),
             cand("pgp:4", shelfmark="T-S 93.1 + T-S 90.2"),
             cand("pgp:5", shelfmark="T-S 95.3"),
             cand("ktiv:6", shelfmark="Cambridge University Library, Cambridge, England Ms. T-S AS 999.2", source="ktiv")]
    out = decide(cands, [passed(c.row_id) for c in cands], res)
    assert out[("lit:1", "recto")].reasons == [f"benchmark:{REGISTRY}:Cambridge_CUL_T_S_99_123"]
    assert out[("pgp:4", "recto")].reasons == [f"benchmark:{REGISTRY}:pgpid:999001"]
    assert out[("pgp:5", "recto")].reasons == ["benchmark:pgp131:Cambridge_CUL_T_S_95_3"]
    assert out[("ktiv:6", "recto")].reasons == ["benchmark:religious:KTIV_990000000000000088_FL5"]


def test_religious_manuscript_holds_out_every_page(registry_dir: Path) -> None:
    """Any page of a religious-benchmark manuscript is held out.

    :param registry_dir: Directory of the synthetic registry.
    """
    res = resources(registry_dir)
    page = cand("990000000000000088:FL6", source="ktiv", ktiv_id="990000000000000088")
    out = decide([page], [passed(page.row_id, "page")], res)
    assert out[(page.row_id, "page")].reasons == ["benchmark:religious:KTIV_990000000000000088_FL5"]


def test_index_resolves_registered_pgp_ids(registry_dir: Path) -> None:
    """Through the index a registered PGP id holds out the ES id it links to.

    :param registry_dir: Directory of the synthetic registry.
    """
    index = shelf.ShelfmarkIndex([{"canonical_id": "Cambridge_CUL_T_S_97_5", "shelfmark_display": "T-S 97.5",
                                   "institution": "Cambridge University Library", "pgpids": ["999001"]}])
    res = resources(registry_dir, index=index)
    linked = cand("lit:2", ids=["Cambridge_CUL_T_S_97_5"], source="literature")
    out = decide([linked], [passed(linked.row_id)], res)
    assert f"benchmark:{REGISTRY}:pgpid:999001" in out[(linked.row_id, "recto")].reasons


def test_text_shingle_hits(registry_dir: Path) -> None:
    """Shared Hebrew 25-letter or rasm 20-letter shingles hold a side out naming the GT document.

    :param registry_dir: Directory of the synthetic registry.
    """
    res = resources(registry_dir)
    hebrew = cand("pgp:7")
    arabic = cand("apd_v3:stem_7", source="apd_v3")
    clean = cand("pgp:8")
    quality = [passed("pgp:7", text=words(HEBREW, 100, seed=5) + " " + GT_DOC[40:160]),
               passed("apd_v3:stem_7", text=GT_ARABIC[:200]),
               passed("pgp:8")]
    out = decide([hebrew, arabic, clean], quality, res)
    assert out[("pgp:7", "recto")].reasons == ["text:genizah_test_v1:gt_doc_1"]
    assert out[("apd_v3:stem_7", "recto")].reasons == ["text:arabic_script_benchmark_v0:ar_doc_1"]
    assert not out[("pgp:8", "recto")].held_out


def test_named_exclusion(registry_dir: Path) -> None:
    """Named exclusions match by PGP id and by shelf mark; the built-in list is empty and a file supplies names.

    :param registry_dir: Directory of the synthetic registry.
    """
    res = resources(registry_dir)
    by_pgpid = cand("pgp:9", pgpids=["999003"])
    by_mark = cand("lit:9", shelfmark="T-S 96.1", source="literature")
    out = decide([by_pgpid, by_mark], [passed("pgp:9"), passed("lit:9")], res)
    assert out[("pgp:9", "recto")].reasons == ["exclusion:named:named_doc"]
    assert out[("lit:9", "recto")].reasons == ["exclusion:named:named_doc"]
    assert ho.NAMED_EXCLUSIONS == {}  # the names are a private list, passed in by file or env
    from_file = {n.name: n for n in ho.compile_named({"named_a": {"pgpids": ["999001"], "marks": ["T-S 1.1"]},
                                                       "named_b": {"marks": ["T-S 2.2"]}})}
    assert set(from_file) == {"named_a", "named_b"} and from_file["named_a"].pgpids == {"999001"}


def test_apd_exclusion_file(registry_dir: Path, tmp_path: Path) -> None:
    """APD exclusion lists match by document id, row stem and bare id lines.

    :param registry_dir: Directory of the synthetic registry.
    :param tmp_path: Pytest temporary directory.
    """
    path = tmp_path / "apd_exclusions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"source": "apd", "doc_id": "Synthetic_Doc_1", "reason": "benchmark_holdout"},
        {"source": "apd", "stem": "apd_Synthetic_Doc_2_0_page", "ids": [], "pgpids": []},
        "Synthetic_Doc_3"]) + "\n", encoding="utf-8")
    res = resources(registry_dir, apd=ho.load_apd_exclusions([path]))
    cands = [cand("apd_v3:apd_Synthetic_Doc_1_0_page", ids=["apd:Synthetic_Doc_1"], source="apd_v3"),
             cand("apd_v3:apd_Synthetic_Doc_2_0_page", ids=["apd:Synthetic_Doc_2"], source="apd_v3",
                  notes={"stem": "apd_Synthetic_Doc_2_0_page"}),
             cand("apd_v3:apd_Synthetic_Doc_3_0_page", ids=["apd:Synthetic_Doc_3"], source="apd_v3"),
             cand("apd_v3:apd_Synthetic_Doc_4_0_page", ids=["apd:Synthetic_Doc_4"], source="apd_v3")]
    out = decide(cands, [passed(c.row_id) for c in cands], res)
    assert out[(cands[0].row_id, "recto")].reasons == ["exclusion:derive_v25a_apd:Synthetic_Doc_1"]
    assert out[(cands[1].row_id, "recto")].reasons == ["exclusion:derive_v25a_apd:apd_Synthetic_Doc_2_0_page"]
    assert out[(cands[2].row_id, "recto")].reasons == ["exclusion:derive_v25a_apd:Synthetic_Doc_3"]
    assert not out[(cands[3].row_id, "recto")].held_out


def test_identity_holds_out_the_whole_group(registry_dir: Path) -> None:
    """An identity hit on one candidate holds out every side of its document group.

    :param registry_dir: Directory of the synthetic registry.
    """
    res = resources(registry_dir)
    bench = cand("pgp:10", group="group:joined", ids=["Cambridge_CUL_T_S_98_7"])
    other = cand("fjp:10", group="group:joined", source="fjp")
    out = decide([bench, other], [passed("fjp:10", "recto"), passed("fjp:10", "verso")], res)
    assert {r.split for r in out.values()} == {"excluded"}
    assert out[("fjp:10", "verso")].reasons == [f"benchmark:{REGISTRY}:Cambridge_CUL_T_S_98_7"]
    assert ("pgp:10", "recto") not in out


def test_split_is_deterministic_and_keeps_groups_together(registry_dir: Path) -> None:
    """The split is a function of the document group: repeatable, and every side of a group shares it.

    :param registry_dir: Directory of the synthetic registry.
    """
    res = resources(registry_dir)
    cands = [cand(f"row:{i}:{part}", group=f"group:{i}") for i in range(300) for part in ("a", "b")]
    quality = [passed(c.row_id, side) for c in cands for side in ("recto", "verso")]
    first = decide(cands, quality, res)
    again = decide(cands, quality, res)
    assert first == again
    splits: Dict[str, set] = {}
    for c in cands:
        for side in ("recto", "verso"):
            splits.setdefault(c.document_group, set()).add(first[(c.row_id, side)].split)
    assert all(len(s) == 1 for s in splits.values())
    assert {next(iter(s)) for s in splits.values()} == {"train", "val"}
    assert ho.split_of("group:1") == ho.split_of("group:1")


def test_val_share_is_about_five_percent() -> None:
    """About 5 % of groups land in validation (and the share follows ``val_share``)."""
    share = sum(ho.split_of(f"synthetic-group-{i}") == "val" for i in range(2000)) / 2000
    assert 0.035 <= share <= 0.065
    assert sum(ho.split_of(f"synthetic-group-{i}", 0.2) == "val" for i in range(2000)) / 2000 == pytest.approx(0.2, abs=0.03)


def test_rnl_is_copied_and_failed_sides_get_no_record(registry_dir: Path) -> None:
    """The RNL flag is copied; quality-failed sides get no hold-out record.

    :param registry_dir: Directory of the synthetic registry.
    """
    res = resources(registry_dir)
    rnl = cand("fjp:11", rnl=True, source="fjp")
    plain = cand("fjp:12", source="fjp")
    failed = QualityResult("fjp:12", "verso", False, ["hygiene_chars"], "hebrew", 50, 0, None, "x")
    out = decide([rnl, plain], [passed("fjp:11"), passed("fjp:12"), failed], res)
    assert out[("fjp:11", "recto")].rnl is True and out[("fjp:12", "recto")].rnl is False
    assert ("fjp:12", "verso") not in out


def test_main_writes_the_stage_file(registry_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI reads the stage files, passes its options to the resources and writes the holdout stage.

    :param registry_dir: Directory of the synthetic registry.
    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    work = tmp_path / "work"
    cands = [cand("pgp:20", ids=["Cambridge_CUL_T_S_98_7"]), cand("pgp:21", rnl=True)]
    write_jsonl(paths.stage_path(work, "images"), cands)
    write_jsonl(paths.stage_path(work, "quality"), [passed("pgp:20"), passed("pgp:21")])
    seen: Dict[str, Any] = {}

    def fake_resources(**kwargs: Any) -> ho.HoldoutResources:
        """Synthetic resources; records the arguments.

        :param kwargs: ``load_resources`` arguments.
        :return: The resources.
        """
        seen.update(kwargs)
        return resources(registry_dir)

    monkeypatch.setattr(ho, "load_resources", fake_resources)
    monkeypatch.delenv(ho.APD_EXCLUSIONS_ENV, raising=False)
    ho.main(["--work-dir", str(work), "--no-index", "--val-share", "0.05"])
    out = {r.row_id: r for r in read_jsonl(paths.stage_path(work, "holdout"), HoldoutResult)}
    assert out["pgp:20"].split == "excluded" and out["pgp:21"].rnl
    assert seen["use_index"] is False and seen["apd_files"] == []
