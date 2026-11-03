"""T7 registry integrity (slow, data-gated) and unit tests of the registry repair tool (migration plan PR-5).

The data tests are skipped when the merged corpus or the decontamination registries are absent (both are
git-ignored data). They check, against the current ES ids (the ``canonical_id`` of every record of
``merged_shelfmarks.jsonl``):

* every id of every decontamination id registry is an ES id;
* every id of the sealed registry is an ES id or a sealed KTIV page id of one of its manuscripts;
* every key registry holds the facade ``loose_key`` of each of its ids; ``clean_v2_keys`` holds nothing else, and
  the only other keys of ``benchmark_keys`` are the institution-free keys an older engine wrote (kept as they
  are, read by the sealed build's FROZEN fragment key), none of them an Oxford key;
* every JSON file under the decontam directory is accounted for, and the repair tool's dry run changes nothing.

The unit tests run the tool on a small fixture repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set

import pytest

from src.datasets import shelfmarks
from src.datasets.evaluations.helper_eval_scripts import repair_decontam_registries as rdr

REPO = Path(__file__).resolve().parents[1]
MERGED = REPO / rdr.MERGED_JSONL
DECONTAM = REPO / rdr.DECONTAM_DIR
SEALED = REPO / rdr.SEALED_REGISTRY

needs_data = pytest.mark.skipif(not MERGED.exists() or not DECONTAM.is_dir(),
                                reason="merged corpus or decontamination registries not present")


def _registry_ids(path: Path) -> List[str]:
    """Ids of a registry file (a JSON list, or ``{"ids": [...]}``).

    :param path: Registry file.
    :type path: Path
    :return: The ids.
    :rtype: List[str]
    """
    return rdr.ids_of(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def es_ids() -> Set[str]:
    """The current ES ids.

    :return: The id set.
    :rtype: Set[str]
    """
    return rdr.load_es_ids(MERGED)


# ------------------------------------------------------------------------------------------------ T7 (data)


@needs_data
@pytest.mark.parametrize("name", rdr.ID_REGISTRIES)
def test_every_registry_id_is_an_es_id(name: str, es_ids: Set[str]) -> None:
    """Every id of a decontamination id registry is a current ES id (fails on pre-fix Oxford ids, F5).

    :param name: Registry file name.
    :type name: str
    :param es_ids: Current ES ids.
    :type es_ids: Set[str]
    """
    path = DECONTAM / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    missing = [i for i in _registry_ids(path) if i not in es_ids]
    assert not missing, f"{name}: {len(missing)} ids are not ES ids, e.g. {missing[:5]}"


@needs_data
def test_sealed_registry_ids_are_es_ids_or_its_ktiv_pages(es_ids: Set[str]) -> None:
    """Every sealed id is an ES id, or a KTIV page id of a manuscript the sealed registry lists.

    :param es_ids: Current ES ids.
    :type es_ids: Set[str]
    """
    if not SEALED.exists():
        pytest.skip("sealed registry not present")
    reg = json.loads(SEALED.read_text(encoding="utf-8"))
    sys_nums = {str(s) for s in reg.get("sys_nums", [])}
    bad = []
    for cid in reg["ids"]:
        page = rdr.KTIV_PAGE_ID.fullmatch(cid)
        if cid not in es_ids and not (page and page.group(1) in sys_nums):
            bad.append(cid)
    assert not bad, f"sealed ids neither ES ids nor its KTIV pages: {bad[:5]}"


@needs_data
@pytest.mark.parametrize("key_name,id_name", sorted(rdr.KEY_REGISTRIES.items()))
def test_key_registry_holds_the_facade_key_of_each_id(key_name: str, id_name: str) -> None:
    """Each id's facade ``loose_key`` is in its key registry; any other key is an institution-free legacy key.

    :param key_name: Key registry file name.
    :type key_name: str
    :param id_name: The id registry it keys.
    :type id_name: str
    """
    if not (DECONTAM / key_name).exists() or not (DECONTAM / id_name).exists():
        pytest.skip("key registry not present")
    keys = json.loads((DECONTAM / key_name).read_text(encoding="utf-8"))
    ids = _registry_ids(DECONTAM / id_name)
    facade = {shelfmarks.loose_key(i) for i in ids}
    assert "" not in facade
    assert facade <= set(keys), f"missing facade keys: {sorted(facade - set(keys))[:5]}"
    assert keys == sorted(set(keys)), "key registry must stay a sorted set"
    other = set(keys) - facade
    if key_name == "clean_v2_keys.json":
        assert not other, f"keys that are not the facade key of an id: {sorted(other)[:5]}"
    heads = {(shelfmarks.split_id(i)[0] or "").replace("_", "").lower() for i in ids} - {""}
    qualified = sorted(k for k in other if k.startswith(rdr.OXFORD_WORD) or any(k.startswith(h) for h in heads))
    assert not qualified, f"institution-qualified keys of no current id (stale): {qualified[:5]}"


@needs_data
def test_every_decontam_file_is_accounted_for() -> None:
    """No JSON file under the decontam directory escapes the repair tool's classification."""
    assert rdr.unclassified_files(REPO) == []


@needs_data
def test_repair_dry_run_changes_nothing(es_ids: Set[str]) -> None:
    """After the PR-5 repair, a dry run plans no change to any registry.

    :param es_ids: Current ES ids.
    :type es_ids: Set[str]
    """
    if not all((DECONTAM / n).exists() for n in list(rdr.ID_REGISTRIES) + list(rdr.KEY_REGISTRIES)):
        pytest.skip("a registry is not present")
    plans = rdr.build_plan(REPO, es_ids, rdr.load_oxford_map(REPO / rdr.OXFORD_MAP_CSV))
    changed = [p.path for p in plans if p.changed]
    assert not changed, f"registries the repair would still change: {changed}"


# ------------------------------------------------------------------------------------------------ unit (fixture)

OLD_OX = "Oxford_Bodleian_MS_heb_a_2_4"
NEW_OX = "Oxford_Bodleian_Bodl_MS_heb_a_2_4"
CASE_OX = "Oxford_Bodleian_Bodl_MS_Heb_d_66_13"
NEW_CASE_OX = "Oxford_Bodleian_Bodl_MS_heb_d_66_13"
CUR = "Cambridge_CUL_T_S_12_301"
UNKNOWN = "Cambridge_CUL_T_S_99_999"
FIXTURE_ES = {NEW_OX, NEW_CASE_OX, CUR}


def _write(path: Path, text: str) -> None:
    """Write a fixture file, creating its directory.

    :param path: File path.
    :type path: Path
    :param text: Content.
    :type text: str
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def fixture_repo(tmp_path: Path) -> Path:
    """A miniature repository with a merged file, every registry and an id map CSV.

    :param tmp_path: Pytest temporary directory.
    :type tmp_path: Path
    :return: The fixture repository root.
    :rtype: Path
    """
    _write(tmp_path / rdr.MERGED_JSONL, "".join(json.dumps({"canonical_id": c, "x": 1}) + "\n" for c in sorted(FIXTURE_ES)))
    _write(tmp_path / rdr.OXFORD_MAP_CSV,
           "old_canonical_id,new_canonical_id,sources,change_type,line,shelfmark_display,note\n"
           f"{OLD_OX},{NEW_OX},pgp,collapsed_into,1,x,\n{CASE_OX},{NEW_CASE_OX},pgp,collapsed_into,2,y,\n")
    d = tmp_path / rdr.DECONTAM_DIR
    bench_ids = sorted([CUR, OLD_OX, UNKNOWN])
    _write(d / "benchmark_ids.json", json.dumps(bench_ids))
    bench_keys = sorted({shelfmarks.loose_key(CUR), shelfmarks.loose_key(UNKNOWN), "oxfordbodleianmsheba2|4",
                         "oxfordmsheba2|4", "ts12|301"})
    _write(d / "benchmark_keys.json", json.dumps(bench_keys))
    _write(d / "clean_v2_ids.json", json.dumps([CUR]))
    _write(d / "clean_v2_keys.json", json.dumps([shelfmarks.loose_key(CUR)]))
    _write(d / "arabic_script_benchmark_v0.json",
           json.dumps({"benchmark": "a", "built": "2026-10-02", "ids": [CUR, CASE_OX], "pgpids": ["1", "2"]}, indent=1))
    for name in ("held_out_edition_pages_v0.json", "offbench_sample_v0.json", "two_edition_documents_v0.json"):
        _write(d / name, json.dumps({"benchmark": name[:-5], "built": "2026-10-04", "ids": [CUR], "pgpids": ["1"]},
                                    indent=1))
    _write(d / "ktiv_v19a_train_sysnums.json", json.dumps(["990001410200205171"], indent=0) + "\n")
    return tmp_path


def _sha(path: Path) -> str:
    """SHA-256 of a file.

    :param path: File.
    :type path: Path
    :return: Hex digest.
    :rtype: str
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_map_registry_id_cases() -> None:
    """Old-form Oxford ids map to the current form; a current id is untouched; others are reported, not changed."""
    oxford_map: Dict[str, Set[str]] = {OLD_OX: {NEW_OX}}
    assert rdr.map_registry_id(OLD_OX, FIXTURE_ES, oxford_map) == rdr.Resolution(OLD_OX, NEW_OX, rdr.MAPPED,
                                                                                  "id map agrees")
    assert rdr.map_registry_id(CASE_OX, FIXTURE_ES, {}).new == NEW_CASE_OX
    assert rdr.map_registry_id(CUR, FIXTURE_ES, oxford_map) == rdr.Resolution(CUR, CUR, rdr.CURRENT)
    unknown = rdr.map_registry_id(UNKNOWN, FIXTURE_ES, oxford_map)
    assert unknown.status == rdr.UNMAPPED and unknown.new is None
    assert rdr.map_registry_id("Oxford_Bodleian_MS_heb_z_1_1", FIXTURE_ES, {}).status == rdr.UNMAPPED
    assert rdr.map_registry_id(OLD_OX, FIXTURE_ES, {OLD_OX: {"Oxford_Bodleian_Bodl_MS_heb_a_2_5"}}).status == rdr.UNMAPPED
    page = rdr.map_registry_id("KTIV_990001505520205171_FL201488923", FIXTURE_ES, {})
    assert page.status == rdr.KTIV_PAGE and page.new is None


def test_regenerate_keys_drops_only_stale_oxford_keys() -> None:
    """Pre-fix Oxford keys go, facade keys come in, every other key stays; a non-Oxford addition stops the tool."""
    old = sorted(["oxfordbodleianmsheba2|4", "oxfordmsheba2|4", "ts12|301", shelfmarks.loose_key(CUR)])
    new, removed, added = rdr.regenerate_keys(old, [CUR, NEW_OX])
    assert removed == ["oxfordbodleianmsheba2|4", "oxfordmsheba2|4"]
    assert added == [shelfmarks.loose_key(NEW_OX)]
    assert new == sorted(["ts12|301", shelfmarks.loose_key(CUR), shelfmarks.loose_key(NEW_OX)])
    with pytest.raises(ValueError):
        rdr.regenerate_keys(old, [CUR, NEW_OX, "Cambridge_CUL_T_S_8_1"])


def test_detect_format_round_trips() -> None:
    """The detected format reproduces compact, indented and newline-terminated files byte for byte."""
    for text in (json.dumps(["a", "b"]), json.dumps({"ids": ["a"]}, indent=1), json.dumps(["a"], indent=0) + "\n"):
        assert rdr.dump_json(json.loads(text), rdr.detect_format(text, json.loads(text))) == text
    with pytest.raises(ValueError):
        rdr.detect_format('["a",  "b"]', ["a", "b"])


def test_run_dry_run_writes_nothing(fixture_repo: Path) -> None:
    """A dry run plans the mapping and leaves every file as it was.

    :param fixture_repo: Fixture repository.
    :type fixture_repo: Path
    """
    d = fixture_repo / rdr.DECONTAM_DIR
    before = {p.name: _sha(p) for p in d.glob("*.json")}
    plans, entries, report = rdr.run(fixture_repo, apply=False)
    assert entries == [] and not (fixture_repo / rdr.BACKUP_DIR).exists()
    assert {p.name: _sha(p) for p in d.glob("*.json")} == before
    by_name = {Path(p.path).name: p for p in plans}
    assert by_name["benchmark_ids.json"].mapped == [(OLD_OX, NEW_OX)]
    assert [r.old for r in by_name["benchmark_ids.json"].unmapped_after] == [UNKNOWN]
    assert by_name["arabic_script_benchmark_v0.json"].mapped == [(CASE_OX, NEW_CASE_OX)]
    assert "| `benchmark_ids.json` | ids | 3 | 2 | 1 | 1 |" in report


def test_run_apply_repairs_backs_up_and_is_idempotent(fixture_repo: Path) -> None:
    """Apply maps the Oxford ids in place, keeps formats, backs up byte-exactly, and a re-run changes nothing.

    :param fixture_repo: Fixture repository.
    :type fixture_repo: Path
    """
    d = fixture_repo / rdr.DECONTAM_DIR
    before = {p.name: (p.read_bytes(), _sha(p)) for p in d.glob("*.json")}
    plans, entries, _ = rdr.run(fixture_repo, apply=True)
    changed = {Path(e["path"]).name for e in entries}
    assert changed == {"benchmark_ids.json", "benchmark_keys.json", "arabic_script_benchmark_v0.json"}

    assert json.loads((d / "benchmark_ids.json").read_text()) == [CUR, UNKNOWN, NEW_OX]  # in place; unknown kept
    arabic = json.loads((d / "arabic_script_benchmark_v0.json").read_text())
    assert list(arabic) == ["benchmark", "built", "ids", "pgpids"] and arabic["ids"] == [CUR, NEW_CASE_OX]
    assert (d / "arabic_script_benchmark_v0.json").read_text() == json.dumps(arabic, indent=1)
    keys = json.loads((d / "benchmark_keys.json").read_text())
    assert keys == sorted({shelfmarks.loose_key(CUR), shelfmarks.loose_key(UNKNOWN), shelfmarks.loose_key(NEW_OX),
                           "ts12|301"})
    for name, (raw, sha) in before.items():
        if name not in changed:
            assert (d / name).read_bytes() == raw

    backup = fixture_repo / rdr.BACKUP_DIR
    manifest = json.loads((backup / rdr.BACKUP_MANIFEST).read_text())
    for e in manifest["files"]:
        name = Path(e["path"]).name
        assert (backup / name).read_bytes() == before[name][0]
        assert e["sha256_before"] == before[name][1] and e["sha256_after"] == _sha(d / name)
    assert {Path(e["path"]).name: e["entries_changed"]["ids_mapped"] for e in manifest["files"]}["benchmark_ids.json"] == 1

    plans2, entries2, _ = rdr.run(fixture_repo, apply=True)
    assert entries2 == [] and not any(p.changed for p in plans2)


def test_apply_never_overwrites_a_differing_backup(fixture_repo: Path) -> None:
    """An existing backup that differs from the file stops the apply before anything is written.

    :param fixture_repo: Fixture repository.
    :type fixture_repo: Path
    """
    _write(fixture_repo / rdr.BACKUP_DIR / "benchmark_ids.json", "[]")
    target = fixture_repo / rdr.DECONTAM_DIR / "benchmark_ids.json"
    raw = target.read_bytes()
    with pytest.raises(FileExistsError):
        rdr.run(fixture_repo, apply=True)
    assert target.read_bytes() == raw
