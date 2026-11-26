"""Tests of the release audit: an independent, stratified sample of the final export, an enforced target, a bound marker."""
from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from src.finetuning.qwen_hebrew.ja_ar_build import adjudicate, audit_contract
from src.finetuning.qwen_hebrew.ja_ar_build import release_audit as ra

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "logs" / "next_round" / "ja_ar_v1" / "novel1119" / "audit" / "release_audit.py"
TEXTS = ["בראשית ברא\nאלהים את השמים", "ואת הארץ והארץ\nהיתה תהו ובהו", "وقال الملك\nفي المدينة",
         "וחשך על פני\nתהום ורוח"]


def manifest_record(k: int) -> Dict[str, Any]:
    """A synthetic manifest record; strata cycle with ``k``.

    :param k: Row number.
    :return: Record.
    """
    by = ["rule", "judges", "judges"][k % 3]
    flags = ["audit_lines_dropped"] if by == "judges" and k % 6 == 1 else []
    if k % 7 == 0:
        flags.append("unread_lines_kept")
    if k == 11:
        flags.append("several_runs")
    return {"stem": f"jaar_s_{k:03d}", "pair_id": f"p{k:03d}", "row_id": f"r{k:03d}", "split": "val" if k % 10 == 3
            else "train_page", "source": "src_a" if k % 4 else "src_b", "document_group": f"g{k // 3:03d}",
            "approval": {"by": by, "fingerprint": f"fp{k:03d}", "scope": "page_text_v1"}, "gate_flags": flags,
            "remedy_class": ["assemble", "assemble_split", "assemble_crop"][k % 3 if k % 5 else 0],
            "script": "arabic" if k % 4 == 2 else "hebrew", "ja_signal": k % 4 == 1,
            "open_lines_settled": 2 if k == 13 else 0, "scrambled_lines": 1 if k == 17 else 0, "clean_changes": 0}


def make_export(tmp: Path, n: int = 40, repeat: bool = True, big_image: bool = False
                ) -> Tuple[Path, List[Dict[str, Any]]]:
    """A synthetic training export (rows/train.parquet with repeated pages, rows/val.parquet, images/) and manifest.

    :param tmp: Folder.
    :param n: Rows.
    :param repeat: Repeat every train row (a training export draws a page more than once).
    :param big_image: Make the first image 3000 px wide.
    :return: ``(export folder, manifest records)``.
    """
    export = tmp / "export"
    (export / "rows").mkdir(parents=True)
    (export / "images").mkdir()
    manifest = [manifest_record(k) for k in range(n)]
    rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for k, m in enumerate(manifest):
        buf = io.BytesIO()
        Image.new("RGB", (3000 if big_image and k == 0 else 40, 30), (k % 251, k * 47 % 251, k * 83 % 251)).save(
            buf, format="JPEG")
        data = buf.getvalue()
        sha = hashlib.sha1(data).hexdigest()
        (export / "images" / f"{sha}.jpg").write_bytes(data)
        split = "train" if m["split"] == "train_page" else "val"
        row = {"stem": m["stem"], "answer": TEXTS[k % 4] + f"\nשורה {k}", "image_sha1": sha, "question": "q"}
        rows[split] += [row, row] if repeat and split == "train" else [row]
    for split, table in rows.items():
        pq.write_table(pa.Table.from_pylist(table), export / "rows" / f"{split}.parquet")
    (export / "manifest.json").write_text(json.dumps({"sha_column": "image_sha1"}), encoding="utf-8")
    return export, manifest


def write_manifest(build: Path, manifest: List[Dict[str, Any]]) -> Path:
    """Write a build folder's manifest.jsonl.

    :param build: Build report folder.
    :param manifest: Records.
    :return: Manifest path.
    """
    build.mkdir(parents=True, exist_ok=True)
    path = build / "manifest.jsonl"
    path.write_text("".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")
    return path


def judgment(item_id: str, verdict: str = "complete_pair", **extra: Any) -> Dict[str, Any]:
    """A judge's record.

    :param item_id: Item id.
    :param verdict: Verdict.
    :param extra: Further fields (concerns).
    :return: Record.
    """
    return {"item_id": item_id, "verdict": verdict, "confidence": 5, "share_on_photo": "all",
            "uncovered_writing": "none",
            "uncovered_kind": "none", "lines_not_on_photo": [], "supplied_text": "none", "order_problem": "none",
            "garbled_lines": [], "open_lines": [], **extra}


def keys_of(n: int) -> List[Dict[str, Any]]:
    """Synthetic key lines.

    :param n: Items.
    :return: Key lines.
    """
    return [{"item_id": f"rel_{k:03d}", "pair_id": f"p{k}", "stem": f"s{k}", "stratum": f"rule:assemble|{'heb' if k % 2 else 'ja'}",
             "decision": "accept", "letters": 50, "contract": "v2", "fingerprint": "f", "facets": ["split:train"],
             "split": "train", "approval_kind": "rule", "remedy_class": "assemble", "script_class": "heb",
             "source": "a"} for k in range(n)]


def results_for(key: List[Dict[str, Any]], bad: int, bad_record: Optional[Dict[str, Any]] = None,
                judges: int = 2) -> List[Dict[str, Any]]:
    """Scored results where the first ``bad`` items are not complete.

    :param key: Key lines.
    :param bad: Items not complete.
    :param bad_record: Extra fields of a bad item's record (default: verdict photo_has_extra).
    :param judges: Judges per item (both give the same record).
    :return: Results.
    """
    out = []
    for k, line in enumerate(key):
        rec = judgment(line["item_id"], **(bad_record or {"verdict": "photo_has_extra"})) if k < bad \
            else judgment(line["item_id"])
        out.append(adjudicate.score_item(line, rec, dict(rec) if judges > 1 else None))
    return out


# ----------------------------------------------------------------------------- draw


def test_draw_covers_every_stratum_and_facet(tmp_path: Path) -> None:
    """Every primary stratum gets its rows and every facet its minimum (or all its rows when fewer exist)."""
    export, manifest = make_export(tmp_path, 60)
    rows = ra.join(manifest, ra.load_export(export))
    sample = ra.draw(manifest, ra.load_export(export), per_stratum=2, seed=7, min_items=10, per_facet=3)
    strata = Counter(r["stratum"] for r in rows)
    got = Counter(r["stratum"] for r in sample)
    assert set(got) == set(strata)
    for name, n in strata.items():
        assert got[name] >= min(2, n)
    facets = Counter(f for r in rows for f in r["facets"])
    have = Counter(f for r in sample for f in r["facets"])
    for facet, n in facets.items():
        assert have[facet] >= min(3, n), facet
    for facet in ("split:val", "split:train", "approval:judges_edit", "approval:rule", "flag:open_lines_settled",
                  "flag:scrambled_lines", "flag:unread_lines_kept", "flag:several_runs", "source:src_b",
                  "script:arabic", "remedy:assemble_crop"):
        assert facet in have, facet
    assert len({r["stem"] for r in sample}) == len(sample)
    assert len(sample) >= 10


def test_draw_is_deterministic_and_uses_the_export(tmp_path: Path) -> None:
    """Same seed, same sample, whatever the input order; the lines are the exported answer's."""
    export, manifest = make_export(tmp_path, 40)
    rows = ra.load_export(export)
    a = ra.draw(manifest, rows, 2, 11, 15, 2)
    shuffled_m, shuffled_r = list(manifest), list(rows)
    random.Random(1).shuffle(shuffled_m)
    random.Random(2).shuffle(shuffled_r)
    b = ra.draw(shuffled_m, shuffled_r, 2, 11, 15, 2)
    assert [r["stem"] for r in a] == [r["stem"] for r in b]
    c = ra.draw(manifest, rows, 2, 12, 15, 2)
    assert [r["stem"] for r in c] != [r["stem"] for r in a]
    by_stem = {r["stem"]: r for r in rows}
    for r in a:
        assert r["answer"] == by_stem[r["stem"]]["answer"]
        assert r["image_path"].endswith(f"{by_stem[r['stem']]['image_sha1']}.jpg")


def test_draw_one_row_per_document_group_inside_a_stratum(tmp_path: Path) -> None:
    """Inside a stratum, a second row of a document group is taken only when no unseen group is left."""
    export, manifest = make_export(tmp_path, 60)
    rows = ra.join(manifest, ra.load_export(export))
    groups = defaultdict(set)
    for r in rows:
        groups[r["stratum"]].add(r["document_group"])
    sample = ra.draw(manifest, ra.load_export(export), per_stratum=3, seed=3, min_items=0, per_facet=0)
    for name in groups:
        drawn = [r["document_group"] for r in sample if r["stratum"] == name and r["drawn_for"] == "stratum"]
        if len(groups[name]) >= 3:
            assert len(drawn) == len(set(drawn)) == 3, name


def test_export_rows_must_be_in_the_manifest_and_consistent(tmp_path: Path) -> None:
    """An export stem the manifest lacks, or a repeated stem with another answer, stops the draw."""
    export, manifest = make_export(tmp_path, 12)
    with pytest.raises(ValueError, match="not in the manifest"):
        ra.draw(manifest[1:], ra.load_export(export), 2, 1, 0, 0)
    with pytest.raises(ValueError, match="differ"):
        ra.dedupe_export([{"stem": "a", "answer": "x", "image_sha1": "1", "split": "train"},
                          {"stem": "a", "answer": "y", "image_sha1": "1", "split": "train"}])


# ----------------------------------------------------------------------------- items and key


def test_items_and_key_follow_the_audit_conventions(tmp_path: Path) -> None:
    """Items carry exactly the contract-v2 fields; key lines score and load as any audit's."""
    export, manifest = make_export(tmp_path, 24, big_image=True)
    sample = ra.draw(manifest, ra.load_export(export), 2, 5, 0, 1)
    items_path, key_path = tmp_path / "audit" / "judge_items_rel01.jsonl", tmp_path / "keys" / "rel01_key.jsonl"
    keys = ra.write_audit(sample, "rel01", items_path, key_path, tmp_path / "images" / "rel01")
    items = ra.load_jsonl(items_path)
    assert [i["item_id"] for i in items] == [k["item_id"] for k in keys] == [f"rel01_{n:03d}" for n in
                                                                              range(1, len(sample) + 1)]
    by_stem = {r["stem"]: r for r in ra.load_export(export)}
    for item, key in zip(items, keys):
        assert set(item) == {"item_id", "image_path", "lines", "open_lines"} and item["open_lines"] == []
        assert item["lines"] == [ln for ln in by_stem[key["stem"]]["answer"].split("\n") if ln.strip()]
        with Image.open(item["image_path"]) as im:
            assert max(im.size) <= ra.LONG_SIDE
        m = next(r for r in manifest if r["stem"] == key["stem"])
        assert key["contract"] == audit_contract.CONTRACT and key["decision"] == "accept"
        assert key["fingerprint"] == m["approval"]["fingerprint"] and key["pair_id"] == m["pair_id"]
        assert len(key["stratum"].split("|")) == 2                      # score.py's remedy and script columns
        for field in ("split", "approval_kind", "remedy_class", "script_class", "source", "facets", "letters"):
            assert field in key
    assert any(Image.open(i["image_path"]).size[0] == ra.LONG_SIDE for i in items) or \
        all(k["stem"] != "jaar_s_000" for k in keys)
    folder = tmp_path / "audit"
    (folder / "judgeA_rel01_batch01.json").write_text(json.dumps([judgment(i["item_id"]) for i in items]),
                                                      encoding="utf-8")
    a = adjudicate.load_judgments(folder, "rel01", "A")
    results = [adjudicate.score_item(k, a[k["item_id"]], None) for k in keys]
    (folder / "audit_rel01_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results),
                                                      encoding="utf-8")
    loaded = adjudicate.load_audit(folder, "rel01")
    assert set(loaded) == {k["pair_id"] for k in keys}
    assert all(v["contract"] == "v2" and v["fingerprint"] for v in loaded.values())
    with pytest.raises(FileExistsError):
        ra.write_audit(sample, "rel01", items_path, key_path, tmp_path / "images" / "rel01")


# ----------------------------------------------------------------------------- verify


def test_verify_passes_at_the_threshold_and_fails_just_under() -> None:
    """19 of 20 is 95 % and passes; 18 of 20 fails; too few judged items fail whatever the rate."""
    key = keys_of(20)
    ok = ra.verify(results_for(key, 1), key, 0.95, 20)
    assert ok["pass"] and ok["rate"] == pytest.approx(0.95) and ok["judged"] == 20 and ok["complete"] == 19
    assert 0 < ok["wilson_lower_95"] < 0.95
    assert set(ok["by_stratum"]) == {"rule:assemble|heb", "rule:assemble|ja"} and "split:train" in ok["by_facet"]
    low = ra.verify(results_for(key, 2), key, 0.95, 20)
    assert not low["pass"] and low["rate"] == pytest.approx(0.90)
    few = ra.verify(results_for(key, 0), key, 0.95, 21)
    assert not few["pass"] and any("fewer than" in r for r in few["reasons"])


def test_a_concern_on_a_complete_verdict_counts_against_the_rate() -> None:
    """complete_pair with uncovered writing (or low confidence) is not complete."""
    key = keys_of(20)
    res = ra.verify(results_for(key, 2, {"uncovered_writing": "some", "uncovered_kind": "main_text_lines"}), key,
                    0.95, 20)
    assert not res["pass"] and res["complete"] == 18 and res["concerns"]["uncovered:main_text_lines"] == 4
    res = ra.verify(results_for(key, 1, {"confidence": 1}), key, 0.95, 20)
    assert res["complete"] == 19 and res["concerns"] == {"low_confidence": 2}


def test_one_judge_per_item_releases_nothing() -> None:
    """Every item needs as many judges as an approval does; one judge's "complete" does not pass a release audit."""
    key = keys_of(20)
    one = ra.verify(results_for(key, 0, judges=1), key, 0.95, 20)
    assert not one["pass"] and len(one["under_judged"]) == 20 and one["min_judges"] == adjudicate.JUDGES_FOR_APPROVAL
    assert any("fewer than 2 judges" in r for r in one["reasons"]) and one["complete"] == 20
    assert ra.verify(results_for(key, 0, judges=1), key, 0.95, 20, min_judges=1)["pass"]
    two = ra.verify(results_for(key, 0), key, 0.95, 20)
    assert two["pass"] and two["under_judged"] == []


def test_cannot_judge_and_unjudged_items_do_not_count_as_complete() -> None:
    """An item no judge could judge stays in the denominator; an item without a record fails the audit."""
    key = keys_of(21)
    res = ra.verify(results_for(key, 1, {"verdict": "cannot_judge"}), key, 0.95, 20)
    assert res["cannot_judge_items"] == ["rel_000"] and res["judged"] == 20 and res["rate"] == pytest.approx(20 / 21)
    assert res["pass"]
    res = ra.verify(results_for(key, 2, {"verdict": "cannot_judge"}), key, 0.95, 19)
    assert not res["pass"] and res["rate"] == pytest.approx(19 / 21)
    res = ra.verify(results_for(key, 0)[1:], key, 0.95, 20)
    assert not res["pass"] and res["unjudged"] == ["rel_000"]


def test_wilson_lower_bound() -> None:
    """The bound is below the point estimate and grows with the sample."""
    assert ra.wilson_lower(0, 0) == 0.0
    assert ra.wilson_lower(95, 100) < 0.95 < ra.wilson_lower(990, 1000) + 0.02
    assert ra.wilson_lower(95, 100) < ra.wilson_lower(950, 1000)


# ----------------------------------------------------------------------------- marker


def passing_marker(tmp_path: Path) -> Tuple[Path, Path, List[Path]]:
    """A build folder with a passing marker bound to a synthetic export.

    :param tmp_path: Folder.
    :return: ``(build folder, export folder, export row files)``.
    """
    export, manifest = make_export(tmp_path, 105)
    build = tmp_path / "ds_build"
    manifest_path = write_manifest(build, manifest)
    rows = ra.load_export(export)
    sample = ra.draw(manifest, rows)
    key = [ra.item_and_key(row, f"rel01_{i:03d}", Path(row["image_path"]))[1] for i, row in enumerate(sample)]
    verdict = ra.verify(results_for(key, 0), key)
    files = ra.export_row_files(export)
    ra.write_marker(build, ra.marker("rel01", verdict, key, manifest_path, files))
    return build, export, files


def test_marker_binds_to_the_export_digests(tmp_path: Path) -> None:
    """The marker passes its own files and fails once one byte of an export file changes."""
    build, export, files = passing_marker(tmp_path)
    content = ra.require_release(build, files)
    assert content["audit"] == "rel01" and len(content["stems"]) == 100
    data = bytearray(files[0].read_bytes())
    data[len(data) // 2] ^= 0xFF
    files[0].write_bytes(bytes(data))
    with pytest.raises(SystemExit, match="NOT RELEASED"):
        ra.require_release(build, files)


def test_consumer_checks_unsampled_photos_and_independent_production_floor(tmp_path: Path) -> None:
    """An unsampled byte change, extra image or marker with a reduced floor prevents release."""
    build, export, files = passing_marker(tmp_path)
    content = ra.require_release(build, files)
    sampled = {f"{item['image_sha1']}.jpg" for item in content["items"]}
    other = next(path for path in (export / "images").iterdir() if path.name not in sampled)
    original = other.read_bytes()
    other.write_bytes(original + b"changed")
    with pytest.raises(SystemExit, match="image content"):
        ra.require_release(build, files)
    other.unlink()
    with pytest.raises(SystemExit, match="missing or extra"):
        ra.require_release(build, files)
    other.write_bytes(original)
    extra = export / "images" / "extra.jpg"
    extra.write_bytes(original)
    with pytest.raises(SystemExit, match="missing or extra"):
        ra.require_release(build, files)
    extra.unlink()
    content["verdict"]["min_items"] = 1
    (build / ra.MARKER_NAME).write_text(json.dumps(content))
    with pytest.raises(SystemExit, match="floor"):
        ra.require_release(build, files)


def test_require_release_without_marker_failed_or_changed_manifest(tmp_path: Path) -> None:
    """No marker, a failed marker, a changed manifest or no file given: refused."""
    export, manifest = make_export(tmp_path, 12)
    files = ra.export_row_files(export)
    build = tmp_path / "b"
    manifest_path = write_manifest(build, manifest)
    with pytest.raises(SystemExit, match="no release audit"):
        ra.require_release(build, files)
    key = keys_of(20)
    ra.write_marker(build, ra.marker("rel02", ra.verify(results_for(key, 5), key), key, manifest_path,
                                     files))
    with pytest.raises(SystemExit, match="failed"):
        ra.require_release(build, files)
    with pytest.raises(FileExistsError):
        ra.write_marker(build, ra.marker("rel03", ra.verify(results_for(key, 0), key), key, manifest_path,
                                         files))
    good, _, good_files = passing_marker(tmp_path / "x")
    with pytest.raises(SystemExit, match="no export file"):
        ra.require_release(good, [])
    (good / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="manifest"):
        ra.require_release(good, good_files)


def test_bind_export_only_with_identical_content(tmp_path: Path) -> None:
    """A further export binds to a passing marker only when it holds exactly the audited rows."""
    build, export, files = passing_marker(tmp_path)
    audited = ra.load_export(export)
    content = json.loads((build / ra.MARKER_NAME).read_text(encoding="utf-8"))
    other = tmp_path / "other" / "export"
    other.parent.mkdir()
    make_export(other.parent, 105, repeat=False)
    val_path = other / "rows" / "val.parquet"
    pq.write_table(pq.read_table(val_path), val_path, compression="gzip")
    with pytest.raises(SystemExit, match="NOT RELEASED"):
        ra.require_release(build, ra.export_row_files(other))
    ra.bind_export(build, content, audited, ra.load_export(other), ra.export_row_files(other))
    ra.require_release(build, ra.export_row_files(other))
    with pytest.raises(SystemExit, match="all splits"):
        ra.require_release(build, files[:1])
    (other / "rows" / "train.parquet").write_bytes(files[0].read_bytes())
    with pytest.raises(SystemExit, match="mixture of exports"):
        ra.require_release(build, ra.export_row_files(other))
    changed = [dict(r) for r in audited]
    changed[0]["answer"] += "\nתוספת"
    with pytest.raises(SystemExit, match="REFUSED"):
        ra.bind_export(build, content, audited, changed, files)


# ----------------------------------------------------------------------------- command line


def load_cli() -> ModuleType:
    """The thin command line in the audit folder.

    :return: Module.
    """
    spec = importlib.util.spec_from_file_location("release_audit_cli", str(CLI))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("min_items", [12, 100])
def test_cli_draw_verify_and_refusals(tmp_path: Path, min_items: int) -> None:
    """Only the full CLI audit can release the export. The smaller diagnostic follows the same judging checks.

    :param tmp_path: Temporary build and audit files.
    :param min_items: Requested sample size.
    """
    cli = load_cli()
    export, manifest = make_export(tmp_path, max(30, min_items))
    build = tmp_path / "ds_build"
    write_manifest(build, manifest)
    common = ["--name", "rel09", "--build-dir", str(build), "--export", str(export), "--items-dir",
              str(tmp_path / "audit"), "--scratch", str(tmp_path / "scratch"), "--keys-dir",
              str(tmp_path / "scratch" / "audit1119_key")]
    cli.main(["draw", *common, "--min-items", str(min_items), "--per-stratum", "1", "--per-facet", "1"])
    with pytest.raises(SystemExit, match="exists"):
        cli.main(["draw", *common])
    items = ra.load_jsonl(tmp_path / "audit" / "judge_items_rel09.jsonl")
    assert len(items) >= min_items
    (tmp_path / "audit" / "judgeA_rel09_batch01.json").write_text(
        json.dumps([judgment(i["item_id"]) for i in items[:-1]]), encoding="utf-8")
    with pytest.raises(SystemExit, match="no judge record"):
        cli.main(["verify", *common])
    assert not (build / ra.MARKER_NAME).exists()
    (tmp_path / "audit" / "judgeA_rel09_batch02.json").write_text(json.dumps([judgment(items[-1]["item_id"])]),
                                                                  encoding="utf-8")
    with pytest.raises(SystemExit, match="fewer than 2 judges"):      # one judge's word releases nothing
        cli.main(["verify", *common])
    assert not (build / ra.MARKER_NAME).exists()
    (tmp_path / "audit" / "judgeB_rel09_batch01.json").write_text(
        json.dumps([judgment(i["item_id"]) for i in items]), encoding="utf-8")
    cli.main(["verify", *common])
    if min_items == ra.DEFAULT_MIN_ITEMS:
        assert ra.require_release(build, ra.export_row_files(export))["scope"] == "release"
    else:
        assert json.loads((build / "DIAGNOSTIC_AUDIT.json").read_text())["scope"] == "diagnostic"
        assert not (build / ra.MARKER_NAME).exists()
        with pytest.raises(SystemExit, match="no release audit"):
            ra.require_release(build, ra.export_row_files(export))
    with pytest.raises(FileExistsError):
        cli.main(["verify", *common])
