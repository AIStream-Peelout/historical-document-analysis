"""Tests for the ``--htr-cache-key`` provenance stamp on the normal two-reader read path.

The normal path is run end to end through ``main_async`` (``--ids --no-resolve``
with local images) with Kraken and LM Studio mocked; an autouse fixture points
the raw cache at a temp dir and fails any Kraken, LM Studio, download or
Elasticsearch call a test did not mock, so nothing here touches the services,
the production raw cache or its records.
"""
import argparse
import asyncio
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from PIL import Image

from src.datasets.consensus import two_reader_lines as trl

KEY = "MiDRASH_Gen_01@k7.0.3-ktivseg-e10-mv"
INDEX = "genizah_merged_v6"
REV = "rev-test"
W, H = 100, 200
OK_DOC, KRAKEN_FAIL_DOC, VLM_FAIL_DOC, NEW_DOC = "Cambridge_T-S_1", "Cambridge_T-S_2", "Cambridge_T-S_3", "Cambridge_T-S_4"
REPLY = json.dumps([{"text": "אבגדה הוזחט", "bbox_2d": [100, 100, 900, 200]},
                    {"text": "יכלמנ סעפצק", "bbox_2d": [100, 300, 900, 400]}], ensure_ascii=False)
KRAKEN_LINES = [{"text": "אבגדה הוזחט", "bbox": [10, 20, 90, 40], "confidence": 0.9},
                {"text": "יכלמנ סעפצק", "bbox": [10, 60, 90, 80], "confidence": 0.8}]
FRAGS = [{"text": "אבגדה הוזחט", "conf": 0.9, "box": [100.0, 100.0, 900.0, 200.0]},
         {"text": "יכלמנ סעפצק", "conf": 0.8, "box": [100.0, 300.0, 900.0, 400.0]}]
STALE_FRAGS = [{"text": "שששש", "conf": 0.5, "box": [150.0, 110.0, 850.0, 190.0]}]
CACHE_NAME = f"{OK_DOC}__0__{trl.VLM_MODEL}.json"


async def _unexpected_async(*args: Any, **kwargs: Any) -> None:
    """Stand-in for a service a test did not mock.

    :param args: Ignored.
    :param kwargs: Ignored.
    :raises AssertionError: Always.
    """
    raise AssertionError("unexpected Kraken / LM Studio call")


def _unexpected(*args: Any, **kwargs: Any) -> None:
    """Stand-in for a download or Elasticsearch request a test did not mock.

    :param args: Ignored.
    :param kwargs: Ignored.
    :raises AssertionError: Always.
    """
    raise AssertionError("unexpected network call")


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the raw cache at a temp dir and fail every service call a test did not mock.

    :param tmp_path: Pytest temp dir.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(trl, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", _unexpected_async)
    monkeypatch.setattr(trl, "transcribe_with_lm_studio", _unexpected_async)
    monkeypatch.setattr(trl, "download", _unexpected)
    monkeypatch.setattr(trl, "_es_request", _unexpected)


class FakeKraken:
    """Stand-in for ``transcribe_with_kraken_lines``: fixed lines, or a failed call for chosen images."""

    def __init__(self, fail_stems: Tuple[str, ...] = ()) -> None:
        """:param fail_stems: Prepared-image stems to answer like a failed service call."""
        self.fail_stems = fail_stems
        self.calls: List[str] = []

    async def __call__(self, model_path: str, image_path: str,
                       timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Record the image stem; the image must exist while it is read.

        :param model_path: Kraken model sent to the service.
        :param image_path: Image sent to the service.
        :param timeout: Request timeout (ignored).
        :returns: A ``/transcribe_lines`` response, or None.
        """
        assert Path(image_path).exists()
        stem = Path(image_path).stem
        self.calls.append(stem)
        return None if stem in self.fail_stems else {"lines": KRAKEN_LINES, "polygon_failures": 0,
                                                     "image_size": [W, H]}


class FakeVLM:
    """Stand-in for ``transcribe_with_lm_studio``: the fixed grounded reply, or a failed call for chosen images."""

    def __init__(self, fail_stems: Tuple[str, ...] = ()) -> None:
        """:param fail_stems: Prepared-image stems to answer like a failed LM Studio call."""
        self.fail_stems = fail_stems

    async def __call__(self, model: str, image_path: str, prompt: str, max_tokens: int = 0) -> Optional[str]:
        """Answer the fixed reply.

        :param model: LM Studio key.
        :param image_path: Image sent to the model.
        :param prompt: Prompt (must be the grounded one).
        :param max_tokens: Token budget (ignored).
        :returns: The reply, or None.
        """
        assert prompt == trl.P_GROUNDED and Path(image_path).exists()
        return None if Path(image_path).stem in self.fail_stems else REPLY


def _jpeg(shade: int) -> bytes:
    """A ``W`` x ``H`` grey JPEG (a different shade gives different bytes).

    :param shade: Grey level 0-255.
    :returns: JPEG bytes.
    """
    buf = io.BytesIO()
    Image.new("RGB", (W, H), (shade, shade, shade)).save(buf, format="JPEG")
    return buf.getvalue()


def _shade(doc_id: str) -> int:
    """Grey level of a document's test image (fixed per document, so re-runs see the same bytes).

    :param doc_id: Document id.
    :returns: Grey level.
    """
    return 255 - 40 * int(doc_id.rsplit("_", 1)[1])


def _normal_run(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
                htr_cache_key: Optional[str] = None, docs: Tuple[str, ...] = (OK_DOC,),
                kraken: Optional[FakeKraken] = None) -> Dict[str, Any]:
    """Run the normal read path (``main_async`` with the CLI's arguments) over one image per document.

    ``KRAKEN_FAIL_DOC`` fails at Kraken and ``VLM_FAIL_DOC`` at LM Studio.
    :param root: Run directory (``raw/``, ``images/``, ``jobs.jsonl``, ``ai_reads.jsonl`` live there).
    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest capture fixture.
    :param htr_cache_key: ``--htr-cache-key`` (None = flag absent).
    :param docs: Documents to read, in job order.
    :param kraken: Kraken stand-in (default one failing ``KRAKEN_FAIL_DOC``).
    :returns: ``raw`` dir, ``out`` and ``failures`` paths, the run's ``log`` text and the ``kraken`` stand-in.
    """
    root.mkdir(parents=True, exist_ok=True)
    kraken = kraken or FakeKraken(fail_stems=(f"{KRAKEN_FAIL_DOC}__0",))
    monkeypatch.setattr(trl, "RAW_DIR", root / "raw")
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", kraken)
    monkeypatch.setattr(trl, "transcribe_with_lm_studio", FakeVLM(fail_stems=(f"{VLM_FAIL_DOC}__0",)))
    jobs = root / "jobs.jsonl"
    with jobs.open("w") as fh:
        for doc in docs:
            img = root / f"source_{doc}.jpg"
            img.write_bytes(_jpeg(_shade(doc)))
            fh.write(json.dumps({"doc_id": doc, "image_index": 0, "image_url": f"https://example.org/{doc}.jpg",
                                 "local_path": str(img)}) + "\n")
    args = argparse.Namespace(                          # argparse's defaults for everything not set here
        from_consensus=None, ids=str(jobs), limit=0, out=str(root / "ai_reads.jsonl"),
        work_dir=str(root / "images"), source_index=INDEX, rematch=False, restamp=False, rekraken=False,
        kraken_model=None, htr_model_name=None, kraken_cache_suffix=None, all_cache=False, force=False,
        raw_dir=None, site_env=str(root / "missing.env"), vlm_model=trl.VLM_MODEL, vlm_revision=REV,
        htr_cache_key=htr_cache_key, min_free_gb=0.0, no_resolve=True, keep_images=False)
    capsys.readouterr()
    asyncio.run(trl.main_async(args))
    return {"raw": root / "raw", "out": root / "ai_reads.jsonl", "failures": root / "ai_reads.failures.jsonl",
            "log": capsys.readouterr().out, "root": root, "kraken": kraken}


def _masked_log(run: Dict[str, Any]) -> List[str]:
    """A run's log lines with timings, the finish time and the run directory masked.

    :param run: Output of :func:`_normal_run`.
    :returns: Comparable lines.
    """
    out = []
    for line in run["log"].splitlines():
        line = re.sub(r"\[dl [^\]]*\]", "[timing]", line.replace(str(run["root"]), "<root>"))
        out.append(re.sub(r"^done \S+ ->", "done <time> ->", line))
    return out


def _untimed(text: str) -> str:
    """Records file text with every ``ai_read.decoded_at`` blanked (re-serialised as the pipeline writes it).

    :param text: JSONL of records.
    :returns: Comparable text.
    """
    rows = []
    for line in text.splitlines():
        r = json.loads(line)
        r["ai_read"]["decoded_at"] = ""
        rows.append(json.dumps(r, ensure_ascii=False) + "\n")
    return "".join(rows)


def _failures(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A run's failure log without its timing fields.

    :param run: Output of :func:`_normal_run`.
    :returns: Failure rows.
    """
    rows = [json.loads(l) for l in run["failures"].read_text().splitlines()]
    return [{k: v for k, v in r.items() if not k.endswith("_s")} for r in rows]


def test_default_run_writes_the_legacy_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture) -> None:
    """Flag absent: raw cache, record, failure log and log lines are exactly what the pre-option code wrote."""
    run = _normal_run(tmp_path, monkeypatch, capsys, docs=(OK_DOC, KRAKEN_FAIL_DOC, VLM_FAIL_DOC))
    data = _jpeg(_shade(OK_DOC))
    _, vlm_lines = trl.parse_grounded(REPLY)
    expected_ai_read = trl.sidecar(vlm_lines, FRAGS, True, trl.VLM_MODEL, REV)
    assert [p.name for p in run["raw"].iterdir()] == [CACHE_NAME]      # failed jobs leave no entry
    assert (run["raw"] / CACHE_NAME).read_text() == json.dumps(dict(
        doc_id=OK_DOC, image_index=0, image_url=f"https://example.org/{OK_DOC}.jpg", width=W, height=H,
        sha256=hashlib.sha256(data).hexdigest(), vlm_model=trl.VLM_MODEL, vlm_raw=REPLY, vlm_lines=vlm_lines,
        parsed=True, frags=FRAGS, htr_model="MiDRASH_Gen_01"), ensure_ascii=False)
    [record] = [json.loads(l) for l in run["out"].read_text().splitlines()]
    assert list(record) == ["doc_id", "source_index", "image_index", "image_url", "image_width", "image_height",
                            "image_sha256", "ai_read"]
    assert list(record["ai_read"]) == list(expected_ai_read)            # no htr_cache_key
    assert record["ai_read"]["htr_model"] == "MiDRASH_Gen_01"
    expected_ai_read["decoded_at"] = record["ai_read"]["decoded_at"]
    assert record == {"doc_id": OK_DOC, "source_index": INDEX, "image_index": 0,
                      "image_url": f"https://example.org/{OK_DOC}.jpg", "image_width": W, "image_height": H,
                      "image_sha256": hashlib.sha256(data).hexdigest(), "ai_read": expected_ai_read}
    assert [(f["doc_id"], f["failure"]) for f in _failures(run)] == [(KRAKEN_FAIL_DOC, "kraken"),
                                                                       (VLM_FAIL_DOC, "vlm")]
    assert list(_failures(run)[1]) == ["doc_id", "image_index", "width", "height", "sha256", "failure"]
    assert _masked_log(run) == [
        f"3 jobs taken as given (--no-resolve; source_index stamp {INDEX})",
        f"3 jobs resolved against {INDEX}; 0 done; 3 to run; vlm={trl.VLM_MODEL} rule={trl.RULE_VERSION}",
        f"  1/3 {OK_DOC}#0: lines {expected_ai_read['n_lines']} agreed {expected_ai_read['n_agreed']} "
        f"frags {len(FRAGS)} parsed=True  [timing]",
        f"  2/3 {KRAKEN_FAIL_DOC}#0: FAILED kraken",
        f"  3/3 {VLM_FAIL_DOC}#0: FAILED vlm",
        "done <time> -> <root>/ai_reads.jsonl"]


def test_flag_writes_what_the_retroactive_stamp_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture) -> None:
    """With vs without the flag on the same pages: the only differences are the two stamps and the header note.

    The flagged outputs equal the unflagged ones after ``stamp_htr_cache_key.py``'s
    edit (raw ``htr_model`` := KEY, rewritten with ``write_json_atomic``; record
    ``ai_read.htr_cache_key`` := KEY), byte for byte but for ``decoded_at``.
    """
    docs = (OK_DOC, KRAKEN_FAIL_DOC, VLM_FAIL_DOC)
    plain = _normal_run(tmp_path / "plain", monkeypatch, capsys, docs=docs)
    keyed = _normal_run(tmp_path / "keyed", monkeypatch, capsys, htr_cache_key=KEY, docs=docs)
    assert keyed["kraken"].calls == plain["kraken"].calls
    entry = json.loads((plain["raw"] / CACHE_NAME).read_text())
    entry["htr_model"] = KEY
    trl.write_json_atomic(plain["raw"] / CACHE_NAME, entry)
    assert [p.name for p in keyed["raw"].iterdir()] == [CACHE_NAME]
    assert (keyed["raw"] / CACHE_NAME).read_bytes() == (plain["raw"] / CACHE_NAME).read_bytes()
    keyed_entry = json.loads((keyed["raw"] / CACHE_NAME).read_text())
    assert "frags_by_htr" not in keyed_entry
    assert trl.cached_frags(keyed_entry, KEY) == FRAGS
    assert trl.cached_frags(keyed_entry, trl.HTR_MODEL_NAME) is None    # never answers as the legacy reader
    stamped = []
    for line in plain["out"].read_text().splitlines():
        r = json.loads(line)
        r["ai_read"]["htr_cache_key"] = KEY
        stamped.append(json.dumps(r, ensure_ascii=False))
    assert _untimed(keyed["out"].read_text()) == _untimed("\n".join(stamped) + "\n")
    ar = json.loads(keyed["out"].read_text())["ai_read"]
    assert (ar["htr_model"], ar["htr_cache_key"], list(ar)[-1]) == (trl.HTR_MODEL_NAME, KEY, "htr_cache_key")
    assert _failures(keyed) == _failures(plain)
    plain_log, keyed_log = _masked_log(plain), _masked_log(keyed)
    assert keyed_log[1] == plain_log[1] + f" htr_cache_key={KEY}"
    assert keyed_log[:1] + keyed_log[2:] == plain_log[:1] + plain_log[2:]


def test_key_equal_to_the_model_name_is_the_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture) -> None:
    """``--htr-cache-key MiDRASH_Gen_01`` names the default key: raw cache and records as without the flag."""
    plain = _normal_run(tmp_path / "plain", monkeypatch, capsys)
    named = _normal_run(tmp_path / "named", monkeypatch, capsys, htr_cache_key=trl.HTR_MODEL_NAME)
    assert (named["raw"] / CACHE_NAME).read_bytes() == (plain["raw"] / CACHE_NAME).read_bytes()
    assert _untimed(named["out"].read_text()) == _untimed(plain["out"].read_text())


def test_rematch_keeps_the_key_of_flagged_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture) -> None:
    """``--rematch`` rebuilds a flagged record from its KEY fragments without Kraken; KEY and legacy never mix."""
    run = _normal_run(tmp_path, monkeypatch, capsys, htr_cache_key=KEY)
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", _unexpected_async)
    cache = run["raw"] / CACHE_NAME
    cache_before, keyed_record = cache.read_bytes(), run["out"].read_text()
    asyncio.run(trl.rematch(run["out"], tmp_path / "images", INDEX, REV))
    assert _untimed(run["out"].read_text()) == _untimed(keyed_record)      # same fragments, same stamps
    assert cache.read_bytes() == cache_before
    # a legacy record of the same image is not matched against the KEY fragments: kept as is
    legacy = json.loads(keyed_record)
    del legacy["ai_read"]["htr_cache_key"]
    legacy_record = json.dumps(legacy, ensure_ascii=False) + "\n"
    run["out"].write_text(legacy_record)
    capsys.readouterr()
    asyncio.run(trl.rematch(run["out"], tmp_path / "images", INDEX, REV))
    assert run["out"].read_text() == legacy_record
    assert f"rematch SKIPPED, no {trl.HTR_MODEL_NAME} fragments" in capsys.readouterr().out
    # and a KEY record is not matched against legacy fragments (an entry nobody stamped): kept as is
    entry = json.loads(cache_before)
    entry["htr_model"] = trl.HTR_MODEL_NAME
    entry["frags"] = STALE_FRAGS
    cache.write_text(json.dumps(entry, ensure_ascii=False))
    run["out"].write_text(keyed_record)
    asyncio.run(trl.rematch(run["out"], tmp_path / "images", INDEX, REV))
    assert run["out"].read_text() == keyed_record
    assert f"rematch SKIPPED, no {KEY} fragments" in capsys.readouterr().out


def test_rekraken_under_the_key_reuses_the_flagged_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture) -> None:
    """A later ``--rekraken --kraken-cache-suffix KEY`` finds the flagged fragments: no read, the same record."""
    run = _normal_run(tmp_path, monkeypatch, capsys, htr_cache_key=KEY)
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", _unexpected_async)
    model_path, htr_model, cache_key = trl.rekraken_names(None, None, KEY)
    assert (htr_model, cache_key) == (trl.HTR_MODEL_NAME, KEY)
    cache_before, record_before = (run["raw"] / CACHE_NAME).read_bytes(), run["out"].read_text()
    counts = asyncio.run(trl.rekraken(run["out"], tmp_path / "images", run["raw"], model_path, htr_model,
                                      cache_key, min_free_gb=0.0))
    assert (counts["skipped"], counts["krakened"], counts["rebuilt"]) == (1, 0, 1)
    assert _untimed(run["out"].read_text()) == _untimed(record_before)
    assert (run["raw"] / CACHE_NAME).read_bytes() == cache_before


def test_resume_skips_recorded_jobs_whatever_their_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture) -> None:
    """A flagged restart over a legacy run's ``--out``: its records are kept as they are, only new jobs are read."""
    first = _normal_run(tmp_path, monkeypatch, capsys)                           # legacy run: one record
    legacy_record, legacy_cache = first["out"].read_text(), (first["raw"] / CACHE_NAME).read_bytes()
    kraken = FakeKraken()
    run = _normal_run(tmp_path, monkeypatch, capsys, htr_cache_key=KEY, docs=(OK_DOC, NEW_DOC), kraken=kraken)
    assert kraken.calls == [f"{NEW_DOC}__0"]
    assert f"2 jobs resolved against {INDEX}; 1 done; 1 to run;" in run["log"]
    lines = run["out"].read_text().splitlines(keepends=True)
    assert lines[0] == legacy_record                                             # not re-read, not relabelled
    assert (run["raw"] / CACHE_NAME).read_bytes() == legacy_cache
    new = json.loads(lines[1])
    assert (new["doc_id"], new["ai_read"]["htr_cache_key"]) == (NEW_DOC, KEY)
    assert json.loads((run["raw"] / f"{NEW_DOC}__0__{trl.VLM_MODEL}.json").read_text())["htr_model"] == KEY


def test_legacy_cache_entry_is_reread_not_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture) -> None:
    """A job not yet in ``--out`` whose raw entry holds legacy fragments is Kraken-read afresh and filed under KEY."""
    first = _normal_run(tmp_path, monkeypatch, capsys)
    first["out"].unlink()                                    # e.g. a crash between the cache write and the append
    cache = first["raw"] / CACHE_NAME
    entry = json.loads(cache.read_text())
    entry["frags"] = STALE_FRAGS
    cache.write_text(json.dumps(entry, ensure_ascii=False))
    kraken = FakeKraken()
    run = _normal_run(tmp_path, monkeypatch, capsys, htr_cache_key=KEY, kraken=kraken)
    assert kraken.calls == [f"{OK_DOC}__0"]
    entry = json.loads(cache.read_text())
    assert (entry["htr_model"], entry["frags"]) == (KEY, FRAGS)
    ar = json.loads(run["out"].read_text())["ai_read"]
    assert [ln["htr_text"] for ln in ar["lines"]] == [f["text"] for f in FRAGS]
    assert ar["htr_cache_key"] == KEY


def test_htr_cache_key_error() -> None:
    """The flag goes with normal runs only; the rewrite modes are refused with their reason."""
    def ns(**kw: Any) -> argparse.Namespace:
        """:param kw: Overrides.
        :returns: CLI arguments with no mode and no key unless overridden."""
        return argparse.Namespace(**{"htr_cache_key": None, "rekraken": False, "rematch": False, "restamp": False,
                                     **kw})

    assert trl.htr_cache_key_error(ns()) is None
    assert trl.htr_cache_key_error(ns(htr_cache_key=KEY)) is None
    assert trl.htr_cache_key_error(ns(rematch=True, rekraken=True)) is None
    assert trl.htr_cache_key_error(ns(htr_cache_key=KEY, rekraken=True)) == (
        "--htr-cache-key is for normal runs: --rekraken names its fragments with --kraken-cache-suffix")
    msg = trl.htr_cache_key_error(ns(htr_cache_key=KEY, rematch=True, restamp=True))
    assert msg.startswith("--htr-cache-key is for normal runs: --rematch ") and "; --restamp " in msg
    assert trl.htr_cache_key_error(argparse.Namespace(rekraken=True, rematch=False, restamp=False)) is None


def test_cli_offers_the_flag_and_refuses_it_with_a_rewrite_mode(tmp_path: Path) -> None:
    """``--help`` lists ``--htr-cache-key``; with ``--rematch`` it is an argument error before anything runs."""
    repo = Path(trl.__file__).resolve().parents[3]
    cmd = [sys.executable, "-m", "src.datasets.consensus.two_reader_lines"]
    shown = subprocess.run(cmd + ["--help"], cwd=repo, capture_output=True, text=True, timeout=120)
    assert shown.returncode == 0 and "--htr-cache-key KEY" in shown.stdout
    scratch = ["--out", str(tmp_path / "out.jsonl"), "--work-dir", str(tmp_path / "images"),
               "--source-index", INDEX, "--site-env", str(tmp_path / "missing.env")]
    refused = subprocess.run(cmd + ["--rematch", "--htr-cache-key", KEY] + scratch, cwd=repo, capture_output=True,
                             text=True, timeout=120)
    assert refused.returncode == 2
    assert "--htr-cache-key is for normal runs: --rematch" in refused.stderr
    assert not (tmp_path / "images").exists()
