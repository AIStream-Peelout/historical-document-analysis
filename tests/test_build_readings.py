"""Tests for the per-page readings sidecars (``src/datasets/consensus/build_readings.py``, schema ``readings-v1``).

Pages are synthetic but their policy traces are the real thing: the decode policy is run (``pdp.decode_page``, a fake
reader instead of LM Studio) over pseudo-random Hebrew pages, and the raw-cache entry and record are assembled the way
``two_reader_lines.stage2_policy`` / ``revisit`` assemble them. An autouse fixture fails any LM Studio, Kraken,
download or Elasticsearch call, so nothing here touches the services or the production records.
"""
import asyncio
import copy
import hashlib
import json
import random
from collections import Counter
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from src.datasets.consensus import build_readings as br
from src.datasets.consensus import decode_policy as dp
from src.datasets.consensus import pipeline_decode_policy as pdp
from src.datasets.consensus import two_reader_lines as trl

HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"
W, H = 1000, 1200
LINE_H, PITCH, TOP = 30, 40, 100
MODEL = "qwen3-vl-8b-heb-v22i-a50"
KEY = "MiDRASH_Gen_01@k7.0.3-ktivseg-e10-mv"
INDEX = "genizah_merged_v8"
REV = "interp-test"
TH = dp.thresholds_for(trl.VLM_MAX_TOKENS)
BUILT_AT = "2026-10-13T22:00:00+00:00"
T0, T1, T2 = "2026-10-13T10:00:00+00:00", "2026-10-13T12:00:00+00:00", "2026-10-13T14:00:00+00:00"
PxLine = Tuple[str, List[int]]
Pair = Tuple[Dict[str, Any], Dict[str, Any]]


# ----------------------------------------------------------------------------- synthetic pages and policy runs


def random_line(seed: int, n_letters: int = 30) -> str:
    """A line of pseudo-random Hebrew words (distinct per seed).

    :param seed: Random seed.
    :param n_letters: Letters in the line.
    :returns: Line text.
    """
    rng = random.Random(seed)
    letters = [rng.choice(HEBREW) for _ in range(n_letters)]
    words, i = [], 0
    while i < len(letters):
        k = rng.randint(3, 6)
        words.append("".join(letters[i:i + k]))
        i += k
    return " ".join(words)


def column(x0: int, x1: int, n: int, seed: int, top: int = TOP) -> List[PxLine]:
    """Lines of one column, one pitch apart, with pixel boxes.

    :param x0: Left edge (px).
    :param x1: Right edge (px).
    :param n: Number of lines.
    :param seed: Seed of the first line's text.
    :param top: y of the first line (px).
    :returns: ``(text, [x0, y0, x1, y1])`` per line, top to bottom.
    """
    return [(random_line(seed + i), [x0, top + i * PITCH, x1, top + i * PITCH + LINE_H]) for i in range(n)]


RIGHT = column(550, 950, 20, seed=100)
LEFT = column(50, 450, 20, seed=500)
TWO_COLUMNS = LEFT + RIGHT
SINGLE = column(100, 900, 20, seed=900)


def frags_of(lines: Sequence[PxLine]) -> List[Dict[str, Any]]:
    """The pipeline's Kraken fragments of some lines (``run_kraken``'s 0-1000 scaling).

    :param lines: ``(text, pixel box)`` per line.
    :returns: ``{"text", "conf", "box"}`` per line.
    """
    return [{"text": t, "conf": 0.9, "box": [1000 * b[0] / W, 1000 * b[1] / H, 1000 * b[2] / W, 1000 * b[3] / H]}
            for t, b in lines]


def reply_of(lines: Sequence[PxLine], crop: Optional[Sequence[int]] = None) -> str:
    """A grounded JSON reply reading some lines, boxes in the crop's 0-1000 frame (the page's when ``crop`` is None).

    :param lines: ``(text, pixel box)`` per line.
    :param crop: Pixel crop the reply was read from.
    :returns: Reply text.
    """
    x0, y0, x1, y1 = crop or (0, 0, W, H)
    entries = [{"text": t, "bbox_2d": [round(1000 * (b[0] - x0) / (x1 - x0)), round(1000 * (b[1] - y0) / (y1 - y0)),
                                      round(1000 * (b[2] - x0) / (x1 - x0)), round(1000 * (b[3] - y0) / (y1 - y0))]}
               for t, b in lines]
    return json.dumps(entries, ensure_ascii=False)


def loop_reply(lines: Sequence[PxLine], crop: Optional[Sequence[int]] = None) -> str:
    """A looping reply: three real lines, then the fourth over and over.

    :param lines: The region's lines.
    :param crop: Pixel crop the reply was read from.
    :returns: Reply text.
    """
    return reply_of(list(lines[:3]) + [lines[3]] * 30, crop)


def parsed_lines(reply: str) -> List[Dict[str, Any]]:
    """Lines the pipeline parses from a reply.

    :param reply: Reply text.
    :returns: ``{"text", "box"}`` per line.
    """
    return trl.parse_grounded(reply)[1]


def lines_in(lines: Sequence[PxLine], crop: Sequence[int]) -> List[PxLine]:
    """The lines whose box centre lies inside a crop.

    :param lines: ``(text, pixel box)`` per line.
    :param crop: Pixel crop.
    :returns: Those lines.
    """
    return [(t, b) for t, b in lines if crop[0] <= (b[0] + b[2]) / 2 <= crop[2] and crop[1] <= (b[1] + b[3]) / 2 <= crop[3]]


def units_of(frags: List[Dict[str, Any]], whole: Optional[str] = None, gap: Optional[int] = None) -> Dict[str, Tuple[int, int, int, int]]:
    """Crops of a page's plan by unit name, with the column halves of every column unit.

    :param frags: Fragments.
    :param whole: Whole-page reply (None: none yet).
    :param gap: Ink gap.
    :returns: ``unit -> pixel crop``.
    """
    plan = pdp.plan_from_frags(frags, W, H, whole, parsed_lines(whole) if whole else None, gap, TH)
    units, _, _ = pdp.units_from_frags(frags, W, H)
    crops = {u.name: u.crop for u in plan.units}
    for u in plan.units:
        if u.role == dp.ROLE_COLUMN:
            crops.update({h.name: h.crop for h in pdp.column_halves(plan, u, units, TH)})
    return crops


class FakeReader:
    """Stand-in for the pipeline's page reader in :func:`pdp.decode_page`: canned replies, parsed and mapped."""

    def __init__(self, replies: Dict[str, Optional[str]]) -> None:
        """:param replies: ``unit -> reply``."""
        self.replies = replies

    async def read(self, unit: str, role: str, crop: Optional[Tuple[int, int, int, int]]) -> Optional[pdp.Reply]:
        """Answer like ``PolicyPageReader.read``.

        :param unit: Unit name.
        :param role: Unit role.
        :param crop: Pixel crop (None = the page).
        :returns: The reply.
        """
        raw = self.replies[unit]
        parsed, lines = trl.parse_grounded(raw)
        if crop is not None:
            lines = pdp.crop_lines_to_page(lines, crop, W, H)
        return pdp.Reply(unit, role, crop, raw, parsed, lines)


def decode(lines: Sequence[PxLine], replies: Dict[str, Optional[str]], whole: Optional[pdp.Reply] = None) -> pdp.PolicyResult:
    """Run the real decode policy over a synthetic page.

    :param lines: The page's lines (Kraken reads them all).
    :param replies: ``unit -> reply`` for every unit the policy may read.
    :param whole: The cached whole-page read of a revisit.
    :returns: The decoded page.
    """
    result = asyncio.run(pdp.decode_page(frags_of(lines), W, H, TH, FakeReader(replies).read, lambda: 500, whole=whole))
    assert result is not None
    return result


def make_pair(doc_id: str, image_index: int, frags: List[Dict[str, Any]], vlm_raw: Any, vlm_lines: List[Dict[str, Any]],
              parsed: bool, trace: Optional[Dict[str, Any]] = None, decoded_at: str = T0,
              key: Optional[str] = KEY) -> Pair:
    """A record and its raw-cache entry, assembled as ``stage2`` / ``stage2_policy`` assemble them.

    :param doc_id: Document id.
    :param image_index: Image position.
    :param frags: Kraken fragments.
    :param vlm_raw: The entry's ``vlm_raw``.
    :param vlm_lines: The entry's ``vlm_lines`` (the lines the record is built from).
    :param parsed: The entry's ``parsed``.
    :param trace: The policy trace (None: a page read without the policy).
    :param decoded_at: The record's ``decoded_at``.
    :param key: HTR cache key (None: an entry and record that predate keys).
    :returns: ``(record, entry)``.
    """
    url = f"https://example.org/{doc_id}/{image_index}.jpg"
    sha = hashlib.sha256(url.encode()).hexdigest()
    entry: Dict[str, Any] = dict(doc_id=doc_id, image_index=image_index, image_url=url, width=W, height=H, sha256=sha,
                                 vlm_model=MODEL, vlm_raw=vlm_raw, vlm_lines=vlm_lines, parsed=parsed, frags=frags)
    ai_read = trl.sidecar(vlm_lines, frags, parsed, MODEL, REV)
    ai_read["decoded_at"] = decoded_at
    if key:
        entry["htr_model"] = key
        ai_read["htr_cache_key"] = key
    if trace is not None:
        entry["decode_policy"] = trace
        ai_read["decode_policy"] = pdp.policy_stamp(trace)
    record = {"doc_id": doc_id, "source_index": INDEX, "image_index": image_index, "image_url": url,
              "image_width": W, "image_height": H, "image_sha256": sha, "ai_read": ai_read}
    return record, entry


def legacy_pair(doc_id: str = "Cambridge_T-S_1", image_index: int = 0, decoded_at: str = T0,
                key: Optional[str] = KEY) -> Pair:
    """A single-column page read whole, without the policy.

    :param doc_id: Document id.
    :param image_index: Image position.
    :param decoded_at: Record time.
    :param key: HTR cache key.
    :returns: ``(record, entry)``.
    """
    raw = reply_of(SINGLE)
    parsed, lines = trl.parse_grounded(raw)
    return make_pair(doc_id, image_index, frags_of(SINGLE), raw, lines, parsed, None, decoded_at, key)


def policy_pair(lines: Sequence[PxLine], replies: Dict[str, Optional[str]], doc_id: str, image_index: int = 0,
                decoded_at: str = T0, whole: Optional[pdp.Reply] = None, old_whole: Optional[List[Dict[str, Any]]] = None) -> Pair:
    """A page read under the policy (live, or a revisit when ``whole`` is given).

    :param lines: The page's lines.
    :param replies: ``unit -> reply``.
    :param doc_id: Document id.
    :param image_index: Image position.
    :param decoded_at: Record time.
    :param whole: The cached whole-page read of a revisit.
    :param old_whole: Lines stored with that cached read (``cached_whole_page``), as ``revisit`` keeps them.
    :returns: ``(record, entry)``.
    """
    result = decode(lines, replies, whole)
    trace = {**result.trace, "decided_at": decoded_at}
    if whole is not None:
        trace["cached_whole_page"] = {"vlm_lines": old_whole if old_whole is not None else whole.lines, "parsed": True}
    return make_pair(doc_id, image_index, frags_of(lines), result.vlm_raw, result.lines, result.parsed, trace, decoded_at)


def columns_pair(doc_id: str = "Manchester_JRL_B_1", image_index: int = 0, decoded_at: str = T0) -> Pair:
    """A two-column page whose column crops read healthy (requests: col0, col1).

    :param doc_id: Document id.
    :param image_index: Image position.
    :param decoded_at: Record time.
    :returns: ``(record, entry)``.
    """
    crops = units_of(frags_of(TWO_COLUMNS), gap=500)
    replies = {name: reply_of(lines_in(TWO_COLUMNS, crop), crop) for name, crop in crops.items()}
    return policy_pair(TWO_COLUMNS, replies, doc_id, image_index, decoded_at)


def stuck_columns_pair(doc_id: str = "Manchester_JRL_C_1", image_index: int = 0, decoded_at: str = T0) -> Pair:
    """A two-column page whose left column loops as a crop and as halves: its share of the whole-page read is used.

    Requests: col0, col1, col1.top, col1.bottom, page (the whole-page read comes last).

    :param doc_id: Document id.
    :param image_index: Image position.
    :param decoded_at: Record time.
    :returns: ``(record, entry)``.
    """
    crops = units_of(frags_of(TWO_COLUMNS), gap=500)
    replies = {name: reply_of(lines_in(TWO_COLUMNS, crop), crop) for name, crop in crops.items()}
    for name in ("col1", "col1.top", "col1.bottom"):
        replies[name] = loop_reply(lines_in(TWO_COLUMNS, crops[name]), crops[name])
    replies["page"] = reply_of(TWO_COLUMNS)
    return policy_pair(TWO_COLUMNS, replies, doc_id, image_index, decoded_at)


def rescue_replies(failing_halves: bool = False) -> Dict[str, Optional[str]]:
    """Replies of a rescued single-column page: a looping whole-page read, then the two halves.

    :param failing_halves: Make both halves loop too.
    :returns: ``unit -> reply``.
    """
    crops = units_of(frags_of(SINGLE), loop_reply(SINGLE))
    make = loop_reply if failing_halves else (lambda lines, crop: reply_of(lines, crop))
    return {"page": loop_reply(SINGLE), **{name: make(lines_in(SINGLE, crop), crop) for name, crop in crops.items()}}


def rescue_pair(doc_id: str = "Paris_AIU_IX_B_1", image_index: int = 0, decoded_at: str = T0,
                failing_halves: bool = False) -> Pair:
    """A single-column page whose whole-page read loops, rescued as two halves (requests: page, top, bottom).

    :param doc_id: Document id.
    :param image_index: Image position.
    :param decoded_at: Record time.
    :param failing_halves: Both halves loop too: the page falls back to the whole-page read.
    :returns: ``(record, entry)``.
    """
    return policy_pair(SINGLE, rescue_replies(failing_halves), doc_id, image_index, decoded_at)


def build(pair: Pair, relpath: str = "raw/x.json") -> Dict[str, Any]:
    """The sidecar of a pair with a fixed build time.

    :param pair: ``(record, entry)``.
    :param relpath: Raw-cache path stored in the provenance.
    :returns: The sidecar.
    """
    return br.readings_for_page(pair[0], pair[1], relpath, BUILT_AT)


def reader(doc: Dict[str, Any], rid: str) -> Dict[str, Any]:
    """One reader of a sidecar by id.

    :param doc: The sidecar.
    :param rid: Reader id.
    :returns: The reader.
    """
    return next(r for r in doc["readers"] if r["id"] == rid)


def ids(doc: Dict[str, Any]) -> List[str]:
    """Reader ids of a sidecar, in order.

    :param doc: The sidecar.
    :returns: The ids.
    """
    return [r["id"] for r in doc["readers"]]


# ----------------------------------------------------------------------------- service guard and file helpers


async def _unexpected_async(*args: Any, **kwargs: Any) -> None:
    """Stand-in for a service a test did not mock.

    :param args: Ignored.
    :param kwargs: Ignored.
    :raises AssertionError: Always.
    """
    raise AssertionError("unexpected Kraken / LM Studio call")


def _unexpected(*args: Any, **kwargs: Any) -> None:
    """Stand-in for a download or Elasticsearch request.

    :param args: Ignored.
    :param kwargs: Ignored.
    :raises AssertionError: Always.
    """
    raise AssertionError("unexpected network call")


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail every service call: the builder reads files only.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(trl, "transcribe_with_kraken_lines", _unexpected_async)
    monkeypatch.setattr(trl, "transcribe_with_lm_studio", _unexpected_async)
    monkeypatch.setattr(trl, "download", _unexpected)
    monkeypatch.setattr(trl, "_es_request", _unexpected)


def write_raw(raw_dir: Path, pair: Pair, suffixless: bool = False) -> Path:
    """Write a pair's raw-cache entry where the pipeline would.

    :param raw_dir: Raw-cache directory.
    :param pair: ``(record, entry)``.
    :param suffixless: Use the model-less name of the first pilot runs.
    :returns: The file.
    """
    record, entry = pair
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = trl.raw_cache_path(record, MODEL, for_write=True, raw_dir=raw_dir)
    if suffixless:
        path = raw_dir / f"{record['doc_id']}__{record['image_index']}.json"
    path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    return path


def write_records(path: Path, records: Sequence[Dict[str, Any]], tail: str = "") -> Path:
    """Write a records JSONL.

    :param path: File.
    :param records: Records in file order.
    :param tail: Text appended after the last newline (a record still being written).
    :returns: The file.
    """
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records) + tail, encoding="utf-8")
    return path


def workspace(tmp_path: Path, pairs: Sequence[Pair]) -> Tuple[Path, Path, Path, Path]:
    """A records file, raw cache, sidecar directory and manifest path holding some pages.

    :param tmp_path: Pytest temp dir.
    :param pairs: The pages, in file order.
    :returns: ``(records file, raw dir, out dir, manifest)``.
    """
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "readings"
    for pair in pairs:
        write_raw(raw_dir, pair)
    records = write_records(tmp_path / "records.jsonl", [p[0] for p in pairs])
    return records, raw_dir, out_dir, tmp_path / "readings_manifest.jsonl"


def sidecar_files(out_dir: Path) -> List[Path]:
    """The sidecars in a directory.

    :param out_dir: Sidecar directory.
    :returns: Sorted ``*.json`` files.
    """
    return sorted(out_dir.glob("*.json"))


def manifest_rows(manifest: Path) -> List[Dict[str, Any]]:
    """The manifest's lines.

    :param manifest: Manifest JSONL.
    :returns: Parsed lines (empty without a file).
    """
    return [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()] if manifest.exists() else []


# ----------------------------------------------------------------------------- boxes


def test_int_box_clamps_rounds_and_orders_the_corners() -> None:
    """Boxes leave as ints inside the 0-1000 frame with min corner first."""
    assert br.int_box([10.4, 20.5, 99.6, 30.2]) == [10, 20, 100, 30]
    assert br.int_box([-5, -1.2, 1200, 1000.4]) == [0, 0, 1000, 1000]
    assert br.int_box([300, 80, 100, 20]) == [100, 20, 300, 80]


def test_reader_lines_map_crop_boxes_to_the_page_frame_numerically() -> None:
    """A box in a crop's frame lands where the pixel arithmetic says: crop (500,0,1000,1000) of a 1000x2000 page."""
    lines = [{"text": "אבג", "box": [100.0, 200.0, 900.0, 300.0]}, {"text": "דהו", "box": [0.0, 1000.0, 500.0, 1100.0]}]
    mapped = br.reader_lines(lines, (500, 0, 1000, 1000), 1000, 2000)
    assert mapped[0] == {"index": 0, "text": "אבג", "box": [550, 100, 950, 150]}     # x: 500+500*.1 .. 500+500*.9; y: 200..300 px / 2
    assert mapped[1]["box"] == [500, 500, 750, 550]                                 # a box may run past the crop edge (1100 px / 2)
    bottom = br.reader_lines([{"text": "זחט", "box": [0.0, 900.0, 1000.0, 1200.0]}], (0, 1500, 1000, 2000), 1000, 2000)
    assert bottom[0]["box"] == [0, 975, 1000, 1000]                                 # y2 = (1500+500*1.2)/2 = 1050 clamps to the page
    assert br.reader_lines(lines) == [{"index": 0, "text": "אבג", "box": [100, 200, 900, 300]},
                                      {"index": 1, "text": "דהו", "box": [0, 1000, 500, 1000]}]
    assert br.crop_box_1000((500, 0, 1000, 1000), 1000, 2000) == [500, 0, 1000, 500]
    assert br.crop_box_1000(None, 1000, 2000) is None


# ----------------------------------------------------------------------------- legacy pages


def test_legacy_page_has_exactly_two_readers_and_the_final_lines_verbatim() -> None:
    """A page read whole: the page read and Kraken, no policy, ``final.source`` legacy, the record's lines untouched."""
    record, entry = legacy_pair()
    assert isinstance(entry["vlm_raw"], str)
    doc = build((record, entry), "raw/Cambridge_T-S_1__0__m.json")
    assert list(doc) == ["schema_version", "doc_id", "image_index", "image_url", "image_sha256", "image_width",
                         "image_height", "source_index", "decoded_at", "rule_version", "vlm_model", "vlm_revision",
                         "policy", "readers", "final", "provenance"]
    assert doc["schema_version"] == "readings-v1" and doc["policy"] is None
    assert (doc["doc_id"], doc["image_index"], doc["image_sha256"], doc["source_index"]) == (
        "Cambridge_T-S_1", 0, record["image_sha256"], INDEX)
    assert (doc["image_width"], doc["image_height"], doc["decoded_at"], doc["vlm_revision"]) == (W, H, T0, REV)
    assert doc["rule_version"] == record["ai_read"]["rule_version"] and doc["vlm_model"] == MODEL
    assert ids(doc) == [f"vlm:{MODEL}:page", f"kraken:{KEY}"]
    page, kraken = doc["readers"]
    assert (page["kind"], page["scope"], page["unit"], page["reply_index"], page["parsed"]) == ("vlm", "page", None, 0, True)
    assert page["health_label"] is None and page["crop_box_px"] is None and page["role"] == "page"
    assert [ln["text"] for ln in page["lines"]] == [t for t, _ in SINGLE]
    assert all(isinstance(v, int) for ln in page["lines"] for v in ln["box"])
    assert (kraken["kind"], kraken["scope"], kraken["htr_model"], kraken["cache_key"]) == ("htr", "page", "MiDRASH_Gen_01", KEY)
    assert [(k["index"], k["text"], k["conf"]) for k in kraken["lines"]] == [(i, t, 0.9) for i, (t, _) in enumerate(SINGLE)]
    assert kraken["lines"][0]["box"] == [100, 83, 900, 108]                            # 100/1000, 100/1200 .. 130/1200 of 1000
    assert doc["final"] == {"source": "legacy", "lines": record["ai_read"]["lines"]}
    assert doc["provenance"] == {"raw_cache": "raw/Cambridge_T-S_1__0__m.json", "built_at": BUILT_AT,
                                 "builder": br.BUILDER, "reply_mapping": "single", "notes": []}
    json.dumps(doc, ensure_ascii=False)


def test_legacy_page_shows_the_stored_parse_the_final_lines_came_from() -> None:
    """A parser that improved since the page was read must not make the page reader disagree with the final lines."""
    record, entry = legacy_pair()
    entry["vlm_lines"] = entry["vlm_lines"][:5]                      # the parse stored at read time kept 5 of 20 lines
    assert len(trl.parse_grounded(entry["vlm_raw"])[1]) == 20       # today's parser finds all 20
    doc = build((record, entry))
    assert len(reader(doc, f"vlm:{MODEL}:page")["lines"]) == 5


def test_legacy_entry_rebuilt_from_a_record_has_no_reply_index() -> None:
    """``--rematch`` entries made from a record have ``vlm_raw`` None: the stored lines stand, with a note."""
    record, entry = legacy_pair()
    entry["vlm_raw"] = None
    page = reader(build((record, entry)), f"vlm:{MODEL}:page")
    assert page["reply_index"] is None and "no raw reply cached" in page["note"] and len(page["lines"]) == 20


def test_pre_key_entry_names_its_kraken_reader_after_the_model() -> None:
    """A record without ``htr_cache_key`` (and an entry without ``htr_model``): Kraken is ``kraken:MiDRASH_Gen_01``."""
    record, entry = legacy_pair(key=None)
    assert "htr_cache_key" not in record["ai_read"] and "htr_model" not in entry
    doc = build((record, entry))
    assert ids(doc) == [f"vlm:{MODEL}:page", "kraken:MiDRASH_Gen_01"]
    assert reader(doc, "kraken:MiDRASH_Gen_01")["cache_key"] == "MiDRASH_Gen_01"
    assert len(reader(doc, "kraken:MiDRASH_Gen_01")["lines"]) == 20


def test_empty_kraken_read_is_an_htr_reader_with_no_lines() -> None:
    """Kraken finding nothing is a reading (zero lines), not a missing reader."""
    record, entry = legacy_pair()
    entry["frags"] = []
    record["ai_read"]["lines"] = []
    assert reader(build((record, entry)), f"kraken:{KEY}")["lines"] == []


def test_missing_fragments_for_the_key_leave_a_note_and_no_htr_reader() -> None:
    """The pure function never invents a Kraken reader; the walker refuses such a pair (``htr_key``)."""
    record, entry = legacy_pair()
    entry["htr_model"] = "MiDRASH_Gen_01@another-service"
    doc = build((record, entry))
    assert ids(doc) == [f"vlm:{MODEL}:page"] and "no Kraken fragments cached" in doc["provenance"]["notes"][0]
    assert br.entry_mismatch(record, entry) == "htr_key"


# ----------------------------------------------------------------------------- policy pages


def test_policy_keep_page_is_the_page_read_alone() -> None:
    """A healthy single-column read is kept: policy action keep, no units, the page reader labelled by the plan."""
    result = decode(SINGLE, {"page": reply_of(SINGLE)})
    record, entry = make_pair("Paris_AIU_1", 0, frags_of(SINGLE), result.vlm_raw, result.lines, result.parsed,
                              {**result.trace, "decided_at": T0})
    assert isinstance(entry["vlm_raw"], str)
    doc = build((record, entry))
    assert doc["policy"]["action"] == "keep" and doc["policy"]["units"] == [] and doc["policy"]["fallback"] is None
    assert (doc["policy"]["mode"], doc["policy"]["calls"], doc["policy"]["final_source"]) == ("live", 1, "whole_page")
    assert ids(doc) == [f"vlm:{MODEL}:page", f"kraken:{KEY}"]
    page = doc["readers"][0]
    assert (page["reply_index"], page["health_label"], page["parsed"]) == (0, "ok", True)
    assert doc["final"]["source"] == "whole_page" and doc["provenance"]["reply_mapping"] == "by_request"


def test_policy_columns_page_maps_each_crop_to_the_page_frame() -> None:
    """Two healthy column crops: one unit reader each (right column first), boxes in the page frame, no page reader."""
    record, entry = columns_pair()
    assert isinstance(entry["vlm_raw"], list) and len(entry["vlm_raw"]) == 2
    doc = build((record, entry))
    assert ids(doc) == [f"vlm:{MODEL}:col0:crop", f"vlm:{MODEL}:col1:crop", f"kraken:{KEY}"]
    col0, col1 = doc["readers"][:2]
    assert (col0["scope"], col0["unit"], col0["role"], col0["reply_index"], col0["health_label"]) == (
        "unit", "col0", "column", 0, "ok")
    assert (col1["unit"], col1["reply_index"]) == ("col1", 1)
    assert [ln["text"] for ln in col0["lines"]] == [t for t, _ in RIGHT] and [ln["text"] for ln in col1["lines"]] == [t for t, _ in LEFT]
    for ln, (_, px) in zip(col0["lines"] + col1["lines"], RIGHT + LEFT):          # page frame: pixels / page size * 1000
        assert ln["box"] == pytest.approx([1000 * px[0] / W, 1000 * px[1] / H, 1000 * px[2] / W, 1000 * px[3] / H], abs=1.01)
    crop0 = entry["decode_policy"]["requests"][0]["crop"]
    assert col0["crop_box_px"] == crop0 and col0["crop_box_1000"] == [round(1000 * crop0[0] / W), round(1000 * crop0[1] / H),
                                                                      round(1000 * crop0[2] / W), round(1000 * crop0[3] / H)]
    policy = doc["policy"]
    assert (policy["action"], policy["reason"], policy["final_source"], policy["calls"], policy["fallback"]) == (
        "columns", "2_columns", "assembled", 2, None)
    assert [(u["name"], u["role"], u["resolved_by"], u["health_label"]) for u in policy["units"]] == [
        ("col0", "column", "crop", "ok"), ("col1", "column", "crop", "ok")]
    assert [a["reply_index"] for u in policy["units"] for a in u["attempts"]] == [0, 1]
    assert policy["units"][0]["attempts"][0]["source"] == "crop" and policy["units"][0]["attempts"][0]["label"] == "ok"
    assert doc["final"]["source"] == "assembled" and doc["final"]["lines"] == record["ai_read"]["lines"]
    assert doc["provenance"]["reply_mapping"] == "by_request" and doc["provenance"]["notes"] == []


def test_replies_are_attached_by_request_unit_not_by_position() -> None:
    """A column page that needed the whole page gets it LAST: the page reader is still the page read, listed first."""
    record, entry = stuck_columns_pair()
    requests = entry["decode_policy"]["requests"]
    assert [r["unit"] for r in requests] == ["col0", "col1", "col1.top", "col1.bottom", "page"]
    assert [r["cached"] for r in requests] == [False] * 5
    doc = build((record, entry))
    assert ids(doc) == [f"vlm:{MODEL}:page", f"vlm:{MODEL}:col0:crop", f"vlm:{MODEL}:col1:crop",
                        f"vlm:{MODEL}:col1.top:crop", f"vlm:{MODEL}:col1.bottom:crop", f"kraken:{KEY}"]
    page = doc["readers"][0]
    assert page["reply_index"] == 4 and [ln["text"] for ln in page["lines"]] == [t for t, _ in TWO_COLUMNS]
    assert page["health_label"] == "ok" and page["scope"] == "page" and page["crop_box_px"] is None
    assert [r["reply_index"] for r in doc["readers"][:-1]] == [4, 0, 1, 2, 3]
    assert [r["role"] for r in doc["readers"][:-1]] == ["page", "column", "column", "half", "half"]
    assert [r["health_label"] for r in doc["readers"][:-1]] == ["ok", "ok", "loop", "loop", "loop"]
    top = reader(doc, f"vlm:{MODEL}:col1.top:crop")
    assert top["crop_box_px"] == requests[2]["crop"] and top["crop_box_1000"][3] > top["crop_box_1000"][1]
    col1_unit = doc["policy"]["units"][1]
    assert (col1_unit["name"], col1_unit["resolved_by"], col1_unit["resolved"]) == ("col1", "share", True)
    crop, halves, share = col1_unit["attempts"]
    assert (crop["source"], crop["label"], crop["reply_index"]) == ("crop", "loop", 1)
    assert (halves["source"], halves["label"], halves["reply_index"], halves["reply_indices"]) == ("halves", "loop", None, [2, 3])
    assert (share["source"], share["reply_index"]) == ("share", 4) and "whole-page read" in share["note"]
    assert col1_unit["health_label"] == share["label"]                           # the reading that resolved the unit
    assert doc["policy"]["calls"] == 5 and doc["policy"]["final_source"] == "assembled"
    assert doc["provenance"]["reply_mapping"] == "by_request"


def test_rescue_page_reads_the_page_then_both_halves() -> None:
    """A looping whole-page read rescued as halves: page, top, bottom, Kraken; the plan's verdicts label each read."""
    record, entry = rescue_pair()
    assert [r["unit"] for r in entry["decode_policy"]["requests"]] == ["page", "top", "bottom"]
    doc = build((record, entry))
    assert ids(doc) == [f"vlm:{MODEL}:page", f"vlm:{MODEL}:top:crop", f"vlm:{MODEL}:bottom:crop", f"kraken:{KEY}"]
    page, top, bottom = doc["readers"][:3]
    assert (page["reply_index"], page["health_label"], page["parsed"]) == (0, "loop", True)
    assert (top["reply_index"], top["health_label"], top["role"], bottom["reply_index"]) == (1, "ok", "half", 2)
    assert len(page["lines"]) == 33 and sum(1 for ln in top["lines"] + bottom["lines"]) > 20     # the loop is shown as read
    policy = doc["policy"]
    assert (policy["action"], policy["reason"], policy["final_source"], policy["fallback"]) == (
        "rescue", "whole_page_loop", "assembled", None)
    assert [(u["name"], u["role"], u["resolved_by"], u["health_label"], u["resolved"]) for u in policy["units"]] == [
        ("top", "half", "crop", "ok", True), ("bottom", "half", "crop", "ok", True)]
    assert [a["reply_index"] for u in policy["units"] for a in u["attempts"]] == [1, 2]
    assert top["crop_box_px"] == policy["units"][0]["crop_box_px"] and top["crop_box_1000"] == policy["units"][0]["crop_box_1000"]
    assert top["crop_box_px"][1] < bottom["crop_box_px"][1] < top["crop_box_px"][3]            # the halves overlap
    assert [ln["text"] for ln in doc["final"]["lines"]] == [t for t, _ in SINGLE]
    assert doc["final"]["source"] == "assembled"


def test_rescue_whose_halves_fail_falls_back_to_the_whole_page_read() -> None:
    """Both halves loop: ``fallback`` names why, the final source is the whole page, units are ``unresolved``."""
    record, entry = rescue_pair(failing_halves=True)
    doc = build((record, entry))
    policy = doc["policy"]
    assert (policy["final_source"], policy["fallback"]) == ("whole_page", "all_units_unresolved")
    assert [(u["resolved_by"], u["resolved"], u["health_label"]) for u in policy["units"]] == [
        ("unresolved", False, "loop"), ("unresolved", False, "loop")]
    assert doc["final"]["source"] == "whole_page" and ids(doc)[1:3] == [f"vlm:{MODEL}:top:crop", f"vlm:{MODEL}:bottom:crop"]


def test_revisit_entry_keeps_the_cached_whole_page_and_its_old_lines() -> None:
    """A revisit's first reply is the cached whole-page read; the lines stored with it stand, not a fresh parse."""
    whole = reply_of(TWO_COLUMNS)
    old = parsed_lines(whole)[:-2]                                  # what an older parser stored for that reply
    cached = pdp.Reply("page", "page", None, whole, True, parsed_lines(whole), cached=True)
    crops = units_of(frags_of(TWO_COLUMNS), whole, gap=500)
    replies = {name: reply_of(lines_in(TWO_COLUMNS, crop), crop) for name, crop in crops.items()}
    record, entry = policy_pair(TWO_COLUMNS, replies, "Oxford_Bodl_1", whole=cached, old_whole=old)
    assert entry["decode_policy"]["mode"] == "revisit" and entry["decode_policy"]["requests"][0]["cached"] is True
    assert entry["vlm_raw"][0] == whole
    doc = build((record, entry))
    assert ids(doc) == [f"vlm:{MODEL}:page", f"vlm:{MODEL}:col0:crop", f"vlm:{MODEL}:col1:crop", f"kraken:{KEY}"]
    page = doc["readers"][0]
    assert page["reply_index"] == 0 and len(page["lines"]) == len(old) == len(parsed_lines(whole)) - 2
    assert doc["policy"]["mode"] == "revisit" and doc["policy"]["calls"] == 2
    assert [r["reply_index"] for r in doc["readers"][:3]] == [0, 1, 2]
    assert doc["provenance"]["reply_mapping"] == "by_request" and doc["provenance"]["notes"] == []   # no drift note for the cached read


def test_a_parse_that_differs_from_the_trace_is_noted_not_a_mismatch() -> None:
    """The parser changed since the page was read: the reply is still attributed (by its length), with a note."""
    record, entry = rescue_pair()
    entry["decode_policy"]["requests"][1]["n_lines"] += 3             # the trace recorded 3 more lines than the reply parses to
    doc = build((record, entry))
    top = reader(doc, f"vlm:{MODEL}:top:crop")
    assert doc["provenance"]["reply_mapping"] == "by_request" and top["reply_index"] == 1 and top["lines"]
    assert len(doc["provenance"]["notes"]) == 1 and "reply of top" in doc["provenance"]["notes"][0]
    assert "parser changed" in doc["provenance"]["notes"][0]
    entry["decode_policy"]["requests"][1]["reply_chars"] += 1         # but a reply of another length is another reply
    assert build((record, entry))["provenance"]["reply_mapping"] == "partial"


def test_a_list_shorter_than_the_trace_attributes_only_the_verified_leading_page_read() -> None:
    """``vlm_raw`` with a reply missing: the page read (first, verified) is kept; the rest is null, never guessed."""
    record, entry = rescue_pair()
    entry["vlm_raw"] = entry["vlm_raw"][:2]
    doc = build((record, entry))
    page, top, bottom = doc["readers"][:3]
    assert page["reply_index"] == 0 and len(page["lines"]) == 33
    for r in (top, bottom):
        assert r["reply_index"] is None and r["lines"] is None and r["parsed"] is None and "could not be attributed" in r["note"]
        assert r["crop_box_px"] is not None and r["health_label"] == "ok"                 # the trace still says what was read
    assert [a["reply_index"] for u in doc["policy"]["units"] for a in u["attempts"]] == [None, None]
    assert doc["provenance"]["reply_mapping"] == "partial" and "ambiguous" in doc["provenance"]["notes"][0]
    assert doc["final"]["lines"] == record["ai_read"]["lines"]


def test_a_reply_that_does_not_match_its_request_makes_the_mapping_partial() -> None:
    """Same length, but the reply at an index is not the one the trace describes (its length differs)."""
    record, entry = rescue_pair()
    entry["vlm_raw"][1], entry["vlm_raw"][2] = entry["vlm_raw"][2][:-3] + "]  ", entry["vlm_raw"][1]
    doc = build((record, entry))
    assert doc["provenance"]["reply_mapping"] == "partial"
    assert [r["reply_index"] for r in doc["readers"][:3]] == [0, None, None]
    assert doc["readers"][0]["lines"] is not None and doc["readers"][1]["lines"] is None


def test_a_column_page_whose_first_reply_is_not_the_page_attributes_nothing_when_ambiguous() -> None:
    """Without a whole-page request first there is no unambiguous reply to keep."""
    record, entry = columns_pair()
    entry["vlm_raw"] = entry["vlm_raw"][:1]
    doc = build((record, entry))
    assert [r["reply_index"] for r in doc["readers"][:2]] == [None, None]
    assert doc["provenance"]["reply_mapping"] == "partial"


def test_a_trace_without_requests_is_flagged_not_fatal() -> None:
    """An unknown trace layout still gives a sidecar: the policy block, Kraken, and a note instead of readers."""
    record, entry = rescue_pair()
    del entry["decode_policy"]["requests"]
    doc = build((record, entry))
    assert ids(doc) == [f"kraken:{KEY}"] and doc["provenance"]["reply_mapping"] == "partial"
    assert doc["policy"]["action"] == "rescue" and [a["reply_index"] for u in doc["policy"]["units"] for a in u["attempts"]] == [None, None]


def test_the_sidecar_does_not_alias_or_alter_its_inputs() -> None:
    """The builder only reads: records and entries are unchanged after building."""
    pair = rescue_pair()
    before = copy.deepcopy(pair)
    build(pair)
    assert pair == before


# ----------------------------------------------------------------------------- record / entry agreement


@pytest.mark.parametrize("change,reason", [
    (lambda r, e: e.update(image_url="https://example.org/other.jpg"), "image_url"),
    (lambda r, e: e.update(vlm_model="another-model"), "vlm_model"),
    (lambda r, e: e.update(sha256="0" * 64), "sha256"),
    (lambda r, e: e.update(width=W + 1), "size"),
    (lambda r, e: r["ai_read"].pop("decode_policy"), "policy_stamp"),
    (lambda r, e: e.pop("decode_policy"), "policy_stamp"),
    (lambda r, e: r["ai_read"]["decode_policy"].update(calls=99), "policy_stamp"),
    (lambda r, e: e.update(htr_model="MiDRASH_Gen_01@elsewhere"), "htr_key"),
    (lambda r, e: None, None),
])
def test_entry_mismatch_names_why_an_entry_is_not_the_records_page(change: Any, reason: Optional[str]) -> None:
    """The raw entry must be this image's, this model's, in this record's policy state, with its Kraken key.

    :param change: Edits a rescue page's ``(record, entry)`` pair in place.
    :param reason: The mismatch :func:`entry_mismatch` must name (None: the pair still agrees).
    """
    record, entry = rescue_pair()
    change(record, entry)
    assert br.entry_mismatch(record, entry) == reason


def test_entry_without_a_policy_trace_agrees_with_a_record_without_a_stamp() -> None:
    """A legacy pair agrees; the legacy pair's key may be absent on both sides."""
    assert br.entry_mismatch(*legacy_pair()) is None
    assert br.entry_mismatch(*legacy_pair(key=None)) is None


# ----------------------------------------------------------------------------- walking the records


def pages() -> List[Pair]:
    """One page of each kind (legacy, columns, rescue).

    :returns: Pairs in file order.
    """
    return [legacy_pair("Cambridge_T-S_1", 0, T0), columns_pair("Manchester_JRL_B_1", 0, T1), rescue_pair("Paris_AIU_IX_B_1", 0, T2)]


def test_backfill_writes_one_sidecar_per_page_and_a_manifest(tmp_path: Path) -> None:
    """Sidecars are named after the raw-cache stems, hold the schema, and the manifest lists each once."""
    records, raw_dir, out_dir, manifest = workspace(tmp_path, pages())
    counts = br.build_readings([records], raw_dir, out_dir, manifest)
    names = [p.name for p in sidecar_files(out_dir)]
    assert names == sorted(f"{d}__0__{MODEL}.json" for d in ("Cambridge_T-S_1", "Manchester_JRL_B_1", "Paris_AIU_IX_B_1"))
    assert not list(out_dir.glob("*.tmp"))
    assert (counts["records"], counts["built"], counts["built_new"], counts["rebuilt"]) == (3, 3, 3, 0)
    assert (counts["built_legacy"], counts["built_policy"], counts["policy_columns"], counts["policy_rescue"]) == (1, 2, 1, 1)
    assert (counts["readers_vlm"], counts["readers_htr"], counts["ambiguous"]) == (1 + 2 + 3, 3, 0)
    assert counts["bytes_written"] == sum(p.stat().st_size for p in sidecar_files(out_dir))
    doc = json.loads((out_dir / f"Manchester_JRL_B_1__0__{MODEL}.json").read_text(encoding="utf-8"))
    assert doc["schema_version"] == "readings-v1" and doc["provenance"]["raw_cache"] == f"raw/Manchester_JRL_B_1__0__{MODEL}.json"
    assert doc["provenance"]["builder"] == br.BUILDER and doc["policy"]["action"] == "columns"
    rows = manifest_rows(manifest)
    assert [(r["stem"], r["doc_id"], r["image_index"], r["readers"], r["policy_action"]) for r in rows] == [
        (f"Cambridge_T-S_1__0__{MODEL}", "Cambridge_T-S_1", 0, 2, None),
        (f"Manchester_JRL_B_1__0__{MODEL}", "Manchester_JRL_B_1", 0, 3, "columns"),
        (f"Paris_AIU_IX_B_1__0__{MODEL}", "Paris_AIU_IX_B_1", 0, 4, "rescue")]
    for row in rows:                                                       # the manifest carries the sidecar's own build time
        built = json.loads((out_dir / f"{row['stem']}.json").read_text(encoding="utf-8"))
        assert row["built_at"] == built["provenance"]["built_at"]


def test_second_run_with_nothing_new_writes_nothing(tmp_path: Path) -> None:
    """Idempotent: no file rewritten (mtimes and bytes unchanged), no manifest line, everything counted up to date."""
    records, raw_dir, out_dir, manifest = workspace(tmp_path, pages())
    br.build_readings([records], raw_dir, out_dir, manifest)
    snapshot = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in sidecar_files(out_dir)}
    manifest_before = manifest.read_bytes()
    counts = br.build_readings([records], raw_dir, out_dir, manifest)
    assert (counts["built"], counts["up_to_date"], counts["records"]) == (0, 3, 3)
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in sidecar_files(out_dir)} == snapshot
    assert manifest.read_bytes() == manifest_before


def test_page_without_a_raw_entry_is_counted_and_skipped(tmp_path: Path) -> None:
    """A record whose raw-cache entry is gone gets no sidecar and no manifest line; the others are built."""
    records, raw_dir, out_dir, manifest = workspace(tmp_path, pages())
    (raw_dir / f"Manchester_JRL_B_1__0__{MODEL}.json").unlink()
    counts = br.build_readings([records], raw_dir, out_dir, manifest)
    assert (counts["built"], counts["raw_missing"]) == (2, 1)
    assert not (out_dir / f"Manchester_JRL_B_1__0__{MODEL}.json").exists()
    assert [r["doc_id"] for r in manifest_rows(manifest)] == ["Cambridge_T-S_1", "Paris_AIU_IX_B_1"]


def test_load_json_returns_only_objects(tmp_path: Path) -> None:
    """Absent, truncated and non-object JSON files all read as None (the walk retries them next run)."""
    path = tmp_path / "x.json"
    assert br.load_json(path) is None
    path.write_text('{"a": 1', encoding="utf-8")
    assert br.load_json(path) is None
    path.write_text("[1, 2]", encoding="utf-8")
    assert br.load_json(path) is None
    path.write_text('{"a": "אבג"}', encoding="utf-8")
    assert br.load_json(path) == {"a": "אבג"}


def test_unreadable_and_foreign_raw_entries_are_counted_with_their_reason(tmp_path: Path) -> None:
    """A truncated entry, another image's entry and an entry ahead of its record are skipped, not built from."""
    records, raw_dir, out_dir, manifest = workspace(tmp_path, pages())
    (raw_dir / f"Cambridge_T-S_1__0__{MODEL}.json").write_text('{"doc_id": "Cambridge_T-S_1", "vlm_raw": "[{', encoding="utf-8")
    foreign = json.loads((raw_dir / f"Manchester_JRL_B_1__0__{MODEL}.json").read_text(encoding="utf-8"))
    foreign["image_url"] = "https://example.org/another.jpg"
    (raw_dir / f"Manchester_JRL_B_1__0__{MODEL}.json").write_text(json.dumps(foreign, ensure_ascii=False), encoding="utf-8")
    ahead = json.loads((raw_dir / f"Paris_AIU_IX_B_1__0__{MODEL}.json").read_text(encoding="utf-8"))
    del ahead["decode_policy"]                                      # the entry lost its trace while the record keeps its stamp
    (raw_dir / f"Paris_AIU_IX_B_1__0__{MODEL}.json").write_text(json.dumps(ahead, ensure_ascii=False), encoding="utf-8")
    counts = br.build_readings([records], raw_dir, out_dir, manifest)
    assert counts["built"] == 0 and counts["raw_unreadable"] == 1 and counts["raw_mismatch"] == 2
    assert (counts["raw_mismatch:image_url"], counts["raw_mismatch:policy_stamp"]) == (1, 1)
    assert sidecar_files(out_dir) == [] and manifest_rows(manifest) == []


def test_force_rewrites_current_sidecars_and_lists_them_again(tmp_path: Path) -> None:
    """``--force`` rebuilds pages that are current; the manifest gets a second line each."""
    records, raw_dir, out_dir, manifest = workspace(tmp_path, pages())
    br.build_readings([records], raw_dir, out_dir, manifest)
    counts = br.build_readings([records], raw_dir, out_dir, manifest, force=True)
    assert (counts["built"], counts["rebuilt"], counts["built_new"], counts["up_to_date"]) == (3, 3, 0, 0)
    assert len(manifest_rows(manifest)) == 6


def test_since_selects_records_decoded_at_or_after_it(tmp_path: Path) -> None:
    """Records before the cut are not even looked at; a ``Z`` time and a bare date both parse."""
    records, raw_dir, out_dir, manifest = workspace(tmp_path, pages())                 # decoded at 10:00, 12:00, 14:00
    counts = br.build_readings([records], raw_dir, out_dir, manifest, since=br.parse_time("2026-10-13T12:00:00Z"))
    assert (counts["records"], counts["before_since"], counts["built"]) == (3, 1, 2)
    assert [r["doc_id"] for r in manifest_rows(manifest)] == ["Manchester_JRL_B_1", "Paris_AIU_IX_B_1"]
    counts = br.build_readings([records], raw_dir, out_dir, manifest, since=br.parse_time("2026-10-14"))
    assert (counts["before_since"], counts["built"]) == (3, 0)
    assert br.parse_time("2026-10-13T12:00:00Z") == br.parse_time("2026-10-13T12:00:00+00:00")
    assert br.parse_time("2026-10-13").tzinfo is timezone.utc


def test_incremental_run_picks_up_only_the_pages_appended_since(tmp_path: Path) -> None:
    """The half-hourly loop: run, append records, run with ``--since`` before them: only the new pages are built."""
    first = pages()
    records, raw_dir, out_dir, manifest = workspace(tmp_path, first)
    br.build_readings([records], raw_dir, out_dir, manifest)
    more = [legacy_pair("Cambridge_T-S_2", 0, "2026-10-13T16:00:00+00:00"), rescue_pair("Paris_AIU_IX_B_2", 0, "2026-10-13T16:30:00+00:00")]
    for pair in more:
        write_raw(raw_dir, pair)
    write_records(records, [p[0] for p in first + more])
    counts = br.build_readings([records], raw_dir, out_dir, manifest, since=br.parse_time("2026-10-13T15:00:00Z"))
    assert (counts["records"], counts["before_since"], counts["built"], counts["up_to_date"]) == (5, 3, 2, 0)
    assert len(sidecar_files(out_dir)) == 5 and len(manifest_rows(manifest)) == 5


def test_a_rewritten_record_rebuilds_its_stale_sidecar(tmp_path: Path) -> None:
    """A page re-read under the policy (record and entry rewritten, newer ``decoded_at``) replaces its legacy sidecar."""
    records, raw_dir, out_dir, manifest = workspace(tmp_path, [rescue_pair("Paris_AIU_IX_B_1", 0, T0)])
    old = legacy_pair("Paris_AIU_IX_B_1", 0, T0)
    write_raw(raw_dir, old)
    write_records(records, [old[0]])
    br.build_readings([records], raw_dir, out_dir, manifest)
    path = out_dir / f"Paris_AIU_IX_B_1__0__{MODEL}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["policy"] is None
    new = rescue_pair("Paris_AIU_IX_B_1", 0, T2)
    write_raw(raw_dir, new)
    write_records(records, [new[0]])
    counts = br.build_readings([records], raw_dir, out_dir, manifest)
    assert (counts["built"], counts["rebuilt"], counts["built_new"]) == (1, 1, 0)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["policy"]["action"] == "rescue" and doc["decoded_at"] == T2
    assert br.build_readings([records], raw_dir, out_dir, manifest)["built"] == 0


def test_entry_ahead_of_its_record_is_not_built_until_the_record_catches_up(tmp_path: Path) -> None:
    """A revisit that rewrote the entry but not yet the record: skipped (policy_stamp), built once the record follows."""
    legacy = legacy_pair("Paris_AIU_IX_B_1", 0, T0)
    records, raw_dir, out_dir, manifest = workspace(tmp_path, [legacy])
    write_raw(raw_dir, rescue_pair("Paris_AIU_IX_B_1", 0, T2))                     # entry rewritten first
    counts = br.build_readings([records], raw_dir, out_dir, manifest)
    assert (counts["built"], counts["raw_mismatch:policy_stamp"]) == (0, 1)
    write_records(records, [rescue_pair("Paris_AIU_IX_B_1", 0, T2)[0]])           # record flushed
    assert br.build_readings([records], raw_dir, out_dir, manifest)["built"] == 1


def test_a_page_recorded_twice_is_built_from_the_later_record_and_stays_put(tmp_path: Path) -> None:
    """Duplicate records (two runs seconds apart): the later one wins, and later runs do not flip-flop."""
    early, late = legacy_pair("Oxford_Bodl_1", 0, T0), legacy_pair("Oxford_Bodl_1", 0, T1)
    late[0]["ai_read"]["n_agreed"] = 7
    records, raw_dir, out_dir, manifest = workspace(tmp_path, [early])
    write_records(records, [early[0], late[0]])
    counts = br.build_readings([records], raw_dir, out_dir, manifest)
    assert (counts["records"], counts["built_new"], counts["rebuilt"]) == (2, 1, 1)
    assert json.loads(sidecar_files(out_dir)[0].read_text(encoding="utf-8"))["decoded_at"] == T1
    again = br.build_readings([records], raw_dir, out_dir, manifest)
    assert (again["built"], again["up_to_date"]) == (0, 2)


def test_a_sidecar_of_another_page_is_never_current_for_this_one(tmp_path: Path) -> None:
    """Two doc ids that differ only by case share a file name on a case-insensitive disk: the file is one page's."""
    mine, other = legacy_pair("Cambridge_T-S_J24b", 0, T1), legacy_pair("Cambridge_T-S_J24B", 0, T1)
    path = tmp_path / "Cambridge_T-S_J24b__0.json"
    assert br.sidecar_status(path, mine[0]) == "missing"
    br.write_json_atomic(path, build(mine))
    assert br.sidecar_status(path, mine[0]) == "current"
    assert br.sidecar_status(path, other[0]) == "stale"                                  # different doc_id: not its sidecar
    foreign_model = legacy_pair("Cambridge_T-S_J24b", 0, T1)[0]
    foreign_model["ai_read"]["vlm_model"] = "qwen3-vl-8b-heb-v21b-step1200"
    assert br.sidecar_status(path, foreign_model) == "stale"
    older = legacy_pair("Cambridge_T-S_J24b", 0, T0)[0]
    assert br.sidecar_status(path, older) == "current"                                    # built from a newer record
    assert br.sidecar_status(path, legacy_pair("Cambridge_T-S_J24b", 0, T2)[0]) == "stale"


def test_a_new_builder_version_makes_every_sidecar_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bumping the builder is the way to roll a change out: sidecars from the old version are rebuilt once."""
    records, raw_dir, out_dir, manifest = workspace(tmp_path, pages())
    br.build_readings([records], raw_dir, out_dir, manifest)
    monkeypatch.setattr(br, "BUILDER", "build_readings 9.9.9")
    counts = br.build_readings([records], raw_dir, out_dir, manifest)
    assert (counts["rebuilt"], counts["up_to_date"]) == (3, 0)
    assert json.loads(sidecar_files(out_dir)[0].read_text(encoding="utf-8"))["provenance"]["builder"] == "build_readings 9.9.9"


def test_the_manifest_directory_is_created_when_missing(tmp_path: Path) -> None:
    """``--manifest`` may name a path in a directory that does not exist yet."""
    records, raw_dir, out_dir, _ = workspace(tmp_path, pages())
    manifest = tmp_path / "sync" / "state" / "manifest.jsonl"
    assert br.build_readings([records], raw_dir, out_dir, manifest)["built"] == 3
    assert len(manifest_rows(manifest)) == 3


def test_limit_stops_after_n_sidecars(tmp_path: Path) -> None:
    """``--limit 2`` writes two; the next run (no limit) writes the rest."""
    records, raw_dir, out_dir, manifest = workspace(tmp_path, pages())
    assert br.build_readings([records], raw_dir, out_dir, manifest, limit=2)["built"] == 2
    assert len(sidecar_files(out_dir)) == 2
    assert br.build_readings([records], raw_dir, out_dir, manifest)["built"] == 1


def test_a_record_still_being_written_is_not_a_record_yet(tmp_path: Path) -> None:
    """The pipeline appends one JSON line at a time: an unterminated tail is skipped and counted."""
    first = pages()
    records, raw_dir, out_dir, manifest = workspace(tmp_path, first)
    write_records(records, [p[0] for p in first], tail=json.dumps(legacy_pair("Cambridge_T-S_9")[0])[:200])
    counts = br.build_readings([records], raw_dir, out_dir, manifest)
    assert (counts["records"], counts["built"], counts["partial_tail"]) == (3, 3, 1)


def test_two_records_files_are_walked_in_turn(tmp_path: Path) -> None:
    """The documentary and literary files can go in one call."""
    a, b = legacy_pair("Cambridge_T-S_1"), legacy_pair("Cambridge_T-S_2")
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "readings"
    for pair in (a, b):
        write_raw(raw_dir, pair)
    files = [write_records(tmp_path / "a.jsonl", [a[0]]), write_records(tmp_path / "b.jsonl", [b[0]])]
    counts = br.build_readings(files, raw_dir, out_dir)
    assert (counts["records"], counts["built"]) == (2, 2) and len(sidecar_files(out_dir)) == 2


def test_older_raw_entry_without_the_model_suffix_is_found_as_the_pipeline_finds_it(tmp_path: Path) -> None:
    """``raw_cache_path`` falls back to the model-less name of the first pilot runs; the sidecar takes that stem."""
    pair = legacy_pair("Cambridge_T-S_5")
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "readings"
    old = write_raw(raw_dir, pair, suffixless=True)
    assert old.name == "Cambridge_T-S_5__0.json"
    records = write_records(tmp_path / "records.jsonl", [pair[0]])
    counts = br.build_readings([records], raw_dir, out_dir)
    assert counts["built"] == 1 and [p.name for p in sidecar_files(out_dir)] == ["Cambridge_T-S_5__0.json"]
    assert json.loads(sidecar_files(out_dir)[0].read_text(encoding="utf-8"))["provenance"]["raw_cache"] == "raw/Cambridge_T-S_5__0.json"


def test_cli_prints_a_summary_and_keeps_the_default_manifest_in_the_sidecar_directory(tmp_path: Path,
                                                                                      capsys: pytest.CaptureFixture) -> None:
    """``main`` takes the documented flags; the default manifest is ``<out-dir>/manifest.jsonl`` (not a ``*.json``)."""
    records, raw_dir, out_dir, _ = workspace(tmp_path, pages())
    argv = ["--records", str(records), "--raw-dir", str(raw_dir), "--out-dir", str(out_dir)]
    assert br.main(argv) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("summary: "))
    assert (summary["built"], summary["records"]) == (3, 3) and "elapsed_s" in summary
    assert len(manifest_rows(out_dir / "manifest.jsonl")) == 3 and len(sidecar_files(out_dir)) == 3
    assert sorted(p.name for p in out_dir.iterdir() if p not in sidecar_files(out_dir)) == ["manifest.jsonl"]
    assert br.main(argv + ["--since", "2026-10-13T13:00:00Z"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("summary: "))["before_since"] == 2
    assert br.main(argv + ["--force", "--limit", "1", "--manifest", str(tmp_path / "other.jsonl")]) == 0
    assert len(manifest_rows(tmp_path / "other.jsonl")) == 1
    with pytest.raises(SystemExit):                                       # a records file that does not exist
        br.main(["--records", str(tmp_path / "missing.jsonl")])
    with pytest.raises(SystemExit):                                       # a --since that is not a time
        br.main(argv + ["--since", "yesterday"])


def test_counts_by_reader_kind_include_partial_pages(tmp_path: Path) -> None:
    """The run's ``ambiguous`` count is the pages with ``reply_mapping`` partial."""
    odd = rescue_pair("Paris_AIU_IX_B_7")
    odd[1]["vlm_raw"] = odd[1]["vlm_raw"][:2]
    records, raw_dir, out_dir, manifest = workspace(tmp_path, [odd, columns_pair()])
    counts = br.build_readings([records], raw_dir, out_dir, manifest)
    assert Counter({k: counts[k] for k in ("built", "ambiguous", "readers_vlm", "readers_htr")}) == Counter(
        built=2, ambiguous=1, readers_vlm=3 + 2, readers_htr=2)
