# File name: v22_task_eval.py
# Date: 9/27/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Task-level evaluation of a v22 checkpoint on the held-out QA and grounding rows.

The training loop only ever sees the mixture's validation split as one aggregate
eval loss.  This harness decodes those held-out rows through LM Studio and scores
each task the way ``docs/v22_dataset.md`` §5 says it should be scored:

* **QA** (``pgp_qa`` rows, answers ``{"line": N, "text": "..."}`` or
  ``{"answer": "not stated"}``): abstention accuracy on both branches, line-index
  hit rate, CER of the quoted span against the target (the headline number for
  full-line families) and exact match (whitespace-, nikud- and final-letter-
  normalised) for the short-span families.
* **Box tasks** (``locate``, ``locate_word``, ``line_index``, ``line_of_phrase``):
  IoU and the centre-hit rule of ``grounding_eval.py`` (prediction centre inside
  the target box), plus span CER where the target also carries text.
* **Text tasks** (``read_box``, ``read_box_word``): strict CER.

Multi-line ``grounded_*`` tasks are decoded but left unscored (``unsupported``).

The validation split shares no page image with training by construction
(``build_v22_mixture.py``), so every row here is unseen.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.eval_harness.v22_task_eval \\
        --data-dir /Volumes/home/studio_offload/datasets/genizah_v22_pilot \\
        --model qwen3-vl-8b-heb-v22a-step1300 \\
        --sources pgp_qa,documentary_grounding,ktiv_grounding \\
        --out logs/v22_task_eval/v22a-1300_val.jsonl

Re-running with the same ``--out`` resumes (rows already decoded are skipped);
``--rescore`` only re-scores the stored raw outputs without touching LM Studio.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.datasets.evaluations.metrics import cer_pair, normalize_whitespace, strip_nikud
from src.models.ocr.lms_transcriber import transcribe_with_lm_studio

BOX_TASKS = {"locate", "locate_word", "line_index", "line_of_phrase"}
TEXT_TASKS = {"read_box", "read_box_word"}
QA_SOURCE = "pgp_qa"
# Families whose target is a short span, where exact match is the intended metric (§5).
QA_EXACT_FAMILIES = {"qa_date_month", "qa_date_year", "qa_place", "qa_person", "qa_ketubah_name",
                     "qa_place_sent", "qa_place_written"}
ABSTAIN_TEXT = "not stated"
MAX_TOKENS = {"qa": 256, "box": 96, "text": 512, "other": 1024}

_FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
_QUOTES = str.maketrans({"׳": "'", "’": "'", "‘": "'", "״": '"', "“": '"', "”": '"'})
_JSON_BOX = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_FENCE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------
def normalise_hebrew(text: str) -> str:
    """Fold a Hebrew string for comparison: nikud stripped, typographic quotes
    mapped to ASCII, final letters folded to their medial form, whitespace collapsed.

    :param text: Raw text.
    :type text: str
    :return: Normalised text.
    :rtype: str
    """
    return normalize_whitespace(strip_nikud(text or "").translate(_QUOTES).translate(_FINALS))


def extract_json(raw: str) -> Optional[Any]:
    """Return the first JSON value (object or array) embedded in a model reply.

    Tolerates code fences and leading/trailing prose: every ``{`` / ``[`` is tried
    as a start position and the first successful decode wins.

    :param raw: Raw model output.
    :type raw: str
    :return: The decoded value, or None when nothing parses.
    :rtype: Optional[Any]
    """
    if not raw:
        return None
    text = _FENCE.sub(" ", raw)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            return value
        except ValueError:
            continue
    return None


def parse_box(raw: str) -> Optional[List[int]]:
    """First ``[x1, y1, x2, y2]`` integer box in a string (same regex as grounding_eval).

    :param raw: Text that may contain a box.
    :type raw: str
    :return: The four integers, or None.
    :rtype: Optional[List[int]]
    """
    m = _JSON_BOX.search(raw or "")
    return [int(v) for v in m.groups()] if m else None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``[x0, y0, x1, y1]`` boxes.

    :param a: First box.
    :type a: Sequence[float]
    :param b: Second box.
    :type b: Sequence[float]
    :return: IoU in [0, 1].
    :rtype: float
    """
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def centre_hit(pred: Sequence[float], gold: Sequence[float]) -> bool:
    """grounding_eval's hit rule: the predicted box's centre lies inside the target box.

    :param pred: Predicted box.
    :type pred: Sequence[float]
    :param gold: Target box.
    :type gold: Sequence[float]
    :return: True on a hit.
    :rtype: bool
    """
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    return gold[0] <= cx <= gold[2] and gold[1] <= cy <= gold[3]


def span_cer(pred: str, gold: str) -> float:
    """Strict CER between two spans after :func:`normalise_hebrew`.

    :param pred: Predicted span.
    :type pred: str
    :param gold: Target span.
    :type gold: str
    :return: Character error rate (may exceed 1 when the prediction is much longer).
    :rtype: float
    """
    return float(cer_pair(normalise_hebrew(pred), normalise_hebrew(gold))[0])


# ---------------------------------------------------------------------------
# QA scoring
# ---------------------------------------------------------------------------
def _qa_items(value: Any) -> Tuple[bool, List[Dict[str, Any]]]:
    """Normalise a QA answer value to ``(is_abstain, [{"line", "text"}, ...])``.

    :param value: Decoded JSON (dict, list) or None.
    :type value: Any
    :return: Abstain flag and the list of quoted items (empty when abstaining/unparsed).
    :rtype: Tuple[bool, List[Dict[str, Any]]]
    """
    if isinstance(value, dict):
        if "answer" in value and "text" not in value:
            return ABSTAIN_TEXT in str(value.get("answer", "")).lower(), []
        return False, [value]
    if isinstance(value, list):
        return False, [v for v in value if isinstance(v, dict)]
    return False, []


def _line_of(item: Dict[str, Any]) -> Optional[int]:
    """Integer ``line`` of a QA item, or None.

    :param item: One ``{"line", "text"}`` object.
    :type item: Dict[str, Any]
    :return: The line number if it parses as an int.
    :rtype: Optional[int]
    """
    try:
        return int(item.get("line"))
    except (TypeError, ValueError):
        return None


def score_qa(raw_pred: str, gold_answer: str, task: str) -> Dict[str, Any]:
    """Score one QA row per docs/v22_dataset.md §5.

    :param raw_pred: Raw model output.
    :type raw_pred: str
    :param gold_answer: The row's target answer (JSON string).
    :type gold_answer: str
    :param task: Row task/family (``qa_date`` ...).
    :type task: str
    :return: Per-row metrics: ``parsed``, ``gold_abstain``, ``pred_abstain``,
        ``abstain_correct``, ``line_hit``, ``span_cer``, ``exact``, ``contains``.
    :rtype: Dict[str, Any]
    """
    gold_abstain, gold_items = _qa_items(json.loads(gold_answer))
    value = extract_json(raw_pred)
    pred_abstain, pred_items = _qa_items(value)
    if value is None and ABSTAIN_TEXT in (raw_pred or "").lower():
        pred_abstain = True
    out: Dict[str, Any] = {"parsed": value is not None or pred_abstain, "gold_abstain": gold_abstain,
                           "pred_abstain": pred_abstain, "abstain_correct": pred_abstain == gold_abstain,
                           "line_hit": None, "span_cer": None, "exact": None, "contains": None,
                           "exact_family": task in QA_EXACT_FAMILIES}
    if gold_abstain or not gold_items:
        return out
    if pred_abstain or not pred_items:
        out.update(line_hit=False, span_cer=1.0, exact=False, contains=False)
        return out
    gold_lines = {_line_of(g) for g in gold_items} - {None}
    pred_lines = {_line_of(p) for p in pred_items} - {None}
    out["line_hit"] = bool(gold_lines & pred_lines) if gold_lines else None
    # Greedy best-CER match of each gold item against the predicted items.
    cers, exacts, contains = [], [], []
    for g in gold_items:
        g_text = str(g.get("text", ""))
        best = min(pred_items, key=lambda p: span_cer(str(p.get("text", "")), g_text))
        p_text = str(best.get("text", ""))
        cers.append(span_cer(p_text, g_text))
        gn, pn = normalise_hebrew(g_text), normalise_hebrew(p_text)
        exacts.append(gn == pn)
        contains.append(bool(gn) and bool(pn) and (re.search(rf"(?:^|\s){re.escape(gn)}(?:\s|$)", pn) is not None
                                                   or re.search(rf"(?:^|\s){re.escape(pn)}(?:\s|$)", gn) is not None))
    out.update(span_cer=sum(cers) / len(cers), exact=all(exacts), contains=all(contains))
    return out


# ---------------------------------------------------------------------------
# grounding scoring
# ---------------------------------------------------------------------------
def score_box(raw_pred: str, gold_answer: str) -> Dict[str, Any]:
    """Score a box-answer row (``locate`` family): IoU, centre hit, optional text CER.

    :param raw_pred: Raw model output.
    :type raw_pred: str
    :param gold_answer: Target JSON with ``bbox_2d`` and optionally ``text``.
    :type gold_answer: str
    :return: ``parsed``, ``iou``, ``hit``, ``span_cer`` (None when the target has no text).
    :rtype: Dict[str, Any]
    """
    gold = json.loads(gold_answer)
    gold_box = gold["bbox_2d"]
    pred_box = parse_box(raw_pred)
    out: Dict[str, Any] = {"parsed": pred_box is not None, "iou": 0.0, "hit": False, "span_cer": None}
    if pred_box is not None:
        out.update(iou=iou(pred_box, gold_box), hit=centre_hit(pred_box, gold_box))
    if gold.get("text"):
        value = extract_json(raw_pred)
        pred_text = str(value.get("text", "")) if isinstance(value, dict) else ""
        out["span_cer"] = span_cer(pred_text, gold["text"]) if pred_text else 1.0
    return out


def score_text(raw_pred: str, gold_answer: str) -> Dict[str, Any]:
    """Score a text-answer row (``read_box`` family) with strict CER.

    :param raw_pred: Raw model output.
    :type raw_pred: str
    :param gold_answer: Target text.
    :type gold_answer: str
    :return: ``parsed`` (non-empty output) and ``span_cer``.
    :rtype: Dict[str, Any]
    """
    pred = (raw_pred or "").strip()
    return {"parsed": bool(pred), "span_cer": span_cer(pred, gold_answer) if pred else 1.0}


def score_row(row: Dict[str, Any], raw_pred: Optional[str]) -> Dict[str, Any]:
    """Dispatch one row to its scorer.

    :param row: Dataset row (needs ``source``, ``task``, ``answer``).
    :type row: Dict[str, Any]
    :param raw_pred: Raw model output (None when the decode failed).
    :type raw_pred: Optional[str]
    :return: ``{"kind": ..., **metrics}``; ``kind`` is ``qa`` / ``box`` / ``text`` / ``unsupported``.
    :rtype: Dict[str, Any]
    """
    raw = raw_pred or ""
    if row["source"] == QA_SOURCE:
        return {"kind": "qa", **score_qa(raw, row["answer"], row["task"])}
    if row["task"] in BOX_TASKS:
        return {"kind": "box", **score_box(raw, row["answer"])}
    if row["task"] in TEXT_TASKS:
        return {"kind": "text", **score_text(raw, row["answer"])}
    return {"kind": "unsupported", "parsed": bool(raw.strip())}


def max_tokens_for(row: Dict[str, Any]) -> int:
    """Generation budget by task kind.

    :param row: Dataset row.
    :type row: Dict[str, Any]
    :return: ``max_tokens`` for the LM Studio call.
    :rtype: int
    """
    if row["source"] == QA_SOURCE:
        return MAX_TOKENS["qa"]
    if row["task"] in BOX_TASKS:
        return MAX_TOKENS["box"]
    if row["task"] in TEXT_TASKS:
        return MAX_TOKENS["text"]
    return MAX_TOKENS["other"]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _rate(values: Iterable[Optional[bool]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-(source, task) summary plus per-source roll-ups.

    :param records: Scored records (each with ``source``, ``task`` and ``score``).
    :type records: Sequence[Dict[str, Any]]
    :return: Mapping ``"source/task"`` (and ``"source/ALL"``) to summary numbers.
    :rtype: Dict[str, Dict[str, Any]]
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[f"{r['source']}/{r['task']}"].append(r)
        groups[f"{r['source']}/ALL"].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for key, rs in sorted(groups.items()):
        s = [r["score"] for r in rs]
        kinds = {x["kind"] for x in s}
        row: Dict[str, Any] = {"n": len(rs), "parse_rate": _rate(x.get("parsed") for x in s)}
        if "qa" in kinds:
            qa = [x for x in s if x["kind"] == "qa"]
            gold_abs = [x for x in qa if x["gold_abstain"]]
            gold_ans = [x for x in qa if not x["gold_abstain"]]
            row.update({
                "abstain_acc_on_abstain_rows": _rate(x["abstain_correct"] for x in gold_abs),
                "false_abstain_rate": _rate(x["pred_abstain"] for x in gold_ans),
                "line_hit": _rate(x["line_hit"] for x in gold_ans),
                "exact": _rate(x["exact"] for x in gold_ans),
                "contains": _rate(x["contains"] for x in gold_ans),
                "span_cer_median": _median(x["span_cer"] for x in gold_ans),
                "span_cer_le_0.2": _rate((x["span_cer"] is not None and x["span_cer"] <= 0.2) for x in gold_ans),
            })
        if "box" in kinds:
            bx = [x for x in s if x["kind"] == "box"]
            row.update({"iou_median": _median(x["iou"] for x in bx), "iou_ge_0.5": _rate(x["iou"] >= 0.5 for x in bx),
                        "hit": _rate(x["hit"] for x in bx), "box_text_cer_median": _median(x["span_cer"] for x in bx)})
        if "text" in kinds:
            tx = [x for x in s if x["kind"] == "text"]
            row.update({"cer_median": _median(x["span_cer"] for x in tx),
                        "cer_le_0.2": _rate(x["span_cer"] <= 0.2 for x in tx)})
        if kinds == {"unsupported"}:
            row["note"] = "decoded, not scored"
        out[key] = row
    return out


def format_summary(summary: Dict[str, Dict[str, Any]], model: str) -> str:
    """Plain-text table of :func:`aggregate` output.

    :param summary: Aggregated numbers.
    :type summary: Dict[str, Dict[str, Any]]
    :param model: Model key (header).
    :type model: str
    :return: Multi-line string.
    :rtype: str
    """
    def fmt(v: Any) -> str:
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = [f"== {model} =="]
    for key, row in summary.items():
        parts = [f"{k}={fmt(v)}" for k, v in row.items() if k != "n"]
        lines.append(f"{key:42s} n={row['n']:3d}  " + "  ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# data + decoding
# ---------------------------------------------------------------------------
def row_key(row: Dict[str, Any]) -> str:
    """Stable id for a row: sha1 of image sha1, task and question.

    :param row: Dataset row.
    :type row: Dict[str, Any]
    :return: 16-hex key.
    :rtype: str
    """
    h = hashlib.sha1(f"{row['image_sha1']}|{row['task']}|{row['question']}".encode()).hexdigest()
    return h[:16]


def load_rows(data_dir: Path, split: str, sources: Sequence[str], tasks: Optional[Sequence[str]],
              limit: Optional[int], exclude_train: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Rows of the split restricted to the requested sources/tasks.

    Works on the mixture directory or on a single component's ``*_images_once``
    export (same layout, no ``source`` column: the first entry of ``sources`` is
    assumed).  ``exclude_train`` drops every row whose page image also appears in
    that parquet, so a component's own validation split can be used as extra
    held-out data without leaking pilot training pages.

    :param data_dir: Mixture or images-once directory (``rows/<split>.parquet`` + ``images/``).
    :type data_dir: Path
    :param split: ``val`` or ``train``.
    :type split: str
    :param sources: Component names to keep (``pgp_qa`` ...).
    :type sources: Sequence[str]
    :param tasks: Optional task filter.
    :type tasks: Optional[Sequence[str]]
    :param limit: Optional cap (rows are taken round-robin per task so a small
        limit still covers every task).
    :type limit: Optional[int]
    :param exclude_train: Optional parquet whose ``image_sha1`` values are excluded.
    :type exclude_train: Optional[Path]
    :return: Row dicts with ``key`` and ``image_path`` added.
    :rtype: List[Dict[str, Any]]
    """
    df = pd.read_parquet(data_dir / "rows" / f"{split}.parquet")
    if "source" not in df.columns:
        df = df.assign(source=list(sources)[0])
    df = df[df["source"].isin(list(sources))]
    if exclude_train is not None:
        seen = set(pd.read_parquet(exclude_train, columns=["image_sha1"])["image_sha1"])
        df = df[~df["image_sha1"].isin(seen)]
    if tasks:
        df = df[df["task"].isin(list(tasks))]
    rows = [dict(r) for r in df.to_dict("records")]
    for r in rows:
        r["key"] = row_key(r)
        r["image_path"] = str(data_dir / "images" / f"{r['image_sha1']}.jpg")
    if limit:
        by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r)
        picked: List[Dict[str, Any]] = []
        while len(picked) < limit and any(by_task.values()):
            for t in sorted(by_task):
                if by_task[t] and len(picked) < limit:
                    picked.append(by_task[t].pop(0))
        rows = picked
    return rows


async def decode_rows(rows: Sequence[Dict[str, Any]], model: str, out_path: Path, temperature: float,
                      done: Dict[str, Dict[str, Any]], min_free_gb: float = 10.0,
                      served_check=None, free_check=None) -> List[Dict[str, Any]]:
    """Decode every row not already in ``done`` through LM Studio, appending to ``out_path``.

    Two safeguards for the shared machine: before each request the run waits while
    local free disk is under ``min_free_gb`` (LM Studio's worker leaks ~32 MB of
    temp per multimodal request), and after a failed request it stops as soon as
    LM Studio no longer lists the model (the run is resumable, so nothing is lost).

    :param rows: Rows to decode.
    :type rows: Sequence[Dict[str, Any]]
    :param model: LM Studio model key.
    :type model: str
    :param out_path: JSONL of records (append mode).
    :type out_path: Path
    :param temperature: Sampling temperature.
    :type temperature: float
    :param done: Records already on disk keyed by row key.
    :type done: Dict[str, Dict[str, Any]]
    :param min_free_gb: Disk floor in GiB; the run pauses below it.
    :type min_free_gb: float
    :param served_check: ``model -> bool`` (default :func:`model_served`; injectable for tests).
    :param free_check: ``() -> GiB free`` (default :func:`free_gb`; injectable for tests).
    :return: All records (existing + new), in row order.
    :rtype: List[Dict[str, Any]]
    """
    served_check = served_check or model_served
    free_check = free_check or free_gb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    todo = [r for r in rows if r["key"] not in done]
    print(f"{len(rows)} rows; {len(done)} already decoded; decoding {len(todo)} with {model}", flush=True)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            if r["key"] in done:
                records.append(done[r["key"]])
                continue
            while free_check() < min_free_gb:
                print(f"  disk floor: {free_check():.1f} GiB free < {min_free_gb} GiB; waiting 60 s", flush=True)
                await asyncio.sleep(60)
            t1 = time.time()
            raw = await transcribe_with_lm_studio(model, r["image_path"], r["question"],
                                                  temperature=temperature, max_tokens=max_tokens_for(r))
            if raw is None and not served_check(model):
                print(f"  {model} is no longer served by LM Studio; stopping after {i - 1} rows (resumable)", flush=True)
                break
            rec = {"key": r["key"], "model": model, "source": r["source"], "task": r["task"],
                   "image_sha1": r["image_sha1"], "question": r["question"], "answer": r["answer"],
                   "raw": raw, "secs": round(time.time() - t1, 1), "score": score_row(r, raw)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
            sc = rec["score"]
            brief = {k: sc[k] for k in ("exact", "span_cer", "iou", "hit", "pred_abstain") if k in sc and sc[k] is not None}
            print(f"  {i:3d}/{len(rows)} {r['source']}/{r['task']:18s} {rec['secs']:5.1f}s {brief}", flush=True)
    print(f"decoded in {(time.time() - t0) / 60:.1f} min", flush=True)
    return records


def free_gb(path: str = "/System/Volumes/Data") -> float:
    """Free space on the volume holding ``path``, in GiB.

    :param path: Any path on the volume.
    :type path: str
    :return: Free GiB.
    :rtype: float
    """
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def model_served(model: str, base_url: str = "http://localhost:1234/v1") -> bool:
    """Whether LM Studio currently lists ``model`` (loaded or loadable).

    :param model: LM Studio model key.
    :type model: str
    :param base_url: LM Studio OpenAI-compatible base URL.
    :type base_url: str
    :return: True when the key appears in ``/models``; False on any error.
    :rtype: bool
    """
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
            ids = {m.get("id") for m in json.load(resp).get("data", [])}
        return model in ids
    except (OSError, ValueError):
        return False


def load_done(out_path: Path) -> Dict[str, Dict[str, Any]]:
    """Records already written to ``out_path`` (for resume / rescore).

    :param out_path: JSONL path.
    :type out_path: Path
    :return: Mapping row key to record.
    :rtype: Dict[str, Dict[str, Any]]
    """
    done: Dict[str, Dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("raw") is None:
                    continue  # a failed decode is retried on the next run
                done[rec["key"]] = rec
    return done


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point.

    :param argv: Arguments (None = ``sys.argv``).
    :type argv: Optional[Sequence[str]]
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", required=True, help="LM Studio model key")
    ap.add_argument("--sources", default="pgp_qa,documentary_grounding",
                    help="comma list of mixture components (pgp_qa, documentary_grounding, ktiv_grounding)")
    ap.add_argument("--tasks", default=None, help="optional comma list of tasks")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--out", type=Path, required=True, help="JSONL of per-row records (resumable)")
    ap.add_argument("--rescore", action="store_true", help="re-score stored raw outputs only; no decoding")
    ap.add_argument("--min-free-gb", type=float, default=10.0, help="pause decoding while local free disk is below this")
    ap.add_argument("--exclude-train", type=Path, default=None,
                    help="parquet whose image_sha1 values are dropped (e.g. the pilot's rows/train.parquet)")
    args = ap.parse_args(argv)

    rows = load_rows(args.data_dir, args.split, args.sources.split(","),
                     args.tasks.split(",") if args.tasks else None, args.limit, args.exclude_train)
    done = load_done(args.out)
    if args.rescore:
        by_key = {r["key"]: r for r in rows}
        records = []
        for key, rec in done.items():
            if key in by_key:
                rec["score"] = score_row(by_key[key], rec.get("raw"))
                records.append(rec)
    else:
        records = asyncio.run(decode_rows(rows, args.model, args.out, args.temperature, done, args.min_free_gb))
    summary = aggregate(records)
    print(format_summary(summary, args.model))
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"model": args.model, "split": args.split, "n": len(records),
                                        "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"summary -> {summary_path}")
    failed = sum(1 for r in records if r.get("raw") is None)
    if failed:
        print(f"WARNING: {failed} rows returned no output (LM Studio failure); re-run to retry them", file=sys.stderr)


if __name__ == "__main__":
    main()
