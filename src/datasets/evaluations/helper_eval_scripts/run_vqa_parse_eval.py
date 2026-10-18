# File name: run_vqa_parse_eval.py
# Date: 10/5/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Evaluate "parse the page, then answer from the image plus a parse" on the held-out pages of ``pgp_vqa_parse_v1``.

For every val page of the set (the edition manifest's held-out documents), a local LM Studio vision
model is asked, one request at a time (one LM Studio consumer at a time):

1. ``parse``: the page-parse prompt of the training set (image -> JSON array of lines);
2. for each fields / question / look-up row of the page, the row's request twice:

   * ``own``: with the model's OWN parse from step 1 in the prompt (the way the model is used:
     parse first, then ask), and
   * ``gold``: with the row's prompt as built (the edition lines, or the cached reading the row
     was built with): the ceiling a perfect parse would allow.

Replies go to ``<out>/<model>.jsonl`` (resumable: a request that has a reply is skipped). The scorer
(``--score``) prints, per condition: how often the reply is valid JSON, the character error of the
parse against the edition lines, and the share of answers whose text equals the edition's after the
QA builder's folding (``build_pgp_qa.normalise_answer``), abstentions included.

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.run_vqa_parse_eval --model <LM Studio key> [--limit-pages N]
    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.run_vqa_parse_eval --score [--models a b]
"""
import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq

from src.datasets.evaluations.metrics import cer_pair, normalize_ink_hypothesis, normalize_whitespace
from src.finetuning.qwen_hebrew import build_pgp_qa as qa
from src.finetuning.qwen_hebrew.build_vqa_parse import CONTEXT_FAMILIES, PARSE_LEAD, context_prompt
from src.finetuning.qwen_hebrew.eval_harness.v22_task_eval import free_gb, model_served
from src.finetuning.qwen_hebrew.images_once import SHA_COLUMN
from src.models.ocr.lms_transcriber import transcribe_with_lm_studio

DEFAULT_EXPORT = Path("/Volumes/home/studio_offload/datasets/pgp_vqa_parse_v1_images_once")
DEFAULT_OUT = Path(__file__).resolve().parents[4] / "logs/next_round/vqa/eval"
PARSE_MAX_TOKENS = 4500
ANSWER_MAX_TOKENS = 400
CONDITIONS = ("own", "gold")
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_PARSE_END = "\n]\n\n"
# one line element of a page parse, found by its shape: the text runs to the closing of its element, so a plain `"`
# inside it (Hebrew abbreviations) does not end it
_LINE_ELEMENT = re.compile(r'\{\s*"n"\s*:\s*\d+\s*,\s*"text"\s*:\s*"(.*?)"\s*(?:,\s*"bbox_2d"\s*:\s*\[[^\]]*\]\s*)?\}(?=\s*(?:,\s*\{|\]|$))',
                           re.S)


# ----------------------------------------------------------------------------- replies

def reply_json(text: Optional[str]) -> Any:
    """The JSON value of a model reply, or ``None`` when it is not JSON.

    :param text: Raw reply (may carry a Markdown code fence).
    :type text: Optional[str]
    :return: The decoded value; ``None`` for a missing, empty or malformed reply. A JSON ``null`` reply also
        gives ``None``: no training target is a bare null.
    :rtype: Any
    """
    if not text:
        return None
    body = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def parse_lines(reply: Any) -> Optional[List[str]]:
    """The line texts of a page-parse reply.

    :param reply: Decoded reply (:func:`reply_json`).
    :type reply: Any
    :return: Line texts in order, or ``None`` when the reply is not a list of ``{"text": str}`` objects.
    :rtype: Optional[List[str]]
    """
    if not isinstance(reply, list) or not reply:
        return None
    if not all(isinstance(e, dict) and isinstance(e.get("text"), str) for e in reply):
        return None
    return [e["text"] for e in reply]


def reply_lines(text: Optional[str]) -> Optional[List[str]]:
    """The lines a page-parse reply holds, read leniently: what the model's own parse is taken to be.

    A reply that is a valid JSON parse gives its lines (:func:`parse_lines`). Otherwise the line elements are found
    by their shape, so a reply with a plain ``"`` inside a text, a reply cut off by the token limit, or a single
    element without the surrounding array still yields the lines it does contain.

    :param text: Raw reply.
    :type text: Optional[str]
    :return: Line texts in order, or ``None`` when the reply holds no line element at all.
    :rtype: Optional[List[str]]
    """
    strict = parse_lines(reply_json(text))
    if strict:
        return strict
    found = [m.group(1) for m in _LINE_ELEMENT.finditer(_FENCE.sub("", (text or "").strip()))]
    lines = []
    for field in found:
        try:
            lines.append(json.loads(f'"{field}"'))
        except json.JSONDecodeError:
            lines.append(field.replace('\\"', '"').replace("\\n", " "))
    return lines or None


def request_of(question: str) -> str:
    """The request part of a context-family prompt (what follows the parse it shows).

    :param question: A fields / question / look-up prompt of the set.
    :type question: str
    :return: The text after the parse.
    :rtype: str
    :raises ValueError: When the prompt does not open with the lead and a parse.
    """
    if not question.startswith(PARSE_LEAD) or _PARSE_END not in question:
        raise ValueError("not a context-family prompt")
    return question.split(_PARSE_END, 1)[1]


def answer_texts(value: Any) -> Optional[List[str]]:
    """Folded texts a decoded answer asserts, sorted; ``[]`` for "not stated".

    Covers every target shape of the set: ``{"answer": str|None|[{"text"}], "line"}``, a look-up's
    ``{"line", "text"}``, and one field value (``{"text", "line"}``, a list of them, or ``None``).

    :param value: Decoded answer (or field value).
    :type value: Any
    :return: Sorted folded texts, or ``None`` when the value has none of the agreed shapes.
    :rtype: Optional[List[str]]
    """
    if value is None:
        return []
    if isinstance(value, dict) and "answer" in value:
        value = value["answer"]
        if value is None:
            return []
        if isinstance(value, str):
            return [qa.normalise_answer(value)]
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return [qa.normalise_answer(value["text"])]
    if isinstance(value, list) and value and all(isinstance(i, dict) and isinstance(i.get("text"), str) for i in value):
        return sorted(qa.normalise_answer(i["text"]) for i in value)
    return None


def answer_matches(task: str, gold: str, reply: Any) -> Tuple[int, int]:
    """Score one context-family reply against its target.

    :param task: ``fields_from_parse``, ``question_from_parse`` or ``lookup_from_parse``.
    :type task: str
    :param gold: The row's target (JSON text).
    :type gold: str
    :param reply: Decoded model reply (``None`` when it was not JSON).
    :type reply: Any
    :return: ``(answers right, answers asked)``: one per field for a fields row, else one. An answer is
        right when its folded text(s) equal the target's; "not stated" is right only when the reply says so.
    :rtype: Tuple[int, int]
    """
    target = json.loads(gold)
    if task == "fields_from_parse":
        got = reply if isinstance(reply, dict) else {}
        return sum(k in got and answer_texts(got[k]) == answer_texts(v) for k, v in target.items()), len(target)
    if not isinstance(reply, dict):
        return 0, 1
    return int(answer_texts(reply) == answer_texts(target)), 1


# ----------------------------------------------------------------------------- jobs

def val_pages(export: Path, limit_pages: Optional[int] = None, families: Sequence[str] = CONTEXT_FAMILIES,
              only_with_rows: bool = False) -> List[Dict[str, Any]]:
    """The held-out pages with their rows.

    :param export: Images-once export of ``pgp_vqa_parse_v1``.
    :type export: Path
    :param limit_pages: Only the first N pages (in image-hash order).
    :type limit_pages: Optional[int]
    :param families: Context families whose rows are kept.
    :type families: Sequence[str]
    :param only_with_rows: Leave out the pages that have no row of ``families`` (a short run on the pages
        that carry facts).
    :type only_with_rows: bool
    :return: One dict per page: ``{"image": path, "sha": ..., "parse": row, "context": [rows]}``; pages
        without a ``parse_lines`` row are left out.
    :rtype: List[Dict[str, Any]]
    """
    rows = pq.read_table(export / "rows" / "val.parquet").to_pylist()
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[row[SHA_COLUMN]].append(row)
    pages = []
    for sha in sorted(by_image):
        parse = [r for r in by_image[sha] if r["task"] == "parse_lines"]
        context = [r for r in by_image[sha] if r["task"] in families]
        if parse and (context or not only_with_rows):
            pages.append({"image": str(export / "images" / f"{sha}.jpg"), "sha": sha, "parse": parse[0], "context": context})
    return pages[:limit_pages] if limit_pages else pages


def load_replies(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Replies already stored for a model.

    :param path: ``<out>/<model>.jsonl``.
    :type path: Path
    :return: ``(stem, condition) -> record`` for records that hold a reply.
    :rtype: Dict[Tuple[str, str], Dict[str, Any]]
    """
    done: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if isinstance(rec.get("reply"), str):
                        done[(rec["stem"], rec["condition"])] = rec
    return done


async def run(model: str, export: Path = DEFAULT_EXPORT, out_dir: Path = DEFAULT_OUT, limit_pages: Optional[int] = None,
              min_free_gb: float = 10.0, transcribe: Optional[Callable[..., Awaitable[Optional[str]]]] = None,
              served_check: Optional[Callable[[str], bool]] = None, free_check: Optional[Callable[[], float]] = None,
              wait_s: float = 60.0, families: Sequence[str] = CONTEXT_FAMILIES, conditions: Sequence[str] = CONDITIONS,
              only_with_rows: bool = False) -> Dict[str, int]:
    """Parse every held-out page, then ask its rows with the model's own parse and with the row's parse.

    :param model: LM Studio model key.
    :type model: str
    :param export: Images-once export of ``pgp_vqa_parse_v1``.
    :type export: Path
    :param out_dir: Folder of the reply files.
    :type out_dir: Path
    :param limit_pages: Only the first N pages.
    :type limit_pages: Optional[int]
    :param min_free_gb: Disk floor in GiB; the run waits below it.
    :type min_free_gb: float
    :param transcribe: ``(model, image_path, prompt, max_tokens=...) -> text or None`` (injectable for tests).
    :param served_check: ``model -> bool`` (default: LM Studio lists the model).
    :param free_check: ``() -> GiB free`` (default: the local data volume).
    :param wait_s: Seconds between disk checks while under the floor.
    :type wait_s: float
    :param families: Context families to ask (default: all three).
    :type families: Sequence[str]
    :param conditions: ``own`` and / or ``gold`` (default: both).
    :type conditions: Sequence[str]
    :param only_with_rows: Only the pages that have a row of ``families``.
    :type only_with_rows: bool
    :return: ``{"asked": n, "failed": n, "skipped": already answered, "stopped": 1 if the model vanished}``.
    :rtype: Dict[str, int]
    """
    transcribe = transcribe or transcribe_with_lm_studio
    served_check = served_check or model_served
    free_check = free_check or free_gb
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model.replace('/', '_')}.jsonl"
    done = load_replies(path)
    pages = val_pages(export, limit_pages, families, only_with_rows)
    stats = {"asked": 0, "failed": 0, "skipped": 0, "stopped": 0}
    print(f"{model}: {len(pages)} held-out pages, {sum(len(p['context']) for p in pages)} context rows, {len(done)} replies stored", flush=True)
    started = time.time()

    with open(path, "a", encoding="utf-8") as fh:
        async def ask(stem: str, condition: str, image: str, prompt: str, max_tokens: int) -> Optional[str]:
            """Ask once unless a reply is stored; store the reply."""
            if (stem, condition) in done:
                stats["skipped"] += 1
                return done[(stem, condition)]["reply"]
            while free_check() < min_free_gb:
                print(f"  disk floor: {free_check():.1f} GiB free < {min_free_gb} GiB; waiting", flush=True)
                await asyncio.sleep(wait_s)
            t0 = time.time()
            reply = await transcribe(model, image, prompt, max_tokens=max_tokens)
            fh.write(json.dumps({"stem": stem, "condition": condition, "reply": reply, "secs": round(time.time() - t0, 1),
                                 "model": model}, ensure_ascii=False) + "\n")
            fh.flush()
            stats["asked" if isinstance(reply, str) else "failed"] += 1
            return reply

        for n, page in enumerate(pages, 1):
            raw = await ask(page["parse"]["stem"], "parse", page["image"], page["parse"]["question"], PARSE_MAX_TOKENS)
            if raw is None and not served_check(model):
                print(f"  {model} is no longer served by LM Studio; stopping at page {n} (resumable)", flush=True)
                stats["stopped"] = 1
                break
            own = reply_lines(raw)                    # lenient: the lines the reply holds, valid JSON or not
            for row in page["context"]:
                if "gold" in conditions:
                    await ask(row["stem"], "gold", page["image"], row["question"], ANSWER_MAX_TOKENS)
                if own and "own" in conditions:       # without a usable parse of its own the model cannot be asked this way
                    await ask(row["stem"], "own", page["image"], context_prompt(own, request_of(row["question"])), ANSWER_MAX_TOKENS)
            valid = parse_lines(reply_json(raw)) is not None
            print(f"  {n:3d}/{len(pages)} {page['parse']['stem']}: parse {'valid JSON' if valid else 'NOT valid JSON'}, "
                  f"{len(own) if own else 0} lines; {len(page['context'])} context rows", flush=True)
    print(f"done: {stats} in {(time.time() - started) / 60:.1f} min -> {path}", flush=True)
    return stats


# ----------------------------------------------------------------------------- scoring

def visible_text(lines: Iterable[str]) -> str:
    """Scoring form of a page's lines: the benchmark scorer's visible-ink form (no gap or doubt markers), one space
    between words, applied to edition lines and model lines alike.

    :param lines: Line texts in order.
    :type lines: Iterable[str]
    :return: Text to compare.
    :rtype: str
    """
    return normalize_whitespace(normalize_ink_hypothesis("\n".join(lines)))


def score_model(pages: Sequence[Dict[str, Any]], replies: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """Scores of one model's stored replies.

    :param pages: Held-out pages (:func:`val_pages`).
    :type pages: Sequence[Dict[str, Any]]
    :param replies: The model's replies (:func:`load_replies`).
    :type replies: Dict[Tuple[str, str], Dict[str, Any]]
    :return: ``{"parse": {"pages", "json", "cer_median", "cer_pooled"}, "<family>|<condition>": {"right", "asked",
        "json", "rows", "missing"}}``. ``json`` counts strictly valid JSON; parse CER (nikud ignored) is over every
        page with a reply, on the lines the reply holds (:func:`reply_lines`; none = CER 1). A context row without a
        stored reply counts as asked and wrong (``missing``), so conditions stay comparable.
    :rtype: Dict[str, Any]
    """
    cers: List[float] = []
    err = ref = 0.0
    parsed = answered = 0
    out: Dict[str, Any] = defaultdict(lambda: {"right": 0, "asked": 0, "json": 0, "rows": 0, "missing": 0})
    for page in pages:
        rec = replies.get((page["parse"]["stem"], "parse"))
        if rec:
            answered += 1
            parsed += parse_lines(reply_json(rec["reply"])) is not None
            gold = visible_text(e["text"] for e in json.loads(page["parse"]["answer"]))
            cer = cer_pair(visible_text(reply_lines(rec["reply"]) or []), gold)[1]      # without nikud, as the benchmarks report it
            cers.append(cer)
            err += cer * len(gold)
            ref += len(gold)
        for row in page["context"]:
            for condition in CONDITIONS:
                cell = out[f"{row['task']}|{condition}"]
                rec = replies.get((row["stem"], condition))
                reply = reply_json(rec["reply"]) if rec else None
                right, asked = answer_matches(row["task"], row["answer"], reply)
                cell["right"] += right
                cell["asked"] += asked
                cell["rows"] += 1
                cell["json"] += reply is not None
                cell["missing"] += rec is None
    cers.sort()
    result = dict(out)
    result["parse"] = {"pages": answered, "json": parsed, "cer_median": round(cers[len(cers) // 2], 4) if cers else None,
                       "cer_pooled": round(err / ref, 4) if ref else None}
    return result


def score(out_dir: Path = DEFAULT_OUT, export: Path = DEFAULT_EXPORT, models: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Print and return the scores of every model with stored replies.

    :param out_dir: Folder of the reply files.
    :type out_dir: Path
    :param export: Images-once export of ``pgp_vqa_parse_v1``.
    :type export: Path
    :param models: Model names (file stems); default: every file in ``out_dir``.
    :type models: Optional[Iterable[str]]
    :return: ``model -> scores`` (:func:`score_model`).
    :rtype: Dict[str, Dict[str, Any]]
    """
    pages = val_pages(export)
    names = list(models) if models else sorted(p.stem for p in out_dir.glob("*.jsonl"))
    results = {}
    print("| model | parse pages (JSON) | parse CER median / pooled | "
          + " | ".join(f"{fam.split('_')[0]} own | {fam.split('_')[0]} gold" for fam in CONTEXT_FAMILIES) + " |")
    print("|---|---|---|" + "---|" * (2 * len(CONTEXT_FAMILIES)))
    for name in names:
        res = score_model(pages, load_replies(out_dir / f"{name}.jsonl"))
        results[name] = res
        cells = []
        for fam in CONTEXT_FAMILIES:
            for condition in CONDITIONS:
                c = res.get(f"{fam}|{condition}", {"right": 0, "asked": 0, "rows": 0, "missing": 0})
                cells.append(f"{c['right']}/{c['asked']}" if c["asked"] and c["missing"] < c["rows"] else "-")   # "-" = never asked
        p = res["parse"]
        print(f"| {name} | {p['pages']} ({p['json']}) | {p['cer_median']} / {p['cer_pooled']} | " + " | ".join(cells) + " |")
    return results


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="LM Studio model key to evaluate")
    ap.add_argument("--score", action="store_true", help="score the stored replies instead of asking")
    ap.add_argument("--models", nargs="*", default=None, help="with --score: only these reply files")
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit-pages", type=int, default=None)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--families", nargs="+", choices=CONTEXT_FAMILIES, default=list(CONTEXT_FAMILIES), help="context families to ask")
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS),
                    help="own = the model's own parse in the prompt, gold = the row's prompt as built")
    ap.add_argument("--fact-pages-only", action="store_true", help="only the pages that have a row of the chosen families")
    args = ap.parse_args(argv)
    if args.score:
        score(args.out, args.export, args.models)
    elif args.model:
        asyncio.run(run(args.model, args.export, args.out, args.limit_pages, args.min_free_gb, families=args.families,
                        conditions=args.conditions, only_with_rows=args.fact_pages_only))
    else:
        ap.error("give --model to evaluate or --score to score")


if __name__ == "__main__":
    main()
