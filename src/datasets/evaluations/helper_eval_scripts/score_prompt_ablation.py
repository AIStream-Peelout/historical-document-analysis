# File name: score_prompt_ablation.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Score the Arabic-script prompt ablation on its panel (offline, no model needed).

Every arm of ``run_prompt_ablation.py`` (one model, one prompt variant, one output budget) answered the
same images of a panel built by ``build_prompt_ablation_panel.py``. This module scores each arm by image
class:

* ``covered`` images are scored with the covered-only scorer (:func:`score_arabic_covered.score_covered_model`:
  per document, the answers to its covered images against the edition sections they carry; clipped 5-gram
  precision / recall / F1 and LER), restricted to the panel's documents, plus the share of covered images
  on which the model abstained;
* the uncovered classes (``hebrew_side``, ``blank_faint``, ``arabic_unedited``, ``other_uncovered``) are scored
  by behaviour: Arabic and Hebrew letters emitted, the share of images with at least
  :data:`score_arabic_covered.OVERREAD_MIN_ARABIC` Arabic letters ("read"), the share that abstained (the
  marker of the stopping prompt, an empty answer, or an answer with no script letter such as a run of
  ``[...]``), loops (``arabic_script.repeat_share`` above the scorer's threshold), Hebrew-script output,
  and the generation's end: ``finish_reason`` (``stop`` = end token emitted, ``length`` = budget cut) and
  completion tokens.

Within a model the prompt variants are compared document by document on the covered class
(:func:`score_arabic_covered.paired_comparison`, bootstrap intervals), and the two output budgets of one
prompt likewise. An earlier full run's answers (``--reference TAG=PATH``) can be scored on the same panel
images as a reference arm (no finish reason or token counts in those rows).

Outputs (``--out-dir``): ``images.csv`` (every arm, every panel image), ``documents.csv`` (covered
documents per arm), ``summary.json`` and the Markdown report (``--report``).

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.score_prompt_ablation \\
        --panel <benchmark>/panels/prompt_ablation_v1.jsonl [--arms-dir <benchmark>/outputs_prompt_ablation] \\
        [--reference v25a-1200-cached=<benchmark>/outputs/qwen3-vl-8b-heb-v25a-step1200__trained.jsonl] \\
        [--out-dir DIR] [--report FILE]
"""
import argparse
import csv
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import score_arabic_benchmark as sab
from src.datasets.evaluations.helper_eval_scripts import score_arabic_covered as sac
from src.datasets.evaluations.helper_eval_scripts.build_prompt_ablation_panel import CLASSES, COVERED, load_panel
from src.datasets.evaluations.helper_eval_scripts.run_prompt_ablation import ABSTAIN_MARKER, DEFAULT_OUT_DIR, VARIANTS

ARM_RE = re.compile(r"^(?P<model>.+?)__(?P<prompt>[A-Za-z0-9-]+)__mt(?P<max_tokens>\d+)$")
GAP_RE = re.compile(r"\[\.\.\.\]|\[…\]")
MODEL_RE = re.compile(r"qwen3-vl-8b-heb-v(?P<major>\d)(?P<minor>\d)(?P<letter>[a-z])-step(?P<step>\d+)")
DEFAULT_SCORES_DIR = "scores_prompt_ablation"
UNCOVERED_CLASSES = tuple(c for c in CLASSES if c != COVERED)
IMAGE_FIELDS = ["arm", "model", "prompt", "max_tokens", "doc_id", "image_index", "image_class", "answered", "abstain", "chars",
                "arabic_letters", "hebrew_letters", "gap_markers", "repeat_share", "loop", "hebrew_output", "ge40",
                "finish_reason", "prompt_tokens", "completion_tokens", "secs"]
DOC_FIELDS = ["arm", "model", "prompt", "max_tokens"] + [f for f in sac.DOC_FIELDS if f != "model"]
ABSTAIN_KINDS = ("marker", "empty", "no_letters")


@dataclass
class Arm:
    """One decoded condition.

    :param tag: File stem (``<model>__<prompt>__mt<max_tokens>`` or a reference name).
    :param model: Model key.
    :param prompt: Prompt variant.
    :param max_tokens: Output budget.
    :param rows: Answer rows by ``(doc_id, image_index)``.
    """
    tag: str
    model: str
    prompt: str
    max_tokens: int
    rows: Dict[Tuple[str, int], Dict[str, Any]] = field(default_factory=dict)


def short_model(model: str) -> str:
    """``qwen3-vl-8b-heb-v25a-step1200`` -> ``v2.5a-1200`` (a suffix after the step, e.g. ``-ctx16k``, is kept); other keys unchanged.

    :param model: Model key.
    :type model: str
    :return: Short name.
    :rtype: str
    """
    m = MODEL_RE.search(model)
    return f"v{m['major']}.{m['minor']}{m['letter']}-{m['step']}{model[m.end():]}" if m else model


def parse_arm_tag(tag: str) -> Optional[Tuple[str, str, int]]:
    """``(model, prompt, max_tokens)`` of an arm file stem, or None when the stem is not one.

    :param tag: File stem.
    :type tag: str
    :return: The parts, or None.
    :rtype: Optional[Tuple[str, str, int]]
    """
    m = ARM_RE.match(tag)
    return (m["model"], m["prompt"], int(m["max_tokens"])) if m else None


def load_rows(path: Path, keep: Optional[Sequence[Tuple[str, int]]] = None) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Answer rows of one file by image (the last row for an image wins).

    :param path: ``.jsonl`` of ``run_prompt_ablation.py`` or ``run_arabic_benchmark.py``.
    :type path: Path
    :param keep: Only these images (default: all).
    :type keep: Optional[Sequence[Tuple[str, int]]]
    :return: ``{(doc_id, image_index): row}``.
    :rtype: Dict[Tuple[str, int], Dict[str, Any]]
    """
    wanted = None if keep is None else set(keep)
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                key = (row["doc_id"], int(row["image_index"]))
                if wanted is None or key in wanted:
                    rows[key] = row
    return rows


def discover_arms(arms_dir: Path, panel_keys: Sequence[Tuple[str, int]]) -> List[Arm]:
    """Every arm file of a directory, in model / prompt / budget order.

    :param arms_dir: ``outputs_prompt_ablation``.
    :type arms_dir: Path
    :param panel_keys: Panel images (rows for other images are dropped).
    :type panel_keys: Sequence[Tuple[str, int]]
    :return: Arms sorted by model, then prompt in :data:`run_prompt_ablation.VARIANTS` order, then budget descending.
    :rtype: List[Arm]
    """
    arms = []
    for path in sorted(arms_dir.glob("*.jsonl")):
        parts = parse_arm_tag(path.stem)
        if parts is not None:
            arms.append(Arm(path.stem, *parts, rows=load_rows(path, panel_keys)))
    order = {v: n for n, v in enumerate(VARIANTS)}
    return sorted(arms, key=lambda a: (a.model, order.get(a.prompt, len(order)), -a.max_tokens))


def reference_arm(name: str, path: Path, panel_keys: Sequence[Tuple[str, int]]) -> Arm:
    """An earlier full run restricted to the panel, as an arm.

    :param name: Tag for the tables.
    :type name: str
    :param path: ``outputs/<model>__<prompt>.jsonl`` of ``run_arabic_benchmark.py``.
    :type path: Path
    :param panel_keys: Panel images.
    :type panel_keys: Sequence[Tuple[str, int]]
    :return: The arm (``model`` and ``prompt`` from the file stem, budget 2,500 as every earlier Arabic run).
    :rtype: Arm
    """
    model, _, prompt = path.stem.partition("__")
    return Arm(name, model, f"{prompt}(cached)", 2500, load_rows(path, panel_keys))


def abstain_kind(text: Optional[str]) -> Optional[str]:
    """How an answer abstained, or None when it read something (or was not answered).

    :param text: The answer.
    :type text: Optional[str]
    :return: ``marker`` (the stopping prompt's marker), ``empty`` (nothing), ``no_letters`` (no Arabic or Hebrew
        letter, e.g. a run of ``[...]``), or None.
    :rtype: Optional[str]
    """
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return "empty"
    arabic, hebrew, _ = ar.script_share(stripped)
    if ABSTAIN_MARKER in stripped and arabic + hebrew == 0:
        return "marker"
    if arabic + hebrew == 0:
        return "no_letters"
    return None


def measure(row: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Behaviour measures of one answer row.

    :param row: Answer row, or None when the image has no row.
    :type row: Optional[Mapping[str, Any]]
    :return: :func:`score_arabic_covered.image_measures` of the text plus ``abstain``, ``chars``, ``gap_markers``,
        ``finish_reason``, ``prompt_tokens``, ``completion_tokens``, ``secs``.
    :rtype: Dict[str, Any]
    """
    text = row.get("text") if row else None
    text = text if isinstance(text, str) else None
    out = sac.image_measures(text)
    out.update({"abstain": abstain_kind(text), "chars": len(text) if text else 0, "gap_markers": len(GAP_RE.findall(text)) if text else 0,
                "finish_reason": row.get("finish_reason") if row else None, "prompt_tokens": row.get("prompt_tokens") if row else None,
                "completion_tokens": row.get("completion_tokens") if row else None, "secs": row.get("secs") if row else None})
    return out


def image_table(arm: Arm, panel: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One measured row per panel image for an arm.

    :param arm: The arm.
    :type arm: Arm
    :param panel: Panel rows.
    :type panel: Sequence[Mapping[str, Any]]
    :return: Rows with :data:`IMAGE_FIELDS`.
    :rtype: List[Dict[str, Any]]
    """
    out = []
    for p in panel:
        key = (p["doc_id"], int(p["image_index"]))
        m = measure(arm.rows.get(key))
        out.append({"arm": arm.tag, "model": arm.model, "prompt": arm.prompt, "max_tokens": arm.max_tokens, "doc_id": p["doc_id"],
                    "image_index": int(p["image_index"]), "image_class": p["image_class"], **{k: m[k] for k in IMAGE_FIELDS if k in m}})
    return out


def covered_documents(benchmark: Mapping[str, Mapping[str, Any]], coverage: Mapping[str, Mapping[int, Mapping[str, Any]]],
                      arm: Arm, panel: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Covered-only document rows of an arm on the panel's covered documents.

    :param benchmark: :func:`score_arabic_benchmark.load_benchmark` result.
    :type benchmark: Mapping[str, Mapping[str, Any]]
    :param coverage: :func:`score_arabic_covered.load_coverage` result.
    :type coverage: Mapping[str, Mapping[int, Mapping[str, Any]]]
    :param arm: The arm.
    :type arm: Arm
    :param panel: Panel rows.
    :type panel: Sequence[Mapping[str, Any]]
    :return: Rows with :data:`DOC_FIELDS`.
    :rtype: List[Dict[str, Any]]
    """
    docs = sorted({p["doc_id"] for p in panel if p["image_class"] == COVERED})
    answers: Dict[str, Dict[int, str]] = {}
    for p in panel:
        if p["image_class"] == COVERED:
            row = arm.rows.get((p["doc_id"], int(p["image_index"])))
            if row and isinstance(row.get("text"), str):
                answers.setdefault(p["doc_id"], {})[int(p["image_index"])] = row["text"]
    doc_rows, _ = sac.score_covered_model(benchmark, coverage, answers, arm.tag, only=docs)
    return [{"arm": arm.tag, "model": arm.model, "prompt": arm.prompt, "max_tokens": arm.max_tokens,
             **{k: v for k, v in r.items() if k != "model"}} for r in doc_rows]


def _share(part: int, whole: int) -> Optional[float]:
    """``part / whole`` to three places, None for an empty whole."""
    return round(part / whole, 3) if whole else None


def _mean(values: Sequence[Any]) -> Optional[float]:
    """Mean of the non-None values to one place, None when there are none."""
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 1) if vals else None


def summarise_class(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Behaviour summary of the image rows of one class and arm.

    :param rows: :func:`image_table` rows of one class.
    :type rows: Sequence[Mapping[str, Any]]
    :return: ``images``, ``answered``, ``abstain`` (share of answered), ``abstain_kinds``, ``arabic_mean`` /
        ``arabic_median`` (letters per answered image), ``hebrew_mean``, ``ge40`` / ``loop`` / ``hebrew_output`` /
        ``length`` (shares of answered), ``finish`` (counts), ``tokens_mean``, ``chars_mean``, ``gap_markers_mean``.
    :rtype: Dict[str, Any]
    """
    answered = [r for r in rows if r["answered"]]
    arabic = [r["arabic_letters"] for r in answered]
    return {"images": len(rows), "answered": len(answered),
            "abstain": _share(sum(r["abstain"] is not None for r in answered), len(answered)),
            "abstain_kinds": dict(Counter(r["abstain"] for r in answered if r["abstain"])),
            "arabic_mean": _mean(arabic), "arabic_median": sac._median(arabic),
            "hebrew_mean": _mean([r["hebrew_letters"] for r in answered]),
            "ge40": _share(sum(r["ge40"] for r in answered), len(answered)),
            "loop": _share(sum(r["loop"] for r in answered), len(answered)),
            "hebrew_output": _share(sum(r["hebrew_output"] for r in answered), len(answered)),
            "length": _share(sum(r["finish_reason"] == "length" for r in answered), len(answered)) if any(r["finish_reason"] for r in answered) else None,
            "finish": dict(Counter(str(r["finish_reason"]) for r in answered)),
            "tokens_mean": _mean([r["completion_tokens"] for r in answered]),
            "chars_mean": _mean([r["chars"] for r in answered]),
            "gap_markers_mean": _mean([r["gap_markers"] for r in answered])}


def summarise_covered(doc_rows: Sequence[Mapping[str, Any]], image_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recognition summary of an arm on the covered class.

    :param doc_rows: :func:`covered_documents` rows.
    :type doc_rows: Sequence[Mapping[str, Any]]
    :param image_rows: :func:`image_table` rows of the covered class.
    :type image_rows: Sequence[Mapping[str, Any]]
    :return: ``documents``, medians of precision / recall / F1 / LER, ``f1_mean``, ``docs_f1_ge_0_2``, ``status``
        counts, and the class summary of the images under ``images``.
    :rtype: Dict[str, Any]
    """
    f1 = [r["ngram_f1"] for r in doc_rows if r["ngram_f1"] is not None]
    return {"documents": len(doc_rows),
            "precision": sac._median([r["ngram_precision"] for r in doc_rows if r["ngram_precision"] is not None]),
            "recall": sac._median([r["ngram_recall"] for r in doc_rows if r["ngram_recall"] is not None]),
            "f1": sac._median(f1), "f1_mean": round(statistics.mean(f1), 4) if f1 else None,
            "ler": sac._median([r["ler"] for r in doc_rows if r["ler"] is not None]),
            "docs_f1_ge_0_2": sum(v >= 0.2 for v in f1), "status": dict(Counter(r["status"] for r in doc_rows)),
            "images": summarise_class(image_rows)}


def score_arm(arm: Arm, panel: Sequence[Mapping[str, Any]], benchmark: Mapping[str, Mapping[str, Any]],
              coverage: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> Dict[str, Any]:
    """Everything the report needs for one arm.

    :param arm: The arm.
    :type arm: Arm
    :param panel: Panel rows.
    :type panel: Sequence[Mapping[str, Any]]
    :param benchmark: Benchmark records.
    :type benchmark: Mapping[str, Mapping[str, Any]]
    :param coverage: Coverage map.
    :type coverage: Mapping[str, Mapping[int, Mapping[str, Any]]]
    :return: ``images`` (rows), ``documents`` (rows), ``by_class`` (summaries; ``covered`` from :func:`summarise_covered`).
    :rtype: Dict[str, Any]
    """
    images = image_table(arm, panel)
    documents = covered_documents(benchmark, coverage, arm, panel)
    by_class = {c: summarise_class([r for r in images if r["image_class"] == c]) for c in UNCOVERED_CLASSES}
    by_class[COVERED] = summarise_covered(documents, [r for r in images if r["image_class"] == COVERED])
    return {"images": images, "documents": documents, "by_class": by_class}


def _f(v: Any, places: int = 3) -> str:
    """Table cell."""
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{places}f}"
    return str(v)


def _pct(v: Optional[float]) -> str:
    """Share as a percentage cell."""
    return "–" if v is None else f"{100 * v:.0f} %"


def arm_label(arm: Arm) -> str:
    """``v2.5a-1200 / trained / 2500``."""
    return f"{short_model(arm.model)} / {arm.prompt} / {arm.max_tokens}"


def table_main(arms: Sequence[Arm], scored: Mapping[str, Mapping[str, Any]]) -> str:
    """Prompt × model × image class: the headline table.

    :param arms: Arms in order.
    :type arms: Sequence[Arm]
    :param scored: ``{tag: score_arm(...)}``.
    :type scored: Mapping[str, Mapping[str, Any]]
    :return: Markdown table.
    :rtype: str
    """
    lines = ["| model / prompt / max_tokens | covered: docs | F1 | LER | F1≥0.2 | abstain | Hebrew sides: Ar letters | ≥40 Ar | abstain | loop "
             "| blank: Ar letters | ≥40 Ar | abstain | unedited Arabic: ≥40 Ar | letters | other uncovered: ≥40 Ar | letters |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        b = scored[arm.tag]["by_class"]
        c, h, k, u, o = b[COVERED], b["hebrew_side"], b["blank_faint"], b["arabic_unedited"], b["other_uncovered"]
        lines.append(f"| {arm_label(arm)} | {c['documents']} | {_f(c['f1'])} | {_f(c['ler'])} | {c['docs_f1_ge_0_2']} | {_pct(c['images']['abstain'])} "
                     f"| {_f(h['arabic_mean'], 0)} | {_pct(h['ge40'])} | {_pct(h['abstain'])} | {_pct(h['loop'])} "
                     f"| {_f(k['arabic_mean'], 0)} | {_pct(k['ge40'])} | {_pct(k['abstain'])} "
                     f"| {_pct(u['ge40'])} | {_f(u['arabic_mean'], 0)} | {_pct(o['ge40'])} | {_f(o['arabic_mean'], 0)} |")
    return "\n".join(lines)


def table_finish(arms: Sequence[Arm], scored: Mapping[str, Mapping[str, Any]]) -> str:
    """finish_reason distribution per arm and class (``stop`` / ``length`` / other), with mean completion tokens.

    :param arms: Arms.
    :type arms: Sequence[Arm]
    :param scored: ``{tag: score_arm(...)}``.
    :type scored: Mapping[str, Mapping[str, Any]]
    :return: Markdown table.
    :rtype: str
    """
    head = "| model / prompt / max_tokens | " + " | ".join(f"{c}: stop / length / other, mean tokens" for c in CLASSES) + " | all: length share |"
    lines = [head, "|" + "---|" * (len(CLASSES) + 2)]
    for arm in arms:
        cells = []
        total = Counter()
        for c in CLASSES:
            s = scored[arm.tag]["by_class"][c]
            s = s["images"] if c == COVERED else s
            fin = Counter(s["finish"])
            total.update(fin)
            other = sum(v for k, v in fin.items() if k not in ("stop", "length"))
            cells.append(f"{fin.get('stop', 0)} / {fin.get('length', 0)} / {other}, {_f(s['tokens_mean'], 0)}")
        n = sum(total.values())
        lines.append(f"| {arm_label(arm)} | " + " | ".join(cells) + f" | {_pct(total.get('length', 0) / n) if n and total.get('length') is not None else '–'} |")
    return "\n".join(lines)


def paired_tables(arms: Sequence[Arm], scored: Mapping[str, Mapping[str, Any]]) -> str:
    """Within-model paired comparisons on the covered documents: each prompt against ``trained`` at the same
    budget, and 1,200 against 2,500 tokens for each prompt.

    :param arms: Arms.
    :type arms: Sequence[Arm]
    :param scored: ``{tag: score_arm(...)}``.
    :type scored: Mapping[str, Mapping[str, Any]]
    :return: Markdown table.
    :rtype: str
    """
    by_key = {(a.model, a.prompt, a.max_tokens): a for a in arms}
    lines = ["| comparison (new vs old) | docs | F1 new / old | median ΔF1 [95 % CI] | W / L | LER new / old | median ΔLER [95 % CI] | W / L |", "|---|---|---|---|---|---|---|---|"]

    def row(new: Arm, old: Arm) -> str:
        f = sac.paired_comparison(scored[new.tag]["documents"], scored[old.tag]["documents"], "ngram_f1", True)
        l = sac.paired_comparison(scored[new.tag]["documents"], scored[old.tag]["documents"], "ler", False)
        if not f.get("documents"):
            return f"| {arm_label(new)} vs {arm_label(old)} | 0 | – | – | – | – | – | – |"
        return (f"| {arm_label(new)} vs {arm_label(old)} | {f['documents']} | {_f(f['new_median'])} / {_f(f['old_median'])} "
                f"| {f['median_delta']:+.3f} [{f['median_ci'][0]:+.3f}, {f['median_ci'][1]:+.3f}] | {f['wins']} / {f['losses']} "
                f"| {_f(l['new_median'])} / {_f(l['old_median'])} | {l['median_delta']:+.3f} [{l['median_ci'][0]:+.3f}, {l['median_ci'][1]:+.3f}] | {l['wins']} / {l['losses']} |")

    for arm in arms:
        if arm.prompt != VARIANTS[0] and (arm.model, VARIANTS[0], arm.max_tokens) in by_key:
            lines.append(row(arm, by_key[(arm.model, VARIANTS[0], arm.max_tokens)]))
    for arm in arms:
        if arm.max_tokens != 2500 and (arm.model, arm.prompt, 2500) in by_key:
            lines.append(row(arm, by_key[(arm.model, arm.prompt, 2500)]))
    return "\n".join(lines)


def table_hebrew_detail(arms: Sequence[Arm], scored: Mapping[str, Mapping[str, Any]]) -> str:
    """Per arm, the uncovered classes in detail: Hebrew letters, Hebrew-script output, gap markers, loops, chars.

    :param arms: Arms.
    :type arms: Sequence[Arm]
    :param scored: ``{tag: score_arm(...)}``.
    :type scored: Mapping[str, Mapping[str, Any]]
    :return: Markdown table.
    :rtype: str
    """
    lines = ["| model / prompt / max_tokens | class | n | abstain (kinds) | Ar letters mean / median | He letters mean | Hebrew output | loop | gap markers mean | chars mean | length-cut |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        for c in UNCOVERED_CLASSES:
            s = scored[arm.tag]["by_class"][c]
            kinds = ", ".join(f"{k} {v}" for k, v in sorted(s["abstain_kinds"].items())) or "–"
            lines.append(f"| {arm_label(arm)} | {c} | {s['answered']} | {_pct(s['abstain'])} ({kinds}) | {_f(s['arabic_mean'], 0)} / {_f(s['arabic_median'], 0)} "
                         f"| {_f(s['hebrew_mean'], 0)} | {_pct(s['hebrew_output'])} | {_pct(s['loop'])} | {_f(s['gap_markers_mean'], 0)} | {_f(s['chars_mean'], 0)} | {_pct(s['length'])} |")
    return "\n".join(lines)


def panel_summary(panel: Sequence[Mapping[str, Any]]) -> str:
    """One line per class with image and document counts."""
    parts = []
    for c in CLASSES:
        rows = [p for p in panel if p["image_class"] == c]
        parts.append(f"{c} {len(rows)} images / {len({p['doc_id'] for p in rows})} docs")
    return "; ".join(parts)


def build_report(arms: Sequence[Arm], scored: Mapping[str, Mapping[str, Any]], panel: Sequence[Mapping[str, Any]], panel_path: Path) -> str:
    """The Markdown report.

    :param arms: Arms in table order.
    :type arms: Sequence[Arm]
    :param scored: ``{tag: score_arm(...)}``.
    :type scored: Mapping[str, Mapping[str, Any]]
    :param panel: Panel rows.
    :type panel: Sequence[Mapping[str, Any]]
    :param panel_path: Where the panel lives (named in the header).
    :type panel_path: Path
    :return: Report text.
    :rtype: str
    """
    complete = [a for a in arms if sum(1 for r in scored[a.tag]["images"] if r["answered"]) == len(panel)]
    partial = [f"{a.tag} ({sum(1 for r in scored[a.tag]['images'] if r['answered'])}/{len(panel)})" for a in arms if a not in complete]
    return "\n".join([
        "# Arabic-script prompt ablation on the fixed panel", "",
        f"Panel `{panel_path.name}`: {len(panel)} images — {panel_summary(panel)}. Covered documents are scored with the covered-only scorer "
        "(medians over documents); the other classes by behaviour (shares of answered images). ≥40 Ar = at least 40 Arabic letters emitted; "
        "abstain = the stopping marker, an empty answer or no script letter at all.", "",
        f"Arms with every panel image answered: {len(complete)}; partial: {', '.join(partial) if partial else 'none'}.", "",
        "## Table 1: prompt × model × image class", "", table_main(arms, scored), "",
        "## Table 2: how generations ended (finish_reason stop / length / other; mean completion tokens)", "", table_finish(arms, scored), "",
        "## Table 3: paired comparisons on the covered documents (bootstrap of documents)", "", paired_tables(arms, scored), "",
        "## Table 4: uncovered classes in detail", "", table_hebrew_detail(arms, scored), ""])


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Write rows to a CSV with the given columns.

    :param path: Output file.
    :type path: Path
    :param rows: Rows.
    :type rows: Sequence[Mapping[str, Any]]
    :param fields: Columns.
    :type fields: Sequence[str]
    """
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=sab.DEFAULT_BENCHMARK)
    parser.add_argument("--coverage", type=Path, default=None, help=f"default: <benchmark>/{sac.DEFAULT_COVERAGE}")
    parser.add_argument("--arms-dir", type=Path, default=None, help=f"default: <benchmark>/{DEFAULT_OUT_DIR}")
    parser.add_argument("--reference", nargs="*", default=[], help="NAME=PATH: an earlier full run scored on the panel images")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default: <benchmark>/{DEFAULT_SCORES_DIR}")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report (default: <out-dir>/report.md)")
    args = parser.parse_args(argv)
    panel = load_panel(args.panel)
    keys = [(p["doc_id"], int(p["image_index"])) for p in panel]
    benchmark = sab.load_benchmark(args.benchmark)
    coverage = sac.load_coverage(args.coverage or args.benchmark / sac.DEFAULT_COVERAGE)
    arms = discover_arms(args.arms_dir or args.benchmark / DEFAULT_OUT_DIR, keys)
    for ref in args.reference:
        name, _, path = ref.partition("=")
        arms.append(reference_arm(name, Path(path), keys))
    scored = {a.tag: score_arm(a, panel, benchmark, coverage) for a in arms}
    out_dir = args.out_dir or args.benchmark / DEFAULT_SCORES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "images.csv", [r for a in arms for r in scored[a.tag]["images"]], IMAGE_FIELDS)
    write_csv(out_dir / "documents.csv", [r for a in arms for r in scored[a.tag]["documents"]], DOC_FIELDS)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({a.tag: {"model": a.model, "prompt": a.prompt, "max_tokens": a.max_tokens, **scored[a.tag]["by_class"]} for a in arms},
                  fh, ensure_ascii=False, indent=1)
    report = build_report(arms, scored, panel, args.panel)
    (args.report or out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
