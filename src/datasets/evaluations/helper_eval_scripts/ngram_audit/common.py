"""Shared loaders and parameterised metric variants for the n-gram audit.

The paper scorer (``score_genizah_offline.py``) hard-codes n=5, a Hebrew-only
letter set, unclipped n-gram counting and four behaviour thresholds.  This
module exposes each of those as a parameter of :class:`ScoringConfig` so the
audit analyses can sweep them without touching the paper's reproduction path.

Expensive per-(fragment, system) features that do not depend on the config
(normalised texts, aligned P/R/F1, CER, WER) are computed once by
:func:`load_features`; the cheap n-gram / classification / tier step is
recomputed per config by :func:`score`.
"""

import collections
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO))

from src.datasets.evaluations.metrics import (  # noqa: E402
    cer_pair,
    genizah_visible_ink_gt,
    normalize_ink_hypothesis,
    wer_pair,
)
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import (  # noqa: E402
    ABSTAIN_MIN_CHARS,
    HALLUCINATION_NGRAM,
    LOOP_REPEAT_RATIO,
    NGRAM_N,
    TIER_A_OVERLAP,
    _ORDER_BLIND,
    _REFUSAL_RE,
    _VLM_EVIDENCE,
    aligned_prf,
    asserted_letters,
    load_benchmark,
)

BENCH_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
OUTPUTS = _REPO / "src/datasets/evaluations/transcription_raw_outputs"
JUDGE_JSON = _REPO / "src/datasets/evaluations/talmud_results/genizah_error_analysis.json"
PAPER_SUMMARY = (_REPO / "src/datasets/evaluations/transcription_results/paper_table"
                 / "genizah_offline_summary.csv")

# Systems that appear in the paper's Genizah tables.
PAPER_SYSTEMS = [
    "kraken_seg", "kraken_raw", "gemini_pro", "gemini_flash", "claude_opus_4_8",
    "claude_sonnet_5", "gpt_5_6_sol", "qwen3_vl_8b_heb_v17_step800",
    "qwen3_vl_8b_heb_v16_step1100", "qwen3_vl_8b", "vision_ocr_seg",
]
FRONTIER_VLMS = ["gemini_pro", "gemini_flash", "claude_opus_4_8",
                 "claude_sonnet_5", "gpt_5_6_sol"]
HEBVL17 = "qwen3_vl_8b_heb_v17_step800"

# Letter sets.  ``hebrew`` is the paper's (audit_genizah_benchmark._HEB);
# ``semitic`` is the union used by the cleaning module (SEMITIC_RE).
LETTER_RES = {
    "hebrew": re.compile(r"[֐-׿]+"),
    "semitic": re.compile(r"[֐-׿؀-ۿ]+"),
}
_MARKER_RE = re.compile(r"\[\.\.\.\]|\(!\)|\[\?\]")


def letters(text: str, letter_set: str = "hebrew") -> str:
    """Reduce text to a bare letter string under the chosen letter set.

    Mirrors ``audit_genizah_benchmark.letters_only`` (marker strip, then
    concatenate every run matched by the block regex).

    :param text: Any transcription text.
    :type text: str
    :param letter_set: ``hebrew`` or ``semitic``.
    :type letter_set: str
    :return: Concatenated letters.
    :rtype: str
    """
    text = _MARKER_RE.sub(" ", text)
    return "".join(LETTER_RES[letter_set].findall(text))


def ngrams(s: str, n: int) -> List[str]:
    """All overlapping character n-grams of ``s``.

    :param s: Letter string.
    :type s: str
    :param n: Window size.
    :type n: int
    :return: List of windows (empty when ``len(s) < n``).
    :rtype: list
    """
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def ngram_precision(hyp_letters: str, gt_letters: str, n: int = NGRAM_N,
                    clip: bool = False) -> float:
    """Share of hypothesis n-grams found in the reference.

    With ``clip=False`` this is byte-for-byte the paper metric
    (``score_genizah_offline.ngram_precision``): membership in the SET of
    reference windows.  With ``clip=True`` it is BLEU-style modified
    precision: a reference window may be matched at most as many times as it
    occurs in the reference, so text copied repeatedly from elsewhere on the
    same page stops earning credit once the reference's occurrences are used.

    :param hyp_letters: Hypothesis reduced to letters.
    :type hyp_letters: str
    :param gt_letters: Reference reduced to letters.
    :type gt_letters: str
    :param n: Window size.
    :type n: int
    :param clip: Apply count clipping.
    :type clip: bool
    :return: Precision in [0, 1]; 0.0 when either side is shorter than ``n``.
    :rtype: float
    """
    if len(hyp_letters) < n or len(gt_letters) < n:
        return 0.0
    hyp_grams = ngrams(hyp_letters, n)
    if not clip:
        gt_set = set(ngrams(gt_letters, n))
        return sum(g in gt_set for g in hyp_grams) / len(hyp_grams)
    gt_counts = collections.Counter(ngrams(gt_letters, n))
    hyp_counts = collections.Counter(hyp_grams)
    matched = sum(min(c, gt_counts.get(g, 0)) for g, c in hyp_counts.items())
    return matched / len(hyp_grams)


def loop_ratio(letters_str: str, span: int = 12) -> float:
    """Share of the text covered by its single most repeated ``span``-gram.

    Identical to the paper's ``loop_ratio``.

    :param letters_str: Hypothesis reduced to asserted letters.
    :type letters_str: str
    :param span: Repeated-unit length.
    :type span: int
    :return: Ratio in [0, 1].
    :rtype: float
    """
    if len(letters_str) < span * 3:
        return 0.0
    counts = collections.Counter(ngrams(letters_str, span))
    _top, freq = counts.most_common(1)[0]
    return (freq * span) / len(letters_str)


@dataclasses.dataclass(frozen=True)
class ScoringConfig:
    """One point in the metric design space.

    :param n: Character n-gram size.
    :param clip: BLEU-style count clipping.
    :param letter_set: ``hebrew`` (paper) or ``semitic`` (Hebrew+Arabic union).
    :param halluc_cutoff: n-gram precision below this = hallucinated.
    :param tier_cutoff: per-system precision needed for Tier A convergence.
    :param loop_cutoff: loop ratio above this = loop collapse.
    :param loop_span: repeated-unit length for loop detection.
    :param abstain_chars: asserted letters below this = abstained.
    """
    n: int = NGRAM_N
    clip: bool = False
    letter_set: str = "hebrew"
    halluc_cutoff: float = HALLUCINATION_NGRAM
    tier_cutoff: float = TIER_A_OVERLAP
    loop_cutoff: float = LOOP_REPEAT_RATIO
    loop_span: int = 12
    abstain_chars: int = ABSTAIN_MIN_CHARS

    def label(self) -> str:
        """Compact human-readable tag for tables.

        :return: e.g. ``n5 unclipped hebrew h0.10 t0.25 l0.45 a25``.
        :rtype: str
        """
        return (f"n{self.n} {'clipped' if self.clip else 'unclipped'} "
                f"{self.letter_set} h{self.halluc_cutoff:.2f} t{self.tier_cutoff:.2f} "
                f"l{self.loop_cutoff:.2f} a{self.abstain_chars}")


PAPER_CONFIG = ScoringConfig()


@dataclasses.dataclass
class Fragment:
    """One benchmark fragment with its config-independent reference forms."""
    doc_id: str
    gt_raw: str
    gt_ink: str
    gt_letters: Dict[str, str]          # letter_set -> letters
    script_bucket: str
    repaired: bool


@dataclasses.dataclass
class Output:
    """One (fragment, system) output with config-independent features."""
    doc_id: str
    model: str
    raw: str
    hyp: str
    hyp_letters: Dict[str, str]         # letter_set -> letters
    asserted: str
    aligned_precision: float
    aligned_recall: float
    aligned_f1: float
    cer: float
    cer_lenient: float
    wer: float


def load_fragments(which: str = "verified") -> List[Fragment]:
    """Load the benchmark exactly as the paper scorer does (GT repaired,
    fragments with < 50 Hebrew letters dropped, only fragments with outputs).

    :param which: ``verified`` (131) or ``frozen`` (150).
    :type which: str
    :return: Fragments in benchmark order.
    :rtype: list
    """
    tags_path = BENCH_DIR / "genizah_test_v1_script_tags.json"
    tags = {}
    if tags_path.exists():
        tags = {k: v.get("bucket", "untagged") for k, v in json.load(open(tags_path)).items()}
    frags = []
    for d in load_benchmark(which):
        if not (OUTPUTS / d["doc_id"]).is_dir():
            continue
        gt_ink = genizah_visible_ink_gt(d["gt"])
        gl = {ls: letters(gt_ink, ls) for ls in LETTER_RES}
        if len(gl["hebrew"]) < 50:
            continue
        frags.append(Fragment(d["doc_id"], d["gt"], gt_ink, gl,
                              tags.get(d["doc_id"], "untagged"),
                              bool(d.get("_repaired_duplication"))))
    return frags


def load_features(frags: Iterable[Fragment], systems: Optional[Iterable[str]] = None,
                  with_alignment: bool = True) -> List[Output]:
    """Read every saved output and compute config-independent features.

    :param frags: Fragments from :func:`load_fragments`.
    :type frags: Iterable[Fragment]
    :param systems: Restrict to these model stems (default: every non-flat,
        non-GT, non-consensus file, as the paper scorer does).
    :type systems: Iterable[str] or None
    :param with_alignment: Compute difflib alignment and CER/WER (slow-ish).
    :type with_alignment: bool
    :return: One :class:`Output` per (fragment, system).
    :rtype: list
    """
    wanted = set(systems) if systems is not None else None
    outs = []
    for fr in frags:
        for f in sorted((OUTPUTS / fr.doc_id).glob("*.txt")):
            model = f.stem
            if model in ("ground_truth", "consensus") or model.endswith("_flat"):
                continue
            if wanted is not None and model not in wanted:
                continue
            raw = f.read_text(errors="replace")
            hyp = normalize_ink_hypothesis(raw)
            hl = {ls: letters(hyp, ls) for ls in LETTER_RES}
            if with_alignment:
                p, r, f1 = aligned_prf(hyp, fr.gt_ink)
                cer_s, cer_l = cer_pair(hyp, fr.gt_ink)
                wer_s, _ = wer_pair(hyp, fr.gt_ink)
            else:
                p = r = f1 = cer_s = cer_l = wer_s = float("nan")
            outs.append(Output(fr.doc_id, model, raw, hyp, hl, asserted_letters(hyp),
                               p, r, f1, cer_s, cer_l, wer_s))
    return outs


def classify(raw: str, asserted: str, ngram_p: float, cfg: ScoringConfig) -> str:
    """Paper's ``classify`` with every threshold taken from ``cfg``.

    :param raw: Raw saved output.
    :type raw: str
    :param asserted: Script-agnostic asserted letters.
    :type asserted: str
    :param ngram_p: n-gram precision under ``cfg``.
    :type ngram_p: float
    :param cfg: Thresholds.
    :type cfg: ScoringConfig
    :return: abstained / loop_collapse / hallucinated / substantive.
    :rtype: str
    """
    if len(asserted) < cfg.abstain_chars or _REFUSAL_RE.search(raw[:400]):
        return "abstained"
    if loop_ratio(asserted, cfg.loop_span) > cfg.loop_cutoff:
        return "loop_collapse"
    if ngram_p < cfg.halluc_cutoff:
        return "hallucinated"
    return "substantive"


def score(frags: List[Fragment], outs: List[Output], cfg: ScoringConfig) -> List[dict]:
    """Score every output under ``cfg`` and assign tiers.

    :param frags: Fragments.
    :type frags: list
    :param outs: Outputs with precomputed features.
    :type outs: list
    :param cfg: Metric configuration.
    :type cfg: ScoringConfig
    :return: Long-format rows (one per output) with ``failure_mode``,
        ``ngram_precision``, ``ngram_precision_unclipped`` (always the
        set-membership value at the same n / letter set), ``tier`` and the
        config-independent metrics.
    :rtype: list
    """
    by_doc = {fr.doc_id: fr for fr in frags}
    rows_by_doc = collections.defaultdict(dict)
    for o in outs:
        fr = by_doc[o.doc_id]
        gt_l = fr.gt_letters[cfg.letter_set]
        hyp_l = o.hyp_letters[cfg.letter_set]
        p_unclipped = ngram_precision(hyp_l, gt_l, cfg.n, clip=False)
        p = ngram_precision(hyp_l, gt_l, cfg.n, clip=True) if cfg.clip else p_unclipped
        mode = classify(o.raw, o.asserted, p, cfg)
        rows_by_doc[o.doc_id][o.model] = dict(
            fragment_id=o.doc_id, model=o.model, failure_mode=mode,
            ngram_precision=p, ngram_precision_unclipped=p_unclipped,
            aligned_precision=o.aligned_precision, aligned_recall=o.aligned_recall,
            aligned_f1=o.aligned_f1, cer=o.cer, cer_lenient=o.cer_lenient, wer=o.wer,
            gt_letters=len(gt_l), hyp_letters=len(hyp_l),
            script_bucket=fr.script_bucket, repaired_gt=fr.repaired,
        )
    rows = []
    for doc_id, per_model in rows_by_doc.items():
        kraken_ok = per_model.get(_ORDER_BLIND, {}).get("ngram_precision", 0) >= cfg.tier_cutoff
        vlm_ok = any(per_model.get(m, {}).get("ngram_precision", 0) >= cfg.tier_cutoff
                     for m in _VLM_EVIDENCE)
        tier = "A" if (kraken_ok and vlm_ok) else "B"
        for r in per_model.values():
            r["tier"] = tier
            rows.append(r)
    return rows


def summarise(rows: List[dict], systems: Optional[Iterable[str]] = None) -> Dict[str, dict]:
    """Per-system behaviour distribution and medians (paper summary shape).

    :param rows: Output of :func:`score`.
    :type rows: list
    :param systems: Restrict / order the systems.
    :type systems: Iterable[str] or None
    :return: model -> summary dict.
    :rtype: dict
    """
    import statistics
    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    names = list(systems) if systems is not None else sorted(by_model)
    out = {}
    for m in names:
        rs = by_model.get(m, [])
        if not rs:
            continue
        modes = collections.Counter(r["failure_mode"] for r in rs)
        subst = [r for r in rs if r["failure_mode"] == "substantive"]
        out[m] = dict(
            model=m, n=len(rs),
            pct_abstained=modes["abstained"] / len(rs),
            pct_loop_collapse=modes["loop_collapse"] / len(rs),
            pct_hallucinated=modes["hallucinated"] / len(rs),
            pct_substantive=modes["substantive"] / len(rs),
            ngram_precision_median=statistics.median(r["ngram_precision"] for r in rs),
            aligned_f1_median=statistics.median(r["aligned_f1"] for r in rs),
            cer_median_substantive=(statistics.median(r["cer"] for r in subst)
                                    if subst else None),
            n_substantive=len(subst),
        )
    return out


def tier_a_count(rows: List[dict]) -> int:
    """Number of Tier A fragments in a scored row set.

    :param rows: Output of :func:`score`.
    :type rows: list
    :return: Count.
    :rtype: int
    """
    return len({r["fragment_id"] for r in rows if r["tier"] == "A"})


def headline_claims(summary: Dict[str, dict]) -> Dict[str, bool]:
    """Evaluate the three paper claims the reviewers questioned.

    :param summary: Output of :func:`summarise` (must contain the paper systems).
    :type summary: dict
    :return: claim name -> holds?
    :rtype: dict
    """
    def sub(m: str) -> float:
        return summary.get(m, {}).get("pct_substantive", float("nan"))
    frontier = [sub(m) for m in FRONTIER_VLMS if m in summary]
    return {
        "kraken_ge_frontier_substantive": sub("kraken_seg") >= max(frontier),
        "hebvl17_gt_frontier_substantive": sub(HEBVL17) > max(frontier),
        "gemini_pro_best_frontier_substantive": sub("gemini_pro") >= max(frontier),
    }


def load_judge() -> Dict[tuple, dict]:
    """Load the Gemini Flash error-taxonomy judge output keyed by (doc, model).

    :return: (doc_id, model) -> judge record with ``errors`` list, or empty
        dict when the file is absent.
    :rtype: dict
    """
    if not JUDGE_JSON.exists():
        return {}
    return {(r["doc_id"], r["model"]): r for r in json.load(open(JUDGE_JSON))}


def cached_features(cache_path: Path, systems: Optional[Iterable[str]] = None,
                    which: str = "verified") -> tuple:
    """Load (fragments, outputs) from a pickle, computing and saving on a miss.

    :param cache_path: Pickle location (scratchpad, never the repo).
    :type cache_path: Path
    :param systems: Systems to include when computing (default PAPER_SYSTEMS).
    :type systems: Iterable[str] or None
    :param which: Benchmark variant.
    :type which: str
    :return: (list of Fragment, list of Output).
    :rtype: tuple
    """
    import pickle
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    frags = load_fragments(which)
    outs = load_features(frags, PAPER_SYSTEMS if systems is None else systems)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump((frags, outs), fh)
    return frags, outs
