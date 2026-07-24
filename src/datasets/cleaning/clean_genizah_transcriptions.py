# File name: clean_genizah_transcriptions.py
# Date: 7/22/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""Clean and triage FJP-scraped Genizah transcriptions for OCR training use.

The merged index mixes two very different transcription lineages:

* **Diplomatic transcriptions** (PGP lineage: Goitein, Gil, Elbaum, ...) —
  line-faithful text with a small editorial vocabulary: ``[...]`` lacunae,
  bracketed best-guess reconstructions, ``(?)`` uncertainty, ``{x}`` supplied
  (never-written) letters, ``//x//`` or ``\\x\\`` interlinear insertions.
* **Edition scrapes** (anonymous ``FJP N`` editors) — OCR of *printed critical
  editions*, including catalog headers, apparatus footnotes, parallel-witness
  reconstructions and Hebrew-font mojibake (final letters -> ``& # $``,
  aleph -> ``?``, flipped brackets like ``]4[``).

For image-grounded (OCR/HTR) training only visible ink is usable ground
truth, so cleaning is *strict*: editorial reconstructions become gap tokens,
supplied letters are dropped, apparatus and headers are dropped, and docs
whose only text is an edition scrape are triaged out for manual work.
"""
import argparse
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GAP = "␣gap␣"

HEBREW_RE = re.compile(r"[֐-׿]")
SEMITIC_RE = re.compile(r"[֐-׿؀-ۿ]")
EDITOR_HEADER_RE = re.compile(r"Editors?\s*:", re.I)
FOOTNOTE_RULE_RE = re.compile(r"_{10,}")
FLIPPED_BRACKET_RE = re.compile(r"\]\s*\d+\s*\[")
EDITION_REF_RE = re.compile(r"מהד\s*'|עמ\s*'?\s*\d+|ק\"ג |כרך|תרגום ד\"?ר|השלמתי")
NUMBERED_FOOTNOTE_RE = re.compile(r"^\s*\d{1,2}\s*\)|(?<=[.\s])\d{1,2}\)\s", re.M)
MOJIBAKE_FINAL_RE = re.compile(r"(?<=[֐-׿])[&#$]|[&#$](?=[֐-׿])")
DOTS_RUN_RE = re.compile(r"(?:[.·…]\s*){2,}")
UNCERTAIN_RE = re.compile(r"\(\?+\)|(?<=[֐-׿؀-ۿ])\?(?![֐-׿])")
SUPPLIED_RE = re.compile(r"\{[^{}\n]*\}")
INTERLINEAR_RE = re.compile(r"//([^/\n]{1,60})//|\\\\?([^\\\n]{1,60})\\\\?")
COLUMN_MARKER_RE = re.compile(r"טור_[א-ת]")
SIDE_MARKER_RE = re.compile(r"\b(recto|verso|margin)\b:?", re.I)


@dataclass
class SectionReport:
    """Cleaning outcome for a single transcription section.

    :param editor: Editor attribution as stored in the index.
    :param kind: ``diplomatic``, ``edition_scrape`` or ``junk_header``.
    :param kept: Whether the section survives cleaning/deduplication.
    :param drop_reason: Why the section was dropped (empty when kept).
    :param corruption: Mojibake flags detected (``aleph_qmark``, ``final_symbol``).
    :param cleaned_text: Strict-mode text (visible ink + gap tokens) when kept.
    :param recon_chars: Characters of editorial reconstruction replaced by gaps.
    :param visible_chars: Semitic-script characters remaining after cleaning.
    """

    editor: str
    kind: str
    kept: bool = False
    drop_reason: str = ""
    corruption: List[str] = field(default_factory=list)
    cleaned_text: str = ""
    recon_chars: int = 0
    visible_chars: int = 0


def classify_section(editor: str, text: str) -> str:
    """Classify a transcription section by lineage.

    :param editor: Editor attribution string.
    :param text: Raw section text.
    :returns: ``junk_header``, ``edition_scrape`` or ``diplomatic``.
    """
    if not SEMITIC_RE.search(text):
        return "junk_header"
    score = 0
    if re.match(r"^\s*FJP\s*\d+", editor or ""):
        score += 2
    if FOOTNOTE_RULE_RE.search(text):
        score += 2
    if FLIPPED_BRACKET_RE.search(text):
        score += 1
    if EDITION_REF_RE.search(text):
        score += 1
    if len(NUMBERED_FOOTNOTE_RE.findall(text)) >= 3:
        score += 2
    return "edition_scrape" if score >= 2 else "diplomatic"


def detect_corruption(text: str) -> List[str]:
    """Detect Hebrew-font mojibake introduced by edition scraping.

    :param text: Raw or cleaned section text.
    :returns: List of corruption flags (possibly empty).
    """
    flags = []
    letters = len(SEMITIC_RE.findall(text))
    if letters:
        alephs = text.count("א")
        qmarks = len(re.findall(r"(?<=[֐-׿])\?|\?(?=[֐-׿])", text))
        if qmarks >= 5 and alephs / letters < 0.02:
            flags.append("aleph_qmark")
        if len(MOJIBAKE_FINAL_RE.findall(text)) >= 3:
            flags.append("final_symbol")
    return flags


def _strip_letters(text: str) -> str:
    """Reduce text to Semitic letters with mojibake wildcards for fuzzy matching.

    :param text: Section text.
    :returns: Letters-only string; ``?``/``&``/``#``/``$`` map to ``*``.
    """
    out = []
    for ch in text:
        if SEMITIC_RE.match(ch):
            out.append(ch)
        elif ch in "?&#$":
            out.append("*")
    return "".join(out)


def near_duplicates(a: str, b: str, n: int = 4, threshold: float = 0.4) -> bool:
    """Decide whether two sections transcribe the same text.

    Uses character n-gram containment (shared n-grams over the smaller
    section's n-grams) so that partial rescrapes still match, computed on a
    mojibake-tolerant alphabet: alef/nun and wildcard symbols are removed
    because the known corruption modes destroy exactly those characters.

    :param a: Letters-only representation of the first section.
    :param b: Letters-only representation of the second section.
    :param n: Character n-gram size.
    :param threshold: Containment above which sections count as duplicates.
    :returns: ``True`` when one section largely reproduces the other.
    """
    a = re.sub(r"[אנן*]", "", a)
    b = re.sub(r"[אנן*]", "", b)
    grams_a = {a[i:i + n] for i in range(len(a) - n + 1)}
    grams_b = {b[i:i + n] for i in range(len(b) - n + 1)}
    if not grams_a or not grams_b:
        return False
    return len(grams_a & grams_b) / min(len(grams_a), len(grams_b)) > threshold


def clean_diplomatic(text: str) -> Tuple[str, int, int]:
    """Strict-clean a diplomatic transcription to visible-ink ground truth.

    Editorial reconstructions (bracketed letters, dotted lacunae, edge
    brackets) become :data:`GAP` tokens; supplied-letter braces are removed;
    interlinear-insertion markers are unwrapped (the ink is on the page);
    uncertainty markers and layout noise are stripped.

    :param text: Raw diplomatic section text.
    :returns: Tuple of (cleaned text, reconstruction chars replaced,
        visible Semitic chars kept).
    """
    text = unicodedata.normalize("NFC", text).replace("\xa0", " ")
    recon_chars = 0
    lines_out = []
    for line in text.splitlines():
        line = SIDE_MARKER_RE.sub(" ", line)
        line = COLUMN_MARKER_RE.sub(" ", line)
        line = SUPPLIED_RE.sub("", line)
        line = INTERLINEAR_RE.sub(lambda m: m.group(1) or m.group(2) or "", line)

        def _bracket(m: "re.Match[str]") -> str:
            nonlocal recon_chars
            inner = m.group(1)
            # Short spans are editorial reconstructions of damaged ink. Long
            # spans mean two independent edge-damage markers that happened to
            # pair up in flat (newline-less) text: keep the real ink between
            # them and record damage at both edges.
            if len(inner) <= 40:
                recon_chars += len(SEMITIC_RE.findall(inner))
                return f" {GAP} "
            return f" {GAP} {inner} {GAP} "

        line = re.sub(r"\[([^\[\]\n]*)\]", _bracket, line)
        # Unpaired edge brackets mark text lost at a damaged line edge.
        stripped = line.strip()
        if stripped.startswith("]"):
            head, line = line.split("]", 1)
            recon_chars += len(SEMITIC_RE.findall(head))
            line = f" {GAP} " + line
        if stripped.endswith("["):
            line, tail = line.rsplit("[", 1)
            recon_chars += len(SEMITIC_RE.findall(tail))
            line = line + f" {GAP} "
        # Any remaining stray brackets are edition debris.
        line = line.replace("[", " ").replace("]", " ")
        line = UNCERTAIN_RE.sub("", line)
        line = DOTS_RUN_RE.sub(f" {GAP} ", line)
        line = re.sub(rf"(?:{re.escape(GAP)}\s*)+", f"{GAP} ", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line and line != GAP:
            lines_out.append(line)
    cleaned = "\n".join(lines_out)
    return cleaned, recon_chars, len(SEMITIC_RE.findall(cleaned))


def process_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Clean all transcription sections of one document and triage it.

    Triage classes:

    * ``A`` — one image, one clean kept section, low reconstruction share.
    * ``B`` — clean text but image/section alignment work remains.
    * ``C`` — only corrupted text survives (needs repair or manual pass).
    * ``D`` — nothing usable (edition scrapes / headers only).

    :param doc: Raw merged-index document (``transcriptions``, ``image_urls``...).
    :returns: Dict with ``doc_id``, ``triage``, ``sections`` reports and
        ``cleaned_text`` for kept sections.
    """
    sections: List[SectionReport] = []
    for t in doc.get("transcriptions") or []:
        editor = (t.get("editor") or "").strip()
        raw = t.get("text") or ""
        kind = classify_section(editor, raw)
        rep = SectionReport(editor=editor, kind=kind)
        if kind == "junk_header":
            rep.drop_reason = "no transcription content"
        elif kind == "edition_scrape":
            rep.drop_reason = "printed-edition scrape (apparatus/footnotes/mojibake)"
        else:
            rep.corruption = detect_corruption(raw)
            rep.cleaned_text, rep.recon_chars, rep.visible_chars = clean_diplomatic(raw)
            rep.kept = rep.visible_chars >= 40
            if not rep.kept:
                rep.drop_reason = "too little visible text after cleaning"
        sections.append(rep)

    # Deduplicate: prefer uncorrupted twins of the same text.
    kept = [s for s in sections if s.kept]
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            if not (a.kept and b.kept):
                continue
            if near_duplicates(_strip_letters(a.cleaned_text), _strip_letters(b.cleaned_text)):
                key = lambda s: (len(s.corruption), -s.visible_chars)
                worse = max(a, b, key=key)
                worse.kept = False
                worse.drop_reason = "duplicate of cleaner section"

    kept = [s for s in sections if s.kept]
    n_images = len(doc.get("image_urls") or [])
    visible = sum(s.visible_chars for s in kept)
    recon = sum(s.recon_chars for s in kept)
    recon_frac = recon / max(recon + visible, 1)

    if not kept:
        triage, reason = "D", "no usable diplomatic transcription"
    elif all(s.corruption for s in kept):
        triage, reason = "C", "only corrupted text available: " + ",".join(kept[0].corruption)
    elif n_images == 1 and len(kept) == 1 and recon_frac < 0.25:
        triage, reason = "A", "single image, single clean section"
    else:
        parts = []
        if n_images != 1:
            parts.append(f"{n_images} images")
        if len(kept) != 1:
            parts.append(f"{len(kept)} kept sections")
        if recon_frac >= 0.25:
            parts.append(f"reconstruction {recon_frac:.0%}")
        triage, reason = "B", "alignment/review needed: " + ", ".join(parts)

    return {
        "doc_id": doc["doc_id"],
        "shelf_mark": doc.get("shelf_mark"),
        "triage": triage,
        "triage_reason": reason,
        "n_images": n_images,
        "recon_frac": round(recon_frac, 4),
        "visible_chars": visible,
        "cleaned_text": "\n\n".join(s.cleaned_text for s in kept),
        "sections": [s.__dict__ for s in sections],
    }


def is_training_clean(result: Dict[str, Any], max_recon_frac: float = 0.10,
                      min_visible_chars: int = 150) -> Tuple[bool, str]:
    """Conservative gate: is a cleaned document safe to fine-tune on?

    Stricter than triage ``A``: rejects any residual mixed signal (editorial
    reconstruction share, Latin debris, leftover markers, short texts) so the
    training set contains only high-confidence visible-ink ground truth.

    :param result: One output record of :func:`process_document`.
    :param max_recon_frac: Maximum tolerated share of reconstructed chars.
    :param min_visible_chars: Minimum visible Semitic characters.
    :returns: Tuple of (verdict, reason for rejection — empty when clean).
    """
    if result["triage"] != "A":
        return False, f"triage {result['triage']}: {result['triage_reason']}"
    if result["recon_frac"] > max_recon_frac:
        return False, f"reconstruction share {result['recon_frac']:.0%}"
    if result["visible_chars"] < min_visible_chars:
        return False, f"only {result['visible_chars']} visible chars"
    text = result["cleaned_text"].replace(GAP, " ")
    kept = [s for s in result["sections"] if s["kept"]]
    if any(s["corruption"] for s in kept):
        return False, "mojibake in kept section"
    latin_words = re.findall(r"[A-Za-z]{3,}", text)
    if latin_words:
        return False, f"latin debris: {latin_words[:3]}"
    if re.search(r"[&#$_{}<>\\/]|�", text):
        return False, "residual editorial/mojibake symbols"
    gap_count = result["cleaned_text"].count(GAP)
    if gap_count > max(6, len(text) // 150):
        return False, f"too fragmentary: {gap_count} gaps"
    return True, ""


def main(in_path: str, out_path: str) -> Dict[str, int]:
    """Run cleaning over a JSONL export and write per-doc results.

    :param in_path: Input JSONL of merged-index documents.
    :param out_path: Output JSONL of cleaning/triage results.
    :returns: Counter of triage classes.
    """
    counts: Dict[str, int] = {}
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            result = process_document(json.loads(line))
            counts[result["triage"]] = counts.get(result["triage"], 0) + 1
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="merged-index JSONL export")
    parser.add_argument("--output", required=True, help="cleaned/triaged JSONL")
    args = parser.parse_args()
    print(main(args.input, args.output))
