"""
talmud_gt_parser.py
Parse Talmud ground truth files with --- Section --- delimiters.

Expected format:
    --- גמרא ---
    [gemara text]
    --- רשי ---
    [rashi text]
    --- תוספות ---
    [tosafot text]
"""

import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

# Canonical section keys used throughout the eval pipeline
SECTION_KEYS: Tuple[str, ...] = ("gemara", "rashi", "tosafot")

# Hebrew names as they appear in the delimiter lines → canonical keys
SECTION_MARKERS: Dict[str, str] = {
    "גמרא": "gemara",
    "רשי": "rashi",    # file uses רשי (no geresh), not רש"י
    'רש"י': "rashi",   # handle the geresh form too
    "תוספות": "tosafot",
}

# Nikud (vowel-point) Unicode block: U+05B0–U+05C7
_NIKUD_RE = re.compile(r"[\u05B0-\u05C7\u05F0-\u05F4\uFB1D-\uFB4F]")


def strip_nikud(text: str) -> str:
    """Remove Hebrew vowel points (nikud) and cantillation marks for lenient CER."""
    return _NIKUD_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace sequences to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def parse_talmud_gt(text: str) -> Dict[str, str]:
    """Parse raw ground truth text into a dict keyed by section name.

    Returns dict with keys 'gemara', 'rashi', 'tosafot' (any may be absent
    if the corresponding section is not found in the file).
    """
    sections: Dict[str, str] = {}
    delimiter_re = re.compile(r"---\s*(.*?)\s*---", re.MULTILINE)
    matches = list(delimiter_re.finditer(text))

    for i, match in enumerate(matches):
        header = match.group(1).strip()

        # Resolve header to canonical section key
        section_key: Optional[str] = None
        for marker, key in SECTION_MARKERS.items():
            if marker in header:
                section_key = key
                break

        if section_key is None:
            continue  # Unknown section header — skip

        # Body is everything between this delimiter and the next (or EOF)
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        # If same section appears twice (e.g., two Rashi blocks), concatenate
        if section_key in sections:
            sections[section_key] += "\n" + body
        else:
            sections[section_key] = body

    return sections


def load_gt_file(path: Path) -> Dict[str, str]:
    """Load and parse a single ground truth .txt file."""
    return parse_talmud_gt(path.read_text(encoding="utf-8"))


def load_gt_directory(gt_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load all .txt GT files from a directory.

    Returns {stem → section_dict}, e.g. {"01_2": {"gemara": ..., "rashi": ..., "tosafot": ...}}
    """
    result: Dict[str, Dict[str, str]] = {}
    for txt_file in sorted(gt_dir.glob("*.txt")):
        sections = load_gt_file(txt_file)
        if sections:
            result[txt_file.stem] = sections
    return result


def get_section_summary(sections: Dict[str, str]) -> str:
    """Quick diagnostic summary of what was parsed."""
    lines = []
    for key in ("gemara", "rashi", "tosafot"):
        if key in sections:
            char_count = len(sections[key])
            word_count = len(sections[key].split())
            lines.append(f"  {key:10s}: {char_count:5d} chars, {word_count:4d} words")
        else:
            lines.append(f"  {key:10s}: (not found)")
    return "\n".join(lines)