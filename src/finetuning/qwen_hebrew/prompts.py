"""Training prompts for the Qwen3-VL Hebrew fine-tuning pipeline.

Two prompt families:

- ``CROP_TRANSCRIBE_PROMPTS`` — Task A: transcribe a single *section crop*
  (gemara / rashi / tosafot region image). New prompts, mirroring the eval
  prompts' instructions minus the full-page layout framing (which would be
  wrong for a crop).
- ``SECTION_EXTRACT_PROMPTS`` — Task B: extract one section from a *full page*
  image. This is a byte-exact copy of
  ``src.datasets.evaluations.talmud_evaluation.SECTION_EXTRACT_PROMPTS``.
  It is copied rather than imported because ``talmud_evaluation`` imports
  heavy eval-only dependencies (wandb, kraken, google-cloud-vision) that must
  not become training-environment requirements.
  ``tests/test_qwen_hebrew_prompts.py`` asserts the copies stay byte-equal.
- Task C: whole-page Genizah fragment transcription. ``FRAGMENT_TRANSCRIBE_PROMPT``
  (asserts Hebrew script) and ``ARABIC_FRAGMENT_PROMPT`` (asserts Arabic script)
  are the prompts the training rows were built with; they are kept for building
  rows and for reproducing past evaluations. ``SCRIPT_NEUTRAL_PROMPT`` is the same
  prompt without the script assertion and is the one prompt the production reader
  (``src.models.ocr.page_reader``) sends for every page, whatever its script: the
  prompt ablation of 2026-10-25 showed that asserting the script makes the reader
  write Arabic on Hebrew-script sides, at no gain on Arabic pages.
"""

from typing import Dict

SECTIONS = ("gemara", "rashi", "tosafot")

# ── Task A: section-crop transcription ────────────────────────────────────────

CROP_TRANSCRIBE_PROMPTS: Dict[str, str] = {
    "gemara": """This image is the Gemara section cropped from a Babylonian Talmud page \
(Vilna edition) — large square Hebrew/Aramaic script.

Transcribe the text exactly as written. Mark unclear characters with [?].
Preserve nikud and line breaks. Do NOT correct or complete from memory.

Return ONLY the transcription.""",

    "rashi": """This image is the Rashi commentary section cropped from a Babylonian Talmud \
page (Vilna edition) — smaller semi-cursive Rashi script.

Transcribe the text exactly as written. Mark unclear characters with [?].
Preserve nikud and line breaks. Do NOT correct or complete from memory.

Return ONLY the transcription.""",

    "tosafot": """This image is the Tosafot commentary section cropped from a Babylonian \
Talmud page (Vilna edition) — small square Hebrew script.

Transcribe the text exactly as written. Mark unclear characters with [?].
Preserve nikud and line breaks. Do NOT correct or complete from memory.

Return ONLY the transcription.""",
}

# ── Task B: full-page section extraction ──────────────────────────────────────
# Byte-exact copy of talmud_evaluation.py — do not edit here without editing
# the original (the eval prompt wording is empirically load-bearing).

_LAYOUT = """This is a full page from the Babylonian Talmud (Vilna edition).
The page has three spatially distinct sections:
  • GEMARA (גמרא)    — centre column, large square Hebrew/Aramaic script
  • RASHI (רש"י)     — inner margin, smaller semi-cursive Rashi script
  • TOSAFOT (תוספות) — outer margin, smaller square script"""

SECTION_EXTRACT_PROMPTS: Dict[str, str] = {
    "gemara": f"""{_LAYOUT}

Extract ONLY the main Gemara text from the centre column.
Do not include Rashi or Tosafot commentary.
Transcribe exactly as written. Mark unclear characters with [?].
Preserve nikud and line breaks. Do NOT correct or complete from memory.

Return ONLY the Gemara transcription.""",

    "rashi": f"""{_LAYOUT}

Extract ONLY the Rashi commentary from the inner margin (semi-cursive Rashi script).
Do not include the Gemara or Tosafot.
Transcribe exactly as written. Mark unclear characters with [?].
Preserve nikud and line breaks. Do NOT correct or complete from memory.

Return ONLY the Rashi transcription.""",

    "tosafot": f"""{_LAYOUT}

Extract ONLY the Tosafot commentary from the outer margin (small square script).
Do not include the Gemara or Rashi.
Transcribe exactly as written. Mark unclear characters with [?].
Preserve nikud and line breaks. Do NOT correct or complete from memory.

Return ONLY the Tosafot transcription.""",
}


# ── Task C: Genizah fragment transcription ────────────────────────────────────
# Targets come from src.datasets.cleaning.clean_genizah_transcriptions: flat
# reading-order text (line breaks were lost upstream), damage gaps rendered as
# "[...]". The prompt therefore asks for reading order, not line preservation.

FRAGMENT_TRANSCRIBE_PROMPT = """This image is a manuscript fragment from the Cairo Genizah — \
handwritten Hebrew script (the language may be Hebrew, Judeo-Arabic, or Aramaic).

Transcribe the text exactly as written, in reading order. Mark unclear characters with [?].
Where text is lost or illegible due to damage, write [...].
Do NOT correct, restore, or complete from memory.

Return ONLY the transcription."""

# The script clause of the fragment prompt. The two asserting prompts differ from each other, and from
# the script-neutral one, in this clause alone; the instructions after it are shared byte for byte.
HEBREW_SCRIPT_CLAUSE = "handwritten Hebrew script (the language may be Hebrew, Judeo-Arabic, or Aramaic)"
ARABIC_SCRIPT_CLAUSE = "handwritten Arabic script"
SCRIPT_NEUTRAL_CLAUSE = "handwritten text in its original script"

# The Arabic-script training rows' prompt (``build_pgp_arabic_editions`` and the book / external
# edition builders): kept for building rows and for reproducing the Arabic-script benchmark runs.
ARABIC_FRAGMENT_PROMPT = FRAGMENT_TRANSCRIBE_PROMPT.replace(HEBREW_SCRIPT_CLAUSE, ARABIC_SCRIPT_CLAUSE)

# The production reader's prompt, for every page of every script (never asserts the script).
SCRIPT_NEUTRAL_PROMPT = """This image is a manuscript fragment from the Cairo Genizah — \
handwritten text in its original script.

Transcribe the text exactly as written, in reading order. Mark unclear characters with [?].
Where text is lost or illegible due to damage, write [...].
Do NOT correct, restore, or complete from memory.

Return ONLY the transcription."""

assert HEBREW_SCRIPT_CLAUSE in FRAGMENT_TRANSCRIBE_PROMPT, "the fragment prompt no longer holds its script clause"
assert SCRIPT_NEUTRAL_PROMPT == FRAGMENT_TRANSCRIBE_PROMPT.replace(HEBREW_SCRIPT_CLAUSE, SCRIPT_NEUTRAL_CLAUSE), \
    "the script-neutral prompt drifted from the fragment prompt in more than the script clause"

# The script-asserting prompts by the script they assert: only for reproducing past evaluations.
LEGACY_SCRIPT_PROMPTS: Dict[str, str] = {"hebrew": FRAGMENT_TRANSCRIBE_PROMPT, "arabic": ARABIC_FRAGMENT_PROMPT}


def prompt_for(task: str, section: str) -> str:
    """Return the training prompt for a task/section pair.

    :param task: Either ``crop_transcribe`` (Task A) or ``page_extract`` (Task B).
    :param section: One of ``gemara``, ``rashi``, ``tosafot``.
    :returns: The prompt string.
    :raises KeyError: If the task or section is unknown.
    """
    prompts = {
        "crop_transcribe": CROP_TRANSCRIBE_PROMPTS,
        "page_extract": SECTION_EXTRACT_PROMPTS,
    }[task]
    return prompts[section]
