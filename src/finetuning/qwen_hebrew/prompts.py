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
