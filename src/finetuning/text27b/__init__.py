# File name: __init__.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""text27b: the multi-task text fine-tune dataset for Qwen3.8-27B (build ``v0_2``).

Core modules (phase 0; every other module imports them):

* :mod:`~src.finetuning.text27b.paths` -- ``load_paths()`` → :class:`Paths` from the git-ignored
  ``logs/text_finetune_27b/paths.json``; no absolute machine paths in committed code.
* :mod:`~src.finetuning.text27b.rows` -- row schema and constants (families, conditions, splits),
  prompts, JSONL I/O, exact token counts and the chat-template render contract
  (``enable_thinking=False``: one empty think block, no reasoning-effort sentence), ``row_id``,
  ``fingerprint``.
* :mod:`~src.finetuning.text27b.textnorm` -- CER normalisation, read cleaning, Latin-line strip,
  Arabic→Judaeo-Arabic transliteration, target strips (line markers, citations, project-name strings),
  the grounding test.
* :mod:`~src.finetuning.text27b.shelfmarks` -- shelfmark normalisation keys
  (``key`` / ``keys`` / ``canonical_to_key``), corpus matching (``match_shelfmark``) and the
  located-in-text rule (``locate``).
* :mod:`~src.finetuning.text27b.pages` -- literature page iteration (one structuring per page,
  OCR text joined by file stem, shingle containment) and page-text analysis helpers.

Submodules are imported explicitly (``from src.finetuning.text27b import rows``); importing the
package itself loads nothing heavy.
"""

__all__ = ["paths", "rows", "textnorm", "shelfmarks", "pages"]
