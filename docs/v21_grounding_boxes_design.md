# v2.1 grounding: accurate line/word boxes (design, 2026-09-09)

## Problem (measured, not assumed)
v2.0a's grounded-page line boxes are a **layout prior, not a measurement**:
on the 24 KTIV probe pages only 58% of boxes sat on the line they transcribe,
median line IoU 0.40, template rate 0.54; on 87 of 115 pilot site pages ≥50%
of lines shared one x-range. Text quality is fine (median line CER 0.14). The
model satisfies the sequential grounded-page loss with "top margin + average
pitch", so the box does not depend on the ink. `locate` (phrase→box), which
cannot be guessed, already works (IoU ~0.48, 70-78% center hits) at just 8%
grounding share — so the capability exists; the *targets* let the model cheat.
Merger capacity is **not** the lever: v20b (merger LoRA) and v19c (full merger)
did not make boxes accurate (see project_qwen_finetune / project_roadmap).

## New row families (build_ktiv_dataset.py; geometry-mode pages only, 0-1000
## normalized to the original scan frame, same decontam gates as transcription)
- **locate_word** (4/page): unique Hebrew word (letters-unique on the page,
  ≥4 letters, longest preferred) → `{"bbox_2d":[...]}`. Word position is not
  predictable from a page template — the core signal.
- **read_box_word** (2/page): word box → the single word. Fine-grained
  region reading.
- **line_index** (2/page): "line N of the {right|left} column (from the
  top|bottom)" → `{"bbox_2d":[...], "text":...}`. Forces counting a specific
  line rather than emitting the average pitch. 1- and 2-column pages only.
- **line_of_phrase** (≤1/page): phrase → host-line box + text. Grounded
  upgrade of the old text-only find-line QA.
- **grounded_detect** (all pages 4-30 lines): whole page, JSON array with
  **bbox_2d BEFORE text** — localization precedes transcription. Distinct
  prompt from the v2.0 `grounded_page` (which stays text-first and is kept for
  eval comparability), so the two formats don't bleed.
- **grounded_crop** (1/page): a random per-column band of lines cropped to its
  own image, boxes re-normalized to the crop. A mid-column line now appears at
  a different y with a different margin, so the box can't be a page template.
  This is the ~9GB image adder (one extra JPEG per page).

## Mixture (notebook genizah_v21b_dispatchfix.ipynb — formerly genizah_v21.ipynb — from v20a)
Raise grounding share 8% → ~20%. Suggested within-grounding weights:
locate 0.16, locate_word 0.18, line_index 0.14, line_of_phrase 0.06,
read_box 0.08, read_box_word 0.08, grounded_detect 0.14, grounded_crop 0.10,
layout_qa 0.03, grounded_page(text-first) 0.03. Transcription/replay families
keep v2.0a's proportions, scaled to the remaining 80%. One variable vs v2.0a =
the grounding data (plus warm-start choice, below).

## Gate (all exist)
grounding_eval/box_quality.py: template_rate, box_on_right_line, vertical_iou,
width_ratio, drift — plus the grounding trio and the religious-140/PGP-131
transcription no-regression pair. New eval queries to add: word-locate and
line-index over the 24 probe pages (build from their word geometry).
Targets: locate IoU 0.65-0.75 / hit ≥85%; right-line ≥90%; template <0.15;
line IoU 0.65-0.75; word-locate 0.5-0.6.

## Open decisions (flagged for the user)
1. **Warm start v2.0a-1800 vs v2.0b-2000** — gated on the v20b final eval.
2. **Hub push of genizah_ktiv_v3** — external publish + ~size; build is
   verified locally, push on go.
3. **Shuffled-order grounded variant** — deliberately dropped for now: teaching
   shuffled output risks reading-order regressions; the crop variant already
   breaks the positional prior. Revisit if template rate stays high.
