# Prompt: two-reader line confirmation — probe results and what to build

Paste this into the session that is building the site features. It supersedes
the realtime-first framing of `website_grounding_playground_prompt.md`: the
plan is now **offline batch, surfaced on selected documents**, with a
per-line "confirmed by a second reader" badge. Numbers below were measured on
2026-09-08 in the sibling repo `historical-document-analysis`.

## What was tested

24 Hebrew manuscript pages (Talmud fragments with KTIV editorial line
transcriptions and line geometry; 474 ground-truth lines) were read by two
independent readers:

* **VLM:** `qwen3-vl-8b-heb-v20a-step1800`, grounded-page prompt (JSON lines
  with `bbox_2d`, 0–1000 normalised). 21 of 24 pages returned parseable JSON;
  375 lines.
* **Kraken:** the site's Kraken microservice (`kraken-linewise`, `:8002`,
  `POST /transcribe_lines`, MiDRASH_Gen_01 model) → per-line `{text, bbox
  (px), confidence}`. ~20 s per page on CPU.

Matching rule (this is what the pipeline should implement): a Kraken
fragment belongs to a VLM line when ≥ 50% of its height lies inside the
line's vertical band and ≥ 50% of its width inside its horizontal extent;
each fragment goes to one line only (best vertical overlap); fragments are
concatenated right-to-left. Agreement = 1 − Levenshtein / max(len) on
**Hebrew letters only** (strip everything else). Ground truth was assigned by
geometry (largest vertical overlap), so "correct" below means both the text
and the box, unless stated.

Reference implementation:
`historical-document-analysis/src/datasets/evaluations/grounding_eval/line_agreement_probe.py`
(`assign_fragments`, `sim`, `geometric_gt`); results in
`grounding_eval/preds/line_agreement_probe.json`.

## Results

**1. Agreement is a precise signal.** Sweep of the agreement threshold τ
(v2.0a, 294 scorable lines):

| τ | lines accepted | share of lines | text CER ≤ 0.10 | text CER ≤ 0.20 | CER > 0.5 leaked | box on the right line |
|---|---|---|---|---|---|---|
| 0.6 | 64 | 22% | 78% (81% text-only) | 92% (95%) | 4.7% (0%) | 95% |
| 0.7 | 57 | 19% | 79% (83%) | 95% (98%) | 3.5% (0%) | — |
| **0.8** | **45** | **15%** | **84% (89%)** | **96% (100%)** | **4.4% (0%)** | **96%** |
| 0.9 | 30 | 10% | 97% (97%) | 100% (100%) | 0% (0%) | — |

Parenthesised values score the text against the line it actually transcribes
(text alignment) rather than the line under the box; the difference is boxes
that drifted one line. **Use τ = 0.8.** At that threshold nothing with a
badly wrong reading leaks through, the box sits on the right line 96% of the
time (median vertical IoU 0.78), and the remaining error is minor letter
confusions both readers share (ד/ר, ב/כ, ם/ס), which is why the badge must
say "two readers agree", not "verified".

**2. Coverage is low on this genre because Kraken barely reads it.** Kraken
produced 928 fragments with a median of 4 letters (54% under 5 letters); 16%
of ground-truth lines got no fragment at all, and two pages were almost
blank for it. 31% of VLM lines had no Kraken fragment to compare with. So on
Talmud manuscripts the badge lands on roughly **1 line in 8**. Documentary
Genizah hands (the PGP material) are what MiDRASH was trained on, so expect
higher coverage there; it was not measurable at line level (no line ground
truth exists for those).

**3. The VLM's line boxes are not display-ready on their own.** Only **58%**
of v2.0a's grounded boxes sit on the line whose text they carry; the rest
drift one line down the page (the boxes look evenly spaced rather than
measured). Text quality is much better than box quality: median line CER 0.14
against the line actually transcribed, 41% of lines at CER ≤ 0.10. Snapping a
box to its Kraken fragments does not fix this (it helps only the already
confirmed lines: 96% → 98%) and makes the box a thin strip. So: draw boxes
only for confirmed lines; show unconfirmed lines as text.

**4. Same rule on the previous model (v1.9a)** gives the same precision
(τ 0.8: 31 lines, 94% at CER ≤ 0.10) with fewer lines, so the rule is not
tuned to one checkpoint.

## What to build

**Offline pipeline (batch, not realtime), per document image:**

1. Kraken `/transcribe_lines` (existing service) → fragments.
2. v2.0a grounded read (LM Studio, the exact grounded prompt from the
   playground prompt; 25–90 s per page on the Studio; run at night or on the
   workstation; one request at a time).
3. Match and score with the rule above; keep everything, badge lines with
   agreement ≥ 0.8.
4. Write one sidecar per image, ingested into the index:

```json
{"ai_read": {
  "vlm_model": "qwen3-vl-8b-heb-v20a-step1800", "vlm_revision": "af9df6a0",
  "htr_model": "MiDRASH_Gen_01", "rule_version": "lines-v1-20260908",
  "decoded_at": "2026-09-08T16:31:00", "parsed": true,
  "n_lines": 16, "n_agreed": 3,
  "lines": [
    {"index": 0, "text": "...", "bbox": [104, 237, 928, 356],
     "agreement": 0.86, "status": "agreed",
     "htr_text": "...", "htr_fragments": [[x1,y1,x2,y2], ...]},
    {"index": 1, "text": "...", "bbox": [104, 356, 928, 472],
     "agreement": 0.22, "status": "unconfirmed", "htr_text": "..."}
  ]}}
```

**Surfacing rules:**

* Show the read on a document only when `parsed` is true and it has at least
  3 agreed lines (or ≥ 25% of lines agreed). Everything else stays hidden
  from visitors but remains queryable for the maintainer.
* On the image, draw boxes only for `agreed` lines, numbered in reading
  order, hover → text. Unconfirmed lines appear in the text panel in grey
  with no box.
* Page header: "N of M lines confirmed by a second reader (Kraken)". Line
  badge tooltip: "Two independent readers produced the same text (agreement
  0.86). Both can still share small letter confusions; not a scholarly
  transcription."
* Keep the model and rule versions visible in a details disclosure so the
  page can be regenerated when v2.1 lands.

**Not in scope now:** realtime reads, the old page-level consensus gate
(that remains the acceptance path for whole-page documentary transcriptions),
and any use of unconfirmed boxes as highlights.

## Caveats to carry into the UI copy

Single decode per model; a second decode of the same page moves individual
lines. Ground-truth geometry is word-union boxes, so vertical IoU numbers are
conservative. Agreement says the two readers concur, not that the line is
right; both readers share ד/ר, ב/כ and ם/ס confusions.
