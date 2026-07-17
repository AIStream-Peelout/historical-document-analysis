# Talmud OCR Segmentation Pipeline

**Code:** `src/datasets/evaluations/talmud_evaluation.py` — `_SEGMENTATION_PROMPTS`,
`_call_ocr_segmentation()`
**Last updated:** 2026-07-13

This documents the full OCR → section-text pipeline used by the Talmud
benchmark, including the quality-control layer added in July 2026. It is no
longer a single "segment this" LLM call — read this before touching the
prompts or the guards.

---

## 1. Why segmentation exists

The benchmark compares two model families on Vilna-edition Talmud pages:

| Family | Path to scored text |
|---|---|
| **End-to-end VLMs** (Gemini Pro/Flash, Claude, LM Studio VLMs) | image → per-section transcription (3 calls, one per section) |
| **Specialized OCR + text-LLM extraction** (Google Vision, Kraken/MiDRASH) | image → flat OCR text → segmentation LLM → per-section text |

Raw OCR output is **never scored**: both engines read the page column by
column, so their output interleaves the marginal apparatus, Gemara, Rashi,
and Tosafot out of reading order. Scoring it directly would conflate OCR
character accuracy with layout order. The raw text is saved
(`transcription_raw_outputs/<doc>/{kraken,vision_ocr}_flat.txt`) and shown in
inspection-only W&B columns (`kraken_raw`, `vision_ocr_raw`), so OCR-engine
failures can be told apart from segmentation failures.

Both OCR engines are routed through the **same** segmentation model
(`--segmentation_model`, default Gemini Flash; `--segmentation_base_url`
switches it to an LM Studio text model) so the two OCR pipelines differ only
in the OCR engine itself.

## 2. Stage 1 — three independent extraction calls

One text-only LLM call per section (gemara / rashi / tosafot), each asked to
extract only its section from the first 15,000 chars of the flat OCR text.
Three separate calls keep each response ~⅓ the size, avoiding
`finish_reason=2` (max-tokens) truncation.

The three prompts deliberately use **two different framings**:

- **Gemara and Tosafot** use `_OCR_STRUCTURE` — a *positional* description
  ("the labelled blocks come first, the remainder is Gemara mixed with
  Tosafot"). This wording is **byte-frozen**: empirically, even mild
  paraphrases (or appending "copy exactly / do not correct" clauses) make
  gemara extraction bimodal, collapsing to ~⅓ of the section in a large
  fraction of samples. Do not edit it without re-running the offline harness
  (§6).
- **Rashi** uses `_CONTENT_TAXONOMY` — a *content-shape* description (short
  glosses: headword ending in a period, explanation ending in a colon).
  History: until 2026-07-12 the rashi prompt claimed the labelled marginal
  apparatus (מסורת הש"ס, הגהות הב"ח, גליון הש"ס…) *was* the Rashi — those
  blocks lead the OCR stream but are not Rashi's commentary, so rashi CER sat
  above 1.0 for months. The content-based prompt dropped kraken rashi CER
  from 1.10/2.27 to 0.05/0.07 on the worst pages.

## 3. Stage 2 — quality control (the "detectors")

The segmentation LLM misfires in two distinct, **bimodal** ways: the same
prompt on the same input yields a good sample on one draw and a bad one on
the next, even at temperature 0.1. Prompt engineering demonstrably makes
this worse (§2), so quality is enforced by **selection**: check each sample,
resample at most once when a check fires, keep the better sample.

A "detector" here is nothing exotic — a single computed statistic with a
threshold.

### 3a. Collapse / over-extraction detector (length vs. GT)

**Failure mode:** the model extracts a fraction of the section (e.g. 1,358
of 2,354 GT chars → CER 0.43) or dumps several sections into one (1,615
chars against a 278-char GT → CER 3.8).

**Check:** for sections with more than 200 GT chars, flag if

```
len(output) < 0.75 × len(gt_section)   or   len(output) > 2.0 × len(gt_section)
```

⚠️ **Oracle disclosure:** this check reads the *length* (never the content)
of the ground-truth section. It is an evaluation-harness stabilizer that
would not be available in deployment. If a strictly GT-blind pipeline is
needed (e.g. for a reviewer), replace the bounds with fractions of the raw
OCR input length; the recitation detector below is already GT-free.

### 3b. Canonical-recitation detector (shingle overlap, GT-free)

**Failure mode:** the Talmud is heavily represented in LLM training data.
Instead of *copying* text out of the noisy OCR, the model sometimes
*recites* the canonical Vilna text from memory — visible as modern
punctuation (question marks, commas, quotation marks) that appears nowhere
in the OCR input, and as silent "corrections" of OCR errors. Against a GT
that follows the actual page, this inflates CER (observed: 0.126 on a page
whose faithful extraction scores 0.033). This is the same
`canonical_completion` failure mode the benchmark taxonomy tracks for
end-to-end VLMs, appearing inside the text-only extraction stage.

**Check:** the output is supposed to be a *copy* of input text, so measure
the fraction of the output's 8-character shingles (whitespace-normalized,
sampled at stride 4) that occur verbatim in the OCR input:

```
overlap = |{output shingles} ∩ {input shingles}| / |{output shingles}|
```

Flag if `overlap < 0.75`. Empirical separation (Berakhot sample pages):

| Sample | Overlap |
|---|---|
| Perfect copy (raw vs. itself) | 1.00 |
| Faithful extractions | 0.92–0.95 |
| Independent text (GT vs. raw OCR) | 0.87 |
| Recited sample (caught in production) | 0.55–0.57 |

### 3c. Resample-and-select policy

If either detector fires: make **one** additional call with the identical
prompt, then keep whichever sample has the lower badness score

```
score(sample) = |len(sample) − len(gt)| / len(gt)  +  (1 − overlap(sample))
```

(i.e. relative length deviation plus recitation evidence). At most one
resample per section per document, so the worst-case cost is one extra
Flash call per section. The console logs every firing:
`[gemara] low OCR overlap (0.55 — reciting?) … resampling once...`

## 4. Empirical validation

Measured offline against saved raw OCR + GT (see §6), and live in eval runs:

| Case | Before | After |
|---|---|---|
| kraken rashi 01_10 (apparatus-prompt bug) | 1.102 | 0.053 |
| kraken rashi 01_10b | 2.273 | 0.070 |
| kraken gemara 01_11 (collapse sample) | 0.433 | 0.094 |
| kraken gemara 01_10 (recitation sample) | 0.126 | 0.030–0.038 typical |

## 5. Known limitations

- **Residual sampling noise:** Flash segmentation still wobbles roughly
  ±0.1 CER run-to-run on gemara (e.g. mild partial omissions at ~0.9× GT
  length pass both detectors). Per-model **means across pages** are the
  stable quantity; single-page single-run numbers are not.
- **The segmentation model is part of the system under test.** Its variance
  and its susceptibility to canonical recitation are findings about the
  "OCR + LLM extraction" pipeline, not harness bugs — worth reporting.
- Kraken rashi on some pages (e.g. 01_11 at ~0.72 across all runs) is an
  OCR-quality limitation, not a segmentation failure — check the saved
  `kraken_flat.txt` before blaming the segmenter.
- Input truncated to 15,000 chars (raw pages run ~9–12k, so currently no
  loss).

## 6. Iterating safely (offline harness)

Segmentation prompts and guards can be tuned **without any image calls**:
every eval run saves the flat OCR per document, and GT parses locally.
Each experiment costs one Gemini Flash text call:

```python
import asyncio
from pathlib import Path
import src.datasets.evaluations.talmud_evaluation as tal
from src.datasets.document_models.talmud_gt_parser import load_gt_directory

gt = load_gt_directory(Path("src/datasets/raw_data/cairo_genizah/evaluations/talmud_sample/texts"))
flat = Path("transcription_raw_outputs/01_10_page_001/kraken_flat.txt").read_text()
sections, metrics = asyncio.run(
    tal._call_ocr_segmentation(flat, "gemini-3.5-flash", None, gt["01_10"])
)
```

Because behavior is bimodal, **run each variant several times** before
concluding anything, and always check gemara for regressions when touching
anything shared — it is the most prompt-sensitive section.
