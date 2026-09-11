# Prompt: two-reader rule v2 — what changed and what the site must change

For the site session. It follows up the "narrow boxes / partial Kraken rows"
finding on L-G Ar. II.70. Findings verified in the pipeline repo on
2026-09-09; nothing has been loaded into Elasticsearch yet.

## Diagnosis, corrected

* **The page is two columns.** The "measured ink extent 155–803" spans both
  columns; the model's box 648–795 sits inside the right column, whose lines
  Kraken reads at 572–821. So the box covered about 60% of its line, not a
  fifth, and the model was reading the right column first (correct order).
* **The boxes are still a template.** 14 of the first 16 lines carry the
  identical x-range, and across the first 115 pilot pages **87 have ≥ 50% of
  their lines sharing one x-range**. The model emits one x-range per page or
  column and steps it down the page; the boxes are a layout prior, not a
  measurement. This matches the probe's drift finding.
* **The confirmed lines were legitimately confirmed.** Agreement is
  `1 − Levenshtein / max(len)`, so a partial Kraken row lowers agreement; it
  cannot raise it. The four agreed lines had Kraken rows of 21–22 letters
  against 21–24 in the model's text. The real cost of the old horizontal
  test was recall: edge fragments outside the narrow box were dropped, e.g.
  line 7 lost "פאן קבלת" (12-letter row instead of 19, agreement 0.48
  instead of 0.76) and line 14 went from unconfirmed to confirmed once its
  row was complete.

## Rule v2 (`lines-v2-20260909`), implemented pipeline-side

1. **Assignment is band-first.** Candidates are the Kraken fragments with at
   least half their height inside the line's vertical band. Candidates are
   clustered by horizontal gaps (> 60‰ of the page width starts a new
   column); the cluster overlapping the model box most (else the nearest to
   its centre) is taken whole. One line per fragment, best vertical overlap
   wins. Two-column pages stay separated; narrow boxes no longer truncate
   rows.
2. **`bbox` is now the evidence box:** union of the model box and the line's
   assigned fragments, or the model box alone when there are none. On the
   flagged page line 14's box became 581–815 (the full Kraken row) instead of
   648–795. Your proposed client-side union is therefore unnecessary; keep
   drawing boxes only for `agreed` lines.
3. `htr_fragments` now holds the whole row of the line's column;
   `agreement` and `AGREED_MIN = 0.8` are unchanged, so `line_status` is
   unchanged.

Validation: on the probe pages the rule keeps its precision (100% of
confirmed lines at CER ≤ 0.20, box on the right line 98% vs 96%) and leaves
fewer lines without evidence (95 → 59 of 340). On L-G Ar. II.70 confirmed
lines go 13 → 14 of 30.

## What to change on the site

* `src/backend/ai_transcriptions.py`: `RULE_VERSION = "lines-v2-20260909"`
  (the `_known_rule` validator currently rejects every v2 record). Update
  the module docstring's bbox sentence: the box is the evidence box (model
  box ∪ Kraken fragments), 0–1000 normalised to the oriented image.
* `docs/planned_features/ai-transcriptions.md`: add the v2 paragraph above
  and the template-box prevalence (87 of 115 pages).
* Nothing else: the envelope, keys, statuses and surfacing rule are the same.

## Pipeline state

All 118 records written under v1 are being rebuilt under v2 from a new raw
cache (Kraken re-run only; the model is not re-read), then the 228-image
pilot resumes. The output file is the same:
`historical-document-analysis/src/datasets/raw_data/cairo_genizah/ai_reads/ai_reads_qwen3-vl-8b-heb-v20a-step1800.jsonl`.
Future rule changes are a `--rematch` over the cache, seconds per page.
