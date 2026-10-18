# Audit of the order-independent n-gram precision metric (ARR #2809)

Date: 2026-09-16. Repo revision 6722dc6 (scorer last changed at bfcf03c).
Benchmark: `genizah_test_v1_verified.json` (131 fragments), raw outputs in
`src/datasets/evaluations/transcription_raw_outputs/`. Every number below is
recomputed offline from those files; no model was called.

Reviewer prompts this answers: jfAJ (validity as a hallucination measure,
threshold sensitivity), Wf2b ("invariant to layer choice" overreaches, no
thresholds in the paper), RbRu (frame behaviour classification as a filter on
CER).

Tooling: `src/datasets/evaluations/helper_eval_scripts/ngram_audit/`
(`common.py` parameterised metric, `score_variant.py` flag-driven scorer,
`analysis_[a-f]_*.py` one per section). The paper path
`score_genizah_offline.py` is untouched; `score_variant.py --check-paper`
asserts the default configuration reproduces the committed summary CSV cell
for cell (verified 2026-09-16: OK, Tier A = 55).

## 1. What the metric is (from code)

`score_genizah_offline.py`:

- Letter reduction: `letters_only()` (imported from
  `audit_genizah_benchmark.py:44,54`) strips `[...]`, `(!)`, `[?]` then
  concatenates every run of the Hebrew block `[֐-׿]+`. Spaces,
  punctuation and Arabic script are dropped **before** windowing, so windows
  cross word boundaries; nikud and cantillation are kept (inside the block).
- `ngram_precision()` (lines 86-105): reference windows go into a set (103),
  hypothesis windows into a list (104), score = share of hypothesis windows
  present in the set (105). No count clipping. Returns 0.0 when either side
  has fewer than `n` letters. `NGRAM_N = 5` (line 73).
- `loop_ratio()` (line 108): frequency of the single most common 12-gram
  times 12, divided by letter count. Because windows overlap the value can
  exceed 1.0 (a pure `אב` repetition scores 5.25); it is a repetition score,
  not a share.
- `classify()` (line 168), first match wins: abstained if asserted letters
  < 25 or a refusal regex hits the first 400 raw characters; loop_collapse if
  loop_ratio > 0.45; hallucinated if ngram_precision < 0.10; else
  substantive. Thresholds at lines 69-73.
- Asserted letters use Hebrew + Arabic + Syriac-range blocks (`_ASSERTED_RE`,
  line 153) but ngram_precision uses Hebrew only, so a mixed-script output is
  graded on its Hebrew half alone.
- Tier A (lines 277-280): kraken_seg precision >= 0.25 **and** any of seven
  VLMs (`_VLM_EVIDENCE`, line 80) >= 0.25.
- Audit exclusion (`audit_genizah_benchmark.py:141-177`): max over eight
  evidence models (line 47; includes vision_ocr_seg, excludes gpt_5_6_sol,
  the reverse of the tier list) of the same 5-gram containment < 0.12
  excludes the fragment as misaligned. 19 fragments were excluded this way.
- Inputs: hypothesis = `normalize_ink_hypothesis(raw)` (`metrics.py:319`),
  reference = `genizah_visible_ink_gt(gt)` (`metrics.py:300`).
- None of 0.10 / 0.12 / 0.25 / n=5 / span 12 / 0.45 / 25 is swept anywhere
  in the repo.

## 2. Summary and recommendation

**Verdict: keep the metric, change its definition in two places, and report
the sensitivity tables.** The order-independent n-gram precision does what the
paper claims for *off-page* text: text copied from a different manuscript
scores near zero (Section 3), the score ranks outputs almost exactly as
alignment and CER do (Spearman 0.80-0.83, Section 7), and it has the same
discriminative power against the LLM judge as those metrics (AUC 0.72 vs
0.73 / 0.68). Three weaknesses are real and measurable:

1. **No count clipping** lets an output recycle a few genuine windows
   thousands of times. This inflates HebVL-1.7 most (14 of its 87 substantive
   outputs are 3x-24x the reference length and collapse to ~0 under
   clipping), and it is why 4 of the 55 Tier A pages are Tier A at all
   (Section 4). It also makes the metric *rise* when a model patches a hard
   passage with text copied from elsewhere on the same page (Section 8).
2. **Hebrew-only letter set** grades mixed-script output on its Hebrew
   residue and made it arithmetically impossible for Arabic-script pages to
   pass the benchmark audit (Section 6).
3. **Canonical interpolation is invisible by construction**: canonical text
   shares windows with the fragment's own reference, so 34 of 44
   judge-flagged canonical completions are classed substantive with a median
   score six times the corpus median (Section 7). No page-level order-blind
   metric can catch this; it belongs to the judge / scholar workstream.

Recommended definition for the resubmission:

| choice | recommendation | evidence |
|---|---|---|
| n | **5 (keep)** | n=3 is unusable (wrong-page median 0.12-0.23, 64-97 % of wrong pairs clear 0.10); n=4 still admits 83 Tier A pages with a 4-9 % wrong-pair rate; n>=6 breaks near-miss reads (Kraken substantive 0.975 -> 0.951 / 0.885 at n=6 / 8) and halves Tier A. At n=5 the cross-document null has median 0.002-0.007 and p99 <= 0.18 (Section 3). |
| count clipping | **yes** | Zero cost to Kraken (0 flips, mean gap 0.008); removes the runaway-repetition channel the loop detector misses; converts the in-page-substitution response from +0.14 (rewarded) to -0.02 (weakly penalised). Cost: HebVL-1.7 0.664 -> 0.557, Tier A 55 -> 50 (Sections 4, 8). |
| letter set | **Hebrew+Arabic union** | Near-inert on reported numbers (7 of 1,352 outputs change class, all substantive->hallucinated, Tier A unchanged); removes the "graded on the Hebrew half" objection; makes the audit fair to Arabic-script pages (Section 6). |
| hallucination cutoff 0.10 | **keep, now justified** | Sits at the p99 of the cross-document null for the well-behaved systems (0.96 % of 78,130 wrong-page pairs reach it). C1/C3/C4 hold at every cutoff 0.05-0.60; C2 first fails at 0.20 (Section 5). |
| Tier A cutoff 0.25 | **keep, now justified** | 0.12 % of wrong-page pairs reach it. Tier A count is 80/68/55/46/26 at 0.15/0.20/0.25/0.30/0.40; it is the most threshold-sensitive number in the paper and should be reported with that range. |
| loop 0.45 / span 12, abstention 25 | **keep** | Inert: widest swing <= 1.6 pp on any system, no claim or Tier A count moves (Section 5). |
| audit exclusion 0.12 | keep, but re-run under the union set | Re-admits 4 of 19 excluded fragments (3 Arabic-script + 1). Benchmark membership change = user decision; see Section 6. |

Under the recommended definition (n=5, clipped, union set, cutoffs
unchanged) the paper table becomes:

| system | n | subst (paper) | subst (rec.) | halluc (paper) | halluc (rec.) | ngram med (paper) | ngram med (rec.) | CER* (paper) | CER* (rec.) |
|---|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.975 | 0.975 | 0.016 | 0.016 | 0.474 | 0.466 | 0.410 | 0.410 |
| kraken_raw | 131 | 0.969 | 0.969 | 0.023 | 0.023 | 0.476 | 0.473 | 0.344 | 0.344 |
| gemini_pro | 68 | 0.588 | 0.544 | 0.412 | 0.456 | 0.174 | 0.165 | 0.432 | 0.396 |
| gemini_flash | 122 | 0.426 | 0.377 | 0.385 | 0.434 | 0.064 | 0.051 | 0.542 | 0.487 |
| claude_opus_4_8 | 131 | 0.405 | 0.374 | 0.290 | 0.321 | 0.065 | 0.061 | 0.416 | 0.397 |
| claude_sonnet_5 | 131 | 0.221 | 0.206 | 0.679 | 0.695 | 0.029 | 0.026 | 0.435 | 0.372 |
| gpt_5_6_sol | 131 | 0.275 | 0.275 | 0.214 | 0.214 | 0.000 | 0.000 | 0.736 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.664 | 0.557 | 0.298 | 0.405 | 0.167 | 0.123 | 0.379 | 0.366 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.137 | 0.137 | 0.687 | 0.687 | 0.010 | 0.002 | 0.373 | 0.373 |
| qwen3_vl_8b | 131 | 0.015 | 0.015 | 0.282 | 0.282 | 0.000 | 0.000 | 0.493 | 0.493 |
| vision_ocr_seg | 123 | 0.309 | 0.276 | 0.545 | 0.577 | 0.024 | 0.014 | 0.646 | 0.617 |

Tier A 55 -> 50. Demoted: Oxford_Bodleian_MS_heb_d_66_32,
Cambridge_CUL_T_S_16_125, Cambridge_CUL_T_S_13J17_12 (only VLM evidence was
a HebVL-1.7 output 9-21x the reference length, clipped precision 0.00-0.01),
Cambridge_CUL_T_S_16_138 (same, 8.5x; Opus and Pro sit at 0.20-0.21, just
under the cutoff), Cambridge_CUL_T_S_16_307 (HebVL-1.7 0.28 -> 0.21, normal
length; a genuine borderline).

### Which paper claims are sensitive

| claim | paper setting | recommended setting | sensitivity |
|---|---|---|---|
| C1 Kraken >= every frontier VLM on substantive share | 0.975 vs 0.588 (+38.7 pp) | 0.975 vs 0.544 | Holds in 42 of 49 sweep settings; fails only at n=3 where Gemini Pro saturates at 1.000. **Robust.** |
| C3 Gemini Pro is the best frontier VLM | 0.588 vs 0.426 (+16.2 pp) | 0.544 vs 0.377 | Holds in 49 of 49. **Robust.** |
| C4 Kraken n-gram median > every frontier VLM | 0.474 vs 0.174 | 0.466 vs 0.165 | Cutoff-independent, holds at every n. **Robust.** |
| C2 HebVL-1.7 > every frontier VLM on substantive share | 0.664 vs 0.588 (+7.6 pp) | 0.557 vs 0.544 (+1.3 pp) | Fails in 25 of 49 settings (n=3, 4, 8; cutoff >= 0.20; clipped at h=0.05 or 0.20). **On the 68 fragments Gemini Pro actually answered, it inverts under clipping: HebVL-1.7 35/68 vs Gemini Pro 37/68 (paper setting: 44 vs 40).** Fragile; rephrase as "matches the best frontier VLM" or report the paired comparison. |
| C5 HebVL-1.7 n-gram median > every frontier VLM | 0.167 vs 0.174 | 0.123 vs 0.165 | **Already false at the paper setting** (true only at n=8 unclipped). Any prose implying it must go. |
| "55 of 131 fragments are Tier A" | 55 | 50 | Ranges 24-120 across n, 26-80 across the tier cutoff. Report as a range with the definition. |
| "invariant to layer choice" | — | — | True only for layers present in the reference; an untranscribed layer (marginalia, the under-text of a palimpsest) scores as hallucination. Rephrase: *invariant to reading order and to the editor's linearisation; a layer the editor did not transcribe still scores as off-page.* |

### 2b. The current best model (v21b), not in the paper

The paper's HebVL-1.7 (v17) is the checkpoint whose claim is fragile. The
current flagship, qwen3_vl_8b_heb_v21b_step1200, does not depend on the
metric definition at all (131 fragments; v21b-1700 and v20a-1500 shown for
context; alignment features cached in the scratchpad `features_v21.pkl`):

| system | subst (paper) | subst (rec.) | halluc (paper) | halluc (rec.) | ngram med (paper) | ngram med (rec.) | F1 med | CER* (rec.) | clip gap mean / p90 | outputs > 2x ref length |
|---|---|---|---|---|---|---|---|---|---|---|
| kraken_seg (n=122) | 0.975 | 0.975 | 0.016 | 0.016 | 0.474 | 0.466 | 0.679 | 0.410 | 0.008 / 0.008 | 0 |
| gemini_pro (n=68) | 0.588 | 0.544 | 0.412 | 0.456 | 0.174 | 0.165 | 0.421 | 0.396 | 0.009 / 0.019 | 0 |
| HebVL-1.7 (v17-800) | 0.664 | 0.557 | 0.298 | 0.405 | 0.167 | 0.123 | 0.553 | 0.366 | 0.033 / 0.089 | 34 |
| v20a-1500 | 0.992 | 0.985 | 0.000 | 0.008 | 0.649 | 0.642 | 0.876 | 0.177 | 0.009 / 0.016 | 2 |
| **v21b-1200** | **0.992** | **0.985** | 0.008 | 0.015 | **0.661** | **0.651** | **0.880** | 0.179 | 0.011 / 0.021 | 2 |
| v21b-1700 | 0.969 | 0.947 | 0.023 | 0.046 | 0.651 | 0.647 | 0.883 | 0.176 | 0.014 / 0.016 | 7 |

Paired on the 68 fragments Gemini Pro answered: v21b-1200 substantive 67/68
(paper) and 66/68 (recommended) vs Gemini Pro 40/68 and 37/68 vs Kraken
65/68 both ways. Paired on Kraken's 122: v21b-1200 121/122 vs Kraken
119/122 under both definitions. n-gram median 0.65 vs Kraken 0.47 vs Gemini
Pro 0.17 under either definition.

So every claim that is fragile for HebVL-1.7 is robust for v21b: it beats
every frontier VLM and Kraken on substantive share, n-gram median and
aligned F1 under the paper definition, the recommended definition, and on
the paired subsets, with margins of 30 to 45 points rather than 1 to 8. Its
clipping gap (mean 0.011) is at Kraken's level, because the runaway-repetition
channel that inflated v17 (34 outputs over twice the reference length) is
nearly gone (2 outputs: Cambridge_CUL_T_S_16_117 at 4.1x, unclipped 0.672 ->
clipped 0.10, and Cambridge_CUL_T_S_20_162 at 12.9x, already 0.000 under
both). v21b-1700 shows the rotating failure set already recorded for the
v19c/v21b arc: 7 runaway outputs, of which Cambridge_CUL_T_S_10J27_10 is the
clearest metric artefact (unclipped 0.500, clipped 0.003, 17.8x length).
Its cross-document floor is ordinary (wrong-page median 0.008, 1.1 % of
pairs >= 0.10).

Implication for the resubmission: if v21b replaces or joins HebVL-1.7 in the
tables, the fine-tuned-model claims stop being threshold-sensitive and the
metric change is cost-free; the sensitivity discussion then concerns only
the frontier VLM rows and the Tier A count. This is an argument for
including v21b, not for keeping the unclipped metric.

## 3. Analysis A: cross-document floor (`A_floor.md`)

Every substantive output (601 under the paper config) scored against the
other 130 references at n=5, unclipped: 78,130 wrong-page pairs.

| system | n subst | wrong med | wrong p90 | wrong p99 | wrong max | share >= 0.10 | share >= 0.25 | own med |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 119 | 0.006 | 0.026 | 0.078 | 0.300 | 0.006 | 0.000 | 0.477 |
| kraken_raw | 127 | 0.006 | 0.025 | 0.075 | 0.293 | 0.005 | 0.000 | 0.477 |
| gemini_pro | 40 | 0.006 | 0.034 | 0.114 | 0.316 | 0.014 | 0.001 | 0.294 |
| gemini_flash | 52 | 0.005 | 0.033 | 0.145 | 0.596 | 0.020 | 0.004 | 0.265 |
| claude_opus_4_8 | 53 | 0.002 | 0.011 | 0.045 | 0.238 | 0.001 | 0.000 | 0.203 |
| claude_sonnet_5 | 29 | 0.002 | 0.010 | 0.047 | 0.190 | 0.001 | 0.000 | 0.190 |
| gpt_5_6_sol | 36 | 0.005 | 0.043 | 0.181 | 0.643 | 0.036 | 0.005 | 0.295 |
| qwen3_vl_8b_heb_v17_step800 | 87 | 0.007 | 0.031 | 0.128 | 0.447 | 0.016 | 0.003 | 0.246 |
| qwen3_vl_8b_heb_v16_step1100 | 18 | 0.002 | 0.009 | 0.028 | 0.145 | 0.000 | 0.000 | 0.198 |
| vision_ocr_seg | 38 | 0.002 | 0.013 | 0.054 | 0.236 | 0.003 | 0.000 | 0.204 |

Floor versus n (wrong-page median / p90 / share >= 0.10):

| system | n=3 | n=4 | n=5 | n=6 | n=8 |
|---|---|---|---|---|---|
| kraken_seg | 0.220 / 0.339 / 0.964 | 0.036 / 0.084 / 0.057 | 0.006 / 0.026 / 0.006 | 0.001 / 0.011 / 0.003 | 0.000 / 0.003 / 0.001 |
| gemini_pro | 0.213 / 0.357 / 0.955 | 0.035 / 0.095 / 0.089 | 0.006 / 0.034 / 0.014 | 0.001 / 0.017 / 0.006 | 0.000 / 0.006 / 0.002 |
| gpt_5_6_sol | 0.224 / 0.381 / 0.963 | 0.038 / 0.114 / 0.133 | 0.005 / 0.043 / 0.036 | 0.000 / 0.021 / 0.019 | 0.000 / 0.003 / 0.011 |
| HebVL-1.7 | 0.225 / 0.361 / 0.969 | 0.040 / 0.099 / 0.098 | 0.007 / 0.031 / 0.016 | 0.001 / 0.013 / 0.007 | 0.000 / 0.003 / 0.003 |
| claude_opus_4_8 | 0.166 / 0.277 / 0.870 | 0.020 / 0.049 / 0.012 | 0.002 / 0.011 / 0.001 | 0.000 / 0.004 / 0.000 | 0.000 / 0.000 / 0.000 |

Findings:

- The average floor is small: 0.96 % of wrong-page pairs reach the 0.10
  cutoff and 0.12 % reach 0.25. The cross-document distribution is an
  empirical null, and the paper's cutoffs sit at roughly its p99 and p99.9.
  This is the validity argument jfAJ asked for.
- The per-output worst case is where the gate is weakest: 121 of 601
  substantive outputs (20 %) score >= 0.10 against at least one manuscript
  they never saw, 31 (5 %) score >= 0.25. GPT-5.6 Sol is worst (61 % / 22 %),
  the Claude models and v1.6 are floor-free.
- Mechanism is short output plus formulaic Judaeo-Arabic: outputs under 300
  letters reach 0.10 against a wrong page 49 % of the time (vs 17 % for
  longer ones); 13 of the 15 most formulaic references are Judaeo-Arabic or
  untagged epistolary documents and 11 of them are Tier A. Extreme case:
  Cambridge_CUL_T_S_16_117 / gemini_flash emits 27 letters (CER 0.99) yet
  scores 0.261 on its own reference and 0.522 against eight other fragments.
- Clipping does not change the floor (p90 moves <= 0.010); n does. The
  Kraken and Claude own-median sits 18-20x above the wrong-page p90; for
  Gemini, GPT and HebVL-1.7 the ratio is only 7-9x, and their own-reference
  p10 sits at 0.10-0.15, right on the gate.

Implication: a minimum asserted length for *Tier A evidence* (not for the
behaviour class) would remove the degenerate short-output cases; not adopted
here because it changes the tier rule rather than the metric, but worth a
sentence in the paper.

## 4. Analysis B: count clipping (`B_clip.md`, pairs in scratchpad `B/pairs/`)

| system | n | mean gap | median | p90 | max | gap >= 0.05 | gap >= 0.10 | subst -> halluc flips |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.001 | 0.008 | 0.223 | 4 | 3 | 0 |
| kraken_raw | 131 | 0.002 | 0.001 | 0.006 | 0.013 | 0 | 0 | 0 |
| gemini_pro | 68 | 0.009 | 0.006 | 0.019 | 0.060 | 1 | 0 | 3 |
| gemini_flash | 122 | 0.020 | 0.004 | 0.025 | 0.421 | 10 | 5 | 6 |
| claude_opus_4_8 | 131 | 0.003 | 0.000 | 0.003 | 0.153 | 2 | 1 | 1 |
| claude_sonnet_5 | 131 | 0.003 | 0.000 | 0.004 | 0.131 | 2 | 1 | 2 |
| gpt_5_6_sol | 131 | 0.003 | 0.000 | 0.011 | 0.040 | 0 | 0 | 0 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.033 | 0.004 | 0.089 | 0.415 | 21 | 13 | 14 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.005 | 0.000 | 0.005 | 0.111 | 5 | 1 | 0 |
| qwen3_vl_8b | 131 | 0.001 | 0.000 | 0.000 | 0.125 | 1 | 1 | 0 |
| vision_ocr_seg | 123 | 0.000 | 0.000 | 0.001 | 0.007 | 0 | 0 | 0 |

Findings:

- Two mechanisms. HebVL-1.7's gap is runaway generation: its 15 largest-gap
  outputs are 3.3x-24x the reference (Cambridge_CUL_T_S_13J17_12: 13,099 vs
  1,401 letters, one window repeated 654 times against 4 in the reference).
  These pass the loop gate because the repeated unit is longer and more
  varied than the 12-letter span, then earn 0.19-0.39 unclipped precision.
  Kraken's four large gaps are normal-length outputs (ratio 0.44-1.11) with
  duplicated lines or repeated formulaic openings inside one page; they stay
  substantive under clipping.
- Gemini Flash's worst cases are short outputs (ratio 0.43-0.57) repeating one
  formula 10-32 times (New_York_JTS_ENA_NS_50_32: 0.443 -> 0.022).
- Drivers: 33 of the 46 outputs with gap >= 0.05 are longer than the
  reference; 13 are not (8 Flash, 3 Kraken, 1 Pro, 1 HebVL-1.7). Pooled
  Spearman of gap with length ratio 0.29.
- The gap is 5x larger at n=3 and vanishes at n=8; n=5 sits where the
  unclipped shortcut is mostly, not entirely, harmless.
- Note for the record: the analysis B report quotes Gemini Pro's clipped
  share as 0.565; the recomputed value is 0.544 (37 of 68). The tables in
  this document use the recomputed value.

## 5. Analysis C: sweeps (`C_sweeps.md`, `sweep_summary_long.csv`)

Substantive share under one-factor sweeps from the paper setting (Tier
cutoff t changes only the Tier A count; loop l and abstention a shown at
their extremes):

| system | paper | n=3 | n=4 | n=6 | n=8 | h=0.05 | h=0.15 | h=0.20 | h=0.30 | l=0.30 | l=0.60 | a=15 | a=50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kraken_seg | 0.975 | 0.992 | 0.992 | 0.951 | 0.885 | 0.992 | 0.959 | 0.943 | 0.803 | 0.967 | 0.975 | 0.984 | 0.975 |
| kraken_raw | 0.969 | 0.992 | 0.985 | 0.954 | 0.885 | 0.985 | 0.962 | 0.931 | 0.786 | 0.969 | 0.969 | 0.977 | 0.969 |
| gemini_pro | 0.588 | 1.000 | 0.897 | 0.515 | 0.441 | 0.794 | 0.515 | 0.441 | 0.279 | 0.588 | 0.588 | 0.588 | 0.588 |
| gemini_flash | 0.426 | 0.803 | 0.672 | 0.344 | 0.262 | 0.525 | 0.328 | 0.262 | 0.180 | 0.418 | 0.434 | 0.434 | 0.410 |
| claude_opus_4_8 | 0.405 | 0.649 | 0.557 | 0.313 | 0.198 | 0.565 | 0.305 | 0.214 | 0.084 | 0.405 | 0.405 | 0.405 | 0.397 |
| claude_sonnet_5 | 0.221 | 0.885 | 0.359 | 0.168 | 0.069 | 0.351 | 0.153 | 0.107 | 0.053 | 0.221 | 0.221 | 0.221 | 0.214 |
| gpt_5_6_sol | 0.275 | 0.489 | 0.374 | 0.252 | 0.176 | 0.321 | 0.237 | 0.198 | 0.137 | 0.275 | 0.275 | 0.305 | 0.252 |
| HebVL-1.7 | 0.664 | 0.931 | 0.840 | 0.534 | 0.366 | 0.817 | 0.527 | 0.420 | 0.214 | 0.649 | 0.679 | 0.664 | 0.664 |
| HebVL-1.6 | 0.137 | 0.748 | 0.237 | 0.107 | 0.061 | 0.221 | 0.092 | 0.061 | 0.015 | 0.137 | 0.137 | 0.137 | 0.137 |
| qwen3_vl_8b | 0.015 | 0.031 | 0.023 | 0.015 | 0.008 | 0.023 | 0.015 | 0.008 | 0.000 | 0.015 | 0.015 | 0.015 | 0.015 |
| vision_ocr_seg | 0.309 | 0.642 | 0.431 | 0.203 | 0.138 | 0.407 | 0.220 | 0.154 | 0.089 | 0.293 | 0.309 | 0.317 | 0.285 |
| Tier A | 55 | 120 | 83 | 37 | 24 | 55 | 55 | 55 | 55 | 55 | 55 | 55 | 55 |
| C1 Kraken >= frontier | T | **F** | T | T | T | T | T | T | T | T | T | T | T |
| C2 HebVL-1.7 > frontier | T | **F** | **F** | T | **F** | T | T | **F** | **F** | T | T | T | T |
| C3 Pro best frontier | T | T | T | T | T | T | T | T | T | T | T | T | T |

Tier A count vs tier cutoff: 0.15 -> 80, 0.20 -> 68, 0.25 -> 55, 0.30 -> 46,
0.40 -> 26. The 27-setting grid (n x cutoff x {unclipped, clipped, union})
and every per-setting behaviour table are in `C_sweeps.md`; the two
condensed tables above are the appendix candidates.

Margins and first failure (n=5 unclipped, cutoff 0.05-0.60):

| claim | margin at paper setting | first fails at |
|---|---|---|
| C1 | +38.7 pp | never |
| C2 | +7.6 pp | 0.20 |
| C3 | +16.2 pp (over Flash) | never |
| C4 (ngram medians) | +30.0 pp | cutoff-independent |
| C5 (HebVL-1.7 ngram median) | -0.7 pp | false at the paper setting |

Caveat: each share is over the system's own denominator (Gemini Pro 68,
kraken_seg / Flash 122, vision_ocr_seg 123, others 131), as in the paper.
Gemini Pro's 68 are the weakest leg of C1-C3; the paired comparison in
Section 2 addresses that for C2.

## 6. Analysis D: letter set (`D_letterset.md`)

- Union set on the verified 131: 7 of 1,352 outputs change class (3 Opus, 4
  vision_ocr_seg), all substantive -> hallucinated; 40 outputs move at all
  (mean -0.030); Tier A 55 -> 55. The union set never inflates a number,
  because only 3 of 131 references contain any Arabic-block letter (4-17
  letters each).
- The paper metric silently grades mixed-script output on its Hebrew half:
  60 outputs are >= 20 % Arabic-block (50 of them vision_ocr_seg, which
  emits Arabic characters in 47 % of its outputs; Opus in 10 %). For 53 of
  the 60 the Hebrew residue is under 5 letters, so the paper score is 0.000
  and reads as "invented" when it means "wrote Arabic".
- The four Arabic-script exclusions have zero Hebrew-block GT letters, so
  under the paper's letter set no system could ever pass the 0.12 audit.
  Under the union set: T_S_Ar_4_10 best 0.379 (Gemini Pro; aligned F1 0.651,
  CER 0.423) and T_S_Ar_38_2 0.249 (Gemini Pro) pass; ENA_NS_2_29 0.069 and
  T_S_Ar_19_23 0.022 do not (the latter has only 4 of 8 evidence systems on
  disk). Kraken and the Hebrew-tuned Qwens score 0.000 on all four under
  both sets.
- Union-set re-audit of all 19 exclusions readmits 4: Manchester_JRL_Genizah_
  Ar_806 (0.034 -> 0.631), T_S_Ar_4_10, T_S_Ar_38_2, T_S_NS_320_42 (0.026 ->
  0.162). The other 15 are genuine misalignments. Readmission is a benchmark
  change (131 -> 135, and `load_fragments` would need a union-set minimum
  instead of 50 Hebrew letters); left for the user to decide.
- Script tags: 6 of 150 are `arabic_script`; 5 of the 6 are among the
  exclusions; Cambridge_CUL_T_S_Ar_4_10 is tagged judaeo_arabic despite an
  all-Arabic GT, so the tag under-counts.

## 7. Analysis E: agreement with other signals and the judge (`E_agreement.md`)

Correlation of 5-gram precision with the other metrics (paper config):

| subset | rho aligned P | rho aligned F1 | rho CER-lenient | Pearson aligned P | Pearson CER-lenient |
|---|---|---|---|---|---|
| pooled, all outputs | 0.802 | 0.833 | -0.768 | 0.708 | -0.254 |
| pooled, Tier A | 0.814 | 0.803 | -0.734 | 0.705 | -0.289 |

Per-system Spearman with aligned precision 0.62-0.83 (degenerate only for
base Qwen, 90 % of whose outputs score 0). Pearson with CER is weak because
CER is unbounded on runaway outputs.

Disagreements: (a) ngram >= 0.25 but aligned precision <= 0.5: 33 outputs
(2.4 %), 15 of them kraken_seg, i.e. reading-order casualties the metric is
designed to forgive. (b) ngram < 0.10 but aligned precision >= 0.5: 139
(10.3 %), concentrated in Sonnet, GPT, vision_ocr_seg, HebVL-1.6. Half of (b)
are stubs (length ratio < 0.15, median aligned recall 0.03) where aligned
precision is the misleading number; the rest are near-miss reads whose every
5-gram breaks: median precision climbs 0.046 -> 0.111 (n=4) -> 0.301 (n=3).
This is the cost of n=5 and the reason n=3/4 were tempting; Section 3 shows
why they are not acceptable.

Judge (Gemini Flash taxonomy, 1,215 of 1,352 outputs judged; kraken_raw not
judged at all):

- Severe "hallucination" fires on 78 % of judged outputs; 29 % of those
  examples complain about inserted rafeh / [?] / nikud marks rather than
  invented words. The judge is a noisy binary partner.
- Conditioned distributions: severe-flagged median precision 0.030 (p25
  0.000, p75 0.155, n=953) vs not-flagged 0.228 (0.052-0.495, n=262). By
  judged quality: unusable 0.006 (n=830), poor 0.277 (350), fair 0.622 (29),
  good 0.749 (6). The metric separates the judge's classes cleanly as an
  ordinal signal.
- 2x2 (metric hallucinated vs judge severe hallucination, n=935): agreement
  0.641, kappa 0.284; 300 judge-only positives vs 36 metric-only. The metric
  is far more conservative than the judge.
- AUC for predicting severe hallucination: n-gram 0.717, aligned precision
  0.726, CER-lenient 0.678. The order-blind metric loses nothing.
- Canonical completion (44 outputs): 34 classed substantive, median
  precision 0.310 vs 0.050 corpus-wide; Manchester_JRL_A_960 scores
  0.74-0.87 across six systems. Interpolated canonical text shares windows
  with the page's own reference. **This is the metric's true blind spot and
  cannot be fixed by any of the knobs above.**

Scholar-validation sample: `scholar_sample.csv` (155 outputs: 120 by
precision bin with 30 each from [0.05,0.10), [0.10,0.20), [0.20,0.35) and 10
from each other bin, plus all 44 canonical-completion outputs; per-system cap
25 % per bin; seed 20260916; image URLs included). Hypothesis/reference text
pairs are in the scratchpad `E/scholar_sample_pairs/`.

## 8. Analysis F: simulated in-page substitution (`F_substitution.md`)

55 Tier A pages, kraken_seg output; middle 30 % of lines replaced by
pseudo-lines from the last 20 % of the same reference.

| metric | before | after | mean delta | n alarmed (moved > 0.02 the wrong way) |
|---|---|---|---|---|
| 5-gram precision, unclipped (paper) | 0.554 | 0.697 | **+0.142** | 0 / 55 (rose on 55 / 55) |
| 5-gram precision, clipped | 0.539 | 0.520 | -0.019 | 29 / 55 |
| aligned precision | 0.749 | 0.519 | -0.230 | 48 / 55 |
| aligned F1 | 0.674 | 0.463 | -0.211 | 45 / 55 |
| CER lenient | 0.413 | 0.520 | +0.107 | 42 / 55 |
| loop ratio | 0.024 | 0.047 | +0.023 | 28 / 55 (max 0.104, never near 0.45) |

In-page vs cross-page (donor lines from a different Tier A page), n=5:
unclipped +0.142 vs -0.197; clipped -0.019 vs -0.197. Hallucination gate:
0/55 in every setting including cross-page; Tier A evidence cutoff: 0/55
in-page, 14/55 cross-page. Larger n does not help (n=8 unclipped rises
+0.187); only clipping changes the sign, and at 50 % substitution clipped
precision becomes decisive (-0.126, 43/55).

Single closing formula inserted once mid-page: unclipped +0.011, clipped
+0.004, aligned precision median -0.020, CER median +0.013, loop 2/55. One
spurious line in ~37 is below every page-level threshold; detecting it needs
per-line alignment (the two-reader-lines machinery), not a page aggregate.

Conclusion for the paper: the n-gram metric is an *off-page* detector and
must be described as one. In-page substitution and canonical interpolation
are detected by order-sensitive metrics (aligned F1, CER) on Tier A pages
and by nothing on Tier B pages; the behaviour classification therefore
cannot claim to catch "plausible but wrong" text that is plausible because
it is on the page.

## 9. What to say to each reviewer

- **jfAJ (validity, threshold sensitivity):** validity via the
  cross-document null (Section 3: cutoffs at p99 / p99.9 of the null;
  Spearman 0.80 with alignment; AUC parity with alignment and CER against the
  judge). Sensitivity: the two condensed tables of Section 5 go in the
  appendix; C1, C3, C4 are robust, C2 is fragile and will be rephrased, the
  Tier A count is reported as a range.
- **Wf2b (layer invariance, thresholds absent):** rephrase as invariance to
  reading order and linearisation among transcribed layers; add the
  threshold values (0.10, 0.25, loop 0.45 / span 12, 25 letters, n=5) and
  the null-model justification to the methods section.
- **RbRu (classification as a filter on CER):** RbRu is right, and the
  sweep shows why it matters. The scorer reports CER only over substantive
  attempts, so the hallucination cutoff is a CER admission filter, and the
  conditional CER median moves with it (`sweep_summary_long.csv`, column
  cer_median_substantive, n=5 unclipped):

  | system | h=0.05 | h=0.10 (paper) | h=0.15 | h=0.20 | h=0.30 |
  |---|---|---|---|---|---|
  | kraken_seg | 0.436 | 0.410 | 0.407 | 0.407 | 0.348 |
  | gemini_pro | 0.516 | 0.432 | 0.390 | 0.347 | 0.249 |
  | HebVL-1.7 | 0.429 | 0.379 | 0.361 | 0.333 | 0.279 |

  A stricter filter admits fewer, better pages and lowers every VLM's CER
  median by up to 0.27, while Kraken's barely moves until 0.30. The
  resubmission should present CER* explicitly as "CER conditional on
  passing the 0.10 off-page filter", state the filter in the table caption,
  and give this row of the sweep in the appendix; system rankings on CER*
  are otherwise not comparable across filter settings.

## 10. Files, reproduction, and what was not done

Code (all new, paper path untouched):

- `src/datasets/evaluations/helper_eval_scripts/ngram_audit/common.py` —
  parameterised metric, loaders, feature cache.
- `.../ngram_audit/score_variant.py` — flag-driven scorer;
  `--check-paper` reproduces the committed summary CSV cell for cell.
- `.../ngram_audit/analysis_{a_floor,b_clip,c_sweeps,d_letterset,
  e_agreement,f_substitution}.py` — one per section; each reads the feature
  cache and writes to `$SCRATCH/<letter>/`.

Run (repo root, `.venv`):

```bash
.venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.ngram_audit.score_variant --check-paper
```

```bash
.venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.ngram_audit.score_variant --clip --letter-set semitic --paper-systems-only --out-dir /tmp/rec
```

Supporting reports copied into `docs/ngram_metric_audit/` (the six per-analysis
markdown files with every table, plus the small CSVs). Large intermediates
(78k-row pair matrix, 1,352-row long CSVs, the 45 clipping pair files, the
155 scholar-sample pair files, 5 substitution examples) are in the session
scratchpad under `ngram_audit/` and are regenerated by the scripts.

Not done / caveats:

- No model calls, so nothing was re-transcribed; the judge file lacks
  kraken_raw entirely and a handful of HebVL-1.7 / base-Qwen outputs.
- Readmitting the 4 union-set audit survivors was not applied (benchmark
  change).
- The Tier A evidence minimum-length idea (Section 3) was not implemented.
- Analysis F builds reference pseudo-lines by equal word chunks because the
  GT has no line breaks; substitution positions are therefore approximate.
- One reporting slip in the analysis B narrative (Gemini Pro clipped share)
  was caught by recomputation; the numbers in this document are the
  recomputed ones.
