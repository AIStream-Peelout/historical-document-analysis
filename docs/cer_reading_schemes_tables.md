# Transcription CER by reading scheme: PGP-131 and religious-140 (raw numbers)

Compiled 2026-10-12, read-only, from `/Users/isaac/Documents/GitHub/historical-document-analysis` (paths below are relative to it). No model, Kraken or network call was made. `memory:` = `~/.claude/projects/-Users-isaac-Documents-GitHub-historical-document-analysis/memory/`. "computed here" = recomputed by me from the cited file; the scratch scripts I cite are under `scripts/` in this scratchpad. All values have 3 decimals; "P" = PGP-131, "R" = religious-140, "–" = not run / not present.

## 0. Benchmarks and CER definitions

| item | definition / value | source |
|---|---|---|
| PGP-131 ("verified") | 131 documentary fragments; Tier A 55 / B 76; script buckets JA 82, Hebrew 20, Aramaic 6, untagged 22, Arabic-script 1; GT flattened to one line | `tier` column of `genizah_offline_long.csv` (computed here); `logs/next_round/consensus/rover_pgp131.md` l.49; `slot_oracle.md` l.7 |
| religious-140 | 140 KTIV pages from 82 MSS; 33 multi-col / 107 single-col; `is_talmud` on 138; GT rev 1.1 | `logs/next_round/consensus/rover_religious140.md` l.5, l.11; `docs/kraken7_segmenter_ab.md` l.47 |
| per-page CER (both) | strict `cer_pair(normalize_ink_hypothesis(raw), genizah_visible_ink_gt(gt))[0]` = Levenshtein(hyp, ref) / len(ref) after NFC + whitespace normalisation, not clamped (can be > 1); pages with ≥ 50 GT letters | `src/datasets/evaluations/metrics.py` l.88–127; P: `src/datasets/evaluations/helper_eval_scripts/score_genizah_offline.py` l.245–261 (self-duplicated GTs repaired at l.209–210); R: `.../helper_eval_scripts/run_religious_benchmark.py` l.226–241 |
| failure modes | abstained (< 25 asserted letters, or a refusal) / loop_collapse (one 12-gram covers > 0.45 of the text) / hallucinated (5-gram precision vs GT < 0.10) / substantive | `score_genizah_offline.py` l.69–71, l.168–189; R imports the same `classify` (`run_religious_benchmark.py` l.57–60, l.239) |
| CER* (P, `cer_median_substantive`) | median per-page CER over substantive pages only | `score_genizah_offline.py` l.310–311, l.321 |
| CERsub (R) | same, per slice ALL / talmud / multi-col / single-col | `run_religious_benchmark.py` l.269, l.277; slices l.299–302 |
| all-page mean / median | mean / median of the per-page `cer` over every scored page, all failure modes included (computed here from the long CSVs). The ROVER and slot-oracle reports use the same per-page CER and report 0 mismatches against the CSVs | `rover_pgp131.md` l.7; `rover_religious140.md` l.7 |
| P GT caveat | 7,284 GT characters that no reader wrote (31.3 % of the 23,269 such characters) are "not on the photographed text, or a GT defect", on 8 pages (plus T-S 6J3.23, counted by the short-read rule). That is 0.043 of the 170,532 pooled GT characters (0.041 without T-S 6J3.23; the report calls it an upper estimate). The memory note puts it as "≈0.04 CER inflating every model's score". R: 207 characters = 0.002 | `logs/next_round/consensus/unreachable_sample.md` l.22, l.39, l.41; `.../consensus/unreachable/profile.json` (`totals.gt`); `unreachable_sample_notes.md` l.3–8, l.25, l.32; memory:`project_pgp131_unreachable_and_gt_defects.md` l.3, l.14 |

## (a) Whole-page VLM read, per model version

P: CER*, n_sub, abstained / loop / hallucinated as a % of n, all-page mean / median. R: subst = substantive share, CERsub per slice, all-page mean / median and multi-col median.

| version | P n | P CER* | P n_sub | P abst% | P loop% | P hall% | P mean | P med | R n | R subst | R CERsub ALL | R talmud | R multi | R single | R mean | R med | R multi med |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qwen3_vl_8b | 131 | 0.493 | 2 | 11.5 | 58.8 | 28.2 | 5.913 | 5.357 | – | – | – | – | – | – | – | – | – |
| v16_step1100 | 131 | 0.372 | 18 | 0.8 | 16.8 | 68.7 | 5.628 | 0.937 | – | – | – | – | – | – | – | – | – |
| v17_step800 | 131 | 0.379 | 87 | 0.8 | 3.1 | 29.8 | 3.837 | 0.501 | – | – | – | – | – | – | – | – | – |
| v18a_step700 | 131 | 0.407 | 98 | 0.0 | 3.1 | 22.1 | 3.640 | 0.479 | – | – | – | – | – | – | – | – | – |
| v18b_step400 | 131 | 0.253 | 117 | 0.0 | 2.3 | 8.4 | 2.002 | 0.287 | – | – | – | – | – | – | – | – | – |
| v18b_step700 | 131 | 0.249 | 118 | 0.0 | 0.8 | 9.2 | 1.330 | 0.263 | – | – | – | – | – | – | – | – | – |
| v19a_step700 | 131 | 0.199 | 127 | 0.0 | 0.8 | 2.3 | 0.419 | 0.201 | – | – | – | – | – | – | – | – | – |
| v19a_step1300 | 131 | 0.196 | 128 | 0.0 | 1.5 | 0.8 | 0.674 | 0.198 | 140 | 0.914 | 0.216 | 0.215 | 0.255 | 0.205 | 0.345 | 0.239 | 0.327 |
| v19b_step1300 | 131 | 0.186 | 125 | 0.0 | 2.3 | 2.3 | 0.971 | 0.190 | 140 | 0.914 | 0.234 | 0.229 | 0.333 | 0.216 | 0.345 | 0.256 | 0.357 |
| v19b_ablmerger | 131 | 0.190 | 113 | 10.7 | 0.8 | 2.3 | 0.956 | 0.203 | 140 | 0.929 | 0.230 | 0.225 | 0.316 | 0.223 | 0.334 | 0.246 | 0.345 |
| v19c_step400 | 30 | 0.257 | 29 | 0.0 | 0.0 | 3.3 | 0.540 | 0.261 | 36 | 0.806 | 0.335 | 0.334 | 0.586 | 0.331 | 0.498 | 0.443 | 0.643 |
| v19c_step700 | 131 | 0.203 | 128 | 0.0 | 0.8 | 1.5 | 0.842 | 0.205 | 140 | 0.879 | 0.211 | 0.210 | 0.204 | 0.211 | 0.352 | 0.254 | 0.266 |
| v19c_step1000 | 33 | 0.204 | 28 | 6.1 | 0.0 | 9.1 | 1.169 | 0.232 | 41 | 0.756 | 0.301 | 0.290 | 0.398 | 0.290 | 0.484 | 0.398 | 0.536 |
| v19c_step1200 | 131 | 0.190 | 126 | 0.0 | 1.5 | 2.3 | 0.719 | 0.191 | 140 | 0.907 | 0.218 | 0.216 | 0.537 | 0.200 | 0.359 | 0.249 | 0.547 |
| v19c_step1800 | 131 | 0.189 | 125 | 0.8 | 1.5 | 2.3 | 0.965 | 0.192 | 140 | 0.921 | 0.225 | 0.222 | 0.355 | 0.211 | 0.346 | 0.243 | 0.496 |
| v20a_step1500 | 131 | 0.178 | 130 | 0.0 | 0.8 | 0.0 | 0.378 | 0.180 | 140 | 0.957 | 0.196 | 0.193 | 0.239 | 0.186 | 0.291 | 0.205 | 0.239 |
| v20a_step1800 | 131 | 0.182 | 129 | 0.0 | 0.8 | 0.8 | 0.393 | 0.184 | 140 | 0.943 | 0.184 | 0.183 | 0.218 | 0.179 | 0.304 | 0.196 | 0.243 |
| v21b_step600 | 30 | 0.166 | 27 | 0.0 | 3.3 | 6.7 | 1.167 | 0.180 | 36 | 0.917 | 0.247 | 0.226 | 0.178 | 0.258 | 0.425 | 0.258 | 0.178 |
| v21b_step900 | 30 | 0.183 | 29 | 0.0 | 0.0 | 3.3 | 0.752 | 0.186 | 36 | 0.944 | 0.266 | 0.253 | 0.278 | 0.253 | 0.324 | 0.278 | 0.278 |
| v21b_step1200 | 131 | 0.177 | 130 | 0.0 | 0.0 | 0.8 | 0.329 | 0.177 | 140 | 0.971 | 0.165 | 0.164 | 0.164 | 0.166 | 0.276 | 0.167 | 0.164 |
| v21b_step1300 | 30 | 0.182 | 29 | 0.0 | 0.0 | 3.3 | 0.384 | 0.186 | 36 | 0.889 | 0.245 | 0.242 | 0.139 | 0.249 | 0.306 | 0.260 | 0.166 |
| v21b_step1400 | 30 | 0.169 | 28 | 0.0 | 0.0 | 6.7 | 0.782 | 0.176 | 36 | 0.917 | 0.249 | 0.246 | 0.135 | 0.261 | 0.322 | 0.262 | 0.162 |
| v21b_step1500 | 30 | 0.180 | 29 | 0.0 | 0.0 | 3.3 | 0.662 | 0.181 | 36 | 0.917 | 0.250 | 0.236 | 0.168 | 0.252 | 0.437 | 0.252 | 0.168 |
| v21b_step1600 | 30 | 0.166 | 25 | 0.0 | 3.3 | 13.3 | 1.343 | 0.198 | 36 | 0.889 | 0.258 | 0.249 | 0.169 | 0.266 | 0.348 | 0.279 | 0.290 |
| v21b_step1700 | 131 | 0.177 | 127 | 0.0 | 0.8 | 2.3 | 0.774 | 0.178 | 140 | 0.971 | 0.175 | 0.173 | 0.208 | 0.169 | 0.261 | 0.181 | 0.212 |
| v22a_step1000 | 69 | 0.159 | 68 | 0.0 | 1.4 | 0.0 | 0.337 | 0.164 | – | – | – | – | – | – | – | – | – |
| v22a_step1300 | 131 | 0.164 | 131 | 0.0 | 0.0 | 0.0 | 0.227 | 0.164 | 140 | 0.957 | 0.186 | 0.185 | 0.221 | 0.182 | 0.271 | 0.192 | 0.221 |
| v22b_step900 | 131 | 0.161 | 128 | 0.8 | 0.8 | 0.8 | 0.404 | 0.163 | 140 | 0.971 | 0.184 | 0.183 | 0.246 | 0.172 | 0.295 | 0.191 | 0.246 |
| v22b_step1200 | 131 | 0.160 | 128 | 0.0 | 1.5 | 0.8 | 0.312 | 0.161 | 140 | 0.964 | 0.181 | 0.177 | 0.222 | 0.170 | 0.265 | 0.185 | 0.243 |
| v22i_a25 | 131 | 0.166 | 131 | 0.0 | 0.0 | 0.0 | 0.207 | 0.166 | 140 | 0.979 | 0.172 | 0.170 | 0.153 | 0.183 | 0.262 | 0.186 | 0.158 |
| v22i_a50 | 131 | 0.158 | 129 | 0.0 | 0.8 | 0.8 | 0.300 | 0.161 | 140 | 0.979 | 0.172 | 0.172 | 0.204 | 0.172 | 0.261 | 0.178 | 0.204 |
| v22i_a75 | 131 | 0.175 | 127 | 0.0 | 2.3 | 0.8 | 0.486 | 0.177 | 140 | 0.971 | 0.168 | 0.167 | 0.196 | 0.167 | 0.280 | 0.175 | 0.196 |
| v23a_step600 | 131 | 0.169 | 130 | 0.0 | 0.0 | 0.8 | 0.219 | 0.172 | – | – | – | – | – | – | – | – | – |
| v23a_step1200 | 131 | 0.153 | 129 | 0.0 | 0.0 | 1.5 | 0.369 | 0.155 | – | – | – | – | – | – | – | – | – |
| v23a_step1500 | 131 | 0.155 | 131 | 0.0 | 0.0 | 0.0 | 0.214 | 0.155 | – | – | – | – | – | – | – | – | – |

- Stem names: P uses `qwen3_vl_8b_heb_<version>` (the base model is `qwen3_vl_8b`) and R uses `<version>`. An n below 131 / 140 means a subset run (the 30-doc / 36-page quick sets, v19c_step1000 at 33 / 41, v22a_step1000 at 69). Every row is a plain whole-page read with the benchmark prompt. `v22i_aXX` (weight interpolations of v21b-1200 and v22a-1300; a50 = 50/50, `rover_religious140.md` l.175) and `v19b_ablmerger` (a merger-ablation checkpoint) are model variants, not reading schemes.
- The only VLM reading-scheme stem in either CSV is `qwen3_vl_8b_heb_v18b_sections` (section d). No `*_linewise`, `*_adv`, `*_blind`, halves or `*_decode_policy` stem exists in the CSVs or in `src/datasets/evaluations/transcription_raw_outputs/`. v23a has no R outputs (`rover_religious140.md` l.5). Cross-check: v20a_step1800's R multi-col all-page median of 0.243 equals the column probe's A_page median.
- source: `src/datasets/evaluations/transcription_results/paper_table/genizah_offline_summary.csv` + `genizah_offline_long.csv` (written 2026-10-07 04:05); `src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1/religious_scores_summary.csv` + `religious_scores_long.csv` (written 2026-10-03 02:06).

## (b) Prompt-only column prompts and (c) column crops at the ink gap: R, 33 two-column pages, v20a-1800

| condition | n | CER median | CER mean | under-read (len ratio < 0.8) | over-gen (> 1.2) | median len ratio | paired vs A: median ΔCER; better / worse (±0.02) |
|---|--:|--:|--:|--:|--:|--:|---|
| A whole page, benchmark prompt (existing benchmark read) | 33 | 0.243 | 0.322 | 12/33 | 1/33 | 0.97 | – |
| B "ONLY the right column" + "ONLY the left column" prompts, full image | 33 | 0.256 | 0.334 | 1/33 | 8/33 | 1.00 | +0.001; 9 / 8 of 33 (paired mean ΔCER +0.012) |
| C page split at its widest vertical ink gap, page prompt per half | 30 | 0.151 | 0.282 | 4/30 | 1/30 | 0.98 | −0.005; 11 / 7 of 30 |

| paired subset (computed here from the CSV; reproduces `docs/v21b_offline_status.md` l.45–50) | n | median A → other | mean A → other |
|---|--:|---|---|
| C on pages A under-read | 10 | 0.612 → 0.239 | 0.559 → 0.278 |
| C on pages A read on target (len ratio ≥ 0.8) | 20 | 0.126 → 0.121 | – |
| C, all pages with a crop | 30 | 0.218 → 0.151 | 0.312 → 0.282 (paired mean ΔCER −0.030; doc: 10 wins > 0.05, 4 losses, 16 within ±0.05) |
| B on pages A under-read | 12 | 0.607 → 0.448 | 0.558 → 0.436 |

- Per-page notes from the doc (l.48–52): B transcribes both columns twice on pages A had already read fully (len ratio ≈ 1.5 on 4 pages). C regressions: FL203725568's crop looped (CER 1.28 → 2.79, 3.6× the GT length), FL203737423 split badly (0.14 → 0.57), and 3 pages had no detectable gap (hence n = 30). Two pages A under-read have no crop (FL202432495, FL201858807; computed here).
- CER definition: strict `cer_pair`, all pages; len ratio = hyp / GT letters (`src/datasets/evaluations/helper_eval_scripts/probe_column_transcription.py` l.105–114; paired thresholds l.177–181).
- source: `src/datasets/evaluations/probes/column_transcription_qwen3_vl_8b_heb_v20a_step1800.md` l.3–11 and `.csv` (96 rows); `docs/v21b_offline_status.md` l.34–57; `docs/todo_by_track.md` l.14 ("prompt-only sectioning does not work").

## (d) Horizontal band "sections" reads: P, v18b

| stem | n | CER* | n_sub | abst% | loop% | hall% | mean | median | F1 med | ngramP med |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| v18b_step700 | 131 | 0.249 | 118 | 0.0 | 0.8 | 9.2 | 1.330 | 0.263 | 0.807 | 0.475 |
| v18b_step400 | 131 | 0.253 | 117 | 0.0 | 2.3 | 8.4 | 2.002 | 0.287 | 0.794 | 0.450 |
| v18b_sections | 131 | 0.445 | 121 | 0.0 | 3.8 | 3.8 | 1.240 | 0.465 | 0.725 | 0.461 |

| sections vs whole page (base = v18b_step700; computed here from `genizah_offline_long.csv`) | value |
|---|---|
| whole-page failures | 13 (12 hallucinated, 1 loop_collapse). 7 are substantive under sections (the commit's "rescued 7 of 13"; per-page CER median 8.760 → 0.465), 6 still fail |
| sections fails where the whole page was substantive | 4: JRL L.18 0.488 → 1.199, ENA NS 45.31 0.926 → 0.953, T-S 16.117 3.649 → 1.540 (loop), Bodl. d.79.36 0.528 → 5.268 (loop) |
| pages substantive in both | 114; median 0.246 (whole page) vs 0.443 (sections) |
| paired, all 131 pages (±0.005) | sections lower on 14, higher on 117 |
| same with base v18b_step400 | 14 failures, 9 rescued, 5 broken |
| sidecars | 131 pages, 385 bands. Bands per page: 1 band ×1 page, 2 ×31, 3 ×80, 4 ×18, 10 ×1 (median 3). `bisect` flag on 54 bands / 23 pages, `truncate` on 28 bands / 7 pages |

- Method (commit 7932bb3): overlapping ~550 px horizontal bands read with the calibrated page prompt and de-duplicated at the overlaps. A band loop triggers a token cap, then one bisection, then truncation.
- source: the P CSVs; `src/datasets/evaluations/transcription_results/linewise_details/*.sections.jsonl`; `git show -s 7932bb3 c751e58`; `src/datasets/consensus/retry_escalates.py` l.3–5; memory:`project_v19_consensus.md` l.25.

## (e) Per-line VLM reads of Kraken line crops (blind / adversarial)

| page (P) | Kraken lines | per-line blind v18b: CER / F1 | Kraken text of the same lines: CER / F1 | v18b_step700 whole page: CER / F1 | v18b_sections: CER / F1 |
|---|--:|---|---|---|---|
| T-S 10J12.4 | 35 | 1.145 / 0.274 | 0.322 / 0.765 | 0.263 / 0.803 | 0.527 / 0.716 |
| T-S 8.4 | 77 | 1.647 / 0.127 | 0.557 / 0.499 | 0.255 / 0.823 | 0.375 / 0.749 |

- Computed here: the sidecar `vlm_text` lines are joined with "\n" in Kraken order (as in `src/datasets/consensus/linewise.py` l.167) and scored with `cer_pair` and a copy of `aligned_prf`. The copy reproduces the CSV values of the three reference stems exactly (`scripts/linewise_blind_cer.py`). The sidecar's Kraken text is the k4 `/transcribe_lines` read; it scores identically to the `kraken_raw` row on both pages.
- This was a smoke test on 2 pages only. The 131-page blind run (`qwen3_vl_8b_heb_v18b_linewise`, `run_linewise_benchmark.py` l.40) was never run. Adversarial mode (`linewise.py` l.9–17, key `..._linewise_adv`, `run_linewise_benchmark.py` l.71) was never run either: no sidecar, output or CSV row exists. Memory: "F1 0.13–0.27 vs whole-page 0.80+", Kraken segments "median 145×43px" (memory:`project_v19_consensus.md` l.25). Commit 7932bb3: "kraken segments are word-fragments on this material and v18b hallucinates on tiny crops".
- source: `src/datasets/evaluations/transcription_results/linewise_details/Cambridge_CUL_T_S_10J12_4.blind.jsonl`, `Cambridge_CUL_T_S_8_4.blind.jsonl`; commits 5b2fda4, 7932bb3.

## (f) Half-page re-reads as a rescue of failed reads: 35 held-out failed reads of v22i-a50 (not on P / R pages)

CER = Levenshtein / edition characters after the visible-ink normalisation. 34 jobs are scored at document level against the parsed PGP edition (the document's other images keep their cached reads) and 1 against its side-verified page text. Pooled = Σerr / Σref. better / worse / same use ±0.02 (`src/datasets/consensus/reread.py` l.35). Failed-read classes: 28 loop, 4 capped, 3 skipped_text.

| variant | n | healthy re-reads | better / worse / same | median CER: re-read vs failed | pooled CER: re-read vs failed | pooled, keep re-read only if healthy |
|---|--:|--:|---|---|---|--:|
| same request again | 35 | 18 | 18 / 6 / 11 | 0.639 vs 0.700 | 0.592 vs 0.688 | 0.580 |
| temperature 0.3 + seed | 35 | 17 | 21 / 7 / 7 | 0.601 vs 0.700 | 0.596 vs 0.688 | 0.585 |
| two halves (12 % overlap, joined) | 35 | 18 | 23 / 7 / 5 | 0.618 vs 0.700 | 0.571 vs 0.688 | 0.606 |
| … halves, loop class | 28 | 13 | 18 / 6 / 4 | 0.591 vs 0.705 | 0.578 vs 0.727 | 0.617 |
| … halves, capped class | 4 | 3 | 4 / 0 / 0 | 0.464 vs 0.519 | 0.411 vs 0.485 | 0.439 |
| … halves, skipped_text class | 3 | 2 | 1 / 1 / 1 | 0.747 vs 0.700 | 1.020 vs 0.654 | 1.020 |
| max_tokens 6000 (capped class only) | 4 | 4 | 4 / 0 / 0 | 0.480 vs 0.519 | 0.417 vs 0.485 | 0.417 |
| other model's cached read (v21b-1200) | 35 | 16 | 17 / 13 / 5 | 0.691 vs 0.700 | 0.677 vs 0.688 | 0.646 |

| chain "keep the first healthy read" (computed here from the rows; better / worse with no margin) | healthy | pooled CER | median CER | better / worse |
|---|--:|--:|--:|---|
| same | 18/35 | 0.580 | 0.646 | 16 / 1 |
| same → seed | 22/35 | 0.577 | 0.600 | 19 / 2 |
| same → seed → halves | 28/35 | 0.552 | 0.598 | 24 / 3 |
| same → seed → halves → cached v21b | 29/35 | 0.550 | 0.595 | 25 / 3 |

- The failed reads themselves: pooled 0.688, median 0.700. The `repeat_penalty` variant is defined (`logs/next_round/failed_reads/reread_experiment.py` l.16) but was not run (`run_reread_paused.log` l.2). Chain values match memory:`project_next_round_v23.md` l.68.
- source: `logs/next_round/failed_reads/reread_runs/qwen3-vl-8b-heb-v22i-a50/reread_scores.json` (`base_check`: max |Δ| vs `cer_slices` 5e-5); `logs/next_round/failed_reads/score_reread.py` l.1–20.

## (g) Kraken line recognisers and MiDRASH Zenodo reads

| stem `kraken_…` | P n | P CER* | P n_sub | P abst% | P hall% | P mean | P med | R n | R subst | R CERsub ALL | R talmud | R multi | R single | R mean | R med | R multi med |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| raw | 131 | 0.344 | 127 | 0.8 | 2.3 | 0.394 | 0.358 | 140 | 0.600 | 0.519 | 0.519 | 0.623 | 0.499 | 0.614 | 0.680 | 0.679 |
| seg | 122 | 0.410 | 119 | 0.8 | 1.6 | 0.455 | 0.437 | 140 | 0.586 | 0.375 | 0.375 | 0.315 | 0.382 | 0.561 | 0.641 | 0.504 |
| raw_ktivft | 131 | 0.362 | 129 | 0.8 | 0.8 | 0.409 | 0.363 | 140 | 0.643 | 0.521 | 0.521 | 0.625 | 0.493 | 0.607 | 0.666 | 0.664 |
| seg_ktivft | – | – | – | – | – | – | – | 140 | 0.650 | 0.358 | 0.358 | 0.312 | 0.395 | 0.547 | 0.602 | 0.346 |
| raw_k7_default | 131 | 0.342 | 128 | 0.8 | 1.5 | 0.385 | 0.347 | 140 | 0.586 | 0.434 | 0.434 | 0.449 | 0.424 | 0.577 | 0.657 | 0.615 |
| seg_k7_default | 131 | 0.337 | 128 | 0.8 | 1.5 | 0.364 | 0.340 | 140 | 0.593 | 0.364 | 0.364 | 0.292 | 0.388 | 0.546 | 0.592 | 0.409 |
| raw_k7_blla2026 | 131 | 0.291 | 128 | 1.5 | 0.8 | 0.346 | 0.297 | 140 | 0.664 | 0.420 | 0.420 | 0.343 | 0.424 | 0.552 | 0.587 | 0.564 |
| seg_k7_blla2026 | 131 | 0.290 | 128 | 1.5 | 0.8 | 0.335 | 0.295 | 140 | 0.664 | 0.333 | 0.333 | 0.290 | 0.363 | 0.528 | 0.574 | 0.399 |
| raw_k7_blla2026_rgb | 131 | 0.299 | 127 | 1.5 | 1.5 | 0.352 | 0.305 | 140 | 0.643 | 0.415 | 0.415 | 0.359 | 0.459 | 0.604 | 0.655 | 0.568 |
| seg_k7_blla2026_rgb | 131 | 0.281 | 127 | 1.5 | 1.5 | 0.342 | 0.297 | 140 | 0.643 | 0.342 | 0.342 | 0.291 | 0.385 | 0.582 | 0.630 | 0.383 |
| raw_k7_blla2026_mv | 131 | 0.269 | 129 | 0.8 | 0.8 | 0.312 | 0.272 | 140 | 0.793 | 0.390 | 0.390 | 0.495 | 0.348 | 0.472 | 0.462 | 0.554 |
| seg_k7_blla2026_mv | 131 | 0.287 | 129 | 0.8 | 0.8 | 0.320 | 0.287 | 140 | 0.793 | 0.310 | 0.310 | 0.258 | 0.328 | 0.434 | 0.397 | 0.299 |
| raw_k7_orli | 131 | 0.355 | 130 | 0.0 | 0.8 | 0.377 | 0.356 | 140 | 0.521 | 0.391 | 0.391 | 0.285 | 0.490 | 0.634 | 0.703 | 0.393 |
| seg_k7_orli | 131 | 0.395 | 130 | 0.0 | 0.8 | 0.407 | 0.396 | 140 | 0.521 | 0.391 | 0.391 | 0.332 | 0.469 | 0.637 | 0.706 | 0.460 |
| raw_k7_ktivseg_e03_rgb | 131 | 0.260 | 129 | 1.5 | 0.0 | 0.303 | 0.270 | 140 | 0.743 | 0.315 | 0.314 | 0.221 | 0.333 | 0.475 | 0.410 | 0.316 |
| seg_k7_ktivseg_e03_rgb | 131 | 0.273 | 129 | 1.5 | 0.0 | 0.312 | 0.275 | 140 | 0.743 | 0.312 | 0.310 | 0.221 | 0.331 | 0.475 | 0.406 | 0.316 |
| raw_k7_ktivseg_e9_rgb | 131 | 0.261 | 129 | 1.5 | 0.0 | 0.304 | 0.264 | 140 | 0.729 | 0.315 | 0.315 | 0.228 | 0.328 | 0.474 | 0.428 | 0.316 |
| seg_k7_ktivseg_e9_rgb | 131 | 0.272 | 129 | 1.5 | 0.0 | 0.311 | 0.273 | 140 | 0.736 | 0.312 | 0.312 | 0.228 | 0.327 | 0.470 | 0.411 | 0.316 |
| raw_k7_ktivseg_e10_rgb | 131 | 0.261 | 128 | 0.8 | 1.5 | 0.301 | 0.264 | 140 | 0.764 | 0.317 | 0.317 | 0.217 | 0.336 | 0.467 | 0.409 | 0.315 |
| seg_k7_ktivseg_e10_rgb | 131 | 0.265 | 128 | 0.8 | 1.5 | 0.309 | 0.267 | 140 | 0.764 | 0.308 | 0.308 | 0.217 | 0.333 | 0.464 | 0.399 | 0.315 |
| raw_k7_ktivseg_e10_rgb_mv | 131 | 0.251 | 129 | 0.0 | 1.5 | 0.288 | 0.253 | 140 | 0.864 | 0.307 | 0.307 | 0.206 | 0.318 | 0.397 | 0.336 | 0.253 |
| seg_k7_ktivseg_e10_rgb_mv | 131 | 0.267 | 129 | 0.0 | 1.5 | 0.304 | 0.267 | 140 | 0.864 | 0.301 | 0.301 | 0.206 | 0.312 | 0.393 | 0.334 | 0.248 |
| raw_mvsmoke_e9 | – | – | – | – | – | – | – | 10 | 0.800 | 0.252 | 0.252 | 0.182 | 0.272 | 0.413 | 0.272 | 0.182 |
| seg_mvsmoke_e9 | – | – | – | – | – | – | – | 10 | 0.800 | 0.226 | 0.226 | 0.182 | 0.252 | 0.407 | 0.252 | 0.182 |
| raw_prodcand_e10mv | 4 | 0.166 | 4 | 0.0 | 0.0 | 0.168 | 0.165 | 8 | 0.750 | 0.167 | 0.167 | 0.118 | 0.208 | 0.289 | 0.208 | 0.118 |
| seg_prodcand_e10mv | 4 | 0.160 | 4 | 0.0 | 0.0 | 0.165 | 0.160 | 8 | 0.750 | 0.166 | 0.166 | 0.118 | 0.208 | 0.338 | 0.208 | 0.118 |

- Variant key (`docs/kraken7_segmenter_ab.md`): `raw` = segmenter reading order, `seg` = lines re-ordered by geometry (l.29–31); recognition is MiDRASH_Gen_01 throughout (l.28). Plain `raw` / `seg` = the kraken 4.3.13 service; `ktivft` = Gen_01 recognition fine-tuned on 45k KTIV lines (`docs/handoff_kraken5_segmenter.md` l.18).
- `k7_default` / `k7_blla2026` / `k7_orli` = kraken 7.0.3 with each segmenter; `_rgb` = colour input; `_mv` = test-time multi-view reading (scale 0.5 / 0.33, 90°/270°; l.266–270); `k7_ktivseg_eN_rgb` = the Genizah segmenter fine-tune at epoch N (e10 chosen, l.263–264; e10 + mv in production since 2026-09-25, l.331); `mvsmoke_e9` (10 pages) and `prodcand_e10mv` (4 P / 8 R pages) are smoke subsets.
- Loop is 0 % for every Kraken row on P; on R, blla2026_mv raw / seg each have 1 loop page. `docs/training_mixture_lineage.md` l.243 gives P CER* 0.344 → 0.251 (k4 → k7 ktivseg_e10).

| R, 138 pages matched to the MiDRASH Zenodo transcriptions (sys_num + FL position, FL gap 3–32) | CER median | CER mean | F1 median |
|---|--:|--:|--:|
| MiDRASH Zenodo (kraken 5.3.1.dev56, MiDRASH segmenters + recognition) | 0.240 | 0.298 | 0.834 |
| kraken_raw (k4) | 0.677 | 0.611 | 0.424 |
| kraken_seg (k4) | 0.635 | 0.557 | 0.500 |
| kraken_seg_ktivft | 0.592 | 0.543 | 0.541 |
| v21b-1200 (VLM, reference) | 0.167 | 0.278 | 0.869 |

- Computed here from the per-page file. Zenodo has the lower CER than kraken_raw on 130/138 pages, than kraken_seg on 129/138, and than v21b on 31/138. Medians by layout (Zenodo / k4 seg / ktivft / v21b): two-col (33) 0.124 / 0.504 / 0.346 / 0.164; single-col (105) 0.255 / 0.643 / 0.621 / 0.167.
- The later configurations against the same 0.240 (doc l.276–281; R all-page medians raw / seg): k4 0.680 / 0.641, blla2026 0.587 / 0.574, blla2026 + mv 0.462 / 0.397, e10 0.409 / 0.399, e10 + mv 0.336 / 0.334. These equal the R med column above.
- source: the P / R CSVs as in (a); `docs/kraken7_segmenter_ab/zenodo_vs_kraken_religious.json` (138 per-page rows); `docs/handoff_kraken5_segmenter.md` l.13–21; `docs/kraken7_segmenter_ab.md` l.35–45, l.276–281.

## (h) Multi-read voting

**(h1) Majority vote of reads (vote probe, P).** CER here counts Hebrew letters and spaces only: Levenshtein / len(GT) (`logs/next_round/vote_probe.py` l.20–34, l.86–93). Documents qualify with ≥ 60 GT characters, all 4 reads present and every read's CER < 0.45, which leaves 112. The pivot is v22i_a50; better / worse means ±0.005 against the pivot (l.100).

| system | n | median | mean | vs pivot better / worse |
|---|--:|--:|--:|---|
| v22i_a50 (pivot) / v22b_step1200 / v22a_step1300 / v22b_step900 | 112 | 0.145 / 0.145 / 0.144 / 0.139 | 0.156 / 0.155 / 0.155 / 0.156 | – |
| vote of 3 (a50 + v22b-1200 + v22a-1300) | 112 | 0.139 | 0.150 | 57 / 1 |
| vote of 4 (+ v22b-900) | 112 | 0.140 | 0.152 | 46 / 0 |

source: stdout was never saved. Re-derived here by executing that file's `bare` / `vote` and the scorer's `load_benchmark` verbatim, without importing the scorer module (`scripts/vote_probe_rerun.py`). memory:`project_next_round_v23.md` l.21: "0.156 -> 0.150 … better on 57 of 112 documents, worse on 1".

**(h2) ROVER over checkpoints, P (131 docs; every consensus row has 0 abstained / loop / hallucinated pages).** Cells are all-page mean / median. † = the failure filter uses the GT (upper bound). ‡ = subset chosen on this benchmark. Tags are directories under `logs/next_round/consensus/`.

| system (tag) | 5 VLMs: v21b-1200, v22a-1300, v22b-1200, v22i a50, v23a-1200 | 6 VLMs: + v23a-1500 |
|---|---|---|
| best single by mean; by median | v22a-1300 0.227 / 0.163; v23a-1200 0.369 / 0.155 | v23a-1500 0.214 / 0.155; v23a-1200 0.369 / 0.155 |
| page oracle, best VLM per page | 0.187 / 0.148 | 0.185 / 0.142 |
| backbone only (MBR selection, no vote) | 0.201 / 0.159 | 0.198 / 0.152 |
| word, unweighted (`word_unweighted_*`) | 0.199 / 0.153 | 0.196 / 0.149 |
| word, w = 1/median CER, LOO (`word_invmed_*`; per bucket `word_invmedbucket_*` identical) | 0.198 / 0.151 | 0.195 / 0.147 |
| word, filter GT † / peer (`word_filtgt_*`, `word_filtpeer_*`) | 0.198 / 0.153 | 0.195 / 0.149 |
| word, filter GT † or peer + bucket weights (`word_filt*_invmedbucket_5vlm`) | 0.197 / 0.151 | – |
| word + Kraken seg or raw, unweighted (`word_unweighted_*_kseg`, `_kraw`) | 0.200 / 0.153 | 0.197 / 0.147 |
| word + Kraken seg or raw, filter GT † or peer (`word_filt*_*_kseg`, `_kraw`) | 0.199 / 0.153 | GT †: 0.195 / 0.147 (kseg), 0.196 / 0.147 (kraw); peer: – |
| char, unweighted / filter GT † / filter peer (`char_unweighted_*`, `char_filtgt_*`, `char_filtpeer_*`) | 0.191 / 0.145 | 0.188 / 0.140 |
| best 3-subset ‡, word, filter GT † / peer (`sub3_filt*_5vlm_124`, `sub3_filt*_6vlm_125`) | 0.195 / 0.149; peer 0.196 / 0.149 | 0.194 / 0.148 |

- Kraken as a reader (not a voter): seg e10 mv 0.304 / 0.267, raw e10 mv 0.288 / 0.253. Gap sensitivity, 5 VLMs: word gap 0.25 → 0.206 / 0.158, word gap 0.50 → 0.196 / 0.153, char gap 0.75 → 0.191 / 0.147 (`word_unweighted_5vlm_gap025`, `_gap050`, `char_unweighted_5vlm_gap075`).
- Four VLMs (no v23a; tags `religious140/pgp131_*_4vlm*`): best single v22a-1300 0.227 (mean) and v22b-1200 0.161 (median); oracle 0.195 / 0.152; word 0.201 / 0.157; char 0.197 / 0.153; char + Kraken seg 0.191 / 0.147 (oracle incl. Kraken 0.189 / 0.152).
- Paired bootstrap, char unweighted vs best single. 5 VLMs: Δmean −0.036 [−0.084, −0.011], 112 pages better / 16 worse; Δmedian vs v23a-1200 −0.010 [−0.018, −0.001]. 6 VLMs: Δmean vs v23a-1500 −0.026 [−0.063, −0.006], 100 / 25; Δmedian −0.015 [−0.022, −0.005].
- source: `logs/next_round/consensus/rover_pgp131.md` l.15–43, l.97, l.133–139, l.147–172, l.219; `<tag>/summary.json`; script `src/datasets/evaluations/helper_eval_scripts/rover_consensus.py`.

**(h3) ROVER over checkpoints, R (140 pages; voters v21b-1200, v22a-1300, v22b-1200, v22i a50).**

| system (tag under `logs/next_round/consensus/religious140/`) | mean | median | CER* |
|---|--:|--:|--:|
| v21b-1200 / v22a-1300 / v22b-1200 / v22i a50 | 0.276 / 0.271 / 0.265 / 0.261 | 0.167 / 0.192 / 0.185 / 0.178 | 0.165 / 0.186 / 0.181 / 0.172 |
| v20a-1500 (fifth-voter check only) | 0.291 | 0.205 | 0.196 |
| Kraken seg / raw (k7 e10 mv) | 0.393 / 0.397 | 0.334 / 0.336 | 0.301 / 0.307 |
| page oracle, best of 4 VLMs (with Kraken as a 5th reader: 0.199 / 0.157) | 0.220 | 0.159 | 0.157 |
| backbone only (MBR) | 0.252 | 0.170 | 0.167 |
| word unweighted (`word_unweighted_4vlm`) / filter peer / filter GT † | 0.243 / 0.248 / 0.246 | 0.170 | 0.169 |
| char unweighted (`char_unweighted_4vlm`) / filter peer / filter GT † | 0.243 / 0.246 / 0.245 | 0.166 | 0.162 / 0.166 / 0.166 |
| word + Kraken seg / raw, unweighted (`word_unweighted_4vlm_kseg`, `_kraw`); either + filter peer | 0.249 / 0.250; 0.247 | 0.179; 0.178 | 0.176; 0.175 |
| char + Kraken seg / raw, unweighted (`char_unweighted_4vlm_kseg`, `_kraw`) | 0.231 / 0.232 | 0.159 | 0.157 / 0.158 |
| word / char + v20a-1500 (`*_unweighted_4vlm_v20a`) | 0.247 / 0.240 | 0.172 / 0.158 | 0.169 / 0.158 |

- By column class, single-col (107) and multi-col (33), mean / median: v21b 0.271 / 0.167 and 0.294 / 0.164; char ROVER 0.235 / 0.167 and 0.271 / 0.158; char + Kraken seg 0.222 / 0.161 and 0.261 / 0.155; oracle-4 0.217 / 0.163 and 0.228 / 0.147.
- Paired bootstrap, Δmean vs v22i a50: char −0.017 [−0.038, +0.002] (95 better / 15 worse); char + Kraken seg −0.030 [−0.049, −0.015] (105 / 19).
- source: `logs/next_round/consensus/rover_religious140.md` l.15–41, l.45–69, l.79, l.86; `religious140/<tag>/summary.json`.

**(h4) Slot oracle: per-slot ceiling over the ROVER network (headline table, verbatim).** Cells are mean / median CER.

| configuration | pages | best single | page oracle | ROVER word | ROVER char | slot oracle word (attested words) | slot oracle char | headroom char = ROVER char − slot oracle char | headroom word |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PGP-131, six VLMs | 131 | 0.214 / 0.155 | 0.185 / 0.142 | 0.196 / 0.149 | 0.188 / 0.140 | 0.148 / 0.100 | 0.137 / 0.091 | 0.052 / 0.050 | 0.049 / 0.050 |
| PGP-131, six VLMs + Kraken seg | 131 | 0.214 / 0.155 | 0.184 / 0.142 | 0.197 / 0.147 | 0.185 / 0.136 | 0.136 / 0.086 | 0.124 / 0.073 | 0.062 / 0.063 | 0.061 / 0.061 |
| religious-140, four VLMs | 140 | 0.261 / 0.167 | 0.220 / 0.159 | 0.243 / 0.170 | 0.243 / 0.166 | 0.184 / 0.123 | 0.169 / 0.111 | 0.074 / 0.055 | 0.059 / 0.046 |
| religious-140, four VLMs + Kraken seg | 140 | 0.261 / 0.167 | 0.199 / 0.157 | 0.249 / 0.179 | 0.231 / 0.159 | 0.159 / 0.117 | 0.133 / 0.093 | 0.097 / 0.066 | 0.090 / 0.062 |

- Char-level ROVER-wrong slots that are learnable (some reader had the GT entry) vs unreachable: P six VLMs 27.9 % / 72.1 %, P + Kraken seg 32.7 / 67.3, R four VLMs 31.6 / 68.4, R + Kraken seg 48.8 / 51.2 (l.44–50).
- source: `logs/next_round/consensus/slot_oracle.md` l.22–29, l.44–51; `logs/next_round/consensus/slot_oracle/*.json`; script `src/datasets/evaluations/helper_eval_scripts/slot_oracle.py`.

## (i) Frontier / API models and other OCR (P only)

| stem | P n | CER* | n_sub | abst% | loop% | hall% | mean | median |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| claude_opus_4_8 | 131 | 0.416 | 53 | 28.2 | 2.3 | 29.0 | 0.762 | 0.694 |
| claude_sonnet_5 | 131 | 0.435 | 29 | 8.4 | 1.5 | 67.9 | 0.656 | 0.619 |
| gemini_pro | 68 | 0.432 | 40 | 0.0 | 0.0 | 41.2 | 0.548 | 0.589 |
| gemini_flash | 122 | 0.542 | 52 | 11.5 | 7.4 | 38.5 | 0.706 | 0.725 |
| gpt_5_6_sol | 131 | 0.736 | 36 | 50.4 | 0.8 | 21.4 | 0.886 | 0.977 |
| gpt_5_2 | 1 | – | 0 | 100.0 | 0.0 | 0.0 | 1.000 | 1.000 |
| gemma_4_31b_it_mlx | 1 | – | 0 | 0.0 | 0.0 | 100.0 | 1.738 | 1.738 |
| vision_ocr_raw | 131 | 0.649 | 39 | 13.0 | 0.0 | 57.3 | 0.833 | 0.912 |
| vision_ocr_seg | 123 | 0.645 | 38 | 13.8 | 0.8 | 54.5 | 0.835 | 0.914 |

- No frontier or API rows exist on R (the R CSVs hold only Kraken and the qwen fine-tunes). gemini_pro (68) and gemini_flash (122) have partial coverage; gpt_5_2 and gemma_4_31b_it_mlx have 1 page each. `vision_ocr_seg` = "Vision OCR → LLM extraction" (`src/datasets/evaluations/fragment_evals.py` l.806); `vision_ocr_raw` = the raw OCR.
- Legacy W&B table `paper_table/genizah_paper_summary.csv` (2026-07-29, run hn9vq3xp, `cer_ink` median): claude_opus_4_8 0.748 (n=14), claude_sonnet_5 0.689 (14), gemini_flash 0.690 (14), gemini_pro 0.649 (6), gpt_5_6_sol 0.943 (14), kraken_seg 0.322 (15), v17_step800 0.423 (14). Crop stems `*_gemara` / `*_rashi` / `*_tosafot` exist in `transcription_raw_outputs/` for the printed-Talmud benchmark only and are not P rows.
- source: `genizah_offline_summary.csv` + `genizah_offline_long.csv`.

## Could not find / not present

- No 131-page per-line run and no adversarial-mode output anywhere; only the 2 blind smoke sidecars in (e).
- No reading-scheme run on R other than the column probe (33 two-column pages, v20a-1800 only). There are no sections, per-line, half-page or vote-of-reads runs on R, and no v23a or frontier outputs.
- decode_policy has no results yet. The untracked code is in place: `src/datasets/consensus/decode_policy.py`, the driver `src/datasets/evaluations/helper_eval_scripts/run_decode_policy.py` (appeared 01:32 today, while I was compiling), `tests/test_decode_policy.py` and `logs/decode_policy/overnight_1012.sh`. But there are no `*_decode_policy` outputs, scores or run logs.
- Half-page re-reads were never run on P or R pages, only on the 35 held-out failed reads scored against PGP editions.
- The vote-probe output file does not exist (re-derived above). The ≈ 0.04 GT-inflation figure is stated only in memory; the repo gives the character counts it is derived from (section 0).
- The "4 at accept quality" part of the sections rescue (`retry_escalates.py` l.4) has no per-document gate file that I found; not verified.
- `genizah_offline_by_script.csv` has no CER column (failure modes, n-gram precision and F1 only). The MiDRASH Zenodo comparison exists for R only, not P.
