# Analysis C — threshold and n sweeps

Benchmark: 131 verified fragments x 11 paper systems = 1352 outputs. Paper config: `n5 unclipped hebrew h0.10 t0.25 l0.45 a25` (Tier A = 55).

Claim keys used throughout:

* **C1 kraken_seg %subst >= best frontier** (`kraken_ge_frontier_substantive`)
* **C2 HebVL-1.7 %subst > best frontier** (`hebvl17_gt_frontier_substantive`)
* **C3 gemini_pro %subst = best frontier** (`gemini_pro_best_frontier_substantive`)
* **C4 kraken_seg ngram median > every frontier** (`kraken_ngram_gt_frontier`)
* **C5 HebVL-1.7 ngram median > every frontier** (`hebvl17_ngram_gt_frontier`)

`%subst` = share of that system's outputs classified substantive; `ngram med` = median n-gram precision over all its outputs; `F1 med` = median aligned F1 (config-independent, shown for reference); `CER med (subst)` = median CER over substantive outputs only.

**Coverage caveat:** systems do not all cover all 131 fragments — kraken_seg 122, kraken_raw 131, gemini_pro 68, gemini_flash 122, claude_opus_4_8 131, claude_sonnet_5 131, gpt_5_6_sol 131, qwen3_vl_8b_heb_v17_step800 131, qwen3_vl_8b_heb_v16_step1100 131, qwen3_vl_8b 131, vision_ocr_seg 123. Every share below is over the system's own denominator, exactly as in the paper.

## T1 — sensitivity of substantive share (single-factor settings)

Cells = %substantive. `=` marks columns that cannot change the behaviour classification by construction (`tier_cutoff` only feeds tier assignment).

| system | paper | n=3 | n=4 | n=5 | n=6 | n=8 | h=0.05 | h=0.10 | h=0.15 | h=0.20 | h=0.30 | t=0.15 | t=0.20 | t=0.25 | t=0.30 | t=0.40 | l=0.30 | l=0.45 | l=0.60 | a=15 | a=25 | a=50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kraken_seg | 0.975 | 0.992 | 0.992 | 0.975 | 0.951 | 0.885 | 0.992 | 0.975 | 0.959 | 0.943 | 0.803 | = | = | = | = | = | 0.967 | 0.975 | 0.975 | 0.984 | 0.975 | 0.975 |
| kraken_raw | 0.969 | 0.992 | 0.985 | 0.969 | 0.954 | 0.885 | 0.985 | 0.969 | 0.962 | 0.931 | 0.786 | = | = | = | = | = | 0.969 | 0.969 | 0.969 | 0.977 | 0.969 | 0.969 |
| gemini_pro | 0.588 | 1.000 | 0.897 | 0.588 | 0.515 | 0.441 | 0.794 | 0.588 | 0.515 | 0.441 | 0.279 | = | = | = | = | = | 0.588 | 0.588 | 0.588 | 0.588 | 0.588 | 0.588 |
| gemini_flash | 0.426 | 0.803 | 0.672 | 0.426 | 0.344 | 0.262 | 0.525 | 0.426 | 0.328 | 0.262 | 0.180 | = | = | = | = | = | 0.418 | 0.426 | 0.434 | 0.434 | 0.426 | 0.410 |
| claude_opus_4_8 | 0.405 | 0.649 | 0.557 | 0.405 | 0.313 | 0.198 | 0.565 | 0.405 | 0.305 | 0.214 | 0.084 | = | = | = | = | = | 0.405 | 0.405 | 0.405 | 0.405 | 0.405 | 0.397 |
| claude_sonnet_5 | 0.221 | 0.885 | 0.359 | 0.221 | 0.168 | 0.069 | 0.351 | 0.221 | 0.153 | 0.107 | 0.053 | = | = | = | = | = | 0.221 | 0.221 | 0.221 | 0.221 | 0.221 | 0.214 |
| gpt_5_6_sol | 0.275 | 0.489 | 0.374 | 0.275 | 0.252 | 0.176 | 0.321 | 0.275 | 0.237 | 0.198 | 0.137 | = | = | = | = | = | 0.275 | 0.275 | 0.275 | 0.305 | 0.275 | 0.252 |
| qwen3_vl_8b_heb_v17_step800 | 0.664 | 0.931 | 0.840 | 0.664 | 0.534 | 0.366 | 0.817 | 0.664 | 0.527 | 0.420 | 0.214 | = | = | = | = | = | 0.649 | 0.664 | 0.679 | 0.664 | 0.664 | 0.664 |
| qwen3_vl_8b_heb_v16_step1100 | 0.137 | 0.748 | 0.237 | 0.137 | 0.107 | 0.061 | 0.221 | 0.137 | 0.092 | 0.061 | 0.015 | = | = | = | = | = | 0.137 | 0.137 | 0.137 | 0.137 | 0.137 | 0.137 |
| qwen3_vl_8b | 0.015 | 0.031 | 0.023 | 0.015 | 0.015 | 0.008 | 0.023 | 0.015 | 0.015 | 0.008 | 0.000 | = | = | = | = | = | 0.015 | 0.015 | 0.015 | 0.015 | 0.015 | 0.015 |
| vision_ocr_seg | 0.309 | 0.642 | 0.431 | 0.309 | 0.203 | 0.138 | 0.407 | 0.309 | 0.220 | 0.154 | 0.089 | = | = | = | = | = | 0.293 | 0.309 | 0.309 | 0.317 | 0.309 | 0.285 |
| **Tier A count** | 55 | 120 | 83 | 55 | 37 | 24 | 55 | 55 | 55 | 55 | 55 | 80 | 68 | 55 | 46 | 26 | 55 | 55 | 55 | 55 | 55 | 55 |
| **Tier changes vs paper** | 0 | 65 | 28 | 0 | 18 | 31 | 0 | 0 | 0 | 0 | 0 | 25 | 13 | 0 | 9 | 29 | 0 | 0 | 0 | 0 | 0 | 0 |
| C1 kraken_seg %subst >= best frontier | TRUE | FALSE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C2 HebVL-1.7 %subst > best frontier | TRUE | FALSE | FALSE | TRUE | TRUE | FALSE | TRUE | TRUE | TRUE | FALSE | FALSE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C3 gemini_pro %subst = best frontier | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C4 kraken_seg ngram median > every frontier | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C5 HebVL-1.7 ngram median > every frontier | FALSE | FALSE | TRUE | FALSE | FALSE | TRUE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE |

## T2 — Tier A count and claims under the 27-setting n x halluc grid

### unclipped, Hebrew letter set (paper metric family)

| system | unclip n3 h0.05 | unclip n3 h0.10 | unclip n3 h0.20 | unclip n5 h0.05 | unclip n5 h0.10 | unclip n5 h0.20 | unclip n8 h0.05 | unclip n8 h0.10 | unclip n8 h0.20 |
|---|---|---|---|---|---|---|---|---|---|
| kraken_seg | 0.992 | 0.992 | 0.992 | 0.992 | 0.975 | 0.943 | 0.959 | 0.885 | 0.721 |
| kraken_raw | 0.992 | 0.992 | 0.985 | 0.985 | 0.969 | 0.931 | 0.954 | 0.885 | 0.718 |
| gemini_pro | 1.000 | 1.000 | 0.971 | 0.794 | 0.588 | 0.441 | 0.515 | 0.441 | 0.221 |
| gemini_flash | 0.803 | 0.803 | 0.795 | 0.525 | 0.426 | 0.262 | 0.369 | 0.262 | 0.156 |
| claude_opus_4_8 | 0.672 | 0.649 | 0.588 | 0.565 | 0.405 | 0.214 | 0.321 | 0.198 | 0.053 |
| claude_sonnet_5 | 0.893 | 0.885 | 0.626 | 0.351 | 0.221 | 0.107 | 0.160 | 0.069 | 0.038 |
| gpt_5_6_sol | 0.489 | 0.489 | 0.435 | 0.321 | 0.275 | 0.198 | 0.244 | 0.176 | 0.107 |
| qwen3_vl_8b_heb_v17_step800 | 0.939 | 0.931 | 0.916 | 0.817 | 0.664 | 0.420 | 0.534 | 0.366 | 0.130 |
| qwen3_vl_8b_heb_v16_step1100 | 0.802 | 0.748 | 0.527 | 0.221 | 0.137 | 0.061 | 0.099 | 0.061 | 0.015 |
| qwen3_vl_8b | 0.038 | 0.031 | 0.023 | 0.023 | 0.015 | 0.008 | 0.015 | 0.008 | 0.000 |
| vision_ocr_seg | 0.659 | 0.642 | 0.577 | 0.407 | 0.309 | 0.154 | 0.203 | 0.138 | 0.065 |
| **Tier A count** | 120 | 120 | 120 | 55 | 55 | 55 | 24 | 24 | 24 |
| **Tier changes vs paper** | 65 | 65 | 65 | 0 | 0 | 0 | 31 | 31 | 31 |
| C1 kraken_seg %subst >= best frontier | FALSE | FALSE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C2 HebVL-1.7 %subst > best frontier | FALSE | FALSE | FALSE | TRUE | TRUE | FALSE | TRUE | FALSE | FALSE |
| C3 gemini_pro %subst = best frontier | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C4 kraken_seg ngram median > every frontier | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C5 HebVL-1.7 ngram median > every frontier | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | TRUE | TRUE | TRUE |

### BLEU-style clipped counts, Hebrew letter set

| system | clip n3 h0.05 | clip n3 h0.10 | clip n3 h0.20 | clip n5 h0.05 | clip n5 h0.10 | clip n5 h0.20 | clip n8 h0.05 | clip n8 h0.10 | clip n8 h0.20 |
|---|---|---|---|---|---|---|---|---|---|
| kraken_seg | 0.992 | 0.992 | 0.984 | 0.992 | 0.975 | 0.943 | 0.959 | 0.885 | 0.705 |
| kraken_raw | 0.992 | 0.992 | 0.985 | 0.985 | 0.969 | 0.931 | 0.954 | 0.885 | 0.718 |
| gemini_pro | 1.000 | 1.000 | 0.882 | 0.735 | 0.544 | 0.426 | 0.515 | 0.441 | 0.221 |
| gemini_flash | 0.787 | 0.730 | 0.590 | 0.467 | 0.377 | 0.254 | 0.344 | 0.254 | 0.139 |
| claude_opus_4_8 | 0.672 | 0.649 | 0.573 | 0.565 | 0.397 | 0.206 | 0.313 | 0.198 | 0.053 |
| claude_sonnet_5 | 0.893 | 0.878 | 0.489 | 0.344 | 0.206 | 0.092 | 0.153 | 0.069 | 0.038 |
| gpt_5_6_sol | 0.489 | 0.473 | 0.435 | 0.321 | 0.275 | 0.198 | 0.244 | 0.176 | 0.107 |
| qwen3_vl_8b_heb_v17_step800 | 0.725 | 0.718 | 0.695 | 0.672 | 0.557 | 0.359 | 0.473 | 0.328 | 0.122 |
| qwen3_vl_8b_heb_v16_step1100 | 0.496 | 0.443 | 0.290 | 0.198 | 0.137 | 0.061 | 0.099 | 0.061 | 0.015 |
| qwen3_vl_8b | 0.023 | 0.023 | 0.023 | 0.023 | 0.015 | 0.008 | 0.015 | 0.008 | 0.000 |
| vision_ocr_seg | 0.659 | 0.634 | 0.561 | 0.407 | 0.309 | 0.154 | 0.203 | 0.138 | 0.065 |
| **Tier A count** | 108 | 108 | 108 | 50 | 50 | 50 | 21 | 21 | 21 |
| **Tier changes vs paper** | 55 | 55 | 55 | 5 | 5 | 5 | 34 | 34 | 34 |
| C1 kraken_seg %subst >= best frontier | FALSE | FALSE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C2 HebVL-1.7 %subst > best frontier | FALSE | FALSE | FALSE | FALSE | TRUE | FALSE | FALSE | FALSE | FALSE |
| C3 gemini_pro %subst = best frontier | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C4 kraken_seg ngram median > every frontier | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C5 HebVL-1.7 ngram median > every frontier | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE |

### unclipped, semitic (Hebrew+Arabic) letter set

| system | sem n3 h0.05 | sem n3 h0.10 | sem n3 h0.20 | sem n5 h0.05 | sem n5 h0.10 | sem n5 h0.20 | sem n8 h0.05 | sem n8 h0.10 | sem n8 h0.20 |
|---|---|---|---|---|---|---|---|---|---|
| kraken_seg | 0.992 | 0.992 | 0.992 | 0.992 | 0.975 | 0.943 | 0.959 | 0.885 | 0.721 |
| kraken_raw | 0.992 | 0.992 | 0.985 | 0.985 | 0.969 | 0.931 | 0.954 | 0.885 | 0.718 |
| gemini_pro | 1.000 | 1.000 | 0.971 | 0.794 | 0.588 | 0.441 | 0.515 | 0.441 | 0.221 |
| gemini_flash | 0.803 | 0.803 | 0.795 | 0.525 | 0.426 | 0.262 | 0.369 | 0.262 | 0.156 |
| claude_opus_4_8 | 0.664 | 0.618 | 0.550 | 0.550 | 0.382 | 0.206 | 0.298 | 0.183 | 0.053 |
| claude_sonnet_5 | 0.893 | 0.885 | 0.626 | 0.351 | 0.221 | 0.107 | 0.160 | 0.069 | 0.038 |
| gpt_5_6_sol | 0.489 | 0.489 | 0.427 | 0.321 | 0.275 | 0.198 | 0.244 | 0.176 | 0.107 |
| qwen3_vl_8b_heb_v17_step800 | 0.939 | 0.931 | 0.916 | 0.817 | 0.664 | 0.420 | 0.534 | 0.366 | 0.130 |
| qwen3_vl_8b_heb_v16_step1100 | 0.802 | 0.748 | 0.527 | 0.221 | 0.137 | 0.061 | 0.099 | 0.061 | 0.015 |
| qwen3_vl_8b | 0.038 | 0.031 | 0.023 | 0.023 | 0.015 | 0.008 | 0.015 | 0.008 | 0.000 |
| vision_ocr_seg | 0.593 | 0.553 | 0.463 | 0.374 | 0.276 | 0.154 | 0.195 | 0.138 | 0.065 |
| **Tier A count** | 120 | 120 | 120 | 55 | 55 | 55 | 24 | 24 | 24 |
| **Tier changes vs paper** | 65 | 65 | 65 | 0 | 0 | 0 | 31 | 31 | 31 |
| C1 kraken_seg %subst >= best frontier | FALSE | FALSE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C2 HebVL-1.7 %subst > best frontier | FALSE | FALSE | FALSE | TRUE | TRUE | FALSE | TRUE | FALSE | FALSE |
| C3 gemini_pro %subst = best frontier | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C4 kraken_seg ngram median > every frontier | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE |
| C5 HebVL-1.7 ngram median > every frontier | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | TRUE | TRUE | TRUE |

## Claim margins at the paper setting

| claim | subject value | best comparison | comparison system | gap (pp) |
|---|---|---|---|---|
| C1 kraken_seg %subst >= best frontier | 0.975 | 0.588 | gemini_pro | +38.7 |
| C2 HebVL-1.7 %subst > best frontier | 0.664 | 0.588 | gemini_pro | +7.6 |
| C3 gemini_pro %subst = best frontier | 0.588 | 0.426 | gemini_flash | +16.2 |
| C4 kraken_seg ngram median > every frontier | 0.474 | 0.174 | gemini_pro | +30.0 |
| C5 HebVL-1.7 ngram median > every frontier | 0.167 | 0.174 | gemini_pro | -0.7 |

C3's comparison excludes gemini_pro itself (it is a frontier VLM), so the gap is against the runner-up frontier system.

## First hallucination cutoff at which each claim fails

Searched 0.05-0.60 in steps of 0.05, n=5 unclipped, all other thresholds at the paper values.

| claim | holds at paper setting | first failing cutoff |
|---|---|---|
| C1 kraken_seg %subst >= best frontier | TRUE | never fails in range |
| C2 HebVL-1.7 %subst > best frontier | TRUE | 0.20 |
| C3 gemini_pro %subst = best frontier | TRUE | never fails in range |
| C4 kraken_seg ngram median > every frontier | TRUE | never fails in range |
| C5 HebVL-1.7 ngram median > every frontier | FALSE | 0.05 |

C4 and C5 compare medians of n-gram precision, which do not depend on the hallucination cutoff at all: C4 holds and C5 fails at every cutoff, including the paper's.

## Per-setting detail

### paper — `n5 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### n=3 — `n3 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.715 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.000 | 0.992 | 0.711 | 0.730 | 0.356 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.000 | 1.000 | 0.498 | 0.421 | 0.589 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.008 | 0.803 | 0.435 | 0.196 | 0.678 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.046 | 0.649 | 0.305 | 0.366 | 0.505 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.015 | 0.885 | 0.247 | 0.366 | 0.597 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.000 | 0.489 | 0.286 | 0.034 | 0.847 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.031 | 0.931 | 0.480 | 0.553 | 0.488 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.076 | 0.748 | 0.233 | 0.083 | 0.709 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.267 | 0.031 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.211 | 0.642 | 0.262 | 0.108 | 0.834 |

Tier A = **120** (paper = 55); fragments whose tier changed vs paper: **65** (ids in `tier_changes.csv`).

Claims — C1: FALSE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### n=4 — `n4 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.564 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.008 | 0.985 | 0.568 | 0.730 | 0.354 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.103 | 0.897 | 0.249 | 0.421 | 0.532 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.139 | 0.672 | 0.174 | 0.196 | 0.646 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.137 | 0.557 | 0.119 | 0.366 | 0.463 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.542 | 0.359 | 0.071 | 0.366 | 0.480 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.115 | 0.374 | 0.075 | 0.034 | 0.844 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.122 | 0.840 | 0.256 | 0.553 | 0.438 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.588 | 0.237 | 0.053 | 0.083 | 0.419 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.275 | 0.023 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.423 | 0.431 | 0.069 | 0.108 | 0.742 |

Tier A = **83** (paper = 55); fragments whose tier changed vs paper: **28** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: TRUE

### n=5 — `n5 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### n=6 — `n6 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.041 | 0.951 | 0.411 | 0.679 | 0.391 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.038 | 0.954 | 0.411 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.485 | 0.515 | 0.122 | 0.421 | 0.390 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.467 | 0.344 | 0.039 | 0.196 | 0.468 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.382 | 0.313 | 0.039 | 0.366 | 0.397 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.733 | 0.168 | 0.014 | 0.366 | 0.343 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.237 | 0.252 | 0.000 | 0.034 | 0.646 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.427 | 0.534 | 0.118 | 0.553 | 0.363 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.718 | 0.107 | 0.000 | 0.083 | 0.346 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.650 | 0.203 | 0.000 | 0.108 | 0.483 |

Tier A = **37** (paper = 55); fragments whose tier changed vs paper: **18** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### n=8 — `n8 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.107 | 0.885 | 0.320 | 0.679 | 0.361 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.107 | 0.885 | 0.319 | 0.730 | 0.330 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.559 | 0.441 | 0.055 | 0.421 | 0.347 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.549 | 0.262 | 0.017 | 0.196 | 0.385 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.496 | 0.198 | 0.015 | 0.366 | 0.352 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.832 | 0.069 | 0.003 | 0.366 | 0.274 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.313 | 0.176 | 0.000 | 0.034 | 0.646 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.595 | 0.366 | 0.061 | 0.553 | 0.320 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.763 | 0.061 | 0.000 | 0.083 | 0.330 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.290 | 0.008 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.715 | 0.138 | 0.000 | 0.108 | 0.423 |

Tier A = **24** (paper = 55); fragments whose tier changed vs paper: **31** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: TRUE

### h=0.05 — `n5 unclipped hebrew h0.05 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.474 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.008 | 0.985 | 0.476 | 0.730 | 0.354 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.206 | 0.794 | 0.174 | 0.421 | 0.516 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.287 | 0.525 | 0.064 | 0.196 | 0.582 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.130 | 0.565 | 0.065 | 0.366 | 0.466 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.550 | 0.351 | 0.029 | 0.366 | 0.480 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.168 | 0.321 | 0.000 | 0.034 | 0.775 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.145 | 0.817 | 0.167 | 0.553 | 0.429 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.603 | 0.221 | 0.010 | 0.083 | 0.414 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.275 | 0.023 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.447 | 0.407 | 0.024 | 0.108 | 0.732 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### h=0.10 — `n5 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### h=0.15 — `n5 unclipped hebrew h0.15 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.033 | 0.959 | 0.474 | 0.679 | 0.407 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.031 | 0.962 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.485 | 0.515 | 0.174 | 0.421 | 0.390 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.484 | 0.328 | 0.064 | 0.196 | 0.447 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.389 | 0.305 | 0.065 | 0.366 | 0.396 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.748 | 0.153 | 0.029 | 0.366 | 0.338 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.252 | 0.237 | 0.000 | 0.034 | 0.646 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.435 | 0.527 | 0.167 | 0.553 | 0.361 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.733 | 0.092 | 0.010 | 0.083 | 0.330 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.634 | 0.220 | 0.024 | 0.108 | 0.516 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### h=0.20 — `n5 unclipped hebrew h0.20 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.049 | 0.943 | 0.474 | 0.679 | 0.407 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.061 | 0.931 | 0.476 | 0.730 | 0.340 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.559 | 0.441 | 0.174 | 0.421 | 0.347 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.549 | 0.262 | 0.064 | 0.196 | 0.382 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.481 | 0.214 | 0.065 | 0.366 | 0.352 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.794 | 0.107 | 0.029 | 0.366 | 0.284 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.290 | 0.198 | 0.000 | 0.034 | 0.689 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.542 | 0.420 | 0.167 | 0.553 | 0.333 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.763 | 0.061 | 0.010 | 0.083 | 0.323 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.290 | 0.008 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.699 | 0.154 | 0.024 | 0.108 | 0.431 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### h=0.30 — `n5 unclipped hebrew h0.30 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.189 | 0.803 | 0.474 | 0.679 | 0.348 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.206 | 0.786 | 0.476 | 0.730 | 0.305 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.721 | 0.279 | 0.174 | 0.421 | 0.249 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.631 | 0.180 | 0.064 | 0.196 | 0.321 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.611 | 0.084 | 0.065 | 0.366 | 0.261 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.847 | 0.053 | 0.029 | 0.366 | 0.255 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.351 | 0.137 | 0.000 | 0.034 | 0.688 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.748 | 0.214 | 0.167 | 0.553 | 0.279 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.809 | 0.015 | 0.010 | 0.083 | 0.211 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.298 | 0.000 | 0.000 | 0.010 | -- |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.764 | 0.089 | 0.024 | 0.108 | 0.388 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### t=0.15 — `n5 unclipped hebrew h0.10 t0.15 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **80** (paper = 55); fragments whose tier changed vs paper: **25** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### t=0.20 — `n5 unclipped hebrew h0.10 t0.20 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **68** (paper = 55); fragments whose tier changed vs paper: **13** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### t=0.25 — `n5 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### t=0.30 — `n5 unclipped hebrew h0.10 t0.30 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **46** (paper = 55); fragments whose tier changed vs paper: **9** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### t=0.40 — `n5 unclipped hebrew h0.10 t0.40 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **26** (paper = 55); fragments whose tier changed vs paper: **29** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### l=0.30 — `n5 unclipped hebrew h0.10 t0.25 l0.30 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.008 | 0.016 | 0.967 | 0.474 | 0.679 | 0.408 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.098 | 0.369 | 0.418 | 0.064 | 0.196 | 0.541 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.015 | 0.206 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.107 | 0.237 | 0.649 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.336 | 0.519 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.725 | 0.145 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.041 | 0.528 | 0.293 | 0.024 | 0.108 | 0.627 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### l=0.45 — `n5 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### l=0.60 — `n5 unclipped hebrew h0.10 t0.25 l0.60 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.041 | 0.410 | 0.434 | 0.064 | 0.196 | 0.543 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.008 | 0.687 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.015 | 0.298 | 0.679 | 0.167 | 0.553 | 0.385 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.038 | 0.817 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.450 | 0.420 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### a=15 — `n5 unclipped hebrew h0.10 t0.25 l0.45 a15`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.000 | 0.000 | 0.016 | 0.984 | 0.474 | 0.679 | 0.423 |
| kraken_raw | 131 | 0.000 | 0.000 | 0.023 | 0.977 | 0.476 | 0.730 | 0.349 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.098 | 0.074 | 0.393 | 0.434 | 0.064 | 0.196 | 0.543 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.458 | 0.008 | 0.229 | 0.305 | 0.000 | 0.034 | 0.757 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.089 | 0.008 | 0.585 | 0.317 | 0.024 | 0.108 | 0.654 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### a=25 — `n5 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### a=50 — `n5 unclipped hebrew h0.10 t0.25 l0.45 a50`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.139 | 0.074 | 0.377 | 0.410 | 0.064 | 0.196 | 0.533 |
| claude_opus_4_8 | 131 | 0.298 | 0.023 | 0.282 | 0.397 | 0.065 | 0.366 | 0.408 |
| claude_sonnet_5 | 131 | 0.099 | 0.015 | 0.672 | 0.214 | 0.029 | 0.366 | 0.403 |
| gpt_5_6_sol | 131 | 0.588 | 0.008 | 0.153 | 0.252 | 0.000 | 0.034 | 0.646 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.015 | 0.031 | 0.290 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.268 | 0.008 | 0.439 | 0.285 | 0.024 | 0.108 | 0.618 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### unclip n3 h0.05 — `n3 unclipped hebrew h0.05 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.715 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.000 | 0.992 | 0.711 | 0.730 | 0.356 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.000 | 1.000 | 0.498 | 0.421 | 0.589 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.008 | 0.803 | 0.435 | 0.196 | 0.678 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.023 | 0.672 | 0.305 | 0.366 | 0.512 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.008 | 0.893 | 0.247 | 0.366 | 0.598 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.000 | 0.489 | 0.286 | 0.034 | 0.847 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.023 | 0.939 | 0.480 | 0.553 | 0.490 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.023 | 0.802 | 0.233 | 0.083 | 0.731 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.260 | 0.038 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.195 | 0.659 | 0.262 | 0.108 | 0.842 |

Tier A = **120** (paper = 55); fragments whose tier changed vs paper: **65** (ids in `tier_changes.csv`).

Claims — C1: FALSE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### unclip n3 h0.10 — `n3 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.715 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.000 | 0.992 | 0.711 | 0.730 | 0.356 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.000 | 1.000 | 0.498 | 0.421 | 0.589 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.008 | 0.803 | 0.435 | 0.196 | 0.678 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.046 | 0.649 | 0.305 | 0.366 | 0.505 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.015 | 0.885 | 0.247 | 0.366 | 0.597 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.000 | 0.489 | 0.286 | 0.034 | 0.847 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.031 | 0.931 | 0.480 | 0.553 | 0.488 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.076 | 0.748 | 0.233 | 0.083 | 0.709 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.267 | 0.031 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.211 | 0.642 | 0.262 | 0.108 | 0.834 |

Tier A = **120** (paper = 55); fragments whose tier changed vs paper: **65** (ids in `tier_changes.csv`).

Claims — C1: FALSE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### unclip n3 h0.20 — `n3 unclipped hebrew h0.20 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.715 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.008 | 0.985 | 0.711 | 0.730 | 0.354 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.029 | 0.971 | 0.498 | 0.421 | 0.580 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.016 | 0.795 | 0.435 | 0.196 | 0.678 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.107 | 0.588 | 0.305 | 0.366 | 0.478 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.275 | 0.626 | 0.247 | 0.366 | 0.522 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.053 | 0.435 | 0.286 | 0.034 | 0.849 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.046 | 0.916 | 0.480 | 0.553 | 0.483 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.298 | 0.527 | 0.233 | 0.083 | 0.551 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.275 | 0.023 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.276 | 0.577 | 0.262 | 0.108 | 0.808 |

Tier A = **120** (paper = 55); fragments whose tier changed vs paper: **65** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### unclip n5 h0.05 — `n5 unclipped hebrew h0.05 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.474 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.008 | 0.985 | 0.476 | 0.730 | 0.354 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.206 | 0.794 | 0.174 | 0.421 | 0.516 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.287 | 0.525 | 0.064 | 0.196 | 0.582 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.130 | 0.565 | 0.065 | 0.366 | 0.466 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.550 | 0.351 | 0.029 | 0.366 | 0.480 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.168 | 0.321 | 0.000 | 0.034 | 0.775 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.145 | 0.817 | 0.167 | 0.553 | 0.429 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.603 | 0.221 | 0.010 | 0.083 | 0.414 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.275 | 0.023 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.447 | 0.407 | 0.024 | 0.108 | 0.732 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### unclip n5 h0.10 — `n5 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.290 | 0.405 | 0.065 | 0.366 | 0.416 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### unclip n5 h0.20 — `n5 unclipped hebrew h0.20 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.049 | 0.943 | 0.474 | 0.679 | 0.407 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.061 | 0.931 | 0.476 | 0.730 | 0.340 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.559 | 0.441 | 0.174 | 0.421 | 0.347 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.549 | 0.262 | 0.064 | 0.196 | 0.382 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.481 | 0.214 | 0.065 | 0.366 | 0.352 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.794 | 0.107 | 0.029 | 0.366 | 0.284 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.290 | 0.198 | 0.000 | 0.034 | 0.689 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.542 | 0.420 | 0.167 | 0.553 | 0.333 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.763 | 0.061 | 0.010 | 0.083 | 0.323 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.290 | 0.008 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.699 | 0.154 | 0.024 | 0.108 | 0.431 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### unclip n8 h0.05 — `n8 unclipped hebrew h0.05 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.033 | 0.959 | 0.320 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.038 | 0.954 | 0.319 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.485 | 0.515 | 0.055 | 0.421 | 0.390 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.443 | 0.369 | 0.017 | 0.196 | 0.482 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.374 | 0.321 | 0.015 | 0.366 | 0.396 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.740 | 0.160 | 0.003 | 0.366 | 0.340 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.244 | 0.244 | 0.000 | 0.034 | 0.645 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.427 | 0.534 | 0.061 | 0.553 | 0.363 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.725 | 0.099 | 0.000 | 0.083 | 0.336 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.650 | 0.203 | 0.000 | 0.108 | 0.483 |

Tier A = **24** (paper = 55); fragments whose tier changed vs paper: **31** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: TRUE

### unclip n8 h0.10 — `n8 unclipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.107 | 0.885 | 0.320 | 0.679 | 0.361 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.107 | 0.885 | 0.319 | 0.730 | 0.330 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.559 | 0.441 | 0.055 | 0.421 | 0.347 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.549 | 0.262 | 0.017 | 0.196 | 0.385 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.496 | 0.198 | 0.015 | 0.366 | 0.352 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.832 | 0.069 | 0.003 | 0.366 | 0.274 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.313 | 0.176 | 0.000 | 0.034 | 0.646 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.595 | 0.366 | 0.061 | 0.553 | 0.320 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.763 | 0.061 | 0.000 | 0.083 | 0.330 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.290 | 0.008 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.715 | 0.138 | 0.000 | 0.108 | 0.423 |

Tier A = **24** (paper = 55); fragments whose tier changed vs paper: **31** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: TRUE

### unclip n8 h0.20 — `n8 unclipped hebrew h0.20 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.270 | 0.721 | 0.320 | 0.679 | 0.336 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.275 | 0.718 | 0.319 | 0.730 | 0.284 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.779 | 0.221 | 0.055 | 0.421 | 0.229 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.656 | 0.156 | 0.017 | 0.196 | 0.259 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.641 | 0.053 | 0.015 | 0.366 | 0.220 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.863 | 0.038 | 0.003 | 0.366 | 0.252 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.382 | 0.107 | 0.000 | 0.034 | 0.688 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.832 | 0.130 | 0.061 | 0.553 | 0.233 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.809 | 0.015 | 0.000 | 0.083 | 0.211 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.298 | 0.000 | 0.000 | 0.010 | -- |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.789 | 0.065 | 0.000 | 0.108 | 0.365 |

Tier A = **24** (paper = 55); fragments whose tier changed vs paper: **31** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: TRUE

### clip n3 h0.05 — `n3 clipped hebrew h0.05 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.663 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.000 | 0.992 | 0.672 | 0.730 | 0.356 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.000 | 1.000 | 0.411 | 0.421 | 0.589 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.025 | 0.787 | 0.281 | 0.196 | 0.675 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.023 | 0.672 | 0.257 | 0.366 | 0.512 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.008 | 0.893 | 0.196 | 0.366 | 0.598 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.000 | 0.489 | 0.259 | 0.034 | 0.847 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.237 | 0.725 | 0.370 | 0.553 | 0.397 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.328 | 0.496 | 0.064 | 0.083 | 0.538 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.275 | 0.023 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.195 | 0.659 | 0.253 | 0.108 | 0.842 |

Tier A = **108** (paper = 55); fragments whose tier changed vs paper: **55** (ids in `tier_changes.csv`).

Claims — C1: FALSE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### clip n3 h0.10 — `n3 clipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.663 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.000 | 0.992 | 0.672 | 0.730 | 0.356 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.000 | 1.000 | 0.411 | 0.421 | 0.589 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.082 | 0.730 | 0.281 | 0.196 | 0.667 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.046 | 0.649 | 0.257 | 0.366 | 0.505 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.023 | 0.878 | 0.196 | 0.366 | 0.595 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.015 | 0.473 | 0.259 | 0.034 | 0.847 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.244 | 0.718 | 0.370 | 0.553 | 0.394 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.382 | 0.443 | 0.064 | 0.083 | 0.511 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.275 | 0.023 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.220 | 0.634 | 0.253 | 0.108 | 0.833 |

Tier A = **108** (paper = 55); fragments whose tier changed vs paper: **55** (ids in `tier_changes.csv`).

Claims — C1: FALSE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### clip n3 h0.20 — `n3 clipped hebrew h0.20 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.008 | 0.984 | 0.663 | 0.679 | 0.423 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.008 | 0.985 | 0.672 | 0.730 | 0.354 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.118 | 0.882 | 0.411 | 0.421 | 0.531 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.221 | 0.590 | 0.281 | 0.196 | 0.625 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.122 | 0.573 | 0.257 | 0.366 | 0.469 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.412 | 0.489 | 0.196 | 0.366 | 0.492 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.053 | 0.435 | 0.259 | 0.034 | 0.849 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.267 | 0.695 | 0.370 | 0.553 | 0.386 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.534 | 0.290 | 0.064 | 0.083 | 0.463 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.275 | 0.023 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.293 | 0.561 | 0.253 | 0.108 | 0.808 |

Tier A = **108** (paper = 55); fragments whose tier changed vs paper: **55** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### clip n5 h0.05 — `n5 clipped hebrew h0.05 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.467 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.008 | 0.985 | 0.473 | 0.730 | 0.354 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.265 | 0.735 | 0.165 | 0.421 | 0.500 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.344 | 0.467 | 0.051 | 0.196 | 0.552 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.130 | 0.565 | 0.065 | 0.366 | 0.466 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.557 | 0.344 | 0.026 | 0.366 | 0.480 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.168 | 0.321 | 0.000 | 0.034 | 0.775 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.290 | 0.672 | 0.123 | 0.553 | 0.382 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.626 | 0.198 | 0.002 | 0.083 | 0.400 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.275 | 0.023 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.447 | 0.407 | 0.024 | 0.108 | 0.732 |

Tier A = **50** (paper = 55); fragments whose tier changed vs paper: **5** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### clip n5 h0.10 — `n5 clipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.467 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.473 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.456 | 0.544 | 0.165 | 0.421 | 0.396 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.434 | 0.377 | 0.051 | 0.196 | 0.487 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.298 | 0.397 | 0.065 | 0.366 | 0.408 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.695 | 0.206 | 0.026 | 0.366 | 0.372 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.405 | 0.557 | 0.123 | 0.553 | 0.366 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.002 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.545 | 0.309 | 0.024 | 0.108 | 0.646 |

Tier A = **50** (paper = 55); fragments whose tier changed vs paper: **5** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### clip n5 h0.20 — `n5 clipped hebrew h0.20 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.049 | 0.943 | 0.467 | 0.679 | 0.407 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.061 | 0.931 | 0.473 | 0.730 | 0.340 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.574 | 0.426 | 0.165 | 0.421 | 0.346 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.557 | 0.254 | 0.051 | 0.196 | 0.382 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.489 | 0.206 | 0.065 | 0.366 | 0.350 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.809 | 0.092 | 0.026 | 0.366 | 0.274 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.290 | 0.198 | 0.000 | 0.034 | 0.689 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.603 | 0.359 | 0.123 | 0.553 | 0.320 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.763 | 0.061 | 0.002 | 0.083 | 0.323 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.290 | 0.008 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.699 | 0.154 | 0.024 | 0.108 | 0.431 |

Tier A = **50** (paper = 55); fragments whose tier changed vs paper: **5** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### clip n8 h0.05 — `n8 clipped hebrew h0.05 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.033 | 0.959 | 0.315 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.038 | 0.954 | 0.319 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.485 | 0.515 | 0.055 | 0.421 | 0.390 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.467 | 0.344 | 0.010 | 0.196 | 0.465 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.382 | 0.313 | 0.015 | 0.366 | 0.396 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.748 | 0.153 | 0.003 | 0.366 | 0.338 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.244 | 0.244 | 0.000 | 0.034 | 0.645 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.489 | 0.473 | 0.045 | 0.553 | 0.351 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.725 | 0.099 | 0.000 | 0.083 | 0.336 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.650 | 0.203 | 0.000 | 0.108 | 0.483 |

Tier A = **21** (paper = 55); fragments whose tier changed vs paper: **34** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### clip n8 h0.10 — `n8 clipped hebrew h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.107 | 0.885 | 0.315 | 0.679 | 0.361 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.107 | 0.885 | 0.319 | 0.730 | 0.330 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.559 | 0.441 | 0.055 | 0.421 | 0.347 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.557 | 0.254 | 0.010 | 0.196 | 0.382 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.496 | 0.198 | 0.015 | 0.366 | 0.352 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.832 | 0.069 | 0.003 | 0.366 | 0.274 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.313 | 0.176 | 0.000 | 0.034 | 0.646 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.634 | 0.328 | 0.045 | 0.553 | 0.314 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.763 | 0.061 | 0.000 | 0.083 | 0.330 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.290 | 0.008 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.715 | 0.138 | 0.000 | 0.108 | 0.423 |

Tier A = **21** (paper = 55); fragments whose tier changed vs paper: **34** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### clip n8 h0.20 — `n8 clipped hebrew h0.20 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.287 | 0.705 | 0.315 | 0.679 | 0.335 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.275 | 0.718 | 0.319 | 0.730 | 0.284 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.779 | 0.221 | 0.055 | 0.421 | 0.229 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.672 | 0.139 | 0.010 | 0.196 | 0.259 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.641 | 0.053 | 0.015 | 0.366 | 0.220 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.863 | 0.038 | 0.003 | 0.366 | 0.252 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.382 | 0.107 | 0.000 | 0.034 | 0.688 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.840 | 0.122 | 0.045 | 0.553 | 0.228 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.809 | 0.015 | 0.000 | 0.083 | 0.211 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.298 | 0.000 | 0.000 | 0.010 | -- |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.789 | 0.065 | 0.000 | 0.108 | 0.365 |

Tier A = **21** (paper = 55); fragments whose tier changed vs paper: **34** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### sem n3 h0.05 — `n3 unclipped semitic h0.05 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.715 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.000 | 0.992 | 0.711 | 0.730 | 0.356 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.000 | 1.000 | 0.498 | 0.421 | 0.589 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.008 | 0.803 | 0.435 | 0.196 | 0.678 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.031 | 0.664 | 0.282 | 0.366 | 0.509 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.008 | 0.893 | 0.247 | 0.366 | 0.598 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.000 | 0.489 | 0.285 | 0.034 | 0.847 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.023 | 0.939 | 0.480 | 0.553 | 0.490 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.023 | 0.802 | 0.233 | 0.083 | 0.731 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.260 | 0.038 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.260 | 0.593 | 0.197 | 0.108 | 0.828 |

Tier A = **120** (paper = 55); fragments whose tier changed vs paper: **65** (ids in `tier_changes.csv`).

Claims — C1: FALSE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### sem n3 h0.10 — `n3 unclipped semitic h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.715 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.000 | 0.992 | 0.711 | 0.730 | 0.356 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.000 | 1.000 | 0.498 | 0.421 | 0.589 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.008 | 0.803 | 0.435 | 0.196 | 0.678 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.076 | 0.618 | 0.282 | 0.366 | 0.488 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.015 | 0.885 | 0.247 | 0.366 | 0.597 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.000 | 0.489 | 0.285 | 0.034 | 0.847 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.031 | 0.931 | 0.480 | 0.553 | 0.488 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.076 | 0.748 | 0.233 | 0.083 | 0.709 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.267 | 0.031 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.301 | 0.553 | 0.197 | 0.108 | 0.802 |

Tier A = **120** (paper = 55); fragments whose tier changed vs paper: **65** (ids in `tier_changes.csv`).

Claims — C1: FALSE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### sem n3 h0.20 — `n3 unclipped semitic h0.20 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.715 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.008 | 0.985 | 0.711 | 0.730 | 0.354 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.029 | 0.971 | 0.498 | 0.421 | 0.580 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.016 | 0.795 | 0.435 | 0.196 | 0.678 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.145 | 0.550 | 0.282 | 0.366 | 0.462 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.275 | 0.626 | 0.247 | 0.366 | 0.522 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.061 | 0.427 | 0.285 | 0.034 | 0.847 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.046 | 0.916 | 0.480 | 0.553 | 0.483 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.298 | 0.527 | 0.233 | 0.083 | 0.551 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.275 | 0.023 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.390 | 0.463 | 0.197 | 0.108 | 0.750 |

Tier A = **120** (paper = 55); fragments whose tier changed vs paper: **65** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### sem n5 h0.05 — `n5 unclipped semitic h0.05 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.000 | 0.992 | 0.474 | 0.679 | 0.436 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.008 | 0.985 | 0.476 | 0.730 | 0.354 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.206 | 0.794 | 0.174 | 0.421 | 0.516 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.287 | 0.525 | 0.064 | 0.196 | 0.582 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.145 | 0.550 | 0.061 | 0.366 | 0.462 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.550 | 0.351 | 0.029 | 0.366 | 0.480 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.168 | 0.321 | 0.000 | 0.034 | 0.775 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.145 | 0.817 | 0.167 | 0.553 | 0.429 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.603 | 0.221 | 0.010 | 0.083 | 0.414 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.275 | 0.023 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.480 | 0.374 | 0.014 | 0.108 | 0.728 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### sem n5 h0.10 — `n5 unclipped semitic h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.016 | 0.975 | 0.474 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.023 | 0.969 | 0.476 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.412 | 0.588 | 0.174 | 0.421 | 0.432 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.385 | 0.426 | 0.064 | 0.196 | 0.542 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.313 | 0.382 | 0.061 | 0.366 | 0.399 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.679 | 0.221 | 0.029 | 0.366 | 0.435 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.214 | 0.275 | 0.000 | 0.034 | 0.736 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.298 | 0.664 | 0.167 | 0.553 | 0.379 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.687 | 0.137 | 0.010 | 0.083 | 0.373 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.577 | 0.276 | 0.014 | 0.108 | 0.617 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: FALSE

### sem n5 h0.20 — `n5 unclipped semitic h0.20 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.049 | 0.943 | 0.474 | 0.679 | 0.407 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.061 | 0.931 | 0.476 | 0.730 | 0.340 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.559 | 0.441 | 0.174 | 0.421 | 0.347 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.549 | 0.262 | 0.064 | 0.196 | 0.382 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.489 | 0.206 | 0.061 | 0.366 | 0.350 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.794 | 0.107 | 0.029 | 0.366 | 0.284 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.290 | 0.198 | 0.000 | 0.034 | 0.689 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.542 | 0.420 | 0.167 | 0.553 | 0.333 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.763 | 0.061 | 0.010 | 0.083 | 0.323 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.290 | 0.008 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.699 | 0.154 | 0.014 | 0.108 | 0.431 |

Tier A = **55** (paper = 55); fragments whose tier changed vs paper: **0** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: FALSE

### sem n8 h0.05 — `n8 unclipped semitic h0.05 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.033 | 0.959 | 0.318 | 0.679 | 0.410 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.038 | 0.954 | 0.319 | 0.730 | 0.344 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.485 | 0.515 | 0.055 | 0.421 | 0.390 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.443 | 0.369 | 0.017 | 0.196 | 0.482 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.397 | 0.298 | 0.015 | 0.366 | 0.388 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.740 | 0.160 | 0.003 | 0.366 | 0.340 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.244 | 0.244 | 0.000 | 0.034 | 0.645 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.427 | 0.534 | 0.061 | 0.553 | 0.363 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.725 | 0.099 | 0.000 | 0.083 | 0.336 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.282 | 0.015 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.659 | 0.195 | 0.000 | 0.108 | 0.480 |

Tier A = **24** (paper = 55); fragments whose tier changed vs paper: **31** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: TRUE · C3: TRUE · C4: TRUE · C5: TRUE

### sem n8 h0.10 — `n8 unclipped semitic h0.10 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.107 | 0.885 | 0.318 | 0.679 | 0.361 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.107 | 0.885 | 0.319 | 0.730 | 0.330 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.559 | 0.441 | 0.055 | 0.421 | 0.347 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.549 | 0.262 | 0.017 | 0.196 | 0.385 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.511 | 0.183 | 0.015 | 0.366 | 0.341 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.832 | 0.069 | 0.003 | 0.366 | 0.274 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.313 | 0.176 | 0.000 | 0.034 | 0.646 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.595 | 0.366 | 0.061 | 0.553 | 0.320 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.763 | 0.061 | 0.000 | 0.083 | 0.330 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.290 | 0.008 | 0.000 | 0.010 | 0.493 |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.715 | 0.138 | 0.000 | 0.108 | 0.423 |

Tier A = **24** (paper = 55); fragments whose tier changed vs paper: **31** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: TRUE

### sem n8 h0.20 — `n8 unclipped semitic h0.20 t0.25 l0.45 a25`

| system | n | %abst | %loop | %halluc | %subst | ngram med | F1 med | CER med (subst) |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.000 | 0.270 | 0.721 | 0.318 | 0.679 | 0.336 |
| kraken_raw | 131 | 0.008 | 0.000 | 0.275 | 0.718 | 0.319 | 0.730 | 0.284 |
| gemini_pro | 68 | 0.000 | 0.000 | 0.779 | 0.221 | 0.055 | 0.421 | 0.229 |
| gemini_flash | 122 | 0.115 | 0.074 | 0.656 | 0.156 | 0.017 | 0.196 | 0.259 |
| claude_opus_4_8 | 131 | 0.282 | 0.023 | 0.641 | 0.053 | 0.015 | 0.366 | 0.220 |
| claude_sonnet_5 | 131 | 0.084 | 0.015 | 0.863 | 0.038 | 0.003 | 0.366 | 0.252 |
| gpt_5_6_sol | 131 | 0.504 | 0.008 | 0.382 | 0.107 | 0.000 | 0.034 | 0.688 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.008 | 0.031 | 0.832 | 0.130 | 0.061 | 0.553 | 0.233 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.008 | 0.168 | 0.809 | 0.015 | 0.000 | 0.083 | 0.211 |
| qwen3_vl_8b | 131 | 0.115 | 0.588 | 0.298 | 0.000 | 0.000 | 0.010 | -- |
| vision_ocr_seg | 123 | 0.138 | 0.008 | 0.789 | 0.065 | 0.000 | 0.108 | 0.365 |

Tier A = **24** (paper = 55); fragments whose tier changed vs paper: **31** (ids in `tier_changes.csv`).

Claims — C1: TRUE · C2: FALSE · C3: TRUE · C4: TRUE · C5: TRUE

