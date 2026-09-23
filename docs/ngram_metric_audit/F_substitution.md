# Analysis F — simulated in-page substitution vs cross-page hallucination

Scoring: `PAPER_CONFIG` (n5 unclipped hebrew h0.10 t0.25 l0.45 a25).  System: `kraken_seg` (the only system with real per-line segmentation).  Fragments: 55 Tier A fragments whose kraken_seg output is substantive and non-empty.

Hypothesis lines = `Output.raw.splitlines()` with blanks dropped.  Reference pseudo-lines = `Fragment.gt_ink` split on whitespace and dealt into the same number of equal-word chunks.  After perturbation the lines are re-joined with newlines and pushed through `normalize_ink_hypothesis`, i.e. the measurement path is the scorer's.  "Alarm" = the metric moved more than 0.02 in the direction that signals a problem (precision / F1 down, CER / loop ratio up).

## 1. Main setting — middle 30 % of lines replaced by the reference's LAST 20 %

| metric | mean before | mean after | mean Δ | median Δ | n alarmed (Δ>0.02 wrong way) | n moved the other way |
| --- | --- | --- | --- | --- | --- | --- |
| 5-gram precision (unclipped, paper) | 0.554 | 0.697 | 0.142 | 0.136 | 0/55 | 55/55 |
| 5-gram precision (clipped) | 0.539 | 0.520 | -0.019 | -0.021 | 29/55 | 18/55 |
| aligned precision | 0.749 | 0.519 | -0.230 | -0.256 | 48/55 | 5/55 |
| aligned recall | 0.628 | 0.424 | -0.204 | -0.215 | 44/55 | 8/55 |
| aligned F1 | 0.674 | 0.463 | -0.211 | -0.228 | 45/55 | 7/55 |
| CER strict | 0.415 | 0.521 | 0.107 | 0.119 | 42/55 | 8/55 |
| CER lenient | 0.413 | 0.520 | 0.107 | 0.120 | 42/55 | 8/55 |
| loop ratio (asserted) | 0.024 | 0.047 | 0.023 | 0.021 | 28/55 | 1/55 |

## 2. Variants (aggregate only)

### in-page, donor = FIRST 20 %

| metric | mean before | mean after | mean Δ | median Δ | n alarmed (Δ>0.02 wrong way) | n moved the other way |
| --- | --- | --- | --- | --- | --- | --- |
| 5-gram precision (unclipped, paper) | 0.554 | 0.712 | 0.158 | 0.154 | 0/55 | 55/55 |
| 5-gram precision (clipped) | 0.539 | 0.491 | -0.048 | -0.062 | 36/55 | 15/55 |
| aligned precision | 0.749 | 0.493 | -0.256 | -0.294 | 49/55 | 5/55 |
| aligned recall | 0.628 | 0.421 | -0.206 | -0.236 | 45/55 | 8/55 |
| aligned F1 | 0.674 | 0.451 | -0.223 | -0.269 | 46/55 | 6/55 |
| CER strict | 0.415 | 0.535 | 0.121 | 0.139 | 45/55 | 7/55 |
| CER lenient | 0.413 | 0.534 | 0.122 | 0.140 | 45/55 | 7/55 |
| loop ratio (asserted) | 0.024 | 0.047 | 0.022 | 0.024 | 31/55 | 2/55 |

### in-page, 15 % of lines

| metric | mean before | mean after | mean Δ | median Δ | n alarmed (Δ>0.02 wrong way) | n moved the other way |
| --- | --- | --- | --- | --- | --- | --- |
| 5-gram precision (unclipped, paper) | 0.554 | 0.630 | 0.075 | 0.066 | 0/55 | 52/55 |
| 5-gram precision (clipped) | 0.539 | 0.559 | 0.020 | 0.016 | 17/55 | 27/55 |
| aligned precision | 0.749 | 0.567 | -0.182 | -0.199 | 44/55 | 7/55 |
| aligned recall | 0.628 | 0.468 | -0.160 | -0.169 | 43/55 | 10/55 |
| aligned F1 | 0.674 | 0.508 | -0.167 | -0.173 | 43/55 | 10/55 |
| CER strict | 0.415 | 0.469 | 0.055 | 0.062 | 39/55 | 6/55 |
| CER lenient | 0.413 | 0.468 | 0.055 | 0.063 | 39/55 | 6/55 |
| loop ratio (asserted) | 0.024 | 0.035 | 0.011 | 0.012 | 10/55 | 0/55 |

### in-page, 50 % of lines

| metric | mean before | mean after | mean Δ | median Δ | n alarmed (Δ>0.02 wrong way) | n moved the other way |
| --- | --- | --- | --- | --- | --- | --- |
| 5-gram precision (unclipped, paper) | 0.554 | 0.792 | 0.238 | 0.230 | 0/55 | 55/55 |
| 5-gram precision (clipped) | 0.539 | 0.413 | -0.126 | -0.133 | 43/55 | 6/55 |
| aligned precision | 0.749 | 0.427 | -0.322 | -0.338 | 52/55 | 2/55 |
| aligned recall | 0.628 | 0.352 | -0.276 | -0.324 | 48/55 | 6/55 |
| aligned F1 | 0.674 | 0.384 | -0.290 | -0.330 | 48/55 | 5/55 |
| CER strict | 0.415 | 0.574 | 0.159 | 0.188 | 42/55 | 12/55 |
| CER lenient | 0.413 | 0.573 | 0.161 | 0.189 | 42/55 | 12/55 |
| loop ratio (asserted) | 0.024 | 0.060 | 0.036 | 0.034 | 43/55 | 1/55 |

### cross-page — donor pseudo-lines come from a DIFFERENT Tier A page (classic cross-page hallucination; this is what the n-gram guard was designed to catch)

| metric | mean before | mean after | mean Δ | median Δ | n alarmed (Δ>0.02 wrong way) | n moved the other way |
| --- | --- | --- | --- | --- | --- | --- |
| 5-gram precision (unclipped, paper) | 0.554 | 0.358 | -0.197 | -0.178 | 55/55 | 0/55 |
| 5-gram precision (clipped) | 0.539 | 0.342 | -0.197 | -0.178 | 55/55 | 0/55 |
| aligned precision | 0.749 | 0.535 | -0.214 | -0.204 | 50/55 | 1/55 |
| aligned recall | 0.628 | 0.463 | -0.164 | -0.164 | 51/55 | 1/55 |
| aligned F1 | 0.674 | 0.482 | -0.193 | -0.185 | 50/55 | 1/55 |
| CER strict | 0.415 | 0.653 | 0.238 | 0.182 | 49/55 | 3/55 |
| CER lenient | 0.413 | 0.652 | 0.239 | 0.183 | 49/55 | 3/55 |
| loop ratio (asserted) | 0.024 | 0.038 | 0.014 | 0.013 | 12/55 | 0/55 |

## 3. n-gram precision at n = 3 / 5 / 8 — baseline vs in-page vs cross-page

Both perturbations replace the same 30 % band of lines; only the donor page differs.

| n | counting | baseline mean | in-page mean | in-page Δ | cross-page mean | cross-page Δ |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | unclipped | 0.753 | 0.829 | 0.076 | 0.562 | -0.191 |
| 3 | clipped | 0.708 | 0.645 | -0.063 | 0.488 | -0.220 |
| 5 | unclipped | 0.554 | 0.697 | 0.142 | 0.358 | -0.197 |
| 5 | clipped | 0.539 | 0.520 | -0.019 | 0.342 | -0.197 |
| 8 | unclipped | 0.410 | 0.597 | 0.187 | 0.260 | -0.150 |
| 8 | clipped | 0.401 | 0.441 | 0.040 | 0.252 | -0.149 |

## 4. Insertion simulation — the reference's closing pseudo-line emitted once more mid-page (nothing removed)

| metric | mean before | mean after | mean Δ | median Δ | n alarmed (Δ>0.02 wrong way) | n moved the other way |
| --- | --- | --- | --- | --- | --- | --- |
| 5-gram precision (unclipped, paper) | 0.554 | 0.565 | 0.011 | 0.006 | 0/55 | 6/55 |
| 5-gram precision (clipped) | 0.539 | 0.543 | 0.004 | 0.002 | 2/55 | 3/55 |
| aligned precision | 0.749 | 0.675 | -0.074 | -0.020 | 28/55 | 0/55 |
| aligned recall | 0.628 | 0.587 | -0.041 | 0.000 | 14/55 | 0/55 |
| aligned F1 | 0.674 | 0.622 | -0.053 | -0.009 | 15/55 | 0/55 |
| CER strict | 0.415 | 0.428 | 0.014 | 0.013 | 19/55 | 2/55 |
| CER lenient | 0.413 | 0.426 | 0.014 | 0.013 | 19/55 | 2/55 |
| loop ratio (asserted) | 0.024 | 0.026 | 0.002 | -0.000 | 2/55 | 0/55 |

## 5. Do the paper's own gates ever fire?

Counts of fragments whose unclipped n=5 precision falls below the hallucination cutoff / below the Tier-A evidence cutoff.

| setting | hallucinated before (<0.10) | hallucinated after | below Tier-A cutoff before (<0.25) | below Tier-A cutoff after |
| --- | --- | --- | --- | --- |
| in-page | 0/55 | 0/55 | 0/55 | 0/55 |
| in-page, donor = FIRST 20 % | 0/55 | 0/55 | 0/55 | 0/55 |
| in-page, 15 % of lines | 0/55 | 0/55 | 0/55 | 0/55 |
| in-page, 50 % of lines | 0/55 | 0/55 | 0/55 | 0/55 |
| cross-page | 0/55 | 0/55 | 0/55 | 14/55 |
| insertion (closing formula mid-page) | 0/55 | 0/55 | 0/55 | 0/55 |

## 6. Per-fragment table (main setting)

| fragment | lines | subst. | 5g uncl. before | 5g uncl. after | 5g clip. before | 5g clip. after | F1 before | F1 after | CER before | CER after | loop before | loop after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cambridge_CUL_Or_1080_J262 | 54 | 17 | 0.649 | 0.689 | 0.647 | 0.529 | 0.806 | 0.505 | 0.260 | 0.518 | 0.034 | 0.044 |
| Cambridge_CUL_T_S_10J12_9 | 21 | 6 | 0.791 | 0.872 | 0.791 | 0.612 | 0.948 | 0.452 | 0.066 | 0.266 | 0.016 | 0.049 |
| Cambridge_CUL_T_S_10J13_11 | 48 | 15 | 0.584 | 0.720 | 0.584 | 0.582 | 0.354 | 0.492 | 0.472 | 0.572 | 0.025 | 0.072 |
| Cambridge_CUL_T_S_10J14_1 | 27 | 8 | 0.564 | 0.651 | 0.562 | 0.540 | 0.865 | 0.672 | 0.167 | 0.311 | 0.020 | 0.061 |
| Cambridge_CUL_T_S_10J16_19 | 27 | 8 | 0.749 | 0.813 | 0.744 | 0.614 | 0.839 | 0.410 | 0.269 | 0.458 | 0.013 | 0.039 |
| Cambridge_CUL_T_S_10J21_1 | 42 | 13 | 0.556 | 0.666 | 0.383 | 0.444 | 0.417 | 0.328 | 0.868 | 0.956 | 0.059 | 0.055 |
| Cambridge_CUL_T_S_10J2_3 | 51 | 16 | 0.340 | 0.614 | 0.340 | 0.414 | 0.565 | 0.369 | 0.558 | 0.591 | 0.027 | 0.067 |
| Cambridge_CUL_T_S_10J4_10 | 35 | 10 | 0.685 | 0.771 | 0.684 | 0.603 | 0.838 | 0.661 | 0.241 | 0.368 | 0.030 | 0.048 |
| Cambridge_CUL_T_S_10J4_12 | 38 | 11 | 0.677 | 0.716 | 0.675 | 0.545 | 0.843 | 0.519 | 0.244 | 0.552 | 0.019 | 0.052 |
| Cambridge_CUL_T_S_10J5_19 | 39 | 12 | 0.309 | 0.600 | 0.307 | 0.450 | 0.647 | 0.579 | 0.480 | 0.522 | 0.018 | 0.034 |
| Cambridge_CUL_T_S_12_3 | 38 | 11 | 0.501 | 0.647 | 0.493 | 0.469 | 0.823 | 0.457 | 0.250 | 0.458 | 0.008 | 0.027 |
| Cambridge_CUL_T_S_13J13_16 | 51 | 16 | 0.664 | 0.728 | 0.660 | 0.562 | 0.857 | 0.564 | 0.208 | 0.378 | 0.014 | 0.030 |
| Cambridge_CUL_T_S_13J14_7 | 38 | 11 | 0.742 | 0.822 | 0.741 | 0.611 | 0.732 | 0.341 | 0.337 | 0.428 | 0.016 | 0.046 |
| Cambridge_CUL_T_S_13J17_12 | 36 | 11 | 0.396 | 0.582 | 0.394 | 0.464 | 0.660 | 0.512 | 0.466 | 0.530 | 0.013 | 0.023 |
| Cambridge_CUL_T_S_13J23_7 | 12 | 3 | 0.505 | 0.594 | 0.503 | 0.445 | 0.819 | 0.359 | 0.232 | 0.336 | 0.022 | 0.063 |
| Cambridge_CUL_T_S_13J26_14 | 43 | 12 | 0.316 | 0.509 | 0.316 | 0.408 | 0.727 | 0.529 | 0.363 | 0.539 | 0.016 | 0.058 |
| Cambridge_CUL_T_S_13J26_15 | 29 | 8 | 0.688 | 0.813 | 0.681 | 0.678 | 0.823 | 0.439 | 0.275 | 0.403 | 0.037 | 0.054 |
| Cambridge_CUL_T_S_13J2_14 | 26 | 7 | 0.446 | 0.597 | 0.444 | 0.458 | 0.744 | 0.360 | 0.335 | 0.436 | 0.016 | 0.031 |
| Cambridge_CUL_T_S_13J5_3 | 30 | 9 | 0.520 | 0.731 | 0.520 | 0.644 | 0.502 | 0.415 | 0.636 | 0.627 | 0.020 | 0.073 |
| Cambridge_CUL_T_S_13J6_23 | 38 | 11 | 0.469 | 0.611 | 0.467 | 0.442 | 0.749 | 0.412 | 0.359 | 0.477 | 0.024 | 0.064 |
| Cambridge_CUL_T_S_13J8_2 | 49 | 14 | 0.520 | 0.672 | 0.520 | 0.503 | 0.766 | 0.412 | 0.344 | 0.482 | 0.026 | 0.039 |
| Cambridge_CUL_T_S_13J8_6 | 27 | 8 | 0.670 | 0.792 | 0.669 | 0.653 | 0.781 | 0.463 | 0.332 | 0.482 | 0.024 | 0.022 |
| Cambridge_CUL_T_S_16_117 | 23 | 6 | 0.398 | 0.742 | 0.370 | 0.661 | 0.222 | 0.343 | 0.819 | 0.739 | 0.047 | 0.038 |
| Cambridge_CUL_T_S_16_124 | 43 | 12 | 0.618 | 0.723 | 0.615 | 0.508 | 0.847 | 0.438 | 0.212 | 0.536 | 0.011 | 0.045 |
| Cambridge_CUL_T_S_16_125 | 75 | 22 | 0.465 | 0.617 | 0.465 | 0.442 | 0.674 | 0.380 | 0.487 | 0.513 | 0.013 | 0.040 |
| Cambridge_CUL_T_S_16_128 | 26 | 7 | 0.415 | 0.601 | 0.414 | 0.393 | 0.801 | 0.493 | 0.256 | 0.430 | 0.015 | 0.051 |
| Cambridge_CUL_T_S_16_138 | 32 | 9 | 0.577 | 0.808 | 0.492 | 0.415 | 0.218 | 0.309 | 0.798 | 0.748 | 0.053 | 0.041 |
| Cambridge_CUL_T_S_16_170 | 32 | 9 | 0.746 | 0.847 | 0.746 | 0.824 | 0.496 | 0.485 | 0.683 | 0.622 | 0.016 | 0.025 |
| Cambridge_CUL_T_S_16_256 | 36 | 11 | 0.686 | 0.746 | 0.684 | 0.503 | 0.886 | 0.523 | 0.160 | 0.392 | 0.015 | 0.049 |
| Cambridge_CUL_T_S_16_307 | 32 | 9 | 0.541 | 0.617 | 0.538 | 0.429 | 0.758 | 0.530 | 0.410 | 0.581 | 0.015 | 0.033 |
| Cambridge_CUL_T_S_18J1_14 | 32 | 9 | 0.457 | 0.600 | 0.453 | 0.483 | 0.793 | 0.598 | 0.271 | 0.387 | 0.021 | 0.041 |
| Cambridge_CUL_T_S_18J1_32 | 44 | 13 | 0.744 | 0.821 | 0.740 | 0.666 | 0.560 | 0.559 | 0.628 | 0.623 | 0.026 | 0.042 |
| Cambridge_CUL_T_S_18J2_16 | 35 | 10 | 0.606 | 0.706 | 0.605 | 0.545 | 0.895 | 0.491 | 0.136 | 0.323 | 0.009 | 0.029 |
| Cambridge_CUL_T_S_20_110 | 37 | 12 | 0.671 | 0.747 | 0.670 | 0.568 | 0.837 | 0.481 | 0.233 | 0.505 | 0.022 | 0.039 |
| Cambridge_CUL_T_S_20_21 | 36 | 11 | 0.717 | 0.799 | 0.709 | 0.548 | 0.902 | 0.542 | 0.165 | 0.337 | 0.012 | 0.026 |
| Cambridge_CUL_T_S_20_4 | 13 | 4 | 0.549 | 0.876 | 0.512 | 0.600 | 0.253 | 0.392 | 0.830 | 0.708 | 0.029 | 0.042 |
| Cambridge_CUL_T_S_20_63 | 43 | 12 | 0.469 | 0.719 | 0.467 | 0.550 | 0.235 | 0.271 | 0.756 | 0.728 | 0.016 | 0.045 |
| Cambridge_CUL_T_S_20_98 | 41 | 12 | 0.375 | 0.569 | 0.371 | 0.408 | 0.723 | 0.509 | 0.367 | 0.466 | 0.013 | 0.044 |
| Cambridge_CUL_T_S_24_13 | 75 | 22 | 0.538 | 0.679 | 0.534 | 0.514 | 0.729 | 0.662 | 0.358 | 0.399 | 0.017 | 0.024 |
| Cambridge_CUL_T_S_8J24_6 | 35 | 10 | 0.636 | 0.701 | 0.636 | 0.524 | 0.792 | 0.420 | 0.352 | 0.529 | 0.025 | 0.078 |
| Cambridge_CUL_T_S_8J4_18 | 30 | 9 | 0.727 | 0.792 | 0.724 | 0.582 | 0.762 | 0.516 | 0.336 | 0.521 | 0.018 | 0.056 |
| Cambridge_CUL_T_S_8_143 | 57 | 18 | 0.433 | 0.686 | 0.433 | 0.543 | 0.613 | 0.580 | 0.529 | 0.534 | 0.019 | 0.036 |
| Cambridge_CUL_T_S_AS_147_38 | 59 | 18 | 0.312 | 0.519 | 0.312 | 0.400 | 0.602 | 0.430 | 0.601 | 0.726 | 0.036 | 0.042 |
| Cambridge_CUL_T_S_Misc_35_11 | 34 | 11 | 0.492 | 0.631 | 0.269 | 0.380 | 0.460 | 0.330 | 0.782 | 0.710 | 0.025 | 0.031 |
| Cambridge_Mosseri_VII_67_2 | 47 | 14 | 0.430 | 0.630 | 0.430 | 0.484 | 0.438 | 0.492 | 0.684 | 0.673 | 0.033 | 0.101 |
| Cambridge_Mosseri_V_355 | 51 | 16 | 0.500 | 0.625 | 0.498 | 0.492 | 0.758 | 0.608 | 0.344 | 0.463 | 0.013 | 0.027 |
| Manchester_JRL_A_960 | 15 | 4 | 0.740 | 0.803 | 0.733 | 0.670 | 0.858 | 0.611 | 0.219 | 0.400 | 0.054 | 0.091 |
| New_York_JTS_ENA_NS_50_32 | 37 | 12 | 0.461 | 0.651 | 0.461 | 0.446 | 0.741 | 0.451 | 0.357 | 0.511 | 0.011 | 0.036 |
| Oxford_Bodleian_MS_heb_a_2_4 | 32 | 9 | 0.463 | 0.633 | 0.264 | 0.364 | 0.183 | 0.256 | 0.765 | 0.743 | 0.129 | 0.104 |
| Oxford_Bodleian_MS_heb_b_11_28 | 36 | 11 | 0.735 | 0.782 | 0.735 | 0.574 | 0.889 | 0.575 | 0.177 | 0.360 | 0.016 | 0.048 |
| Oxford_Bodleian_MS_heb_b_11_7 | 45 | 14 | 0.606 | 0.789 | 0.606 | 0.617 | 0.419 | 0.256 | 0.732 | 0.695 | 0.016 | 0.022 |
| Oxford_Bodleian_MS_heb_c_28_66 | 20 | 6 | 0.755 | 0.818 | 0.750 | 0.539 | 0.935 | 0.441 | 0.095 | 0.355 | 0.012 | 0.040 |
| Oxford_Bodleian_MS_heb_d_66_32 | 47 | 14 | 0.522 | 0.651 | 0.522 | 0.480 | 0.723 | 0.566 | 0.345 | 0.441 | 0.021 | 0.063 |
| Oxford_Bodleian_MS_heb_d_68_101 | 74 | 23 | 0.331 | 0.538 | 0.331 | 0.369 | 0.335 | 0.316 | 0.700 | 0.693 | 0.025 | 0.049 |
| Oxford_Bodleian_MS_heb_d_74_40 | 58 | 17 | 0.444 | 0.640 | 0.444 | 0.413 | 0.645 | 0.381 | 0.452 | 0.579 | 0.029 | 0.080 |

## Files

- `per_fragment.csv` — every metric, before / after / delta, main setting.
- `examples/` — before/after normalised text for the 5 fragments with the largest CER rise: Cambridge_CUL_T_S_16_124, Cambridge_CUL_T_S_10J4_12, Cambridge_CUL_T_S_20_110, Oxford_Bodleian_MS_heb_c_28_66, Cambridge_CUL_Or_1080_J262.
