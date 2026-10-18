# Analysis E — agreement of the 5-gram precision with other signals and with the LLM judge

Config: `n5 unclipped hebrew h0.10 t0.25 l0.45 a25` (paper reproduction). 1352 outputs = 131 verified fragments x 11 paper systems (Tier A fragments: 55). scipy 1.15.3 was available in `.venv`, so Spearman uses `scipy.stats.spearmanr` (tie-corrected); AUC and kappa are implemented with numpy in this script.

## Part 1 — correlation with the alignment metrics

`cer_lenient` correlations are expected negative (higher n-gram precision = lower CER).

### All outputs

| system | n | rho(aligned P) | r(aligned P) | rho(aligned F1) | r(aligned F1) | rho(CER-lenient) | r(CER-lenient) |
|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.622 | 0.504 | 0.644 | 0.604 | -0.605 | -0.603 |
| kraken_raw | 131 | 0.816 | 0.727 | 0.809 | 0.791 | -0.777 | -0.782 |
| gemini_pro | 68 | 0.821 | 0.820 | 0.857 | 0.875 | -0.798 | -0.847 |
| gemini_flash | 122 | 0.655 | 0.597 | 0.693 | 0.708 | -0.652 | -0.651 |
| claude_opus_4_8 | 131 | 0.833 | 0.740 | 0.831 | 0.722 | -0.722 | -0.568 |
| claude_sonnet_5 | 131 | 0.762 | 0.680 | 0.837 | 0.698 | -0.681 | -0.409 |
| gpt_5_6_sol | 131 | 0.716 | 0.606 | 0.795 | 0.516 | -0.773 | -0.477 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.762 | 0.612 | 0.757 | 0.647 | -0.695 | -0.353 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.802 | 0.703 | 0.777 | 0.721 | -0.742 | -0.368 |
| qwen3_vl_8b | 131 | 0.187 | 0.817 | 0.197 | 0.829 | 0.154 | -0.044 |
| vision_ocr_seg | 123 | 0.692 | 0.613 | 0.739 | 0.818 | -0.733 | -0.822 |
| POOLED | 1352 | 0.802 | 0.708 | 0.833 | 0.752 | -0.768 | -0.254 |

### Tier A outputs only

| system | n | rho(aligned P) | r(aligned P) | rho(aligned F1) | r(aligned F1) | rho(CER-lenient) | r(CER-lenient) |
|---|---|---|---|---|---|---|---|
| kraken_seg | 55 | 0.561 | 0.401 | 0.538 | 0.411 | -0.515 | -0.428 |
| kraken_raw | 55 | 0.789 | 0.765 | 0.696 | 0.604 | -0.658 | -0.555 |
| gemini_pro | 31 | 0.935 | 0.846 | 0.913 | 0.839 | -0.886 | -0.839 |
| gemini_flash | 54 | 0.548 | 0.528 | 0.665 | 0.673 | -0.666 | -0.664 |
| claude_opus_4_8 | 55 | 0.870 | 0.805 | 0.903 | 0.825 | -0.677 | -0.557 |
| claude_sonnet_5 | 55 | 0.784 | 0.697 | 0.769 | 0.666 | -0.597 | -0.342 |
| gpt_5_6_sol | 55 | 0.792 | 0.656 | 0.631 | 0.387 | -0.603 | -0.356 |
| qwen3_vl_8b_heb_v17_step800 | 55 | 0.716 | 0.463 | 0.650 | 0.477 | -0.601 | -0.245 |
| qwen3_vl_8b_heb_v16_step1100 | 55 | 0.954 | 0.755 | 0.933 | 0.774 | -0.899 | -0.485 |
| qwen3_vl_8b | 55 | 0.301 | 0.820 | 0.322 | 0.835 | 0.172 | -0.067 |
| vision_ocr_seg | 55 | 0.696 | 0.642 | 0.805 | 0.857 | -0.810 | -0.866 |
| POOLED | 580 | 0.814 | 0.705 | 0.803 | 0.707 | -0.734 | -0.289 |

Tie caveat: systems whose n-gram precision is exactly 0 for most outputs have a near-degenerate rank vector, so Spearman collapses while Pearson stays high (the zeros still line up with low alignment scores). Share of outputs at exactly 0: qwen3_vl_8b 90.1%, gpt_5_6_sol 51.9%, vision_ocr_seg 44.7%.

## Part 2 — disagreement lists

**List (a)** n-gram >= 0.25 but aligned P <= 0.5: **33** outputs (2.4% of all).

**List (b)** n-gram < 0.10 but aligned P >= 0.5: **139** outputs (10.3% of all).

### List (a) — per-system counts

| system | count | share of that system's outputs |
|---|---|---|
| kraken_seg | 15 | 0.123 |
| kraken_raw | 1 | 0.008 |
| gemini_pro | 3 | 0.044 |
| gemini_flash | 4 | 0.033 |
| claude_opus_4_8 | 2 | 0.015 |
| claude_sonnet_5 | 1 | 0.008 |
| gpt_5_6_sol | 0 | 0.000 |
| qwen3_vl_8b_heb_v17_step800 | 7 | 0.053 |
| qwen3_vl_8b_heb_v16_step1100 | 0 | 0.000 |
| qwen3_vl_8b | 0 | 0.000 |
| vision_ocr_seg | 0 | 0.000 |
| TOTAL | 33 | 0.024 |

### List (b) — per-system counts

| system | count | share of that system's outputs |
|---|---|---|
| kraken_seg | 1 | 0.008 |
| kraken_raw | 1 | 0.008 |
| gemini_pro | 5 | 0.074 |
| gemini_flash | 6 | 0.049 |
| claude_opus_4_8 | 13 | 0.099 |
| claude_sonnet_5 | 30 | 0.229 |
| gpt_5_6_sol | 28 | 0.214 |
| qwen3_vl_8b_heb_v17_step800 | 9 | 0.069 |
| qwen3_vl_8b_heb_v16_step1100 | 17 | 0.130 |
| qwen3_vl_8b | 1 | 0.008 |
| vision_ocr_seg | 28 | 0.228 |
| TOTAL | 139 | 0.103 |

### List (a) rows

| doc_id | model | ngram | aligned P | aligned R | CER-len | len ratio | tier | script | failure_mode |
|---|---|---|---|---|---|---|---|---|---|
| Cambridge_CUL_T_S_20_28 | kraken_seg | 0.691 | 0.401 | 0.298 | 0.656 | 0.33 | B | judaeo_arabic | substantive |
| Oxford_Bodleian_MS_heb_a_2_4 | gemini_flash | 0.605 | 0.189 | 0.153 | 0.737 | 0.57 | A | aramaic | substantive |
| Cambridge_CUL_T_S_10J13_11 | kraken_seg | 0.584 | 0.430 | 0.300 | 0.470 | 0.69 | A | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_10J24_3 | kraken_seg | 0.580 | 0.319 | 0.267 | 0.684 | 0.82 | B | hebrew | substantive |
| Cambridge_CUL_T_S_16_138 | kraken_seg | 0.577 | 0.316 | 0.167 | 0.798 | 0.28 | A | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_10J21_1 | kraken_seg | 0.556 | 0.367 | 0.483 | 0.862 | 0.85 | A | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_20_21 | gemini_flash | 0.553 | 0.185 | 0.010 | 0.969 | 0.03 | A | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_16_106 | claude_opus_4_8 | 0.545 | 0.135 | 0.076 | 0.902 | 0.02 | B | hebrew | substantive |
| Cambridge_CUL_T_S_12_146 | gemini_flash | 0.517 | 0.162 | 0.141 | 0.782 | 0.45 | B | hebrew | substantive |
| Cambridge_CUL_T_S_Misc_35_11 | kraken_seg | 0.492 | 0.420 | 0.509 | 0.782 | 1.11 | A | hebrew | substantive |
| Cambridge_CUL_T_S_20_63 | kraken_seg | 0.469 | 0.330 | 0.183 | 0.750 | 0.42 | A | judaeo_arabic | substantive |
| Oxford_Bodleian_MS_heb_a_2_4 | kraken_seg | 0.463 | 0.210 | 0.162 | 0.765 | 0.44 | A | aramaic | substantive |
| New_York_JTS_ENA_NS_50_32 | gemini_flash | 0.443 | 0.066 | 0.049 | 0.737 | 0.47 | A | hebrew | loop_collapse |
| Cambridge_CUL_T_S_16_125 | qwen3_vl_8b_heb_v17_step800 | 0.419 | 0.024 | 0.340 | 13.349 | 14.48 | A | untagged | loop_collapse |
| Cambridge_CUL_T_S_6J3_2 | kraken_seg | 0.418 | 0.494 | 0.328 | 0.669 | 0.64 | B | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_16_117 | kraken_seg | 0.398 | 0.474 | 0.145 | 0.819 | 0.23 | A | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_16_138 | qwen3_vl_8b_heb_v17_step800 | 0.390 | 0.069 | 0.575 | 7.378 | 8.50 | A | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_20_162 | kraken_seg | 0.390 | 0.455 | 0.268 | 0.593 | 0.60 | B | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_20_162 | kraken_raw | 0.389 | 0.452 | 0.268 | 0.594 | 0.60 | B | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_AS_149_9 | kraken_seg | 0.354 | 0.315 | 0.193 | 0.660 | 0.60 | B | untagged | substantive |
| New_York_JTS_ENA_NS_50_32 | claude_opus_4_8 | 0.341 | 0.387 | 0.643 | 1.084 | 1.51 | A | hebrew | substantive |
| Oxford_Bodleian_MS_heb_d_68_101 | kraken_seg | 0.331 | 0.380 | 0.300 | 0.700 | 0.74 | A | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_13J17_12 | qwen3_vl_8b_heb_v17_step800 | 0.325 | 0.009 | 0.084 | 8.378 | 9.35 | A | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_8_4 | kraken_seg | 0.322 | 0.282 | 0.217 | 0.659 | 0.74 | B | judaeo_arabic | substantive |
| Oxford_Bodleian_MS_heb_d_66_32 | qwen3_vl_8b_heb_v17_step800 | 0.314 | 0.027 | 0.581 | 20.708 | 21.15 | A | untagged | substantive |
| Cambridge_CUL_T_S_20_4 | qwen3_vl_8b_heb_v17_step800 | 0.308 | 0.128 | 0.415 | 2.544 | 3.29 | A | judaeo_arabic | substantive |
| Oxford_Bodleian_MS_heb_a_2_4 | claude_sonnet_5 | 0.304 | 0.355 | 0.673 | 1.295 | 1.68 | A | aramaic | substantive |
| Oxford_Bodleian_MS_heb_b_11_7 | gemini_pro | 0.301 | 0.429 | 0.079 | 0.897 | 0.10 | A | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_16_332 | gemini_pro | 0.289 | 0.427 | 0.348 | 0.640 | 0.82 | B | untagged | substantive |
| Oxford_Bodleian_MS_heb_f_108_63 | kraken_seg | 0.287 | 0.086 | 0.067 | 0.675 | 0.76 | B | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_16_110 | gemini_pro | 0.284 | 0.491 | 0.373 | 0.611 | 0.77 | B | untagged | substantive |
| Oxford_Bodleian_MS_heb_b_11_7 | qwen3_vl_8b_heb_v17_step800 | 0.269 | 0.066 | 0.445 | 5.932 | 6.81 | A | judaeo_arabic | substantive |
| Cambridge_CUL_T_S_16_117 | qwen3_vl_8b_heb_v17_step800 | 0.261 | 0.011 | 0.043 | 3.250 | 3.41 | A | judaeo_arabic | substantive |

### List (b) rows (with shorter windows)

| doc_id | model | ngram | aligned P | aligned R | CER-len | len ratio | tier | script | failure_mode | ngram n=4 | ngram n=3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cambridge_CUL_T_S_13J23_4 | gpt_5_6_sol | 0.000 | 1.000 | 0.004 | 0.996 | 0.00 | B | judaeo_arabic | abstained | 1.000 | 1.000 |
| New_York_JTS_ENA_NS_45_31 | gpt_5_6_sol | 0.000 | 1.000 | 0.010 | 0.990 | 0.01 | B | judaeo_arabic | abstained | 0.000 | 0.100 |
| Cambridge_CUL_T_S_16_235 | gpt_5_6_sol | 0.000 | 1.000 | 0.006 | 0.994 | 0.01 | B | judaeo_arabic | abstained | 0.250 | 0.600 |
| Cambridge_CUL_T_S_Misc_35_11 | gemini_flash | 0.000 | 1.000 | 0.005 | 0.995 | 0.00 | A | hebrew | abstained | 1.000 | 1.000 |
| Cambridge_CUL_T_S_AS_147_38 | gemini_flash | 0.000 | 1.000 | 0.002 | 0.998 | 0.00 | A | judaeo_arabic | abstained | 0.000 | 0.000 |
| Cambridge_CUL_T_S_16_128 | gpt_5_6_sol | 0.048 | 0.968 | 0.029 | 0.970 | 0.03 | A | judaeo_arabic | hallucinated | 0.318 | 0.609 |
| Cambridge_CUL_T_S_10J12_4 | gpt_5_6_sol | 0.000 | 0.941 | 0.011 | 0.988 | 0.01 | B | judaeo_arabic | abstained | 0.091 | 0.417 |
| New_York_JTS_ENA_NS_45_31 | gemini_flash | 0.000 | 0.935 | 0.022 | 0.976 | 0.02 | B | judaeo_arabic | hallucinated | 0.227 | 0.609 |
| Cambridge_CUL_T_S_18J1_14 | vision_ocr_seg | 0.000 | 0.923 | 0.016 | 0.984 | 0.02 | A | untagged | abstained | 0.000 | 0.125 |
| Cambridge_CUL_T_S_8J8_22 | gpt_5_6_sol | 0.000 | 0.900 | 0.022 | 0.977 | 0.03 | B | judaeo_arabic | abstained | 0.000 | 0.000 |
| Cambridge_CUL_T_S_10J14_1 | gpt_5_6_sol | 0.083 | 0.899 | 0.092 | 0.904 | 0.10 | A | hebrew | hallucinated | 0.164 | 0.419 |
| Cambridge_CUL_T_S_13J2_9 | vision_ocr_seg | 0.000 | 0.896 | 0.045 | 0.951 | 0.05 | B | judaeo_arabic | hallucinated | 0.000 | 0.167 |
| Cambridge_CUL_T_S_10J9_15 | gpt_5_6_sol | 0.000 | 0.886 | 0.046 | 0.948 | 0.06 | B | hebrew | hallucinated | 0.214 | 0.483 |
| Cambridge_CUL_T_S_10J14_13 | vision_ocr_seg | 0.000 | 0.879 | 0.035 | 0.964 | 0.04 | B | judaeo_arabic | hallucinated | 0.000 | 0.083 |
| Cambridge_CUL_T_S_10J13_9 | gpt_5_6_sol | 0.045 | 0.871 | 0.021 | 0.976 | 0.03 | B | judaeo_arabic | hallucinated | 0.130 | 0.333 |
| Cambridge_CUL_T_S_16_307 | gpt_5_6_sol | 0.030 | 0.870 | 0.040 | 0.954 | 0.05 | A | judaeo_arabic | hallucinated | 0.088 | 0.286 |
| Oxford_Bodleian_MS_heb_b_11_7 | gpt_5_6_sol | 0.000 | 0.857 | 0.003 | 0.997 | 0.00 | A | judaeo_arabic | abstained | 0.000 | 0.000 |
| Cambridge_CUL_T_S_16_110 | gpt_5_6_sol | 0.052 | 0.831 | 0.035 | 0.960 | 0.04 | B | untagged | hallucinated | 0.119 | 0.367 |
| Oxford_Bodleian_MS_heb_f_108_63 | gpt_5_6_sol | 0.000 | 0.812 | 0.035 | 0.957 | 0.04 | B | judaeo_arabic | hallucinated | 0.136 | 0.565 |
| Cambridge_CUL_T_S_20_162 | qwen3_vl_8b_heb_v17_step800 | 0.034 | 0.805 | 0.021 | 0.974 | 0.03 | B | judaeo_arabic | hallucinated | 0.067 | 0.226 |
| Cambridge_CUL_T_S_10J5_9 | vision_ocr_seg | 0.000 | 0.800 | 0.026 | 0.969 | 0.03 | B | judaeo_arabic | hallucinated | 0.000 | 0.097 |
| Cambridge_CUL_T_S_13J24_1 | gpt_5_6_sol | 0.085 | 0.765 | 0.125 | 0.865 | 0.17 | B | judaeo_arabic | hallucinated | 0.196 | 0.463 |
| Cambridge_CUL_T_S_13J26_14 | gpt_5_6_sol | 0.099 | 0.764 | 0.112 | 0.873 | 0.16 | A | untagged | hallucinated | 0.174 | 0.368 |
| New_York_JTS_ENA_NS_45_31 | gemini_pro | 0.000 | 0.762 | 0.036 | 0.953 | 0.05 | B | judaeo_arabic | hallucinated | 0.000 | 0.188 |
| Cambridge_CUL_T_S_13J20_17 | vision_ocr_seg | 0.074 | 0.757 | 0.127 | 0.860 | 0.17 | B | judaeo_arabic | hallucinated | 0.131 | 0.304 |
| Cambridge_CUL_T_S_10J16_19 | gpt_5_6_sol | 0.000 | 0.750 | 0.004 | 0.994 | 0.01 | A | judaeo_arabic | abstained | 0.250 | 0.600 |
| Cambridge_CUL_T_S_10J4_12 | claude_sonnet_5 | 0.096 | 0.730 | 0.640 | 0.396 | 0.87 | A | untagged | hallucinated | 0.169 | 0.349 |
| Cambridge_CUL_T_S_6J3_2 | gpt_5_6_sol | 0.000 | 0.727 | 0.010 | 0.986 | 0.02 | B | judaeo_arabic | abstained | 0.286 | 0.500 |
| Cambridge_CUL_T_S_16_307 | vision_ocr_seg | 0.043 | 0.726 | 0.121 | 0.866 | 0.15 | A | judaeo_arabic | hallucinated | 0.085 | 0.246 |
| Cambridge_CUL_T_S_16_237 | vision_ocr_seg | 0.024 | 0.710 | 0.115 | 0.868 | 0.15 | B | judaeo_arabic | hallucinated | 0.039 | 0.188 |
| New_York_JTS_ENA_NS_45_31 | claude_sonnet_5 | 0.000 | 0.704 | 0.038 | 0.946 | 0.05 | B | judaeo_arabic | hallucinated | 0.020 | 0.220 |
| Cambridge_CUL_T_S_10J5_4 | claude_sonnet_5 | 0.054 | 0.704 | 0.579 | 0.480 | 0.82 | B | judaeo_arabic | hallucinated | 0.112 | 0.300 |
| New_York_JTS_ENA_NS_45_31 | qwen3_vl_8b_heb_v17_step800 | 0.070 | 0.701 | 0.052 | 0.929 | 0.07 | B | judaeo_arabic | hallucinated | 0.181 | 0.370 |
| Cambridge_CUL_T_S_10J14_13 | gpt_5_6_sol | 0.089 | 0.696 | 0.133 | 0.844 | 0.19 | B | judaeo_arabic | hallucinated | 0.200 | 0.429 |
| Cambridge_CUL_T_S_12_111 | vision_ocr_seg | 0.000 | 0.691 | 0.193 | 0.799 | 0.22 | B | judaeo_arabic | hallucinated | 0.035 | 0.226 |
| Cambridge_CUL_Or_1080_J262 | vision_ocr_seg | 0.099 | 0.691 | 0.364 | 0.619 | 0.49 | A | judaeo_arabic | hallucinated | 0.166 | 0.334 |
| Cambridge_CUL_T_S_8J24_6 | qwen3_vl_8b_heb_v16_step1100 | 0.094 | 0.690 | 0.661 | 0.390 | 0.95 | A | judaeo_arabic | hallucinated | 0.162 | 0.306 |
| Cambridge_CUL_T_S_13J2_14 | vision_ocr_seg | 0.000 | 0.688 | 0.020 | 0.974 | 0.03 | A | judaeo_arabic | abstained | 0.050 | 0.333 |
| Cambridge_CUL_T_S_20_98 | vision_ocr_seg | 0.083 | 0.688 | 0.084 | 0.892 | 0.12 | A | untagged | hallucinated | 0.157 | 0.378 |
| Cambridge_Mosseri_VII_67_2 | vision_ocr_seg | 0.050 | 0.681 | 0.251 | 0.731 | 0.36 | A | untagged | hallucinated | 0.111 | 0.293 |
| Cambridge_CUL_T_S_16_110 | qwen3_vl_8b_heb_v16_step1100 | 0.015 | 0.681 | 0.035 | 0.951 | 0.05 | B | untagged | hallucinated | 0.045 | 0.373 |
| Cambridge_CUL_T_S_18J2_16 | qwen3_vl_8b_heb_v16_step1100 | 0.066 | 0.680 | 0.614 | 0.408 | 0.89 | A | arabic_script | hallucinated | 0.128 | 0.339 |
| Cambridge_CUL_T_S_20_98 | gpt_5_6_sol | 0.000 | 0.679 | 0.025 | 0.963 | 0.04 | A | untagged | hallucinated | 0.000 | 0.140 |
| Cambridge_CUL_T_S_16_232 | claude_opus_4_8 | 0.098 | 0.674 | 0.642 | 0.415 | 0.93 | B | judaeo_arabic | hallucinated | 0.166 | 0.342 |
| Cambridge_CUL_T_S_10J5_4 | qwen3_vl_8b_heb_v17_step800 | 0.092 | 0.668 | 0.649 | 0.433 | 0.98 | B | judaeo_arabic | hallucinated | 0.172 | 0.357 |
| Cambridge_CUL_T_S_8J33_7 | vision_ocr_seg | 0.034 | 0.667 | 0.047 | 0.943 | 0.06 | B | judaeo_arabic | hallucinated | 0.067 | 0.226 |
| Cambridge_CUL_T_S_8J4_18 | vision_ocr_seg | 0.038 | 0.667 | 0.022 | 0.967 | 0.04 | A | judaeo_arabic | hallucinated | 0.111 | 0.250 |
| Cambridge_CUL_T_S_20_110 | vision_ocr_seg | 0.082 | 0.661 | 0.112 | 0.862 | 0.16 | A | judaeo_arabic | hallucinated | 0.136 | 0.319 |
| Cambridge_CUL_T_S_10J9_15 | gemini_flash | 0.000 | 0.655 | 0.028 | 0.959 | 0.04 | B | hebrew | abstained | 0.000 | 0.095 |
| Cambridge_CUL_T_S_16_232 | gpt_5_6_sol | 0.000 | 0.650 | 0.020 | 0.971 | 0.03 | B | judaeo_arabic | abstained | 0.000 | 0.200 |
| Cambridge_CUL_T_S_16_256 | claude_sonnet_5 | 0.058 | 0.647 | 0.612 | 0.464 | 0.88 | A | judaeo_arabic | hallucinated | 0.122 | 0.341 |
| Cambridge_CUL_T_S_13J23_12 | claude_sonnet_5 | 0.053 | 0.646 | 0.552 | 0.460 | 0.86 | B | hebrew | hallucinated | 0.104 | 0.296 |
| Cambridge_CUL_T_S_10J24_3 | vision_ocr_seg | 0.014 | 0.641 | 0.104 | 0.875 | 0.15 | B | hebrew | hallucinated | 0.056 | 0.200 |
| Cambridge_CUL_T_S_10J13_11 | claude_sonnet_5 | 0.056 | 0.640 | 0.523 | 0.483 | 0.80 | A | judaeo_arabic | hallucinated | 0.120 | 0.310 |
| New_York_JTS_ENA_NS_2_31 | claude_opus_4_8 | 0.098 | 0.640 | 0.581 | 0.469 | 0.81 | B | judaeo_arabic | hallucinated | 0.177 | 0.328 |
| Cambridge_CUL_Or_1080_J262 | qwen3_vl_8b | 0.078 | 0.640 | 0.562 | 0.463 | 0.87 | A | judaeo_arabic | hallucinated | 0.143 | 0.288 |
| Cambridge_CUL_T_S_13J6_23 | qwen3_vl_8b_heb_v16_step1100 | 0.076 | 0.637 | 0.554 | 0.428 | 0.88 | A | judaeo_arabic | hallucinated | 0.139 | 0.279 |
| Cambridge_CUL_T_S_10J21_1 | vision_ocr_seg | 0.092 | 0.634 | 0.184 | 0.788 | 0.26 | A | judaeo_arabic | hallucinated | 0.156 | 0.326 |
| Cambridge_CUL_T_S_10J21_1 | claude_sonnet_5 | 0.050 | 0.633 | 0.360 | 0.668 | 0.56 | A | judaeo_arabic | hallucinated | 0.112 | 0.340 |
| Cambridge_CUL_T_S_16_110 | kraken_seg | 0.085 | 0.633 | 0.397 | 0.600 | 0.61 | B | untagged | hallucinated | 0.150 | 0.413 |
| Cambridge_CUL_T_S_13J26_14 | vision_ocr_seg | 0.085 | 0.629 | 0.166 | 0.808 | 0.26 | A | untagged | hallucinated | 0.147 | 0.329 |
| Cambridge_CUL_T_S_13J17_9 | kraken_raw | 0.077 | 0.628 | 0.202 | 0.746 | 0.31 | B | hebrew | hallucinated | 0.153 | 0.332 |
| Cambridge_CUL_T_S_10J7_3 | vision_ocr_seg | 0.060 | 0.627 | 0.047 | 0.931 | 0.07 | B | judaeo_arabic | hallucinated | 0.176 | 0.346 |
| Cambridge_CUL_T_S_12_146 | gpt_5_6_sol | 0.000 | 0.626 | 0.045 | 0.932 | 0.07 | B | hebrew | hallucinated | 0.037 | 0.256 |
| Cambridge_CUL_T_S_10J4_12 | qwen3_vl_8b_heb_v16_step1100 | 0.078 | 0.625 | 0.449 | 0.550 | 0.67 | A | untagged | hallucinated | 0.121 | 0.242 |
| Cambridge_CUL_T_S_Ar_18_49 | gpt_5_6_sol | 0.026 | 0.624 | 0.065 | 0.906 | 0.10 | B | untagged | hallucinated | 0.104 | 0.372 |
| Cambridge_CUL_T_S_8J4_18 | claude_sonnet_5 | 0.013 | 0.622 | 0.299 | 0.666 | 0.47 | A | judaeo_arabic | hallucinated | 0.056 | 0.273 |
| Cambridge_CUL_T_S_20_110 | claude_sonnet_5 | 0.044 | 0.622 | 0.554 | 0.487 | 0.88 | A | judaeo_arabic | hallucinated | 0.092 | 0.309 |
| Cambridge_CUL_T_S_16_149 | gemini_flash | 0.058 | 0.622 | 0.049 | 0.926 | 0.08 | B | judaeo_arabic | hallucinated | 0.115 | 0.350 |
| Cambridge_CUL_T_S_8J22_30 | claude_opus_4_8 | 0.067 | 0.622 | 0.599 | 0.427 | 0.92 | B | judaeo_arabic | hallucinated | 0.119 | 0.317 |
| Cambridge_CUL_T_S_13J8_2 | qwen3_vl_8b_heb_v16_step1100 | 0.068 | 0.621 | 0.503 | 0.499 | 0.81 | A | hebrew | hallucinated | 0.121 | 0.297 |
| Cambridge_CUL_T_S_10J13_17 | claude_sonnet_5 | 0.000 | 0.617 | 0.039 | 0.955 | 0.04 | B | judaeo_arabic | hallucinated | 0.000 | 0.125 |
| Cambridge_CUL_T_S_NS_184_57 | claude_opus_4_8 | 0.076 | 0.610 | 0.474 | 0.521 | 0.76 | B | judaeo_arabic | hallucinated | 0.149 | 0.305 |
| Cambridge_CUL_T_S_24_51 | claude_opus_4_8 | 0.082 | 0.608 | 0.446 | 0.519 | 0.68 | B | judaeo_arabic | hallucinated | 0.148 | 0.364 |
| Cambridge_CUL_T_S_13J25_5 | qwen3_vl_8b_heb_v17_step800 | 0.090 | 0.607 | 0.596 | 0.449 | 0.99 | B | untagged | hallucinated | 0.160 | 0.426 |
| Oxford_Bodleian_MS_heb_f_108_63 | qwen3_vl_8b_heb_v16_step1100 | 0.051 | 0.607 | 0.537 | 0.495 | 0.90 | B | judaeo_arabic | hallucinated | 0.097 | 0.242 |
| Cambridge_CUL_T_S_20_162 | claude_sonnet_5 | 0.000 | 0.600 | 0.006 | 0.994 | 0.00 | B | judaeo_arabic | abstained | 0.000 | 0.000 |
| Oxford_Bodleian_MS_heb_d_74_40 | gpt_5_6_sol | 0.000 | 0.600 | 0.004 | 0.995 | 0.01 | A | judaeo_arabic | abstained | 0.000 | 0.000 |
| Cambridge_CUL_T_S_Ar_30_255 | gemini_pro | 0.061 | 0.599 | 0.109 | 0.854 | 0.18 | B | judaeo_arabic | hallucinated | 0.144 | 0.423 |
| Oxford_Bodleian_MS_heb_d_74_40 | claude_sonnet_5 | 0.033 | 0.598 | 0.601 | 0.463 | 1.01 | A | judaeo_arabic | hallucinated | 0.087 | 0.264 |
| Cambridge_CUL_T_S_10J9_15 | qwen3_vl_8b_heb_v16_step1100 | 0.066 | 0.597 | 0.400 | 0.573 | 0.68 | B | hebrew | hallucinated | 0.120 | 0.236 |
| Cambridge_CUL_T_S_6J6_11 | qwen3_vl_8b_heb_v16_step1100 | 0.048 | 0.596 | 0.531 | 0.514 | 0.84 | B | judaeo_arabic | hallucinated | 0.085 | 0.206 |
| Cambridge_CUL_T_S_16_128 | claude_sonnet_5 | 0.046 | 0.595 | 0.554 | 0.475 | 0.92 | A | judaeo_arabic | hallucinated | 0.108 | 0.306 |
| Cambridge_CUL_T_S_16_149 | claude_opus_4_8 | 0.061 | 0.594 | 0.467 | 0.529 | 0.72 | B | judaeo_arabic | hallucinated | 0.106 | 0.287 |
| Cambridge_CUL_T_S_20_21 | claude_sonnet_5 | 0.095 | 0.587 | 0.562 | 0.477 | 0.97 | A | judaeo_arabic | hallucinated | 0.182 | 0.497 |
| Cambridge_CUL_T_S_8_137 | claude_sonnet_5 | 0.057 | 0.585 | 0.547 | 0.496 | 0.90 | B | judaeo_arabic | hallucinated | 0.118 | 0.278 |
| Cambridge_CUL_T_S_13J8_6 | vision_ocr_seg | 0.067 | 0.584 | 0.074 | 0.901 | 0.12 | A | judaeo_arabic | hallucinated | 0.115 | 0.313 |
| Cambridge_CUL_T_S_10J21_1 | qwen3_vl_8b_heb_v16_step1100 | 0.019 | 0.582 | 0.140 | 0.832 | 0.23 | A | judaeo_arabic | hallucinated | 0.062 | 0.252 |
| Cambridge_CUL_T_S_18J1_27 | vision_ocr_seg | 0.029 | 0.579 | 0.073 | 0.914 | 0.08 | B | untagged | hallucinated | 0.070 | 0.208 |
| Oxford_Bodleian_MS_heb_f_56_12 | claude_opus_4_8 | 0.065 | 0.579 | 0.418 | 0.552 | 0.65 | B | judaeo_arabic | hallucinated | 0.119 | 0.270 |
| Cambridge_CUL_T_S_13J20_17 | claude_sonnet_5 | 0.060 | 0.579 | 0.513 | 0.483 | 0.90 | B | judaeo_arabic | hallucinated | 0.106 | 0.272 |
| Manchester_JRL_A_960 | claude_sonnet_5 | 0.014 | 0.579 | 0.895 | 0.132 | 1.67 | A | aramaic | hallucinated | 0.018 | 0.049 |
| Cambridge_CUL_T_S_10J14_13 | qwen3_vl_8b_heb_v17_step800 | 0.076 | 0.576 | 0.532 | 0.482 | 0.91 | B | judaeo_arabic | hallucinated | 0.159 | 0.378 |
| Cambridge_CUL_T_S_13J2_6 | gpt_5_6_sol | 0.021 | 0.574 | 0.014 | 0.976 | 0.03 | B | judaeo_arabic | hallucinated | 0.083 | 0.367 |
| Cambridge_CUL_T_S_8J33_7 | qwen3_vl_8b_heb_v17_step800 | 0.067 | 0.574 | 0.612 | 0.573 | 1.08 | B | judaeo_arabic | hallucinated | 0.135 | 0.316 |
| Cambridge_CUL_T_S_20_28 | vision_ocr_seg | 0.086 | 0.573 | 0.261 | 0.691 | 0.42 | B | judaeo_arabic | hallucinated | 0.136 | 0.362 |
| Cambridge_CUL_T_S_13J2_9 | gemini_flash | 0.073 | 0.572 | 0.095 | 0.861 | 0.16 | B | judaeo_arabic | hallucinated | 0.161 | 0.376 |
| Oxford_Bodleian_MS_heb_d_68_101 | gpt_5_6_sol | 0.024 | 0.571 | 0.149 | 0.806 | 0.26 | A | judaeo_arabic | hallucinated | 0.107 | 0.272 |
| Cambridge_CUL_T_S_18J1_14 | claude_sonnet_5 | 0.049 | 0.571 | 0.515 | 0.497 | 0.90 | A | untagged | hallucinated | 0.092 | 0.283 |
| Cambridge_CUL_T_S_12_3 | claude_sonnet_5 | 0.075 | 0.570 | 0.519 | 0.527 | 0.91 | A | judaeo_arabic | hallucinated | 0.153 | 0.400 |
| Cambridge_CUL_T_S_16_149 | vision_ocr_seg | 0.013 | 0.569 | 0.029 | 0.951 | 0.05 | B | judaeo_arabic | hallucinated | 0.065 | 0.205 |
| Cambridge_CUL_T_S_13J20_17 | qwen3_vl_8b_heb_v16_step1100 | 0.043 | 0.568 | 0.490 | 0.476 | 0.86 | B | judaeo_arabic | hallucinated | 0.093 | 0.237 |
| New_York_JTS_ENA_NS_50_32 | gpt_5_6_sol | 0.014 | 0.568 | 0.157 | 0.799 | 0.27 | A | hebrew | hallucinated | 0.066 | 0.356 |
| Cambridge_CUL_T_S_16_128 | claude_opus_4_8 | 0.069 | 0.567 | 0.579 | 0.463 | 1.03 | A | judaeo_arabic | hallucinated | 0.123 | 0.322 |
| Cambridge_CUL_T_S_AS_147_38 | vision_ocr_seg | 0.030 | 0.566 | 0.128 | 0.845 | 0.22 | A | judaeo_arabic | hallucinated | 0.059 | 0.200 |
| Cambridge_CUL_T_S_8_4 | claude_opus_4_8 | 0.070 | 0.566 | 0.488 | 0.509 | 0.86 | B | judaeo_arabic | hallucinated | 0.136 | 0.320 |
| New_York_JTS_ENA_4020_16 | claude_opus_4_8 | 0.067 | 0.566 | 0.512 | 0.481 | 0.85 | B | judaeo_arabic | hallucinated | 0.129 | 0.380 |
| Cambridge_CUL_T_S_13J24_1 | claude_sonnet_5 | 0.033 | 0.565 | 0.555 | 0.526 | 0.88 | B | judaeo_arabic | hallucinated | 0.081 | 0.224 |
| Cambridge_CUL_T_S_8_143 | qwen3_vl_8b_heb_v16_step1100 | 0.054 | 0.560 | 0.468 | 0.495 | 0.82 | A | judaeo_arabic | hallucinated | 0.103 | 0.253 |
| Oxford_Bodleian_MS_heb_d_66_22 | claude_sonnet_5 | 0.030 | 0.559 | 0.511 | 0.479 | 0.92 | B | aramaic | hallucinated | 0.077 | 0.251 |
| Cambridge_Mosseri_VII_67_2 | qwen3_vl_8b_heb_v16_step1100 | 0.037 | 0.556 | 0.549 | 0.475 | 0.91 | A | untagged | hallucinated | 0.081 | 0.260 |
| Cambridge_CUL_T_S_13J4_10 | qwen3_vl_8b_heb_v17_step800 | 0.094 | 0.553 | 0.521 | 0.485 | 0.93 | B | judaeo_arabic | hallucinated | 0.141 | 0.317 |
| Oxford_Bodleian_MS_heb_f_108_63 | claude_sonnet_5 | 0.039 | 0.551 | 0.490 | 0.516 | 0.90 | B | judaeo_arabic | hallucinated | 0.076 | 0.223 |
| Cambridge_CUL_T_S_16_232 | claude_sonnet_5 | 0.019 | 0.551 | 0.474 | 0.533 | 0.83 | B | judaeo_arabic | hallucinated | 0.058 | 0.178 |
| Cambridge_CUL_T_S_16_232 | gemini_pro | 0.077 | 0.548 | 0.468 | 0.514 | 0.86 | B | judaeo_arabic | hallucinated | 0.166 | 0.397 |
| Cambridge_CUL_T_S_10J7_3 | claude_sonnet_5 | 0.074 | 0.546 | 0.387 | 0.653 | 0.71 | B | judaeo_arabic | hallucinated | 0.145 | 0.363 |
| Cambridge_CUL_T_S_13J14_7 | qwen3_vl_8b_heb_v16_step1100 | 0.031 | 0.544 | 0.460 | 0.529 | 0.81 | A | judaeo_arabic | hallucinated | 0.061 | 0.186 |
| Cambridge_CUL_T_S_13J23_12 | qwen3_vl_8b_heb_v16_step1100 | 0.035 | 0.539 | 0.463 | 0.501 | 0.83 | B | hebrew | hallucinated | 0.077 | 0.228 |
| Oxford_Bodleian_MS_heb_f_108_63 | vision_ocr_seg | 0.000 | 0.538 | 0.087 | 0.867 | 0.14 | B | judaeo_arabic | hallucinated | 0.013 | 0.148 |
| Cambridge_CUL_T_S_16_235 | qwen3_vl_8b_heb_v17_step800 | 0.074 | 0.537 | 0.534 | 0.544 | 1.00 | B | judaeo_arabic | hallucinated | 0.167 | 0.434 |
| Cambridge_CUL_T_S_16_149 | claude_sonnet_5 | 0.023 | 0.537 | 0.360 | 0.595 | 0.67 | B | judaeo_arabic | hallucinated | 0.061 | 0.240 |
| Cambridge_CUL_T_S_13J4_10 | claude_opus_4_8 | 0.061 | 0.529 | 0.433 | 0.493 | 0.68 | B | judaeo_arabic | hallucinated | 0.103 | 0.264 |
| Cambridge_CUL_T_S_13J35_1 | gemini_pro | 0.044 | 0.523 | 0.028 | 0.949 | 0.05 | B | hebrew | hallucinated | 0.199 | 0.482 |
| Cambridge_CUL_T_S_10J14_1 | vision_ocr_seg | 0.029 | 0.522 | 0.249 | 0.680 | 0.46 | A | hebrew | hallucinated | 0.076 | 0.273 |
| Cambridge_CUL_T_S_10J24_3 | claude_sonnet_5 | 0.058 | 0.520 | 0.433 | 0.517 | 0.84 | B | hebrew | hallucinated | 0.088 | 0.280 |
| Oxford_Bodleian_MS_heb_d_74_40 | qwen3_vl_8b_heb_v16_step1100 | 0.011 | 0.519 | 0.450 | 0.538 | 0.88 | A | judaeo_arabic | hallucinated | 0.034 | 0.166 |
| Cambridge_CUL_T_S_13J5_3 | claude_sonnet_5 | 0.046 | 0.519 | 0.207 | 0.745 | 0.40 | A | judaeo_arabic | hallucinated | 0.100 | 0.301 |
| Cambridge_CUL_T_S_10J9_15 | claude_sonnet_5 | 0.072 | 0.516 | 0.441 | 0.530 | 0.87 | B | hebrew | hallucinated | 0.112 | 0.237 |
| Oxford_Bodleian_MS_heb_d_68_101 | qwen3_vl_8b_heb_v16_step1100 | 0.045 | 0.512 | 0.444 | 0.539 | 0.82 | A | judaeo_arabic | hallucinated | 0.078 | 0.200 |
| Oxford_Bodleian_MS_heb_f_56_12 | gpt_5_6_sol | 0.000 | 0.512 | 0.037 | 0.935 | 0.07 | B | judaeo_arabic | hallucinated | 0.094 | 0.303 |
| Cambridge_CUL_T_S_10J14_1 | claude_opus_4_8 | 0.085 | 0.509 | 0.509 | 0.403 | 1.00 | A | hebrew | hallucinated | 0.133 | 0.352 |
| Cambridge_CUL_T_S_13J2_6 | claude_opus_4_8 | 0.066 | 0.508 | 0.308 | 0.660 | 0.51 | B | judaeo_arabic | hallucinated | 0.117 | 0.323 |
| Cambridge_CUL_T_S_10J13_11 | vision_ocr_seg | 0.014 | 0.505 | 0.110 | 0.833 | 0.21 | A | judaeo_arabic | hallucinated | 0.069 | 0.262 |
| Cambridge_CUL_T_S_13J25_5 | claude_sonnet_5 | 0.026 | 0.505 | 0.477 | 0.559 | 0.96 | B | untagged | hallucinated | 0.076 | 0.311 |
| Cambridge_CUL_T_S_8J8_22 | qwen3_vl_8b_heb_v17_step800 | 0.068 | 0.505 | 0.460 | 0.503 | 0.91 | B | judaeo_arabic | hallucinated | 0.160 | 0.371 |
| Cambridge_CUL_T_S_13J5_3 | gpt_5_6_sol | 0.005 | 0.504 | 0.069 | 0.885 | 0.13 | A | judaeo_arabic | hallucinated | 0.042 | 0.285 |
| Cambridge_CUL_T_S_10J13_11 | gemini_pro | 0.077 | 0.504 | 0.403 | 0.585 | 0.80 | A | judaeo_arabic | hallucinated | 0.152 | 0.388 |
| Cambridge_CUL_T_S_8J33_7 | claude_sonnet_5 | 0.049 | 0.502 | 0.465 | 0.580 | 0.83 | B | judaeo_arabic | hallucinated | 0.089 | 0.223 |
| Oxford_Bodleian_MS_heb_d_68_101 | vision_ocr_seg | 0.000 | 0.500 | 0.004 | 0.996 | 0.00 | A | judaeo_arabic | abstained | 0.000 | 0.000 |

#### List (b) split by hypothesis length

Aligned precision is length-blind: a 20-character stub that happens to match scores ~0.9. Splitting list (b) by `len(hyp letters)/len(GT letters)` separates those stubs from genuine full-length near misses.

| band | n | median aligned R | median n=5 | median n=4 | median n=3 | n=3 >= 0.10 | metric-abstained |
|---|---|---|---|---|---|---|---|
| len ratio < 0.15 (stub) | 51 | 0.029 | 0.000 | 0.070 | 0.285 | 42 | 17 |
| 0.15-0.50 | 24 | 0.145 | 0.056 | 0.121 | 0.312 | 24 | 0 |
| 0.50-0.80 | 13 | 0.403 | 0.066 | 0.120 | 0.305 | 13 | 0 |
| >= 0.80 (full length) | 51 | 0.532 | 0.058 | 0.118 | 0.300 | 50 | 0 |

Shorter-window rescue, list (b): median n=5 0.046, n=4 0.111, n=3 0.301; 129/139 clear the 0.10 hallucination cutoff at n=3 and 79/139 at n=4.

## Part 3 — conditioning on the Gemini Flash judge

Judge coverage: 1215/1352 outputs (89.9%). Unjudged per system: kraken_raw 131, qwen3_vl_8b_heb_v17_step800 4, qwen3_vl_8b 2. All tables below use judged rows only.

Judged flags: severe hallucination 953 (78.4%), any-severity hallucination 1135 (93.4%), canonical_completion 44.

### (i) n-gram precision conditioned on judged hallucination

**severe hallucination**

| system | n flagged | median | p25 | p75 | n not flagged | median | p25 | p75 |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 53 | 0.396 | 0.297 | 0.473 | 69 | 0.559 | 0.455 | 0.677 |
| kraken_raw | 0 | — | — | — | 0 | — | — | — |
| gemini_pro | 64 | 0.116 | 0.057 | 0.291 | 4 | 0.749 | 0.706 | 0.792 |
| gemini_flash | 112 | 0.059 | 0.023 | 0.185 | 10 | 0.262 | 0.050 | 0.449 |
| claude_opus_4_8 | 87 | 0.034 | 0.000 | 0.111 | 44 | 0.184 | 0.028 | 0.292 |
| claude_sonnet_5 | 99 | 0.020 | 0.004 | 0.049 | 32 | 0.126 | 0.045 | 0.266 |
| gpt_5_6_sol | 84 | 0.009 | 0.000 | 0.163 | 47 | 0.000 | 0.000 | 0.327 |
| qwen3_vl_8b_heb_v17_step800 | 119 | 0.151 | 0.072 | 0.273 | 8 | 0.363 | 0.282 | 0.439 |
| qwen3_vl_8b_heb_v16_step1100 | 109 | 0.002 | 0.000 | 0.028 | 22 | 0.126 | 0.044 | 0.198 |
| qwen3_vl_8b | 127 | 0.000 | 0.000 | 0.000 | 2 | 0.089 | 0.044 | 0.133 |
| vision_ocr_seg | 99 | 0.000 | 0.000 | 0.099 | 24 | 0.136 | 0.070 | 0.340 |
| POOLED | 953 | 0.030 | 0.000 | 0.155 | 262 | 0.228 | 0.052 | 0.495 |

**hallucination, any severity**

| system | n flagged | median | p25 | p75 | n not flagged | median | p25 | p75 |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 111 | 0.475 | 0.335 | 0.606 | 11 | 0.469 | 0.262 | 0.731 |
| kraken_raw | 0 | — | — | — | 0 | — | — | — |
| gemini_pro | 66 | 0.143 | 0.059 | 0.300 | 2 | 0.750 | 0.690 | 0.809 |
| gemini_flash | 118 | 0.070 | 0.027 | 0.257 | 4 | 0.000 | 0.000 | 0.050 |
| claude_opus_4_8 | 114 | 0.065 | 0.000 | 0.174 | 17 | 0.031 | 0.000 | 0.239 |
| claude_sonnet_5 | 121 | 0.026 | 0.005 | 0.062 | 10 | 0.128 | 0.055 | 0.288 |
| gpt_5_6_sol | 102 | 0.018 | 0.000 | 0.190 | 29 | 0.000 | 0.000 | 0.000 |
| qwen3_vl_8b_heb_v17_step800 | 126 | 0.158 | 0.076 | 0.290 | 1 | 0.266 | 0.266 | 0.266 |
| qwen3_vl_8b_heb_v16_step1100 | 127 | 0.008 | 0.000 | 0.046 | 4 | 0.221 | 0.134 | 0.328 |
| qwen3_vl_8b | 128 | 0.000 | 0.000 | 0.000 | 1 | 0.000 | 0.000 | 0.000 |
| vision_ocr_seg | 122 | 0.025 | 0.000 | 0.127 | 1 | 0.000 | 0.000 | 0.000 |
| POOLED | 1135 | 0.050 | 0.000 | 0.221 | 80 | 0.083 | 0.000 | 0.385 |

### (ii) n-gram precision conditioned on overall_quality

| system | quality | n | median | p25 | p75 |
|---|---|---|---|---|---|
| kraken_seg | unusable | 26 | 0.303 | 0.215 | 0.413 |
| kraken_seg | poor | 87 | 0.499 | 0.399 | 0.597 |
| kraken_seg | fair | 8 | 0.726 | 0.690 | 0.740 |
| kraken_seg | good | 1 | 0.791 | 0.791 | 0.791 |
| gemini_pro | unusable | 39 | 0.065 | 0.037 | 0.111 |
| gemini_pro | poor | 22 | 0.300 | 0.234 | 0.380 |
| gemini_pro | fair | 4 | 0.622 | 0.610 | 0.649 |
| gemini_pro | good | 3 | 0.766 | 0.749 | 0.817 |
| gemini_flash | unusable | 93 | 0.042 | 0.008 | 0.117 |
| gemini_flash | poor | 24 | 0.286 | 0.166 | 0.413 |
| gemini_flash | fair | 5 | 0.660 | 0.499 | 0.719 |
| claude_opus_4_8 | unusable | 77 | 0.000 | 0.000 | 0.061 |
| claude_opus_4_8 | poor | 47 | 0.189 | 0.116 | 0.241 |
| claude_opus_4_8 | fair | 5 | 0.432 | 0.194 | 0.473 |
| claude_opus_4_8 | good | 2 | 0.017 | 0.013 | 0.022 |
| claude_sonnet_5 | unusable | 86 | 0.011 | 0.001 | 0.032 |
| claude_sonnet_5 | poor | 43 | 0.125 | 0.051 | 0.217 |
| claude_sonnet_5 | fair | 2 | 0.305 | 0.160 | 0.450 |
| gpt_5_6_sol | unusable | 111 | 0.000 | 0.000 | 0.059 |
| gpt_5_6_sol | poor | 18 | 0.334 | 0.284 | 0.388 |
| gpt_5_6_sol | fair | 2 | 0.668 | 0.598 | 0.738 |
| qwen3_vl_8b_heb_v17_step800 | unusable | 61 | 0.078 | 0.023 | 0.140 |
| qwen3_vl_8b_heb_v17_step800 | poor | 64 | 0.252 | 0.168 | 0.337 |
| qwen3_vl_8b_heb_v17_step800 | fair | 2 | 0.525 | 0.504 | 0.546 |
| qwen3_vl_8b_heb_v16_step1100 | unusable | 108 | 0.001 | 0.000 | 0.024 |
| qwen3_vl_8b_heb_v16_step1100 | poor | 23 | 0.162 | 0.099 | 0.248 |
| qwen3_vl_8b | unusable | 126 | 0.000 | 0.000 | 0.000 |
| qwen3_vl_8b | poor | 3 | 0.000 | 0.000 | 0.108 |
| vision_ocr_seg | unusable | 103 | 0.000 | 0.000 | 0.078 |
| vision_ocr_seg | poor | 19 | 0.273 | 0.204 | 0.342 |
| vision_ocr_seg | fair | 1 | 0.622 | 0.622 | 0.622 |
| POOLED | unusable | 830 | 0.006 | 0.000 | 0.062 |
| POOLED | poor | 350 | 0.277 | 0.159 | 0.419 |
| POOLED | fair | 29 | 0.622 | 0.499 | 0.719 |
| POOLED | good | 6 | 0.749 | 0.202 | 0.785 |

### (iii) 2x2: metric class vs judged severe hallucination

Metric positive = `failure_mode == hallucinated`; abstained and loop-collapse rows excluded.

| system | n | both halluc | metric only | judge only | neither | agreement | kappa |
|---|---|---|---|---|---|---|---|
| kraken_seg | 121 | 1 | 1 | 52 | 67 | 0.562 | 0.005 |
| kraken_raw | 0 | 0 | 0 | 0 | 0 | — | — |
| gemini_pro | 68 | 28 | 0 | 36 | 4 | 0.471 | 0.084 |
| gemini_flash | 99 | 47 | 0 | 46 | 6 | 0.535 | 0.110 |
| claude_opus_4_8 | 91 | 30 | 8 | 24 | 29 | 0.648 | 0.318 |
| claude_sonnet_5 | 118 | 80 | 9 | 10 | 19 | 0.839 | 0.561 |
| gpt_5_6_sol | 64 | 25 | 3 | 27 | 9 | 0.531 | 0.130 |
| qwen3_vl_8b_heb_v17_step800 | 122 | 38 | 0 | 76 | 8 | 0.377 | 0.062 |
| qwen3_vl_8b_heb_v16_step1100 | 108 | 82 | 8 | 5 | 13 | 0.880 | 0.594 |
| qwen3_vl_8b | 39 | 37 | 0 | 1 | 1 | 0.974 | 0.655 |
| vision_ocr_seg | 105 | 60 | 7 | 23 | 15 | 0.714 | 0.319 |
| POOLED | 935 | 428 | 36 | 300 | 171 | 0.641 | 0.284 |

### (iv) rank AUC for predicting judged severe hallucination

Predictors oriented so that higher = more likely hallucination (n-gram precision and aligned precision negated; CER-lenient as is). AUC 0.5 = no signal.

| system | n | pos | neg | AUC n-gram P | AUC aligned P | AUC CER-lenient |
|---|---|---|---|---|---|---|
| kraken_seg | 122 | 53 | 69 | 0.753 | 0.667 | 0.605 |
| kraken_raw | 0 | 0 | 0 | — | — | — |
| gemini_pro | 68 | 64 | 4 | 0.984 | 0.973 | 0.973 |
| gemini_flash | 122 | 112 | 10 | 0.629 | 0.669 | 0.371 |
| claude_opus_4_8 | 131 | 87 | 44 | 0.725 | 0.672 | 0.726 |
| claude_sonnet_5 | 131 | 99 | 32 | 0.765 | 0.772 | 0.685 |
| gpt_5_6_sol | 131 | 84 | 47 | 0.498 | 0.534 | 0.324 |
| qwen3_vl_8b_heb_v17_step800 | 127 | 119 | 8 | 0.817 | 0.868 | 0.887 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 109 | 22 | 0.892 | 0.939 | 0.875 |
| qwen3_vl_8b | 129 | 127 | 2 | 0.722 | 0.528 | 0.929 |
| vision_ocr_seg | 123 | 99 | 24 | 0.756 | 0.792 | 0.742 |
| POOLED | 1215 | 953 | 262 | 0.717 | 0.726 | 0.678 |

### (v) canonical_completion outputs

| doc_id | model | ngram | aligned P | CER-len | failure_mode | substantive? | quality | tier |
|---|---|---|---|---|---|---|---|---|
| Cambridge_CUL_T_S_24_13 | gemini_flash | 0.000 | 0.107 | 0.826 | hallucinated | no | unusable | A |
| Cambridge_CUL_T_S_13J23_7 | gpt_5_6_sol | 0.010 | 0.240 | 0.727 | hallucinated | no | unusable | A |
| Manchester_JRL_A_960 | claude_sonnet_5 | 0.014 | 0.579 | 0.132 | hallucinated | no | fair | A |
| Cambridge_CUL_T_S_13J2_6 | qwen3_vl_8b_heb_v16_step1100 | 0.018 | 0.384 | 0.744 | hallucinated | no | unusable | B |
| Cambridge_CUL_T_S_12_3 | claude_opus_4_8 | 0.026 | 0.090 | 0.676 | loop_collapse | no | good | A |
| Cambridge_CUL_T_S_13J14_7 | gemini_flash | 0.036 | 0.098 | 0.815 | hallucinated | no | unusable | A |
| Cambridge_CUL_T_S_10J21_1 | gpt_5_6_sol | 0.041 | 0.390 | 0.846 | hallucinated | no | unusable | A |
| Oxford_Bodleian_MS_heb_d_68_101 | qwen3_vl_8b_heb_v16_step1100 | 0.045 | 0.512 | 0.539 | hallucinated | no | unusable | A |
| Cambridge_CUL_T_S_8J33_7 | qwen3_vl_8b_heb_v17_step800 | 0.067 | 0.574 | 0.573 | hallucinated | no | poor | B |
| Cambridge_Mosseri_VII_67_2 | gpt_5_6_sol | 0.149 | 0.397 | 0.626 | substantive | yes | unusable | A |
| Cambridge_CUL_T_S_20_110 | gemini_flash | 0.184 | 0.549 | 0.492 | substantive | yes | unusable | A |
| Cambridge_CUL_T_S_8_137 | gpt_5_6_sol | 0.228 | 0.636 | 0.849 | substantive | yes | unusable | B |
| Cambridge_CUL_T_S_12_111 | gemini_pro | 0.229 | 0.600 | 0.461 | substantive | yes | poor | B |
| Cambridge_CUL_T_S_8J33_7 | gemini_flash | 0.235 | 0.701 | 0.698 | substantive | yes | unusable | B |
| Cambridge_CUL_T_S_18J1_14 | gpt_5_6_sol | 0.237 | 0.820 | 0.939 | substantive | yes | unusable | A |
| Cambridge_CUL_T_S_10J13_11 | gpt_5_6_sol | 0.265 | 0.806 | 0.888 | substantive | yes | unusable | A |
| Cambridge_Mosseri_VII_67_2 | gemini_flash | 0.272 | 0.745 | 0.346 | substantive | yes | poor | A |
| Cambridge_CUL_T_S_13J17_9 | gpt_5_6_sol | 0.283 | 0.728 | 0.626 | substantive | yes | poor | B |
| New_York_JTS_ENA_NS_50_32 | qwen3_vl_8b_heb_v17_step800 | 0.302 | 0.664 | 0.352 | substantive | yes | poor | A |
| Cambridge_CUL_T_S_13J26_14 | gemini_flash | 0.303 | 0.774 | 0.440 | substantive | yes | poor | A |
| Cambridge_CUL_T_S_8_137 | gemini_flash | 0.309 | 0.761 | 0.387 | substantive | yes | poor | B |
| Cambridge_CUL_T_S_8J24_6 | gemini_pro | 0.310 | 0.714 | 0.347 | substantive | yes | poor | A |
| Cambridge_Mosseri_VII_67_2 | gemini_pro | 0.311 | 0.747 | 0.332 | substantive | yes | poor | A |
| Oxford_Bodleian_MS_heb_d_66_22 | gpt_5_6_sol | 0.320 | 0.786 | 0.746 | substantive | yes | poor | B |
| Cambridge_CUL_T_S_16_110 | gemini_flash | 0.333 | 0.687 | 0.895 | substantive | yes | unusable | B |
| Cambridge_CUL_T_S_13J23_7 | gemini_pro | 0.387 | 0.846 | 0.390 | substantive | yes | poor | A |
| Cambridge_CUL_T_S_10J12_4 | kraken_seg | 0.405 | 0.797 | 0.320 | substantive | yes | poor | B |
| Oxford_Bodleian_MS_heb_a_2_4 | gpt_5_6_sol | 0.427 | 0.715 | 0.402 | substantive | yes | poor | A |
| Cambridge_CUL_T_S_Misc_35_11 | gemini_pro | 0.469 | 0.863 | 0.240 | substantive | yes | poor | A |
| Cambridge_CUL_T_S_10J2_3 | gemini_flash | 0.496 | 0.881 | 0.204 | substantive | yes | fair | A |
| Cambridge_CUL_T_S_12_3 | kraken_seg | 0.501 | 0.834 | 0.244 | substantive | yes | poor | A |
| Cambridge_CUL_T_S_10J2_3 | gpt_5_6_sol | 0.589 | 0.918 | 0.644 | substantive | yes | poor | A |
| New_York_JTS_ENA_4007_6 | gemini_pro | 0.594 | 0.898 | 0.197 | substantive | yes | poor | B |
| Oxford_Bodleian_MS_heb_a_2_4 | gemini_pro | 0.601 | 0.851 | 0.210 | substantive | yes | fair | A |
| Cambridge_CUL_T_S_13J17_9 | gemini_pro | 0.613 | 0.907 | 0.282 | substantive | yes | fair | B |
| Cambridge_CUL_T_S_16_124 | gemini_flash | 0.660 | 0.910 | 0.217 | substantive | yes | fair | A |
| Manchester_JRL_A_960 | vision_ocr_seg | 0.709 | 0.896 | 0.159 | substantive | yes | poor | A |
| Oxford_Bodleian_MS_heb_c_28_66 | gemini_pro | 0.731 | 0.934 | 0.094 | substantive | yes | good | A |
| Manchester_JRL_A_960 | kraken_seg | 0.740 | 0.914 | 0.219 | substantive | yes | fair | A |
| Manchester_JRL_A_960 | claude_opus_4_8 | 0.760 | 0.939 | 0.128 | substantive | yes | fair | A |
| Manchester_JRL_A_960 | gemini_flash | 0.770 | 0.915 | 0.136 | substantive | yes | fair | A |
| Manchester_JRL_A_960 | gpt_5_6_sol | 0.808 | 0.930 | 0.118 | substantive | yes | fair | A |
| Cambridge_CUL_T_S_12_3 | gpt_5_6_sol | 0.850 | 0.967 | 0.986 | abstained | no | unusable | A |
| Manchester_JRL_A_960 | gemini_pro | 0.868 | 0.958 | 0.059 | substantive | yes | good | A |

34/44 canonical-completion outputs are classed **substantive** by the metric; median n-gram precision 0.310 vs 0.050 over all judged outputs.

### Judge 'hallucination' label: marks vs invented text

| example flavour | count | share |
|---|---|---|
| invented_text_hebrew_pair | 452 | 0.474 |
| apparatus_marks | 274 | 0.288 |
| invented_text_described | 227 | 0.238 |

Apparatus-mark complaints: 274/953 (28.8%) of severe-hallucination examples; invented text (both flavours) 679/953 (71.2%). Matched tokens: `rafeh`, `rafe`, `[?]`, `nikud`, `niqqud`, `vowel`, `mark`, `diacrit`, `dagesh`, `cantillation`, `punctuation`, `ֿ`, `ְ`.

## Part 4 — scholar validation sample

| bin | population | target | sampled |
|---|---|---|---|
| [0,0.05) | 610 | 10 | 16 |
| [0.05,0.10) | 121 | 30 | 31 |
| [0.10,0.20) | 161 | 30 | 31 |
| [0.20,0.35) | 192 | 30 | 41 |
| [0.35,0.60) | 178 | 10 | 16 |
| [0.60,1] | 90 | 10 | 20 |
| TOTAL | 1352 | 120 | 155 |

Draw reasons: canonical_completion 35, bin_quota 120; seed 20260916. Bins can exceed their target because every judged canonical_completion output is added on top of the quota draw.

Secondary stratification by system (quota draws only):

| bin | quota drawn | per-system cap | heaviest system | its count | its share of bin |
|---|---|---|---|---|---|
| [0,0.05) | 10 | ceil(0.25 x 10) = 3 | qwen3_vl_8b_heb_v16_step1100 | 3 | 0.300 |
| [0.05,0.10) | 30 | ceil(0.25 x 30) = 8 | gemini_pro | 7 | 0.233 |
| [0.10,0.20) | 30 | ceil(0.25 x 30) = 8 | gemini_flash | 7 | 0.233 |
| [0.20,0.35) | 30 | ceil(0.25 x 30) = 8 | kraken_raw | 6 | 0.200 |
| [0.35,0.60) | 10 | ceil(0.25 x 10) = 3 | kraken_raw | 3 | 0.300 |
| [0.60,1] | 10 | ceil(0.25 x 10) = 3 | kraken_raw | 3 | 0.300 |

| system | in sample | share |
|---|---|---|
| kraken_seg | 13 | 0.084 |
| kraken_raw | 15 | 0.097 |
| gemini_pro | 22 | 0.142 |
| gemini_flash | 23 | 0.148 |
| claude_opus_4_8 | 10 | 0.065 |
| claude_sonnet_5 | 13 | 0.084 |
| gpt_5_6_sol | 17 | 0.110 |
| qwen3_vl_8b_heb_v17_step800 | 21 | 0.135 |
| qwen3_vl_8b_heb_v16_step1100 | 11 | 0.071 |
| qwen3_vl_8b | 3 | 0.019 |
| vision_ocr_seg | 7 | 0.045 |

Written to `E/scholar_sample.csv` with hyp/GT text files in `E/scholar_sample_pairs/`.
