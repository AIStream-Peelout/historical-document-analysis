# Analysis B — clipped vs unclipped n-gram precision

Metric: character 5-gram precision, `hebrew` letter set, 131 verified fragments x 11 paper systems. `unclipped` = paper metric (window ∈ set of reference windows); `clipped` = BLEU-style modified precision (each window earns credit at most as often as the reference contains it). `gap = unclipped - clipped`.

## 1a. Gap over all outputs, per system

| system | n | mean gap | median gap | p90 gap | max gap | gap>=0.05 | gap>=0.10 | gap>=0.20 |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 122 | 0.008 | 0.001 | 0.008 | 0.223 | 4 | 3 | 1 |
| kraken_raw | 131 | 0.002 | 0.001 | 0.006 | 0.013 | 0 | 0 | 0 |
| gemini_pro | 68 | 0.009 | 0.006 | 0.019 | 0.060 | 1 | 0 | 0 |
| gemini_flash | 122 | 0.020 | 0.004 | 0.025 | 0.421 | 10 | 5 | 3 |
| claude_opus_4_8 | 131 | 0.003 | 0.000 | 0.003 | 0.153 | 2 | 1 | 0 |
| claude_sonnet_5 | 131 | 0.003 | 0.000 | 0.004 | 0.131 | 2 | 1 | 0 |
| gpt_5_6_sol | 131 | 0.003 | 0.000 | 0.011 | 0.040 | 0 | 0 | 0 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0.033 | 0.004 | 0.089 | 0.415 | 21 | 13 | 9 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0.005 | 0.000 | 0.005 | 0.111 | 5 | 1 | 0 |
| qwen3_vl_8b | 131 | 0.001 | 0.000 | 0.000 | 0.125 | 1 | 1 | 0 |
| vision_ocr_seg | 123 | 0.000 | 0.000 | 0.001 | 0.007 | 0 | 0 | 0 |

## 1b. Same, restricted to outputs substantive under PAPER_CONFIG

| system | n | mean gap | median gap | p90 gap | max gap | gap>=0.05 | gap>=0.10 | gap>=0.20 |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 119 | 0.008 | 0.001 | 0.008 | 0.223 | 4 | 3 | 1 |
| kraken_raw | 127 | 0.002 | 0.001 | 0.006 | 0.013 | 0 | 0 | 0 |
| gemini_pro | 40 | 0.011 | 0.008 | 0.019 | 0.060 | 1 | 0 | 0 |
| gemini_flash | 52 | 0.031 | 0.006 | 0.073 | 0.419 | 8 | 4 | 2 |
| claude_opus_4_8 | 53 | 0.006 | 0.000 | 0.006 | 0.153 | 2 | 1 | 0 |
| claude_sonnet_5 | 29 | 0.009 | 0.001 | 0.008 | 0.131 | 2 | 1 | 0 |
| gpt_5_6_sol | 36 | 0.009 | 0.001 | 0.024 | 0.040 | 0 | 0 | 0 |
| qwen3_vl_8b_heb_v17_step800 | 87 | 0.036 | 0.004 | 0.147 | 0.362 | 13 | 11 | 8 |
| qwen3_vl_8b_heb_v16_step1100 | 18 | 0.002 | 0.001 | 0.004 | 0.015 | 0 | 0 | 0 |
| qwen3_vl_8b | 2 | 0.003 | 0.003 | 0.004 | 0.005 | 0 | 0 | 0 |
| vision_ocr_seg | 38 | 0.001 | 0.000 | 0.003 | 0.007 | 0 | 0 | 0 |

## 1c. Behaviour re-classification under clipping (same 0.10 cutoff)

| system | substantive (paper) | substantive (clipped) | subst -> hallucinated flips | flipped doc_ids |
|---|---|---|---|---|
| kraken_seg | 119 | 119 | 0 | — |
| kraken_raw | 127 | 127 | 0 | — |
| gemini_pro | 40 | 37 | 3 | Cambridge_CUL_T_S_10J14_13, Cambridge_CUL_T_S_10J16_19, New_York_JTS_ENA_4020_16 |
| gemini_flash | 52 | 46 | 6 | Cambridge_CUL_T_S_10J13_9, Cambridge_CUL_T_S_10J21_1, Cambridge_CUL_T_S_13J17_12, Cambridge_CUL_T_S_13J21_33, Cambridge_CUL_T_S_18J1_32, Cambridge_CUL_T_S_20_63 |
| claude_opus_4_8 | 53 | 52 | 1 | Cambridge_CUL_T_S_13J8_6 |
| claude_sonnet_5 | 29 | 27 | 2 | Cambridge_CUL_T_S_13J26_14, Cambridge_Mosseri_VII_67_2 |
| gpt_5_6_sol | 36 | 36 | 0 | — |
| qwen3_vl_8b_heb_v17_step800 | 87 | 73 | 14 | Cambridge_CUL_T_S_10J13_9, Cambridge_CUL_T_S_10J5_19, Cambridge_CUL_T_S_13J17_12, Cambridge_CUL_T_S_13J2_14, Cambridge_CUL_T_S_13J2_9, Cambridge_CUL_T_S_13J6_23, Cambridge_CUL_T_S_16_117, Cambridge_CUL_T_S_16_138, Cambridge_CUL_T_S_20_28, Cambridge_CUL_T_S_20_4, Cambridge_CUL_T_S_24_51, New_York_JTS_ENA_NS_39_9, Oxford_Bodleian_MS_heb_b_11_7, Oxford_Bodleian_MS_heb_d_66_32 |
| qwen3_vl_8b_heb_v16_step1100 | 18 | 18 | 0 | — |
| qwen3_vl_8b | 2 | 2 | 0 | — |
| vision_ocr_seg | 38 | 38 | 0 | — |

## 1d. Tier A (0.25 convergence cutoff)

| scoring | Tier A fragments |
|---|---|
| unclipped (paper) | 55 |
| clipped | 50 |

## 2. Largest-gap fragments per system

### kraken_seg

Top 5-gram column = occurrences of the hypothesis's single most repeated 5-gram in the hypothesis / in the reference. The window itself (Hebrew text) is in the pair files.

| # | doc_id | unclipped | clipped | gap | hyp L | gt L | len ratio | aligned P | aligned R | tier | bucket | top 5-gram hyp/gt | distinct over-used 5-grams |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Cambridge_CUL_T_S_Misc_35_11 | 0.492 | 0.269 | 0.223 | 974 | 878 | 1.109 | 0.420 | 0.509 | A | hebrew | 6/3 | 588 |
| 2 | Oxford_Bodleian_MS_heb_a_2_4 | 0.463 | 0.264 | 0.199 | 557 | 1262 | 0.441 | 0.210 | 0.162 | A | aramaic | 10/1 | 270 |
| 3 | Cambridge_CUL_T_S_10J21_1 | 0.556 | 0.383 | 0.173 | 615 | 726 | 0.847 | 0.367 | 0.483 | A | judaeo_arabic | 5/0 | 282 |
| 4 | Cambridge_CUL_T_S_16_138 | 0.577 | 0.492 | 0.085 | 453 | 1644 | 0.276 | 0.316 | 0.167 | A | judaeo_arabic | 4/1 | 199 |
| 5 | Cambridge_CUL_T_S_20_101 | 0.141 | 0.102 | 0.039 | 835 | 1369 | 0.610 | 0.152 | 0.116 | B | aramaic | 22/0 | 230 |
| 6 | Cambridge_CUL_T_S_20_4 | 0.549 | 0.512 | 0.037 | 840 | 4125 | 0.204 | 0.669 | 0.156 | A | judaeo_arabic | 5/1 | 372 |
| 7 | Cambridge_CUL_T_S_16_117 | 0.398 | 0.370 | 0.028 | 760 | 3267 | 0.233 | 0.474 | 0.145 | A | judaeo_arabic | 3/0 | 399 |
| 8 | Cambridge_CUL_T_S_Misc_8_103 | 0.241 | 0.228 | 0.013 | 1318 | 2083 | 0.633 | 0.350 | 0.222 | B | judaeo_arabic | 5/2 | 999 |
| 9 | New_York_JTS_ENA_NS_18_25 | 0.559 | 0.549 | 0.011 | 1143 | 1201 | 0.952 | 0.748 | 0.711 | B | judaeo_arabic | 5/6 | 512 |
| 10 | Cambridge_CUL_T_S_16_235 | 0.503 | 0.494 | 0.009 | 1026 | 980 | 1.047 | 0.824 | 0.871 | B | judaeo_arabic | 3/2 | 514 |
| 11 | Cambridge_CUL_T_S_20_21 | 0.717 | 0.709 | 0.008 | 1946 | 1949 | 0.998 | 0.901 | 0.902 | A | judaeo_arabic | 5/5 | 562 |
| 12 | Cambridge_CUL_T_S_13J21_33 | 0.615 | 0.607 | 0.008 | 753 | 905 | 0.832 | 0.873 | 0.727 | B | judaeo_arabic | 2/1 | 294 |
| 13 | Cambridge_CUL_T_S_10J13_9 | 0.534 | 0.526 | 0.008 | 899 | 1019 | 0.882 | 0.881 | 0.789 | B | judaeo_arabic | 6/8 | 424 |
| 14 | Cambridge_CUL_T_S_12_3 | 0.501 | 0.493 | 0.008 | 1581 | 1607 | 0.984 | 0.834 | 0.812 | A | judaeo_arabic | 3/0 | 789 |
| 15 | Cambridge_CUL_T_S_13J23_4 | 0.477 | 0.470 | 0.007 | 825 | 906 | 0.911 | 0.839 | 0.775 | B | judaeo_arabic | 3/3 | 435 |

### gemini_pro

Top 5-gram column = occurrences of the hypothesis's single most repeated 5-gram in the hypothesis / in the reference. The window itself (Hebrew text) is in the pair files.

| # | doc_id | unclipped | clipped | gap | hyp L | gt L | len ratio | aligned P | aligned R | tier | bucket | top 5-gram hyp/gt | distinct over-used 5-grams |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Cambridge_CUL_T_S_16_332 | 0.289 | 0.229 | 0.060 | 1553 | 1887 | 0.823 | 0.427 | 0.348 | B | untagged | 7/3 | 927 |
| 2 | Cambridge_CUL_T_S_13J21_33 | 0.087 | 0.050 | 0.037 | 1086 | 905 | 1.200 | 0.195 | 0.235 | B | judaeo_arabic | 8/2 | 810 |
| 3 | Cambridge_CUL_T_S_10J16_19 | 0.108 | 0.073 | 0.035 | 948 | 1149 | 0.825 | 0.214 | 0.179 | A | judaeo_arabic | 12/0 | 666 |
| 4 | Cambridge_CUL_T_S_10J14_13 | 0.118 | 0.088 | 0.029 | 513 | 672 | 0.763 | 0.482 | 0.368 | B | judaeo_arabic | 4/1 | 427 |
| 5 | Cambridge_CUL_T_S_13J5_3 | 0.235 | 0.206 | 0.029 | 557 | 1448 | 0.385 | 0.522 | 0.201 | A | judaeo_arabic | 3/0 | 408 |
| 6 | Cambridge_CUL_T_S_Ar_18_49 | 0.094 | 0.068 | 0.027 | 756 | 768 | 0.984 | 0.279 | 0.275 | B | untagged | 6/4 | 565 |
| 7 | Cambridge_CUL_T_S_16_208 | 0.036 | 0.016 | 0.020 | 1067 | 1248 | 0.855 | 0.119 | 0.107 | B | untagged | 28/0 | 638 |
| 8 | Cambridge_CUL_T_S_13J2_14 | 0.061 | 0.043 | 0.018 | 772 | 891 | 0.866 | 0.248 | 0.217 | A | judaeo_arabic | 7/1 | 652 |
| 9 | Oxford_Bodleian_MS_heb_a_2_4 | 0.601 | 0.584 | 0.017 | 1268 | 1262 | 1.005 | 0.851 | 0.850 | A | aramaic | 8/8 | 502 |
| 10 | Cambridge_CUL_T_S_Misc_8_103 | 0.052 | 0.035 | 0.017 | 119 | 2083 | 0.057 | 0.395 | 0.022 | B | judaeo_arabic | 3/1 | 108 |
| 11 | Cambridge_CUL_T_S_AS_147_38 | 0.300 | 0.282 | 0.017 | 528 | 798 | 0.662 | 0.620 | 0.426 | A | judaeo_arabic | 5/4 | 366 |
| 12 | New_York_JTS_ENA_4020_16 | 0.114 | 0.098 | 0.016 | 1421 | 1352 | 1.051 | 0.299 | 0.315 | B | judaeo_arabic | 11/0 | 1015 |
| 13 | Cambridge_CUL_Or_1080_J262 | 0.430 | 0.415 | 0.016 | 766 | 788 | 0.972 | 0.813 | 0.787 | A | judaeo_arabic | 6/4 | 408 |
| 14 | Manchester_JRL_A_960 | 0.868 | 0.853 | 0.016 | 513 | 506 | 1.014 | 0.958 | 0.973 | A | aramaic | 3/3 | 75 |
| 15 | New_York_JTS_ENA_NS_18_25 | 0.059 | 0.043 | 0.016 | 1024 | 1201 | 0.853 | 0.189 | 0.161 | B | judaeo_arabic | 7/3 | 745 |

### qwen3_vl_8b_heb_v17_step800

Top 5-gram column = occurrences of the hypothesis's single most repeated 5-gram in the hypothesis / in the reference. The window itself (Hebrew text) is in the pair files.

| # | doc_id | unclipped | clipped | gap | hyp L | gt L | len ratio | aligned P | aligned R | tier | bucket | top 5-gram hyp/gt | distinct over-used 5-grams |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Cambridge_CUL_T_S_16_125 | 0.419 | 0.005 | 0.415 | 16748 | 1157 | 14.475 | 0.024 | 0.340 | A | untagged | 805/2 | 545 |
| 2 | Cambridge_CUL_T_S_16_138 | 0.390 | 0.029 | 0.362 | 13969 | 1644 | 8.497 | 0.069 | 0.575 | A | judaeo_arabic | 46/2 | 1013 |
| 3 | Cambridge_CUL_T_S_13J17_12 | 0.325 | 0.002 | 0.322 | 13099 | 1401 | 9.350 | 0.009 | 0.084 | A | judaeo_arabic | 654/4 | 60 |
| 4 | Oxford_Bodleian_MS_heb_d_66_32 | 0.314 | 0.008 | 0.306 | 13264 | 627 | 21.155 | 0.027 | 0.581 | A | untagged | 543/2 | 446 |
| 5 | Cambridge_CUL_T_S_20_4 | 0.308 | 0.048 | 0.260 | 13590 | 4125 | 3.295 | 0.128 | 0.415 | A | judaeo_arabic | 192/6 | 1533 |
| 6 | Cambridge_CUL_T_S_16_117 | 0.261 | 0.002 | 0.259 | 11131 | 3267 | 3.407 | 0.011 | 0.043 | A | judaeo_arabic | 580/4 | 144 |
| 7 | Oxford_Bodleian_MS_heb_b_11_7 | 0.269 | 0.013 | 0.257 | 11560 | 1698 | 6.808 | 0.066 | 0.445 | A | judaeo_arabic | 251/0 | 1023 |
| 8 | Cambridge_CUL_T_S_10J13_9 | 0.233 | 0.008 | 0.225 | 14106 | 1019 | 13.843 | 0.033 | 0.467 | B | judaeo_arabic | 712/8 | 626 |
| 9 | New_York_JTS_ENA_NS_39_9 | 0.222 | 0.005 | 0.217 | 13101 | 592 | 22.130 | 0.017 | 0.383 | B | judaeo_arabic | 243/5 | 208 |
| 10 | Cambridge_CUL_T_S_13J6_23 | 0.192 | 0.002 | 0.189 | 12846 | 683 | 18.808 | 0.010 | 0.177 | A | judaeo_arabic | 273/0 | 150 |
| 11 | Cambridge_CUL_T_S_13J2_14 | 0.120 | 0.001 | 0.119 | 11770 | 891 | 13.210 | 0.006 | 0.082 | A | judaeo_arabic | 353/0 | 184 |
| 12 | Cambridge_CUL_T_S_16_208 | 0.115 | 0.000 | 0.114 | 12521 | 1248 | 10.033 | 0.008 | 0.084 | B | untagged | 482/0 | 77 |
| 13 | Cambridge_CUL_T_S_20_28 | 0.107 | 0.000 | 0.106 | 13278 | 1523 | 8.718 | 0.003 | 0.026 | B | judaeo_arabic | 565/1 | 53 |
| 14 | Oxford_Bodleian_MS_heb_d_79_36 | 0.091 | 0.002 | 0.089 | 12883 | 1141 | 11.291 | 0.015 | 0.173 | B | judaeo_arabic | 385/0 | 221 |
| 15 | Cambridge_CUL_T_S_AS_149_9 | 0.091 | 0.002 | 0.088 | 12438 | 516 | 24.105 | 0.020 | 0.481 | B | untagged | 277/2 | 305 |

### claude_opus_4_8 (short form)

| # | doc_id | unclipped | clipped | gap | len ratio | top 5-gram hyp/gt | distinct over-used 5-grams |
|---|---|---|---|---|---|---|---|
| 1 | New_York_JTS_ENA_NS_50_32 | 0.341 | 0.188 | 0.153 | 1.506 | 4/3 | 912 |
| 2 | Cambridge_CUL_T_S_13J8_6 | 0.158 | 0.084 | 0.074 | 1.882 | 6/4 | 1262 |
| 3 | New_York_JTS_ENA_4007_6 | 0.292 | 0.268 | 0.024 | 0.933 | 6/6 | 501 |
| 4 | Cambridge_CUL_T_S_24_13 | 0.271 | 0.251 | 0.019 | 1.065 | 5/6 | 1227 |
| 5 | Cambridge_CUL_T_S_13J23_7 | 0.341 | 0.326 | 0.015 | 0.829 | 3/1 | 310 |
| 6 | Cambridge_CUL_T_S_20_63 | 0.195 | 0.189 | 0.006 | 0.789 | 3/1 | 1130 |
| 7 | Cambridge_CUL_T_S_8_137 | 0.207 | 0.202 | 0.006 | 0.920 | 2/0 | 389 |
| 8 | Cambridge_CUL_T_S_16_128 | 0.069 | 0.065 | 0.005 | 1.031 | 2/1 | 791 |
| 9 | Cambridge_CUL_T_S_18J2_16 | 0.296 | 0.291 | 0.005 | 0.987 | 3/2 | 890 |
| 10 | Manchester_JRL_A_960 | 0.760 | 0.756 | 0.004 | 0.931 | 3/3 | 114 |
| 11 | Cambridge_CUL_T_S_20_110 | 0.121 | 0.117 | 0.004 | 0.220 | 2/3 | 218 |
| 12 | Cambridge_CUL_T_S_18J1_14 | 0.345 | 0.342 | 0.003 | 0.958 | 2/0 | 383 |
| 13 | Oxford_Bodleian_MS_heb_a_2_4 | 0.432 | 0.429 | 0.003 | 0.979 | 5/0 | 677 |
| 14 | Cambridge_CUL_T_S_20_21 | 0.231 | 0.228 | 0.003 | 0.953 | 3/0 | 1383 |
| 15 | Cambridge_CUL_T_S_16_208 | 0.060 | 0.057 | 0.003 | 0.826 | 4/0 | 941 |

### gemini_flash (short form)

| # | doc_id | unclipped | clipped | gap | len ratio | top 5-gram hyp/gt | distinct over-used 5-grams |
|---|---|---|---|---|---|---|---|
| 1 | New_York_JTS_ENA_NS_50_32 | 0.443 | 0.022 | 0.421 | 0.466 | 24/2 | 25 |
| 2 | Oxford_Bodleian_MS_heb_a_2_4 | 0.605 | 0.186 | 0.419 | 0.574 | 10/1 | 186 |
| 3 | Cambridge_CUL_T_S_12_146 | 0.517 | 0.204 | 0.312 | 0.446 | 6/2 | 187 |
| 4 | Cambridge_CUL_T_S_13J21_33 | 0.183 | 0.024 | 0.159 | 1.160 | 32/1 | 454 |
| 5 | Cambridge_CUL_T_S_20_63 | 0.138 | 0.005 | 0.133 | 0.432 | 28/0 | 60 |
| 6 | Cambridge_CUL_T_S_10J13_9 | 0.105 | 0.025 | 0.080 | 0.790 | 24/1 | 353 |
| 7 | Cambridge_CUL_T_S_13J17_12 | 0.113 | 0.039 | 0.074 | 0.550 | 7/0 | 212 |
| 8 | Cambridge_CUL_T_S_18J1_32 | 0.149 | 0.086 | 0.062 | 0.895 | 15/2 | 941 |
| 9 | Cambridge_CUL_T_S_10J21_1 | 0.137 | 0.080 | 0.057 | 0.800 | 8/0 | 197 |
| 10 | Cambridge_CUL_T_S_8J8_22 | 0.068 | 0.011 | 0.057 | 1.143 | 28/0 | 199 |
| 11 | Cambridge_CUL_T_S_Misc_8_103 | 0.058 | 0.012 | 0.046 | 0.323 | 13/0 | 168 |
| 12 | Cambridge_CUL_T_S_18J1_27 | 0.175 | 0.148 | 0.027 | 0.984 | 6/4 | 693 |
| 13 | Cambridge_CUL_T_S_16_208 | 0.042 | 0.016 | 0.025 | 0.698 | 20/0 | 281 |
| 14 | Cambridge_CUL_T_S_20_110 | 0.184 | 0.160 | 0.024 | 0.887 | 4/2 | 813 |
| 15 | Cambridge_CUL_T_S_10J13_3 | 0.033 | 0.010 | 0.024 | 1.048 | 15/1 | 341 |

## 4. What drives the gap

Spearman rho of gap against hypothesis length inflation, per system (all 131 outputs each). `len_ratio` = hyp letters / gt letters; `len_diff` = hyp letters - gt letters.

| system | rho(gap, len_ratio) | p | rho(gap, len_diff) | p | mean gap, len_ratio<=1.05 | mean gap, len_ratio>1.05 | n short/normal | n long |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 0.217 | 0.0165 | 0.160 | 0.0788 | 0.006 | 0.112 | 120 | 2 |
| kraken_raw | 0.382 | 0.0000 | 0.237 | 0.0064 | 0.002 | 0.001 | 129 | 2 |
| gemini_pro | 0.304 | 0.0117 | 0.193 | 0.1156 | 0.009 | 0.018 | 65 | 3 |
| gemini_flash | 0.342 | 0.0001 | 0.287 | 0.0013 | 0.019 | 0.037 | 115 | 7 |
| claude_opus_4_8 | 0.514 | 0.0000 | 0.456 | 0.0000 | 0.001 | 0.025 | 121 | 10 |
| claude_sonnet_5 | 0.419 | 0.0000 | 0.355 | 0.0000 | 0.001 | 0.040 | 124 | 7 |
| gpt_5_6_sol | 0.629 | 0.0000 | 0.511 | 0.0000 | 0.003 | n/a | 131 | 0 |
| qwen3_vl_8b_heb_v17_step800 | 0.411 | 0.0000 | 0.383 | 0.0000 | 0.005 | 0.096 | 91 | 40 |
| qwen3_vl_8b_heb_v16_step1100 | -0.008 | 0.9304 | 0.028 | 0.7518 | 0.001 | 0.009 | 68 | 63 |
| qwen3_vl_8b | -0.082 | 0.3525 | 0.017 | 0.8436 | 0.000 | 0.002 | 17 | 114 |
| vision_ocr_seg | 0.475 | 0.0000 | 0.352 | 0.0001 | 0.000 | n/a | 123 | 0 |

Pooled over all 1352 outputs: rho(gap, len_ratio) = 0.291 (p = 0.0000), rho(gap, len_diff) = 0.244 (p = 0.0000). Mean gap for len_ratio <= 1.05: 0.004 (n = 1104); for len_ratio > 1.05: 0.023 (n = 248).

**Reading.** Of the 46 outputs with gap >= 0.05, 33 (72%) are longer than the reference (len_ratio > 1.05) — the over-generation / loop channel, dominated by HebVL-1.7 — but 13 are NOT: gemini_flash x8, kraken_seg x3, gemini_pro x1, qwen3_vl_8b_heb_v17_step800 x1. Those are short or normal-length outputs (len_ratio 0.28-1.04) that recycle a formula internally, so clipping bites there too. The correlation is real but modest (pooled rho = 0.291), and the per-system rho is driven by a handful of long outputs: for the two Qwen baselines, where most outputs are long for unrelated reasons, rho is ~0.

## 5. Gap at n=3 and n=8

| system | mean gap n=3 | p90 gap n=3 | mean gap n=5 | p90 gap n=5 | mean gap n=8 | p90 gap n=8 |
|---|---|---|---|---|---|---|
| kraken_seg | 0.040 | 0.055 | 0.008 | 0.008 | 0.004 | 0.001 |
| kraken_raw | 0.033 | 0.051 | 0.002 | 0.006 | 0.000 | 0.001 |
| gemini_pro | 0.082 | 0.142 | 0.009 | 0.019 | 0.002 | 0.005 |
| gemini_flash | 0.121 | 0.286 | 0.020 | 0.025 | 0.008 | 0.004 |
| claude_opus_4_8 | 0.026 | 0.057 | 0.003 | 0.003 | 0.001 | 0.000 |
| claude_sonnet_5 | 0.037 | 0.066 | 0.003 | 0.004 | 0.001 | 0.000 |
| gpt_5_6_sol | 0.019 | 0.064 | 0.003 | 0.011 | 0.001 | 0.000 |
| qwen3_vl_8b_heb_v17_step800 | 0.156 | 0.466 | 0.033 | 0.089 | 0.010 | 0.018 |
| qwen3_vl_8b_heb_v16_step1100 | 0.123 | 0.318 | 0.005 | 0.005 | 0.000 | 0.000 |
| qwen3_vl_8b | 0.041 | 0.124 | 0.001 | 0.000 | 0.000 | 0.000 |
| vision_ocr_seg | 0.013 | 0.037 | 0.000 | 0.001 | 0.000 | 0.000 |
