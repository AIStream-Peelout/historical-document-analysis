# Analysis A — the formulaic floor of the order-independent 5-gram metric

Every output classified **substantive** under the paper config (`n5 unclipped hebrew h0.10 t0.25 l0.45 a25`) is scored with the exact paper metric (character n-grams over Hebrew-block letters, set membership in the reference, unclipped) against **every other** benchmark fragment's ground truth.  A wrong-reference score is pure floor: the hypothesis and the reference come from different manuscripts.

- substantive outputs: **601** across 11 systems
- references: **131** fragments (each output scored against 130 wrong references + its own)
- wrong-reference pairs at n=5: **78,130**
- cutoffs: hallucination `< 0.10`, Tier A `>= 0.25`

## 1. Per-system wrong-reference floor (n=5, unclipped)

| system | n subst. | wrong pairs | wrong med | wrong p90 | wrong p99 | wrong max | share >= 0.10 | share >= 0.25 | own med |
|---|---|---|---|---|---|---|---|---|---|
| kraken_seg | 119 | 15,470 | 0.006 | 0.026 | 0.078 | 0.300 | 0.006 | 0.000 | 0.477 |
| kraken_raw | 127 | 16,510 | 0.006 | 0.025 | 0.075 | 0.293 | 0.005 | 0.000 | 0.477 |
| gemini_pro | 40 | 5,200 | 0.006 | 0.034 | 0.114 | 0.316 | 0.014 | 0.001 | 0.294 |
| gemini_flash | 52 | 6,760 | 0.005 | 0.033 | 0.145 | 0.596 | 0.020 | 0.004 | 0.265 |
| claude_opus_4_8 | 53 | 6,890 | 0.002 | 0.011 | 0.045 | 0.238 | 0.001 | 0.000 | 0.203 |
| claude_sonnet_5 | 29 | 3,770 | 0.002 | 0.010 | 0.047 | 0.190 | 0.001 | 0.000 | 0.190 |
| gpt_5_6_sol | 36 | 4,680 | 0.005 | 0.043 | 0.181 | 0.643 | 0.036 | 0.005 | 0.295 |
| qwen3_vl_8b_heb_v17_step800 | 87 | 11,310 | 0.007 | 0.031 | 0.128 | 0.447 | 0.016 | 0.003 | 0.246 |
| qwen3_vl_8b_heb_v16_step1100 | 18 | 2,340 | 0.002 | 0.009 | 0.028 | 0.145 | 0.000 | 0.000 | 0.198 |
| qwen3_vl_8b | 2 | 260 | 0.000 | 0.008 | 0.034 | 0.052 | 0.000 | 0.000 | 0.197 |
| vision_ocr_seg | 38 | 4,940 | 0.002 | 0.013 | 0.054 | 0.236 | 0.003 | 0.000 | 0.204 |

### 1b. Per-output maximum over the 130 wrong references

"Would this output have passed the gate against the best wrong manuscript in the benchmark?"

| system | n subst. | max-wrong med | max-wrong p90 | max-wrong max | n outputs max >= 0.10 | n outputs max >= 0.25 |
|---|---|---|---|---|---|---|
| kraken_seg | 119 | 0.051 | 0.156 | 0.300 | 18 (0.151) | 4 (0.034) |
| kraken_raw | 127 | 0.050 | 0.134 | 0.293 | 19 (0.150) | 4 (0.031) |
| gemini_pro | 40 | 0.056 | 0.193 | 0.316 | 11 (0.275) | 4 (0.100) |
| gemini_flash | 52 | 0.079 | 0.244 | 0.596 | 17 (0.327) | 5 (0.096) |
| claude_opus_4_8 | 53 | 0.030 | 0.095 | 0.238 | 4 (0.075) | 0 (0.000) |
| claude_sonnet_5 | 29 | 0.024 | 0.059 | 0.190 | 1 (0.034) | 0 (0.000) |
| gpt_5_6_sol | 36 | 0.122 | 0.337 | 0.643 | 22 (0.611) | 8 (0.222) |
| qwen3_vl_8b_heb_v17_step800 | 87 | 0.056 | 0.161 | 0.447 | 22 (0.253) | 6 (0.069) |
| qwen3_vl_8b_heb_v16_step1100 | 18 | 0.026 | 0.070 | 0.145 | 1 (0.056) | 0 (0.000) |
| qwen3_vl_8b | 2 | 0.040 | 0.050 | 0.052 | 0 (0.000) | 0 (0.000) |
| vision_ocr_seg | 38 | 0.035 | 0.110 | 0.236 | 6 (0.158) | 0 (0.000) |

## 2. Per-reference floor (n=5, unclipped)

For each fragment used as a reference: mean and max n-gram precision of all substantive outputs **of other fragments** (all systems pooled).

**15 most formulaic references (highest mean wrong-source precision)**

| ref doc_id | script bucket | gt letters | Tier A | pairs | mean wrong | max wrong | share >= 0.10 |
|---|---|---|---|---|---|---|---|
| Cambridge_CUL_T_S_16_117 | judaeo_arabic | 3267 | yes | 595 | 0.034 | 0.340 | 0.081 |
| Cambridge_CUL_T_S_20_21 | judaeo_arabic | 1949 | yes | 593 | 0.029 | 0.522 | 0.057 |
| Cambridge_CUL_T_S_13J2_6 | judaeo_arabic | 2013 | no | 598 | 0.028 | 0.262 | 0.038 |
| Cambridge_CUL_T_S_20_4 | judaeo_arabic | 4125 | yes | 596 | 0.028 | 0.269 | 0.013 |
| Cambridge_CUL_T_S_13J8_6 | judaeo_arabic | 1388 | yes | 596 | 0.027 | 0.596 | 0.049 |
| Cambridge_CUL_T_S_20_63 | judaeo_arabic | 1826 | yes | 596 | 0.027 | 0.277 | 0.032 |
| Cambridge_CUL_T_S_16_138 | judaeo_arabic | 1644 | yes | 594 | 0.026 | 0.435 | 0.052 |
| New_York_JTS_ENA_4020_16 | judaeo_arabic | 1352 | no | 598 | 0.025 | 0.293 | 0.043 |
| Cambridge_CUL_T_S_16_170 | judaeo_arabic | 1488 | yes | 596 | 0.022 | 0.522 | 0.049 |
| Cambridge_CUL_T_S_10J4_10 | judaeo_arabic | 878 | yes | 597 | 0.021 | 0.522 | 0.039 |
| Cambridge_CUL_T_S_18J1_32 | judaeo_arabic | 1403 | yes | 596 | 0.021 | 0.319 | 0.023 |
| Cambridge_CUL_T_S_20_98 | untagged | 1144 | yes | 595 | 0.020 | 0.522 | 0.022 |
| Cambridge_CUL_T_S_16_208 | untagged | 1248 | no | 599 | 0.019 | 0.404 | 0.030 |
| Cambridge_CUL_T_S_13J5_3 | judaeo_arabic | 1448 | yes | 595 | 0.019 | 0.522 | 0.022 |
| Cambridge_CUL_T_S_18J1_27 | untagged | 917 | no | 596 | 0.017 | 0.348 | 0.018 |

**15 least formulaic references**

| ref doc_id | script bucket | gt letters | Tier A | pairs | mean wrong | max wrong | share >= 0.10 |
|---|---|---|---|---|---|---|---|
| Cambridge_CUL_T_S_13J23_7 | hebrew | 578 | yes | 592 | 0.002 | 0.037 | 0.000 |
| Cambridge_CUL_T_S_18J3_18 | untagged | 648 | no | 600 | 0.003 | 0.100 | 0.002 |
| Cambridge_CUL_T_S_13J26_14 | untagged | 870 | yes | 594 | 0.003 | 0.045 | 0.000 |
| Cambridge_CUL_T_S_Misc_35_11 | hebrew | 878 | yes | 592 | 0.003 | 0.028 | 0.000 |
| Cambridge_CUL_T_S_10J24_3 | hebrew | 969 | no | 596 | 0.003 | 0.029 | 0.000 |
| Oxford_Bodleian_MS_heb_b_11_28 | hebrew | 814 | yes | 591 | 0.004 | 0.071 | 0.000 |
| Cambridge_CUL_T_S_10J12_9 | hebrew | 761 | yes | 591 | 0.004 | 0.114 | 0.002 |
| Cambridge_CUL_T_S_10J14_1 | hebrew | 615 | yes | 596 | 0.004 | 0.057 | 0.000 |
| Cambridge_CUL_T_S_10J9_15 | hebrew | 547 | no | 597 | 0.004 | 0.074 | 0.000 |
| New_York_JTS_ENA_NS_2_31 | judaeo_arabic | 497 | no | 599 | 0.004 | 0.085 | 0.000 |
| Cambridge_CUL_T_S_6J3_23 | hebrew | 769 | no | 599 | 0.004 | 0.031 | 0.000 |
| Cambridge_CUL_T_S_13J23_12 | hebrew | 855 | no | 595 | 0.004 | 0.043 | 0.000 |
| Oxford_Bodleian_MS_heb_f_108_63 | judaeo_arabic | 576 | no | 596 | 0.004 | 0.055 | 0.000 |
| Cambridge_CUL_T_S_16_232 | judaeo_arabic | 522 | no | 598 | 0.004 | 0.183 | 0.003 |
| Cambridge_CUL_T_S_10J13_11 | judaeo_arabic | 687 | yes | 596 | 0.004 | 0.042 | 0.000 |

Across all 131 references: mean-of-means 0.011, median-of-means 0.010, p90-of-means 0.019, max-of-means 0.034; 104 of 131 references are matched at >= 0.10 by at least one foreign output (45 at >= 0.25).

## 3. How the floor moves with n (unclipped, same outputs)

Wrong-reference pairs only.  n=5 is the paper's setting.

| system | n=3 med / p90 / share>=0.10 | n=4 med / p90 / share>=0.10 | n=5 med / p90 / share>=0.10 | n=6 med / p90 / share>=0.10 | n=8 med / p90 / share>=0.10 |
|---|---|---|---|---|---|
| kraken_seg | 0.220 / 0.339 / 0.964 | 0.036 / 0.084 / 0.057 | 0.006 / 0.026 / 0.006 | 0.001 / 0.011 / 0.003 | 0.000 / 0.003 / 0.001 |
| kraken_raw | 0.220 / 0.339 / 0.967 | 0.036 / 0.084 / 0.056 | 0.006 / 0.025 / 0.005 | 0.001 / 0.011 / 0.003 | 0.000 / 0.003 / 0.001 |
| gemini_pro | 0.213 / 0.357 / 0.955 | 0.035 / 0.095 / 0.089 | 0.006 / 0.034 / 0.014 | 0.001 / 0.017 / 0.006 | 0.000 / 0.006 / 0.002 |
| gemini_flash | 0.216 / 0.367 / 0.948 | 0.033 / 0.099 / 0.099 | 0.005 / 0.033 / 0.020 | 0.000 / 0.013 / 0.010 | 0.000 / 0.002 / 0.006 |
| claude_opus_4_8 | 0.166 / 0.277 / 0.870 | 0.020 / 0.049 / 0.012 | 0.002 / 0.011 / 0.001 | 0.000 / 0.004 / 0.000 | 0.000 / 0.000 / 0.000 |
| claude_sonnet_5 | 0.164 / 0.274 / 0.895 | 0.019 / 0.045 / 0.011 | 0.002 / 0.010 / 0.001 | 0.000 / 0.003 / 0.000 | 0.000 / 0.000 / 0.000 |
| gpt_5_6_sol | 0.224 / 0.381 / 0.963 | 0.038 / 0.114 / 0.133 | 0.005 / 0.043 / 0.036 | 0.000 / 0.021 / 0.019 | 0.000 / 0.003 / 0.011 |
| qwen3_vl_8b_heb_v17_step800 | 0.225 / 0.361 / 0.969 | 0.040 / 0.099 / 0.098 | 0.007 / 0.031 / 0.016 | 0.001 / 0.013 / 0.007 | 0.000 / 0.003 / 0.003 |
| qwen3_vl_8b_heb_v16_step1100 | 0.152 / 0.255 / 0.842 | 0.017 / 0.042 / 0.004 | 0.002 / 0.009 / 0.000 | 0.000 / 0.003 / 0.000 | 0.000 / 0.000 / 0.000 |
| qwen3_vl_8b | 0.119 / 0.223 / 0.642 | 0.012 / 0.035 / 0.000 | 0.000 / 0.008 / 0.000 | 0.000 / 0.003 / 0.000 | 0.000 / 0.000 / 0.000 |
| vision_ocr_seg | 0.171 / 0.299 / 0.886 | 0.020 / 0.055 / 0.024 | 0.002 / 0.013 / 0.003 | 0.000 / 0.004 / 0.000 | 0.000 / 0.000 / 0.000 |

## 4. Same table with clipping (n=5, clip=True)

| system | n subst. | wrong med | wrong p90 | wrong p99 | wrong max | share >= 0.10 | share >= 0.25 | own med |
|---|---|---|---|---|---|---|---|---|
| kraken_seg | 119 | 0.006 | 0.023 | 0.071 | 0.298 | 0.005 | 0.000 | 0.470 |
| kraken_raw | 127 | 0.006 | 0.022 | 0.070 | 0.290 | 0.005 | 0.000 | 0.475 |
| gemini_pro | 40 | 0.006 | 0.028 | 0.104 | 0.307 | 0.012 | 0.000 | 0.282 |
| gemini_flash | 52 | 0.004 | 0.024 | 0.110 | 0.596 | 0.013 | 0.003 | 0.262 |
| claude_opus_4_8 | 53 | 0.002 | 0.010 | 0.045 | 0.223 | 0.001 | 0.000 | 0.201 |
| claude_sonnet_5 | 29 | 0.002 | 0.008 | 0.035 | 0.190 | 0.001 | 0.000 | 0.174 |
| gpt_5_6_sol | 36 | 0.005 | 0.043 | 0.181 | 0.616 | 0.034 | 0.005 | 0.278 |
| qwen3_vl_8b_heb_v17_step800 | 87 | 0.005 | 0.021 | 0.057 | 0.229 | 0.002 | 0.000 | 0.211 |
| qwen3_vl_8b_heb_v16_step1100 | 18 | 0.001 | 0.008 | 0.027 | 0.145 | 0.000 | 0.000 | 0.195 |
| qwen3_vl_8b | 2 | 0.000 | 0.006 | 0.030 | 0.044 | 0.000 | 0.000 | 0.194 |
| vision_ocr_seg | 38 | 0.002 | 0.012 | 0.053 | 0.220 | 0.003 | 0.000 | 0.203 |

Largest absolute change in wrong-reference p90 from clipping: 0.010.

## 5. Floor vs signal — own-reference distribution (n=5, unclipped)

| system | n subst. | own med | own p10 | own min | own max | wrong p90 | own med / wrong p90 |
|---|---|---|---|---|---|---|---|
| kraken_seg | 119 | 0.477 | 0.232 | 0.126 | 0.791 | 0.026 | 18.555 |
| kraken_raw | 127 | 0.477 | 0.233 | 0.126 | 0.790 | 0.025 | 19.018 |
| gemini_pro | 40 | 0.294 | 0.118 | 0.106 | 0.868 | 0.034 | 8.561 |
| gemini_flash | 52 | 0.265 | 0.118 | 0.105 | 0.770 | 0.033 | 8.131 |
| claude_opus_4_8 | 53 | 0.203 | 0.119 | 0.105 | 0.760 | 0.011 | 18.389 |
| claude_sonnet_5 | 29 | 0.190 | 0.125 | 0.100 | 0.596 | 0.010 | 19.920 |
| gpt_5_6_sol | 36 | 0.295 | 0.146 | 0.100 | 0.808 | 0.043 | 6.793 |
| qwen3_vl_8b_heb_v17_step800 | 87 | 0.246 | 0.121 | 0.101 | 0.566 | 0.031 | 7.855 |
| qwen3_vl_8b_heb_v16_step1100 | 18 | 0.198 | 0.110 | 0.103 | 0.464 | 0.009 | 22.272 |
| qwen3_vl_8b | 2 | 0.197 | 0.181 | 0.177 | 0.216 | 0.008 | 26.006 |
| vision_ocr_seg | 38 | 0.204 | 0.115 | 0.101 | 0.709 | 0.013 | 16.150 |

## 6. Where the floor bites

Top 15 wrong-reference pairs (n=5, unclipped).  `own n-gram P` is the score the paper actually reports for that output; `CER` and `aligned F1` are the order-aware metrics on its own reference.

| source doc | system | wrong reference | wrong P | hyp letters | ref gt letters | own n-gram P | own CER | own aligned F1 |
|---|---|---|---|---|---|---|---|---|
| Cambridge_CUL_T_S_10J2_3 | gpt_5_6_sol | Manchester_JRL_A_960 | 0.643 | 267 | 676 | 0.589 | 0.644 | 0.517 |
| Cambridge_CUL_T_S_20_21 | gemini_flash | Cambridge_CUL_T_S_13J8_6 | 0.596 | 51 | 1949 | 0.553 | 0.969 | 0.019 |
| Cambridge_CUL_T_S_16_117 | gemini_flash | Cambridge_CUL_T_S_13J8_6 | 0.522 | 27 | 3267 | 0.261 | 0.992 | 0.014 |
| Cambridge_CUL_T_S_16_117 | gemini_flash | Cambridge_CUL_T_S_13J5_3 | 0.522 | 27 | 3267 | 0.261 | 0.992 | 0.014 |
| Cambridge_CUL_T_S_16_117 | gemini_flash | Cambridge_CUL_T_S_16_170 | 0.522 | 27 | 3267 | 0.261 | 0.992 | 0.014 |
| Cambridge_CUL_T_S_16_117 | gemini_flash | Cambridge_CUL_T_S_13J2_9 | 0.522 | 27 | 3267 | 0.261 | 0.992 | 0.014 |
| Cambridge_CUL_T_S_16_117 | gemini_flash | Cambridge_CUL_T_S_10J4_10 | 0.522 | 27 | 3267 | 0.261 | 0.992 | 0.014 |
| Cambridge_CUL_T_S_16_117 | gemini_flash | Cambridge_CUL_T_S_20_21 | 0.522 | 27 | 3267 | 0.261 | 0.992 | 0.014 |
| Cambridge_CUL_T_S_16_117 | gemini_flash | Cambridge_CUL_T_S_20_98 | 0.522 | 27 | 3267 | 0.261 | 0.992 | 0.014 |
| Cambridge_CUL_T_S_10J5_19 | gpt_5_6_sol | Cambridge_CUL_T_S_10J4_10 | 0.522 | 73 | 903 | 0.304 | 0.922 | 0.118 |
| Cambridge_CUL_T_S_10J5_19 | gpt_5_6_sol | Cambridge_CUL_T_S_13J5_3 | 0.507 | 73 | 903 | 0.304 | 0.922 | 0.118 |
| Cambridge_CUL_T_S_8_137 | gpt_5_6_sol | Cambridge_CUL_T_S_16_124 | 0.485 | 105 | 565 | 0.228 | 0.849 | 0.195 |
| Cambridge_CUL_T_S_10J5_19 | gpt_5_6_sol | Cambridge_CUL_T_S_16_170 | 0.478 | 73 | 903 | 0.304 | 0.922 | 0.118 |
| Cambridge_CUL_T_S_10J5_19 | gpt_5_6_sol | Cambridge_CUL_T_S_13J2_9 | 0.464 | 73 | 903 | 0.304 | 0.922 | 0.118 |
| Cambridge_CUL_T_S_10J5_19 | gpt_5_6_sol | Cambridge_CUL_T_S_13J8_6 | 0.449 | 73 | 903 | 0.304 | 0.922 | 0.118 |

Short hypotheses carry most of the floor risk:

| hypothesis length | n outputs | median max-wrong | share max-wrong >= 0.10 | share max-wrong >= 0.25 |
|---|---|---|---|---|
| < 300 Hebrew letters | 61 | 0.096 | 0.492 | 0.180 |
| >= 300 Hebrew letters | 540 | 0.049 | 0.169 | 0.037 |
