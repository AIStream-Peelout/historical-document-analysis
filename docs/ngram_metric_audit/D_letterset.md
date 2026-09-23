# Analysis D — effect of the letter set (Hebrew block vs Hebrew+Arabic union)

Paper metric: 5-gram precision, unclipped, letters reduced to the Hebrew block `U+0590-U+05FF`. Union ('semitic') set adds `U+0600-U+06FF`. All numbers 3 dp.

## Headline

- On the verified 131 the union letter set changes the behaviour class of **7 / 1352** outputs (0.52%), all of them `substantive -> hallucinated`, and never the reverse.
- Tier A is unchanged: **55** under the paper letter set, **55** under the union set (0 fragments change tier).
- **60 / 1352** outputs are >= 20% Arabic-block by letter count and are therefore graded on a Hebrew fragment of themselves under the paper metric; `vision_ocr_seg` writes Arabic-block characters in 58/123 of its outputs.
- Only **3 / 131** references in the verified 131 contain any Arabic-block letter (max 17 letters), so the union set can only ever cost a system precision here, never earn it.
- Re-running the audit's 0.12 best-evidence-overlap test under the union set readmits **4 / 19** excluded fragments: `Cambridge_CUL_T_S_Ar_38_2` (0.000 -> 0.249), `Cambridge_CUL_T_S_Ar_4_10` (0.000 -> 0.379), `Cambridge_CUL_T_S_NS_320_42` (0.026 -> 0.162), `Manchester_JRL_Genizah_Ar_806` (0.034 -> 0.631).

## Part 1 — verified 131, 11 paper systems

Outputs scored: **1352** saved outputs over 131 fragments and 11 systems (not every system has a file for every fragment).

### Class changes per system (Hebrew set -> union set)

| system | n outputs | class changes | from->to |
|---|---|---|---|
| kraken_seg | 122 | 0 | - |
| kraken_raw | 131 | 0 | - |
| gemini_pro | 68 | 0 | - |
| gemini_flash | 122 | 0 | - |
| claude_opus_4_8 | 131 | 3 | substantive->hallucinated x3 |
| claude_sonnet_5 | 131 | 0 | - |
| gpt_5_6_sol | 131 | 0 | - |
| qwen3_vl_8b_heb_v17_step800 | 131 | 0 | - |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0 | - |
| qwen3_vl_8b | 131 | 0 | - |
| vision_ocr_seg | 123 | 4 | substantive->hallucinated x4 |

Total class changes: **7 / 1352** (0.518%).

### Every output whose class changes

| doc_id | model | bucket | P(heb) | P(sem) | class heb | class sem | hyp heb | hyp sem | hyp ar-only | gt heb | gt sem | gt ar-only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cambridge_CUL_T_S_10J7_3 | claude_opus_4_8 | judaeo_arabic | 0.170 | 0.045 | substantive | hallucinated | 198 | 563 | 365 | 788 | 788 | 0 |
| Cambridge_CUL_T_S_13J17_9 | claude_opus_4_8 | hebrew | 0.239 | 0.057 | substantive | hallucinated | 502 | 1000 | 498 | 736 | 736 | 0 |
| Cambridge_CUL_T_S_20_110 | claude_opus_4_8 | judaeo_arabic | 0.121 | 0.058 | substantive | hallucinated | 251 | 398 | 147 | 1140 | 1140 | 0 |
| Cambridge_CUL_T_S_10J16_19 | vision_ocr_seg | judaeo_arabic | 0.167 | 0.008 | substantive | hallucinated | 10 | 123 | 113 | 1149 | 1149 | 0 |
| Cambridge_CUL_T_S_16_110 | vision_ocr_seg | untagged | 0.176 | 0.017 | substantive | hallucinated | 38 | 365 | 327 | 1449 | 1449 | 0 |
| Cambridge_CUL_T_S_16_117 | vision_ocr_seg | judaeo_arabic | 0.115 | 0.040 | substantive | hallucinated | 108 | 301 | 193 | 3267 | 3267 | 0 |
| Cambridge_CUL_T_S_16_138 | vision_ocr_seg | judaeo_arabic | 0.158 | 0.009 | substantive | hallucinated | 23 | 354 | 331 | 1644 | 1644 | 0 |

### Precision movement

- Outputs with any change in 5-gram precision: **40** (2.959%).

- Mean delta over those: -0.0297; min -0.1817, max +0.0001.

| doc_id | model | bucket | P(heb) | P(sem) | delta | hyp ar-only | gt ar-only |
|---|---|---|---|---|---|---|---|
| Cambridge_CUL_T_S_13J17_9 | claude_opus_4_8 | hebrew | 0.239 | 0.057 | -0.182 | 498 | 0 |
| Cambridge_CUL_T_S_16_110 | vision_ocr_seg | untagged | 0.176 | 0.017 | -0.160 | 327 | 0 |
| Cambridge_CUL_T_S_10J16_19 | vision_ocr_seg | judaeo_arabic | 0.167 | 0.008 | -0.158 | 113 | 0 |
| Cambridge_CUL_T_S_16_138 | vision_ocr_seg | judaeo_arabic | 0.158 | 0.009 | -0.149 | 331 | 0 |
| Cambridge_CUL_T_S_10J7_3 | claude_opus_4_8 | judaeo_arabic | 0.170 | 0.045 | -0.125 | 365 | 0 |
| Cambridge_CUL_T_S_16_117 | vision_ocr_seg | judaeo_arabic | 0.115 | 0.040 | -0.075 | 193 | 0 |
| Cambridge_CUL_T_S_20_110 | claude_opus_4_8 | judaeo_arabic | 0.121 | 0.058 | -0.063 | 147 | 0 |
| Cambridge_CUL_T_S_13J26_15 | claude_opus_4_8 | hebrew | 0.149 | 0.103 | -0.046 | 183 | 0 |
| Oxford_Bodleian_MS_heb_d_66_32 | claude_opus_4_8 | untagged | 0.057 | 0.017 | -0.040 | 360 | 0 |
| Cambridge_CUL_T_S_13J4_10 | vision_ocr_seg | judaeo_arabic | 0.042 | 0.006 | -0.036 | 139 | 0 |
| Manchester_JRL_A_960 | gemini_flash | aramaic | 0.770 | 0.744 | -0.026 | 12 | 0 |
| Cambridge_CUL_T_S_13J6_23 | claude_opus_4_8 | judaeo_arabic | 0.049 | 0.032 | -0.017 | 85 | 0 |
| Cambridge_CUL_T_S_20_63 | kraken_seg | judaeo_arabic | 0.469 | 0.453 | -0.016 | 0 | 10 |
| Cambridge_CUL_T_S_16_332 | vision_ocr_seg | untagged | 0.014 | 0.001 | -0.013 | 865 | 0 |
| Cambridge_CUL_T_S_8J8_22 | claude_opus_4_8 | judaeo_arabic | 0.017 | 0.007 | -0.010 | 288 | 0 |
| Oxford_Bodleian_MS_heb_f_108_63 | claude_opus_4_8 | judaeo_arabic | 0.009 | 0.000 | -0.009 | 167 | 0 |
| Cambridge_CUL_T_S_18J1_27 | vision_ocr_seg | untagged | 0.029 | 0.020 | -0.008 | 29 | 0 |
| Cambridge_CUL_T_S_20_63 | kraken_raw | judaeo_arabic | 0.552 | 0.544 | -0.008 | 0 | 10 |
| Cambridge_CUL_T_S_8_143 | vision_ocr_seg | judaeo_arabic | 0.140 | 0.133 | -0.007 | 12 | 0 |
| Cambridge_CUL_T_S_8J24_6 | vision_ocr_seg | judaeo_arabic | 0.059 | 0.054 | -0.006 | 23 | 0 |
| Cambridge_CUL_T_S_10J21_1 | vision_ocr_seg | judaeo_arabic | 0.092 | 0.086 | -0.006 | 12 | 0 |
| Cambridge_CUL_T_S_20_28 | vision_ocr_seg | judaeo_arabic | 0.086 | 0.081 | -0.005 | 39 | 0 |
| New_York_JTS_ENA_4007_6 | vision_ocr_seg | hebrew | 0.273 | 0.269 | -0.004 | 4 | 0 |
| New_York_JTS_ENA_4046_1 | kraken_seg | judaeo_arabic | 0.405 | 0.402 | -0.003 | 0 | 17 |
| Cambridge_CUL_T_S_16_237 | claude_opus_4_8 | judaeo_arabic | 0.007 | 0.005 | -0.003 | 82 | 0 |
| Cambridge_CUL_T_S_10J4_12 | vision_ocr_seg | untagged | 0.112 | 0.109 | -0.002 | 5 | 0 |
| Cambridge_CUL_T_S_13J2_6 | claude_opus_4_8 | judaeo_arabic | 0.066 | 0.064 | -0.002 | 27 | 0 |
| Cambridge_CUL_Or_1080_J262 | gemini_flash | judaeo_arabic | 0.382 | 0.380 | -0.002 | 3 | 0 |
| Cambridge_CUL_T_S_10J14_13 | gpt_5_6_sol | judaeo_arabic | 0.089 | 0.087 | -0.001 | 2 | 0 |
| Cambridge_CUL_T_S_16_237 | gemini_flash | judaeo_arabic | 0.036 | 0.035 | -0.001 | 29 | 0 |
| Cambridge_CUL_T_S_16_232 | vision_ocr_seg | judaeo_arabic | 0.028 | 0.027 | -0.001 | 7 | 0 |
| Cambridge_CUL_T_S_16_138 | claude_opus_4_8 | judaeo_arabic | 0.214 | 0.213 | -0.001 | 3 | 0 |
| Cambridge_CUL_T_S_20_63 | claude_opus_4_8 | judaeo_arabic | 0.195 | 0.194 | -0.001 | 1 | 10 |
| Cambridge_CUL_T_S_16_125 | vision_ocr_seg | untagged | 0.021 | 0.020 | -0.001 | 12 | 0 |
| Cambridge_CUL_Or_1080_J262 | vision_ocr_seg | judaeo_arabic | 0.099 | 0.098 | -0.001 | 2 | 0 |
| Cambridge_CUL_T_S_10J27_10 | kraken_raw | untagged | 0.233 | 0.233 | -0.000 | 1 | 0 |
| Cambridge_CUL_T_S_10J9_8 | gpt_5_6_sol | judaeo_arabic | 0.023 | 0.023 | -0.000 | 2 | 0 |
| Cambridge_CUL_T_S_8J22_30 | claude_opus_4_8 | judaeo_arabic | 0.067 | 0.067 | -0.000 | 3 | 0 |
| Cambridge_CUL_T_S_16_256 | claude_sonnet_5 | judaeo_arabic | 0.058 | 0.058 | -0.000 | 2 | 0 |
| New_York_JTS_ENA_4046_1 | qwen3_vl_8b_heb_v17_step800 | judaeo_arabic | 0.000 | 0.000 | +0.000 | 15299 | 17 |

### Tier A

- Tier A under PAPER_CONFIG (hebrew): **55**

- Tier A under letter_set='semitic': **55**

- Fragments changing tier: **0**

### Arabic-block characters present at all

| system | n outputs | hyps with >=1 Arabic-block letter | share | total Arabic-block letters | max in one output |
|---|---|---|---|---|---|
| kraken_seg | 122 | 0 | 0.000 | 0 | 0 |
| kraken_raw | 131 | 1 | 0.008 | 1 | 1 |
| gemini_pro | 68 | 0 | 0.000 | 0 | 0 |
| gemini_flash | 122 | 3 | 0.025 | 44 | 29 |
| claude_opus_4_8 | 131 | 13 | 0.099 | 2209 | 498 |
| claude_sonnet_5 | 131 | 1 | 0.008 | 2 | 2 |
| gpt_5_6_sol | 131 | 3 | 0.023 | 25 | 21 |
| qwen3_vl_8b_heb_v17_step800 | 131 | 3 | 0.023 | 42651 | 15299 |
| qwen3_vl_8b_heb_v16_step1100 | 131 | 0 | 0.000 | 0 | 0 |
| qwen3_vl_8b | 131 | 1 | 0.008 | 2 | 2 |
| vision_ocr_seg | 123 | 58 | 0.472 | 7242 | 865 |


References (GT) containing Arabic-block letters: **3 / 131**

| doc_id | bucket | gt hebrew letters | gt semitic letters | gt Arabic-only letters |
|---|---|---|---|---|
| New_York_JTS_ENA_4046_1 | judaeo_arabic | 558 | 575 | 17 |
| Cambridge_CUL_T_S_20_63 | judaeo_arabic | 1826 | 1836 | 10 |
| New_York_JTS_ENA_NS_45_31 | judaeo_arabic | 1047 | 1051 | 4 |

### Mixed-script outputs (>= 20% of hypothesis letters are Arabic-block)

Count: **60 / 1352**.

| doc_id | model | bucket | Arabic share of hyp | P(heb) | P(sem) | delta | class heb | class sem | hyp heb | hyp ar-only | gt ar-only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cambridge_CUL_T_S_10J9_8 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 124 | 0 |
| Cambridge_CUL_T_S_8J6_21 | vision_ocr_seg | aramaic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 96 | 0 |
| Oxford_Bodleian_MS_heb_b_11_7 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 221 | 0 |
| Cambridge_CUL_T_S_8J22_30 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | abstained | abstained | 0 | 12 | 0 |
| Cambridge_CUL_T_S_10J27_10 | vision_ocr_seg | untagged | 1.000 | 0.000 | 0.000 | +0.000 | abstained | abstained | 0 | 19 | 0 |
| Oxford_Bodleian_MS_heb_d_66_32 | vision_ocr_seg | untagged | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 222 | 0 |
| Cambridge_CUL_T_S_20_162 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | abstained | abstained | 0 | 8 | 0 |
| Cambridge_CUL_T_S_10J13_17 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 96 | 0 |
| New_York_JTS_ENA_NS_18_25 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 193 | 0 |
| Cambridge_CUL_T_S_Misc_8_103 | qwen3_vl_8b_heb_v17_step800 | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 12296 | 0 |
| Cambridge_CUL_T_S_Misc_8_103 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 652 | 0 |
| New_York_JTS_ENA_4046_1 | qwen3_vl_8b_heb_v17_step800 | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 15299 | 17 |
| Cambridge_CUL_T_S_24_51 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 30 | 0 |
| Cambridge_CUL_T_S_8J5_7 | vision_ocr_seg | untagged | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 124 | 0 |
| Cambridge_CUL_T_S_10J13_9 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 111 | 0 |
| Cambridge_CUL_T_S_13J21_33 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 100 | 0 |
| Cambridge_CUL_T_S_10J13_3 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 43 | 0 |
| New_York_JTS_ENA_NS_22_6 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 30 | 0 |
| Cambridge_CUL_T_S_AS_149_9 | vision_ocr_seg | untagged | 1.000 | 0.000 | 0.000 | +0.000 | abstained | abstained | 0 | 21 | 0 |
| New_York_JTS_ENA_NS_2_30 | vision_ocr_seg | untagged | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 191 | 0 |
| Cambridge_CUL_T_S_AS_149_4 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 234 | 0 |
| Cambridge_CUL_T_S_16_208 | vision_ocr_seg | untagged | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 220 | 0 |
| Cambridge_CUL_T_S_8J8_22 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | abstained | abstained | 0 | 16 | 0 |
| Oxford_Bodleian_MS_heb_d_74_40 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 34 | 0 |
| Cambridge_Mosseri_V_355 | qwen3_vl_8b_heb_v17_step800 | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 15056 | 0 |
| Cambridge_Mosseri_V_355 | vision_ocr_seg | judaeo_arabic | 1.000 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 0 | 108 | 0 |
| Cambridge_CUL_T_S_13J5_3 | vision_ocr_seg | judaeo_arabic | 0.993 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 1 | 145 | 0 |
| Cambridge_CUL_T_S_Ar_18_49 | vision_ocr_seg | untagged | 0.981 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 5 | 263 | 0 |
| New_York_JTS_ENA_4020_16 | vision_ocr_seg | judaeo_arabic | 0.975 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 6 | 236 | 0 |
| Cambridge_CUL_T_S_13J14_7 | vision_ocr_seg | judaeo_arabic | 0.967 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 4 | 116 | 0 |
| Cambridge_CUL_T_S_Ar_30_255 | vision_ocr_seg | judaeo_arabic | 0.947 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 8 | 143 | 0 |
| Cambridge_CUL_T_S_16_138 | vision_ocr_seg | judaeo_arabic | 0.935 | 0.158 | 0.009 | -0.149 | substantive | hallucinated | 23 | 331 | 0 |
| Cambridge_CUL_T_S_10J16_19 | vision_ocr_seg | judaeo_arabic | 0.919 | 0.167 | 0.008 | -0.158 | substantive | hallucinated | 10 | 113 | 0 |
| Cambridge_CUL_T_S_16_110 | vision_ocr_seg | untagged | 0.896 | 0.176 | 0.017 | -0.160 | substantive | hallucinated | 38 | 327 | 0 |
| Manchester_JRL_L_18 | vision_ocr_seg | hebrew | 0.884 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 106 | 811 | 0 |
| Cambridge_CUL_T_S_13J17_12 | vision_ocr_seg | judaeo_arabic | 0.859 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 11 | 67 | 0 |
| Cambridge_CUL_T_S_16_332 | vision_ocr_seg | untagged | 0.856 | 0.014 | 0.001 | -0.013 | hallucinated | hallucinated | 146 | 865 | 0 |
| New_York_JTS_ENA_4046_1 | vision_ocr_seg | judaeo_arabic | 0.854 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 7 | 41 | 17 |
| Cambridge_CUL_T_S_8_4 | vision_ocr_seg | judaeo_arabic | 0.846 | 0.000 | 0.000 | +0.000 | abstained | abstained | 2 | 11 | 0 |
| Cambridge_CUL_T_S_13J4_10 | vision_ocr_seg | judaeo_arabic | 0.832 | 0.042 | 0.006 | -0.036 | hallucinated | hallucinated | 28 | 139 | 0 |
| New_York_JTS_ENA_2557_1 | vision_ocr_seg | judaeo_arabic | 0.821 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 5 | 23 | 0 |
| Cambridge_CUL_T_S_13J23_12 | vision_ocr_seg | hebrew | 0.792 | 0.000 | 0.000 | +0.000 | abstained | abstained | 5 | 19 | 0 |
| Oxford_Bodleian_MS_heb_a_3_32 | vision_ocr_seg | untagged | 0.776 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 30 | 104 | 0 |
| Cambridge_CUL_T_S_16_235 | vision_ocr_seg | judaeo_arabic | 0.723 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 28 | 73 | 0 |
| Cambridge_CUL_T_S_6J3_2 | vision_ocr_seg | judaeo_arabic | 0.718 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 11 | 28 | 0 |
| Cambridge_CUL_T_S_16_237 | gpt_5_6_sol | judaeo_arabic | 0.700 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 9 | 21 | 0 |
| New_York_JTS_ENA_NS_39_9 | vision_ocr_seg | judaeo_arabic | 0.667 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 19 | 38 | 0 |
| Cambridge_CUL_T_S_10J7_3 | claude_opus_4_8 | judaeo_arabic | 0.648 | 0.170 | 0.045 | -0.125 | substantive | hallucinated | 198 | 365 | 0 |
| Cambridge_CUL_T_S_16_117 | vision_ocr_seg | judaeo_arabic | 0.641 | 0.115 | 0.040 | -0.075 | substantive | hallucinated | 108 | 193 | 0 |
| Oxford_Bodleian_MS_heb_d_66_32 | claude_opus_4_8 | untagged | 0.608 | 0.057 | 0.017 | -0.040 | hallucinated | hallucinated | 232 | 360 | 0 |
| Oxford_Bodleian_MS_heb_f_108_63 | claude_opus_4_8 | judaeo_arabic | 0.601 | 0.009 | 0.000 | -0.009 | loop_collapse | loop_collapse | 111 | 167 | 0 |
| Cambridge_CUL_T_S_16_220 | vision_ocr_seg | untagged | 0.553 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 21 | 26 | 0 |
| Cambridge_CUL_T_S_13J17_9 | claude_opus_4_8 | hebrew | 0.498 | 0.239 | 0.057 | -0.182 | substantive | hallucinated | 502 | 498 | 0 |
| Cambridge_CUL_T_S_18J1_32 | vision_ocr_seg | judaeo_arabic | 0.418 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 64 | 46 | 0 |
| Cambridge_CUL_T_S_8J8_22 | claude_opus_4_8 | judaeo_arabic | 0.408 | 0.017 | 0.007 | -0.010 | hallucinated | hallucinated | 418 | 288 | 0 |
| Cambridge_CUL_T_S_16_237 | claude_opus_4_8 | judaeo_arabic | 0.371 | 0.007 | 0.005 | -0.003 | hallucinated | hallucinated | 139 | 82 | 0 |
| Cambridge_CUL_T_S_20_110 | claude_opus_4_8 | judaeo_arabic | 0.369 | 0.121 | 0.058 | -0.063 | substantive | hallucinated | 251 | 147 | 0 |
| Cambridge_CUL_T_S_13J6_23 | claude_opus_4_8 | judaeo_arabic | 0.336 | 0.049 | 0.032 | -0.017 | hallucinated | hallucinated | 168 | 85 | 0 |
| Cambridge_CUL_T_S_NS_184_57 | vision_ocr_seg | judaeo_arabic | 0.327 | 0.000 | 0.000 | +0.000 | hallucinated | hallucinated | 35 | 17 | 0 |
| Cambridge_CUL_T_S_18J1_27 | vision_ocr_seg | untagged | 0.282 | 0.029 | 0.020 | -0.008 | hallucinated | hallucinated | 74 | 29 | 0 |


## Part 2 — the Arabic-script fragments the audit excluded

### GT letter counts

| doc_id | bucket | gt hebrew | gt semitic | gt Arabic-only | GT repaired? |
|---|---|---|---|---|---|
| New_York_JTS_ENA_NS_2_29 | arabic_script | 0 | 1104 | 1104 | False |
| Cambridge_CUL_T_S_Ar_38_2 | arabic_script | 0 | 594 | 594 | False |
| Cambridge_CUL_T_S_Ar_4_10 | judaeo_arabic | 0 | 677 | 677 | False |
| Cambridge_CUL_T_S_Ar_19_23 | arabic_script | 0 | 1312 | 1312 | False |


#### New_York_JTS_ENA_NS_2_29 (bucket `arabic_script`, GT 0 heb / 1104 sem / 1104 Arabic-only letters)

| system | P5(heb) | P5(sem) | delta | aligned F1 | CER | hyp heb | hyp sem | hyp ar-only |
|---|---|---|---|---|---|---|---|---|
| gemini_pro | 0.000 | 0.069 | +0.069 | 0.210 | 0.682 | 0 | 917 | 917 |
| gemini_flash | 0.000 | 0.029 | +0.029 | 0.160 | 0.657 | 0 | 889 | 889 |
| vision_ocr_seg | 0.000 | 0.027 | +0.027 | 0.217 | 0.701 | 0 | 558 | 558 |
| qwen3_vl_8b_heb_v17_step800 | 0.000 | 0.001 | +0.001 | 0.019 | 13.708 | 0 | 15704 | 15704 |
| claude_opus_4_8 | 0.000 | 0.000 | +0.000 | 0.040 | 0.921 | 0 | 0 | 0 |
| claude_sonnet_5 | 0.000 | 0.000 | +0.000 | 0.054 | 0.904 | 0 | 0 | 0 |
| gpt_5_6_sol | 0.000 | 0.000 | +0.000 | 0.004 | 0.998 | 19 | 19 | 0 |
| kraken_raw | 0.000 | 0.000 | +0.000 | 0.060 | 0.962 | 76 | 91 | 15 |
| kraken_seg | 0.000 | 0.000 | +0.000 | 0.047 | 0.974 | 60 | 75 | 15 |
| qwen3_vl_8b | 0.000 | 0.000 | +0.000 | 0.062 | 4.870 | 6035 | 6035 | 0 |
| qwen3_vl_8b_heb_v16_step1100 | 0.000 | 0.000 | +0.000 | 0.029 | 11.599 | 13035 | 13035 | 0 |


#### Cambridge_CUL_T_S_Ar_38_2 (bucket `arabic_script`, GT 0 heb / 594 sem / 594 Arabic-only letters)

| system | P5(heb) | P5(sem) | delta | aligned F1 | CER | hyp heb | hyp sem | hyp ar-only |
|---|---|---|---|---|---|---|---|---|
| gemini_pro | 0.000 | 0.249 | +0.249 | 0.426 | 0.591 | 0 | 361 | 361 |
| gpt_5_6_sol | 0.000 | 0.231 | +0.231 | 0.372 | 0.664 | 0 | 299 | 299 |
| gemini_flash | 0.000 | 0.168 | +0.168 | 0.386 | 0.532 | 0 | 474 | 474 |
| claude_sonnet_5 | 0.000 | 0.090 | +0.090 | 0.300 | 0.548 | 0 | 425 | 425 |
| vision_ocr_seg | 0.000 | 0.077 | +0.077 | 0.347 | 0.677 | 0 | 288 | 288 |
| qwen3_vl_8b_heb_v17_step800 | 0.000 | 0.001 | +0.001 | 0.010 | 20.871 | 0 | 13449 | 13449 |
| claude_opus_4_8 | 0.000 | 0.000 | +0.000 | 0.170 | 1.197 | 0 | 0 | 0 |
| kraken_raw | 0.000 | 0.000 | +0.000 | 0.044 | 0.976 | 52 | 52 | 0 |
| kraken_seg | 0.000 | 0.000 | +0.000 | 0.044 | 0.976 | 52 | 52 | 0 |
| qwen3_vl_8b | 0.000 | 0.000 | +0.000 | 0.016 | 12.003 | 0 | 8205 | 8205 |
| qwen3_vl_8b_heb_v16_step1100 | 0.000 | 0.000 | +0.000 | 0.125 | 0.907 | 294 | 294 | 0 |


#### Cambridge_CUL_T_S_Ar_4_10 (bucket `judaeo_arabic`, GT 0 heb / 677 sem / 677 Arabic-only letters)

| system | P5(heb) | P5(sem) | delta | aligned F1 | CER | hyp heb | hyp sem | hyp ar-only |
|---|---|---|---|---|---|---|---|---|
| gemini_pro | 0.000 | 0.379 | +0.379 | 0.651 | 0.423 | 0 | 576 | 576 |
| gemini_flash | 0.000 | 0.297 | +0.297 | 0.467 | 0.465 | 0 | 519 | 519 |
| claude_sonnet_5 | 0.000 | 0.108 | +0.108 | 0.488 | 0.575 | 0 | 428 | 428 |
| qwen3_vl_8b_heb_v17_step800 | 0.000 | 0.098 | +0.098 | 0.227 | 0.644 | 0 | 482 | 482 |
| gpt_5_6_sol | 0.000 | 0.096 | +0.096 | 0.254 | 0.751 | 0 | 244 | 244 |
| vision_ocr_seg | 0.000 | 0.092 | +0.092 | 0.413 | 0.598 | 0 | 417 | 417 |
| claude_opus_4_8 | 0.000 | 0.000 | +0.000 | 0.059 | 0.934 | 0 | 0 | 0 |
| kraken_raw | 0.000 | 0.000 | +0.000 | 0.061 | 0.949 | 114 | 127 | 13 |
| kraken_seg | 0.000 | 0.000 | +0.000 | 0.044 | 0.951 | 114 | 127 | 13 |
| qwen3_vl_8b | 0.000 | 0.000 | +0.000 | 0.004 | 10.711 | 0 | 8938 | 8938 |
| qwen3_vl_8b_heb_v16_step1100 | 0.000 | 0.000 | +0.000 | 0.035 | 9.437 | 4681 | 4681 | 0 |


#### Cambridge_CUL_T_S_Ar_19_23 (bucket `arabic_script`, GT 0 heb / 1312 sem / 1312 Arabic-only letters)

| system | P5(heb) | P5(sem) | delta | aligned F1 | CER | hyp heb | hyp sem | hyp ar-only |
|---|---|---|---|---|---|---|---|---|
| gpt_5_6_sol | 0.000 | 0.533 | +0.533 | 0.027 | 0.986 | 0 | 19 | 19 |
| qwen3_vl_8b_heb_v17_step800 | 0.000 | 0.022 | +0.022 | 0.009 | 13.025 | 0 | 17507 | 17507 |
| claude_opus_4_8 | 0.000 | 0.000 | +0.000 | 0.109 | 0.914 | 0 | 0 | 0 |
| claude_sonnet_5 | 0.000 | 0.000 | +0.000 | 0.001 | 0.999 | 0 | 0 | 0 |
| kraken_raw | 0.000 | 0.000 | +0.000 | 0.095 | 0.943 | 233 | 240 | 7 |
| qwen3_vl_8b | 0.000 | 0.000 | +0.000 | 0.043 | 7.440 | 10236 | 10236 | 0 |
| qwen3_vl_8b_heb_v16_step1100 | 0.000 | 0.000 | +0.000 | 0.125 | 0.881 | 801 | 801 | 0 |


### Audit re-check on the four (evidence systems only, threshold 0.12)

| doc_id | evidence systems present / 8 | best overlap (heb) | best model (heb) | passes heb? | best overlap (sem) | best model (sem) | passes sem? |
|---|---|---|---|---|---|---|---|
| New_York_JTS_ENA_NS_2_29 | 8 | 0.000 | - | no | 0.069 | gemini_pro | no |
| Cambridge_CUL_T_S_Ar_38_2 | 8 | 0.000 | - | no | 0.249 | gemini_pro | yes |
| Cambridge_CUL_T_S_Ar_4_10 | 8 | 0.000 | - | no | 0.379 | gemini_pro | yes |
| Cambridge_CUL_T_S_Ar_19_23 | 4 | 0.000 | - | no | 0.022 | qwen3_vl_8b_heb_v17_step800 | no |


`Cambridge_CUL_T_S_Ar_19_23` has only 4/8 evidence systems on disk (missing: kraken_seg, gemini_pro, gemini_flash, vision_ocr_seg).


### All 19 excluded fragments under the union set

| doc_id | bucket | evidence systems / 8 | gt heb | gt ar-only | best overlap (heb) | best model (heb) | best overlap (sem) | best model (sem) | readmitted at 0.12? |
|---|---|---|---|---|---|---|---|---|---|
| Manchester_JRL_Genizah_Ar_806 | judaeo_arabic | 8 | 341 | 261 | 0.034 | kraken_seg | 0.631 | claude_opus_4_8 | yes |
| Cambridge_CUL_T_S_Ar_4_10 | judaeo_arabic | 8 | 0 | 677 | 0.000 | - | 0.379 | gemini_pro | yes |
| Cambridge_CUL_T_S_Ar_38_2 | arabic_script | 8 | 0 | 594 | 0.000 | - | 0.249 | gemini_pro | yes |
| Cambridge_CUL_T_S_NS_320_42 | arabic_script | 8 | 119 | 903 | 0.026 | kraken_seg | 0.162 | gemini_pro | yes |
| Cambridge_CUL_T_S_20_99 | untagged | 4 | 1557 | 0 | 0.111 | qwen3_vl_8b_heb_v17_step800 | 0.111 | qwen3_vl_8b_heb_v17_step800 | no |
| New_York_JTS_ENA_NS_I_38 | judaeo_arabic | 8 | 483 | 0 | 0.110 | qwen3_vl_8b_heb_v17_step800 | 0.110 | qwen3_vl_8b_heb_v17_step800 | no |
| Cambridge_CUL_T_S_12_708 | judaeo_arabic | 8 | 533 | 0 | 0.106 | gemini_flash | 0.106 | gemini_flash | no |
| New_York_JTS_ENA_2738_34 | judaeo_arabic | 8 | 1049 | 0 | 0.085 | gemini_pro | 0.085 | gemini_pro | no |
| Cambridge_CUL_T_S_10J30_7 | judaeo_arabic | 4 | 490 | 0 | 0.081 | qwen3_vl_8b_heb_v17_step800 | 0.081 | qwen3_vl_8b_heb_v17_step800 | no |
| New_York_JTS_ENA_NS_2_29 | arabic_script | 8 | 0 | 1104 | 0.000 | - | 0.069 | gemini_pro | no |
| Cambridge_Mosseri_V_336_1 | judaeo_arabic | 8 | 525 | 0 | 0.029 | qwen3_vl_8b_heb_v17_step800 | 0.029 | qwen3_vl_8b_heb_v17_step800 | no |
| Cambridge_CUL_T_S_28_24 | untagged | 4 | 3800 | 0 | 0.022 | qwen3_vl_8b_heb_v17_step800 | 0.022 | qwen3_vl_8b_heb_v17_step800 | no |
| Cambridge_CUL_T_S_Ar_19_23 | arabic_script | 4 | 0 | 1312 | 0.000 | - | 0.022 | qwen3_vl_8b_heb_v17_step800 | no |
| Oxford_Bodleian_MS_heb_f_39_30 | hebrew | 8 | 415 | 0 | 0.022 | vision_ocr_seg | 0.020 | vision_ocr_seg | no |
| Oxford_Bodleian_MS_heb_d_66_68 | judaeo_arabic | 6 | 505 | 0 | 0.009 | qwen3_vl_8b_heb_v17_step800 | 0.009 | qwen3_vl_8b_heb_v17_step800 | no |
| Cambridge_CUL_T_S_8J17_17 | judaeo_arabic | 4 | 504 | 0 | 0.007 | claude_sonnet_5 | 0.007 | claude_sonnet_5 | no |
| Cambridge_CUL_T_S_20_117 | judaeo_arabic | 4 | 1047 | 0 | 0.005 | qwen3_vl_8b_heb_v17_step800 | 0.005 | qwen3_vl_8b_heb_v17_step800 | no |
| Cambridge_CUL_T_S_13J8_16 | arabic_script | 8 | 664 | 0 | 0.000 | - | 0.000 | - | no |
| New_York_JTS_ENA_NS_I_62 | judaeo_arabic | 8 | 368 | 0 | 0.000 | - | 0.000 | - | no |


Readmitted under the union set at 0.12: **4 / 19** — Cambridge_CUL_T_S_Ar_38_2, Cambridge_CUL_T_S_Ar_4_10, Cambridge_CUL_T_S_NS_320_42, Manchester_JRL_Genizah_Ar_806


## Note — the `arabic_script` script_tags bucket

| bucket | fragments | share of 150 |
|---|---|---|
| judaeo_arabic | 93 | 0.620 |
| untagged | 24 | 0.160 |
| hebrew | 21 | 0.140 |
| aramaic | 6 | 0.040 |
| arabic_script | 6 | 0.040 |

| doc_id | in verified subset? | in the scored 131? | in excluded_misaligned? |
|---|---|---|---|
| Cambridge_CUL_T_S_13J8_16 | no | no | yes |
| Cambridge_CUL_T_S_18J2_16 | yes | yes | no |
| Cambridge_CUL_T_S_Ar_19_23 | no | no | yes |
| Cambridge_CUL_T_S_Ar_38_2 | no | no | yes |
| Cambridge_CUL_T_S_NS_320_42 | no | no | yes |
| New_York_JTS_ENA_NS_2_29 | no | no | yes |


`arabic_script` = **6/150 = 4.000%** of the tagged benchmark; 5 of the 6 are in `audit.excluded_misaligned`, 1 survive into the scored 131.
