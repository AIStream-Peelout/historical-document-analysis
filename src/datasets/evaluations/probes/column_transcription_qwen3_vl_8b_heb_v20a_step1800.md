# Column-transcription probe — qwen3-vl-8b-heb-v20a-step1800 on the 33 two-column religious-140 pages

| condition | n | median CER | mean CER | under-read (<0.8) | over-gen (>1.2) | median len_ratio |
|---|---|---|---|---|---|---|
| A_page | 33 | 0.243 | 0.322 | 12/33 | 1/33 | 0.97 |
| B_prompt | 33 | 0.256 | 0.334 | 1/33 | 8/33 | 1.00 |
| C_crop | 30 | 0.151 | 0.282 | 4/30 | 1/30 | 0.98 |

B_prompt vs A_page: paired median ΔCER +0.001, better on 9, worse on 8 of 33

C_crop vs A_page: paired median ΔCER -0.005, better on 11, worse on 7 of 30
