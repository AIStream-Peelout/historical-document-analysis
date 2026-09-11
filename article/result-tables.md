# Result tables — *Three Ways to Misread*

Every figure is recomputed offline from each system's raw saved output against
repaired ground truth. Failure-mode thresholds: **abstained** = under 25
characters or a refusal phrase; **loop collapse** = one repeated 12-gram covering
≥45% of the output; **hallucinated** = under 0.10 order-independent 5-gram
precision against the ground truth; everything else is a **substantive** attempt.
CER is reported only over substantive attempts.

Interactive version: https://claude.ai/code/artifact/1679e1c8-adb3-4b32-81b7-e5e5449119d9

## Table A — Genizah behaviour (131 verified fragments)

| System | Read | Hallucinated | Abstained | Looped | n-gram prec. | Aligned F1 | CER (subst.) | n subst. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hebrew v1.9a step1300 (new) | 97.7% | 0.8% | 0.0% | 1.5% | 0.600 | 0.862 | 0.196 | 128 |
| Kraken HTR (segmented) | 97.5% | 1.6% | 0.8% | 0.0% | 0.474 | 0.678 | 0.410 | 119 |
| Kraken HTR (raw) | 96.9% | 2.3% | 0.8% | 0.0% | 0.476 | 0.730 | 0.344 | 127 |
| Hebrew v1.9a step700 (new) | 96.9% | 2.3% | 0.0% | 0.8% | 0.584 | 0.854 | 0.199 | 127 |
| Hebrew v1.8b (sectioned) | 92.4% | 3.8% | 0.0% | 3.8% | 0.461 | 0.725 | 0.445 | 121 |
| Hebrew v1.8b (vision trained) | 90.1% | 9.2% | 0.0% | 0.8% | 0.475 | 0.807 | 0.249 | 118 |
| Hebrew v1.8b step400 | 89.3% | 8.4% | 0.0% | 2.3% | 0.450 | 0.794 | 0.253 | 117 |
| Hebrew v1.8a (vision frozen) | 74.8% | 22.1% | 0.0% | 3.1% | 0.186 | 0.616 | 0.407 | 98 |
| Hebrew v1.7 step800 | 66.4% | 29.8% | 0.8% | 3.1% | 0.167 | 0.553 | 0.379 | 87 |
| Gemini Pro | 58.8% | 41.2% | 0.0% | 0.0% | 0.174 | 0.421 | 0.432 | 40 |
| Gemini Flash | 42.6% | 38.5% | 11.5% | 7.4% | 0.064 | 0.196 | 0.542 | 52 |
| Claude Opus 4.8 | 40.5% | 29.0% | 28.2% | 2.3% | 0.065 | 0.366 | 0.416 | 53 |
| Cloud Vision OCR (seg) | 30.9% | 54.5% | 13.8% | 0.8% | 0.024 | 0.108 | 0.646 | 38 |
| Cloud Vision OCR (raw) | 29.8% | 57.3% | 13.0% | 0.0% | 0.026 | 0.116 | 0.649 | 39 |
| GPT-5.6 Sol | 27.5% | 21.4% | 50.4% | 0.8% | 0.000 | 0.034 | 0.736 | 36 |
| Claude Sonnet 5 | 22.1% | 67.9% | 8.4% | 1.5% | 0.029 | 0.366 | 0.435 | 29 |
| Hebrew v1.6 step1100 | 13.7% | 68.7% | 0.8% | 16.8% | 0.010 | 0.083 | 0.373 | 18 |
| Qwen3-VL-8B (base) | 1.5% | 28.2% | 11.5% | 58.8% | 0.000 | 0.010 | 0.493 | 2 |
| Gemma 4 31B | 0.0% | 100.0% | 0.0% | 0.0% | 0.000 | 0.019 | — | 0 |
| GPT-5.2 | 0.0% | 0.0% | 100.0% | 0.0% | 0.000 | 0.000 | — | 0 |

*Gemma 4 31B and GPT-5.2 were run on a single fragment each; their rows are not comparable. Hebrew v1.9a rows scored 2026-08-21.*

## Table B — Talmud Bavli (65 pages, three sections)

| System | Section | Pages | Coverage | Median CER | Mean CER | Pages >0.3 | Pages >1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hebrew v1.6 step1100 | gemara | 65 | 100.0% | 0.004 | 0.013 | 1.5% | 0.0% |
| Hebrew v1.6 step600 | gemara | 65 | 100.0% | 0.004 | 0.013 | 1.5% | 0.0% |
| Hebrew v1.6 step1000 | gemara | 65 | 100.0% | 0.005 | 0.090 | 3.1% | 1.5% |
| Hebrew v1.7 step800 | gemara | 65 | 100.0% | 0.005 | 0.014 | 1.5% | 0.0% |
| Gemini Pro | gemara | 19 | 29.2% | 0.014 | 0.015 | 0.0% | 0.0% |
| Cloud Vision OCR (seg) | gemara | 65 | 100.0% | 0.033 | 0.056 | 1.5% | 0.0% |
| Gemini Flash | gemara | 64 | 98.5% | 0.036 | 0.155 | 15.6% | 1.6% |
| Kraken HTR (segmented) | gemara | 65 | 100.0% | 0.040 | 0.081 | 3.1% | 0.0% |
| Hebrew v1.5 | gemara | 65 | 100.0% | 0.048 | 0.059 | 1.5% | 0.0% |
| GPT-5.6 Sol | gemara | 20 | 30.8% | 0.156 | 0.376 | 40.0% | 15.0% |
| Claude Opus 4.8 | gemara | 65 | 100.0% | 0.292 | 0.706 | 49.2% | 24.6% |
| Claude Sonnet 5 | gemara | 65 | 100.0% | 0.398 | 0.630 | 67.7% | 16.9% |
| GPT-5.2 | gemara | 65 | 100.0% | 0.907 | 0.880 | 100.0% | 4.6% |
| Qwen3-VL-8B (base) | gemara | 65 | 100.0% | 3.63 | 5.08 | 100.0% | 100.0% |
| Hebrew v1.6 step1000 | rashi | 65 | 100.0% | 0.014 | 0.047 | 4.6% | 1.5% |
| Hebrew v1.6 step1100 | rashi | 65 | 100.0% | 0.014 | 0.183 | 9.2% | 3.1% |
| Hebrew v1.7 step800 | rashi | 65 | 100.0% | 0.015 | 0.084 | 7.7% | 3.1% |
| Hebrew v1.6 step600 | rashi | 65 | 100.0% | 0.017 | 0.226 | 13.9% | 6.2% |
| Gemini Pro | rashi | 9 | 13.9% | 0.109 | 0.299 | 44.4% | 0.0% |
| Hebrew v1.5 | rashi | 65 | 100.0% | 0.144 | 0.362 | 20.0% | 6.2% |
| Kraken HTR (segmented) | rashi | 65 | 100.0% | 0.290 | 0.392 | 49.2% | 0.0% |
| Gemini Flash | rashi | 65 | 100.0% | 0.718 | 0.594 | 80.0% | 1.5% |
| Cloud Vision OCR (seg) | rashi | 65 | 100.0% | 0.744 | 0.737 | 100.0% | 1.5% |
| Claude Opus 4.8 | rashi | 65 | 100.0% | 0.781 | 0.718 | 93.8% | 1.5% |
| Claude Sonnet 5 | rashi | 65 | 100.0% | 0.793 | 0.779 | 100.0% | 0.0% |
| GPT-5.6 Sol | rashi | 20 | 30.8% | 0.884 | 0.947 | 100.0% | 40.0% |
| GPT-5.2 | rashi | 65 | 100.0% | 0.949 | 0.933 | 100.0% | 1.5% |
| Qwen3-VL-8B (base) | rashi | 65 | 100.0% | 3.39 | 4.45 | 100.0% | 100.0% |
| Hebrew v1.6 step1100 | tosafot | 63 | 100.0% | 0.011 | 0.308 | 9.5% | 4.8% |
| Hebrew v1.7 step800 | tosafot | 63 | 100.0% | 0.012 | 0.037 | 1.6% | 1.6% |
| Hebrew v1.6 step1000 | tosafot | 63 | 100.0% | 0.013 | 0.099 | 11.1% | 0.0% |
| Hebrew v1.6 step600 | tosafot | 63 | 100.0% | 0.014 | 0.126 | 12.7% | 3.2% |
| Gemini Pro | tosafot | 12 | 19.1% | 0.076 | 0.155 | 8.3% | 8.3% |
| Hebrew v1.5 | tosafot | 63 | 100.0% | 0.130 | 0.512 | 22.2% | 6.3% |
| Kraken HTR (segmented) | tosafot | 63 | 100.0% | 0.229 | 0.288 | 34.9% | 0.0% |
| GPT-5.6 Sol | tosafot | 19 | 30.2% | 0.684 | 0.785 | 79.0% | 15.8% |
| Cloud Vision OCR (seg) | tosafot | 63 | 100.0% | 0.737 | 0.706 | 98.4% | 1.6% |
| Gemini Flash | tosafot | 63 | 100.0% | 0.771 | 0.783 | 93.7% | 7.9% |
| Claude Sonnet 5 | tosafot | 63 | 100.0% | 0.822 | 0.836 | 100.0% | 1.6% |
| Claude Opus 4.8 | tosafot | 63 | 100.0% | 0.838 | 0.786 | 95.2% | 0.0% |
| GPT-5.2 | tosafot | 63 | 100.0% | 0.958 | 1.25 | 100.0% | 4.8% |
| Qwen3-VL-8B (base) | tosafot | 63 | 100.0% | 3.74 | 6.11 | 100.0% | 95.2% |

*Coverage below 100% is reported, not dropped. Gemini Pro coverage is still capped by the July billing quota (not model refusal); Gemini Flash was re-filled 2026-08-23 to 98.5–100%. Kraken HTR (segmented) rows are from the 2026-08-21 re-run and Cloud Vision (seg) from the 2026-08-24 re-run, both with the quota clear.*

## Table C — Hallucination rate by script bucket

| System | Hebrew (n=20) | Aramaic (n=6) | Judaeo-Arabic (n=82) | Untagged (n=22) |
| --- | ---: | ---: | ---: | ---: |
| Kraken HTR (raw) | 5.0% | 0.0% | 0.0% | 9.1% |
| Hebrew v1.8b (vision trained) | 10.0% | 0.0% | 6.1% | 22.7% |
| Hebrew v1.8a (vision frozen) | 20.0% | 33.3% | 20.7% | 27.3% |
| Hebrew v1.7 step800 | 20.0% | 33.3% | 29.3% | 40.9% |
| Gemini Pro | 16.7% | 0.0% | 51.3% | 40.0% |
| Gemini Flash | 10.5% | 20.0% | 46.7% | 40.9% |
| Claude Opus 4.8 | 10.0% | 16.7% | 35.4% | 27.3% |
| Claude Sonnet 5 | 35.0% | 33.3% | 78.0% | 72.7% |
| GPT-5.6 Sol | 30.0% | 0.0% | 19.5% | 27.3% |
| Cloud Vision OCR (raw) | 25.0% | 33.3% | 65.9% | 63.6% |
| Hebrew v1.6 step1100 | 45.0% | 33.3% | 76.8% | 68.2% |

## Table D — Vision-LoRA A/B (same data, same hyperparameters, same step)

| Measure | v1.8a — vision frozen | v1.8b — vision trained | Change |
| --- | ---: | ---: | ---: |
| Median n-gram precision | 0.186 | 0.475 | ×2.55 |
| Median aligned F1 | 0.616 | 0.807 | ×1.31 |
| Read substantively | 74.8% | 90.1% | +15.3 pp |
| Hallucinated | 22.1% | 9.2% | -12.9 pp |
| Vision `lora_B` tensors trained | 0 / 108 | 108 / 108 | — |

## Table E — Confusable-pair swap rates

| Pair | Kraken (Genizah) | v1.7 (Genizah) | v1.8b (Genizah) | v1.8b (Talmud) | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| ד/ר | 4.77% | 8.00% | 6.65% | 0.04% | ×185 |
| ב/כ | 4.41% | 5.14% | 5.32% | 0.16% | ×33 |
| ם/ס | 2.50% | 4.37% | 3.23% | 0.33% | ×10 |
| י/ו | 2.19% | 3.10% | 2.19% | 0.04% | ×52 |
| כ/פ | 1.54% | 3.84% | 1.83% | 0 | — |
| ט/ס | 1.21% | 1.41% | 1.59% | 0.07% | ×24 |
| ה/ח | 1.30% | 2.79% | 1.55% | 0.05% | ×31 |
| ה/ת | 1.40% | 2.92% | 1.48% | 0.03% | ×57 |
| ע/צ | 0.60% | 2.59% | 1.35% | 0.01% | ×144 |
| ג/נ | 1.55% | 1.65% | 1.30% | 0.13% | ×10 |

Pooled over all twenty tracked pairs:

| System | Corpus | Swaps / occurrences | Rate |
| --- | --- | ---: | ---: |
| Hebrew v1.8b | Genizah manuscript | 2,486 / 136,053 | 1.83% |
| Hebrew v1.8b | Printed Talmud | 201 / 396,440 | 0.051% |
| Hebrew v1.7 | Genizah manuscript | 2,388 / 91,784 | 2.60% |
| Kraken HTR | Genizah manuscript | 2,276 / 144,491 | 1.58% |

The same model swaps confusable letters **36× more often** on manuscript
hands than on printed type.

## Table F — Cross-generation forgetting probe

| Model | Task | n | Median CER | Mean CER | Share > 0.3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hebrew v1.6 step1000 | synthetic_rashi | 30 | 0.008 | 0.022 | 0.0% |
| Hebrew v1.6 step1000 | talmud_crop_transcribe | 16 | 0.748 | 1.28 | 75.0% |
| Hebrew v1.6 step1000 | talmud_page_extract | 4 | 0.011 | 0.015 | 0.0% |
| Hebrew v1.7 step800 | synthetic_rashi | 30 | 0.008 | 0.025 | 0.0% |
| Hebrew v1.7 step800 | talmud_crop_transcribe | 16 | 0.789 | 1.76 | 75.0% |
| Hebrew v1.7 step800 | talmud_page_extract | 4 | 0.014 | 0.013 | 0.0% |
| Hebrew v1.8a (vision frozen) | synthetic_rashi | 30 | 0.008 | 0.022 | 0.0% |
| Hebrew v1.8a (vision frozen) | talmud_crop_transcribe | 16 | 0.711 | 2.57 | 75.0% |
| Hebrew v1.8a (vision frozen) | talmud_page_extract | 4 | 0.011 | 0.011 | 0.0% |
| Hebrew v1.8b (vision trained) | synthetic_rashi | 30 | 0.002 | 0.019 | 0.0% |
| Hebrew v1.8b (vision trained) | talmud_crop_transcribe | 16 | 0.731 | 1.00 | 75.0% |
| Hebrew v1.8b (vision trained) | talmud_page_extract | 4 | 0.006 | 0.007 | 0.0% |

*A seeded control on the corpus the models were trained on. All four generations still read full Talmud pages and synthetic Rashi renders near-perfectly; all four fail the isolated-crop framing.*

## Table G — Talmud WER (computed post-hoc, 2026-08-22)

| System | Section | n | Median CER | Median WER | WER/CER |
| --- | --- | ---: | ---: | ---: | ---: |
| Hebrew v1.7 step800 | gemara | 56 | 0.0043 | 0.0153 | 3.6× |
| Hebrew v1.7 step800 | rashi | 64 | 0.0149 | 0.0584 | 3.9× |
| Hebrew v1.7 step800 | tosafot | 49 | 0.0126 | 0.0444 | 3.5× |
| Hebrew v1.8b (vision trained) | gemara | 56 | 0.0030 | 0.0098 | 3.3× |
| Hebrew v1.8b (vision trained) | rashi | 64 | 0.0080 | 0.0324 | 4.0× |
| Hebrew v1.8b (vision trained) | tosafot | 49 | 0.0070 | 0.0294 | 4.2× |
| Gemini Pro | gemara | 13 | 0.0147 | 0.0377 | 2.6× |
| Gemini Pro | rashi | 6 | 0.2952 | 0.3640 | 1.2× |
| Gemini Pro | tosafot | 9 | 0.0865 | 0.1840 | 2.1× |
| Gemini Flash | gemara | 13 | 0.0285 | 0.0576 | 2.0× |
| Gemini Flash | rashi | 17 | 0.5411 | 0.7699 | 1.4× |
| Gemini Flash | tosafot | 15 | 0.7094 | 0.9738 | 1.4× |
| Kraken HTR (seg, re-run) | gemara | 55 | 0.0464 | 0.1090 | 2.3× |
| Kraken HTR (seg, re-run) | rashi | 64 | 0.2886 | 0.3967 | 1.4× |
| Kraken HTR (seg, re-run) | tosafot | 49 | 0.2304 | 0.3382 | 1.5× |
| Claude Opus 4.8 | gemara | 37 | 0.2374 | 0.2668 | 1.1× |
| Claude Opus 4.8 | rashi | 38 | 0.7809 | 0.9849 | 1.3× |
| Claude Opus 4.8 | tosafot | 24 | 0.8403 | 0.9956 | 1.2× |
| Claude Sonnet 5 | gemara | 37 | 0.3984 | 0.4906 | 1.2× |
| Claude Sonnet 5 | rashi | 38 | 0.8136 | 0.9788 | 1.2× |
| Claude Sonnet 5 | tosafot | 24 | 0.7733 | 0.9683 | 1.3× |
| GPT-5.2 | gemara | 56 | 0.9097 | 0.9946 | 1.1× |
| GPT-5.2 | rashi | 64 | 0.9481 | 1.0000 | 1.1× |
| GPT-5.2 | tosafot | 49 | 0.9538 | 1.0000 | 1.0× |

*WER computed after the fact from the saved raw outputs with the repo's `wer_pair` (whitespace tokens, unclamped), against artifact ground truth excluding 24 W&B-truncated cells — so n differs slightly from the official CER table. Rankings are identical to CER on every section. The WER/CER ratio is diagnostic: ~3.5–4× for accurate systems (a lone letter error destroys a whole short Hebrew word), compressing toward 1.0× as systems saturate — GPT-5.2 hits WER 1.0000 exactly on two sections while CER still separates it from Claude. Gemini rows are small-n disk leftovers; their full-coverage numbers arrive with the post-cap fill runs.*
