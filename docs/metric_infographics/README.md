# Metric infographics (blog series for the Genizah transcription paper)

Five self-contained HTML pages (pages 2, 3 and 4 also carry the post-paper checkpoints HebVL-1.8/1.9/2.0/2.1 and the KTIV-fine-tuned Kraken, marked †), one per evaluation metric, sharing one design system
(slate paper, verdigris = correct / on the page, madder = wrong / invented; Newsreader,
IBM Plex, Frank Ruhl Libre via Google Fonts). Every number is computed with the repo's own
functions; the worked example is one line of Oxford, Bodleian MS heb. a 2/4 against a
constructed model output with one misread letter and an invented closing blessing.

| page | metric | artifact |
|---|---|---|
| 1_ngram_precision.html | order-independent 5-gram precision | https://claude.ai/artifact/3TkT7j4agETFhhFZpobySR |
| 2_behaviour_classes.html | abstained / loop / hallucinated / substantive gates | https://claude.ai/artifact/7nsjNUbBuv9TDWLEpMXc4L |
| 3_aligned_prf.html | aligned character precision / recall / F1 | https://claude.ai/artifact/95Zwak1sbiZ9R7TLgzKUeo |
| 4_cer_wer.html | CER / WER, strict and lenient, unclamped, conditional | https://claude.ai/artifact/Jje1c6bqwZKKtobM62DvFB |
| 5_tier_ab.html | Tier A / Tier B convergence rule | https://claude.ai/artifact/Cpp5wUN31oKFQbEJDQSM4T |

SPEC_COMMON.md is the shared build spec. Sources: score_genizah_offline.py, metrics.py,
and docs/ngram_metric_audit.md (2026-09 audit) for the sensitivity figures.
