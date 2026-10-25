# Training-mixture lineage, transfer and sample efficiency: Qwen3-VL-8B Hebrew (HebVL) fine-tunes

Internal report, 2026-10-02, built read-only from the Hub dataset revisions, notebooks, W&B configs/histories and the result CSVs listed under Sources. Consumed-page counts come from local sampler/stream replays (session scratch scripts, not kept in the repo; rerunnable on request).

**Conventions**
- *page*: one real source page image.
  - Talmud: one amud scan, stem `vol_folio[b]`.
  - PGP: one document photograph, deduplicated by image hash.
  - KTIV: `ktiv_<sysnum>_<FL>`.
- Crops and lines of a page count once.
- *samples*: steps × effective batch. Every run here uses an effective batch of 8 (1 per device × 8 grad-accum × 1 A100; W&B config).
- *mixture epoch*: one pass over the interleaved (oversampled) mixture. This is what W&B `train/epoch` reports. It is **not** a pass over unique rows.
- Consumed-page counts for map-style runs come from an exact replay: `interleave_datasets` with `all_exhausted`, plus accelerate `SeedableRandomSampler` with `data_seed` 3407. The replayed mixture lengths match W&B/trainer_state to the row. The remaining assumption is that Colab's `torch.randperm` matches local torch 2.8.
- Where a count is an expectation rather than an exact replay, it is marked "~".

---

## 1. Direct answer: initial Talmud pages

**Page** = one single-page scan (amud) of the HebrewBooks.org Vilna-layout Talmud (37 volumes; `talmud_full/talmud_complete`, 5,373 scans). The scans carry a ©Moznaim watermark marked "No commercial use allowed".

| Item | Count | Source |
|---|---|---|
| Corpus scans | 5,373 | `talmud_complete/{pdfs,converted_images,texts}` |
| Benchmark held out (by stem, hard leak assert) | 66 GT stems: vol 01 ff. 2–15b (28) + vol 02 ff. 2–20b (38); 65 have images | `build_dataset.py:289-294`; 0/66 in any hub split |
| Val pages (0 train overlap) | 100 | hub metadata |
| Pages with no usable GT / crop | 53 | 52 no gemara GT, 1 manifest miss (03_102) |
| **v1 train pages** | **5,154** | 5,154 + 100 + 53 + 66 = 5,373 |
| v1 train rows | 17,476 = crops 14,696 (gemara 5,154 / rashi 4,772 / tosafot 4,770) + page_extract 2,780 (972 / 904 / 904, on 972 pages) | hub @8c26a322; notebook cell 2 asserts |
| Distinct train images | 15,668 (14,696 crop strips + 972 full pages reused ×≤3 prompts) | audit |
| Val / smoke rows | 340 (100 pages) / 32 (11 train pages) | hub card |
| Synthetic | 0 | — |

The holdout is page-disjoint, not volume-disjoint: 01_16 through 01_20b are in training.

**What each run actually consumed (initial era):**

| Run | Rows / pages available | Consumed | Notes |
|---|---|---|---|
| MLX overfit-8 gate (configs/smoke.yaml) | smoke split, shuffle(seed 0).select(8) | 8 rows from **7** pages, 320 iters (config; no log) | Corrected by the audit; the report said 3 pages. Rows were mostly rashi/tosafot apparatus crops. |
| Colab overfit-8 gate (notebook cell 6) | smoke[:8] | 8 rows / 3 pages, 300 steps | Same in-memory `model` as cell 8. Whether v1 inherited these 300 steps is **unverifiable**. |
| MLX scouts (5p7j44ui, jeye33w7, nexqfnvn joint; ohhks8pu, ew0eclnd, w31jdn6r text-only) | 2,000 crop rows ≤1,800 target tokens, shuffle seed 0 → **1,741 pages** (728 / 671 / 601) | joint nexqfnvn 1,920 iters (crashed; ckpt 1500 evaluated); text-only 1,000 + 1,360 iters. ~1,352–1,681 pages seen (expectation; mlx-vlm permutation unseeded) | README says max_samples 4000, but every logged run used 2000. All scouts were worse than base (CER 3.44 / 4.08 vs 2.92). Nothing carried forward. |
| stage2_fullpage | — | never launched | config only |
| **v1 Colab (0yz2y8j2)** | 17,476 rows / 5,154 pages; interleave crops 0.85 / pages 0.15, all_exhausted → 18,738 rows/epoch (15,958 crop draws incl. 1,262 repeats + 2,780 page rows) | 4,686 steps = 37,476 samples (2.00 mixture epochs). **Promoted step 3800 = 30,394 samples** (1.622 mixture epochs; 1.74× unique rows; every page seen, since epoch 1 ends at step 2343). Epoch-1 model step 2400 = 19,194. | lr 1e-4 cosine; QLoRA r16; vision LoRA dead (lora_B = 0), so language-side only |
| v1.5 (lhfus88b) | same v1 dataset + synthetic_rashi v1 (18,960) | 16,000 samples; ~4,375 Talmud pages redrawn, **0 new** | warm v1@3800 |
| v1.6 (4s4u4f0c) | talmud_finetune_v2: 20,297 rows on 5,206 pages (151 not in v1 train: 99 ex-v1-val + 52 rashi/tosafot-only) | 8,800 samples to step 1100; ~3,710 pages drawn, of which **~94 new** | warm v1.5@2000 |

**Cumulative:**
- About **5,248** distinct real Talmud pages were drawn by the end of v1.6 (5,154 + ~94), out of a 5,305-page union pool.
- Samples along v1@3800 → v1.5 → v1.6@1100: **55,194**. Of these, ~46,158 were real Talmud rows and ~9,036 were synthetic.
- Learnable real Talmud draws at v1@3800 were only **~13,575**: 30,394 × (5,589 gemara crops + 2,780 page rows) / 18,738. The other ~16.8k were apparatus-strip rashi/tosafot crops whose target text is mostly absent from the image.

---

## 2. Lineage table

Abbreviations:
- **T** = Talmud print; **G** = Genizah PGP documentary photographs; **K** = KTIV manuscripts.
- **"new / cum"** = distinct real pages consumed that no earlier link in the warm-start chain had drawn, and the cumulative consumed total along the chain.
- **Samples** are counted at the promoted (or evaluated) checkpoint.
- **[k]** marks a killed, failed, never-trained or side-arm row.

| Version | Dates | Warm start | New data this step | Real pages new / cum (T · G · K) | Synthetic rows (built; drawn) | Train rows (built → mixture len) | 8 × steps = samples (epochs) | Promoted ckpt | Headline |
|---|---|---|---|---|---|---|---|---|---|
| MLX scouts [k] | 07-14/15 | stock bf16 | 2k-row crop subset | T ≤1,741 (subset) · – · – | 0 | 2,000 | 1 × 1,920 iters max (0.96 subset ep) | none (iter 1500 eval) | CER 3.44 / 4.08 vs base 2.92 |
| v1 | 07-16 → 07-17 | stock 4-bit (overfit-8 inheritance ?¹) | talmud_finetune v1 | T 5,154 / 5,154 | 0 | 17,476 → 18,738 | 8 × 3800 = **30,394** (1.62 mix ep) | 3800 `e2f85dde` | 30 val crops mean 1.44 (gemara 0.91; gemara best at 2000: 0.65). No 65-page row² |
| v1.5 | 07-19 → 07-20 | v1@3800 | synthetic_rashi v1 (11–30px) | T 0 / 5,154 | 18,960; ~6,394 | 36,436 → 47,434 | 8 × 2000 = 16,000 (0.337) | 2000 `e050aca8` | Talmud 65pp means g/r/t 0.059 / 0.362 / 0.512 (med 0.048 / 0.144 / 0.130) |
| v1.6 | 07-21 → 07-22 | v1.5@2000 | talmud_v2 (full-page rashi/tosafot, crops dropped), synth v2 (18–48px), 6.5–7 MP | T ~94 / ~5,248 | 18,988; ~2,642 | 39,285 → 63,223 | 8 × 1100 = 8,800 (0.139) | 1100 `b9f47f32` (paper); 1000 `d6bb5dd3` (card) | 1100: 0.0130 / 0.183 / 0.308 (med 0.004 / 0.014 / 0.011); 600: 0.013 / 0.226 / 0.126; 1000: 0.090 / 0.047 / 0.099. PGP-131 F1 0.083 (zero-shot) |
| v1.7 | 07-24 | v1.6@1100 | genizah_clean_v1 (1,077 docs) at 0.35 | T ≤57³ / ~5,248–5,305 · G 962 / 962 | 18,988 (v2); 1,302 | 40,185 → 103,012 | 8 × 800 = 6,400 (0.062; G 2.06 passes) | 800 `068a2cc1` | PGP-131 F1 0.553, CER* 0.379; Talmud 0.014 / 0.084 / 0.037 |
| v1.8a [side arm] | 08-09/10 | v1.7@800 | clean_v2 (repaired, 1,055) | G 133 / 1,095 | 1,141 drawn | 40,163 → 103,012 | 8 × 700 = 5,600 (0.054) | 700 `9dbd0070` | F1 0.616 (data repair alone +0.063) |
| v1.8b | 08-10 | v1.7@800 | same rows and order as 1.8a; **vision LoRA live** | G 133 / 1,095 | 1,141 drawn | same | 8 × 700 = 5,600 (0.054; G 1.85) | 700 `c80313f8` | F1 0.807, CER* 0.249; step 400: 0.794. Talmud means 0.089 / 0.088 / 0.057, median 0.006 |
| v1.8.1 [k] | — | — | never launched (killed by the step-400 read) | 0 | 0 | 0 | 0 | — | — |
| v1.9a | 08-20/21 | v18b@700 | KTIV v1 (1,607 pp / 861 mss), synthetic_hebrew_v3 | G ≤34 / ~1,095–1,129 · K 1,527 / 1,527 | 11,040; ~2,071 | 35,359 → 103,012 | 8 × 1300 = 10,400 (0.101; K 1.94, G 1.98 passes) | 1300 `43e21bd7` (700 `10c8cbf8`) | PGP F1 0.862 / CER* 0.196 (700: 0.854 / 0.199); religious-140 F1 0.816 / CERsub 0.216; locate 42/72 zero-shot |
| v1.9b [side arm] | 08-28/29 | v18b@700 | identical data; merger LoRA | same as 1.9a | same | same | 8 × 1300 = 10,400 | 1300 `f8618c28` (not adopted) | PGP 0.868; religious 0.810 / 0.234 |
| v1.9c [side arm] | 08-30 → 09-01 | v18b@700 | identical data; full-weight merger | exact 4,427 pages drawn at 1800 (K 1,585 · G 1,005 · T 1,837) | ~2,867 | same | 8 × 1800 = 14,400 (0.140) | 1800 `6620fb0d` (not adopted) | PGP 0.870; religious 0.811 / 0.225 |
| v2.0a | 09-03 → 09-05 | v19a@1300 | KTIV v2 (3,150 pp / 1,484 mss) + 8% grounding (12,776 rows) | K ~1.6k / ~3.1k⁴ | 11,040; 2,082 | 56,562 → 160,104 | 8 × 1800 = 14,400 (0.090; K rows 0.25) | 1800 `af9df6a0` | PGP 0.875 / 0.1825; religious 0.849 / 0.184; locate 53/72, read_box 0.129 |
| v2.0b [side arm] | 09-06 → 09-09 | v19a@1300 | identical to v20a; merger LoRA | 5,875 drawn @2000 (no new) | 2,319 | same | 8 × 2000 = 16,000 | none (final `4cae9d52`) | eval loss 0.668; boxes never beat v20a; transcription eval aborted |
| v2.0c [k] | — | — | never trained | 0 | 0 | 0 | 0 | — | — |
| v2.1 [k] | 09-11 | v20a@1800 | KTIV v3, streamed; **image-blind** (dispatcher truncation) | text-only | 765 | 84,298 rows, streamed | 8 × 700 = 5,600 (blind) | none | eval loss rose 0.683 → 0.764 |
| v2.1b | 09-11 → 09-15 | v20a@1800 | KTIV v3: +27,638 rows in 6 new grounding families on the **same pages**; grounding 8% → 20% | K **0 new train pages** / ~3.2–3.3k · G all 1,055 clean_v2 drawn (cum ~1,120) | 11,040; ~1,334 | 84,298 (streamed) | 8 × 1200 = 9,600 (0.114 of source rows; K frag 0.81 pass, G 1.45) | 1200 `6724c32c` (**model of record**) | PGP 0.880 / 0.1766; religious 0.869 / 0.165 (two-col 0.164); boxes on right line 0.800 |
| v2.2a pilot | 09-25/26 | v21b@1200 | PGP editions, documentary grounding (pseudo-labels), PGP QA, KTIV v4 subset; no Talmud, no synthetic | K +269 / ~3.5k · G +1,221 / ~2.34k | 0 | 8,000 (materialised) | 8 × 1300 = 10,400 (**1.30 ep**) | 1300 `f1ad6300` (evaluated, not flagship) | PGP 0.889 / 0.1635; religious 0.860 / 0.186 (regression); doc-grounding right-line 97.7% |
| v2.2i a50 | 09-28 | weight blend 0.5·v21b-1200 + 0.5·v22a-1300 | none | 0 | 0 | 0 | 0 | a50 (routed documentary reader) | PGP CER* 0.158 / F1 0.891; religious 0.172 |
| v2.2b (running) | 10-02 → | v21b@1200 (not v22a) | genizah_v22_full rev2 `89fb5af4`: 20,000 rows, one pass, lr 3e-5 | planned K 3,268 · G 3,163 · T 1,459 (vs v22a: +4,479); consumed so far 3,979 | 1,400 | 20,000 | 8 × 760 = 6,080 live (0.304); plan 8 × 2500 = 20,000 (1.0) | none yet (last Hub ckpt 700 `e1cc69c9`) | eval loss 0.630 → 0.575 @700 (own val); no hard evals |

**Footnotes**
1. v1 base is unverifiable. Notebook cell 6 (overfit-8) trains the same in-memory `model` that cell 8 trains. Whether the user re-ran cell 6 in the launch session is not logged.
2. v1 was never run on the 65-page Talmud benchmark. All v1 numbers come from 30 val crops in the MLX quick_eval harness, where rashi/tosafot crops are invalid.
3. v1.7 drew 2,261 talmud_v2 pages. How many of v1.6's ~57 undrawn new pages were among them was not computed.
4. KTIV cumulative after v20a is an estimate. The union of v19a's 1,527 v1 pages and v20a's 2,888 v2 pages is roughly 2,888 + ~114 v1/v2-shared pages drawn only by v19a + ~156 v1-only pages ≈ 3.1–3.2k. The exact union was not computed.

---

## 3. Cumulative exposure by domain (flagship warm-start chain)

The chain is v1@3800 → v1.5 → v1.6@1100 → v1.7@800 → v1.8b@700 → v1.9a@1300 → v2.0a@1800 → v2.1b@1200 → v2.2a@1300. Side arms are excluded. Samples per domain come from the replayed draws; the v1.9a split uses expectations, giving ±~20.

| After | Talmud pages (cum, consumed) | Genizah PGP pages | KTIV pages | Talmud samples | Synthetic samples | PGP samples | KTIV samples | Total samples |
|---|---|---|---|---|---|---|---|---|
| v1@3800 | 5,154 | 0 | 0 | 30,394 | 0 | 0 | 0 | 30,394 |
| v1.5 | 5,154 | 0 | 0 | 40,000 | 6,394 | 0 | 0 | 46,394 |
| v1.6@1100 | ~5,248 | 0 | 0 | 46,158 | 9,036 | 0 | 0 | 55,194 |
| v1.7@800 | ~5,248–5,305 | 962 | 0 | 49,039 | 10,338 | 2,217 | 0 | 61,594 |
| v1.8b@700 | ≤5,305 | 1,095 | 0 | 51,548 | 11,479 | 4,167 | 0 | 67,194 |
| v1.9a@1300 | ≤5,305 | ~1,095–1,129 | 1,527 | ~53,115 | ~13,550 | ~6,273 | ~4,676 | ~77,594 |
| v2.0a@1800 | ≤5,305 | ~1,095–1,129 | ~3.1k | ~55,183 | ~15,632 | ~9,059 | ~12,140 | ~91,994 |
| v2.1b@1200 | ≤5,305 | ~1,120 | ~3.2–3.3k | ~56,300 | ~16,966 | ~10,592 | ~17,756 | ~101,594 |
| v2.2a@1300 | ≤5,305 | ~2,340 | ~3.5k | ~56,300 | ~16,966 | ~16,312 | ~22,436 | ~111,994 |
| v2.2b plan (from v21b) | ≤5,305 (1,459 replayed) | ~3,645 | ~3.6k | +1,600 | +1,400 | +8,800 | +8,200 | 101,594 + 20,000 |

Notes:
- Talmud never grows beyond the 5,305-page union after v1.6. Later versions only replay it, at a share of 45% (v1.7/1.8), 15% (v1.9), 14% (v2.0a), ~12% (v2.1b), 0% (v2.2a) and 8% (v2.2b).
- **Half of all samples the flagship has ever seen are Talmud print** (56k of 112k). The documentary PGP domain the model is mostly judged on received ~16k samples.
- v1.8 consumed exactly the Talmud and synthetic rows of v1.7's first 5,600 samples (same seed and lengths), so it added no new replay content.
- v2.2a PGP samples assume pilot shares × 1.3 epochs (editions 2,000 + documentary grounding 800 + QA 1,600 per epoch). KTIV = (2,400 + 1,200) × 1.3.

---

## 4. Transfer learning evidence

| Transfer | Model (trained on) | Benchmark | Result | Caveats |
|---|---|---|---|---|
| Baseline | stock Qwen3-VL-8B, LM Studio 4-bit, native resolution | Talmud 65pp; PGP-131; Arabic v0 | Talmud means 5.08 / 4.45 / 6.11 (all 193 sections > 0.3). PGP F1 0.010, 1.5% substantive, 58.8% loops. Arabic 5-gram F1 0.003, 51/58 loops | Base is not quantisation- or resolution-matched to the 8-bit 6.5–7 MP fine-tunes |
| **Print → Genizah handwriting (zero-shot)** | v1.6@1100 (Vilna print + synthetic Rashi; 55,194 samples) | PGP-131 | F1 0.083 vs 0.010; substantive 13.7% vs 1.5%; loops 16.8%; hallucinated 68.7%; Tier A F1 0.417 | Strongly language-dependent: Hebrew (n=20) F1 0.591 / 50% substantive, Aramaic (6) 0.551, Judaeo-Arabic (82) 0.051 / 3.7%. Transfer is real for Hebrew script and nearly absent for JA |
| Print + Genizah stage | v1.7 (+2,217 Genizah draws on 962 pages) | PGP-131 | F1 0.553, substantive 66.4% | ~2.2k in-domain samples bought +0.47 F1 on top of the print prior. There is no from-scratch Genizah-only control, so the print prior's contribution is not isolated |
| **Genizah → Talmud print retention** | v1.7 (45% Talmud / 20% synthetic replay) vs parent v1.6@1100, same eval run ldykmhn4 | Talmud 65pp | means 0.014 / 0.084 / 0.037 vs 0.013 / 0.183 / 0.308; medians unchanged (~0.005 / 0.015 / 0.012) | Retained. The mean gain comes from fewer collapses, not better typical reading. The paper's "no collapse pages" is false (§8) |
| | v1.8b (vision LoRA trained on handwriting) | Talmud 65pp (W&B 8toyje16) | means 0.089 / 0.088 / 0.057; per-page median 0.006 (best); 7 collapse pages | Not in the paper CSV. **No Talmud page run exists for v1.9a or later**, so retention after v1.8b is unknown |
| | v1.6 → v1.8b | forgetting probe | synthetic Rashi median 0.0077 → 0.0017; Talmud page_extract 0.011 → 0.0065 | n = 30 / 4 |
| Print → modern Hebrew / unseen fonts | v1.5 / v1.7 / v1.8b | anecdote; 20-face render probe | v1.5 news screenshot near-perfect (no CER). v1.7 reads 16 unseen square faces at median CER 0.000–0.030; Solitreo collapses (0.677; v1.8b 0.759) | Synthetic renders, not scans |
| Print + documentary → Talmud manuscripts (religious-140) | — | religious-140 | **unknown**: no model before v1.9a was scored | v1.9a already trained on KTIV v1 (decontaminated by manuscript), so 0.816 F1 is in-family, not zero-shot |
| Literary KTIV → documentary PGP | v18b@700 → v1.9a@700 (same 5,600 samples; KTIV v1 added, synthetic swapped) | PGP-131 | F1 0.807 → 0.854; ngramP 0.475 → 0.584; hallucinated 9% → 2%; CER* 0.249 → 0.199 | Positive cross-genre transfer. Confounded with synthetic_v3 (12 faces) and reweighting. Genizah draws were similar (1,950 vs 1,130) |
| KTIV v2 + grounding → PGP | v1.9a → v2.0a | PGP-131 | F1 0.862 → 0.875; CER* 0.196 → 0.1825 | Small. Also +14,400 samples and new tasks |
| **Grounding zero-shot** | v1.9a@1300 (no box rows ever) | grounding eval | locate centre-hit 42/72 (chance 2–4%), IoU 0.288; grounded JSON 19/24; line-IoU 0.150 | v1.9b merger LoRA raised line-IoU to 0.437 with no grounding data, while text CER worsened |
| Grounding: literary → documentary | v2.1b@1200 (box rows only from KTIV religious pages) | 801 held-out documentary rows | locate hit 93.5%, IoU ≥ 0.5 78.7%, read_box CER 0.069; v2.2a with documentary rows: 97.7% / 82.8% / 0.057 | Documentary box targets are v21b+Kraken pseudo-labels (self-distillation) |
| Documentary grounding → KTIV locate (reverse) | v2.2a (0 new religious grounding pages) | grounding eval | locate 56 → 65/72; grounded line-IoU 0.623 → 0.699 | v2.2a still had 1,200 KTIV grounding rows (v4), mostly on known pages |
| Mixture shift → literary regression | v2.1b → v2.2a | religious-140 | CERsub 0.165 → 0.186 (regression) while PGP improved 0.177 → 0.164 | The eval timeline says v2.2a had "no new religious data". In fact KTIV was 45% of its rows (mixture.json), but full-page KTIV transcription fell to ~395 rows (~514 samples) vs ~2,557 page draws in v2.1b. **Hypothesis**: loss of full-page literary transcription share, not KTIV absence. Weight interpolation recovers most of the gap (a75 0.168) |
| QA | v2.1b (no QA) → v2.2a (~1,600 QA samples per epoch over 895 rows) | pilot val, n = 37 | exact 0 → 17%; line-hit 10 → 24%; abstain 1/8 both | Format learned, retrieval not; place answers dominated by one label |
| Arabic script | v2.1b (0.03% Arabic letters in targets) | first look | 5-gram F1 0.051 vs base 0.003 | No Arabic-script training anywhere. Formal run in progress |

---

## 5. Sample efficiency

### 5a. Between consecutive versions: PGP-131 (the only benchmark scored for v1.6 → v2.2a)

F1 = median aligned F1; CER* = median CER over substantive pages; post-2026-09-22 scoring throughout.

| Step | ΔF1 | ΔCER* | New real pages (first drawn) | In-domain (PGP) draws | Samples | ΔF1 per 1k samples | Main confounds |
|---|---|---|---|---|---|---|---|
| v1.6-1100 → v1.7-800 | 0.0833 → 0.553 = **+0.470** | (subst 13.7 → 66.4%) | G +962 | 2,217 | 6,400 | **0.073** (per 1k PGP draws: 0.212; per 100 new pages: 0.049) | none major (first domain data) |
| v1.7 → v1.8a-700 | 0.553 → 0.6162 = +0.063 | 0.379 → 0.407 | G +133 | 1,950 | 5,600 | 0.011 | GT repair (45 docs), 23 dropped, LR schedule completed |
| v1.8a → v1.8b (matched rows) | 0.6162 → 0.8074 = **+0.191** | 0.407 → 0.249 | 0 | 0 extra | 0 extra | — | single variable: vision LoRA live |
| v1.8b-400 → 700 | 0.7944 → 0.8074 = +0.013 | 0.2533 → 0.2487 | G +201 (695 → 896) | +813 | +2,400 | 0.0054 | — |
| v1.8b-700 → v1.9a-700 | 0.8074 → 0.8541 = +0.047 | 0.249 → 0.199 | K +1,281 (other genre) | 1,130 | 5,600 | 0.0084 | synthetic swap (rashi → v3 12 faces), reweighting |
| v1.9a-700 → 1300 | 0.8541 → 0.8624 = +0.008 | 0.199 → 0.196 | K +246, G +227 | +976 | +4,800 | 0.0017 | — |
| v1.9a-1300 → v2.0a-1800 | 0.8624 → 0.8751 = +0.013 | 0.1957 → 0.1825 | K ~+1.6k | 2,786 | 14,400 | 0.0009 | grounding tasks, KTIV v2 GT-mash fixes |
| v2.0a-1800 → v2.1b-1200 | 0.8751 → 0.8797 = +0.005 | 0.1825 → 0.1766 | **0** train pages | 1,533 | 9,600 | 0.0005 | streamed order, grounding 20% |
| v2.1b-1200 → v2.2a-1300 | 0.8797 → 0.8892 = +0.0095 | 0.1766 → 0.1635 | G +1,221 (+269 K) | ~5,720 (editions + doc-grounding + QA) | 10,400 | 0.0009 (ΔCER* per 1k new PGP pages: −0.011) | new task formats, no replay |

Reading the table:
- Gains per sample fall by about two orders of magnitude from v1.7 to v2.x.
- The only steps that moved PGP by more than 0.04 F1 were:
  - first in-domain pages (+0.47);
  - turning the vision tower on (+0.19);
  - adding a second handwriting corpus (+0.047 at matched samples).
- After v1.9a, ~38k further samples bought +0.027 F1 in total.
- F1 noise is about ±0.008 (T = 0.1 sampling), so every step from v1.9a-700 onward is within 1–2× noise individually.

### 5b. Religious-140 (v1.9a onward)

| Step | ΔCERsub | ΔF1 | New K pages | K draws | Samples | Note |
|---|---|---|---|---|---|---|
| v1.9a-1300 → v2.0a-1800 | 0.2164 → 0.1840 = **−0.032** | 0.816 → 0.849 = +0.033 | ~+1.6k (v2 has +1,707 vs v1 train) | 7,464 (frag 4,244) | 14,400 | ≈ −0.020 CER per 1k new KTIV pages |
| v2.0a → v2.1b-1200 | 0.1840 → 0.1651 = −0.019 | +0.021 | 0 | 5,616 (1,912 grounding) | 9,600 | two-column 0.218 → 0.164; gain from task mix, not pages |
| v2.1b → v2.2a-1300 | 0.1651 → 0.1856 = **+0.021** | −0.010 | +269 | ~4,680 (mostly lines / regions / grounding) | 10,400 | regression; see §4 |

### 5c. Talmud 65-page benchmark (strict CER means, medians in brackets)

| Model | Cum. samples | Gemara | Rashi | Tosafot |
|---|---|---|---|---|
| base | 0 | 5.08 | 4.45 | 6.11 |
| v1.5-2000 | 46,394 | 0.059 (0.048) | 0.362 (0.144) | 0.512 (0.130) |
| v1.6-600 | 51,194 | 0.013 (0.004) | 0.226 (0.017) | 0.126 (0.014) |
| v1.6-1000 | 54,394 | 0.090 (0.005) | 0.047 (0.014) | 0.099 (0.013) |
| v1.6-1100 | 55,194 | 0.013 (0.004) | 0.183 (0.014) | 0.308 (0.011) |
| v1.7-800 | 61,594 | 0.014 (0.005) | 0.084 (0.015) | 0.037 (0.012) |
| v1.8b-700 | 67,194 | 0.089 | 0.088 | 0.057 (all-section median 0.006) |

The biggest per-sample jump is v1.5 → v1.6@600: rashi median 0.144 → 0.017 after 4,800 samples (~1,700 small-script page draws). It coincides with three simultaneous changes:
- dropping the apparatus crops;
- training full-page small-script extraction;
- 3.2 → 6.5 MP resolution.

It is a data-quality and resolution effect, not a data-quantity one. After v1.6@600, medians are flat; means move only through 1–7 collapse pages.

### 5d. Within-run saturation (where the gain lands)

| Run | Curve | Saturation point |
|---|---|---|
| v1 (30 val crops, gemara only valid) | gemara mean 1.03 @1000 / 0.65 @2000 / 0.84 @2900 / 0.91 @3800 / 0.89 @4686 | gemara best at 16,000 samples (~0.85 mixture epoch). Step 3800 was promoted on a 30-crop mean of which 19 crops are invalid |
| v1.6 | eval loss 0.0891 @100 → 0.0616 @1100, still falling; benchmark medians flat 600–1100 | ≤4,800 samples |
| v1.7 | eval loss 1.083 → 0.781 @800, still falling; only step 800 benchmarked | unknown (stopped at 800/1500 for budget) |
| v1.8b | F1 0.794 @400 vs 0.807 @700 | 95% of the gain over v1.7 by 3,200 samples (1.08 Genizah passes) |
| v1.9a / v1.9c | PGP 0.854 @700 → 0.862 @1300 → (v1.9c) 0.870 @1800; religious (v1.9c) 0.800 @700 / 0.811 @1200 / 0.811 @1800 | ~5,600–9,600 samples (~1–1.8 passes over KTIV and Genizah pages); loss keeps falling (0.579 → 0.531) |
| v2.0a | PGP 0.876 @1500 / 0.875 @1800; religious 0.851 / 0.849; eval loss flat from 1700 | ≤12,000 samples |
| v2.1b | boxes on right line 0.848 @600 / 0.742 @900 / 0.800 @1200; full 1200 vs 1700: religious 0.869 vs 0.864, PGP 0.880 vs 0.8825 | grounding by 4,800 samples (945 grounding draws on ~434 pages / 220 mss); transcription by 9,600 |
| v2.2a | eval loss −0.10 over epoch 1 (0.738 → 0.638), −0.003 over 300 steps of epoch 2; train loss 0.33–0.55 on repeats | 1 pass over 8,000 rows; epoch 2 is memorisation |

**Pattern.** After each data change, transcription benchmarks saturate within ~3–6k samples, which is about 1–2 passes over the small in-domain real-page source. Teacher-forced eval loss keeps falling well past benchmark saturation (v18b, v19c, v20a). Directly supervised capabilities need very few pages: v21b grounding came from ~434 pages, and v1.7's handwriting gain from 962 documents.

### 5e. Data-scale arms

- **There is no clean data-scale ablation in the lineage.** Commit 5545e38's "data-scale" notebook (v1.9a) is a single arm: +KTIV v1 and +synthetic v3 vs v18b.
- Clean single-variable pairs that do exist are vision (v1.8a/b) and merger (v1.9a/b/c, v2.0a/b). Data and step order are identical within each pair.
- v2.2b vs v2.2a is the closest data-scale comparison: 20,000 rows / 7,890 pages / 1 pass vs 8,000 rows / 3,766 pages / 1.3 passes. It is confounded by LR (3e-5 vs 5e-5), Talmud and synthetic replay (only in v2.2b), QA v1 → v2, and different val sets. Hard evals are pending.

---

## 6. Comparability caveats

- **Harness.** MLX quick_eval numbers (scouts, v1 sweep, v1 epoch-1/2, v1.5 MLX) are not comparable with LM Studio numbers. v1.5's MLX numbers were shown to be invalid. The v1 sweep was never re-checked and may understate v1, which also trained via formatting_func.
- **Talmud crops.** All rashi/tosafot crop rows and crop CERs (v1/v1.5 training, the 30-crop sweep, the 195-crop benchmark, the forgetting probe crop column) use apparatus-strip geometry. Only gemara crops and page-level numbers count.
- **Talmud 65pp.** Means are dominated by collapse pages, so medians are the robust statistic. Closed-model rows changed after submission (August fills on the uncommitted `camera-ready-fills` branch). Gemini Pro and GPT-5.6 coverage is 14–31% and biased toward easy pages.
- **PGP-131.**
  - Scorer v3 (2026-08-04) changed abstain/hallucination labels.
  - The 2026-09-22 editorial-mark change altered GT on 35/131 docs: v21b 0.180 → 0.1766, v20a 0.185 → 0.1825.
  - n-gram clipping (not default) moves Tier A from 55 to 50.
  - Lite slices (30 PGP / 36 religious) are not comparable with full sets.
- **Religious-140.**
  - GT rev 1.0 → 1.1 (2026-08-29): v1.9a F1 0.806 → 0.816.
  - Two CER definitions circulate. CERsub (cards, "CER*") gives v1.9a 0.216 / v2.0a 0.184 / v2.1b 0.165 / v2.2a 0.186. The all-page median gives 0.239 / 0.196 / 0.167 / 0.192. Rankings of the interpolations flip between them, so never mix the two.
- **Decode noise.** T = 0.1 sampling with a single training seed. Per-page flips are noise. Median noise is about ±0.008 F1 (PGP) and about ±0.02 CER (religious).
- **Eval loss.** It is not comparable across versions: each version has its own val set.
  - talmud_v2 val is 99/100 v1 train pages.
  - clean_v2 val is 53/56 clean_v1 train docs (32 of the 40 eval docs were consumed by v1.7).
  - KTIV v2 val holds 78 v1-train pages, and KTIV v3 val holds 82 v2-train pages.
  - v1 had no eval set.
- **Epoch semantics.** W&B `train/epoch` is relative to the oversampled interleave (or step/max_steps for streamed v2.1/v2.1b). It is not a count of passes over unique rows.
- **Consumed counts.**
  - Map-style runs (v1–v2.0b, v2.2a/b): exact replay under the seeded-sampler assumption.
  - v1.9 counts were corrected to exact values: 3,836 pages @1300 (K 1,527 / G 928 / T 1,381) and 4,427 @1800.
  - v2.1/v2.1b streaming counts are a replication of datasets 4.3.0 semantics (no shard shuffle, sys_num-ordered files, 512-row buffer). Draw totals are plausible; page identity is unverified.
- **Base row.** It is 4-bit at native resolution. A matched 8-bit 7 MP base exists but has not been run on these benchmarks.
- **Kraken baselines moved** (kraken 4 blla → k7 ktivseg_e10): PGP CER* 0.344 → 0.251. Do not use Kraken as a fixed reference across dates.
- **Benchmark scope.** PGP-131 is documentary and 62% Judaeo-Arabic. Religious-140 is 138/140 Talmud manuscripts. Neither covers cursive/Solitreo or Arabic script.
- **Leakage notes.**
  - The synthetic_rashi word pool (first 400 sorted GT stems) contains words from 63 of the 66 Talmud benchmark stems, as shuffled single words.
  - KTIV manuscript 990001398720205171 (13 train pages) matches benchmark text on one re-scraped page.
  - 2 unverified frozen-150 PGP docs (T-S 20.99 pgpid 3558; T-S 12.708 pgpid 3199) are in the v2.2 mixtures. PGP-131 itself is clean.

---

## 7. Gaps and how to measure them

| Gap | Cheapest recovery |
|---|---|
| Whether v1 inherited the 300-step overfit-8 adapter | W&B 0yz2y8j2 `sampledHistory` of train/loss at steps 1–20, compared with v1.5's fresh-start loss (a heavily overfit start shows a spike or anomaly). Otherwise unrecoverable; treat as caveat |
| Colab torch / datasets versions behind the replays | W&B run files (`requirements.txt`, `wandb-metadata.json`) for 0yz2y8j2, vjpkl6rh, bdtsxqio, genizah-v20a, kozus0pg. This confirms the randperm assumption and the v2.1b no-shuffle stream order |
| v1 on the 65-page Talmud benchmark | Evaluate isaacmg/qwen3-vl-8b-hebrew-ckpt @e2f85dde (and step 2000 / 2400) in LM Studio, one model at a time. Completes the print curve from 0 → 30k samples |
| Talmud print retention after v1.8b | One 65-page run each for v1.9a-1300, v2.0a-1800, v2.1b-1200, v2.2a-1300 (sequential queue) |
| Print → manuscript zero-shot | religious-140 for v1.6-1100, v1.7-800, v1.8b-700 (and the matched base). This is the missing first point of the KTIV transfer curve |
| Matched base | qwen3-vl-8b-base-8bit-7mp on Talmud 65pp, PGP-131 and religious-140 |
| v2.0b transcription quality | Rerun the aborted PGP / religious hard eval on v20b-2000 |
| v2.2a epoch-1 vs epoch-2 | Finish the abandoned v22a-1000 PGP eval (stopped at 60/131) and add religious |
| v2.2b | Hard evals at 1000 / 1500 / 2500 once trained. This gives the only data-scale-ish comparison with v2.2a |
| v2.1b realised page identity | Unrecoverable from logs. The v2.2b session_plan approach fixes this going forward |
| Exact cumulative unions (KTIV after v20a; Talmud after v1.7) | Join the existing replay outputs (v19x.py, v20_consumed, v21_consumed) on page ids. Pure scratch computation |

**Clean sample-efficiency experiments.** Colab is used for training only; evals run locally, one at a time.

1. **Checkpoint-density curves at zero training cost.** Every run has Hub checkpoints every 100 steps. Evaluate v1.7 (100–800), v1.8b (100–700) and v1.9a (100–1300) on full PGP-131, and v1.6 (100–1100) on Talmud 65pp, at ~5 points each. This gives samples-to-saturation curves for already-trained data.
2. **Nested page-count ablation at fixed steps.** From v2.1b-1200, train 4 arms on nested PGP-edition subsets (250 / 500 / 1,000 / 2,229 pages) with fixed 400 steps (3,200 samples, inside the saturation window seen in v1.8b) and an identical non-PGP remainder. Score PGP-131 only. Cost is about 4 × 7.2 A100-h.
3. **Unique pages vs passes at fixed samples.** Two arms at 8,000 samples:
   - A: 2,000 pages × 4 passes;
   - B: 8,000 pages × 1 pass (drawn from the v4 / editions / documentary-grounding pools).
   Same LR, same warm start. This directly tests the v2.2a lesson "more unique rows, not more passes".
4. **Talmud-replay share ablation.** Replay at 0 / 8 / 15% at fixed steps from v2.1b-1200. Score religious-140 plus Talmud 65pp. This separates the v2.2a religious regression (replay / full-page share) from data novelty.

---

## 8. Reconciliation with the paper

| Paper statement | Found | Status |
|---|---|---|
| 08_finetuning.tex:5-9: "printed Talmud pages and section crops" (no counts) | v1: 5,154 train pages, 17,476 rows (14,696 crops + 2,780 page rows on 972 pages). v1.6: talmud_v2, 20,297 rows on 5,206 pages (crops reduced to gemara-only). ~5,248 distinct pages drawn by HebVL-1.6. 9,542 v1 crop rows had apparatus geometry | Add counts and the crop-defect disclosure |
| "20k synthetic Rashi-script images" | Two different 20k sets: synthetic_rashi v1 (18,960 train + 1,040 eval; 11–30px; modes 9,421 / 4,826 / 4,713) used in v1.5, and v2 (18,988 + 1,012; 18–48px) used in v1.6–v1.8. Consumed draws by HebVL-1.7: ~6,394 + ~2,642 + 1,302 ≈ 10.3k | Imprecise: "two 20k sets, ~10k drawn" |
| "1,145 cleaned Genizah image-transcription pairs" | 1,145 is the manifest. genizah_clean_v1 @9f6e289d = 1,077 train + 57 val (1,134; 11 lost to dead or small images). HebVL-1.7 consumed **962** distinct docs over 2,217 draws | Over-states training docs by ~16% (1,145 vs 962 drawn / 1,077 built) |
| :24-29: HebVL-1.7 = "identical recipe plus Genizah handwriting" | Warm-started from v1.6@1100. New mixture: Genizah 0.35 / Talmud 0.45 / synthetic 0.20; 6,400 samples (0.062 mixture epoch); stopped at 800/1,500 | Reword as "continued training from HebVL-1.6 on a mixture adding …" |
| :29-32: v1.7 strongest on printed Talmud "with no collapse pages" | talmud_paper_long.csv: v1.7 has 7 sections > 0.3 and 3 > 1.0 (worst 1.65, 02_15b rashi). Memory's "worst 0.485" is also stale | **Incorrect** as written |
| :15-22: all 108 vision LoRA matrices B == 0 in every checkpoint | True for v1, v1.5, v1.6, v1.7 (and v1.8a). Live from v1.8b onward | Correct for the models in the paper |
| :34-37: 3.9% docs / 10.1% chars self-duplicated | 45/1,145 docs, repaired in clean_v2 | Consistent (denominator is the manifest) |
| Talmud train/test disjointness (not stated; reviewer Wf2b) | Enforced in code: 66 stems excluded with a leak assert; 0/66 in any split; all 65 evaluated pages held out. Page-disjoint only: adjacent folios 01_16–01_20b are in train. Synthetic word pool includes single words from 63/66 benchmark pages | State it, with both nuances |
| Genizah disjoint at document / shelfmark level | Re-verified 0 overlaps for clean_v1 and clean_v2 vs genizah_test_v1 (150) and the verified 131 | Correct. For future versions: 2 frozen-150 docs are in the v2.2 mixtures (PGP-131 unaffected) |
| appendix C: HebVL-1.6 = v16_step1100 | Model card calls step 1000 "official"; memory called step 1000 the v1.6 row. CSV holds both (1000: 0.090 / 0.047 / 0.099; 1100: 0.013 / 0.183 / 0.308) | Paper is internally consistent (1100). Align the card |
| Memory "v1 = 17.3k mixture", "37,488 samples", "ckpts every 250 steps" | Mixture epoch 18,738; 37,476 samples over 2 epochs; promoted 30,394; ckpts every 100 steps from step 100 | Memory stale |

---

## Sources

**Notebooks and code**
- `src/finetuning/qwen_hebrew/colab/`:
  - `qwen3_vl_hebrew_unsloth.ipynb` (cells 2 / 6 / 8)
  - `rashi_focus_v15.ipynb`, `pages_focus_v16.ipynb`, `genizah_focus_v17.ipynb`, `genizah_focus_v18.ipynb`
  - `genizah_focus_v19{a,b,c}.ipynb`, `genizah_v20{a,b,c}.ipynb`
  - `genizah_v21b_dispatchfix.ipynb`, `genizah_v22a.ipynb`, `genizah_v22b.ipynb`
- Run copies with outputs: `src/datasets/raw_data/cairo_genizah/ktiv/genizah_v20a_5.ipynb`, `genizah_v21(1).ipynb`
- Builders: `build_dataset.py` (@25d5d34 for v1; leak assert :289-294), `talmud_crop.py:76-84`, `synthetic_rashi.py`, `build_genizah_dataset.py`, `build_ktiv_dataset.py`, `build_v22_mixture.py`, `images_once.py`, `mlx_train.py`, `configs/{smoke,stage1_joint,ablation_text_only,text_only_continue,stage2_fullpage}.yaml`

**HF datasets** (revisions)
- `isaacmg/talmud_finetune@8c26a322`, `talmud_finetune_v2@813dd440` (stats.json)
- `synthetic_rashi@884bed8c` (v1) / `@b4e1254e` (v2)
- `genizah_clean_v1@9f6e289d`, `genizah_clean_v2@57366ad3`
- `genizah_ktiv_v1@ccf3a25e`, `genizah_ktiv_v2@1bb20e20`, `genizah_ktiv_v3@45fe477b`
- `synthetic_hebrew_v3@59abcf7c`
- `genizah_v22_pilot@c8c15251`, `genizah_v22_full@89fb5af4`

**HF checkpoints**
- `qwen3-vl-8b-hebrew-ckpt@e2f85dde` (v1-3800; trainer_state)
- `-rashi-ckpt@e050aca8`, `-v16-ckpt@b9f47f32` / `@d6bb5dd3`, `-v17-ckpt@068a2cc1`
- `-v18a@9dbd0070`, `-v18b@c80313f8` / `@7fbb7d40`
- `-v19a@43e21bd7` / `@10c8cbf8`, `-v19b@f8618c28`, `-v19c@6620fb0d` / `@34e7d8d8`
- `-v20a@af9df6a0`, `-v20b@4cae9d52`
- `-v21b@6724c32c`, `-v22a@f1ad6300`, `-v22b@e1cc69c9` (+ `session_plan.json`)

**W&B `igodfried/qwen-hebrew-finetune`** (run ids)
- Scouts: 5p7j44ui, jeye33w7, nexqfnvn, ohhks8pu, ew0eclnd, w31jdn6r
- v1 to v1.8: 0yz2y8j2, lhfus88b, 4s4u4f0c, vjpkl6rh, 8knu1jz8, hbt2sfls, 8yqryixh
- v1.9: bdtsxqio, z74xg18k, j5ba4acb, ugu8llz6
- v2.0 / v2.1: genizah-v20a, aey7ug4i, 45gofe1g, s21stpko, js3ku3tw, kozus0pg, jmnbgc9g, e4nbikcc, w7eme256, a6c0q20m
- v2.2: amvy4kfz, 8h6pgqdk
- Hard-eval runs: v19c-, v20a-, v21b-, v22a-hard-evals

**W&B `igodfried/cairo-genizah-transcription`**: nk5d9rpy, nzk9kn1n, g4ckcoof, ldykmhn4, 8toyje16, 2uj4keee

**NAS** (`/Volumes/home/studio_offload/datasets/`): `talmud_finetune_v2_images_once`, `genizah_ktiv_v{2,3}` (stats.json), `genizah_ktiv_v4_images_once`, `genizah_v22_pilot/` and `genizah_v22_full/` (mixture.json, stats.json, manifest.json), `pgp_editions_v1`, `documentary_grounding_v1`, `pgp_qa_v{1,2}`

**Local**: `src/datasets/processed/{qwen_hebrew_genizah, qwen_hebrew_genizah_v2, genizah_ktiv_v1, synthetic_hebrew_v3, synthetic_rashi}`; `src/datasets/raw_data/genizah_cleanup/` (manifests, `forgetting_probe.log`)

**Results**
- `src/datasets/evaluations/talmud_results/paper_table/talmud_paper_{summary,long}.csv`
- `.../transcription_results/paper_table/genizah_offline_{summary,long,by_script}.csv`
- `.../genizah_religious_v1/religious_scores_{summary,long}.csv`
- `.../transcription_results/qwen_finetune_evals/sweep_results/`
- `.../grounding_eval/*_results.json`
- `logs/v22_task_eval/`, `logs/hard_eval_v22*_full.log`

**Docs and paper**
- Docs: `docs/v22_dataset.md`, `docs/v21b_offline_status.md`, `docs/postmortem_v21_dispatch_truncation.md`
- Paper: `~/Documents/GitHub/genizah-paper/sections/08_finetuning.tex`, `03_benchmarks.tex`, `appendix/C_snapshots.tex`, `tables/talmud.tex`

**Memory**: `project_qwen_finetune.md`, `project_v18_prep.md`, `project_v19_consensus.md`, `project_v21b_results.md`, `project_v22_scope.md`, `project_v22b_exact_resume.md`, `project_genizah_final_benchmark.md`
