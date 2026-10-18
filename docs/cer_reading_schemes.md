# Transcription CER by reading scheme: PGP-131 and religious-140

Status 2026-10-12, written for review. Every number comes from the scorer CSVs, probe reports, ROVER and oracle reports
in this repo; the raw tables with file and line sources are in `docs/cer_reading_schemes_tables.md`. Tonight's decode-policy
runs will add rows; their report lands in `logs/decode_policy/MORNING_REPORT.md`.

## 0. How to read the numbers

- **Two CER definitions circulate; never mix them.** *CER\** (PGP-131) and *CERsub* (religious-140) are the median per-page
  CER over *substantive* pages only, so pages that abstained (< 25 letters), loop-collapsed (one 12-gram covers > 45 % of the
  text) or hallucinated (5-gram precision < 0.10) are left out. *All-page mean / median* include every page; per-page CER is
  unclamped, so a looping page can score 5 or 10 and the mean is dominated by such pages. The cards show CER\*.
- **Benchmarks.** PGP-131: 131 documentary fragments (Judaeo-Arabic 82, Hebrew 20, Aramaic 6, untagged 22, Arabic script 1),
  ground truth flattened to one line, resolution about ±0.010. Religious-140: 140 KTIV pages from 82 manuscripts, 33
  two-column, 138 Talmud, ground truth rev 1.1, resolution about ±0.03 because two-column order flips move whole pages.
- **Ground-truth caveat.** On PGP-131, 7,284 ground-truth characters on 8 pages are not in the photograph or are GT defects:
  0.043 of all GT characters. Every PGP CER carries that; relative comparisons stand, absolute numbers are too high, and a
  sub-10 % *mean* is out of reach until those 8 pages are fixed. On religious-140 the same figure is 0.002.

## 1. Where a single whole-page read stands

| model | PGP CER\* | PGP mean / median (all pages) | religious CERsub ALL | two-column (33) | single-column (107) | religious mean / median |
|---|---|---|---|---|---|---|
| v2.0a (v20a-1800) | 0.182 | 0.393 / 0.184 | 0.184 | 0.218 | 0.179 | 0.304 / 0.196 |
| **v2.1b (v21b-1200)** | 0.177 | 0.329 / 0.177 | **0.165** | 0.164 | 0.166 | 0.276 / 0.167 |
| v2.2a (v22a-1300) | 0.164 | 0.227 / 0.164 | 0.186 | 0.221 | 0.182 | 0.271 / 0.192 |
| v2.2b (v22b-1200) | 0.160 | 0.312 / 0.161 | 0.181 | 0.222 | 0.170 | 0.265 / 0.185 |
| blend a50 (0.5 v21b + 0.5 v22a) | 0.158 | 0.300 / 0.161 | 0.172 | 0.204 | 0.172 | 0.261 / 0.178 |
| blend a25 | 0.166 | **0.207** / 0.166 | 0.172 | **0.153** | 0.183 | 0.262 / 0.186 |
| v2.3a-1200 | **0.153** | 0.369 / 0.155 | not run | | | |
| v2.3a-1500 | 0.155 | 0.214 / 0.155 | not run (tonight) | | | |

Reads: the best card numbers are 0.153 on PGP (v2.3a-1200) and 0.165 on religious (v2.1b). The spread between a model's
median and mean is its loop pages: v2.3a-1200 has the best median and a mean of 0.369, v2.3a-1500 nearly the same median
with a mean of 0.214. No later checkpoint of v2.1b beat step 1200 (step 1700: 0.175 ALL, 0.208 two-column). The a25 blend
has the best two-column number of any single read, 0.153, and the lowest PGP mean, at the price of a worse median.

Other readers, for scale. Frontier API models on PGP-131: Claude Opus CER\* 0.416 with 28 % abstained and 29 % hallucinated,
Claude Sonnet 0.435 (68 % hallucinated), Gemini Pro 0.432 (68 pages), Gemini Flash 0.542, GPT 0.736 (50 % abstained),
Apple Vision OCR 0.649. Kraken line recognition improved from CER\* 0.344 (kraken 4, default segmenter) to 0.251 (kraken 7,
our fine-tuned segmenter, multi-view) on PGP and from 0.519 to 0.301 on religious. The MiDRASH Zenodo transcriptions, read
with their own segmenters, score a median 0.240 on the 138 religious pages we could match, and **0.124 on the 33 two-column
pages against v2.1b's 0.164**; on single-column pages they are far behind (0.255 against 0.167).

## 2. Reading schemes tried

### 2a. Column prompts without cropping: no

33 two-column religious pages, v2.0a. Whole page 0.243 median; "transcribe ONLY the right column" plus "ONLY the left
column" on the full image 0.256. Paired: 9 pages better, 8 worse. On the 12 pages the whole-page read had under-read, the
prompts helped (0.607 to 0.448), but on 4 pages the model transcribed both columns twice (length ratio about 1.5) and 8 of 33
over-generated. In the section-prompt experiment on documentary openings the "only the left page" prompt did recover the
named page's text, but its line geometry drifted by up to 2.5 row heights. The prompt steers coverage; it does not fix layout.

### 2b. Column crops at the ink gap: the one scheme that moved the number

Same pages, the image split at its widest vertical ink gap, page prompt per half, right half first: **0.151 median on the 30
pages with a detectable gap, from 0.243**. On the 10 pages the whole-page read had under-read: 0.612 to 0.239. On the 20
pages it had read on target: 0.126 to 0.121, neutral. Paired mean change −0.030; 10 wins over 0.05, 4 losses. Failure modes
to guard: one crop looped (1.28 to 2.79), one page split badly (0.14 to 0.57), three pages had no detectable gap. Part of the
gain is mechanical: the processor upsamples every image to about 7 megapixels, so a column crop is read at about 1.35× the
linear resolution of the whole page. This scheme sat as an "idea" row in the tracker until tonight.

### 2c. Horizontal bands ("sections") as the default read: no; as a rescue: yes

PGP-131, v1.8b. Reading every page as overlapping 550-px bands: CER\* 0.249 to 0.445, higher CER on 117 of 131 pages, 54
bands bisected and 28 truncated. But of the 13 pages the whole-page read failed on, 7 became substantive (median 8.76 to
0.465) while 4 good pages broke. That is the shape the sections-retry runner already uses: whole page first, bands only for
pages that fail the gate.

### 2d. Per-line crops: dead on this material

Two smoke pages, v1.8b blind per-line reads of Kraken segments: CER 1.145 and 1.647 against whole-page 0.263 and 0.255,
worse even than Kraken's own text on the same lines (0.322, 0.557). Kraken's segments on Genizah pages are word fragments
(median about 145 × 43 px) and the VLM hallucinates on tiny crops. The 131-page run and the "adversarial" mode (the VLM sees
Kraken's read and corrects it) were never run. Line-level reading stays Kraken's job; the VLM needs a region with context.

### 2e. Half-page re-reads as a rescue for failed reads: cheap and effective

35 held-out failed reads of the blend (28 loops, 4 capped, 3 under-read), scored against PGP editions rather than the
benchmark pages. Re-reading the same request: 18 better, 6 worse. Temperature 0.3 with a seed: 21 / 7. Two halves with 12 %
overlap, joined: **23 / 7, pooled CER 0.688 to 0.571**. Chaining "same, then seed, then halves" and keeping the first healthy
read leaves 28 of 35 healthy at pooled 0.552. Raising max_tokens to 6,000 fixed all 4 capped reads. Loops are the one class
where re-reading does not reliably help (13 of 28 healthy after halves).

### 2f. Line recognisers as readers

Kraken is not a competitor to the VLM on single-column pages (0.318 against 0.166), but on two-column religious pages the
line recognisers beat it: our fine-tuned segmenter 0.206 and MiDRASH 0.124 against v2.1b's 0.164. The VLM's two-column
weakness is structural, not optical, and an independent second reader for exactly those pages already exists.

### 2g. Voting over several reads

- Majority vote of three reads of the same page (blend, v2.2b, v2.2a; 112 PGP documents): mean 0.156 to 0.150, median
  0.145 to 0.139, better on 57 documents and worse on 1. A free gain wherever three reads are affordable.
- ROVER over six checkpoints on PGP-131, character level: **0.188 mean / 0.140 median** against the best single reader's
  0.214 / 0.155; the per-page oracle (best reader per page) is 0.185 / 0.142, so the vote already matches it. Bootstrap:
  median change −0.015 [−0.022, −0.005]. Adding Kraken as a voter: 0.185 / 0.136.
- ROVER on religious-140 (four checkpoints, v2.3a missing): character level 0.243 / 0.166, no better than v2.1b's median;
  **with Kraken as a voter 0.231 / 0.159**, better on 105 pages and worse on 19; two-column median 0.164 to 0.155.
- Slot oracle, mean / median. **These are not results of any system.** The oracle picks, in every alignment slot,
  the candidate that matches the ground truth whenever at least one reader wrote it, so it is the ceiling for any
  selector over these readers; the realistic expectation for a learned voter lies between ROVER and this line, and
  most of the gap is lone-dissenter cases a voter cannot identify from agreement alone (lone dissenters are right
  3 % of the time on PGP, 8 to 15 % on religious):

| configuration | best single | ROVER char | slot oracle | headroom |
|---|---|---|---|---|
| PGP-131, six VLMs | 0.214 / 0.155 | 0.188 / 0.140 | 0.137 / **0.091** | 0.052 / 0.050 |
| PGP-131, six VLMs + Kraken | 0.214 / 0.155 | 0.185 / 0.136 | 0.124 / **0.073** | 0.062 / 0.063 |
| religious-140, four VLMs | 0.261 / 0.167 | 0.243 / 0.166 | 0.169 / 0.111 | 0.074 / 0.055 |
| religious-140, four VLMs + Kraken | 0.261 / 0.167 | 0.231 / 0.159 | 0.133 / **0.093** | 0.097 / 0.066 |

  Of the characters ROVER still gets wrong, 72 % on PGP are wrong in every reader (no candidate holds the truth); on religious
  with Kraken the split is 49 % learnable / 51 % unreachable. Diversity of readers, not voter cleverness, is the binding
  constraint, and prompt variants of one model are correlated readers.

## 3. What this says about the 10 % target

Today: medians 0.155 (PGP) and 0.165 (religious) for one read, 0.140 and 0.159 with the vote. The remaining distance splits
into four parts, in the order they are worth attacking:

1. **Structural failures of the single read**: two columns, under-read pages, loops. The column-crop probe (0.243 to 0.151)
   and the rescue experiments (23 of 35 failed reads recovered) are the evidence; tonight's decode policy puts both into one
   read path with health guards and a whole-page fallback. First live pages: a two-column Talmud leaf went from 609 to 1,217
   letters against 1,223 in the ground truth; a looping PGP page from 1,087 to 1,484 against 1,523.
2. **Reader diversity for the vote.** The oracle says a median near 0.09 is attainable on both benchmarks with the candidates
   that already exist, and the line recognisers supply exactly the two-column strength the VLM lacks. Column-crop reads,
   band reads, a second model family and MiDRASH's reads on NLI pages are the cheap additions; a learned slot voter is the
   step after ROVER.
3. **Ground truth.** The 8 PGP pages with text absent from the image add 0.043 to every PGP mean. Fix them, report medians
   and the cleaned mean.
4. **The letter-level floor.** On religious pages the substitution component is 0.075 on two-column and 0.143 on
   single-column pages; once structure is fixed, what remains is model quality. For literary pages the next lever is
   canonical alignment (the text is known; the model's job becomes finding where the leaf deviates), for documentary pages
   more descriptive training data in the v11 round.

A single read under 0.10 on both benchmarks is not on the table with the current models; the vote over structurally repaired
reads is the path with evidence behind it, and the oracle gives it room.

## 4. Running tonight (2026-10-12 → 13), one sequential LM Studio consumer

| stage | what | expected |
|---|---|---|
| 1 | religious-140, v2.1b decode policy: 28 pages read as two column crops, 3 rescued, 109 kept | two-column CERsub 0.164 → about 0.15 |
| 2 | PGP-131, v2.1b: rescue of the 3 failed pages (no PGP page is two-column) | means down, medians unchanged |
| 3 | PGP-131, v2.3a-1500: rescue of its 3 failed pages | same |
| 4 | religious-140, v2.3a-1500 whole page: the missing baseline | first literary number for v2.3a |
| 5 | religious-140, v2.3a-1500 decode policy | |
| 6 | ROVER on both benchmarks with the new readers as voters, with and without Kraken | part of the oracle headroom |

Column detection on the cached Kraken lines, judged against the benchmark's layout labels (the policy never reads them):
26 of 33 two-column pages found, 2 false positives (both two-page spreads with two real text blocks), 7 missed, 6 of them
pages Kraken never splits. Outputs land as new model stems that the existing scorers pick up; every page carries a JSON trace
of what was read and why under `src/datasets/evaluations/transcription_results/decode_policy_details/`. The queue log is
`logs/decode_policy/overnight_1012.log`; the report is assembled into `logs/decode_policy/MORNING_REPORT.md` when it ends.

## 5. Results of the overnight decode-policy runs (2026-10-12, 01:45–04:24; part 2 from 04:26)

Every stage finished with return code 0; the policy made 76 model calls on religious-140 per reader (28 two-column pages
read as two crops, 3 rescues) and 6 on PGP-131 (3 rescues). Scorer rows: `religious_scores_summary.csv`,
`genizah_offline_summary.csv`; traces per page under `transcription_results/decode_policy_details/`.

### Religious-140 (final, after the guard fix and re-decode of the two affected pages)

| reader | ALL CERsub / F1 | two-column (33) | single-column (107) | paired vs whole page (±0.02) |
|---|---|---|---|---|
| v2.1b whole page | 0.165 / 0.869 | 0.164 / F1 0.864 | 0.166 | |
| **v2.1b decode policy** | **0.163 / 0.877** | **0.115** / F1 0.918 | 0.167 | 17 better / 2 worse before the fix; the 2 are repaired |
| a50 blend whole page (production documentary reader) | 0.172 / 0.862 | 0.204 / F1 0.863 | 0.172 | |
| **a50 decode policy** | **0.167 / 0.867** | **0.107** / F1 0.923 | 0.172 | |
| v2.3a-1500 whole page (first literary number) | 0.188 / 0.851 | 0.259 / F1 0.814 | 0.177 | vs v2.1b: 30 better / 35 worse |
| v2.3a-1500 decode policy | 0.170 / 0.870 | 0.120 / F1 0.913 | 0.179 | |

The two-column class moves from 0.164 to 0.115 (v2.1b), 0.204 to 0.107 (a50) and 0.259 to 0.120 (v2.3a): the column crop
beats the probe's 0.151 and nearly erases the differences between the three models on that class. Single-column pages are
unchanged, as designed (the policy only touches a single-column page when its whole-page read fails health). The biggest
gains are loop pages the rescue repaired (1.808 → 0.142, 1.348 → 0.129) and the two two-page spreads the detector flagged as
columns (1.35 → 0.71, 1.71 → 0.78). v2.3a is the worse literary reader (all-page median 0.214 against 0.167), which settles
whether it should replace v2.1b there: it should not.

**The two regressions and their fix.** FL202443168 (0.246 → 0.775) and FL203739117 (0.148 → 0.541), identical on every
reader, shared one cause: Kraken under-segmented both pages (178 and 231 letters where the page holds about 1,150), so the
over-generation guard, which budgets a crop read against Kraken's letters, rejected correct column reads of about 550
letters, rejected the band re-reads the same way, and each column fell back to a thin share of the whole-page read (326 and
558 letters against healthy whole-page reads of 1,144 and 1,091). Two rules now prevent it, both without a model call:
never return less than 0.8× a healthy whole-page read, and when Kraken read under half of what the whole-page read did,
judge over-generation against 1.5× the whole-page read instead of 3× Kraken. A replay over every trace showed these were
the only two pages per reader the rules touch; re-decoded, they assemble at 970 to 1,100 letters. The policy's own
transcript of each page (which units were read, which guard fired, where it fell back) is the JSON trace beside the output.

### PGP-131

| reader | CER\* | all-page mean / median | substantive pages |
|---|---|---|---|
| v2.1b whole page | 0.177 | 0.329 / 0.177 | 130 |
| v2.1b decode policy | 0.177 | **0.222** / 0.177 | 131 |
| v2.3a-1500 whole page | 0.155 | 0.214 / 0.155 | 131 |
| v2.3a-1500 decode policy | **0.152** | **0.194** / 0.152 | 131 |

As predicted from the dry run, PGP has no two-column pages, so the policy only rescues the three failed reads per reader:
medians stay within noise, the means drop by a third because the loop pages are gone, and every page is now substantive.

### ROVER with the new readers (character level, all-page mean / median; final)

| configuration | religious-140 | PGP-131 |
|---|---|---|
| best single read before tonight (v2.1b whole page) | 0.276 / 0.167 | 0.214 / 0.155 (v2.3a-1500) |
| best single read now (v2.1b decode policy) | 0.227 / 0.163 | 0.194 / 0.152 (v2.3a-1500 decode policy) |
| previous best vote (whole-page VLMs + Kraken) | 0.231 / 0.159 | 0.185 / 0.136 |
| three decode-policy readers only (v2.1b, a50, v2.3a) | 0.205 / 0.156 | |
| **decode-policy v2.1b, a50, v2.3a + whole-page v2.2a, v2.2b** | **0.201 / 0.145** | |
| same + Kraken | 0.200 / 0.151 | |
| four whole-page VLMs + three decode-policy readers + Kraken | 0.208 / 0.153 | 0.188 / 0.140 |
| per-page oracle over the best set | 0.195 / 0.147 | 0.185 / 0.141 |

On religious the vote over column-correct readers reaches 0.201 mean / 0.145 median, from 0.231 / 0.159 for the previous
best vote and 0.276 / 0.167 for the single v2.1b read a week ago; on the median it now beats its own per-page oracle, so the
combination produces pages better than any one reader. Adding Kraken or the whole-page readers to that set no longer helps,
which is the column-order drag made visible. On PGP the policy changes three pages per reader and the vote is unchanged at
its per-page oracle; gains there need a slot-level voter or genuinely new readers.

### Slot oracle recomputed on tonight's best religious set (decode-policy v2.1b, a50, v2.3a + whole-page v2.2a, v2.2b)

| | without Kraken | with Kraken |
|---|---|---|
| best single read (v2.1b decode policy) | 0.227 / 0.163 | same |
| per-page oracle | 0.195 / 0.147 | 0.190 / 0.147 |
| ROVER (real) | **0.201 / 0.145** | 0.200 / 0.151 |
| slot oracle (ceiling, uses the ground truth) | 0.139 / 0.099 | 0.121 / 0.083 |

Ground-truth characters: 81.9 % the vote already gets right, 5.2 % some reader got right but the vote did not pick,
12.9 % no reader produced (11.5 % with Kraken in the pool). Lone-dissenter precision 7.9 %. The recoverable 5 % needs an
external signal (language model or canonical text scoring the candidates); the 12 % needs better or more diverse readers.
