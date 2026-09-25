# v22 dataset — design, gates, counts, training estimate

Status: build in progress (started 2026-09-22). Counts marked *pending* are filled in by the builders'
`stats.json` files as they finish; everything else was measured on the current files.

## 1. Why v22

v21b-1200 is the flagship (religious CER 0.167, PGP CER 0.180 on the benchmarks). Its weaknesses are
measured, not guessed:

- Documentary reads: 44% of human lines right at 0.8 similarity on PGP edition pages; page CER 0.21 on
  pages that decode normally, 0.27 with the 3,500-token cap, parser failures and collapse pages.
- Grounding on documentary photos drifts (boxes stretched 1.45×, 43% of rows below the ink) because
  every grounding row so far was a literary KTIV page.
- The model has never seen a documentary line break: `genizah_clean_v2` (1,055 pages, 992 documents,
  15% of the PGP Hebrew-script editions) was flattened to one line by our own indexer.
- 53 of the 56 documentary validation documents were training documents in the previous set, so the
  documentary eval loss has not been held out since v1.8b.

v22 adds three things: line-broken documentary pages, documentary grounding rows, and extractive
question answering. The user's rules for the QA family: every answer must be text the model can point
to on the page (line index now, bounding box later); no catalogue-style answers; conservative
generation; no LLM in the loop on Judaeo-Arabic.

## 2. Sources and what each contributes

| Source | State | Contributes |
|---|---|---|
| KTIV scrape (`raw_data/cairo_genizah/ktiv`) | 2,070 transcribed mss incl. 185 new since the v3 build (546 pages, 494k letters) | literary transcription + all word-box grounding families; new `short_fragment` family |
| PGP digital editions (`pgp_raw/data/footnotes.csv`) | 6,454 docs ≥100 letters, 6.44M letters, 212k lines; 4,289 usable (image in the served index, outside the benchmark) | documentary page transcription with line breaks; line-structure tasks; the text QA answers are quoted from |
| PGP metadata (`documents.csv`, person relations, `people.csv`) | 3,560 person links on usable docs; dates on 862; descriptions on all | validators for QA (never the answer) |
| Two-reader consensus batch (`ai_reads/…v21b-step1200.jsonl`, rule v3) | 11,592 agreed lines with Kraken-derived geometry | documentary `locate` / `read_box` rows |
| synthetic_hebrew_v3 | unchanged (11,040 renders) | transcription |
| talmud_finetune_v2 | unchanged, unpinned, © HebrewBooks scans | **decision needed**: keep at ≤5% or drop |

## 3. Builders and gates

### 3.1 `genizah_ktiv_v4` (`build_ktiv_dataset.py`, NAS `datasets/genizah_ktiv_v4`)
Same 15 families and gates as v3 (≥150 letters, ≥3 lines, gaps ≤40%, boxes in frame, benchmark
decontamination by id, loose key and 25-letter shingles) plus one new family:

- `short_fragment` (`page_short`): pages with 40–149 letters, damage share ≤0.30 (gap tokens + tokens
  with dots/brackets over all tokens), image in frame, not contaminated. Page-transcription rows only,
  own split so it can be weighted separately. Audit estimate 2026-09-14: 169 pages / 109 mss.
- Images-once export (`images_once.py`): images stored once by content hash + parquet rows, with a
  map-style loader that reconstructs the original rows. v3 on the hub is 77 GB because every row embeds
  its page JPEG; this is what the notebook should load in v22 (train-path change, gated, see §6).

Counts: v3 = 3,700 pages / 1,588 mss / 60k rows / 2.2M letters. **v4 (built 2026-09-22): 4,230 pages
/ 1,750 mss / 67,303 rows (63,966 train, 3,337 val) / 2.57M letters in page rows; `page_short` 189
rows on 214 short pages from 122 mss (38 pages rejected as too damaged, 44 too short, 25
contaminated); 130 mss removed as contaminated, 82 by the religious benchmark.** On the NAS the
DatasetDict is 86 GB (page JPEG embedded in every row) against 11.7 GB of distinct page images, which
is why the images-once export exists. Export result: 67,303 rows → 18,383 distinct JPEGs (11.69 GB) +
a 10 MB rows parquet, a 7.4× reduction, round-trip verified on all of val and 2,748 train rows with 0
mismatches. The 18,383 loose files should be packed into tar shards before going to Colab, and the
NAS DatasetDict must not be opened with `load_from_disk` (it pulls ~30 GB into memory).

**Validation-split fix (2026-09-22):** the builder reshuffles the split whenever the manuscript set
changes, so 77 of the 90 v4 val manuscripts were v3 train manuscripts (seen by v21b) and 75 of v3's
78 val manuscripts landed in v4 train. **Fixed 2026-09-23:** v4 val = all 78 v3 val manuscripts +
12 of the 242 manuscripts new in v4 (seeded) → train 64,266 rows / 1,714 mss, val 3,037 rows / 90
mss (`page_short` 177 train / 12 val); both overlaps now 0 (`split_manifest.json`). Applied to the
images-once parquets and, shard by shard, to the DatasetDict (4,083 rows cross-checked). The builder
now takes `--val-manuscripts` and writes `val_manuscripts.json` on every build, so future rebuilds pin
the split instead of reshuffling. Pre-fix shards are parked in `genizah_ktiv_v4/_pre_split_fix`
(35 GB, deletable). One manuscript (990001398720205171, re-scraped 09-14) now matches benchmark text
on a new page and is dropped from v4; v21b trained on 13 of its other pages.

### 3.2 `pgp_editions_v1` (`build_pgp_editions.py`, NAS `datasets/pgp_editions_v1`)
The risk with editions is the side: an edition covers every side of a document, the training image is
one side. Metadata and labels are not enough — on pages with reader evidence, "Recto" came first in
only 16 of 22 clear two-image cases and a third were mixed. So inclusion is evidence-based:

1. Reader evidence per image: the two-reader pipeline's raw outputs (VLM lines + Kraken fragments).
   The pipeline queue was re-ordered on 2026-09-22 to read every image of the 4,289 usable edition
   documents first, and again on 2026-09-23 (user decision) to read the 2,015 pages of the 1,619
   QA-relevant documents (exact-located persons, PGP-dated, ketubot) before the other 3,166 edition
   pages; one page in flight (no concurrency, by decision). Measured cost on edition pages: Kraken 23 s
   and VLM 38 s median per page (17 lines), overlapped → 74–95 pages/h; QA-relevant set ≈1.5 days,
   all edition pages ≈3 days.
2. Block presence: an edition block is kept when its Hebrew 4-gram presence in the union of the two
   readers' text beats a per-page null (50 unrelated pages), with ≥3 matched n-grams and one line read
   at ≥0.5 similarity. Measured false-keep rate against unrelated pages: 0.09%.
3. Page inclusion: kept blocks must cover ≥90% of the letters of the side that belongs to the image
   (labelled editions) or of the whole edition (unlabelled = single-sided). Both-sides-kept, incomplete
   side, joins/multifragment, and documents with >3 images are dropped.
4. Cleaning: `clean_diplomatic` (restorations → `[...]`), editor commentary lines removed, label lines
   removed, ≥100 letters and ≥3 lines after cleaning.
5. Decontamination: benchmark ids, join partners, 25-letter shingles against benchmark GT.
6. Split: 5% of documents to `val`, join partners kept together — a genuinely held-out documentary
   eval loss for the first time since v1.8b.

Rows per page: `page` (same prompt as KTIV page rows, answer with line breaks), `line_by_number` (≤2),
`line_of_phrase_text` (1). No boxes: these teach line structure without geometry.

Two reader cross-checks were added during the build: a kept block may not contain a run of ≥6 lines
that neither reader read, and a page is dropped when both readers read a line the edition lacks (the
editor skipped lines, or the leaf carries a second text). Editorial marks that `clean_diplomatic`
leaves behind — sic marks `(!)`, parenthesised expansions, `|` break markers — are resolved or the
line is dropped. (The v21b documentary set `clean_v2` still carries them: 123 of 1,055 answers with
`(!)`, 89 with parenthesised expansions.)

Decontamination decisions taken by the builder (documented here because they differ from the
literal v3 rules): benchmark fragments are matched by an institution-agnostic fragment key (11
benchmark fragments sit under `Oxford_Bodleian_Bodl_MS_heb_*` ids that an id check misses — the
grounding set was re-checked with the same key: 0 leaks); join partners are found through shared PGP
ids and true sub-parts rather than by stripping `_<n>` (which would merge whole boxes and drop 1,448
documents); shingle matching excludes near-copies of a benchmark text rather than any two shared
25-letter shingles (legal and ketubba formulae would otherwise drop 1,043 documents;
`--shingle-containment 0` restores the strict rule).

**First cut 2026-09-22 (evidence for 315 of 5,760 usable pages, 5.5%):** 245 eligible pages → 124
kept (51%): dropped for side rule 92 (no block kept 45, side incomplete 34, complex labels 5,
unlabelled incomplete 4, both sides 4), editorial notation 22, cross-checks 6, too short 1. Rows: 124
`page`, 154 `line_by_number`, 85 `line_of_phrase_text`; train 337 / val 26 (val only from documents
never in `clean_v1`/`v2`). Documents removed by the document gates: 209 joins, 33 with >3 images, 15
flattened editions, 12 benchmark duplicates.

**After the QA-relevant block (2026-09-24 01:17; evidence for 2,593 of 5,760 usable pages, 2,270
eligible):** 1,022 pages included from 954 documents (45% of eligible pages with evidence), 825k
letters, 20,964 lines; rows 970 `page` + 1,449 `line_by_number` + 757 `line_of_phrase_text` + 149
val = 3,325. Page outcomes: side incomplete 447, no block kept 410, both sides kept 84, complex side
labels 67, editorial notation 91, unlabelled incomplete 42, reader cross-checks 55, too short 24.
Remaining 3,166 edition pages (documents without located persons or PGP dates) read next.

### 3.3 `documentary_grounding_v1` (`build_documentary_grounding.py`, NAS `datasets/documentary_grounding_v1`)
From agreed lines only (agreement ≥0.8, ≥8 letters, no gap). The line box is the union of the Kraken
fragments assigned to the line (ink geometry), never the VLM box or the evidence union. Rows use the
KTIV `locate` and `read_box` prompts byte-for-byte (same task, different `label_source`), ≤4 locate and
≤2 read-box rows per page, pages with ≥2 usable lines. Two extra filters: lines with letters from
another script (garbled reads) and lines whose text repeats on the page (ambiguous locate targets).

Built 2026-09-22 from an 8,171-record snapshot of the batch: 1,985 pages had agreed lines, 1,317 kept
(668 had <2 usable lines); 11,807 agreed lines → 10,511 usable (727 with gaps, 325 short, 179 box too
tall, 34 other-script, 31 repeated); **7,217 rows = 4,583 locate + 2,634 read_box; 6,909 train / 308
val (55 docs, seeded hash split); 158k letters**; 1,204 PGP photos + 113 KTIV scans; 0 download
failures (sha256-verified); 0 docs excluded by decontam (volume-level matching would drop 536 more,
left off: `--volume-level-decontam`). Arrow 7.7 GB (page embedded ~5.5× per page), images 1.4 GB.
Next step (not in this build): edition lines aligned to Kraken line segments, which would give
full-page grounded rows on documentary pages.

### 3.4 `pgp_qa_v1` (`build_pgp_qa.py`, NAS `datasets/pgp_qa_v1`) — IN the mixture (user decision 2026-09-24)
Every answer is **whole token(s) of one line of the page's own transcription, exactly as written —
never part of a word**: a place written "בדמשק" is quoted "בדמשק", not "דמשק" (prefix letters and
glued punctuation stay with their token). It is returned as `{"line": N, "text": "..."}`; the build
checks every emitted answer against `(?:^|\s)text(?:\s|$)` on its line and raises on a violation
(tests cover it). Metadata only validates.

| Family | Answer | Validation | Pool measured |
|---|---|---|---|
| `qa_person` (sender, recipient, witness, party, validating judge; never scribe, and **not "mentioned"** — dropped after review: many valid answers per page, and the sample's hit was a signatory) | the name as written, whole tokens (an attached ו/ל/ב … kept) | PGP relation row located by the romanization inverter, exact token match, tier 1 or kunya+given only; roles with several PGP entries skipped | 1,727 exact of 3,560 rows, 981 docs; ~95% precision on spot checks |
| `qa_date` | the date line | Hebrew month in the line matches PGP `doc_date_original` **and the line carries the year** (a token after שנת/בשנת/משנתינו or the Judaeo-Arabic סנה/סנת, or ליצירה/לשטרות/לבריאת/למניין) | 738 of 1,493 date-line docs |
| `qa_ketubah_parties` | the formula line naming groom and bride | ≥1 party name located exactly on that line | ~22 docs today (8 with both) |
| `qa_party` | the acknowledgment line | a Party/Witness relation located on that line | ≤402 docs |
| `qa_line_by_number`, `qa_line_of_phrase_text` | the line | by construction | unlimited |
| `qa_abstain` | `{"answer": "not stated"}` | no date line AND no PGP date | ≤10% of QA rows |

Caps: ≤3 QA rows per page image; answer lines with `[...]` skipped; the QA page set is the
`pgp_editions_v1` page set (side-verified). A 250-row stratified review sample
(`qa_review_sample.md`) is produced for the user as a check; **QA rows are part of the v22 mixture from the start (user decision 2026-09-24; the earlier hold-out was the assistant's conservatism, not a requirement).**

After the QA-relevant block (2026-09-24, 1,022 side-verified pages): **293 rows — 194 `qa_date`, 63
`qa_person`, 6 `qa_party`, 29 `qa_abstain`, 1 `qa_ketubah_parties`; train 276 / val 17; 272 pages
carry a QA row.** Funnel: person — 268 exact-located relation rows on included pages, 189 skipped
because PGP lists several people for the role (witnesses, parties), 22 uncertain, 14 gap in line → 63;
date — 402 date-line pages, 85 skipped for a gap elsewhere in the line, 59 for the year on the next
line, 40 for an ambiguous PGP month → 194; party — 601 acknowledgment lines skipped because no
PGP Party/Witness relation is located on that line → 6; ketubah — 56 marriage pages, names not
parsed 26 / not located 23 → 1. **Projection at full coverage is well under the 1,000-row threshold;
revisit per the decision above** (levers: complete-list answers for multi-holder roles, date lines
with a gap elsewhere, formula-validated party lines, place names vs origin/destination).

**Rebuild 2026-09-24 02:00 with the user's additions (same 1,022 pages): 812 rows.** New families:
`qa_date_month` 312 and `qa_date_year` 128 (quote just the month token / the year expression, so a gap
elsewhere on the line or a year on the next line no longer blocks the page; number-word years need a
closing word or gershayim, months must be standalone tokens); `qa_witnesses_list` 19 (complete list
only: every PGP witness located exactly AND every signature-shaped name in the signature region is a
located witness) + `qa_witness_line` 19 (fallback: one signing line); `qa_parties_list` 1 +
`qa_party_line` 15; `qa_ketubah_groom` 2, `qa_ketubah_bride` 1 (name located on any line of a marriage
page). Unchanged: `qa_date` 194, `qa_person` 63, `qa_party` 6, `qa_ketubah_parties` 1; `qa_abstain` 51
(10% cap). Funnel: witnesses — 57 documents with 2+ holders → 39 all located → 21 pass completeness
→ 19 lists; parties — 39 → 14 → 1. 206 tests. Review sample regenerated (200 rows, 13 families).
Duplicate-photo fix (02:30): 42 pages dropped as `duplicate_side_photo` (same image hash or identical
kept lines within a document); with 67 new evidence pages the edition set is now 1,017 pages (rows
1,017 page / 1,499 line_by_number / 785 line_of_phrase_text) and QA is **794 rows** (date 186, month 301,
year 125, person 61, witnesses list 18 + line 19, parties list 1 + line 15, party 6, ketubah 4, abstain
58); 209 tests.

**Levers pulled 2026-09-24 (user: QA must be in the mixture; stop waiting on the remaining levers):**
`qa_party_formula` (acknowledgment lines validated by the formula alone, included only if a 50-line
spot check reads ≥90% correct) and `qa_place` (origin/destination place names located exactly via
`places.csv` variants + a table of common Genizah toponyms).

**Rebuilt 2026-09-24 11:05 with both levers: 929 rows** (date 186, month 270, year 121, place 171,
person 60, witnesses list 18 + line 19, parties list 1 + line 11, party 6, party formula 4, ketubah 4,
abstain 58; train 889 / val 40; 482 pages carry a QA row). `qa_party_formula`: only 22 lines on the
1,017 pages matched the acknowledgment pattern, so all 22 were judged instead of 50; round 1 read 10/22
correct (court and witness "we" formulas introduce the party in the third person); the tightened rule
(singular אנא or מודה/מודים directly followed by the name, no court or witness wording on the line,
legal documents only, pages with two matching lines skipped) read 8/8 and yields 4 rows
(`qa_party_formula_spotcheck.md`, both rounds line by line). `qa_place`: 503 documents with an origin,
destination or location → 305 with the place name located exactly → 171 rows (160 "written", 11
"sent"); skipped 69 with the name on several lines, 70 with a gap in the line, 3 with several PGP
places; guards: names that are also ordinary words (צור, רום, חלב, דן …) never match alone, נוחו עדן and
יציאת מצרים are not Aden or Fustat, extra spellings פסטט and אל קאהרה. Month, year, person and party-line
counts moved down slightly because place rows take some of the 3 slots per page. 232 tests.
Review: `qa_review_sample.md` (250 rows, 15 families) and the visual page `qa_review.html` beside it
(all 929 rows: the page image with the answer line boxed from the Kraken row union at similarity ≥0.5,
else the VLM line box ≥0.6 — 578 Kraken / 253 VLM / 66 without a box; the transcription with the answer
line highlighted; good / wrong / unsure marks kept in the browser and exportable as JSON).

**Whole-token answers (rebuild 2026-09-24 19:36, user rule above).** Place, name and month prompts
now ask for the text "including any attached prefix letter". Places are quoted as the whole token(s)
matched (prefix included); a located name widens to whole tokens only over a prefix particle (a
hyphen-joined or longer word would skip the row: 0 such skips); month tokens with a prefix (בניסן,
לאדר, דאייר) are quoted instead of skipped (a month token with brackets inside is still skipped); the
year keeps its last token's punctuation. **935 rows** (date 186, month 276, year 121, place 171,
person 60, witnesses list 18 + line 19, parties list 1 + line 11, party 6, party formula 4, ketubah 4,
abstain 58; train 895 / val 40; 484 pages carry a QA row). 119 answer texts changed in 118 rows: 115
place (113 gained ב, 1 ו, 1 ל), 1 person (ועמאר בר פראח), 1 parties list (לסעיד בן יוסף, למפצל בן
סלאמה), 1 ketubah bride (לרייסה בת עמרם). The 7 prefixed month tokens that were skipped became
candidates and gave 6 new month rows (one lost to the 3-row cap). Invariant on the full build: 0
violations in 903 answer spans. 261 tests. `qa_review_sample.md` and `qa_review.html` regenerated (935
rows: 583 Kraken / 254 VLM / 66 without a box).

First cut 2026-09-22 (same 124 pages): 20 rows — 14 `qa_date`, 3 `qa_person`, 1 `qa_party`, 2
`qa_abstain`, 0 `qa_ketubah_parties`; train 17 / val 3; 169 tests. Small by design: exact name
matches only, single-holder roles only, and senders/recipients mostly sit in the verso address, which
enters when the verso pages are read. Review sample: `/Volumes/home/studio_offload/datasets/pgp_qa_v1/qa_review_sample.md`.

**Decision (user, 2026-09-22): if the validated QA families total fewer than 1,000 rows at full
evidence coverage, revisit and look for ways to increase the count.** Levers in order of preference
(none lowers name-match precision): ketubah parties from the opening formula once those lines are
read; place names validated against PGP origin/destination; the letter address line validated
against the recipient relation; senders/recipients from verso pages as they are read. Projection at
today's rates: 500–1,000 validated rows plus ~5,000 line-structure rows.

## 4. Proposed mixture (sampling probabilities, by row)

| Bucket | Share | Inside |
|---|---|---|
| Transcription | 0.55 | KTIV v4 page/region/section/line/page_short 0.28 · PGP editions page + line-structure 0.22 · synthetic 0.05 (· Talmud replay ≤0.05 if kept, taken from KTIV) |
| Grounding | 0.25 | KTIV word/line families 0.15 · documentary locate/read_box 0.10 |
| QA | 0.20 | extractive families above (validated rows weighted ≈0.05, line-structure rows the rest) |

Placement augmentation (random margins, offsets, scale, page composites) is applied at training time
to every grounding row to break the two-column page prior.

## 5. Evaluation

- CER: the religious and PGP (131) benchmarks, unchanged, decontaminated.
- Grounding: the existing KTIV grounding eval + a new documentary line-grounding eval from held-out
  agreed lines (on-line rate, IoU).
- Documentary transcription: the new `pgp_editions_v1` val split (line-broken, held out by document).
- QA: ~300 held-out documents stratified by family, scored with partial credit (decided 2026-09-24,
  after the user asked how a full-line answer is penalised when single letters are illegible):
  line-index hit rate; CER of the quoted span against the target (the headline number for full-line
  families); exact match only for short spans (month, year, place, names; whitespace- and
  final-letter-normalised); abstention accuracy on both branches. Exact match on full lines is
  reported but is not a target. Why: the training loss is token-level cross-entropy with teacher
  forcing, so an illegible letter costs its own token and nothing else, exactly as on the page
  transcription rows; only the metric can over-penalise. Answer targets never contain letters the
  editor restored: `clean_diplomatic` turns restorations into gaps and the QA builder skips gapped
  lines (spans must be gap-free). v21b's cached page reads of the answer lines already sit close to
  the targets on formulaic families and far on names: date lines median CER 0.077 (59% ≤0.10, 17%
  exact, 7% >0.5 = whole-page read failures), party lines 0.05; month present 74%, place 73%, year
  expressions 29%, person names 42%, witness lists 44%, witness signature lines median 0.27.
- Train-path gates from the v21 lessons: prepared-dataloader batch check before step 1 (class, shapes
  vs grid, prepared == direct loss), loss-level alarm vs the v21b reference (>2× = stop), warm-start
  regression check on two evals, merger frozen, one name per run.

## 6. Training plan and time estimate

Reference: v21b ran at 62 s/step training, 67.5 s/step wall including evals, batch 1×8 on the Colab GPU
(W&B `kozus0pg`: 1,240 steps, 9,600 samples, 23.3 h). Warm start from v21b-1200.

| Run | Steps | Samples | Colab wall time (67.5 s/step) | Notes |
|---|---|---|---|---|
| Pilot (no QA) | 800 | 6,400 | ~15 h | validates the new sources, images-once loader, documentary grounding; go/no-go on eval |
| Full v22 | 3,000 | 24,000 | ~56 h (≈3 Colab sessions, resumable every 100 steps) | ~0.4 epochs of the ~60k-row transcription pool, ~1.5 passes of the QA rows at 0.20 |
| Full v22 on the ROCm box (R9700, ETA late Oct) | 3,000 | 24,000 | 95–140 h at an assumed 0.4–0.6× of the Colab GPU | unverified stack (ROCm + Unsloth); smoke test first |

The images-once loader changes the train path; it must pass the prepared-dataloader gate on the
notebook before step 1, like the v21b dispatch fix did.

## 7. Open decisions

1. Accept the 0.55 / 0.25 / 0.20 mixture (QA share parked in transcription until the review).
2. Keep or drop the Talmud replay (`talmud_finetune_v2`: unpinned, copyrighted scans, 12% of v21b).
3. Run the full v22 on Colab in sessions, or wait for the ROCm box.
4. Side gate strictness: ≥90% of the side's letters (current) vs ≥80% (more pages, more risk).

## 8. Build log

- 2026-09-22: measurements (lineage, reader accuracy vs editions, romanization inverter, side-label
  validation) done; pipeline re-ordered to edition documents first; builders launched for KTIV v4,
  documentary grounding, PGP editions + QA.

## 9. v2.2a launch (2026-09-24 evening — user: "we are starting training tonight")

The consensus pipeline is NOT a prerequisite: its remaining ~1,400 edition pages only add edition/QA rows
to a later rebuild. The pilot trains on what exists.

**Pilot mixture** `isaacmg/genizah_v22_pilot` (private), built by `build_v22_mixture.py` into
`/Volumes/home/studio_offload/datasets/genizah_v22_pilot` (images-once: `rows/train.parquet`,
`rows/val.parquet`, `images/<sha1>.jpg`, `mixture.json`, `manifest.json`, `README.md`): 8,000 materialised
train rows — KTIV v4 transcription 0.30, PGP editions 0.25, KTIV grounding 0.15, documentary grounding 0.10,
PGP QA 0.20 (QA rows repeat ~1.8×; passes recorded in `mixture.json`) — and 200 val rows across all sources.
Shuffled once with seed 3407 so every window of the map-style dataset carries the mixture. Only the images
those rows reference are copied. Synthetic and Talmud replay are OUT of the pilot (decision: the pilot
measures the new levers; replay can return in the full run if Talmud eval regresses).

**Notebook** `src/finetuning/qwen_hebrew/colab/genizah_v22a.ipynb` (tests `tests/test_colab_notebook_v22a.py`):
v2.1b's install triplet, resolution contract (6.5–7 MP), LoRA r16 tower+language, merger FROZEN, adamw_8bit,
cosine 5e-5, 1×8 batches, eval/save every 100 steps, `max_steps=2000` (8,000 rows = 1,000 steps/epoch; stop
at 800 for the pilot verdict if needed). Warm start = v2.1b step 1200
(`isaacmg/qwen3-vl-8b-hebrew-v21b-ckpt@6724c32c`). Map-style loader (`ImagesOnceDataset` mirror), no
streaming; the prepared-dataloader gate compares the first prepared batch with a direct collate of the same
(recorded) row; resume sets `ignore_data_skip` instead of replaying batches. Checkpoints to the NEW repo
`isaacmg/qwen3-vl-8b-hebrew-v22a-ckpt`, W&B run `genizah_v22a`.

**Local smoke test before launch** `src/finetuning/qwen_hebrew/v22_smoke_check.py` runs sampled rows through
the real Qwen3-VL processor on the Mac: decode, template, patch rows == grid product, ~25k patches per page,
answer after the assistant header, sequence ≤ 12,288.

**Launch checklist (user):**
1. `.venv/bin/hf auth login` (once), then `bash logs/push_v22_pilot.sh` (upload-large-folder, resumable; prints
   the revision to pin as `V22_REVISION`).
2. Open the notebook in Colab (A100), secrets `HF_TOKEN`, `WANDB_API_KEY`; run cells 1→6. Cells 2, 4, 5 and the
   gate in 6 are the go/no-go: shares, hygiene, collator masking on a page and a QA row, tower-LoRA gradients,
   prepared == direct loss.
3. Watch W&B `qwen-hebrew-finetune/genizah_v22a`: train loss LEVEL vs v21b at the same step (>2× = stop),
   eval loss at 100/200 must not rise on the KTIV/editions rows (warm-start regression rule).
4. Hard evals at 400/800 with the LM Studio harness (`hard_eval_ckpt.sh`), QA scored with partial credit (§5).
