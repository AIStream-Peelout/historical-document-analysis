# Arabic-script benchmark (v0)

Built 2026-10-02. Measures whether a model can read Genizah documents written in **Arabic script**
(not Judaeo-Arabic, which is Arabic in Hebrew letters and is what the fine-tunes are trained on).
Arabic script is 0.03 % of the letters in the v22 training mixture, and none of the existing
benchmarks can score it: their letter counting is Hebrew-only, so an Arabic ground truth is skipped.

## 1. What is in it

| | |
|---|---|
| Location | `/Volumes/home/studio_offload/datasets/arabic_script_benchmark_v0/` (NAS) |
| Documents | 187 (69 state documents, 56 legal documents, 54 letters, 8 other) |
| Images | 358 (228 from the image store, 130 from library IIIF manifests: Cambridge 126, Princeton 2, PGP-hosted 2) |
| Ground truth | 122,046 Arabic letters; median 484 per document, range 201 to 5,816 |
| Holding libraries | Cambridge 151, JTS 24, Bodleian 6, Rylands 3, Penn 2, Princeton 1 |
| Images and edition cover the same sides | 57 documents (see section 4) |

Files:

- `benchmark.jsonl`: one record per document (`id`, `pgpids`, `shelfmark`, `library`, `doc_type`,
  `doc_date`, `pgp_side`, `images[{file, image_index, label, side, source}]`,
  `gt{sections[{side, lines}], text, arabic_letters, hebrew_letters, editions}`).
  The file is in a fixed pseudo-random order (sorted by a hash of the id), so **the first N
  records are an unbiased sample**; the runner's `--limit N` relies on this.
- `images/<doc_id>__<n>.jpg`, `images_manifest.json` (source URL, bytes, sha256 or the error of
  every image), `build_report.json` (counts and every excluded document with its reason).
- `outputs/<model>__<prompt>.jsonl` (model answers), `scores/` (per-document CSVs and summaries).
- `images_excluded/`: images fetched for documents that were excluded afterwards.

The ids are registered in `raw_data/cairo_genizah/decontam/arabic_script_benchmark_v0.json`
(`ids`, `pgpids`; rewritten by every full build). `benchmark_registry.registered_benchmark_documents()`
returns them, and `build_pgp_editions.py` and `build_documentary_grounding.py` now add them to the
documents they hold out. Any new builder of training rows must do the same.

## 2. How the documents were chosen

Source: tier 1 of `merged/arabic_scrape_priority_queue.csv` (`arabic_scrape_queue.py`): a Princeton
Geniza Project edition with at least 200 Arabic letters, an image we may use, not in
`genizah_clean_v2`. 314 PGP documents are in that tier; 219 resolve to a document in the merged
index. Of those 219:

| Excluded | Documents | Why |
|---|---|---|
| Joins | 11 | the edition covers several fragments, the images show one |
| In a training source set | 3 | `Cambridge_CUL_T_S_10J10_10` (PGP edition page in v22), `Cambridge_CUL_T_S_13J13_2` and `Cambridge_CUL_T_S_NS_320_42` (agreed documentary lines) |
| Fewer than 200 visible letters | 10 | after removing the editor's restorations |
| No usable image | 4 | Bodleian whole-volume manifest (2), Library of Congress refuses the request (1), image missing from the store (1) |
| Text too dense for the image | 4 | more than 1,000 edition letters per megapixel (for example 3,074 letters on an 849 x 1,440 photograph): letters a few pixels wide, unreadable for any reader |

The training check reads the manifests of the source sets the mixtures are sampled from
(`pgp_editions_v1`, `pgp_qa_v1`, `pgp_qa_v2`, `documentary_grounding_v1`) and
`decontam/clean_v2_ids.json`, by canonical id and by PGP id. The build stops if a manifest is
missing. None of the 187 documents has a KTIV transcription or is in the PGP-131 benchmark.

The other 95 tier-1 PGP documents have no record in the merged index (their images are reachable
through PGP's IIIF links). They are not in v0.

## 3. Ground truth

PGP editions are scholarly editions, not diplomatic transcriptions.
`arabic_script.clean_edition_line` reduces each line to what the scribe wrote:

- removed: restorations and lacuna marks `[...]` (also when the bracket spans lines), struck text
  `[[...]]` / `⟦...⟧`, letters supplied by the editor `<...>`, every parenthesised span (`(؟)`,
  `(!)`, alternative readings, notes), dot runs, elongation strokes, leading line numbers;
- kept: text the scribe inserted (`\\...\\`, `//...//`) and words the editor marked as superfluous `{...}`.

`split_sections` cuts the edition at "Recto" / "Verso" labels; other labels (margins, address) stay
with their side. When a document has several editions, the one with the most visible Arabic
letters is used.

## 4. Scoring

`score_arabic_benchmark.py`. A document's answer is the text of its images in image order. All
metrics are on Arabic letters after folding (presentation forms, Persian letter variants, vowel
marks, hamza carriers, alef maqsura, ta marbuta), so vocalisation a model adds and orthography an
editor normalised do not count as errors.

| Metric | Meaning |
|---|---|
| 5-gram F1 / precision / recall | clipped letter 5-gram overlap with the edition; order-robust (sides may be read in either order), a repeated phrase counts once. **Headline metric.** |
| LER | letter error rate (edit distance / edition length), best of the two side orders. Above 1 = longer than the text and mostly wrong. |
| floor F1 | the same answer scored against another document's edition (formulae give unrelated texts a small overlap; about 0.01 to 0.02) |
| status | `read`, `loop` (an image's answer repeats itself: more than half of its 12-letter runs are repeats), `wrong_script` (the Arabic text came back in Hebrew letters), `empty` |

**Side coverage.** An edition and the images do not always cover the same sides. An edition of
both sides with one image caps recall; a second image holding Arabic text nobody edited lowers
precision. Comparisons between models are unaffected (same images, same text). Absolute values
are meaningful on the 57 documents where the record shows the two match (one image and a one-side
edition, or two images and an edition with recto and verso); the tables report that subset
separately ("sides-matched").

## 5. Running it

```bash
# decode (LM Studio, one model at a time; resumable; --limit N = the first N documents)
.venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.run_arabic_benchmark \
    --model qwen/qwen3-vl-8b --prompt arabic --limit 60
# score every outputs/*.jsonl; --paired keeps the documents all listed runs answered
.venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.score_arabic_benchmark --paired
# rebuild (images are cached; manifests are fetched again)
PYTHONPATH=. .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.build_arabic_benchmark
```

Protocol: `lms_transcriber.transcribe_with_lm_studio`, 2,500 output tokens, temperature 0.1, model
loaded by LM Studio on the first request (8,192 context), as in the religious benchmark. Two
prompts:

- `arabic`: the fragment prompt with "The text is written in Arabic script". Can the model read
  Arabic script at all? **Primary condition.**
- `standard`: `consensus_gate.build_fragment_prompt` unchanged, which says the text is in Hebrew
  script. What happens to an Arabic page in today's pipelines?
- `trained` (added 2026-10-05): `build_pgp_arabic_editions.ARABIC_FRAGMENT_PROMPT`, the wording of every
  Arabic-script training row (Geniza editions and outside handwriting sets). A model fine-tuned on Arabic
  script is read with this prompt first (the v23a checkpoints: `logs/v23a_eval_chain.sh`); the `arabic`
  prompt is run next to it at a few steps so the fine-tune can be paired with the baselines above.

Outputs from another reader (an API model, Kraken) can be scored by writing the same JSONL layout
(`{"doc_id", "image_index", "text"}`) into `outputs/`.

## 6. Known limits and open items

- **Side coverage** (section 4): only 57 of 187 documents can be shown to have matching images and
  edition. A pass that assigns each edition section to its image (by a strong reader, checked by
  hand on a sample) would make every document usable for absolute scores.
- **Benchmark size against training data.** The benchmark took every usable tier-1 document.
  Cutting it to a prefix (for example the first 100 records, which is an unbiased sample) would
  release about 87 documents to training. Undecided.
- Before the registry existed, three tier-1 documents had already entered v22 source sets (a mixed
  Hebrew and Arabic edition page, two pages with agreed Hebrew lines). They were dropped from the
  benchmark rather than from training. From now on the two builders exclude the registered ids.
- **Base model in LM Studio.** `qwen/qwen3-vl-8b` is a 4-bit quantisation and reads images at
  their native size (65k to 16.7M pixels); the fine-tunes are 8-bit and resize every image to
  6.5 to 7 megapixels. A fair base is the stock weights converted to 8-bit MLX (8.99 bits per
  weight) with the fine-tunes' `preprocessor_config.json`: folder `qwen3-vl-8b-base-8bit-7mp`
  (master on the NAS under `v19b_merge/`, built by `logs/arabic_eval_1002b.sh`), which LM Studio
  serves under the key **`qwen3-vl-8b-base-7mp`** (it drops the quantisation token from the name).
- Image sizes differ by source: store images are 1,440 to 14,176 px on the long side, library
  images 2,000 to 4,000 px (Cambridge serves at most 2,000).
- No API-model baseline yet (Gemini, Claude): needs a go on cost.

## 7. Other readers, scored offline

Answer files of readers that cover only part of the benchmark live in `outputs_other/` and are
scored with `--outputs outputs_other`, so they do not shrink the paired comparison of the local
models.

- **MiDRASH automatic transcriptions** (Zenodo 17734473, Kraken): 43 benchmark documents (88
  images) are NLI-hosted and have a published read, matched by system number and page. 25
  documents came back in Hebrew letters; the 16 answered in Arabic letters have 5-gram F1 median
  0.04 (best 0.39). Not a usable second reader for Arabic script.
  Across the whole dump (948,549 pages, scanned 2026-10-02; per-page script counts in
  `midrash_zenodo/derived/pages.csv` on the NAS): 16,149 pages (1.7 % of pages and of letters,
  4,011 records) are in Arabic script, and they come from a different recognition model than
  `MiDRASH_Gen_01`. Their output has ذ ض ظ, decomposed hamza and Persian letters (چ گ پ ک), and
  `Gen_01`'s alphabet has none of these. The record says `Gen_01` made "the Hebrew script items".
  The model is chosen per image (pages are almost never mixed: 167 of about 800k pages with text),
  so a mixed-script page gets one script. Routing to Arabic is precise but misses a lot: 3 of 1,038
  Judaeo-Arabic-only PGP records got an Arabic page, but 321 of 539 Arabic-only PGP records with
  text got none (Hebrew-letter output instead). Word check against the PGP Arabic vocabulary: 45 %
  of Zenodo Arabic tokens are known words, against 84 % for held-out editions and 18 % for random
  letters; book hands (wisdom literature, Karaite copies at the NLR and BL) are readable, but
  documentary hands are not. On the 16 benchmark documents, F1 on the dotless letter skeleton is
  0.08, against 0.04 with dots and 0.01 for the wrong document.
- **Our Kraken service** (`kraken-linewise` :8002, Kraken 7.0.3, `MiDRASH_Gen_01`, fine-tuned
  segmenter; `outputs_other/kraken_service_midrash_gen01.jsonl`, first 20 documents, 36 images):
  F1 0.0 on every document. It wrote 7,863 Hebrew letters and 52 Arabic letters: Arabic script
  comes out as Hebrew-letter noise, although the model's alphabet holds 38 Arabic characters.
  Independently of the reader, the consensus rule (`line_rule.letters`) compares Hebrew letters
  only, so an Arabic line can never be agreed in the two-reader pipeline as it stands.
- **Blind reading pilot, Claude Opus 5.5** (two subagents, no access to the editions, images capped
  at 2,400 px, free to crop and enlarge; first 20 documents, 36 images;
  `outputs_other/claude_opus_5_5_blind_pilot.jsonl`): 5-gram F1 median 0.27, precision 0.49,
  recall 0.18; 4 documents at F1 0.5 or more (best 0.83 with a letter error rate of 0.10), 12 at
  0.2 or more. On the same 20 documents v21b is at 0.006 and the stock base at 0.001. So the
  images and editions do correspond and the task is readable; a frontier model reads clear hands
  well and transcribes little of the hard chancery cursive, but half of what it writes is right.
  This is an agent with a zoom tool, not an API run on full-resolution images, and it is slow and
  costly (about 20k tokens and 4 minutes per image): fine for a pilot, not for a full run.
  The readers found no Arabic script on 9 of the 36 images (a Hebrew-script text or a blank
  side): a quarter of the benchmark's images are such second sides, where a reader should write
  no Arabic at all.

## 8. Training pages: `pgp_arabic_editions_v1`

`src/finetuning/qwen_hebrew/build_pgp_arabic_editions.py` builds page-transcription rows from
the PGP Arabic editions that no benchmark holds out
(`/Volumes/home/studio_offload/datasets/pgp_arabic_editions_v1`, images-once export next to it).

| | |
|---|---|
| Rows | 220 pages of 209 documents (210 train, 10 val), 58,317 Arabic letters, 1,749 lines |
| Prompt | the training fragment prompt with "handwritten Arabic script" (`ARABIC_FRAGMENT_PROMPT`) |
| Target | visible text of the page, one line per line, `[...]` where the editor restored or marked a loss |
| Images | library IIIF only (Cambridge, JTS, Rylands, Bodleian), because their labels (`1r`, `1v`) say which side an image shows |

A page is written only when the record says which text is on which image: one image and a
one-side edition; or an image label that matches the edition's Recto / Verso label; or PGP's
`side` field for an unlabelled edition. What was left out of the 422 candidate documents:

| Reason | Documents |
|---|---|
| Edition without side labels and two or more images | 113 |
| Fewer than 40 visible Arabic letters | 53 |
| Image labels do not name the edition's side | 22 |
| Mixed script (under 90 % Arabic letters) | 10 |
| Text outside the side labels, whole-volume manifest, request refused | 7 |
| Pages: two documents on one image (6), under 40 letters (7), too dense (2), duplicate record (1) | 16 pages |

The 113 unlabelled editions are the largest recoverable group (the same side-assignment pass as
for the benchmark would recover them). The set is not in any mixture: `build_v22_mixture.py`
needs an `arabic_editions` component and a share, which is a decision for the round after v22b.
38 of the documents are fragments that other training sets already use for their Hebrew-script
text; they are kept out of `val`.

## 9. Results (2026-10-02)

Arabic-script prompt, the first 60 benchmark documents (58 common to all four runs), medians per document.
Per-document CSVs under `<benchmark>/scores/`; tables are also posted to the status page.

| Model (prompt) | 5-gram F1 | Precision | Recall | Floor F1 | F1 >= 0.2 | Loops | Wrong script |
|---|---|---|---|---|---|---|---|
| Blend v22i-a50, documentary pass model (Arabic) | 0.037 | 0.033 | 0.053 | 0.013 | 1 | 29 | 2 |
| v21b-1200 (standard Hebrew-script prompt) | 0.029 | 0.017 | 0.058 | 0.009 | 2 | 33 | 3 |
| v22b step 1200 (Arabic) | 0.020 | 0.011 | 0.058 | 0.011 | 2 | 33 | 1 |
| v20a-1800 (Arabic) | 0.015 | 0.008 | 0.053 | 0.010 | 1 | 38 | 3 |
| v21b-1200 (Arabic) | 0.011 | 0.008 | 0.053 | 0.009 | 3 | 33 | 4 |
| v19a-1300 (Arabic) | 0.010 | 0.006 | 0.048 | 0.008 | 1 | 42 | 3 |
| Base, 8-bit, 6.5-7 MP, fair (Arabic) | 0.007 | 0.004 | 0.012 | 0.002 | 1 | 44 | 0 |
| Base, stock LM Studio 4-bit, native size (Arabic) | 0.003 | 0.002 | 0.006 | 0.002 | 1 | 51 | 0 |
| Base, 8-bit, 6.5-7 MP (standard prompt) | 0.000 | 0.000 | 0.001 | 0.000 | 2 | 44 | 0 |

For scale, on the same first 20 documents (section 7): Claude Opus 5.5 reading blind, F1 0.27 /
precision 0.49 / recall 0.18; our Kraken service 0.0.

What this says:

- **No local model reads Genizah Arabic script.** All four sit at or just above their own
  wrong-document floor; no document reaches F1 0.5; half or more of the documents end in a
  repetition loop (the 2,500-token cap is hit at `finish_reason: length`).
- **The capability was never there to lose.** The stock Qwen3-VL-8B under the fine-tunes' exact
  conditions (8-bit, 6.5-7 megapixels) is as bad as the 4-bit one, so Hebrew fine-tuning did not
  remove an Arabic ability; the blend's small edge (2.5x its floor) is consistent with the few
  Arabic-letter edition rows the v22a side saw. Arabic script has to be trained in.
- **The task is readable**: a frontier model gets clear hands nearly right (best document F1 0.83,
  letter error rate 0.10) and stays precise on the rest, which also validates the ground truth.
- The models know the script they see: v21b's own answers tell an Arabic-script side from a
  Hebrew or blank side on 30 of the 36 pilot images, which can serve side assignment later.

- Across the fine-tune generations the numbers drift up with the amount of Judaeo-Arabic
  documentary data (v19a 0.010, v20a 0.015, v21b 0.011, v22b-1200 0.020, blend 0.037) but stay
  within a few times the floor; the prompt condition changes nothing for any model.

**All 187 documents** (2026-10-03), Arabic-script prompt, the three primary models:

| Model | 5-gram F1 | Precision | Recall | Floor F1 | F1 >= 0.2 | F1 >= 0.5 | Loops | Wrong script |
|---|---|---|---|---|---|---|---|---|
| Blend v22i-a50 | 0.046 | 0.039 | 0.062 | 0.012 | 9 | 0 | 79 | 8 |
| v21b-1200 | 0.026 | 0.021 | 0.066 | 0.010 | 14 | 0 | 93 | 12 |
| Base, 8-bit, 6.5-7 MP | 0.006 | 0.004 | 0.012 | 0.002 | 3 | 0 | 143 | 0 |

Same picture at three times the sample: no document above F1 0.5 for any model, 42 to 76 % of
documents ending in a loop.
