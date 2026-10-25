# Code map: what lives where

One line per area, grouped by job. Module docstrings carry the details; tests mirror modules as
`tests/test_<module>.py`. See `readme.md` for setup and data sources.

## 1. Corpus: merge, normalise, clean

| Where | What it does |
|---|---|
| `src/datasets/merging/merge_shelfmarks.py` | Merges the catalogue sources into one consolidated record per shelfmark (the "merged file" everything downstream reads). |
| `src/datasets/document_models/` | Document classes (`genizah_document`, `bibliography_document`) and the normalisers for shelfmarks, institutions, people, places and scholars, plus the scrub that strips legacy-catalogue URLs from anything served. |
| `src/datasets/merging/*_images.py`, `upload_*_images.py` | Map KTIV and Bodleian scans to stored image objects and upload them. |
| `src/datasets/cleaning/` | Triage and repair of scraped transcriptions; `verify_decontamination` checks training data is disjoint from the benchmarks. |

## 2. Search index and embeddings

| Where | What it does |
|---|---|
| `src/datasets/indexing/index_merged_genizah.py` | Builds the fragment search index from the merged file. |
| `src/datasets/indexing/update_*.py`, `backfill_alt_titles.py` | Update an existing index in place (catalogue fields, generated descriptions, alternate titles). |
| `src/datasets/indexing/verify_reembedded_indexes.py` | Checks a built index against its embedding and text contract (`_meta`). |
| `src/embeddings/text_representation.py` | The one versioned "document → embedding text" function every indexer uses. |
| `src/embeddings/semantic_text.py`, `qwen_text_embedding.py` | `semantic_v3` / `semantic_v4` text rules and the Qwen3 text embedder; `build_golden_fixtures.py` pins them on masked real records (`tests/data/`). |

## 3. Knowledge graph and secondary literature

| Where | What it does |
|---|---|
| `src/datasets/indexing/neo4j/` | Graph imports and enrichment passes (fragments, people, places, citations). New graph work happens in the web-app repo (`genizah_search`). |
| `src/datasets/indexing/bibliography/` | Academic-PDF pipeline: OCR + pass 1 (`fill_initial_stages`), stage ledger, indexing, coverage and availability reports. |
| `src/models/llm/academic/` | LLM passes over the literature: structuring, entity tagging, coreference, relation extraction. |
| `src/models/ocr/book_ocr_service.py`, `script_routing.py` | Book OCR, choosing per page between a PDF's text layer and Cloud Vision. |

## 4. Reading manuscripts (production readers)

| Where | What it does |
|---|---|
| `src/models/ocr/page_reader.py` | The production whole-page VLM read: script-neutral prompt, 1,200-token budget, loop guard with one retry, ink-based blank-page gate. |
| `src/models/ocr/lms_transcriber.py` | LM Studio client (`transcribe_with_lm_studio[_meta]` returns the text plus finish reason and token counts). |
| `src/models/ocr/lms_yield_gate.py` | Makes batch jobs pause while a production chat turn is running; every LM Studio consumer goes through it. |
| `src/models/ocr/kraken_transcriber.py`, `src/services/kraken_microservice/` | Kraken line segmentation + recognition, served as a small HTTP service (`k7/` is the Kraken 7 build). |
| `src/finetuning/qwen_hebrew/prompts.py` | Every reader and training prompt, in one place. |

## 5. Consensus: two readers and a gate

`src/datasets/consensus/`: Kraken lines + grounded VLM lines per page (`two_reader_lines`, `line_rule`), the
confidence gate (`consensus_gate`), the corpus runner, re-reads of failed pages (`read_health`, `reread`,
`retry_escalates`), the decode policy (column crops / rescue halves / whole page), character-level ROVER over cached
reads (`rover_reads`), per-page reading sidecars for the site's reader-comparison view (`build_readings`), and
catalogue descriptions generated from reads (`describe_from_reads`).

## 6. Benchmarks and evaluation

| Where | What it does |
|---|---|
| `src/datasets/evaluations/metrics.py`, `fragment_evals.py`, `talmud_evaluation.py` | Shared metrics (CER, n-gram precision, failure flags) and the main eval pipelines. |
| `src/datasets/evaluations/benchmark_registry.py` | Every held-out benchmark document, so dataset builders can exclude them. |
| `helper_eval_scripts/build_*_benchmark.py` | Build the benchmarks: PGP-131, religious, Arabic-script, and `sealed_v1` (never trained or tuned on). |
| `helper_eval_scripts/run_*.py` | Decode a benchmark with a model (LM Studio, Kraken, API models). |
| `helper_eval_scripts/score_*.py` | Offline scoring of saved outputs; `score_genizah_offline.py` is the source of truth for Genizah results. |
| `helper_eval_scripts/arabic_bench_coverage.py`, `score_arabic_covered.py` | Which edition side each Arabic benchmark image carries, and scores on covered images only. |
| `helper_eval_scripts/*prompt_ablation*.py`, `run_reader_dryrun.py`, `loop_guard_outputs.py` | The reader prompt / output-budget ablation and the production-reader dry run. |
| `helper_eval_scripts/rover_consensus.py`, `slot_oracle.py`, `unreachable_profile.py` | Output-space consensus and its ceilings. |
| `helper_eval_scripts/ngram_audit/` | Audit of the paper's n-gram metric. |
| `grounding_eval/` | Box-quality and line-agreement evals for grounded reads. |

## 7. Training the VLM reader (Qwen3-VL)

| Where | What it does |
|---|---|
| `src/finetuning/qwen_hebrew/build_*.py` | Dataset builders, one per source or task (KTIV, PGP editions, Arabic editions, public Arabic sets, grounding, QA, page parse). |
| `build_v22_mixture.py`, `subset_mixture.py`, `images_once.py` | Assemble a training mixture as one images-once dataset; derive ablation subsets. |
| `colab/derive_v*.py` → `colab/genizah_v*.ipynb` | Each run's notebooks are generated by its `derive_*.py` from the previous run's; regenerate them, don't hand-edit (the `test_colab_notebook_*` tests check them). `host_lambda.py` makes the Lambda Cloud edition. |
| `side_match.py` | Matches PGP edition sides to library images by reading them. |
| `eval_harness/` | Checkpoint evals, run watchers and alarms, merges and weight interpolation. |
| `train_session.py`, `export_lmstudio.py`, `verify_vision_lora.py` | Session-safe training helpers, export to LM Studio, vision-LoRA sanity check. |

## 8. Training Kraken

`src/finetuning/kraken/`:
- **Genizah Hebrew:** export KTIV lines and page XML for recognition and segmenter training (`export_ktiv_*`,
  `seg_*`, `filter_manifests`).
- **Arabic-script:** line sets from public Arabic HTR datasets (`build_arabic_lines`, `fetch_page_images`,
  `build_arabic_seg_set`), the pre-training gate and fine-tune driver (`gate_lines`, `ketos_frozen_bn`,
  `finetune_arabic.sh`, `Dockerfile.k711`).
- **Forced alignment:** human edition lines placed onto segmenter lines to make line ground truth (`line_extent`,
  `page_layout`, `align_edition_lines`, `filter_aligned_lines`, `segment_edition_pages`).

## 9. Text model (Qwen 27B: translation and catalogue)

`src/finetuning/text27b/`: the multi-task text dataset (translation, catalogue JSON, document analysis, literature
reading, RAG rows), its hold-outs and contamination gates, eval sets and scorer. `colab/derive_t27.py` generates the
t27 notebooks. `paths.py` reads machine-local configuration.

## 10. Legacy

`src/finetuning/donut_hebrew_model/`, `legacy_gemma/`, `clip_hebrew/`, `vqa_finetune/`: earlier model families, kept
for reference.
