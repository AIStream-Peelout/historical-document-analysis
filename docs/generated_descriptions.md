# Generated catalogue descriptions for vector-less records

**Purpose.** 14,461 records of `genizah_merged_v9` have `semantic_eligible: false` and no vector, because their catalogue text is empty, short or a boilerplate label (`docs/v9_index_embedding.md`). The consensus pipeline (`src/datasets/consensus/two_reader_lines.py`) reads their pages. `src/datasets/consensus/describe_from_reads.py` has a text-only model (`qwen/qwen3.8-27b` on LM Studio) turn one document's reads (the first 4 pages) into a short catalogue description in the PGP register. `src/datasets/indexing/update_generated_descriptions.py` embeds that description with the v9 embedder, through the indexer's own `semantic_v3` text path, and writes it into the index.

**Principles.**
- The generator never sees an image. The prompt holds known catalogue metadata (shelf mark, institution, PGP type and languages, KTIV title, Bodleian notes; links are dropped) and the transcription, with each line marked `[agreed]` (both readers agree) or `[unverified]`.
- Every person, place, date, commodity and institution in the reply must occur verbatim in that transcription. Before comparison, both sides are lightly normalised: marks dropped, quotes unified, punctuation turned into spaces, Hebrew final letters folded. Spans that fail are dropped and recorded, and the confidence is capped at 0.5. Original-script quotes in the summary that are absent are flagged.
- Outputs are labelled machine-derived. The embedded text ends its description with `[machine-generated from automatic transcription]`, and rows and index documents carry `description_source: generated_from_consensus` with full provenance.
- Generated text is never a training target for any reader.
- An identified work stays `unverified` ("Possible identification (unverified): ...") unless a symbolic check confirms it. No such check is wired in yet.

**Schema.** The model returns `genre`, `document_type`, `language[]`, `script`, `summary` (2–4 sentences), `identified_work` (`{title, section, confidence}` or null), `entities` (`people`, `places`, `dates`, `commodities`, `institutions`: verbatim spans), `themes`, `confidence` and `notes`. This is validated against `OUTPUT_SCHEMA`, with one repair turn.

Each `--out` row adds provenance: `doc_id`, `n_pages`, `n_lines`, `agreed_lines_pct`, `readers`, `source_files`, `pages`, `model`, `prompt_version`, `reasoning`, `generated_at` (UTC), the `grounding` report, `description_text`, `embed_eligible`/`embed_reason` and `description_source`.

The index gets `embedding_vector`, `semantic_eligible: true`, `text_representation_version`, `description_source`, `generated_description` and `generated_description_meta`.

**CLIs.**
```bash
.venv/bin/python -m src.datasets.consensus.describe_from_reads --reads <ai_reads*.jsonl ...> \
    --pool logs/next_round/consensus_pool/jobs_vectorless_pool.jsonl --catalogue <merged_shelfmarks.jsonl> \
    --out logs/next_round/consensus_pool/generated_descriptions.jsonl [--dry-run] [--reasoning off|on] [--limit N]
.venv/bin/python -m src.datasets.indexing.update_generated_descriptions --generated <out> [--dry-run] [--index ...]
```
`--dry-run` on the generator writes only `<out stem>.prompts.jsonl`. A real run is sequential and resumable, and logs failures to `<out stem>.failures.jsonl`. Run it only when no other LM Studio consumer is active.

The updater's `--dry-run` checks the guards and prints its decision for each document without loading the embedder or writing anything. It refuses:
- rows with `embed_eligible: false`;
- documents that already have a catalogue vector;
- any index whose `_meta` (model, revision, text version) or canary does not match the embedder.

Rebuilding the index drops these vectors, so re-run the updater afterwards (it is idempotent). The verifier's self-consistency check must re-embed `generated_description` for these documents (`embedded_text_from_source`).

**Review loop.** Agreed lines are the citable layer. The site shows the description labelled as machine-derived, alongside its provenance. Reviewers sample rows by `embed_reason`, dropped spans, capped confidence and `agreed_lines_pct`. A confirmed or corrected description becomes catalogue text. Any prompt change bumps `PROMPT_VERSION`. Regenerated rows replace older ones because their `generated_at` is newer. Withdrawing a rejected description from the index is not automated yet.

## Voted reads (character-level ROVER)

**What it is.** `src/datasets/consensus/rover_reads.py` merges the reads of one page by several VLM checkpoints, plus Kraken's fragments, into one read. It uses the alignment and vote of `rover_consensus.py` in the configuration that scored best offline: character level, unweighted, unfiltered, with Kraken's text as one more voter. It writes one record per page in the pipeline's own schema, so the description stage reads it unchanged.

The pilot's three-voter setup was also checked offline on PGP-131 (mean / median CER):

| voters | mean | median |
|---|--:|--:|
| v21b-1200 + v23a-1500 + Kraken | 0.191 | 0.142 |
| blend + v23a-1500 + Kraken | 0.187 | 0.140 |
| best single reader (v23a-1500) | 0.214 | 0.155 |

On religious-140, two VLMs plus Kraken improve the mean but only tie the best single median. Without Kraken, a two-VLM vote just copies the first reader.

**Feeding the description stage.** Pass the voted file to `describe_from_reads --reads`, either alone or with the single-reader files plus `--prefer-reader rover3 rover2`. Without that preference, `page_rank` keeps whichever read of a page has the most agreed lines, which may be a single read.
- A page with one usable VLM read passes through unchanged (`rover.voted: false`; model `rover2` with Kraken, `rover1` without).
- `[...]` and other bracket marks without letters are voted as one token and written back, so the grounding check still refuses spans across a gap.
- Line `status` is re-derived with the pipeline's own rule (`line_rule`: geometric fragment assignment, letters-only similarity ≥ 0.8). Boxes come from the page's geometry reader: the VLM whose own lines pair best with Kraken's fragments.
- With Kraken as a voter, the vote leans towards Kraken where the VLMs disagree, so `[agreed]` is slightly optimistic there. `kraken_only` counts those characters.

**Provenance fields.**
- `ai_read`: `vlm_model` is `rover<N>` (N voters, Kraken included). `vlm_revision` is `<reader>=<revision>; …; kraken=<htr cache key>`. `htr_model` and `htr_cache_key` are carried over, and `rule_version` is the current line rule.
- Top-level `rover` block: `version`, `voted` / `reason`, `unit`, `voters`, `backbone`, `geometry_reader` / `geometry_scores`, `skipped` (reads without lines), `job`, plus:
  - `sources`: each single-reader record as `<reads file>:<line>`, with its revision, decode time, `parsed`, line and agreed counts, and box source;
  - `kraken`: the record whose fragments were used, their source, key and count;
  - page counts: `n_hybrid_lines`, `n_words`, `n_non_input_words`, `n_kraken_only`, `n_loop_duplicates`.
- `rover.lines[i]`, aligned with `ai_read.lines`:
  - `attested_by`: characters each reader backed;
  - `n_tokens`, `kraken_only`;
  - `full`: readers who wrote the whole line; `hybrid`: nobody did;
  - `non_input_words`: words no reader wrote anywhere on the page;
  - `source_lines`, `box_from`.

Description rows already record `readers` and `pages` with `vlm_model` and `vlm_revision`, so a description built from voted reads traces back to `rover.sources`. A grounded span can still land on a non-input word, i.e. text no single reader wrote. The grounding check does not test for this yet, so reviewers should compare spans against `non_input_words`.

**CLI.**
```bash
A=src/datasets/raw_data/cairo_genizah/ai_reads; P=logs/next_round/consensus_pool
.venv/bin/python -m src.datasets.consensus.rover_reads --reads $A/<pass A>.jsonl $A/<pass B>.jsonl $A/<pass C>.jsonl \
    --jobs $P/jobs_vectorless_pilot200_cap4_all.jsonl --out $P/ai_reads_rover_vectorless_pilot200.jsonl [--dry-run]
```
`--dry-run` writes nothing. It reports:
- pages and readers per page;
- page counts per voter combination;
- Kraken coverage;
- agreed lines per reader against the same fragments;
- a re-derivation check: single reads re-matched from the raw cache must reproduce their own records.

The full rules are in the module docstring.

## Comparing description runs

`src/datasets/consensus/compare_descriptions.py` joins two or more labelled sets of generator rows on `doc_id` and writes one markdown report. On the pilot, `logs/next_round/consensus_pool/run_post_pilot.sh` runs it as `single` (passes A and B, one reader per page) against `voted` (the same documents described from the ROVER reads).

```bash
P=logs/next_round/consensus_pool
.venv/bin/python -m src.datasets.consensus.compare_descriptions \
    single=$P/generated_descriptions_pilot_passA.jsonl+$P/generated_descriptions_pilot_passB.jsonl \
    voted=$P/generated_descriptions_pilot_rover.jsonl --out $P/pilot_description_comparison.md [--dry-run]
```

**How to read it.** Start with the inputs. Documents present in one label only are listed and left out of the paired figures, and a warning appears when the generator, prompt version or reasoning mode differ, since the reads are then not the only change. The per-label tables describe each run on its own. The paired sections answer whether the reads changed the descriptions, with Δ always the later label minus the earlier. Embedding-decision flips matter most, because those documents would gain or lose a vector. Confidences sit on a few round values, so the median Δ and its bootstrap interval are often exactly 0 even when many documents moved: read the mean Δ and the higher / equal / lower counts with it. An interval that excludes 0 shows a consistent shift on these documents, not which run is right. The two labels describe different transcriptions, so a summary similarity well below 1 is normal rewording. The signal is in the low tail, especially with a genre change, and the divergence sample puts those summaries side by side for review. With Kraken as a voter, voted reads mark `[agreed]` slightly optimistically (see above) and the generator weights agreed lines, so part of a higher agreed share or confidence under `voted` can come from that. Plumbing check: a file compared with itself must show identical columns, zero deltas and no divergent document.
