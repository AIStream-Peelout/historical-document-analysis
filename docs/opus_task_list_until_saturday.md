# Task list for Opus 4.8 sessions until the Saturday usage reset

Written 2026-09-09 by Fable after reviewing Opus's v2.1 builder work. The
review found the geometry and wiring correct and three misses, all of the
same kind: consequences that are not visible in the code being edited (the
hub/Colab size effect of per-row image embedding; a fragment passing a
substring check while not being a whole word; awkward prompt grammar). So the
tasks below are chosen to be **self-contained, mechanically verifiable, and
reversible**, each with a definition of done and a check to run. Anything
that needs a judgment call about training, evaluation interpretation, prod,
or public text waits for Saturday.

## Standing rules for every task (paste with the task)
- The Mac Studio is shared production: never stop/restart/prune containers,
  never eject LM Studio models you did not load, never kill cloudflared.
  Check `df -h`, `~/.lmstudio/bin/lms ps`, `docker ps` before any long job;
  keep ≥ 15 GB free on the data volume; big outputs go to
  `/Volumes/home/studio_offload/`.
- Never move, rename or delete anything under `src/datasets/raw_data/`.
- Never name Friedberg/FJMS/FJP on anything public (model cards, READMEs).
- Do not run `--apply` against Elasticsearch, do not commit or push git, do
  not delete files outside the scratchpad, do not modify `line_rule.py`'s
  rule, do not launch Colab training. Propose; the user executes.
- "Done" means: tests written and passing, the verification command below run
  and its output pasted in the final message, and any decision point flagged
  rather than resolved.

## Green: do these (each fully specified)

### 1. v2.0b model card and series-table rows
Write `README.md` for `isaacmg/qwen3-vl-8b-hebrew-v20b-ckpt` in the style of
the v20a card (`scratchpad/v20a_card.md` in the 09-05 session, or copy the
v20a repo's README via `hf_hub_download`). Facts: final step 2000, commit
`4cae9d52d9cd167cb0d1ddd0248ed728c7308f29`, eval loss 0.6684, merger LoRA r16,
warm start v1.9a-1300; grounding trio and box geometry numbers are in
`src/finetuning/qwen_hebrew/eval_harness/final_eval_v20b.log`; the
transcription hard eval did not run (say so). Verdict text: "the merger LoRA
did not change the training objective (loss identical to v2.0a to three
decimals) and did not improve box geometry; not adopted." Then add a
`| **v2.0b** |` row to the series table of the v19a, v19b, v19c and v20a
cards (check with the literal row string before patching, as the 09-05
session learned). Verify: `grep -c "v2.0b" README.md` on each card after
download; run the public-attribution grep from the standing rule on every card (must be empty).

### 2. Word-locate and indexed-line eval queries
Extend `src/datasets/evaluations/grounding_eval/grounding_eval.py --build`
with two new modes over the same 24 pages: `locate_word` (targets from
`build_ktiv_dataset.word_candidates`, 3 per page) and `line_index` (targets
from `build_ktiv_dataset._eligible_columns`, 2 per page; prompt string copied
byte-for-byte from `_LINE_INDEX_PROMPT`). Write them to a NEW file
`grounding_eval_v21.jsonl` (never rewrite `grounding_eval.jsonl`). Add
`--run locate_word` (IoU + center hit, like locate) and `--run line_index`
(box IoU vs the target line + CER of the text field). Tests in
`tests/test_grounding_eval_v21.py` with synthetic pages. Then run both modes
on `qwen3-vl-8b-heb-v20a-step1800` (it is loaded in LM Studio; ~25 min) so
v2.1 has a baseline. Verify: the two `*_results.json` files exist and the
printed medians are in the final message.

### 3. Box-geometry metrics in the eval chain
Add a `box_quality.py --preds <model>` call to `final_eval_v20a.sh`'s
successor template (`final_eval_v20b.sh` is the newest) right after the
grounding trio, and log its dict to the W&B run `<ver>-hard-evals` via
`compare_series.py`'s existing wandb block (one `wandb.log({"box/...": v})`).
Do not change any threshold or metric definition. Verify: run
`box_quality.py --preds qwen3-vl-8b-heb-v20a-step1800` and show the dict;
run the shell script with `bash -n` for syntax.

### 4. Rebuild the grounding viewer with v2.0b-2000
`preds/grounded_qwen3-vl-8b-heb-v20b-step2000.json` (and locate/read_box)
already exist from the final eval. Run
`.venv/bin/python src/datasets/evaluations/grounding_eval/viewer/build_viewer.py`
(and the `--max-side 1300` hosted build to the scratchpad). Verify: the
page loads at `http://localhost:8765/grounding_viewer.html` with three model
checkboxes; send the hosted file with SendUserFile. Do not publish an Artifact
(blocked by policy in this project; deliver the file).

### 5. Two-reader pipeline: the 1,000-letter slice (only after the user says go)
`nohup .venv/bin/python -m src.datasets.consensus.two_reader_lines --ids
src/datasets/raw_data/cairo_genizah/ai_reads/jobs_letters_1000_v5.jsonl
--min-free-gb 8 > .../ai_reads/run_letters_1000.log 2>&1 &` at night, with
LM Studio showing no other user of `qwen3-vl-8b-heb-v20a-step1800`. When it
ends: loader dry run
(`cd ~/Documents/GitHub/genizah_search && PYTHONPATH=. .venv/bin/python
scripts/load_ai_transcriptions.py --input <file>`), then a summary table:
records, parsed, lines, agreed, surfaced, and agreed-share by document type
(join `jobs_letters_1000_v5.jsonl`'s `document_type`). Never `--apply`.

### 6. KTIV catalog metadata pull (v2.2 prerequisite)
For every manuscript in the v3 training build (sys_num from the image dir
`/Volumes/home/studio_offload/datasets/ktiv_dataset_v3_images/<sys_num>/`),
fetch the NLI IIIF manifest
(`https://iiif.nli.org.il/IIIFv21/DOCID/PNX_MANUSCRIPTS<sys_num>-1/manifest`)
and write `src/datasets/indexing/merged/ktiv_catalog_metadata.csv` with
`sys_num,title,language,date,subject,shelfmark` parsed from the manifest's
`metadata` labels (rate limit 2 req/s, resumable, 3 retries, log failures).
Test the parser on one saved manifest. Verify: row count, share with a
non-empty language field, top 10 titles.

### 7. Sefaria canonical text cache (v2.3 prerequisite, mechanical part only)
From the CSV of task 6, map titles that name a Talmud tractate or a Mishneh
Torah book to Sefaria refs, and fetch the canonical Hebrew text via
`https://www.sefaria.org/api/texts/<ref>?context=0&commentary=0` into
`src/datasets/raw_data/cairo_genizah/canonical/sefaria/<ref>.json` (create
this new subfolder; nothing else under raw_data changes). Whole tractates
only (all dapim), rate-limited, resumable. No alignment, no scoring. Verify:
number of refs fetched, total characters, one sample ref printed.

### 8. Disk hygiene report (report only)
List, with sizes: `~/.cache/huggingface/hub/*`, `~/.cache/huggingface/datasets`,
`models/*`, `~/.lmstudio/models/isaacmg/*`, `/Volumes/home/studio_offload/v19b_merge/*`,
and mark each as flagship (keep), NAS-backed (deletable locally), or
regenerable. Deliver as a table. Delete nothing.

### 9. Docker restart policies and the tunnel service (write, do not apply)
In the site repo, draft the `docker-compose.yml` diff adding
`restart: unless-stopped` to every `genizah_search-*` service and a
`com.cloudflare.cloudflared.plist` LaunchDaemon (or the
`cloudflared service install` command) for tunnel
`adc9e5c6-54fc-4476-84db-806b4c61f6dc`. Deliver the diff and the exact
commands; applying restarts prod, so the user runs them.

### 10. Camera-ready mechanical LaTeX (paper repo, branch `camera-ready-fills`)
Add `\FloatBarrier` (placeins) before `\section*{Limitations}` so Tables 4/5
no longer float into it; confirm Limitations contains no results sentences
(move any to Results); recompile; report page count and where References
start. Do not touch the title or any prose beyond moving sentences.

## Red: not until Saturday (needs judgment or is irreversible)
- The v2.1 notebook (`genizah_v21.ipynb`): mixture weights, the streaming
  loader change, warm-start pin. Fable writes it once `genizah_ktiv_v3` has a
  hub revision.
- Any change to `src/datasets/consensus/line_rule.py` or the loader contract.
- Interpreting v2.1 results into promote/hold decisions; flagship promotion.
- Gemini Pro tagging pilot design and prompts (v2.2).
- Paper title choice, author-response wording, any public prose.
- `--apply` to Elasticsearch, deletions, commits/pushes, LM Studio ejects,
  Colab launches.
