# Readings sidecars (`readings-v1`)

One JSON file per page that lists **every reader's lines** for the website's advanced reader-comparison view: the
whole-page VLM read, each decode-policy unit read (column crops, rescue halves, the halves of a failed column), Kraken's
fragments, the policy trace and the final merged lines. Built by
`src/datasets/consensus/build_readings.py` from files the consensus pipeline already writes. It is a reshaping: no model,
Kraken, Elasticsearch or network call, and it never writes to the records or to the raw cache.

```
src/datasets/raw_data/cairo_genizah/ai_reads/
  ai_reads_<model>.jsonl            records (input, read only)
  raw/<stem>.json                   raw cache (input, read only)
  readings/<stem>.json              one sidecar per page (output)
  readings/manifest.jsonl           one line per sidecar written (output, append only; not a *.json file)
```

`<stem>` is the raw-cache stem: `re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id) + f"__{image_index}__{vlm_model}"`. A page read
by two models has two sidecars. Entries of the first pilot runs lack the `__{vlm_model}` part; the builder finds them the way
`two_reader_lines.raw_cache_path` does and names the sidecar after the raw file it used (none of the current records needs it).

## Schema

All boxes are `[x1, y1, x2, y2]` **ints in 0-1000 of the oriented page** (the frame of `image_width` x `image_height`),
corners ordered, clamped to the page. Pixel crop boxes are in the same oriented page's pixels.

```jsonc
{
  "schema_version": "readings-v1",
  "doc_id": "...", "image_index": 1, "image_url": "...", "image_sha256": "...",
  "image_width": 4017, "image_height": 5350,
  "source_index": "genizah_merged_v8",            // from the record
  "decoded_at": "2026-10-12T19:21:05+00:00",      // the record's ai_read.decoded_at (the build is current while this holds)
  "rule_version": "lines-v3-20260917",
  "vlm_model": "qwen3-vl-8b-heb-v22i-a50", "vlm_revision": "interp-v21b1200-v22a1300",
  "policy": null | { ... },                       // null = read without the decode policy ("legacy" page)
  "readers": [ { ... }, ... ],
  "final": { "source": "whole_page|assembled|assembled_partial|legacy", "lines": [ ... ] },
  "provenance": { "raw_cache": "raw/<stem>.json", "built_at": "...", "builder": "build_readings 1.0.0",
                  "reply_mapping": "single|by_request|partial", "notes": [] }
}
```

### `readers[]`

Order: the whole-page read (when the page was read whole), then the unit reads in request order, then Kraken.

VLM readers (`kind: "vlm"`):

| field | meaning |
|---|---|
| `id` | `vlm:<model>:page` for the whole-page read, `vlm:<model>:<unit>:crop` for a unit read (`col0`, `col1`, `col0.top`, `top`, ...) |
| `scope` / `unit` / `role` | `page` / null / `page`, or `unit` / the unit name / `column` or `half` |
| `reply_index` | position of the reply in the raw entry's `vlm_raw` (`vlm_raw[reply_index]` when it is a list, `vlm_raw` itself when it is a string, so 0). null = no reply text to point at (see the mapping rules) |
| `parsed` | the reply parsed as a grounded JSON array (`parse_grounded`) |
| `health_label` | the policy's verdict on this read: `ok`, `loop`, `capped`, `near_empty`, `skipped_text`, `overgen`, `parse_loss`. null for pages read without the policy (no verdict was recorded) |
| `crop_box_px`, `crop_box_1000` | the crop the model saw, in pixels and in the page frame (both null for the page read) |
| `lines` | `[{index, text, box}]`, `index` = position in that reply (before any de-duplication); boxes mapped to the page frame. null only when the reply could not be attributed |
| `note` | present when `reply_index` / `lines` are null: why |

Kraken reader (`kind: "htr"`): `id` `kraken:<htr_cache_key or htr_model>`, `scope` `page`, `unit` null, `htr_model`
(recognition model), `cache_key` (the service key of the fragments used, e.g. `MiDRASH_Gen_01@k7.0.3-ktivseg-e10-mv`), and
`lines: [{index, text, box, conf}]` = the cached fragments of the record's key (`index` = position in the fragment list;
`conf` may be null; an empty list means Kraken found no line).

### `policy` (null for legacy pages)

`version`, `action` (`keep` | `rescue` | `columns`), `reason` (`whole_page_ok`, `whole_page_loop`, `2_columns`,
`whole_page_loop_kraken_blank`, ...), `final_source` (`whole_page` | `assembled` | `assembled_partial`), `calls` (model
requests made), `fallback` (null, or why the page fell back to the whole-page read: `unit_unresolved`,
`all_units_unresolved`, `assembled_shorter_than_whole_page`), `mode` (`live`, or `revisit`: the whole-page read was cached
and only the units were read), `decided_at`, and `units`:

```jsonc
{ "name": "col1", "role": "column", "crop_box_px": [...], "crop_box_1000": [...],
  "resolved_by": "crop|halves|share|unresolved", "resolved": true,
  "health_label": "loop",                 // verdict on the reading that resolved the unit, else on the last one tried
  "attempts": [                           // in the order tried: crop, then halves (columns only), then share
    {"source": "crop",   "label": "loop", "flags": ["loop"], "letters": 175, "n_lines": 30, "reply_index": 3},
    {"source": "halves", "label": "loop", "flags": [...], "letters": 1822, "n_lines": 98, "reply_index": null,
     "reply_indices": [4, 5], "note": "joined from the replies in reply_indices (top half, bottom half)"},
    {"source": "share",  "label": "ok",   "flags": [], "letters": 540, "n_lines": 22, "reply_index": 6,
     "note": "lines of the whole-page read (reply_index) whose box centre lies in the unit's crop"} ] }
```

A `keep` page has `units: []`. A `rescue` page has two `half` units (`top`, `bottom`, 12 % overlap). A `columns` page has one
`column` unit per Kraken column, rightmost first (`col0` is the right column). On an assembled page a unit contributes
the lines of the attempt named by `resolved_by`; an `unresolved` unit contributes its best effort (halves, else crop,
else share) when the page is `assembled_partial`, and nothing when the page falls back to the whole-page read. The other
attempts are the evidence for why.

### `final`

`lines` is the record's `ai_read.lines` **verbatim** (index, text, evidence `bbox`, `agreement`, `status`, `htr_text`,
`htr_fragments`): the lines the website shows. They are not a copy of one reader: loop duplicates are dropped, and for
an assembled page the text of each line comes from a unit reader (the longer reading when two halves read the same line).
`source` is the policy's `final_source`, or `legacy` for a page read without it.

### `provenance`

`raw_cache` (path of the raw entry relative to `ai_reads/`), `built_at`, `builder` (`build_readings <version>`; a new
version makes every sidecar stale, see below), `reply_mapping` (`single`: a legacy page, one reply and one page reader;
`by_request`: every reply of a policy page verified against its request; `partial`: see the mapping rules) and `notes`
(empty unless something is missing or ambiguous).

## How replies are mapped to units, and where that stops

A page read under `--decode-policy` stores `vlm_raw` as one string when it took a single request (a `keep` page) and
otherwise as a list **parallel to `decode_policy.requests`** (`PolicyResult.vlm_raw`): reply `i` answers request `i`, and
the request names its `unit`, `role` and pixel `crop`. The mapping uses the unit name, never the position alone:

* the whole-page reply is first for `keep` and `rescue` pages (and for a revisit, where it is the cached read), but
  **last** for a column page that needed the whole page for a unit's share or the page fallback (6 of 17 column pages in
  the 2026-10-13 data, e.g. `col0, col1, col1.top, col1.bottom, page`); the other 11 column pages have no whole-page read
  and therefore no page reader;
* the halves of a failed column are replies of their own (`col0.top`, `col0.bottom`): readers with `role: half`;
* a `share` attempt has no reply of its own (it is a slice of the whole-page reply, `reply_index` points at that reply);
  a `halves` attempt joins two replies (`reply_indices`).

Each attribution is **verified** against the request before it is used: reply length (`reply_chars`), `parsed` and
`n_lines` must match the trace (the cached whole-page read of a revisit is checked by length only). All 3,028 replies of
the 2,204 policy pages present at the first check matched. After the backfill a one-off script (not in the repo) compared
all 39,220 sidecars with their records and raw entries independently of the builder: reader and request counts, reply
indices, line counts, crops, `final.lines` verbatim, Kraken texts, and that every final line's text is found in the
readers it should come from. It found no disagreement.

When the list and the trace disagree (different lengths, repeated unit names, or a reply that does not match its
request) the mapping is **partial**: only a verified leading whole-page reply is attributed, every other VLM reader keeps
its trace facts (`crop_box_*`, `health_label`) but has `reply_index: null`, `lines: null` and a `note`, and
`provenance.reply_mapping` is `partial` with an explanation in `provenance.notes`. Nothing is guessed. No page in the
current data is partial.

Where the lines of a reader come from:

* **crop and policy whole-page replies**: `two_reader_lines.parse_grounded(reply)`; crop boxes go through
  `reread.to_page_box` (crop 0-1000 frame to page 0-1000 frame), then are rounded and clamped to ints (error at most
  0.5 on the 0-1000 scale; a model box that runs past the page edge is clamped);
* **the whole-page read of a legacy page** (and the cached whole-page read of a revisit): the parse **stored with the
  entry** (`vlm_lines`, or `decode_policy.cached_whole_page.vlm_lines`), because that is what the final lines were built
  from. Today's `parse_grounded` finds more lines than the stored parse on 280 of the 37,000 legacy records (+4,155
  lines; 42 of them have zero stored lines), because the parser improved after those pages were read. Showing the new
  parse would make the page reader disagree with `final`; re-parsing those records is a separate decision (a
  `--rematch`-style pass, no inference);
* **Kraken**: `frags` for the record's key (`ai_read.htr_cache_key`, else `htr_model`).

Other limits:

* the sidecar holds parsed lines, not reply text: use `reply_index` into `raw/<stem>.json` for the text of a reply;
* the lines of a `halves` or `share` attempt are not stored apart: they are the halves' readers joined / the page
  reader's lines whose centre lies in `crop_box_px`;
* legacy pages have no `health_label` (the pipeline did not record one before the decode policy);
* policy coordinates assume the pipeline's frame (Kraken view rotation 0, the oriented page);
* a record is only built when its raw entry is **its own**: same image URL, model, hash, size, policy state
  (the entry's trace and the record's `decode_policy` stamp agree) and Kraken key; otherwise the page is counted and
  skipped. In the data this skips 2 records: `Cambridge_CUL_T_S_NS_J24B` image 0 (both models), whose raw file name is
  shared on a case-insensitive disk with `Cambridge_CUL_T_S_NS_J24b` image 0, so the file holds the other document's read;
* a page recorded twice (two records of one page, e.g. two runs seconds apart) is built from the later record, once.

## Building

Repo root, CPU only (a full pass over 39,224 records takes about 25 s):

```bash
.venv/bin/python -m src.datasets.consensus.build_readings \
  --records src/datasets/raw_data/cairo_genizah/ai_reads/ai_reads_qwen3-vl-8b-heb-v22i-a50.jsonl \
            src/datasets/raw_data/cairo_genizah/ai_reads/ai_reads_qwen3-vl-8b-heb-v21b-step1200.jsonl
```

Options: `--raw-dir` (default `ai_reads/raw`), `--out-dir` (default `ai_reads/readings`), `--since ISO` (only records whose
`decoded_at` is at or after), `--limit N` (stop after N sidecars written), `--force` (rewrite current sidecars),
`--manifest PATH` (default `ai_reads/readings/manifest.jsonl`). The run prints a
`summary:` JSON line: `records`, `built` (`built_new` + `rebuilt`), `built_policy` / `built_legacy`, `policy_<action>`,
`readers_vlm` / `readers_htr`, `ambiguous` (partial mappings), `up_to_date`, `before_since`, `raw_missing`,
`raw_unreadable`, `raw_mismatch` (+ `raw_mismatch:<reason>`), `partial_tail`, `bytes_written`, `elapsed_s`.

A sidecar is **current**, and skipped, when it is this page's (same `doc_id`, `image_index`, `vlm_model`), has the same
schema and builder version, and was built from this record or a newer one (`decoded_at`). So:

* a second run with no new pages writes nothing and no manifest line;
* a page whose record was rewritten (`--revisit` turning a legacy page into a policy page, `--rematch`, `--rekraken`) has a
  newer `decoded_at` and is rebuilt on the next run;
* bumping `BUILDER_VERSION` after changing the builder rebuilds everything once; `--force` does so on demand;
* the records file is read as the pipeline appends to it: a last line without its newline is not a record yet
  (`partial_tail`); sidecars are written to a temp file and renamed, so a reader never sees half a file; the manifest line
  is appended after the sidecar. The pipeline writes a page's raw entry before its record, so a visible record always
  has its raw entry.

Incremental loop (every 30 minutes; the overlap costs only a skipped-file check):

```bash
SINCE=$(date -u -v-2H +%Y-%m-%dT%H:%M:%SZ)
.venv/bin/python -m src.datasets.consensus.build_readings --records <both files> --since "$SINCE"
```

`readings/manifest.jsonl` has one line per sidecar written, `{stem, doc_id, image_index, built_at, readers, policy_action}`;
a rebuilt page appears again (the last line wins), so a sync job can follow it by line count.

## Backfill of 2026-10-13 (v22i-a50 documentary file and v21b-step1200 literary file)

As of the backfill (the live pipeline keeps appending pages): 39,224 records walked, 39,220 sidecars on disk: 36,998 legacy and 2,222 policy pages (1,813 `keep`, 392 `rescue`, 17
`columns`), holding 40,049 VLM readers (39,209 page reads, 840 unit reads) and 39,220 Kraken readers. The run's counters
(39,222 writes, 40,051 VLM readers) include the two pages recorded twice, written once per record with the later record
last. 0 partial mappings, 0 missing raw entries, 2 records skipped (`raw_mismatch:image_url`, the `J24B` pair above),
364 MB of JSON (422 MB on disk), 26 s. A second run wrote nothing (39,222 up to date, 2 mismatches again).

## Example sidecars

* legacy (one page read, Kraken): `ai_reads/readings/Cambridge_CUL_Or_1080_15_50__1__qwen3-vl-8b-heb-v22i-a50.json`
* columns (`2_columns`, two column crops, no whole-page read): `ai_reads/readings/Manchester_JRL_B_2289__1__qwen3-vl-8b-heb-v22i-a50.json`;
  with failed crops, halves and the whole-page reply last: `ai_reads/readings/Manchester_JRL_C_131__2__qwen3-vl-8b-heb-v22i-a50.json`
* rescue (looping whole-page read, `top` + `bottom` halves): `ai_reads/readings/Manchester_JRL_A_127__1__qwen3-vl-8b-heb-v22i-a50.json`
