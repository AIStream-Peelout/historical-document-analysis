# Prompt for the genizah_search session: advanced "readers" view (who read this page which way)

Paste from the line below into the site session. Written 2026-10-14. Data side is done in historical-document-analysis
(`docs/readings_sidecar.md` is the contract; three example files are named at the end).

---

Task: add an advanced, off-by-default "Readers" panel to the document view that shows, for one page image, every reader's
transcription lines drawn on the image and listed beside it: the whole-page VLM read, each decode-policy unit read
(column crops, rescue halves), Kraken's line fragments, and the final merged consensus lines with their agreement. Add
a per-line expert correction affordance on the same panel. Audience: ML people and Genizah scholars; the panel must be
invisible to ordinary visitors. Work on a branch (never master); backend + frontend + a browser check. Never name or link
the legacy catalogue store in anything you write (run the project's attribution grep over your output before finishing).

## The data (produced on the Studio, one JSON per page image)
Schema `readings-v1`, documented in `/Users/isaac/Documents/GitHub/historical-document-analysis/docs/readings_sidecar.md`
(read it first). Files live at `src/datasets/raw_data/cairo_genizah/ai_reads/readings/<doc_id>__<image_index>__<vlm_model>.json`
on the Studio; 39,223 pages today (36,998 "legacy" pages with two readers, 2,225 policy pages with up to eight), median
7 KB, largest 94 KB, 430 MB total, rebuilt incrementally every half hour. The essentials:

- Top level: `doc_id`, `image_index`, `image_url`, `image_sha256`, `image_width`, `image_height`, `vlm_model`,
  `vlm_revision`, `decoded_at`, `rule_version`, `policy`, `readers[]`, `final`, `provenance`.
- Every box is `[x1, y1, x2, y2]` in **0–1000 of the oriented page** (the same frame the existing consensus overlay uses).
- `readers[]`: `{id, kind: "vlm"|"htr", scope: "page"|"unit", unit, role, reply_index, parsed, health_label,
  crop_box_1000, lines: [{index, text, box}] | null}`. Kraken readers add `conf` per line and `cache_key`. Ids look like
  `vlm:qwen3-vl-8b-heb-v22i-a50:page`, `vlm:…:col0:crop`, `vlm:…:col1.top:crop`, `kraken:MiDRASH_Gen_01@k7.0.3-ktivseg-e10-mv`.
  `lines: null` means the reply could not be attributed (none today); `[]` means the reader found nothing.
- `policy`: null on legacy pages; otherwise `{action: keep|rescue|columns, reason, final_source, calls, fallback,
  mode, units: [{name, role: column|half, crop_box_1000, resolved_by: crop|halves|share|unresolved, health_label,
  attempts: [{source, label, letters, n_lines, reply_index}]}]}`. `col0` is the right column.
- `final`: `{source: whole_page|assembled|assembled_partial|legacy, lines: [{index, text, bbox, agreement, status,
  htr_text, htr_fragments}]}` — identical to what the site already shows as the consensus read.
- Forward compatibility: new reader kinds will appear later (several VLM checkpoints voting; a `vote` block with per-slot
  candidates). Render unknown reader ids generically from `kind`, `scope`, `role`; do not hard-code the current ids.

## Transport (decide with the person; recommendation first)
1. **Recommended:** a new Elasticsearch index `ai_readings` on the production ES, one document per sidecar with `_id` =
   `<doc_id>__<image_index>__<vlm_model>`, indexed fields `doc_id` (keyword), `image_index` (integer), `vlm_model`
   (keyword), `image_sha256` (keyword), `decoded_at` (date), `policy_action` (keyword), `n_readers` (integer), and the
   whole sidecar stored under `readings` with `enabled: false` (stored, not indexed). The Studio side will push it with
   the same half-hourly loop that pushes consensus records (that loader lives in this repo under the YK loader scripts;
   the push script for readings will be written on the Studio side once you confirm the index name and mapping).
2. Alternative with no backend work: the Studio uploads the sidecars to the public image bucket under a `readings/`
   prefix and the frontend fetches them directly; costs bucket egress and needs CORS, and it leaks nothing new (the files
   hold only our own reads). Only if 1 is refused.

## Backend
- `GET /documents/{doc_id}/readings?image_index=N` → the stored sidecar (404 when absent; 409-style message when the
  record exists under two models: return both, keyed by `vlm_model`). Cache for an hour. No ranking, no search.
- `POST /documents/{doc_id}/readings/corrections` with `{image_index, line_index, reader_id, bbox, original_text,
  corrected_text, note, contributor}` → stored in a new index `line_corrections` with `created_at`, `page_hash`
  (`image_sha256`), `source_reader` and the record's `vlm_model`/`rule_version`, plus `GET …/corrections` for the page and
  an export route (`/admin/corrections.jsonl`, token-protected) that the Studio pulls as human ground truth. The site has
  no accounts: gate the panel and the POST behind a shared advanced-access token kept in local storage after the user
  enters it once (`?advanced=1` opens the prompt); store the contributor name the user types; rate-limit the POST.

## Frontend (document view, behind the advanced token)
- A "Readers" tab beside the existing consensus view. Left: the page image with overlays; right: a reader list.
- Reader list: one row per reader with a colour swatch, kind/scope/role, line count, `health_label` (as a badge: ok,
  loop, capped, near_empty, skipped_text, overgen, parse_loss), and a checkbox to show or hide its boxes. The final
  consensus is one more row, on by default; every other reader is off by default.
- Overlay: each visible reader's boxes in its colour; hovering a box shows the line text; clicking a final line pins it
  and lists, for every reader, the line whose box overlaps it most (IoU), so the user sees all readings of one line.
- Policy strip above the image for policy pages: `action · reason · final source · calls · fallback`, and the unit crops
  drawn as dashed rectangles with their `resolved_by` and `health_label` (e.g. "col1: loop → halves → share").
- Compare mode: pick two readers, show their lines side by side aligned by box overlap, differences highlighted at
  character level (a plain diff is fine).
- Correction: on a pinned final line, "Suggest a reading" opens a field prefilled with the final text; submit posts the
  correction; existing corrections for the page show as a third column with contributor and time. Hebrew and Arabic are
  RTL: `dir="auto"` everywhere text is rendered.
- Keep the panel out of the default bundle path if it adds weight (lazy load).

## Known data caveats (do not try to fix them on the site)
- 42 legacy pages have zero lines in `final` because of an older parser; they will be re-parsed on the Studio.
- Two doc ids differing only in letter case collide in the cache (`…J24B` / `…J24b`); one of them has no sidecar.
- A page recorded by two models has two sidecars; show the one matching the record the page view is displaying.

## Acceptance
- `Manchester_JRL_C_131` image 2 (columns page that fell back): eight readers listed, two dashed column crops with
  loop badges, final = whole page.
- `Manchester_JRL_A_127` image 1 (rescue page): two half units, the half reads overlaid, final assembled.
- `Cambridge_CUL_Or_1080_15_50` image 1 (legacy page): two readers, no policy strip.
- A correction posted on any line appears for the page and in the export; without the advanced token the tab, the
  endpoint responses and the POST are all unavailable.
- Ordinary visitors see no change anywhere. Report back with the branch, screenshots of the three pages, and the exact
  index name and mapping you created so the Studio-side push can be written against it.
