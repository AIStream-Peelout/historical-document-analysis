# Prompt: "Read this fragment" playground for cairogenizah.ai

Paste everything below this line into a Claude Code session opened in
`~/Documents/GitHub/genizah_search` (the website repo). It is written for the
agent doing the work; the facts about the model come from the sibling repo
`historical-document-analysis` and were measured on 2026-09-08.

---

## Context and goal

Build an experimental, public **"Read this fragment"** playground for the Cairo
Genizah AI site: a visitor picks a fragment image the site already indexes (or
uploads one), chooses a task, and watches our fine-tuned vision-language model
transcribe it **while bounding boxes for each line appear on the image in real
time**. It must open with a prominent "experimental, do not trust" notice. It is
also the maintainer's own tool for probing the model on arbitrary pages, so a
"show raw model output" toggle and the timing of each request matter.

Before touching anything read `AGENTS.md`, `CLAUDE.MD` and
`docs/SHARED_RUNTIME.md`. This machine is production. Never eject, unload or load
extra LM Studio models, never rebuild the backend container without explicit
approval (a rebuild is a deploy), never touch the databases. Develop against
`scripts/dev_backend_local.py` on :8010. There is no Node on the host; the
frontend builds only through `docker compose build frontend` (ask first).

## The model

* **LM Studio key:** `qwen3-vl-8b-heb-v20a-step1800` (MLX, 8-bit, 9.9 GB; usually
  resident, JIT-loads in 10–20 s if not). Configure it through an env var
  (`PLAYGROUND_MODEL`) — never hard-code, and never fall back to a different
  model if it is missing; return a "playground offline" status instead.
* **What it is:** Qwen3-VL-8B-Instruct with a LoRA fine-tune (language LoRA r16
  + vision-tower LoRA; the vision→language merger frozen) for Hebrew-script
  manuscripts. Version 2.0a is the first checkpoint trained with **grounding
  tasks** (phrase → box, box → text, page → lines with boxes) alongside plain
  transcription. Training data: Cairo Genizah documentary fragments with
  Princeton Geniza Project transcriptions, Hebrew religious/Talmudic manuscript
  pages with editorial transcriptions and word geometry accessed through the
  National Library of Israel's KTIV project, printed Vilna Talmud pages, and
  synthetic renders. Credit exactly those sources in any public text (NLI KTIV,
  Princeton Geniza Project, holding libraries such as Cambridge University
  Library, JTS, the British Library). Do not name any other manuscript
  database.
* **What it can do (single decode, temperature 0.1, held-out pages):**

  | task | measure | v2.0a | parent v1.9a |
  |---|---|---|---|
  | page transcription, Genizah religious set (140) | median aligned F1 | 0.849 | 0.816 |
  | page transcription, PGP documentary set (131) | median aligned F1 | 0.875 | 0.862 |
  | locate phrase → box (72 queries) | median IoU / boxes with IoU ≥ 0.5 / predicted centre inside true box | 0.47 / 49% / 69–78% | 0.29 / 25% / 54% |
  | read box → text (37 lines) | median CER | 0.13–0.17 | 0.58–0.63 |
  | grounded page → lines+boxes (24 pages) | pages with parseable JSON / median line-box IoU / median line CER | 21 of 24 / 0.37–0.41 / 0.27–0.31 | 17 of 24 / 0.15–0.23 / 0.21–0.31 |

  Ranges are two independent decodes of the same model; the spread is normal
  decode noise, so **the same image can give slightly different output twice**.
  Say so in the UI.
* **Known failure modes to design around:**
  1. It can hallucinate fluent, plausible Hebrew for damaged or blank regions
     (about 1% of pages fully, more often a few words). Boxes do not make text
     more trustworthy.
  2. Repetition loops on hard pages (the same line emitted again and again).
     Detect server-side (a 24-character window repeating ≥ 6 times) and stop the
     stream with a "the model started looping" notice.
  3. Grounded-page boxes are placed on the right line but are usually **much
     wider than the written line** (the model draws the full column width).
     Position is reliable; width is not. Locate-phrase boxes are tighter.
  4. About 1 page in 8 in grounded mode returns text that is not a valid JSON
     array (missing brackets, trailing prose). Parse incrementally (below) and
     show whatever lines did parse plus the raw text.
  5. Scripts it was **not** trained on: Arabic script, Latin, Greek, Syriac,
     Coptic. Judaeo-Arabic (Arabic language in Hebrew letters) is fine. It does
     transcribe printed Talmud pages well.
  6. Locate works best with 2–3 consecutive words that occur once on the page
     (that is how it was trained); single common words are ambiguous by
     construction.

## How it is served

* OpenAI-compatible `POST {LLM_STUDIO_URL}/v1/chat/completions`
  (`host.docker.internal:1234` from the backend container, `127.0.0.1:1234`
  from the dev server). Reuse the existing LM Studio plumbing in
  `src/backend/lms_agentic_search.py`: model resolution/JIT loading
  (`resolve_model`), the TTL field, the request timeout, and above all the
  **admission semaphore** (`LM_STUDIO_MAX_CONCURRENCY`, default 1). One
  vision request at a time is the right default: the image prefill saturates
  the GPU, and two concurrent requests just double both users' latency.
* Message shape (exactly what the benchmarks use):

  ```json
  {"model": "<PLAYGROUND_MODEL>", "temperature": 0.1, "max_tokens": 3500, "stream": true,
   "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "text", "text": "<task prompt>"}]}]}
  ```
* **Streaming works.** Measured on the production Studio with a 2800×1785 page
  (0.9 MB JPEG): first token after 19 s (that is the image prefill — nothing to
  optimise in the app), then ~46 tokens/s, i.e. **one completed line object
  roughly every second** in grounded mode; the whole 6-line page took 24 s.
  Full pages of 30–40 lines take 60–90 s. A locate query is ~15 s (almost all
  prefill), a read-box query ~15 s. Design the UI around a ~20 s "reading the
  image…" phase followed by lines appearing one by one.
* **Image policy.** The model's preprocessor wants 6.5–7 megapixels (it
  upscales smaller images and downscales larger ones itself). Send the largest
  image the site has for the fragment (the `actual_image_url` / first of
  `image_urls` from the index, fetched **server-side**, never by the browser)
  as JPEG; cap uploads at 12 MP / 10 MB and convert PNG/TIFF to JPEG. Do not
  pre-shrink below ~6 MP. Server-side fetches must be restricted to the image
  hosts already present in the index (allow-list the hostnames; never fetch an
  arbitrary user-supplied URL).
* **Coordinates.** Every box is `[x1, y1, x2, y2]` with values **0–1000
  normalised to the width and height of the image the model received**. So
  the browser may display a smaller copy of the same image and scale boxes by
  the displayed width/height. The one hard rule: the displayed image and the
  inferred image must have the same framing (no crop, no padding, same
  rotation). If the viewer shows a IIIF region, send that region.

## The tasks and their exact prompts

The wording below is what the model was trained and benchmarked with. Keep it
byte-identical (store the strings in one backend module with a test that
asserts they never change). Phrase and box values are substituted into the
`{}` slots.

**1. Transcribe the page** (plain text, no boxes). Training prompt:

```
This image is a manuscript fragment from the Cairo Genizah — handwritten Hebrew script (the language may be Hebrew, Judeo-Arabic, or Aramaic).

Transcribe the text exactly as written, in reading order. Mark unclear characters with [?].
Where text is lost or illegible due to damage, write [...].
Do NOT correct, restore, or complete from memory.

Return ONLY the transcription.
```
Output: plain lines, right-to-left text. `max_tokens` 4096.

**2. Transcribe with boxes** (the realtime one):

```
Transcribe this manuscript page line by line. Respond with ONLY a JSON array; each element {"text": "...", "bbox_2d": [x1, y1, x2, y2]} gives one line's transcription and its bounding box, coordinates normalized to 0-1000. Preserve reading order.
```
Output: `[{"text": "...", "bbox_2d": [105, 248, 936, 347]}, ...]`, one object per
line, streamed in reading order. `max_tokens` 3500.

**3. Find a phrase** (locate):

```
Locate the exact Hebrew phrase "{phrase}" on this manuscript page. Respond with ONLY a JSON object {"bbox_2d": [x1, y1, x2, y2]} giving the phrase's bounding box, coordinates normalized to 0-1000. No other text.
```
Output: `{"bbox_2d": [x1, y1, x2, y2]}`. `max_tokens` 64. Parse with the regex
`\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]` on the whole
reply (the benchmark does exactly this), clamp to 0–1000, reject if
`x2 <= x1` or `y2 <= y1`.

**4. Read a region** (read box) — the user draws a rectangle on the image:

```
Transcribe ONLY the Hebrew text inside the region bbox_2d = [{x0}, {y0}, {x1}, {y1}] (coordinates normalized 0-1000) on this manuscript page, exactly as written. Mark unclear characters with [?]. Return the text alone, no commentary.
```
Output: plain text. `max_tokens` 512.

**5. Layout questions** (optional extras, trained but simpler):

```
How many columns of text does this manuscript page have? Answer with the number only.
Transcribe ONLY the {first|last} line of this manuscript page, exactly as written. Return the text alone.
Which line of this manuscript page contains the phrase "{phrase}"? Return that full line's transcription alone, exactly as written.
```

## Backend

Add `src/backend/playground_service.py` (prompts, image fetch/validation,
LM Studio streaming call, incremental JSON parsing, loop detection, timing)
and routes in `app.py`:

* `POST /playground/read` → **SSE** (`StreamingResponse`, `text/event-stream`,
  same pattern as `/chat-stream`). Body: `{doc_id | image_upload, task,
  phrase?, box?}`. Events, each a JSON object:
  * `status` — `{"phase": "queued", "position": n}` / `"loading_model"` /
    `"reading_image"` / `"generating"`; send a heartbeat comment every 10 s so
    proxies keep the connection open.
  * `delta` — `{"text": "..."}` raw token text as it arrives (for the raw view
    and for plain transcription).
  * `line` — `{"index": i, "text": "...", "bbox": [x1,y1,x2,y2]}` emitted the
    moment one array element closes (grounded task).
  * `box` — `{"bbox": [...]}` (locate task).
  * `done` — `{"text": "...", "lines": [...], "timing": {"ttft_s": .., "total_s": ..}, "finish_reason": "stop|length|loop"}`.
  * `error` — `{"message": "..."}` (model not available, image rejected, timeout).
* `GET /playground/status` → whether the feature is enabled, the model is
  resident, and current queue depth.
* Incremental parsing of the streamed array: keep a buffer; scan characters
  tracking `in_string` (respect `\"` escapes) and brace depth; when depth
  returns to 0 after a `{`, `json.loads` that slice, validate `text` (str) and
  `bbox_2d` (4 numbers, clamp 0–1000), emit `line`. Ignore everything outside
  objects. This tolerates a missing closing `]`, trailing prose and partial
  last objects.
* Guardrails: feature flag `PLAYGROUND_ENABLED`; per-IP rate limit (start with
  8 requests per 15 min and 1 in flight); global queue through the existing
  semaphore with position reporting; hard 180 s cap per request; uploads are
  held in memory only and discarded after the response; log timings and the
  doc id, never the image or the transcription; the "loop" detector above.
* Add `PLAYGROUND_MODEL`, `PLAYGROUND_ENABLED` and the limits to
  `.env.example` and to `docs/SHARED_RUNTIME.md`. While there, fix the stale
  line "Never eject `qwen3-vl-8b-heb-v18b-step700`": the models the sibling
  project keeps resident are now `qwen3-vl-8b-heb-v19a-step1300` and
  `qwen3-vl-8b-heb-v20a-step1800`.

## Frontend

A new route `/read` (React 18, `react-router-dom`; the existing document page
`src/frontend/src/core_results/DocumentModel.jsx` should get a "Read with AI"
button that deep-links to `/read?doc=<id>`).

* Layout: image stage on the left (the fragment with an SVG overlay), text
  panel on the right (right-to-left, the site's Hebrew font), task picker and
  the warning banner on top, a "raw model output" disclosure at the bottom.
* Overlay: an `<svg viewBox="0 0 W H">` sized to the displayed image, boxes as
  `<rect>` with `vector-effect: non-scaling-stroke`, numbered in reading
  order; hover a box → highlight its text line and vice versa; toggles for
  boxes / numbers; wheel zoom and drag pan. A working reference for exactly
  this overlay, zoom/pan and coordinate mapping is
  `~/Documents/GitHub/historical-document-analysis/src/datasets/evaluations/grounding_eval/viewer/viewer.template.html`
  (functions `drawOverlay`, `applyTransform`, `fit`, `zoomAt`, `toPx`); lift
  the mechanics, not its data model.
* Realtime: consume the SSE; during `reading_image` show a progress hint
  ("reading the image, ~20 s"); on each `line` event append the text line and
  draw its box with a short fade-in; `done` swaps in the final text. For the
  plain transcription task stream the text only.
* Locate: a text field ("paste 2–3 consecutive words") plus click-on-a-word
  in an existing transcription; result is a single highlighted box with a
  "centre inside / IoU unknown" caveat (there is no ground truth here).
* Read a region: drag a rectangle on the image; send its normalised box.
* Banner copy (edit freely, keep the substance):
  > **Experimental.** This model reads handwritten Hebrew-script manuscripts
  > and is often right, but it also invents plausible words where the page is
  > damaged, sometimes repeats itself, and its boxes are approximate (about
  > half of phrase boxes overlap the true location by 50% or more). Nothing
  > here is checked by a person. Do not cite it; use it to explore. The same
  > image can read slightly differently twice. Images you upload are not
  > stored.

## Acceptance checks

1. `GET /playground/status` reports enabled + model resident on the dev
   server (:8010) without loading anything new (`curl localhost:1234/api/v0/models`
   before and after must list the same models).
2. Grounded read of
   `~/Documents/GitHub/historical-document-analysis/src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1/images/990025086880205171_FL202082468.jpg`
   streams about 6 `line` events, the first within 30 s, all boxes inside
   0–1000, total under 40 s.
3. Boxes land on the written lines at two different display widths (the
   normalisation is right if they do).
4. A reply with a missing `]` still yields its lines; a looping reply ends
   with `finish_reason: "loop"`.
5. Two simultaneous requests: the second receives `queued, position 1` and
   then completes; a ninth request within 15 minutes from one IP is refused
   with a clear message.
6. The prompt-string test passes and the banner is visible on first load.

## Later (not now)

Export lines as W3C/IIIF annotations (Mirador is already a dependency, so
"open in Mirador" with the boxes is cheap); a "compare with v1.9a" toggle;
thumbs-up/down per line to collect a correction set for v2.1; share links.
