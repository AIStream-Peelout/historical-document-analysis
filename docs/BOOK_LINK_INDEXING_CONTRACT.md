# BookArticle direct-link indexing contract

What the genizah_search backend reads when it builds "open this work" links in
the chat UI. Store the fields exactly as specified — the serving side does not
guess, and a wrongly-shaped value produces a broken link rather than an error.

## Where it is read

Neo4j, `(:BookArticle)` node properties, fetched by
`neo4j_service.find_book_article()` and served by `GET /book-info`.

## Fields

| Property | Type | Format | Example |
|---|---|---|---|
| `doi` | string | **Bare DOI only.** Starts `10.`. No `https://doi.org/`, no `doi:` prefix, no trailing punctuation. | `10.2307/162589` |
| `isbn` | string | Digits only, hyphens stripped, 10 or 13 chars (trailing `X` allowed on ISBN-10). Books only. | `9780691048698` |
| `url` | string | Full absolute URL to the work's landing page. `https://` preferred. No trailing period. | `https://www.jstor.org/stable/162589` |
| `journal` | string | Journal name — **presence of this field is how the UI decides the work is an article rather than a book**, which changes the link layout. Leave absent/null for books. | `Journal of the Economic and Social History of the Orient` |
| `volume`, `pages` | string | Free text, display only. | `12`, `1-25` |
| `publisher`, `year` | string / int | Display only. `year` as int if possible. | `Princeton University Press`, `1973` |

## How the link is chosen (priority order)

```
1. doi     → https://doi.org/{doi}                      ← best
2. isbn    → https://search.worldcat.org/isbn/{isbn}    ← direct item page
3. url     → used verbatim
4. (none)  → falls back to a WorldCat / Google Scholar *search* — the poor
             experience we are trying to eliminate
```

Only one direct link is shown, taken from the highest tier available. Searches
still appear as secondary buttons.

## Critical: do not store a URL in `doi`

The backend builds `https://doi.org/{doi}`. Storing
`https://doi.org/10.2307/162589` in `doi` yields
`https://doi.org/https://doi.org/10.2307/162589` → broken.
(The serving side now defensively strips `https://doi.org/`, `http://dx.doi.org/`
and `doi:` prefixes, but treat that as a seatbelt, not the contract.)

If a source only gives you a DOI as a URL, strip it to the bare form before
storing. If a value does not start with `10.`, it is not a DOI — put it in
`url` instead.

## Coverage targets (measured 2026-07-31, 6,309 BookArticle nodes)

- `doi`: 0 → **this is the biggest single win available**
- `isbn`: 0
- `url`: 222 (JSTOR 79, Cambridge T-S 37, Academia.edu 35, doi.org 23, rest
  Persée / NLI / Princeton)

~96% of works currently have no direct identifier.

## Backfill approach

1. **DOI via CrossRef** — query `https://api.crossref.org/works?query.bibliographic=`
   with title + first author + year; accept a match only on a high title
   similarity (guard against CrossRef's fuzzy fallbacks). Store `message.DOI`
   verbatim (already bare).
2. **ISBN for books** — OpenLibrary (`https://openlibrary.org/search.json?title=&author=`)
   or the publisher record. Store digits only.
3. **`url` for the rest** — where the source bibliography prints a stable URL,
   capture it at ingestion instead of discarding it; that is where the existing
   222 came from, and the same extractor should run over every work.

## Title join (why some works get no metadata at all)

`/book-info` is called with the title as it appears in the **ES bibliography
index**, and matched against `BookArticle.title` by a leading-word probe
(first ~6 words, case-insensitive, both directions). ES and KG titles differ
(subtitles, editor suffixes, bilingual `English / עברית` forms), so this is
fuzzy and sometimes misses.

**Durable fix:** store the bibliography index's book identifier on the
BookArticle node (e.g. `es_book_id`, matching the prefix of the ES `doc_id`
such as `malk-ish-1-426` or `genizah_kedem_3vol`), or an `es_title` property
holding the exact ES title string. Then the join becomes exact instead of
probabilistic. This also fixes the work→manuscript bridge (item 7).

## Verifying your work

```bash
curl -s -G --data-urlencode "title=<exact ES title>" http://localhost:8000/book-info | jq
```
Expect `direct_url` populated and `direct_url_label` naming the destination
(e.g. "View on JSTOR"). If `found_in_graph` is false, the title join failed.
