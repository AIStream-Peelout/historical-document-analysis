# v10 festival queries: what the result cards hide (2026-10-11)

Scope: the production comparison of `genizah_merged_v8` and `genizah_merged_v10` on "Rosh Hashanah" and "Sukkot"
(Semantic tab). Scripts and raw logs: `logs/v10/holiday/` in this repo (`festival_panel.py` is the reusable one).

## Finding

The six v10 cards for each query are correct hits. Their KTIV alternate titles are
"Common Prayers: Rosh ha-Shanah", "Common Prayers: Musaf Rosh ha-Shanah Amidah", "Common Prayers: Sukkot Amidah",
"Common Prayers: Sukkot". The card shows only `description`, which for these records is the KTIV catalogue
*title*, and NLI's title for Genizah fragments is a semicolon list of subject headings
("פיוט;תפילה וברכות;תפילות קבע"). The identification lives in `alt_titles` (and `scholarly_frames`), which the
card does not display and the lexical leg does not search.

Judged on the full record (description + alt_titles + scholarly_frames + subjects), top-10 precision:

| query | v10 (fine-tuned s0211) | v8 (base model) |
|---|---|---|
| Rosh Hashanah | 10/10 | 6/10 (two "Rosh ha-Seder" name hits, a letter, Shabbat Rosh Hodesh) |
| Sukkot | 10/10 | 10/10 |
| Yom Kippur | 10/10 | 10/10 |
| Passover | 10/10 | 10/10 |
| Hanukkah | 10/10 | 7/10 |
| Purim | 10/10 | 10/10 |
| Shavuot | 9/10 | 9/10 |
| shofar | 4/10 | 1/10 |
| Rosh Hashanah piyyut | 9/10 | 6/10 |
| **total** | **82/90** | **69/90** |

So v10 is not worse than v8 on festivals; it is better, and the regression impression comes from the card.

## Serving is correct

- Local kNN replays with the pinned fine-tuned model (`af0252255d3e`, query prompt) return the same six ids
  with the same two score tiers as production; the base-model replay against v8 reproduces the v8 screenshot
  (MIAC 74, ENA 2708.46/47, T-S AS 161.14, the two Rosh ha-Seder records) in order.
- The card's "% match" is the raw cosine × 100. For T-S NS 152.45 the fine-tuned query vector against the stored
  v10 vector is 0.5709 (card: 57.1 %); the base-model query vector would give 0.31. Production therefore embeds
  v10 queries with the fine-tuned model and v8 queries with the base model, per index. (That routing is not in the
  `genizah_search` checkout on this machine; production runs from the MBP checkout.)

## What the fine-tune changed (same texts, both models, cosine to "Rosh Hashanah")

| text | base | fine-tuned |
|---|---|---|
| "Halakhic monograph written by Yosef Rosh ha-Seder" (distractor) | 0.589 | 0.094 |
| "Common Prayers: Shabbat Amidah" (wrong festival) | 0.442 | 0.113 |
| labels only "פיוט;תפילה וברכות;תפילות קבע" | 0.336 | 0.218 |
| labels + "Common Prayers: Rosh ha-Shanah; תפילות קבע: ראש השנה" (the card records) | 0.542 | 0.601 |
| same without the label prefix | 0.611 | 0.559 |
| "Common Prayers: Rosh ha-Shanah" alone, no Hebrew | 0.656 | 0.422 |
| MIAC 74 catalogue sentence ("Prayers and blessings on the occasion of Rosh ha-Shanah …") | 0.674 | 0.540 |
| T-S AS 161.14 ("דיני שופר בראש השנה … treatise on Ros ha-sana, the sofar") | 0.623 | 0.396 |
| Hebrew query "ראש השנה" vs MIAC 74 | 0.226 | 0.528 |

Read: the fine-tune pushed name and wrong-festival distractors far down and made Hebrew queries work, but it scores
long descriptive English sentences lower than short title-like records (MIAC 74 fell from rank 1 to 12, T-S AS 161.14
from 4 to 187 on "Rosh Hashanah"), and it leans a little on the KTIV label prefix that the training texts carried
(−0.04 when the prefix is removed, −0.14 when the Hebrew title is also absent).

## Index-side cause and fix (this repo, implemented 2026-10-11 evening, uncommitted)

`merge_shelfmarks.py` set `description = PGP description → KTIV basic title → TEI → legacy description`. In the
2026-10-11 01:47 merge, 16,894 records showed a KTIV title as their description; 5,909 of those titles are
semicolon label strings and 10,985 are single headings ("קטע גניזה" 2,009, "מקרא [טקסט" 1,093, "פיוט" 774,
"תלמוד בבלי [טקסט" 683, "משנה תורה" 623 …), and 8,747 of them have an English varying title.

The rule now in the code (`genizah_document.py`: `is_ktiv_label_title`, `ktiv_heading_parts`, `ktiv_varying_title`,
`merged_description`; `merge_shelfmarks.py`: `ktiv_description`; tests `tests/test_ktiv_title_precedence.py`,
v10 set 297 green):
1. a KTIV title that is a label string or a bare heading is not a description: its headings become `subjects`
   (19,052 records gain them, Hebrew collection-wide headings dropped);
2. the catalogued varying title replaces it as the description, English before Hebrew, unless that title is itself
   only a label ("Piyyut", "Mishneh Torah"); the Hebrew form stays in `alt_titles`, so the embedded text opens
   with the identification once and no longer with the heading prefix;
3. the document model applies the rule at index/updater time, so the current merged file needs no re-merge; new
   merges carry it natively.

Corpus effect on the 01:47 merge (`logs/v10/holiday/label_rule_corpus_scan.log`): 7,366 descriptions change
(5,419 English, 1,947 Hebrew); eligibility 7,170 stay eligible, 39 become boilerplate-only (their new description is a
bare genre label such as "Liturgy and Brakhot"; the updater refuses to drop vectors, so they keep today's), 157 stay
ineligible. Nothing gains a vector.

Re-ranking simulation with the unchanged v10 model on today's top-500 candidates
(`logs/v10/holiday/strip_labels_sim.log`): P@10 69/70 → 70/70 over the seven festivals; page-1 "terse heading
cards" Rosh Hashanah 10 → 3, Sukkot 10 → 0, Yom Kippur 10 → 0, Passover 9 → 7 (Haggadah titles), Hanukkah 8 → 5,
Purim 5 → 1, Shavuot 4 → 1. Rosh Hashanah page 1 becomes four Musaf Rosh ha-Shanah leaves (now labelled as such),
three Talmud Rosh ha-Shanah leaves, MIAC 74, a yotzer and a mahzor; Yom Kippur opens with piyyut collections, a
yotzer and a letter about the fast before the amidah leaves. The amidah leaves' cosines fall by 0.04 (Rosh
Hashanah) to 0.13 (Sukkot) because the model learned the heading prefix; v11 training on the cleaned text removes
that dependence.

Applying it to v10 in place (user's commands; the dry run against the 01:47 copy is in
`logs/v10/holiday/label_rule_dry_run.log`: unchanged 63,120 | update/reembed 7,170 | update/fields_only 9,195 |
refused:would_lose_vector 39 | document canary unchanged):

    python -m src.datasets.indexing.update_catalogue_enrichment --index genizah_merged_v10 --dry-run --report logs/v10/holiday/label_rule_dry_run_latest.json
    python -m src.datasets.indexing.update_catalogue_enrichment --index genizah_merged_v10 --i-know-this-is-live --report logs/v10/holiday/label_rule_run.json
    python -m src.datasets.indexing.verify_reembedded_indexes   # self-consistency + canaries afterwards

The default merged path is the latest merge, so a real run also applies any re-merge delta since 01:47. v8 is the
rollback index and is left alone. For the lead: the v4 text function, boilerplate list, admission list and golden
fixtures are unchanged; only the document model's description/subjects changed, so eval texts regenerated from the
merge will differ for these ~7.4k records and the frozen-eval re-scores should be re-run before v11 training.

Still owed for v11: training positives that include long descriptive catalogue sentences and name distractors
(Rosh ha-Seder, Rosh Hodesh), and the festival panel (`festival_panel.py`, judged on record fields) as a
`verify_reembedded_indexes.py` stage with per-query thresholds.

## Site-side asks (genizah_search, not editable from here)

1. Result card: show the first Latin-script `alt_titles` entry (or the Hebrew one) as the identification line when
   `description` is a Hebrew heading list; today the identification never reaches the user.
2. Lexical leg: add `alt_titles^2`, `scholarly_frames^2`, `subjects^1.5` to the multi_match; the current field
   list (`title`, `transcription_full_text`, `translation_full_text`, `content_type`) names fields the merged
   indices do not have, so BM25 effectively searches `description` only.
3. Hybrid tab: the function_score uses the multi_match as a filter with a flat +0.5, discarding BM25 scores, and
   any filter zeroes every score (known since the v9 handoff). Until fixed, the Semantic tab is the honest one.
4. Optional: `collapse` on `embedded_text_sha256` for the kNN leg (identical texts share a vector; largest
   eligible cluster in v10 is 147 records).
