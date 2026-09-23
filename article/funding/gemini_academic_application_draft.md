# Gemini Academic Program — application draft

Apply at: https://ai.google.dev/gemini-api/docs/gemini-for-research ("Apply now")
Reviewed monthly. Fill placeholders in [brackets] before submitting.

---

**Applicant:** Isaac Godfried — Independent researcher (historical-document AI); contact via personal email / GitHub
**Affiliation:** Research published under PolyAgent (https://www.polyagent.co), a
non-profit open-source research lab and initiative of Radical Philanthropies, a
501(c)(3) established 2020. No institutional email; the applicant self-funds the
compute and is the sole maintainer of the benchmark and public resource below.
**Country:** United States

## Research summary (~200 words)

We build and maintain an open multimodal benchmark for machine transcription of
historical Hebrew-script documents: the Cairo Genizah (~400,000 manuscript
fragments, 9th–13th c., largely Judaeo-Arabic — a low-resource language written
in Hebrew letters) and printed Talmud pages. The benchmark's contribution is
behavioral evaluation: every system output is classified as a substantive
reading, an abstention, a loop collapse, or a fluent hallucination *before* any
error rate is computed, using ground-truth-free detectors. This separates
reading ability from the shape of failure — on damaged manuscripts, frontier
VLMs hallucinate fluent, plausible, entirely invented text at rates up to 68%,
which conventional CER conflates with honest error.

Against this benchmark we fine-tune open 8B vision-language models that now
exceed all evaluated frontier systems on both corpora, and we operate a public
search and consensus-transcription pipeline over the corpus
(https://cairogenizah.ai). All benchmark code, scoring pipelines, and result
tables are open source; a paper ("Plausible but Wrong") is under review at ACL Rolling Review (ARR #2809);
reviewers have requested full frontier-model coverage for the next-cycle resubmission.

## Fit to program priority areas

- **Evaluations & benchmarks** — a frozen, verified, publicly documented
  multimodal benchmark with a novel behavior-first failure taxonomy.
- **Multimodal understanding** — vision-language reading of degraded historical
  documents; quantifying when models look versus when they generate from prior.
- (Secondary) **low-resource languages** — Judaeo-Arabic and rabbinic Hebrew,
  aligned with PolyAgent's language-preservation research area.

## How Gemini is used, and why credits matter

1. **Frontier baselines:** Gemini Pro and Flash are the strongest general-model
   baselines in the benchmark; keeping their rows complete and current requires
   periodic full-coverage evaluation runs (65 Talmud pages and 131 Genizah
   fragments × 3 sections/output modes).
2. **Pipeline component:** Gemini Flash performs the section-segmentation stage
   of the OCR comparison pipeline (Kraken HTR and Cloud Vision are only
   comparable when routed through the same segmenter).
3. **Reviewer-driven coverage completion (immediate need):** ARR reviewers
   (September 2026) explicitly flagged that the Gemini Pro rows are partial —
   26/65 Talmud pages and 68/131 Genizah fragments — because the runs hit the
   applicant's personal billing cap, not because of any research decision.
   Completing them (plus repeated-timeout retries and a human-validation sample
   of the Flash-based judge, also requested by reviewers) is required for the
   resubmission.
4. The project is currently self-funded by the applicant; measured Gemini API
   spend for benchmark maintenance was $150 in August 2026 alone (bursty:
   five evaluation days), and frontier-model coverage is constrained by a
   personal spending cap rather than by the research design. Planned follow-on
   work (entity-extraction QA over ~1,600 KTIV manuscripts, v2.2) is gated on
   the same cap.

**Requested:** $5,000 in Gemini API credits over 12 months (≈ two full frontier-coverage
cycles for the resubmission plus the v2.2 entity-QA pass) and research rate limits
to permit batch evaluation runs.

## Links

- Open-source benchmark & pipelines: https://github.com/AIStream-Peelout/historical-document-analysis
- Public resource: https://cairogenizah.ai
- Organization: https://www.polyagent.co
- Paper: under review at ACL Rolling Review (#2809, OpenReview VnDz6qmOXS); PDF available on request [add preprint link if posted]
- Interactive results: [artifact links if public]
