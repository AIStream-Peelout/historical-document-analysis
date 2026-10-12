# Promoting "Transcribing the Cairo Genizah with Multi-Modal AI"

Article: https://medium.com/deep-data-science/transcribing-the-cairo-genizah-with-multi-modal-ai-ad4cd9cbe980
Target queries: "transcribing Cairo Genizah", "Cairo Genizah AI transcription", "Genizah HTR", "Hebrew manuscript OCR / VLM", "Judaeo-Arabic handwriting recognition".

## Where the search results stand (checked 2026-10-04)

- "transcribing Cairo Genizah AI": every result is MiDRASH / National Library of Israel coverage from Nov 2025 (JNS syndication across ~6 Jewish papers, JPost, Ynet, Zenodo, PaleoJudaica). We don't show up.
- Our properties that already rank: cairogenizah.ai (the "Cairo Genizah Search" homepage), the genizah_search GitHub repo, `isaacmg/qwen3-vl-8b-hebrew-v20a-merged` and `isaacmg/kraken-genizah-segmenter` on HF, and the earlier Medium piece (search & clustering).
- None of those link to the new article. Inbound links from pages we already control and that already rank are the quickest gain.
- Rule check: none of the copy below names or links the legacy catalogue store. Re-run the project's attribution grep on any edited version before posting.

## Tier 1: our own properties (no account needed; fastest SEO gain)

Done 2026-10-04: all 20 public HF cards (18 models + `synthetic_rashi`, `synthetic_hebrew_v3`) have a link block under the title, marked `<!-- cairo-genizah-ai-transcription-writeup -->`. Anchor text: "Cairo Genizah AI Transcription — formal analysis and write-up of the project", plus "Cairo Genizah AI" → cairogenizah.ai. The historical-document-analysis `readme.md` on master has the same block; its homepage is set to the article and it has 8 topics. genizah_search has topics set, and its readme link is in AIStream-Peelout/genizah_search#29. Still open: cairogenizah.ai page link and Search Console. TRL pushes regenerate cards, so re-add the block after any new checkpoint upload.

1. **cairogenizah.ai**: add a "Research / How the transcriptions are made" page or footer link that points to the article. It ranks for the brand query already, so a link from it carries the most weight. Also add the site to Google Search Console and Bing Webmaster Tools, submit a sitemap, and request indexing.
2. **HF model cards** (v20a-merged, newest flagship, kraken-genizah-segmenter, Genizah Reader Space): add a "Read more" line linking to the article.
3. **GitHub READMEs** (historical-document-analysis, genizah_search): add the article link near the top, and set the repo topics `cairo-genizah`, `hebrew-ocr`, `htr`, `vision-language-model`, `digital-humanities`.
4. **Medium**: add tags (max 5: `Cairo Genizah`, `Machine Learning`, `Digital Humanities`, `OCR`, `Jewish History`), and link this article and the earlier search/clustering article to each other.
5. **arXiv / HF Papers**: once the ARR paper is camera-ready, an arXiv preprint and an HF Papers page would be the strongest long-term ranking signal for "Genizah transcription". Link the article from the abstract page or code links.

## Tier 2: community posts (drafts below; each needs your account and an OK)

| Venue | Audience | Notes |
|---|---|---|
| Hacker News | ML + history-curious devs | Submit the URL with the title as-is. Weekday morning US Eastern. One submission only; don't ask for upvotes. |
| r/MachineLearning | ML researchers | Needs the `[P]` tag. Self-promo is OK if you answer questions. |
| r/LocalLLaMA | People running local VLMs | Lead with the open 8B Qwen3-VL fine-tune and LM Studio/MLX use. |
| r/Judaism, r/AcademicBiblical, r/DigitalHumanities | Subject audience | Lead with the manuscripts, not the model. Read each sub's self-promo rules first. |
| LinkedIn | Professional network | Long-form post with 1–2 images (a page + its transcription). |
| Bluesky / Mastodon (#DigitalHumanities, #Genizah, #JewishStudies) | DH scholars | The DH and Jewish-studies academic community is very active on both. |
| X | ML crowd | A thread with images. |
| Emails (not posts) | Jim Davila (PaleoJudaica), Princeton Geniza Lab, the MiDRASH team, Humanist list, H-Judaic | PaleoJudaica already covered MiDRASH, so it's a natural follow-up. A short, personal note works best. |

## Drafts

### Hacker News
Title: `Transcribing the Cairo Genizah with Multi-Modal AI`
URL: the article. No text body. Optional first comment from you:
> Author here. I fine-tuned an open 8B vision-language model (Qwen3-VL) to read Hebrew, Aramaic and Judaeo-Arabic manuscript fragments from the Cairo Genizah, and scored it against held-out benchmarks of documentary letters/contracts and religious pages. The weights and the segmenter are on Hugging Face, and the transcriptions feed a public search site (cairogenizah.ai). Happy to answer questions about the data, the failure modes (hallucination on damaged regions, looping), or evaluating CER on medieval Hebrew.

### r/MachineLearning
Title: `[P] Fine-tuning Qwen3-VL to transcribe medieval Hebrew/Judaeo-Arabic manuscripts from the Cairo Genizah`
> I've been working on reading the Cairo Genizah, roughly 400k medieval manuscript fragments in Hebrew, Aramaic and Judaeo-Arabic, with an open VLM instead of a classic line-based HTR pipeline. This write-up covers:
> - building training data from scholarly editions without leaking the benchmark pages
> - comparing a fine-tuned Qwen3-VL-8B with Kraken line recognition and frontier models on held-out documentary and religious benchmarks
> - the failure modes we found (hallucinating plausible text on damaged regions, looping, box drift) and how we measure them
>
> Article: <link> · Model: https://huggingface.co/isaacmg/qwen3-vl-8b-hebrew-v20a-merged · Code: https://github.com/AIStream-Peelout/historical-document-analysis
>
> Feedback on the eval methodology is especially welcome.

(Swap in the current flagship model link and the headline CER numbers, quoted exactly as the article gives them.)

### r/LocalLLaMA
Title: `An 8B Qwen3-VL fine-tune that reads medieval Hebrew manuscripts (runs in LM Studio/MLX)`
> Open weights, built for diplomatic transcription of Cairo Genizah fragments (Hebrew, Aramaic, Judaeo-Arabic), plus phrase-to-box grounding. The write-up covers the training data, the benchmarks, and what broke along the way (including a dead vision-LoRA bug in Unsloth's Qwen3-VL adapters). <article> · <HF link>

### r/Judaism / r/AcademicBiblical
Title: `Using AI to read the Cairo Genizah: what works, and what still needs a human scholar`
> I've written up a project that uses a vision-language model to transcribe Genizah fragments: letters, contracts and liturgical and Talmudic pages. It's honest about where the model fails, such as damaged fragments and invented text. The transcriptions are searchable at cairogenizah.ai. I'd love to hear from anyone who works with Genizah material. <article>

### LinkedIn
> The Cairo Genizah is a medieval storeroom in Fustat that preserved roughly 400,000 fragments of Jewish life: letters, contracts, poetry, Talmud. Most of it has never been transcribed.
>
> I've published a long write-up on how we're teaching an open multi-modal model to read it: fine-tuning Qwen3-VL on scholarly editions, benchmarking against held-out fragments, and building ways to catch the model when it invents text.
>
> 📖 <article>
> 🔎 Search the corpus: https://cairogenizah.ai
> 🤗 Models: <HF link>
>
> #DigitalHumanities #MachineLearning #JewishStudies #OCR #ComputerVision

### Bluesky / Mastodon (≤300 chars)
> New write-up: transcribing the Cairo Genizah with an open vision-language model. Hebrew, Aramaic and Judaeo-Arabic fragments, held-out benchmarks, and an honest look at failure modes. <article> #DigitalHumanities #Genizah #JewishStudies

### Email: PaleoJudaica / Geniza Lab
Subject: `Open VLM transcription of Genizah fragments: write-up + benchmarks`
> Dear Dr. Davila,
> You covered MiDRASH's automatic Genizah transcriptions last year. I've just published a complementary approach: an open vision-language model fine-tuned to read documentary and literary Genizah fragments, evaluated on held-out benchmarks, with the results searchable at cairogenizah.ai. In case it's of interest to your readers: <article>
> Best, Isaac Godfried
