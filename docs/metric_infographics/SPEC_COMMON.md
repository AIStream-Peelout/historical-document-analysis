# Common spec for the metric infographic series (Genizah paper blog post)

Design system = docs/metric_infographics/1_ngram_precision.html. Read it fully first and REUSE ITS CSS VERBATIM
(same tokens, fonts link, classes: .wrap .eyebrow .lede .pipeline .step .num .step-body .sample .sample.ref .two
.letters .chips .chip.hit/.miss .legend .score .bar .gates .gate .why .why-grid .why-card .stat .null .track
.limits .limit .dot footer, .heb for Hebrew (dir rtl), .mono for numbers). Add new small CSS rules only when the
spec needs a component the first page lacks (e.g. a stacked bar); keep them in the same token vocabulary
(--paper --paper-2 --ink --ink-2 --ink-3 --rule --hit --hit-bg --miss --miss-bg --gold --chip-bg) so both themes work.
Structure is the same on every page: eyebrow + question-title + lede; numbered pipeline steps with the worked
example threaded through; a "why" section with 3 stat cards; a "what it does not see" limits list; footer.
Rules:
- No <!doctype>/<html>/<head>/<body>; the file starts with <title> then the fonts <link> then <style>, like page 1.
- Title = a short question, like "Is it on the page?". Eyebrow names the metric.
- Use ONLY the numbers given in the spec. Do not invent or round differently. Hebrew strings exactly as given.
- Colour: verdigris (--hit) = correct/matched/kept, madder (--miss) = wrong/invented/dropped, --gold marks the reference.
- No emoji, no gradients, no cards-with-shadows; numbering only where the content is a real sequence.
- Phone width must work (the first page's media queries handle it if you keep its classes).
- Save to the path given in the spec. Do not publish, do not open a browser, do not run servers. Reply with the
  path and three lines on what you built. Keep the reply short and do not paste Hebrew into the reply.
Shared worked example (same fragment as page 1, Oxford, Bodleian MS heb. a 2/4, Judaeo-Aramaic ketubba):
  Reference, visible ink (29 chars, 5 words):  בנחשא טבא מעליא ויבנו ויצליחו
  Model output, normalised (40 chars, 7 words): בנחשא טבא מעליא ויכנו ויצליחו שלום וברכה
  (one misread letter: כ where the page has ב; and an invented closing blessing שלום וברכה)
