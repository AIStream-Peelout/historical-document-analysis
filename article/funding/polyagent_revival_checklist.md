# PolyAgent minimal revival — so it can verify researchers under its umbrella

Goal: enough of an organizational footprint that Google (Gemini Academic Program,
Google for Nonprofits / Cloud credits) and arXiv/venues can verify "Isaac Godfried,
researcher at PolyAgent (Radical Philanthropies, 501(c)(3))". Not a relaunch.

Legal entity (verified 2026-09-16, ProPublica/Charity Navigator):
- Radical Philanthropies, EIN 85-2701136, Irvine CA; 501(c)(3) since Oct 2020
- Form 990 filed for 2022, 2023, 2024 → exempt status current
- Directors of record: Artur Kiulian, Anastasiia Dvorzhanska
- PolyAgent = an initiative of Radical Philanthropies (site already says so)

## Steps (owner → effort)

1. Google for Nonprofits account in Radical Philanthropies' name — Artur (or Isaac
   with Artur's written authorization) → ~30 min + Goodstack verification 2–14 business days.
   Needs: EIN, IRS determination letter (or Goodstack pulls it), requester attests authority.
   Check first whether an account ALREADY exists from the earlier credits.
2. Google Workspace for Nonprofits (free) on polyagent.co → whoever controls DNS adds
   one TXT record → isaac@polyagent.co + artur@polyagent.co. This is the affiliation
   proof every Google program accepts, and it fixes the "no org email" gap.
3. Research page on polyagent.co — Isaac → 1–2 hours. Lab blurb, named researchers
   (those who agree), publications (ARR #2809 "Plausible but Wrong", PyData Boston
   tutorial), links: cairogenizah.ai, GitHub org, Radical Philanthropies. Refresh
   the latest-activity date (site currently stops at Sept 2024; "coming soon" button
   should go).
4. One-paragraph affiliation letter signed by Artur as Director — template below →
   5 min. Fallback proof for Goodstack / Google / a venue.
5. Add Isaac as admin on the Google for Nonprofits account so nothing depends on
   Artur afterwards.
6. THEN submit the Gemini Academic application with the polyagent.co email and the
   research page link; optionally request Cloud credits from the nonprofit account.

## Message to Artur (paste-ready)

Subject: 1-hour favor — reviving PolyAgent's nonprofit verification for research credits

Hi Artur,

Quick one. I'm publishing the Cairo Genizah benchmark work under the PolyAgent
umbrella (paper under review at ACL Rolling Review; public site at cairogenizah.ai)
and I'm applying for Gemini API research credits and Google's nonprofit Cloud
credits. Both need the organization to be verifiable, and right now nobody has a
polyagent.co email and the site has no researcher page.

Radical Philanthropies is in good standing (990s through 2024, you're director of
record), so this is small:

1. Does a Google for Nonprofits account already exist for Radical Philanthropies
   from when we had credits before? If yes, could you add me as admin? If no, could
   you request one (EIN 85-2701136; Google's partner Goodstack verifies in a few
   days) — or send me a one-line authorization and I'll do it.
2. Who controls polyagent.co DNS? I need one TXT record added so we get free
   Workspace for Nonprofits email on the domain (isaac@ / artur@polyagent.co).
3. Could you sign a one-paragraph affiliation letter as Director (I'll draft it)?

I'll do everything else: the research page on the site, the applications, and
being the ongoing admin so it doesn't come back to you. Happy to jump on a call
if easier.

Thanks,
Isaac

## Affiliation letter (for Artur's signature)

To whom it may concern,

Radical Philanthropies (EIN 85-2701136), a 501(c)(3) non-profit organization,
operates PolyAgent, a non-profit open-source research lab. I confirm that Isaac
Godfried is a researcher affiliated with PolyAgent and that his research on
machine transcription and evaluation of historical Hebrew-script documents,
including the associated open benchmark and the public resource cairogenizah.ai,
is conducted and published under the PolyAgent umbrella.

Artur Kiulian, Director, Radical Philanthropies — [date]
