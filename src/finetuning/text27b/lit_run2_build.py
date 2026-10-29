# File name: lit_run2_build.py
# Date: 10/27/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Assembly step of the run-2 literature mining pass (:mod:`lit_run2`): candidates per pipeline, image lookups,
output files and the inventory / report / sample documents.

Output layout (run directory from the local config; files over ``large_file_mb`` go to the configured NAS directory
and leave a pointer file behind)::

    translate_en.jsonl, translate_he.jsonl     L5 rows (rows schema) with ``meta.run2`` provenance
    vlm_pages_hebrew_ja.jsonl, vlm_pages_arabic.jsonl   book_editions-layout entries + resolution + images
    catalogue_descriptions.jsonl               candidates (+ ``row``: describe_pgp schema when an input exists)
    analysis_commentary.jsonl                  candidates (+ ``row``: analyse_document schema) and the L5 analysis rows
    INVENTORY.md, REPORT.md, SAMPLES.md, run2_summary.json
    book_pairs_run2/                           the L5 builder's own manifest / audit for the run-2 span sources

Public API: :func:`work_meta`, :func:`licence_class_of`, :func:`translation_rows`, :func:`vlm_candidates`,
:func:`catalogue_candidates`, :func:`commentary_candidates`, :func:`attach_images`, :func:`gate`,
:func:`write_pipeline_files`, :func:`inventory_md`, :func:`report_md`, :func:`samples_md`, :func:`build_all`.
"""
from __future__ import annotations

import collections
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.finetuning.text27b import book_pairs as _bp
from src.finetuning.text27b import builder as _bd
from src.finetuning.text27b import lit_run2 as L
from src.finetuning.text27b import lit_segment as ls
from src.finetuning.text27b import rows as _rows

LOG = logging.getLogger(__name__)
SAMPLE_N = 30


# ===================================================================================== metadata
def work_meta(root: Path, work: str, part: str) -> Dict[str, Any]:
    """Title, licence and language of a literature work from its ``<part>_metadata.json``.

    :param root: Literature root.
    :param work: Top-level directory.
    :param part: Work slug (first part component).
    :returns: ``{"title", "licence", "language", "authors", "year"}`` (None where absent).
    """
    for path in (root / work / part / f"{part}_metadata.json", root / work / f"{part}_metadata.json"):
        if path.exists():
            m = json.loads(path.read_text(encoding="utf-8"))
            src = m.get("source") if isinstance(m.get("source"), dict) else {}
            pub = m.get("publication") if isinstance(m.get("publication"), dict) else {}
            authors = m.get("authors")
            if isinstance(authors, list):
                authors = "; ".join(str(a.get("name") if isinstance(a, dict) else a) for a in authors)
            return {"title": m.get("title"), "licence": src.get("licence") or m.get("licence"),
                    "language": pub.get("language") or m.get("language"), "authors": authors,
                    "year": pub.get("year") or m.get("year")}
    return {"title": None, "licence": None, "language": None, "authors": None, "year": None}


def licence_class_of(licence: Optional[str]) -> str:
    """Licence class (:data:`rows.LICENCE_CLASSES`) of a licence string.

    :param licence: Licence text.
    :returns: ``pd`` / ``cc-by`` / ``cc-by-nc`` / ``copyright`` / ``unknown``.
    """
    s = (licence or "").lower()
    if not s:
        return "unknown"
    if "public domain" in s:
        return "pd"
    if "cc by-nc" in s or "cc-by-nc" in s:
        return "cc-by-nc"
    if "cc by" in s or "cc-by" in s:
        return "cc-by"
    return "copyright"


# ===================================================================================== translation
def _source_work(sources: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Tuple[str, str]]:
    """Span-source slug → (work dir, work slug).

    :param sources: Span sources.
    :param cfg: Config.
    :returns: Mapping.
    """
    out: Dict[str, Tuple[str, str]] = {}
    parts = {s["slug"]: s["part"] for s in (cfg.get("doc_headed") or []) + (cfg.get("anthology") or [])}
    for s in sources:
        slug = s["slug"]
        base = slug[:-len("_formularies")] if slug.endswith("_formularies") else slug
        out[slug] = (s.get("work"), s.get("work_slug") or parts.get(base) or parts.get(slug) or slug)
    return out


def existing_family_pgpids(paths: Sequence[Path]) -> Dict[str, Dict[str, str]]:
    """PGP ids that the t27b L5 / web families already pair, per family.

    :param paths: Family files.
    :returns: ``{family: {pgpid: file name}}``.
    """
    out: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    for p in paths:
        if not Path(p).exists():
            continue
        for r in _rows.read_jsonl(p):
            m = r["meta"]
            if m.get("condition") != "edition":
                continue
            for pid in list(m.get("pgpids") or []) + ([m["pgpid"]] if m.get("pgpid") else []):
                out[m["family"]][str(pid)] = Path(p).name
    return out


def translation_rows(ctx: Any, rows: Sequence[Dict[str, Any]], docs: Sequence[Mapping[str, Any]],
                     sources: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Attach run-2 provenance / hold-out / quality to the L5 rows; split translate and analyse rows.

    :param ctx: Run context.
    :param rows: L5 rows (from the run-2 build).
    :param docs: L5 document records.
    :param sources: Span sources (for notes and OCR quality).
    :returns: ``(translate rows, analyse rows)``.
    """
    sw = _source_work(sources, ctx.cfg)
    notes: Dict[str, Mapping[str, Any]] = {}
    for s in sources:
        for d in s.get("docs") or []:
            notes[f"book:{s['slug']}:{d['doc']}"] = {**d, "_source": s}
    existing = existing_family_pgpids([ctx.p.repo / f for f in ctx.cfg.get("existing_families") or []])
    tr, an = [], []
    for r in rows:
        m = r["meta"]
        slug = str(m.get("source", "")).split(":", 1)[-1]
        work, wslug = sw.get(slug, (None, slug))
        d = notes.get(m.get("doc_id"), {})
        shelf = [str(x) for x in d.get("shelfmarks") or []] or ([m["shelfmark"]] if m.get("shelfmark") else [])
        res = ctx.resolver.resolve(shelf)
        if m.get("pgpids") and not res["pgpids"]:
            res = {**res, "pgpids": [str(x) for x in m["pgpids"]], "resolved": True}
        user = r["messages"][1]["content"]
        target = r["messages"][2]["content"]
        mention = ctx.resolver.mention_reason([target, user], ctx.resolver.holdouts)
        flags: List[str] = []
        score = 1.0
        oq = d.get("ocr_quality")
        if oq == "poor":
            score -= 0.4
            flags.append("ocr_quality:poor")
        elif oq == "fair":
            score -= 0.15
            flags.append("ocr_quality:fair")
        if slug.endswith("_formularies"):
            # read 2026-10-27: the translation span carries the editor's note on how the formulary was rebuilt and
            # the typescript's line-number column interleaved with the text; clean by hand before use
            flags += ["composite_formulary", "editor_note_in_target", "line_number_column_interleaved"]
            score -= 0.45
        if d.get("not_paired"):
            flags.append("not_paired_note")
        if "list" in str(d.get("note", "")).lower() or ls.numbered_share(target.split("\n")) > 0.4:
            flags.append("list_document")
            score -= 0.1
        lr = m.get("length_ratio")
        if isinstance(lr, (int, float)) and not 0.7 <= lr <= 1.8:
            flags.append(f"length_ratio:{lr}")
            score -= 0.1
        fam = m["family"]
        dup = sorted({existing.get(fam, {}).get(str(pid)) for pid in (m.get("pgpids") or []) if existing.get(fam, {}).get(str(pid))})
        if dup and m.get("condition") == "edition":
            flags.append("dup_existing_t27b:" + ",".join(dup))
        pages = list(dict.fromkeys(k for v in ((m.get("provenance") or {}).get("work_pages") or {}).values() for k in (v or [])))
        held = m.get("split") == "heldout" or bool(mention)
        if mention and m.get("split") != "heldout":
            m["split"] = "heldout"
            m["split_reason"] = "span_mention"
            m["holdout_reason"] = mention
        m["run2"] = {"work": work, "work_slug": wslug, "page_keys": pages,
                     "printed_pages": L.printed_pages(ctx.pages_by_key, pages),
                     "shelfmark_raw": shelf, "canonical_ids": res["canonical_ids"], "pgpids": res["pgpids"],
                     "held_out": held, "held_out_reason": m.get("holdout_reason") or (f"span_mention:{mention}" if mention else None),
                     "licence": m.get("licence"), "licence_class": m.get("licence_class"),
                     "extraction_method": (m.get("provenance") or {}).get("located_by") or "agent_read_pages_D15",
                     "quality": {"score": round(max(0.0, score), 2), "flags": flags}, "note": d.get("note")}
        (tr if fam.startswith("translate_") else an).append(r)
    return tr, an


def translation_only_candidates(ctx: Any, sources: Sequence[Mapping[str, Any]], built: Set[str]) -> List[Dict[str, Any]]:
    """Human translations whose document has no PGP digital edition (the L5 builder drops them as
    ``no_pgp_edition``): kept as candidates for a read-conditioned or image-conditioned translation family (a
    machine read may be an INPUT; the human translation stays the target).

    :param ctx: Run context.
    :param sources: Span sources with ``input: pgp_edition``.
    :param built: L5 document ids that reached the builder's document list.
    :returns: Candidates.
    """
    page_texts = {k: p.ocr_text for k, p in ctx.pages_by_key.items()}
    out = []
    for src in sources:
        if src.get("input") != "pgp_edition":
            continue
        for d in src.get("docs") or []:
            doc_id = f"book:{src['slug']}:{d['doc']}"
            span = (d.get("spans") or {}).get("translation")
            if doc_id in built or not span:
                continue
            raw, info = _bp.cut_span(span, page_texts)
            if raw is None:
                continue
            text, ops = _bp.normalise(raw, span.get("normalise") or [])
            target, cops = _bd.clean_target(text, f"translate_{src.get('target_lang', 'en')}")
            pages = info.get("pages") or []
            rec = {"pipeline": "translate_en_no_edition", "source": f"book:{src['slug']}", "doc": d["doc"],
                   "doc_id": doc_id, "shelfmark": (d.get("shelfmarks") or [None])[0], "target": target.strip(),
                   "target_raw": raw if (ops or cops) else None, "strip_ops": list(ops) + list(cops),
                   "lang_class": d.get("lang_class"), "note": d.get("note"),
                   "provenance": {"work": src.get("work"), "work_slug": _source_work([src], ctx.cfg)[src["slug"]][1],
                                  "page_keys": pages, "printed_pages": L.printed_pages(ctx.pages_by_key, pages),
                                  "licence": src.get("licence"), "licence_class": src.get("licence_class"),
                                  "extraction_method": "anthology_sections"},
                   "flags": ["no_pgp_edition_input"], "quality": {"score": 0.7}}
            L._finish(ctx, rec, d.get("shelfmarks") or [], [target])
            out.append(rec)
    return out


# ===================================================================================== VLM
def vlm_candidates(ctx: Any, dh_entries: Sequence[Dict[str, Any]], span_sources: Sequence[Mapping[str, Any]]
                   ) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    """All edition entries: doc-headed edition part, agent spans, web spans, generic numbered runs.

    :param ctx: Run context.
    :param dh_entries: Doc-headed entries (:func:`lit_run2.build_doc_headed`).
    :param span_sources: Agent-located span sources (with ``edition_sides``).
    :returns: ``(entries, drops)``.
    """
    drops: collections.Counter = collections.Counter()
    page_texts = {k: p.ocr_text for k, p in ctx.pages_by_key.items()}
    e1, d1 = L.span_vlm_entries({"sources": list(span_sources)}, page_texts, ctx.pages_by_key)
    drops.update(d1)
    web_spans = json.loads((ctx.p.repo / ctx.cfg["web_pairs_spans"]).read_text(encoding="utf-8")) \
        if ctx.cfg.get("web_pairs_spans") else {}
    lic = {d: work_meta(ctx.p.academic_literature, d, "_").get("licence") for d in ctx.cfg.get("web_dirs") or []}
    e2, d2 = L.web_edition_entries(web_spans, ctx.p.academic_literature, lic)
    drops.update({f"web:{k}": v for k, v in d2.items()})
    e3, d3 = L.generic_edition_entries(ctx.pages, ctx.index_rows, ctx.cfg.get("generic_vlm_works") or [])
    drops.update({f"generic:{k}": v for k, v in d3.items()})
    out = []
    for e in list(dh_entries) + e1 + e2 + e3:
        lines = [t["text"] for t in e["transcription_lines"]]
        if "quality" not in e:
            st = e["provenance"].get("clean_stats") or {"raw": len(lines), "kept": len(lines),
                                                        "short": sum(1 for x in lines if len(x.split()) <= 2)}
            q, fl = L.quality_flags_edition(lines, st)
            if e["provenance"].get("ocr_quality") == "poor":
                q, fl = max(0.0, q - 0.4), fl + ["ocr_quality:poor"]
            elif e["provenance"].get("ocr_quality") == "fair":
                q, fl = max(0.0, q - 0.15), fl + ["ocr_quality:fair"]
            if e["extraction_method"] == "generic_numbered_rtl_run":
                q, fl = max(0.0, q - 0.3), fl + ["auto_block_unreviewed", "document_attribution_by_proximity"]
            if e["extraction_method"] == "web_pairs_span_D15":
                fl = fl + ["coverage_unknown"]
            e["quality"] = {"score": round(q, 2), "flags": fl}
        if any(s == "unknown" for s in (t["side"] for t in e["transcription_lines"])):
            e["quality"]["flags"].append("side_unknown")
        rev = ctx.reversed_layer(f"{e['provenance'].get('work')}/{e['provenance'].get('work_slug')}") \
            if e["extraction_method"] == "generic_numbered_rtl_run" else None
        if rev:
            e["quality"]["flags"].append(rev)
            e["quality"]["score"] = 0.0
        e["doc_id"] = f"{e['source']}:{e['doc']}"
        if not e["provenance"].get("licence") and e["provenance"].get("work") and e["provenance"].get("work_slug"):
            meta = work_meta(ctx.p.academic_literature, e["provenance"]["work"], e["provenance"]["work_slug"])
            e["provenance"]["licence"] = meta["licence"]
        e["provenance"]["licence_class"] = e["provenance"].get("licence_class") or licence_class_of(e["provenance"].get("licence"))
        L._finish(ctx, e, e["shelfmarks"], ["\n".join(lines)])
        e["pipeline"] = "vlm_pages_arabic" if e["script"] == "arabic" else "vlm_pages_hebrew_ja"
        out.append(e)
    return out, drops


# ===================================================================================== catalogue
def catalogue_candidates(ctx: Any, weiss: Sequence[Dict[str, Any]], vlm: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Catalogue candidates (doc-headed catalogue part + numbered catalogues) with ``describe_pgp`` rows.

    :param ctx: Run context.
    :param weiss: Doc-headed catalogue candidates.
    :param vlm: Edition entries (the doc-headed work's own edition is the preferred input).
    :returns: Candidates.
    """
    own_ed = {e.get("catalogue_doc_id"): e for e in vlm if e.get("catalogue_doc_id")}
    out = list(weiss)
    for spec in ctx.cfg.get("catalogue") or []:
        wp = f"{spec['work']}/{spec['part']}"
        meta = work_meta(ctx.p.academic_literature, spec["work"], spec["part"])
        for r in L.catalogue_records(ctx.parts.get(wp) or [], spec):
            if len(r["text"]) < 60:
                continue
            c = {"pipeline": "catalogue_descriptions", "source": f"lit:{spec['slug']}", "doc": r["doc"],
                 "doc_id": f"lit:{spec['slug']}:{r['doc']}", "shelfmark": r["shelfmarks"][0], "description": r["text"],
                 "description_ops": r["ops"], "fields": ls.parse_description(r["text"]),
                 "provenance": {"work": spec["work"], "work_slug": spec["part"], "page_keys": r["page_keys"],
                                "printed_pages": L.printed_pages(ctx.pages_by_key, r["page_keys"]),
                                "licence": meta["licence"], "licence_class": licence_class_of(meta["licence"]),
                                "extraction_method": "numbered_catalogue_entries"},
                 "flags": list(r["flags"]) + ["hebrew_quotations_garbled_in_ocr"],
                 "quality": {"score": 0.6, "note": "PD catalogue; Hebrew incipits lost in the OCR text layer"}}
            L._finish(ctx, c, r["shelfmarks"], [r["text"]])
            out.append(c)
    for c in out:
        target, ops = _bd.clean_target(c["description"], "describe_pgp")
        target, _ = ls.scrub_stamps(target)
        c["target"] = target.strip()
        c["target_raw"] = c["description"] if ops else None
        c["strip_ops"] = ops
        res = c["resolution"]
        ed = own_ed.get(c["doc_id"])
        inp, src = None, None
        if ed and ed["quality"]["score"] >= 0.5:
            inp = "\n".join(t["text"] for t in ed["transcription_lines"])
            src = f"book_edition:{c['source'].split(':', 1)[1]}"
        elif res["resolved"] and not res["broad"] and not res.get("ambiguous"):
            got = ctx.resolver.pgp_edition(res["pgpids"])
            if got:
                inp, src = got[0], f"pgp_digital_edition:{got[1]}"
        c["input_source"] = src
        c["row"] = L.render_describe_row(c, inp, src) if (inp and len(c["target"]) >= 60) else None
    return out


# ===================================================================================== commentary
def commentary_candidates(ctx: Any, anth_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Commentary candidates: anthology introductions, single-fragment web posts, located discussion passages.

    :param ctx: Run context.
    :param anth_records: Anthology entry records (with ``intro_text``).
    :returns: Candidates with an ``analyse_document`` row where PGP holds an edition of the anchor.
    """
    cands: List[Dict[str, Any]] = []
    for r in anth_records:
        if not r.get("intro_text"):
            continue
        text, _ = _bp.normalise(r["intro_text"], ["drop_footnote_refs", "prose_paragraphs"])
        cands.append({"source": r["source"], "doc": r["doc"], "shelfmarks": [r["shelfmark_printed"]], "text": text,
                      "page_keys": r["page_keys"], "anchor_source": "anthology_entry_heading", "flags": [],
                      "provenance_base": r["provenance"], "lang": "en"})
    web_spans = json.loads((ctx.p.repo / ctx.cfg["web_pairs_spans"]).read_text(encoding="utf-8")) \
        if ctx.cfg.get("web_pairs_spans") else {}
    wcfg = json.loads((ctx.p.repo / ctx.cfg["web_pairs_config"]).read_text(encoding="utf-8")) \
        if ctx.cfg.get("web_pairs_config") else {}
    skip = {d: list(v) for d, v in (wcfg.get("skip_items") or {}).items()}
    for it in web_spans.get("items") or []:
        if any(d.get("not_placed") == "not_genizah" for d in it.get("docs") or []):
            skip.setdefault(it["dir"], []).append(it["slug"])
    dirs = ctx.cfg.get("commentary_post_dirs") or []
    parts = {k: v for k, v in ctx.parts.items() if k.split("/")[0] in dirs}
    titles = {k: work_meta(ctx.p.academic_literature, *k.split("/", 1)).get("title") for k in parts}
    hebrew_dirs = list(ctx.cfg.get("commentary_hebrew_dirs") or [])
    for c in L.web_commentary(parts, web_spans, dirs, titles, skip, hebrew_dirs):
        c["lang"] = "he" if c["source"].split(":", 1)[1] in hebrew_dirs else "en"
        cands.append(c)
    cands += L.index_passages(ctx.pages_by_key, ctx.index_rows, ctx.cfg.get("passage_works") or [])
    out = []
    for c in cands:
        wk, wslug = (c["page_keys"][0].split("/")[:2] if c.get("page_keys") else (None, None))
        meta = work_meta(ctx.p.academic_literature, wk, wslug) if wk else {"licence": None}
        prov = dict(c.get("provenance_base") or {})
        prov.update({"work": wk, "work_slug": wslug, "page_keys": c["page_keys"],
                     "printed_pages": L.printed_pages(ctx.pages_by_key, c["page_keys"]),
                     "licence": prov.get("licence") or meta.get("licence") or (ctx.cfg.get("dir_licence") or {}).get(wk, {}).get("licence"),
                     "extraction_method": c["anchor_source"]})
        prov["licence_class"] = prov.get("licence_class") or licence_class_of(prov["licence"])
        target, ops = _bd.clean_target(c["text"], "analyse_document")
        target, _ = ls.scrub_stamps(target)
        rec = {"pipeline": "analysis_commentary", "source": c["source"], "doc": c["doc"],
               "doc_id": f"{c['source']}:{c['doc']}", "shelfmark": c["shelfmarks"][0], "anchors": c["shelfmarks"],
               "title": c.get("title"), "commentary": c["text"], "target": target.strip(),
               "target_raw": c["text"] if ops else None, "strip_ops": ops, "lang": c.get("lang") or "en",
               "provenance": prov, "flags": list(c.get("flags") or [])}
        L._finish(ctx, rec, c["shelfmarks"], [target])
        res = rec["resolution"]
        if len(rec["target"]) < L.COMMENTARY_MIN_CHARS:
            rec["flags"].append("short_after_strip")
        score = 0.8 if c["anchor_source"] in ("anthology_entry_heading", "spans_file_commentary_docs") else 0.6
        if c["anchor_source"] == "literature_index_located_discuss":
            score = 0.45
        if res["broad"] or not res["resolved"]:
            score -= 0.3
        if any(f.startswith("long_target") or f.startswith("many_anchors") for f in rec["flags"]):
            score -= 0.15
        rev = ctx.reversed_layer(f"{wk}/{wslug}") if rec["lang"] != "en" else None
        if rev:
            rec["flags"].append(rev)
            score = 0.0
        rec["quality"] = {"score": round(max(0.0, score), 2)}
        inp, src = None, None
        if res["resolved"] and not res["broad"]:
            first = ctx.resolver.resolve(c["shelfmarks"][:1])
            got = ctx.resolver.pgp_edition(first["pgpids"] if not first["broad"] else [])
            if got:
                inp, src = got[0], f"pgp_digital_edition:{got[1]}"
        rec["input_source"] = src
        rec["row"] = L.render_analyse_row(rec, inp, src) if (inp and len(rec["target"]) >= L.COMMENTARY_MIN_CHARS
                                                               and len(rec["target"]) <= L.TARGET_MAX_CHARS) else None
        out.append(rec)
    return out


# ===================================================================================== images and gates
def _cids_of(rec: Mapping[str, Any]) -> List[str]:
    """Canonical ids of a candidate (none for over-broad matches; ids synthesised from the PGP fragment shelfmark
    when PGP ids resolved without a merged record, :func:`lit_run2.pgp_fragment_cids`).

    :param rec: Candidate or row.
    :returns: Ids.
    """
    res = rec.get("resolution") or {}
    if res.get("broad"):
        return []
    return list(res.get("canonical_ids") or []) or list(res.get("canonical_ids_synth") or [])


def attach_images(ctx: Any, groups: Mapping[str, Sequence[Dict[str, Any]]], extra_cids: Iterable[str] = ()) -> int:
    """Look up images for every candidate's canonical ids (read-only) and attach a summary.

    VLM entries keep the per-id image URLs (the page-row builder needs them); other candidates keep counts only.

    :param ctx: Run context.
    :param groups: pipeline → candidates (translation rows carry ``meta.run2``).
    :param extra_cids: Further ids to look up (literature index, for the inventory).
    :returns: Number of ids newly looked up.
    """
    allc: Set[str] = set(extra_cids)
    for recs in groups.values():
        for r in recs:
            allc |= set(_cids_of(r.get("meta", {}).get("run2") or r) if "meta" in r else _cids_of(r))
    n = ctx.images.fetch(allc)
    for name, recs in groups.items():
        for r in recs:
            if "meta" in r:
                run2 = r["meta"]["run2"]
                s = ctx.images.summary(run2.get("canonical_ids") or [])
                run2["images"] = {k: v for k, v in s.items() if k != "per_id"}
            else:
                s = ctx.images.summary(_cids_of(r))
                r["images"] = s if name.startswith("vlm") else {k: v for k, v in s.items() if k != "per_id"}
    return n


def gate(name: str, rec: Mapping[str, Any]) -> Tuple[bool, bool, List[str]]:
    """Hold-out and quality gate of one candidate.

    :param name: Pipeline name.
    :param rec: Candidate (or L5 row for translation files).
    :returns: ``(passes hold-out, passes quality gate, reasons)``; the quality gate implies the hold-out gate.
    """
    why: List[str] = []
    if "meta" in rec:
        m = rec["meta"]
        r2 = m["run2"]
        ho = not r2["held_out"] and m.get("split") == "train"
        if m.get("condition") != "edition":
            why.append(f"condition:{m.get('condition')}")
        if r2["quality"]["score"] < 0.6:
            why.append("score<0.6")
        if any(f.startswith("dup_existing") for f in r2["quality"]["flags"]):
            why.append("dup_existing_t27b")
        if not ho:
            why.append(f"split:{m.get('split')}")
        return ho, ho and not why, why
    ho = not rec.get("held_out")
    if not ho:
        why.append("held_out")
    q = (rec.get("quality") or {}).get("score", 0)
    res = rec.get("resolution") or {}
    unres = ("piece_unresolved_anchored" if res.get("anchors") and not res.get("resolved") else "unresolved_or_broad") \
        if (res.get("broad") or not res.get("resolved")) else None
    if name.endswith("_no_edition") or name.startswith("vlm"):
        if name.startswith("vlm") and q < 0.6:
            why.append("score<0.6")
        if not (rec.get("images") or {}).get("with_images"):
            why.append("no_served_images")
        if unres:
            why.append(unres)
        elif res.get("ambiguous"):
            why.append("piece_ambiguous")
        if name.endswith("_no_edition") and q < 0.5:
            why.append("score<0.5")
    else:
        if not rec.get("row"):
            why.append("no_input_edition")
        if q < 0.5:
            why.append("score<0.5")
        if unres:
            why.append(unres)
    return ho, ho and not why, why


# ===================================================================================== writing
#: Prose fields of candidate records that are scrubbed of project names (identifier fields such as shelfmarks are not).
PROSE_FIELDS: Tuple[str, ...] = ("target", "target_raw", "commentary", "description", "title", "note")


def scrub_record(rec: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove scrubbed project names / hosts (:func:`textnorm.strip_source_names`) from a record's prose fields, the
    way the t27b rows are scrubbed (F4); rows (``meta`` present) go through :func:`rows.scrub_meta` and their messages
    are scrubbed too.

    :param rec: Candidate record or row.
    :returns: The scrubbed copy.
    """
    from src.finetuning.text27b import textnorm as tn
    out = json.loads(json.dumps(rec, ensure_ascii=False, default=list))
    strip = (lambda t: tn.strip_source_names(t)[0] if isinstance(t, str) else t)
    if "meta" in out:
        for m in out.get("messages") or []:
            m["content"] = strip(m["content"])
        _rows.scrub_meta(out)
        if isinstance(out["meta"].get("run2"), dict) and out["meta"]["run2"].get("note"):
            out["meta"]["run2"]["note"] = strip(out["meta"]["run2"]["note"])
        return out
    for k in PROSE_FIELDS:
        if isinstance(out.get(k), str):
            out[k] = strip(out[k])
    if isinstance(out.get("row"), dict):
        out["row"] = scrub_record(out["row"])
    if isinstance(out.get("provenance"), dict) and isinstance(out["provenance"].get("note"), str):
        out["provenance"]["note"] = strip(out["provenance"]["note"])
    return out


def _write_jsonl(path: Path, recs: Sequence[Mapping[str, Any]]) -> int:
    """Write records as JSONL (UTF-8, one object per line; prose scrubbed of project names, :func:`scrub_record`).

    :param path: Destination.
    :param recs: Records.
    :returns: Bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(scrub_record(r), ensure_ascii=False, default=list) + "\n")
    return path.stat().st_size


def write_pipeline_files(ctx: Any, groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Write one JSONL per pipeline (large files to the NAS with a pointer file in the run directory).

    :param ctx: Run context.
    :param groups: pipeline → records.
    :returns: pipeline → ``{"path", "bytes", "records"}``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    limit = int(ctx.cfg.get("large_file_mb", 50)) * 1_000_000
    for name, recs in groups.items():
        path = ctx.out / f"{name}.jsonl"
        n = _write_jsonl(path, recs)
        if n > limit and ctx.cfg.get("nas_dir") and Path(ctx.cfg["nas_dir"]).parent.exists():
            nas = Path(ctx.cfg["nas_dir"]) / path.name
            _write_jsonl(nas, recs)
            path.unlink()
            path.with_suffix(".jsonl.on_nas").write_text(str(nas) + "\n", encoding="utf-8")
            path = nas
        out[name] = {"path": str(path), "bytes": n, "records": len(recs)}
    return out


def _clip(s: Optional[str], n: int) -> str:
    """Shorten a text for the samples document.

    :param s: Text.
    :param n: Maximum characters.
    :returns: The text, cut with an ellipsis marker.
    """
    s = s or ""
    return s if len(s) <= n else s[:n] + " […]"


def samples_md(groups: Mapping[str, Sequence[Mapping[str, Any]]], n: int = SAMPLE_N, seed: int = 3407) -> str:
    """Hand-check samples: ``n`` seeded records per pipeline with source text and target.

    :param groups: pipeline → records.
    :param n: Records per pipeline.
    :param seed: Random seed.
    :returns: Markdown.
    """
    lines = ["# Run-2 literature candidates — samples for review (2026-10-27)", "",
             "INTERNAL USE ONLY: edition and translation text from in-copyright works; do not publish or paste outside.",
             f"Up to {n} seeded records per pipeline (train-eligible first, then the rest). Texts are clipped.", ""]
    rng = random.Random(seed)
    for name, recs in groups.items():
        recs = list(recs)
        ok = [r for r in recs if gate(name, r)[1]]
        rest = [r for r in recs if not gate(name, r)[1]]
        pick = rng.sample(ok, min(n, len(ok)))
        if len(pick) < n:
            pick += rng.sample(rest, min(n - len(pick), len(rest)))
        lines += [f"## {name} ({len(pick)} of {len(recs)})", ""]
        for i, r in enumerate(pick, 1):
            if "meta" in r:
                m = r["meta"]
                r2 = m["run2"]
                head = (f"### {i}. {m.get('doc_id')} — {', '.join(r2['shelfmark_raw'])} — split {m.get('split')} — "
                        f"{m.get('condition')} — q {r2['quality']['score']} {r2['quality']['flags']}")
                src_txt = r["messages"][1]["content"].split("<document>", 1)[-1].split("</document>")[0]
                lines += [head, "", "**source (input)**", "", "```", _clip(src_txt.strip(), 900), "```", "",
                          "**target**", "", "```", _clip(r["messages"][2]["content"], 900), "```", ""]
            elif name.startswith("vlm"):
                txt = "\n".join(f"[{t['side']}] {t['text']}" for t in r["transcription_lines"])
                im = r.get("images") or {}
                lines += [f"### {i}. {r['doc_id']} — {r['shelfmark']} — {r['extraction_method']} — q {r['quality']['score']} "
                          f"{r['quality']['flags']} — held_out {r.get('held_out')} — ids {r['resolution']['canonical_ids'][:3]} "
                          f"— images {im.get('n_images', 0)}", "", "```", _clip(txt, 900), "```", ""]
            else:
                body = r.get("description") or r.get("commentary") or r.get("target")
                row = r.get("row")
                inp = row["messages"][1]["content"].split("<document>", 1)[-1].split("</document>")[0] if row else None
                lines += [f"### {i}. {r['doc_id']} — {r.get('shelfmark')} — {r['provenance'].get('extraction_method')} — "
                          f"q {(r.get('quality') or {}).get('score')} — held_out {r.get('held_out')} — pgpids "
                          f"{r['resolution']['pgpids'][:4]} — input {r.get('input_source')}", ""]
                if inp:
                    lines += ["**input (edition)**", "", "```", _clip(inp.strip(), 500), "```", ""]
                lines += ["**target**", "", "```", _clip(r.get("target") or body, 1100), "```", ""]
    return "\n".join(lines)


def build_all(ctx: Any, with_book_pairs: bool = True) -> Dict[str, Any]:
    """Run every extractor, look up images, gate, and write the outputs.

    :param ctx: Run context (:class:`lit_run2.Run2Context`).
    :param with_book_pairs: Run the L5 builder on the span sources.
    :returns: Summary.
    """
    from src.finetuning.text27b import lit_run2_report as R
    cats_dh, vlm_dh, rf_sources = L.build_doc_headed(ctx)
    anth_sources, anth_records = L.build_anthologies(ctx)
    span_sources: List[Dict[str, Any]] = []
    for f in ctx.cfg.get("span_files") or []:
        span_sources += json.loads((ctx.p.repo / f).read_text(encoding="utf-8")).get("sources") or []
    sources = rf_sources + anth_sources + span_sources
    (ctx.out / "spans").mkdir(parents=True, exist_ok=True)
    (ctx.out / "spans" / "programmatic_spans.json").write_text(
        json.dumps({"sources": rf_sources + anth_sources}, ensure_ascii=False, indent=1), encoding="utf-8")
    tr_rows: List[Dict[str, Any]] = []
    tr_only: List[Dict[str, Any]] = []
    an_rows: List[Dict[str, Any]] = []
    bp_docs: List[Dict[str, Any]] = []
    bp_report: Dict[str, Any] = {}
    if with_book_pairs:
        bpo = L.run_book_pairs(ctx, sources, ctx.out / "book_pairs_run2")
        rows = list(_rows.read_jsonl(ctx.out / "book_pairs_run2" / "families" / "book_pairs.jsonl"))
        bp_docs = list(bpo["result"].get("docs") or [])
        bp_report = bpo["result"].get("report") or {}
        tr_rows, an_rows = translation_rows(ctx, rows, bp_docs, sources)
    built = {d.get("doc_id") for d in bp_docs}
    tr_only = translation_only_candidates(ctx, anth_sources, built)
    vlm, vdrops = vlm_candidates(ctx, vlm_dh, span_sources)
    cats = catalogue_candidates(ctx, cats_dh, vlm)
    comm = commentary_candidates(ctx, anth_records)
    for r in an_rows:
        r["meta"]["run2"]["kind"] = "l5_book_pairs_analyse_row"
    groups: Dict[str, List[Dict[str, Any]]] = {
        "translate_en": [r for r in tr_rows if r["meta"]["family"] == "translate_en"],
        "translate_he": [r for r in tr_rows if r["meta"]["family"] == "translate_he"],
        "translate_en_no_edition": tr_only,
        "vlm_pages_hebrew_ja": [e for e in vlm if e["pipeline"] == "vlm_pages_hebrew_ja"],
        "vlm_pages_arabic": [e for e in vlm if e["pipeline"] == "vlm_pages_arabic"],
        "catalogue_descriptions": cats,
        "analysis_commentary": comm,
    }
    index_cids = {c for r in ctx.index_rows for c in (r.get("merged_doc_ids") or [])}
    looked = attach_images(ctx, {**groups, "l5_analyse": an_rows}, index_cids)
    for name, recs in groups.items():
        for r in recs:
            ho, ok, why = gate(name, r)
            tgt = r["meta"]["run2"] if "meta" in r else r
            tgt["gate"] = {"holdout_ok": ho, "quality_ok": ok, "reasons": why}
    files = write_pipeline_files(ctx, {**groups, "analysis_l5_rows": an_rows})
    safety = R.safety_scan(ctx.out, files)
    summary = R.summarise(ctx, groups, an_rows, bp_docs, bp_report, vdrops, looked, files, safety)
    (ctx.out / "INVENTORY.md").write_text(R.inventory_md(ctx, groups, an_rows), encoding="utf-8")
    (ctx.out / "SAMPLES.md").write_text(samples_md({k: [scrub_record(r) for r in v] for k, v in groups.items()}),
                                        encoding="utf-8")
    (ctx.out / "REPORT.md").write_text(R.report_md(ctx, summary), encoding="utf-8")
    return summary
