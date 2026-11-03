# File name: lit_run2_report.py
# Date: 10/27/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Summary, safety scan, INVENTORY.md and REPORT.md of the run-2 literature mining pass (:mod:`lit_run2`).

The documents are written into the git-ignored run directory; they name works and shelfmarks (internal notes) and
must not be copied to a public surface.

Public API: :data:`PIPELINES`, :func:`safety_scan`, :func:`counts`, :func:`summarise`, :func:`inventory_md`,
:func:`report_md`.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from src.finetuning.text27b import lit_segment as ls
from src.finetuning.text27b import textnorm as tn

PIPELINES: Tuple[str, ...] = ("translate_en", "translate_he", "translate_en_no_edition", "vlm_pages_hebrew_ja",
                              "vlm_pages_arabic", "catalogue_descriptions", "analysis_commentary")
_SHORT = {"translate_en": "tr-en", "translate_he": "tr-he", "vlm_pages_hebrew_ja": "vlm-he/ja",
          "vlm_pages_arabic": "vlm-ar", "catalogue_descriptions": "cat", "analysis_commentary": "ana"}


def safety_scan(out: Path, files: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Count stamp / IP-address / project-name hits in every written pipeline file.

    :param out: Run directory.
    :param files: Output of :func:`lit_run2_build.write_pipeline_files`.
    :returns: file → ``{"ip_lines", "stamp_lines", "project_name_lines"}`` (project names in prose fields / messages).
    """
    res: Dict[str, Dict[str, int]] = {}
    for name, f in files.items():
        c = {"ip_lines": 0, "stamp_lines": 0, "project_name_lines": 0}
        with open(f["path"], encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                rec.pop("images", None)
                if "meta" in rec:
                    rec["meta"].get("run2", {}).pop("images", None)
                text = json.dumps(rec, ensure_ascii=False)
                c["ip_lines"] += int(ls.has_ip_address(re.sub(r"\d+\.\d+\.\d+-\d+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?=\.jpg)", "", text)))
                c["stamp_lines"] += int(any(ls.is_stamp_line(x) for x in text.split("\\n")))
                prose = [rec.get(k) for k in ("target", "target_raw", "commentary", "description")] + \
                    [m.get("content") for m in rec.get("messages") or []]
                c["project_name_lines"] += int(any(tn.has_source_names(x) for x in prose if isinstance(x, str)))
        res[name] = c
    return res


def _tgt(rec: Mapping[str, Any]) -> Mapping[str, Any]:
    """The record carrying provenance / gate fields (``meta.run2`` for rows).

    :param rec: Candidate or row.
    :returns: The mapping.
    """
    return rec["meta"]["run2"] if "meta" in rec else rec


def _work_of(rec: Mapping[str, Any]) -> str:
    """``<work>/<work slug>`` of a candidate.

    :param rec: Candidate or row.
    :returns: Key.
    """
    t = _tgt(rec)
    prov = t if "meta" in rec else (rec.get("provenance") or {})
    return f"{prov.get('work')}/{prov.get('work_slug')}"


def counts(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Dict[str, int]]:
    """Candidates / after hold-out / after quality gate per pipeline (translation: edition rows only for the gate
    columns; all conditions in ``rows``).

    :param groups: pipeline → records.
    :returns: pipeline → counts.
    """
    out = {}
    for name, recs in groups.items():
        if recs and "meta" in recs[0]:
            ed = [r for r in recs if r["meta"].get("condition") == "edition"]
            docs = {r["meta"].get("doc_id") for r in ed}
            out[name] = {"rows_all_conditions": len(recs), "candidates": len(ed), "documents": len(docs),
                         "after_holdout": sum(1 for r in ed if _tgt(r)["gate"]["holdout_ok"]),
                         "after_quality_gate": sum(1 for r in ed if _tgt(r)["gate"]["quality_ok"])}
        else:
            out[name] = {"candidates": len(recs),
                         "after_holdout": sum(1 for r in recs if _tgt(r)["gate"]["holdout_ok"]),
                         "after_quality_gate": sum(1 for r in recs if _tgt(r)["gate"]["quality_ok"]),
                         "with_row": sum(1 for r in recs if r.get("row")) if not name.startswith("vlm") else None}
    return out


def summarise(ctx: Any, groups: Mapping[str, Sequence[Mapping[str, Any]]], an_rows: Sequence[Mapping[str, Any]],
              bp_docs: Sequence[Mapping[str, Any]], bp_report: Mapping[str, Any], vdrops: Mapping[str, int],
              looked: int, files: Mapping[str, Any], safety: Mapping[str, Any]) -> Dict[str, Any]:
    """Machine-readable summary of the run.

    :param ctx: Run context.
    :param groups: pipeline → records.
    :param an_rows: L5 analysis rows.
    :param bp_docs: L5 document records.
    :param bp_report: L5 report.
    :param vdrops: VLM extraction drops.
    :param looked: Image ids looked up in this run.
    :param files: Written files.
    :param safety: Safety scan.
    :returns: Summary dict.
    """
    held: List[Dict[str, Any]] = []
    gate_reasons: Dict[str, collections.Counter] = {}
    per_work: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for name, recs in groups.items():
        gr = collections.Counter()
        for r in recs:
            t = _tgt(r)
            if "meta" in r and r["meta"].get("condition") != "edition":
                continue
            per_work[_work_of(r)][name] += 1
            if t["gate"]["quality_ok"]:
                per_work[_work_of(r)][name + ":ok"] += 1
            gr.update(t["gate"]["reasons"] or ["ok"])
            if t.get("held_out"):
                res = t if "meta" in r else r.get("resolution", {})
                held.append({"pipeline": name, "doc_id": r["meta"].get("doc_id") if "meta" in r else r.get("doc_id"),
                             "shelfmark": (t.get("shelfmark_raw") if "meta" in r else r.get("shelfmark")),
                             "pgpids": (res.get("pgpids") or [])[:6], "reason": t.get("held_out_reason")})
        gate_reasons[name] = gr
    l5_drops = collections.Counter()
    for d in bp_docs:
        for k in ("translate", "analyse"):
            v = d.get(k) or {}
            if isinstance(v, dict) and v.get("drop"):
                l5_drops[f"{k}:{v['drop']}"] += 1
    return {"counts": counts(groups), "l5_analyse_rows": len(an_rows),
            "l5_analyse_train": sum(1 for r in an_rows if r["meta"].get("split") == "train" and not r["meta"]["run2"]["held_out"]),
            "l5_docs": len(bp_docs), "l5_drops": dict(l5_drops), "l5_report_drops": (bp_report or {}).get("drops"),
            "gate_reasons": {k: dict(v) for k, v in gate_reasons.items()}, "held_out": held,
            "per_work": {k: dict(v) for k, v in per_work.items()}, "vlm_drops": dict(vdrops),
            "image_ids_looked_up": looked, "image_cache_ids": len(ctx.images.data), "files": dict(files),
            "safety_scan": dict(safety), "es_index": ctx.cfg.get("es_index"), "config": ctx.cfg.get("_path")}


def _work_rows(ctx: Any, groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """One inventory row per work part.

    :param ctx: Run context.
    :param groups: pipeline → records.
    :returns: Rows.
    """
    from src.finetuning.text27b import lit_run2_build as B
    cand_cids: Dict[str, Set[str]] = collections.defaultdict(set)
    per: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for name, recs in groups.items():
        for r in recs:
            if "meta" in r and r["meta"].get("condition") != "edition":
                continue
            w = _work_of(r)
            per[w][name] += 1
            if _tgt(r)["gate"]["quality_ok"]:
                per[w][name + ":ok"] += 1
            res = _tgt(r) if "meta" in r else r.get("resolution") or {}
            if not res.get("broad"):
                cand_cids[w] |= set(res.get("canonical_ids") or [])
    idx_cids: Dict[str, Set[str]] = collections.defaultdict(set)
    for r in ctx.index_rows:
        if r.get("located_in_text") and r.get("merged_doc_ids") and len(r["merged_doc_ids"]) <= 3:
            idx_cids[f"{r['book']}/{r['part'].split('/')[0]}"] |= set(r["merged_doc_ids"])
    rows = []
    for wp, pgs in sorted(ctx.parts.items()):
        work, part = wp.split("/", 1)
        meta = B.work_meta(ctx.p.academic_literature, work, part)
        conf = (ctx.cfg.get("works") or {}).get(wp) or (ctx.cfg.get("dir_defaults") or {}).get(work) or {}
        cids = idx_cids.get(wp, set()) | cand_cids.get(wp, set())
        with_img = sum(1 for c in cids if ctx.images.n_images(c))
        he = sum(ls.letters(p.ocr_text or p.full_main_text or "")[0] for p in pgs)
        ar = sum(ls.letters(p.ocr_text or p.full_main_text or "")[1] for p in pgs)
        text = "\n".join(p.ocr_text or p.full_main_text or "" for p in pgs)
        health = []
        wr, lr = ls.word_order_reversal(text), ls.reversed_hebrew_share(text)
        if wr is not None and wr > 0.3:
            health.append(f"HEBREW WORD ORDER REVERSED ({wr})")
        if lr is not None and lr > 0.3:
            health.append(f"HEBREW LETTERS REVERSED/GARBLED ({lr})")
        rows.append({"work": work, "part": part, "title": meta.get("title"), "year": meta.get("year"),
                     "health": health,
                     "lang": conf.get("lang") or meta.get("language"), "contains": conf.get("contains", ""),
                     "pipelines": conf.get("pipelines", []), "note": conf.get("quality_note", ""), "pages": len(pgs),
                     "he_letters": he, "ar_letters": ar, "resolved_ids": len(cids), "with_images": with_img,
                     "per": dict(per.get(wp, {})), "licence": meta.get("licence")})
    return rows


def _cell(per: Mapping[str, int], name: str) -> str:
    """``candidates/ok`` cell.

    :param per: Counter of the work.
    :param name: Pipeline.
    :returns: Cell text ('' when zero).
    """
    n = per.get(name, 0)
    return f"{n}/{per.get(name + ':ok', 0)}" if n else ""


def inventory_md(ctx: Any, groups: Mapping[str, Sequence[Mapping[str, Any]]], an_rows: Sequence[Mapping[str, Any]]) -> str:
    """INVENTORY.md: one row per work with language, contents, resolved shelfmarks, images and candidates.

    :param ctx: Run context.
    :param groups: pipeline → records.
    :param an_rows: L5 analysis rows (counted per work in the notes).
    :returns: Markdown.
    """
    rows = _work_rows(ctx, groups)
    l5 = collections.Counter(f"{r['meta']['run2'].get('work')}/{r['meta']['run2'].get('work_slug')}" for r in an_rows)
    head = ["# Run-2 literature inventory (2026-10-27 ingest)", "",
            "INTERNAL: names holdings and shelfmarks; keep off public surfaces.", "",
            "Columns: **ids** = distinct canonical ids our corpus resolves from shelfmarks LOCATED in the work's text "
            "(literature index, ≤ 3 ids per mention) or from the extracted candidates; **img** = how many of those have "
            f"images in the served fragment index (read-only lookup, `{ctx.cfg.get('es_index')}`). Pipeline cells = "
            "`candidates/after quality gate` (translation: edition rows). `l5-ana` = analysis rows the L5 builder made "
            "from located introductions.", ""]
    hdr = ("| work | lang | contains | pp | HE/AR letters | ids | img | tr-en | tr-he | tr-en no-ed | vlm-he/ja | vlm-ar | "
           "cat | ana | l5-ana | can feed | quality note |")
    sep = "|" + "---|" * 17
    out = head + ["## Downloads of 2026-10-27 (JSTOR, ProQuest, book)", "", hdr, sep]
    main = [r for r in rows if r["work"] in ("dissertations", "jstor_alumni", "cohen_2005_voice_of_the_poor")]
    web = [r for r in rows if r not in main]

    def line(r: Mapping[str, Any]) -> str:
        """Table row.

        :param r: Inventory row.
        :returns: Markdown line.
        """
        per = r["per"]
        t = (r["title"] or r["part"]).replace("|", "/")
        return (f"| `{r['part']}`<br>{t[:70]} | {r['lang'] or ''} | {r['contains'][:160]} | {r['pages']} | "
                f"{r['he_letters']}/{r['ar_letters']} | {r['resolved_ids']} | {r['with_images']} | "
                + " | ".join(_cell(per, n) for n in ("translate_en", "translate_he", "translate_en_no_edition",
                                                     "vlm_pages_hebrew_ja", "vlm_pages_arabic", "catalogue_descriptions",
                                                     "analysis_commentary"))
                + f" | {l5.get(r['work'] + '/' + r['part'], '') or ''} | {', '.join(r['pipelines'])} | "
                + f"{'; '.join(r['health'] + ([r['note'][:260]] if r['note'] else []))} |")

    out += [line(r) for r in main]
    by_dir: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in web:
        by_dir[r["work"]].append(r)
    out += ["", "## Web batch (fetched 10-26, materialised 10-27; translation pairs already mined into t27b web_pairs)", ""]
    dd = ctx.cfg.get("dir_defaults") or {}
    out += ["| source dir | works | pp | ids | img | vlm-he/ja | vlm-ar | cat | ana | can feed | quality note |",
            "|" + "---|" * 11]
    for d, rs in sorted(by_dir.items()):
        agg = collections.Counter()
        for r in rs:
            agg.update(r["per"])
        conf = dd.get(d, {})
        out.append(f"| `{d}` | {len(rs)} | {sum(r['pages'] for r in rs)} | {sum(r['resolved_ids'] for r in rs)} | "
                   f"{sum(r['with_images'] for r in rs)} | {_cell(agg, 'vlm_pages_hebrew_ja')} | {_cell(agg, 'vlm_pages_arabic')} | "
                   f"{_cell(agg, 'catalogue_descriptions')} | {_cell(agg, 'analysis_commentary')} | "
                   f"{', '.join(conf.get('pipelines', []))} | {conf.get('quality_note', '')} |")
    out += ["", "### Web works, one row each", "", hdr, sep]
    for d, rs in sorted(by_dir.items()):
        out += [line(r) for r in rs]
    return "\n".join(out) + "\n"


def report_md(ctx: Any, s: Mapping[str, Any]) -> str:
    """REPORT.md: counts per pipeline, hold-out collisions, what needs the user's eyes, re-run commands, caveats.

    :param ctx: Run context.
    :param s: Summary (:func:`summarise`).
    :returns: Markdown (the hand-written findings in ``REPORT_NOTES.md`` of the run directory, when present, are
        placed before the generated tables, so a re-run keeps them).
    """
    c = s["counts"]
    out = ["# Run-2 literature mining — report (2026-10-27)", "",
           "Generated by `src/finetuning/text27b/lit_run2.py` (CPU parsing + read-only Elasticsearch lookups; no LM "
           "Studio, no paid API, no Hub, no mixture or eval-manifest change). INTERNAL.", "",
           "## Counts per pipeline", "",
           "| pipeline | candidates | after hold-out | after quality gate | notes |", "|---|---:|---:|---:|---|"]
    for name in PIPELINES:
        v = c.get(name, {})
        note = ""
        if "documents" in v:
            note = f"{v.get('documents', 0)} documents; {v.get('rows_all_conditions', 0)} rows incl. noised / rescript twins"
        elif name.endswith("_no_edition"):
            note = "human translation, no PGP edition: gate = resolved + served images (read/image-conditioned family)"
        elif v.get("with_row") is not None:
            note = f"{v.get('with_row')} with a rendered row (input edition found)"
        out.append(f"| {name} | {v.get('candidates', 0)} | {v.get('after_holdout', 0)} | {v.get('after_quality_gate', 0)} | {note} |")
    out += ["", f"L5 analysis rows from located introductions (book_pairs `analyse_document`): {s['l5_analyse_rows']} "
                f"({s['l5_analyse_train']} train, not held).", ""]
    out += ["## Gate reasons (why candidates fail the quality gate)", ""]
    for name, gr in s["gate_reasons"].items():
        out.append(f"* **{name}**: " + ", ".join(f"{k} {v}" for k, v in sorted(gr.items(), key=lambda x: -x[1])))
    out += ["", "## L5 builder drops (documents)", "", "```", json.dumps(s["l5_drops"], indent=1), "```", "",
            f"## Hold-out collisions ({len(s['held_out'])} candidates)", "",
            "| pipeline | doc | shelfmark | pgpids | reason |", "|---|---|---|---|---|"]
    for h in s["held_out"][:400]:
        sm = h["shelfmark"]
        sm = ", ".join(sm) if isinstance(sm, list) else (sm or "")
        out.append(f"| {h['pipeline']} | {h['doc_id']} | {sm[:60]} | {', '.join(h['pgpids'])} | {h['reason']} |")
    out += ["", "## Safety scan of the written files", "", "```", json.dumps(s["safety_scan"], indent=1), "```", "",
            "## Files", ""]
    for name, f in s["files"].items():
        out.append(f"* `{name}`: {f['records']} records, {f['bytes'] / 1e6:.1f} MB → `{f['path']}`")
    notes = Path(ctx.out) / "REPORT_NOTES.md"
    if notes.exists():
        out = out[:2] + ["", notes.read_text(encoding="utf-8").strip(), "", "---", "", "# Generated tables", ""] + out[2:]
    return "\n".join(out) + "\n"
