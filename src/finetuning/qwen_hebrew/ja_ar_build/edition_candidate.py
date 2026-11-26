"""Prepare private, source-reviewed edition pairs for the normal page builder.

The source ledger indexes *all* printed lines. A placement alone never approves a
page. A review binds the final answer, visible image, PDF and exact line order.
The independent release audit remains a separate operation on the frozen export.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from . import approval, build_rows as br, textnorm
from .edition_assembly import view_key, write_records
from .edition_lines import digest


def key(row: dict) -> tuple:
    """Identify one edition and exact photograph transformation.

    :param row: A view or its exposure record.
    :return: Edition ID and the crop-then-rotate identity.
    """
    return (row["row_id"], *view_key(row))


def index_unique(rows: list[dict], field: str) -> dict:
    """Index records and reject an ambiguous ID.

    :param rows: Input records.
    :param field: Unique field.
    :return: Indexed input.
    """
    result = {r[field]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {field}")
    return result


def candidate(edition: dict, original: dict) -> tuple[dict, dict]:
    """Index source-bearing lines by printed PDF page, including currently unplaced text.

    :param edition: Physical baseline ledger.
    :param original: Staged historical document identity and photographs.
    :return: Normal builder candidate and mapping from physical line ID to side and line index.
    """
    if original["document_group"] != edition["document_group"] or original["source"] not in br.LEGACY_STORE_SOURCES:
        raise ValueError(f"Edition identity mismatch for {edition['row_id']}")
    if Path(edition["pdf_path"]).is_file() and br.file_sha256(edition["pdf_path"]) != edition["pdf_sha256"]:
        raise ValueError(f"Changed edition PDF for {edition['row_id']}")
    targets, refs = {}, {}
    for line in edition["lines"]:
        if not line["text"] or not line["text"].strip():
            continue
        side = f"pdf{line['pdf_page']}"
        target = targets.setdefault(side, {"side_key": side, "lines": [], "line_ids": []})
        line_id = line["line_id"]
        if line_id in refs or digest(line["raw"]) != line["raw_sha256"]:
            raise ValueError(f"Invalid printed-line provenance for {line_id}")
        refs[line_id] = [side, len(target["lines"])]
        target["lines"].append(line["text"])
        target["line_ids"].append(line_id)
    for side in targets.values():
        side["text"] = "\n".join(side["lines"])
        side["pointer"] = {"file": Path(edition["pdf_path"]).name, "sha256": edition["pdf_sha256"]}
    return {**original, "targets": list(targets.values())}, refs


def make_pair(view: dict, edition: dict, cand: dict, refs: dict, check: dict) -> dict:
    """Translate placed ledger lines into the ordinary builder contract.

    :param view: Proposed view carrying its screened target and source line IDs.
    :param edition: Edition ledger.
    :param cand: Staged edition candidate.
    :param refs: Physical baseline index.
    :param check: Exact transformed-image and text exposure evidence.
    :return: Unapproved assembly pair.
    """
    ids = view["draft_line_ids"]
    placed = [line for line in edition["lines"] if line["placement"] and view_key(line["placement"]) == view_key(view)]
    readable = [line for line in placed if (line["text"] or "").strip()]
    unread = [line["line_id"] for line in placed if not (line["text"] or "").strip()]
    if ([l["line_id"] for l in placed] != view["line_ids"] or
            [l["line_id"] for l in readable] != ids or unread != view["placed_without_reading"]
            or len(set(view["line_ids"])) != len(placed)):
        raise ValueError(f"Placed line IDs differ for {key(view)}")
    lines = [line["text"] for line in readable]
    expected = "\n".join(lines)
    photos = {(p.get("sha256"), p.get("path")) for p in cand["images"]}
    if ((view["image_sha256"], view["image_path"]) not in photos or key(check) != key(view)
            or check["view_sha256"] != view["review_image_sha256"] or view["draft_text"] != expected
            or digest(expected) != view["draft_sha256"] or check["draft_sha256"] != view["draft_sha256"]
            or view["document_group"] != cand["document_group"] or check["document_group"] != cand["document_group"]
            or view["source"] != cand["source"]):
        raise ValueError(f"Stale photograph, document or source text for {key(view)}")
    line_refs = [refs[i] for i in ids]
    vid = digest(json.dumps(key(view)))[:20]
    return {**{k: view[k] for k in ("row_id", "document_group", "source", "script", "image_sha256",
                                      "image_path", "crop", "rotation", "flags")},
            "pair_id": f"{view['row_id']}|edition|{vid}", "lines": lines, "line_refs": line_refs,
            "side_keys": list(dict.fromkeys(ref[0] for ref in line_refs)),
            "remedies": ["assemble"] + (["crop"] if view["crop"] else []),
            "decision": "review", "reasons": ["source_review_pending"],
            "open_lines": ([{"lines": view["placed_without_reading"]}] if view["placed_without_reading"] else []),
            "evidence": {"target_letters": textnorm.letters(expected)},
            "edition": {"pdf_sha256": edition["pdf_sha256"], "line_ids": ids,
                        "view_sha256": check["view_sha256"], "documentary_lead": view["documentary_catalogue_lead"]}}


def review_key(pair: dict, answer: br.Answer, cands: dict) -> str:
    """Bind a source inspection to the exact target, source lines and delivered photo bytes.

    :param pair: Corrected pair.
    :param answer: What the row builder would actually write.
    :param cands: Canonical candidates.
    :return: Review digest.
    """
    return approval.edition_review_key(pair, br.pair_fingerprint(pair, answer, cands))


def apply_review(pair: dict, answer: br.Answer, record: dict, cands: dict, models: dict, mirrored: set) -> None:
    """Apply a scope-specific photo-and-edition inspection before release sampling.

    :param pair: Pair to update.
    :param answer: Final builder answer seen by reviewer.
    :param record: Signed-by-content inspection disposition and concrete observation.
    :param cands: Corrected candidate table.
    :param models: Text tools used by the row builder.
    :param mirrored: Mirrored edition keys.
    """
    if (record["review_key"] != review_key(pair, answer, cands) or not record.get("observation")
            or not record.get("reviewer") or record["decision"] not in {"complete", "hold"}):
        raise ValueError(f"Invalid or stale inspection for {pair['pair_id']}")
    if record["decision"] == "hold":
        pair["reasons"] = ["source_inspection_hold"]
        return
    if answer.fails or not answer.letters_kept:
        raise ValueError(f"Source text changed in final answer for {pair['pair_id']}")
    for issue in pair["open_lines"]:
        issue["settled"] = record["observation"]
    reason = br.stamp_approval(pair, cands, models, mirrored, "source_review", answer_seen=True,
                               review_key=record["review_key"], reviewer=record["reviewer"])
    if reason:
        raise ValueError(f"{pair['pair_id']}: {reason}")
    pair.update(decision="accept", reasons=[])


def prepare(args: argparse.Namespace, stage: Path) -> dict:
    """Write canonical candidates, pairs, holdout decisions and exact final-answer review queue.

    :param args: Paths to source, exposure and review evidence.
    :param stage: Fresh private directory.
    :return: Counts and input digests.
    """
    from . import holdout_v2 as h
    views = br.load_jsonl(args.views)
    checks = br.load_jsonl(args.checks)
    exposure = {key(c): c for c in checks}
    positive = ("cached_exact_photo", "cached_photo_hits", "previous_export_near_photo", "open_eval_text",
                "historical_text", "novel_v2_text", "historical_photo", "novel_v2_photo")
    held = {c["document_group"] for c in checks if c["group_held"] or any(c[col] for col in positive)}
    held.update(v["document_group"] for v in views if v["exposure_status"] == "manuscript_overlap_hold")
    active = [v for v in views if v["exposure_status"] == "passed_completed_checks"]
    if (len(exposure) != len(checks) or len({key(v) for v in active}) != len(active)
            or any(v["document_group"] in held for v in active)):
        raise ValueError("Conflicting exposure records or duplicate proposed view")
    editions = index_unique(br.load_jsonl(args.editions), "row_id")
    wanted = {v["row_id"] for v in active}
    originals = index_unique([c for c in br.load_jsonl(args.candidates) if c["row_id"] in wanted], "row_id")
    old_verdicts = {(v["row_id"], v["side_key"]): v for v in br.load_jsonl(args.verdicts)}
    checks_by_rid = defaultdict(list)
    for v in active:
        checks_by_rid[v["row_id"]].append(v)
    cands, pairs = {}, []
    for rid, group_views in checks_by_rid.items():
        cands[rid], refs = candidate(editions[rid], originals[rid])
        cands[rid]["rnl"] = any(v["rnl_train_only"] for v in group_views) or originals[rid]["rnl"]
        pairs.extend(make_pair(v, editions[rid], cands[rid], refs, exposure[key(v)]) for v in group_views)
    models, mirrored = br.load_text_tools(cands)
    reviews = index_unique(br.load_jsonl(args.reviews), "pair_id") if args.reviews else {}
    sealed = index_unique(br.load_jsonl(args.sealed_checks), "pair_id") if args.sealed_checks else {}
    by_pair = index_unique(pairs, "pair_id")
    if (set(reviews) | set(sealed)) - by_pair.keys():
        raise ValueError("Evidence refers to another candidate pool")
    sealed_holds = {by_pair[pid]["document_group"] for pid, result in sealed.items() if result["held"]}
    queue, verdicts = [], {}
    for p in pairs:
        rid, pid = p["row_id"], p["pair_id"]
        answer = br.final_answer(p, cands, models, mirrored)
        binding = review_key(p, answer, cands)
        old = [(t, old_verdicts[(rid, t["side_key"])]) for t in originals[rid]["targets"]]
        if any(v["document_group"] != p["document_group"] or v["tier"] != "novel_strict"
               or v["eval_holdout"] or v["pair_tiers"].get(p["image_sha256"]) != "novel_strict" for _, v in old):
            raise ValueError(f"Historical document verdict does not cover {pid}")
        screen = exposure[key(p)]
        if (h.registry_letters(answer.text) != h.registry_letters("\n".join(p["lines"]))
                or textnorm.skeleton(answer.text) != textnorm.skeleton("\n".join(p["lines"]))):
            status = "changed_after_open_text_screen"
        else:
            status = "open_checks_passed"
        identical = [t["side_key"] for t, v in old if textnorm.skeleton(t["text"]) == textnorm.skeleton(answer.text)]
        sealed_status = "same_historically_checked_input" if identical else "pending"
        if pid in sealed:
            result = sealed[pid]
            if result["answer_sha256"] != digest(answer.text) or not isinstance(result["held"], bool):
                raise ValueError(f"Stale blind result for {pid}")
            sealed_status = "blind_check_hold" if result["held"] else "blind_check_passed"
        if p["document_group"] in sealed_holds:
            sealed_status = "blind_group_hold"
        queue.append({"pair_id": pid, "review_key": binding, "answer": answer.text,
                      "answer_sha256": digest(answer.text), "view_sha256": screen["view_sha256"],
                      "image_path": p["image_path"], "crop": p["crop"], "rotation": p["rotation"],
                      "pdf_path": editions[rid]["pdf_path"], "pdf_sha256": editions[rid]["pdf_sha256"],
                      "document_group": p["document_group"], "documentary_lead": p["edition"]["documentary_lead"],
                      "draft_sha256": screen["draft_sha256"], "sealed_status": sealed_status,
                      "same_original_sides": identical, "exposure_status": status,
                      "source_inspection": reviews.get(pid, {}).get("decision", "pending")})
        if not Path(editions[rid]["pdf_path"]).is_file():
            status = "source_pdf_unavailable"
            queue[-1]["exposure_status"] = status
        if pid in reviews:
            apply_review(p, answer, reviews[pid], cands, models, mirrored)
        if status != "open_checks_passed" or sealed_status in {"pending", "blind_group_hold", "blind_check_hold"}:
            p.pop("approval", None)
            p.update(decision="review", reasons=list(dict.fromkeys(p["reasons"] + [status if status != "open_checks_passed" else sealed_status])))
        if p["decision"] == "accept":
            for sk in p["side_keys"]:
                verdicts[rid, sk] = dict(row_id=rid, side_key=sk, tier="novel_strict", eval_holdout=False)
    for name, rows in [("candidates", cands.values()), ("pairs", pairs), ("verdicts", verdicts.values()), ("review_queue", queue)]:
        write_records(stage / f"{name}.jsonl", rows)
    files = {p.name: br.file_sha256(str(p)) for p in stage.iterdir() if p.is_file()}
    inputs = {name: {"path": str(path.resolve()), "sha256": br.file_sha256(str(path))}
              for name, path in vars(args).items() if isinstance(path, Path) and name != "out"}
    return {"kind": "edition_candidate", "views": len(queue), "groups": len({p["document_group"] for p in pairs}),
            "accepted_before_row_filters": sum(p["decision"] == "accept" for p in pairs), "files": files, "inputs": inputs}


def main() -> None:
    """Prepare a new immutable edition-candidate bundle from explicit inputs."""
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("editions", "views", "checks", "candidates", "verdicts", "reviews", "sealed-checks", "out"):
        ap.add_argument("--" + name, type=Path, required=name not in {"reviews", "sealed-checks"})
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        report = prepare(args, stage)
        (stage / "run.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(args.out)
    print(f"{report['views']} views, {report['accepted_before_row_filters']} accepted before row filters")


if __name__ == "__main__":
    main()
