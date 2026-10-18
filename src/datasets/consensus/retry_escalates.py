"""Second-stage rescue: sections-mode re-read of corpus escalate documents.

Benchmark evidence (2026-08-17, verified 131): sections mode rescued 7 of 13
whole-page VLM failures to substantive (4 at accept quality) while every doc
it degrades is whole-page-substantive — so running it ONLY on gate escalates
strictly adds acceptance.  This runner re-reads escalated pilot documents
with :func:`transcribe_sections` and re-gates against the STORED kraken text
(no kraken re-run), appending to ``retry_results.jsonl``.

Skipped escalates (cannot be rescued by a better VLM read):
* ``htr_failed`` and weak-kraken docs (< 50 kraken letters) — the gate has
  nothing to confirm against; these need the JRL imaging/preprocessing fix.

Usage (from repo root):
    PYTHONPATH=. python -m src.datasets.consensus.retry_escalates --limit 240
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.datasets.consensus.consensus_gate import GATE_VERSION, evaluate_pair
from src.datasets.consensus.corpus_runner import (
    DEFAULT_DIR,
    VLM_MODEL_ID,
    download_image,
)
from src.datasets.consensus.linewise import transcribe_sections
from src.models.ocr.lms_transcriber import check_lm_studio_health

MIN_HTR_LETTERS = 50


def rescue_candidates(results_path: Path) -> list:
    """Select escalated rows a better VLM read could plausibly rescue.

    :param results_path: Path to pilot_results.jsonl.
    :type results_path: Path
    :return: Candidate result rows, original order.
    :rtype: list
    """
    rows = [json.loads(l) for l in open(results_path)]
    out = []
    for r in rows:
        if r["tier"] != "escalate":
            continue
        if r["reason"] in ("htr_failed",) or r.get("error"):
            continue
        if r["htr_letters"] < MIN_HTR_LETTERS:
            continue
        out.append(r)
    return out


async def run(out_dir: Path, limit: int) -> None:
    """Re-read candidates with sections mode and re-gate.

    :param out_dir: Consensus pilot directory.
    :type out_dir: Path
    :param limit: Max documents this run (0 = all).
    :type limit: int
    """
    models = await check_lm_studio_health()
    if VLM_MODEL_ID not in models:
        raise RuntimeError(f"{VLM_MODEL_ID} not served by LM Studio")

    results_path = out_dir / "pilot_results.jsonl"
    retry_path = out_dir / "retry_results.jsonl"
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    done = set()
    if retry_path.exists():
        for line in open(retry_path):
            done.add(json.loads(line)["canonical_id"])

    todo = [r for r in rescue_candidates(results_path)
            if r["canonical_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"rescue candidates to process: {len(todo)}", flush=True)

    counts = {}
    with open(retry_path, "a", encoding="utf-8") as fh:
        for i, orig in enumerate(todo, 1):
            cid = orig["canonical_id"]
            t0 = time.time()
            image_path = images_dir / orig["image_file"]
            if not download_image(orig["image_url"], image_path):
                continue
            try:
                result = await transcribe_sections(
                    str(image_path), cid, VLM_MODEL_ID)
            except Exception as exc:
                print(f"[{i}/{len(todo)}] {cid} FAILED: {exc}", flush=True)
                continue
            gate = evaluate_pair(result["text"], orig["htr_text"])
            image_path.unlink(missing_ok=True)
            fh.write(json.dumps(dict(
                canonical_id=cid, stage="sections_retry",
                original_reason=orig["reason"],
                original_agreement=orig["agreement"],
                tier=gate.tier, reason=gate.reason,
                agreement=gate.agreement,
                agree_vlm_in_htr=gate.agree_vlm_in_htr,
                vlm_loop_ratio=gate.vlm_loop_ratio,
                vlm_letters=gate.vlm_letters, htr_letters=gate.htr_letters,
                vlm_text=result["text"],
                n_bands=len(result["bands"]),
                band_flags=[b["flags"] for b in result["bands"]],
                gate_version=GATE_VERSION, vlm_model=VLM_MODEL_ID,
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
            ), ensure_ascii=False) + "\n")
            fh.flush()
            counts[gate.tier] = counts.get(gate.tier, 0) + 1
            print(f"[{i}/{len(todo)}] {cid}: escalate -> {gate.tier} "
                  f"(agree {orig['agreement']:.2f} -> {gate.agreement:.2f}) "
                  f"in {time.time() - t0:.0f}s  running: {counts}", flush=True)
    print(f"done; results in {retry_path}", flush=True)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args.out_dir, args.limit))


if __name__ == "__main__":
    main()
