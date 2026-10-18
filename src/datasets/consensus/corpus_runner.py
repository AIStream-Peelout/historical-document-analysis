"""Consensus corpus runner: Kraken + VLM + confidence gate over an inventory.

Sequential, resumable, local-only.  For each inventory record: download the
image at native resolution, transcribe with Kraken (docker microservice) and
the calibrated VLM (LM Studio), apply the consensus gate, append one JSONL
result row.  Sequential on purpose — LM Studio must never see two inference
requests at once (module lock in ``lms_transcriber`` enforces it), and Kraken
shares the box.

The VLM defaults to the calibrated engine (v18b step-700) with the exact
benchmark prompt; changing model or prompt invalidates the calibrated
thresholds in ``consensus_gate``.

Usage (from repo root, long runs via nohup + absolute paths):
    PYTHONPATH=. python -m src.datasets.consensus.corpus_runner \\
        --inventory src/datasets/raw_data/cairo_genizah/consensus_pilot/pilot_inventory.jsonl \\
        --limit 500
"""

import argparse
import asyncio
import json
import shutil
import time
import urllib.request
from pathlib import Path

from src.datasets.consensus.consensus_gate import (
    GATE_VERSION,
    build_fragment_prompt,
    evaluate_pair,
)
from src.models.ocr.kraken_transcriber import (
    preload_kraken_model,
    transcribe_with_kraken,
)
from src.models.ocr.lms_transcriber import (
    check_lm_studio_health,
    transcribe_with_lm_studio,
)

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/consensus_pilot"
KRAKEN_MODEL = str(
    _REPO / "src/datasets/raw_data/cairo_genizah/custom_model_weights/"
            "MiDRASH_Gen_01.mlmodel"
)
VLM_MODEL_ID = "qwen3-vl-8b-heb-v18b-step700"
KRAKEN_TIMEOUT_S = 180.0


def load_done_ids(results_path: Path) -> set:
    """Return canonical ids already present in the results file.

    :param results_path: Path to the results JSONL (may not exist).
    :type results_path: Path
    :return: Set of processed canonical ids.
    :rtype: set
    """
    done = set()
    if results_path.exists():
        with open(results_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["canonical_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def download_image(url: str, dest: Path, timeout: float = 60.0) -> bool:
    """Download an image to *dest* unless it is already there.

    Images are fetched at native resolution — the resolution policy lives in
    the model's exported processor, never in preprocessing here.

    :param url: Source URL.
    :type url: str
    :param dest: Destination file path.
    :type dest: Path
    :param timeout: Request timeout in seconds.
    :type timeout: float
    :return: True when the file is available locally afterwards.
    :rtype: bool
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"    image download failed: {exc}")
        return False


async def process_doc(rec: dict, images_dir: Path) -> dict:
    """Run both engines and the gate for one inventory record.

    :param rec: Inventory row (canonical_id, image_url, ...).
    :type rec: dict
    :param images_dir: Directory for downloaded images.
    :type images_dir: Path
    :return: Result row ready for JSONL append.
    :rtype: dict
    """
    cid = rec["canonical_id"]
    timings = {}
    image_path = images_dir / rec["image_file"]

    t0 = time.time()
    ok = download_image(rec["image_url"], image_path)
    timings["download_s"] = round(time.time() - t0, 1)
    if not ok:
        gate = evaluate_pair("", "")
        return _row(rec, "", "", gate, timings, error="image_unavailable")

    t0 = time.time()
    htr_text = await transcribe_with_kraken(
        KRAKEN_MODEL, str(image_path), timeout=KRAKEN_TIMEOUT_S
    ) or ""
    timings["kraken_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    vlm_text = await transcribe_with_lm_studio(
        model_id=VLM_MODEL_ID,
        image_path=str(image_path),
        prompt=build_fragment_prompt(cid),
    ) or ""
    timings["vlm_s"] = round(time.time() - t0, 1)

    gate = evaluate_pair(vlm_text, htr_text)
    return _row(rec, vlm_text, htr_text, gate, timings)


def _row(rec: dict, vlm_text: str, htr_text: str, gate, timings: dict,
         error: str = "") -> dict:
    """Assemble one results row.

    :param rec: Inventory row.
    :type rec: dict
    :param vlm_text: Raw VLM transcription.
    :type vlm_text: str
    :param htr_text: Raw Kraken transcription.
    :type htr_text: str
    :param gate: Gate decision for the pair.
    :type gate: consensus_gate.GateResult
    :param timings: Per-stage wall-clock seconds.
    :type timings: dict
    :param error: Non-engine failure note (e.g. image_unavailable).
    :type error: str
    :return: JSON-serialisable result row.
    :rtype: dict
    """
    return dict(
        canonical_id=rec["canonical_id"],
        shelfmark_display=rec.get("shelfmark_display"),
        institution=rec.get("institution"),
        image_file=rec.get("image_file"),
        image_url=rec.get("image_url"),
        tier=gate.tier, reason=gate.reason,
        agreement=gate.agreement,
        agree_vlm_in_htr=gate.agree_vlm_in_htr,
        agree_htr_in_vlm=gate.agree_htr_in_vlm,
        vlm_loop_ratio=gate.vlm_loop_ratio,
        vlm_letters=gate.vlm_letters, htr_letters=gate.htr_letters,
        vlm_text=vlm_text, htr_text=htr_text,
        timings=timings, error=error,
        gate_version=GATE_VERSION, vlm_model=VLM_MODEL_ID,
        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


async def run(inventory: Path, out_dir: Path, limit: int,
              delete_images: bool = False, min_free_gb: float = 0.0) -> None:
    """Process the inventory sequentially with per-document resume.

    :param inventory: Inventory JSONL path.
    :type inventory: Path
    :param out_dir: Output directory (results + images).
    :type out_dir: Path
    :param limit: Maximum number of new documents to process (0 = all).
    :type limit: int
    :param delete_images: Remove each image after processing (long unattended
        runs; images are re-downloadable from GCS).
    :type delete_images: bool
    :param min_free_gb: Stop gracefully when free disk falls below this
        (0 disables the guard).
    :type min_free_gb: float
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    results_path = out_dir / "pilot_results.jsonl"

    models = await check_lm_studio_health()
    if VLM_MODEL_ID not in models:
        raise RuntimeError(f"{VLM_MODEL_ID} not served by LM Studio: {models}")
    preload_kraken_model(KRAKEN_MODEL)

    records = [json.loads(line) for line in open(inventory)]
    done = load_done_ids(results_path)
    todo = [r for r in records if r["canonical_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"inventory {len(records)}, done {len(done)}, processing {len(todo)}")

    counts = {}
    with open(results_path, "a", encoding="utf-8") as fh:
        for i, rec in enumerate(todo, 1):
            if min_free_gb:
                free_gb = shutil.disk_usage(out_dir).free / 1e9
                if free_gb < min_free_gb:
                    print(f"STOPPING: {free_gb:.1f}GB free < "
                          f"--min-free-gb {min_free_gb}", flush=True)
                    break
            t0 = time.time()
            row = await process_doc(rec, images_dir)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if delete_images:
                (images_dir / rec["image_file"]).unlink(missing_ok=True)
            counts[row["tier"]] = counts.get(row["tier"], 0) + 1
            print(f"[{i}/{len(todo)}] {rec['canonical_id']}: "
                  f"{row['tier']} ({row['reason']}, agree {row['agreement']:.2f}) "
                  f"in {time.time() - t0:.0f}s  running: {counts}", flush=True)
    print(f"done; results in {results_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path,
                        default=DEFAULT_DIR / "pilot_inventory.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="max new documents this run (0 = all)")
    parser.add_argument("--delete-images", action="store_true",
                        help="remove each image after processing")
    parser.add_argument("--min-free-gb", type=float, default=0.0,
                        help="stop gracefully below this much free disk")
    args = parser.parse_args()
    asyncio.run(run(args.inventory, args.out_dir, args.limit,
                    delete_images=args.delete_images,
                    min_free_gb=args.min_free_gb))


if __name__ == "__main__":
    main()
