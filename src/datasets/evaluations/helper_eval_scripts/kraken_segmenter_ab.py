"""Kraken segmenter A/B: one ``/transcribe_lines`` call per page, cached under a tag.

Writes, per benchmark doc, the same files the benchmark runners score:

* ``kraken_lines_<tag>.json`` — service response plus ``seconds`` (wall time),
* ``kraken_raw_<tag>.txt`` — line texts in the segmenter's reading order,
* ``kraken_seg_<tag>.txt`` — the same lines re-ordered by geometry
  (``ktiv_layout.reorder_ocr_lines``, as ``run_religious_benchmark.py`` does).

``/transcribe`` and ``/transcribe_lines`` run the identical two-pass policy, so
deriving ``kraken_raw`` from the lines response is equivalent to a separate
``/transcribe`` call and halves segmentation time. Resumable: docs whose lines
file exists are skipped. Point it at a test service with ``--url`` (never the
:8002 production service while the consensus pipeline is using it).

Score afterwards with ``run_religious_benchmark.py --score-only`` and
``score_genizah_offline.py --no-wandb --out-dir <scratch>``.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))

_EVAL = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations"
_BENCHES = {
    "religious": (_EVAL / "genizah_religious_v1/genizah_religious_v1.json",
                  _EVAL / "genizah_religious_v1/raw_outputs"),
    "pgp": (_EVAL / "genizah_test_v1/genizah_test_v1_verified.json",
            _REPO / "src/datasets/evaluations/transcription_raw_outputs"),
}
_KRAKEN_MODEL = str(_REPO / "src/datasets/raw_data/cairo_genizah/custom_model_weights/MiDRASH_Gen_01.mlmodel")


def doc_image(bench: str, doc: dict) -> Path:
    """Local image path of a benchmark doc.

    :param bench: ``religious`` or ``pgp``.
    :type bench: str
    :param doc: Benchmark doc entry.
    :type doc: dict
    :return: Image path (religious docs carry it; PGP images are ``images/<doc_id>.jpg``).
    :rtype: Path
    """
    if bench == "religious":
        return Path(doc["image"])
    return _EVAL / "genizah_test_v1/images" / f"{doc['doc_id']}.jpg"


async def run(bench: str, tag: str, model: str, limit: int, ids: set) -> None:
    """Transcribe every doc of a benchmark through the service and cache the outputs.

    :param bench: ``religious`` or ``pgp``.
    :type bench: str
    :param tag: Output key suffix, e.g. ``k7_default``.
    :type tag: str
    :param model: Recognition model path (host path; mapped by basename in the service).
    :type model: str
    :param limit: Process at most this many docs (0 = all).
    :type limit: int
    :param ids: Restrict to these doc ids (empty = all).
    :type ids: set
    """
    # Imported here so KRAKEN_MICROSERVICE_URL (set in main) is read by the client.
    from src.finetuning.qwen_hebrew.ktiv_layout import reorder_ocr_lines
    from src.models.ocr.kraken_transcriber import preload_kraken_model, transcribe_with_kraken_lines

    spec, out_root = _BENCHES[bench]
    docs = json.load(open(spec))["docs"]
    if ids:
        docs = [d for d in docs if d["doc_id"] in ids]
    if limit:
        docs = docs[:limit]
    preload_kraken_model(model)
    for i, d in enumerate(docs, 1):
        outdir = out_root / d["doc_id"]
        lines_f = outdir / f"kraken_lines_{tag}.json"
        if lines_f.exists():
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        res = await transcribe_with_kraken_lines(model, str(doc_image(bench, d)), timeout=900.0)
        if res is None:
            print(f"  FAILED {d['doc_id']}", flush=True)
            continue
        res["seconds"] = round(time.time() - t0, 1)
        lines = res.get("lines", [])
        (outdir / f"kraken_raw_{tag}.txt").write_text("\n".join(l["text"] for l in lines), encoding="utf-8")
        (outdir / f"kraken_seg_{tag}.txt").write_text(reorder_ocr_lines(lines), encoding="utf-8")
        lines_f.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print(f"  {i}/{len(docs)} {d['doc_id']} {len(lines)} lines {res['seconds']}s", flush=True)


def main() -> None:
    """Parse arguments and run the A/B transcription pass."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--benchmark", choices=sorted(_BENCHES), required=True)
    p.add_argument("--tag", required=True, help="output key suffix, e.g. k7_default")
    p.add_argument("--url", default="http://localhost:8003", help="Kraken service base URL")
    p.add_argument("--kraken-model", default=_KRAKEN_MODEL)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", default="", help="comma-separated doc ids")
    args = p.parse_args()
    if args.url.rstrip("/").endswith(":8002"):
        sys.exit("refusing :8002 — that is the production service the consensus pipeline uses")
    os.environ["KRAKEN_MICROSERVICE_URL"] = args.url
    asyncio.run(run(args.benchmark, args.tag, args.kraken_model, args.limit,
                    {x for x in args.ids.split(",") if x}))


if __name__ == "__main__":
    main()
