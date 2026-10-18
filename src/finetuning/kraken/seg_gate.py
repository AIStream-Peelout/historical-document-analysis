"""blla2026 predictions for segmenter-export candidates (runs INSIDE the kraken 7 container).

For every page in ``<data>/candidates.jsonl`` without ``<data>/preds/<page_id>.json``:
segment the RGB page image (the same input ``ketos segtrain`` trains on) with the
given blla model, mark each detected line covered / uncovered against the GT
line boxes (:func:`seg_geometry.is_covered`), and recognise only the uncovered
lines with the recognition model (on the nlbin-binarized page, as the service
does) so the exporter can reject pages with untranscribed text.

Needs only kraken + this directory on ``PYTHONPATH`` (no repo imports), e.g.:
    docker run --rm -v <repo>/src/finetuning/kraken:/code:ro -v <data>:/data \\
      -v <weights>:/app/models:ro -v <segmodels>:/segmodels:ro -e PYTHONPATH=/code \\
      kraken-service:k7-base python /code/seg_gate.py --data /data \\
      --seg-model /segmodels/blla_2026/blla.mlmodel --rec-model /app/models/MiDRASH_Gen_01.mlmodel
"""
import argparse
import dataclasses
import json
import logging
import statistics
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from kraken import binarization, rpred  # noqa: E402
from kraken.configs import SegmentationInferenceConfig  # noqa: E402
from kraken.lib import models  # noqa: E402
from kraken.models import load_models  # noqa: E402
from kraken.tasks.segmentation import SegmentationTaskModel  # noqa: E402

from seg_geometry import is_covered  # noqa: E402

log = logging.getLogger("seg_gate")


def page_pitch(rec: dict) -> float:
    """Median GT line-box height of a candidate page (the gate's pitch proxy).

    :param rec: Candidate record.
    :type rec: dict
    :return: Pitch, px.
    :rtype: float
    """
    return float(statistics.median(ln["box"][3] - ln["box"][1] for col in rec["columns"] for ln in col))


def gate_page(rec: dict, data: Path, seg: SegmentationTaskModel, cfg: SegmentationInferenceConfig, rec_model) -> dict:
    """Segment one page and read its uncovered lines.

    :param rec: Candidate record.
    :type rec: dict
    :param data: Dataset root.
    :type data: Path
    :param seg: Segmentation task.
    :type seg: SegmentationTaskModel
    :param cfg: Segmentation config.
    :type cfg: SegmentationInferenceConfig
    :param rec_model: Recognition model.
    :return: ``{"page_id", "lines": [{"baseline", "boundary", "covered", "text", "conf"}], "seconds"}``.
    :rtype: dict
    """
    t0 = time.time()
    im = Image.open(data / rec["image"]).convert("RGB")
    result = seg.predict(im, cfg)
    gt_boxes = [tuple(ln["box"]) for col in rec["columns"] for ln in col]
    pitch = page_pitch(rec)
    lines, uncovered = [], []
    for ln in result.lines:
        baseline = [[int(p[0]), int(p[1])] for p in ln.baseline]
        boundary = [[int(p[0]), int(p[1])] for p in ln.boundary] if ln.boundary is not None else []
        cov = is_covered(baseline, gt_boxes, pitch)
        lines.append({"baseline": baseline, "boundary": boundary, "covered": cov, "text": None, "conf": None})
        if not cov and boundary:
            ln.boundary = [tuple(p) for p in boundary]
            uncovered.append((len(lines) - 1, ln))
    if uncovered:
        bim = binarization.nlbin(im)
        sub = dataclasses.replace(result, lines=[ln for _, ln in uncovered])
        for (i, _), r in zip(uncovered, rpred.rpred(rec_model, bim, sub)):
            confs = list(r.confidences or [])
            lines[i]["text"] = r.prediction
            lines[i]["conf"] = sum(confs) / len(confs) if confs else None
    return {"page_id": rec["page_id"], "lines": lines, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--seg-model", required=True)
    p.add_argument("--rec-model", required=True)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", default="0/1", help="k/N: process candidates whose index % N == k")
    p.add_argument("--max-w", type=int, default=2600,
                   help="skip pages wider than this (the exporter rejects them for training memory; very wide "
                        "pages also exceed the gate's own memory cap)")
    a = p.parse_args()
    shard_k, shard_n = (int(v) for v in a.shard.split("/"))
    torch.set_num_threads(a.threads)
    seg = SegmentationTaskModel(load_models(a.seg_model, tasks=["segmentation"]))
    cfg = SegmentationInferenceConfig(text_direction="horizontal-rl", accelerator="cpu", device=1, precision="32-true")
    rec_model = models.load_any(a.rec_model)
    out = a.data / "preds"
    out.mkdir(exist_ok=True)
    recs = [json.loads(l) for l in (a.data / "candidates.jsonl").open(encoding="utf-8")]
    todo = [r for i, r in enumerate(recs)
            if i % shard_n == shard_k and r["image_size"][0] <= a.max_w and not (out / f"{r['page_id']}.json").exists()]
    if a.limit:
        todo = todo[:a.limit]
    log.info("%d candidates, %d to gate", len(recs), len(todo))
    for k, rec in enumerate(todo, 1):
        res = gate_page(rec, a.data, seg, cfg, rec_model)
        tmp = out / f"{rec['page_id']}.json.tmp"
        tmp.write_text(json.dumps(res, ensure_ascii=False))
        tmp.rename(out / f"{rec['page_id']}.json")
        n_unc = sum(not ln["covered"] for ln in res["lines"])
        log.info("%d/%d %s: %d lines (%d uncovered) %.1fs", k, len(todo), rec["page_id"],
                 len(res["lines"]), n_unc, res["seconds"])


if __name__ == "__main__":
    main()
