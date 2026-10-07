"""Lite hard-eval for one v19c checkpoint: flip pages + seeded samples → W&B.

The decision-relevant slice: every page/fragment where v19b flipped vs v19a
(the rescues and breaks the merger caused or didn't), plus fixed seeded
samples for a median. ~66 items ≈ 40 min of LM Studio inference. Outputs are
written to the same per-model cache files the full runners use, so nothing is
transcribed twice.

Logs to W&B run `v19c_hard_evals` (project qwen-hebrew-finetune) with
v19a/v19b full-benchmark reference values as flat lines.
"""
import argparse
import asyncio
import csv
import random
import statistics
import sys
from pathlib import Path

REPO = Path("/Users/isaac/Documents/GitHub/historical-document-analysis")
sys.path.insert(0, str(REPO))
import dotenv

dotenv.load_dotenv(REPO / ".env")
import os  # noqa: E402
import json  # noqa: E402

from src.datasets.evaluations.metrics import cer_pair, genizah_visible_ink_gt, \
    normalize_ink_hypothesis  # noqa: E402
from src.datasets.evaluations.helper_eval_scripts.score_genizah_offline import \
    aligned_prf  # noqa: E402
from src.datasets.consensus.consensus_gate import build_fragment_prompt  # noqa: E402
from src.models.ocr.lms_transcriber import check_lm_studio_health, \
    transcribe_with_lm_studio  # noqa: E402

REL_BENCH = REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_religious_v1"
REL_CSV = REL_BENCH / "religious_scores_long.csv"
PGP_BENCH = REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1"
PGP_CSV = REPO / "src/datasets/evaluations/transcription_results/paper_table/genizah_offline_long.csv"
PGP_OUT = REPO / "src/datasets/evaluations/transcription_raw_outputs"
SEED = 20260830
N_EXTRA = 15


def flip_and_sample(csv_path, key, a, b, all_ids):
    """(flip ids, seeded extra ids) from the recorded a/b scores."""
    by = {}
    for r in csv.DictReader(open(csv_path)):
        if r["model"] in (a, b):
            by.setdefault(r[key], {})[r["model"]] = float(r["aligned_f1"])
    flips = sorted(d for d, v in by.items()
                   if len(v) == 2 and abs(v[b] - v[a]) > 0.05)
    rest = sorted(set(all_ids) - set(flips))
    rng = random.Random(SEED)
    return flips, rng.sample(rest, min(N_EXTRA, len(rest))), {
        d: v for d, v in by.items() if len(v) == 2}


async def run(model_name: str, step: int) -> dict:
    """Transcribe the lite slice (cached) and score it."""
    served = await check_lm_studio_health()
    assert model_name in served, f"{model_name} not served: {served}"
    key = model_name.replace("-", "_")

    rel_docs = {d["doc_id"]: d for d in json.load(open(REL_BENCH / "genizah_religious_v1.json"))["docs"]}
    rel_flips, rel_extra, rel_ab = flip_and_sample(
        REL_CSV, "doc_id", "v19a_step1300", "v19b_step1300", rel_docs)
    pgp_docs = {d["doc_id"]: d for d in json.load(open(PGP_BENCH / "genizah_test_v1_verified.json"))["docs"]}
    pgp_flips, pgp_extra, pgp_ab = flip_and_sample(
        PGP_CSV, "fragment_id",
        "qwen3_vl_8b_heb_v19a_step1300", "qwen3_vl_8b_heb_v19b_step1300", pgp_docs)

    async def transcribe(doc_id, image, outdir, max_tokens):
        outdir.mkdir(parents=True, exist_ok=True)
        f = outdir / f"{key}.txt"
        if f.exists():
            return f.read_text(errors="replace")
        txt = await transcribe_with_lm_studio(
            model_name, image, build_fragment_prompt(doc_id), max_tokens=max_tokens) or ""
        f.write_text(txt, encoding="utf-8")
        return txt

    def score(hyp, gt):
        gt_ink = genizah_visible_ink_gt(gt)
        _, _, f1 = aligned_prf(normalize_ink_hypothesis(hyp), gt_ink)
        cer, _ = cer_pair(normalize_ink_hypothesis(hyp), gt_ink)
        return f1, cer

    rel_scores, pgp_scores = {}, {}
    for i, did in enumerate(rel_flips + rel_extra, 1):
        d = rel_docs[did]
        txt = await transcribe(did, d["image"], REL_BENCH / "raw_outputs" / did, 2500)
        rel_scores[did] = score(txt, d["gt"])
        if i % 10 == 0:
            print(f"  religious {i}/{len(rel_flips) + len(rel_extra)}", flush=True)
    for i, did in enumerate(pgp_flips + pgp_extra, 1):
        d = pgp_docs[did]
        img = PGP_BENCH / "images" / f"{did}.jpg"
        txt = await transcribe(did, str(img), PGP_OUT / did, 8192)
        pgp_scores[did] = score(txt, d["gt"])
        if i % 10 == 0:
            print(f"  pgp {i}/{len(pgp_flips) + len(pgp_extra)}", flush=True)

    def verdicts(flips, scores, ab, alab, blab):
        kept = dropped = lost = still = 0
        for did in flips:
            fa, fb = ab[did][alab], ab[did][blab]
            fc = scores[did][0]
            if fa < 0.5 <= fb:
                kept += fc >= 0.5
                lost += fc < 0.5
            elif fb < 0.5 <= fa:
                dropped += fc >= 0.5
                still += fc < 0.5
        return kept, lost, dropped, still

    rk, rl, rd, rs = verdicts(rel_flips, rel_scores, rel_ab,
                              "v19a_step1300", "v19b_step1300")
    pk, pl, pd_, ps = verdicts(pgp_flips, pgp_scores, pgp_ab,
                               "qwen3_vl_8b_heb_v19a_step1300",
                               "qwen3_vl_8b_heb_v19b_step1300")
    m = {
        "lite/religious_F1_median": statistics.median(v[0] for v in rel_scores.values()),
        "lite/religious_CER_median": statistics.median(v[1] for v in rel_scores.values()),
        "lite/pgp_F1_median": statistics.median(v[0] for v in pgp_scores.values()),
        "lite/pgp_CER_median": statistics.median(v[1] for v in pgp_scores.values()),
        "flips/kept_rescues": rk + pk, "flips/lost_rescues": rl + pl,
        "flips/dropped_breaks": rd + pd_, "flips/still_broken": rs + ps,
        "ref/v19a_religious_F1": 0.816, "ref/v19b_religious_F1": 0.810,
        "ref/v19a_pgp_F1": 0.862, "ref/v19b_pgp_F1": 0.868,
    }
    return m


def log_wandb(metrics: dict, step: int) -> None:
    """Append one point to the shared hard-evals W&B run."""
    import wandb
    run = wandb.init(project="qwen-hebrew-finetune", id="v19c-hard-evals",
                     name="v19c_hard_evals", resume="allow",
                     settings=wandb.Settings(silent=True))
    wandb.log(metrics, step=step)
    wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--model-name", required=True)
    args = ap.parse_args()
    metrics = asyncio.run(run(args.model_name, args.step))
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=1))
    log_wandb(metrics, args.step)
    print(f"logged to W&B v19c_hard_evals @ step {args.step}")
