# File name: probe_forgetting.py
# Date: 8/10/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.
"""Cross-generation forgetting probe: typed/printed scripts vs Genizah gains.

v1.8b is the first model whose vision tower actually trained, which makes it
the first that *could* forget the typed-script competence (Vilna gemara,
printed Rashi, synthetic Rashi renders) the earlier language-only adapters
banked. This probe runs a fixed seeded sample of Talmud val rows and
synthetic-Rashi eval renders through every local generation via the LM Studio
API (the only valid harness for unsloth-trained checkpoints) and reports
strict CER per domain per model.

Usage (from src/datasets/evaluations, PYTHONPATH=repo root):
    python helper_eval_scripts/probe_forgetting.py \\
        [--models m1,m2,...] [--n-talmud 20] [--n-synth 30]
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))

from src.datasets.evaluations.metrics import cer_pair  # noqa: E402

LM_STUDIO_URL = "http://localhost:1234/v1"
SEED = 20260810
DEFAULT_MODELS = [
    "qwen3-vl-8b-heb-v16-step1000",
    "qwen3-vl-8b-heb-v17-step800",
    "qwen3-vl-8b-heb-v18a-step700",
    "qwen3-vl-8b-heb-v18b-step700",
]
OUT_PATH = _REPO / "src/datasets/evaluations/transcription_results/forgetting_probe.jsonl"


def image_to_data_url(img: Any) -> str:
    """Encode a PIL image as a base64 JPEG data URL.

    :param img: PIL image from a datasets row.
    :returns: ``data:image/jpeg;base64,...`` string at native resolution.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def call_model(client: Any, model: str, question: str, data_url: str,
               max_tokens: int = 3000, retries: int = 4) -> str:
    """One serial LM Studio transcription call with JIT-load retries.

    :param client: OpenAI-compatible client bound to LM Studio.
    :param model: LM Studio model identifier.
    :param question: Prompt text (the row's own training question).
    :param data_url: Base64 image data URL.
    :param max_tokens: Generation cap.
    :param retries: Attempts (model-swap evictions 400 until JIT load).
    :returns: Model output text ("" after exhausted retries).
    """
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": question},
                ]}],
                max_tokens=max_tokens, temperature=0.0, timeout=420,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            if attempt == retries - 1:
                print(f"    FAILED after {retries}: {type(exc).__name__}: {str(exc)[:120]}")
                return ""
            time.sleep(20)
    return ""


def build_samples(n_talmud: int, n_synth: int) -> List[Dict[str, Any]]:
    """Draw the fixed seeded probe samples from local dataset caches.

    :param n_talmud: Talmud val rows (mixed crop/page, short targets).
    :param n_synth: Synthetic-Rashi eval renders.
    :returns: List of dicts with domain/section/stem/question/answer/image.
    """
    from datasets import load_dataset

    talmud_val = load_dataset("isaacmg/talmud_finetune", split="val")
    talmud_val = talmud_val.filter(lambda c: 0 < c < 2500, input_columns="target_chars")
    talmud_val = talmud_val.shuffle(seed=SEED).select(range(min(n_talmud, len(talmud_val))))

    synth_eval = load_dataset(
        "isaacmg/synthetic_rashi", split="eval",
        revision="b4e1254e1b51ff93958599ccd46ecf2992cd2e97")
    synth_eval = synth_eval.shuffle(seed=SEED).select(range(min(n_synth, len(synth_eval))))

    samples = []
    for row in talmud_val:
        samples.append(dict(domain=f"talmud_{row['task']}", section=row["section"],
                            stem=row["stem"], question=row["question"],
                            answer=row["answer"], image=row["image"]))
    for i, row in enumerate(synth_eval):
        samples.append(dict(domain="synthetic_rashi", section="synth",
                            stem=f"synth_{SEED}_{i}", question=row["question"],
                            answer=row["answer"], image=row["image"]))
    return samples


def main() -> None:
    """Run the probe and print the per-domain CER table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--n-talmud", type=int, default=20)
    parser.add_argument("--n-synth", type=int, default=30)
    args = parser.parse_args()
    models = args.models.split(",")

    from openai import OpenAI
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    samples = build_samples(args.n_talmud, args.n_synth)
    print(f"{len(samples)} probe samples x {len(models)} models", flush=True)

    done = set()
    if OUT_PATH.exists():
        done = {(r["model"], r["stem"]) for r in map(json.loads, open(OUT_PATH))}
    with open(OUT_PATH, "a") as out:
        for model in models:
            for i, s in enumerate(samples):
                if (model, s["stem"]) in done:
                    continue
                text = call_model(client, model, s["question"],
                                  image_to_data_url(s["image"]))
                cer_s, _ = cer_pair(text, s["answer"])
                out.write(json.dumps(dict(model=model, domain=s["domain"],
                                          section=s["section"], stem=s["stem"],
                                          cer=round(cer_s, 4),
                                          out_chars=len(text)), ensure_ascii=False) + "\n")
                out.flush()
                if (i + 1) % 10 == 0:
                    print(f"  {model}: {i + 1}/{len(samples)}", flush=True)

    rows = [json.loads(l) for l in open(OUT_PATH)]
    import statistics
    print(f"\n{'model':32s} {'domain':22s} {'n':>3s} {'CER med':>8s}")
    for model in models:
        for domain in sorted({r['domain'] for r in rows}):
            sel = [r["cer"] for r in rows if r["model"] == model and r["domain"] == domain]
            if sel:
                print(f"{model:32s} {domain:22s} {len(sel):3d} {statistics.median(sel):8.3f}")


if __name__ == "__main__":
    main()
