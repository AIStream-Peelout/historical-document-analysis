"""Preflight validator for the final Cairo Genizah benchmark run.

Verifies every provider and every input BEFORE the paid, run-once benchmark
starts: one tiny call per paid model (Gemini Flash/Pro, Claude, ChatGPT),
LM Studio model availability, Google Vision credentials, W&B auth, and
that all benchmark documents have extractable ground truth and a locally
staged primary image.

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/preflight_final_benchmark.py \
        --catalog <frozen benchmark json> \
        --lm-studio-models qwen/qwen3-vl-8b,... \
        [--check-openai] [--check-claude]

Exits non-zero if any check fails.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import dotenv

dotenv.load_dotenv()

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
# fragment_evals lives one level up (the evaluations dir, also the run CWD)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.llm.transcription.genizah_fragment_agent import (  # noqa: E402
    AgentConfig,
    call_gemini_text_only,
    download_image,
)
from src.models.ocr.lms_transcriber import check_lm_studio_health  # noqa: E402

PASS, FAIL, WARN = "✅", "❌", "⚠️ "


async def check_gemini(results: list) -> None:
    """Tiny text-only call against Flash and Pro to prove quota/cap headroom.

    :param results: Mutable list of (ok, label) tuples to append to.
    :type results: list
    """
    for model in (AgentConfig.GEMINI_FLASH_MODEL, AgentConfig.GEMINI_PRO_MODEL):
        text = await call_gemini_text_only(model, "Reply with the single word: ok", timeout=60)
        ok = bool(text and "ok" in text.lower())
        results.append((ok, f"Gemini {model}: {'reachable' if ok else 'FAILED (spending cap / quota?)'}"))


async def check_claude(models: list, results: list) -> None:
    """One ~5-token call per Claude model.

    :param models: Claude model IDs to verify.
    :type models: list
    :param results: Mutable list of (ok, label) tuples to append to.
    :type results: list
    """
    import anthropic
    client = anthropic.AsyncAnthropic(timeout=60.0, max_retries=1)
    for model in models:
        try:
            msg = await client.messages.create(
                model=model, max_tokens=32,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            )
            text = "".join(b.text for b in msg.content if b.type == "text")
            ok = "ok" in text.lower()
        except Exception as exc:
            ok, text = False, f"{type(exc).__name__}: {str(exc)[:120]}"
        results.append((ok, f"Claude {model}: {'reachable' if ok else text}"))


async def check_openai(models: list, results: list) -> None:
    """Verify each OpenAI model is available to this key and answers.

    :param models: OpenAI model IDs to verify.
    :type models: list
    :param results: Mutable list of (ok, label) tuples to append to.
    :type results: list
    """
    from src.models.llm.transcription.openai_transcriber import _get_client
    client = _get_client()
    try:
        available = {m.id for m in (await client.models.list()).data}
    except Exception as exc:
        results.append((False, f"OpenAI auth/models.list: {type(exc).__name__}: {str(exc)[:120]}"))
        return
    for model in models:
        if model not in available:
            close = sorted(m for m in available if "gpt-5" in m)[:8]
            results.append((False, f"OpenAI {model}: NOT available. gpt-5* models: {close}"))
            continue
        try:
            # Generous budget: GPT-5-family reasoning tokens count against
            # max_completion_tokens and can starve tiny budgets.
            resp = await client.chat.completions.create(
                model=model, max_completion_tokens=512,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            )
            ok = "ok" in (resp.choices[0].message.content or "").lower()
        except Exception as exc:
            ok = False
        results.append((ok, f"OpenAI {model}: {'reachable' if ok else 'call failed'}"))


async def check_lm_studio(models: list, base_url: str, results: list) -> None:
    """Confirm the LM Studio server is up and lists every benchmark model.

    :param models: LM Studio model IDs the run will use.
    :type models: list
    :param base_url: LM Studio API base URL.
    :type base_url: str
    :param results: Mutable list of (ok, label) tuples to append to.
    :type results: list
    """
    try:
        available = await check_lm_studio_health(base_url)
    except RuntimeError as exc:
        results.append((False, f"LM Studio: {exc}"))
        return
    for m in models:
        results.append((m in available, f"LM Studio model {m}: {'listed' if m in available else 'MISSING'}"))


async def check_docs_and_images(catalog_path: str, images_dir: Path, results: list) -> None:
    """Validate GT extraction and stage the primary image for every doc.

    :param catalog_path: Path to the frozen benchmark catalog JSON.
    :type catalog_path: str
    :param images_dir: Local image cache directory used by the eval.
    :type images_dir: Path
    :param results: Mutable list of (ok, label) tuples to append to.
    :type results: list
    """
    from fragment_evals import extract_ground_truth, has_ground_truth
    catalog = json.load(open(catalog_path))
    bad = []
    for doc_id, meta in catalog.items():
        if not meta.get("images"):
            bad.append(f"{doc_id}: no images")
            continue
        if not has_ground_truth(meta):
            bad.append(f"{doc_id}: no usable ground truth")
            continue
        try:
            path = await download_image(meta["images"][0], doc_id, images_dir)
            if not path.exists() or path.stat().st_size < 10_000:
                bad.append(f"{doc_id}: image missing/tiny at {path}")
        except Exception as exc:
            bad.append(f"{doc_id}: image download failed — {type(exc).__name__}: {str(exc)[:80]}")
    results.append((not bad, f"Docs/images: {len(catalog) - len(bad)}/{len(catalog)} staged"
                    + (f" — problems: {bad}" if bad else "")))


def check_static(results: list) -> None:
    """Non-network checks: credentials files and env vars.

    :param results: Mutable list of (ok, label) tuples to append to.
    :type results: list
    """
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    results.append((bool(creds and Path(creds).exists()),
                    f"GOOGLE_APPLICATION_CREDENTIALS: {creds or 'unset'}"))
    results.append((bool(os.getenv("GEMINI_API_KEY")), "GEMINI_API_KEY set"))
    results.append((bool(os.getenv("WANDB_API_KEY")), "WANDB_API_KEY set"))
    results.append((bool(os.getenv("ANTHROPIC_API_KEY")), "ANTHROPIC_API_KEY set"))
    key = os.getenv("OPEN_AI_API_KEY", "")
    results.append((bool(key) and len(key) > 20, f"OPEN_AI_API_KEY plausible (len {len(key)})"))


async def main() -> int:
    """Run all preflight checks and print a pass/fail report.

    :return: 0 when every check passed, 1 otherwise.
    :rtype: int
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--lm-studio-models", type=lambda s: [m for m in s.split(",") if m], default=[])
    parser.add_argument("--lm-studio-base-url", default="http://localhost:1234/v1")
    parser.add_argument("--claude-models", type=lambda s: [m for m in s.split(",") if m],
                        default=["claude-opus-4-8", "claude-sonnet-5"])
    parser.add_argument("--openai-models", type=lambda s: [m for m in s.split(",") if m],
                        default=["gpt-5.2"])
    parser.add_argument("--check-claude", action="store_true")
    parser.add_argument("--check-openai", action="store_true")
    parser.add_argument("--images-dir", default="./genizah_images")
    args = parser.parse_args()

    results: list = []
    check_static(results)
    await check_gemini(results)
    if args.check_claude:
        await check_claude(args.claude_models, results)
    if args.check_openai:
        await check_openai(args.openai_models, results)
    if args.lm_studio_models:
        await check_lm_studio(args.lm_studio_models, args.lm_studio_base_url, results)
    await check_docs_and_images(args.catalog, Path(args.images_dir), results)

    print("\n" + "=" * 70)
    print("PREFLIGHT REPORT")
    print("=" * 70)
    failures = 0
    for ok, label in results:
        print(f"{PASS if ok else FAIL} {label}")
        failures += 0 if ok else 1
    print("=" * 70)
    print("ALL CHECKS PASSED — safe to launch" if not failures else
          f"{failures} CHECK(S) FAILED — do NOT launch the paid run")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
