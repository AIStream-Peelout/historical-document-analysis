"""Build a single scrollable HTML page for eyeballing the Genizah benchmark.

Renders every fragment in ``genizah_test_v1`` as image + ground truth side by
side, with its selection metadata (visible characters, gap count,
reconstruction fraction), so the benchmark's legibility can be reviewed
without opening 150 files.  Optionally splices in each model's transcription
for fragments that have already been evaluated.

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/make_benchmark_review_page.py \\
        [--with-outputs] [--sort gaps|visible|id] [--out review.html]
"""

import argparse
import html
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_BENCH_DIR = (_REPO / "src/datasets/raw_data/cairo_genizah/evaluations/genizah_test_v1")
_OUTPUTS_DIR = _REPO / "src/datasets/evaluations/transcription_raw_outputs"

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0;
       background: #f5f5f7; color: #1d1d1f; }
header { position: sticky; top: 0; background: #fff; padding: 14px 24px;
         border-bottom: 1px solid #d2d2d7; z-index: 10; }
h1 { margin: 0 0 4px; font-size: 18px; }
.meta { color: #6e6e73; font-size: 13px; }
.frag { display: grid; grid-template-columns: minmax(280px, 40%) 1fr;
        gap: 20px; padding: 22px 24px; border-bottom: 1px solid #d2d2d7;
        background: #fff; margin: 14px; border-radius: 10px; }
.frag img { width: 100%; border-radius: 6px; border: 1px solid #d2d2d7;
            background: #fff; }
.gt { direction: rtl; text-align: right; white-space: pre-wrap;
      font-family: "SBL Hebrew", "Times New Roman", serif; font-size: 17px;
      line-height: 1.9; background: #f0fff0; padding: 12px; border-radius: 6px; }
.model { direction: rtl; text-align: right; white-space: pre-wrap;
         font-family: "SBL Hebrew", "Times New Roman", serif; font-size: 15px;
         line-height: 1.7; background: #f7f7fa; padding: 10px;
         border-radius: 6px; margin-top: 8px; max-height: 260px;
         overflow-y: auto; }
h2 { font-size: 15px; margin: 0 0 6px; }
h3 { font-size: 13px; margin: 12px 0 2px; color: #6e6e73; direction: ltr; }
.badge { display: inline-block; background: #eef; color: #334; padding: 2px 8px;
         border-radius: 10px; font-size: 12px; margin-right: 6px; }
"""


def load_outputs(doc_id: str) -> list:
    """Load saved model transcriptions for one fragment.

    :param doc_id: Fragment identifier.
    :type doc_id: str
    :return: Sorted list of (model_key, text) pairs; empty when not evaluated.
    :rtype: list
    """
    d = _OUTPUTS_DIR / doc_id
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("*.txt")):
        if f.stem in ("ground_truth",) or f.stem.endswith(("_raw", "_flat")):
            continue
        text = f.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            out.append((f.stem, text))
    return out


def main() -> None:
    """Render the review page."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-outputs", action="store_true",
                        help="Include model transcriptions for evaluated fragments")
    parser.add_argument("--sort", choices=("gaps", "visible", "id"), default="gaps")
    parser.add_argument("--out", type=Path, default=_BENCH_DIR / "review.html")
    args = parser.parse_args()

    spec = json.load(open(_BENCH_DIR / "genizah_test_v1.json"))
    docs = spec["docs"]
    key = {
        "gaps": lambda d: int(d["gaps"]),
        "visible": lambda d: -int(d["visible_chars"]),
        "id": lambda d: d["doc_id"],
    }[args.sort]
    docs = sorted(docs, key=key)

    parts = [
        # Explicit charset: without it browsers sniff the Hebrew UTF-8 bytes as
        # Latin-1 and every letter renders as "×"-prefixed mojibake.
        "<!DOCTYPE html>",
        "<html lang='he'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>genizah_test_v1 — benchmark review</title>",
        f"<style>{_CSS}</style></head><body>",
        "<header><h1>genizah_test_v1 — benchmark review</h1>",
        f"<div class='meta'>{len(docs)} fragments · frozen {spec['frozen_date']} · "
        f"sorted by {args.sort} · gaps rendered <code>{spec['gap_rendering']}</code>"
        f"<br>{html.escape(spec['criteria'])}</div></header>",
    ]

    for i, doc in enumerate(docs, 1):
        doc_id = doc["doc_id"]
        img = f"images/{doc_id}.jpg"
        parts.append("<div class='frag'>")
        parts.append(f"<div><img src='{html.escape(img)}' loading='lazy'></div>")
        parts.append("<div>")
        parts.append(
            f"<h2>{i}. {html.escape(doc['shelf_mark'])}</h2>"
            f"<div class='meta'>{html.escape(doc['institution'])} · "
            f"<code>{html.escape(doc_id)}</code></div>"
            f"<div style='margin:8px 0'>"
            f"<span class='badge'>{doc['visible_chars']} visible chars</span>"
            f"<span class='badge'>{doc['gaps']} gaps</span>"
            f"<span class='badge'>recon {doc['recon_frac']}</span></div>"
        )
        parts.append(f"<h3>GROUND TRUTH</h3><div class='gt'>{html.escape(doc['gt'])}</div>")
        if args.with_outputs:
            for model, text in load_outputs(doc_id):
                parts.append(
                    f"<h3>{html.escape(model)}</h3>"
                    f"<div class='model'>{html.escape(text[:4000])}</div>"
                )
        parts.append("</div></div>")

    parts.append("</body></html>")
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {args.out} ({len(docs)} fragments)")
    print(f"Open with:  open {args.out}")


if __name__ == "__main__":
    main()
