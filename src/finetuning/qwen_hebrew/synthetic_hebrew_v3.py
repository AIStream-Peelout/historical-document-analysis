"""Synthetic multi-font Hebrew dataset v3 — the v1.9a font-generalization arm.

Successor to :mod:`synthetic_rashi` (v2, Noto Rashi only), built from the
2026-08-19/20 font-probe findings (600 samples x v17/v18b x 20 faces):

* True cursive (Solitreo) is a capability CLIFF for both models (CER 0.68-0.76,
  <=9/30 substantive reads) -> cursive gets the largest weight.
* v18b's handwriting-tuned vision REGRESSED on unseen square print for two
  glyph pairs (ס→ם up to 41%, ג→נ 10-16% on the Arial family) -> square faces
  return as a drill bucket with those pairs weighted highest.
* ה/ח is calligraphic-face-specific (StamSefarad v17 6.2% -> v18b 2.1%) ->
  STA"M faces + ה/ח drills keep pushing it down.

Reuses v2's render/degradation machinery and text modes; adds a weighted font
bank with per-font cmap filtering (probe lesson: letters-only coverage misses
geresh/gershayim -> tofu) and measured drill-pair weights.  The eval split is
the pure-perception multi-font benchmark; the frozen probe slice under
``transcription_results/confusable_probe/fonts`` stays the before/after gate.

Fonts: Noto Rashi ships in assets/noto_rashi (see synthetic_rashi.py); the
Rashi-type extras fetch per confusable_probe_fonts.py's docstring; square
faces are macOS system fonts (rendered locally, never redistributed).

Usage (from repo root):
    PYTHONPATH=. python -m src.finetuning.qwen_hebrew.synthetic_hebrew_v3 \\
        [--n_samples 12000] [--preview] [--push-to-hub isaacmg/synthetic_hebrew_v3]
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, Features, Image, Value
from fontTools.ttLib import TTCollection, TTFont

from src.finetuning.qwen_hebrew.synthetic_rashi import (
    FINAL_FORMS,
    FONT_DIR as NOTO_DIR,
    HEBREW_LETTERS,
    MARKS,
    MODES,
    _corpus_words,
    render_block,
)

_REPO = Path(__file__).resolve().parents[3]
_EXTRA = _REPO / "src/finetuning/qwen_hebrew/assets/extra_fonts"
DEFAULT_OUT = _REPO / "src/datasets/processed/synthetic_hebrew_v3"
SEED = 20260820

SYNTHETIC_HEBREW_PROMPT = """This image shows Hebrew text printed in type. The typeface may be square, \
semi-cursive (Rashi), or cursive script.
The text may not form meaningful sentences — transcribe the characters exactly
as written, right to left, preserving line breaks. Do NOT correct spelling or
substitute familiar words. Return ONLY the transcription."""

# (name, path or dir-glob, style, weight).  Weights follow the measured
# priority: cursive 0.30, rashi-type anchor 0.20, calligraphic 0.20, square
# drill bucket 0.30 (split evenly across available square faces).
FONT_BANK: List[Tuple[str, str, str, float]] = [
    ("Solitreo", str(_EXTRA / "Solitreo-Regular.ttf"), "cursive", 0.30),
    ("NotoRashi", str(NOTO_DIR / "*.ttf"), "rashi", 0.20),
    ("StamAshkenaz", str(_EXTRA / "StamAshkenazCLM.ttf"), "calligraphic", 0.10),
    ("StamSefarad", str(_EXTRA / "StamSefaradCLM.ttf"), "calligraphic", 0.10),
]
_SQUARE_FACES = [
    ("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf"),
    ("ArialHebrew", "/System/Library/Fonts/ArialHB.ttc"),
    ("Tahoma", "/System/Library/Fonts/Supplemental/Tahoma.ttf"),
    ("TimesNewRoman", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
    ("MicrosoftSansSerif", "/System/Library/Fonts/Supplemental/Microsoft Sans Serif.ttf"),
    ("SFHebrew", "/System/Library/Fonts/SFHebrew.ttf"),
    ("NewPeninimMT", "/System/Library/Fonts/Supplemental/NewPeninimMT.ttc"),
    ("Raanana", "/System/Library/Fonts/Supplemental/Raanana.ttc"),
]
SQUARE_BUCKET_WEIGHT = 0.30

# Measured drill priorities (probe 2026-08-20): the two v18b print
# regressions first, then the manuscript/calligraphic pairs.
DRILL_PAIRS: List[Tuple[str, str, float]] = [
    ("ס", "ם", 0.25), ("ג", "נ", 0.20), ("ב", "כ", 0.15), ("ה", "ח", 0.15),
    ("ד", "ר", 0.10), ("ו", "ן", 0.05), ("ם", "מ", 0.05), ("ט", "מ", 0.05),
]
MODE_WEIGHTS = {"shuffled_words": 0.45, "random_chars": 0.20,
                "confusable_drill": 0.35}

FEATURES = Features({
    "image": Image(),
    "question": Value("string"),
    "answer": Value("string"),
    "task": Value("string"),
    "section": Value("string"),
    "stem": Value("string"),
    "label_source": Value("string"),
    "font": Value("string"),
    "style": Value("string"),
    "mode": Value("string"),
    "image_width": Value("int32"),
    "image_height": Value("int32"),
})


def font_cmap(path: str) -> set:
    """Codepoints a font file can render.

    :param path: TTF/TTC path.
    :type path: str
    :return: Set of supported codepoints.
    :rtype: set
    """
    font = (TTCollection(path).fonts[0] if path.lower().endswith(".ttc")
            else TTFont(path, lazy=True))
    return set((font.getBestCmap() or {}).keys())


def load_font_bank() -> List[Dict]:
    """Resolve the configured bank to available font files with cmaps.

    :return: Entries ``{name, files, style, weight, cmap}``; missing files
        are skipped with a warning so the generator degrades gracefully.
    :rtype: List[Dict]
    """
    import glob as _glob

    bank: List[Dict] = []
    entries = list(FONT_BANK) + [
        (name, path, "square", SQUARE_BUCKET_WEIGHT / len(_SQUARE_FACES))
        for name, path in _SQUARE_FACES
    ]
    for name, spec, style, weight in entries:
        files = sorted(_glob.glob(spec)) if "*" in spec else \
            ([spec] if Path(spec).exists() else [])
        if not files:
            print(f"WARNING: font {name} missing ({spec}) — skipped")
            continue
        cmap = font_cmap(files[0])
        for f in files[1:]:
            cmap &= font_cmap(f)
        bank.append(dict(name=name, files=[Path(f) for f in files],
                         style=style, weight=weight, cmap=cmap))
    return bank


def weighted_drill_text(rng: random.Random, words: List[str]) -> str:
    """Confusable drill with measured pair weights.

    :param rng: Seeded RNG.
    :type rng: random.Random
    :param words: Corpus word pool.
    :type words: List[str]
    :return: Text alternating base words and swapped variants.
    :rtype: str
    """
    pairs = [(a, b) for a, b, _ in DRILL_PAIRS]
    weights = [w for _, _, w in DRILL_PAIRS]
    out: List[str] = []
    for _ in range(rng.randint(10, 30)):
        w = rng.choice(words)
        a, b = rng.choices(pairs, weights=weights)[0]
        if a in w:
            out.extend([w, w.replace(a, b, 1)])
        elif b in w:
            out.extend([w, w.replace(b, a, 1)])
        else:
            pos = rng.randrange(len(w))
            out.extend([w[:pos] + a + w[pos + 1:], w[:pos] + b + w[pos + 1:]])
    return " ".join(out)


def generate(n_samples: int, out_dir: Path, seed: int,
             eval_fraction: float = 0.08,
             push_to_hub: Optional[str] = None) -> DatasetDict:
    """Render the dataset, save to disk, optionally push to the hub.

    :param n_samples: Total samples across all fonts.
    :type n_samples: int
    :param out_dir: Output directory (images/ + arrow dataset).
    :type out_dir: Path
    :param seed: RNG seed.
    :type seed: int
    :param eval_fraction: Share reserved for the eval split.
    :type eval_fraction: float
    :param push_to_hub: Private hub repo id, or None.
    :type push_to_hub: Optional[str]
    :return: The train/eval DatasetDict.
    :rtype: DatasetDict
    """
    rng = random.Random(seed)
    bank = load_font_bank()
    words = _corpus_words()
    print(f"fonts: {[(b['name'], round(b['weight'], 3)) for b in bank]}; "
          f"{len(words)} corpus words")

    mode_fns = {m[0]: m[1] for m in MODES}
    mode_fns["confusable_drill"] = weighted_drill_text
    mode_names = list(MODE_WEIGHTS)
    mode_w = [MODE_WEIGHTS[m] for m in mode_names]
    font_w = [b["weight"] for b in bank]

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    allowed_base = set(HEBREW_LETTERS + FINAL_FORMS + MARKS + " \n")
    for i in range(n_samples):
        fb = rng.choices(bank, weights=font_w)[0]
        mode = rng.choices(mode_names, weights=mode_w)[0]
        text = mode_fns[mode](rng, words)
        text = "".join(ch for ch in text
                       if ch in " \n" or (ch in allowed_base and ord(ch) in fb["cmap"]))
        img, gt = render_block(text, rng, fb["files"])
        name = f"synv3_{i:06d}.jpg"
        img.save(images_dir / name, "JPEG", quality=92)
        rows.append(dict(
            image=str(images_dir / name), question=SYNTHETIC_HEBREW_PROMPT,
            answer=gt, task="synthetic_hebrew", section=fb["style"],
            stem=f"synv3_{i:06d}", label_source="synthetic",
            font=fb["name"], style=fb["style"], mode=mode,
            image_width=img.width, image_height=img.height,
        ))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_samples}", flush=True)

    rng_split = random.Random(seed + 1)
    eval_idx = set(rng_split.sample(range(len(rows)),
                                    int(len(rows) * eval_fraction)))
    train = [r for i, r in enumerate(rows) if i not in eval_idx]
    eval_ = [r for i, r in enumerate(rows) if i in eval_idx]
    import collections
    print("per-font counts:", dict(collections.Counter(r["font"] for r in rows)))
    print(f"train {len(train)} / eval {len(eval_)}")

    dsd = DatasetDict({
        "train": Dataset.from_list(train, features=FEATURES),
        "eval": Dataset.from_list(eval_, features=FEATURES),
    })
    dsd.save_to_disk(str(out_dir / "dataset"))
    with open(out_dir / "stats.json", "w") as fh:
        json.dump(dict(n=len(rows),
                       per_font=dict(collections.Counter(r["font"] for r in rows)),
                       per_mode=dict(collections.Counter(r["mode"] for r in rows)),
                       seed=seed), fh, indent=2)
    if push_to_hub:
        import os

        import dotenv
        dotenv.load_dotenv(str(_REPO / ".env"))
        dsd.push_to_hub(push_to_hub, private=True,
                        token=os.environ.get("HF1_TOKEN"))
        print(f"pushed to {push_to_hub} (private)")
    return dsd


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=12000)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval_fraction", type=float, default=0.08)
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.preview:
        args.n_samples = 24
    generate(args.n_samples, args.out_dir, args.seed, args.eval_fraction,
             args.push_to_hub)


if __name__ == "__main__":
    main()
