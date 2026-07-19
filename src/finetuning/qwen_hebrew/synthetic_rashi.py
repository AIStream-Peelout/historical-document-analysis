"""Synthetic Rashi-script training data: random Hebrew rendered in Rashi type.

Renders text the model CANNOT know from its language prior — shuffled corpus
words, random character strings, and confusable-letter minimal-pair drills —
in Noto Rashi Hebrew across weights/sizes/degradations. Success on this data
requires reading glyphs, which directly counters the canonical-recitation
failure mode (`canonical_completion` in the benchmark taxonomy).

Also emits a held-out ``eval`` slice usable as a *pure-perception* benchmark
track (language prior is useless on random text — CER measures perception).

Rendering note: this Pillow build lacks libraqm, so RTL is handled manually
by reversing each visual line before drawing. That is glyph-safe for Hebrew
(non-connecting script); the charset deliberately excludes characters with
asymmetric paired forms (parentheses, brackets).

Fonts (gitignored, fetch once):
  curl -sL https://github.com/notofonts/hebrew/releases/download/NotoRashiHebrew-v1.007/NotoRashiHebrew-v1.007.zip \\
      -o /tmp/rashi.zip && unzip -q /tmp/rashi.zip -d src/finetuning/qwen_hebrew/assets/noto_rashi

Usage
-----
  python -m src.finetuning.qwen_hebrew.synthetic_rashi --n_samples 20 --preview
  python -m src.finetuning.qwen_hebrew.synthetic_rashi --n_samples 20000
"""

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from src.datasets.document_models.talmud_gt_parser import load_gt_directory

_REPO = Path(__file__).resolve().parents[3]
FONT_DIR = Path(__file__).parent / "assets/noto_rashi/NotoRashiHebrew/hinted/ttf"
GT_DIR = _REPO / "src/datasets/raw_data/cairo_genizah/evaluations/talmud_full/talmud_complete/texts"
DEFAULT_OUT = _REPO / "src/datasets/processed/synthetic_rashi"

HEBREW_LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"
FINAL_FORMS = "ךםןףץ"
# Noto Rashi Hebrew covers ONLY the Hebrew block: ASCII '"":. have no glyphs
# (tofu). Corpus text is translated to the Hebrew geresh/gershayim and the
# unsupported marks are dropped, so GT always matches what is drawn.
MARKS = "׳״"
_TRANSLATE = str.maketrans({"'": "׳", '"': "״", ":": None, ".": None})
# Pairs the OCR error taxonomy shows are confused in Rashi script
CONFUSABLE_PAIRS = [("ד", "ר"), ("ה", "ח"), ("ו", "ן"), ("ב", "כ"),
                    ("ג", "נ"), ("ם", "ס"), ("ט", "מ")]

SYNTHETIC_RASHI_PROMPT = """This image shows Hebrew text printed in Rashi (semi-cursive) script.
The text may not form meaningful sentences — transcribe the characters exactly
as written, right to left, preserving line breaks. Do NOT correct spelling or
substitute familiar words. Return ONLY the transcription."""


def _corpus_words(gt_dir: Path = GT_DIR, max_pages: int = 400) -> List[str]:
    """Collect a word pool from the real GT corpus (natural char statistics).

    :param gt_dir: Ground-truth text directory.
    :param max_pages: Pages to sample words from (keeps startup fast).
    :returns: List of Hebrew words.
    """
    gt = load_gt_directory(gt_dir)
    words: List[str] = []
    allowed = set(HEBREW_LETTERS + FINAL_FORMS + MARKS)
    for stem in sorted(gt)[:max_pages]:
        for text in gt[stem].values():
            for raw in text.split():
                w = raw.translate(_TRANSLATE)
                if 1 < len(w) <= 12 and w and set(w) <= allowed:
                    words.append(w)
    return words


def _shuffled_words_text(rng: random.Random, words: List[str]) -> str:
    """Sample real words in random order — natural glyphs, broken syntax.

    :param rng: Seeded RNG.
    :param words: Corpus word pool.
    :returns: Space-joined scrambled text.
    """
    n = rng.randint(15, 70)
    return " ".join(rng.choice(words) for _ in range(n))


def _random_chars_text(rng: random.Random, words: List[str]) -> str:
    """Generate random character 'words' — no language signal at all.

    :param rng: Seeded RNG.
    :param words: Unused (uniform signature).
    :returns: Random pseudo-word text with final forms and marks.
    """
    out = []
    for _ in range(rng.randint(15, 60)):
        length = rng.randint(2, 9)
        chars = [rng.choice(HEBREW_LETTERS) for _ in range(length - 1)]
        last = rng.choice(HEBREW_LETTERS + FINAL_FORMS)
        word = "".join(chars) + last
        if rng.random() < 0.15:
            pos = rng.randint(1, len(word) - 1)
            word = word[:pos] + rng.choice(MARKS) + word[pos:]
        out.append(word)
    return " ".join(out)


def _confusable_drill_text(rng: random.Random, words: List[str]) -> str:
    """Minimal pairs over confusable letters — adjacent contrasts.

    :param rng: Seeded RNG.
    :param words: Corpus word pool to mutate.
    :returns: Text alternating base words and confusable-swapped variants.
    """
    out = []
    for _ in range(rng.randint(10, 30)):
        w = rng.choice(words)
        a, b = rng.choice(CONFUSABLE_PAIRS)
        if a in w:
            out.extend([w, w.replace(a, b, 1)])
        elif b in w:
            out.extend([w, w.replace(b, a, 1)])
        else:
            pos = rng.randrange(len(w))
            out.extend([w[:pos] + a + w[pos + 1:], w[:pos] + b + w[pos + 1:]])
    return " ".join(out)


MODES: List[Tuple[str, Callable[[random.Random, List[str]], str], float]] = [
    ("shuffled_words", _shuffled_words_text, 0.5),
    ("random_chars", _random_chars_text, 0.25),
    ("confusable_drill", _confusable_drill_text, 0.25),
]


def _wrap(text: str, font: ImageFont.FreeTypeFont, width_px: int) -> List[str]:
    """Word-wrap text (logical order) to a pixel width.

    :param text: Logical-order text.
    :param font: Rendering font.
    :param width_px: Column width budget.
    :returns: Wrapped lines in logical order.
    """
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if font.getlength(cand) <= width_px or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def render_block(text: str, rng: random.Random, fonts: List[Path]) -> Tuple[Image.Image, str]:
    """Render a text block in Rashi type with augmentations.

    :param text: Logical-order source text.
    :param rng: Seeded RNG.
    :param fonts: Candidate font files.
    :returns: (augmented image, ground-truth text with line breaks).
    """
    font_size = rng.randint(11, 30)
    font = ImageFont.truetype(str(rng.choice(fonts)), font_size)
    col_width = rng.randint(280, 900)
    margin = rng.randint(8, 30)
    line_gap = int(font_size * rng.uniform(0.25, 0.6))

    lines = _wrap(text, font, col_width)
    gt = "\n".join(lines)

    line_h = font_size + line_gap
    height = margin * 2 + line_h * len(lines)
    bg = rng.randint(228, 252)
    img = Image.new("L", (col_width + margin * 2, height), color=bg)
    draw = ImageDraw.Draw(img)

    ink = rng.randint(10, 70)
    for i, line in enumerate(lines):
        visual = line[::-1]  # manual RTL (no libraqm)
        x = margin + col_width - draw.textlength(visual, font=font)
        draw.text((x, margin + i * line_h), visual, font=font, fill=ink)

    if rng.random() < 0.7:
        img = img.rotate(rng.uniform(-1.2, 1.2), expand=True,
                         fillcolor=bg, resample=Image.BICUBIC)
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    if rng.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.15))
    if rng.random() < 0.3:  # print-degradation cycle
        w, h = img.size
        f = rng.uniform(0.6, 0.85)
        img = img.resize((int(w * f), int(h * f))).resize((w, h))
    return img.convert("RGB"), gt


def generate(
    n_samples: int,
    out_dir: Path,
    seed: int,
    eval_fraction: float = 0.05,
) -> None:
    """Generate the synthetic dataset (images + JSONL manifest).

    :param n_samples: Total samples to render.
    :param out_dir: Output directory (``images/`` + ``manifest.jsonl``).
    :param seed: RNG seed for full reproducibility.
    :param eval_fraction: Fraction reserved for the pure-perception eval split.
    """
    rng = random.Random(seed)
    fonts = sorted(FONT_DIR.glob("*.ttf"))
    if not fonts:
        raise FileNotFoundError(f"No Rashi fonts in {FONT_DIR}")
    words = _corpus_words()
    print(f"{len(fonts)} fonts, {len(words)} corpus words")

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    mode_names = [m[0] for m in MODES]
    mode_fns = {m[0]: m[1] for m in MODES}
    mode_weights = [m[2] for m in MODES]

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as sink:
        for i in range(n_samples):
            mode = rng.choices(mode_names, weights=mode_weights)[0]
            img, gt = render_block(mode_fns[mode](rng, words), rng, fonts)
            name = f"synth_{i:06d}.png"
            img.save(images_dir / name)
            sink.write(json.dumps({
                "image": str(images_dir / name),
                "question": SYNTHETIC_RASHI_PROMPT,
                "answer": gt,
                "task": "synthetic_rashi",
                "section": "rashi",
                "stem": f"synth_{i:06d}",
                "label_source": "synthetic",
                "mode": mode,
                "split": "eval" if rng.random() < eval_fraction else "train",
                "image_width": img.width,
                "image_height": img.height,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{n_samples}")
    print(f"✅ {n_samples} samples → {out_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    :param argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Generate synthetic Rashi-script data.")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--preview", action="store_true",
                        help="Render a handful of samples for visual inspection only.")
    args = parser.parse_args(argv)

    if args.preview:
        args.n_samples = min(args.n_samples, 20)
    generate(args.n_samples, args.out_dir, args.seed, args.eval_fraction)


if __name__ == "__main__":
    main()
