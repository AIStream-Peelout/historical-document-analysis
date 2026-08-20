"""Confusable probe, model half: unseen-font synthetic slice through LM Studio.

Answers "is the ה/ח (etc.) problem a FONT-GENERALIZATION gap or a PERCEPTION
gap?"  The free half (``confusable_probe.py``) showed v18b's ה/ח swap rate is
~0 on Vilna print (in-distribution) — so this half renders confusable-drill
and unmemorizable text in fonts the model has NEVER seen (macOS system Hebrew
faces incl. a semi-cursive one) alongside the seen control (Noto Rashi), runs
each generation through LM Studio at temperature 0, and reports per-font CER
and per-pair swap rates.  Re-run unchanged after v1.9a for the before/after.

Queueing: the job waits until LM Studio reports no model GENERATING for
IDLE_POLLS consecutive polls (so it never competes with a user experiment),
with a hard cap after which it proceeds anyway.

Rashi-type faces are fetched into assets/extra_fonts (gitignored, like Noto):
    curl -sL -o Solitreo-Regular.ttf \\
      https://github.com/google/fonts/raw/main/ofl/solitreo/Solitreo-Regular.ttf
    curl -sL https://downloads.sourceforge.net/project/culmus/culmus/0.140/culmus-0.140.tar.gz \\
      | tar xz --strip-components=1 culmus-0.140/StamAshkenazCLM.ttf culmus-0.140/StamSefaradCLM.ttf

Usage (from src/datasets/evaluations):
    PYTHONPATH=<repo> python helper_eval_scripts/confusable_probe_fonts.py \\
        [--models qwen3-vl-8b-heb-v18b-step700,qwen3-vl-8b-heb-v17-step800] \\
        [--per-font 30] [--no-wait]
"""

import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.evaluations.metrics import cer_pair  # noqa: E402
from src.datasets.evaluations.helper_eval_scripts.confusable_probe import (  # noqa: E402
    CANONICAL_PAIRS,
    confusion_events,
    letters,
)
from src.datasets.evaluations.helper_eval_scripts.probe_forgetting import (  # noqa: E402
    call_model,
    image_to_data_url,
)
from src.finetuning.qwen_hebrew.synthetic_rashi import (  # noqa: E402
    FONT_DIR as NOTO_DIR,
    MODES,
    SYNTHETIC_RASHI_PROMPT,
    _corpus_words,
    render_block,
)

OUT_DIR = _REPO / "src/datasets/evaluations/transcription_results/confusable_probe/fonts"
LMS_BIN = os.path.expanduser("~/.lmstudio/bin/lms")
SEED = 20260819
IDLE_POLLS = 5
POLL_S = 60
WAIT_CAP_S = 6 * 3600

_ASSETS = str(_REPO / "src/finetuning/qwen_hebrew/assets/extra_fonts")
UNSEEN_FONTS = {
    # Rashi-type / manuscript-adjacent faces (the user's actual failure case;
    # downloaded OFL/GPL-FE, see assets/extra_fonts): Solitreo is Ladino
    # cursive — the nearest open relative of Genizah documentary hands; the
    # Culmus STA"M faces are sofer calligraphic square.
    "Solitreo": f"{_ASSETS}/Solitreo-Regular.ttf",
    "StamAshkenaz": f"{_ASSETS}/StamAshkenazCLM.ttf",
    "StamSefarad": f"{_ASSETS}/StamSefaradCLM.ttf",
    "ArialUnicode": "/Library/Fonts/Arial Unicode.ttf",
    "ArialHebrew": "/System/Library/Fonts/ArialHB.ttc",
    "LucidaGrande": "/System/Library/Fonts/LucidaGrande.ttc",
    "SFHebrew": "/System/Library/Fonts/SFHebrew.ttf",
    "SFHebrewRounded": "/System/Library/Fonts/SFHebrewRounded.ttf",
    "Arial": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "ArialBold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "ArialItalic": "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "CorsivaHebrew": "/System/Library/Fonts/Supplemental/Corsiva.ttc",
    "CourierNew": "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "MicrosoftSansSerif": "/System/Library/Fonts/Supplemental/Microsoft Sans Serif.ttf",
    "NewPeninimMT": "/System/Library/Fonts/Supplemental/NewPeninimMT.ttc",
    "Raanana": "/System/Library/Fonts/Supplemental/Raanana.ttc",
    "Tahoma": "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "TimesNewRoman": "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "TimesNewRomanBold": "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
}


def lms_busy() -> bool:
    """Return True when LM Studio reports any model GENERATING/PROCESSING.

    :return: Busy flag (False if ``lms`` is unavailable).
    :rtype: bool
    """
    try:
        out = subprocess.run([LMS_BIN, "ps"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False
    return any(tok in out for tok in ("GENERATING", "PROCESSING", "LOADING"))


def wait_for_idle(log) -> None:
    """Block until LM Studio has been idle for IDLE_POLLS polls (or the cap).

    :param log: Callable taking a message string.
    :type log: callable
    """
    quiet = 0
    t0 = time.time()
    while quiet < IDLE_POLLS:
        if time.time() - t0 > WAIT_CAP_S:
            log(f"wait cap {WAIT_CAP_S}s reached — proceeding while LM Studio may be busy")
            return
        busy = lms_busy()
        quiet = 0 if busy else quiet + 1
        log(f"lms busy={busy} quiet_polls={quiet}/{IDLE_POLLS} waited={int(time.time()-t0)}s")
        if quiet < IDLE_POLLS:
            time.sleep(POLL_S)


def build_slice(per_font: int, rng: random.Random) -> list:
    """Render the held-out multi-font slice.

    :param per_font: Samples per font.
    :type per_font: int
    :param rng: Seeded RNG.
    :type rng: random.Random
    :return: List of dicts (font, seen, mode, gt, image_path).
    :rtype: list
    """
    words = _corpus_words()
    fonts = {"NotoRashi": [Path(p) for p in sorted(glob.glob(str(NOTO_DIR / "*.ttf")))]}
    for name, path in UNSEEN_FONTS.items():
        if os.path.exists(path):
            fonts[name] = [Path(path)]
    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    mode_names = [m[0] for m in MODES]
    mode_fns = {m[0]: m[1] for m in MODES}
    mode_w = [m[2] for m in MODES]
    rows = []
    for name, files in fonts.items():
        for i in range(per_font):
            mode = rng.choices(mode_names, weights=mode_w)[0]
            text = mode_fns[mode](rng, words)
            img, gt = render_block(text, rng, files)
            path = img_dir / f"{name}_{i:03d}.jpg"
            img.save(path, "JPEG", quality=92)
            rows.append(dict(font=name, seen=(name == "NotoRashi"), mode=mode, gt=gt,
                             image_path=str(path), width=img.width, height=img.height))
    with open(OUT_DIR / "slice.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def top_up_slice(rows: list, per_font: int) -> list:
    """Render samples for configured fonts missing from an existing slice.

    Fonts added to :data:`UNSEEN_FONTS` after a slice was built (e.g. the
    downloaded Rashi-type faces) get their own rows appended; existing rows
    and their model outputs stay valid.  Each font's RNG is seeded stably
    from its name so later additions never perturb earlier renders.

    :param rows: Current slice rows.
    :type rows: list
    :param per_font: Samples per font.
    :type per_font: int
    :return: Slice rows including any newly rendered fonts.
    :rtype: list
    """
    import zlib

    have = {r["font"] for r in rows}
    missing = {name: path for name, path in UNSEEN_FONTS.items()
               if name not in have and os.path.exists(path)}
    if not missing:
        return rows
    words = _corpus_words()
    img_dir = OUT_DIR / "images"
    mode_names = [m[0] for m in MODES]
    mode_fns = {m[0]: m[1] for m in MODES}
    mode_w = [m[2] for m in MODES]
    new_rows = []
    for name, path in missing.items():
        # Only feed characters the font can render — letters-only coverage was
        # verified, but punctuation like geresh/gershayim may be absent and
        # would render as tofu boxes, unfairly punishing the font's CER.
        from fontTools.ttLib import TTCollection, TTFont
        font_obj = (TTCollection(path).fonts[0] if path.lower().endswith(".ttc")
                    else TTFont(path, lazy=True))
        cmap = set((font_obj.getBestCmap() or {}).keys())
        rng = random.Random(SEED ^ zlib.crc32(name.encode()))
        for i in range(per_font):
            mode = rng.choices(mode_names, weights=mode_w)[0]
            text = mode_fns[mode](rng, words)
            text = "".join(ch for ch in text
                           if ch in " \n" or ord(ch) in cmap)
            img, gt = render_block(text, rng, [Path(path)])
            ipath = img_dir / f"{name}_{i:03d}.jpg"
            img.save(ipath, "JPEG", quality=92)
            new_rows.append(dict(font=name, seen=False, mode=mode, gt=gt,
                                 image_path=str(ipath), width=img.width, height=img.height))
    rows = rows + new_rows
    with open(OUT_DIR / "slice.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def run_model(model: str, rows: list, log) -> Path:
    """Transcribe every slice image with one model (resumable JSONL).

    :param model: LM Studio model id.
    :type model: str
    :param rows: Slice rows.
    :type rows: list
    :param log: Logger callable.
    :type log: callable
    :return: Path of the output JSONL.
    :rtype: Path
    """
    from openai import OpenAI
    from PIL import Image

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    out_path = OUT_DIR / f"outputs_{model.replace('/', '_')}.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            done.add(json.loads(line)["image_path"])
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            if r["image_path"] in done:
                continue
            t0 = time.time()
            data_url = image_to_data_url(Image.open(r["image_path"]))
            out = call_model(client, model, SYNTHETIC_RASHI_PROMPT, data_url, max_tokens=800)
            fh.write(json.dumps(dict(r, model=model, output=out,
                                     seconds=round(time.time() - t0, 1)),
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                log(f"{model}: {i}/{len(rows)} done ({time.time()-t0:.0f}s last)")
    return out_path


def score(model: str, out_path: Path, log) -> None:
    """Per-font CER and confusable-pair swap rates for one model's outputs.

    :param model: Model id (for file naming).
    :type model: str
    :param out_path: Output JSONL from :func:`run_model`.
    :type out_path: Path
    :param log: Logger callable.
    :type log: callable
    """
    rows = [json.loads(l) for l in open(out_path)]
    by_font = collections.defaultdict(list)
    for r in rows:
        by_font[r["font"]].append(r)
    table = []
    log(f"\n== {model} ==")
    log(f"{'font':20s} {'seen':>4s} {'n':>3s} {'CER':>6s} {'reads':>5s}  " +
        " ".join(f"{a}/{b:>4s}" for a, b in CANONICAL_PAIRS[:9]))
    for font, rs in sorted(by_font.items(), key=lambda kv: (not kv[1][0]["seen"], kv[0])):
        cers, ev, glc = [], collections.Counter(), collections.Counter()
        reads = 0
        for r in rs:
            h, g = letters(r["output"]), letters(r["gt"])
            if not g:
                continue
            c, _ = cer_pair(r["output"].strip(), r["gt"].strip())
            cers.append(min(c, 2.0))
            if h and c < 0.6:
                reads += 1
                ev.update(confusion_events(h, g))
                glc.update(g)
        med = sorted(cers)[len(cers) // 2] if cers else float("nan")
        cells = []
        rec = dict(model=model, font=font, seen=rs[0]["seen"], n=len(rs), cer_median=round(med, 4),
                   substantive=reads)
        for a, b in CANONICAL_PAIRS:
            na, nb = glc[a], glc[b]
            rate = (ev[(a, b)] + ev[(b, a)]) / (na + nb) if (na + nb) else None
            rec[f"{a}/{b}"] = round(rate, 4) if rate is not None else None
            if (a, b) in CANONICAL_PAIRS[:9]:
                cells.append(f"{rate*100:5.1f}%" if rate is not None else "    —")
        table.append(rec)
        log(f"{font:20s} {str(rs[0]['seen']):>4s} {len(rs):3d} {med:6.3f} {reads:5d}  " + " ".join(cells))
    with open(OUT_DIR / f"scores_{model.replace('/', '_')}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)


def main() -> None:
    """Wait for LM Studio idle, build the slice, run models, score."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen3-vl-8b-heb-v18b-step700,qwen3-vl-8b-heb-v17-step800")
    parser.add_argument("--per-font", type=int, default=30)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "probe_fonts.log"

    def log(msg: str) -> None:
        """Print and append a timestamped line to the log file."""
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    rng = random.Random(SEED)
    slice_path = OUT_DIR / "slice.jsonl"
    rows = [json.loads(l) for l in open(slice_path)] if slice_path.exists() else build_slice(args.per_font, rng)
    rows = top_up_slice(rows, args.per_font)
    log(f"slice: {len(rows)} samples over {len({r['font'] for r in rows})} fonts")
    if not args.no_wait:
        wait_for_idle(log)
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        log(f"starting {model}")
        out_path = run_model(model, rows, log)
        score(model, out_path, log)
    log("DONE")


if __name__ == "__main__":
    main()
