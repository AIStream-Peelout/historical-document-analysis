"""Render each article exhibit to a 2x PNG (auto-cropped) in article/assets.

Headless Firefox with a profile that sets ``layout.css.devPixelsPerPx=2``
screenshots ``misreadings.html#export=<exhibit id>`` (the page's export mode
isolates one exhibit); the PNG is then cropped to content with a 28px pad.
Reconstructed from the 2026-08-20 scratchpad pipeline, extended to all 14
exhibits.

Usage: python3 article/build/export.py [exhibit-id ...]   (default: all)
"""
import pathlib
import subprocess
import sys

from PIL import Image, ImageChops

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "article/assets"
PROFILE = REPO / "article/build/ffprofile"
FF = "/Applications/Firefox.app/Contents/MacOS/firefox"
URL = f"file://{REPO}/article/misreadings.html"
PAD = 28

EXHIBITS = [
    ("ex-specimen", "01-specimen-four-failures"),
    ("ex-taxonomy", "02-failure-taxonomy"),
    ("ex-talmud", "03-talmud-cer"),
    ("ex-meanmed", "04-mean-vs-median"),
    ("ex-arc", "05-finetune-arc"),
    ("ex-cliff", "06-cross-corpus-cliff"),
    ("ex-script", "07-hallucination-by-script"),
    ("ex-lora", "08-vision-lora-ab"),
    ("ex-conf", "09-letter-confusions"),
    ("ex-bench", "10-benchmark-provenance"),
    ("ex-loop", "11-loop-collapse-rashi"),
    ("ex-talmud-final", "12-talmud-final-results"),
    ("ex-kraken-decomp", "13-kraken-decomposition"),
    ("ex-genizah-final", "14-genizah-final-results"),
]


def export(eid: str, name: str) -> None:
    """Screenshot one exhibit and crop it to content.

    :param eid: The exhibit's DOM id (``ex-...``).
    :param name: Output basename under article/assets.
    """
    dst = OUT / f"{name}.png"
    subprocess.run([FF, "--headless", "--profile", str(PROFILE),
                    "--screenshot", str(dst), "--window-size=2560,3000",
                    f"{URL}#export={eid}"], capture_output=True, timeout=180)
    if not dst.exists():
        print(f"  !! {name}: no output")
        return
    im = Image.open(dst).convert("RGB")
    bg = Image.new("RGB", im.size, im.getpixel((2, 2)))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        l, t, r, b = box
        im = im.crop((max(l - PAD, 0), max(t - PAD, 0),
                      min(r + PAD, im.width), min(b + PAD, im.height)))
    im.save(dst, optimize=True)
    print(f"  {name}.png  {im.size[0]}x{im.size[1]}  {dst.stat().st_size / 1024:.0f} KB")


def main() -> None:
    """Export the requested exhibits (all by default)."""
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = set(sys.argv[1:])
    for eid, name in EXHIBITS:
        if wanted and eid not in wanted and name not in wanted:
            continue
        export(eid, name)


if __name__ == "__main__":
    main()
