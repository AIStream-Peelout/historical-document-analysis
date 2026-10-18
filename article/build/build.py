"""Inject article/build/data.json into the article template.

Reconstructed from the (wiped) 2026-08-20 scratchpad pipeline: the template
carries a ``const DATA = /*__DATA__*/;`` placeholder that receives the whole
dataset (talmud / genizah / genizah_script / confusions / forgetting /
benchmark / specimen / loop / kraken_decomp). ``refresh_data.py`` updates the
result-driven keys from the pipeline CSVs; this script only injects.

Usage: python3 article/build/build.py
"""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "article/misreadings.template.html"
DATA = REPO / "article/build/data.json"
DEST = REPO / "article/misreadings.html"


def main() -> None:
    """Build article/misreadings.html from the template and data.json."""
    tpl = TEMPLATE.read_text()
    data = json.load(open(DATA))
    data.pop("_consistency", None)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    assert "/*__DATA__*/" in tpl, "placeholder missing from template"
    DEST.write_text(tpl.replace("/*__DATA__*/", payload))
    print(f"built {DEST.relative_to(REPO)}  ({DEST.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
