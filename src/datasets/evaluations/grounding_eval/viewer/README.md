# Grounding viewer

A single-file web tool that overlays a model's predicted boxes on the eval
page images next to the KTIV ground-truth boxes (toggle either layer, show the
overlap, compare models side by side, inspect the raw model output).

## Rebuild after a new decode

1. Decode the model(s) for the viewer — this writes `../preds/<mode>_<model>.json`
   (raw output + parsed boxes/text + per-query scores) and, because of
   `--no-results`, leaves the canonical `../grounding_eval_*_results.json`
   scores untouched:

       viewer/decode_preds.sh qwen3-vl-8b-heb-v20a-step1800 [more LM Studio keys]

   (`grounding_eval.py --run MODE --model KEY` without `--no-results` does the
   canonical scoring AND writes the same preds dump.)

2. Build the page (from the repo root):

       .venv/bin/python src/datasets/evaluations/grounding_eval/viewer/build_viewer.py

   Output: `grounding_viewer.html` (open it directly — no server needed).
   `--max-side 1300` keeps the file under the 16 MB Artifact limit for a
   hosted copy; the default 1800 is for local use.

## Conventions

* Coordinates are 0–1000 normalized to the scan (the eval's convention);
  boxes are drawn on the embedded, downscaled image by scaling those.
* The header stats are recomputed from the decode in the page; the muted
  "card run" numbers are the canonical results files for the same model, so
  the gap between them is decode noise at temperature 0.1.
* Grounded mode uses the eval's alignment: each predicted line is matched to
  the GT line with the highest text similarity, matched only if ≥ 0.5.
