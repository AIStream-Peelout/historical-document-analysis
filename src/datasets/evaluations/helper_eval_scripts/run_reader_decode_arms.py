# File name: run_reader_decode_arms.py
# Date: 10/29/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Decode-time arms of the production page reader on the Arabic-script benchmark's first 60 documents (117 images).

Every arm is :func:`page_reader.read_page` with a :class:`page_reader.ReaderSettings` built from this CLI's flags
(the options of :mod:`src.models.ocr.reader_options`), so an arm is a reproducible reader setting, not a one-off
script::

    arm0  production reader (script-neutral prompt, 1,200 tokens, letter loop guard + T=0 retry on length, ink gate)
    arm1  --loop-detector markers           (``[...]`` floods and non-letter repeats are loops; looped reads retried)
    arm2  --orientation probe               (four 48-token probes with log-probabilities, read in the chosen turn)
          --orientation gated               (probe only after a suspect upright read: token limit, loop, markers)
    arm3  --crops always|auto               (band crops of the fragment box, joined, kept when not worse)
    arm4  --script-filter                   (post-filter of Latin / CJK / other-script characters, retry when >= 3)
    arm5  all of the above

**Request cache.** Every request (model, image bytes, prompt, temperature, budget, extra payload) is answered from
``<out>/request_cache.jsonl`` when the same request was made before, by any arm, and stored there otherwise. Arms
therefore differ only where their decode decisions differ: arm 1 re-uses arm 0's first reads and pays only for its
extra retries, a replay of an arm with other thresholds costs no request (``--cache-only`` refuses a miss), and the
generation noise of a re-read is measured once, separately, by ``--no-cache`` (``arm0_rerun``).

**Orientation calibration** (``--calibrate-orientation N``): N upright images of documents *outside* the 60 are
probed in their four turns; for every turn the page could have arrived in, the choice of
:func:`reader_options.choose_orientation` is checked against the truth, and the probe statistic and margin that
keep every upright page upright while turning the most turned pages are written to
``orientation_calibration.json`` (read by ``--orientation-margin-from``).

Outputs (``--out-dir``, default ``<benchmark>/outputs_decode_1029``): ``<arm>.jsonl`` (one row per image: the panel
fields and every :data:`page_reader.READ_FIELDS` key, ``text`` scored by ``score_arabic_covered``), ``panel.jsonl``,
``request_cache.jsonl``, ``views/`` (turned and cropped images) and the calibration files. Resumable: an image with a
stored read is skipped.

Shared machine: the run never pauses for other LM Studio work (user 2026-11-08: run in parallel; set
``LMS_YIELD_MODE=dryrun`` so the transcriber's yield gate only logs). It waits only below a disk floor and
unloads only its own model when free disk falls under ``--unload-below-gb`` (the VL worker's temp file grows with each
image request and is released only when the worker exits).

Usage::

    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.run_reader_decode_arms --arm arm0
    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.run_reader_decode_arms --arm arm1 --loop-detector markers
    .venv/bin/python -m src.datasets.evaluations.helper_eval_scripts.run_reader_decode_arms --calibrate-orientation 12
"""
import argparse
import asyncio
import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.datasets.evaluations.helper_eval_scripts import run_prompt_ablation as rpa
from src.datasets.evaluations.helper_eval_scripts import score_arabic_benchmark as sab
from src.datasets.evaluations.helper_eval_scripts import score_arabic_covered as sac
from src.finetuning.qwen_hebrew.eval_harness.v22_task_eval import free_gb, model_served
from src.models.ocr import page_reader as pr
from src.models.ocr import reader_options as ro
from src.models.ocr.lms_transcriber import transcribe_with_lm_studio_meta

DEFAULT_MODEL = "qwen3-vl-8b-heb-v25a-step1200"
DEFAULT_OUT_DIR = "outputs_decode_1029"
PANEL_SOURCE = "outputs/qwen3-vl-8b-heb-v25a-step1200__trained.jsonl"   # the 117 images of the first 60 documents
CACHE_FILE = "request_cache.jsonl"
CALIBRATION_ROWS = "orientation_calibration.jsonl"
CALIBRATION_FILE = "orientation_calibration.json"
CALIBRATION_SEED = 20261029
STATISTICS = ("mean_logprob", "first_logprob")
MARGINS = tuple(round(0.05 * i, 2) for i in range(0, 41))          # 0.00 .. 2.00 nats per token
PROTECTED_MODELS = ("qwen/qwen3-4b-2507", "qwen/qwen3.6-35b-a3b", "qwen/qwen3.8-27b", "qwen3-vl-8b-heb-v21b-step1200")


class CacheMiss(RuntimeError):
    """A ``--cache-only`` run asked for a request no arm has made."""


class RequestCache:
    """A transcriber that answers repeated requests from a JSONL file (see the module docstring).

    :param path: The cache file (appended to).
    :type path: Path
    :param transcribe: The real transcriber (default :func:`lms_transcriber.transcribe_with_lm_studio_meta`).
    :param cache_only: Raise :class:`CacheMiss` instead of calling the model.
    :type cache_only: bool
    :param enabled: False: never answer from the file and never write it (a fresh read of every request).
    :type enabled: bool
    """

    def __init__(self, path: Path, transcribe: Optional[Callable[..., Awaitable[Optional[Dict[str, Any]]]]] = None,
                 cache_only: bool = False, enabled: bool = True) -> None:
        self.path, self.cache_only, self.enabled = path, cache_only, enabled
        self.transcribe = transcribe or transcribe_with_lm_studio_meta
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.hits = self.misses = 0
        self._digests: Dict[Tuple[str, int, float], str] = {}
        if enabled and path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self.entries[row["key"]] = row["meta"]

    def digest(self, image_path: str) -> str:
        """SHA-1 of an image file's bytes (memoised by path, size and modification time).

        :param image_path: Image file.
        :type image_path: str
        :return: Hex digest.
        :rtype: str
        """
        stat = Path(image_path).stat()
        key = (str(image_path), stat.st_size, stat.st_mtime)
        if key not in self._digests:
            self._digests[key] = hashlib.sha1(Path(image_path).read_bytes()).hexdigest()
        return self._digests[key]

    def key(self, model_id: str, image_path: str, prompt: str, temperature: float, max_tokens: int,
            extra_payload: Optional[Mapping[str, Any]] = None) -> str:
        """The cache key of a request.

        :param model_id: Model key.
        :type model_id: str
        :param image_path: Image file (its bytes, not its name, enter the key).
        :type image_path: str
        :param prompt: Prompt text.
        :type prompt: str
        :param temperature: Sampling temperature.
        :type temperature: float
        :param max_tokens: Output budget.
        :type max_tokens: int
        :param extra_payload: Further request fields.
        :type extra_payload: Optional[Mapping[str, Any]]
        :return: Hex digest.
        :rtype: str
        """
        body = json.dumps({"model": model_id, "image": self.digest(image_path), "prompt": prompt, "temperature": float(temperature),
                           "max_tokens": int(max_tokens), "extra": dict(extra_payload or {})}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(body.encode()).hexdigest()

    async def __call__(self, model_id: str, image_path: str, prompt: str, temperature: float = 0.1, max_tokens: int = 1200,
                       **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Answer a request from the cache or the model (the :func:`page_reader.read_page` ``transcribe`` signature).

        :param model_id: Model key.
        :type model_id: str
        :param image_path: Image file.
        :type image_path: str
        :param prompt: Prompt text.
        :type prompt: str
        :param temperature: Sampling temperature.
        :type temperature: float
        :param max_tokens: Output budget.
        :type max_tokens: int
        :param kwargs: Passed to the transcriber (``extra_payload`` enters the key).
        :type kwargs: Any
        :return: Completion metadata, None when the request failed (failures are not cached).
        :rtype: Optional[Dict[str, Any]]
        :raises CacheMiss: With ``cache_only`` and no stored answer.
        """
        key = self.key(model_id, image_path, prompt, temperature, max_tokens, kwargs.get("extra_payload")) if self.enabled else None
        if key is not None and key in self.entries:
            self.hits += 1
            return dict(self.entries[key])
        if self.cache_only:
            raise CacheMiss(f"no cached answer for {Path(image_path).name} T={temperature} max_tokens={max_tokens}")
        self.misses += 1
        meta = await self.transcribe(model_id, image_path, prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
        if meta is not None and key is not None:
            self.entries[key] = meta
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "image": Path(image_path).name, "temperature": temperature, "max_tokens": max_tokens,
                                     "extra": kwargs.get("extra_payload"), "ts": round(time.time(), 1), "meta": meta},
                                    ensure_ascii=False) + "\n")
        return meta


# --- panel --------------------------------------------------------------------------------------------------------------------

def build_panel(benchmark: Path, source: str = PANEL_SOURCE, coverage_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """The arms' images: every image the saved trained-prompt v2.5a-1200 run read (the first 60 documents), in that order.

    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :param source: Answer file whose images form the panel.
    :type source: str
    :param coverage_path: Coverage map (default ``<benchmark>/coverage_v2.jsonl``).
    :type coverage_path: Optional[Path]
    :return: ``{"doc_id", "image_index", "image_file", "image_class" (the map's role), "script"}`` per image.
    :rtype: List[Dict[str, Any]]
    """
    coverage = sac.load_coverage(coverage_path or benchmark / sac.DEFAULT_COVERAGE)
    panel = []
    with open(benchmark / source, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                cov = coverage[row["doc_id"]][int(row["image_index"])]
                panel.append({"doc_id": row["doc_id"], "image_index": int(row["image_index"]), "image_file": cov["image_file"],
                              "image_class": cov.get("role") or cov.get("status"), "script": cov.get("script")})
    return panel


def calibration_images(benchmark: Path, exclude_docs: Sequence[str], n: int, seed: int = CALIBRATION_SEED,
                       coverage_path: Optional[Path] = None, hebrew_share: float = 0.25) -> List[Dict[str, Any]]:
    """Upright images of documents outside the panel for the orientation calibration.

    Arabic-script covered images and Hebrew-script uncovered sides whose map row names no turn (no override mentioning
    rotation, upside down or degrees), sampled with a fixed seed.

    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :param exclude_docs: Documents of the arms' panel.
    :type exclude_docs: Sequence[str]
    :param n: Images to take.
    :type n: int
    :param seed: Sampling seed.
    :type seed: int
    :param coverage_path: Coverage map.
    :type coverage_path: Optional[Path]
    :param hebrew_share: Share of the images taken from Hebrew-script sides.
    :type hebrew_share: float
    :return: Rows like :func:`build_panel`'s.
    :rtype: List[Dict[str, Any]]
    """
    coverage = sac.load_coverage(coverage_path or benchmark / sac.DEFAULT_COVERAGE)
    skip = set(exclude_docs)
    turned = ("rotat", "upside", "degree", "turned", "vertical")
    arabic, hebrew = [], []
    for doc_id, images in sorted(coverage.items()):
        if doc_id in skip:
            continue
        for index, cov in sorted(images.items()):
            note = (cov.get("override_reason") or "").lower()
            if any(t in note for t in turned):
                continue
            row = {"doc_id": doc_id, "image_index": index, "image_file": cov["image_file"],
                   "image_class": cov.get("role") or cov.get("status"), "script": cov.get("script")}
            if row["image_class"] == sac.COVERED and cov.get("script") == "arabic":
                arabic.append(row)
            elif row["image_class"] == sac.UNCOVERED and cov.get("script") == "hebrew":
                hebrew.append(row)
    rng = random.Random(seed)
    n_hebrew = min(len(hebrew), int(round(n * hebrew_share)))
    return rng.sample(arabic, min(len(arabic), n - n_hebrew)) + rng.sample(hebrew, n_hebrew)


# --- orientation calibration --------------------------------------------------------------------------------------------------

def presented_probes(absolute: Mapping[int, Mapping[str, Any]], presented: int) -> Dict[int, Mapping[str, Any]]:
    """The probes a page would get had it arrived turned ``presented`` degrees (counter-clockwise).

    A probe applies its own turn ``r`` to the page as it arrived, so it shows the upright page turned
    ``presented + r``: the four probes of the upright page answer every presentation.

    :param absolute: Probe features of the upright page turned 0 / 90 / 180 / 270 degrees.
    :type absolute: Mapping[int, Mapping[str, Any]]
    :param presented: The turn the page arrived in.
    :type presented: int
    :return: ``{r: features}`` for the presented page.
    :rtype: Dict[int, Mapping[str, Any]]
    """
    return {r: absolute[(presented + r) % 360] for r in ro.ROTATIONS}


def calibration_outcomes(rows: Sequence[Mapping[str, Any]], statistic: str, margin: float,
                         min_script_share: float = ro.MIN_PROBE_SCRIPT_SHARE) -> Dict[str, int]:
    """How a rule decides the calibration pages in every presentation.

    :param rows: Calibration rows (``probes``: absolute turn -> features).
    :type rows: Sequence[Mapping[str, Any]]
    :param statistic: Probe statistic.
    :type statistic: str
    :param margin: Margin in nats per token.
    :type margin: float
    :param min_script_share: Probe script-letter floor.
    :type min_script_share: float
    :return: ``upright_kept`` / ``upright_total`` (pages that arrived upright and stayed), ``turned_fixed`` /
        ``turned_total`` (pages that arrived turned and were turned back exactly).
    :rtype: Dict[str, int]
    """
    out = {"upright_kept": 0, "upright_total": 0, "turned_fixed": 0, "turned_total": 0}
    for row in rows:
        absolute = {int(k): v for k, v in row["probes"].items()}
        for presented in ro.ROTATIONS:
            chosen, _ = ro.choose_orientation(presented_probes(absolute, presented), margin, min_script_share, statistic)
            correct = chosen == (-presented) % 360
            if presented == 0:
                out["upright_total"] += 1
                out["upright_kept"] += correct
            else:
                out["turned_total"] += 1
                out["turned_fixed"] += correct
    return out


def calibrate(rows: Sequence[Mapping[str, Any]], statistics: Sequence[str] = STATISTICS,
              margins: Sequence[float] = MARGINS) -> Dict[str, Any]:
    """Choose the probe statistic and margin on the calibration pages.

    Rules that keep every upright page upright are ranked by the turned pages they fix; among the best, the statistic
    listed first wins and the margin is the middle of its tied margins (away from both failure edges). When no rule
    keeps every upright page, the rule with the most upright pages kept (then turned fixed) is taken.

    :param rows: Calibration rows.
    :type rows: Sequence[Mapping[str, Any]]
    :param statistics: Statistics tried, in preference order.
    :type statistics: Sequence[str]
    :param margins: Margins tried.
    :type margins: Sequence[float]
    :return: ``{"statistic", "margin", "outcome", "grid": [{statistic, margin, upright_kept, ...}]}``.
    :rtype: Dict[str, Any]
    """
    grid = [dict(statistic=s, margin=m, **calibration_outcomes(rows, s, m)) for s in statistics for m in margins]
    safe = [g for g in grid if g["upright_kept"] == g["upright_total"]]
    pool = safe or grid
    best_key = max((g["upright_kept"], g["turned_fixed"]) for g in pool)
    for statistic in statistics:
        tied = [g for g in pool if g["statistic"] == statistic and (g["upright_kept"], g["turned_fixed"]) == best_key]
        if tied:
            chosen = tied[len(tied) // 2]
            return {"statistic": statistic, "margin": chosen["margin"], "safe": bool(safe),
                    "outcome": {k: chosen[k] for k in ("upright_kept", "upright_total", "turned_fixed", "turned_total")},
                    "grid": grid}
    raise RuntimeError("no calibration rule")      # unreachable: pool is never empty


async def run_calibration(images: Sequence[Mapping[str, Any]], model: str, benchmark: Path, out_dir: Path,
                          settings: pr.ReaderSettings, cache: RequestCache) -> Dict[str, Any]:
    """Probe each calibration image in its four turns, then choose the rule (:func:`calibrate`).

    :param images: :func:`calibration_images` rows.
    :type images: Sequence[Mapping[str, Any]]
    :param model: Model key.
    :type model: str
    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :param out_dir: Output directory (``orientation_calibration.jsonl`` / ``.json``).
    :type out_dir: Path
    :param settings: Reader settings (probe budget, view directory).
    :type settings: pr.ReaderSettings
    :param cache: Request cache.
    :type cache: RequestCache
    :return: The calibration (also written).
    :rtype: Dict[str, Any]
    """
    _, prompt = pr.select_prompt()
    rows = []
    for n, image in enumerate(images, 1):
        path = str(benchmark / "images" / image["image_file"])
        _, orientation = await pr.probe_orientation(model, path, prompt, settings, cache)
        rows.append(dict(image, probes=orientation["probes"]))
        summary = {r: (p["mean_logprob"], p["script_share"]) for r, p in orientation["probes"].items()}
        print(f"  calibration {n}/{len(images)} {image['doc_id']}#{image['image_index']} [{image['script']}]: {summary}", flush=True)
    with open(out_dir / CALIBRATION_ROWS, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    result = calibrate(rows)
    result["images"] = len(rows)
    (out_dir / CALIBRATION_FILE).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"calibration: {result['statistic']} margin {result['margin']} -> {result['outcome']}", flush=True)
    return result


# --- arms -----------------------------------------------------------------------------------------------------------------------

def row_for(panel_row: Mapping[str, Any], read: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    """One output row: the panel fields, the arm name and every read field.

    :param panel_row: Panel row.
    :type panel_row: Mapping[str, Any]
    :param read: :func:`page_reader.read_page` result.
    :type read: Mapping[str, Any]
    :param arm: Arm name.
    :type arm: str
    :return: The row.
    :rtype: Dict[str, Any]
    """
    return {"doc_id": panel_row["doc_id"], "image_index": int(panel_row["image_index"]), "image_file": panel_row["image_file"],
            "image_class": panel_row.get("image_class"), "arm": arm, **{k: read.get(k) for k in pr.READ_FIELDS}}


def unload_own_model(model: str, lms: Path = rpa.LMS) -> str:
    """Unload this run's own model (never one of :data:`PROTECTED_MODELS`) to release its worker's temp file.

    :param model: Model key.
    :type model: str
    :param lms: The ``lms`` executable.
    :type lms: Path
    :return: The command's last output line.
    :rtype: str
    :raises ValueError: For a protected model.
    """
    if model in PROTECTED_MODELS:
        raise ValueError(f"refusing to unload the production model {model}")
    proc = subprocess.run([str(lms), "unload", model], capture_output=True, text=True, timeout=120)
    lines = (proc.stdout + proc.stderr).strip().splitlines()
    return lines[-1] if lines else ""


async def run_arm(panel: Sequence[Mapping[str, Any]], model: str, arm: str, out_dir: Path, benchmark: Path,
                  settings: pr.ReaderSettings, cache: RequestCache, limit: Optional[int] = None, min_free_gb: float = 15.0,
                  unload_below_gb: float = 0.0, wait_s: float = 30.0,
                  free_check: Optional[Callable[[], float]] = None,
                  served_check: Optional[Callable[[str], bool]] = None,
                  unload: Optional[Callable[[str], str]] = None) -> Dict[str, int]:
    """Read every panel image of one arm that has no stored read yet.

    :param panel: Panel rows.
    :type panel: Sequence[Mapping[str, Any]]
    :param model: Model key.
    :type model: str
    :param arm: Arm name (``<out_dir>/<arm>.jsonl``).
    :type arm: str
    :param out_dir: Output directory.
    :type out_dir: Path
    :param benchmark: Benchmark directory.
    :type benchmark: Path
    :param settings: Reader settings of the arm.
    :type settings: pr.ReaderSettings
    :param cache: Request cache (the reader's transcriber).
    :type cache: RequestCache
    :param limit: Only the first N images.
    :type limit: Optional[int]
    :param min_free_gb: Disk floor; the run waits below it.
    :type min_free_gb: float
    :param unload_below_gb: Unload the model when free disk falls under this (0: never).
    :type unload_below_gb: float
    :param wait_s: Seconds between disk checks.
    :type wait_s: float
    :param free_check: ``() -> GiB free`` (default the data volume).
    :param served_check: ``model -> bool``.
    :param unload: ``model -> output`` (default :func:`unload_own_model`).
    :return: ``answered``, ``failed``, ``skipped``, ``stopped``, ``hits``, ``misses``, ``unloads``.
    :rtype: Dict[str, int]
    """
    free_check = free_check or free_gb
    served_check = served_check or model_served
    unload = unload or unload_own_model
    out_path = out_dir / f"{arm}.jsonl"
    jobs, skipped = rpa.pending_jobs(panel, out_path, limit)
    stats = {"answered": 0, "failed": 0, "skipped": skipped, "stopped": 0, "unloads": 0}
    hits0, misses0 = cache.hits, cache.misses
    print(f"{arm} [{pr.options_tag(settings) or 'production'}]: {len(jobs)} images to read, {skipped} stored", flush=True)
    started = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for n, row in enumerate(jobs, 1):
            # Other LM Studio work never pauses the run. A cache-only replay sends no request, so it never unloads.
            if unload_below_gb and not cache.cache_only and free_check() < unload_below_gb:
                print(f"  {free_check():.1f} GiB free < {unload_below_gb}: unloading {model}: {unload(model)}", flush=True)
                stats["unloads"] += 1
            while free_check() < min_free_gb:
                print(f"  disk floor: {free_check():.1f} GiB free < {min_free_gb}; waiting", flush=True)
                await asyncio.sleep(wait_s)
            read = await pr.read_page(model, str(benchmark / "images" / row["image_file"]), settings, transcribe=cache)
            if read["reason"] == pr.REASON_FAILED and not served_check(model):
                print(f"  {model} is no longer served; stopping after {n - 1} images (resumable)", flush=True)
                stats["stopped"] = 1
                break
            fh.write(json.dumps(row_for(row, read, arm), ensure_ascii=False) + "\n")
            fh.flush()
            stats["answered" if isinstance(read["text"], str) else "failed"] += 1
            crop = read.get("crop") or {}
            what = ("BLANK (gated)" if read["blank"] else "FAILED" if read["text"] is None else
                    f"{len(read['text'])} chars, finish={read['finish_reason']}"
                    + (f" (first {read['first_finish_reason']})" if read["first_finish_reason"] != read["finish_reason"] else "")
                    + (", LOOPED" if read["looped"] else "") + (", retried" if read["retried"] else "")
                    + (f", turned {read['rotation']}" if read["rotation"] else "")
                    + (f", bands {len(crop.get('bands', []))} {crop.get('accept_reason')}" if crop else "")
                    + (f", stripped {read['stripped_chars']}" if read["stripped_chars"] else ""))
            print(f"  {n:4d}/{len(jobs)} {row['doc_id']}#{row['image_index']} [{row.get('image_class')}]: {what} in {read['secs']}s",
                  flush=True)
    stats.update(hits=cache.hits - hits0, misses=cache.misses - misses0)
    print(f"done {arm}: {stats} in {(time.time() - started) / 60:.1f} min -> {out_path}", flush=True)
    return stats


def settings_from_args(args: argparse.Namespace, out_dir: Path) -> pr.ReaderSettings:
    """The arm's reader settings from the CLI flags.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :param out_dir: Output directory (views go to ``<out_dir>/views``).
    :type out_dir: Path
    :return: Settings.
    :rtype: pr.ReaderSettings
    """
    margin, statistic = args.orientation_margin, args.orientation_statistic
    if args.orientation_margin_from:
        calibration = json.loads(Path(args.orientation_margin_from).read_text(encoding="utf-8"))
        margin, statistic = calibration["margin"], calibration["statistic"]
    return replace(pr.DEFAULT_SETTINGS, loop_detector=args.loop_detector, script_filter=args.script_filter,
                   orientation=args.orientation, orientation_margin=margin, orientation_statistic=statistic,
                   crops=args.crops, crop_min_mp=args.crop_min_mp, crop_max_density=args.crop_max_density,
                   view_dir=str(out_dir / "views"))


def main(argv: Optional[List[str]] = None) -> None:
    """CLI.

    :param argv: Arguments.
    :type argv: Optional[List[str]]
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=sab.DEFAULT_BENCHMARK)
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default <benchmark>/{DEFAULT_OUT_DIR}")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm", default=None, help="arm name (output <out-dir>/<arm>.jsonl)")
    parser.add_argument("--loop-detector", choices=[ro.LOOP_LETTERS, ro.LOOP_MARKERS], default=ro.LOOP_LETTERS)
    parser.add_argument("--script-filter", action="store_true")
    parser.add_argument("--orientation", choices=[ro.ORIENTATION_OFF, ro.ORIENTATION_PROBE, ro.ORIENTATION_GATED],
                        default=ro.ORIENTATION_OFF)
    parser.add_argument("--orientation-margin", type=float, default=ro.ORIENTATION_MARGIN)
    parser.add_argument("--orientation-statistic", choices=STATISTICS, default="mean_logprob")
    parser.add_argument("--orientation-margin-from", default=None, help="calibration JSON (statistic and margin)")
    parser.add_argument("--crops", choices=[ro.CROPS_OFF, ro.CROPS_ALWAYS, ro.CROPS_AUTO], default=ro.CROPS_OFF)
    parser.add_argument("--crop-min-mp", type=float, default=pr.DEFAULT_SETTINGS.crop_min_mp)
    parser.add_argument("--crop-max-density", type=float, default=pr.DEFAULT_SETTINGS.crop_max_density)
    parser.add_argument("--cache-only", action="store_true", help="answer every request from the cache; a miss is an error")
    parser.add_argument("--no-cache", action="store_true", help="fresh requests, not cached (the re-read noise arm)")
    parser.add_argument("--calibrate-orientation", type=int, default=0, help="probe N held-out upright images and choose the rule")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=15.0)
    parser.add_argument("--unload-below-gb", type=float, default=0.0)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or args.benchmark / DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = build_panel(args.benchmark)
    (out_dir / "panel.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in panel), encoding="utf-8")
    settings = settings_from_args(args, out_dir)
    cache = RequestCache(out_dir / CACHE_FILE, cache_only=args.cache_only, enabled=not args.no_cache)
    if args.calibrate_orientation:
        images = calibration_images(args.benchmark, sorted({r["doc_id"] for r in panel}), args.calibrate_orientation)
        asyncio.run(run_calibration(images, args.model, args.benchmark, out_dir, settings, cache))
        return
    if not args.arm:
        parser.error("--arm is required")
    (out_dir / f"{args.arm}.settings.json").write_text(json.dumps(asdict(settings), indent=1, default=str), encoding="utf-8")
    stats = asyncio.run(run_arm(panel, args.model, args.arm, out_dir, args.benchmark, settings, cache, args.limit, args.min_free_gb,
                                args.unload_below_gb))
    print(json.dumps({"arm": args.arm, "stats": stats, "loaded": rpa.model_info(rpa.lms_models(), args.model)}), flush=True)


if __name__ == "__main__":
    main()
