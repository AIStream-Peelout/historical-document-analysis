# File name: test_colab_notebook_novel_v1.py
# Date: 11/3/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for the novel_v1 notebook derivation, its training-rows step and the round's gates.

The Lambda edition must keep v2.5a's training stack and gates, differ only where planned (data repository and pin
guard, this mixture's data-cell gates, warm start v2.5a step 1200, the guardrail batch, the schedule, the step-0 eval,
the Hub-confirmed quarter-point milestones, the train-loss alarm, names), refuse an unpinned dataset, and equal a fresh
derivation at the revision and plan written into it. The rows step applies v2.5a's pass rule to the novel train pages
(every page 2 or 3 times, no source above 2.5 passes, a whole number of steps), sets the script-neutral prompt on every
row and leaves Russian National Library rows out. The round's files (``logs/next_round/novel_v1``, git-ignored) are
tested when they are on this machine.
"""
import ast
import copy
import difflib
import importlib.util
import json
import math
import re
import shlex
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from src.finetuning.qwen_hebrew import build_v22_mixture as bvm
from src.finetuning.qwen_hebrew.build_pgp_editions import save_dataset
from src.finetuning.qwen_hebrew.colab import derive_novel_v1 as dn
from src.finetuning.qwen_hebrew.colab import host_lambda as hl
from src.finetuning.qwen_hebrew.images_once import export_images_once
from src.finetuning.qwen_hebrew.novel_build import build as novel_build
from src.finetuning.qwen_hebrew.novel_build import funnel
from src.finetuning.qwen_hebrew.novel_build.targets import measure_script
from src.finetuning.qwen_hebrew.prompts import (ARABIC_FRAGMENT_PROMPT, FRAGMENT_TRANSCRIBE_PROMPT,
                                                SCRIPT_NEUTRAL_PROMPT)
from src.finetuning.qwen_hebrew.train_session import cosine_warmup_lr, warmup_steps_for

SRC = Path("src/finetuning/qwen_hebrew/colab/genizah_v22b.ipynb")
COLAB_DIR = Path("src/finetuning/qwen_hebrew/colab")
COMMITTED = COLAB_DIR / "genizah_novel_v1_lambda.ipynb"
V25A = COLAB_DIR / "genizah_v25a_lambda.ipynb"
ROUND_DIR = Path("logs/next_round/novel_v1")
README = ROUND_DIR / "README.md"
GATES = ROUND_DIR / "run_notebook_gates_novel_v1.py"
EVAL_CHAIN = ROUND_DIR / "eval_chain_novel_v1.sh"
SHA = "c" * 40
ROWS, PAGES = 2200, 1000
HEBREW = "כתאבי אליך אטאל אללה בקאך ואדאם עזך וסלאמתך [...] שלום רב לאהובי ולכל ביתו ולכל אשר לו אמן\nואעלם אן אלכתב וצלת"
ARABIC = "كتابي اليك اطال الله بقاك وادام عزك وتاييدك وسعادتك وسلامتك من مصر لخمس خلون من شهر رمضان [...]\nوالحمد لله وحده"
MIXED = HEBREW + "\n" + "بسم الله الرحمن الرحيم والحمد لله رب العالمين"
#: names a tracked file may not carry (the builder's list, plus a storage-path prefix), compared lower-case
TRACKED_FORBIDDEN = (*bvm.FORBIDDEN_CARD_STRINGS, "/" + "volumes" + "/")
LEGACY_SOURCE = next(k for k, v in novel_build.LABEL_SOURCES.items() if v == "legacy_store_diplomatic")


@pytest.fixture(scope="module")
def editions() -> tuple:
    """(v2.5a Lambda edition as committed, novel_v1 Colab edition, novel_v1 Lambda edition) at 2,200 rows / 1,000 pages."""
    nb = json.load(open(SRC, encoding="utf-8"))
    colab = dn.derive(nb, SHA, ROWS, PAGES)
    return json.load(open(V25A, encoding="utf-8")), colab, dn.to_lambda(colab, ROWS)


def _src(nb: dict, i: int) -> str:
    """Source of cell ``i``.

    :param nb: Notebook.
    :param i: Cell index.
    :return: The joined source.
    """
    return "".join(nb["cells"][i]["source"])


def _changed(a: dict, b: dict, i: int) -> tuple:
    """(lines only in ``a``, lines only in ``b``) of cell ``i``.

    :param a: First notebook.
    :param b: Second notebook.
    :param i: Cell index.
    :return: Two lists of lines.
    """
    diff = [l for l in difflib.unified_diff(_src(a, i).splitlines(), _src(b, i).splitlines(), lineterm="", n=0)
            if not l.startswith(("---", "+++", "@@"))]
    return [l[1:] for l in diff if l.startswith("-")], [l[1:] for l in diff if l.startswith("+")]


def _parses(nb: dict) -> None:
    """Every code cell parses (IPython magics and shell lines left out).

    :param nb: Notebook.
    """
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))))


# ----------------------------------------------------------------------------- structure


def test_cells_compile_and_only_the_planned_cells_change(editions: tuple) -> None:
    """Every cell parses; install, grad gate and bench are v2.5a's; helpers and collator change only where planned."""
    v25a, colab, lam = editions
    assert len(colab["cells"]) == 10 and len(lam["cells"]) == len(v25a["cells"]) == 11 and lam["metadata"] == v25a["metadata"]
    _parses(colab)
    _parses(lam)
    for i in (1, 5, 6):                                          # install gate, grad gate (108 tower lora_B), bench
        assert _src(lam, i) == _src(v25a, i), i
    out, new = _changed(v25a, lam, 7)                            # helpers: the run's length and its safety stop only
    assert [l.split(" =")[0] for l in out] == [l.split(" =")[0] for l in new] == ["MAX_STEPS", "TIME_BUDGET_H"], (out, new)
    out, new = _changed(v25a, lam, 4)                            # collator: the guardrail picks and their message only
    assert [l for l in new if not l.startswith("#")][:4] == [l for l in dn.NEW_PICKS.splitlines() if not l.startswith("#")]
    assert len(out) == 5 and any(dn.OLD_PICK_MESSAGE in l for l in out) and any(dn.NEW_PICK_MESSAGE in l for l in new)


def test_derivation_is_deterministic_and_leaves_the_source_alone(editions: tuple) -> None:
    """Deriving twice gives the same notebook and never edits the v2.2b source."""
    nb = json.load(open(SRC, encoding="utf-8"))
    assert dn.derive(copy.deepcopy(nb), SHA, ROWS, PAGES) == dn.derive(nb, SHA, ROWS, PAGES) == editions[1]
    assert nb == json.load(open(SRC, encoding="utf-8"))
    assert dn.lambda_edition(SHA, ROWS, PAGES) == editions[2]


def test_committed_placeholder_is_a_fresh_derivation() -> None:
    """The committed Lambda edition equals a fresh derivation at the revision and plan written into it (the placeholder
    carries the reference plan; the push regenerates it pinned with the built one)."""
    committed = json.load(open(COMMITTED, encoding="utf-8"))
    plan = dn.notebook_plan(committed)
    assert committed == dn.lambda_edition(plan["revision"], plan["train_rows"], plan["pages"])
    if plan["revision"] == dn.PLACEHOLDER:
        assert (plan["pages"], plan["train_rows"]) == (dn.REFERENCE_PAGES, dn.auto_rows({"pages": dn.REFERENCE_PAGES}))
        assert dn.notebook_problems(committed) == [f"dataset revision {dn.PLACEHOLDER!r} is not pinned: run {dn.PUSH_SCRIPT}"]
    else:
        assert dn.notebook_problems(committed) == []
    assert (COLAB_DIR / dn.SETUP_SCRIPT).read_text(encoding="utf-8") == hl.setup_script()


# ----------------------------------------------------------------------------- data cell


def _pin_block(nb: dict) -> str:
    """The data cell's revision lines (assignment + guard), ready to exec.

    :param nb: Notebook.
    :return: Source.
    """
    c2 = _src(nb, 2)
    return c2[c2.index("V22_REVISION = "):c2.index("LOCAL = ")]


def test_notebook_refuses_an_unpinned_dataset(editions: tuple) -> None:
    """The data cell raises on the placeholder and on anything but a 40-hex revision."""
    _, _, lam = editions
    block = _pin_block(lam)
    assert "HfApi().dataset_info" not in _src(lam, 2) and dn.PIN_GUARD in block
    exec(compile(block, "<pin guard>", "exec"), {})
    for bad in (dn.PLACEHOLDER, "c" * 39, "C" * 40, "main"):
        with pytest.raises(RuntimeError, match="not a pinned 40-hex sha"):
            exec(compile(block.replace(SHA, bad), "<pin guard>", "exec"), {})
    with pytest.raises(RuntimeError, match="push_novel_v1.sh"):
        exec(compile(_pin_block(dn.lambda_edition(dn.PLACEHOLDER, ROWS, PAGES)), "<pin guard>", "exec"), {})


def test_data_cell_names_and_gates(editions: tuple) -> None:
    """The data cell names this run's repository and folder, carries the plan and drops v2.5a's gates."""
    _, colab, lam = editions
    c2 = _src(lam, 2)
    assert f'V22_REPO = "{dn.DATA_REPO}"' in c2 and f'V22_REVISION = "{SHA}"' in c2
    assert f'LOCAL = os.path.expanduser("~/genizah_work/{dn.LOCAL_LEAF}")' in c2 and f'LOCAL = "/content/{dn.LOCAL_LEAF}"' in _src(colab, 2)
    assert "(2,200 train rows over 1,000 novel pages)" in c2 and "\nNOVEL_TRAIN_ROWS = 2200\n" in c2 and "\nNOVEL_TRAIN_PAGES = 1000\n" in c2
    for gone in ('"Arabic script" in q', "for _s, _lo in", "Arabic mixture incomplete", "val must cover every train source",
                 "the replay rows are not Hebrew script", "genizah_v23a_arabic", "genizah_v25a"):
        assert gone not in c2, gone
    assert f"NEUTRAL_PROMPT = {SCRIPT_NEUTRAL_PROMPT!r}" in c2


def _exec_cell2_tail(c2: str, train: List[Dict[str, Any]], val: List[Dict[str, Any]], mix: Dict[str, Any]) -> Dict[str, Any]:
    """Run the data cell's gates (from the plan gate on) on fake datasets.

    :param c2: Data cell source.
    :param train: Train rows.
    :param val: Val rows.
    :param mix: ``mixture.json``.
    :return: The namespace.
    """
    class _Fake:
        """A dataset with ``column`` and ``len``."""

        def __init__(self, rows: List[Dict[str, Any]]) -> None:
            """Hold the rows.

            :param rows: Rows.
            """
            self.rows = rows

        def __len__(self) -> int:
            """Number of rows.

            :return: Rows.
            """
            return len(self.rows)

        def column(self, name: str) -> List[Any]:
            """One column, as the notebook's dataset gives it.

            :param name: Column.
            :return: Values.
            """
            return [r[name] for r in self.rows]

    start = c2.index("# novel_v1 plan gate")
    ns = {"train_ds": _Fake(train), "eval_ds": _Fake(val), "MIX": mix, "Counter": Counter, "json": json}
    exec(compile(c2[start:], "<data gates>", "exec"), ns)
    return ns


def _fake_rows(stems_counts: Dict[str, int], sources: Dict[str, str], answers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Rows of pages repeated as counted.

    :param stems_counts: stem -> rows.
    :param sources: stem -> source.
    :param answers: stem -> answer.
    :return: Rows.
    """
    return [{"stem": s, "source": sources[s], "task": dn.TASK, "section": dn.SECTION, "question": SCRIPT_NEUTRAL_PROMPT,
             "answer": answers[s]} for s, n in stems_counts.items() for _ in range(n)]


def test_data_cell_gates_run_as_written(capsys: pytest.CaptureFixture) -> None:
    """The plan gate, the pass rule, the neutral prompt, novel page rows, answer hygiene and the script print."""
    stems = {f"a{i}": "pgp_edition" for i in range(6)} | {f"b{i}": "apd_edition" for i in range(2)}
    counts = dn.page_counts({s: [st for st, v in stems.items() if v == s] for s in set(stems.values())}, 16)
    answers = {s: (ARABIC if stems[s] == "apd_edition" else HEBREW) for s in stems}
    train = _fake_rows(counts, stems, answers)
    val = [dict(train[0], stem="v0")]
    mix = {"config": {"train_rows": 16, "train_pages": 8, "val_rows": 1,
                      "pages_by_source": dict(Counter(stems.values())),
                      "rows_by_source": {s: sum(n for st, n in counts.items() if stems[st] == s) for s in set(stems.values())}}}
    c2 = _src(dn.lambda_edition(SHA, 16, 8), 2)
    ns = _exec_cell2_tail(c2, train, val, mix)
    out = capsys.readouterr().out
    assert "hygiene OK" in out and "Arabic-script rows" in out and "no floors" in out and ns["_scripts"]["arabic"] > 0

    def broken(rows: Any = None, vals: Any = None, mx: Any = None) -> Dict[str, Any]:
        """Run the gates on a variation.

        :param rows: Train rows (default: the good ones).
        :param vals: Val rows (default: the good ones).
        :param mx: ``mixture.json`` (default: the good one).
        :return: The namespace.
        """
        return _exec_cell2_tail(c2, rows if rows is not None else train, vals if vals is not None else val, mx or mix)

    for rows, message in ((train[:-1], "train rows"), ([dict(train[0], question=FRAGMENT_TRANSCRIBE_PROMPT)] + train[1:], "script-neutral"),
                          ([dict(train[0], answer=train[0]["answer"] + " مِنْ")] + train[1:], "harakat"),
                          ([dict(train[0], answer=train[0]["answer"] + " `x")] + train[1:], "backtick"),
                          ([dict(train[0], section="pgp_arabic_page")] + train[1:], "novel page"),
                          ([dict(train[0], answer="abc 123")] + train[1:], "mojibake|without Hebrew or Arabic letters")):
        with pytest.raises(AssertionError, match=message):
            broken(rows=rows)
    third = next(s for s, n in counts.items() if n == 2)
    with pytest.raises(AssertionError, match="2-3 rows|plan"):
        broken(rows=train + _fake_rows({third: 2}, stems, answers))
    with pytest.raises(AssertionError, match="a val page is also a train page"):
        broken(vals=[dict(train[0])])
    weak = next(i for i, r in enumerate(train) if r["source"] == "apd_edition")
    with pytest.raises(AssertionError, match="not mainly Arabic script"):
        broken(rows=train[:weak] + [dict(train[weak], answer="كتابي اليك اطال الله بقاك وادام [...]")] + train[weak + 1:])


# ----------------------------------------------------------------------------- the neutral prompt


def test_the_rows_step_sets_the_neutral_prompt_and_refuses_a_script_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every question becomes the neutral prompt; a prompt naming a script never passes."""
    assert not dn.script_clause_hits([SCRIPT_NEUTRAL_PROMPT])
    assert dn.script_clause_hits([FRAGMENT_TRANSCRIBE_PROMPT, ARABIC_FRAGMENT_PROMPT, SCRIPT_NEUTRAL_PROMPT]) == sorted(
        [FRAGMENT_TRANSCRIBE_PROMPT, ARABIC_FRAGMENT_PROMPT])
    table = pa.table({"stem": ["x", "y", "z"], "question": [FRAGMENT_TRANSCRIBE_PROMPT, ARABIC_FRAGMENT_PROMPT, SCRIPT_NEUTRAL_PROMPT]})
    out, rewritten = dn.neutral_questions(table)
    assert out["question"].to_pylist() == [SCRIPT_NEUTRAL_PROMPT] * 3 and rewritten == 2
    monkeypatch.setattr(dn, "SCRIPT_NEUTRAL_PROMPT", ARABIC_FRAGMENT_PROMPT)       # a prompt that names a script never passes
    with pytest.raises(ValueError, match="names a script"):
        dn.neutral_questions(table)


# ----------------------------------------------------------------------------- the pass plan


@pytest.mark.parametrize("pages", [{"a": 1000}, {"a": 600, "b": 250, "c": 45, "d": 40, "e": 50, "f": 15}, {"a": 3, "b": 2, "c": 9},
                                   {"x": 37, "y": 2}])
def test_pass_plan_meets_the_rule_at_every_whole_step(pages: Dict[str, int]) -> None:
    """Every whole-step row count of the range keeps every page at 2 or 3 rows and every source at most 2.5 passes."""
    stems = {s: [f"{s}{i:04d}" for i in range(n)] for s, n in pages.items()}
    lo8, hi8 = dn.step_rows(pages)
    low, high = dn.feasible_rows(pages)
    assert low == 2 * sum(pages.values()) and high == low + sum(int(math.floor(0.5 * n)) for n in pages.values())
    for rows in range(lo8, hi8 + 1, 8):
        counts = dn.page_counts(stems, rows)
        assert sum(counts.values()) == rows and set(counts.values()) <= {2, 3}
        for s, group in stems.items():
            assert sum(counts[x] for x in group) <= 2.5 * len(group)
        assert counts == dn.page_counts({s: list(reversed(v)) for s, v in reversed(list(stems.items()))}, rows)   # order-free
    auto = dn.auto_rows(pages)
    assert auto % 8 == 0 and lo8 <= auto <= hi8 and abs(auto - 2.2 * sum(pages.values())) <= 4 + 8 * (auto in (lo8, hi8))
    for bad in (lo8 - 8, hi8 + 8, lo8 + 4, "2.2", "-8"):
        with pytest.raises(ValueError):
            dn.resolve_rows(bad, pages)
    assert dn.resolve_rows("AUTO", pages) == auto and dn.resolve_rows(str(lo8), pages) == lo8


def test_the_reference_plan_is_the_design() -> None:
    """1,000 pages: 2,200 rows = 275 steps, 9 warm-up steps, milestones 100 / 150 / 250 / 275, evals 150 / 250 / 275, 7 h."""
    assert dn.auto_rows({"a": 1000}) == 2200 and dn.step_rows({"a": 1000}) == (2000, 2496)
    assert dn.max_steps_for(2200) == 275 and dn.warmup_steps(2200) == 9 == warmup_steps_for(275, 0.03)
    assert dn.milestone_steps(275) == (100, 150, 250, 275) and dn.eval_steps(275) == (150, 250, 275)
    assert dn.lambda_budget_h(2200) == 7.0 == math.ceil(1.3 * 275 * 63.6 / 3600)
    assert cosine_warmup_lr(9, 5e-5, 9, 275) == pytest.approx(5e-5)
    with pytest.raises(ValueError):
        dn.max_steps_for(2204)
    with pytest.raises(ValueError, match="no multiple of 8"):                  # 26 to 31 rows: no whole step
        dn.auto_rows({"a": 3, "b": 1, "c": 9})


def test_pass_plan_draws_the_extras_uniformly_over_all_pages() -> None:
    """The extras are the first pages of one seeded permutation of every page of every source (novel_v1/extras); a page
    of a source at its cap is skipped; the plan repeats under its seed."""
    pages = {"a": 10, "b": 4}
    plan = dn.pass_plan(pages, 2 * 14 + 4)
    index = [(s, i) for s, n in pages.items() for i in range(n)]
    caps, given, want = {"a": 5, "b": 2}, {"a": 0, "b": 0}, {"a": [2] * 10, "b": [2] * 4}
    for k in bvm.component_rng(dn.SEED, "novel_v1/extras").permutation(len(index)):
        s, i = index[k]
        if sum(given.values()) < 4 and given[s] < caps[s]:
            want[s][i] += 1
            given[s] += 1
    assert {s: v.tolist() for s, v in plan.items()} == want
    assert dn.pass_plan(pages, 32)["a"].tolist() == plan["a"].tolist()                      # seeded: the same plan again
    big = {"x": 600, "y": 400}
    assert dn.pass_plan(big, 2200, seed=1)["x"].tolist() != dn.pass_plan(big, 2200, seed=2)["x"].tolist()
    assert dn.pass_plan({"a": 3, "b": 1}, 9)["b"].tolist() == [2]                          # b's cap is 0 extras: skipped
    with pytest.raises(ValueError, match="cannot be met"):
        dn.pass_plan(pages, 2 * 14 + 8)                                                     # the caps allow 7 extras


#: a plausible 1,000-page composition: source -> script -> train pages
REALISTIC_MIX = {"pgp_edition": {"hebrew": 560, "arabic": 30, "mixed": 10}, "legacy_store_diplomatic": {"hebrew": 250},
                 "ktiv_nli": {"hebrew": 45}, "scanned_edition": {"hebrew": 25, "arabic": 10, "mixed": 5},
                 "apd_edition": {"arabic": 50}, "book_edition": {"arabic": 15}}


def test_uniform_extras_keep_the_natural_script_shares_on_a_realistic_mix() -> None:
    """On the realistic mix every whole-step row count keeps every script within the gates' 0.5 points of its page share
    and every source within 2 to 2.5 passes (round-robin by source gave Arabic +1.05 points at the auto rows)."""
    by_source: Dict[str, List[str]] = {}
    scripts: Dict[str, str] = {}
    for source, by_script in REALISTIC_MIX.items():
        for script, n in by_script.items():
            for i in range(n):
                stem = f"{source}_{script}_{i:04d}"
                by_source.setdefault(source, []).append(stem)
                scripts[stem] = script
    pages = {s: len(v) for s, v in by_source.items()}
    assert sum(pages.values()) == 1000 and dn.auto_rows(pages) == 2200
    lo8, hi8 = dn.step_rows(pages)
    for rows in range(lo8, hi8 + 1, 8):
        counts = dn.page_counts(by_source, rows)
        assert not dn.script_drift_problems(dn.script_table(scripts, counts)), rows
        assert all(2 <= sum(counts[x] for x in stems) / len(stems) <= 2.5 for stems in by_source.values()), rows
    table = dn.script_table(scripts, dn.page_counts(by_source, 2200))
    assert table["arabic"]["page_share"] == 0.105 and abs(table["arabic"]["drift"]) < 0.001


def test_script_drift_is_measured_and_bounded() -> None:
    """script_table measures a script's row share against its page share; the tolerance and the in-tolerance rows see it."""
    stems = {"big": [f"h{i}" for i in range(90)], "small": [f"r{i}" for i in range(10)]}
    scripts = {s: "hebrew" for s in stems["big"]} | {s: "arabic" for s in stems["small"]}
    skewed = {s: 2 for s in scripts} | {f"r{i}": 3 for i in range(5)} | {f"h{i}": 3 for i in range(11)}   # 5 of 16 extras Arabic
    table = dn.script_table(scripts, skewed)
    assert table["arabic"]["page_share"] == 0.1 and table["arabic"]["drift"] == pytest.approx(25 / 216 - 0.1, abs=1e-6)
    assert dn.script_drift_problems(table) and not dn.script_drift_problems(table, 0.02)
    flat = dn.script_table(scripts, dn.page_counts(stems, 200))              # every page twice: no drift
    assert all(v["drift"] == 0 for v in flat.values()) and not dn.script_drift_problems(flat)
    assert 200 in dn.rows_within_tolerance(stems, scripts)


# ----------------------------------------------------------------------------- warm start, schedule, names


def test_warm_start_is_v25a_step_1200(editions: tuple) -> None:
    """The model cell pins v2.5a step 1200 by commit and keeps LoRA r16 with the merger frozen."""
    v25a, _, lam = editions
    c3 = _src(lam, 3)
    assert dn.warm_start_pinned() and dn.WARM_STEP == 1200 and dn.WARM_CKPT_REPO == "isaacmg/qwen3-vl-8b-hebrew-v25a-ckpt"
    assert f'WARM_REVISION = "{dn.WARM_REVISION}"' in c3 and "\nWARM_STEP = 1200\n" in c3 and f'WARM_CKPT_REPO = "{dn.WARM_CKPT_REPO}"' in c3
    assert dn.NEW_WARM_DOWNLOAD in c3 and "v24a-ckpt" not in c3 and 'MERGER_MODE = "frozen"' in c3
    assert 'r=16, lora_alpha=16, lora_dropout=0.0, bias="none", random_state=3407,' in c3
    assert 'assert len(WARM_REVISION) == 40, "invalid revision SHA"' in c3                 # a placeholder warm start refuses
    out, new = _changed(v25a, lam, 3)
    assert len(out) == len(new) == 6                      # title, warm-start comment (2 lines), repo, revision, the check's comment


def test_warm_start_step_check_runs_as_written(tmp_path: Path) -> None:
    """The trainer_state step check passes step 1200 and refuses another step."""
    (tmp_path / "last-checkpoint").mkdir()
    check = dn.NEW_WARM_DOWNLOAD.split("\n", 2)[2]
    ns = {"json": json, "local": str(tmp_path), "WARM_STEP": 1200, "WARM_CKPT_REPO": "r", "WARM_REVISION": "0a0e7e76"}
    (tmp_path / "last-checkpoint" / "trainer_state.json").write_text(json.dumps({"global_step": 1200}))
    exec(compile(check, "<warm step>", "exec"), dict(ns))
    (tmp_path / "last-checkpoint" / "trainer_state.json").write_text(json.dumps({"global_step": 900}))
    with pytest.raises(AssertionError, match="is step 900, expected 1200"):
        exec(compile(check, "<warm step>", "exec"), dict(ns))


def test_schedule_and_trainer_settings(editions: tuple) -> None:
    """Steps, budget, LR, warm-up, seed, saves, evals, privacy and the no-dispatch rule are as planned."""
    _, _, lam = editions
    c7, c8 = _src(lam, 7), _src(lam, 8)
    assert "\nMAX_STEPS = 275\n" in c7 and "\nORDER_SEED = 3407\n" in c7
    assert "\nTIME_BUDGET_H = 7.0   # Lambda, one session: 275 steps x ~63.6 s" in c7
    assert len(re.findall(r"^\s*learning_rate=5e-5,", c8, re.M)) == 1 and "learning_rate=1e-4," not in c8
    for needle in ('warmup_ratio=0.03, lr_scheduler_type="cosine", weight_decay=0.01,   # 9 warm-up steps of 275',
                   'optim="adamw_8bit", seed=3407, data_seed=3407, output_dir=OUT_DIR,', "per_device_train_batch_size=1, gradient_accumulation_steps=8,",
                   "save_steps=50, save_total_limit=2,", 'eval_strategy="steps", eval_steps=100,', 'hub_strategy="checkpoint", hub_private_repo=True,',
                   "hub_always_push=True,", 'accelerator_config={"dispatch_batches": False},', "# ---- DATALOADER GATE",
                   "_order = lock_session_plan(CKPT_REPO, _plan, _order, has_checkpoint=resume_dir is not None)",
                   "TimeBudgetTrainerCallback(budget_s=TIME_BUDGET_H * 3600)", "private by default"):
        assert needle in c8, needle
    assert "Muharaf" not in c8


def test_names_repositories_and_privacy(editions: tuple) -> None:
    """Checkpoint, merged and W&B names are this run's; v2.5a appears only as the warm-start repository."""
    _, _, lam = editions
    c8, c9, c10 = _src(lam, 8), _src(lam, 9), _src(lam, 10)
    assert dn.CKPT_REPO == "isaacmg/qwen3-vl-8b-hebrew-novel-v1-ckpt" and dn.MERGED_REPO == "isaacmg/qwen3-vl-8b-hebrew-novel-v1"
    assert dn.DATA_REPO == "isaacmg/genizah_novel_v1" and dn.RUN_NAME == "genizah_novel_v1" and dn.NOTEBOOK == "genizah_novel_v1_lambda.ipynb"
    for needle in (f'CKPT_REPO = "{dn.CKPT_REPO}"', 'OUT_DIR = "outputs_novel-v1"', 'RESUME_ROOT = os.path.expanduser("~/genizah_work/novel-v1_resume")',
                   'run_name="genizah_novel_v1"', 'wandb.init(project="qwen-hebrew-finetune", name="genizah_novel_v1", ', dn.TRAIN_TITLE):
        assert needle in c8, needle
    assert f'MERGED_REPO = "{dn.MERGED_REPO}"' in c9 and "private=True" in c9 and c9.count('"novel-v1-merged"') == 2
    assert 'wandb.alert(title="genizah_novel_v1: notebook finished"' in c10
    code = "\n".join(hl.code_cells(lam))
    for stale in ("v25a-ckpt\"  #", "genizah_v25a", "outputs_v25a", "v25a_resume", "v25a-merged", "push_v25a", "arabic_genizah",
                  "genizah_v23a_arabic", "v23a-ckpt", "-merged\"\n"):
        assert stale not in code, stale
    assert code.count("v25a-ckpt") == 1                                       # only as the warm-start repository


def test_intro_describes_this_run(editions: tuple) -> None:
    """The intro states the warm start, the data, the plan, the gates and the session."""
    _, colab, lam = editions
    intro = _src(colab, 0)
    for needle in (f"# Genizah {dn.ROUND}", dn.DATA_REPO, dn.CKPT_REPO, dn.MERGED_REPO, "**2,200 / 275**", "1,000 train pages",
                   "v2.5a step 1200", dn.WARM_REVISION[:8], "5e-5", "(9 steps)", "100, 150, 250, 275", "150, 250, 275",
                   "safety stop 7 h", "refuses to run", "run every cell again", "replay quota 0", "SCRIPT_NEUTRAL_PROMPT",
                   "2 x max(0.60, the step-0 eval loss), at least 1.20", "step-0 eval", "uniformly over all pages"):
        assert needle in intro, needle
    lam_intro = _src(lam, 0)
    assert "**Lambda Cloud edition**" in lam_intro.splitlines()[0] and dn.NOTEBOOK in lam_intro and lam_intro.endswith(intro)


def test_conversation_and_collator_are_v25a_s(editions: tuple) -> None:
    """The EOS supervision finding of tests/test_v25a_eos_supervision.py carries over: same to_conversation, same collator."""
    v25a, _, lam = editions
    a, b = _src(v25a, 4), _src(lam, 4)
    head = a.index("# guardrail")
    assert b.startswith(a[:head]) and "train_on_responses_only=True" in a[:head] and 'response_part="<|im_start|>assistant\\n"' in a[:head]


# ----------------------------------------------------------------------------- step-0 eval, milestones, loss alarm


class _FakeCallback:
    """Stands in for ``transformers.TrainerCallback``."""

    def __init__(self) -> None:
        """Nothing to set up."""


def _block(nb: dict, start: str, end: str) -> str:
    """A training-cell block from its comment to (and including) a closing line.

    :param nb: Notebook.
    :param start: Its first line's start.
    :param end: Its last line.
    :return: Source.
    """
    c8 = _src(nb, 8)
    i = c8.index(start)
    return c8[i:c8.index(end, i) + len(end)]


def test_step0_eval_runs_on_a_fresh_start_and_is_read_back_on_resume(editions: tuple, tmp_path: Path) -> None:
    """A fresh start evaluates the warm start and stores the loss; a resume reads it back without evaluating."""
    _, _, lam = editions
    c8 = _src(lam, 8)
    assert c8.index(dn.SESSION_LINE) < c8.index(dn.STEP0_START) < c8.index("_gate = ResumeGateTrainerCallback()")
    block = _block(lam, dn.STEP0_START, 'on {len(eval_ds)} val rows")\n')
    calls = []

    class _Api:
        """HfApi stand-in."""

        def upload_file(self, **kw: Any) -> None:
            """Record an upload.

            :param kw: The call's keyword arguments.
            """
            calls.append(("upload", kw["path_in_repo"], json.loads(kw["path_or_fileobj"])))

    import huggingface_hub
    trainer = SimpleNamespace(evaluate=lambda: calls.append(("evaluate",)) or {"eval_loss": 1.0625})
    model_calls = []
    ns = {"resume_dir": None, "trainer": trainer, "json": json, "WARM_CKPT_REPO": "w", "WARM_REVISION": "r", "CKPT_REPO": "c",
          "eval_ds": [1, 2, 3], "model": "m", "FastVisionModel": SimpleNamespace(for_training=model_calls.append)}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(huggingface_hub, "HfApi", _Api)
        exec(compile(block, "<step0>", "exec"), ns)
    assert ns["STEP0_EVAL_LOSS"] == 1.0625 and calls[0] == ("evaluate",) and model_calls == ["m"]
    assert calls[1][:2] == ("upload", dn.STEP0_FILE) and calls[1][2]["eval_loss"] == 1.0625
    stored = tmp_path / "step0.json"
    stored.write_text(json.dumps({"eval_loss": 0.9}))
    resumed = dict(ns, resume_dir="ckpt", trainer=SimpleNamespace(evaluate=lambda: pytest.fail("a resume must not re-evaluate")))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: str(stored))
        exec(compile(block, "<step0>", "exec"), resumed)
    assert resumed["STEP0_EVAL_LOSS"] == 0.9


def _callbacks(nb: dict, ensure: Any = None, alerts: Any = None, steps: int = 275, step0: float = 1.0) -> Dict[str, Any]:
    """Execute the training cell's callbacks block with fakes.

    :param nb: Notebook.
    :param ensure: ensure_hub_checkpoint stand-in.
    :param alerts: List collecting W&B alerts.
    :param steps: MAX_STEPS.
    :param step0: The step-0 eval loss (STEP0_EVAL_LOSS).
    :return: The namespace.
    """
    trainer = SimpleNamespace(args=SimpleNamespace(save_steps=50), push_in_progress=None, callbacks=[])
    trainer.add_callback = trainer.callbacks.append
    alerts = alerts if alerts is not None else []
    wandb = SimpleNamespace(run=object(), alert=lambda title, text: alerts.append(title))
    ns = {"TrainerCallback": _FakeCallback, "trainer": trainer, "CKPT_REPO": "repo", "OUT_DIR": "out", "MAX_STEPS": steps,
          "ensure_hub_checkpoint": ensure or (lambda repo, out, step: None), "math": math, "wandb": wandb, "STEP0_EVAL_LOSS": step0}
    exec(compile(_block(nb, dn.CALLBACKS_START, "trainer.add_callback(WarmStartWatchCallback(STEP0_EVAL_LOSS))\n"), "<callbacks>", "exec"), ns)
    return ns


def test_milestones_are_the_quarter_points_pushed_and_confirmed(editions: tuple) -> None:
    """Quarter points rounded up to 50 (at least 100, the last the final step) are confirmed on the Hub, each once."""
    _, colab, lam = editions
    assert dn.milestone_steps(1200) == (300, 600, 900, 1200)                  # v2.5a's own
    assert dn.milestone_steps(137) == (100, 137) and dn.milestone_steps(60) == (60,) and dn.milestone_steps(400) == (100, 200, 300, 400)
    for steps in range(1, 2000, 7):
        ms = dn.milestone_steps(steps)
        assert ms[-1] == steps and all(m % 50 == 0 for m in ms[:-1]) and all(m >= min(100, steps) for m in ms)
    for nb in (colab, lam):
        c8 = _src(nb, 8)
        assert "\nMILESTONES = (100, 150, 250, 275)\n" in c8 and dn.SAVE_LINE in c8
        assert c8.index(dn.BUDGET_CALLBACK) < c8.index(dn.CALLBACKS_START) < c8.index("_final_step = train_with_exact_resume(")
    calls: List[tuple] = []
    ns = _callbacks(lam, ensure=lambda repo, out, step: calls.append(("confirm", step)))
    cb = ns["_milestones"]
    assert len(ns["trainer"].callbacks) == 3 and ns["MILESTONES"] == (100, 150, 250, 275)
    for step in (50, 200, 300):
        cb.on_save(None, SimpleNamespace(global_step=step), "ctl")
    assert calls == []
    ns["trainer"].push_in_progress = SimpleNamespace(wait_until_done=lambda: calls.append(("wait",)))
    for step in (100, 100, 150, 275):
        cb.on_save(None, SimpleNamespace(global_step=step), "ctl")
    cb.on_train_begin(None, SimpleNamespace(global_step=250), "ctl")
    assert [c[1] for c in calls if c[0] == "confirm"] == [100, 150, 275, 250] and calls[0] == ("wait",)
    with pytest.raises(AssertionError, match="must be save steps"):
        _callbacks(lam, steps=300)                                             # MILESTONES no longer end at MAX_STEPS


@pytest.mark.parametrize("step0, threshold", [(0.5, 1.2), (0.6, 1.2), (0.9, 1.8), (1.07, 2.14)])
def test_the_first_train_loss_above_the_alarm_threshold_stops_the_run(editions: tuple, capsys: pytest.CaptureFixture,
                                                                      step0: float, threshold: float) -> None:
    """The threshold is 2 x max(0.60, the step-0 eval loss), printed as one line before training; a logged train loss
    above it, or a non-finite one, raises a never-retried LossAlarmError."""
    _, _, lam = editions
    alerts: List[str] = []
    ns = _callbacks(lam, alerts=alerts, step0=step0)
    assert (ns["LOSS_ALARM_REFERENCE"], ns["LOSS_ALARM_FACTOR"]) == (dn.LOSS_ALARM_REFERENCE, dn.LOSS_ALARM_FACTOR) == (0.6, 2.0)
    assert ns["LOSS_ALARM_THRESHOLD"] == pytest.approx(threshold) and dn.loss_alarm_threshold(step0) == pytest.approx(threshold)
    assert f"loss alarm threshold {threshold:.2f} = 2.0 x max(0.60, step-0 eval {step0:.3f})" in capsys.readouterr().out
    alarm = ns["trainer"].callbacks[1]
    assert issubclass(ns["LossAlarmError"], AssertionError)                    # train_with_exact_resume never retries it
    for logs in ({"loss": 0.62}, {"loss": threshold - 0.01}, {"eval_loss": 9.0}, {"train_loss": 9.0}, None):
        assert alarm.on_log(None, SimpleNamespace(global_step=10), "ctl", logs=logs) == "ctl"
    for bad in (threshold + 0.01, 9.0, float("nan"), float("inf")):
        with pytest.raises(ns["LossAlarmError"], match="the run stops"):
            alarm.on_log(None, SimpleNamespace(global_step=10), "ctl", logs={"loss": bad})
    assert alerts and all("loss alarm" in a for a in alerts)


def test_warm_start_watch_alerts_after_two_evals_above_step0(editions: tuple) -> None:
    """Two consecutive evals above the step-0 loss raise one W&B alert; the run goes on."""
    _, _, lam = editions
    alerts: List[str] = []
    watch = _callbacks(lam, alerts=alerts)["trainer"].callbacks[2]
    for step, loss in ((0, 5.0), (100, 1.1), (200, 0.9), (300, 1.2)):
        watch.on_evaluate(None, SimpleNamespace(global_step=step), "ctl", metrics={"eval_loss": loss})
    assert alerts == []
    watch.on_evaluate(None, SimpleNamespace(global_step=400), "ctl", metrics={"eval_loss": 1.3})
    watch.on_evaluate(None, SimpleNamespace(global_step=500), "ctl", metrics={"eval_loss": 1.4})
    assert len(alerts) == 1 and "warm-start stop rule" in alerts[0]


# ----------------------------------------------------------------------------- the rows step on a synthetic export


def _synthetic_export(root: Path, rnl_stem: str = "") -> Path:
    """A tiny novel dataset (KTIV FEATURES via save_dataset) with its manifest, stats and images-once export.

    :param root: Folder.
    :param rnl_stem: A stem whose document is a Russian National Library one.
    :return: The export.
    """
    plan = [("pgp", HEBREW, 8, "train"), ("pgp", ARABIC, 2, "train"), (LEGACY_SOURCE, HEBREW, 4, "train"), ("apd_v3", ARABIC, 3, "train"),
            ("literature", MIXED, 2, "train"), ("pgp", HEBREW, 2, "val"), ("apd_v3", ARABIC, 1, "val")]
    rows: Dict[str, List[Dict[str, Any]]] = {"train_page": [], "val": []}
    manifest = []
    (root / "src").mkdir(parents=True)
    n = 0
    for source, text, count, split in plan:
        for _ in range(count):
            n += 1
            stem = f"novel_{source}_{n:03d}_r_page"
            path = root / "src" / f"{n}.jpg"
            rng = np.random.default_rng(n)
            Image.fromarray(rng.integers(0, 255, (100, 82, 3), dtype=np.uint8)).resize((820, 1000)).save(path, "JPEG", quality=60)
            answer = f"{text}\n{n}"
            rows[novel_build.SPLIT_NAMES[split]].append({
                "image": str(path), "question": SCRIPT_NEUTRAL_PROMPT, "answer": answer, "task": dn.TASK, "section": dn.SECTION,
                "stem": stem, "label_source": novel_build.LABEL_SOURCES[source], "target_chars": len(answer), "target_tokens": 0,
                "image_width": 820, "image_height": 1000})
            manifest.append({"stem": stem, "row_id": f"{source}:{n}", "source": source, "document_group": f"g{n}", "split": split,
                             "canonical_ids": ["StPetersburg_NLR_Test_1" if stem == rnl_stem else f"Test_Collection_{n}"],
                             "pgpids": [], "script": measure_script(answer), "verdict": "novel", "rnl": stem == rnl_stem,
                             "shelfmark": f"Test Coll. {n}", "image_path": str(path)})
    per: Dict[str, Dict[str, Dict[str, int]]] = {}
    for m in manifest:
        cell = per.setdefault(funnel.source_label(m["source"]), {}).setdefault(m["script"], {})
        cell[novel_build.SPLIT_NAMES[m["split"]]] = cell.get(novel_build.SPLIT_NAMES[m["split"]], 0) + 1
    stats = {"name": "genizah_novel_v1", "built": "2026-11-03T00:00:00Z", "rows": {k: len(v) for k, v in rows.items()},
             "rows_by_source_script_split": per, "private": True, "private_reasons": ["test rows from the legacy store"]}
    dataset = root / "genizah_novel_v1"
    save_dataset(rows, dataset, {"README.md": "# test\n", "stats.json": json.dumps(stats), "PRIVATE.md": "private\n"})
    (dataset / "manifest.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8")
    export_images_once(dataset, root / "genizah_novel_v1_images_once")
    return root / "genizah_novel_v1_images_once"


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """(export, rows folder) of a tiny synthetic novel dataset, one Russian National Library row left out."""
    root = tmp_path_factory.mktemp("novel")
    export = _synthetic_export(root, rnl_stem="novel_pgp_001_r_page")
    out = root / "train_out"
    dn.build_rows(export, out, workers=2)
    return export, out


def test_rows_step_writes_the_plan(synthetic: tuple) -> None:
    """The rows step writes the pass plan, the neutral prompt, the source column, the images and a clean card."""
    export, out = synthetic
    mix = json.loads((out / "mixture.json").read_text(encoding="utf-8"))
    config = mix["config"]
    train = pq.read_table(out / "rows" / "train.parquet").to_pylist()
    val = pq.read_table(out / "rows" / "val.parquet").to_pylist()
    pages = {"apd_edition": 3, "legacy_store_diplomatic": 4, "pgp_edition": 9, "scanned_edition": 2}
    assert config["pages_by_source"] == pages and config["train_pages"] == 18 and config["rnl_dropped"] == {"train": 1, "val": 0}
    assert config["train_rows"] == dn.auto_rows(pages) == len(train) and len(train) % 8 == 0 and config["max_steps"] == len(train) // 8
    counts = Counter(r["stem"] for r in train)
    by_source = {s: sorted({r["stem"] for r in train if r["source"] == s}) for s in pages}
    assert dict(counts) == dn.page_counts(by_source, len(train)) and "novel_pgp_001_r_page" not in counts
    assert all(r["question"] == SCRIPT_NEUTRAL_PROMPT and r["source"] == r["label_source"] for r in train + val)
    assert len(val) == config["val_rows"] == 3 and not {r["stem"] for r in val} & set(counts)
    meta = json.loads(pq.read_table(out / "rows" / "train.parquet").schema.metadata[b"images_once"])
    assert meta["columns"][-1] == "source" and set(meta["columns"]) - {"image"} <= set(train[0])
    assert {p.name[:-4] for p in (out / "images").glob("*.jpg")} == {r["image_sha1"] for r in train + val}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == export.name and manifest["splits"]["train"]["rows"] == len(train)
    public = "".join((out / f).read_text(encoding="utf-8") for f in ("README.md", "mixture.json", "manifest.json")).lower()
    assert not any(word in public for word in dn.FORBIDDEN_CARD) and str(export.parent).lower() not in public
    assert "**PRIVATE." in (out / "README.md").read_text(encoding="utf-8")
    assert config["milestones"] == list(dn.milestone_steps(config["max_steps"])) and config["warm_start"].startswith(dn.WARM_CKPT_REPO)
    with pytest.raises(FileExistsError):
        dn.build_rows(export, out)


def test_rows_step_refuses_an_export_its_manifest_does_not_describe(synthetic: tuple, tmp_path: Path) -> None:
    """A manifest out of step with the export, or rows the pages cannot carry, stop the rows step."""
    export, _ = synthetic
    manifest = json.loads(json.dumps([json.loads(l) for l in dn.dataset_manifest_path(export).read_text(encoding="utf-8").splitlines()]))
    bad = tmp_path / "manifest.jsonl"
    bad.write_text("".join(json.dumps(dict(m, split="val") if i == 3 else m) + "\n" for i, m in enumerate(manifest)), encoding="utf-8")
    with pytest.raises(ValueError, match="another split"):
        dn.prepare_rows(export, manifest_path=bad)
    bad.write_text("".join(json.dumps(m) + "\n" for m in manifest[1:]), encoding="utf-8")
    with pytest.raises(ValueError, match="no manifest record"):
        dn.prepare_rows(export, manifest_path=bad)
    kept = dn.prepare_rows(export, keep_rnl=True)
    assert kept.record["train_pages"] == 19 and kept.record["keep_rnl"] and kept.record["rnl_dropped"] == {"train": 0, "val": 0}
    with pytest.raises(ValueError, match="multiples of 8"):
        dn.prepare_rows(export, request="44")


def _load_gates() -> Any:
    """The round's gates runner as a module.

    :return: The module.
    """
    spec = importlib.util.spec_from_file_location("run_notebook_gates_novel_v1", GATES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not GATES.is_file(), reason="the round's gates runner is git-ignored and not on this machine")
def test_gates_runner_passes_the_synthetic_rows_and_catches_breakage(synthetic: tuple, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The round's gates pass the synthetic rows and fail on drift, a held-out mark and a script prompt."""
    pytest.importorskip("src.finetuning.qwen_hebrew.colab.derive_v25a")
    gates = _load_gates()
    export, built = synthetic
    out = tmp_path / "rows"
    shutil.copytree(built, out)
    empty = lambda cairo_dir=None: gates.d5.HeldOut()
    marker = gates.main(out, export=export, tolerance=0.1, workers=2, held_out=empty)     # a tiny set: a wider tolerance
    text = capsys.readouterr().out
    assert "ALL GATES PASSED" in text and "guardrail rows" in text and "hygiene OK" in text
    assert marker["train_sha256"] == dn.file_sha256(out / "rows" / "train.parquet") and (out / gates.MARKER).is_file()
    with pytest.raises(SystemExit):                                            # no tolerance: the sampled extras leave a gap
        gates.main(out, export=export, tolerance=0.0, workers=2, held_out=empty)
    assert "gate 4 scripts FAILED" in capsys.readouterr().out
    held = lambda cairo_dir=None: gates.d5.HeldOut(keys=frozenset({gates.d5.shelfmark_key("Test Coll. 2")}))
    with pytest.raises(SystemExit):
        gates.main(out, export=export, tolerance=0.1, workers=2, held_out=held)
    assert "gate 8 ledger and hold-outs FAILED" in capsys.readouterr().out
    rows = pq.read_table(out / "rows" / "train.parquet")
    i = rows.schema.get_field_index("question")
    broken = rows.set_column(i, rows.schema.field(i), pa.array([ARABIC_FRAGMENT_PROMPT] + rows["question"].to_pylist()[1:], rows.schema.field(i).type))
    bvm.write_rows(broken, json.loads(rows.schema.metadata[b"images_once"]), out / "rows" / "train.parquet")
    with pytest.raises(SystemExit):
        gates.main(out, export=export, tolerance=0.1, workers=2, held_out=empty)
    assert "gate 2 prompt FAILED" in capsys.readouterr().out


# ----------------------------------------------------------------------------- CLI and public-surface hygiene


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The notebook subcommand writes only the Lambda edition; show prints a pinned notebook's fields."""
    dn.main(["notebook", "--revision", SHA, "--pages", "1000", "--out-dir", str(tmp_path)])
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([dn.SETUP_SCRIPT, dn.NOTEBOOK])      # the Lambda edition only
    assert json.loads((tmp_path / dn.NOTEBOOK).read_text(encoding="utf-8")) == dn.lambda_edition(SHA, ROWS, PAGES)
    plan = tmp_path / "mixture.json"
    plan.write_text(json.dumps({"config": {"train_rows": 2016, "train_pages": 1000}}))
    dn.main(["notebook", "--revision", SHA, "--plan", str(plan), "--out-dir", str(tmp_path)])
    assert dn.notebook_plan(json.loads((tmp_path / dn.NOTEBOOK).read_text(encoding="utf-8")))["train_rows"] == 2016
    for argv in (["notebook", "--revision", "not-a-sha", "--pages", "1000"], ["notebook", "--revision", SHA],
                 ["notebook", "--revision", SHA, "--pages", "1000", "--plan", str(plan)]):
        with pytest.raises(SystemExit):
            dn.main(argv + ["--out-dir", str(tmp_path)])
    capsys.readouterr()
    dn.main(["show", "--notebook", str(tmp_path / dn.NOTEBOOK), "--field", "eval_steps", "--require-pinned"])
    assert capsys.readouterr().out.split() == [str(s) for s in dn.eval_steps(252)]
    dn.main(["notebook", "--revision", dn.PLACEHOLDER, "--pages", "1000", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="not pinned"):
        dn.main(["show", "--notebook", str(tmp_path / dn.NOTEBOOK), "--require-pinned"])


def test_no_internal_names_in_the_tracked_files(editions: tuple) -> None:
    """The tracked files carry no restricted attribution, storage path, token or Colab line."""
    for path in (Path(dn.__file__), COMMITTED, Path(__file__)):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(word in text for word in TRACKED_FORBIDDEN), path
    text = json.dumps(editions[2], ensure_ascii=False)
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text) and "google.colab" not in text and "/content" not in text
    assert not any(word in dn.PRIVATE_BANNER.lower() for word in dn.FORBIDDEN_CARD)


# ----------------------------------------------------------------------------- the round's files (git-ignored)

needs_readme = pytest.mark.skipif(not README.is_file(), reason="the round's README is git-ignored and not on this machine")


def _bash_lines() -> List[str]:
    """The README's bash lines, line continuations joined.

    :return: Lines.
    """
    blocks = re.findall(r"```bash\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    return [l for b in blocks for l in b.replace("\\\n", " ").splitlines()]


@needs_readme
def test_readme_marks_the_hub_write_and_the_launch_user_go() -> None:
    """The runbook numbers its steps and marks exactly the push and the launch USER GO."""
    heads = [l for l in README.read_text(encoding="utf-8").splitlines() if l.startswith("### ")]
    assert [int(h.split()[1].rstrip(".")) for h in heads] == list(range(len(heads)))
    assert "USER GO" in next(h for h in heads if "Push" in h) and "USER GO" in next(h for h in heads if "Launch" in h)
    assert sum("USER GO" in h for h in heads) == 2
    assert heads.index(next(h for h in heads if "Score the checkpoints" in h)) == heads.index(next(h for h in heads if "Launch" in h)) + 1


@needs_readme
def test_readme_commands_meet_the_real_interfaces() -> None:
    """The runbook's commands parse with the real command-line interfaces."""
    lines = _bash_lines()
    assert any(l.startswith("D=") for l in lines) and "DATASET=$D/genizah_novel_v1" in lines and "EXPORT=$D/genizah_novel_v1_images_once" in lines
    calls = [l for l in lines if "-m src.finetuning.qwen_hebrew.colab.derive_novel_v1" in l]
    assert {shlex.split(l.split("&&")[-1])[shlex.split(l.split("&&")[-1]).index("-m") + 2] for l in calls} == {"plan", "rows", "notebook"}
    for line in calls:
        tokens = shlex.split(line.split("&&")[-1].split("2>&1")[0].replace("$REV", SHA), comments=True)
        args = dn.build_parser().parse_args(tokens[tokens.index("-m") + 2:])
        assert args.command in ("plan", "rows", "notebook")
    assert any("run_notebook_gates_novel_v1.py $OUT --cairo-dir $CAIRO" in l for l in lines)
    assert any("token_lengths.py $OUT" in l for l in lines) and any("v22_smoke_check --data-dir $OUT" in l for l in lines)
    assert any("test_v25a_eos_supervision.py" in l for l in lines) and any(l.startswith("REV=$(cat $R/push_novel_v1.revision)") for l in lines)
    assert sum("drive_novel_v1.sh" in l for l in lines) == 4 and any("push_novel_v1.sh" in l for l in lines)
    evals = [l for l in lines if "eval_chain_novel_v1.sh" in l]
    assert len(evals) == 1 and evals[0].startswith("cd $MAIN && screen -dmS novel_v1_eval ") and "nohup" not in evals[0]
    for script in ("run_notebook_gates_novel_v1.py", "push_novel_v1.sh", "drive_novel_v1.sh", "eval_chain_novel_v1.sh", "run_prompt_arm.py"):
        assert (ROUND_DIR / script).is_file(), script


@needs_readme
def test_round_scripts_name_this_run() -> None:
    """Push and drive scripts name this run, stay private and load secrets inside Python only."""
    push = (ROUND_DIR / "push_novel_v1.sh").read_text(encoding="utf-8")
    drive = (ROUND_DIR / "drive_novel_v1.sh").read_text(encoding="utf-8")
    assert f"HUB={dn.DATA_REPO}" in push and "private=True" in push and "2.4 * 2**30" in push and "dotenv_values" in push
    assert "notebook --revision $REV --plan $SRC/mixture.json" in push and "dn.warm_start_pinned()" in push
    assert "is PUBLIC" in push and "push_novel_v1.revision" in push and "documents_sha256" in push
    assert "genizah_novel_v1_lambda.ipynb" in drive and "genizah_novel_v1_lambda.py" in drive and "dn.notebook_problems(nb)" in drive
    assert "dotenv_values" in drive and "v25a" not in drive.replace("drive_v25a.sh", "")
    for text in (push, drive):
        assert not re.search(r"(cat|grep|sed|head|echo)[^\n|]*\.env\b", text)


@pytest.mark.skipif(not EVAL_CHAIN.is_file(), reason="the round's eval chain is git-ignored and not on this machine")
def test_eval_chain_order_prompt_arm_and_flagships() -> None:
    """The eval chain scores PGP-131 first, unloads between stages, defaults to the neutral arm and keeps the flagships."""
    chain = EVAL_CHAIN.read_text(encoding="utf-8")
    assert "PROMPT_ARM=${PROMPT_ARM:-neutral}" in chain and "\nVER=novel-v1 " in chain
    assert "derive_novel_v1 show --notebook $NB --field eval_steps --require-pinned" in chain
    assert "FLAGSHIPS=(qwen3-vl-8b-heb-v24a-step2000 qwen3-vl-8b-heb-v25a-step1200)" in chain and "KEEP=($PINNED $FLAGSHIPS)" in chain
    body = chain[chain.index("score_model() {"):chain.index("eval_step() {")]
    stages = [body.index(k) for k in ("run_prompt_arm.py pgp131", "run_prompt_arm.py religious", "run_prompt_arm.py arabic")]
    assert stages == sorted(stages) and body.count("unload_model $KEY") == 3                       # PGP-131 first; unload between stages
    code = "\n".join(l for l in chain.splitlines() if not l.lstrip().startswith("#"))
    assert "lms load" not in code and "$LMS load" not in code and "unload --all" not in code and 'startswith(f"Training in progress, step {step}, checkpoint")' in chain
    assert "judaeo_arabic" in chain and "score_arabic_covered" in chain and "--out-dir $REPO/$SCORES" in chain


@pytest.mark.skipif(not ROUND_DIR.is_dir() or shutil.which("zsh") is None, reason="the round's scripts or zsh are not here")
@pytest.mark.parametrize("script", ["push_novel_v1.sh", "drive_novel_v1.sh", "eval_chain_novel_v1.sh"])
def test_round_scripts_parse(script: str) -> None:
    """Each round script parses under zsh -n."""
    if not (ROUND_DIR / script).is_file():
        pytest.skip(f"{script} is not on this machine")
    assert subprocess.run(["zsh", "-n", str(ROUND_DIR / script)], capture_output=True).returncode == 0
