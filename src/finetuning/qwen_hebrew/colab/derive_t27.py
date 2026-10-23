# File name: derive_t27.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Derive the text27b notebooks (Qwen3.8-27B text fine-tune: t27a translation pilot, t27b joint run) from v2.5a's.

Plan 2026-10-21 section 4 (amendment A1, decision R0: non-thinking SFT on human targets). The v2.5a notebook is the
source of the training stack that carries over unchanged: the session helpers (``train_session.py`` inlined: schedule
lock, fixed row order, resume gate, time budget), the Hub checkpoint helpers, the Hub-confirmed milestone pushes,
``make_sft_config`` / ``make_trainer`` and the prepared-dataloader recording wrapper. Those blocks are taken from the
v2.5a Colab notebook by asserted anchors (:func:`kept_blocks`), so they cannot drift silently. Everything that was
vision-specific is replaced:

* **data** (cell 2): chat rows of the v0_2 mixture (``<arm>/train.jsonl``, materialised and seeded by
  ``text27b.mixture``; trained once, in file order) and the E3 dev monitor from a private dataset at a pinned revision;
  the cell refuses to run unless the revision (40-hex) and the mixture's text fingerprint (64-hex,
  :func:`text_fingerprint`, the same digest as ``text27b.rows.fingerprint``) are pinned;
* **model** (cell 3): ``unsloth/Qwen3.8-27B-unsloth-bnb-4bit`` (``Qwen/Qwen3.5-2B`` on rung 1, the smoke flag) with a
  language-only LoRA regex (:data:`LORA_REGEX`: DeltaNet ``in_proj_qkv`` / ``in_proj_z`` / ``out_proj``, attention
  ``q/k/v/o``, MLP; never the vision tower, the MTP head or the DeltaNet ``in_proj_a`` / ``in_proj_b``);
* **collator** (cell 4): every row rendered once with ``enable_thinking=False`` and tokenized once; labels mask the
  prompt prefix (system + user + assistant header + the empty think block); nothing is truncated;
* tower / pixel / merger checks are gone; the prepared-vs-direct loss check stays.

**Gates** (cells; each raises on failure):

* G1 (cell 1) the kernel satisfies ``host_lambda.PIN_SETS["text27b"]`` and the fla / causal-conv1d kernels import;
* G2 (cell 2) pinned revision + fingerprint, v0_2 schema, ``split == "train"`` only, the arm's families, dev monitor
  disjoint from train and free of reader-exposed reads;
* G3 (cell 3) 0 vision / MTP / ``in_proj_a|b`` LoRA tensors, the expected ``lora_A`` count per module group (400 on
  the 27B), no trainable non-LoRA parameter;
* G4 (cell 3) the Gated DeltaNet layers run the fla / causal-conv1d kernels, not the torch fallback;
* G5 (cell 4) render contract on every row (one empty think block, no "Reasoning effort" sentence), 0 rows above
  6,144 tokens, mask guardrail (answer = supervised tokens; prompt and empty think block masked);
* G6 (cell 5) a gradient on every language ``lora_B``, per module group (DeltaNet / attention / MLP), none elsewhere;
* G7 (cell 8) 32 rows per family decoded from the PREPARED train dataloader equal the cell-4 tokens and masks;
* G8 (cell 8) prepared-vs-direct loss within 5 %, and the warm-up survived ``make_sft_config``;
* G9 (cell 6, rungs) 30 timed steps (the worst-case batch first), batch-geometry sweep, peak VRAM, GO estimate;
  (cell 8, every fresh start) step-0 dev loss per family = the 2x loss-alarm reference;
* G10 (cells 8-9, rungs) a kill at step 10 and an exact resume from the Hub's copy in a fresh trainer, the
  end-of-run save at a non-save step, scratch push, merge test (rung 1 also reloads the merge and compares the loss).

**Stages** (``T27_STAGE``; a separate GO for each): ``rung1`` (the 2B smoke), ``rung2`` (the 27B gates and timing),
``train`` (the run; ``T27_ARM`` = ``t27a`` or ``t27b``). W&B project ``genizah-text27b`` (private), never
``qwen-hebrew-finetune``. New private checkpoint and merged repositories per arm; scratch repositories for the rungs.

``PIN_SETS["text27b"]`` records the resolved smoke-environment versions. G1 asserts the kernel pins at runtime.
Setup preserves the pinned torch/CUDA build and compiles causal-conv1d against it. The model cell verifies both
attribute-based and module-level kernel dispatch, repairing the known stale Unsloth recurrent wrapper before G4.

Also here: **rung 0** (:func:`rung0`), the notebook's G2 / G5 logic run locally on a mixture file with the local
tokenizer and chat template (no GPU, no network).

``derive_v25a`` is not imported (it needs ``derive_v24a``, which is not on every branch): the v2.5a Colab notebook is
read as JSON.

Usage::

    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_t27 [--t27a-revision <40-hex> --t27a-mixture FILE]
        [--t27b-revision <40-hex> --t27b-mixture FILE] [--t27a-fingerprint <64-hex>] [--template FILE | --no-template]
        [--fresh] [--host both|colab|lambda] [--src colab/genizah_v25a.ipynb] [--out-dir DIR]
    .venv/bin/python -m src.finetuning.qwen_hebrew.colab.derive_t27 --rung0 MIXTURE.jsonl [--rung0-arm t27a|t27b]
        [--rung0-split train|dev] [--rung0-limit N] [--rung0-tokenizer tokenizer.json] [--rung0-template chat_template.jinja]
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from src.finetuning.qwen_hebrew.colab import host_lambda
from src.finetuning.qwen_hebrew.colab.derive_v22b import _replace_line, _replace_once
from src.finetuning.text27b import rows as t27rows

HERE = Path(__file__).resolve().parent

SOURCE_NOTEBOOK = "genizah_v25a.ipynb"
NOTEBOOK = "genizah_t27.ipynb"
PLACEHOLDER = "PIN-AFTER-PUSH"
PIN_SET = "text27b"
SETUP_SCRIPT = "lambda_setup_t27.sh"
VENV = "$HOME/genizah-t27-venv"
KERNEL_NAME = "genizah-t27"
KERNEL_DISPLAY = "genizah t27 (venv)"
RESOLVED_PINS_FILE = "t27_resolved_pins.txt"

DATA_REPO = "isaacmg/genizah_text27b"          # private dataset: the local v0_2 mixtures + the dev monitor, pushed by the user
LOCAL_LEAF = "genizah_text27b"
DEV_FILE = "dev/dev_monitor.jsonl"              # E3 dev monitor rows (eval_sets), one file for both arms
BASE_MODEL = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
SMOKE_BASE_MODEL = "Qwen/Qwen3.5-2B"             # rung 1: the same qwen3_5 architecture at 2B
MAX_SEQ = 6144
LORA_RANK = 16
LORA_ALPHA = 16
LR = "1e-4"
WARMUP_RATIO = "0.03"
WEIGHT_DECAY = "0.01"
OPTIM = "adamw_8bit"
SEED = 3407
ROWS_PER_STEP = 16
SAVE_STEPS = 50
EVAL_STEPS = 100
LADDER_STEPS = 15                                # a rung: killed at step 10, resumed to 15 (not a save step: the end-of-run save)
KILL_AT = 10
TIMED_STEPS = 30                                 # rung 2
SMOKE_TIMED_STEPS = 5                            # rung 1: the code path only
BATCH_GEOMETRIES: Tuple[Tuple[int, int], ...] = ((1, 16), (2, 8), (4, 4))
DECODE_ROWS_PER_FAMILY = 32
LOSS_TOLERANCE = 0.05
ALARM_FACTOR = 2.0
#: 27B: 48 DeltaNet layers x 3 + 16 attention layers x 4 + 64 MLPs x 3 (plan section 4; the local copy's config agrees)
EXPECTED_LORA_A_27B = 400
WANDB_PROJECT = "genizah-text27b"
PUBLIC_WANDB_PROJECT = "qwen-hebrew-finetune"
PRICE_PER_H = 1.99                               # A100 40 GB on demand (plan section 4)
BUDGET_FACTOR = 1.3                              # Lambda safety stop = the plan's upper hours x this
LADDER_BUDGET_H = 3.0
COLAB_BUDGET_H = 7.5                             # Colab ends runtimes at ~8 h (derive_v23a.TIME_BUDGET_H)
MERGED_GB_27B = 55.6
MERGED_GB_SMOKE = 4.5
STAGES: Tuple[str, ...] = ("rung1", "rung2", "train")

#: PEFT full-matches a string ``target_modules`` against module names: language projections only. The leading look-ahead
#: keeps the vision tower and the MTP head out even where their names would otherwise match (``mtp.layers.0.self_attn``).
LORA_REGEX = (r"(?!.*(?:visual|vision|mtp)).*\.layers\.\d+\."
              r"(?:linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
              r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
              r"|mlp\.(?:gate_proj|up_proj|down_proj))")

EMPTY_THINK = t27rows.EMPTY_THINK
EFFORT_MARKER = t27rows.EFFORT_MARKER
IM_END = "<|im_end|>\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n"


@dataclass(frozen=True)
class Arm:
    """One of the two text27b runs.

    :param name: ``t27a`` or ``t27b`` (the value of ``T27_ARM``).
    :param families: Families its train rows may carry.
    :param required: Families its train rows must carry.
    :param push_every: Hub-confirmed milestone every this many steps (a multiple of :data:`SAVE_STEPS`).
    :param milestone_fractions: Further milestones at these fractions of the run (rounded to a save step).
    :param plan_hours: The plan's training-hours range on the A100 (the GO number replaces it).
    :param merge_hours: Merge + push hours after training.
    """
    name: str
    families: Tuple[str, ...]
    required: Tuple[str, ...]
    push_every: int
    milestone_fractions: Tuple[float, ...]
    plan_hours: Tuple[float, float]
    merge_hours: float

    @property
    def run_name(self) -> str:
        """W&B and trainer run name.

        :returns: ``genizah_<name>``.
        """
        return f"genizah_{self.name}"

    @property
    def ckpt_repo(self) -> str:
        """Checkpoint repository (new, private).

        :returns: ``isaacmg/qwen3.8-27b-genizah-<name>-ckpt``.
        """
        return f"isaacmg/qwen3.8-27b-genizah-{self.name}-ckpt"

    @property
    def merged_repo(self) -> str:
        """Merged-model repository (new, private).

        :returns: ``isaacmg/qwen3.8-27b-genizah-<name>-merged``.
        """
        return f"isaacmg/qwen3.8-27b-genizah-{self.name}-merged"

    @property
    def lambda_budget_h(self) -> float:
        """Lambda safety stop of the run.

        :returns: The plan's upper hours x :data:`BUDGET_FACTOR`, one decimal.
        """
        return round(self.plan_hours[1] * BUDGET_FACTOR, 1)


T27A = Arm("t27a", ("translate_en", "translate_he"), ("translate_en", "translate_he"), 100, (), (4.0, 10.0), 1.5)
T27B = Arm("t27b", tuple(t27rows.FAMILIES),
           ("translate_en", "translate_he", "catalogue_json", "describe_pgp", "extract_relations"), 250, (0.5,), (9.5, 25.0), 2.0)
ARMS: Dict[str, Arm] = {"t27a": T27A, "t27b": T27B}


@dataclass
class Pins:
    """The values a notebook is pinned to (each a hex digest or :data:`PLACEHOLDER`).

    :param revisions: Arm -> 40-hex revision of :data:`DATA_REPO`.
    :param fingerprints: Arm -> 64-hex :func:`text_fingerprint` of ``<arm>/train.jsonl``.
    :param template_sha256: sha256 of the local chat template rung 0 checked (informational in the notebook), or None.
    """
    revisions: Dict[str, str] = field(default_factory=lambda: {a: PLACEHOLDER for a in ARMS})
    fingerprints: Dict[str, str] = field(default_factory=lambda: {a: PLACEHOLDER for a in ARMS})
    template_sha256: Optional[str] = None

    def validate(self) -> None:
        """Check every value.

        :returns: None.
        :raises ValueError: On an unknown arm or a value that is neither a hex digest of the right length nor the placeholder.
        """
        for what, values, n in (("revision", self.revisions, 40), ("fingerprint", self.fingerprints, 64)):
            if set(values) != set(ARMS):
                raise ValueError(f"{what}s must name exactly the arms {sorted(ARMS)}, got {sorted(values)}")
            for arm, value in values.items():
                if value != PLACEHOLDER and not re.fullmatch(rf"[0-9a-f]{{{n}}}", value):
                    raise ValueError(f"{arm} {what} must be a {n}-hex digest or {PLACEHOLDER}, got {value!r}")
        if self.template_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.template_sha256):
            raise ValueError(f"template sha256 must be 64-hex or None, got {self.template_sha256!r}")


# ===================================================================================== functions inlined into the notebook
# Self-contained (stdlib + typing only, literal defaults): inspect.getsource copies them into the cells, and the tests run
# them here. Notebook globals they rely on are passed in as arguments.


def check_pin(value: str, length: int, what: str, fix: str) -> str:
    """Refuse anything but a lowercase hex digest of ``length`` characters.

    :param value: The pinned value (a revision or a fingerprint).
    :param length: Expected number of hex characters (40 for a revision, 64 for a sha256).
    :param what: Name of the value, for the message.
    :param fix: What to do about it, for the message.
    :returns: ``value``.
    :raises RuntimeError: When ``value`` is a placeholder or malformed.
    """
    if len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError(f"{what} {value!r} is not a pinned {length}-hex value: {fix}")
    return value


def pin_problems(specifiers: Dict[str, str], get_version: Callable[[str], str]) -> List[str]:
    """Distributions that are missing or outside their PEP 440 specifier.

    :param specifiers: Distribution name -> specifier ('' = any version).
    :param get_version: ``importlib.metadata.version`` (or a stand-in); raises ``ImportError`` for a missing distribution.
    :returns: One line per problem; empty when the stack satisfies every specifier.
    """
    from packaging.specifiers import SpecifierSet
    problems: List[str] = []
    for name, spec in specifiers.items():
        try:
            have = get_version(name)
        except ImportError:   # importlib.metadata.PackageNotFoundError is an ImportError: a missing distribution is reported
            problems.append(f"{name}: not installed")
            continue
        if spec and not SpecifierSet(spec).contains(have, prereleases=True):
            problems.append(f"{name}: {have} does not satisfy {spec!r}")
    return problems


def read_jsonl(path: Any) -> List[Dict[str, Any]]:
    """Rows of a JSONL file (blank lines skipped).

    :param path: File path.
    :returns: Parsed rows in file order.
    """
    import json
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def text_fingerprint(rows: Iterable[Dict[str, Any]], keys: Sequence[str] = ("system", "user", "assistant")) -> str:
    """Order-sensitive sha256 over the message texts of rows (the digest ``text27b.rows.fingerprint`` computes).

    :param rows: Rows in their materialised order.
    :param keys: Message roles hashed per row.
    :returns: Hex digest.
    """
    import hashlib
    import json
    h = hashlib.sha256()
    for row in rows:
        for k in keys:
            value = next((m.get("content") for m in row.get("messages") or [] if m.get("role") == k), None)
            text = "" if value is None else (value if isinstance(value, str) else
                                             json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            h.update(k.encode("utf-8"))
            h.update(b"\x1f")
            h.update(text.encode("utf-8"))
            h.update(b"\x1e")
        h.update(b"\x1d")
    return h.hexdigest()


def row_problems(row: Dict[str, Any], split: str, families: Sequence[str], conditions: Sequence[str],
                 build: str = "v0_2") -> List[str]:
    """Schema problems of one chat row (G2).

    :param row: ``{"messages": [system, user, assistant], "meta": {...}}``.
    :param split: The split every row of this file must carry.
    :param families: Families the row may belong to.
    :param conditions: Allowed input conditions.
    :param build: Required ``meta.build``.
    :returns: One line per problem; empty for a valid row.
    """
    problems: List[str] = []
    msgs = row.get("messages")
    roles = [m.get("role") for m in msgs] if isinstance(msgs, list) else None
    if roles != ["system", "user", "assistant"]:
        problems.append(f"roles {roles}, expected system / user / assistant")
    elif any(not isinstance(m.get("content"), str) for m in msgs):
        problems.append("a message content is not a string")
    elif not msgs[-1]["content"].strip():
        problems.append("empty assistant target")
    meta = row.get("meta") or {}
    if meta.get("split") != split:
        problems.append(f"split {meta.get('split')!r}, expected {split!r}")
    if meta.get("build") != build:
        problems.append(f"build {meta.get('build')!r}, expected {build!r}")
    if meta.get("family") not in families:
        problems.append(f"family {meta.get('family')!r} is not one of {list(families)}")
    if meta.get("condition") not in conditions:
        problems.append(f"condition {meta.get('condition')!r} is not one of {list(conditions)}")
    if meta.get("held_out_vision_benchmark") is True:
        problems.append("a vision-benchmark hold-out")
    if not meta.get("row_id"):
        problems.append("no row_id")
    return problems


def module_group(name: str) -> Optional[str]:
    """Module group of a parameter or module name.

    :param name: Dotted name (``...layers.3.linear_attn.in_proj_qkv.lora_A.default.weight``).
    :returns: ``vision``, ``mtp``, ``deltanet``, ``attention``, ``mlp``, or None outside those.
    """
    if "visual" in name or "vision" in name:
        return "vision"
    if "mtp" in name:
        return "mtp"
    if ".linear_attn." in name:
        return "deltanet"
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    return None


def expected_lora_counts(layer_types: Sequence[str]) -> Dict[str, int]:
    """LoRA-adapted modules per group for a hybrid qwen3_5 stack with dense MLPs.

    DeltaNet layers adapt ``in_proj_qkv``, ``in_proj_z`` and ``out_proj`` (3), attention layers ``q/k/v/o`` (4), every
    layer's MLP ``gate/up/down`` (3).

    :param layer_types: ``config.layer_types`` (``linear_attention`` / ``full_attention`` per layer).
    :returns: ``{"deltanet": n, "attention": n, "mlp": n}``.
    :raises ValueError: On a layer type this rule does not know.
    """
    unknown = sorted(set(layer_types) - {"linear_attention", "full_attention"})
    if unknown:
        raise ValueError(f"unknown layer types {unknown}: the expected LoRA count covers linear_attention / full_attention")
    n_linear = sum(1 for t in layer_types if t == "linear_attention")
    n_full = sum(1 for t in layer_types if t == "full_attention")
    return {"deltanet": 3 * n_linear, "attention": 4 * n_full, "mlp": 3 * len(layer_types)}


def lora_problems(lora_a_names: Sequence[str], layer_types: Sequence[str],
                  expected_total: Optional[int] = None) -> Tuple[Dict[str, int], List[str]]:
    """G3: where the adapter sits, against the language-only plan.

    :param lora_a_names: Names of the ``lora_A`` parameters (or of the moved ``lora_B`` mapped to ``lora_A``).
    :param layer_types: ``config.layer_types`` of the language model.
    :param expected_total: Required total over the language groups (400 on the 27B), or None.
    :returns: ``(counts per group, problems)``.
    """
    import re
    counts = {"deltanet": 0, "attention": 0, "mlp": 0, "vision": 0, "mtp": 0, "other": 0}
    others: List[str] = []
    for n in lora_a_names:
        group = module_group(n)
        counts[group or "other"] += 1
        if group is None:
            others.append(n)
    problems: List[str] = []
    if counts["vision"]:
        problems.append(f"{counts['vision']} LoRA tensors on the vision tower (must be 0)")
    if counts["mtp"]:
        problems.append(f"{counts['mtp']} LoRA tensors on the MTP head (must be 0)")
    if others:
        problems.append(f"{len(others)} LoRA tensors outside the language groups, e.g. {others[:3]}")
    gates = [n for n in lora_a_names if re.search(r"\.linear_attn\.in_proj_[ab]\.", n)]
    if gates:
        problems.append(f"{len(gates)} LoRA tensors on DeltaNet in_proj_a / in_proj_b (must be 0), e.g. {gates[:2]}")
    expected = expected_lora_counts(layer_types)
    for group, n in expected.items():
        if counts[group] != n:
            problems.append(f"{group}: {counts[group]} lora_A tensors, expected {n}")
    total = sum(counts[g] for g in expected)
    if expected_total is not None and total != expected_total:
        problems.append(f"{total} language lora_A tensors, expected {expected_total}")
    return counts, problems


def repair_recurrent_kernel(modeling_module: Any, kernel_decorator: Callable[..., Any]) -> bool:
    """Rebind Unsloth's stale recurrent wrapper using Transformers' correct FLA kernel name.

    :param modeling_module: The loaded model's modeling module.
    :param kernel_decorator: Transformers' ``use_kernel_func_from_hub_with_fallback`` factory.
    :returns: Whether a known fallback wrapper was rebuilt; G4 still verifies the selected implementation.
    """
    import inspect
    wrapper = getattr(modeling_module, "torch_recurrent_gated_delta_rule", None)
    if not inspect.isfunction(wrapper) or not hasattr(wrapper, "__wrapped__"):
        return False
    if inspect.getclosurevars(wrapper).nonlocals.get("is_new_implementation") is not False:
        return False
    modeling_module.torch_recurrent_gated_delta_rule = kernel_decorator("fused_recurrent_gated_delta_rule", "fla")(wrapper.__wrapped__)
    return True


def fast_path_problems(layer: Any, module_flag: Optional[bool], modeling_module: Optional[Any] = None) -> List[str]:
    """G4: whether a Gated DeltaNet layer runs the fla / causal-conv1d kernels.

    :param layer: One ``linear_attn`` module (it holds its kernels as attributes, the torch fallbacks named ``torch_*``).
    :param module_flag: ``is_fast_path_available`` of its modeling module, or None when the module has none.
    :param modeling_module: Modeling module for Transformers versions with module-level kernel dispatch wrappers.
    :returns: One line per problem; empty when the fast path is in use.
    """
    kernels = ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "recurrent_gated_delta_rule")
    found = {k: getattr(layer, k) for k in kernels if hasattr(layer, k)}
    problems: List[str] = []
    if module_flag is False:
        problems.append("the modeling module reports is_fast_path_available = False (fla / causal-conv1d not importable there)")
    if not found and modeling_module is not None:
        import inspect
        wrappers = {"causal_conv1d_fn": "causal_conv1d", "causal_conv1d_update": "causal_conv1d",
                    "torch_chunk_gated_delta_rule": "fla", "torch_recurrent_gated_delta_rule": "fla"}
        for name, package in wrappers.items():
            fn = getattr(modeling_module, name, None)
            seen = set()
            verified = False
            while callable(fn) and id(fn) not in seen:
                seen.add(id(fn))
                captured = inspect.getclosurevars(fn).nonlocals if inspect.isfunction(fn) else {}
                if "is_new_implementation" in captured:
                    implementation = captured.get("implementation")
                    verified = (captured["is_new_implementation"] is True and callable(implementation)
                                and getattr(implementation, "__module__", "").split(".")[0] == package)
                    break
                fn = getattr(fn, "__wrapped__", None)
            if not verified:
                problems.append(f"{name}: cannot verify an active {package} kernel in the dispatch wrapper")
        return problems
    if not found and module_flag is None:
        problems.append(f"{type(layer).__name__} exposes neither the kernel attributes {kernels} nor is_fast_path_available: "
                        "check the fast path by hand before the run")
    for name, fn in found.items():
        if fn is None:
            problems.append(f"{name} is None (its kernel library is missing)")
        elif getattr(fn, "__name__", "").startswith("torch_"):
            problems.append(f"{name} is the torch fallback {fn.__name__}")
    return problems


def nfc_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copies of chat turns with NFC-normalised string contents.

    The tokenizer normalises text to NFC, so the supervised tokens of a decomposed answer (``t`` + U+0323) decode
    to the composed form; rendering NFC text keeps the mask guardrail's answer comparison exact (fix pass F3).

    :param messages: Chat turns.
    :returns: New turn dicts; non-string contents are copied unchanged.
    """
    import unicodedata
    return [dict(m, content=unicodedata.normalize("NFC", m["content"])) if isinstance(m.get("content"), str) else dict(m)
            for m in messages]


def render_problems(rendered: str, messages: Sequence[Dict[str, Any]], empty_think: str = "<think>\n\n</think>\n\n",
                    effort_marker: str = "Reasoning effort", im_end: str = "<|im_end|>\n") -> List[str]:
    """G5: the ``enable_thinking=False`` render contract (the rule of ``text27b.rows.check_render``).

    :param rendered: The chat template's output for a training row.
    :param messages: The rendered messages (ending in the assistant target).
    :param empty_think: The empty think block.
    :param effort_marker: Start of the sentence the template injects when thinking is on.
    :param im_end: End-of-turn marker.
    :returns: One line per broken rule; empty when the render is right.
    """
    problems: List[str] = []
    if effort_marker in rendered:
        problems.append("the reasoning-effort sentence is in the render (enable_thinking was not honoured)")
    n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
    if rendered.count(empty_think) != n_assistant or rendered.count("<think>") != n_assistant:
        problems.append(f"{rendered.count(empty_think)} empty think blocks and {rendered.count('<think>')} <think> tags, "
                        f"expected {n_assistant} of each")
    if not messages or messages[-1].get("role") != "assistant":
        problems.append("the last turn is not the assistant target")
    elif not rendered.endswith(empty_think + str(messages[-1].get("content") or "").strip() + im_end):
        problems.append("the render does not end with the empty think block + the answer + <|im_end|>")
    return problems


def split_rendered(rendered: str, content: str, im_end: str = "<|im_end|>\n",
                   assistant_open: str = "<|im_start|>assistant\n",
                   empty_think: str = "<think>\n\n</think>\n\n") -> Tuple[str, str]:
    """Split a rendered row into the masked prompt and the supervised completion.

    :param rendered: The chat template's output.
    :param content: The assistant target (the template strips it).
    :param im_end: End-of-turn marker.
    :param assistant_open: Assistant header.
    :param empty_think: The empty think block.
    :returns: ``(prompt, completion)``; the prompt ends with the assistant header and the empty think block, the
        completion is the stripped answer + ``<|im_end|>\\n``.
    :raises ValueError: When the render does not end with the completion or the prompt does not end with the header.
    """
    completion = str(content or "").strip() + im_end
    if not rendered.endswith(completion):
        raise ValueError("the render does not end with the answer + <|im_end|>")
    prompt = rendered[: len(rendered) - len(completion)]
    if not prompt.endswith(assistant_open + empty_think):
        raise ValueError("the prompt does not end with the assistant header and the empty think block")
    return prompt, completion


def prompt_token_count(offsets: Sequence[Sequence[int]], boundary: int) -> int:
    """Number of leading tokens that lie in the prompt (character offsets end at or before ``boundary``).

    :param offsets: ``(start, end)`` character offsets of every token, in order.
    :param boundary: Length of the prompt in characters.
    :returns: The prompt's token count; every later token starts at or after ``boundary``.
    :raises ValueError: When a token straddles the boundary or a prompt token follows an answer token.
    """
    n, in_answer = 0, False
    for start, end in offsets:
        if end <= boundary and not in_answer:
            n += 1
        elif start >= boundary:
            in_answer = True
        else:
            raise ValueError(f"token ({start}, {end}) straddles the prompt / answer boundary at {boundary}")
    return n


def supervised_problems(ids: Sequence[int], labels: Sequence[int], n_prompt: int, prompt: str, completion: str,
                        decode: Callable[[Sequence[int]], str], think_ids: Sequence[int] = (),
                        ignore_index: int = -100) -> List[str]:
    """G5 / G7 mask guardrail on one unpadded row: the answer is exactly the supervised part.

    :param ids: Input ids of the row.
    :param labels: Its labels.
    :param n_prompt: Prompt token count (the masked prefix).
    :param prompt: The prompt text (system + user + assistant header + empty think block).
    :param completion: The answer + ``<|im_end|>\\n``.
    :param decode: Token ids -> text, lossless (special tokens kept).
    :param think_ids: Ids of ``<think>`` / ``</think>``; none may be supervised.
    :param ignore_index: The masked label value.
    :returns: One line per problem; empty when the mask is right.
    """
    ids, labels = list(ids), list(labels)
    problems: List[str] = []
    if len(ids) != len(labels):
        problems.append(f"{len(ids)} ids but {len(labels)} labels")
        return problems
    masked = [i for i, label in enumerate(labels) if label == ignore_index]
    if masked != list(range(n_prompt)):
        problems.append(f"labels mask {len(masked)} tokens, not exactly the {n_prompt}-token prompt prefix")
    if labels[n_prompt:] != ids[n_prompt:]:
        problems.append("supervised labels differ from the input ids")
    if not ids[n_prompt:]:
        problems.append("nothing is supervised")
    if decode(ids[:n_prompt]) != prompt:
        problems.append("the masked prefix does not decode to the prompt (system + user + assistant header + empty think block)")
    if decode(ids[n_prompt:]) != completion:
        problems.append("the supervised tokens do not decode to the answer + <|im_end|>")
    if set(think_ids) & set(ids[n_prompt:]):
        problems.append("a think tag is supervised")
    return problems


def prepared_row_problems(ids: Sequence[int], labels: Sequence[int], attention: Sequence[int], expected_ids: Sequence[int],
                          n_prompt: int, prompt: str, completion: str, decode: Callable[[Sequence[int]], str],
                          think_ids: Sequence[int] = ()) -> List[str]:
    """G7: one row of a prepared (collated, right-padded) batch against the row tokenized in cell 4.

    :param ids: The batch row's input ids (padded).
    :param labels: Its labels (padded with -100).
    :param attention: Its attention mask.
    :param expected_ids: The row's ids as tokenized in cell 4.
    :param n_prompt: The row's prompt token count.
    :param prompt: The row's prompt text.
    :param completion: The row's answer + ``<|im_end|>\\n``.
    :param decode: Token ids -> text, lossless.
    :param think_ids: Ids of the think tags.
    :returns: One line per problem; empty when the prepared row is the planned one.
    """
    attention = list(attention)
    n = sum(1 for a in attention if a)
    if attention != [1] * n + [0] * (len(attention) - n):
        return ["the attention mask is not a right-padded prefix"]
    ids, labels = list(ids)[:n], list(labels)[:n]
    if ids != list(expected_ids):
        return [f"the prepared ids ({n} tokens) are not the row's cell-4 tokens ({len(expected_ids)})"]
    return supervised_problems(ids, labels, n_prompt, prompt, completion, decode, think_ids)


def grad_group_problems(entries: Iterable[Tuple[str, Optional[float]]],
                        expected: Dict[str, int]) -> Tuple[Dict[str, Tuple[int, float]], List[str]]:
    """G6: gradient norm per module group over the ``lora_B`` tensors after one backward pass.

    :param entries: ``(parameter name, gradient L2 norm or None)`` per ``lora_B`` tensor.
    :param expected: Tensors per language group that must carry a non-zero gradient (:func:`expected_lora_counts`).
    :returns: ``(group -> (tensors with a non-zero gradient, L2 norm of the group), problems)``.
    """
    import math
    n: Dict[str, int] = {}
    sq: Dict[str, float] = {}
    for name, norm in entries:
        group = module_group(name) or "other"
        if norm is None or not norm > 0:
            continue
        n[group] = n.get(group, 0) + 1
        sq[group] = sq.get(group, 0.0) + float(norm) ** 2
    summary = {g: (n[g], math.sqrt(sq[g])) for g in n}
    problems: List[str] = []
    for group in sorted(set(n) - set(expected)):
        problems.append(f"{n[group]} {group} lora_B tensors got a gradient (only the language groups may)")
    for group, want in expected.items():
        if n.get(group, 0) != want:
            problems.append(f"{group}: {n.get(group, 0)} of {want} lora_B tensors got a non-zero gradient")
    return summary, problems


def family_quota(counts: Dict[str, int], k: int = 32) -> Dict[str, int]:
    """Rows to decode per family in G7.

    :param counts: Train rows per family.
    :param k: Rows per family (fewer when a family has fewer).
    :returns: ``family -> min(k, rows)``.
    """
    return {family: min(k, int(n)) for family, n in counts.items()}


def milestone_steps(max_steps: int, save_steps: int, push_every: int, fractions: Sequence[float] = ()) -> List[int]:
    """Steps whose checkpoint is pushed and confirmed on the Hub before training goes on.

    :param max_steps: Steps of the run (the last is always a milestone: the trainer saves at the end of training).
    :param save_steps: Save interval (every milestone but the last is a save step).
    :param push_every: Milestone interval (a multiple of ``save_steps``).
    :param fractions: Further milestones at these fractions of the run, rounded to the nearest save step.
    :returns: Sorted milestone steps.
    :raises ValueError: On a non-positive size or a push interval that is not a multiple of the save interval.
    """
    if max_steps <= 0 or save_steps <= 0 or push_every <= 0:
        raise ValueError("max_steps, save_steps and push_every must be positive")
    if push_every % save_steps:
        raise ValueError(f"push_every {push_every} is not a multiple of save_steps {save_steps}")
    steps = set(range(push_every, max_steps + 1, push_every))
    for f in fractions:
        if not 0 < f <= 1:
            raise ValueError(f"milestone fraction {f} is not in (0, 1]")
        steps.add(min(max(int(round(f * max_steps / save_steps)) * save_steps, save_steps), max_steps))
    steps.add(max_steps)
    return sorted(steps)


def pick_geometry(results: Dict[Tuple[int, int], Optional[float]]) -> Tuple[int, int]:
    """The fastest batch geometry that did not run out of memory.

    :param results: ``(per-device batch, accumulation) -> seconds per step``, None for OOM.
    :returns: The fastest geometry (ties: the smaller per-device batch).
    :raises RuntimeError: When every geometry ran out of memory.
    """
    ok = {g: t for g, t in results.items() if t is not None}
    if not ok:
        raise RuntimeError("every batch geometry ran out of memory")
    return min(ok, key=lambda g: (ok[g], g[0]))


def go_estimate(s_per_step: float, steps: int, price_per_h: float, extra_h: float) -> Dict[str, float]:
    """The GO number: hours and dollars of a run from a measured step time.

    :param s_per_step: Seconds per optimizer step.
    :param steps: Steps of the run.
    :param price_per_h: Instance price per hour.
    :param extra_h: Hours after training (merge, push).
    :returns: ``{"train_h", "total_h", "usd"}``.
    """
    train_h = s_per_step * steps / 3600
    total_h = train_h + extra_h
    return {"train_h": round(train_h, 2), "total_h": round(total_h, 2), "usd": round(total_h * price_per_h, 2)}


def loss_alarm(value: float, reference: Optional[float], factor: float = 2.0) -> bool:
    """Whether a loss crossed ``factor`` x its step-0 reference (the gate-the-train-path rule).

    :param value: The logged loss.
    :param reference: The step-0 dev loss it is compared with (None or <= 0: no alarm).
    :param factor: Alarm level.
    :returns: True when ``value > factor * reference``.
    """
    return reference is not None and reference > 0 and value > factor * reference


def merged_dir_problems(files: Sequence[str], weight_keys: Sequence[str], config: Dict[str, Any],
                        has_chat_template: bool) -> List[str]:
    """G10 merge test on a merged (bf16) model folder.

    :param files: File names in the folder.
    :param weight_keys: Tensor names of its weights.
    :param config: Its ``config.json``.
    :param has_chat_template: Whether the chat template was saved with it.
    :returns: One line per problem; empty when the folder is a complete, adapter-free model.
    """
    problems: List[str] = []
    if "config.json" not in files:
        problems.append("no config.json")
    if not any(f.endswith(".safetensors") for f in files):
        problems.append("no safetensors weights")
    if "qwen3_5" not in str(config.get("model_type", "")):
        problems.append(f"model_type {config.get('model_type')!r} is not qwen3_5")
    lora = [k for k in weight_keys if "lora_" in k]
    if lora:
        problems.append(f"{len(lora)} LoRA tensors left in the merge, e.g. {lora[:2]}")
    if not any(".layers." in k for k in weight_keys):
        problems.append("no language-layer weights")
    if config.get("vision_config") and not any("visual" in k or "vision" in k for k in weight_keys):
        problems.append("the config has a vision tower but the merge has no vision weights")
    if not has_chat_template:
        problems.append("no chat template saved with the merge")
    return problems


NOTEBOOK_FUNCTIONS: Tuple[Callable[..., Any], ...] = (
    check_pin, pin_problems, read_jsonl, text_fingerprint, row_problems, module_group, expected_lora_counts, lora_problems,
    fast_path_problems, render_problems, split_rendered, prompt_token_count, supervised_problems, prepared_row_problems,
    grad_group_problems, family_quota, milestone_steps, pick_geometry, go_estimate, loss_alarm, merged_dir_problems)


# ===================================================================================== derive-time helpers


def inline_source(*funcs: Callable[..., Any]) -> str:
    """Source of functions for a notebook cell.

    :param funcs: Module-level functions of this module.
    :returns: Their source after a blank line, separated by blank lines.
    """
    return "\n" + "\n\n".join(inspect.getsource(f).rstrip("\n") + "\n" for f in funcs)


def lora_targets(names: Iterable[str], regex: str = LORA_REGEX) -> List[str]:
    """Module names PEFT adapts for a string ``target_modules`` (``re.fullmatch`` on every module name).

    :param names: Module names.
    :param regex: The target regex.
    :returns: The names it selects, in input order.
    """
    pattern = re.compile(regex)
    return [n for n in names if pattern.fullmatch(n)]


def _raw_literal(text: str) -> str:
    """A raw string literal for a regex in generated code.

    :param text: The regex.
    :returns: ``r'...'``.
    :raises ValueError: When the text cannot be written as a raw single-quoted literal.
    """
    if "'" in text or "\n" in text or text.endswith("\\"):
        raise ValueError("the regex cannot be written as a raw single-quoted literal")
    return "r'" + text + "'"


def _fill(template: str, **values: str) -> str:
    """Replace ``«KEY»`` markers; every marker must be used and none may remain.

    :param template: Text with markers.
    :param values: Marker values.
    :returns: The filled text.
    :raises ValueError: On an unused value or a marker left over.
    """
    for key, value in values.items():
        marker = f"«{key}»"
        if marker not in template:
            raise ValueError(f"marker {marker} is not in the template")
        template = template.replace(marker, value)
    left = re.findall(r"«[A-Z0-9_]+»", template)
    if left:
        raise ValueError(f"markers left unfilled: {sorted(set(left))}")
    return template


def _extract(text: str, start: str, end: str, what: str) -> str:
    """The block of ``text`` from ``start`` (included) to ``end`` (excluded); both anchors must occur exactly once.

    :param text: Source text.
    :param start: Opening anchor.
    :param end: Closing anchor (searched after ``start``).
    :param what: Label for the error message.
    :returns: The block.
    :raises ValueError: When an anchor is missing or ambiguous.
    """
    for anchor in (start, end):
        if text.count(anchor) != 1:
            raise ValueError(f"{what}: anchor found {text.count(anchor)} times: {anchor[:80]!r}")
    i = text.index(start)
    j = text.find(end, i)
    if j < 0:
        raise ValueError(f"{what}: the closing anchor precedes the opening one")
    return text[i:j]


def _src(cell: Dict[str, Any]) -> str:
    """Joined source of a notebook cell.

    :param cell: Notebook cell.
    :returns: Its source text.
    """
    return "".join(cell["source"])


def _with_source(cell: Dict[str, Any], text: str) -> Dict[str, Any]:
    """A copy of ``cell`` with new source.

    :param cell: Notebook cell (its type and metadata are kept).
    :param text: New source.
    :returns: The new cell.
    """
    out = copy.deepcopy(cell)
    out["source"] = text.splitlines(keepends=True)
    if out["cell_type"] == "code":
        out["outputs"], out["execution_count"] = [], None
    return out


def _requirement(name: str, spec: str, extra: str = "") -> str:
    """A quoted pip requirement.

    :param name: Distribution name.
    :param spec: PEP 440 specifier ('' = any).
    :param extra: Extras, e.g. ``[colab-new]``.
    :returns: ``"name[extra]spec"`` in double quotes.
    """
    return f'"{name}{extra}{spec}"'


def pin_set(name: str = PIN_SET) -> Dict[str, Any]:
    """A pin set of ``host_lambda.PIN_SETS``.

    :param name: Set name.
    :returns: The set.
    :raises ValueError: When the unsloth specifier is not an ``==`` one (the Colab install line and the host substitution
        need ``unsloth[colab-new]==...``) or the kernel gate names a distribution the set does not specify.
    """
    pins = host_lambda.PIN_SETS[name]
    specs = pins["specifiers"]
    if not str(specs.get("unsloth", "")).startswith("=="):
        raise ValueError(f"PIN_SETS[{name!r}]['specifiers']['unsloth'] must start with '==' (the install line is asserted)")
    missing = [d for d in pins["kernel_gate"] if d not in specs]
    if missing:
        raise ValueError(f"PIN_SETS[{name!r}] kernel gate names unspecified distributions: {missing}")
    return pins


def kernel_pins(name: str = PIN_SET) -> Dict[str, str]:
    """The specifiers G1 asserts.

    :param name: Set name.
    :returns: Kernel-gate distribution -> specifier.
    """
    pins = pin_set(name)
    return {d: pins["specifiers"][d] for d in pins["kernel_gate"]}


# ===================================================================================== the cells

INTRO_MD = """# Genizah text27b — Qwen3.8-27B text fine-tune: t27a (translation pilot) and t27b (joint run)

Answer-only SFT on human targets (plan 2026-10-21 section 4, amendment A1; non-thinking): **Qwen3.8-27B** from
`«BASE»`, a **language-only LoRA** (r«RANK», alpha «ALPHA», dropout 0) on the Gated DeltaNet `in_proj_qkv` / `in_proj_z` /
`out_proj`, the attention `q/k/v/o` and the MLP `gate/up/down` projections — never the vision tower, the MTP head or the
DeltaNet `in_proj_a` / `in_proj_b`. Every row is rendered with **`enable_thinking=False`** (no "Reasoning effort" sentence,
one empty think block, masked with the prompt) and the loss is on the answer only. At most «MAX_SEQ» tokens a row
(nothing is truncated: a longer row fails G5), «ROWS» rows a step, LR «LR» cosine with «WARMUP_PCT» warm-up, weight decay
«WD», «OPTIM», seed «SEED», «SAVE»-step saves, dev-monitor eval every «EVAL» steps.

Kept from v2.5a: the session helpers (`train_session.py`: schedule lock, fixed row order, resume gate, time budget), the
Hub checkpoint helpers, the Hub-confirmed milestone pushes, `make_sft_config` / `make_trainer`, the prepared-dataloader
gate. The row order is the mixture file's own (materialised and seeded when the mixture was built), trained once.

| | t27a | t27b |
|---|---|---|
| Data (`«DATA_REPO»`, private) | `t27a/train.jsonl`: translate_en + translate_he, every condition, x2 | `t27b/train.jsonl`: the joint mixture (plan section 9.3) |
| Families | «A_FAMILIES» | «B_FAMILIES» |
| Steps | train rows // «ROWS» (plan: about 384) | train rows // «ROWS» (plan: about 1,227) |
| Run / repositories | `«A_RUN»`; `«A_CKPT»`, `«A_MERGED»` (new, private) | `«B_RUN»`; `«B_CKPT»`, `«B_MERGED»` (new, private) |
| Hub-confirmed milestones | every «A_PUSH» steps and the last | every «B_PUSH» steps, 50 % and the last |
| Lambda safety stop | «A_BUDGET» h | «B_BUDGET» h |

**Knobs** (cell 1, or the environment of a headless run): `T27_ARM` (t27a | t27b), `T27_STAGE` (rung1 | rung2 | train;
default rung1), `T27_BATCH` (per-device batch 1, 2 or 4; accumulation makes «ROWS» rows), `T27_SCRATCH_TAG` (names a rung's
scratch repositories; default: the launch time).

**The ladder — every stage is its own GO:**

1. `rung1` — smoke on `«SMOKE_BASE»` (the same qwen3_5 architecture): the pins resolve, G1–G8 pass, «LADDER» steps with a
   kill at step «KILL» and an exact resume from the Hub's copy, scratch push, merge and reload test (G9 code path, G10).
2. `rung2` — the 27B: every gate, «TIMED» timed steps (the worst-case batch first), the batch-geometry sweep, peak VRAM,
   step-0 dev loss per family, kill-and-resume at step «KILL», merge test (local, not pushed). Its GO estimate
   (measured s/step x steps x $«PRICE»/h) prices the run.
3. `train` — the run: t27a first; t27b after t27a's evaluation.

| Gate | Cell | What must hold |
|---|---|---|
| G1 | 1 | the kernel satisfies `host_lambda.PIN_SETS["«PIN_SET»"]`; fla and causal_conv1d import |
| G2 | 2 | pinned revision + mixture fingerprint; v0_2 schema; train split only; the arm's families; dev monitor disjoint from train, no reader-exposed reads |
| G3 | 3 | 0 LoRA tensors on vision, MTP, DeltaNet in_proj_a / in_proj_b; lora_A per group = the layer plan («EXPECTED» on the 27B) |
| G4 | 3 | the DeltaNet layers run the fla / causal-conv1d kernels, not the torch fallback |
| G5 | 4 | render contract on every row; 0 rows over «MAX_SEQ» tokens; answer supervised, prompt + empty think block masked |
| G6 | 5 | a non-zero gradient on every language lora_B, per group (DeltaNet / attention / MLP), none elsewhere |
| G7 | 8 | «DECODE_K» rows per family decoded from the PREPARED train dataloader equal the cell-4 tokens and masks |
| G8 | 8 | prepared-vs-direct loss within «TOL_PCT»; the warm-up survived `make_sft_config` |
| G9 | 6, 8 | rungs: timed steps, geometry sweep, peak VRAM, GO estimate; every fresh start: step-0 dev loss per family (the 2x alarm reference) |
| G10 | 8, 9 | rungs: kill at step «KILL», exact resume from the Hub copy, end-of-run save, merge test (rung 1: reload + loss, scratch push) |

**The pins are «PIN_STATUS».** `host_lambda.PIN_SETS["«PIN_SET»"]` holds the package specifications («PIN_LIST»).
G1 asserts the specifications at runtime. Rung 1 resolves placeholder ranges; its versions are recorded as exact pins
and this notebook regenerated before the run.

**Data pins.** The data cell refuses to run until the arm's dataset revision (40-hex) and mixture fingerprint (64-hex) are
pinned: push the local mixture, then `derive_t27.py --«PLACEHOLDER_ARM»-revision <sha> --«PLACEHOLDER_ARM»-mixture <the pushed file>`
regenerates both notebooks.

**W&B:** project `«WANDB»`, created as a private project before the first run (its visibility is set in W&B, not here);
never `«PUBLIC_WANDB»`.

**To resume after a session ends: run every cell again, unedited, with the same `T27_ARM` / `T27_STAGE`.** The notebook
finds the newest checkpoint on the Hub, checks the stored schedule and row order, and continues with the rows not yet
trained. The export cell merges and pushes only a finished run.
"""


def intro_md() -> str:
    """Markdown intro of the text27b notebook.

    :returns: Text of cell 0 (the Lambda edition puts its launch notes in front of it).
    """
    specs = pin_set()["specifiers"]
    listed = ", ".join(f"{d} {specs[d] or 'any'}" for d in pin_set()["kernel_gate"])
    return _fill(
        INTRO_MD, BASE=BASE_MODEL, RANK=str(LORA_RANK), ALPHA=str(LORA_ALPHA), MAX_SEQ=f"{MAX_SEQ:,}", ROWS=str(ROWS_PER_STEP),
        LR=LR, WARMUP_PCT=f"{float(WARMUP_RATIO):.0%}", WD=WEIGHT_DECAY, OPTIM=OPTIM, SEED=str(SEED), SAVE=str(SAVE_STEPS),
        EVAL=str(EVAL_STEPS), DATA_REPO=DATA_REPO, A_FAMILIES=", ".join(T27A.families), B_FAMILIES=", ".join(T27B.families),
        A_RUN=T27A.run_name, A_CKPT=T27A.ckpt_repo, A_MERGED=T27A.merged_repo, B_RUN=T27B.run_name, B_CKPT=T27B.ckpt_repo,
        B_MERGED=T27B.merged_repo, A_PUSH=str(T27A.push_every), B_PUSH=str(T27B.push_every), A_BUDGET=str(T27A.lambda_budget_h),
        B_BUDGET=str(T27B.lambda_budget_h), SMOKE_BASE=SMOKE_BASE_MODEL, LADDER=str(LADDER_STEPS), KILL=str(KILL_AT),
        TIMED=str(TIMED_STEPS), PRICE=f"{PRICE_PER_H}", PIN_SET=PIN_SET, EXPECTED=str(EXPECTED_LORA_A_27B),
        DECODE_K=str(DECODE_ROWS_PER_FAMILY), TOL_PCT=f"{LOSS_TOLERANCE:.0%}", PIN_LIST=listed, PLACEHOLDER_ARM="t27a",
        WANDB=WANDB_PROJECT, PUBLIC_WANDB=PUBLIC_WANDB_PROJECT, PIN_STATUS=str(pin_set()["status"]))


INSTALL_TEMPLATE = r'''# Cell 1 — installs + env + run knobs + G1 (pin gate)
# The text27b pin set (host_lambda.PIN_SETS["text27b"]) controls the package versions.
# G1 asserts its specifications at runtime; resolved smoke versions become exact pins before training.
import os
import time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
WANDB_PROJECT = «WANDB»   # private project; never the public «PUBLIC_WANDB» (its watcher follows the newest run)
assert WANDB_PROJECT != «PUBLIC_WANDB», "text27b runs must not log to the public project"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
# ---- run knobs (edit here, or set them in the environment of a headless run) ----
ARM = os.environ.get("T27_ARM", "t27a")          # t27a = translation pilot, t27b = joint run
STAGE = os.environ.get("T27_STAGE", "rung1")     # rung1 = 2B smoke, rung2 = 27B gates + timing, train = the run (each its own GO)
assert ARM in «ARMS», f"T27_ARM must be one of «ARMS», got {ARM!r}"
assert STAGE in «STAGES», f"T27_STAGE must be one of «STAGES», got {STAGE!r}"
SMOKE = STAGE == "rung1"                         # the smoke flag: the base becomes «SMOKE_BASE» (same qwen3_5 architecture)
SCRATCH_TAG = os.environ.get("T27_SCRATCH_TAG") or time.strftime("%m%d%H%M", time.gmtime())   # names a rung's scratch repos
ROWS_PER_STEP = «ROWS»                           # rows per optimizer step
PER_DEVICE_BATCH = int(os.environ.get("T27_BATCH", "1"))   # rung 2's geometry sweep may pick 2 or 4
assert PER_DEVICE_BATCH > 0 and ROWS_PER_STEP % PER_DEVICE_BATCH == 0, f"T27_BATCH must divide {ROWS_PER_STEP}"
GRAD_ACCUM = ROWS_PER_STEP // PER_DEVICE_BATCH
LADDER_STEPS, KILL_AT = «LADDER», «KILL»         # a rung: killed at step «KILL», resumed from the Hub to step «LADDER» (G10)
print(f"arm {ARM}, stage {STAGE}, batch {PER_DEVICE_BATCH} x {GRAD_ACCUM}" + (f", scratch tag {SCRATCH_TAG}" if STAGE != "train" else ""))
«INSTALL»
# ---- G1: the installed stack satisfies the text27b pin set ----
from importlib.metadata import version
from importlib.util import find_spec
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
T27_PIN_STATUS = «STATUS»
T27_KERNEL_PINS = «KERNEL_PINS»
«FUNCS»
_problems = pin_problems(T27_KERNEL_PINS, version)
assert not _problems, "G1 pin gate FAILED (this kernel is not the text27b stack):\n  - " + "\n  - ".join(_problems)
_missing = [_m for _m in ("fla", "causal_conv1d") if find_spec(_m) is None]
assert not _missing, f"G1: the Gated DeltaNet fast-path kernels are not importable: {_missing}"
RESOLVED_PINS = {_n: version(_n) for _n in T27_KERNEL_PINS}
print(f"G1 OK ({T27_PIN_STATUS} pin set): {RESOLVED_PINS}")
if T27_PIN_STATUS == "placeholder":
    print("   PLACEHOLDER ranges: write these resolved versions into host_lambda.PIN_SETS['text27b'] as exact pins, "
          "then regenerate this notebook with derive_t27.py")

import torch
«GPU»
'''

COLAB_GPU = 'assert torch.cuda.is_available(), "No GPU — switch runtime to A100"'
LAMBDA_GPU = ('assert torch.cuda.is_available(), "No GPU — this kernel does not see the instance\'s GPU"\n'
              '_p = torch.cuda.get_device_properties(0)\n'
              'print(f"GPU: {_p.name}, {_p.total_memory / 2**30:.0f} GiB")')


def install_lines(host: str) -> str:
    """The install part of cell 1.

    :param host: ``colab`` (``%pip`` from the pin set) or ``lambda`` (the venv of :data:`SETUP_SCRIPT`; nothing installed).
    :returns: Source lines.
    :raises ValueError: On an unknown host.
    """
    if host == "lambda":
        return (f'# Lambda: the kernel is the venv {SETUP_SCRIPT} built from host_lambda.PIN_SETS["{PIN_SET}"]; '
                "nothing is installed here")
    if host != "colab":
        raise ValueError(f"unknown host {host!r}")
    specs = pin_set()["specifiers"]
    main = [_requirement("unsloth", specs["unsloth"], "[colab-new]")]
    main += [_requirement(d, s) for d, s in specs.items() if d not in ("unsloth", "causal-conv1d")]
    lines = ["%pip install -q " + " ".join(main)]
    if "causal-conv1d" in specs:
        lines.append("%pip install -q --no-build-isolation " + _requirement("causal-conv1d", specs["causal-conv1d"]) +
                     "   # builds against the installed torch")
    return "\n".join(lines)


def install_cell(host: str) -> str:
    """Cell 1: installs (Colab) or the venv note (Lambda), env, run knobs and G1.

    :param host: ``colab`` or ``lambda``.
    :returns: Source of cell 1.
    """
    pins = pin_set()
    return _fill(INSTALL_TEMPLATE, WANDB=repr(WANDB_PROJECT), PUBLIC_WANDB=repr(PUBLIC_WANDB_PROJECT), ARMS=repr(tuple(ARMS)),
                 STAGES=repr(STAGES), SMOKE_BASE=SMOKE_BASE_MODEL, ROWS=str(ROWS_PER_STEP), LADDER=str(LADDER_STEPS),
                 KILL=str(KILL_AT), INSTALL=install_lines(host), STATUS=repr(pins["status"]), KERNEL_PINS=repr(kernel_pins()),
                 FUNCS=inline_source(pin_problems), GPU=LAMBDA_GPU if host == "lambda" else COLAB_GPU)


DATA_TEMPLATE = r'''# Cell 2 — data: the pinned text27b mixture (chat rows, MAP-STYLE, materialised order — no streaming) + G2 (data gate)
import json
from collections import Counter
from pathlib import Path
from google.colab import userdata
from huggingface_hub import login, snapshot_download

login(token=userdata.get("HF_TOKEN"))

DATA_REPO = «DATA_REPO»   # private: the local v0_2 mixtures (<arm>/train.jsonl + mixture_manifest.json) and the dev monitor
# revision of DATA_REPO per arm, and the text fingerprint of <arm>/train.jsonl in file order (derive_t27.text_fingerprint)
DATA_REVISIONS = «REVISIONS»
MIXTURE_FINGERPRINTS = «FINGERPRINTS»
ARM_FAMILIES = «ARM_FAMILIES»
ARM_REQUIRED = «ARM_REQUIRED»
FAMILIES = «FAMILIES»
CONDITIONS = «CONDITIONS»
BUILD = «BUILD»
DEV_FILE = «DEV_FILE»   # E3 dev monitor: all dev translate + 40 catalogue + 20 relations rows, reader-exposed reads removed
TRAIN_FILE, MANIFEST_FILE = f"{ARM}/train.jsonl", f"{ARM}/mixture_manifest.json"
«FUNCS»
# never train on an unpinned or unverified mixture: derive_t27.py writes both pins after the push
DATA_REVISION = check_pin(DATA_REVISIONS[ARM], 40, f"{ARM} dataset revision",
                          f"push the mixture, then regenerate this notebook with derive_t27.py --{ARM}-revision <sha> --{ARM}-mixture <file>")
MIXTURE_FINGERPRINT = check_pin(MIXTURE_FINGERPRINTS[ARM], 64, f"{ARM} mixture fingerprint",
                                f"regenerate this notebook with derive_t27.py --{ARM}-mixture <the pushed file>")
LOCAL = "/content/«LOCAL_LEAF»"
DATA_DIR = Path(snapshot_download(DATA_REPO, repo_type="dataset", revision=DATA_REVISION, local_dir=LOCAL,
                                  allow_patterns=[TRAIN_FILE, MANIFEST_FILE, DEV_FILE, "README.md"]))
train_rows = read_jsonl(DATA_DIR / TRAIN_FILE)
dev_rows = read_jsonl(DATA_DIR / DEV_FILE)
MIX = json.loads((DATA_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
print(f"{ARM}: {len(train_rows):,} train rows, {len(dev_rows):,} dev-monitor rows; mixture manifest keys {sorted(MIX)}")

# ---- G2: the pinned mixture (fingerprint), train split only, this arm's families, the v0_2 schema; a clean dev monitor ----
_fp = text_fingerprint(train_rows)
assert _fp == MIXTURE_FINGERPRINT, f"G2: mixture fingerprint {_fp[:16]}… != pinned {MIXTURE_FINGERPRINT[:16]}… — not the pushed file"
_bad = [(_i, _p) for _i, _r in enumerate(train_rows) for _p in row_problems(_r, "train", ARM_FAMILIES[ARM], CONDITIONS, BUILD)]
assert not _bad, f"G2: {len(_bad)} train-row problems, e.g. {_bad[:5]}"
_bad = [(_i, _p) for _i, _r in enumerate(dev_rows) for _p in row_problems(_r, "dev", FAMILIES, CONDITIONS, BUILD)]
assert not _bad, f"G2: {len(_bad)} dev-monitor problems, e.g. {_bad[:5]}"
assert not [_r for _r in dev_rows if _r["meta"].get("reader_exposed") is True], "G2: reader-exposed reads in the dev monitor"
_train_docs = {str(_r["meta"]["pgpid"]) for _r in train_rows if _r["meta"].get("pgpid") not in (None, "")}
_shared = _train_docs & {str(_r["meta"]["pgpid"]) for _r in dev_rows if _r["meta"].get("pgpid") not in (None, "")}
assert not _shared, f"G2: {len(_shared)} documents in both train and the dev monitor, e.g. {sorted(_shared)[:5]}"
_shared = {_r["meta"]["row_id"] for _r in train_rows} & {_r["meta"]["row_id"] for _r in dev_rows}
assert not _shared, f"G2: {len(_shared)} row ids in both train and the dev monitor"
_fam = Counter(_r["meta"]["family"] for _r in train_rows)
assert set(ARM_REQUIRED[ARM]) <= set(_fam), f"G2: {ARM} needs {ARM_REQUIRED[ARM]}, the mixture has {sorted(_fam)}"
assert len(train_rows) >= ROWS_PER_STEP * LADDER_STEPS, "G2: fewer rows than a ladder rung needs"
_cond = Counter((_r["meta"]["family"], _r["meta"]["condition"]) for _r in train_rows)
print("G2 OK: train rows by family", dict(_fam))
print("   by family x condition", dict(sorted(_cond.items())))
print("   by language", dict(Counter(_r["meta"].get("lang_class") for _r in train_rows)))
print("   dev monitor by family", dict(Counter(_r["meta"]["family"] for _r in dev_rows)))
'''


def data_cell(pins: Pins) -> str:
    """Cell 2: the pinned mixture and G2.

    :param pins: The pinned revisions and fingerprints.
    :returns: Source of cell 2.
    """
    return _fill(DATA_TEMPLATE, DATA_REPO=repr(DATA_REPO), REVISIONS=repr(dict(pins.revisions)),
                 FINGERPRINTS=repr(dict(pins.fingerprints)), ARM_FAMILIES=repr({a.name: a.families for a in ARMS.values()}),
                 ARM_REQUIRED=repr({a.name: a.required for a in ARMS.values()}), FAMILIES=repr(tuple(t27rows.FAMILIES)),
                 CONDITIONS=repr(tuple(t27rows.CONDITIONS)), BUILD=repr(t27rows.BUILD), DEV_FILE=repr(DEV_FILE),
                 FUNCS="from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple\n"
                       + inline_source(check_pin, read_jsonl, text_fingerprint, row_problems),
                 LOCAL_LEAF=LOCAL_LEAF)


MODEL_TEMPLATE = r'''# Cell 3 — model: Qwen3.8-27B (Unsloth bnb 4-bit; Qwen3.5-2B on rung 1) + language-only LoRA + G3 (adapter gate) + G4 (kernel fast path)
import inspect as _inspect
import re
from unsloth import FastModel

BASE_MODEL = «SMOKE_BASE» if SMOKE else «BASE»
MAX_SEQ = «MAX_SEQ»
LORA_RANK, LORA_ALPHA = «RANK», «ALPHA»
# language layers only: DeltaNet in_proj_qkv / in_proj_z / out_proj, attention q / k / v / o, MLP gate / up / down; never the
# vision tower, the MTP head or DeltaNet in_proj_a / in_proj_b (PEFT full-matches this regex on every module name)
LORA_REGEX = «REGEX»
EXPECTED_LORA_A = None if SMOKE else «EXPECTED»   # 27B: 48 DeltaNet x 3 + 16 attention x 4 + 64 MLP x 3 (rung 1: from the config)

model, tokenizer = FastModel.from_pretrained(BASE_MODEL, load_in_4bit=True, max_seq_length=MAX_SEQ,
                                             use_gradient_checkpointing="unsloth", full_finetuning=False)
model = FastModel.get_peft_model(
    model, target_modules=LORA_REGEX,
    finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none", random_state=«SEED»,
    use_gradient_checkpointing="unsloth",
)
«FUNCS»

# ---- G3: a language-only adapter — 0 vision / MTP / in_proj_a|b LoRA tensors, the planned lora_A count per module group ----
_text_cfg = getattr(model.config, "text_config", None) or model.config
LAYER_TYPES = list(_text_cfg.layer_types)
assert not getattr(_text_cfg, "num_experts", None), "G3: a MoE config — the expected count assumes dense MLPs"
_lora_a = [_n for _n, _ in model.named_parameters() if "lora_A" in _n]
LORA_COUNTS, _problems = lora_problems(_lora_a, LAYER_TYPES, EXPECTED_LORA_A)
if _problems:
    print("linear_attn modules found:", sorted({re.sub(r"\.\d+\.", ".N.", _n) for _n, _ in model.named_modules() if ".linear_attn." in _n})[:20])
assert not _problems, "G3 adapter gate FAILED:\n  - " + "\n  - ".join(_problems)
_trainable_other = [_n for _n, _p in model.named_parameters() if _p.requires_grad and "lora_" not in _n]
assert not _trainable_other, f"G3: non-LoRA parameters are trainable: {_trainable_other[:5]}"
print(f"G3 OK: {sum(LORA_COUNTS[_g] for _g in ('deltanet', 'attention', 'mlp'))} language lora_A tensors {LORA_COUNTS} on {len(LAYER_TYPES)} layers")

# ---- G4: the Gated DeltaNet layers run the fla / causal-conv1d kernels, not the torch fallback ----
REQUIRE_FAST_PATH = True
_lin = [_m for _n, _m in model.named_modules() if _n.endswith(".linear_attn")]
assert _lin, "G4: no linear_attn modules — not a qwen3_5 model?"
_modeling_module = _inspect.getmodule(type(_lin[0]))
from transformers.integrations.hub_kernels import use_kernel_func_from_hub_with_fallback
if repair_recurrent_kernel(_modeling_module, use_kernel_func_from_hub_with_fallback):
    print("Rebound the recurrent DeltaNet wrapper to fused_recurrent_gated_delta_rule; G4 verifies its implementation")
_problems = fast_path_problems(_lin[0], getattr(_modeling_module, "is_fast_path_available", None), _modeling_module)
if _problems and REQUIRE_FAST_PATH:
    raise AssertionError("G4 kernel fast-path gate FAILED:\n  - " + "\n  - ".join(_problems))
print("G4 OK: the DeltaNet layers run the fla / causal-conv1d kernels" if not _problems else f"⚠️ G4 fast path missing: {_problems}")
print(f"base {BASE_MODEL}; trainable {sum(_p.numel() for _p in model.parameters() if _p.requires_grad) / 1e6:.1f}M parameters")
'''


def model_cell() -> str:
    """Cell 3: model, language-only LoRA, G3 and G4.

    :returns: Source of cell 3.
    """
    return _fill(MODEL_TEMPLATE, SMOKE_BASE=repr(SMOKE_BASE_MODEL), BASE=repr(BASE_MODEL), MAX_SEQ=str(MAX_SEQ),
                 RANK=str(LORA_RANK), ALPHA=str(LORA_ALPHA), REGEX=_raw_literal(LORA_REGEX), EXPECTED=str(EXPECTED_LORA_A_27B),
                 SEED=str(SEED), FUNCS=inline_source(module_group, expected_lora_counts, lora_problems, repair_recurrent_kernel, fast_path_problems))


COLLATOR_TEMPLATE = r'''# Cell 4 — render (enable_thinking=False) + tokenize once + response-only labels + collator + G5 (render, length, mask)
import hashlib
import unicodedata
import numpy as np
from torch.utils.data import Dataset as _TorchDataset

_tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer   # the text tokenizer under a processor
CHAT_TEMPLATE = getattr(_tok, "chat_template", None) or getattr(tokenizer, "chat_template", None)
assert CHAT_TEMPLATE and "enable_thinking" in CHAT_TEMPLATE, "G5: the chat template has no enable_thinking switch"
TEMPLATE_SHA256 = hashlib.sha256(CHAT_TEMPLATE.encode("utf-8")).hexdigest()
# sha256 of the local template rung 0 checked: informational (the render contract below is the gate)
LOCAL_TEMPLATE_SHA256 = «LOCAL_SHA»
print(f"chat template sha256 {TEMPLATE_SHA256[:16]}…" + ("" if LOCAL_TEMPLATE_SHA256 is None else
      (" = the local copy's" if TEMPLATE_SHA256 == LOCAL_TEMPLATE_SHA256 else f" != the local copy's {LOCAL_TEMPLATE_SHA256[:16]}…")))
THINK_IDS = [_tok.convert_tokens_to_ids(_t) for _t in ("<think>", "</think>")]
PAD_ID = _tok.pad_token_id if _tok.pad_token_id is not None else _tok.eos_token_id
«FUNCS»


def render(messages):
    """The training render of a row: the chat template with enable_thinking=False (no generation prompt), NFC text
    (the tokenizer normalises to NFC; pass nfc_messages(...) so the answer compares equal)."""
    return unicodedata.normalize("NFC", _tok.apply_chat_template(list(messages), chat_template=CHAT_TEMPLATE,
                                                                 tokenize=False, enable_thinking=False))


def decode(ids):
    """Lossless decode (special tokens kept, no clean-up)."""
    return _tok.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


class TokenizedRows(_TorchDataset):
    """Rows rendered and tokenized once (G5 render contract on every row); labels mask the prompt prefix."""

    def __init__(self, rows):
        self.rows, self.meta, self.ids, self.n_prompt = rows, [r["meta"] for r in rows], [], []
        for k, row in enumerate(rows):
            msgs = nfc_messages(row["messages"])
            text = render(msgs)
            problems = render_problems(text, msgs)
            assert not problems, f"G5 render contract FAILED on row {k} ({row['meta'].get('row_id')}): {problems}"
            prompt, _ = split_rendered(text, msgs[-1]["content"])
            enc = _tok(text, add_special_tokens=False, return_offsets_mapping=True)
            self.n_prompt.append(prompt_token_count(enc["offset_mapping"], len(prompt)))
            self.ids.append(np.asarray(enc["input_ids"], dtype=np.int32))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        ids = torch.from_numpy(self.ids[i].astype(np.int64))
        labels = ids.clone()
        labels[: self.n_prompt[i]] = -100
        return {"input_ids": ids, "labels": labels}

    def column(self, name):
        return [m.get(name) for m in self.meta]

    def parts(self, i):
        """(prompt, completion) text of row i (NFC, as tokenized)."""
        msgs = nfc_messages(self.rows[i]["messages"])
        return split_rendered(render(msgs), msgs[-1]["content"])


def collate(items):
    """Right-padded batch: input_ids, attention_mask, labels (-100 on the prompt and the padding)."""
    n = max(len(it["input_ids"]) for it in items)
    ids = torch.full((len(items), n), PAD_ID, dtype=torch.long)
    labels = torch.full((len(items), n), -100, dtype=torch.long)
    mask = torch.zeros((len(items), n), dtype=torch.long)
    for k, it in enumerate(items):
        m = len(it["input_ids"])
        ids[k, :m], labels[k, :m], mask[k, :m] = it["input_ids"], it["labels"], 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


train_ds = TokenizedRows(train_rows)
dev_sets = {_f: TokenizedRows([_r for _r in dev_rows if _r["meta"]["family"] == _f])
            for _f in sorted({_r["meta"]["family"] for _r in dev_rows})}   # one eval set per family: eval_<family>_loss

# ---- G5: 0 rows over MAX_SEQ (nothing is truncated); the mask guardrail on 2 rows per family ----
_lens = np.array([len(_x) for _x in train_ds.ids])
_over = Counter(_m["family"] for _m, _n in zip(train_ds.meta, _lens) if _n > MAX_SEQ)
_over_dev = Counter(_f for _f, _d in dev_sets.items() for _x in _d.ids if len(_x) > MAX_SEQ)
assert not _over and not _over_dev, f"G5: rows over MAX_SEQ={MAX_SEQ}: train {dict(_over)}, dev {dict(_over_dev)} — fix the mixture"
_guard = {}
for _i, _m in enumerate(train_ds.meta):
    if len(_guard.setdefault(_m["family"], [])) < 2:
        _guard[_m["family"]].append(_i)
for _f, _idx in _guard.items():
    for _i in _idx:
        _item = train_ds[_i]
        _prompt, _completion = train_ds.parts(_i)
        _p = supervised_problems(_item["input_ids"].tolist(), _item["labels"].tolist(), train_ds.n_prompt[_i], _prompt, _completion,
                                 decode, THINK_IDS)
        assert not _p, f"G5 mask guardrail FAILED on {_f} row {_i}: {_p}"
_sup = int(sum(len(_x) - _n for _x, _n in zip(train_ds.ids, train_ds.n_prompt)))
print(f"G5 OK: {len(train_ds):,} rows rendered with enable_thinking=False (one empty think block each, no effort sentence); "
      f"longest {int(_lens.max()):,} tokens (<= {MAX_SEQ:,}); {int(_lens.sum()):,} tokens, {_sup:,} supervised; "
      f"mask guardrail on {sum(len(_v) for _v in _guard.values())} rows; dev monitor {sum(len(_d) for _d in dev_sets.values())} rows "
      f"in {len(dev_sets)} families")
'''


def collator_cell(pins: Pins) -> str:
    """Cell 4: render, tokenize, collator and G5.

    :param pins: Supplies the local template's sha256.
    :returns: Source of cell 4.
    """
    return _fill(COLLATOR_TEMPLATE, LOCAL_SHA=repr(pins.template_sha256),
                 FUNCS=inline_source(nfc_messages, render_problems, split_rendered, prompt_token_count, supervised_problems))


GRAD_TEMPLATE = r'''# Cell 5 — G6 gradient flow per module group (cheap; run BEFORE burning GPU-hours)
«FUNCS»
FastModel.for_training(model)
_vb = {_k: _v.to(model.device) for _k, _v in collate([train_ds[0]]).items()}
model(**_vb).loss.backward()
_entries = [(_n, (_p.grad.detach().float().norm().item() if _p.grad is not None else None))
            for _n, _p in model.named_parameters() if "lora_B" in _n]
model.zero_grad(set_to_none=True)
GRAD_NORMS, _problems = grad_group_problems(_entries, expected_lora_counts(LAYER_TYPES))
assert not _problems, "G6 gradient gate FAILED:\n  - " + "\n  - ".join(_problems)
print("G6 OK: lora_B gradients " + ", ".join(f"{_g} {_n} tensors |g|={_v:.2e}" for _g, (_n, _v) in sorted(GRAD_NORMS.items())))
del _vb
'''


def grad_cell() -> str:
    """Cell 5: G6.

    :returns: Source of cell 5.
    """
    return _fill(GRAD_TEMPLATE, FUNCS=inline_source(grad_group_problems))


BENCH_TEMPLATE = r'''# Cell 6 — G9 throughput on the ladder rungs: timed steps (the worst-case batch first), batch-geometry sweep, peak VRAM, GO estimate
import statistics
RUN_BENCH = STAGE != "train"
TIMED_STEPS = «TIMED» if STAGE == "rung2" else «SMOKE_TIMED»
BATCH_GEOMETRIES = «GEOMETRIES»   # (per-device batch, accumulation): «ROWS» rows a step each
PRICE_PER_H = «PRICE»             # A100 40 GB on demand (plan section 4): the GO number = measured s/step x steps x price
MERGE_H = «MERGE_H»               # merge + push hours after training, per arm
«FUNCS»


def timed_step(rows, per_device, accum):
    """Seconds of one optimizer step's forward + backward passes over ``rows`` (no optimizer update)."""
    torch.cuda.synchronize()
    t0 = time.time()
    for k in range(accum):
        b = {key: v.to(model.device) for key, v in collate([train_ds[i] for i in rows[k * per_device:(k + 1) * per_device]]).items()}
        (model(**b).loss / accum).backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return time.time() - t0


if RUN_BENCH:
    FastModel.for_training(model)
    _lens = [len(_x) for _x in train_ds.ids]
    WORST = sorted(range(len(train_ds)), key=lambda _i: -_lens[_i])[:ROWS_PER_STEP]   # the longest rows: the worst-case step
    timed_step(WORST[:1], 1, 1)                                                          # warm-up: kernel autotune, allocator
    BENCH = {}
    for _bs, _ga in BATCH_GEOMETRIES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            BENCH[(_bs, _ga)] = timed_step(WORST, _bs, _ga)
            print(f"batch {_bs}x{_ga}: worst-case step {BENCH[(_bs, _ga)]:.1f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        except torch.cuda.OutOfMemoryError:   # an expected outcome of the sweep: that geometry does not fit
            BENCH[(_bs, _ga)] = None
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"batch {_bs}x{_ga}: OOM")
    BEST_GEOMETRY = pick_geometry(BENCH)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _n = len(train_ds)
    _times = [timed_step(WORST if _s == 0 else [((_s - 1) * ROWS_PER_STEP + _j) % _n for _j in range(ROWS_PER_STEP)],
                         PER_DEVICE_BATCH, GRAD_ACCUM) for _s in range(TIMED_STEPS)]
    PEAK_GIB = torch.cuda.max_memory_allocated() / 2**30
    S_PER_STEP = statistics.median(_times[1:]) if len(_times) > 1 else _times[0]
    _steps = _n // ROWS_PER_STEP
    GO = go_estimate(S_PER_STEP, _steps, PRICE_PER_H, MERGE_H[ARM])
    print(f"G9: {TIMED_STEPS} timed steps at {PER_DEVICE_BATCH}x{GRAD_ACCUM}: worst-case {_times[0]:.1f} s, median {S_PER_STEP:.1f} s, "
          f"peak VRAM {PEAK_GIB:.1f} GiB; fastest geometry {BEST_GEOMETRY}"
          + ("" if BEST_GEOMETRY == (PER_DEVICE_BATCH, GRAD_ACCUM) else f" — set T27_BATCH={BEST_GEOMETRY[0]} for the run"))
    print(f"GO estimate for {ARM}: {_steps:,} steps x {S_PER_STEP:.1f} s = {GO['train_h']} h + {MERGE_H[ARM]} h merge/push "
          f"= {GO['total_h']} h, about ${GO['usd']} at ${PRICE_PER_H}/h")
else:
    print("G9 timing skipped on the run (rung 2 measured it; W&B times every step)")
'''


def bench_cell() -> str:
    """Cell 6: G9 timing on the rungs.

    :returns: Source of cell 6.
    """
    return _fill(BENCH_TEMPLATE, TIMED=str(TIMED_STEPS), SMOKE_TIMED=str(SMOKE_TIMED_STEPS), GEOMETRIES=repr(list(BATCH_GEOMETRIES)),
                 ROWS=str(ROWS_PER_STEP), PRICE=repr(PRICE_PER_H), MERGE_H=repr({a.name: a.merge_hours for a in ARMS.values()}),
                 FUNCS=inline_source(pick_geometry, go_estimate))


HELPER_MAX_STEPS = ("MAX_STEPS = LADDER_STEPS if STAGE != \"train\" else len(train_ds) // ROWS_PER_STEP   "
                    "# t27: one pass over the pinned mixture in its file order (a remainder of < 16 rows is not trained); a rung: 15")
HELPER_COLAB_BUDGET = (f"TIME_BUDGET_H = {COLAB_BUDGET_H}   # Colab: one session (runtimes end at ~8 h); the run resumes exactly")


def lambda_budget_line() -> str:
    """The helper cell's time-budget line in the Lambda edition.

    :returns: ``TIME_BUDGET_H = ...`` per arm for the run, :data:`LADDER_BUDGET_H` for a rung.
    """
    budgets = {a.name: a.lambda_budget_h for a in ARMS.values()}
    hours = " / ".join(f"{a.plan_hours[0]}-{a.plan_hours[1]}" for a in ARMS.values())
    return (f"TIME_BUDGET_H = {budgets!r}[ARM] if STAGE == \"train\" else {LADDER_BUDGET_H}   # Lambda: the plan's "
            f"{hours} h x {BUDGET_FACTOR} for a run, {LADDER_BUDGET_H} h for a rung; a safety stop only, max_steps ends the run")


def helper_cell(v25a_helpers: str) -> str:
    """Cell 7: v2.5a's inlined session helpers, with the run length and the session budget of this notebook.

    :param v25a_helpers: Source of cell 7 of the v2.5a Colab notebook.
    :returns: Source of cell 7.
    :raises ValueError: When the two lines are not found exactly once.
    """
    c7 = _replace_line(v25a_helpers, r"^MAX_STEPS = \d+", HELPER_MAX_STEPS, "max steps")
    return _replace_line(c7, r"^TIME_BUDGET_H = ", HELPER_COLAB_BUDGET, "time budget")


# anchors of the v2.5a training cell's blocks that carry over
FACTORIES_START = "def make_sft_config(**kw):\n"
RESUME_HEADER = "# ---- v22b: where to resume from (helpers cell) ----\n"
RESUME_ROOT_LINE = 'RESUME_ROOT = "/content/'
RECORDING_START = "class _Recording(_TDataset):\n"
RECORDING_END = "\n_rec = _Recording(train_ds)\n"
MILESTONE_START = "class MilestonePushCallback(TrainerCallback):\n"
MILESTONE_END = "\n\n_milestones = MilestonePushCallback("
#: the planned edits of the milestone callback: a set of milestone steps instead of a fixed interval
MILESTONE_EDITS: Tuple[Tuple[str, str], ...] = (
    ("    def __init__(self, trainer, ckpt_repo, out_dir, every):\n",
     "    def __init__(self, trainer, ckpt_repo, out_dir, steps):\n"),
    ("self.trainer, self.ckpt_repo, self.out_dir, self.every, self.confirmed = trainer, ckpt_repo, out_dir, int(every), []",
     "self.trainer, self.ckpt_repo, self.out_dir, self.steps, self.confirmed = trainer, ckpt_repo, out_dir, {int(s) for s in steps}, []"),
    ("        if step <= 0 or step % self.every or step in self.confirmed:\n",
     "        if step not in self.steps or step in self.confirmed:\n"),
)


def kept_blocks(v25a_train: str) -> Dict[str, str]:
    """The blocks of v2.5a's training cell that carry over (asserted anchors).

    :param v25a_train: Source of cell 8 of the v2.5a Colab notebook.
    :returns: ``factories`` (``make_sft_config`` + ``make_trainer``), ``resume_comment`` (the resume-root rationale),
        ``recording`` (the prepared-dataloader recording wrapper), ``milestone`` (``MilestonePushCallback``, as in v2.5a).
    :raises ValueError: When an anchor is missing or ambiguous.
    """
    return {
        "factories": _extract(v25a_train, FACTORIES_START, RESUME_HEADER, "trainer factories").rstrip("\n") + "\n",
        "resume_comment": _extract(v25a_train, RESUME_HEADER, RESUME_ROOT_LINE, "resume comment"),
        "recording": _extract(v25a_train, RECORDING_START, RECORDING_END, "recording wrapper").rstrip("\n") + "\n",
        "milestone": _extract(v25a_train, MILESTONE_START, MILESTONE_END, "milestone callback").rstrip("\n") + "\n",
    }


def milestone_class(v25a_milestone: str) -> str:
    """v2.5a's ``MilestonePushCallback`` with a set of milestone steps (the planned edits only).

    :param v25a_milestone: The class as :func:`kept_blocks` returns it.
    :returns: The edited class source.
    """
    out = v25a_milestone
    for old, new in MILESTONE_EDITS:
        out = _replace_once(out, old, new, "milestone callback")
    return out


def _runs_literal() -> str:
    """The ``RUNS`` table of the training cell.

    :returns: Source lines of the dict entries.
    """
    lines = []
    for a in ARMS.values():
        lines.append(f'    "{a.name}": dict(run_name="{a.run_name}", ckpt_repo="{a.ckpt_repo}", merged_repo="{a.merged_repo}",\n'
                     f'                 push_every={a.push_every}, milestone_fractions={a.milestone_fractions!r}),')
    return "\n".join(lines)


TRAIN_TEMPLATE = r'''# Cell 8 — text27b training (t27a / t27b, or a ladder rung): the pinned mixture in file order, exact resume, G7 + G8 + G10
import inspect
import wandb
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import Dataset as _TDataset
from trl import SFTConfig, SFTTrainer
from unsloth import is_bf16_supported
«FUNCS»

RUNS = {
«RUNS»
}
RUN = RUNS[ARM]
if STAGE == "train":
    RUN_NAME, CKPT_REPO, MERGED_REPO = RUN["run_name"], RUN["ckpt_repo"], RUN["merged_repo"]   # NEW repos, private
    SAVE_STEPS, EVAL_STEPS, PUSH_EVERY, MILESTONE_FRACTIONS = «SAVE», «EVAL», RUN["push_every"], RUN["milestone_fractions"]
else:   # a ladder rung: scratch repos (private; delete them after the ladder), saves every KILL_AT steps
    _base = "qwen3.5-2b" if SMOKE else "qwen3.8-27b"
    RUN_NAME = f"genizah_t27_{STAGE}_{SCRATCH_TAG}"
    CKPT_REPO = f"isaacmg/{_base}-genizah-t27-{STAGE}-{SCRATCH_TAG}-ckpt"
    MERGED_REPO = f"isaacmg/{_base}-genizah-t27-{STAGE}-{SCRATCH_TAG}-merged"   # rung 1 pushes its merge test here
    SAVE_STEPS, EVAL_STEPS, PUSH_EVERY, MILESTONE_FRACTIONS = KILL_AT, KILL_AT, KILL_AT, ()
OUT_DIR = f"outputs_{RUN_NAME}"
wandb.login(key=userdata.get("WANDB_API_KEY"))

«FACTORIES»
«RESUME_COMMENT»RESUME_ROOT = "/content/t27_resume"
RESUME_ROOT = os.path.join(RESUME_ROOT, RUN_NAME)   # one folder per run, outside OUT_DIR
resume_dir, _resume_step = discover_resume(CKPT_REPO, OUT_DIR, RESUME_ROOT)
print(f"resume point: {resume_dir} (step {_resume_step})" if resume_dir else f"no checkpoint on the Hub or on this machine: fresh start of {RUN_NAME}")
assert STAGE == "train" or resume_dir is None, f"a ladder rung starts from scratch, but {CKPT_REPO} has step {_resume_step}: re-run cell 1 (a new T27_SCRATCH_TAG)"
# transformers 5 may fold the warm-up ratio into warmup_steps (a float below 1); the schedule check below holds either way
_WARMUP = {"warmup_ratio": «WARMUP»} if "warmup_ratio" in inspect.signature(SFTConfig.__init__).parameters else {"warmup_steps": «WARMUP»}


def sft_args(out_dir):
    """The run's SFTConfig, in one place: G10's resumed trainer is built exactly like the first."""
    return make_sft_config(
        per_device_train_batch_size=PER_DEVICE_BATCH, gradient_accumulation_steps=GRAD_ACCUM,   # «ROWS» rows a step
        dataloader_num_workers=2, dataloader_pin_memory=True,
        max_steps=MAX_STEPS,              # one pass over the mixture; sessions end on the time budget and resume exactly
        learning_rate=«LR», lr_scheduler_type="cosine", weight_decay=«WD», **_WARMUP,   # the resume gate knows cosine only
        logging_steps=1 if STAGE != "train" else 10,
        eval_strategy="steps", eval_steps=EVAL_STEPS, per_device_eval_batch_size=1,
        save_steps=SAVE_STEPS, save_total_limit=2,   # a termination costs at most SAVE_STEPS steps (the Hub follows every save)
        push_to_hub=True, hub_model_id=CKPT_REPO,
        hub_strategy="checkpoint", hub_private_repo=True,   # private: dataset, checkpoints and W&B stay private
        hub_always_push=True,             # every save is pushed, also the time-budget stop that lands right after a regular save
        optim=«OPTIM», seed=«SEED», data_seed=«SEED», output_dir=out_dir,
        report_to="wandb", run_name=RUN_NAME,
        bf16=is_bf16_supported(), fp16=not is_bf16_supported(),
        remove_unused_columns=False, dataset_text_field="",
        accelerator_config={"dispatch_batches": False},   # map-style rows: Accelerate does not dispatch, the v2.1 rule stays gated
        dataset_kwargs={"skip_prepare_dataset": True},     # rows are rendered and tokenized in cell 4 (prompt masked)
        max_seq_length=MAX_SEQ,
    )


FastModel.for_training(model)
from datasets import Dataset as _HFDataset


def trl_dataset(rows: TokenizedRows) -> _HFDataset:
    """Expose pretokenized rows through TRL 1.15's supported dataset interface.

    :param rows: Tokenized rows in their original order, with response-only labels.
    :returns: A map-style Hugging Face dataset yielding the same tensors.
    """
    return _HFDataset.from_dict({
        "input_ids": [row.tolist() for row in rows.ids],
        "labels": [rows[i]["labels"].tolist() for i in range(len(rows))],
    }).with_format("torch")


_trl_train_ds = trl_dataset(train_ds)
_trl_dev_sets = {family: trl_dataset(rows) for family, rows in dev_sets.items()}
trainer = make_trainer(model=model, tokenizer=_tok, data_collator=collate, train_dataset=_trl_train_ds, eval_dataset=_trl_dev_sets,
                       args=sft_args(OUT_DIR))
# schedule check (G8): make_sft_config drops kwargs this TRL does not know — the warm-up and the cosine must have survived
assert trainer.args.get_warmup_steps(MAX_STEPS) == warmup_steps_for(MAX_STEPS, «WARMUP»), (
    f"warm-up {trainer.args.get_warmup_steps(MAX_STEPS)} steps != planned {warmup_steps_for(MAX_STEPS, «WARMUP»)}")
assert str(getattr(trainer.args.lr_scheduler_type, "value", trainer.args.lr_scheduler_type)) == "cosine", "not a cosine schedule"
wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config={
    "arm": ARM, "stage": STAGE, "resume_step": _resume_step, "base_model": BASE_MODEL, "data_repo": DATA_REPO,
    "data_revision": DATA_REVISION, "mixture_fingerprint": MIXTURE_FINGERPRINT, "pin_status": T27_PIN_STATUS,
    "resolved_pins": RESOLVED_PINS, "chat_template_sha256": TEMPLATE_SHA256, "lora_regex": LORA_REGEX,
    "lora_counts": LORA_COUNTS, "batch": [PER_DEVICE_BATCH, GRAD_ACCUM], "max_steps": MAX_STEPS, "mixture": MIX}, allow_val_change=True)

# ---- G7: the PREPARED train dataloader (v2.1 lesson) hands out the planned rows: «DECODE_K» rows per family decoded ----
assert trainer.args.accelerator_config.dispatch_batches is False, "dispatch_batches=False did not reach the trainer"

«RECORDING»
_rec = _Recording(train_ds)
trainer.train_dataset = _rec
_saved_workers, trainer.args.dataloader_num_workers = trainer.args.dataloader_num_workers, 0  # in-process fetch ⇒ indices visible
_dl = trainer.get_train_dataloader()
assert "Dispatcher" not in type(_dl).__name__, f"train dataloader is {type(_dl).__name__} — batches would be truncated"
_want = family_quota(Counter(train_ds.column("family")), «DECODE_K»)
_got, _pos, _first = Counter(), 0, None
for _b in _dl:
    if _first is None:
        _first, _first_rows = _b, list(_rec.seen[:_b["input_ids"].shape[0]])
    for _j in range(_b["input_ids"].shape[0]):
        _i = _rec.seen[_pos]
        _pos += 1
        _fam = train_ds.meta[_i]["family"]
        if _got[_fam] >= _want[_fam]:
            continue
        _prompt, _completion = train_ds.parts(_i)
        _p = prepared_row_problems(_b["input_ids"][_j].tolist(), _b["labels"][_j].tolist(), _b["attention_mask"][_j].tolist(),
                                   train_ds.ids[_i].tolist(), train_ds.n_prompt[_i], _prompt, _completion, decode, THINK_IDS)
        assert not _p, f"G7 FAILED on prepared row {_i} ({_fam}): {_p}"
        _got[_fam] += 1
    if all(_got[_f] >= _n for _f, _n in _want.items()):
        break
assert all(_got[_f] >= _n for _f, _n in _want.items()), f"G7: decoded {dict(_got)}, wanted {_want}"
print(f"G7 OK: {sum(_got.values())} prepared rows {dict(_got)} = the cell-4 tokens; prompt + empty think block masked, answer supervised")
# ---- G8: the loss on the first prepared batch equals the loss on the same rows collated directly ----
_direct = {_k: _v.to(model.device) for _k, _v in collate([train_ds[_i] for _i in _first_rows]).items()}
assert torch.equal(_direct["input_ids"], _first["input_ids"].to(model.device)), "G8: prepared batch 0 is not the recorded rows"
model.eval()
with torch.no_grad():
    _lp = model(**{_k: _v.to(model.device) for _k, _v in _first.items()}).loss.item()
    _ld = model(**_direct).loss.item()
assert abs(_lp - _ld) <= «TOL» * max(1.0, _ld), f"G8: prepared-dataloader loss {_lp:.3f} != direct-collate loss {_ld:.3f}"
print(f"G8 OK: loss prepared={_lp:.3f} direct={_ld:.3f} (rows {_first_rows}); warm-up {trainer.args.get_warmup_steps(MAX_STEPS)} steps")
trainer.train_dataset = train_ds
trainer.args.dataloader_num_workers = _saved_workers
del _dl, _b, _first, _direct, _rec
FastModel.for_training(model)

# ---- G9 (every fresh start): the step-0 dev loss per family is the 2x loss-alarm reference ----
STEP0_FILE = "step0_dev_loss.json"
if resume_dir is None:
    _m0 = trainer.evaluate()
    STEP0_DEV_LOSS = {_k[len("eval_"):-len("_loss")]: float(_v) for _k, _v in _m0.items() if _k.startswith("eval_") and _k.endswith("_loss")}
    FastModel.for_training(model)
else:
    with open(hf_hub_download(CKPT_REPO, STEP0_FILE), encoding="utf-8") as _fh:
        STEP0_DEV_LOSS = json.load(_fh)
print("step-0 dev loss per family:", {_k: round(_v, 3) for _k, _v in sorted(STEP0_DEV_LOSS.items())})

# ---- v22b: schedule lock, fixed row order, exact resume (helpers cell) — t27: the order is the mixture file's own ----
_a = trainer.args
_order = list(range(len(train_ds))) * epochs_for(MAX_STEPS, len(train_ds), _a.per_device_train_batch_size, _a.gradient_accumulation_steps)
_plan = build_session_plan(
    _order, order_description="mixture file order", n_rows=len(train_ds), max_steps=_a.max_steps,
    per_device_batch=_a.per_device_train_batch_size, grad_accum=_a.gradient_accumulation_steps,
    learning_rate=_a.learning_rate, warmup_ratio=«WARMUP», lr_scheduler_type=str(getattr(_a.lr_scheduler_type, "value", _a.lr_scheduler_type)),
    weight_decay=_a.weight_decay, optim=str(getattr(_a.optim, "value", _a.optim)), seed=_a.seed,
    dataset_repo=DATA_REPO, dataset_revision=DATA_REVISION, rows_sha256=MIXTURE_FINGERPRINT, base_model=BASE_MODEL,
    lora=f"r{LORA_RANK}/alpha{LORA_ALPHA}/{LORA_REGEX}", max_seq=MAX_SEQ, render="enable_thinking=False")
_order = lock_session_plan(CKPT_REPO, _plan, _order, has_checkpoint=resume_dir is not None)   # first launch stores it; later sessions must match it
if resume_dir is None:
    HfApi().upload_file(path_or_fileobj=json.dumps(STEP0_DEV_LOSS).encode("utf-8"), path_in_repo=STEP0_FILE, repo_id=CKPT_REPO,
                        commit_message="Step-0 dev loss per family (2x loss-alarm reference)")
print(describe_session(len(train_ds), MAX_STEPS, _a.per_device_train_batch_size, _a.gradient_accumulation_steps, _resume_step, _a.seed))

# ---- milestones: pushed AND confirmed on the Hub before training goes on (the Studio evals; t27b's 50 % / 100 % comparison) ----
MILESTONES = milestone_steps(MAX_STEPS, SAVE_STEPS, PUSH_EVERY, MILESTONE_FRACTIONS)
print(f"Hub-confirmed milestones: {MILESTONES}")


«MILESTONE»

class LossAlarmCallback(TrainerCallback):
    """Alarm (print + W&B alert, training goes on) when a loss crosses 2x its step-0 dev reference."""

    def __init__(self, reference, families, factor):
        super().__init__()
        self.reference, self.factor, self.fired = dict(reference), float(factor), set()
        trained = [self.reference[f] for f in families if f in self.reference]
        self.train_reference = sum(trained) / len(trained) if trained else None

    def _alarm(self, key, text):
        if key in self.fired:
            return
        self.fired.add(key)
        print(f"🚨 LOSS ALARM: {text}", flush=True)
        if wandb.run is not None:
            wandb.alert(title=f"{RUN_NAME}: loss alarm", text=text)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        for key, value in (metrics or {}).items():
            family = key[len("eval_"):-len("_loss")] if key.startswith("eval_") and key.endswith("_loss") else None
            if family and loss_alarm(float(value), self.reference.get(family), self.factor):
                self._alarm(("eval", family), f"step {state.global_step}: dev {family} loss {value:.3f} > {self.factor} x step-0 {self.reference[family]:.3f}")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        value = (logs or {}).get("loss")
        if value is not None and loss_alarm(float(value), self.train_reference, self.factor):
            self._alarm(("train", state.global_step // 100), f"step {state.global_step}: train loss {value:.3f} > {self.factor} x step-0 dev {self.train_reference:.3f}")
        return control


class KillAtStepCallback(TrainerCallback):
    """Ladder rungs (G10): end train() at ``step`` with a save, as a killed session would have left it."""

    def __init__(self, step):
        super().__init__()
        self.step = int(step)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_save = True
            control.should_training_stop = True
        return control


def add_callbacks(trainer, kill_at=None):
    """The session's callbacks: resume gate, time budget, milestone pushes, loss alarm (and a rung's kill)."""
    gate = ResumeGateTrainerCallback()
    trainer.add_callback(gate)                                                      # step / LR / optimizer state / rows left, before the first step
    trainer.add_callback(TimeBudgetTrainerCallback(budget_s=TIME_BUDGET_H * 3600))   # clean save + push before the session's end
    trainer.add_callback(MilestonePushCallback(trainer, CKPT_REPO, trainer.args.output_dir, MILESTONES))
    trainer.add_callback(LossAlarmCallback(STEP0_DEV_LOSS, ARM_FAMILIES[ARM], «ALARM»))
    if kill_at is not None:
        trainer.add_callback(KillAtStepCallback(kill_at))
    return gate


_gate = add_callbacks(trainer, KILL_AT if STAGE != "train" else None)
_final_step = train_with_exact_resume(trainer, train_ds, _order, resume_dir, _gate, max_retries=MAX_INSESSION_RETRIES)
if STAGE != "train":
    # ---- G10: the session was "killed" at step KILL_AT; resume exactly from the Hub's copy in a fresh trainer ----
    assert _final_step == KILL_AT, f"G10: the rung stopped at step {_final_step}, expected the kill at {KILL_AT}"
    ensure_hub_checkpoint(CKPT_REPO, OUT_DIR, KILL_AT)
    _resumed_out, _resumed_root = OUT_DIR + "_resumed", RESUME_ROOT + "_resumed"
    _hub_dir, _hub_step = discover_resume(CKPT_REPO, _resumed_out, _resumed_root)   # a new output dir: only the Hub's copy is a candidate
    assert _hub_step == KILL_AT and _hub_dir == os.path.join(_resumed_root, "last-checkpoint"), f"G10: resume point {_hub_dir} (step {_hub_step})"
    _sha = lambda _path: hashlib.sha256(Path(_path).read_bytes()).hexdigest()
    assert _sha(os.path.join(OUT_DIR, f"checkpoint-{KILL_AT}", "adapter_model.safetensors")) == _sha(os.path.join(_hub_dir, "adapter_model.safetensors")), \
        "G10: the Hub's adapter differs from the one this session saved"
    del trainer                                                                      # a new session: nothing of the first trainer survives
    gc.collect()
    torch.cuda.empty_cache()
    trainer = make_trainer(model=model, tokenizer=_tok, data_collator=collate, train_dataset=_trl_train_ds, eval_dataset=_trl_dev_sets,
                           args=sft_args(_resumed_out))
    _order = lock_session_plan(CKPT_REPO, _plan, _order, has_checkpoint=True)      # the stored plan must match this session's
    _gate = add_callbacks(trainer)
    _final_step = train_with_exact_resume(trainer, train_ds, _order, _hub_dir, _gate, max_retries=0)
    assert _final_step == MAX_STEPS and _gate.passed == [KILL_AT], f"G10: resumed run ended at {_final_step}, gate passes {_gate.passed}"
    OUT_DIR = _resumed_out
    print(f"G10 OK: killed at step {KILL_AT}, resumed from the Hub copy (same adapter bytes, resume gate passed), finished at step {_final_step}")
ensure_hub_checkpoint(CKPT_REPO, OUT_DIR, _final_step)                           # the session ends only once the Hub holds this step
print(f"session ended at step {_final_step} of {MAX_STEPS}" + ("" if _final_step >= MAX_STEPS else ": run the notebook again to continue from exactly here"))
'''


def train_cell(v25a_train: str) -> str:
    """Cell 8: training, G7, G8, step-0 reference, milestones, G10.

    :param v25a_train: Source of cell 8 of the v2.5a Colab notebook (its kept blocks are copied).
    :returns: Source of cell 8.
    """
    blocks = kept_blocks(v25a_train)
    return _fill(TRAIN_TEMPLATE, FUNCS="from transformers import TrainerCallback\n" + inline_source(
                     family_quota, prepared_row_problems, milestone_steps, loss_alarm),
                 RUNS=_runs_literal(), SAVE=str(SAVE_STEPS), EVAL=str(EVAL_STEPS), FACTORIES=blocks["factories"],
                 RESUME_COMMENT=blocks["resume_comment"], WARMUP=WARMUP_RATIO, ROWS=str(ROWS_PER_STEP), LR=LR, WD=WEIGHT_DECAY,
                 OPTIM=repr(OPTIM), SEED=str(SEED), DECODE_K=str(DECODE_ROWS_PER_FAMILY), RECORDING=blocks["recording"],
                 TOL=repr(LOSS_TOLERANCE), MILESTONE=milestone_class(blocks["milestone"]), ALARM=repr(ALARM_FACTOR))


EXPORT_TEMPLATE = r'''# Cell 9 — export: shipped-adapter check, merge to bf16 (G10 merge test on the rungs), push (private)
import shutil
«FUNCS»
MERGED_DIR = f"{RUN_NAME}-merged"
MERGED_GB = «SMOKE_GB» if SMOKE else «GB»   # bf16 size of the merged model on disk
if STAGE == "train" and _final_step < MAX_STEPS:
    print(f"run stopped at step {_final_step} of {MAX_STEPS}: nothing exported — run the notebook again to continue")
else:
    # the shipped adapter: every language lora_B moved, none on vision / MTP (the G3 plan)
    _moved = [_n.replace("lora_B", "lora_A") for _n, _p in model.named_parameters() if "lora_B" in _n and _p.detach().abs().max().item() > 0]
    _counts, _problems = lora_problems(_moved, LAYER_TYPES, EXPECTED_LORA_A)
    assert not _problems, "shipped adapter: lora_B tensors that never moved or sit outside the language layers:\n  - " + "\n  - ".join(_problems)
    _free = shutil.disk_usage(".").free / 1e9
    assert _free > 1.2 * MERGED_GB + 5, f"{_free:.0f} GB free; the bf16 merge needs about {MERGED_GB} GB"
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    _files = sorted(os.listdir(MERGED_DIR))
    with open(os.path.join(MERGED_DIR, "config.json"), encoding="utf-8") as _fh:
        _merged_cfg = json.load(_fh)
    if "model.safetensors.index.json" in _files:
        with open(os.path.join(MERGED_DIR, "model.safetensors.index.json"), encoding="utf-8") as _fh:
            _keys = list(json.load(_fh)["weight_map"])
    else:
        from safetensors import safe_open
        with safe_open(os.path.join(MERGED_DIR, "model.safetensors"), "pt") as _fh:
            _keys = list(_fh.keys())
    _has_template = "chat_template.jinja" in _files
    if not _has_template and "tokenizer_config.json" in _files:
        with open(os.path.join(MERGED_DIR, "tokenizer_config.json"), encoding="utf-8") as _fh:
            _has_template = "chat_template" in json.load(_fh)
    _problems = merged_dir_problems(_files, _keys, _merged_cfg, _has_template)
    assert not _problems, "G10 merge test FAILED:\n  - " + "\n  - ".join(_problems)
    print(f"merged to {MERGED_DIR}: {len(_keys):,} tensors, {sum(os.path.getsize(os.path.join(MERGED_DIR, _f)) for _f in _files) / 1e9:.1f} GB")
    if STAGE == "rung1":
        # G10: the merged bf16 model scores a dev row like the 4-bit model + adapter
        import transformers as _tf
        _merged = getattr(_tf, _merged_cfg["architectures"][0]).from_pretrained(MERGED_DIR, torch_dtype=torch.bfloat16, device_map="cuda")
        _b = {_k: _v.to("cuda") for _k, _v in collate([next(iter(dev_sets.values()))[0]]).items()}
        model.eval()
        with torch.no_grad():
            _la, _lm = model(**_b).loss.item(), _merged(**_b).loss.item()
        del _merged
        torch.cuda.empty_cache()
        assert abs(_lm - _la) <= «TOL» * max(1.0, _la), f"G10: merged bf16 loss {_lm:.3f} != 4-bit + adapter loss {_la:.3f}"
        print(f"G10 OK: merged bf16 loss {_lm:.3f}, 4-bit + adapter {_la:.3f}")
    if STAGE in ("rung1", "train"):
        model.push_to_hub_merged(MERGED_REPO, tokenizer, save_method="merged_16bit", private=True)
        print("pushed", MERGED_REPO, "(private)")
    else:
        print(f"rung 2: merge test only — {MERGED_DIR} is not pushed (delete it to free the disk)")
'''


def export_cell() -> str:
    """Cell 9: shipped-adapter check, merge, merge test, push.

    :returns: Source of cell 9.
    """
    return _fill(EXPORT_TEMPLATE, FUNCS=inline_source(merged_dir_problems), SMOKE_GB=repr(MERGED_GB_SMOKE), GB=repr(MERGED_GB_27B),
                 TOL=repr(LOSS_TOLERANCE))


#: first lines of the v2.5a Colab cells this derivation reads or replaces (the source must be that notebook)
SOURCE_TITLES: Tuple[Tuple[int, str], ...] = (
    (1, "# Cell 1 — installs + env (PINNED"),
    (2, "# Cell 2 — data:"),
    (3, "# Cell 3 — model"),
    (4, "# Cell 4 — conversation format + collator"),
    (5, "# Cell 5 — gradient-flow verification"),
    (6, "# Cell 6a — OPTIONAL throughput benchmark"),
    (8, "# Cell 6 — "),
    (9, "# Cell 7 — export merged"),
)


def check_source(nb: Dict[str, Any]) -> None:
    """Refuse anything but the v2.5a Colab notebook's layout.

    :param nb: Source notebook.
    :returns: None.
    :raises ValueError: On a different cell count, a missing cell title, a missing helper cell or a Lambda edition.
    """
    cells = nb.get("cells", [])
    if len(cells) != 10:
        raise ValueError(f"the source must be the 10-cell v2.5a Colab notebook, got {len(cells)} cells")
    for i, title in SOURCE_TITLES:
        if not _src(cells[i]).lstrip("\n").startswith(title):
            raise ValueError(f"cell {i} of the source does not start with {title!r}")
    if "# ---- v22b session helpers" not in _src(cells[7]):
        raise ValueError("cell 7 of the source is not the inlined session-helper cell")
    if "google.colab" not in _src(cells[2]):
        raise ValueError("the source is not a Colab edition (derive from genizah_v25a.ipynb, not the Lambda edition)")


def derive(nb: Dict[str, Any], pins: Optional[Pins] = None) -> Dict[str, Any]:
    """The text27b Colab notebook from the v2.5a Colab notebook.

    :param nb: Notebook JSON of ``genizah_v25a.ipynb`` (not modified).
    :param pins: Pinned revisions, fingerprints and template hash (default: placeholders, the notebook refuses to run).
    :returns: The text27b Colab notebook (10 cells).
    :raises ValueError: When the source is not v2.5a's Colab edition or an anchor is missing.
    """
    pins = pins or Pins()
    pins.validate()
    check_source(nb)
    out = copy.deepcopy(nb)
    src = out["cells"]
    out["cells"] = [
        _with_source(src[0], intro_md()),
        _with_source(src[1], install_cell("colab")),
        _with_source(src[2], data_cell(pins)),
        _with_source(src[3], model_cell()),
        _with_source(src[4], collator_cell(pins)),
        _with_source(src[5], grad_cell()),
        _with_source(src[6], bench_cell()),
        _with_source(src[7], helper_cell(_src(src[7]))),
        _with_source(src[8], train_cell(_src(src[8]))),
        _with_source(src[9], export_cell()),
    ]
    return out


LAUNCH_NOTES = """# {title} — **Lambda Cloud edition**

Derived from the Colab notebook by `derive_t27.py` (through `host_lambda.py`): same data, schedule, repositories and exact
resume; secrets come from the environment or a prompt and are never stored here.

**Instance:** 1x A100 40 GB (x86_64), Lambda Stack image. Check the disk first: the 27B bf16 merge alone is about
{merged_gb} GB. Not GH200 (ARM: untested).

**Once per instance** (JupyterLab → Terminal), after uploading this notebook and `{setup}` with the file browser:

```bash
bash {setup}
```

(a venv from `host_lambda.PIN_SETS["{pin_set}"]`, registered as the Jupyter kernel **{kernel}**. Those pins are PLACEHOLDER
ranges until rung 1: the script writes the resolved versions to `~/genizah_work/{resolved}`, and G1 asserts the ranges.)

**Stages — each is its own GO:** `T27_STAGE=rung1` (2B smoke) → `rung2` (27B gates + timing: its GO estimate prices the
run) → `train` with `T27_ARM=t27a`, later `t27b`.

**Run, interactive:** set `ARM` / `STAGE` in cell 1 (or export `T27_ARM` / `T27_STAGE` before starting JupyterLab), choose
the kernel *{display}*, Run all. It asks for the Hugging Face token (write access) and the W&B API key once.

**Run, headless** (survives everything but the instance):

```bash
T27_ARM=t27a T27_STAGE=rung1 HF_TOKEN=… WANDB_API_KEY=… nohup {venv}/bin/jupyter nbconvert --to notebook --execute \\
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name={kernel} \\
  {notebook} --output t27_rung1.ipynb > t27_rung1.log 2>&1 &
```

**When W&B shows the run finished: terminate the instance in the Lambda console — it bills by the minute.**

---

"""


def to_lambda(nb: Dict[str, Any], notebook_name: str) -> Dict[str, Any]:
    """The Lambda edition of the text27b notebook.

    ``host_lambda.to_lambda`` makes the host substitutions (secrets from the environment or a prompt, the work directory,
    the wrap-up cell, the check that no Colab line remains); this edition then takes its own install cell (the venv of
    :data:`SETUP_SCRIPT`, not the Qwen3-VL pins), its own launch notes, a per-arm safety stop and a wrap-up alert named
    after the running arm.

    :param nb: :func:`derive` output.
    :param notebook_name: File name the edition is written as (quoted in the launch notes).
    :returns: The Lambda edition (11 cells).
    :raises ValueError: When a substitution's anchor is missing or a Colab-specific line remains.
    """
    out = host_lambda.to_lambda(nb, notebook_name)
    cells = out["cells"]
    title = _src(nb["cells"][0]).splitlines()[0].lstrip("# ").strip()
    notes = LAUNCH_NOTES.format(title=title, merged_gb=MERGED_GB_27B, setup=SETUP_SCRIPT, pin_set=PIN_SET, kernel=KERNEL_NAME,
                                resolved=RESOLVED_PINS_FILE, display=KERNEL_DISPLAY, venv=VENV, notebook=notebook_name)
    cells[0] = _with_source(cells[0], notes + _src(nb["cells"][0]))
    cells[1] = _with_source(cells[1], install_cell("lambda"))
    cells[7] = _with_source(cells[7], _replace_line(_src(cells[7]), r"^TIME_BUDGET_H = ", lambda_budget_line(), "time budget"))
    run_name = re.search(r'run_name="([^"]+)"', _src(nb["cells"][8])).group(1)
    cells[-1] = _with_source(cells[-1], _replace_once(_src(cells[-1]), f'title="{run_name}: notebook finished"',
                                                      'title=f"{RUN_NAME}: notebook finished"', "wrap-up alert"))
    leftovers = [(i, line) for i, c in enumerate(cells) for line in _src(c).splitlines()
                 if any(s in line for s in ("google.colab", "/content", "userdata", "%pip"))]
    if leftovers:
        raise ValueError(f"Colab-specific lines remain: {leftovers}")
    return out


def setup_script() -> str:
    """``lambda_setup_t27.sh``: the one-time venv of the text27b pin set on a Lambda instance.

    :returns: Bash source.
    """
    pins = pin_set()
    specs = pins["specifiers"]
    main = " ".join(_requirement(d, s) for d, s in specs.items() if d != "causal-conv1d")
    gate = " ".join(pins["kernel_gate"])
    torch_req = _requirement("torch", pins["torch"]) + " " + _requirement("torchvision", pins["torchvision"])
    cuda_pin = re.search(r"\+(cu\d+)$", str(pins["torch"]))
    cuda_selection = (f"CUDA={cuda_pin.group(1)}" if cuda_pin else
                      'if [ "$DRIVER" -ge 580 ]; then CUDA=cu130; else CUDA=cu128; fi')
    return f"""#!/bin/bash
# {SETUP_SCRIPT} — one-time environment for the text27b notebook ({host_lambda.lambda_files(NOTEBOOK)[0]}) on a Lambda instance.
# Generated by src/finetuning/qwen_hebrew/colab/derive_t27.py from host_lambda.PIN_SETS["{PIN_SET}"] (status: {pins["status"]}).
# Exact pins record the resolved smoke environment; the notebook's G1 gate verifies them at runtime.
# Usage (JupyterLab -> Terminal):   bash {SETUP_SCRIPT}
set -euo pipefail
VENV="{VENV}"
command -v nvidia-smi >/dev/null || {{ echo "nvidia-smi not found: this is not a GPU instance"; exit 1; }}
ARCH=$(uname -m)
[ "$ARCH" = "x86_64" ] || {{ echo "CPU architecture $ARCH is untested (GH200 is ARM): use an x86_64 instance"; exit 1; }}
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
# An exact CUDA wheel pin selects its matching index; otherwise choose by driver capability.
{cuda_selection}
echo "driver $DRIVER -> torch for $CUDA"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python {pins["python"]} "$VENV"
uv pip install --python "$VENV/bin/python" --index-url "https://download.pytorch.org/whl/$CUDA" {torch_req}
uv pip install --python "$VENV/bin/python" {main} {torch_req} packaging ipykernel nbconvert
# Compile against this torch/CUDA build without replacing torch or reusing an incompatible wheel.
uv pip install --python "$VENV/bin/python" setuptools wheel ninja
CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --python "$VENV/bin/python" --no-deps --no-cache --no-binary causal-conv1d --no-build-isolation {_requirement("causal-conv1d", specs.get("causal-conv1d", ""))}
"$VENV/bin/python" -m ipykernel install --user --name {KERNEL_NAME} --display-name "{KERNEL_DISPLAY}"
mkdir -p "$HOME/genizah_work"
uv pip freeze --python "$VENV/bin/python" > "$HOME/genizah_work/{RESOLVED_PINS_FILE}"
"$VENV/bin/python" - {gate} <<'PY'
import sys
from importlib.metadata import version
from importlib.util import find_spec
import torch
p = torch.cuda.get_device_properties(0)
print(f"torch {{torch.__version__}} on {{p.name}} ({{p.total_memory / 2**30:.0f}} GiB)")
for name in sys.argv[1:]:
    print(f"  {{name}} {{version(name)}}")
for module in ("fla", "causal_conv1d"):
    print(f"  {{module}}: {{'importable' if find_spec(module) else 'MISSING'}}")
PY
echo "kernel '{KERNEL_DISPLAY}' ready; resolved versions in ~/genizah_work/{RESOLVED_PINS_FILE}"
echo "rung 1: write the versions above into host_lambda.PIN_SETS['{PIN_SET}'] as exact pins and regenerate the notebook"
"""


# ===================================================================================== pins of a written notebook


def pins_of(nb: Dict[str, Any]) -> Pins:
    """Read the pinned values back from a derived notebook (either edition).

    :param nb: A text27b notebook.
    :returns: Its :class:`Pins`.
    :raises ValueError: When a pin line is missing.
    """
    c2, c4 = _src(nb["cells"][2]), _src(nb["cells"][4])
    found = {}
    for name, text, pattern in (("revisions", c2, r"^DATA_REVISIONS = (\{.*\})$"),
                                ("fingerprints", c2, r"^MIXTURE_FINGERPRINTS = (\{.*\})$"),
                                ("template_sha256", c4, r"^LOCAL_TEMPLATE_SHA256 = (None|'[0-9a-f]+')$")):
        m = re.search(pattern, text, re.M)
        if not m:
            raise ValueError(f"{name}: pin line not found")
        found[name] = ast.literal_eval(m.group(1))
    pins = Pins(**found)
    pins.validate()
    return pins


def current_pins(path: Path) -> Optional[Pins]:
    """Pins of an existing notebook file.

    :param path: A written text27b notebook (Colab edition).
    :returns: Its pins, or None when the file does not exist.
    """
    if not path.exists():
        return None
    return pins_of(json.loads(path.read_text(encoding="utf-8")))


def file_fingerprint(path: Path) -> str:
    """:func:`text_fingerprint` of a JSONL file in file order, streamed.

    :param path: Mixture file.
    :returns: Hex digest.
    """
    def _rows() -> Iterator[Dict[str, Any]]:
        """Rows of the file, one at a time.

        :returns: Iterator over the rows.
        """
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
    return text_fingerprint(_rows())


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes.

    :param path: File.
    :returns: Hex digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ===================================================================================== rung 0 (local gates)


def rung0(rows: Iterable[Dict[str, Any]], render: Callable[[Sequence[Dict[str, Any]]], str],
          encode: Callable[[str], Tuple[List[int], List[Tuple[int, int]]]], decode: Callable[[Sequence[int]], str],
          think_ids: Sequence[int] = (), max_seq: int = MAX_SEQ, guard_per_family: int = 2,
          schema: Optional[Tuple[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """Rung 0: the notebook's G2 schema (optional) and G5 render / length / mask gates on local rows.

    :param rows: Chat rows (rendered and tokenized as NFC text, like the notebook's cell 4: :func:`nfc_messages`).
    :param render: Messages -> rendered text (the chat template with ``enable_thinking=False``).
    :param encode: Text -> ``(ids, character offsets)`` without added special tokens.
    :param decode: Ids -> text, lossless.
    :param think_ids: Ids of ``<think>`` / ``</think>``.
    :param max_seq: Longest allowed row in tokens.
    :param guard_per_family: Rows per family that get the mask guardrail.
    :param schema: ``(split, families)`` to also run the G2 row schema, or None.
    :returns: Report: rows, problems by kind, rows over ``max_seq`` per family, tokens and supervised tokens per family,
        longest row per family, and ``ok``.
    """
    rep: Dict[str, Any] = {"rows": 0, "render_problems": Counter(), "schema_problems": Counter(), "split_problems": 0,
                           "over_max_seq": Counter(), "tokens": Counter(), "supervised": Counter(), "longest": {},
                           "guard_problems": []}
    guarded: Counter = Counter()
    for row in rows:
        rep["rows"] += 1
        meta = row.get("meta") or {}
        family = meta.get("family") or meta.get("task") or "?"
        if schema is not None:
            for p in row_problems(row, schema[0], schema[1], t27rows.CONDITIONS, t27rows.BUILD):
                rep["schema_problems"][" ".join(p.split()[:2])] += 1
        msgs = nfc_messages(row["messages"])
        text = unicodedata.normalize("NFC", render(msgs))
        problems = render_problems(text, msgs)
        for p in problems:
            rep["render_problems"][p] += 1
        if problems:
            continue
        try:
            prompt, completion = split_rendered(text, msgs[-1]["content"])
            ids, offsets = encode(text)
            n_prompt = prompt_token_count(offsets, len(prompt))
        except ValueError:   # a row whose render cannot be split or whose tokens straddle the boundary is counted, not fatal
            rep["split_problems"] += 1
            continue
        rep["tokens"][family] += len(ids)
        rep["supervised"][family] += len(ids) - n_prompt
        rep["longest"][family] = max(rep["longest"].get(family, 0), len(ids))
        if len(ids) > max_seq:
            rep["over_max_seq"][family] += 1
        if guarded[family] < guard_per_family:
            guarded[family] += 1
            labels = [-100] * n_prompt + list(ids[n_prompt:])
            for p in supervised_problems(ids, labels, n_prompt, prompt, completion, decode, think_ids):
                rep["guard_problems"].append((family, meta.get("row_id"), p))
    rep["ok"] = not (rep["render_problems"] or rep["schema_problems"] or rep["split_problems"] or rep["over_max_seq"]
                     or rep["guard_problems"])
    for key in ("render_problems", "schema_problems", "over_max_seq", "tokens", "supervised"):
        rep[key] = dict(rep[key])
    return rep


def _rung0_main(args: argparse.Namespace) -> int:
    """Run rung 0 from the CLI with the local tokenizer and chat template.

    :param args: Parsed arguments.
    :returns: Process exit status (0 when every gate passed).
    """
    tok = t27rows.get_tokenizer(args.rung0_tokenizer)

    def encode(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Local tokenizer encode.

        :param text: Rendered row.
        :returns: ``(ids, offsets)``.
        """
        enc = tok.encode(text, add_special_tokens=False)
        return enc.ids, enc.offsets

    def decode(ids: Sequence[int]) -> str:
        """Local tokenizer decode, special tokens kept.

        :param ids: Token ids.
        :returns: Text.
        """
        return tok.decode(list(ids), skip_special_tokens=False)

    def render(messages: Sequence[Dict[str, Any]]) -> str:
        """Local chat template with ``enable_thinking=False``.

        :param messages: Chat turns.
        :returns: Rendered text.
        """
        return t27rows.render_chat(messages, enable_thinking=False, template_path=args.rung0_template, check=False)

    def rows() -> Iterator[Dict[str, Any]]:
        """Rows of the mixture file, up to the limit.

        :returns: Iterator over the rows.
        """
        with open(args.rung0, encoding="utf-8") as fh:
            for k, line in enumerate(fh):
                if args.rung0_limit and k >= args.rung0_limit:
                    return
                if line.strip():
                    yield json.loads(line)

    schema = (args.rung0_split, ARMS[args.rung0_arm].families if args.rung0_split == "train" else t27rows.FAMILIES) \
        if args.rung0_arm else None
    think_ids = [i for i in (tok.token_to_id("<think>"), tok.token_to_id("</think>")) if i is not None]
    report = rung0(rows(), render, encode, decode, think_ids, MAX_SEQ, schema=schema)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    return 0 if report["ok"] else 1


# ===================================================================================== CLI


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the CLI.

    :returns: The parser.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arm in ARMS:
        ap.add_argument(f"--{arm}-revision", help=f"40-hex revision of {DATA_REPO} holding {arm}/train.jsonl, or {PLACEHOLDER}")
        ap.add_argument(f"--{arm}-fingerprint", help=f"64-hex text fingerprint of {arm}/train.jsonl, or {PLACEHOLDER}")
        ap.add_argument(f"--{arm}-mixture", type=Path, help=f"the pushed {arm} mixture file: its fingerprint is computed")
    ap.add_argument("--template", type=Path, help="chat template whose sha256 the notebook reports (default: paths.json chat_template)")
    ap.add_argument("--no-template", action="store_true", help="record no local template hash")
    ap.add_argument("--fresh", action="store_true", help="ignore the pins of the notebook already in --out-dir")
    ap.add_argument("--src", type=Path, default=HERE / SOURCE_NOTEBOOK)
    ap.add_argument("--host", choices=["both", "colab", "lambda"], default="both",
                    help=f"both (default) = {NOTEBOOK} + {host_lambda.lambda_files(NOTEBOOK)[0]} (+ {SETUP_SCRIPT})")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--rung0", type=Path, help="run rung 0 (G2 schema + G5 render / length / mask gates) on this JSONL instead")
    ap.add_argument("--rung0-arm", choices=sorted(ARMS), help="also run the G2 row schema for this arm")
    ap.add_argument("--rung0-split", choices=["train", "dev"], default="train")
    ap.add_argument("--rung0-limit", type=int, default=0, help="rows to check (0 = all)")
    ap.add_argument("--rung0-tokenizer", type=Path, help="tokenizer.json (default: paths.json tokenizer_json)")
    ap.add_argument("--rung0-template", type=Path, help="chat_template.jinja (default: paths.json chat_template)")
    return ap


def _default_template() -> Optional[Path]:
    """The local chat template from ``paths.json``, when one is configured and present.

    :returns: Its path, or None.
    """
    from src.finetuning.text27b import paths
    try:
        path = paths.load_paths().chat_template
    except (FileNotFoundError, KeyError, AttributeError):   # no machine-local config here: no template hash is recorded
        return None
    return path if path.exists() else None


def resolve_pins(args: argparse.Namespace) -> Pins:
    """Pins for this derivation: the existing notebook's, overridden by the command line.

    :param args: Parsed arguments.
    :returns: Validated pins.
    :raises SystemExit: On a malformed value, or a fingerprint given both directly and through a mixture file.
    """
    base = None if args.fresh else current_pins(args.out_dir / NOTEBOOK)
    pins = copy.deepcopy(base) if base is not None else Pins()
    for arm in ARMS:
        key = arm.replace("-", "_")
        revision, fingerprint, mixture = (getattr(args, f"{key}_revision"), getattr(args, f"{key}_fingerprint"),
                                          getattr(args, f"{key}_mixture"))
        if fingerprint and mixture:
            raise SystemExit(f"give --{arm}-fingerprint or --{arm}-mixture, not both")
        if revision:
            pins.revisions[arm] = revision
        if fingerprint:
            pins.fingerprints[arm] = fingerprint
        if mixture:
            pins.fingerprints[arm] = file_fingerprint(mixture)
    if args.no_template:
        pins.template_sha256 = None
    else:
        template = args.template or _default_template()
        if template is not None:
            pins.template_sha256 = file_sha256(template)
    try:
        pins.validate()
    except ValueError as error:
        raise SystemExit(str(error))
    return pins


def write_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` unless the file already holds it.

    :param path: Destination.
    :param text: Content.
    :returns: Whether the file was written.
    """
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: derive the Colab edition, the Lambda edition (+ the setup script beside it), or both; or run rung 0.

    :param argv: Arguments; None = ``sys.argv[1:]``.
    :returns: Exit status.
    """
    args = build_parser().parse_args(argv)
    if args.rung0 is not None:
        return _rung0_main(args)
    pins = resolve_pins(args)
    colab = derive(json.loads(args.src.read_text(encoding="utf-8")), pins)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    if args.host in ("both", "colab"):
        outputs.append((args.out_dir / NOTEBOOK, colab))
    if args.host in ("both", "lambda"):
        nb_name = host_lambda.lambda_files(NOTEBOOK)[0]
        outputs.append((args.out_dir / nb_name, to_lambda(colab, nb_name)))
        if write_if_changed(args.out_dir / SETUP_SCRIPT, setup_script()):
            print(f"wrote {args.out_dir / SETUP_SCRIPT}")
    for path, nb in outputs:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {path} ({len(nb['cells'])} cells; {DATA_REPO} revisions {pins.revisions}; "
              f"fingerprints { {a: f[:12] for a, f in pins.fingerprints.items()} }; template sha256 "
              f"{(pins.template_sha256 or 'none')[:12]})")
    unpinned = [a for a in ARMS if PLACEHOLDER in (pins.revisions[a], pins.fingerprints[a])]
    if unpinned:
        print(f"NOTE: the notebooks refuse to run {unpinned} until regenerated with the pushed revision and mixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
