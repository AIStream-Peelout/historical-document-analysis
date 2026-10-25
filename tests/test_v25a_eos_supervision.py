# File name: test_v25a_eos_supervision.py
# Date: 10/24/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Does a v2.5a training row teach the model to stop?

One training row is built the way the committed v2.5a notebooks build it: the notebook's own ``to_conversation`` (compiled
from cell 4 of ``genizah_v25a_lambda.ipynb``), the processor's ``apply_chat_template(tokenize=False,
add_generation_prompt=False)`` and ``processor(text, images, padding=True, add_special_tokens=False)`` as
``UnslothVisionDataCollator`` calls them, then the collator's label rule: image / pad tokens to -100, and
``train_on_responses_only`` with the cell's ``instruction_part`` / ``response_part`` (everything before the end of the
assistant header masked, everything from there to the next user header -- or to the end of the row -- copied from the
labels).

unsloth / unsloth_zoo are not installed on the Mac (the run's stack, unsloth 2026.8.9 + unsloth_zoo 2026.8.6, lives on
the training host only), so the label rule is ported here from ``unsloth_zoo.dataset_utils.train_on_responses_only``
(``force_match=True``) and ``unsloth_zoo.vision_utils.get_padding_tokens_ids``; :func:`test_real_unsloth_rule_agrees`
runs the real functions instead and is skipped where unsloth_zoo cannot be imported.

The processor is the one the trainer saved with the v2.5a checkpoint (``last-checkpoint/`` of
``isaacmg/qwen3-vl-8b-hebrew-v25a-ckpt`` in the local Hub cache: eos ``<|im_end|>``, pad ``<|vision_pad|>``), else the
local v2.4a step-2000 export; the tests skip when neither is on disk. Nothing is downloaded.

Finding: the assistant turn renders as ``answer<|im_end|>\\n``, both tokens are in the row and both are supervised
(labels != -100); the user turn's ``<|im_end|>`` is masked. End-of-turn supervision is present.
"""
import ast
import glob
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import pytest

ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "src" / "finetuning" / "qwen_hebrew" / "colab"
NOTEBOOKS = (COLAB / "genizah_v25a_lambda.ipynb", COLAB / "genizah_v25a.ipynb")
MODEL_CELL, COLLATOR_CELL = 3, 4
COLLATOR_CLASS = "UnslothVisionDataCollator"
IGNORE_INDEX = -100
END_OF_TURN = "<|im_end|>"

#: Processor directories, first = the exact training-time tokenizer + preprocessor (saved by the trainer with every
#: v2.5a checkpoint), then the local MLX export of v2.4a step 2000 (same vocabulary and chat template).
PROCESSOR_CANDIDATES = (
    "/Users/isaac/hf_home_merge_local/hub/models--isaacmg--qwen3-vl-8b-hebrew-v25a-ckpt/snapshots/*/last-checkpoint",
    str(ROOT / "models" / "qwen3-vl-8b-heb-v24a-step2000"),
)

#: The Qwen entries of ``unsloth_zoo.vision_utils.IMAGE_TOKENS`` (the other entries are not in this vocabulary).
IMAGE_TOKENS = ("<|image|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>")

HEBREW_ANSWER = "בשם רחמנא\nשלום רב לאדוני ומורי\nוכתבתי לך מן מצרים"
ARABIC_ANSWER = "بسم الله الرحمن الرحيم\nكتابي اليك من مصر\nوالحمد لله وحده"


def processor_dir() -> Optional[Path]:
    """The first processor directory of :data:`PROCESSOR_CANDIDATES` that is on disk.

    :return: Directory holding ``tokenizer.json``, ``chat_template.jinja`` and ``preprocessor_config.json``, or None.
    :rtype: Optional[Path]
    """
    for pattern in PROCESSOR_CANDIDATES:
        for hit in sorted(glob.glob(pattern)):
            path = Path(hit)
            if all((path / f).is_file() for f in ("tokenizer.json", "chat_template.jinja", "preprocessor_config.json")):
                return path
    return None


def cell_source(nb_path: Path, index: int) -> str:
    """Source of one notebook cell.

    :param nb_path: Notebook file.
    :param index: Cell index.
    :return: The cell's source text.
    :rtype: str
    """
    return "".join(json.loads(nb_path.read_text(encoding="utf-8"))["cells"][index]["source"])


def notebook_function(source: str, name: str) -> Callable[..., Any]:
    """Compile one top-level ``def`` of a notebook cell on its own (the rest of the cell needs unsloth and a GPU).

    :param source: Cell source.
    :param name: Function name.
    :return: The function, defined exactly as the cell defines it.
    :rtype: Callable[..., Any]
    """
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace: Dict[str, Any] = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), f"<notebook cell: {name}>", "exec"), namespace)
    return namespace[name]


def collator_kwargs(source: str) -> Dict[str, Any]:
    """Keyword arguments of the cell's ``UnslothVisionDataCollator(...)`` call; literals as values, names as ``str``.

    :param source: Cell source.
    :return: ``{keyword: literal value or variable name}``.
    :rtype: Dict[str, Any]
    """
    call = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == COLLATOR_CLASS)
    return {kw.arg: (kw.value.id if isinstance(kw.value, ast.Name) else ast.literal_eval(kw.value)) for kw in call.keywords}


def cell_constants(source: str, names: Sequence[str]) -> Dict[str, Any]:
    """Literal top-level assignments of a cell (e.g. ``MAX_SEQ``, ``MIN_PIX``).

    :param source: Cell source.
    :param names: Variable names wanted.
    :return: ``{name: value}`` for the names assigned a literal.
    :rtype: Dict[str, Any]
    """
    out: Dict[str, Any] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def padding_token_ids(tokenizer: Any) -> Set[int]:
    """Ids the collator masks before the response rule (``unsloth_zoo.vision_utils.get_padding_tokens_ids``).

    :param tokenizer: The text tokenizer of the processor.
    :return: Image-token ids plus the pad token id.
    :rtype: Set[int]
    """
    tokens = list(IMAGE_TOKENS) + ([tokenizer.image_token] if hasattr(tokenizer, "image_token") else [])
    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens] + [getattr(tokenizer, "pad_token_id", None)]
    return {i for i in ids if i}


def responses_only_labels(input_ids: Sequence[int], old_labels: Sequence[int], instruction_ids: Sequence[int],
                          response_ids: Sequence[int]) -> List[int]:
    """Port of the inner function of ``unsloth_zoo.dataset_utils.train_on_responses_only`` with ``force_match=True``.

    Every label is -100 except the spans that start right after a response header and end at the next instruction
    header, or at the end of the row when the assistant turn is the last one; those spans are copied from
    ``old_labels`` (pad / image tokens already -100).

    :param input_ids: Token ids of one row.
    :param old_labels: Labels after the padding mask.
    :param instruction_ids: Token ids of ``instruction_part``.
    :param response_ids: Token ids of ``response_part``.
    :return: The row's labels.
    :rtype: List[int]
    """
    ids, q, a = list(input_ids), list(instruction_ids), list(response_ids)
    n = len(ids)
    labels = [IGNORE_INDEX] * n
    j = 0
    while j < n:
        if ids[j:j + len(a)] == a:
            assistant_k = j = j + len(a)
            while j < n:
                if j == n - 1 or ids[j:j + len(q)] == q:
                    user_j = n if j == n - 1 else j          # the last turn runs to the end of the row
                    labels[assistant_k:user_j] = old_labels[assistant_k:user_j]
                    j = n - 1 if j == n - 1 else j + len(q)
                    break
                j += 1
        j += 1
    return labels


def collate_row(processor: Any, to_conversation: Callable[[Dict[str, Any]], Dict[str, Any]], sample: Dict[str, Any],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """One row through the notebook's formatting and the collator's text, tokenisation and label steps.

    :param processor: Qwen3-VL processor with the notebook's pixel bounds.
    :param to_conversation: The notebook's ``to_conversation``.
    :param sample: ``{"image", "question", "answer"}``.
    :param kwargs: The notebook's collator keyword arguments.
    :return: ``{"text", "input_ids", "labels", "assistant_start"}``.
    :rtype: Dict[str, Any]
    """
    messages = to_conversation(sample)["messages"]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = processor(text=[text], images=[sample["image"]], padding=True, return_tensors="pt", add_special_tokens=False)
    tok = processor.tokenizer
    ids = enc["input_ids"][0].tolist()
    masked = padding_token_ids(tok)
    labels = [IGNORE_INDEX if t in masked else t for t in ids]
    assert kwargs["train_on_responses_only"] is True
    response_ids = tok(kwargs["response_part"], add_special_tokens=False).input_ids
    labels = responses_only_labels(ids, labels, tok(kwargs["instruction_part"], add_special_tokens=False).input_ids,
                                   response_ids)
    start = next(j for j in range(len(ids)) if ids[j:j + len(response_ids)] == response_ids) + len(response_ids)
    return {"text": text, "input_ids": ids, "labels": labels, "assistant_start": start, "grid": enc["image_grid_thw"]}


@pytest.fixture(scope="module")
def notebook() -> Dict[str, Any]:
    """The collator cell's ``to_conversation`` and keyword arguments, and the model cell's constants.

    :return: ``{"to_conversation", "kwargs", "consts"}``.
    :rtype: Dict[str, Any]
    """
    cells = [cell_source(nb, COLLATOR_CELL) for nb in NOTEBOOKS]
    assert cells[0] == cells[1], "the Colab and Lambda editions differ in the collator cell"
    consts = cell_constants(cell_source(NOTEBOOKS[0], MODEL_CELL), ("MAX_SEQ", "MIN_PIX", "MAX_PIX"))
    return {"to_conversation": notebook_function(cells[0], "to_conversation"), "kwargs": collator_kwargs(cells[0]),
            "consts": consts}


@pytest.fixture(scope="module")
def processor(notebook: Dict[str, Any]) -> Any:
    """The training processor with the notebook's pixel bounds (cell 3: ``tokenizer.image_processor = ...``).

    :param notebook: The :func:`notebook` fixture.
    :return: A Qwen3-VL processor.
    """
    path = processor_dir()
    if path is None:
        pytest.skip("no local Qwen3-VL processor (v2.5a checkpoint cache or the v2.4a export)")
    transformers = pytest.importorskip("transformers")
    proc = transformers.AutoProcessor.from_pretrained(str(path))
    proc.image_processor = transformers.AutoImageProcessor.from_pretrained(
        str(path), min_pixels=notebook["consts"]["MIN_PIX"], max_pixels=notebook["consts"]["MAX_PIX"])
    return proc


def page_sample(answer: str) -> Dict[str, Any]:
    """A page row: a blank page image (upscaled to the 6.5 MP floor by the processor), a prompt and an answer.

    :param answer: The transcription.
    :return: ``{"image", "question", "answer"}``.
    :rtype: Dict[str, Any]
    """
    from PIL import Image

    return {"image": Image.new("RGB", (1200, 1800), "white"), "question": "Transcribe this page.", "answer": answer}


def test_collator_cell_is_responses_only_on_the_assistant_header(notebook: Dict[str, Any]) -> None:
    """The emitted cell trains on responses only, split at the Qwen user / assistant headers."""
    kw = notebook["kwargs"]
    assert kw["train_on_responses_only"] is True and kw["resize"] == "max" and kw["max_seq_length"] == "MAX_SEQ"
    assert (kw["instruction_part"], kw["response_part"]) == ("<|im_start|>user\n", "<|im_start|>assistant\n")
    assert "formatting_func" in kw and kw["formatting_func"] == "to_conversation"


@pytest.mark.parametrize("answer", [HEBREW_ANSWER, ARABIC_ANSWER], ids=["hebrew_page", "arabic_page"])
def test_assistant_turn_ends_with_supervised_im_end(notebook: Dict[str, Any], processor: Any, answer: str) -> None:
    """(i) the templated assistant turn ends with <|im_end|>; (ii) that id closes the row's assistant span;
    (iii) its label is not -100 -- the model is trained to stop. The prompt (and the user turn's <|im_end|>) is masked."""
    tok = processor.tokenizer
    eot = tok.convert_tokens_to_ids(END_OF_TURN)
    newline = tok("\n", add_special_tokens=False).input_ids
    assert tok.eos_token == END_OF_TURN and tok.eos_token_id == eot
    assert tok.pad_token_id != eot, "pad == <|im_end|> would mask every end-of-turn"

    row = collate_row(processor, notebook["to_conversation"], page_sample(answer), notebook["kwargs"])
    ids, labels, start = row["input_ids"], row["labels"], row["assistant_start"]

    assert row["text"].endswith("<|im_start|>assistant\n" + answer + END_OF_TURN + "\n")          # (i)
    assert ids[-1 - len(newline):] == [eot] + newline                                               # (ii)
    assert labels[-1 - len(newline)] == eot != IGNORE_INDEX                                          # (iii)
    assert labels[-len(newline):] == newline                       # the trailing "\n" after <|im_end|> is supervised too
    assert labels[start:] == ids[start:], "the whole assistant span (answer + <|im_end|> + \\n) carries the loss"
    assert all(t == IGNORE_INDEX for t in labels[:start]), "prompt, image tokens and the user <|im_end|> are masked"
    assert tok.decode(labels[start:]) == answer + END_OF_TURN + "\n"
    assert [i for i, t in enumerate(ids) if t == eot] == [ids.index(eot), len(ids) - 1 - len(newline)]
    assert labels[ids.index(eot)] == IGNORE_INDEX                                       # the user turn's end-of-turn
    assert len(ids) <= notebook["consts"]["MAX_SEQ"] and int(row["grid"].prod()) // 4 > 6000   # page res, no truncation


def test_real_unsloth_rule_agrees(notebook: Dict[str, Any], processor: Any) -> None:
    """On a host with unsloth_zoo (the training stack), its own padding ids and response rule give the same labels."""
    dataset_utils = pytest.importorskip("unsloth_zoo.dataset_utils")
    vision_utils = pytest.importorskip("unsloth_zoo.vision_utils")
    import torch

    kw = notebook["kwargs"]
    row = collate_row(processor, notebook["to_conversation"], page_sample(HEBREW_ANSWER), kw)
    eot = processor.tokenizer.convert_tokens_to_ids(END_OF_TURN)
    pad_ids = vision_utils.get_padding_tokens_ids(processor)
    assert eot not in set(pad_ids.tolist())
    ids = torch.tensor([row["input_ids"]])
    old = ids.clone()
    old[torch.isin(old, pad_ids)] = IGNORE_INDEX
    fn = dataset_utils.train_on_responses_only(None, instruction_part=kw["instruction_part"],
                                               response_part=kw["response_part"], force_match=True,
                                               tokenizer=processor, return_function=True)
    assert fn({"input_ids": ids, "labels": old})["labels"][0].tolist() == row["labels"]
