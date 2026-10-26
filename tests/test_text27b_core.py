# File name: test_text27b_core.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Tests for text27b ``paths`` and ``rows``: config resolution, JSONL I/O, ids, fingerprints,
prompts, token counts and the chat-template render contract.

Everything runs on synthetic fixtures in ``tmp_path`` (a tiny word-level tokenizer and small
Jinja templates). Two tests also exercise the real local Qwen3.8 template / tokenizer named in
``paths.json`` and are skipped on machines without them.
"""
import codecs
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

from src.finetuning.text27b import paths as P
from src.finetuning.text27b import rows as R

REQUIRED = ["probe_dir", "text_ft_v0_1", "pgp_parallel", "pgp_export", "merged_pgpid_map", "document_treatments",
            "analysis_pages_index", "exclusions_v1", "samples", "a50_overlap", "benchmark_ids", "academic_literature",
            "ai_reads", "book_editions_nas", "tokenizer_json", "chat_template", "out_dir", "scratch"]

#: A small template with the Qwen3.8 thinking behaviour that matters here.
GOOD_TEMPLATE = (
    "{%- if enable_thinking is undefined or enable_thinking is true %}{%- set effort = 'Reasoning effort is set to xhigh.' %}"
    "{%- else %}{%- set effort = '' %}{%- endif %}"
    "{%- for m in messages %}"
    "{%- if m.role == 'system' %}{{- '<|im_start|>system\\n' + (effort + '\\n\\n' if effort else '') + m.content|trim + '<|im_end|>\\n' }}"
    "{%- elif m.role == 'user' %}{{- '<|im_start|>user\\n' + m.content|trim + '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + m.content|trim + '<|im_end|>\\n' }}{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}")
#: Injects the effort sentence even with thinking off.
EFFORT_TEMPLATE = GOOD_TEMPLATE.replace("{%- set effort = '' %}", "{%- set effort = 'Reasoning effort is set to low.' %}")
#: Drops the empty think block.
NO_THINK_TEMPLATE = GOOD_TEMPLATE.replace("<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + m.content",
                                          "<|im_start|>assistant\\n' + m.content")


def write_config(tmp_path: Path, **overrides: Any) -> Path:
    """Write a synthetic ``paths.json`` with repository-relative entries under ``tmp_path``.

    :param tmp_path: Pytest temporary directory (used as the repo root).
    :param overrides: Entries to replace or add.
    :returns: Path of the config file.
    """
    cfg: Dict[str, Any] = {"repo": str(tmp_path)}
    cfg.update({k: f"data/{k}" for k in REQUIRED})
    cfg["out_dir"] = "out/v0_2"
    cfg.update(overrides)
    f = tmp_path / "paths.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    return f


def make_tokenizer(tmp_path: Path) -> Path:
    """Save a tiny word-level tokenizer (whitespace pre-tokenizer) to ``tmp_path``.

    :param tmp_path: Pytest temporary directory.
    :returns: Path of ``tokenizer.json``.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "<|im_start|>": 3, "<|im_end|>": 4}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    out = tmp_path / "tokenizer.json"
    tok.save(str(out))
    return out


def write_template(tmp_path: Path, text: str, name: str = "chat_template.jinja") -> Path:
    """Write a Jinja template file.

    :param tmp_path: Pytest temporary directory.
    :param text: Template source.
    :param name: File name.
    :returns: Path of the template.
    """
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


def messages(answer: str = "The answer.") -> List[Dict[str, str]]:
    """A three-turn chat row.

    :param answer: Assistant content.
    :returns: Messages.
    """
    return R.make_messages("Translate this.", answer, system="You are a test.")


# ----------------------------------------------------------------------------------- paths
def test_load_paths_resolves_relative_absolute_and_extra(tmp_path: Path) -> None:
    """Relative entries resolve against ``repo``; absolute and ``~`` entries are kept; unknown keys go to ``extra``.

    :param tmp_path: Pytest temporary directory.
    """
    absolute = tmp_path / "elsewhere" / "tok.json"
    cfg = write_config(tmp_path, tokenizer_json=str(absolute), new_dir="logs/new", label="plain")
    p = P.load_paths(cfg)
    assert p.repo == tmp_path
    assert p.pgp_parallel == tmp_path / "data" / "pgp_parallel"
    assert p.tokenizer_json == absolute
    assert p.get("new_dir") == tmp_path / "logs" / "new"
    assert p.get("label") == "plain"
    assert p.get("missing", 7) == 7
    assert p.get("out_dir") == tmp_path / "out" / "v0_2"
    assert p.merged_shelfmarks == tmp_path / P.OPTIONAL_DEFAULTS["merged_shelfmarks"]
    assert p.source_file == cfg.resolve()


def test_paths_out_and_mkdir(tmp_path: Path) -> None:
    """``Paths.out`` joins below ``out_dir`` and creates the parent only on request.

    :param tmp_path: Pytest temporary directory.
    """
    p = P.load_paths(write_config(tmp_path))
    target = p.out("families", "x.jsonl")
    assert target == tmp_path / "out" / "v0_2" / "families" / "x.jsonl"
    assert not target.parent.exists()
    p.out("families", "x.jsonl", mkdir=True)
    assert target.parent.is_dir()


def test_load_paths_missing_entry_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing required entry raises; ``$TEXT27B_PATHS`` selects the file.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    cfg = write_config(tmp_path)
    data = json.loads(cfg.read_text())
    del data["samples"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(KeyError):
        P.load_paths(bad)
    monkeypatch.setenv(P.ENV_VAR, str(cfg))
    assert P.load_paths().samples == tmp_path / "data" / "samples"


def test_load_paths_picks_up_edits(tmp_path: Path) -> None:
    """The cache is keyed on modification time, so an edited config is re-read.

    :param tmp_path: Pytest temporary directory.
    """
    cfg = write_config(tmp_path)
    assert P.load_paths(cfg).samples == tmp_path / "data" / "samples"
    data = json.loads(cfg.read_text())
    data["samples"] = "other/samples.json"
    cfg.write_text(json.dumps(data))
    os.utime(cfg, ns=(cfg.stat().st_atime_ns, cfg.stat().st_mtime_ns + 10_000_000))
    assert P.load_paths(cfg).samples == tmp_path / "other" / "samples.json"


# ----------------------------------------------------------------------------------- constants / prompts
def test_constants_cover_brief_values() -> None:
    """Families, conditions, splits and exposure levels match the build brief."""
    assert set(R.PGP_FAMILIES) == {"translate_en", "translate_he", "catalogue_json", "describe_pgp",
                                   "extract_relations", "analyse_document"}
    assert {"domain_reading", "rag_context"} <= set(R.FAMILIES)
    assert set(R.CONDITIONS) == {"edition", "noised_edition", "read", "read_hard", "read_partial",
                                 "edition_rescript", "literature"}
    assert {"train", "dev", "test_indist", "heldout_vision", "heldout"} <= set(R.SPLITS)
    assert set(R.EXPOSURE_LEVELS) == {"page", "pgpid", "fragment", "none"}


def test_read_like_conditions_share_one_amended_note() -> None:
    """Every read-like condition uses the single amended note; the v0 notes stay verbatim."""
    notes = {R.COND_NOTE[c] for c in R.READ_LIKE_CONDITIONS}
    assert notes == {R.READ_NOTE_V2}
    assert "skip lines" in R.READ_NOTE_V2 and "neighbouring document" in R.READ_NOTE_V2
    assert R.COND_NOTE_V0["noised_edition"] == R.COND_NOTE_V0["read"]
    assert R.COND_NOTE["edition"] == R.COND_NOTE_V0["edition"]


def test_user_prompt_layout() -> None:
    """The user prompt places instruction, note, document and extra blocks in order."""
    u = R.user_prompt("translate_en", "read_hard", "arabic", "نص", extra="Context: x")
    assert u.startswith(R.INSTRUCTION["translate_en"].format(script="Arabic script"))
    assert f"\n\n{R.READ_NOTE_V2}\n\n<document>\nنص\n</document>\n\nContext: x" in u
    v0 = R.user_prompt("describe_pgp", "noised_edition", "unknown", "t", notes=R.COND_NOTE_V0)
    assert R.COND_NOTE_V0["read"] in v0 and "(Hebrew script)" in v0
    with pytest.raises(KeyError):
        R.user_prompt("translate_en", "nonsense", "hebrew", "t")


# ----------------------------------------------------------------------------------- I/O
def test_jsonl_roundtrip_atomic(tmp_path: Path) -> None:
    """``write_jsonl`` writes atomically and ``read_jsonl`` streams the rows back.

    :param tmp_path: Pytest temporary directory.
    """
    rows = [{"a": 1, "t": "ש"}, {"a": 2, "t": None}]
    f = tmp_path / "sub" / "rows.jsonl"
    assert R.write_jsonl(f, iter(rows)) == 2
    assert list(R.read_jsonl(f)) == rows
    assert R.load_jsonl(f) == rows
    assert not [x for x in f.parent.iterdir() if x.name.endswith(".tmp")]
    f.write_text(f.read_text() + "\n\n")
    assert len(R.load_jsonl(f)) == 2


def test_write_jsonl_failure_leaves_old_file(tmp_path: Path) -> None:
    """A failure while writing keeps the previous file and removes the temporary one.

    :param tmp_path: Pytest temporary directory.
    """
    f = tmp_path / "rows.jsonl"
    R.write_jsonl(f, [{"a": 1}])

    def bad() -> Iterator[Dict[str, int]]:
        """Yield one row, then fail.

        :returns: Generator of rows.
        """
        yield {"a": 2}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        R.write_jsonl(f, bad())
    assert R.load_jsonl(f) == [{"a": 1}]
    assert [x.name for x in tmp_path.iterdir()] == ["rows.jsonl"]


# ----------------------------------------------------------------------------------- ids
def test_row_id_stable_and_discriminating() -> None:
    """Row ids are 16 hex chars, deterministic, and change with each component."""
    a = R.row_id("translate_en", "123", "read", "a50")
    assert a == R.row_id("translate_en", 123, "read", "a50")
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)
    assert R.row_id("translate_en", "123", "read", None) == R.row_id("translate_en", "123", "read", "")
    others = {R.row_id("translate_he", "123", "read", "a50"), R.row_id("translate_en", "124", "read", "a50"),
              R.row_id("translate_en", "123", "edition", "a50"), R.row_id("translate_en", "123", "read", "v21b"),
              R.row_id("translate_en", "123", "read", "a50", "distractor")}
    assert a not in others and len(others) == 5


def test_fingerprint_order_and_columns() -> None:
    """Fingerprints depend on row order and on the chosen columns only."""
    r1 = {"messages": messages("one"), "meta": {"split": "train"}}
    r2 = {"messages": messages("two"), "meta": {"split": "dev"}}
    assert R.fingerprint([r1, r2]) == R.fingerprint([dict(r1), dict(r2)])
    assert R.fingerprint([r1, r2]) != R.fingerprint([r2, r1])
    assert R.fingerprint([r1], keys=("user",)) == R.fingerprint([r2], keys=("user",))
    assert R.fingerprint([r1], keys=("meta.split",)) != R.fingerprint([r2], keys=("meta.split",))
    assert R.fingerprint([]) == R.fingerprint([], keys=("assistant",))


def test_new_meta_and_validate_row() -> None:
    """``new_meta`` fills the identity fields; ``validate_row`` reports schema problems."""
    meta = R.new_meta("translate_en", "read_hard", "train", "77", read_model="a50", lang_class="JA")
    assert meta["task"] == "translate_en" and meta["build"] == R.BUILD
    assert meta["row_id"] == R.row_id("translate_en", "77", "read_hard", "a50")
    for k in R.V01_META_FIELDS:
        meta.setdefault(k, None)
    row = {"messages": messages(), "meta": meta}
    assert R.validate_row(row) == []
    bad = {"messages": messages() + [{"role": "user", "content": "x"}], "meta": dict(meta, condition="ocr", split="x")}
    probs = R.validate_row(bad)
    assert any("messages" in p for p in probs) and any("condition" in p for p in probs) and any("split" in p for p in probs)
    unstripped = {"messages": [dict(m) for m in messages()], "meta": meta}
    unstripped["messages"][2]["content"] = "answer \n"
    assert any("whitespace" in p for p in R.validate_row(unstripped))
    assert any("v0_1" in p for p in R.validate_row({"messages": messages(), "meta": R.new_meta("describe_pgp", "edition", "dev", 1)}))


# ----------------------------------------------------------------------------------- tokens
def test_count_tokens_with_explicit_tokenizer(tmp_path: Path) -> None:
    """Token counts use the given ``tokenizer.json`` without special tokens.

    :param tmp_path: Pytest temporary directory.
    """
    tok = make_tokenizer(tmp_path)
    assert R.count_tokens("hello world hello", tok) == 3
    assert R.count_tokens("", tok) == 0
    assert R.count_tokens_batch(["hello", "hello world", ""], tok) == [1, 2, 0]
    assert R.count_tokens_batch([], tok) == []
    with pytest.raises(FileNotFoundError):
        R.count_tokens("x", tmp_path / "missing.json")


# ----------------------------------------------------------------------------------- render
def test_render_chat_contract_with_synthetic_template(tmp_path: Path) -> None:
    """A conforming template renders one empty think block before the answer and no effort sentence.

    :param tmp_path: Pytest temporary directory.
    """
    tpl = write_template(tmp_path, GOOD_TEMPLATE)
    msgs = messages("  The answer.  ")
    out = R.render_chat(msgs, template_path=tpl)
    assert out.endswith(R.EMPTY_THINK + "The answer.<|im_end|>\n")
    assert out.count(R.EMPTY_THINK) == 1 and R.EFFORT_MARKER not in out
    thinking = R.render_chat(msgs, enable_thinking=True, template_path=tpl)
    assert R.EFFORT_MARKER in thinking
    with pytest.raises(R.RenderError):
        R.check_render(thinking, msgs)
    gen = R.render_chat(msgs[:2], add_generation_prompt=True, template_path=tpl)
    assert gen.endswith("<|im_start|>assistant\n" + R.EMPTY_THINK)
    prompt, completion = R.split_prompt_completion(msgs, template_path=tpl)
    assert prompt + completion == out and completion == "The answer.<|im_end|>\n"
    assert prompt == gen


def test_render_chat_rejects_broken_templates(tmp_path: Path) -> None:
    """An effort sentence or a missing think block with thinking off raises :class:`RenderError`.

    :param tmp_path: Pytest temporary directory.
    """
    for name, text in (("effort.jinja", EFFORT_TEMPLATE), ("nothink.jinja", NO_THINK_TEMPLATE)):
        tpl = write_template(tmp_path, text, name)
        with pytest.raises(R.RenderError):
            R.render_chat(messages(), template_path=tpl)
        assert R.render_chat(messages(), template_path=tpl, check=False)
    with pytest.raises(R.RenderError):
        R.split_prompt_completion(messages()[:2], template_path=write_template(tmp_path, GOOD_TEMPLATE, "g.jinja"))


def test_check_render_suffix_mismatch() -> None:
    """A render whose tail is not the assistant content fails the contract."""
    msgs = messages("Real answer.")
    fake = "<|im_start|>assistant\n" + R.EMPTY_THINK + "Other answer.<|im_end|>\n"
    with pytest.raises(R.RenderError):
        R.check_render(fake, msgs)
    double = "<|im_start|>assistant\n" + R.EMPTY_THINK + R.EMPTY_THINK + "Real answer.<|im_end|>\n"
    with pytest.raises(R.RenderError):
        R.check_render(double, msgs)


def test_tokens_of_row_counts_rendered_text(tmp_path: Path) -> None:
    """``tokens_of_row`` counts the rendered conversation with the given tokenizer.

    :param tmp_path: Pytest temporary directory.
    """
    tpl = write_template(tmp_path, GOOD_TEMPLATE)
    tok = make_tokenizer(tmp_path)
    row = {"messages": messages("hello world"), "meta": {}}
    rendered = R.render_chat(row["messages"], template_path=tpl)
    assert R.tokens_of_row(row, tokenizer_path=tok, template_path=tpl) == R.count_tokens(rendered, tok)


def _real_paths() -> Any:
    """The machine's configured paths, or None when the config / model files are absent.

    :returns: :class:`Paths` or None.
    """
    try:
        p = P.load_paths()
    except (FileNotFoundError, KeyError):
        return None
    if not (p.chat_template.exists() and p.tokenizer_json.exists()):
        return None
    return p


@pytest.mark.skipif(_real_paths() is None, reason="local Qwen3.8 chat template / tokenizer not available")
def test_real_qwen_template_contract() -> None:
    """The real Qwen3.8 template with thinking off: answer after exactly one empty think block, no effort sentence."""
    msgs = R.make_messages("Catalogue this document.", "{\"document_type\": \"Letter\"}", system=R.SYSTEM)
    out = R.render_chat(msgs)
    assert out.endswith(R.EMPTY_THINK + msgs[2]["content"] + "<|im_end|>\n")
    assert out.count(R.EMPTY_THINK) == 1 and out.count("<think>") == 1
    assert "Reasoning effort" not in out
    assert "Reasoning effort" in R.render_chat(msgs, enable_thinking=True, check=False)
    prompt, completion = R.split_prompt_completion(msgs)
    assert prompt + completion == out
    assert R.tokens_of_row({"messages": msgs}) == R.count_tokens(out) > R.count_tokens(msgs[2]["content"])


def test_make_messages_nfc_and_non_nfc_roles() -> None:
    """F3: every message content is NFC when a row is made; :func:`rows.non_nfc_roles` names decomposed turns.

    :returns: None.
    """
    decomposed = "  Fusṭāṭ letter  "
    msgs = R.make_messages(decomposed, decomposed, system=decomposed)
    assert all(m["content"] == "Fus\u1e6d\u0101\u1e6d letter" for m in msgs)
    assert R.non_nfc_roles({"messages": msgs}) == []
    raw = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                        {"role": "assistant", "content": decomposed.strip()}]}
    assert R.non_nfc_roles(raw) == ["assistant"] and R.nfc(decomposed.strip()) == "Fus\u1e6d\u0101\u1e6d letter"


def test_write_jsonl_scrubs_target_raw(tmp_path: Path) -> None:
    """F4: ``meta.target_raw`` loses its project-name strings at write time (plain text and JSON targets); clean rows and
    other JSON objects are written unchanged.

    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    name, acro = codecs.decode("Sevrqoret", "rot_13"), codecs.decode("SWZF", "rot_13")  # F12: decoded at runtime
    cat_raw = json.dumps({"summary": f"A deed. See the {acro} images online.", "date": None})
    rows = [{"messages": [], "meta": {"row_id": "a", "target_raw": f"A deed. Images at the {name} site. Clean end."}},
            {"messages": [], "meta": {"row_id": "b", "target_raw": cat_raw}},
            {"messages": [], "meta": {"row_id": "c", "target_raw": "A clean raw target."}},
            {"pgpid": "1", "page_key": "x"}]
    f = tmp_path / "rows.jsonl"
    assert R.write_jsonl(f, rows) == 4
    back = R.load_jsonl(f)
    assert back[0]["meta"]["target_raw"] == "A deed. Clean end."
    assert json.loads(back[1]["meta"]["target_raw"]) == {"summary": "A deed.", "date": None}
    assert back[2]["meta"]["target_raw"] == "A clean raw target." and back[3] == {"pgpid": "1", "page_key": "x"}
    assert name.lower() not in f.read_text().lower() and acro.lower() not in f.read_text().lower()


def test_write_jsonl_nfc_target_raw(tmp_path: Path) -> None:
    """F12: a decomposed ``meta.target_raw`` is written in NFC; :func:`rows.non_nfc_roles` reports it before.

    :param tmp_path: Pytest temporary directory.
    :returns: None.
    """
    decomposed = "Fus" + "t\u0323" + "a\u0304" + "t\u0323 (ll. 3\u20134)"
    row = {"messages": [{"role": "assistant", "content": "Fus\u1e6d\u0101\u1e6d"}], "meta": {"target_raw": decomposed}}
    assert R.non_nfc_roles(row) == ["meta.target_raw"]
    f = tmp_path / "rows.jsonl"
    R.write_jsonl(f, [row])
    back = R.load_jsonl(f)[0]
    assert back["meta"]["target_raw"] == "Fus\u1e6d\u0101\u1e6d (ll. 3\u20134)" and R.non_nfc_roles(back) == []
