# File name: test_sealed_benchmark_runners.py
# Date: 10/12/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""The PGP and religious runners / scorers read another spec (the sealed set) through flags, defaults unchanged."""
import asyncio
import csv
import json
import sys
from pathlib import Path

import pytest

from src.datasets.evaluations.helper_eval_scripts import run_religious_benchmark as rrb
from src.datasets.evaluations.helper_eval_scripts import score_genizah_offline as sgo
from src.finetuning.qwen_hebrew.eval_harness import run_pgp131_v19b as pgp

GT = ("ידוע יהיה לכם כי הגיע המכתב ושמחנו מאד בשלומכם ובשלום ביתכם "
      "ועתה בקשתי ממך לשלוח את הספרים אשר השארתי בבית הכנסת הגדול")


def _spec(tmp_path: Path, docs: list) -> Path:
    """Write a spec file.

    :param tmp_path: Folder.
    :param docs: Spec documents.
    :return: Its path.
    """
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"docs": docs}, ensure_ascii=False), encoding="utf-8")
    return path


def _output(root: Path, doc_id: str, model: str, text: str) -> None:
    """Write one model output.

    :param root: Output root.
    :param doc_id: Document.
    :param model: Output stem.
    :param text: Transcription.
    """
    (root / doc_id).mkdir(parents=True, exist_ok=True)
    (root / doc_id / f"{model}.txt").write_text(text, encoding="utf-8")


def test_scorer_reads_another_spec_and_output_root(tmp_path, monkeypatch):
    spec = _spec(tmp_path, [{"doc_id": "D1", "gt": GT}, {"doc_id": "D2", "gt": GT}])
    assert [d["doc_id"] for d in sgo.load_benchmark("verified", spec)] == ["D1", "D2"]
    default = sgo.load_benchmark("verified") if (sgo._BENCH_DIR / "genizah_test_v1_verified.json").exists() else None
    if default is not None:
        assert len(default) == 131                                   # the default is still PGP-131
    outputs = tmp_path / "outputs"
    _output(outputs, "D1", "model_a", GT)                            # D2 has no output: not scored
    tags = tmp_path / "tags.json"
    tags.write_text(json.dumps({"D1": {"bucket": "hebrew"}}), encoding="utf-8")
    out_dir = tmp_path / "scores"
    monkeypatch.setattr(sys, "argv", ["score", "--spec", str(spec), "--outputs-dir", str(outputs),
                                      "--out-dir", str(out_dir), "--script-tags", str(tags), "--no-wandb"])
    sgo.main()
    rows = list(csv.DictReader(open(out_dir / "genizah_offline_long.csv", encoding="utf-8")))
    assert [(r["fragment_id"], r["model"], r["script_bucket"]) for r in rows] == [("D1", "model_a", "hebrew")]
    assert float(rows[0]["cer"]) == 0.0 and rows[0]["failure_mode"] == "substantive"


def test_religious_runner_scores_another_spec_next_to_it(tmp_path, monkeypatch):
    doc = {"doc_id": "KTIV_990000000000000001_FL1", "gt": GT, "is_talmud": True, "n_columns": 2, "genre": "talmud/halakha"}
    spec = _spec(tmp_path, [doc])
    outputs = tmp_path / "raw_outputs"
    _output(outputs, doc["doc_id"], "qwen3_vl_8b_heb_test", GT)
    rows = rrb.score([doc], outputs)
    assert [(r["model"], r["columns"], r["cer"]) for r in rows] == [("test", "multi", 0.0)]
    default_csv = rrb._BENCH / "religious_scores_long.csv"
    before = default_csv.stat().st_mtime if default_csv.exists() else None
    monkeypatch.setattr(sys, "argv", ["run", "--score-only", "--spec", str(spec), "--outputs-dir", str(outputs)])
    rrb.main()
    assert (tmp_path / "religious_scores_long.csv").exists() and (tmp_path / "religious_scores_summary.csv").exists()
    assert (default_csv.stat().st_mtime if default_csv.exists() else None) == before   # religious-140 results untouched
    assert rrb._OUT == rrb._BENCH / "raw_outputs"                                      # defaults unchanged


def test_pgp_runner_flags_default_to_pgp131():
    args = pgp.parse_args([])
    assert args.model == pgp.DEFAULT_MODEL and args.spec == pgp.BENCH / "genizah_test_v1_verified.json"
    assert args.images_dir == pgp.BENCH / "images" and args.outputs_dir == pgp.OUT
    args = pgp.parse_args(["qwen3-vl-8b-heb-v21b-step1200", "--spec", "s.json", "--images-dir", "i", "--outputs-dir", "o"])
    assert (args.model, args.spec, args.images_dir, args.outputs_dir) == \
        ("qwen3-vl-8b-heb-v21b-step1200", Path("s.json"), Path("i"), Path("o"))


def test_pgp_runner_reads_the_given_spec_images_and_outputs(tmp_path, monkeypatch):
    spec = _spec(tmp_path, [{"doc_id": "D1"}, {"doc_id": "D2"}])
    images = tmp_path / "images"
    images.mkdir()
    (images / "D1.jpg").write_bytes(b"jpg")                          # D2 has no image: reported, skipped
    calls = []

    async def health():
        """Fake LM Studio model list.

        :return: Served models.
        """
        return ["m-1"]

    async def transcribe(model, image, prompt, max_tokens):
        """Fake read.

        :param model: Model id.
        :param image: Image path.
        :param prompt: Prompt.
        :param max_tokens: Token cap.
        :return: Text.
        """
        calls.append((model, Path(image).name, max_tokens))
        return "קריאה"

    monkeypatch.setattr(pgp, "check_lm_studio_health", health)
    monkeypatch.setattr(pgp, "transcribe_with_lm_studio", transcribe)
    args = pgp.parse_args(["m-1", "--spec", str(spec), "--images-dir", str(images), "--outputs-dir", str(tmp_path / "out")])
    asyncio.run(pgp.main(args))
    assert calls == [("m-1", "D1.jpg", pgp.MAX_TOKENS)]
    assert (tmp_path / "out" / "D1" / "m_1.txt").read_text(encoding="utf-8") == "קריאה"
    asyncio.run(pgp.main(args))                                      # resumable: D1 is not read again
    assert len(calls) == 1


def test_pgp_runner_refuses_an_unserved_model(tmp_path, monkeypatch):
    spec = _spec(tmp_path, [{"doc_id": "D1"}])

    async def health():
        """Fake LM Studio model list.

        :return: Served models.
        """
        return ["other"]

    monkeypatch.setattr(pgp, "check_lm_studio_health", health)
    args = pgp.parse_args(["m-1", "--spec", str(spec), "--outputs-dir", str(tmp_path / "out")])
    with pytest.raises(RuntimeError):
        asyncio.run(pgp.main(args))
