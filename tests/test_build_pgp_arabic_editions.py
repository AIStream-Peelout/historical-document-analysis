# File name: test_build_pgp_arabic_editions.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the Arabic-script training-page builder."""
import csv
import io
import json

import pytest
from datasets import load_from_disk
from PIL import Image

from src.datasets.evaluations.helper_eval_scripts import build_arabic_benchmark as bab
from src.finetuning.qwen_hebrew import build_documentary_grounding as bdg
from src.finetuning.qwen_hebrew import build_pgp_arabic_editions as bpa
from src.finetuning.qwen_hebrew.prompts import FRAGMENT_TRANSCRIBE_PROMPT

QUEUE_COLUMNS = ["rank", "tier", "pgpid", "shelfmark", "single_fragment", "canonical_id", "library", "doc_type", "side", "doc_date",
                 "iiif_urls", "suggested_iiif"]
RECTO = ["بسم الله الرحمن الرحيم", "هذا ما استاجر افرايم بن عالي الاسرائيلي من ابي الفتوح [بن سعيد]", "جميع الدار المعروفة به بمصر"]
VERSO = ["الى الشيخ ابو سعد اطال الله بقاه وادام عزه وتاييده", "من عبده ومملوكه يقبل الارض ..."]


def _queue(tmp_path, rows):
    path = tmp_path / "queue.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, {"single_fragment": "True", "tier": "1"}.get(c, "")) for c in QUEUE_COLUMNS})
    return path


def test_the_prompt_is_the_fragment_prompt_with_the_script_changed():
    assert "handwritten Arabic script" in bpa.ARABIC_FRAGMENT_PROMPT and "Hebrew" not in bpa.ARABIC_FRAGMENT_PROMPT
    assert bpa.ARABIC_FRAGMENT_PROMPT.splitlines()[1:] == FRAGMENT_TRANSCRIBE_PROMPT.splitlines()[1:]      # same instructions


def test_candidate_documents_keeps_what_may_be_trained_on(tmp_path):
    path = _queue(tmp_path, [
        {"pgpid": "1", "canonical_id": "A", "shelfmark": "T-S Ar.1.1", "iiif_urls": "https://lib/m1 ; https://lib/m1b", "side": "verso"},
        {"pgpid": "2", "canonical_id": "", "suggested_iiif": "https://lib/m2", "tier": "2"},
        {"pgpid": "3", "canonical_id": "Held", "iiif_urls": "https://lib/m3"},
        {"pgpid": "4", "canonical_id": "B", "iiif_urls": "https://lib/m4"},
        {"pgpid": "5", "canonical_id": "C", "iiif_urls": "https://lib/m5", "tier": "6"},
        {"pgpid": "6", "canonical_id": "D", "iiif_urls": "https://lib/m6", "single_fragment": "False"},
        {"pgpid": "7", "canonical_id": "E"},
    ])
    docs, skipped = bpa.candidate_documents(path, {"Held"}, {"4"})
    assert list(docs) == ["1", "2"] and docs["1"]["manifests"] == ["https://lib/m1", "https://lib/m1b"] and docs["1"]["pgp_side"] == "verso"
    assert docs["2"]["canonical_id"] == "" and docs["2"]["tier"] == "2"
    assert skipped == {"3": "held out by a benchmark", "4": "held out by a benchmark", "5": "no permitted image",
                       "6": "join: the edition covers several fragments", "7": "no library manifest"}


def test_best_sections_marks_losses_and_takes_the_fullest_edition():
    short = "Recto\n" + RECTO[0]
    full = "Recto\n" + "\n".join(RECTO) + "\nVerso\n" + "\n".join(VERSO)
    sections = bpa.best_sections([short, full])
    assert [side for side, _ in sections] == ["recto", "verso"]
    assert sections[0][1][1] == "هذا ما استاجر افرايم بن عالي الاسرائيلي من ابي الفتوح [...]" and sections[1][1][1].endswith("الارض [...]")
    assert bpa.best_sections([]) == [] and bpa.best_sections(["Recto\n[...]"]) == []


@pytest.mark.parametrize("value, side", [("verso", "verso"), ("Recto", "recto"), ("verso ; verso", "verso"), ("recto and verso", ""), ("", "")])
def test_single_side(value, side):
    assert bpa.single_side(value) == side


def test_assign_pages_only_where_the_record_says_which_image():
    both = [("recto", RECTO), ("verso", VERSO)]
    assert bpa.assign_pages(both, ["recto", "verso"]) == ({0: RECTO, 1: VERSO}, "")
    assert bpa.assign_pages(both, ["verso", "recto", ""]) == ({1: RECTO, 0: VERSO}, "")                  # labels, not positions
    assert bpa.assign_pages([("recto", RECTO[:1]), ("verso", VERSO), ("recto", RECTO[1:])], ["recto", "verso"])[0] == {0: RECTO, 1: VERSO}
    assert bpa.assign_pages(both, ["recto"]) == ({0: RECTO}, "0 images labelled verso")                # the verso was not photographed
    assert bpa.assign_pages(both, [""]) == ({}, "0 images labelled recto; 0 images labelled verso")       # one image cannot hold both sides
    assert bpa.assign_pages(both, ["recto", "verso", "recto", "verso"])[0] == {}                           # two folios: which recto?
    assert bpa.assign_pages([("verso", VERSO)], [""]) == ({0: VERSO}, "")                                 # one side, one image
    assert bpa.assign_pages([("verso", VERSO)], ["recto"]) == ({}, "0 images labelled verso")
    assert bpa.assign_pages([("", RECTO)], [""]) == ({0: RECTO}, "") and bpa.assign_pages([("", RECTO)], ["verso"]) == ({0: RECTO}, "")
    assert bpa.assign_pages([("", RECTO)], ["recto", "verso"]) == ({}, "edition without side labels and several images")
    assert bpa.assign_pages([("", RECTO)], ["recto", "verso"], pgp_side="verso") == ({1: RECTO}, "")         # PGP says which side
    assert bpa.assign_pages([("", RECTO)], ["recto", "verso"], pgp_side="recto and verso")[0] == {}
    assert bpa.assign_pages([("", RECTO), ("verso", VERSO)], ["recto", "verso"]) == ({}, "edition has text outside its side labels")
    assert bpa.assign_pages([], ["recto"]) == ({}, "no edition text") and bpa.assign_pages(both, []) == ({}, "no image")


def test_resolve_shared_images_keeps_one_page_per_image():
    first = {"url": "u1", "text": "\n".join(RECTO), "pgpid": "1"}
    again = {"url": "u1", "text": "\n".join(RECTO + VERSO[:1]), "pgpid": "2"}            # a second, fuller record of the same text
    other = {"url": "u2", "text": "\n".join(RECTO), "pgpid": "3"}
    stranger = {"url": "u2", "text": "\n".join(VERSO), "pgpid": "4"}                     # a different document on the same side
    alone = {"url": "u3", "text": "\n".join(VERSO), "pgpid": "5"}
    kept, dropped = bpa.resolve_shared_images([first, again, other, stranger, alone])
    assert [page["pgpid"] for page in kept] == ["2", "5"]
    assert dropped == {"another edition of the same page is fuller": 1, "several documents on one image": 2}
    assert bpa.resolve_shared_images([]) == ([], {})


def test_split_of_is_stable_and_close_to_the_share():
    assert bpa.split_of("19642", 0.05) == bpa.split_of("19642", 0.05)
    val = sum(bpa.split_of(str(n), 0.05) == "val" for n in range(4000))
    assert 150 < val < 250 and bpa.split_of("19642", 0.0) == "train_page" and bpa.split_of("19642", 1.0) == "val"


def _manifest(labels):
    return {"sequences": [{"canvases": [
        {"label": label, "width": 1200, "height": 1600,
         "images": [{"resource": {"@id": f"https://img/{name}-{n}/full/full/0/default.jpg", "service": {"@id": f"https://img/{name}-{n}"}}}]}
        for n, (name, label) in enumerate(labels)]}]}


def _jpeg(colour, size=(600, 800)):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_build_writes_side_matched_pages_and_reports_the_rest(tmp_path, monkeypatch):
    both = "Recto\n" + "\n".join(RECTO) + "\nVerso\n" + "\n".join(VERSO)
    documents = {                                   # pgpid: (queue fields, edition, canvas labels)
        "10": ({"canonical_id": "Two_sided", "shelfmark": "T-S Ar.1.10"}, both, ["1r", "1v"]),
        "11": ({"canonical_id": "", "shelfmark": "ENA 1.11", "tier": "2"}, "\n".join(VERSO), ["1"]),
        "12": ({"canonical_id": "Unlabelled", "shelfmark": "T-S Ar.1.12", "side": "verso"}, "\n".join(RECTO), ["1r", "1v"]),
        "13": ({"canonical_id": "Ambiguous", "shelfmark": "T-S Ar.1.13"}, "\n".join(RECTO), ["1r", "1v"]),
        "14": ({"canonical_id": "Hebrew_benchmark_doc", "shelfmark": "T-S 8.4"}, both, ["1r", "1v"]),
        "15": ({"canonical_id": "Mixed", "shelfmark": "T-S Ar.1.15"}, "\n".join(RECTO) + "\n" + "לכבוד מרנא ורבנא אדוננו " * 3, ["1r"]),
        "16": ({"canonical_id": "Short", "shelfmark": "T-S Ar.1.16"}, "[بسم الله الرحمن الرحيم] الحمد لله", ["1r"]),
        "17": ({"canonical_id": "Held", "shelfmark": "T-S Ar.1.17"}, both, ["1r", "1v"]),
        "18": ({"canonical_id": "Volume", "shelfmark": "Bodl. 1"}, both, [str(n) for n in range(40)]),
        "19": ({"canonical_id": "Trained_elsewhere", "shelfmark": "T-S Ar.1.19"}, "Verso\n" + "\n".join(VERSO), ["1r", "1v"]),
        "20": ({"canonical_id": "Scroll", "shelfmark": "T-S 28.8"}, "\n".join(RECTO * 8), ["1"]),
        "21": ({"canonical_id": "Twice_in_PGP", "shelfmark": "CUL Or.1080 J7"}, "\n".join(RECTO), ["1"]),
        "22": ({"canonical_id": "Twice_in_PGP", "shelfmark": "CUL Or.1080 J7", "manifest": "21"}, "\n".join(RECTO + VERSO[:1]), ["1"]),
    }
    monkeypatch.setattr(bab, "QUEUE_CSV", _queue(tmp_path, [dict(fields, pgpid=pid, iiif_urls=f"https://lib/m{fields.get('manifest', pid)}")
                                                           for pid, (fields, _, _) in documents.items()]))
    footnotes = tmp_path / "footnotes.csv"
    with open(footnotes, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["document_id", "doc_relation", "content"])
        writer.writeheader()
        for pid, (_, edition, _) in documents.items():
            writer.writerow({"document_id": pid, "doc_relation": "Edition", "content": edition})
    monkeypatch.setattr(bab, "FOOTNOTES_CSV", footnotes)
    fetched = []

    def fake_get(url, timeout=90):
        fetched.append(url)
        if url.startswith("https://lib/m"):
            pid = url.rsplit("m", 1)[1]
            return json.dumps(_manifest([(pid, label) for label in documents[pid][2]])).encode()
        return _jpeg("white", (150, 2000)) if "/20-" in url else _jpeg("white" if "-0/" in url else "grey")

    monkeypatch.setattr(bab, "http_get", fake_get)
    out = tmp_path / "pgp_arabic"
    stats = bpa.build(out, delay_s=0, val_share=0.0, held_out=({"Held"}, set()),
                      benchmark_index=bdg.build_benchmark_index(["Hebrew_benchmark_doc"]), in_training={"Trained_elsewhere": "clean_v2_ids"})
    assert stats["documents"] == 5 and stats["pages"] == 6 and stats["rows"] == {"train_page": 6, "val": 0}
    assert stats["skipped_documents"] == {
        "13": "edition without side labels and several images", "14": "fragment of a benchmark (benchmark_id)", "15": "mixed script",
        "16": "fewer than 40 visible Arabic letters", "17": "held out by a benchmark", "18": "whole volume (40 canvases)"}
    assert stats["page_skips"] == {"text too dense for the image": 1,                       # the scroll: 0.3 megapixels for 700 letters
                                   "another edition of the same page is fuller": 1}
    assert stats["documents_already_in_another_training_set"] == 1 and stats["by_tier"] == {"1": 5, "2": 1}
    assert "https://lib/m17" not in fetched                               # nothing is requested for a held-out document
    rows = {r["stem"]: r for r in load_from_disk(str(out))["train_page"]}
    assert sorted(rows) == ["pgpar_10_0_page", "pgpar_10_1_page", "pgpar_11_0_page", "pgpar_12_1_page", "pgpar_19_1_page",
                            "pgpar_22_0_page"]
    assert rows["pgpar_22_0_page"]["answer"].splitlines()[-1] == VERSO[0]                 # the fuller of PGP's two records
    recto = rows["pgpar_10_0_page"]
    assert recto["answer"].splitlines() == [RECTO[0], "هذا ما استاجر افرايم بن عالي الاسرائيلي من ابي الفتوح [...]", RECTO[2]]
    assert recto["question"] == bpa.ARABIC_FRAGMENT_PROMPT and recto["task"] == "fragment_transcribe" and recto["section"] == "pgp_arabic_page"
    assert recto["label_source"] == "pgp_edition" and (recto["image_width"], recto["image_height"]) == (600, 800)
    assert recto["target_chars"] == len(recto["answer"]) and recto["image"].size == (600, 800)
    assert rows["pgpar_10_1_page"]["answer"].splitlines()[0] == VERSO[0] and rows["pgpar_12_1_page"]["answer"].startswith(RECTO[0])
    manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(m["pgpid"], m["image_index"], m["side"]) for m in manifest] == [("10", 0, "recto"), ("10", 1, "verso"), ("11", 0, ""),
                                                                              ("12", 1, "verso"), ("19", 1, "verso"), ("22", 0, "")]
    assert json.loads((out / "stats.json").read_text(encoding="utf-8"))["pages"] == 6
    assert sorted(p.name for p in (tmp_path / "pgp_arabic_images").iterdir()) == ["10_0.jpg", "10_1.jpg", "11_0.jpg", "12_1.jpg", "19_1.jpg",
                                                                                "20_0.jpg", "22_0.jpg"]
    everything_val = bpa.build(tmp_path / "all_val", delay_s=0, val_share=1.0, held_out=({"Held"}, set()),
                               benchmark_index=bdg.build_benchmark_index(["Hebrew_benchmark_doc"]), in_training={"Trained_elsewhere": "clean_v2_ids"})
    assert everything_val["rows"] == {"train_page": 1, "val": 5}            # a fragment trained on elsewhere never validates
