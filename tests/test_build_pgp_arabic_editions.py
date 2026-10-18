# File name: test_build_pgp_arabic_editions.py
# Date: 10/2/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for the Arabic-script training-page builder."""
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from datasets import load_from_disk
from PIL import Image

from src.datasets.evaluations.helper_eval_scripts import build_arabic_benchmark as bab
from src.datasets.indexing import arabic_scrape_queue as asq
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


# --- side assignments (v2.4a) -------------------------------------------------------------------------------------------

READER = "claude-opus-5-5"


def _sides(edition: str) -> Dict[str, List[str]]:
    """The edition's cleaned lines by side, as the builder fingerprints them."""
    return bpa.sides_of(bpa.best_sections([edition]))


def _canvases(labels: List[str], pid: str = "9") -> List[Dict[str, Any]]:
    """Images as ``bab.planned_images`` returns them, one per canvas label."""
    return [{"url": f"https://img/{pid}-{n}/full/full/0/default.jpg", "label": label, "side": bab.side_of_label(label), "source": "m"}
            for n, label in enumerate(labels)]


def _assignment(sides: Dict[str, List[str]], canvases: List[Dict[str, Any]], pages: List[Tuple[int, str]], decision: str = "accept",
                fingerprint: Optional[str] = None) -> Dict[str, Any]:
    """A side matcher's decision for document 9: ``pages`` are ``(canvas index, assign)``."""
    return {"pgpid": "9", "edition_fingerprint": fingerprint or bpa.edition_fingerprint(sides), "decision": decision, "reader": READER,
            "pages": [{"canvas_index": index, "image_url": canvases[index]["url"], "assign": assign} for index, assign in pages],
            "evidence": {"f1": 0.9}}


def test_edition_fingerprint_names_the_cleaned_text_and_its_sides():
    labelled = _sides("Recto\n" + "\n".join(RECTO) + "\nVerso\n" + "\n".join(VERSO))
    expected = "@recto\n" + "\n".join(labelled["recto"]) + "\n@verso\n" + "\n".join(labelled["verso"])
    assert bpa.edition_fingerprint(labelled) == hashlib.sha1(expected.encode("utf-8")).hexdigest()
    unlabelled = _sides("\n".join(RECTO + VERSO))
    assert list(unlabelled) == [""] and bpa.edition_fingerprint(unlabelled) != bpa.edition_fingerprint(labelled)   # same lines, no sides
    assert bpa.edition_fingerprint(_sides("\n".join(RECTO))) != bpa.edition_fingerprint(unlabelled)                 # other text
    assert bpa.sides_of([("recto", ["a"]), ("verso", ["b"]), ("recto", ["c"])]) == {"recto": ["a", "c"], "verso": ["b"]}


@pytest.mark.parametrize("problem, side", [("edition without side labels and several images", True),
                                           ("edition has text outside its side labels", True),
                                           ("0 images labelled recto; 2 images labelled verso", True), ("1 images labelled verso", True),
                                           ("no edition text", False), ("no image", False), ("", False)])
def test_is_side_problem(problem, side):
    assert bpa.is_side_problem(problem) is side


def test_apply_side_assignment_all_puts_the_unlabelled_edition_on_one_image():
    sides, canvases = _sides("\n".join(RECTO)), _canvases(["00000001.tif", "00000002.tif"])
    assert bpa.apply_side_assignment(sides, canvases, _assignment(sides, canvases, [(1, "all")])) == ({1: sides[""]}, "")
    split = bpa.apply_side_assignment(sides, canvases, _assignment(sides, canvases, [(0, "all"), (1, "all")]))
    assert split == ({}, "side assignment invalid: 'all' must be the only page")                    # never split across images
    as_sides = bpa.apply_side_assignment(sides, canvases, _assignment(sides, canvases, [(0, "recto"), (1, "verso")]))
    assert as_sides == ({}, "side assignment invalid: text outside the side labels can only be assigned with 'all'")
    labelled_image = _canvases(["1r", "00000002.tif"])                                            # an unlabelled edition takes any image
    assert bpa.apply_side_assignment(sides, labelled_image, _assignment(sides, labelled_image, [(0, "all")])) == ({0: sides[""]}, "")


@pytest.mark.parametrize("edition", [
    "\n".join(RECTO) + "\nVerso\n" + "\n".join(VERSO),                 # unlabelled text (an implicit recto) + a verso: two sides
    "Recto\n" + "\n".join(RECTO) + "\nVerso\n" + "\n".join(VERSO),     # both sides
    "Verso\n" + "\n".join(VERSO),                                       # one labelled side: placed with 'verso', where labels are checked
])
def test_apply_side_assignment_all_is_only_for_an_edition_without_side_labels(edition):
    sides = _sides(edition)
    for labels in (["00000001.tif", "00000002.tif"], ["1r"]):
        canvases = _canvases(labels)
        assert bpa.apply_side_assignment(sides, canvases, _assignment(sides, canvases, [(0, "all")])) == \
            ({}, "side assignment invalid: 'all' is only for an edition without side labels")


def test_apply_side_assignment_labels_win_for_a_one_sided_edition():
    verso_only, recto_image = _sides("Verso\n" + "\n".join(VERSO)), _canvases(["1r"])
    assert bpa.assign_pages(bpa.best_sections(["Verso\n" + "\n".join(VERSO)]), ["recto"]) == ({}, "0 images labelled verso")
    for assign, problem in (("verso", "an image label names the other side"),     # the only image is labelled recto:
                            ("all", "'all' is only for an edition without side labels")):    # no verso text goes on it
        assert bpa.apply_side_assignment(verso_only, recto_image, _assignment(verso_only, recto_image, [(0, assign)])) == \
            ({}, f"side assignment invalid: {problem}")
    figgy = _canvases(["00000001.tif", "00000002.tif"])
    assert bpa.apply_side_assignment(verso_only, figgy, _assignment(verso_only, figgy, [(1, "verso")])) == ({1: verso_only["verso"]}, "")


def test_apply_side_assignment_maps_labelled_sections_to_unnamed_images():
    both = _sides("Recto\n" + "\n".join(RECTO) + "\nVerso\n" + "\n".join(VERSO))
    canvases = _canvases(["00000001.tif", "00000002.tif", "colour chart"])
    assert bpa.apply_side_assignment(both, canvases, _assignment(both, canvases, [(1, "recto"), (0, "verso")])) == \
        ({1: both["recto"], 0: both["verso"]}, "")
    assert bpa.apply_side_assignment(both, canvases, _assignment(both, canvases, [(0, "recto")])) == \
        ({0: both["recto"]}, "side match left verso unassigned")                                         # the verso was not photographed
    assert bpa.apply_side_assignment(both, canvases, _assignment(both, canvases, [(0, "recto"), (1, "recto")]))[1] == \
        "side assignment invalid: a side is assigned to several images"
    assert bpa.apply_side_assignment(both, canvases, _assignment(both, canvases, [(0, "recto"), (0, "verso")]))[1] == \
        "side assignment invalid: an image is assigned twice"
    assert bpa.apply_side_assignment(both, canvases, _assignment(both, canvases, [(0, "margin")]))[1].startswith("side assignment invalid")
    verso_only = _sides("Verso\n" + "\n".join(VERSO))
    assert bpa.apply_side_assignment(verso_only, canvases, _assignment(verso_only, canvases, [(0, "recto")]))[1] == \
        "side assignment invalid: the edition has no recto"
    labelled = _canvases(["1v", "00000002.tif"])
    assert bpa.apply_side_assignment(both, labelled, _assignment(both, labelled, [(0, "recto"), (1, "verso")]))[1] == \
        "side assignment invalid: an image label names the other side"
    assert bpa.apply_side_assignment(both, canvases, _assignment(both, canvases, []))[1] == "side assignment invalid: no page"


def test_apply_side_assignment_stale_or_dropped_gives_no_page():
    sides, canvases = _sides("\n".join(RECTO)), _canvases(["00000001.tif", "00000002.tif"])
    assert bpa.apply_side_assignment(sides, canvases, _assignment(sides, canvases, [(1, "all")], fingerprint="0" * 40)) == \
        ({}, "side assignment stale")
    changed = _assignment(sides, canvases, [(1, "all")])
    changed["pages"][0]["image_url"] = "https://img/other/full/full/0/default.jpg"
    assert bpa.apply_side_assignment(sides, canvases, changed) == ({}, "side assignment stale (images changed)")
    out_of_range = _assignment(sides, canvases, [(1, "all")])
    out_of_range["pages"][0]["canvas_index"] = 5
    assert bpa.apply_side_assignment(sides, canvases, out_of_range)[1] == "side assignment stale (images changed)"
    assert bpa.apply_side_assignment(sides, canvases, _assignment(sides, canvases, [(1, "all")], decision="drop")) == ({}, "side match dropped")
    stale_drop = _assignment(sides, canvases, [], decision="drop", fingerprint="0" * 40)              # dropped under an earlier text
    assert bpa.apply_side_assignment(sides, canvases, stale_drop) == ({}, "side assignment stale")


def test_load_side_assignments_reads_the_matcher_file(tmp_path):
    sides, canvases = _sides("\n".join(RECTO)), _canvases(["a", "b"])
    first = dict(_assignment(sides, canvases, [(0, "all")]), pgpid=9)
    later = _assignment(sides, canvases, [(1, "all")])
    other = dict(_assignment(sides, canvases, [], decision="drop"), pgpid="10")
    path = tmp_path / "side_assignments.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in (first, other, later)) + "\n\n", encoding="utf-8")
    loaded = bpa.load_side_assignments(path)
    assert sorted(loaded) == ["10", "9"] and loaded["9"]["pages"][0]["canvas_index"] == 1 and loaded["10"]["decision"] == "drop"
    for broken, message in (({k: v for k, v in later.items() if k != "reader"}, "missing reader"),
                            (dict(later, decision="maybe"), "decision 'maybe'"), (dict(later, pages=[{"canvas_index": 0}]), "pages")):
        path.write_text(json.dumps(broken) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            bpa.load_side_assignments(path)


def test_unassigned_record_has_the_shared_format():
    sides, canvases = _sides("Recto\n" + "\n".join(RECTO)), _canvases(["00000001.tif", "00000002.tif"])
    doc = {"canonical_id": "Cambridge_X", "shelfmark": "T-S Ar.1.9", "library": "Cambridge University Library", "tier": "1"}
    record = bpa.unassigned_record("9", doc, sides, canvases, "1 images labelled recto")
    assert record == {"pgpid": "9", "shelfmark": "T-S Ar.1.9", "canonical_id": "Cambridge_X", "library": "Cambridge University Library",
                      "edition_fingerprint": bpa.edition_fingerprint(sides), "sections": {"recto": sides["recto"]},
                      "canvases": [{"index": 0, "label": "00000001.tif", "image_url": canvases[0]["url"]},
                                   {"index": 1, "label": "00000002.tif", "image_url": canvases[1]["url"]}],
                      "reason": "1 images labelled recto"}
    assert bpa.edition_fingerprint(json.loads(json.dumps(record))["sections"]) == record["edition_fingerprint"]   # recomputable


def test_seed_cache_copies_an_identical_url_instead_of_downloading(tmp_path, monkeypatch):
    v1_images = tmp_path / "v1_images"
    v1_images.mkdir()
    (v1_images / "7_0.jpg").write_bytes(_jpeg("black", (700, 900)))
    (tmp_path / "v1_manifest.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"image_url": "https://img/7-0/full/full/0/default.jpg", "image_path": "/old/place/v1_images/7_0.jpg"},
        {"image_url": "https://img/8-0/full/full/0/default.jpg", "image_path": "/old/place/v1_images/8_0.jpg"})) + "\n", encoding="utf-8")
    seed = bpa.seed_cache_index(v1_images, tmp_path / "v1_manifest.jsonl")
    assert seed == {"https://img/7-0/full/full/0/default.jpg": v1_images / "7_0.jpg"}            # 8_0.jpg is not on disk
    fetched = []
    monkeypatch.setattr(bab, "http_get", lambda url, timeout=90: fetched.append(url) or _jpeg("white"))
    cache = tmp_path / "v2_images"
    cache.mkdir()
    assert bpa.fetch_image("https://img/7-0/full/full/0/default.jpg", cache / "7_0.jpg", seed) == "seed"
    assert bpa.fetch_image("https://img/7-0/full/full/0/default.jpg", cache / "7_0.jpg", seed) == "cache"
    assert bpa.fetch_image("https://img/8-0/full/full/0/default.jpg", cache / "8_0.jpg", seed) == "download"
    assert fetched == ["https://img/8-0/full/full/0/default.jpg"] and Image.open(cache / "7_0.jpg").size == (700, 900)
    assert (v1_images / "7_0.jpg").read_bytes() == (cache / "7_0.jpg").read_bytes() and sorted(p.name for p in v1_images.iterdir()) == ["7_0.jpg"]


def _published(tmp_path: Path) -> Tuple[Path, Tuple[Path, Path, Path]]:
    """A stand-in for the NAS datasets directory holding a v1 build, its image cache and its images-once export."""
    datasets = tmp_path / "datasets"
    v1 = datasets / "pgp_arabic_editions_v1"
    protected = (v1, datasets / "pgp_arabic_editions_v1_images", datasets / "pgp_arabic_editions_v1_images_once")
    for directory in protected:
        directory.mkdir(parents=True)
        (directory / "keep.txt").write_text("published", encoding="utf-8")
    return datasets, protected


def _flip_case(path: Path, part: str) -> Path:
    """``path`` with the component ``part`` spelled in upper case."""
    return Path(*[p.upper() if p == part else p for p in path.parts])


def test_default_out_is_v2_and_the_real_v1_is_protected():
    args = bpa.build_parser().parse_args([])
    assert args.out == bpa.DEFAULT_OUT and bpa.DEFAULT_OUT.name == "pgp_arabic_editions_v2" and bpa.DEFAULT_OUT != bpa.V1_OUT
    assert bpa.V1_OUT.name == "pgp_arabic_editions_v1" and args.image_cache is None and args.pgp_dir is None and args.queue_csv is None
    assert [p.name for p in bpa.PROTECTED_OUTPUTS] == ["pgp_arabic_editions_v1", "pgp_arabic_editions_v1_images",
                                                      "pgp_arabic_editions_v1_images_once"]
    assert {p.parent for p in bpa.PROTECTED_OUTPUTS} == {bab.NAS_DATASETS}
    # check_outputs only compares paths (and stats the ones that exist): it never writes
    for refused in (bpa.V1_OUT, bpa.V1_OUT / "unassigned.jsonl", bab.NAS_DATASETS, bab.NAS_DATASETS.parent, Path("/"),
                    bpa.V1_OUT.with_name("PGP_Arabic_Editions_v1"), bpa.V1_OUT.with_name("pgp_arabic_editions_v1_IMAGES")):
        with pytest.raises(ValueError, match="must not be overwritten"):
            bpa.check_outputs([refused])
    bpa.check_outputs([bpa.DEFAULT_OUT, bpa.DEFAULT_OUT.with_name(bpa.DEFAULT_OUT.name + "_images"), None])


def test_check_outputs_refuses_v1_its_contents_its_ancestors_and_other_spellings(tmp_path):
    datasets, protected = _published(tmp_path)
    v1 = protected[0]
    (tmp_path / "link_to_v1").symlink_to(v1)
    (tmp_path / "link_to_datasets").symlink_to(datasets)
    refused = [v1, v1 / "unassigned.jsonl", v1 / "new" / "dir", protected[1], protected[2] / "x.jpg",
               datasets, tmp_path, Path("/"),                                              # ancestors: save_dataset would replace them
               _flip_case(v1, v1.name), _flip_case(v1 / "x", "datasets"), _flip_case(datasets, "datasets"),
               tmp_path / "link_to_v1", tmp_path / "link_to_datasets" / v1.name, tmp_path / "link_to_datasets"]
    for path in refused:
        with pytest.raises(ValueError, match="must not be overwritten"):
            bpa.check_outputs([tmp_path / "fine", None, path], protected)
    bpa.check_outputs([datasets / "pgp_arabic_editions_v2", datasets / "pgp_arabic_editions_v2_images", datasets / "pgp_arabic_editions_v1x",
                       tmp_path / "elsewhere" / "unassigned.jsonl", None], protected)
    assert all((directory / "keep.txt").read_text(encoding="utf-8") == "published" for directory in protected)


def test_check_outputs_compares_existing_paths_by_file_identity(tmp_path, monkeypatch):
    datasets, protected = _published(tmp_path)
    if not _flip_case(datasets, "datasets").exists():
        pytest.skip("case-sensitive filesystem: no other spelling of the same directory")
    monkeypatch.setattr(bpa, "path_key", lambda path: Path(path).resolve().parts)     # spelling alone no longer matches
    for path in (_flip_case(protected[0], protected[0].name), _flip_case(protected[0] / "x", "datasets"), _flip_case(datasets, "datasets")):
        with pytest.raises(ValueError, match="must not be overwritten"):
            bpa.check_outputs([path], protected)
    bpa.check_outputs([datasets / "pgp_arabic_editions_v2", _flip_case(datasets / "pgp_arabic_editions_v2", "datasets")], protected)


def _forbid_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every step of ``build`` after its checks fail loudly, so a test of the checks can never build anything."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        """Stand-in for a step that must not run."""
        raise AssertionError("build went past its checks")

    for module, name in ((bab, "http_get"), (bab, "save_image"), (bab, "training_documents"), (bab, "load_editions"),
                         (bpa, "save_dataset"), (bpa, "candidate_documents"), (bpa, "registered_benchmark_documents"),
                         (bdg, "load_benchmark_index")):
        monkeypatch.setattr(module, name, forbidden)


def test_build_refuses_v1_before_doing_anything(tmp_path, monkeypatch):
    datasets, protected = _published(tmp_path)
    monkeypatch.setattr(bpa, "PROTECTED_OUTPUTS", protected)                       # the default of check_outputs, read when called
    _forbid_work(monkeypatch)
    queue = _queue(tmp_path, [])
    for out, kwargs in ((protected[0], {}), (datasets, {}), (_flip_case(protected[0], protected[0].name), {}),
                        (tmp_path / "v2", {"image_cache": protected[1]}), (tmp_path / "v2", {"emit_unassigned": protected[0] / "unassigned.jsonl"}),
                        (datasets / "v2", {"image_cache": _flip_case(protected[1], protected[1].name)})):
        with pytest.raises(ValueError, match="must not be overwritten"):
            bpa.build(out, held_out=(set(), set()), benchmark_index=bdg.build_benchmark_index([]), in_training={}, queue_csv=queue,
                      footnotes_csv=tmp_path / "footnotes.csv", **kwargs)
    assert sorted(p.name for p in datasets.iterdir()) == sorted(p.name for p in protected)            # nothing new, nothing removed
    assert all(sorted(p.name for p in directory.iterdir()) == ["keep.txt"] for directory in protected)
    assert not (tmp_path / "v2_images").exists() and not (tmp_path / "datasets_images").exists()


def test_main_threads_the_flags_to_build(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(bpa, "build", lambda out, **kwargs: calls.update(out=out, **kwargs) or {"pages": 0})
    v1_images = tmp_path / "v1_images"
    v1_images.mkdir()
    (v1_images / "1_0.jpg").write_bytes(_jpeg("white"))
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"image_url": "https://u/1", "image_path": "x/1_0.jpg"}) + "\n", encoding="utf-8")
    (tmp_path / "assign.jsonl").write_text(json.dumps({"pgpid": "1", "edition_fingerprint": "f", "decision": "drop", "reader": READER,
                                                       "pages": []}) + "\n", encoding="utf-8")
    bpa.main(["--out", str(tmp_path / "v2"), "--pgp-dir", str(tmp_path / "snapshot"), "--queue-csv", str(tmp_path / "q.csv"),
              "--image-cache", str(tmp_path / "cache"), "--seed-cache-from", str(v1_images), str(tmp_path / "manifest.jsonl"),
              "--side-assignments", str(tmp_path / "assign.jsonl"), "--emit-unassigned", str(tmp_path / "unassigned.jsonl"), "--limit", "3"])
    assert calls["out"] == tmp_path / "v2" and calls["footnotes_csv"] == tmp_path / "snapshot" / "footnotes.csv"
    assert calls["queue_csv"] == tmp_path / "q.csv" and calls["image_cache"] == tmp_path / "cache" and calls["limit"] == 3
    assert calls["seed_cache"] == {"https://u/1": v1_images / "1_0.jpg"} and list(calls["side_assignments"]) == ["1"]
    assert calls["emit_unassigned"] == tmp_path / "unassigned.jsonl"
    bpa.main([])
    assert calls["out"] == bpa.DEFAULT_OUT and calls["footnotes_csv"] is None and calls["queue_csv"] is None
    assert calls["seed_cache"] is None and calls["side_assignments"] is None and calls["emit_unassigned"] is None


def _write_summary(queue: Path, pgp_dir: Optional[Path], commit: str) -> None:
    """The summary ``arabic_scrape_queue`` writes next to a queue, recording the export it was built from."""
    queue.with_name(asq.OUT_SUMMARY.name).write_text(json.dumps({"pgp_dir": str(pgp_dir) if pgp_dir else "", "pgp_export_commit": commit,
                                                                 "listed": 0}), encoding="utf-8")


def _write_inputs(tmp_path: Path, documents: Dict[str, Tuple[Dict[str, str], str, List[str]]]) -> Tuple[Path, Path]:
    """The queue CSV (with its summary) and a pinned-snapshot-like PGP directory (footnotes.csv + COMMIT) for ``documents``."""
    queue = _queue(tmp_path, [dict(fields, pgpid=pid, iiif_urls=f"https://lib/m{pid}") for pid, (fields, _, _) in documents.items()])
    pgp_dir = tmp_path / "pgp_export"
    pgp_dir.mkdir(exist_ok=True)
    with open(pgp_dir / "footnotes.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["document", "document_id", "doc_relation", "content"])
        writer.writeheader()
        for pid, (_, edition, _) in documents.items():
            writer.writerow({"document": f"doc {pid}", "document_id": pid, "doc_relation": "Digital Edition", "content": edition})
    (pgp_dir / "COMMIT").write_text("b2343a17\n", encoding="utf-8")
    _write_summary(queue, pgp_dir, "b2343a17")
    return queue, pgp_dir


def test_check_queue_export_requires_the_queue_and_the_editions_from_one_export(tmp_path):
    pinned, unpinned = tmp_path / "snapshot", tmp_path / "raw"
    pinned.mkdir()
    unpinned.mkdir()
    (pinned / "COMMIT").write_text("b2343a17\n", encoding="utf-8")
    queue = _queue(tmp_path, [])
    for summary, edition_dir in ((None, pinned), (("raw", ""), pinned), (("other", "0bad0bad"), pinned),     # pinned text, other queue
                                 (("snapshot", "b2343a17"), unpinned)):                                        # pinned queue, other text
        queue.with_name(asq.OUT_SUMMARY.name).unlink(missing_ok=True)
        if summary:
            _write_summary(queue, tmp_path / summary[0], summary[1])
        with pytest.raises(ValueError, match="rebuild the queue"):
            bpa.check_queue_export(queue, edition_dir)
    _write_summary(queue, pinned, "b2343a17")
    assert bpa.check_queue_export(queue, pinned) == {"pgp_export_commit": "b2343a17", "queue_pgp_export_commit": "b2343a17",
                                                     "queue_pgp_dir": str(pinned)}
    queue.with_name(asq.OUT_SUMMARY.name).unlink()                                     # the default inputs: neither is pinned
    assert bpa.check_queue_export(queue, unpinned) == {"pgp_export_commit": "", "queue_pgp_export_commit": "", "queue_pgp_dir": ""}


def test_build_refuses_a_queue_from_another_export_before_doing_anything(tmp_path, monkeypatch):
    _forbid_work(monkeypatch)
    queue, pgp_dir = _write_inputs(tmp_path, {})
    _write_summary(queue, tmp_path / "raw_data_export", "")                            # e.g. the old raw_data queue
    with pytest.raises(ValueError, match="rebuild the queue"):
        bpa.build(tmp_path / "v2", held_out=(set(), set()), benchmark_index=bdg.build_benchmark_index([]), in_training={},
                  queue_csv=queue, footnotes_csv=pgp_dir / "footnotes.csv")
    assert not (tmp_path / "v2").exists() and not (tmp_path / "v2_images").exists()


def test_main_refuses_pgp_dir_without_queue_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bpa, "build", lambda out, **kwargs: pytest.fail("build ran"))
    with pytest.raises(SystemExit):
        bpa.main(["--out", str(tmp_path / "v2"), "--pgp-dir", str(tmp_path / "snapshot")])
    assert "--pgp-dir needs --queue-csv" in capsys.readouterr().err


def test_build_applies_side_assignments_and_emits_the_unassigned(tmp_path, monkeypatch):
    unlabelled, both = "\n".join(RECTO), "Recto\n" + "\n".join(RECTO) + "\nVerso\n" + "\n".join(VERSO)
    figgy = ["00000001.tif", "00000002.tif"]
    documents = {                                   # pgpid: (queue fields, edition, canvas labels)
        "30": ({"canonical_id": "All", "shelfmark": "T-S Ar.2.30"}, unlabelled, figgy),
        "31": ({"canonical_id": "Mapped", "shelfmark": "T-S Ar.2.31"}, both, figgy),
        "32": ({"canonical_id": "Stale", "shelfmark": "T-S Ar.2.32"}, unlabelled, figgy),
        "33": ({"canonical_id": "Dropped", "shelfmark": "T-S Ar.2.33"}, unlabelled, figgy),
        "34": ({"canonical_id": "Unassigned", "shelfmark": "T-S Ar.2.34"}, unlabelled, figgy),
        "35": ({"canonical_id": "Single", "shelfmark": "T-S Ar.2.35"}, unlabelled, ["1"]),
        "36": ({"canonical_id": "Labelled", "shelfmark": "T-S Ar.2.36"}, both, ["1r", "1v"]),
        "37": ({"canonical_id": "Invalid", "shelfmark": "T-S Ar.2.37"}, unlabelled, figgy),
        "38": ({"canonical_id": "Dropped_earlier", "shelfmark": "T-S Ar.2.38"}, unlabelled, figgy),
        "39": ({"canonical_id": "Verso_only", "shelfmark": "T-S Ar.2.39"}, "Verso\n" + "\n".join(VERSO), figgy),
        "40": ({"canonical_id": "Labels_win", "shelfmark": "T-S Ar.2.40"}, "Verso\n" + "\n".join(VERSO), ["1r"]),
    }
    queue, pgp_dir = _write_inputs(tmp_path, documents)
    fetched = []

    def fake_get(url, timeout=90):
        fetched.append(url)
        if url.startswith("https://lib/m"):
            pid = url.rsplit("m", 1)[1]
            return json.dumps(_manifest([(pid, label) for label in documents[pid][2]])).encode()
        return _jpeg("white")

    monkeypatch.setattr(bab, "http_get", fake_get)
    url = "https://img/{}-{}/full/full/0/default.jpg".format

    def assignment(pid: str, pages: List[Tuple[int, str]], decision: str = "accept", fingerprint: Optional[str] = None) -> Dict[str, Any]:
        """A side matcher's decision for one of ``documents``."""
        sides = _sides(documents[pid][1])
        return {"pgpid": pid, "edition_fingerprint": fingerprint or bpa.edition_fingerprint(sides), "decision": decision, "reader": READER,
                "pages": [{"canvas_index": n, "image_url": url(pid, n), "assign": assign} for n, assign in pages], "evidence": {}}

    assignments = {"30": assignment("30", [(1, "all")]), "31": assignment("31", [(1, "recto"), (0, "verso")]),
                   "32": assignment("32", [(1, "all")], fingerprint="0" * 40), "33": assignment("33", [], decision="drop"),
                   "36": assignment("36", [(0, "all")]), "37": assignment("37", [(0, "recto"), (1, "verso")]),
                   "38": assignment("38", [], decision="drop", fingerprint="0" * 40),    # dropped under an earlier edition text
                   "39": assignment("39", [(1, "verso")]), "40": assignment("40", [(0, "all")])}
    v1_images = tmp_path / "v1_images"
    v1_images.mkdir()
    (v1_images / "35_0.jpg").write_bytes(_jpeg("black", (700, 900)))
    (tmp_path / "v1_manifest.jsonl").write_text(json.dumps({"image_url": url("35", 0), "image_path": "/nas/v1_images/35_0.jpg"}) + "\n",
                                                encoding="utf-8")
    out, emitted = tmp_path / "pgp_arabic_v2", tmp_path / "match" / "unassigned.jsonl"
    common = dict(delay_s=0, val_share=0.0, held_out=(set(), set()), benchmark_index=bdg.build_benchmark_index(["Hebrew_benchmark_doc"]),
                  in_training={}, queue_csv=queue, footnotes_csv=pgp_dir / "footnotes.csv")
    stats = bpa.build(out, image_cache=tmp_path / "v2_cache", seed_cache=bpa.seed_cache_index(v1_images, tmp_path / "v1_manifest.jsonl"),
                      side_assignments=assignments, emit_unassigned=emitted, **common)
    manifest = {m["stem"]: m for m in map(json.loads, (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    assert {stem: (m["side"], m["side_source"]) for stem, m in manifest.items()} == {
        "pgpar_30_1_page": ("", f"side_match:{READER}"), "pgpar_31_0_page": ("verso", f"side_match:{READER}"),
        "pgpar_31_1_page": ("recto", f"side_match:{READER}"), "pgpar_35_0_page": ("", "single"),
        "pgpar_36_0_page": ("recto", "label"), "pgpar_36_1_page": ("verso", "label"), "pgpar_39_1_page": ("verso", f"side_match:{READER}")}
    rows = {r["stem"]: r for r in load_from_disk(str(out))["train_page"]}
    assert rows["pgpar_30_1_page"]["answer"].splitlines()[0] == RECTO[0] and rows["pgpar_31_1_page"]["answer"].startswith(RECTO[0])
    assert rows["pgpar_31_0_page"]["answer"].splitlines()[0] == VERSO[0] and rows["pgpar_35_0_page"]["image"].size == (700, 900)  # seeded
    assert {pid: stats["skipped_documents"][pid] for pid in ("32", "33", "34", "37", "38", "40")} == {
        "32": "side assignment stale", "33": "side match dropped", "34": "edition without side labels and several images",
        "37": "side assignment invalid: text outside the side labels can only be assigned with 'all'", "38": "side assignment stale",
        "40": "side assignment invalid: 'all' is only for an edition without side labels"}                    # the image says 1r
    assert stats["side_assignments"] == {"loaded": 9, "accepted": 3, "dropped": 1, "stale": 2, "invalid": 2, "unused": 1}   # 36: labels
    assert stats["side_sources"] == {"side_match": 4, "single": 1, "label": 2} and stats["unassigned_emitted"] == 3
    assert stats["images"] == {"seed": 1, "download": 6} and url("35", 0) not in fetched
    assert stats["inputs"]["pgp_export_commit"] == "b2343a17" and stats["inputs"]["footnotes_csv"] == str(pgp_dir / "footnotes.csv")
    assert stats["inputs"]["queue_pgp_export_commit"] == "b2343a17" and stats["inputs"]["queue_pgp_dir"] == str(pgp_dir)
    assert sorted(p.name for p in (tmp_path / "v2_cache").iterdir()) == ["30_1.jpg", "31_0.jpg", "31_1.jpg", "35_0.jpg", "36_0.jpg", "36_1.jpg",
                                                                        "39_1.jpg"]
    assert not (tmp_path / "pgp_arabic_v2_images").exists()
    records = [json.loads(line) for line in emitted.read_text(encoding="utf-8").splitlines()]
    assert [(r["pgpid"], r["reason"]) for r in records] == [
        ("32", "edition without side labels and several images (side assignment stale)"),
        ("34", "edition without side labels and several images"),
        ("38", "edition without side labels and several images (side assignment stale)")]          # a stale drop is offered again
    assert set(records[1]) == {"pgpid", "shelfmark", "canonical_id", "library", "edition_fingerprint", "sections", "canvases", "reason"}
    assert records[1]["sections"] == {"": _sides(unlabelled)[""]} and records[1]["edition_fingerprint"] == bpa.edition_fingerprint(_sides(unlabelled))
    assert records[1]["canvases"] == [{"index": 0, "label": "00000001.tif", "image_url": url("34", 0)},
                                      {"index": 1, "label": "00000002.tif", "image_url": url("34", 1)}]
    # round trip: a matcher answers from the emitted record alone, and the next build takes the page
    answer = {"pgpid": records[1]["pgpid"], "edition_fingerprint": records[1]["edition_fingerprint"], "decision": "accept", "reader": READER,
              "pages": [{"canvas_index": 0, "image_url": records[1]["canvases"][0]["image_url"], "assign": "all"}], "evidence": {"f1": 0.8}}
    (tmp_path / "match" / "side_assignments.jsonl").write_text(json.dumps(answer) + "\n", encoding="utf-8")
    again = bpa.build(tmp_path / "second", side_assignments=bpa.load_side_assignments(tmp_path / "match" / "side_assignments.jsonl"), **common)
    assert "34" not in again["skipped_documents"] and again["side_assignments"]["accepted"] == 1
    assert [m["side_source"] for m in map(json.loads, (tmp_path / "second" / "manifest.jsonl").read_text(encoding="utf-8").splitlines())
            if m["pgpid"] == "34"] == [f"side_match:{READER}"]


def test_mainly_arabic_script_matches_the_notebook_gate():
    """A page passes only with >= 40 Arabic letters and >= 9x as many Arabic as Hebrew letters."""
    arabic = "بسم الله الرحمن الرحيم " * 3                       # 57 letters in U+0620-064A
    assert bpa.mainly_arabic_script(arabic)
    assert not bpa.mainly_arabic_script("بسم الله")                # under 40 Arabic letters
    assert bpa.mainly_arabic_script(arabic + " שלום")             # 57 vs 4 Hebrew: 57 >= 36
    assert not bpa.mainly_arabic_script(arabic + " שלום עליכם")   # 57 vs 9 Hebrew: 57 < 81
