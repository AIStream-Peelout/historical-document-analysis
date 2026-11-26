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

from src.datasets.evaluations import arabic_script as ar
from src.datasets.evaluations.helper_eval_scripts import build_arabic_benchmark as bab
from src.datasets.indexing import arabic_scrape_queue as asq
from src.datasets.document_models.genizah_document import fjp_image_url
from src.finetuning.qwen_hebrew import build_documentary_grounding as bdg
from src.finetuning.qwen_hebrew import build_pgp_arabic_editions as bpa
from src.finetuning.qwen_hebrew import side_match as sm
from src.finetuning.qwen_hebrew.prompts import FRAGMENT_TRANSCRIBE_PROMPT

QUEUE_COLUMNS = ["rank", "tier", "pgpid", "shelfmark", "single_fragment", "canonical_id", "library", "doc_type", "side", "doc_date",
                 "iiif_urls", "suggested_iiif"]
STORE_QUEUE_COLUMNS = QUEUE_COLUMNS + ["fjp_images", "merged_join"]          # the queue with the image-store columns
RECTO = ["بسم الله الرحمن الرحيم", "هذا ما استاجر افرايم بن عالي الاسرائيلي من ابي الفتوح [بن سعيد]", "جميع الدار المعروفة به بمصر"]
VERSO = ["الى الشيخ ابو سعد اطال الله بقاه وادام عزه وتاييده", "من عبده ومملوكه يقبل الارض ..."]


def _queue(tmp_path: Path, rows: List[Dict[str, str]], columns: List[str] = QUEUE_COLUMNS) -> Path:
    """Write a queue CSV with ``columns``; a missing field is empty (``single_fragment`` True, ``tier`` 1).

    :param tmp_path: Directory the CSV is written to (as ``queue.csv``).
    :type tmp_path: Path
    :param rows: Queue rows; missing columns take their defaults.
    :type rows: List[Dict[str, str]]
    :param columns: Header of the CSV.
    :type columns: List[str]
    :return: Path of the CSV.
    :rtype: Path
    """
    path = tmp_path / "queue.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, {"single_fragment": "True", "tier": "1"}.get(c, "")) for c in columns})
    return path


def test_the_prompt_is_the_fragment_prompt_with_the_script_changed():
    assert "handwritten Arabic script" in bpa.ARABIC_FRAGMENT_PROMPT and "Hebrew" not in bpa.ARABIC_FRAGMENT_PROMPT
    assert bpa.ARABIC_FRAGMENT_PROMPT.splitlines()[1:] == FRAGMENT_TRANSCRIBE_PROMPT.splitlines()[1:]      # same instructions


def test_candidate_documents_keeps_what_may_be_trained_on(tmp_path: Path) -> None:
    """Held-out, unpermitted, joined and image-less rows are left out with their reasons; an older queue has no store column.

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    """
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
    assert docs["1"]["store_images"] == [] and docs["2"]["store_images"] == []          # an older queue has no fjp_images column


def test_candidate_documents_takes_store_photographs_when_there_is_no_manifest(tmp_path: Path) -> None:
    """A row without a manifest uses its store photographs (quoted names, URLs as they are); a manifest wins over them.

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    """
    base = "https://storage.googleapis.com/cairo-genizah-es-json/images"
    path = _queue(tmp_path, [
        {"pgpid": "1", "canonical_id": "Store", "tier": "2", "fjp_images": "X_1_1r.jpg; X_1_1v 2.jpg ;https://example.org/X_1_2r.jpg",
         "merged_join": "shelfmark"},
        {"pgpid": "2", "canonical_id": "Both", "iiif_urls": "https://lib/m2", "fjp_images": "X_2_1r.jpg", "merged_join": "pgpid"},
        {"pgpid": "3", "canonical_id": "Neither", "fjp_images": " ; "},
        {"pgpid": "4", "canonical_id": "Store_tier_6", "tier": "6", "fjp_images": "X_4_1r.jpg"},
    ], STORE_QUEUE_COLUMNS)
    docs, skipped = bpa.candidate_documents(path, set(), set())
    assert docs["1"]["manifests"] == [] and docs["1"]["store_images"] == [
        f"{base}/X_1_1r.jpg", f"{base}/X_1_1v%202.jpg", "https://example.org/X_1_2r.jpg"]          # quoted; a URL as it is
    assert docs["2"]["manifests"] == ["https://lib/m2"] and docs["2"]["store_images"] == []        # the manifest wins
    assert skipped == {"3": "no library manifest", "4": "no permitted image"}


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


def _manifest(labels: List[Tuple[str, str]]) -> Dict[str, Any]:
    """A IIIF v2 manifest with one 1200x1600 canvas per ``(name, label)``.

    :param labels: ``(name, canvas label)`` per canvas; the image URL is ``https://img/<name>-<n>/...``.
    :type labels: List[Tuple[str, str]]
    :return: Manifest JSON.
    :rtype: Dict[str, Any]
    """
    return {"sequences": [{"canvases": [
        {"label": label, "width": 1200, "height": 1600,
         "images": [{"resource": {"@id": f"https://img/{name}-{n}/full/full/0/default.jpg", "service": {"@id": f"https://img/{name}-{n}"}}}]}
        for n, (name, label) in enumerate(labels)]}]}


def _jpeg(colour: str, size: Tuple[int, int] = (600, 800)) -> bytes:
    """A plain JPEG.

    :param colour: PIL colour name.
    :type colour: str
    :param size: ``(width, height)`` in pixels.
    :type size: Tuple[int, int]
    :return: JPEG bytes.
    :rtype: bytes
    """
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_build_writes_side_matched_pages_and_reports_the_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Library pages are placed only where the record says which image holds which text; the rest is skipped with a reason.

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    :param monkeypatch: pytest's monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
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

    def fake_get(url: str, timeout: int = 90) -> bytes:
        """Stand-in for ``bab.http_get``: manifests of ``documents``, plain JPEGs (a narrow one for the scroll).

        :param url: Requested URL (recorded in ``fetched``).
        :type url: str
        :param timeout: Ignored.
        :type timeout: int
        :return: Response body.
        :rtype: bytes
        """
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
    """The edition's cleaned lines by side, as the builder fingerprints them.

    :param edition: PGP edition text.
    :type edition: str
    :return: ``{side: lines}``.
    :rtype: Dict[str, List[str]]
    """
    return bpa.sides_of(bpa.best_sections([edition]))


def _canvases(labels: List[str], pid: str = "9") -> List[Dict[str, Any]]:
    """Images as ``bab.planned_images`` returns them, one per canvas label.

    :param labels: Canvas labels.
    :type labels: List[str]
    :param pid: Document id used in the image URLs.
    :type pid: str
    :return: ``[{url, label, side, source}]``.
    :rtype: List[Dict[str, Any]]
    """
    return [{"url": f"https://img/{pid}-{n}/full/full/0/default.jpg", "label": label, "side": bab.side_of_label(label), "source": "m"}
            for n, label in enumerate(labels)]


def _assignment(sides: Dict[str, List[str]], canvases: List[Dict[str, Any]], pages: List[Tuple[int, str]], decision: str = "accept",
                fingerprint: Optional[str] = None) -> Dict[str, Any]:
    """A side matcher's decision for document 9: ``pages`` are ``(canvas index, assign)``.

    :param sides: ``{side: lines}`` of the edition.
    :type sides: Dict[str, List[str]]
    :param canvases: The document's images.
    :type canvases: List[Dict[str, Any]]
    :param pages: ``(canvas index, assign)`` per page.
    :type pages: List[Tuple[int, str]]
    :param decision: ``accept`` or ``drop``.
    :type decision: str
    :param fingerprint: Edition fingerprint; default: that of ``sides``.
    :type fingerprint: Optional[str]
    :return: One ``side_assignments.jsonl`` record.
    :rtype: Dict[str, Any]
    """
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


@pytest.mark.parametrize("problem, side", [("edition without side labels and several images", True), (bpa.STORE_ONE_LEFT, True),
                                           (bpa.STORE_READS_ONLY, True),
                                           ("edition has text outside its side labels", True),
                                           ("0 images labelled recto; 2 images labelled verso", True), ("1 images labelled verso", True),
                                           ("no edition text", False), ("no image", False), ("", False)])
def test_is_side_problem(problem: str, side: bool) -> None:
    """Side reasons (store photographs among them) go to the side matcher; no text or no image does not.

    :param problem: A reason from ``assign_pages`` or the store rules.
    :type problem: str
    :param side: Whether it is a side problem.
    :type side: bool
    """
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


def test_seed_cache_copies_an_identical_url_instead_of_downloading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An image with the URL of an earlier build's image is copied from it (read only), anything else is downloaded.

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    :param monkeypatch: pytest's monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    v1_images = tmp_path / "v1_images"
    v1_images.mkdir()
    (v1_images / "7_0.jpg").write_bytes(_jpeg("black", (700, 900)))
    (tmp_path / "v1_manifest.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"image_url": "https://img/7-0/full/full/0/default.jpg", "image_path": "/old/place/v1_images/7_0.jpg"},
        {"image_url": "https://img/8-0/full/full/0/default.jpg", "image_path": "/old/place/v1_images/8_0.jpg"})) + "\n", encoding="utf-8")
    seed = bpa.seed_cache_index(v1_images, tmp_path / "v1_manifest.jsonl")
    assert seed == {"https://img/7-0/full/full/0/default.jpg": v1_images / "7_0.jpg"}            # 8_0.jpg is not on disk
    fetched = []

    def fake_get(url: str, timeout: int = 90) -> bytes:
        """Stand-in for ``bab.http_get``: a white JPEG for any URL.

        :param url: Requested URL (recorded in ``fetched``).
        :type url: str
        :param timeout: Ignored.
        :type timeout: int
        :return: JPEG bytes.
        :rtype: bytes
        """
        fetched.append(url)
        return _jpeg("white")

    monkeypatch.setattr(bab, "http_get", fake_get)
    cache = tmp_path / "v2_images"
    cache.mkdir()
    assert bpa.fetch_image("https://img/7-0/full/full/0/default.jpg", cache / "7_0.jpg", seed) == "seed"
    assert bpa.fetch_image("https://img/7-0/full/full/0/default.jpg", cache / "7_0.jpg", seed) == "cache"
    assert bpa.fetch_image("https://img/8-0/full/full/0/default.jpg", cache / "8_0.jpg", seed) == "download"
    assert fetched == ["https://img/8-0/full/full/0/default.jpg"] and Image.open(cache / "7_0.jpg").size == (700, 900)
    assert (v1_images / "7_0.jpg").read_bytes() == (cache / "7_0.jpg").read_bytes() and sorted(p.name for p in v1_images.iterdir()) == ["7_0.jpg"]


def _published(tmp_path: Path) -> Tuple[Path, Tuple[Path, Path, Path]]:
    """A stand-in for the NAS datasets directory holding a v1 build, its image cache and its images-once export.

    :param tmp_path: Test directory.
    :type tmp_path: Path
    :return: ``(datasets directory, (v1, v1_images, v1_images_once))``, each holding ``keep.txt``.
    :rtype: Tuple[Path, Tuple[Path, Path, Path]]
    """
    datasets = tmp_path / "datasets"
    v1 = datasets / "pgp_arabic_editions_v1"
    protected = (v1, datasets / "pgp_arabic_editions_v1_images", datasets / "pgp_arabic_editions_v1_images_once")
    for directory in protected:
        directory.mkdir(parents=True)
        (directory / "keep.txt").write_text("published", encoding="utf-8")
    return datasets, protected


def _flip_case(path: Path, part: str) -> Path:
    """``path`` with the component ``part`` spelled in upper case.

    :param path: Any path.
    :type path: Path
    :param part: Component to upper-case.
    :type part: str
    :return: The respelled path.
    :rtype: Path
    """
    return Path(*[p.upper() if p == part else p for p in path.parts])


def test_default_out_is_v3_and_the_real_v1_and_v2_are_protected() -> None:
    """The default output is v3; v1 and v2 (pinned inputs) and their image directories are refused in any spelling."""
    args = bpa.build_parser().parse_args([])
    assert args.out == bpa.DEFAULT_OUT and bpa.DEFAULT_OUT.name == "pgp_arabic_editions_v3" and bpa.DEFAULT_OUT not in (bpa.V1_OUT, bpa.V2_OUT)
    assert bpa.V1_OUT.name == "pgp_arabic_editions_v1" and args.image_cache is None and args.pgp_dir is None and args.queue_csv is None
    assert bpa.V2_OUT.name == "pgp_arabic_editions_v2" and args.store_label_checks is None
    assert [p.name for p in bpa.PROTECTED_OUTPUTS] == ["pgp_arabic_editions_v1", "pgp_arabic_editions_v1_images",
                                                      "pgp_arabic_editions_v1_images_once", "pgp_arabic_editions_v2",
                                                      "pgp_arabic_editions_v2_images", "pgp_arabic_editions_v2_images_once"]
    assert {p.parent for p in bpa.PROTECTED_OUTPUTS} == {bab.NAS_DATASETS}
    # check_outputs only compares paths (and stats the ones that exist): it never writes
    for refused in (bpa.V1_OUT, bpa.V1_OUT / "unassigned.jsonl", bab.NAS_DATASETS, bab.NAS_DATASETS.parent, Path("/"),
                    bpa.V1_OUT.with_name("PGP_Arabic_Editions_v1"), bpa.V1_OUT.with_name("pgp_arabic_editions_v1_IMAGES"),
                    bpa.V2_OUT, bpa.V2_OUT / "manifest.jsonl", bpa.V2_OUT.with_name("pgp_arabic_editions_v2_images"),
                    bpa.V2_OUT.with_name("pgp_arabic_editions_v2_images_once") / "x.jpg", bpa.V2_OUT.with_name("PGP_Arabic_Editions_V2")):
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
    """Make every step of ``build`` after its checks fail loudly, so a test of the checks can never build anything.

    :param monkeypatch: pytest's monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """

    def forbidden(*args: Any, **kwargs: Any) -> None:
        """Stand-in for a step that must not run.

        :param args: Ignored.
        :type args: Any
        :param kwargs: Ignored.
        :type kwargs: Any
        :raises AssertionError: always.
        """
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


def test_main_threads_the_flags_to_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every flag reaches ``build``; the label-check file is passed loaded and as its path (for ``stats.inputs``).

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    :param monkeypatch: pytest's monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    calls: Dict[str, Any] = {}

    def fake_build(out: Path, **kwargs: Any) -> Dict[str, Any]:
        """Stand-in for ``build`` that records its arguments.

        :param out: Output directory.
        :type out: Path
        :param kwargs: The other arguments.
        :type kwargs: Any
        :return: Minimal statistics.
        :rtype: Dict[str, Any]
        """
        calls.update(out=out, **kwargs)
        return {"pages": 0}

    monkeypatch.setattr(bpa, "build", fake_build)
    v1_images = tmp_path / "v1_images"
    v1_images.mkdir()
    (v1_images / "1_0.jpg").write_bytes(_jpeg("white"))
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"image_url": "https://u/1", "image_path": "x/1_0.jpg"}) + "\n", encoding="utf-8")
    (tmp_path / "assign.jsonl").write_text(json.dumps({"pgpid": "1", "edition_fingerprint": "f", "decision": "drop", "reader": READER,
                                                       "pages": []}) + "\n", encoding="utf-8")
    bpa.main(["--out", str(tmp_path / "v2"), "--pgp-dir", str(tmp_path / "snapshot"), "--queue-csv", str(tmp_path / "q.csv"),
              "--image-cache", str(tmp_path / "cache"), "--seed-cache-from", str(v1_images), str(tmp_path / "manifest.jsonl"),
              "--side-assignments", str(tmp_path / "assign.jsonl"), "--emit-unassigned", str(tmp_path / "unassigned.jsonl"), "--limit", "3",
              "--store-label-checks", str(_write_label_checks(tmp_path / "checks.jsonl", [_label_check("X_1_1r.jpg", "yes")]))])
    assert calls["out"] == tmp_path / "v2" and calls["footnotes_csv"] == tmp_path / "snapshot" / "footnotes.csv"
    assert calls["queue_csv"] == tmp_path / "q.csv" and calls["image_cache"] == tmp_path / "cache" and calls["limit"] == 3
    assert calls["seed_cache"] == {"https://u/1": v1_images / "1_0.jpg"} and list(calls["side_assignments"]) == ["1"]
    assert calls["emit_unassigned"] == tmp_path / "unassigned.jsonl"
    assert calls["store_label_checks"] == {fjp_image_url("X_1_1r.jpg"): [_label_check("X_1_1r.jpg", "yes")]}
    assert calls["store_label_checks_path"] == tmp_path / "checks.jsonl"
    bpa.main([])
    assert calls["out"] == bpa.DEFAULT_OUT and calls["footnotes_csv"] is None and calls["queue_csv"] is None
    assert calls["seed_cache"] is None and calls["side_assignments"] is None and calls["emit_unassigned"] is None
    assert calls["store_label_checks"] is None and calls["store_label_checks_path"] is None


def _write_summary(queue: Path, pgp_dir: Optional[Path], commit: str) -> None:
    """The summary ``arabic_scrape_queue`` writes next to a queue, recording the export it was built from.

    :param queue: Queue CSV; the summary is written next to it.
    :type queue: Path
    :param pgp_dir: Export directory recorded (None: empty).
    :type pgp_dir: Optional[Path]
    :param commit: Export commit recorded (``""``: unpinned).
    :type commit: str
    """
    queue.with_name(asq.OUT_SUMMARY.name).write_text(json.dumps({"pgp_dir": str(pgp_dir) if pgp_dir else "", "pgp_export_commit": commit,
                                                                 "listed": 0}), encoding="utf-8")


def _write_inputs(tmp_path: Path, documents: Dict[str, Tuple[Dict[str, str], str, List[str]]],
                  columns: List[str] = QUEUE_COLUMNS) -> Tuple[Path, Path]:
    """The queue CSV (with its summary) and a pinned-snapshot-like PGP directory (footnotes.csv + COMMIT) for ``documents``.

    A document's manifest is ``https://lib/m<pid>`` unless its fields give ``iiif_urls`` (``""``: none).

    :param tmp_path: Test directory.
    :type tmp_path: Path
    :param documents: ``{pgpid: (queue fields, edition text, canvas labels)}``.
    :type documents: Dict[str, Tuple[Dict[str, str], str, List[str]]]
    :param columns: Queue CSV header.
    :type columns: List[str]
    :return: ``(queue CSV, PGP export directory)``.
    :rtype: Tuple[Path, Path]
    """
    queue = _queue(tmp_path, [dict({"iiif_urls": f"https://lib/m{pid}"}, **fields, pgpid=pid) for pid, (fields, _, _) in documents.items()],
                   columns)
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


def test_build_applies_side_assignments_and_emits_the_unassigned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepted assignments place pages where labels say nothing; stale, dropped and invalid ones do not; the rest is emitted.

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    :param monkeypatch: pytest's monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
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

    def fake_get(url: str, timeout: int = 90) -> bytes:
        """Stand-in for ``bab.http_get``: manifests of ``documents``, white JPEGs.

        :param url: Requested URL (recorded in ``fetched``).
        :type url: str
        :param timeout: Ignored.
        :type timeout: int
        :return: Response body.
        :rtype: bytes
        """
        fetched.append(url)
        if url.startswith("https://lib/m"):
            pid = url.rsplit("m", 1)[1]
            return json.dumps(_manifest([(pid, label) for label in documents[pid][2]])).encode()
        return _jpeg("white")

    monkeypatch.setattr(bab, "http_get", fake_get)
    url = "https://img/{}-{}/full/full/0/default.jpg".format

    def assignment(pid: str, pages: List[Tuple[int, str]], decision: str = "accept", fingerprint: Optional[str] = None) -> Dict[str, Any]:
        """A side matcher's decision for one of ``documents``.

        :param pid: Document id.
        :type pid: str
        :param pages: ``(canvas index, assign)`` per page.
        :type pages: List[Tuple[int, str]]
        :param decision: ``accept`` or ``drop``.
        :type decision: str
        :param fingerprint: Edition fingerprint; default: that of the document's edition.
        :type fingerprint: Optional[str]
        :return: One ``side_assignments.jsonl`` record.
        :rtype: Dict[str, Any]
        """
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
    assert stats["side_assignments"] == {"loaded": 9, "accepted": 3, "store_unmatched": 0, "dropped": 1, "stale": 2, "invalid": 2,
                                         "unused": 1}                                                                # 36: labels
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


# --- image-store photographs ----------------------------------------------------------------------------------------------

STORE_BASE = "https://storage.googleapis.com/cairo-genizah-es-json/images"


def _label_check(name: str, verdict: str, resolved_to: str = "", label_read: Optional[str] = None,
                 pgpid: Optional[str] = None) -> Dict[str, Any]:
    """One shelf-mark label check of the store photograph ``name`` (the ``--store-label-checks`` line format).

    :param name: Store file name.
    :type name: str
    :param verdict: ``label_matches``.
    :type verdict: str
    :param resolved_to: Fragment the label names.
    :type resolved_to: str
    :param label_read: Label as read.
    :type label_read: Optional[str]
    :param pgpid: Provenance: the document the check was made for (None: a check of the image alone, no ``pgpid`` key).
    :type pgpid: Optional[str]
    :return: The record.
    :rtype: Dict[str, Any]
    """
    record = {"image_url": fjp_image_url(name), "image": name, "label_read": label_read, "label_matches": verdict,
              "resolved_to": resolved_to, "basis": "slip read", "recorded_at": "2026-11-01T00:00:00Z"}
    return record if pgpid is None else dict(record, pgpid=pgpid)


def _write_label_checks(path: Path, records: List[Dict[str, Any]]) -> Path:
    """Write label-check records as JSONL and return the path.

    :param path: JSONL file to write.
    :type path: Path
    :param records: Label-check records (any JSON value, to test malformed lines).
    :type records: List[Dict[str, Any]]
    :return: ``path``.
    :rtype: Path
    """
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


@pytest.mark.parametrize("name, label", [("AIU_XII.80_1v.jpg", "file:1v"), ("X_2R.JPG", "file:2r"), ("X_1r(1).jpg", "file:1r"),
                                         ("X_1r%202.jpg", "file:1r"), ("5804_image.jpg", ""), ("X_12.jpg", ""), ("X_1r.json", "")])
def test_store_file_label_is_provenance_from_the_file_name(name: str, label: str) -> None:
    """The side token of a store file name becomes a ``file:`` label, or none.

    :param name: Store file name.
    :type name: str
    :param label: Expected provenance label.
    :type label: str
    """
    assert bpa.store_file_label(f"{STORE_BASE}/{name}") == label


def test_store_canvases_never_take_a_side_from_the_file_name() -> None:
    """A store canvas has no side, whatever its file name says, and the side matcher never sees the token."""
    canvases = [{"url": f"{STORE_BASE}/X_1v.jpg", "label": "", "side": "", "source": "image_store"},
                {"url": f"{STORE_BASE}/X_1r.jpg", "label": "", "side": "", "source": "image_store"}]
    assert [(c["label"], c["side"]) for c in bpa.store_canvases(canvases)] == [("file:1v", ""), ("file:1r", "")]
    assert bpa.assign_pages([("verso", VERSO)], [c["side"] for c in bpa.store_canvases(canvases)])[0] == {}      # file names place nothing
    record = bpa.unassigned_record("9", {"canonical_id": "", "shelfmark": "X", "library": ""}, _sides("Verso\n" + "\n".join(VERSO)),
                                   bpa.store_canvases(canvases), "0 images labelled verso")
    assert [c["label"] for c in record["canvases"]] == ["", ""]                    # the matcher never sees the file-name token


def test_load_store_label_checks_validates_every_line(tmp_path: Path) -> None:
    """Every check is kept, one per image and provenance pgpid (a later line for the same pair replaces the earlier one in
    its place); a bad line raises ValueError naming its line number.

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    """
    first, again = _label_check("X_1r.jpg", "no", "X 2"), _label_check("X_1r.jpg", "yes", "X 1", "X 1")
    for_a, for_b = _label_check("X_1r.jpg", "no", "X 2", "X 2", pgpid="1"), _label_check("X_1r.jpg", "yes", "X 2", "X 2", pgpid="2")
    for_a_again = dict(for_a, label_matches="unclear", pgpid=1)                      # an integer pgpid is the same document
    path = _write_label_checks(tmp_path / "checks.jsonl", [first, for_a, _label_check("X_1v.jpg", "no label visible"), again, for_b,
                                                           for_a_again])
    loaded = bpa.load_store_label_checks(path)
    assert sorted(loaded) == [fjp_image_url("X_1r.jpg"), fjp_image_url("X_1v.jpg")]
    assert loaded[fjp_image_url("X_1r.jpg")] == [again, for_a_again, for_b]
    assert loaded[fjp_image_url("X_1v.jpg")] == [_label_check("X_1v.jpg", "no label visible")]
    for broken, message in ((dict(first, label_matches="maybe"), "checks.jsonl:2: label_matches 'maybe'"),
                            ({k: v for k, v in first.items() if k != "resolved_to"}, "checks.jsonl:2: missing resolved_to"),
                            (dict(first, image_url=""), "checks.jsonl:2: image_url"), (dict(first, label_read=3), "checks.jsonl:2: label_read"),
                            (dict(first, pgpid=["1"]), "checks.jsonl:2: pgpid"), (dict(first, pgpid=True), "checks.jsonl:2: pgpid"),
                            (["not", "an", "object"], "checks.jsonl:2: not a JSON object")):
        _write_label_checks(path, [again, broken])
        with pytest.raises(ValueError, match=message):
            bpa.load_store_label_checks(path)


def _store(names: List[str]) -> List[Dict[str, Any]]:
    """Store canvases of the photographs ``names``, as the builder makes them.

    :param names: Store file names.
    :type names: List[str]
    :return: :func:`bpa.store_canvases` of them.
    :rtype: List[Dict[str, Any]]
    """
    return bpa.store_canvases([{"url": fjp_image_url(name), "label": "", "side": "", "source": "image_store"} for name in names])


def _read_scores(pages: List[Tuple[int, str]], recall: float = 0.9, precision: float = 0.9) -> Dict[str, Any]:
    """Side-matcher ``evidence`` whose read of each page's canvas has these scores (the ``side_match.decide_record`` shape).

    :param pages: ``(canvas index, assign)`` per page; ``all`` is scored on the section ``""``.
    :type pages: List[Tuple[int, str]]
    :param recall: Recall of the assigned section in the read.
    :type recall: float
    :param precision: Precision of the read against the edition.
    :type precision: float
    :return: ``{rule, reason, canvases: [{canvas_index, recall: {section: recall}, precision, repeat_share, read_letters}],
        thresholds}``.
    :rtype: Dict[str, Any]
    """
    return {"rule": "test", "reason": "", "thresholds": dict(sm.THRESHOLDS),
            "canvases": [{"canvas_index": index, "recall": {"" if assign == "all" else assign: recall}, "precision": precision,
                          "repeat_share": 0.0, "read_letters": 100} for index, assign in pages]}


def _scored_assignment(pgpid: str, sides: Dict[str, List[str]], url: str, assign: str, read: str) -> Dict[str, Any]:
    """An accepted one-photograph side assignment whose evidence holds the side matcher's real scores of ``read``.

    The decision is made here, not by the matcher's rules, so the builder's store read gate is tested on its own.

    :param pgpid: Document id.
    :type pgpid: str
    :param sides: ``{side: lines}`` of the edition.
    :type sides: Dict[str, List[str]]
    :param url: The photograph (canvas 0).
    :type url: str
    :param assign: ``all`` / ``recto`` / ``verso``.
    :type assign: str
    :param read: The read of the photograph.
    :type read: str
    :return: One ``side_assignments.jsonl`` record.
    :rtype: Dict[str, Any]
    """
    return {"pgpid": pgpid, "edition_fingerprint": bpa.edition_fingerprint(sides), "decision": "accept", "reader": READER,
            "pages": [{"canvas_index": 0, "image_url": url, "assign": assign}],
            "evidence": {"canvases": sm.score_canvases(sides, [0], [read]), "thresholds": dict(sm.THRESHOLDS)}}


def test_store_read_problem_asks_the_read_for_the_text_not_a_formula() -> None:
    """A store page is kept only when the read shows its text: a shared basmala, a Hebrew-script read, too few shared 5-grams
    or missing scores keep it out."""
    sides, url = _sides("\n".join(RECTO)), fjp_image_url("S_1r.jpg")
    faithful = _scored_assignment("9", sides, url, "all", "\n".join(RECTO))
    assert bpa.store_read_problem(sides, faithful, 0) == ""
    formula = _scored_assignment("9", sides, url, "all", RECTO[0] + "\n" + "\n".join(VERSO))      # a basmala, then another letter
    score, grams = formula["evidence"]["canvases"][0], len(ar.arabic_letters("\n".join(sides[""]))) - bpa.STORE_NGRAM + 1
    assert bpa.STORE_MIN_RECALL <= score["recall"][""] < bpa.STORE_SHORT_MIN_RECALL and score["precision"] >= bpa.STORE_MIN_PRECISION
    assert round(score["recall"][""] * grams) < bpa.STORE_MIN_SHARED                  # 16 shared: the basmala and one 5-gram more
    assert bpa.store_read_problem(sides, formula, 0) == bpa.STORE_WEAK_READ
    hebrew = _scored_assignment("9", sides, url, "all", "חשבון של ר' יוסף בן יעקב עשרה דינרים")
    assert bpa.store_read_problem(sides, hebrew, 0) == bpa.STORE_WEAK_READ
    page = [(0, "all")]
    unscored = dict(faithful, evidence={"f1": 0.9})
    other_canvas = dict(faithful, evidence=_read_scores([(1, "all")]))
    other_ngram = dict(faithful, evidence=dict(_read_scores(page), thresholds={"ngram": 4}))
    no_section = dict(faithful, evidence=_read_scores([(0, "recto")]))
    flag = dict(faithful, evidence=_read_scores(page, precision=True))
    for assignment in (unscored, dict(faithful, evidence=None), other_canvas, other_ngram, no_section, flag):
        assert bpa.store_read_problem(sides, assignment, 0) == bpa.STORE_UNSCORED
    longer = _sides("\n".join(RECTO + VERSO))                            # 149 letters, 145 5-grams: 0.15 of them is 22
    assert bpa.store_read_problem(longer, dict(faithful, evidence=_read_scores(page, 0.15, 0.2)), 0) == ""        # at the floors
    assert bpa.store_read_problem(longer, dict(faithful, evidence=_read_scores(page, 0.15, 0.19)), 0) == bpa.STORE_WEAK_READ
    assert bpa.store_read_problem(longer, dict(faithful, evidence=_read_scores(page, 0.14, 0.9)), 0) == bpa.STORE_WEAK_READ
    # 86 letters: 25 shared 5-grams, but a short text needs recall 0.4
    assert bpa.store_read_problem(sides, dict(faithful, evidence=_read_scores(page, 0.3, 0.3)), 0) == bpa.STORE_WEAK_READ
    verso = _sides("Verso\n" + "\n".join(VERSO))                         # a short side: 63 letters, 59 5-grams
    grams = len(ar.arabic_letters("\n".join(verso["verso"]))) - bpa.STORE_NGRAM + 1
    short = dict(_scored_assignment("9", verso, url, "verso", ""), evidence=_read_scores([(0, "verso")], 0.3, 0.9))
    assert round(0.3 * grams) < bpa.STORE_MIN_SHARED <= round(0.36 * grams)
    assert bpa.store_read_problem(verso, short, 0) == bpa.STORE_WEAK_READ                                       # 18 shared
    assert bpa.store_read_problem(verso, dict(short, evidence=_read_scores([(0, "verso")], 0.36, 0.9)), 0) == bpa.STORE_WEAK_READ
    assert bpa.store_read_problem(verso, dict(short, evidence=_read_scores([(0, "verso")], 0.45, 0.9)), 0) == ""   # 27 shared


def _letters_sides(letters: int, section: str = "") -> Dict[str, List[str]]:
    """An edition whose one section holds exactly ``letters`` Arabic letters (the fixture's letters, repeated).

    :param letters: Arabic letters in the section.
    :type letters: int
    :param section: The section's label (``""``, ``recto`` or ``verso``).
    :type section: str
    :return: ``{section: [one line]}``.
    :rtype: Dict[str, List[str]]
    """
    source = ar.arabic_letters("".join(RECTO + VERSO))
    sides = {section: [(source * (letters // len(source) + 1))[:letters]]}
    assert len(ar.arabic_letters(sides[section][0])) == letters
    return sides


@pytest.mark.parametrize("letters, recall, precision, problem", [
    (80, 0.3, 0.5, bpa.STORE_WEAK_READ),        # 23 shared 5-grams, but a short text needs recall 0.4
    (300, 0.3, 0.5, ""),                         # the same recall on a long text passes
    (80, 0.4, 0.5, ""),                          # at the short-text recall floor: 30 shared
    (80, 0.39, 0.5, bpa.STORE_WEAK_READ),
    (99, 0.3, 0.5, bpa.STORE_WEAK_READ),        # the last short length: 28 shared
    (100, 0.3, 0.5, ""),                         # the first length the base recall floor applies to: 29 shared
    (120, 0.17, 0.5, ""),                        # 116 5-grams: 20 shared, at the shared floor
    (120, 0.16, 0.5, bpa.STORE_WEAK_READ),      # 19 shared, though recall and precision pass
    (300, 0.3, 0.2, ""),                         # at the precision floor
    (300, 0.3, 0.199, bpa.STORE_WEAK_READ),
    (300, 0.15, 0.2, ""),                        # at the recall floor: 44 shared
    (300, 0.149, 0.9, bpa.STORE_WEAK_READ),
])
def test_store_read_problem_floors(letters: int, recall: float, precision: float, problem: str) -> None:
    """The store read gate at its floors: recall (higher for a page text under ``STORE_SHORT_LETTERS`` Arabic letters),
    precision and shared 5-grams, each tested where it alone decides.

    :param letters: Arabic letters in the assigned section.
    :type letters: int
    :param recall: Recall of the section in the read.
    :type recall: float
    :param precision: Precision of the read against the edition.
    :type precision: float
    :param problem: Expected :func:`bpa.store_read_problem` result.
    :type problem: str
    """
    assert (bpa.STORE_MIN_RECALL, bpa.STORE_MIN_PRECISION, bpa.STORE_MIN_SHARED) == (0.15, 0.2, 20)
    assert (bpa.STORE_SHORT_LETTERS, bpa.STORE_SHORT_MIN_RECALL) == (100, 0.4)
    sides = _letters_sides(letters, "recto")
    assignment = {"pgpid": "9", "edition_fingerprint": bpa.edition_fingerprint(sides), "decision": "accept", "reader": READER,
                  "pages": [{"canvas_index": 0, "image_url": f"{STORE_BASE}/S_1r.jpg", "assign": "recto"}],
                  "evidence": _read_scores([(0, "recto")], recall, precision)}
    assert bpa.store_read_problem(sides, assignment, 0) == problem


def test_label_checked_needs_a_check_for_every_store_photograph() -> None:
    """One unchecked photograph keeps the whole document out; checked canvases carry their verdicts."""
    canvases, doc = _store(["X_1r.jpg", "X_1v.jpg"]), {"canonical_id": "Paris_X_1", "shelfmark": "X 1"}
    checks = {fjp_image_url("X_1r.jpg"): [_label_check("X_1r.jpg", "yes", "X 1", "X")],
              fjp_image_url("X_1v.jpg"): [_label_check("X_1v.jpg", "no", "Y")]}
    assert bpa.label_checked(canvases, None, "1", doc) == ([], bpa.NOT_LABEL_CHECKED)
    assert bpa.label_checked(canvases, {fjp_image_url("X_1r.jpg"): checks[fjp_image_url("X_1r.jpg")]}, "1", doc) == ([], bpa.NOT_LABEL_CHECKED)
    checked, problem = bpa.label_checked(canvases, checks, "1", doc)
    assert problem == "" and [(c["label_check"], c["label_read"], c["image"]) for c in checked] == [("yes", "X", "X_1r.jpg"), ("no", None, "X_1v.jpg")]
    assert bpa.other_fragment_reason(checked[1]) == "store photograph labelled as another fragment: Y"


def test_label_checked_takes_the_documents_own_check_else_the_image_check() -> None:
    """A photograph filed under two records: each document takes the check recorded for it, else the one check of the image
    alone; checks recorded only for other documents leave the photograph unchecked."""
    canvases, url = _store(["SH_1r.jpg"]), fjp_image_url("SH_1r.jpg")
    doc_a, doc_b = {"canonical_id": "Paris_AIU_XII_60", "shelfmark": "AIU XII.60"}, {"canonical_id": "Paris_AIU_XII_61", "shelfmark": "AIU XII.61"}
    doc_c = {"canonical_id": "Paris_AIU_XII_62", "shelfmark": "AIU XII.62"}
    for_a = _label_check("SH_1r.jpg", "no", "Paris_AIU_XII_61", "AIU XII.61", pgpid="60")
    for_b = _label_check("SH_1r.jpg", "yes", "Paris_AIU_XII_61", "AIU XII.61", pgpid="61")
    by_image = _label_check("SH_1r.jpg", "no label visible")

    def verdicts(checks: Dict[str, List[Dict[str, Any]]], pgpid: str, doc: Dict[str, Any]) -> List[str]:
        """The label verdicts the document's photographs carry.

        :param checks: ``{image_url: [check, ...]}``.
        :type checks: Dict[str, List[Dict[str, Any]]]
        :param pgpid: Document id.
        :type pgpid: str
        :param doc: ``canonical_id`` and ``shelfmark`` of the document.
        :type doc: Dict[str, Any]
        :return: One verdict per canvas (empty when the document is not label-checked).
        :rtype: List[str]
        """
        return [canvas["label_check"] for canvas in bpa.label_checked(canvases, checks, pgpid, doc)[0]]

    assert verdicts({url: [for_a, for_b]}, "60", doc_a) == ["no"] and verdicts({url: [for_a, for_b]}, "61", doc_b) == ["yes"]
    assert bpa.label_checked(canvases, {url: [for_a, for_b]}, "62", doc_c) == ([], bpa.NOT_LABEL_CHECKED)   # other documents' checks only
    assert verdicts({url: [for_a, by_image, for_b]}, "62", doc_c) == ["no label visible"]                   # the check of the image alone
    assert verdicts({url: [by_image, for_a]}, "60", doc_a) == ["no"]                                        # the document's own check wins
    assert bpa.check_for([dict(for_b, pgpid=61)], "61") == dict(for_b, pgpid=61)                            # an integer provenance
    assert bpa.check_for([by_image, dict(by_image, label_matches="yes")], "62") is None                     # two checks of the image: ambiguous
    assert bpa.check_for([], "62") is None


def test_label_checked_counts_yes_only_for_the_document_it_names() -> None:
    """A ``yes`` whose resolved fragment is another record names another fragment and drops the photograph like ``no``; an
    empty one or the document's own counts as ``yes``."""
    canvases, doc = _store(["SH_1r.jpg"]), {"canonical_id": "Paris_AIU_XII_60", "shelfmark": "AIU XII.60"}
    url = fjp_image_url("SH_1r.jpg")
    for resolved_to, verdict in (("Paris_AIU_XII_61", bpa.YES_FOR_ANOTHER), ("AIU XII.6", bpa.YES_FOR_ANOTHER), ("", "yes"),
                                 ("Paris_AIU_XII_60", "yes"), ("AIU XII.60", "yes"), ("aiu  xii-60", "yes")):
        checked, problem = bpa.label_checked(canvases, {url: [_label_check("SH_1r.jpg", "yes", resolved_to, "AIU XII")]}, "60", doc)
        assert (problem, checked[0]["label_check"], checked[0]["resolved_to"]) == ("", verdict, resolved_to)
    unclear = bpa.label_checked(canvases, {url: [_label_check("SH_1r.jpg", "unclear", "Paris_AIU_XII_61")]}, "60", doc)[0]
    assert unclear[0]["label_check"] == "unclear"                                       # only a yes is re-weighed
    assert bpa.YES_FOR_ANOTHER == "yes for another record" and bpa.DROP_VERDICTS == ("no", bpa.YES_FOR_ANOTHER)
    another = bpa.label_checked(canvases, {url: [_label_check("SH_1r.jpg", "yes", "Paris_AIU_XII_61", "AIU XII.61")]}, "60", doc)[0]
    assert bpa.other_fragment_reason(another[0]) == "store photograph labelled as another fragment: Paris_AIU_XII_61"
    assert bpa.names_document(None, doc) and bpa.names_document("  ", doc)
    assert bpa.names_document("T-S Ar.1.10", {"canonical_id": "T_S_Ar_1_10", "shelfmark": ""})
    assert not bpa.names_document("ENA NS 2.26", {"canonical_id": "", "shelfmark": "ENA NS 22.6"})            # digit boundaries kept
    assert not bpa.names_document("?", {"canonical_id": "", "shelfmark": "?"})                                # no loose form: not a match


def test_build_uses_label_checked_store_photographs_and_marks_the_set_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Store documents: the label check gates and drops photographs, sides come from reads only, and the set is private.

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    :param monkeypatch: pytest's monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    both, verso_only = "Recto\n" + "\n".join(RECTO) + "\nVerso\n" + "\n".join(VERSO), "Verso\n" + "\n".join(VERSO)
    store = {"50": ["S_50_1v.jpg"], "51": ["S_51_1r.jpg", "S_51_1v.jpg"], "52": ["S 52_1r.jpg", "S_52_1v.jpg", "S_52_2r.jpg"],
             "53": ["S_53_1r.jpg", "S_53_1v.jpg"], "54": ["S_54_1r.jpg"], "55": ["S_55_1r.jpg"], "56": ["S_56_1r.jpg"],
             "57": ["S_57_1r.jpg", "S_57_1v.jpg"]}
    editions = {"50": "\n".join(RECTO), "51": both, "52": both, "53": "\n".join(RECTO), "54": "\n".join(RECTO), "55": "\n".join(RECTO),
                "56": "\n".join(RECTO), "57": verso_only}
    documents = {pid: ({"canonical_id": f"Store_{pid}", "shelfmark": f"AIU XII.{pid}", "tier": "2", "merged_join": "pgpid",
                        "iiif_urls": "https://lib/m56" if pid == "56" else "", "fjp_images": ";".join(store[pid])}, editions[pid], ["1"])
                 for pid in store}
    queue, pgp_dir = _write_inputs(tmp_path, documents, STORE_QUEUE_COLUMNS)
    fetched = []

    def fake_get(url: str, timeout: int = 90) -> bytes:
        """Stand-in for ``bab.http_get``: the manifest of document 56, white JPEGs.

        :param url: Requested URL (recorded in ``fetched``).
        :type url: str
        :param timeout: Ignored.
        :type timeout: int
        :return: Response body.
        :rtype: bytes
        """
        fetched.append(url)
        if url.startswith("https://lib/m"):
            return json.dumps(_manifest([("56", "1")])).encode()
        return _jpeg("white")

    monkeypatch.setattr(bab, "http_get", fake_get)
    checks = _write_label_checks(tmp_path / "label_checks.jsonl", [
        _label_check("S_50_1v.jpg", "no label visible"), _label_check("S_51_1r.jpg", "yes", "AIU XII.51", "AIU XII.51"),
        _label_check("S_51_1v.jpg", "unclear", "AIU XII.51", "XII.5"), _label_check("S 52_1r.jpg", "no", "AIU XII.53", "AIU XII.53"),
        _label_check("S_52_1v.jpg", "yes", "AIU XII.52", "AIU XII.52"), _label_check("S_52_2r.jpg", "unclear", "AIU XII.52"),
        _label_check("S_53_1r.jpg", "yes", "AIU XII.53"), _label_check("S_53_1v.jpg", "yes", "AIU XII.53"),
        _label_check("S_55_1r.jpg", "no", "", "ENA 1.2"), _label_check("S_56_1r.jpg", "yes", "AIU XII.56"),   # 54: never checked
        _label_check("S_57_1r.jpg", "no", "AIU XII.58"), _label_check("S_57_1v.jpg", "yes", "AIU XII.57")])

    def assignment(pid: str, pages: List[Tuple[str, str]]) -> Dict[str, Any]:
        """A side matcher's decision: ``pages`` are ``(store file name, assign)``, indices counted after the label check; the
        reads match the edition well (``_read_scores``).

        :param pid: Document id.
        :type pid: str
        :param pages: ``(store file name, assign)`` per page.
        :type pages: List[Tuple[str, str]]
        :return: One ``side_assignments.jsonl`` record.
        :rtype: Dict[str, Any]
        """
        kept = [name for name in store[pid] if name != "S 52_1r.jpg"]
        return {"pgpid": pid, "edition_fingerprint": bpa.edition_fingerprint(_sides(editions[pid])), "decision": "accept", "reader": READER,
                "pages": [{"canvas_index": kept.index(name), "image_url": fjp_image_url(name), "assign": assign} for name, assign in pages],
                "evidence": _read_scores([(kept.index(name), assign) for name, assign in pages])}

    assignments = {"51": assignment("51", [("S_51_1r.jpg", "verso"), ("S_51_1v.jpg", "recto")]),       # reads, not file names, say which side
                   "52": assignment("52", [("S_52_1v.jpg", "recto"), ("S_52_2r.jpg", "verso")])}
    out, emitted = tmp_path / "pgp_arabic_v3", tmp_path / "match" / "unassigned.jsonl"
    common = dict(delay_s=0, val_share=0.0, held_out=(set(), set()), benchmark_index=bdg.build_benchmark_index([]), in_training={},
                  queue_csv=queue, footnotes_csv=pgp_dir / "footnotes.csv")
    stats = bpa.build(out, side_assignments=assignments, emit_unassigned=emitted, store_label_checks=bpa.load_store_label_checks(checks),
                      **common)
    manifest = {m["stem"]: m for m in map(json.loads, (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    assert {stem: (m["image_origin"], m["label"], m["side"], m["side_source"], m["label_check"]) for stem, m in manifest.items()} == {
        "pgpar_51_0_page": ("store", "file:1r", "verso", f"side_match:{READER}", "yes"),
        "pgpar_51_1_page": ("store", "file:1v", "recto", f"side_match:{READER}", "unclear"),
        "pgpar_52_1_page": ("store", "file:1v", "recto", f"side_match:{READER}", "yes"),   # image 0 was dropped: names keep their place
        "pgpar_52_2_page": ("store", "file:2r", "verso", f"side_match:{READER}", "unclear"),
        "pgpar_56_0_page": ("manifest", "1", "", "single", None)}                           # a manifest wins over the store
    assert manifest["pgpar_51_0_page"]["label_read"] == "AIU XII.51" and manifest["pgpar_56_0_page"]["label_read"] is None
    assert manifest["pgpar_52_1_page"]["image_url"] == fjp_image_url("S_52_1v.jpg")
    assert {pid: stats["skipped_documents"][pid] for pid in ("50", "53", "54", "55", "57")} == {
        "50": bpa.STORE_READS_ONLY,                                # one photograph, no read: never placed by the single-image rule
        "53": "edition without side labels and several images", "54": "store photograph not label-checked",
        "55": "store photograph labelled as another fragment: ENA 1.2", "57": bpa.STORE_ONE_LEFT}
    assert {reason: n for reason, n in stats["page_skips"].items() if n} == {"store photograph labelled as another fragment: AIU XII.53": 1,
                                                                            "store photograph labelled as another fragment: AIU XII.58": 1}
    assert stats["store_canvases"] == {"no label visible": 1, "yes": 5, "unclear": 2, "no": 3} and stats["store_canvases_unchecked"] == 1
    assert (stats["store_rows"], stats["store_documents"], stats["private"], stats["store_label_checks_loaded"]) == (4, 2, True, 12)
    assert json.loads((out / "stats.json").read_text(encoding="utf-8"))["private"] is True
    assert fjp_image_url("S 52_1r.jpg") not in fetched and fjp_image_url("S_56_1r.jpg") not in fetched    # dropped / not used: never fetched
    assert fjp_image_url("S_50_1v.jpg") not in fetched
    assert sorted(p.name for p in (tmp_path / "pgp_arabic_v3_images").iterdir()) == ["51_0.jpg", "51_1.jpg", "52_1.jpg", "52_2.jpg", "56_0.jpg"]
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "PRIVATE" in readme and "image store" in readme and "Never push" in readme and "never release" in readme
    assert "| 51 | AIU XII.51 | S_51_1r.jpg | yes | AIU XII.51 | row |" in readme and "| 50 |" not in readme
    assert "| 52 | AIU XII.52 | S 52_1r.jpg | no | AIU XII.53 | store photograph labelled as another fragment: AIU XII.53 |" in readme
    assert "| 56 |" not in readme and "| 57 |" not in readme                                                 # store documents with rows only
    records = {r["pgpid"]: r for r in map(json.loads, emitted.read_text(encoding="utf-8").splitlines())}
    assert sorted(records) == ["50", "53", "57"] and records["57"]["reason"] == bpa.STORE_ONE_LEFT
    assert records["50"]["reason"] == bpa.STORE_READS_ONLY and records["50"]["sections"] == {"": _sides(editions["50"])[""]}
    assert records["50"]["canvases"] == [{"index": 0, "label": "", "image_url": fjp_image_url("S_50_1v.jpg")}]
    assert records["53"]["canvases"] == [{"index": 0, "label": "", "image_url": fjp_image_url("S_53_1r.jpg")},
                                         {"index": 1, "label": "", "image_url": fjp_image_url("S_53_1v.jpg")}]
    assert records["57"]["canvases"] == [{"index": 0, "label": "", "image_url": fjp_image_url("S_57_1v.jpg")}]   # only what passed the check
    unchecked = bpa.build(tmp_path / "no_checks", **common)                       # no --store-label-checks: no store photograph is used
    assert {pid: unchecked["skipped_documents"][pid] for pid in store if pid != "56"} == {
        pid: "store photograph not label-checked" for pid in store if pid != "56"}
    assert (unchecked["pages"], unchecked["store_rows"], unchecked["private"], unchecked["store_canvases"]) == (1, 0, False, {})
    assert unchecked["store_canvases_unchecked"] == 12 and unchecked["store_label_checks_loaded"] is None
    assert not (tmp_path / "no_checks" / "README.md").exists()


def test_build_places_store_photographs_only_by_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A store photograph becomes a row only through an accepted side assignment whose read shows the edition's text (the store
    read gate); label checks are picked per document and a ``yes`` for another record drops the photograph; a whole volume of
    store photographs is skipped; ``stats.inputs`` records the label-check file.

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    :param monkeypatch: pytest's monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    unlabelled, verso_only = "\n".join(RECTO), "Verso\n" + "\n".join(VERSO)
    both = "Recto\n" + "\n".join(RECTO) + "\nVerso\n" + "\n".join(VERSO)
    store = {"70": ["70_image.jpg"], "71": ["S_71_1r.jpg"], "72": ["S_72_1v.jpg"], "73": ["S_73_1r.jpg", "S_73_1v.jpg", "S_73_2r.jpg"],
             "74": ["SH_1r.jpg"], "75": ["SH_1r.jpg"], "76": ["S_76_1r.jpg"], "77": ["S_77_1r.jpg"], "78": ["S_78_1r.jpg"],
             "79": ["S_79_1r.jpg"], "80": ["S_80_1r.jpg", "S_80_1v.jpg"]}
    editions = {"70": verso_only, "71": unlabelled, "72": verso_only, "73": unlabelled, "74": unlabelled, "75": unlabelled,
                "76": unlabelled, "77": verso_only, "78": unlabelled, "79": unlabelled, "80": both}
    documents = {pid: ({"canonical_id": f"Paris_AIU_XII_{pid}", "shelfmark": f"AIU XII.{pid}", "tier": "2", "iiif_urls": "",
                        "side": "verso" if editions[pid] == verso_only else "", "fjp_images": ";".join(store[pid]),
                        "merged_join": "shelfmark"}, editions[pid], [])
                 for pid in store}
    queue, pgp_dir = _write_inputs(tmp_path, documents, STORE_QUEUE_COLUMNS)
    fetched: List[str] = []

    def fake_get(url: str, timeout: int = 90) -> bytes:
        """Stand-in for ``bab.http_get``: a white JPEG for any store photograph.

        :param url: Requested URL (recorded in ``fetched``).
        :type url: str
        :param timeout: Ignored.
        :type timeout: int
        :return: JPEG bytes.
        :rtype: bytes
        """
        fetched.append(url)
        return _jpeg("white")

    monkeypatch.setattr(bab, "http_get", fake_get)
    checks_path = _write_label_checks(tmp_path / "label_checks.jsonl", [
        _label_check("70_image.jpg", "unclear", "", "472", pgpid="70"),                   # the label shows only the end of the mark
        _label_check("S_71_1r.jpg", "yes", "Paris_AIU_XII_71", "AIU XII.71", pgpid="71"),
        _label_check("S_72_1v.jpg", "yes", "Paris_AIU_XII_99", "AIU XII.99"),            # a check of the image alone, yes for record 99
        _label_check("SH_1r.jpg", "no", "Paris_AIU_XII_75", "AIU XII.75", pgpid="74"),   # one photograph filed under 74 and 75
        _label_check("SH_1r.jpg", "yes", "Paris_AIU_XII_75", "AIU XII.75", pgpid="75"),
        _label_check("S_76_1r.jpg", "yes", "Paris_AIU_XII_99", "AIU XII.99", pgpid="99"),  # checked for another document only
        _label_check("S_77_1r.jpg", "no label visible", pgpid="77"), _label_check("S_78_1r.jpg", "no label visible", pgpid="78"),
        _label_check("S_79_1r.jpg", "unclear", "", "XII", pgpid="79"), _label_check("S_80_1r.jpg", "no label visible", pgpid="80"),
        _label_check("S_80_1v.jpg", "no label visible", pgpid="80")])

    def assignment(pid: str, pages: List[Tuple[int, str]], decision: str = "accept",
                   evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """A side matcher's decision on the document's photographs.

        :param pid: Document id.
        :type pid: str
        :param pages: ``(canvas index, assign)`` per page.
        :type pages: List[Tuple[int, str]]
        :param decision: ``accept`` or ``drop``.
        :type decision: str
        :param evidence: The matcher's evidence; default: reads that match the edition well (``_read_scores``).
        :type evidence: Optional[Dict[str, Any]]
        :return: One ``side_assignments.jsonl`` record.
        :rtype: Dict[str, Any]
        """
        return {"pgpid": pid, "edition_fingerprint": bpa.edition_fingerprint(_sides(editions[pid])), "decision": decision, "reader": READER,
                "pages": [{"canvas_index": n, "image_url": fjp_image_url(store[pid][n]), "assign": assign} for n, assign in pages],
                "evidence": _read_scores(pages) if evidence is None else evidence}

    one_side_weak = _read_scores([(0, "recto")])
    one_side_weak["canvases"] += _read_scores([(1, "verso")], recall=0.1, precision=0.1)["canvases"]
    basmala_read = RECTO[0] + "\n" + "\n".join(VERSO)                 # a photograph of another letter that opens with the basmala
    assignments = {"71": assignment("71", [(0, "all")]), "72": assignment("72", [(0, "verso")]), "75": assignment("75", [(0, "all")]),
                   "77": assignment("77", [], decision="drop"),
                   "78": _scored_assignment("78", _sides(unlabelled), fjp_image_url("S_78_1r.jpg"), "all", basmala_read),
                   "79": assignment("79", [(0, "all")], evidence={"f1": 0.8}),                  # accepted without read scores
                   "80": assignment("80", [(0, "recto"), (1, "verso")], evidence=one_side_weak)}
    common = dict(delay_s=0, val_share=0.0, held_out=(set(), set()), benchmark_index=bdg.build_benchmark_index([]), in_training={},
                  queue_csv=queue, footnotes_csv=pgp_dir / "footnotes.csv", max_canvases=2,
                  store_label_checks=bpa.load_store_label_checks(checks_path))
    out, emitted = tmp_path / "v3", tmp_path / "match" / "unassigned.jsonl"
    stats = bpa.build(out, side_assignments=assignments, emit_unassigned=emitted, store_label_checks_path=checks_path, **common)
    manifest = {m["stem"]: m for m in map(json.loads, (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    assert {stem: (m["image_origin"], m["side"], m["side_source"], m["label_check"], m["label_read"]) for stem, m in manifest.items()} == {
        "pgpar_71_0_page": ("store", "", f"side_match:{READER}", "yes", "AIU XII.71"),
        "pgpar_75_0_page": ("store", "", f"side_match:{READER}", "yes", "AIU XII.75"),
        "pgpar_80_0_page": ("store", "recto", f"side_match:{READER}", "no label visible", None)}   # its verso read did not match
    assert stats["skipped_documents"] == {
        "70": bpa.STORE_READS_ONLY,                         # one photograph, a one-side edition, PGP says verso: still no row without a read
        "72": "store photograph labelled as another fragment: Paris_AIU_XII_99",                  # a yes for record 99 drops it
        "73": "whole volume (3 store photographs)", "74": "store photograph labelled as another fragment: Paris_AIU_XII_75",
        "76": bpa.NOT_LABEL_CHECKED, "77": "side match dropped", "78": bpa.STORE_WEAK_READ, "79": bpa.STORE_UNSCORED}
    assert {reason: n for reason, n in stats["page_skips"].items() if n} == {bpa.STORE_WEAK_READ: 1}
    assert stats["side_sources"] == {"side_match": 3} and stats["unassigned_emitted"] == 1
    assert stats["side_assignments"] == {"loaded": 7, "accepted": 3, "store_unmatched": 2, "dropped": 1, "stale": 0, "invalid": 0,
                                         "unused": 1}                                                    # 72 never reached its assignment
    assert stats["store_canvases"] == {"unclear": 2, "yes": 2, bpa.YES_FOR_ANOTHER: 1, "no": 1, "no label visible": 4}
    assert (stats["store_canvases_unchecked"], stats["store_label_checks_loaded"], stats["store_rows"], stats["store_documents"]) == (1, 11, 3, 3)
    assert stats["store_read_gate"] == {"ngram": 5, "min_recall": 0.15, "min_precision": 0.2, "min_shared_ngrams": 20,
                                        "short_letters": 100, "short_min_recall": 0.4}
    assert stats["inputs"]["store_label_checks"] == str(checks_path)
    assert stats["inputs"]["store_label_checks_sha256"] == hashlib.sha256(checks_path.read_bytes()).hexdigest()
    assert json.loads((out / "stats.json").read_text(encoding="utf-8"))["inputs"]["store_label_checks"] == str(checks_path)
    assert sorted(fetched) == sorted({fjp_image_url(name) for name in ("S_71_1r.jpg", "SH_1r.jpg", "S_80_1r.jpg")})   # rows only
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "| 80 | AIU XII.80 | S_80_1r.jpg | no label visible |  | row |" in readme
    assert "| 80 | AIU XII.80 | S_80_1v.jpg | no label visible |  | no row |" in readme and "| 72 |" not in readme
    assert ("store read gate (recall of the page text at least 0.15, or at least 0.4 for a page text under 100 Arabic letters; "
            "precision at least 0.2; at least 20 shared letter 5-grams)") in readme
    records = [json.loads(line) for line in emitted.read_text(encoding="utf-8").splitlines()]
    assert records == [{"pgpid": "70", "shelfmark": "AIU XII.70", "canonical_id": "Paris_AIU_XII_70", "library": "",
                        "edition_fingerprint": bpa.edition_fingerprint(_sides(verso_only)), "sections": {"verso": _sides(verso_only)["verso"]},
                        "canvases": [{"index": 0, "label": "", "image_url": fjp_image_url("70_image.jpg")}], "reason": bpa.STORE_READS_ONLY}]
    # without --emit-unassigned the document is skipped for the same reason; nothing records the label-check file
    quiet = bpa.build(tmp_path / "quiet", side_assignments=assignments, **common)
    assert quiet["skipped_documents"]["70"] == bpa.STORE_READS_ONLY and quiet["unassigned_emitted"] is None
    assert (quiet["inputs"]["store_label_checks"], quiet["inputs"]["store_label_checks_sha256"]) == (None, None)
    # round trip: the side matcher reads the one photograph (here: the verso text) and decides from the emitted record alone;
    # the next build places the page, its read having passed the store read gate
    answer = sm.decide_record(records[0], ["\n".join(VERSO)], READER)
    assert answer["decision"] == "accept" and answer["pages"] == [{"canvas_index": 0, "image_url": fjp_image_url("70_image.jpg"),
                                                                   "assign": "verso"}]
    again = bpa.build(tmp_path / "again", side_assignments=dict(assignments, **{"70": json.loads(json.dumps(answer))}), **common)
    placed = [m for m in map(json.loads, (tmp_path / "again" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) if m["pgpid"] == "70"]
    assert "70" not in again["skipped_documents"] and [(m["side"], m["side_source"], m["label_check"]) for m in placed] == [
        ("verso", f"side_match:{READER}", "unclear")]


def test_build_refuses_to_emit_unassigned_inside_the_datasets_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The side matcher caches next to ``unassigned.jsonl``, so the builder never writes it inside the NAS datasets directory.

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    :param monkeypatch: pytest's monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    datasets = tmp_path / "datasets"
    (datasets / "pgp_arabic_editions_v3").mkdir(parents=True)
    monkeypatch.setattr(bab, "NAS_DATASETS", datasets)                 # the default of check_unassigned_location, read when called
    _forbid_work(monkeypatch)
    queue, pgp_dir = _write_inputs(tmp_path, {})
    for emitted in (datasets / "pgp_arabic_editions_v3" / "unassigned.jsonl", datasets / "unassigned.jsonl",
                    _flip_case(datasets, "datasets") / "x" / "unassigned.jsonl"):
        with pytest.raises(ValueError, match="side matcher would keep its caches"):
            bpa.build(tmp_path / "v3", emit_unassigned=emitted, held_out=(set(), set()), benchmark_index=bdg.build_benchmark_index([]),
                      in_training={}, queue_csv=queue, footnotes_csv=pgp_dir / "footnotes.csv")
    assert sorted(p.name for p in datasets.iterdir()) == ["pgp_arabic_editions_v3"] and not (tmp_path / "v3").exists()
    bpa.check_unassigned_location(None)
    bpa.check_unassigned_location(tmp_path / "logs" / "unassigned.jsonl")
    bpa.check_unassigned_location(tmp_path / "datasets_logs" / "unassigned.jsonl")              # a sibling, not inside
