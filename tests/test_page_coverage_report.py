"""Tests for the bibliography page-coverage report.

The report must keep the pipeline stages distinct — a page that was
photographed but never structured is a re-run, not a return trip to the
library — and must read printed page numbers off two-page spreads correctly.
"""

import json

import pytest

from src.datasets.indexing.bibliography.page_coverage_report import (
    PageRecord,
    ScanCoverage,
    _collection_rollup,
    _format_ranges,
    _printed_number,
    assign_spans,
    build_scan_coverage,
    detect_stride,
    find_printed_gaps,
    volume_gaps,
    seam_spans,
    fit_pagination as detect_fit,
    infer_volume,
    known_page_count,
    tail_gap,
    leading_page_number,
    parse_seq,
)


@pytest.mark.parametrize("name,seq", [
    ("page_001_structured.json", 1),
    ("page_042.png", 42),
    ("page_107_entities.json", 107),
    ("cover.png", None),
])
def test_parse_seq(tmp_path, name, seq):
    assert parse_seq(tmp_path / name) == seq


@pytest.mark.parametrize("text,expected", [
    # A spread's verso running head comes first.
    ("188\nEFFECTING MARRIAGE\niii B\nPROPOSAL AND ACCEPTANCE\n189\ntext", 188),
    ("108\nSETTING THE SCENE\n1\nDRAMATIS PERSONAE\n109\npedigrees", 108),
    # Punctuation / whitespace before the number is tolerated.
    ("  200.\nEFFECTING MARRIAGE", 200),
    # No leading number at all.
    ("EFFECTING MARRIAGE\niii B\n188", None),
    ("", None),
    (None, None),
    # Implausible values are rejected rather than trusted.
    ("1980 Tel-Aviv University", None),
    ("0 nothing", None),
])
def test_leading_page_number(text, expected):
    assert leading_page_number(text) == expected


def test_detect_stride_spread_vs_single():
    spread = {1: 108, 2: 110, 3: 112, 4: 114, 5: 116}
    single = {1: 12, 2: 13, 3: 14, 4: 15}
    assert detect_stride(spread) == 2
    assert detect_stride(single) == 1
    assert detect_stride({}) == 1


def test_detect_stride_ignores_jumps_across_missing_sheets():
    """A missing sheet leaves a step of 4+; it must not skew the stride."""
    printed = {1: 100, 2: 102, 3: 104, 5: 112, 6: 114, 7: 116}
    assert detect_stride(printed) == 2


def test_printed_number_ignores_the_extracted_page_field():
    """``extracted_page_number`` names the recto and would break span parity."""
    structured = {
        "full_main_text": "188\nEFFECTING MARRIAGE\nPROPOSAL AND ACCEPTANCE\n189\n",
        "extracted_page_number": "189",
    }
    assert _printed_number(structured, ocr_text=None) == 188
    # No usable structured text: fall back to raw OCR.
    assert _printed_number({"extracted_page_number": "189"}, ocr_text="188\nMARRIAGE") == 188
    # Neither text has a leading number: report nothing rather than the recto,
    # which is what the interpolation in assign_spans is for.
    assert _printed_number({"extracted_page_number": "189"}, ocr_text="MARRIAGE\n189") is None
    assert _printed_number(None, ocr_text=None) is None


def test_fit_pagination_rejects_ocr_noise():
    """A footnote or plate number read as a running head must not set the range."""
    observations = {
        1: 24,    # abbreviations page, not a body page number
        2: 2, 3: 4, 5: 8, 6: 10, 7: 12, 8: 14, 9: 16,
        10: 200,  # a year or citation misread as the running head
    }
    fit = detect_fit(observations)
    assert fit is not None
    assert fit.stride == 2
    assert set(fit.rejected) == {1, 10}
    assert fit.accepted[2] == 2 and fit.accepted[9] == 16


def test_fit_pagination_needs_enough_observations():
    assert detect_fit({}) is None
    assert detect_fit({1: 100, 2: 102}) is None


def test_fit_pagination_handles_single_page_scans():
    fit = detect_fit({1: 10, 2: 11, 3: 12, 4: 13})
    assert fit.stride == 1
    assert not fit.rejected


@pytest.mark.parametrize("scan,volume", [
    ("friedman_108_201_vol_1", 1),
    ("friedman_0_100_vol2", 2),
    ("fridman_100_198_vol_2", 2),
    ("some_book_volume_3", 3),
    ("friedman_vol_intro", None),
    ("india_traders", None),
])
def test_infer_volume(scan, volume):
    assert infer_volume(scan) == volume


def test_infer_volume_metadata_override_beats_the_directory_name():
    """`friedman_vol_intro` is volume 1 pp. 1-107, which its name hides."""
    metadata = {"scan_volumes": {"friedman_vol_intro": 1}}
    assert infer_volume("friedman_vol_intro", metadata) == 1
    # A bad entry falls through to the name rather than raising.
    assert infer_volume("x_vol_2", {"scan_volumes": {"x_vol_2": "not a number"}}) == 2


@pytest.mark.parametrize("metadata,volume,expected", [
    ({"volumes": {"1": {"page_count": 492}, "2": {"page_count": 517}}}, 1, 492),
    ({"volumes": {"1": {"page_count": 492}, "2": {"page_count": 517}}}, 2, 517),
    # No volume stated: fall back to the book-level count.
    ({"page_count": 240}, None, 240),
    ({"publication": {"pages": "312"}}, None, 312),
    # A per-volume count wins over the book-level one.
    ({"page_count": 1000, "volumes": {"1": {"page_count": 492}}}, 1, 492),
    # Unusable or absent values report nothing rather than guessing.
    ({}, 1, None),
    ({"page_count": "many"}, None, None),
    ({"page_count": 0}, None, None),
])
def test_known_page_count(metadata, volume, expected):
    assert known_page_count(metadata, volume) == expected


def test_tail_gap_reports_the_unscanned_remainder():
    """Scans stopping at p. 201 of a 492-page volume leave 202-492 unscanned."""
    cov = ScanCoverage(scan="s", collection="c", volume=1, title="t", book_uuid="u",
                       stride=2, pages=[_page(1, 200)], known_page_count=492)
    assert tail_gap([cov]) == (202, 492)


def test_tail_gap_is_none_when_the_scans_reach_the_end():
    cov = ScanCoverage(scan="s", collection="c", volume=1, title="t", book_uuid="u",
                       stride=2, pages=[_page(1, 238)], known_page_count=239)
    assert tail_gap([cov]) is None


def test_tail_gap_is_none_without_a_known_extent():
    """Silence, not a guess, when the page count was never recorded."""
    cov = ScanCoverage(scan="s", collection="c", volume=1, title="t", book_uuid="u",
                       stride=2, pages=[_page(1, 200)])
    assert tail_gap([cov]) is None


def test_tail_gap_covers_the_whole_book_when_nothing_is_placed():
    cov = ScanCoverage(scan="s", collection="c", volume=1, title="t", book_uuid="u",
                       stride=2, pages=[], known_page_count=100)
    assert tail_gap([cov]) == (1, 100)


def test_rollup_marks_a_volume_incomplete_when_a_tail_remains():
    """Every stage can be clean and the volume still be two-thirds unscanned."""
    cov = ScanCoverage(scan="s_vol_1", collection="work", volume=1, title="t",
                       book_uuid="u", stride=2,
                       pages=[_page(1, 100, has_image=True, ocr_chars=5,
                                    has_structured=True, in_es=True)],
                       known_page_count=492)
    row = _collection_rollup([cov])[0]
    assert cov.is_complete is True          # no stage gap...
    assert row["complete"] is False         # ...but the book is not covered
    assert row["tail_gap"] == [102, 492]
    assert row["tail_pages"] == 391


def _page(seq, start, stride=2, **kwargs):
    """Build a PageRecord covering a printed span.

    :param seq: Sequence index.
    :param start: First printed page on the sheet (None for unreadable).
    :param stride: Printed pages per sheet.
    :param kwargs: Stage flags to override.
    :returns: The record.
    """
    end = None if start is None else start + stride - 1
    return PageRecord(seq=seq, printed_start=start, printed_end=end, **kwargs)


def test_printed_pages_expands_a_spread():
    assert _page(1, 188).printed_pages == [188, 189]
    assert _page(1, 188, stride=1).printed_pages == [188]
    assert _page(1, None).printed_pages == []


def test_find_printed_gaps_reports_pages_behind_missing_sheets():
    pages = [_page(1, 100), _page(2, 102), _page(4, 108), _page(5, 110)]
    # Sheet 3 is absent, so 104-107 were never photographed.
    assert find_printed_gaps(pages) == [104, 105, 106, 107]
    assert find_printed_gaps([]) == []
    assert find_printed_gaps([_page(1, None)]) == []


def test_find_printed_gaps_is_silent_when_every_sheet_is_present():
    """A dense sheet run has no scan gap, whatever the running heads read.

    Sheet 2's span here is one page short of tiling — the kind of drift a
    misread running head causes. Comparing covered pages against the range
    would report 103 as missing; only an absent sheet may raise a gap.
    """
    pages = [_page(1, 100), _page(2, 104), _page(3, 106)]
    assert find_printed_gaps(pages) == []


def test_assign_spans_interpolates_without_breaking_parity():
    """An unread sheet takes the parity of its neighbours, not a rounded fit."""
    observations = {1: 2, 2: 4, 4: 8, 5: 10, 6: 12}  # sheet 3 unreadable
    fit = detect_fit(observations)
    spans = assign_spans([1, 2, 3, 4, 5, 6], fit)
    assert spans[3] == (6, True)          # inferred, verso parity preserved
    assert spans[4] == (8, False)         # read directly
    # Every sheet tiles the printed range with no hole.
    covered = sorted(n for seq, (start, _) in spans.items() for n in (start, start + 1))
    assert covered == list(range(2, 14))


def test_assign_spans_without_a_fit():
    assert assign_spans([1, 2, 3], None) == {}


def test_stage_reached_walks_the_pipeline():
    assert _page(1, 100).stage_reached == "missing"
    assert _page(1, 100, has_image=True).stage_reached == "image_only"
    assert _page(1, 100, has_image=True, ocr_chars=500).stage_reached == "ocr"
    assert _page(
        1, 100, has_image=True, ocr_chars=500, has_structured=True
    ).stage_reached == "structured"
    assert _page(
        1, 100, has_image=True, ocr_chars=500, has_structured=True, in_es=True
    ).stage_reached == "indexed"
    assert _page(
        1, 100, has_image=True, ocr_chars=500, has_structured=True, in_es=True, kg_relations=3
    ).stage_reached == "kg"


def test_scan_coverage_separates_stage_gaps():
    """The three stage gaps must not be conflated — each has a different fix."""
    cov = ScanCoverage(scan="s", collection="c", title="t", book_uuid="u", stride=2, pages=[
        # Fully processed.
        _page(1, 100, has_image=True, ocr_chars=500, has_structured=True, in_es=True),
        # Photographed, OCR failed.
        _page(2, None, has_image=True),
        # OCR'd but never structured — re-run the pipeline, do not re-scan.
        _page(3, 104, has_image=True, ocr_chars=500),
        # Structured but never indexed.
        _page(4, 106, has_image=True, ocr_chars=500, has_structured=True),
    ])
    assert cov.ocr_gaps == [2]
    assert cov.structuring_gaps == [3]
    assert cov.index_gaps == [4]
    assert cov.image_count == 4
    assert cov.structured_count == 2
    assert cov.es_count == 1
    assert cov.printed_span == (100, 107)
    assert cov.is_complete is False


def test_scan_coverage_complete_when_every_stage_passed():
    cov = ScanCoverage(scan="s", collection="c", title="t", book_uuid="u", stride=2, pages=[
        _page(1, 100, has_image=True, ocr_chars=5, has_structured=True, in_es=True),
        _page(2, 102, has_image=True, ocr_chars=5, has_structured=True, in_es=True),
    ])
    assert cov.printed_gaps == find_printed_gaps(cov.pages) == []
    assert cov.is_complete is True


@pytest.mark.parametrize("numbers,expected", [
    ([], ""),
    ([5], "5"),
    ([1, 2, 3], "1-3"),
    ([12, 18, 19, 20, 21, 22, 23, 24, 31], "12, 18-24, 31"),
    ([3, 1, 2], "1-3"),  # unsorted input
])
def test_format_ranges(numbers, expected):
    assert _format_ranges(numbers) == expected


def test_collection_rollup_unions_sibling_scans():
    """Two scans of one work must be judged on their combined coverage."""
    first = ScanCoverage(scan="vol_a", collection="work", title="t", book_uuid="u1", stride=2,
                         pages=[_page(1, 100), _page(2, 102)])
    second = ScanCoverage(scan="vol_b", collection="work", title="t", book_uuid="u2", stride=2,
                          pages=[_page(1, 104), _page(2, 106)])
    rows = _collection_rollup([first, second])
    assert len(rows) == 1
    row = rows[0]
    assert row["printed_span"] == [100, 107]
    assert row["printed_pages_covered"] == 8
    # The seam between the two scans is continuous, so no gap.
    assert row["printed_gaps"] == []
    assert sorted(row["scans"]) == ["vol_a", "vol_b"]


def test_collection_rollup_reports_a_seam_separately_from_confirmed_gaps():
    """A seam is not asserted as missing — the scans may be different works."""
    first = ScanCoverage(scan="vol_a", collection="work", title="t", book_uuid="u1", stride=2,
                         pages=[_page(1, 100)])
    second = ScanCoverage(scan="vol_b", collection="work", title="t", book_uuid="u2", stride=2,
                          pages=[_page(1, 106)])
    row = _collection_rollup([first, second])[0]
    assert row["printed_gaps"] == []
    assert row["seam_spans"] == [[102, 105]]
    assert row["seam_pages"] == 4
    assert row["one_work"] is True


def test_collection_rollup_flags_a_folder_of_separate_works():
    """Distinct titles mean the seams between scans are other people's pages."""
    first = ScanCoverage(scan="art_a", collection="journal", title="Article A",
                         book_uuid="u1", stride=2, pages=[_page(1, 8)])
    second = ScanCoverage(scan="art_b", collection="journal", title="Article B",
                          book_uuid="u2", stride=2, pages=[_page(1, 488)])
    row = _collection_rollup([first, second])[0]
    assert row["printed_gaps"] == []      # nothing confirmed missing
    assert row["one_work"] is False       # ...and the seam is not a gap
    assert row["seam_spans"] == [[10, 487]]


def test_collection_rollup_separates_volumes():
    """Each volume restarts at page 1; pooling them would mask real gaps."""
    one = ScanCoverage(scan="s_vol_1", collection="work", volume=1, title="t",
                       book_uuid="u1", stride=2, pages=[_page(1, 100), _page(2, 102)])
    two = ScanCoverage(scan="s_vol_2", collection="work", volume=2, title="t",
                       book_uuid="u2", stride=2, pages=[_page(1, 1), _page(2, 3)])
    rows = _collection_rollup([one, two])
    assert {row["label"] for row in rows} == {"work vol.1", "work vol.2"}
    by_label = {row["label"]: row for row in rows}
    assert by_label["work vol.1"]["printed_span"] == [100, 103]
    assert by_label["work vol.2"]["printed_span"] == [1, 4]


def test_volume_gaps_ignores_misread_running_heads():
    """Only absent sheets and unmet seams count — not set-subtraction holes."""
    dense = ScanCoverage(scan="s", collection="work", volume=1, title="t", book_uuid="u",
                         stride=2, pages=[_page(1, 100), _page(2, 104), _page(3, 106)])
    dense.printed_gaps = find_printed_gaps(dense.pages)
    assert volume_gaps([dense]) == []


def test_volume_gaps_counts_only_missing_sheets():
    first = ScanCoverage(scan="a", collection="work", volume=1, title="t", book_uuid="u1",
                         stride=2, pages=[_page(1, 100), _page(3, 106)])
    first.printed_gaps = find_printed_gaps(first.pages)   # sheet 2 absent -> 102-105
    second = ScanCoverage(scan="b", collection="work", volume=1, title="t", book_uuid="u2",
                          stride=2, pages=[_page(1, 112)])
    # The 108-111 seam is reported by seam_spans, not asserted as missing.
    assert volume_gaps([first, second]) == [102, 103, 104, 105]
    assert seam_spans([first, second]) == [(108, 111)]


def test_seam_spans_tolerates_overlapping_scans():
    first = ScanCoverage(scan="a", collection="work", volume=1, title="t", book_uuid="u1",
                         stride=2, pages=[_page(1, 100), _page(2, 102)])
    second = ScanCoverage(scan="b", collection="work", volume=1, title="t", book_uuid="u2",
                          stride=2, pages=[_page(1, 102), _page(2, 104)])
    assert volume_gaps([first, second]) == []
    assert seam_spans([first, second]) == []


def test_build_scan_coverage_end_to_end(tmp_path):
    """A book with an unstructured sheet must be attributed to structuring."""
    root = tmp_path / "academic_literature"
    book = root / "work" / "scan_one"
    images = book / "scan_one_images"
    structured = book / "scan_one_structured_gemini"
    images.mkdir(parents=True)
    structured.mkdir(parents=True)
    (root / "work" / "work_metadata.json").write_text(
        json.dumps({"title": "A Study"}), encoding="utf-8"
    )

    for seq in (1, 2, 3):
        (images / f"page_{seq:03d}.png").write_bytes(b"png")
    for seq, verso in ((1, 100), (2, 102)):
        (structured / f"page_{seq:03d}_structured.json").write_text(
            json.dumps({
                "full_main_text": f"{verso}\nRUNNING HEAD\nOTHER HEAD\n{verso + 1}\nbody",
                "extracted_page_number": str(verso + 1),
            }),
            encoding="utf-8",
        )
    (book / "scan_one_ocr_results.json").write_text(
        json.dumps({"pages": [
            {"page_number": 1, "ocr_result": {"full_text": "100\nRUNNING HEAD\n101\nbody"}},
            {"page_number": 2, "ocr_result": {"full_text": "102\nRUNNING HEAD\n103\nbody"}},
            {"page_number": 3, "ocr_result": {"full_text": "104\nRUNNING HEAD\n105\nbody"}},
        ]}),
        encoding="utf-8",
    )

    task = {
        "metadata_file": root / "work" / "work_metadata.json",
        "structured_dir": structured,
        "image_dir": images,
        "book": "scan_one",
    }
    cov = build_scan_coverage(task, root, es_seqs={1}, kg_counts={})

    assert cov.scan == "scan_one"
    assert cov.collection == "work"
    assert cov.title == "A Study"
    assert cov.stride == 2
    assert cov.printed_span == (100, 105)
    assert cov.printed_gaps == []          # sheet 3 has OCR text, so 104-105 are covered
    assert cov.structuring_gaps == [3]     # ...but it was never structured
    assert cov.ocr_gaps == []
    assert cov.index_gaps == [2]           # structured, not in ES
    assert cov.es_count == 1
    assert cov.is_complete is False
