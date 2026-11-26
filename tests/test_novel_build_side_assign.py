"""Tests of stage 2 (sides) of the novel-rows build: placement by reads against a per-side null, review reasons, the
gate.

Everything is synthetic: random words over the Hebrew and Arabic alphabets (not texts), invented row ids, groups and
hashes. A page "read" is the target text with a few letters changed, so its recall is high; the null pool is reads of
random words belonging to no candidate.
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest

from src.finetuning.qwen_hebrew.novel_build import paths, side_assign as sa
from src.finetuning.qwen_hebrew.novel_build.schema import (Candidate, ImageRef, ReadRecord, SideDecision, TargetSide,
                                                           read_jsonl, write_jsonl)

HEBREW = "אבגדהוזחטיכלמנסעפצקרשת"                  # 22 Hebrew letters (an alphabet, not a text)
ARABIC = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"              # 28 Arabic letters (an alphabet, not a text)
KRAKEN = "kraken:test_model"
VLM = "vlm:test_key"
N_BOOT = 20


def words(rng: random.Random, alphabet: str, letters: int, per_line: int = 8) -> str:
    """Random words over an alphabet, about ``letters`` letters, in lines of ``per_line`` words.

    :param rng: Random generator.
    :param alphabet: Letters to draw from.
    :param letters: Approximate letter count.
    :param per_line: Words per line.
    :return: The text.
    """
    out: List[str] = []
    total = 0
    while total < letters:
        word = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 6)))
        out.append(word)
        total += len(word)
    return "\n".join(" ".join(out[i:i + per_line]) for i in range(0, len(out), per_line))


def noisy(rng: random.Random, text: str, alphabet: str, rate: float = 0.04) -> str:
    """The text with a few letters replaced (a decent OCR read of it).

    :param rng: Random generator.
    :param text: Source text.
    :param alphabet: Replacement letters.
    :param rate: Share of letters replaced.
    :return: The noisy copy.
    """
    return "".join(rng.choice(alphabet) if ch in alphabet and rng.random() < rate else ch for ch in text)


def read(sha: str, text: str, reader: str = KRAKEN, finish: str = "ok",
         meta: Optional[Dict[str, object]] = None) -> ReadRecord:
    """A read record.

    :param sha: Photograph hash.
    :param text: Read text.
    :param reader: Reader id.
    :param finish: Finish code.
    :param meta: Reader extras (the page reader's finish fields).
    :return: The record.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    return ReadRecord(sha256=sha, reader=reader, text=text, lines=lines, n_lines=len(lines),
                      letters_hebrew=sum(ch in HEBREW for ch in text), letters_arabic=sum(ch in ARABIC for ch in text),
                      finish=finish, seconds=1.0, read_at="2026-11-03T00:00:00Z", meta=dict(meta or {}))


def null_reads(seed: int, alphabet: str = HEBREW, reader: str = KRAKEN, n: int = 60) -> List[ReadRecord]:
    """Reads of photographs of no candidate (the null pool).

    :param seed: Random seed.
    :param alphabet: Alphabet of the reads.
    :param reader: Reader id.
    :param n: Number of reads.
    :return: Reads.
    """
    rng = random.Random(seed)
    return [read(f"null{seed}_{i}", words(rng, alphabet, rng.randint(150, 280)), reader) for i in range(n)]


def target(side_key: str, text: str, marker: Optional[str] = None) -> TargetSide:
    """A target side.

    :param side_key: Side key.
    :param text: Target text.
    :param marker: The edition's own side marker.
    :return: The target.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    return TargetSide(side_key=side_key, text=text, raw_text=text, lines=lines, pointer={"source": "synthetic"},
                      side_marker=marker)


def image(sha: str, origin: str = "iiif_cudl", label: Optional[str] = None, status: str = "present",
          width: int = 2000, height: int = 3000) -> ImageRef:
    """A photograph.

    :param sha: Its hash.
    :param origin: Origin.
    :param label: Canvas label.
    :param status: Status.
    :param width: Pixel width.
    :param height: Pixel height.
    :return: The reference.
    """
    return ImageRef(image_id=f"{origin}:{sha}", origin=origin, path=f"/tmp/none/{sha}.jpg", sha256=sha, label=label,
                    status=status, width=width, height=height)


def cand(row_id: str, targets: Sequence[TargetSide], images: Sequence[ImageRef], group: Optional[str] = None,
         source: str = "pgp") -> Candidate:
    """A candidate.

    :param row_id: Row id.
    :param targets: Target sides.
    :param images: Photographs.
    :param group: Document group (default: the row id).
    :param source: Source.
    :return: The candidate.
    """
    return Candidate(row_id=row_id, source=source, document_group=group or f"group:{row_id}", canonical_ids=[],
                     pgpids=[], ktiv_id=None, shelfmark="", script="hebrew", languages_stated=None, verdict="novel",
                     witnesses=[], unit="document", n_letters=0, targets=list(targets), images=list(images))


def by_side(decisions: Sequence[SideDecision], row_id: str) -> Dict[str, SideDecision]:
    """Decisions of one candidate by side key.

    :param decisions: Decisions.
    :param row_id: Row id.
    :return: ``{side_key: decision}``.
    """
    return {d.side_key: d for d in decisions if d.row_id == row_id}


def two_sides(seed: int, alphabet: str = HEBREW, letters: int = 220) -> Sequence[str]:
    """Two random side texts.

    :param seed: Random seed.
    :param alphabet: Alphabet.
    :param letters: Letters per side.
    :return: ``(recto, verso)``.
    """
    rng = random.Random(seed)
    return words(rng, alphabet, letters), words(rng, alphabet, letters)


# --- placement --------------------------------------------------------------------------------------------------------

def test_two_image_document_is_placed_by_reads_at_p99() -> None:
    """Recto and verso go to the photographs whose reads hold them (not to the image order), at p99, with coverage."""
    rng = random.Random(1)
    recto, verso = two_sides(11)
    c = cand("pgp:1", [target("recto", recto), target("verso", verso)], [image("img_a"), image("img_b")])
    reads = null_reads(100) + [read("img_a", noisy(rng, verso, HEBREW)), read("img_b", noisy(rng, recto, HEBREW))]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:1")
    assert out["recto"].outcome == "accepted" and out["recto"].image_sha256 == "img_b"
    assert out["verso"].outcome == "accepted" and out["verso"].image_sha256 == "img_a"
    assert out["recto"].basis == "p99" and out["recto"].reasons == []
    assert out["recto"].recall > out["recto"].null_p99 and out["recto"].margin >= sa.MIN_MARGIN
    first = out["recto"].competitors[0]
    assert first["role"] == "chosen" and first["sha256"] == "img_b"
    assert first["coverage"]["reader"] == KRAKEN and 0.8 < first["coverage"]["letters_ratio"] < 1.2
    assert first["coverage"]["kraken_lines"] == first["coverage"]["read_lines"]


def test_arabic_document_uses_rasm_5grams_and_the_page_reader() -> None:
    """Arabic-script sides are scored on rasm 5-grams against a page-reader null and placed by reads."""
    rng = random.Random(2)
    recto, verso = two_sides(12, ARABIC)
    c = cand("pgp:2", [target("recto", recto), target("verso", verso)], [image("ar_a"), image("ar_b")])
    reads = null_reads(101, ARABIC, VLM) + [read("ar_a", noisy(rng, recto, ARABIC), VLM),
                                            read("ar_b", noisy(rng, verso, ARABIC), VLM)]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:2")
    assert out["recto"].outcome == "accepted" and out["recto"].image_sha256 == "ar_a"
    assert out["verso"].outcome == "accepted" and out["verso"].image_sha256 == "ar_b"
    assert out["recto"].reader == VLM
    assert "null_n5" in out["recto"].competitors[0] and "null_n3" in out["recto"].competitors[0]


def test_single_image_below_p99_goes_to_review() -> None:
    """A side with one readable photograph that misses p99 goes to review with ``single_image_below_p99``."""
    rng = random.Random(3)
    recto, _ = two_sides(13)
    c = cand("pgp:3", [target("recto", recto)], [image("solo")])
    reads = null_reads(102) + [read("solo", words(rng, HEBREW, 200))]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:3")
    assert out["recto"].outcome == "needs_review"
    assert out["recto"].reasons == [sa.SINGLE_BELOW_P99]
    assert out["recto"].image_sha256 is None


def test_single_image_at_p99_is_accepted_on_the_single_image_basis() -> None:
    """A side with one readable photograph clearing p99 is accepted on the single-image basis."""
    rng = random.Random(4)
    recto, _ = two_sides(14)
    c = cand("pgp:4", [target("recto", recto)], [image("solo4")])
    reads = null_reads(103) + [read("solo4", noisy(rng, recto, HEBREW))]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:4")
    assert out["recto"].outcome == "accepted" and out["recto"].basis == "single_image_p99"


def test_both_sides_clearing_on_one_image_is_a_conflict() -> None:
    """Two sides whose texts are both read on one photograph go to review with the conflict reasons."""
    rng = random.Random(5)
    recto, verso = two_sides(15)
    c = cand("pgp:5", [target("recto", recto), target("verso", verso)], [image("spread"), image("other")])
    spread = noisy(rng, recto, HEBREW) + "\n" + noisy(rng, verso, HEBREW)
    reads = null_reads(104) + [read("spread", spread), read("other", words(rng, HEBREW, 200))]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:5")
    assert {d.outcome for d in out.values()} == {"needs_review"}
    reasons = [r for d in out.values() for r in d.reasons]
    assert sa.OTHER_SIDE_ON_IMAGE in reasons
    assert sa.ON_IMAGE_OF_OTHER_SIDE in reasons


@pytest.mark.parametrize("finish, text_kind", [("looping", "page"), ("ok", "loop")])
def test_looping_read_goes_to_review(finish: str, text_kind: str) -> None:
    """A looping read (finish code or a repeated phrase) is unusable and raises ``read_looping``.

    :param finish: Finish code of the read.
    :param text_kind: ``page`` (a page read) or ``loop`` (one phrase repeated).
    """
    rng = random.Random(6)
    recto, _ = two_sides(16)
    phrase = words(rng, HEBREW, 30, per_line=50).replace("\n", " ")
    text = noisy(rng, recto, HEBREW) if text_kind == "page" else "\n".join([phrase] * 12)
    c = cand("pgp:6", [target("recto", recto)], [image("loop")])
    reads = null_reads(105) + [read("loop", text, finish=finish)]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:6")
    assert out["recto"].outcome == "needs_review"
    assert sa.READ_LOOPING in out["recto"].reasons and sa.NO_USABLE_READ in out["recto"].reasons


def test_looping_read_of_another_reader_is_reported() -> None:
    """A looping read is reported even when the preferred reader's read failed for another reason."""
    rng = random.Random(61)
    recto, _ = two_sides(26)
    c = cand("pgp:60", [target("recto", recto)], [image("mixed_reads")])
    zero_lines = ReadRecord("mixed_reads", KRAKEN, "", [], 0, 0, 0, "blank", 1.0, "2026-11-03T00:00:00Z")
    looping = read("mixed_reads", noisy(rng, recto, HEBREW), VLM, finish="looping")
    out = by_side(sa.assign_sides([c], null_reads(115) + [zero_lines, looping], n_boot=N_BOOT, seed=3), "pgp:60")
    assert out["recto"].outcome == "needs_review" and out["recto"].reasons == [sa.NO_USABLE_READ, sa.READ_LOOPING]


def test_a_null_too_small_to_calibrate_goes_to_review() -> None:
    """A null of fewer than 20 reads cannot calibrate p95 / p99: review with ``null_too_small``."""
    rng = random.Random(62)
    recto, _ = two_sides(27)
    c = cand("pgp:61", [target("recto", recto)], [image("thin_null")])
    reads = null_reads(116, n=5) + [read("thin_null", noisy(rng, recto, HEBREW))]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:61")
    assert out["recto"].outcome == "needs_review" and out["recto"].reasons == [sa.NULL_TOO_SMALL]


def test_no_read_and_no_image_are_unassigned() -> None:
    """No read, no photograph and no readable photograph each give ``unassigned`` with its reason."""
    recto, _ = two_sides(17)
    unread = cand("pgp:7", [target("recto", recto)], [image("unread")])
    no_image = cand("pgp:8", [target("recto", recto + " אב")], [])
    missing = cand("pgp:9", [target("recto", recto + " גד")], [image("gone", status="missing")])
    out = sa.assign_sides([unread, no_image, missing], null_reads(106), n_boot=N_BOOT, seed=3)
    assert by_side(out, "pgp:7")["recto"].outcome == "unassigned"
    assert by_side(out, "pgp:7")["recto"].reasons == [sa.NO_READ]
    assert by_side(out, "pgp:8")["recto"].reasons == [sa.NO_IMAGE]
    assert by_side(out, "pgp:9")["recto"].reasons == [sa.NO_USABLE_IMAGE]


def test_shorter_duplicate_version_is_dropped() -> None:
    """Of two versions of one text in a document group, one more than 5 % shorter than the other is dropped before
    placement, naming the kept one."""
    rng = random.Random(8)
    recto, _ = two_sides(18, letters=300)
    shorter = " ".join(recto.split()[:-6])
    long_c = cand("pgp:10", [target("recto", recto)], [image("dup_img")], group="group:shared")
    short_c = cand("fjp:10", [target("", shorter)], [image("dup_img2", origin="fjp_local")], group="group:shared",
                   source="fjp")
    reads = null_reads(107) + [read("dup_img", noisy(rng, recto, HEBREW)), read("dup_img2", noisy(rng, recto, HEBREW))]
    out = sa.assign_sides([long_c, short_c], reads, n_boot=N_BOOT, seed=3)
    dropped = by_side(out, "fjp:10")[""]
    assert dropped.outcome == "dropped" and dropped.reasons == [sa.DUPLICATE_SHORTER]
    assert dropped.competitors[0]["role"] == "kept_version" and dropped.competitors[0]["row_id"] == "pgp:10"
    assert by_side(out, "pgp:10")["recto"].outcome == "accepted"


def test_preferred_origin_wins_among_photographs_of_one_page() -> None:
    """Among photographs of one page from different campaigns the preferred origin is chosen."""
    rng = random.Random(9)
    recto, verso = two_sides(19)
    images = [image("cudl_r"), image("cudl_v"), image("nli_r", origin="iiif_nli"),
              image("store_r", origin="fjp_local")]
    c = cand("pgp:11", [target("recto", recto), target("verso", verso)], images)
    reads = null_reads(108) + [read("cudl_r", noisy(rng, recto, HEBREW, 0.02)),
                               read("cudl_v", noisy(rng, verso, HEBREW)),
                               read("nli_r", noisy(rng, recto, HEBREW, 0.08)),
                               read("store_r", noisy(rng, recto, HEBREW, 0.01))]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:11")
    assert out["recto"].outcome == "accepted" and out["recto"].image_sha256 == "nli_r"
    roles = {e["sha256"]: e["role"] for e in out["recto"].competitors}
    assert roles["nli_r"] == "chosen" and roles["cudl_r"] in ("assigned", "alternative")
    assert out["verso"].outcome == "accepted" and out["verso"].image_sha256 == "cudl_v"


def test_exact_tie_goes_to_the_preferred_origin() -> None:
    """Two photographs with the same read: the preferred origin wins the tie."""
    rng = random.Random(10)
    recto, _ = two_sides(20)
    page = noisy(rng, recto, HEBREW)
    c = cand("pgp:12", [target("recto", recto)], [image("tie_store", origin="fjp_bucket", status="fetched"),
                                                  image("tie_cudl")])
    reads = null_reads(109) + [read("tie_store", page), read("tie_cudl", page)]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:12")
    assert out["recto"].outcome == "accepted" and out["recto"].image_sha256 == "tie_cudl"


def test_two_photographs_of_one_campaign_holding_the_side_go_to_review() -> None:
    """A side read on two photographs of one campaign (a split page) goes to review."""
    rng = random.Random(11)
    recto, _ = two_sides(21, letters=400)
    half = len(recto) // 2
    c = cand("pgp:13", [target("", recto)], [image("half_1"), image("half_2")])
    reads = null_reads(110) + [read("half_1", noisy(rng, recto[:half], HEBREW)),
                               read("half_2", noisy(rng, recto[half:], HEBREW))]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:13")
    assert out[""].outcome == "needs_review" and sa.ON_ANOTHER_IMAGE in out[""].reasons


def test_shared_text_goes_to_the_group_whose_photograph_holds_it() -> None:
    """A text attached to two document groups is owned by the group whose photograph holds it."""
    rng = random.Random(12)
    recto, _ = two_sides(22)
    owner = cand("pgp:14", [target("recto", recto)], [image("own_img")])
    other = cand("lit:14", [target("", recto)], [image("other_img")], source="literature")
    reads = null_reads(111) + [read("own_img", noisy(rng, recto, HEBREW)), read("other_img", words(rng, HEBREW, 200))]
    out = sa.assign_sides([owner, other], reads, n_boot=N_BOOT, seed=3)
    assert by_side(out, "pgp:14")["recto"].outcome == "accepted"
    lost = by_side(out, "lit:14")[""]
    assert lost.outcome == "dropped" and lost.reasons == [sa.NOT_OWNER]


def test_photograph_accepted_for_two_documents_goes_to_review() -> None:
    """One photograph accepted for two documents sends both acceptances to review."""
    rng = random.Random(13)
    recto, verso = two_sides(23)
    page = noisy(rng, recto, HEBREW) + "\n" + noisy(rng, verso, HEBREW)
    first = cand("pgp:15", [target("recto", recto)], [image("same")])
    second = cand("pgp:16", [target("recto", verso)], [image("same")])
    reads = null_reads(112) + [read("same", page)]
    out = sa.assign_sides([first, second], reads, n_boot=N_BOOT, seed=3)
    for row in ("pgp:15", "pgp:16"):
        decision = by_side(out, row)["recto"]
        assert decision.outcome == "needs_review" and sa.IMAGE_TWICE_DOCUMENT in decision.reasons


def mirror_case(face_two: Dict[str, Dict[str, object]], second_origin: str) -> Dict[str, Dict[str, object]]:
    """Decide two faces whose face 1 clears p99 on ``img_1`` (0.40) and nothing on ``img_2`` (0.02).

    :param face_two: Face 2's scores per photograph.
    :param second_origin: Origin of ``img_2``.
    :return: Decisions by side key, plus ``"one"`` / ``"two"`` aliases.
    """
    one, two = faces(33, second_origin)
    pairs = {one.key: {"img_1": crafted(0.40, True, True), "img_2": crafted(0.02, False, False)}, two.key: face_two}
    result = sa.decide_candidate([one, two], sa.distinct_images(one.cand.images), pairs, {})
    return {"one": result[one.key], "two": result[two.key]}


def test_mirror_accepts_when_the_conflicting_side_holds_its_own_photograph_at_twice_the_recall() -> None:
    """Face 2 clears p95 on face 1's photograph, but recalls 0.50 on its own (over 2 x 0.15) and less than face 1
    there: face 1 is accepted by the mirrored rule and the pair is recorded."""
    result = mirror_case({"img_2": crafted(0.50, True, True), "img_1": crafted(0.15, True, False)}, "fjp_local")
    one = result["one"]
    assert one["outcome"] == "accepted" and one["basis"] == sa.MIRROR_BASIS
    assert one["reasons"] == [sa.CONFLICT_RATIO_OK]
    record = one["info"][0]
    assert (record["role"], record["row_id"], record["recall_here"], record["recall_own"]) == ("conflict", "store:2",
                                                                                             0.15, 0.50)
    assert result["two"]["outcome"] == "accepted" and result["two"]["reasons"] == [sa.SPILL_RATIO_OK]


def test_mirror_keeps_review_when_the_conflicting_side_is_unplaced() -> None:
    """Face 2 clears p95 on face 1's photograph but is placed nowhere: face 1 stays in review."""
    result = mirror_case({"img_2": crafted(0.05, False, False), "img_1": crafted(0.15, True, False)}, "fjp_local")
    assert result["one"]["outcome"] == "needs_review" and result["one"]["reasons"] == [sa.OTHER_SIDE_ON_IMAGE]
    assert result["one"]["info"][0]["placed"] is False


def test_mirror_keeps_review_when_the_conflicting_side_recalls_more_on_this_photograph() -> None:
    """Face 2 (accepted on a photograph of another campaign, 0.95 >= 2 x 0.45) recalls more on face 1's photograph
    than face 1 does: review."""
    result = mirror_case({"img_2": crafted(0.95, True, True), "img_1": crafted(0.45, True, False)}, "iiif_cudl")
    assert result["two"]["outcome"] == "accepted"
    assert result["one"]["outcome"] == "needs_review" and result["one"]["reasons"] == [sa.OTHER_SIDE_ON_IMAGE]


def test_mirror_keeps_review_at_one_and_a_half_times() -> None:
    """Face 2 is accepted on its own photograph with only 1.5 x its recall on face 1's photograph: review."""
    result = mirror_case({"img_2": crafted(0.30, True, True), "img_1": crafted(0.20, True, False)}, "iiif_cudl")
    assert result["two"]["outcome"] == "accepted"
    assert result["one"]["outcome"] == "needs_review" and result["one"]["reasons"] == [sa.OTHER_SIDE_ON_IMAGE]


def test_two_faces_that_excuse_each_other_are_accepted_together() -> None:
    """Each face spills onto the other's photograph and conflicts there; both ratios hold, so both are accepted."""
    one, two = faces(34)
    pairs = {one.key: {"img_1": crafted(0.60, True, True), "img_2": crafted(0.20, True, False)},
             two.key: {"img_2": crafted(0.50, True, True), "img_1": crafted(0.15, True, False)}}
    result = sa.decide_candidate([one, two], sa.distinct_images(one.cand.images), pairs, {})
    for key in (one.key, two.key):
        assert result[key]["outcome"] == "accepted" and result[key]["basis"] == sa.MIRROR_BASIS
        assert result[key]["reasons"] == [sa.SPILL_RATIO_OK, sa.CONFLICT_RATIO_OK]


def test_faces_edited_as_two_candidates_are_placed_jointly() -> None:
    """Two single-side candidates sharing two photographs resolve to opposite photographs in one assignment."""
    rng = random.Random(70)
    first, second = two_sides(28)
    images = [image("leaf_a", origin="fjp_local"), image("leaf_b", origin="fjp_local")]
    a = cand("store:10", [target("", first)], images, group="group:leaf10", source="fjp")
    b = cand("store:11", [target("", second)], images, group="group:leaf10", source="fjp")
    reads = null_reads(117) + [read("leaf_a", noisy(rng, second, HEBREW)), read("leaf_b", noisy(rng, first, HEBREW))]
    out = sa.assign_sides([a, b], reads, n_boot=N_BOOT, seed=3)
    one, two = by_side(out, "store:10")[""], by_side(out, "store:11")[""]
    assert (one.outcome, one.image_sha256) == ("accepted", "leaf_b")
    assert (two.outcome, two.image_sha256) == ("accepted", "leaf_a")
    other = next(e for e in one.competitors if e.get("sha256") == "leaf_a")
    assert other["assigned_to"] == "store:11#"


def test_candidates_of_one_group_sharing_no_photograph_stay_apart() -> None:
    """Candidates of one document group without a common photograph are decided one by one."""
    rng = random.Random(71)
    first, second = two_sides(29)
    a = cand("store:20", [target("", first)], [image("own_a", origin="fjp_local")], group="group:leaf20", source="fjp")
    b = cand("store:21", [target("", second)], [image("own_b", origin="fjp_local")], group="group:leaf20", source="fjp")
    reads = null_reads(118) + [read("own_a", noisy(rng, first, HEBREW)), read("own_b", noisy(rng, second, HEBREW))]
    out = sa.assign_sides([a, b], reads, n_boot=N_BOOT, seed=3)
    assert [(d.row_id, d.basis, len(d.competitors)) for d in out] == [("store:20", "single_image_p99", 1),
                                                                      ("store:21", "single_image_p99", 1)]
    assert [len(unit) for unit in sa.decision_units([a, b])] == [1, 1]


def test_background_spill_between_faces_is_excused_on_both_sides() -> None:
    """A face whose text is faintly read on the other face's photograph is accepted by the ratio rule, and the other
    face, whose only blocking reason is that conflict, by the mirrored rule (the pair is recorded for the eye sheet)."""
    rng = random.Random(72)
    first, second = two_sides(32, letters=300)
    chunk = " ".join(first.split()[:12])
    images = [image("bg_a", origin="fjp_local"), image("bg_b", origin="fjp_local")]
    a = cand("store:30", [target("", first)], images, group="group:leaf30", source="fjp")
    b = cand("store:31", [target("", second)], images, group="group:leaf30", source="fjp")
    reads = null_reads(119) + [read("bg_a", noisy(rng, first, HEBREW)),
                               read("bg_b", noisy(rng, second, HEBREW) + "\n" + noisy(rng, chunk, HEBREW))]
    out = sa.assign_sides([a, b], reads, n_boot=N_BOOT, seed=3)
    one, two = by_side(out, "store:30")[""], by_side(out, "store:31")[""]
    assert one.outcome == "accepted" and one.basis == sa.RATIO_BASIS and one.reasons == [sa.SPILL_RATIO_OK]
    assert one.image_sha256 == "bg_a"
    assert two.outcome == "accepted" and two.basis == sa.MIRROR_BASIS and two.reasons == [sa.CONFLICT_RATIO_OK]
    pair_record = next(e for e in two.competitors if e.get("role") == "conflict")
    assert (pair_record["row_id"], pair_record["side_key"], pair_record["own_sha256"]) == ("store:30", "", "bg_a")
    assert pair_record["recall_own"] >= sa.RECALL_RATIO * pair_record["recall_here"]


# --- the gate ---------------------------------------------------------------------------------------------------------

def test_gate_sources_check_the_given_photograph() -> None:
    """KTIV and pass-through sides keep their given photograph and pass only when the read reaches p95."""
    rng = random.Random(14)
    page, other = two_sides(24)
    good = cand("990000000000000001:FL1", [target("page", page)], [image("ktiv_ok", origin="ktiv_zip")], source="ktiv")
    bad = cand("990000000000000002:FL2", [target("page", other)], [image("ktiv_bad", origin="ktiv_zip")], source="ktiv")
    apd = cand("apd_v3:stem_1", [target("recto", page + " אב")],
               [image("cudl_extra"), image("apd_img", origin="apd_local")], source="apd_v3")
    reads = null_reads(113) + [read("ktiv_ok", noisy(rng, page, HEBREW)), read("ktiv_bad", words(rng, HEBREW, 200)),
                               read("apd_img", noisy(rng, page, HEBREW)), read("cudl_extra", noisy(rng, page, HEBREW))]
    out = sa.assign_sides([good, bad, apd], reads, n_boot=N_BOOT, seed=3)
    ok = by_side(out, "990000000000000001:FL1")["page"]
    assert ok.outcome == "accepted" and ok.basis == "gate_p95" and ok.image_sha256 == "ktiv_ok"
    assert ok.competitors[0]["coverage"]["target_letters"] > 0
    refused = by_side(out, "990000000000000002:FL2")["page"]
    assert refused.outcome == "needs_review"
    assert refused.reasons == [sa.GATE_BELOW_P95, sa.N3_BELOW_P95, sa.under_matched_reason(4)]
    given = by_side(out, "apd_v3:stem_1")["recto"]
    assert given.outcome == "accepted" and given.image_sha256 == "apd_img"
    assert [e["sha256"] for e in given.competitors] == ["apd_img"]


# --- rules on crafted scores ------------------------------------------------------------------------------------------

def crafted(recall: float, p95: bool, p99: bool, n3: bool = True, matched: int = 50) -> Dict[str, object]:
    """A usable pair score with chosen clears flags.

    :param recall: Primary recall.
    :param p95: Clears p95.
    :param p99: Clears p99.
    :param n3: The 3-gram recall clears its p95.
    :param matched: Matched primary n-grams.
    :return: The score.
    """
    return {"usable": True, "recall": recall, "secondary": recall, "clears_p95": p95, "clears_p99": p99,
            "n3_clears_p95": n3, "matched": matched, "reader": KRAKEN,
            "null_n4": {"p95": 0.1, "p99": 0.2, "n": 60}}


def test_p95_needs_an_agreeing_prior_from_both_edition_and_photograph() -> None:
    """p95 accepts only when the edition's marker and the photograph's label agree on the side."""
    text = words(random.Random(15), HEBREW, 120)
    c = cand("pgp:20", [target("recto", text, marker="Recto")],
             [image("lab_r", label="1r"), image("lab_v", label="1v")])
    side = sa.make_side(c, 0, c.targets[0])
    pairs = {side.key: {"lab_r": crafted(0.15, True, False), "lab_v": crafted(0.01, False, False)}}
    result = sa.decide_candidate([side], sa.distinct_images(c.images), pairs, {})
    assert result[side.key]["outcome"] == "accepted" and result[side.key]["basis"] == "p95_with_prior"
    c.targets[0].side_marker = None
    no_marker = sa.make_side(c, 0, c.targets[0])
    result = sa.decide_candidate([no_marker], sa.distinct_images(c.images), pairs, {})
    assert result[side.key]["outcome"] == "needs_review"
    assert result[side.key]["reasons"] == [sa.NOT_P99_WITHOUT_PRIOR]


def test_margin_n3_and_matched_grams_send_to_review() -> None:
    """A runner-up within the margin (Arabic script only), a weak 3-gram recall or too few matched n-grams review."""
    hebrew = cand("pgp:21", [target("recto", words(random.Random(16), HEBREW, 120))], [image("m_a"), image("m_b")])
    arabic = cand("pgp:22", [target("recto", words(random.Random(17), ARABIC, 120))], [image("m_a"), image("m_b")])
    images = sa.distinct_images(hebrew.images)
    for c, expected in ((arabic, [sa.MARGIN_REASON]), (hebrew, [])):
        side = sa.make_side(c, 0, c.targets[0])
        close = {side.key: {"m_a": crafted(0.50, True, True), "m_b": crafted(0.47, False, False)}}
        assert sa.decide_candidate([side], images, close, {})[side.key]["reasons"] == expected
    side = sa.make_side(hebrew, 0, hebrew.targets[0])
    weak = {side.key: {"m_a": crafted(0.50, True, True, n3=False, matched=3), "m_b": crafted(0.01, False, False)}}
    assert sa.decide_candidate([side], images, weak, {})[side.key]["reasons"] == [sa.N3_BELOW_P95,
                                                                                    sa.under_matched_reason(4)]


def faces(seed: int, second_origin: str = "fjp_local") -> List[sa.Side]:
    """Two single-side candidates of one document group sharing two photographs (``img_1``, ``img_2``).

    :param seed: Random seed of the texts.
    :param second_origin: Origin of ``img_2`` (another photo campaign than ``img_1`` unless ``fjp_local`` /
        ``fjp_bucket``).
    :return: Their sides (face 1, face 2).
    """
    first, second = two_sides(seed)
    images = [image("img_1", origin="fjp_local"), image("img_2", origin=second_origin)]
    a = cand("store:1", [target("", first)], images, group="group:leaf", source="fjp")
    b = cand("store:2", [target("", second)], images, group="group:leaf", source="fjp")
    return [sa.make_side(a, 0, a.targets[0]), sa.make_side(b, 0, b.targets[0])]


@pytest.mark.parametrize("spill, holder_recall, outcome", [(0.20, 0.50, "accepted"), (0.27, 0.50, "needs_review"),
                                                           (0.20, 0.15, "needs_review")])
def test_spill_ratio_rule(spill: float, holder_recall: float, outcome: str) -> None:
    """A spill is excused at twice the recall when the other face holds the photograph with a higher recall there.

    :param spill: Face 1's recall on face 2's photograph (it clears p95 there).
    :param holder_recall: Face 2's recall on its photograph.
    :param outcome: Expected outcome of face 1 (chosen recall 0.40: 2x at 0.20, 1.5x at 0.27).
    """
    one, two = faces(30)
    images = sa.distinct_images(one.cand.images)
    pairs = {one.key: {"img_1": crafted(0.40, True, True), "img_2": crafted(spill, True, False)},
             two.key: {"img_2": crafted(holder_recall, True, True), "img_1": crafted(0.02, False, False)}}
    result = sa.decide_candidate([one, two], images, pairs, {})
    decision = result[one.key]
    assert decision["outcome"] == outcome and decision["image"].sha256 == "img_1"
    if outcome == "accepted":
        assert decision["reasons"] == [sa.SPILL_RATIO_OK] and decision["basis"] == sa.RATIO_BASIS
        assert result[two.key]["outcome"] == "accepted" and result[two.key]["reasons"] == [sa.CONFLICT_RATIO_OK]
    else:
        assert decision["reasons"] == [sa.ON_ANOTHER_IMAGE]
        assert result[two.key]["reasons"] == [sa.OTHER_SIDE_ON_IMAGE]    # face 1 is not accepted: no mirror


def test_spill_onto_an_unassigned_photograph_and_p95_choices() -> None:
    """An unassigned spill photograph is excused at twice the recall only for a target that names its side (an
    unlabelled target may hold both faces); a side resting on p95 is never excused."""
    one, _ = faces(31)
    images = sa.distinct_images(one.cand.images)
    pairs = {one.key: {"img_1": crafted(0.40, True, True), "img_2": crafted(0.20, True, False)}}
    decision = sa.decide_candidate([one], images, pairs, {})[one.key]
    assert decision["outcome"] == "needs_review" and decision["reasons"] == [sa.ON_ANOTHER_IMAGE]
    one.target.side_key = "recto"
    decision = sa.decide_candidate([one], images, pairs, {})[one.key]
    assert decision["outcome"] == "accepted" and decision["basis"] == sa.RATIO_BASIS
    one.target.side_marker, one.cand.images[0].label = "r", "1r"
    pairs = {one.key: {"img_1": crafted(0.40, True, False), "img_2": crafted(0.20, True, False)}}
    decision = sa.decide_candidate([one], images, pairs, {})[one.key]
    assert decision["outcome"] == "needs_review" and decision["reasons"] == [sa.ON_ANOTHER_IMAGE]


def test_side_parsing_and_priors() -> None:
    """Side names are parsed from labels and markers; a bare marker flag falls back to the side key."""
    assert sa.side_of("1r") == "recto" and sa.side_of("f. 2v") == "verso" and sa.side_of('ע"ב') == "verso"
    assert sa.side_of("Recto") == "recto" and sa.side_of("00000001.tif") == "" and sa.side_of(None) == ""
    flagged = target("verso", "אבג", marker="marker")
    assert sa.edition_side(flagged) == "verso"
    assert sa.edition_side(target("verso", "אבג")) == ""


# --- CLI --------------------------------------------------------------------------------------------------------------

def test_main_reads_and_writes_the_stage_files(tmp_path: Path) -> None:
    """The CLI reads the images stage and the reads file and writes one decision per side.

    :param tmp_path: Pytest temporary directory.
    """
    rng = random.Random(17)
    recto, verso = two_sides(25)
    c = cand("pgp:30", [target("recto", recto), target("verso", verso)], [image("m1"), image("m2")])
    other = cand("fjp:30", [target("", verso + " אב")], [], source="fjp")
    write_jsonl(paths.stage_path(tmp_path, "images"), [c, other])
    write_jsonl(tmp_path / sa.READS_FILE, null_reads(114) + [read("m1", noisy(rng, recto, HEBREW)),
                                                              read("m2", noisy(rng, verso, HEBREW))])
    sa.main(["--work-dir", str(tmp_path), "--n-boot", "10", "--sources", "pgp"])
    out = list(read_jsonl(paths.stage_path(tmp_path, "sides"), SideDecision))
    assert [(d.row_id, d.side_key, d.outcome) for d in out] == [("pgp:30", "recto", "accepted"),
                                                               ("pgp:30", "verso", "accepted")]


# --- photographs of one face ------------------------------------------------------------------------------------------

def test_same_face_copies_fold_onto_the_larger_photograph() -> None:
    """Two photographs of one face (a store copy and a bucket copy) whose reads agree fold onto the larger one: the side
    is accepted there instead of going to review for a spill, and the fold is recorded with its containments."""
    rng = random.Random(80)
    recto, verso = two_sides(40)
    images = [image("bucket_r", origin="fjp_bucket", width=1500, height=2000),
              image("store_r", origin="fjp_local", width=4000, height=6000), image("store_v", origin="fjp_local")]
    c = cand("store:40", [target("r", recto), target("v", verso)], images, source="fjp")
    reads = null_reads(120) + [read("bucket_r", noisy(rng, recto, HEBREW, 0.01)),
                               read("store_r", noisy(rng, recto, HEBREW, 0.01)),
                               read("store_v", noisy(rng, verso, HEBREW, 0.01))]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "store:40")
    assert out["r"].outcome == "accepted" and out["r"].image_sha256 == "store_r" and out["r"].reasons == []
    fold = next(e for e in out["r"].competitors if e.get("role") == "same_face")
    assert fold["link"] == "same_face_of:store_r" and fold["sha256"] == "bucket_r"
    assert min(fold["containment_in_kept"], fold["containment_of_kept"]) >= sa.SAME_FACE_CONTAINMENT
    assert fold["shared_grams"] >= sa.SAME_FACE_MIN_SHARED
    assert "bucket_r" not in {e.get("sha256") for e in out["r"].competitors if e.get("role") != "same_face"}
    assert out["v"].outcome == "accepted" and out["v"].image_sha256 == "store_v"


def test_fold_needs_agreeing_usable_reads_and_keeps_the_larger_photograph() -> None:
    """Photographs fold onto the largest agreeing one (equal areas: the preferred origin); a photograph whose read
    disagrees or that has no usable read is never folded."""
    rng = random.Random(81)
    recto, verso = two_sides(41)
    page = noisy(rng, recto, HEBREW, 0.01)
    small = image("small", origin="iiif_cudl", width=1000, height=1500)
    big = image("big", origin="fjp_bucket", width=3000, height=4000)
    twin = image("twin", origin="iiif_nli", width=1000, height=1500)
    other = image("other")
    unread = image("unread", origin="cudl_local", width=5000, height=5000)
    views = sa.views_by_sha([read("small", page), read("big", noisy(rng, recto, HEBREW, 0.01)), read("twin", page),
                             read("other", noisy(rng, verso, HEBREW, 0.01)), read("unread", "", finish="blank")])
    kept, folds = sa.fold_same_faces([small, big, twin, other, unread], views, sa.HEBREW)
    assert [im.sha256 for im in kept] == ["big", "other", "unread"]
    assert {f["sha256"]: f["kept_sha256"] for f in folds} == {"small": "big", "twin": "big"}
    kept, folds = sa.fold_same_faces([small, twin], views, sa.HEBREW)
    assert [im.sha256 for im in kept] == ["twin"] and [f["link"] for f in folds] == ["same_face_of:twin"]


def test_same_face_compares_reads_of_one_reader_family() -> None:
    """When both photographs have usable reads of one family those are compared, not the best read of each."""
    rng = random.Random(82)
    recto, verso = two_sides(42)
    a, b = image("fa"), image("fb", origin="cudl_local")
    views = sa.views_by_sha([read("fa", noisy(rng, recto, HEBREW, 0.01)), read("fa", noisy(rng, verso, HEBREW), VLM),
                             read("fb", noisy(rng, recto, HEBREW, 0.01), VLM)])
    assert sa.same_face(b, a, views, sa.HEBREW) is None
    views["fb"].append(sa.make_view(read("fb", noisy(rng, recto, HEBREW, 0.01))))
    record = sa.same_face(b, a, views, sa.HEBREW)
    assert record is not None and (record["reader"], record["kept_reader"]) == (KRAKEN, KRAKEN)


def test_the_bucket_mirror_is_its_own_campaign() -> None:
    """The PGP bucket mirror is never a sibling of the library photographs: a mirror whose read is too noisy to fold
    is another campaign's photograph, not a spill, and the side is accepted on the library photograph."""
    assert sa.campaign_of("pgpid_mirror") == "pgpid_mirror" and sa.campaign_of("iiif_pgp") == "pgp"
    rng = random.Random(90)
    recto, verso = two_sides(50)
    images = [image("lib_r", origin="iiif_pgp", width=6000, height=8000), image("lib_v", origin="iiif_pgp"),
              image("mirror", origin="pgpid_mirror", width=1200, height=1600)]
    c = cand("pgp:50", [target("recto", recto), target("verso", verso)], images)
    reads = null_reads(130) + [read("lib_r", noisy(rng, recto, HEBREW, 0.02)), read("lib_v", noisy(rng, verso, HEBREW)),
                               read("mirror", noisy(rng, recto, HEBREW, 0.12))]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:50")
    assert out["recto"].outcome == "accepted" and out["recto"].image_sha256 == "lib_r" and out["recto"].reasons == []
    assert not [e for e in out["recto"].competitors if e.get("role") == "same_face"]
    roles = {e["sha256"]: e["role"] for e in out["recto"].competitors if e.get("sha256")}
    assert roles["mirror"] == "alternative"


# --- the read that places a side --------------------------------------------------------------------------------------

def test_the_read_clearing_the_highest_threshold_wins() -> None:
    """A short Kraken read that does not hold the side loses to a page-reader read clearing p99 against its own null:
    the decision names the page reader and lists both reads."""
    rng = random.Random(83)
    recto, _ = two_sides(43)
    c = cand("pgp:43", [target("recto", recto)], [image("two_reads")])
    reads = (null_reads(121) + null_reads(122, reader=VLM)
             + [read("two_reads", words(rng, HEBREW, 40)), read("two_reads", noisy(rng, recto, HEBREW), VLM)])
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:43")["recto"]
    assert out.outcome == "accepted" and out.reader == VLM and out.basis == "single_image_p99"
    first = out.competitors[0]
    assert [r["reader"] for r in first["reads_scored"]] == [KRAKEN, VLM]
    assert first["reads_scored"][1]["level"] == "p99" and first["reads_scored"][0]["level"] == "below"
    assert first["coverage"]["reader"] == VLM


def test_a_read_whose_family_null_is_too_small_cannot_win() -> None:
    """A perfect page-reader read whose family's null has 5 reads cannot win; the Kraken read places the side."""
    rng = random.Random(84)
    recto, _ = two_sides(44)
    c = cand("pgp:44", [target("recto", recto)], [image("thin_family")])
    reads = (null_reads(123) + null_reads(124, reader=VLM, n=5)
             + [read("thin_family", noisy(rng, recto, HEBREW, 0.08)), read("thin_family", recto, VLM)])
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:44")["recto"]
    assert out.outcome == "accepted" and out.reader == KRAKEN
    assert out.competitors[0]["reads_scored"][1] == {"reader": VLM, "why": sa.NULL_TOO_SMALL}


@pytest.mark.parametrize("finish, meta, outcome", [
    ("length", {}, "accepted"),
    ("looping", {"finish_reason": "length", "looped": True, "cut_at": None, "repeat_share": 0.05}, "accepted"),
    ("looping", {"finish_reason": "length", "looped": True, "cut_at": 120, "repeat_share": 0.05}, "needs_review")])
def test_a_capped_read_places_a_side(finish: str, meta: Dict[str, object], outcome: str) -> None:
    """A read the token budget cut without a loop (``length``, or a stored ``looping`` whose meta shows only the cut)
    is evidence and places a side; a read whose loop the detector found does not.

    :param finish: Stored finish code.
    :param meta: The page reader's fields stored with the read.
    :param outcome: Expected outcome.
    """
    rng = random.Random(85)
    recto, _ = two_sides(45)
    c = cand("pgp:45", [target("recto", recto)], [image("capped")])
    reads = null_reads(125, reader=VLM) + [read("capped", noisy(rng, recto, HEBREW), VLM, finish=finish, meta=meta)]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:45")["recto"]
    assert out.outcome == outcome
    if outcome == "accepted":
        assert out.reader == VLM and out.basis == "single_image_p99"
    else:
        assert out.reasons == [sa.NO_USABLE_READ, sa.READ_LOOPING]


# --- versions and editions --------------------------------------------------------------------------------------------

def test_versions_with_different_side_labels_go_to_review() -> None:
    """Two versions of one text labelled recto and verso are a label conflict: both go to review, none is dropped."""
    rng = random.Random(86)
    recto, _ = two_sides(46, letters=300)
    c = cand("pgp:46", [target("recto", recto), target("verso", " ".join(recto.split()[:-1]))],
             [image("vc_a"), image("vc_b")])
    reads = null_reads(126) + [read("vc_a", noisy(rng, recto, HEBREW)), read("vc_b", words(rng, HEBREW, 200))]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:46")
    for key, other in (("recto", "verso"), ("verso", "recto")):
        assert out[key].outcome == "needs_review" and out[key].reasons == [sa.VERSION_CONFLICT]
        record = next(e for e in out[key].competitors if e.get("role") == "version_conflict")
        assert record["side_key"] == other and record["share"] >= sa.DUPLICATE_SHARE
    assert sa.labels_differ(target("1r", "אב"), target("2r", "אב"))
    assert not sa.labels_differ(target("recto", "אב"), target("r", "אב"))
    assert not sa.labels_differ(target("recto", "אב"), target("", "אב"))


def test_the_version_with_fewer_marks_within_five_percent_is_kept() -> None:
    """Of two versions within 5 % of each other's length that placement cannot tell apart (neither is read), the one
    with fewer apparatus marks is kept, even when shorter; the other is dropped with ``duplicate_version_more_marks``
    and a ``version_of`` record holding both versions' recall and marks."""
    recto, _ = two_sides(47, letters=400)
    tokens = recto.split()
    marked = " ".join(w + "?" if i % 5 == 0 else w for i, w in enumerate(tokens)) + " [...]"
    clean = " ".join(tokens[:-1])
    a = cand("pgp:47", [target("recto", marked)], [image("mk_a")], group="group:47")
    b = cand("fjp:47", [target("", clean)], [image("mk_b", origin="fjp_local")], group="group:47", source="fjp")
    out = sa.assign_sides([a, b], null_reads(127), n_boot=N_BOOT, seed=3)
    dropped = by_side(out, "pgp:47")["recto"]
    assert dropped.outcome == "dropped" and dropped.reasons == [sa.DUPLICATE_MORE_MARKS]
    record = next(e for e in dropped.competitors if e.get("role") == "kept_version")
    assert record["link"] == "version_of:fjp:47:" and record["decided_by"] == "marks"
    assert record["recall"] == [None, None] and record["marks"][0] > record["marks"][1]
    assert by_side(out, "fjp:47")[""].outcome != "dropped"


def test_editions_of_one_side_from_two_sources_do_not_compete() -> None:
    """A literature edition of a PGP side (Dice under the version threshold) is decided apart from it; both land on
    the same photograph, the PGP edition is kept and the other dropped as ``duplicate_edition_on_image``."""
    rng = random.Random(88)
    recto, verso = two_sides(48, letters=300)
    excerpt = " ".join(recto.split()[: int(len(recto.split()) * 0.7)]) + "\n" + words(rng, HEBREW, 90)
    images = [image("ed_r"), image("ed_v")]
    pgp = cand("pgp:48", [target("recto", recto), target("verso", verso)], images, group="group:48")
    lit = cand("lit:48", [target("", excerpt)], images, group="group:48", source="literature")
    reads = null_reads(128) + [read("ed_r", noisy(rng, recto, HEBREW)), read("ed_v", noisy(rng, verso, HEBREW))]
    out = sa.assign_sides([pgp, lit], reads, n_boot=N_BOOT, seed=3)
    assert [d.outcome for d in by_side(out, "pgp:48").values()] == ["accepted", "accepted"]
    edition = by_side(out, "lit:48")[""]
    assert edition.outcome == "dropped" and edition.reasons == [sa.DUPLICATE_EDITION]
    record = next(e for e in edition.competitors if e.get("role") == "kept_edition")
    assert record["link"] == "edition_of:pgp:48" and record["sha256"] == "ed_r"
    assert record["containment"] >= sa.EDITION_CONTAINMENT
    assert [len(u) for u in sa.decision_units([pgp, lit])] == [1, 1]


def test_complementary_candidates_of_one_source_stay_joint_and_editions_apart() -> None:
    """Two candidates of one source with different texts are one decision unit; an excerpt of the other's text is
    an edition and is decided apart."""
    recto, _ = two_sides(53, letters=300)
    excerpt = " ".join(recto.split()[: len(recto.split()) // 2])
    images = [image("lx_a"), image("lx_b")]
    full = cand("lit:53a", [target("", recto)], images, group="group:53", source="literature")
    part = cand("lit:53b", [target("", excerpt)], images, group="group:53", source="literature")
    other = cand("lit:53c", [target("", two_sides(54)[0])], images, group="group:53", source="literature")
    assert [len(u) for u in sa.decision_units([full, part])] == [1, 1]
    assert [len(u) for u in sa.decision_units([full, other])] == [2]
    assert [len(u) for u in sa.decision_units([full, other, part])] == [2, 1]


def test_two_texts_accepted_on_one_photograph_are_not_editions() -> None:
    """Different texts of two units accepted on one photograph stay a conflict for review, not an edition drop."""
    first, second = two_sides(49)
    shared = [image("both")]
    a = cand("pgp:49", [target("recto", first)], shared, group="group:49")
    b = cand("lit:49", [target("", second)], shared, group="group:49", source="literature")
    sides = {s.key: s for s in (sa.make_side(a, 0, a.targets[0]), sa.make_side(b, 0, b.targets[0]))}
    decisions = {key: sa._decision("accepted", [], image=shared[0], assigned=shared[0]) for key in sides}
    pairs = {key: {"both": crafted(0.5, True, True)} for key in sides}
    assert sa.resolve_editions(decisions, sides, {key: i for i, key in enumerate(sides)}, pairs) == []
    assert sorted(sa.image_conflicts(decisions, sides)) == sorted(sides)


# --- targets spanning two photographs, the gate -----------------------------------------------------------------------

@pytest.mark.parametrize("side_key, reasons", [("r", [sa.SPANS_TWO_IMAGES]),
                                               ("", [sa.ON_ANOTHER_IMAGE, sa.SPANS_TWO_IMAGES])])
def test_a_target_spanning_two_photographs_goes_to_review(side_key: str, reasons: List[str]) -> None:
    """A target holding a recto's text (68 %) and its verso's (32 %) clears p99 on the recto photograph and spills
    onto the verso one at under half its recall. Labelled, the ratio rule excuses the spill but the second half
    recalls better on the verso photograph (``target_spans_two_images``, half recalls recorded); unlabelled, the spill
    onto a photograph no side holds is not excused either.

    :param side_key: The target's side key.
    :param reasons: Expected blocking reasons.
    """
    front, back = two_sides(51, letters=420)
    front_words, back_words = front.split(), back.split()
    text = (" ".join(front_words[: int(len(front_words) * 0.68)]) + "\n"
            + " ".join(back_words[: int(len(back_words) * 0.32)]))
    images = [image("span_r", origin="fjp_local"), image("span_v", origin="fjp_local")]
    c = cand("store:51", [target(side_key, text)], images, source="fjp")
    reads = null_reads(131) + [read("span_r", front), read("span_v", back)]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "store:51")[side_key]
    assert out.outcome == "needs_review"
    assert [r for r in out.reasons if r not in sa.FLAG_ONLY] == reasons
    halves = next(e for e in out.competitors if e.get("role") == "halves")
    assert halves["spans"] == ["span_v"] and halves["spill"]["span_v"][1] > halves["chosen"][1]


def test_the_gate_also_needs_the_3gram_recall_and_matched_grams() -> None:
    """A gate source's read must also clear the 3-gram p95 and match at least 5 primary n-grams."""
    c = cand("apd_v3:stem_2", [target("recto", words(random.Random(18), HEBREW, 120))],
             [image("gate_img", origin="apd_local")], source="apd_v3")
    side = sa.make_side(c, 0, c.targets[0])
    given = c.images[0]
    assert sa.decide_gate(side, given, crafted(0.5, True, True))["outcome"] == "accepted"
    weak = sa.decide_gate(side, given, crafted(0.5, True, True, n3=False, matched=3))
    assert weak["outcome"] == "needs_review" and weak["reasons"] == [sa.N3_BELOW_P95, sa.under_matched_reason(4)]


def test_third_matches_count_the_target_thirds_found_in_the_read() -> None:
    """The coverage signals carry the letters of the target's thirds and their n-grams found in the read."""
    recto, _ = two_sides(52, letters=330)
    c = cand("pgp:52", [target("recto", recto)], [image("thirds")])
    side = sa.make_side(c, 0, c.targets[0])
    thirds = sa.third_matches(side, sa.make_view(read("thirds", recto[: len(recto) * 2 // 3])))
    assert sum(thirds["letters"]) == len(side.skeleton)
    assert thirds["matched"][0] > 0 and thirds["matched"][2] == 0


# --- version by placement, Arabic 3-grams, coverage read --------------------------------------------------------------

def test_the_version_placed_with_the_higher_recall_is_kept() -> None:
    """Of two versions within 5 % of each other, the one accepted with the higher recall on its photograph is kept even
    with more apparatus marks; the other is dropped with ``duplicate_version_lower_recall`` and both values recorded."""
    rng = random.Random(94)
    recto, _ = two_sides(56, letters=400)
    tokens = recto.split()
    marked = " ".join(w + "?" if i % 6 == 0 else w for i, w in enumerate(tokens))
    clean = " ".join(tokens[:-1])
    sharp = cand("pgp:56", [target("recto", marked)], [image("sharp")], group="group:56")
    blurry = cand("fjp:56", [target("", clean)], [image("blurry", origin="fjp_local")], group="group:56", source="fjp")
    reads = null_reads(134) + [read("sharp", noisy(rng, recto, HEBREW, 0.01)),
                               read("blurry", noisy(rng, recto, HEBREW, 0.12))]
    out = sa.assign_sides([sharp, blurry], reads, n_boot=N_BOOT, seed=3)
    kept, lost = by_side(out, "pgp:56")["recto"], by_side(out, "fjp:56")[""]
    assert kept.outcome == "accepted" and lost.outcome == "dropped"
    assert lost.reasons == [sa.DUPLICATE_LOWER_RECALL]
    record = next(e for e in lost.competitors if e.get("role") == "kept_version")
    assert record["decided_by"] == "recall" and record["recall"][0] < record["recall"][1]
    assert record["marks"][0] < record["marks"][1]
    compared = next(e for e in kept.competitors if e.get("role") == "versions_compared")
    assert [v["kept"] for v in compared["versions"]] == [True, False]


def test_the_3gram_check_is_a_flag_on_arabic_sides() -> None:
    """On an Arabic-script side a 3-gram recall under its p95 is recorded but does not block the side, in a decision
    unit and at the gate; a Hebrew-script side still goes to review."""
    arabic = cand("pgp:57", [target("recto", words(random.Random(19), ARABIC, 120))], [image("ar_1"), image("ar_2")])
    side = sa.make_side(arabic, 0, arabic.targets[0])
    pairs = {side.key: {"ar_1": crafted(0.5, True, True, n3=False), "ar_2": crafted(0.01, False, False)}}
    decision = sa.decide_candidate([side], sa.distinct_images(arabic.images), pairs, {})[side.key]
    assert decision["outcome"] == "accepted" and decision["reasons"] == [sa.N3_BELOW_P95]
    gate = cand("apd_v3:stem_3", [target("recto", words(random.Random(20), ARABIC, 120))],
                [image("apd_g", origin="apd_local")], source="apd_v3")
    gate_side = sa.make_side(gate, 0, gate.targets[0])
    rasm_pair = {**crafted(0.5, True, True, n3=False), "null_n5": {"p95": 0.1, "p99": 0.2, "n": 60}}
    accepted = sa.decide_gate(gate_side, gate.images[0], rasm_pair)
    assert accepted["outcome"] == "accepted" and accepted["reasons"] == [sa.N3_BELOW_P95]
    assert sa.blocks(sa.N3_BELOW_P95, sa.make_side(cand("pgp:58", [target("recto", "אבגדה" * 30)], []), 0,
                                                    target("recto", "אבגדה" * 30)))


def test_coverage_is_computed_on_the_read_that_covers_most_of_the_target() -> None:
    """A Kraken read of 30 % of the target places the side (the page reader's null is too small to win), and the
    coverage signals use the page-reader read that covers the whole target, naming both readers."""
    recto, _ = two_sides(55, letters=300)
    lines = recto.splitlines()
    partial = "\n".join(lines[: max(1, len(lines) * 3 // 10)])
    c = cand("pgp:55", [target("recto", recto)], [image("cov")])
    reads = null_reads(132) + null_reads(133, reader=VLM, n=5) + [read("cov", partial), read("cov", recto, VLM)]
    out = by_side(sa.assign_sides([c], reads, n_boot=N_BOOT, seed=3), "pgp:55")["recto"]
    assert out.outcome == "accepted" and out.reader == KRAKEN
    coverage = out.competitors[0]["coverage"]
    assert (coverage["coverage_reader"], coverage["placement_reader"]) == (VLM, KRAKEN)
    assert coverage["usable_readers"] == ["kraken", "vlm"] and coverage["letters_ratio"] > 0.9


def test_versions_of_one_group_are_one_copy_of_a_shared_text() -> None:
    """Two versions of a text in one document group that another group also holds act as one copy: their group owns
    the text, the other group's copy is dropped, and the version placed with the higher recall is kept."""
    rng = random.Random(95)
    text, _ = two_sides(57, letters=300)
    a = cand("pgp:57", [target("recto", text)], [image("own_a")], group="group:57")
    b = cand("fjp:57", [target("", text + " ?")], [image("own_b", origin="fjp_local")], group="group:57", source="fjp")
    c = cand("lit:57", [target("", text)], [image("far")], group="group:58", source="literature")
    reads = null_reads(135) + [read("own_a", noisy(rng, text, HEBREW, 0.01)),
                               read("own_b", noisy(rng, text, HEBREW, 0.1)), read("far", words(rng, HEBREW, 200))]
    out = sa.assign_sides([a, b, c], reads, n_boot=N_BOOT, seed=3)
    lost = by_side(out, "lit:57")[""]
    assert lost.outcome == "dropped" and lost.reasons == [sa.NOT_OWNER]
    assert by_side(out, "pgp:57")["recto"].outcome == "accepted"
    assert by_side(out, "fjp:57")[""].reasons == [sa.DUPLICATE_LOWER_RECALL]
