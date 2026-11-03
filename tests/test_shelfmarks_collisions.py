"""T2 collision test: two different items never share a ``loose_key``.

* Digit-boundary pairs (the ``[^a-z0-9]`` squash family maps each pair to one key): ENA NS 2.26 / 22.6 (P060 /
  P061), T-S AS 181.163 / 18.1163 (P007 / P045), T-S Misc. 31.15 / 3.115 (P046 / P047).
* Cross-institution pairs (the institution-free ``core_key`` maps each pair to one key: 175 key groups / 402 ES ids
  on 2026-11-02): ``Berlin_JCB_A2`` / ``Manchester_JRL_A_2``, ``Budapest_MTA_12`` / ``Birmingham_Mingana_12``.
* Piece letters stay distinct (``F 1908.44`` / ``F 1908.44A``; a ``base_key``-style key merges them).
* The text27b exposure index (fragment level) keys on the facade, so a read of one item of a digit-boundary pair is
  not marked as exposed by training on the other (the squash ``exposure.id_key`` merged all three pairs).
"""

from __future__ import annotations

from typing import Optional, Tuple

import pytest

from src.datasets import shelfmarks
from src.finetuning.text27b import exposure

#: (mark, institution) pairs naming two different items.
Pair = Tuple[Tuple[str, Optional[str]], Tuple[str, Optional[str]]]

DIGIT_BOUNDARY: Tuple[Pair, ...] = (
    (("ENA NS 2.26", "Jewish Theological Seminary"), ("ENA NS 22.6", "Jewish Theological Seminary")),
    (("T-S AS 181.163", "Cambridge University Library"), ("T-S AS 18.1163", "Cambridge University Library")),
    (("T-S Misc. 31.15", "Cambridge University Library"), ("T-S Misc. 3.115", "Cambridge University Library")),
    (("ENA NS 2.26", None), ("ENA NS 22.6", None)),
    (("T-S AS 181.163", None), ("T-S AS 18.1163", None)),
    (("T-S Misc. 31.15", None), ("T-S Misc. 3.115", None)),
    (("New_York_JTS_ENA_NS_2_26", None), ("New_York_JTS_ENA_NS_22_6", None)),
)

CROSS_INSTITUTION: Tuple[Pair, ...] = (
    (("Berlin_JCB_A2", None), ("Manchester_JRL_A_2", None)),
    (("A2", "Jewish Community of Berlin"), ("JRL A 2", "John Rylands Library")),
    (("Budapest_MTA_12", None), ("Birmingham_Mingana_12", None)),
    (("DK 12", "Budapest Kaufmann"), ("Mingana 12", "Birmingham")),
    (("DK 12", None), ("Birmingham: Mingana 12", None)),
)

PIECE_LETTERS: Tuple[Pair, ...] = (
    (("F 1908.44", "Freer Gallery of Art"), ("F 1908.44A", "Freer Gallery of Art")),
    (("Washington_Freer_F_1908_44", None), ("Washington_Freer_F_1908_44A", None)),
)


def _pair_id(pair: Pair) -> str:
    """A readable test id for a pair.

    :param pair: Two (mark, institution) items.
    :type pair: Pair
    :return: ``"<mark> vs <mark>"``.
    :rtype: str
    """
    return f"{pair[0][0]} vs {pair[1][0]}"


@pytest.mark.parametrize("pair", DIGIT_BOUNDARY + CROSS_INSTITUTION + PIECE_LETTERS,
                         ids=[_pair_id(p) for p in DIGIT_BOUNDARY + CROSS_INSTITUTION + PIECE_LETTERS])
def test_distinct_items_have_distinct_loose_keys(pair: Pair) -> None:
    """Both items get a non-empty, institution-qualified loose key, and the two keys differ.

    :param pair: Two (mark, institution) items naming different fragments.
    :type pair: Pair
    :return: None.
    :rtype: None
    """
    (mark_a, inst_a), (mark_b, inst_b) = pair
    key_a, key_b = shelfmarks.loose_key(mark_a, inst_a), shelfmarks.loose_key(mark_b, inst_b)
    assert key_a and key_b
    assert key_a != key_b
    assert shelfmarks.canonical_id(mark_a, inst_a) != shelfmarks.canonical_id(mark_b, inst_b)


#: The cross-institution pairs written as ES ids.
CROSS_INSTITUTION_IDS: Tuple[Pair, ...] = (CROSS_INSTITUTION[0], CROSS_INSTITUTION[2])


@pytest.mark.parametrize("pair", CROSS_INSTITUTION_IDS, ids=[_pair_id(p) for p in CROSS_INSTITUTION_IDS])
def test_core_key_hazard_is_real(pair: Pair) -> None:
    """The institution-free ``core_key`` does collide on the cross-institution ES-id pairs (documented hazard), which
    is why only the institution-qualified ``loose_key`` is a tracking / join key.

    :param pair: Two ES ids of different institutions with the same core.
    :type pair: Pair
    :return: None.
    :rtype: None
    """
    (id_a, _), (id_b, _) = pair
    assert shelfmarks.core_key(id_a) == shelfmarks.core_key(id_b)
    assert shelfmarks.loose_key(id_a) != shelfmarks.loose_key(id_b)


def test_digit_boundary_keys_are_exact() -> None:
    """The ``|`` digit-boundary marker keeps the dotted numbers apart.

    :return: None.
    :rtype: None
    """
    assert shelfmarks.loose_key("ENA NS 2.26") == "newyorkjtsenans2|26"
    assert shelfmarks.loose_key("ENA NS 22.6") == "newyorkjtsenans22|6"
    assert shelfmarks.loose_key("T-S AS 181.163") == "cambridgecultsas181|163"
    assert shelfmarks.loose_key("T-S AS 18.1163") == "cambridgecultsas18|1163"


#: ES ids without a registry token next to the registry-token id they may duplicate (Tier-1 candidates): each keeps
#: its own key, so ``loose_key`` never folds two ES ids the ES keeps apart.
SLUG_VS_REGISTRY: Tuple[Tuple[str, str], ...] = (
    ("Unknown_DK_123", "Budapest_MTA_DK_123"),
    ("Unknown_DK_239_1", "Budapest_MTA_DK_239_1"),
    ("Unknown_Reinach_II", "Reinach_II"),
    ("BNU_Strasbourg_4038", "Strasbourg_4038"),
)


@pytest.mark.parametrize("pair", SLUG_VS_REGISTRY, ids=[f"{a} vs {b}" for a, b in SLUG_VS_REGISTRY])
def test_slug_and_unknown_ids_keep_keys_of_their_own(pair: Tuple[str, str]) -> None:
    """A slug-token or ``Unknown_…`` ES id and a registry-token ES id get distinct, non-empty loose keys.

    :param pair: (no-registry-token id, registry-token id).
    :type pair: Tuple[str, str]
    :return: None.
    :rtype: None
    """
    key_a, key_b = shelfmarks.loose_key(pair[0]), shelfmarks.loose_key(pair[1])
    assert key_a and key_b and key_a != key_b


#: The three digit-boundary pairs as ES ids (what the exposure index sees: dataset stems, manifest and read ids).
DIGIT_BOUNDARY_IDS: Tuple[Tuple[str, str], ...] = (
    ("New_York_JTS_ENA_NS_2_26", "New_York_JTS_ENA_NS_22_6"),
    ("Cambridge_CUL_T_S_AS_181_163", "Cambridge_CUL_T_S_AS_18_1163"),
    ("Cambridge_CUL_T_S_Misc_31_15", "Cambridge_CUL_T_S_Misc_3_115"),
)


@pytest.mark.parametrize("pair", DIGIT_BOUNDARY_IDS, ids=[f"{a} vs {b}" for a, b in DIGIT_BOUNDARY_IDS])
def test_exposure_index_keeps_digit_boundary_pairs_apart(pair: Tuple[str, str]) -> None:
    """Training on one item of a pair gives fragment exposure to a read of that item only, never of the other.

    :param pair: Two ES ids that the old alphanumeric squash mapped to one key.
    :type pair: Tuple[str, str]
    :return: None.
    :rtype: None
    """
    trained, other = pair
    assert exposure.fragment_key(trained) != exposure.fragment_key(other)
    assert exposure.fragment_key(trained) == shelfmarks.loose_key(trained)
    index = exposure.TrainingIndex()
    index.add("fragment", exposure.fragment_key(trained), "clean_v2")
    mixtures = ("clean_v2",)

    def level(cid: str) -> str:
        """Exposure level of a page-less read of ``cid``.

        :param cid: ES id the read is of.
        :type cid: str
        :return: The read's level.
        :rtype: str
        """
        read = exposure.Read(pgpid="1", model="v21b", pages=[], cids=[cid])
        return exposure.classify_read(read, index, {}, mixtures)[0]["read_level"]

    assert level(trained) == "fragment"
    assert level(other) == "none"


@pytest.mark.parametrize("pair", DIGIT_BOUNDARY[:3], ids=[_pair_id(p) for p in DIGIT_BOUNDARY[:3]])
def test_exposure_key_of_a_raw_mark_matches_its_id(pair: Pair) -> None:
    """A raw mark queried with its institution reaches the key of its own ES id, not its pair's.

    :param pair: Two (mark, institution) items of a digit-boundary pair.
    :type pair: Pair
    :return: None.
    :rtype: None
    """
    (mark_a, inst_a), (mark_b, inst_b) = pair
    id_a, id_b = shelfmarks.canonical_id(mark_a, inst_a), shelfmarks.canonical_id(mark_b, inst_b)
    assert id_a and id_b
    assert exposure.fragment_key(mark_a, inst_a) == exposure.fragment_key(id_a)
    assert exposure.fragment_key(mark_a, inst_a) != exposure.fragment_key(id_b)
