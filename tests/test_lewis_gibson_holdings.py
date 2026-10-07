"""Lewis-Gibson (``L-G``) shelfmarks resolve to Cambridge University Library.

The collection was bought jointly with the Bodleian in 2013 but is housed,
catalogued and digitised at CUL and treated as a CUL sub-collection in the
literature (decision 2026-09-15). The old compound string
``"Cambridge University Library / Bodleian Library Oxford"`` produced a
separate holdings node in the June 2026 graph; it must not come back.
"""

import pytest

from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer


@pytest.mark.parametrize("mark", ["L-G Misc. 119", "L-G Talm. II 97", "MS-L-G Ar. 1.2"])
def test_lewis_gibson_is_a_cul_subcollection(mark):
    info = ShelfmarkNormalizer.get_institution_info(mark)
    assert info["institution"] == "Cambridge University Library"
    assert info["collection"] == "Lewis-Gibson"
    assert "/" not in info["institution"]
