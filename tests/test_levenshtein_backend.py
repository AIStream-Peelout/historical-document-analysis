"""Pin the python-Levenshtein backend assumptions the metrics rely on.

``metrics.py`` routes both CER (strings) and WER (token lists) through the
C-accelerated ``Levenshtein.distance``. That only works because modern
python-Levenshtein (>= 0.21) is rapidfuzz-backed and accepts arbitrary
sequences — the pre-0.21 C extension was strings-only. A code-review bot once
suggested guarding the fast path to strings, which would have silently pushed
all WER onto the ~1000x slower pure-Python fallback; this test makes the
actual contract explicit instead.
"""


def test_c_levenshtein_accepts_token_lists() -> None:
    """python-Levenshtein must compute distance over token lists (WER path)."""
    from Levenshtein import distance

    assert distance(["the", "cat", "sat"], ["the", "dog", "sat"]) == 1
    assert distance([], ["a"]) == 1


def test_c_levenshtein_strings_unchanged() -> None:
    """String distance (CER path) behaves as expected."""
    from Levenshtein import distance

    assert distance("kitten", "sitting") == 3
    assert distance("", "") == 0
