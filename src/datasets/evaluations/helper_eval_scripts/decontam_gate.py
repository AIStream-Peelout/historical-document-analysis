"""Decontamination gate for building KTIV-sourced evaluation slices.

Any manuscript added to a NEW eval slice (e.g. a literary / Talmud-MS
benchmark) must be provably disjoint from everything the models have already
trained on, or the eval is worthless.  KTIV documents are keyed by NLI
``sys_num`` (the 18-digit ``990...`` id), so the primary check is an EXACT id
match — far more reliable than fuzzy shelfmark matching.

Three independent layers, each sufficient to reject:

1. **sys_num** in the v1.9a KTIV training set (906 manuscripts) — exact.
   KTIV is the only source that entered training keyed by sys_num
   (via ``genizah_ktiv_v1``); this is the decisive check for KTIV re-scrapes.
2. **loose shelfmark key** (the institution-qualified, digit-boundary safe
   :func:`src.datasets.shelfmarks.loose_key`, the format of the inventories)
   in ``genizah_clean_v2`` (the legacy-store/PGP documentary training lineage used
   since v1.7) or in the frozen 131-fragment benchmark — catches a doc that
   reached training/eval under a different source's shelfmark. Until
   2026-11 the gate keyed a raw mark with its institution stripped, so this
   check never fired on a raw mark against the institution-qualified
   ``clean_v2_keys`` (audit finding F4).
3. **content shingle** overlap with training GT (optional, applied at build
   time when the candidate's transcription is in hand): >= 2 shared 25-letter
   letter-shingles ⇒ reject (same rule ``build_ktiv_dataset`` used).

The inventory JSONs are produced alongside this module under
``raw_data/cairo_genizah/decontam/`` and are the single source of truth;
regenerate them whenever a new training corpus is added.

Note: ``talmud_finetune_v2`` is Vilna PRINT keyed by tractate_daf, a disjoint
namespace from KTIV manuscripts, so it needs no id gate here — but a Talmud-MS
eval must still be content-checked against it (layer 3) since both carry the
same canonical Talmud text; a manuscript of the same daf is a DIFFERENT image
and is fair eval, but verify the GT is the manuscript's reading, not Vilna's.
"""

import json
import re
from pathlib import Path
from typing import Tuple

from src.datasets import shelfmarks

_DECONTAM = (Path(__file__).resolve().parents[4]
             / "src/datasets/raw_data/cairo_genizah/decontam")
_HEB = re.compile(r"[א-ת]")
_SHINGLE_N = 25
_SHINGLE_MIN_HITS = 2


def _load(name: str) -> set:
    """Load a decontam inventory JSON as a set.

    :param name: File stem under the decontam dir.
    :type name: str
    :return: Set of ids/keys (empty if the file is absent).
    :rtype: set
    """
    p = _DECONTAM / f"{name}.json"
    return set(json.load(open(p))) if p.exists() else set()


class DecontamGate:
    """Reject candidates that overlap any prior training or eval corpus.

    :cvar loose_key: The gate's shelf-mark key, :func:`src.datasets.shelfmarks.loose_key` itself
        (institution-qualified: ``T-S 12.301``, its KTIV form and ``Cambridge_CUL_T_S_12_301`` all give
        ``cambridgecults12|301``; ``""`` when the mark cannot be keyed, e.g. its institution cannot be
        determined).
    """

    loose_key = staticmethod(shelfmarks.loose_key)

    def __init__(self) -> None:
        """Load the three id/key inventories."""
        self.ktiv_train_sys = _load("ktiv_v19a_train_sysnums")
        self.clean_keys = _load("clean_v2_keys")
        self.bench_keys = _load("benchmark_keys")
        if not self.ktiv_train_sys:
            raise FileNotFoundError(
                "decontam inventory missing — regenerate it (see module docstring)")

    def check(self, sys_num: str, shelfmark: str = "") -> Tuple[bool, str]:
        """Return (is_clean, reason) for a candidate KTIV manuscript.

        :param sys_num: NLI system number (18-digit ``990...``).
        :type sys_num: str
        :param shelfmark: Shelfmark for the cross-source fuzzy check.
        :type shelfmark: str
        :return: (True, "clean") when disjoint from all prior corpora, else
            (False, <which layer matched>).
        :rtype: Tuple[bool, str]
        """
        if sys_num and sys_num in self.ktiv_train_sys:
            return False, "sysnum_in_ktiv_v19a_train"
        key = self.loose_key(shelfmark)
        if key and key in self.clean_keys:
            return False, f"shelfmark_in_clean_v2 ({key})"
        if key and key in self.bench_keys:
            return False, f"shelfmark_in_benchmark ({key})"
        return True, "clean"

    def content_contaminated(self, text: str, train_shingles: set) -> bool:
        """Layer-3 content check: shared 25-letter shingles with training GT.

        :param text: Candidate transcription.
        :type text: str
        :param train_shingles: Precomputed set of training-GT letter shingles.
        :type train_shingles: set
        :return: True when >= 2 shingles are shared (contaminated).
        :rtype: bool
        """
        letters = "".join(_HEB.findall(text))
        hits = sum(letters[i:i + _SHINGLE_N] in train_shingles
                   for i in range(len(letters) - _SHINGLE_N + 1))
        return hits >= _SHINGLE_MIN_HITS


def main() -> None:
    """CLI smoke test: check a sys_num[/shelfmark] against the gate."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sys_num")
    ap.add_argument("--shelfmark", default="")
    args = ap.parse_args()
    gate = DecontamGate()
    clean, reason = gate.check(args.sys_num, args.shelfmark)
    print(f"{args.sys_num}: {'CLEAN' if clean else 'CONTAMINATED'} — {reason}")


if __name__ == "__main__":
    main()
