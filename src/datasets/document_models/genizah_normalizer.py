#!/usr/bin/env python3
"""
Shelfmark Normalization for Cairo Genizah Documents

This module provides:
1. Canonical ID generation (underscore-separated UUID)
2. Variant generation (all known formats for search)
3. Bidirectional matching (any format → canonical ID)
4. Institution/collection lookup from shelfmark prefix

Example:
    Input: "T-S AS 18.170"
    Output:
        canonical_id: "T_S_AS_18_170"
        variants: ["T-S AS 18.170", "TS AS 18.170", "CUL: T-S AS 18.170", ...]
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set

from src.datasets.document_models.entity_normalizer import EntityNormalizer


class ShelfmarkNormalizer(EntityNormalizer):
    """Normalize shelfmarks to canonical IDs and generate search variants."""

    # ── EntityNormalizer ABC implementation ───────────────────────────────────

    @classmethod
    def normalize(cls, raw: str) -> str:
        """Return the canonical shelfmark ID for *raw*.

        :param raw: Shelfmark in any known format.
        :returns: Canonical underscore-separated ID (e.g. ``T_S_AS_18_170``).
        """
        return cls.to_canonical_id(raw)

    @classmethod
    def get_variants(cls, canonical: str) -> List[str]:
        """Return known surface-form variants for a canonical shelfmark ID.

        :param canonical: Canonical ID (e.g. ``T_S_AS_18_170``).
        :returns: List of variant display strings.
        """
        return cls.generate_variants(canonical)

    @classmethod
    def get_metadata(cls, raw: str) -> Dict[str, Any]:
        """Return institution / collection info for *raw*.

        :param raw: Shelfmark in any known format.
        :returns: Dict with ``institution``, ``collection``, ``subcollection``.
        """
        return cls.get_institution_info(raw)

    # Collection prefix mappings (variant → canonical)
    COLLECTION_PREFIXES = {
        # Taylor-Schechter (Cambridge)
        'T-S': 'T_S',
        'TS': 'T_S',
        'T S': 'T_S',
        'MS-TS-': 'T_S_',

        # T-S subseries
        'T-S AS': 'T_S_AS',
        'TS AS': 'T_S_AS',
        'MS-TS-AS-': 'T_S_AS_',

        'T-S NS': 'T_S_NS',
        'TS NS': 'T_S_NS',
        'MS-TS-NS-': 'T_S_NS_',

        'T-S K': 'T_S_K',
        'MS-TS-K-': 'T_S_K_',

        'T-S Ar': 'T_S_Ar',
        'MS-TS-AR-': 'T_S_Ar_',

        # ENA (JTS)
        'ENA': 'ENA',
        'ENA NS': 'ENA_NS',

        # Lewis-Gibson
        'L-G': 'L_G',
        'LG': 'L_G',
        'MS-L-G-': 'L_G_',

        # Mosseri
        'Mosseri': 'Mosseri',
        'Moss': 'Mosseri',
        'Moss.': 'Mosseri',
        'MS-MOSSERI-': 'Mosseri_',

        # CUL
        'CUL Or': 'CUL_Or',
        'MS-OR-': 'CUL_Or_',
        'CUL Add': 'CUL_Add',
        'MS-ADD-': 'CUL_Add_',

        # Manchester (Gaster)
        'Gaster A': 'Gaster_A',
        'Gaster B': 'Gaster_B',
        'Gaster P': 'Gaster_P',
        'Gaster L': 'Gaster_L',
        'Gaster C': 'Gaster_C',

        # Bodleian
        'Bodl': 'Bodl',
        'Bodl.': 'Bodl',
        'MS. Heb.': 'MS_Heb',

        # AIU Paris
        'AIU': 'AIU',

        # British Library
        'BL Or': 'BL_Or',
        'BL Add': 'BL_Add',

        # Penn
        'Halper': 'Halper',
        'CAJS': 'CAJS',
    }

    # Institution prefixes that can appear in variants
    INSTITUTION_PREFIXES = [
        'Cambridge',
        'Cambridge CUL',
        'CUL',
        'Cambridge University Library',
        'JTS',
        'Jewish Theological Seminary',
        'Penn',
        'University of Pennsylvania',
        'Manchester',
        'Rylands',
        'Oxford',
        'Bodleian',
        'Paris',
        'AIU',
    ]

    # Residual noise tokens dropped at the token level. Kept deliberately small
    # and multi-character: single letters (f, l, p) are NOT listed here because
    # they collide with collection letters (Lewis-Gibson "L-G", Gaster "P"/"L").
    # Folio designators are stripped earlier by regex (see FOLIO_RE).
    NOISE_TOKENS: Set[str] = {'box', 'folio', 'shelfmark', 'unknown', 'alt'}

    # Folio / leaf / page designators preceding a fragment number. The number is
    # preserved. Single-letter designators require a trailing dot so collection
    # letters survive ("BM Or 10110, f. 23" -> drop "f.", keep "23"; "L-G" kept).
    FOLIO_RE = re.compile(
        r"(?:\b(?:fols?|ff|pp|pag|page|lines?|lin)\b\.?\s)"
        r"|(?:(?<=[\s,])[flp]\.\s*)",
        flags=re.IGNORECASE,
    )

    # Institution / collection renames applied at the token level before
    # canonicalisation. British Museum -> British Library is the historical
    # rename behind every "BM Or ..." historic variant.
    TOKEN_RENAMES: Dict[str, str] = {
        'BM': 'BL',
        'BRITISH': 'BL',
        'MUSEUM': '',
    }

    @staticmethod
    def _strip_institution(shelf_mark: str) -> str:
        """Remove the leading institution / location prefix from *shelf_mark*.

        Handles the three source conventions:

        * **KTIV** ``"<Institution>, <City>, <Country> Ms. <core>"`` — everything
          up to and including the last `` Ms. `` delimiter is dropped.
        * **FJP** ``"<Institution>: <core>"`` — for a single shelfmark there is
          exactly one ``:``; the text after the last ``:`` is the core.
        * **Known leading institution words** from :attr:`INSTITUTION_PREFIXES`
          (e.g. ``"Cambridge CUL "``) are peeled as a fallback.

        :param shelf_mark: Raw shelfmark string (already unicode-normalised).
        :returns: The shelfmark core with institution context removed.
        """
        s = shelf_mark

        # KTIV: split on the " Ms. " delimiter (case-insensitive). Manchester's
        # own "MS B 3567" style has no preceding institution+comma, so only fire
        # when a comma (location list) precedes the delimiter.
        ms_match = re.search(r"^.*,.*?\sMs\.\s+", s, flags=re.IGNORECASE)
        if ms_match:
            s = s[ms_match.end():]

        # FJP: institution prefix terminated by a colon. Take the text after the
        # last colon (single-shelfmark records have exactly one).
        if ":" in s:
            s = s.rsplit(":", 1)[1]

        s = s.strip()

        # Fallback: peel a known leading institution word (no colon present).
        for inst_prefix in ShelfmarkNormalizer.INSTITUTION_PREFIXES:
            pattern = rf'^{re.escape(inst_prefix)}\s*[:\s,]\s*'
            new = re.sub(pattern, '', s, flags=re.IGNORECASE)
            if new != s:
                s = new
                break

        return s.strip()

    @staticmethod
    def _collapse_ts_classmark(tokens: List[str]) -> List[str]:
        """Collapse spaced Taylor-Schechter class-marks to PGP's closed form.

        PGP writes single-letter class-marks closed up (``T-S 10J5.6``,
        ``T-S 8K20.2``, ``T-S K10.10``) while KTIV and many historic variants
        space them out (``T-S 10 J 5.6``). This merges a single class-mark letter
        with an immediately preceding and/or following digit run so both forms
        produce the same id. Multi-letter subseries (``AS``, ``NS``, ``Ar``,
        ``Misc``) are left untouched. Scoped to ``T-S`` only, since other
        collections (e.g. St Petersburg ``EVR II A 11``) use different rules.

        :param tokens: Canonical-id tokens (already cleaned, pre-join).
        :returns: Tokens with T-S class-marks collapsed.
        """
        if len(tokens) < 3 or tokens[0] != "T" or tokens[1] != "S":
            return tokens

        out: List[str] = ["T", "S"]
        rest = tokens[2:]
        i = 0
        while i < len(rest):
            tok = rest[i]
            if len(tok) == 1 and tok.isalpha():
                merged = tok
                if out and out[-1].isdigit():
                    merged = out.pop() + merged
                if i + 1 < len(rest) and rest[i + 1].isdigit():
                    merged += rest[i + 1]
                    i += 1
                out.append(merged)
            else:
                out.append(tok)
            i += 1
        return out

    @staticmethod
    def to_canonical_id(shelfmark: str) -> str:
        """
        Convert any shelfmark format to canonical ID (UUID format).

        Canonical ID uses underscores only, no spaces or punctuation.

        Args:
            shelfmark: Input shelfmark in any format

        Returns:
            Canonical ID (e.g., "T_S_AS_18_170")

        Examples:
            ShelfmarkNormalizer.to_canonical_id('T-S AS 18.170')
            'T_S_AS_18_170'
            ShelfmarkNormalizer.to_canonical_id('Cambridge CUL: T-S AS 18.170')
            'T_S_AS_18_170'
             ShelfmarkNormalizer.to_canonical_id('MS-TS-AS-00018-00170')
            'T_S_AS_18_170'
        """
        if not shelfmark:
            return ""

        # Unicode-normalise and unify the various dash characters to ASCII '-'.
        canonical = unicodedata.normalize("NFKC", shelfmark).strip()
        canonical = re.sub(r"[‐-―−]", "-", canonical)

        # Drop parentheticals ("(shelfmark unknown)", "(Alt: 1)") wholesale.
        canonical = re.sub(r"\([^)]*\)", " ", canonical)

        # John Rylands Library, Manchester. PGP uses "JRL <series>" while FJP and
        # KTIV write "Manchester[:] <series>"; both name the same fragments
        # (e.g. JRL C 121 == Manchester C 121, across all A/B/C/... and Gaster /
        # Genizah series). Unify everything to the canonical "JRL <series>" form.
        if re.search(r"\b(?:JRL|Rylands|Manchester)\b", canonical, flags=re.IGNORECASE):
            core = ShelfmarkNormalizer._strip_institution(canonical)
            core = re.sub(
                r"^\s*(?:JRL|John\s+Rylands(?:\s+Library)?|Rylands|Manchester)\s*:?\s*",
                "", core, flags=re.IGNORECASE,
            )
            canonical = "JRL " + core.strip()

        # Strip institution / location context (KTIV "Ms.", FJP "X:", known
        # leading institution words).
        canonical = ShelfmarkNormalizer._strip_institution(canonical)

        # Strip folio / leaf designators ("f.", "fol.", "Box"); fragment numbers
        # that follow them are preserved.
        canonical = ShelfmarkNormalizer.FOLIO_RE.sub(" ", canonical)
        canonical = re.sub(r"\bBox\b", " ", canonical, flags=re.IGNORECASE)

        # Handle Cambridge TEI encodings
        canonical = canonical.replace("MS-TS-AS-", "T_S_AS_")
        canonical = canonical.replace("MS-TS-NS-", "T_S_NS_")
        canonical = canonical.replace("MS-TS-K-", "T_S_K_")
        canonical = canonical.replace("MS-TS-AR-", "T_S_Ar_")
        canonical = canonical.replace("MS-TS-", "T_S_")
        canonical = canonical.replace("MS-MOSSERI-", "Mosseri_")
        canonical = canonical.replace("MS-L-G-", "L_G_")
        canonical = canonical.replace("MS-OR-", "CUL_Or_")
        canonical = canonical.replace("MS-ADD-", "CUL_Add_")

        # Normalize collection prefix (T-S/TS/T S -> T_S, etc.)
        for variant, standard in sorted(ShelfmarkNormalizer.COLLECTION_PREFIXES.items(),
                                        key=lambda x: len(x[0]), reverse=True):
            if canonical.upper().startswith(variant.upper()):
                rest = canonical[len(variant):]
                canonical = standard + rest
                break

        # Tokenise on every separator (existing underscores from the prefix
        # normalisation are preserved as token boundaries).
        tokens = [t for t in re.split(r"[\s._\-/,:()\[\]]+", canonical) if t]

        normalized_parts: List[str] = []
        for tok in tokens:
            renamed = ShelfmarkNormalizer.TOKEN_RENAMES.get(tok.upper(), tok)
            if renamed == "":
                continue
            tok = renamed
            # Drop folio / leaf designators (the following number is kept).
            if tok.lower() in ShelfmarkNormalizer.NOISE_TOKENS:
                continue
            # Strip leading zeros from pure-digit tokens (00018 -> 18).
            if tok.isdigit():
                tok = str(int(tok))
            normalized_parts.append(tok)

        normalized_parts = ShelfmarkNormalizer._collapse_ts_classmark(normalized_parts)

        return "_".join(normalized_parts)

    @staticmethod
    def generate_variants(canonical_id: str) -> List[str]:
        """
        Generate common variant formats for a canonical ID.

        This helps with search - users can find documents using any variant.

        Args:
            canonical_id: Canonical ID (e.g., "T_S_AS_18_170")

        Returns:
            List of variant formats

        Examples:
             ShelfmarkNormalizer.generate_variants('T_S_AS_18_170')
            ['T-S AS 18.170', 'TS AS 18.170', 'T-S AS 18/170', 'CUL: T-S AS 18.170', ...]
        """
        if not canonical_id:
            return []

        variants: Set[str] = set()

        # Parse the canonical ID
        parts = canonical_id.split('_')

        # Determine collection prefix
        collection_prefix = None
        rest_parts = parts

        # Check for known collections
        if len(parts) >= 2:
            # Try two-part prefix first (T_S, L_G, etc.)
            two_part = f"{parts[0]}_{parts[1]}"
            if two_part in ['T_S', 'L_G', 'BL_Or', 'BL_Add', 'CUL_Or', 'CUL_Add', 'MS_Heb']:
                collection_prefix = two_part
                rest_parts = parts[2:]
            # Try three-part prefix (T_S_AS, T_S_NS, etc.)
            elif len(parts) >= 3:
                three_part = f"{parts[0]}_{parts[1]}_{parts[2]}"
                if three_part in ['T_S_AS', 'T_S_NS', 'T_S_K', 'T_S_Ar', 'ENA_NS', 'Gaster_A', 'Gaster_B']:
                    collection_prefix = three_part
                    rest_parts = parts[3:]
            # Single-part prefix
            elif parts[0] in ['ENA', 'AIU', 'Mosseri', 'Halper', 'CAJS', 'Bodl']:
                collection_prefix = parts[0]
                rest_parts = parts[1:]

        # Generate variant formats
        if collection_prefix and rest_parts:
            # Convert prefix to display formats
            prefix_variants = ShelfmarkNormalizer._get_prefix_variants(collection_prefix)

            # Convert rest to different separator formats
            rest_space = ' '.join(rest_parts)
            rest_dot = '.'.join(rest_parts)
            rest_slash = '/'.join(rest_parts)
            rest_underscore = '_'.join(rest_parts)

            for prefix_var in prefix_variants:
                # Common formats
                variants.add(f"{prefix_var} {rest_space}")
                variants.add(f"{prefix_var} {rest_dot}")
                variants.add(f"{prefix_var} {rest_slash}")

                # With institution prefixes
                if collection_prefix.startswith('T_S'):
                    variants.add(f"CUL: {prefix_var} {rest_space}")
                    variants.add(f"Cambridge CUL: {prefix_var} {rest_space}")
                elif collection_prefix == 'ENA':
                    variants.add(f"JTS: {prefix_var} {rest_space}")

        # Always include the canonical ID itself
        variants.add(canonical_id)

        return sorted(list(variants))

    @staticmethod
    def _get_prefix_variants(canonical_prefix: str) -> List[str]:
        """Get display variants for a canonical prefix"""
        variants = [canonical_prefix.replace('_', '-')]  # T_S → T-S

        # Add common variants
        if canonical_prefix == 'T_S':
            variants.extend(['T-S', 'TS', 'T S'])
        elif canonical_prefix == 'T_S_AS':
            variants.extend(['T-S AS', 'TS AS'])
        elif canonical_prefix == 'T_S_NS':
            variants.extend(['T-S NS', 'TS NS'])
        elif canonical_prefix == 'L_G':
            variants.extend(['L-G', 'LG'])
        elif canonical_prefix == 'ENA':
            variants.append('ENA')
        elif canonical_prefix == 'Mosseri':
            variants.extend(['Mosseri', 'Moss', 'Moss.'])

        return variants

    @staticmethod
    def normalize_document(shelf_mark: str) -> Dict[str, any]:
        """
        Complete normalization: canonical ID + variants + original.

        This is the main function to use when processing documents.

        Args:
            shelf_mark: Input shelf mark in any format

        Returns:
            Dictionary with canonical_id, variants, and original

        Examples:
            ShelfmarkNormalizer.normalize_document('T-S AS 18.170')
            {
                'canonical_id': 'T_S_AS_18_170',
                'variants': ['T-S AS 18.170', 'TS AS 18.170', ...],
                'original': 'T-S AS 18.170'
            }
        """
        canonical_id = ShelfmarkNormalizer.to_canonical_id(shelf_mark)
        variants = ShelfmarkNormalizer.generate_variants(canonical_id)

        return {
            'canonical_id': canonical_id,
            'variants': variants,
            'original': shelf_mark
        }

    @staticmethod
    def are_equivalent(shelfmark1: str, shelfmark2: str) -> bool:
        """
        Check if two shelf marks are equivalent (same canonical ID).

        Args:
            shelfmark1: First shelf mark
            shelfmark2: Second shelf mark

        Returns:
            True if they normalize to the same canonical ID
        """
        id1 = ShelfmarkNormalizer.to_canonical_id(shelfmark1)
        id2 = ShelfmarkNormalizer.to_canonical_id(shelfmark2)
        return id1 == id2

    # Maps known shelfmark prefixes → institution / collection / sub-collection
    INSTITUTION_MAPPING: Dict[str, Dict[str, Optional[str]]] = {
        # Cambridge University Library (Taylor-Schechter)
        "T-S": {"institution": "Cambridge University Library", "collection": "Taylor-Schechter", "subcollection": None},
        "T-S AS": {"institution": "Cambridge University Library", "collection": "Taylor-Schechter", "subcollection": "Additional Series"},
        "T-S NS": {"institution": "Cambridge University Library", "collection": "Taylor-Schechter", "subcollection": "New Series"},
        "T-S Ar": {"institution": "Cambridge University Library", "collection": "Taylor-Schechter", "subcollection": "Arabic"},
        "T-S K": {"institution": "Cambridge University Library", "collection": "Taylor-Schechter", "subcollection": "Miscellaneous"},
        "CUL Add": {"institution": "Cambridge University Library", "collection": "Additional Manuscripts", "subcollection": None},
        "ULC Add": {"institution": "Cambridge University Library", "collection": "Additional Manuscripts", "subcollection": None},
        "CUL Or": {"institution": "Cambridge University Library", "collection": "Oriental Manuscripts", "subcollection": None},
        "Mosseri": {"institution": "Cambridge University Library", "collection": "Mosseri", "subcollection": None},
        "L-G": {"institution": "Cambridge University Library / Bodleian Library Oxford", "collection": "Lewis-Gibson", "subcollection": None},
        # JTS / ENA
        "ENA": {"institution": "Jewish Theological Seminary", "collection": "Elkan Nathan Adler", "subcollection": "Main series"},
        "ENA NS": {"institution": "Jewish Theological Seminary", "collection": "Elkan Nathan Adler", "subcollection": "New Series"},
        "ENA II": {"institution": "Jewish Theological Seminary", "collection": "Elkan Nathan Adler", "subcollection": "Second Adler acquisition"},
        "JTS MS": {"institution": "Jewish Theological Seminary", "collection": "General Manuscripts", "subcollection": None},
        "JTS MS Rabbinica": {"institution": "Jewish Theological Seminary", "collection": "General Manuscripts", "subcollection": "Rabbinic literature from ENA"},
        "JTS MS Lutzki": {"institution": "Jewish Theological Seminary", "collection": "General Manuscripts", "subcollection": "Biblical texts from ENA"},
        "JTS Scroll": {"institution": "Jewish Theological Seminary", "collection": "General Manuscripts", "subcollection": "Scrolls from ENA"},
        "KE": {"institution": "Jewish Theological Seminary", "collection": "Kahle Acquisition", "subcollection": None},
        # Manchester / Rylands (Gaster)
        "Gaster A": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Series A"},
        "Gaster B": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Series B"},
        "Gaster P": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Series P"},
        "Gaster L": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Series L"},
        "Gaster C": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Series C"},
        "Gaster Ar": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Arabic Series"},
        "Gaster Hebrew MS": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Hebrew Manuscripts"},
        "Gaster Hebrew MS Add": {"institution": "John Rylands Library, University of Manchester", "collection": "Gaster", "subcollection": "Hebrew Manuscripts Additional"},
        "Rylands Genizah": {"institution": "John Rylands Library, University of Manchester", "collection": "Pre-Gaster Acquisitions", "subcollection": None},
        "JRL": {"institution": "John Rylands Library, University of Manchester", "collection": "General designation", "subcollection": None},
        # Bodleian
        "MS. Heb.": {"institution": "Bodleian Library, Oxford", "collection": "Hebrew Manuscripts", "subcollection": None},
        "Bodl.": {"institution": "Bodleian Library, Oxford", "collection": "Bodleian Manuscripts", "subcollection": None},
        # Penn
        "Halper": {"institution": "University of Pennsylvania, Katz Center", "collection": "Dropsie College", "subcollection": None},
        "CAJS": {"institution": "University of Pennsylvania, Center for Advanced Judaic Studies", "collection": "CAJS", "subcollection": None},
        "Penn CAJS": {"institution": "University of Pennsylvania, Center for Advanced Judaic Studies", "collection": "CAJS", "subcollection": None},
        # AIU Paris
        "AIU": {"institution": "Alliance Israélite Universelle, Paris", "collection": "Cairo Genizah", "subcollection": None},
        # Budapest
        "Kaufmann": {"institution": "Hungarian Academy of Sciences, Budapest", "collection": "David Kaufmann", "subcollection": None},
        "MS Kaufmann A": {"institution": "Hungarian Academy of Sciences, Budapest", "collection": "David Kaufmann", "subcollection": "Codices"},
        "DKG": {"institution": "Hungarian Academy of Sciences, Budapest", "collection": "David Kaufmann", "subcollection": None},
        # British Library
        "BL Or": {"institution": "British Library, London", "collection": "Oriental Manuscripts", "subcollection": None},
        "BL Add": {"institution": "British Library, London", "collection": "Additional Manuscripts", "subcollection": None},
        # Russia / St. Petersburg
        "RNL": {"institution": "Russian National Library, St. Petersburg", "collection": "Genizah", "subcollection": None},
        "Yevr.": {"institution": "Russian National Library, St. Petersburg", "collection": "Evreiskii (Jewish)", "subcollection": None},
        # Dropsie alias
        "Dropsie": {"institution": "University of Pennsylvania (formerly Dropsie College)", "collection": "Dropsie", "subcollection": None},
    }

    @staticmethod
    def _normalize_for_prefix_match(shelf_mark: str) -> str:
        """Translate raw shelfmark encodings (e.g. TEI MS- prefixes, Manchester format)
        into a human-readable form suitable for prefix lookup against INSTITUTION_MAPPING."""
        if not shelf_mark:
            return ""
        sm = shelf_mark.strip()
        sm = sm.replace("MS-TS-AS", "T-S AS")
        sm = sm.replace("MS-TS-NS", "T-S NS")
        sm = sm.replace("MS-TS-K", "T-S K")
        sm = sm.replace("MS-TS-AR", "T-S Ar")
        sm = sm.replace("MS-TS-", "T-S ")
        sm = sm.replace("MS-MOSSERI", "Mosseri")
        sm = sm.replace("MS-L-G", "L-G")
        sm = sm.replace("MS-OR-", "CUL Or ")
        sm = sm.replace("MS-ADD-", "CUL Add ")
        if sm.startswith("Manchester:") or sm.startswith("Manchester "):
            parts = sm.split(":", 1)
            tail = parts[1].strip() if len(parts) > 1 else sm
            if tail.startswith("A "):
                return "Gaster A"
            if tail.startswith("B "):
                return "Gaster B"
            if tail.startswith("P "):
                return "Gaster P"
            if tail.startswith("L "):
                return "Gaster L"
            if tail.startswith("C "):
                return "Gaster C"
        return sm

    @staticmethod
    def get_institution_info(shelf_mark: str) -> Dict[str, Optional[str]]:
        """Return institution, collection, and subcollection for a shelfmark.

        Matches the shelfmark against INSTITUTION_MAPPING using the longest
        known prefix, after normalising TEI and Manchester encodings.

        Args:
            shelf_mark: Shelfmark in any known format.

        Returns:
            Dict with keys 'institution', 'collection', 'subcollection'
            (all Optional[str]).  Returns all-None dict if no match found.

        Examples:
            ShelfmarkNormalizer.get_institution_info("T-S AS 18.170")
            {'institution': 'Cambridge University Library', 'collection': 'Taylor-Schechter', 'subcollection': 'Additional Series'}
            ShelfmarkNormalizer.get_institution_info("MS-TS-AS-00018-00170")
            {'institution': 'Cambridge University Library', 'collection': 'Taylor-Schechter', 'subcollection': 'Additional Series'}
            ShelfmarkNormalizer.get_institution_info("Paris X.10")
            {'institution': 'Alliance Israélite Universelle, Paris', 'collection': 'Cairo Genizah', 'subcollection': 'Series X'}
        """
        empty: Dict[str, Optional[str]] = {"institution": None, "collection": None, "subcollection": None}
        if not shelf_mark:
            return empty

        sm = shelf_mark.strip()

        # Special handling: Paris / AIU with Roman-numeral sub-collections
        if "Paris" in sm or "AIU" in sm:
            result = {"institution": "Alliance Israélite Universelle, Paris",
                      "collection": "Cairo Genizah",
                      "subcollection": None}
            rest = sm.replace("Paris", "").replace("AIU", "").strip(" ,:")
            roman_match = re.match(r"^([IVX]+)", rest)
            if roman_match:
                result["subcollection"] = f"Series {roman_match.group(1)}"
            return result

        candidate = ShelfmarkNormalizer._normalize_for_prefix_match(sm)
        best_key = None
        for key in sorted(ShelfmarkNormalizer.INSTITUTION_MAPPING.keys(), key=len, reverse=True):
            if candidate.startswith(key):
                best_key = key
                break

        if not best_key:
            return empty

        mapped = ShelfmarkNormalizer.INSTITUTION_MAPPING[best_key]
        return {
            "institution": mapped.get("institution"),
            "collection": mapped.get("collection"),
            "subcollection": mapped.get("subcollection"),
        }


def test_normalization():
    """Test canonical ID generation and variant generation"""
    test_cases = [
        {
            'input': 'T-S AS 18.170',
            'expected_id': 'T_S_AS_18_170',
            'expected_variants_include': ['T-S AS 18.170', 'TS AS 18.170']
        },
        {
            'input': 'Cambridge CUL: T-S AS 18.170',
            'expected_id': 'T_S_AS_18_170',
            'expected_variants_include': ['CUL: T-S AS 18.170']
        },
        {
            'input': 'MS-TS-AS-00018-00170',
            'expected_id': 'T_S_AS_18_170',
            'expected_variants_include': ['T-S AS 18.170']
        },
        {
            'input': 'ENA 2713.17',
            'expected_id': 'ENA_2713_17',
            'expected_variants_include': ['ENA 2713.17']
        },
        {
            'input': 'L-G Ar.I.130',
            'expected_id': 'L_G_Ar_I_130',
            'expected_variants_include': ['L-G Ar.I.130']
        }
    ]

    print("Testing Canonical ID Generation:")
    print("=" * 70)

    for test in test_cases:
        result = ShelfmarkNormalizer.normalize_document(test['input'])
        canonical_id = result['canonical_id']
        variants = result['variants']

        # Check canonical ID
        id_match = canonical_id == test['expected_id']
        print(f"\n{'✓' if id_match else '✗'} Input: {test['input']}")
        print(f"  Canonical ID: {canonical_id}")
        print(f"  Expected:     {test['expected_id']}")

        # Check variants
        print(f"  Variants ({len(variants)}):")
        for var in test['expected_variants_include']:
            if var in variants:
                print(f"    ✓ {var}")
            else:
                print(f"    ✗ {var} (MISSING)")

        # Show sample of all variants
        print(f"  All variants: {variants[:5]}...")

    print("\n" + "=" * 70)

    # Test equivalence
    print("\nTesting Equivalence:")
    print("=" * 70)
    equiv_tests = [
        ('T-S AS 18.170', 'Cambridge CUL: T-S AS 18.170', True),
        ('MS-TS-AS-00018-00170', 'T-S AS 18.170', True),
        ('T-S AS 18.170', 'T-S AS 19.170', False),
    ]

    for sm1, sm2, expected in equiv_tests:
        result = ShelfmarkNormalizer.are_equivalent(sm1, sm2)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{sm1}' ≡ '{sm2}': {result} (expected: {expected})")


if __name__ == "__main__":
    test_normalization()