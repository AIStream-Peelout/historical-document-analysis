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
from typing import Dict, List, Optional, Set


class ShelfmarkNormalizer:
    """Normalize shelfmarks to canonical IDs and generate search variants"""

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

        canonical = shelfmark.strip()

        # Remove institution prefixes first
        for inst_prefix in ShelfmarkNormalizer.INSTITUTION_PREFIXES:
            # Match "Cambridge CUL: " or "Cambridge CUL " or "Cambridge CUL, "
            pattern = rf'^{re.escape(inst_prefix)}\s*[:\s,]\s*'
            canonical = re.sub(pattern, '', canonical, flags=re.IGNORECASE)

        # Handle special Manchester format
        if canonical.startswith("Manchester:") or canonical.startswith("Manchester "):
            parts = canonical.split(":", 1) if ":" in canonical else canonical.split(" ", 1)
            if len(parts) > 1:
                tail = parts[1].strip()
                if tail.startswith("A "):
                    canonical = "Gaster_A_" + tail[2:]
                elif tail.startswith("B "):
                    canonical = "Gaster_B_" + tail[2:]
                elif tail.startswith("P "):
                    canonical = "Gaster_P_" + tail[2:]
                elif tail.startswith("L "):
                    canonical = "Gaster_L_" + tail[2:]
                elif tail.startswith("C "):
                    canonical = "Gaster_C_" + tail[2:]

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

        # Normalize collection prefix
        for variant, standard in sorted(ShelfmarkNormalizer.COLLECTION_PREFIXES.items(),
                                        key=lambda x: len(x[0]), reverse=True):
            if canonical.upper().startswith(variant.upper()):
                # Replace the prefix
                rest = canonical[len(variant):]
                canonical = standard + rest
                break

        # Convert all separators to underscores
        canonical = canonical.replace(' ', '_')
        canonical = canonical.replace('.', '_')
        canonical = canonical.replace('-', '_')
        canonical = canonical.replace('/', '_')
        canonical = canonical.replace(',', '_')
        canonical = canonical.replace(':', '_')

        # Remove leading zeros from numbers (18.170 → 18_170, not 00018_00170)
        parts = canonical.split('_')
        normalized_parts = []
        for part in parts:
            if part.isdigit():
                # Remove leading zeros but keep at least one digit
                normalized_parts.append(str(int(part)))
            else:
                normalized_parts.append(part)
        canonical = '_'.join(normalized_parts)

        # Clean up multiple underscores
        while '__' in canonical:
            canonical = canonical.replace('__', '_')

        # Strip trailing underscores
        canonical = canonical.strip('_')

        return canonical

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