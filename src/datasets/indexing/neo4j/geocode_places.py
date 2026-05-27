#!/usr/bin/env python3
"""
Geocode Place nodes in the Neo4j knowledge graph.

Finds all Place nodes that are missing lat/lng and uses the Nominatim
geocoder (OpenStreetMap, free, no API key required) to add:

  - lat        : float latitude
  - lng        : float longitude
  - city       : city name (if not already set)
  - country    : country name (if not already set)
  - region     : broad region (if not already set)
  - osm_display_name : full display name from OSM for reference

Results are written back to Neo4j. Already-geocoded nodes are skipped.

Usage
-----
python geocode_places.py              # geocode all un-geocoded places
python geocode_places.py --dry-run    # show what would be geocoded
python geocode_places.py --all        # re-geocode everything (overwrite)
python geocode_places.py --limit 50   # process at most 50 places

Requirements
------------
pip install geopy
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import dotenv

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))
dotenv.load_dotenv(project_root / ".env")

from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known historical → modern name overrides for places the LLM extracts using
# historical / transliterated names that Nominatim doesn't recognise.
# ---------------------------------------------------------------------------
HISTORICAL_OVERRIDES: dict[str, str] = {
    "Fustat":         "Old Cairo, Cairo, Egypt",
    "Fusṭāṭ":         "Old Cairo, Cairo, Egypt",
    "al-Fustat":      "Old Cairo, Cairo, Egypt",
    "Qayrawān":       "Kairouan, Tunisia",
    "Qayrawan":       "Kairouan, Tunisia",
    "al-Qahira":      "Cairo, Egypt",
    "Misr":           "Cairo, Egypt",
    "Raqqa":          "Ar-Raqqah, Syria",
    "Ramla":          "Ramla, Israel",
    "Ramle":          "Ramla, Israel",
    "al-Ramla":       "Ramla, Israel",
    "Tyre":           "Tyre, Lebanon",
    "Acre":           "Akko, Israel",
    "Ascalon":        "Ashkelon, Israel",
    "Tinnis":         "Tinnis, Egypt",
    "Damietta":       "Damietta, Egypt",
    "Byzantium":      "Istanbul, Turkey",
    "Constantinople": "Istanbul, Turkey",
    "Tripoli (Libya)":  "Tripoli, Libya",
    "Tripoli (Lebanon)":"Tripoli, Lebanon",
    "Dahlak":             "Dahlak Archipelago, Eritrea",
    "Dahlak Archipelago": "Dahlak Archipelago, Eritrea",
    "al-Andalus":         "Andalusia, Spain",
    "Al-Andalus":         "Andalusia, Spain",
    "Andalus":            "Andalusia, Spain",
    "Sefarad":            "Iberian Peninsula, Spain",

    # Old Cairo / Fustat area — the heart of the Genizah world
    "Qaṣr al-Shamʿ":               "Coptic Cairo, Egypt",
    "Qaṣr al-Shamʿ (neighborhood)": "Coptic Cairo, Egypt",
    "Qasr al-Sham'":               "Coptic Cairo, Egypt",
    "Qasr al-Shama":               "Coptic Cairo, Egypt",
    "al-Fustat":          "Old Cairo, Egypt",
    "Fusṭāṭ":             "Old Cairo, Egypt",
    "Fustat":             "Old Cairo, Egypt",

    # Greater Syria / Levant — "al-Sham" as a region, not a neighbourhood
    "Bilād al-Shām":      "Damascus, Syria",
    "Bilad al-Sham":      "Damascus, Syria",
    "al-Shām":            "Damascus, Syria",
    "al-Sham":            "Damascus, Syria",

    # Other common mismatches
    "Miṣr":               "Cairo, Egypt",
    "Misr":               "Cairo, Egypt",
    "al-Qāhira":          "Cairo, Egypt",
    "Ifrīqiya":           "Kairouan, Tunisia",
    "Ifriqiya":           "Kairouan, Tunisia",
    "Diyār al-Yemen":     "Sana'a, Yemen",
    "Diyar al-Yemen":     "Sana'a, Yemen",
}

# Broad region assignments for countries when the geocoder doesn't return one.
COUNTRY_TO_REGION: dict[str, str] = {
    "Egypt": "North Africa",
    "Tunisia": "North Africa",
    "Libya": "North Africa",
    "Morocco": "North Africa",
    "Algeria": "North Africa",
    "Sudan": "North Africa",
    "Iraq": "Middle East",
    "Syria": "Middle East",
    "Lebanon": "Middle East",
    "Israel": "Middle East",
    "Palestine": "Middle East",
    "Jordan": "Middle East",
    "Yemen": "Middle East",
    "Saudi Arabia": "Middle East",
    "Iran": "Middle East",
    "Turkey": "Middle East",
    "Spain": "Europe",
    "Italy": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "United Kingdom": "Europe",
    "England": "Europe",
    "Netherlands": "Europe",
    "Greece": "Europe",
    "India": "South Asia",
    "Ethiopia": "East Africa",
    "Kenya": "East Africa",
    "Somalia": "East Africa",
    "Afghanistan": "South Asia",
    "Pakistan": "South Asia",
    "China": "East Asia",
}


import re as _re
import unicodedata as _ud

# OSM classes that represent real geographic places.
# Anything else (amenity=restaurant, highway=road, building=yes, etc.) is rejected.
_GEOGRAPHIC_CLASSES = {"place", "boundary", "natural", "waterway", "landuse"}

# Approximate bounding box for the Cairo Genizah world:
# Morocco/West Africa in the west → India/Pakistan in the east
# Sub-Saharan East Africa in the south → Central Asia/Europe in the north
# New York/Cambridge are legitimate (manuscript repositories) so we allow a wide lng range,
# but we use a name-based heuristic to reject US matches for clearly Arabic/Hebrew names.
_GENIZAH_LAT_MIN = -5
_GENIZAH_LAT_MAX = 60
_GENIZAH_LNG_MIN = -20
_GENIZAH_LNG_MAX = 85   # up to western India

# Countries that are plausible false-positive traps but NOT valid Genizah locations.
# A match in one of these for a non-English-looking name is rejected.
_SUSPECT_COUNTRIES = {
    "United States", "Canada", "Mexico", "Brazil", "Argentina", "Australia",
    "New Zealand", "Colombia", "Peru", "Chile", "Venezuela", "Cuba",
    "Guatemala", "Honduras", "Nicaragua", "Costa Rica", "Panama",
}

def _looks_arabic_or_hebrew(name: str) -> bool:
    """True if the name looks like a transliterated Arabic/Hebrew place name.

    Heuristics: contains diacritics (ā, ī, ū, ṭ, ḥ, etc.) or starts with 'al-'/'ibn-'.
    Modern English city names (Cambridge, Oxford, New York) return False.
    """
    arabic_diacritics = set("āīūḥṭṣḍẓġḳḏṯḫṻẗÿāīūĀĪŪḤṬṢḌẒĠḲ")
    if any(c in arabic_diacritics for c in name):
        return True
    lower = name.lower()
    if lower.startswith(("al-", "ibn ", "bab ", "dar ", "diyar", "umm ")):
        return True
    return False


def _is_hebrew_script(name: str) -> bool:
    """True if the name contains Hebrew/Aramaic Unicode characters."""
    return any('א' <= c <= 'ת' or 'יִ' <= c <= 'פֿ' for c in name)


# Hebrew concepts / abstract terms that are not geocodable places.
# Names in this set are silently skipped rather than sent to the geocoder.
_HEBREW_NON_PLACES: set[str] = {
    "גן עדן",       # Garden of Eden
    "גניזה",        # The Genizah (the collection, not a place)
    "גלות",         # Exile / Diaspora
    "ארץ ישראל",    # Land of Israel (too broad)
    "הארץ",         # The Land
    "העולם הבא",    # The World to Come
    "שמים",         # Heaven/Sky
    "ים סוף",       # Red Sea / Sea of Reeds (broad region, not a city)
}

# Hebrew institution prefixes — strip these to get the actual place name for geocoding.
# e.g. "ישיבת סורא" → geocode "סורא" (Sura, Iraq)
_HEBREW_INSTITUTION_PREFIXES = [
    "ישיבת ",    # Yeshiva of
    "בית כנסת ", # Synagogue of
    "בית דין ",  # Court of
    "בית מדרש ", # Study house of
    "קהילת ",    # Community of
    "כנסת ",     # Assembly/Synagogue of
]

# Basic Hebrew consonant → Latin approximation for Nominatim fallback.
# Hebrew is an abjad so this is inherently approximate, but good enough for
# well-known place names (Sura, Pumbedita, Baghdad, etc.)
_HEBREW_TO_LATIN: dict[str, str] = {
    'א': '', 'ב': 'b', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ו': 'v',
    'ז': 'z', 'ח': 'kh', 'ט': 't', 'י': 'y', 'כ': 'k', 'ך': 'k',
    'ל': 'l', 'מ': 'm', 'ם': 'm', 'נ': 'n', 'ן': 'n', 'ס': 's',
    'ע': '', 'פ': 'p', 'ף': 'f', 'צ': 'ts', 'ץ': 'ts', 'ק': 'k',
    'ר': 'r', 'ש': 'sh', 'ת': 't',
    # Final forms already covered above; vowel points / cantillation marks → skip
}

def _transliterate_hebrew(name: str) -> str:
    """Approximate Hebrew→Latin transliteration for Nominatim queries.

    Strips vowel points and maps each consonant to a Latin approximation.
    The result is not scholarly transliteration — it's just a Nominatim-friendly
    ASCII string (e.g. "סורא" → "svra", close enough for Nominatim to find Sura).
    """
    # Strip Hebrew vowel points (niqqud) and cantillation marks (U+05B0–U+05C7)
    cleaned = ''.join(c for c in name if not ('ְ' <= c <= 'ׇ'))
    result = ''.join(_HEBREW_TO_LATIN.get(c, c if c.isascii() else '') for c in cleaned)
    # Collapse multiple spaces/hyphens and strip
    return _re.sub(r'\s+', ' ', result).strip()


def _prepare_hebrew_queries(name: str) -> tuple[list[str], bool]:
    """Return (query_list, is_institution) for a Hebrew-script place name.

    Strips institution prefixes so the geocoder sees the actual place name.
    Returns the Hebrew form first (Google handles it natively), then a
    transliterated ASCII form as fallback for Nominatim.

    Returns ([], False) for names that are known non-places (Garden of Eden etc.)
    """
    if name in _HEBREW_NON_PLACES:
        return [], False

    is_institution = False
    geocode_name = name
    for prefix in _HEBREW_INSTITUTION_PREFIXES:
        if name.startswith(prefix):
            geocode_name = name[len(prefix):].strip()
            is_institution = True
            break

    transliterated = _transliterate_hebrew(geocode_name)
    queries = [geocode_name]
    if transliterated and transliterated != geocode_name:
        queries.append(transliterated)
    return queries, is_institution


def _strip_diacritics(s: str) -> str:
    return _ud.normalize("NFKD", s).encode("ascii", "ignore").decode().strip()


def _parse_princeton_coords(coord_str: str) -> Optional[tuple[float, float]]:
    """Parse Princeton's DMS coordinate string into (lat, lng).

    Handles formats like:
      '15° 49' 48" N, 40° 12' 0" E'
      '31°15′N, 29°55′E'
      '30.0500° N, 31.2333° E'
    Returns (lat, lng) floats or None if unparseable.
    """
    if not coord_str:
        return None

    # Normalise fancy quote characters
    s = coord_str.replace("′", "'").replace("″", '"').replace("°", "°")

    # Try decimal degrees first: '30.0500° N, 31.2333° E'
    dec = _re.findall(r'([\d.]+)\s*°?\s*([NS]),?\s*([\d.]+)\s*°?\s*([EW])', s)
    if dec:
        lat_v, lat_d, lng_v, lng_d = dec[0]
        lat = float(lat_v) * (-1 if lat_d == "S" else 1)
        lng = float(lng_v) * (-1 if lng_d == "W" else 1)
        return lat, lng

    # DMS: degrees° minutes' seconds" N/S, degrees° minutes' seconds" E/W
    dms = _re.findall(
        r'(\d+)\s*°\s*(\d+)\s*[\'′]\s*([\d.]+)?\s*[\"″]?\s*([NS]),?\s*'
        r'(\d+)\s*°\s*(\d+)\s*[\'′]\s*([\d.]+)?\s*[\"″]?\s*([EW])',
        s
    )
    if dms:
        ld, lm, ls, ld_dir, nd, nm, ns, nd_dir = dms[0]
        lat = int(ld) + int(lm)/60 + float(ls or 0)/3600
        lng = int(nd) + int(nm)/60 + float(ns or 0)/3600
        if ld_dir == "S": lat = -lat
        if nd_dir == "W": lng = -lng
        return lat, lng

    return None

def _clean_place_name(name: str) -> tuple[str, str]:
    """Return (cleaned_name, context_hint) where hint comes from parenthetical qualifiers.

    'Abwān (near Damietta)'  →  ('Abwān', 'near Damietta')
    'Abyssinia (region)'     →  ('Abyssinia', 'region')
    'Tripoli (Libya)'        →  ('Tripoli', 'Libya')
    """
    m = _re.search(r'\(([^)]+)\)', name)
    hint = m.group(1).strip() if m else ""
    clean = _re.sub(r'\s*\([^)]*\)', '', name).strip()
    return clean, hint


def _geocode_one(geolocator, name: str) -> Optional[dict]:
    """Geocode a single place name. Returns a result dict or None.

    Tries (in order):
      1. HISTORICAL_OVERRIDES exact match
      2. Name as-is
      3. Name + parenthetical hint as context (e.g. "Abwān, near Damietta")
      4. Cleaned name (parentheticals stripped)
      5. ASCII-stripped version of the cleaned name
    """
    import time as _time

    def _try(query: str):
        try:
            loc = geolocator.geocode(query, exactly_one=True, timeout=10,
                                     addressdetails=True, language="en")
            return loc
        except Exception:
            return None

    # 1. Historical override — curated, always trusted
    if name in HISTORICAL_OVERRIDES:
        loc = _try(HISTORICAL_OVERRIDES[name])
        if loc:
            result = _build_result(loc, place_name=name)
            if result:
                result["geocode_source"]  = "override"
                result["geocode_suspect"] = False
            return result

    clean, hint = _clean_place_name(name)

    # 2. Name as-is (if no parens)
    if clean == name:
        loc = _try(name)
        if loc:
            return _build_result(loc, place_name=name)
    else:
        # 3. Clean name + hint as geographic context
        if hint and not hint.lower().startswith(("region", "area", "province")):
            loc = _try(f"{clean}, {hint}")
            if loc:
                return _build_result(loc)

        # 4. Cleaned name alone
        loc = _try(clean)
        if loc:
            return _build_result(loc, place_name=name)

    # 5. ASCII-stripped fallback
    ascii_name = _strip_diacritics(clean)
    if ascii_name and ascii_name != clean:
        _time.sleep(0.5)   # extra courtesy delay for extra request
        loc = _try(ascii_name)
        if loc:
            return _build_result(loc, place_name=name)

    return None


# OSM types that indicate a sub-city unit (neighbourhood, quarter, etc.)
# These are almost always wrong for historical Genizah place names.
_SUSPECT_OSM_TYPES = {
    "neighbourhood", "neighborhood", "quarter", "suburb",
    "hamlet", "isolated_dwelling", "allotments", "city_block",
}

# Nominatim importance threshold below which we flag as suspect.
# Well-known cities score 0.6+. Random sub-city units score < 0.4.
_IMPORTANCE_THRESHOLD = 0.4


def _build_result(loc, place_name: str = "") -> Optional[dict]:
    """Extract a clean result dict from a geopy Location.

    Returns None if the OSM result is not a geographic place
    (e.g. a restaurant, street, or building with the same name)
    or is geographically implausible for a Genizah place name.

    Sets `geocode_suspect=True` when the result looks like a sub-city
    unit (neighbourhood, quarter) rather than an actual place, so the
    frontend can flag it for manual review.
    """
    osm_class  = loc.raw.get("class",      "")
    osm_type   = loc.raw.get("type",       "")
    importance = float(loc.raw.get("importance", 1.0))

    # Reject non-geographic hits: amenities, buildings, roads, shops, etc.
    if osm_class not in _GEOGRAPHIC_CLASSES:
        logger.debug(f"  Rejecting OSM class='{osm_class}' type='{osm_type}' "
                     f"→ '{loc.address[:60]}'")
        return None

    addr     = loc.raw.get("address", {})
    country  = addr.get("country") or addr.get("state") or ""
    lat, lng = loc.latitude, loc.longitude

    # For Arabic/Hebrew transliterated names, reject results that land in
    # geographically implausible locations (Americas, Pacific, etc.)
    if place_name and _looks_arabic_or_hebrew(place_name):
        if not (_GENIZAH_LAT_MIN <= lat <= _GENIZAH_LAT_MAX and
                _GENIZAH_LNG_MIN <= lng <= _GENIZAH_LNG_MAX):
            logger.debug(f"  Rejecting out-of-region result ({lat:.2f},{lng:.2f}) "
                         f"for '{place_name}'")
            return None
        if country in _SUSPECT_COUNTRIES:
            logger.debug(f"  Rejecting suspect country '{country}' for '{place_name}'")
            return None

    # Flag as suspect if the result is a sub-city unit or has low importance
    suspect = (osm_type in _SUSPECT_OSM_TYPES) or (importance < _IMPORTANCE_THRESHOLD)
    if suspect:
        logger.debug(f"  Flagging as suspect: type='{osm_type}' importance={importance:.2f} "
                     f"→ '{loc.address[:60]}'")

    city   = (addr.get("city") or addr.get("town") or
               addr.get("village") or addr.get("county") or "")
    region = COUNTRY_TO_REGION.get(country, "")
    return {
        "lat":              lat,
        "lng":              lng,
        "city":             city,
        "country":          country,
        "region":           region,
        "osm_display_name": loc.address,
        "geocode_source":   "nominatim",
        "geocode_suspect":  suspect,
    }


def _geocode_one_compat(geolocator, name: str) -> Optional[dict]:
    """Error-catching wrapper around _geocode_one."""
    try:
        return _geocode_one(geolocator, name)
    except Exception as e:
        logger.warning(f"  Geocoding '{name}' failed: {e}")
        return None


def _extract_institution_location(name: str) -> list[str]:
    """Return a ranked list of queries to try geocoding an institution.

    Extracts the most specific locatable part of the name so Nominatim
    can find a real city/university rather than failing on the full name.

    Examples
    --------
    "University of Chicago"                  → ["University of Chicago", "Chicago"]
    "Oxford Centre for Hebrew and Jewish Studies" → ["Oxford Centre for Hebrew and Jewish Studies",
                                                       "Oxford University", "Oxford"]
    "Jewish Theological Seminary"            → ["Jewish Theological Seminary", "New York"]
    "Brandeis University"                    → ["Brandeis University", "Waltham Massachusetts"]
    "Ben-Gurion University of the Negev"     → ["Ben-Gurion University of the Negev", "Beersheba Israel"]
    """
    queries = [name]  # always try the full name first

    lower = name.lower()

    # Handle comma-separated format: "Beeghly Library, University of Heidelberg"
    # Try each comma segment as a separate query (most specific first).
    if "," in name:
        parts = [p.strip() for p in name.split(",") if p.strip()]
        # Add the last segment (usually the parent institution / city) as fallback
        if len(parts) >= 2:
            queries.append(parts[-1])   # e.g. "University of Heidelberg"
            queries.append(parts[0])    # e.g. "Beeghly Library"

    # "University of X" anywhere in the name  →  try "X" as the city
    m = _re.search(r'university of ([^,]+)', lower)
    if m:
        city_part = name[m.start(1):m.start(1) + len(m.group(1))]
        city_part = _re.sub(r'\s*\([^)]*\)', '', city_part).strip()
        if city_part not in queries:
            queries.append(city_part)
        return queries

    # "X University"  →  Nominatim usually knows "X University" for famous ones
    # but also try "X" alone as fallback
    m = _re.search(r'(.+?)\s+university\b', lower)
    if m:
        prefix = name[:m.end(1)].strip()
        if prefix.lower() not in ("the", "a") and prefix not in queries:
            queries.append(prefix)
        return queries

    # Named place at the start: "Oxford Centre for...", "Cambridge Institute of..."
    # Extract first token(s) if they look like a place name (capitalised, not "The/A/An")
    first_word = name.split()[0]
    if first_word not in ("The", "A", "An", "New", "Jewish", "Hebrew", "Islamic",
                          "Center", "Centre", "Institute", "National", "International"):
        if first_word not in queries:
            queries.append(first_word)

    # Well-known institution → city overrides
    _KNOWN: dict[str, str] = {
        "jewish theological seminary":          "New York City",
        "jts":                                  "New York City",
        "bodleian":                             "Oxford",
        "taylor-schechter":                     "Cambridge",
        "cambridge university library":         "Cambridge",
        "john rylands":                         "Manchester",
        "british library":                      "London",
        "bibliotheque nationale":               "Paris",
        "national library of israel":           "Jerusalem",
        "hebrew university":                    "Jerusalem",
        "yad ben zvi":                          "Jerusalem",
        "ben-zvi institute":                    "Jerusalem",
        "dropsie college":                      "Philadelphia",
        "mandel institute":                     "Jerusalem",
        "brandeis":                             "Waltham Massachusetts",
        "ben-gurion university":                "Beersheba Israel",
        "bar-ilan":                             "Ramat Gan Israel",
        "genazim institute":                    "Tel Aviv",
        # Hebrew Union College — Genizah fragments are at the Klau Library,
        # Cincinnati campus. HUC also has campuses in Jerusalem, NY, and LA
        # so geocoders default to Jerusalem without this override.
        "hebrew union college":                 "Klau Library, Cincinnati Ohio",
        "klau library":                         "Klau Library, Cincinnati Ohio",
        "huc-jir":                              "Klau Library, Cincinnati Ohio",
    }
    for key, city in _KNOWN.items():
        if key in lower:
            queries.append(city)
            break

    return queries


def _geocode_google(queries: list[str], gmaps_client,
                    suspect_types: set = None) -> Optional[dict]:
    """Geocode using the Google Geocoding API.

    Tries each query in order, returns the first usable result.
    Sets geocode_suspect=True when Google returns a sub-city unit.

    `suspect_types` — additional Google result types to flag (e.g. for
    historical places we flag 'neighborhood', 'sublocality').
    """
    if suspect_types is None:
        suspect_types = {"neighborhood", "sublocality", "sublocality_level_1",
                         "sublocality_level_2", "premise", "point_of_interest"}

    for query in queries:
        try:
            results = gmaps_client.geocode(query)
            if not results:
                continue

            r        = results[0]
            loc      = r["geometry"]["location"]
            lat, lng = loc["lat"], loc["lng"]
            types    = set(r.get("types", []))

            components = {c["types"][0]: c["long_name"]
                          for c in r.get("address_components", [])
                          if c.get("types")}
            country = components.get("country", "")
            city    = (components.get("locality")
                       or components.get("administrative_area_level_2")
                       or components.get("administrative_area_level_1", ""))
            region  = COUNTRY_TO_REGION.get(country, "")

            # Flag as suspect if Google returned a sub-city unit
            suspect = bool(types & suspect_types) and "locality" not in types

            if query != queries[0]:
                logger.debug(f"  Resolved via Google query '{query}'")

            return {
                "lat":              lat,
                "lng":              lng,
                "city":             city,
                "country":          country,
                "region":           region,
                "osm_display_name": r.get("formatted_address", ""),
                "geocode_source":   "google",
                "geocode_suspect":  suspect,
            }
        except Exception as e:
            logger.debug(f"  Google query '{query}' failed: {e}")
            continue

    return None


def _geocode_institution_google(name: str, gmaps_client) -> Optional[dict]:
    """Geocode an institution — institutions can be sub-city so suspect rules are relaxed."""
    queries = [name] + _extract_institution_location(name)[1:]
    # For institutions, only flag truly ambiguous types as suspect
    return _geocode_google(queries, gmaps_client,
                           suspect_types={"premise", "point_of_interest"})


def _geocode_institution_nominatim(geolocator, name: str) -> Optional[dict]:
    """Geocode an institution using Nominatim as fallback."""
    queries = _extract_institution_location(name)

    for query in queries:
        try:
            loc = geolocator.geocode(query, exactly_one=True, timeout=10,
                                     addressdetails=True, language="en")
            if not loc:
                continue

            osm_class = loc.raw.get("class", "")
            accepted  = _GEOGRAPHIC_CLASSES | {"amenity", "tourism", "office"}
            if osm_class not in accepted:
                logger.debug(f"  Skipping OSM class='{osm_class}' for '{query}'")
                continue

            addr    = loc.raw.get("address", {})
            country = addr.get("country") or ""
            city    = (addr.get("city") or addr.get("town") or
                       addr.get("village") or addr.get("county") or "")
            region  = COUNTRY_TO_REGION.get(country, "")

            if query != name:
                logger.debug(f"  '{name}' resolved via Nominatim query '{query}'")

            return {
                "lat":              loc.latitude,
                "lng":              loc.longitude,
                "city":             city,
                "country":          country,
                "region":           region,
                "osm_display_name": loc.address,
            }
        except Exception as e:
            logger.debug(f"  Nominatim query '{query}' failed: {e}")
            continue

    return None


def _run_institution_geocoding(session, geolocator, args, gmaps_client=None):
    """Geocode all Institution nodes that are missing lat/lng."""
    if args.all:
        query = ("MATCH (i:Institution) "
                 "RETURN i.name AS name ORDER BY i.name")
    else:
        query = ("MATCH (i:Institution) WHERE i.lat IS NULL "
                 "RETURN i.name AS name ORDER BY i.name")

    institutions = [r["name"] for r in session.run(query) if r["name"]]
    if args.limit:
        institutions = institutions[:args.limit]

    logger.info(f"Found {len(institutions)} Institution node(s) to geocode")

    if args.dry_run:
        for name in institutions:
            queries = _extract_institution_location(name)
            print(f"  {name}  →  would try: {queries}")
        return

    backend = "Google" if gmaps_client else "Nominatim"
    logger.info(f"Institution geocoding backend: {backend}")

    ok, failed = 0, []
    for name in institutions:
        if gmaps_client:
            result = _geocode_institution_google(name, gmaps_client)
            # Google has no rate limit concern at this scale, but be polite
            time.sleep(0.1)
        else:
            result = _geocode_institution_nominatim(geolocator, name)
            time.sleep(args.delay)

        if result:
            session.run("""
                MATCH (i:Institution {name: $name})
                SET i.lat              = $lat,
                    i.lng              = $lng,
                    i.city             = CASE WHEN $city    <> '' AND i.city    IS NULL THEN $city    ELSE i.city    END,
                    i.country          = CASE WHEN $country <> '' AND i.country IS NULL THEN $country ELSE i.country END,
                    i.region           = CASE WHEN $region  <> '' AND i.region  IS NULL THEN $region  ELSE i.region  END,
                    i.osm_display_name = $osm_display_name
            """, name=name, **result)
            ok += 1
            logger.info(f"  ✅ {name} → ({result['lat']:.4f}, {result['lng']:.4f}) "
                        f"{result['city']}, {result['country']}")
        else:
            failed.append(name)
            logger.warning(f"  ❌ {name} — no result")

    print(f"\n✅ Geocoded {ok}/{len(institutions)} institutions")
    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for f in failed:
            print(f"  • {f}")


def main():
    parser = argparse.ArgumentParser(
        description="Geocode historical Place nodes and Institution nodes in Neo4j.\n"
                    "By default both are geocoded. Use --places-only / --institutions-only "
                    "to restrict to one type."
    )
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be geocoded without writing.")
    parser.add_argument("--all", action="store_true",
                        help="Re-geocode nodes that already have lat/lng.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of nodes to geocode per type.")
    parser.add_argument("--delay", type=float, default=1.1,
                        help="Seconds between Nominatim requests (default 1.1 — "
                             "Nominatim policy is max 1 req/sec).")
    parser.add_argument("--places-only", action="store_true",
                        help="Geocode historical Place nodes only (skip institutions).")
    parser.add_argument("--institutions-only", action="store_true",
                        help="Geocode Institution nodes only (skip historical places).")
    # Legacy alias so existing scripts/docs still work
    parser.add_argument("--institutions", "-i", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # --institutions is a legacy alias for --institutions-only
    if args.institutions:
        args.institutions_only = True

    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        logger.error("geopy is required: pip install geopy")
        sys.exit(1)

    # Fix macOS SSL certificate verification errors
    import ssl
    try:
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        logger.warning("certifi not found — falling back to unverified SSL (pip install certifi to fix)")
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    neo4j_uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user     = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "genizah-prod")

    if not neo4j_password:
        logger.error("NEO4J_PASSWORD not set")
        sys.exit(1)

    driver    = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    geolocator = Nominatim(user_agent="cairo_genizah_kg_geocoder/1.0", ssl_context=ssl_ctx)

    # Google Geocoding API — primary geocoder when key is available
    gmaps_client = None
    google_key   = os.getenv("GOOGLE_MAPS_API_KEY")
    if google_key:
        try:
            import googlemaps
        except ImportError:
            logger.error("GOOGLE_MAPS_API_KEY is set but googlemaps is not installed. "
                         "Run: pip install googlemaps")
            sys.exit(1)
        try:
            gmaps_client = googlemaps.Client(key=google_key)
            # Verify the key actually works with a cheap test call
            test = gmaps_client.geocode("Cairo, Egypt")
            if not test:
                raise ValueError("Test geocode returned empty result")
            logger.info(f"✅ Google Maps API connected (test: Cairo → "
                        f"{test[0]['geometry']['location']})")
        except Exception as e:
            logger.error(f"Google Maps API failed to connect: {e}")
            logger.error("Check your GOOGLE_MAPS_API_KEY and that the Geocoding API "
                         "is enabled in Google Cloud Console.")
            sys.exit(1)
    else:
        logger.info("No GOOGLE_MAPS_API_KEY in .env — using Nominatim only")

    run_places       = not args.institutions_only
    run_institutions = not args.places_only

    try:
        with driver.session(database=neo4j_database) as session:

            # ── Historical places ────────────────────────────────────────────
            if run_places:
                logger.info("━━━ Geocoding historical Place nodes ━━━")

                if args.all:
                    place_query = ("MATCH (pl:Place) WHERE pl.place_type = 'historical' "
                                   "RETURN pl.name AS name, pl.coordinates AS coordinates, "
                                   "pl.name_variants AS name_variants, "
                                   "pl.data_sources AS data_sources ORDER BY pl.name")
                else:
                    place_query = ("MATCH (pl:Place) "
                                   "WHERE pl.place_type = 'historical' AND pl.lat IS NULL "
                                   "RETURN pl.name AS name, pl.coordinates AS coordinates, "
                                   "pl.name_variants AS name_variants, "
                                   "pl.data_sources AS data_sources ORDER BY pl.name")

                rows = [(r["name"], r["coordinates"], r["name_variants"], r["data_sources"] or [])
                        for r in session.run(place_query) if r["name"]]
                if args.limit:
                    rows = rows[:args.limit]

                logger.info(f"Found {len(rows)} Place node(s) to geocode")

                if args.dry_run:
                    for name, coords, _, data_sources in rows:
                        pgp_only = "pgp" in (data_sources or []) and len(set(data_sources or [])) == 1
                        if pgp_only and not coords:
                            src = "skip (PGP, no Princeton coords)"
                        elif coords:
                            src = "princeton coords"
                        else:
                            src = "geocoder"
                        print(f"  [{src}] {name}")
                else:
                    ok = 0
                    ok_princeton = 0
                    skipped_pgp = 0
                    failed = []
                    for name, princeton_coords, name_variants_str, data_sources in rows:

                        # ── Priority 1: parse Princeton's existing coordinates string ──
                        parsed = _parse_princeton_coords(princeton_coords) if princeton_coords else None
                        if parsed:
                            lat, lng = parsed
                            session.run("""
                                MATCH (pl:Place {name: $name})
                                SET pl.lat             = $lat,
                                    pl.lng             = $lng,
                                    pl.geocode_source  = 'princeton',
                                    pl.geocode_suspect = false
                            """, name=name, lat=lat, lng=lng)
                            ok += 1
                            ok_princeton += 1
                            logger.info(f"  📍 {name} → ({lat:.4f}, {lng:.4f}) [Princeton coords]")
                            continue

                        # ── Skip PGP-only places without Princeton coordinates ──
                        # Princeton deliberately omits coordinates for places they cannot
                        # reliably locate. Geocoding them produces false positives.
                        # Non-PGP places (extracted by LLM) are tried normally since they
                        # tend to use more recognisable modern names.
                        pgp_only = "pgp" in (data_sources or []) and \
                                   all(s == "pgp" for s in (data_sources or ["pgp"]))
                        if pgp_only:
                            skipped_pgp += 1
                            logger.debug(f"  ⏭ {name} — PGP place without Princeton coords, "
                                         f"skipping (likely unlocatable)")
                            continue

                        # Parse name_variants (comma-delimited Princeton string) into a list
                        # of extra queries to try if the canonical name fails.
                        variants = []
                        if name_variants_str:
                            variants = [v.strip() for v in name_variants_str.split(",")
                                        if v.strip() and v.strip() != name]

                        # ── Hebrew-script handling ───────────────────────────────────
                        # For Hebrew/Aramaic names: strip institution prefixes (ישיבת etc.),
                        # skip known non-places (גן עדן etc.), and build a transliterated
                        # fallback for Nominatim (which is weak on Hebrew script).
                        hebrew_nominatim_queries = None
                        if _is_hebrew_script(name):
                            hebrew_queries, is_heb_institution = _prepare_hebrew_queries(name)
                            if not hebrew_queries:
                                logger.info(f"  ⏭ {name} — Hebrew non-place concept, skipping")
                                continue
                            # Override the variant/clean-name query lists with Hebrew-aware ones
                            variants        = hebrew_queries[1:]  # transliterated forms as variants
                            clean_name      = hebrew_queries[0]   # Hebrew form (Google handles it)
                            # For Nominatim, prefer the transliterated form
                            hebrew_nominatim_queries = hebrew_queries
                            if is_heb_institution:
                                logger.info(f"  🔤 Hebrew institution prefix stripped: '{name}' → '{clean_name}'")
                        else:
                            # Non-Hebrew: clean parentheticals as before
                            clean_name, _ = _clean_place_name(name)

                        override_key  = name if name in HISTORICAL_OVERRIDES else (
                                        clean_name if clean_name in HISTORICAL_OVERRIDES else None)

                        # Build Google query list using cleaned/Hebrew-stripped name
                        google_queries = ([HISTORICAL_OVERRIDES[override_key]] if override_key
                                          else [clean_name] + variants[:3])
                        result = None

                        # ── Priority 2: Google (primary, when key available) ──
                        if gmaps_client:
                            display = (f"'{HISTORICAL_OVERRIDES[override_key]}' [override for '{name}']"
                                       if override_key else f"'{clean_name}'")
                            logger.info(f"  🌍 Trying Google: {display}")
                            result = _geocode_google(google_queries, gmaps_client)
                            if result:
                                suspect_flag = "⚠️ suspect" if result.get("geocode_suspect") else "✅"
                                logger.info(f"  {suspect_flag} {name} → "
                                            f"({result['lat']:.4f}, {result['lng']:.4f}) "
                                            f"{result['country']} [google]")
                            else:
                                logger.info(f"  ❌ Google found nothing for '{name}' — trying Nominatim")

                        # ── Priority 3: Nominatim — when Google unavailable/failed/suspect ──
                        use_nominatim = (
                            not result
                            or result.get("geocode_suspect", False)
                        )
                        if use_nominatim:
                            if result and result.get("geocode_suspect"):
                                logger.info(f"  🔄 Google result suspect — trying Nominatim for '{name}'")
                            # For Hebrew names, try transliterated queries with Nominatim
                            # since Nominatim's Hebrew support is weaker than Google's
                            nom_queries = hebrew_nominatim_queries or [name]
                            logger.info(f"  📡 Trying Nominatim: '{nom_queries[0]}'")
                            nom_result = _geocode_one_compat(geolocator, nom_queries[0])
                            if not nom_result:
                                for variant in (nom_queries[1:] + variants):
                                    nom_result = _geocode_one_compat(geolocator, variant)
                                    time.sleep(args.delay)
                                    if nom_result:
                                        logger.info(f"  📡 Nominatim resolved via '{variant}'")
                                        break
                            time.sleep(args.delay)

                            if nom_result and not nom_result.get("geocode_suspect", False):
                                result = nom_result
                                logger.info(f"  ✅ {name} → ({result['lat']:.4f}, {result['lng']:.4f}) "
                                            f"{result['country']} [nominatim]")
                            elif nom_result and not result:
                                result = nom_result
                                logger.info(f"  ⚠️ {name} → ({result['lat']:.4f}, {result['lng']:.4f}) "
                                            f"{result['country']} [nominatim — suspect]")
                            elif not nom_result and result and result.get("geocode_suspect"):
                                logger.info(f"  ⚠️ Keeping suspect Google result for '{name}' "
                                            f"— Nominatim also found nothing")

                        if result:
                            session.run("""
                                MATCH (pl:Place {name: $name})
                                SET pl.lat              = $lat,
                                    pl.lng              = $lng,
                                    pl.osm_display_name = $osm_display_name,
                                    pl.geocode_source   = $geocode_source,
                                    pl.geocode_suspect  = $geocode_suspect,
                                    pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL
                                                      THEN $city    ELSE pl.city    END,
                                    pl.country = CASE WHEN $country <> '' AND pl.country IS NULL
                                                      THEN $country ELSE pl.country END,
                                    pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL
                                                      THEN $region  ELSE pl.region  END
                            """, name=name, **result)
                            ok += 1
                        else:
                            failed.append(name)
                            logger.warning(f"  ❌ {name} — no result from Google or Nominatim")

                    print(f"\n✅ Places: geocoded {ok}/{len(rows)} "
                          f"({ok_princeton} from Princeton coords, "
                          f"{ok - ok_princeton} via geocoder, "
                          f"{skipped_pgp} PGP-only skipped as unlocatable)")
                    if failed:
                        print(f"❌ Failed ({len(failed)}):")
                        for f in failed:
                            print(f"  • {f}")

            # ── Institutions ─────────────────────────────────────────────────
            if run_institutions:
                if run_places and not args.dry_run:
                    print()   # blank line between the two sections
                logger.info("━━━ Geocoding Institution nodes ━━━")
                _run_institution_geocoding(session, geolocator, args, gmaps_client)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
