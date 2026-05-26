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
    "Tripoli (Libya)":"Tripoli, Libya",
    "Tripoli (Lebanon)":"Tripoli, Lebanon",
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

def _strip_diacritics(s: str) -> str:
    return _ud.normalize("NFKD", s).encode("ascii", "ignore").decode().strip()

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

    # 1. Historical override
    if name in HISTORICAL_OVERRIDES:
        loc = _try(HISTORICAL_OVERRIDES[name])
        if loc:
            return _build_result(loc)

    clean, hint = _clean_place_name(name)

    # 2. Name as-is (if no parens)
    if clean == name:
        loc = _try(name)
        if loc:
            return _build_result(loc)
    else:
        # 3. Clean name + hint as geographic context
        if hint and not hint.lower().startswith(("region", "area", "province")):
            loc = _try(f"{clean}, {hint}")
            if loc:
                return _build_result(loc)

        # 4. Cleaned name alone
        loc = _try(clean)
        if loc:
            return _build_result(loc)

    # 5. ASCII-stripped fallback
    ascii_name = _strip_diacritics(clean)
    if ascii_name and ascii_name != clean:
        _time.sleep(0.5)   # extra courtesy delay for extra request
        loc = _try(ascii_name)
        if loc:
            return _build_result(loc)

    return None


def _build_result(loc) -> dict:
    """Extract a clean result dict from a geopy Location."""
    addr    = loc.raw.get("address", {})
    country = addr.get("country") or addr.get("state") or ""
    city    = (addr.get("city") or addr.get("town") or
               addr.get("village") or addr.get("county") or "")
    region  = COUNTRY_TO_REGION.get(country, "")
    return {
        "lat":              loc.latitude,
        "lng":              loc.longitude,
        "city":             city,
        "country":          country,
        "region":           region,
        "osm_display_name": loc.address,
    }


def _geocode_one_compat(geolocator, name: str) -> Optional[dict]:
    """Error-catching wrapper around _geocode_one."""
    try:
        return _geocode_one(geolocator, name)
    except Exception as e:
        logger.warning(f"  Geocoding '{name}' failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Geocode Place nodes in Neo4j.")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be geocoded without writing.")
    parser.add_argument("--all", action="store_true",
                        help="Re-geocode nodes that already have lat/lng.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of places to geocode.")
    parser.add_argument("--delay", type=float, default=1.1,
                        help="Seconds between Nominatim requests (default 1.1 — "
                             "Nominatim policy is max 1 req/sec).")
    args = parser.parse_args()

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

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    geolocator = Nominatim(user_agent="cairo_genizah_kg_geocoder/1.0", ssl_context=ssl_ctx)

    try:
        with driver.session(database=neo4j_database) as session:

            # Fetch places to geocode
            if args.all:
                query = "MATCH (pl:Place) RETURN pl.name AS name ORDER BY pl.name"
            else:
                query = ("MATCH (pl:Place) WHERE pl.lat IS NULL "
                         "RETURN pl.name AS name ORDER BY pl.name")

            places = [r["name"] for r in session.run(query) if r["name"]]
            if args.limit:
                places = places[:args.limit]

            logger.info(f"Found {len(places)} Place node(s) to geocode")

            if args.dry_run:
                for name in places:
                    print(f"  Would geocode: {name}")
                return

            ok = 0
            failed = []
            for name in places:
                result = _geocode_one_compat(geolocator, name)
                time.sleep(args.delay)   # respect Nominatim rate limit

                if result:
                    session.run("""
                        MATCH (pl:Place {name: $name})
                        SET pl.lat              = $lat,
                            pl.lng              = $lng,
                            pl.osm_display_name = $osm_display_name,
                            pl.city    = CASE WHEN $city    <> '' AND pl.city    IS NULL
                                              THEN $city    ELSE pl.city    END,
                            pl.country = CASE WHEN $country <> '' AND pl.country IS NULL
                                              THEN $country ELSE pl.country END,
                            pl.region  = CASE WHEN $region  <> '' AND pl.region  IS NULL
                                              THEN $region  ELSE pl.region  END
                    """, name=name, **result)
                    ok += 1
                    logger.info(f"  ✅ {name} → ({result['lat']:.4f}, {result['lng']:.4f}) {result['country']}")
                else:
                    failed.append(name)
                    logger.warning(f"  ❌ {name} — no geocoding result")

            print(f"\n✅ Geocoded {ok}/{len(places)} places")
            if failed:
                print(f"❌ Failed ({len(failed)}):")
                for f in failed:
                    print(f"  • {f}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
