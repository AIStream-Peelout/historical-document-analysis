"""
Curated coordinate corrections for Place and Institution nodes.

``geocode_places.py`` writes these at the start of every run (also to nodes
that already have a pin) and never geocodes a corrected node, so a fix made
here survives re-geocoding, ``--all`` and a rebuild of the graph from empty. Each entry was checked by
hand against the Princeton Geniza Project record, the documents linked to the
node, and a gazetteer (Wikipedia / Pleiades / the institution's own site); the
full audit with evidence is ``docs/geocode_audit_2026-10-18.tsv``.

A correction with coordinates replaces the node's location fields and records
``geocode_source = 'manual'``. A correction without coordinates
(:func:`excluded`) removes the pin: the node is not a locatable place or
institution (a military corps, a publisher's place string, a placeholder
coordinate) and gets ``geocode_source = 'excluded'`` so it stays off the map.

Node names must match ``name`` exactly (they are the Neo4j merge keys).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GeocodeCorrection:
    """A curated location for one node, or the decision that it has none.

    :param note: Why the correction exists (what was wrong, what the pin means).
    :param lat: Corrected latitude, or ``None`` to remove the pin.
    :param lng: Corrected longitude, or ``None`` to remove the pin.
    :param country: Modern country of the corrected pin ('' when the node is a region).
    :param city: City of the corrected pin ('' when not meaningful).
    :param region: Broad region; '' derives it from ``country``.
    """

    note: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    country: str = ""
    city: str = ""
    region: str = ""

    @property
    def is_excluded(self) -> bool:
        """True when the correction removes the pin instead of moving it."""
        return self.lat is None


def pin(lat: float, lng: float, country: str, city: str, note: str, region: str = "") -> GeocodeCorrection:
    """Build a correction that moves a node to a curated location.

    :param lat: Latitude in decimal degrees.
    :param lng: Longitude in decimal degrees.
    :param country: Modern country ('' for a region spanning several).
    :param city: City or locality ('' when not meaningful).
    :param note: Why the correction exists.
    :param region: Broad region; '' derives it from ``country``.
    :return: The correction.
    """
    return GeocodeCorrection(note=note, lat=lat, lng=lng, country=country, city=city, region=region)


def excluded(note: str) -> GeocodeCorrection:
    """Build a correction that removes a node's pin.

    :param note: Why the node has no valid location.
    :return: The correction.
    """
    return GeocodeCorrection(note=note)


# ---------------------------------------------------------------------------
# Historical places
# ---------------------------------------------------------------------------
PLACE_CORRECTIONS: dict[str, GeocodeCorrection] = {
    # Bare-name geocoder hits on PGP places that Princeton left unlocated.
    "Goshen": pin(30.42, 31.5597, "Egypt", "", note=(
        "Biblical Land of Goshen, eastern Nile Delta. Google had matched Goshen, Indiana. "
        "PGP gives no coordinates; its documents place Bilbays 'adjacent to the Land of Goshen' "
        "(ENA NS 13.34, BL Or 10126(1).6), so the pin is anchored at Bilbays.")),
    "al-Bannā": pin(30.7297, 31.7972, "Egypt", "Fāqūs", note=(
        "PGP: 'Unidentified locality near Fāqūs'. Google had matched a hamlet in Kafr el-Sheikh; "
        "pinned at Fāqūs.")),
    "al-Maghrib": pin(30.0, 5.0, "", "", region="North Africa", note=(
        "The Islamic West (Maghreb) as a region. Google had matched Hay Al Gharb, Sharjah (UAE); "
        "pinned at the Wikidata Maghreb centroid (Q28227).")),
    "גומא מזוד": excluded(
        "PGP: 'Unidentified location likely in Iraq'. Google had matched Migdal on the Sea of Galilee."),
    "Qaṣr al-Shamʿ (neighborhood)": pin(30.0053, 31.2327, "Egypt", "Cairo", note=(
        "Babylon fortress quarter of Fustat (Coptic Cairo). The pin was right but flagged suspect "
        "as a neighbourhood, which hid 77 fragments from the map.")),

    # Region nodes PGP leaves without coordinates; pinned so a rebuild reproduces them.
    "Egypt (region)": pin(26.8206, 30.8025, "Egypt", "", note="Country-level pin for the Egypt region node."),
    "India": pin(20.5937, 78.9629, "India", "", note="Country-level pin for the India region node."),
    "Yemen": pin(15.5527, 48.5164, "Yemen", "", note="Country-level pin for the Yemen region node."),

    # Princeton coordinate strings that are wrong at the source.
    "London": pin(51.5097, -0.1181, "United Kingdom", "London", note=(
        "PGP string reads 0° 7′ 5″ E (east London); London is west of Greenwich.")),
    "Valencia": pin(39.4694, -0.3772, "Spain", "Valencia", note=(
        "PGP string reads 0° 22′ 38″ E, which is in the sea; Valencia is at 0.38° W.")),
    "Darmus": pin(23.5691, 32.8414, "Egypt", "Kalabsha", note=(
        "PGP equates Darmus with Kalabsha/Talmis in Lower Nubia; its 24° 39′ N latitude lands at "
        "Gebel el-Silsila, about 1° too far north. Pinned at Talmis (Pleiades 795868).")),
    "Kalabshā": pin(23.554, 32.865, "Egypt", "Kalabsha", note=(
        "PGP pins New Kalabsha, where the temple was moved in the 1960s; the medieval town "
        "(T-S 10J17.23) stood at the original west-bank site, now under Lake Nasser.")),
    "Dongola": pin(18.2231, 30.7439, "Sudan", "Old Dongola", note=(
        "PGP pins modern Dongola, founded after 1812; the medieval documents mean Dunqula, "
        "the Makurian capital (Old Dongola).")),
    "al-Qulzum": pin(29.9546, 32.574, "Egypt", "Suez", note=(
        "PGP pins a homonymous Qalyubiyya village; the linked documents mean the Red Sea port "
        "(Clysma) at Suez.")),
    "Romania": pin(41.0064, 28.9758, "Turkey", "Istanbul", note=(
        "Hebrew רומנייה in Bodl. MS heb. a 3/27 means Byzantine Rūm, not modern Romania; "
        "pinned at Constantinople.")),
    "Nizāla": excluded("PGP: 'Unknown location; please disregard coordinates.'"),
}


# ---------------------------------------------------------------------------
# Institutions
# ---------------------------------------------------------------------------
_NOT_AN_INSTITUTION = "Not an institution (extraction error); had a bare-name geocoder pin."

INSTITUTION_CORRECTIONS: dict[str, GeocodeCorrection] = {
    # Homonym cities and wrong campuses. Duplicate nodes merged by
    # consolidate_institutions.py ("St. Petersburg, Firkovich collections",
    # "Adler collection", ...) are corrected through their canonical node.
    "Russian National Library, St. Petersburg": pin(59.9335, 30.3349, "Russia", "Saint Petersburg", note=(
        "Holds the Firkovich collections (once pinned in St. Petersburg, Florida); "
        "was the St. Petersburg city centroid.")),
    "Institute of Oriental Manuscripts, Russian Academy of Sciences": pin(
        59.9439, 30.3219, "Russia", "Saint Petersburg", note="Was the RAS Presidium in Moscow."),
    "Westminster College library collection": pin(52.2104, 0.1119, "United Kingdom", "Cambridge", note=(
        "Westminster College, Cambridge (Lewis-Gibson Genizah). Google had matched New Wilmington, PA.")),
    "St. John's College": pin(52.2081, 0.1167, "United Kingdom", "Cambridge", note=(
        "St John's College, Cambridge. Google had matched Annapolis, Maryland.")),
    "Christ's College": pin(52.2062, 0.1227, "United Kingdom", "Cambridge", note=(
        "Christ's College, Cambridge. Google had matched Christchurch, New Zealand.")),
    "Faculty of Asian and Middle Eastern Studies": pin(52.2009, 0.1109, "United Kingdom", "Cambridge", note=(
        "Cambridge faculty. Google had matched Oxford's Oriental Faculty.")),
    "Harvard": pin(42.3741, -71.1167, "United States", "Cambridge", note=(
        "Harvard University. Google had matched the town of Harvard, Massachusetts.")),
    "Harvard University Center for Jewish Studies": pin(42.3781, -71.1139, "United States", "Cambridge", note=(
        "Was the town of Harvard, Massachusetts.")),
    "Beeghly Library, University of Heidelberg": pin(49.4125, 8.7046, "Germany", "Heidelberg", note=(
        "Holds Heidelberg papyri (p. Heid. Hebr./Arab.) and Cod. Heid. Or.; Google had matched "
        "Heidelberg University's Beeghly Library in Tiffin, Ohio.")),
    "Sassoon Library": pin(51.9795, -0.2295, "United Kingdom", "Letchworth", note=(
        "D. S. Sassoon's library (Ohel David), Letchworth. Google had matched the David Sassoon "
        "Library in Mumbai.")),
    "Mandel School": pin(31.7943, 35.2429, "Israel", "Jerusalem", note=(
        "Hebrew University's Mandel School for Advanced Studies in the Humanities. Google had "
        "matched the Mandel School in Cleveland.")),
    "Department of Anthropology": pin(41.7883, -87.6001, "United States", "Chicago", note=(
        "University of Chicago (per the edge evidence). Google had matched Texas State University.")),
    "Paris, BNF": pin(48.8673, 2.3382, "France", "Paris", note=(
        "BnF Richelieu (manuscripts); was the Tolbiac site.")),
    "Consistoire central de Paris": pin(48.8798, 2.3325, "France", "Paris", note=(
        "Fragment on deposit at the AIU library, 45 rue La Bruyère; was the Centre Européen du Judaïsme.")),
    "Paris Consistoire": pin(48.8758, 2.3371, "France", "Paris", note="Was the Paris city centroid."),
    "University of Pennsylvania, Katz Center": pin(39.9473, -75.1485, "United States", "Philadelphia", note=(
        "Katz Center, 420 Walnut St; was the West Philadelphia campus centroid.")),
    "Near Eastern Center of Los Angeles": pin(34.0699, -118.4438, "United States", "Los Angeles", note=(
        "UCLA's Near Eastern Center; was the Los Angeles centroid.")),
    "Ohio State's Department of History": pin(40.0061, -83.0283, "United States", "Columbus", note=(
        "Ohio State University; was the Ohio state centroid.")),

    # Country or city centroids for institutions with a known seat.
    "British Academy": pin(51.5063, -0.1312, "United Kingdom", "London", note="Was the UK centroid."),
    "Egypt Exploration Fund": pin(51.5234, -0.1173, "United Kingdom", "London", note="Was the Egypt centroid."),
    "Egyptian Department of Antiquities": pin(30.048, 31.2336, "Egypt", "Cairo", note="Was the Egypt centroid."),
    "Institute for Jewish History in Austria": pin(48.2025, 15.6287, "Austria", "St. Pölten", note=(
        "Injoest, St. Pölten; was the Austria centroid.")),
    "Royal Library (Det KGL Biblitek) of Denmark": pin(55.6733, 12.5828, "Denmark", "Copenhagen", note=(
        "Was the Denmark centroid.")),
    "Romanian Academy of Sciences": pin(44.4452, 26.0895, "Romania", "Bucharest", note="Was the Romania centroid."),
    "St. Catherine's Monastery at Mount Sinai": pin(28.5556, 33.9761, "Egypt", "Saint Catherine", note=(
        "The monastery; was the Jabal Musa summit.")),
    "Palestinian Congregation": pin(30.0058, 31.231, "Egypt", "Cairo", note=(
        "Congregation of the Palestinians in Fustat; was the Palestine centroid.")),
    "Yeshiva in Egypt": pin(30.0047, 31.2375, "Egypt", "Cairo", note="Fustat; was the Egypt centroid."),

    # Academies of the geonic period (Hebrew names geocoded to modern Israeli addresses).
    "Palestinian Yeshiva": pin(31.7767, 35.2344, "", "Jerusalem", region="Middle East", note=(
        "Yeshiva of Eretz Israel, Jerusalem in the 11th century; was the Palestine centroid.")),
    "מתא מחסיה": pin(31.8831, 44.45, "Iraq", "Sura", note=(
        "Mata Mehasya (Sura academy). Google had matched Moshav Mahseya near Beit Shemesh.")),
    "מתיבתא דמתא מחסיה": pin(31.8831, 44.45, "Iraq", "Sura", note=(
        "Academy of Mata Mehasya (Sura). Google had matched Moshav Mahseya near Beit Shemesh.")),
    "גאון יעקב במתא מחסיא": pin(31.8831, 44.45, "Iraq", "Sura", note=(
        "Gaon of Sura. Google had matched a modern yeshiva in Bnei Brak.")),
    "גאון יעקוב": pin(33.3508, 43.7858, "Iraq", "Pumbedita", note=(
        "Ge'on Ya'aqov, title of the Pumbedita academy. Google had matched Bnei Brak.")),
    "ביתא": pin(33.3508, 43.7858, "Iraq", "Pumbedita", note=(
        "'The house' (academy of Pumbedita). Google had matched Beita in the West Bank.")),
    "בית הדין 'דמרנו ורבנו סעדיה'": pin(33.3153, 44.3661, "Iraq", "Baghdad", note=(
        "Court of Saadia Gaon, Baghdad. Google had matched a Tel Aviv street address.")),
    "מרכז שייסד משה אבי חנוך": pin(37.89, -4.78, "Spain", "Córdoba", note=(
        "Moses b. Ḥanokh's centre in Córdoba. Google had geocoded the word 'מרכז' (Israel's Center District).")),
    "המרכז לחקר בין-תחומי של הגניזה הקהירית": pin(32.7614, 35.0195, "Israel", "Haifa", note=(
        "University of Haifa Genizah research centre; was the Center District centroid.")),

    # Not institutions, or no single location.
    "Private Collection": excluded("Generic label. Google had matched an address in Madison, Mississippi."),
    "Alain Jacquet (Private)": excluded("Private owner. Google had matched 'Alain' to Al Ain, UAE."),
    "Cairo's Janissary corps": excluded("Ottoman military corps. Google had matched Kalamazoo, Michigan."),
    "Çavuşān regiment": excluded("Ottoman military corps. Google had matched Siverek, Türkiye."),
    "Kazdağlı household": excluded("Ottoman-Egyptian household. Google had matched an Ankara address."),
    "Academy's Board of seven": excluded("Office of an academy. Google had matched Hattiesburg, Mississippi."),
    "Berkely, Los Angeles and London": excluded("Publisher's place string, not an institution."),
    "University Press, Cambridge": pin(52.2016, 0.1189, "United Kingdom", "Cambridge", note=(
        "Cambridge University Press (printer W. Lewis, Gottheil & Worrell's Freer volume). "
        "Google had matched Cambridge, Massachusetts.")),
    "la compagnie d'Alam ad-daula": excluded("Historical company. Google had matched Los Angeles."),
    "Friedberg and Safra Foundations": excluded("Two funding bodies. Google had matched Friedberg, Hesse."),
    "Genazim Project": excluded("Friedberg project unit with no physical seat. Google had matched Monsey, NY."),
    "Europe": excluded("Continent, not an institution."),
    "Egyptian Jewry": excluded("Community, not an institution."),
    "Ayyubids of Egypt": excluded("Dynasty, not an institution."),
    "Gaon": excluded("Title, not an institution. Google had matched Gaon, Verona."),
    "Qāḍī al-Qudāt": excluded("Office (chief judge). Google had matched Qadi, KwaZulu-Natal."),
    "Islamic and South Asian": excluded("Catalogue section label. Google had matched the South Asia centroid."),
    "land-tax office": excluded("Office, not a located institution. Google had matched West Bengal."),
    "Ṭālibīs": excluded("Family group (descendants of Abū Ṭālib). Google had matched Guelmim, Morocco."),
    "Trabiz": excluded("A place (Tabriz), not an institution."),
    "אבי נשיאה": excluded(_NOT_AN_INSTITUTION + " (Aby, Lincolnshire)"),
    "אין ברירה": excluded(_NOT_AN_INSTITUTION + " ('no choice'; matched the river Inn)"),
    "מועצה (מַגְ'לִס)": excluded(_NOT_AN_INSTITUTION + " ('council'; matched the Jezreel Valley)"),
    "רכוש העדה המוסלמית ('צאפיה')": excluded(_NOT_AN_INSTITUTION + " ('Muslim community property'; matched Ghawr as-Safi)"),
    "גני שריון 33": excluded(_NOT_AN_INSTITUTION + " (matched Gagny, France)"),
}
