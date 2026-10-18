"""
Curated duplicate merges and edge removals for Institution and Place nodes.

``consolidate_institutions.py`` applies these after the importers (import step 4b,
before ``geocode_places.py``). Every rule is keyed on node names as the importers
write them, so a rebuild of the graph from empty re-creates the duplicates and the
pass cleans them again; a second run on a clean graph changes nothing. The rules
come from the geocode audit (``docs/geocode_audit_2026-10-18.tsv``, rows with
``in_corrections = no``) and a read-only census of all 348 Institution nodes on
2026-10-18; each one was checked against the edge evidence in the graph.

* :data:`INSTITUTION_MERGES` maps a duplicate node name to its canonical node. The
  duplicate's edges move to the canonical (AFFILIATED_WITH edges can be routed to a
  different node, e.g. a university instead of its library), parallel edges are folded
  with the genizah_search edge-merge policy, and the duplicate is deleted.
  ``normalize=True`` names are also mapped by ``InstitutionNormalizer`` at import
  time, so they stop being created at all.
* :data:`EDGE_REMOVALS` deletes edges that are not holdings or geography (provenance
  "from the Cairo Genizah", figure captions, liturgical allusions, collector-of-a-
  collection read as an affiliation). Institution nodes they leave without edges
  are deleted.
* :data:`PLACE_MERGES` and :data:`EDGE_MOVES` fix Place duplicates and edges on the
  wrong homonym.

Node names must match ``name`` exactly (they are the Neo4j merge keys); Fragment
start nodes are matched on ``canonical_shelfmark``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: ``InstitutionMerge.affiliated_with`` value that leaves AFFILIATED_WITH edges on the duplicate.
KEEP_AFFILIATIONS = "<keep>"


@dataclass(frozen=True)
class InstitutionMerge:
    """Where a duplicate Institution node's edges go.

    :param canonical: Name of the node that survives.
    :param note: Why the two nodes are the same institution.
    :param affiliated_with: Target for AFFILIATED_WITH edges: '' for the canonical, another
        Institution name, or :data:`KEEP_AFFILIATIONS` to leave them on the duplicate (which is
        then only deleted once it has no edges left).
    :param normalize: True when ``InstitutionNormalizer`` should map this name to the canonical
        at import time. False for names that need the pass: bare city names (their edge types go
        to different nodes) and names that carry an :data:`EDGE_REMOVALS` rule.
    """

    canonical: str
    note: str
    affiliated_with: str = ""
    normalize: bool = True

    @property
    def affiliation_target(self) -> str:
        """Name of the node AFFILIATED_WITH edges move to (or :data:`KEEP_AFFILIATIONS`)."""
        return self.affiliated_with or self.canonical


@dataclass(frozen=True)
class EdgeRemoval:
    """A curated edge deletion.

    An edge matches when its type is ``rel_type``, its end node has label ``end_label`` and
    name ``end_name``, its start node's key equals ``start_key`` ('' matches any start node) and,
    when ``source_book`` is set, that book is in the edge's ``source_books``.

    :param rel_type: Relationship type.
    :param end_label: Label of the end node (Institution or Place).
    :param end_name: ``name`` of the end node.
    :param note: Why the edge is wrong.
    :param start_key: ``canonical_shelfmark`` of a Fragment or ``name`` of any other start node.
    :param source_book: Only edges extracted from this book.
    """

    rel_type: str
    end_label: str
    end_name: str
    note: str
    start_key: str = ""
    source_book: str = ""


@dataclass(frozen=True)
class EdgeMove:
    """Move one edge's end from a node to another node of the same label.

    :param rel_type: Relationship type.
    :param end_label: Label of both end nodes.
    :param from_name: Current end node name.
    :param to_name: New end node name.
    :param start_key: Key of the start node (see :class:`EdgeRemoval`).
    :param note: Why the edge belongs on the other node.
    """

    rel_type: str
    end_label: str
    from_name: str
    to_name: str
    start_key: str
    note: str


def merge(canonical: str, note: str, affiliated_with: str = "", normalize: bool = True) -> InstitutionMerge:
    """Build an :class:`InstitutionMerge`.

    :param canonical: Name of the surviving node.
    :param note: Why the nodes are the same institution.
    :param affiliated_with: AFFILIATED_WITH target ('' for the canonical).
    :param normalize: Whether the import-time normalizer maps the name too.
    :return: The merge rule.
    """
    return InstitutionMerge(canonical=canonical, note=note, affiliated_with=affiliated_with, normalize=normalize)


# ---------------------------------------------------------------------------
# Institutions
# ---------------------------------------------------------------------------
CUL = "Cambridge University Library"
JTS = "Jewish Theological Seminary"
BODLEIAN = "Bodleian Library, Oxford"
SBB = "Staatsbibliothek zu Berlin"
RNL = "Russian National Library, St. Petersburg"
KATZ = "University of Pennsylvania, Katz Center"
HUJI = "Hebrew University of Jerusalem"
PALESTINIAN_YESHIVA = "Palestinian Yeshiva"
IMHM = "Institute of Microfilmed Hebrew Manuscripts, Jewish National and University Library, Jerusalem"

_COLLECTOR_EDGE = "the edge records a collector or a find in the collection, not an affiliation (see EDGE_REMOVALS)"

INSTITUTION_MERGES: dict[str, InstitutionMerge] = {
    # Cambridge University Library: city names, collection names and punctuation variants.
    "Cambridge": merge(CUL, (
        "Bare city name. Its HELD_AT edges are T-S and 'Camb.' shelfmarks; its affiliations "
        "(Schechter, the librarian Jenkinson) are to the university."),
        affiliated_with="University of Cambridge", normalize=False),
    "Cambridge Genizah": merge(CUL, "Only holding is T-S K10.4, a Taylor-Schechter shelfmark."),
    "Cambridge Genizah Collections": merge(CUL, "Only holding is the Taylor-Schechter box T-S A45."),
    "Cambridge Genizah collection": merge(CUL, "Holds T-S Ar. 1a.54; " + _COLLECTOR_EDGE + ".",
                                          normalize=False),
    "Cambridge, University Library": merge(CUL, "Punctuation variant; 17 T-S shelfmarks. " + _COLLECTOR_EDGE + ".",
                                           normalize=False),
    "University Library, Cambridge": merge(CUL, "Name variant with the same West Road pin; holds T-S 13J25."),
    "University Library, Cambridge, England": merge(CUL, (
        "Name variant (Goitein: 'Collections of Geniza material in the library other than the "
        "Taylor-Schechter Collection'); pinned 280 m off.")),
    "University Library in Cambridge": merge(CUL, "The Manuscript Reading Room of the University Library."),
    "Cambridge University Libraries": merge(CUL, "Plural form (Ben Outhwaite of Cambridge University Libraries)."),
    "T-S": merge(CUL, "Taylor-Schechter abbreviation used as a library name (T-S 16.94)."),
    "T-S Genizah collection": merge(CUL, "Taylor-Schechter collection; " + _COLLECTOR_EDGE + ".", normalize=False),
    "אוסף TS": merge(CUL, "'The TS collection' (T-S 8J1)."),

    # Jewish Theological Seminary: the Adler and Schechter collections and Hebrew names.
    "Adler collection": merge(JTS, "E. N. Adler collection (ENA 4096d); was pinned at Adler, Russia."),
    "Elkan Adler collection": merge(JTS, "E. N. Adler collection; " + _COLLECTOR_EDGE + ".", normalize=False),
    "Elkan Nathan Adler Genizah collection": merge(JTS, "E. N. Adler collection; " + _COLLECTOR_EDGE + ".",
                                                   normalize=False),
    "אוסף אדלר": merge(JTS, "'The Adler collection'."),
    "אוסף שכטר": merge(JTS, "'The Schechter collection at the rabbinical seminary in New York' (malk-ish-1-400)."),
    "הסימנר בנוירק": merge(JTS, "'The seminary in New York'."),
    "בית המדרש לרבנים בניו יורק": merge(JTS, "Hebrew name of JTS (ENA NS 35.15)."),

    # Bodleian, British Library, Staatsbibliothek: city strings and citation abbreviations.
    "Oxford": merge(BODLEIAN, "Bare city name; all four holdings are 'Oxford, Bodl.' shelfmarks.",
                    affiliated_with="University of Oxford", normalize=False),
    "Oxford, Bodl.": merge(BODLEIAN, "Citation abbreviation for the Bodleian (Opp., Mich., Poc. shelfmarks)."),
    "Bodl.": merge(BODLEIAN, "Abbreviation (Bodl. MS heb. b 11/16)."),
    "ספרייה הבריטית": merge("British Library, London", "Hebrew name of the British Library."),
    "BL": merge("British Library, London", "Abbreviation (London, BL Or 2674 and Add 15977)."),
    "Berlin": merge(SBB, "Bare city name; both holdings are 'Berlin, SBB, Or.' shelfmarks.",
                    affiliated_with=KEEP_AFFILIATIONS, normalize=False),
    "Berlin, SBB": merge(SBB, "Citation abbreviation for the Staatsbibliothek (Or. Oct. 3150)."),
    "Königliche Bibliothek in Berlin": merge(SBB, (
        "Predecessor of the Staatsbibliothek; Ms. or. fol. 1233 was bought from Shapira in 1881.")),

    # Holding libraries with a second name (three come from the PGP CSVs).
    "National Library of Russia": merge(RNL, "PGP's other name for the same library (Firkovich manuscripts)."),
    "St. Petersburg, Firkovich collections": merge(RNL, "Firkovich collections of the National Library of Russia."),
    "Firkovitch collection": merge(RNL, "Firkovich collection (II A 333)."),
    "Jewish National and University Library": merge("National Library of Israel", "Pre-2008 name of the NLI."),
    "הספרייה הלאומית": merge("National Library of Israel", "Hebrew name of the NLI."),
    "Katz Center for Advanced Judaic Studies, University of Pennsylvania": merge(KATZ, "PGP's name for the Katz Center."),
    "Herbert D. Katz Center for Advanced Judaic Studies": merge(KATZ, "Full name of the Katz Center (Halper 331)."),
    "University of Pennsylvania Library": merge(KATZ, "Its holdings are Halper fragments, kept at the Katz Center."),
    "Ben Zvi Institute": merge("Yad Izhak Ben-Zvi Institute", "PGP's name for the Ben-Zvi Institute."),
    "University of Manchester Library": merge("John Rylands Library, University of Manchester", (
        "The Genizah holdings are in the John Rylands Library, part of the University of Manchester Library.")),
    "Toronto, University of Toronto Libraries": merge("Thomas Fisher Rare Book Library, University of Toronto", (
        "Friedberg fragment (Fr 5-014), kept at the Fisher library.")),
    "OIM": merge("Oriental Institute Museum", "Oriental Institute Museum (Chicago) registration prefix (OIM E8349)."),
    "Oxford Institute for Papyrology": merge("Oriental Institute Museum", (
        "LLM expansion of 'OIM': its one holding is OIM E8349 (P. 65877/N. 44651), an Oriental Institute "
        "Museum number.")),
    "Jüdischen Gemeindebibliothek Berlin": merge("Jüdische Gemeindebibliothek (Berlin)", (
        "Genitive form from the same Schirmann 1932 title.")),
    "L. A. Mayer Museum of Islamic Art": merge("L. A. Mayer Museum for Islamic Art", "Name variant; same pin."),
    "Musée de l'Art Arabe": merge("Museum of Islamic Art in Cairo", "Its name before 1952."),

    # Universities and projects (affiliations only).
    "The Hebrew University": merge(HUJI, "Name variant; same pin."),
    "The Hebrew University, Jerusalem": merge(HUJI, "Name variant; same pin."),
    "האוניברסיטה העברית": merge(HUJI, "Hebrew name."),
    "האוניברסיטה העברית בירושלים": merge(HUJI, "Hebrew name."),
    "Tel-Aviv University": merge("Tel Aviv University", "Spelling variant; same pin."),
    "University of Tel Aviv": merge("Tel Aviv University", "Name variant; same pin."),
    "UNIVERSITY COLLEGE LONDON": merge("University College London", "All-caps variant."),
    "UCL": merge("University College London", "Abbreviation (the Gaster papers at UCL)."),
    "אוניברסיטת חיפה": merge("University of Haifa", "Hebrew name."),
    "Bar Ilan": merge("Bar-Ilan University", "Short name; same pin."),
    "Ankara Üniversitesi Ilahiyat Fakültesi Yayınları": merge("Ankara Üniversitesi Ilahiyat Fakültesi", (
        "'Yayınları' = the faculty's publication series.")),
    "Princeton Geniza Project database": merge("Princeton Geniza Project", "The project's database."),
    "Faculty of Asian and Middle Eastern Studies Degree Committee": merge(
        "Faculty of Asian and Middle Eastern Studies", "Cambridge thesis boilerplate."),
    "Harvard University Center for Jewish S": merge("Harvard University Center for Jewish Studies",
                                                    "Truncated name."),

    # The Palestinian Yeshiva (Yeshiva of Eretz Israel) under its English and Hebrew names.
    "Palestinian Academy": merge(PALESTINIAN_YESHIVA, "Gottheil & Worrell's name for the same academy."),
    "Yeshiva of Eretz Israel": merge(PALESTINIAN_YESHIVA, "Same academy."),
    "ישיבת ארץ ישראל": merge(PALESTINIAN_YESHIVA, "Hebrew name of the same academy."),
    "אב בית דין בישיבת ארץ ישראל": merge(PALESTINIAN_YESHIVA, (
        "Title of Nathan b. Abraham ('av bet din of the Yeshiva of Eretz Israel'), not a separate institution.")),
}


# ---------------------------------------------------------------------------
# Edges that are not holdings or geography
# ---------------------------------------------------------------------------
def _not_a_holding(name: str, note: str) -> EdgeRemoval:
    """Remove every HELD_AT edge into ``name``.

    :param name: Institution name that never holds fragments.
    :param note: Why its HELD_AT edges are wrong.
    :return: The removal rule.
    """
    return EdgeRemoval("HELD_AT", "Institution", name, note)


def _held_at(start_key: str, name: str, note: str) -> EdgeRemoval:
    """Remove the HELD_AT edge from one Fragment to ``name``.

    :param start_key: Fragment ``canonical_shelfmark``.
    :param name: Institution name.
    :param note: Why the edge is wrong.
    :return: The removal rule.
    """
    return EdgeRemoval("HELD_AT", "Institution", name, note, start_key=start_key)


def _collector(scholar: str, collection: str) -> EdgeRemoval:
    """Remove an AFFILIATED_WITH edge that links a collector or finder to a collection.

    :param scholar: Person or Scholar name.
    :param collection: Institution name of the collection.
    :return: The removal rule.
    """
    return EdgeRemoval("AFFILIATED_WITH", "Institution", collection, (
        "Collector of the collection or found fragments in it; not an affiliation."), start_key=scholar)


_PROVENANCE = "Provenance ('from the Cairo Genizah'), not the current holder."
_HYMN = "Liturgical allusion in a penitential hymn (Gottheil & Worrell no. XVII), not a geographic link."

EDGE_REMOVALS: tuple[EdgeRemoval, ...] = (
    _not_a_holding("Cairo Genizah", _PROVENANCE),
    _not_a_holding("Cairo Geniza", _PROVENANCE),
    _not_a_holding("Geniza", _PROVENANCE),
    _not_a_holding("קהיר", (
        "'Cairo' as a holder: 14 of its 15 edges come from one sentence about a court in Cairo (6david).")),
    _held_at("Maestà_altarpiece", "Cathedral of Siena", "Art-history figure caption (Duccio's Maestà)."),
    _held_at("VI_C_6", "Palestinian synagogue", "Where the elders met (event location), not a holding."),
    _held_at("parallel_manuscript_of_the_siddur", "Leningrad", (
        "'A copy of which is found in Leningrad': not a fragment, a mention of another manuscript.")),
    _held_at("ENA_3012", "Paris Nat. Bibl.", "ENA is a JTS shelfmark; the sentence is about another manuscript."),
    _held_at("Mosseri_Ia_26_2", IMHM, "A microfilm the author consulted; the fragment is at CUL."),
    EdgeRemoval("HELD_AT", "Institution", "אוסף Dropsie", (
        "One sentence ('the Dropsie collection where a copy of the deed is found') attached to every "
        "shelfmark on the page; T-S, ENA and Bodl. fragments are not at Dropsie."),
        source_book="malk-ish-1-155"),
    EdgeRemoval("MENTIONS_PLACE", "Place", "Goshen", _HYMN, start_key="XVII_PENITENTIAL_HYMN"),
    EdgeRemoval("MENTIONS_PLACE", "Place", "Edom", _HYMN, start_key="XVII_PENITENTIAL_HYMN"),
    EdgeRemoval("MENTIONS_PLACE", "Place", "Old Gate", _HYMN, start_key="XVII_PENITENTIAL_HYMN"),
    _collector("Elkan Adler", "Elkan Adler collection"),
    _collector("Elkan Nathan Adler", "Elkan Nathan Adler Genizah collection"),
    _collector("Solomon Schechter", "Cambridge Genizah collection"),
    _collector("Isaacs", "Cambridge, University Library"),
    _collector("Isaacs", "T-S Genizah collection"),
    EdgeRemoval("MENTIONS_PLACE", "Place", "Raḥba", (
        "PGP tags T-S 8.61 to both Raḥbas; a journey from Aleppo to Raḥba is al-Raḥba on the Euphrates, "
        "which keeps its edge."), start_key="T_S_8_61"),
    EdgeRemoval("ORIGINATED_FROM", "Place", "Raḥba", (
        "The nisba 'Ben al-Raḥbī' belongs to his wife Sittūna; he is 'Ben Bānūqa' (zinger_princet p. 421)."),
        start_key="Faraḥ b. Joseph"),
)


# ---------------------------------------------------------------------------
# Places
# ---------------------------------------------------------------------------
PLACE_MERGES: dict[str, InstitutionMerge] = {
    "Turūja": merge("Tarūja", (
        "Same Arabic name (تروجة); PGP calls Turūja 'a town south of Alexandria', i.e. the Buḥayra town "
        "Tarūja. Turūja's 30.5, 30.1 is a rounded point in the desert."), normalize=False),
}

EDGE_MOVES: tuple[EdgeMove, ...] = (
    EdgeMove("ORIGINATED_FROM", "Place", "Raḥba", "al-Raḥba", "Sittūna bt. Ḥayyīm b. Solomon", (
        "Her nisba 'Ben al-Raḥbī' (zinger_princet p. 421) points to al-Raḥba on the Euphrates, the "
        "well-known town, not the Lebanese village PGP defines as Raḥba.")),
)
