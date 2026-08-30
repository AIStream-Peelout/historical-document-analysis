#!/usr/bin/env python3
"""Report every shelfmark in Elasticsearch that has no images, grouped by collection.

The merged Genizah index (``genizah_merged_v4``) unions PGP + FJP + KTIV
shelfmarks, but only records with an FJP image filename or a downloaded KTIV
scan carry ``image_urls``. Everything else is a fragment we hold *metadata* for
and no picture of — the KTIV scrape worklist.

Unlike :mod:`src.datasets.merging.merge_shelfmarks` (which derives its worklist
from the on-disk merged JSONL), this reads the **live index**, so it reflects
what is actually searchable today, including any image backfills applied after
the last merge run.

Grouping is by descriptive institution token (``Cambridge_CUL``,
``Oxford_Bodleian`` …) rather than the raw ``institution`` string, because the
sources spell the same holding library many ways (``Rylands`` /
``John Rylands Library, University of Manchester`` / ``Manchester``). The token
is the same one that prefixes every canonical id, so the grouping matches the
ids in the output. See :mod:`src.datasets.merging.institution_tokens`.

Usage::

    ELASTIC_SEARCH_HOST=localhost ELASTIC_SEARCH_PORT=9200 \\
    ELASTIC_SEARCH_SCHEME=http \\
    python -m src.datasets.indexing.es_image_gap_report --top 15
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import dotenv
from elasticsearch import Elasticsearch

from src.datasets.indexing.elastic_index_genizah import es_config_from_env
from src.datasets.merging.institution_tokens import _REGISTRY, institution_token, resolve_token

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = "genizah_merged_v4"
DEFAULT_OUT_DIR = _REPO_ROOT / "src/datasets/raw_data/cairo_genizah/merged"
PAGE_SIZE = 1000

#: Registry tokens, longest first, so ``Cambridge_Lewis_Gibson`` is tested
#: before ``Cambridge_CUL`` when prefix-matching a canonical id.
_TOKENS_BY_LENGTH: List[str] = sorted((token for token, _ in _REGISTRY), key=len, reverse=True)

#: Fields pulled from ``_source``. Excludes the dense vector and the bulky text
#: bodies — a 21k-row worklist should not drag 200MB of transcriptions with it.
_SOURCE_FIELDS: List[str] = [
    "canonical_id",
    "shelf_mark",
    "institution",
    "collection",
    "sub_collection",
    "source_collection",
    "sources_present",
    "description",
    "has_bib",
    "has_transcriptions",
    "completeness_score",
    "ktiv_iiif_manifest_url",
    "has_ktiv_images",
]


@dataclass
class GapRecord:
    """One shelfmark that exists in the index with no image attached.

    :param canonical_id: Institution-qualified merged id (also the ES ``_id``).
    :param shelf_mark: Display shelfmark as indexed.
    :param institution_group: Descriptive institution token used for grouping.
    :param collection: Raw sub-collection string (``Taylor-Schechter`` …).
    :param institution_raw: Raw institution string as indexed.
    :param sources_present: Which of pgp/fjp/ktiv contributed this record.
    :param ktiv_manifest_url: KTIV IIIF manifest, when the record already
        carries one (these are directly downloadable — no lookup needed).
    :param ktiv_known: True when KTIV metadata exists for the shelfmark.
    :param description_chars: Length of the indexed description.
    :param has_bib: Whether bibliography records are attached.
    :param has_transcriptions: Whether a transcription is attached.
    :param completeness_score: Indexed completeness heuristic.
    """

    canonical_id: str
    shelf_mark: str
    institution_group: str
    collection: str
    institution_raw: str
    sources_present: str
    ktiv_manifest_url: str
    ktiv_known: bool
    description_chars: int
    has_bib: bool
    has_transcriptions: bool
    completeness_score: float

    @property
    def ktiv_ready(self) -> bool:
        """Whether the scan can be fetched straight from a known IIIF manifest.

        :returns: True when a KTIV IIIF manifest URL is already indexed.
        """
        return bool(self.ktiv_manifest_url)

    @property
    def metadata_rich(self) -> bool:
        """Whether the record carries substantive metadata worth imaging.

        Mirrors :func:`src.datasets.merging.merge_shelfmarks.record_is_rich`:
        a fragment we already have a description, transcription or scholarly
        record for is a high-value scrape target.

        :returns: True when description, transcription or bibliography exists.
        """
        return bool(self.description_chars or self.has_transcriptions or self.has_bib)


def build_gap_query() -> Dict[str, Any]:
    """Build the query selecting indexed records with no image.

    ``has_images`` is the explicit flag; ``image_urls`` is asserted empty too
    because an empty array would leave a stale ``has_images`` undetected.

    :returns: An Elasticsearch query DSL fragment.
    """
    return {
        "bool": {
            "filter": [{"term": {"has_images": False}}],
            "must_not": [{"exists": {"field": "image_urls"}}],
        }
    }


def institution_group(canonical_id: str, institution_raw: Optional[str] = None) -> str:
    """Resolve the descriptive institution token for a record.

    Canonical ids are *built* from these tokens, so prefix-matching the id is
    exact wherever the institution is in the registry. Records whose
    institution fell back to a slug (rare / private collections) are resolved
    from the raw institution text instead.

    :param canonical_id: Institution-qualified canonical id.
    :param institution_raw: Raw institution string from the index, if any.
    :returns: A descriptive token, never empty (``Unknown`` at worst).
    """
    for token in _TOKENS_BY_LENGTH:
        if canonical_id == token or canonical_id.startswith(token + "_"):
            return token
    resolved = resolve_token(institution_raw)
    if resolved:
        return resolved
    if institution_raw:
        return institution_token(institution_raw)
    return canonical_id.split("_")[0] or "Unknown"


def _to_record(source: Dict[str, Any], es_id: str) -> GapRecord:
    """Convert an ES ``_source`` hit into a :class:`GapRecord`.

    :param source: The ``_source`` dict of one hit.
    :param es_id: The document ``_id``, used when ``canonical_id`` is absent.
    :returns: The populated gap record.
    """
    canonical_id = source.get("canonical_id") or es_id
    institution_raw = source.get("institution") or ""
    collection = source.get("collection") or source.get("sub_collection") or ""
    sources_present = source.get("sources_present") or []
    return GapRecord(
        canonical_id=canonical_id,
        shelf_mark=source.get("shelf_mark") or "",
        institution_group=institution_group(canonical_id, institution_raw),
        collection=collection.strip(),
        institution_raw=institution_raw.strip(),
        sources_present="+".join(sorted(sources_present)),
        ktiv_manifest_url=source.get("ktiv_iiif_manifest_url") or "",
        ktiv_known="ktiv" in sources_present,
        description_chars=len(source.get("description") or ""),
        has_bib=bool(source.get("has_bib")),
        has_transcriptions=bool(source.get("has_transcriptions")),
        completeness_score=float(source.get("completeness_score") or 0.0),
    )


def iter_gap_documents(
    es: Elasticsearch,
    index: str = DEFAULT_INDEX,
    page_size: int = PAGE_SIZE,
) -> Iterator[GapRecord]:
    """Stream every imageless record out of the index.

    Paginates with ``search_after`` on ``canonical_id`` (a unique keyword), so
    no scroll context is held open and the walk is deterministic.

    :param es: A connected Elasticsearch client.
    :param index: Index to read.
    :param page_size: Hits per request.
    :yields: One :class:`GapRecord` per imageless document.
    """
    query = build_gap_query()
    after: Optional[Sequence[Any]] = None
    while True:
        kwargs: Dict[str, Any] = {
            "index": index,
            "size": page_size,
            "query": query,
            "sort": [{"canonical_id": "asc"}],
            "source_includes": _SOURCE_FIELDS,
        }
        if after is not None:
            kwargs["search_after"] = after
        hits = es.search(**kwargs)["hits"]["hits"]
        if not hits:
            return
        for hit in hits:
            yield _to_record(hit.get("_source") or {}, hit["_id"])
        after = hits[-1]["sort"]


def sort_key(record: GapRecord) -> tuple:
    """Sort worklist rows: collection, then scrape-readiness, then richness.

    :param record: The record to key.
    :returns: A tuple ordering institution > collection > KTIV-ready >
        metadata-rich > description length > id.
    """
    return (
        record.institution_group,
        record.collection or "~",  # unspecified collection sorts last
        not record.ktiv_ready,
        not record.metadata_rich,
        -record.description_chars,
        record.canonical_id,
    )


@dataclass
class GroupSummary:
    """Per-institution rollup of the image gap.

    :param institution_group: Descriptive institution token.
    :param total: Imageless shelfmarks for this institution.
    :param metadata_rich: How many carry description/transcription/bibliography.
    :param ktiv_ready: How many already have a KTIV IIIF manifest URL.
    :param ktiv_known: How many have any KTIV metadata.
    :param collections: Imageless count per raw collection string.
    :param source_mix: Imageless count per source combination.
    """

    institution_group: str
    total: int = 0
    metadata_rich: int = 0
    ktiv_ready: int = 0
    ktiv_known: int = 0
    collections: Dict[str, int] = field(default_factory=dict)
    source_mix: Dict[str, int] = field(default_factory=dict)


def summarize(records: Sequence[GapRecord]) -> List[GroupSummary]:
    """Roll the gap records up per institution, largest gap first.

    :param records: All imageless records.
    :returns: Summaries sorted by descending total.
    """
    groups: Dict[str, GroupSummary] = {}
    collections: Dict[str, Counter] = defaultdict(Counter)
    sources: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        summary = groups.setdefault(
            record.institution_group, GroupSummary(record.institution_group)
        )
        summary.total += 1
        summary.metadata_rich += int(record.metadata_rich)
        summary.ktiv_ready += int(record.ktiv_ready)
        summary.ktiv_known += int(record.ktiv_known)
        collections[record.institution_group][record.collection or "(unspecified)"] += 1
        sources[record.institution_group][record.sources_present or "(none)"] += 1
    for token, summary in groups.items():
        summary.collections = dict(collections[token].most_common())
        summary.source_mix = dict(sources[token].most_common())
    return sorted(groups.values(), key=lambda s: (-s.total, s.institution_group))


def write_csv(records: Sequence[GapRecord], path: Path) -> None:
    """Write the per-shelfmark worklist, sorted by collection.

    :param records: Records to write (sorted internally by :func:`sort_key`).
    :param path: Destination CSV path.
    """
    columns = [
        "institution_group",
        "collection",
        "canonical_id",
        "shelf_mark",
        "sources_present",
        "ktiv_ready",
        "ktiv_known",
        "ktiv_manifest_url",
        "metadata_rich",
        "description_chars",
        "has_bib",
        "has_transcriptions",
        "completeness_score",
        "institution_raw",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in sorted(records, key=sort_key):
            row = asdict(record)
            row["ktiv_ready"] = record.ktiv_ready
            row["metadata_rich"] = record.metadata_rich
            writer.writerow({column: row[column] for column in columns})


def write_summary(
    summaries: Sequence[GroupSummary],
    index: str,
    index_total: int,
    path: Path,
) -> Dict[str, Any]:
    """Write the institution rollup as JSON.

    :param summaries: Per-institution summaries.
    :param index: Index the report was built from.
    :param index_total: Total documents in the index.
    :param path: Destination JSON path.
    :returns: The report dict that was written.
    """
    gap_total = sum(summary.total for summary in summaries)
    report = {
        "index": index,
        "index_total": index_total,
        "imageless_total": gap_total,
        "imaged_total": index_total - gap_total,
        "metadata_rich_total": sum(s.metadata_rich for s in summaries),
        "ktiv_ready_total": sum(s.ktiv_ready for s in summaries),
        "ktiv_known_total": sum(s.ktiv_known for s in summaries),
        "institutions": [asdict(summary) for summary in summaries],
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_rollup(summaries: Sequence[GroupSummary], top: int) -> None:
    """Print the institution rollup to stdout.

    :param summaries: Per-institution summaries (already sorted).
    :param top: How many institutions to expand by collection; 0 for none.
    """
    header = f"{'institution':26s} {'no_image':>9s} {'rich':>7s} {'ktiv_ready':>11s} {'ktiv_meta':>10s}"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary.institution_group:26s} {summary.total:9d} {summary.metadata_rich:7d} "
            f"{summary.ktiv_ready:11d} {summary.ktiv_known:10d}"
        )
    if not top:
        return
    print("\nCollection breakdown (top institutions):")
    for summary in summaries[:top]:
        print(f"\n  {summary.institution_group}  ({summary.total} imageless)")
        for collection, count in list(summary.collections.items())[:12]:
            print(f"    {count:7d}  {collection}")


def main() -> None:
    """CLI entry point: build the imageless-shelfmark worklist from ES."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=DEFAULT_INDEX, help="ES index to read")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for CSV/JSON output"
    )
    parser.add_argument("--prefix", default="es_image_gap", help="Output filename prefix")
    parser.add_argument("--host", help="Override ELASTIC_SEARCH_HOST")
    parser.add_argument("--port", help="Override ELASTIC_SEARCH_PORT")
    parser.add_argument("--scheme", help="Override ELASTIC_SEARCH_SCHEME")
    parser.add_argument(
        "--top", type=int, default=8, help="Institutions to expand by collection in stdout"
    )
    parser.add_argument(
        "--rich-only",
        action="store_true",
        help="Keep only records with description/transcription/bibliography",
    )
    args = parser.parse_args()

    dotenv.load_dotenv(_REPO_ROOT / ".env")
    for flag, env_var in (
        (args.host, "ELASTIC_SEARCH_HOST"),
        (args.port, "ELASTIC_SEARCH_PORT"),
        (args.scheme, "ELASTIC_SEARCH_SCHEME"),
    ):
        if flag:  # CLI wins over .env
            os.environ[env_var] = flag

    es = Elasticsearch(**es_config_from_env(), request_timeout=60)
    index_total = es.count(index=args.index)["count"]
    records = list(iter_gap_documents(es, args.index))
    if args.rich_only:
        records = [record for record in records if record.metadata_rich]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.prefix}_by_collection.csv"
    json_path = args.out_dir / f"{args.prefix}_summary.json"
    summaries = summarize(records)
    write_csv(records, csv_path)
    report = write_summary(summaries, args.index, index_total, json_path)

    print(
        f"{args.index}: {report['imageless_total']} of {index_total} shelfmarks have no image "
        f"({report['metadata_rich_total']} metadata-rich, {report['ktiv_ready_total']} with a "
        f"KTIV manifest already)\n"
    )
    print_rollup(summaries, args.top)
    print(f"\nWrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
