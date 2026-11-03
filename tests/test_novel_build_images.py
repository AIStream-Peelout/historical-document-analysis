"""Tests for the stage-1 image resolver of the novel-rows build (synthetic fixtures; no network, no shared volumes).

Hosts are patched to ``example.invalid`` names; every HTTP call goes through stubs that count their calls.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pytest
from PIL import Image

from src.datasets import shelfmarks
from src.finetuning.qwen_hebrew.novel_build import images as im
from src.finetuning.qwen_hebrew.novel_build import paths
from src.finetuning.qwen_hebrew.novel_build.schema import Candidate, ImageRef, read_jsonl, write_jsonl

BUCKET = "https://bucket.example.invalid/store/images"
CUDL_MANIFEST = "https://cudl.example.invalid/iiif/MS-TEST-00001"
PGP_MANIFEST = "https://pgp.example.invalid/iiif/doc-2.json"
CUDL_SERVICE = "https://img-cudl.example.invalid/iiif/MS-TEST-00001-000-0000{n}.jp2"
LEDGER_LIB = Path(__file__).resolve().parents[1] / "logs/next_round/unseen_v1/registry_lib.py"


@pytest.fixture(autouse=True)
def synthetic_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every host-dependent rule of the module at ``example.invalid`` hosts.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(im, "LEGACY_BUCKET_BASE", BUCKET)
    monkeypatch.setattr(im, "BUCKET_HOSTS", frozenset({"bucket.example.invalid"}))
    monkeypatch.setattr(im, "CUDL_HOSTS", frozenset({"cudl.example.invalid", "img-cudl.example.invalid"}))
    monkeypatch.setattr(im, "PGP_HOSTS", frozenset({"pgp.example.invalid"}))
    monkeypatch.setattr(im, "NLI_HOSTS", frozenset({"nli.example.invalid"}))
    monkeypatch.setattr(im, "BODLEIAN_HOSTS", frozenset({"bodleian.example.invalid"}))


# ----------------------------------------------------------------------------------------------------- helpers


def pattern_image(width: int, height: int, seed: int) -> Image.Image:
    """A deterministic RGB test pattern (different seeds give different pictures).

    :param width: Pixels.
    :param height: Pixels.
    :param seed: Pattern selector.
    :return: Image.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    arr = np.stack([(xx * (seed % 7 + 1) + yy) % 256, (yy * (seed % 5 + 2) + seed * 11) % 256,
                    ((xx // (seed + 3)) * 37 + (yy // 17) * 13) % 256], -1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def encode(img: Image.Image, fmt: str = "JPEG", orientation: Optional[int] = None) -> bytes:
    """Encode an image, optionally with an EXIF orientation tag.

    :param img: Image.
    :param fmt: PIL format name.
    :param orientation: EXIF orientation value or None.
    :return: Encoded bytes.
    """
    buf = io.BytesIO()
    kwargs: Dict[str, Any] = {"quality": 90} if fmt == "JPEG" else {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif.tobytes()
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


def sha(data: bytes) -> str:
    """sha256 hex of bytes.

    :param data: Bytes.
    :return: Hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def cand(row_id: str, source: str, *, cids: Sequence[str] = (), pgpids: Sequence[str] = (),
         ktiv_id: Optional[str] = None, images: Sequence[ImageRef] = ()) -> Candidate:
    """A minimal candidate.

    :param row_id: Row id.
    :param source: Ledger source.
    :param cids: Canonical ids.
    :param pgpids: PGP ids.
    :param ktiv_id: KTIV system number.
    :param images: Stage-0 image references.
    :return: Candidate.
    """
    return Candidate(row_id=row_id, source=source, document_group=f"group:{row_id}", canonical_ids=list(cids),
                     pgpids=list(pgpids), ktiv_id=ktiv_id, shelfmark="Test Lib 1", script="hebrew",
                     languages_stated=None, verdict="novel", witnesses=[], unit="document", n_letters=120,
                     images=list(images))


def manifest_v2() -> Dict[str, Any]:
    """A Presentation 2 manifest with two canvases (a service object, then a service list).

    :return: Manifest.
    """
    first = {"@id": "x1.jpg", "service": {"@id": CUDL_SERVICE.format(n=1)}}
    second = {"@id": "x2.jpg", "service": [{"@id": CUDL_SERVICE.format(n=2) + "/"}]}
    canvases = [{"@id": "c1", "label": "1r", "images": [{"resource": first}]},
                {"@id": "c2", "label": ["1v"], "images": [{"resource": second}]}]
    return {"@context": "http://iiif.io/api/presentation/2/context.json", "sequences": [{"canvases": canvases}]}


def manifest_v3() -> Dict[str, Any]:
    """A Presentation 3 manifest: a canvas with an image service, a canvas with a plain image body.

    :return: Manifest.
    """
    def canvas(label: Dict[str, List[str]], body: Dict[str, Any]) -> Dict[str, Any]:
        """One canvas.

        :param label: Language map.
        :param body: Painting body.
        :return: Canvas.
        """
        return {"type": "Canvas", "label": label,
                "items": [{"type": "AnnotationPage", "items": [{"type": "Annotation", "body": body}]}]}

    return {"@context": "http://iiif.io/api/presentation/3/context.json", "items": [
        canvas({"none": ["recto"]}, {"id": "https://iiif.other.example.invalid/img/a/full/max/0/default.jpg",
                                     "service": [{"id": "https://iiif.other.example.invalid/img/a",
                                                  "type": "ImageService3"}]}),
        canvas({"en": ["verso"]}, {"id": "https://iiif.other.example.invalid/img/b.jpg"}),
    ]}


@dataclass
class World:
    """Every synthetic input of a resolution, on disk under ``tmp_path``."""

    root: Path
    work: Path
    pgp_dir: Path
    merged: Path
    corrections: Path
    legacy: Path
    run2: Path
    ktiv_dir: Path
    ktiv_cache: Path
    bodleian_dir: Path
    files: Dict[str, Path]
    data: Dict[str, bytes]

    def resolve(self, cands: List[Candidate], fetch_manifest: Optional[Callable[[str], Dict[str, Any]]] = None,
                **kw: Any) -> Counter:
        """Run :func:`images.resolve_images` on this world.

        :param cands: Candidates (mutated).
        :param fetch_manifest: Manifest stub or None (offline).
        :param kw: Extra keyword arguments.
        :return: Resolution counts.
        """
        return im.resolve_images(cands, pgp_export_dir=self.pgp_dir, merged_path=self.merged,
                                 corrections_path=self.corrections, work_dir=self.work, legacy_manifest=self.legacy,
                                 run2_local_images=self.run2, ktiv_dir=self.ktiv_dir, ktiv_cache=self.ktiv_cache,
                                 fetch_manifest=fetch_manifest, bodleian_dir=self.bodleian_dir, workers=1, delay=0,
                                 sleep=lambda s: None, **kw)

    def cli_args(self) -> List[str]:
        """Path flags of :func:`images.main` for this world.

        :return: Arguments.
        """
        return ["--work-dir", str(self.work), "--pgp-export-dir", str(self.pgp_dir), "--merged", str(self.merged),
                "--corrections", str(self.corrections), "--legacy-manifest", str(self.legacy),
                "--run2-local-images", str(self.run2), "--ktiv-dir", str(self.ktiv_dir),
                "--ktiv-cache", str(self.ktiv_cache), "--bodleian-dir", str(self.bodleian_dir), "--workers", "1"]


def correction(image: str, src: str, dst: Optional[str], side: Optional[str] = None) -> Dict[str, Any]:
    """A valid image-correction line.

    :param image: File name.
    :param src: From canonical id.
    :param dst: To canonical id (None = detach).
    :param side: Slip side.
    :return: Correction record.
    """
    return {"image": image, "url": f"{BUCKET}/{image}", "sha256": None, "from_canonical_id": src,
            "to_canonical_id": dst, "to_shelfmark_text": None if dst else "Unknown 9", "side": side,
            "basis": "eye", "evidence": "slip", "reviewed": True, "recorded_at": "2026-11-01", "source": "test"}


@pytest.fixture
def world(tmp_path: Path) -> World:
    """Build the synthetic corpus: PGP export, a 3-record merged file, corrections (one move, one detach), a legacy
    manifest with local files, run-2 local downloads, a KTIV image zip and a local Bodleian master.

    :param tmp_path: Pytest temp dir.
    :return: The world.
    """
    files: Dict[str, Path] = {}
    data: Dict[str, bytes] = {}

    def put(rel: str, payload: bytes) -> Path:
        """Write a file under the world root.

        :param rel: Relative path.
        :param payload: Bytes.
        :return: Path.
        """
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files[Path(rel).name] = path
        data[Path(rel).name] = payload
        return path

    for i, name in enumerate(["A_1_1r.jpg", "MISFILED_1r.jpg", "D_4_1v.jpg"]):
        put(f"store/{name}", encode(pattern_image(1000, 1200, i + 1)))
    put("run2/MS-D/01_1r.jpg", encode(pattern_image(900, 1100, 9)))
    put("bodleian/Test_Lib_A_1/A1a.jpg", encode(pattern_image(1000, 1000, 12)))
    put("apd/page.jpg", encode(pattern_image(1000, 1300, 13)))

    pgp_dir = tmp_path / "pgp_export"
    pgp_dir.mkdir()
    with open(pgp_dir / "documents.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pgpid", "iiif_urls", "side", "shelfmark"])
        writer.writeheader()
        writer.writerow({"pgpid": "101", "iiif_urls": f"{CUDL_MANIFEST} ; {PGP_MANIFEST}", "side": "recto",
                         "shelfmark": "Test Lib A 1"})
        writer.writerow({"pgpid": "202", "iiif_urls": "", "side": "", "shelfmark": "Test Lib B 2"})

    merged = tmp_path / "merged.jsonl"
    images_a = {"fjp": ["A_1_1r.jpg", "A_1_1v.jpg", "101_image.jpg"],
                "ktiv": {"sys_num": "990", "image_urls": ["https://bucket.example.invalid/store/KTIV/990/001_1r.jpg"]},
                "bodleian": {"image_urls": ["https://bucket.example.invalid/store/BODLEIAN/Test_Lib_A_1/A1a.jpg"],
                             "images": ["BODLEIAN/Test_Lib_A_1/A1a.jpg"]}}
    rec_a = {"canonical_id": "Test_Lib_A_1", "pgpids": ["101"], "images": images_a,
             "sources": {"pgp": {"fragment": {"iiif_url": CUDL_MANIFEST}}}}
    rec_b = {"_id": "Test_Lib_B_2", "pgpids": [], "images": {"fjp": ["B_2_1r.jpg", "MISFILED_1r.jpg"]}}
    rec_c = {"canonical_id": "Test_Lib_C_3", "pgpids": [], "images": {"fjp": ["C_3_1r.jpg"]}}
    merged.write_text("".join(json.dumps(r) + "\n" for r in (rec_a, rec_b, rec_c)), encoding="utf-8")

    corrections = tmp_path / "image_corrections.jsonl"
    corrections.write_text(json.dumps(correction("MISFILED_1r.jpg", "Test_Lib_B_2", "Test_Lib_A_1", side="1r"))
                           + "\n" + json.dumps(correction("A_1_1v.jpg", "Test_Lib_A_1", None)) + "\n",
                           encoding="utf-8")

    legacy = tmp_path / "images_manifest.jsonl"
    legacy_rows = [
        {"canonical_id": "Test_Lib_A_1", "file_name": "A_1_1r.jpg", "path": str(files["A_1_1r.jpg"]),
         "url": f"{BUCKET}/A_1_1r.jpg", "sha256": sha(data["A_1_1r.jpg"]), "width": 1000, "height": 1200,
         "image_index": 0, "folio": 1, "side_token": "r", "slip_side": "r", "slip_outcome": "confirmed"},
        {"canonical_id": "Test_Lib_A_1", "file_name": "MISFILED_1r.jpg", "path": str(files["MISFILED_1r.jpg"]),
         "url": f"{BUCKET}/MISFILED_1r.jpg", "sha256": sha(data["MISFILED_1r.jpg"]), "width": 1000,
         "height": 1200, "image_index": 1, "side_token": None, "slip_side": None,
         "slip_outcome": "relabel_to_known_document"},
        {"canonical_id": "Test_Lib_D_4", "file_name": "D_4_1v.jpg", "path": str(files["D_4_1v.jpg"]),
         "url": f"{BUCKET}/D_4_1v.jpg", "sha256": sha(data["D_4_1v.jpg"]), "width": 1000, "height": 1200,
         "image_index": 0, "folio": 1, "side_token": "v", "slip_side": "2r", "slip_outcome": "confirmed"},
    ]
    legacy.write_text("".join(json.dumps(r) + "\n" for r in legacy_rows), encoding="utf-8")

    run2 = tmp_path / "local_images.json"
    run2.write_text(json.dumps({"Test_Lib_D_4": [{"path": str(files["01_1r.jpg"]), "canvas_label": "1r",
                                                  "manifest_id": "MS-D"}]}), encoding="utf-8")

    ktiv_dir = tmp_path / "ktiv"
    ktiv_dir.mkdir()
    page = encode(pattern_image(1100, 1500, 21))
    data["ktiv_page"] = page
    with zipfile.ZipFile(ktiv_dir / "ktiv_PNX_MANUSCRIPTS990001-1_images.zip", "w") as zf:
        zf.writestr("0001_FL123.jpg", page)
        zf.writestr("0002_FL124.jpg", encode(pattern_image(1100, 1500, 22)))

    return World(root=tmp_path, work=tmp_path / "work", pgp_dir=pgp_dir, merged=merged, corrections=corrections,
                 legacy=legacy, run2=run2, ktiv_dir=ktiv_dir, ktiv_cache=tmp_path / "ktiv_cache",
                 bodleian_dir=tmp_path / "bodleian", files=files, data=data)


class Http:
    """Stub HTTP layer: per-URL bodies and HEAD statuses, with call counters."""

    def __init__(self, bodies: Optional[Dict[str, bytes]] = None, heads: Optional[Dict[str, int]] = None,
                 manifests: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Create the stub.

        :param bodies: GET bodies by URL (absent -> error).
        :param heads: HEAD statuses by URL (absent -> 200).
        :param manifests: Manifests by URL.
        """
        self.bodies = dict(bodies or {})
        self.heads = dict(heads or {})
        self.manifests = dict(manifests or {})
        self.calls: Counter = Counter()

    def head(self, url: str) -> int:
        """HEAD stub.

        :param url: URL.
        :return: Status.
        """
        self.calls[("head", url)] += 1
        return self.heads.get(url, 200)

    def get(self, url: str) -> bytes:
        """GET stub.

        :param url: URL.
        :return: Body.
        :raises RuntimeError: For an unknown URL.
        """
        self.calls[("get", url)] += 1
        if url not in self.bodies:
            raise RuntimeError("connection refused")
        return self.bodies[url]

    def manifest(self, url: str) -> Dict[str, Any]:
        """Manifest stub.

        :param url: URL.
        :return: Manifest.
        :raises im.PermanentFetchError: For an unknown URL.
        """
        self.calls[("manifest", url)] += 1
        if url not in self.manifests:
            raise im.PermanentFetchError("HTTP 404")
        return self.manifests[url]

    def total(self) -> int:
        """Number of calls of any kind.

        :return: Count.
        """
        return sum(self.calls.values())


def fetch(cands: List[Candidate], work: Path, http: Http, **kw: Any) -> Counter:
    """Run :func:`images.fetch_missing` with the stub and no sleeping.

    :param cands: Candidates.
    :param work: Work dir.
    :param http: Stub.
    :param kw: Extra keyword arguments.
    :return: Counts.
    """
    return im.fetch_missing(cands, work, head=http.head, get=http.get, delay=0, sleep=lambda s: None, **kw)


def by_origin(c: Candidate) -> Counter:
    """Origins of a candidate's images.

    :param c: Candidate.
    :return: Counts.
    """
    return Counter(r.origin for r in c.images)


# ------------------------------------------------------------------------------------------- manifest parsers


def test_manifest_canvases_v2_uses_service_and_cudl_width() -> None:
    """A v2 manifest: canvases in order, labels from a string and a list, the Cambridge service at its width."""
    canvases = im.manifest_canvases(manifest_v2())
    assert [c["canvas_index"] for c in canvases] == [0, 1]
    assert [c["label"] for c in canvases] == ["1r", "1v"]
    assert canvases[0]["image_url"] == CUDL_SERVICE.format(n=1) + "/full/3245,/0/default.jpg"
    assert canvases[1]["image_url"] == CUDL_SERVICE.format(n=2) + "/full/3245,/0/default.jpg"


def test_manifest_canvases_v3_service_and_plain_body() -> None:
    """A v3 manifest: language-map labels, ``service[].id`` -> full image, no service -> the body's own id."""
    canvases = im.manifest_canvases(manifest_v3())
    assert [c["label"] for c in canvases] == ["recto", "verso"]
    assert canvases[0]["image_url"] == "https://iiif.other.example.invalid/img/a/full/full/0/default.jpg"
    assert canvases[1]["image_url"] == "https://iiif.other.example.invalid/img/b.jpg"
    assert canvases[1]["service"] is None


def test_refs_from_manifest_origin_by_host() -> None:
    """Canvas references take the manifest host's origin and keep label and canvas index."""
    cudl = im.refs_from_manifest(CUDL_MANIFEST, manifest_v2())
    pgp = im.refs_from_manifest(PGP_MANIFEST, manifest_v3())
    other = im.refs_from_manifest("https://museum.example.invalid/m.json", manifest_v3())
    assert {r.origin for r in cudl} == {"iiif_cudl"} and {r.origin for r in pgp} == {"iiif_pgp"}
    assert {r.origin for r in other} == {"iiif_other"}
    assert [(r.canvas_index, r.label, r.status) for r in cudl] == [(0, "1r", "listed"), (1, "1v", "listed")]


def test_legacy_label_rule() -> None:
    """The slip side replaces the file-name token; an agreeing token only adds the leaf."""
    assert im.legacy_label("1r", "") == "1r" and im.legacy_label("1v", "v") == "1v"
    assert im.legacy_label("1r", "v") == "v" and im.legacy_label("", "2r") == "2r"


def test_origin_and_url_helpers() -> None:
    """Bucket folders map to their mirrored source; mirrors and image URLs are recognised."""
    assert im.origin_of_url(f"{BUCKET}/X_1_1r.jpg") == "fjp_bucket"
    assert im.origin_of_url(f"{BUCKET}/555_image.jpg") == "pgpid_mirror"
    assert im.origin_of_url("https://bucket.example.invalid/store/KTIV/9/001.jpg") == "iiif_nli"
    assert im.origin_of_url("https://bucket.example.invalid/store/BODLEIAN/X/a.jpg") == "bodleian"
    assert im.is_image_url("https://nli.example.invalid/IIIF/FL1/full/max/0/default.jpg")
    assert not im.is_image_url(CUDL_MANIFEST) and not im.is_image_url(PGP_MANIFEST)
    assert im.split_urls(f" {CUDL_MANIFEST} ;{PGP_MANIFEST}  ") == [CUDL_MANIFEST, PGP_MANIFEST]
    assert im.safe_name("lit:a/b c:1") == "lit_a_b_c_1"
    assert im.side_token("T_S_1_2v.jpg") == "2v" and im.side_token("cover.jpg") is None


def test_split_existing_separates_manifest_url_from_local_path(tmp_path: Path) -> None:
    """A stage-0 reference whose URL is a manifest and whose path is another file becomes two items."""
    ref = ImageRef(image_id="x:0", origin="fjp_store_named", url=CUDL_MANIFEST, path=str(tmp_path / "a.jpg"),
                   sha256="ab" * 32, phash="0" * 16)
    items = im.split_existing(ref)
    assert items[1] == CUDL_MANIFEST
    assert isinstance(items[0], ImageRef) and items[0].url is None and items[0].origin == "local_other"
    assert items[0].sha256 == "ab" * 32
    same = ImageRef(image_id="y:0", origin="fjp_local", url=f"{BUCKET}/a.jpg", path=str(tmp_path / "a.jpg"))
    assert len(im.split_existing(same)) == 1


# ------------------------------------------------------------------------------------------ source resolution


def test_resolve_pgp_lists_every_source_with_corrections(world: World) -> None:
    """PGP: export manifests (v2 + v3) expanded, merged images listed by origin, the move and the detach applied,
    the stage-0 local path merged with its legacy-store listing, and manifests and the merged index cached."""
    pool = ImageRef(image_id="pool:0", origin="fjp_store_named", url=CUDL_MANIFEST,
                    path=str(world.files["A_1_1r.jpg"]), status="present")
    c = cand("pgp:101", "pgp", cids=["Test_Lib_A_1"], pgpids=["101"], images=[pool])
    http = Http(manifests={CUDL_MANIFEST: manifest_v2(), PGP_MANIFEST: manifest_v3()})
    stats = world.resolve([c], fetch_manifest=http.manifest)

    origins = by_origin(c)
    assert origins == Counter({"iiif_cudl": 2, "iiif_pgp": 2, "fjp_local": 2, "pgpid_mirror": 1, "iiif_nli": 1,
                               "bodleian": 1})
    urls = {r.url for r in c.images}
    assert f"{BUCKET}/A_1_1v.jpg" not in urls                      # detached
    assert f"{BUCKET}/B_2_1r.jpg" not in urls                      # another record's photograph
    moved = next(r for r in c.images if r.url == f"{BUCKET}/MISFILED_1r.jpg")
    assert moved.origin == "fjp_local" and moved.label == "1r" and moved.status == "present"
    own = next(r for r in c.images if r.path == str(world.files["A_1_1r.jpg"]))
    assert own.url == f"{BUCKET}/A_1_1r.jpg" and own.label == "1r" and own.slip_outcome == "confirmed"
    assert own.sha256 == sha(world.data["A_1_1r.jpg"]) and own.phash and (own.width, own.height) == (1000, 1200)
    bod = next(r for r in c.images if r.origin == "bodleian")
    assert bod.path == str(world.files["A1a.jpg"]) and bod.status == "present"
    assert next(r for r in c.images if r.origin == "pgpid_mirror").status == "listed"
    assert [r.image_id for r in c.images] == [f"{r.origin}:{i}" for i, r in enumerate(c.images)]
    assert stats["corrections_moved"] == 1 and stats["corrections_detached"] == 1
    notes = c.notes[im.NOTES_KEY]
    assert {m["url"] for m in notes["manifests_expanded"]} == {CUDL_MANIFEST, PGP_MANIFEST}
    assert notes["corrections"] == [{"canonical_id": "Test_Lib_A_1", "moved_in": ["MISFILED_1r.jpg"],
                                     "removed": ["A_1_1v.jpg"]}]
    assert http.total() == 2

    index = json.loads((world.work / "manifests" / im.MERGED_INDEX_FILE).read_text(encoding="utf-8"))
    assert set(index["records"]) == {"Test_Lib_A_1", "Test_Lib_B_2"}   # wanted + correction ids, not C
    again = cand("pgp:101", "pgp", cids=["Test_Lib_A_1"], pgpids=["101"], images=[pool])
    world.resolve([again], fetch_manifest=http.manifest)
    assert http.total() == 2                                        # manifests served from the cache
    assert [r.url for r in again.images] == [r.url for r in c.images]


def test_resolve_offline_leaves_uncached_manifests_unexpanded(world: World) -> None:
    """Without a manifest fetcher, uncached manifests are named in the notes and nothing is requested."""
    c = cand("pgp:101", "pgp", cids=["Test_Lib_A_1"], pgpids=["101"])
    stats = world.resolve([c], fetch_manifest=None)
    unexpanded = c.notes[im.NOTES_KEY]["manifests_unexpanded"]
    assert {u["url"] for u in unexpanded} == {CUDL_MANIFEST, PGP_MANIFEST}
    assert {u["reason"] for u in unexpanded} == {"manifest_not_cached"}
    assert "iiif_cudl" not in by_origin(c) and stats["manifests_unavailable"] == 2


def test_merged_index_cache_is_reused_until_new_ids(world: World) -> None:
    """The merged index is read from its cache while it covers the ids; a new id triggers one rebuild pass."""
    cache = world.root / "idx.json"
    first = im.merged_images_index(world.merged, ["Test_Lib_A_1"], cache)
    assert set(first) == {"Test_Lib_A_1"} and first["Test_Lib_A_1"]["pgp_iiif"] == CUDL_MANIFEST
    payload = json.loads(cache.read_text(encoding="utf-8"))
    payload["records"]["Test_Lib_A_1"]["marker"] = True
    cache.write_text(json.dumps(payload), encoding="utf-8")
    assert im.merged_images_index(world.merged, ["Test_Lib_A_1"], cache)["Test_Lib_A_1"].get("marker") is True
    rebuilt = im.merged_images_index(world.merged, ["Test_Lib_B_2"], cache)
    assert set(rebuilt) == {"Test_Lib_A_1", "Test_Lib_B_2"} and "marker" not in rebuilt["Test_Lib_A_1"]


def test_resolve_legacy_store_candidate_verifies_files(world: World) -> None:
    """Legacy-store candidates: an existing file is present (sha256 and phash computed when missing), an absent
    path falls back to its bucket URL."""
    present = ImageRef(image_id="fjp_local:0", origin="fjp_local", url=f"{BUCKET}/A_1_1r.jpg",
                       path=str(world.files["A_1_1r.jpg"]), label="1r", status="present")
    gone = ImageRef(image_id="fjp_local:1", origin="fjp_local", url=f"{BUCKET}/A_1_9v.jpg",
                    path=str(world.root / "store/A_1_9v.jpg"), label="9v", status="present")
    c = cand("Test_Lib_A_1:0", "fjp", cids=["Test_Lib_A_1"], images=[present, gone])
    world.resolve([c])
    first, second = c.images
    assert first.status == "present" and first.sha256 == sha(world.data["A_1_1r.jpg"]) and len(first.phash) == 16
    assert first.label == "1r"
    assert second.status == "listed" and second.path is None and second.origin == "fjp_bucket"
    assert "path_absent" in second.note


def test_resolve_ktiv_extracts_zip_page_once(world: World) -> None:
    """KTIV: the page is found in the manuscript zip, cached as ``<sys>/<fl>.jpg`` with the member's sha256; a page
    missing from the zip keeps its library URL."""
    url = "https://nli.example.invalid/IIIF/FL123/full/2800,/0/default.jpg"
    c = cand("990001:FL123", "ktiv", ktiv_id="990001",
             images=[ImageRef(image_id="ktiv:0", origin="ktiv", url=url)])
    lost = cand("990001:FL999", "ktiv", ktiv_id="990001",
                images=[ImageRef(image_id="ktiv:0", origin="ktiv",
                                 url="https://nli.example.invalid/IIIF/FL999/full/2800,/0/default.jpg")])
    world.resolve([c, lost])
    (ref,) = c.images
    cached = world.ktiv_cache / "990001" / "FL123.jpg"
    assert ref.origin == "ktiv_zip" and ref.path == str(cached) and ref.url == url and ref.status == "present"
    assert ref.sha256 == sha(world.data["ktiv_page"]) == sha(cached.read_bytes())
    assert "0001_FL123.jpg" in ref.note
    (other,) = lost.images
    assert other.origin == "iiif_nli" and other.status == "listed" and other.path is None
    assert lost.notes[im.NOTES_KEY]["misses"] == ["ktiv_zip_member_not_found"]
    mtime = cached.stat().st_mtime_ns
    again = cand("990001:FL123", "ktiv", ktiv_id="990001", images=[ImageRef(image_id="k", origin="ktiv", url=url)])
    world.resolve([again])
    assert cached.stat().st_mtime_ns == mtime and again.images[0].sha256 == ref.sha256


def test_resolve_literature_adds_run2_and_legacy_files(world: World) -> None:
    """Literature: the candidate's URL, the run-2 local download (label kept) and the legacy-store file of its
    canonical id; a canonical id absent from the merged file is noted."""
    pool = ImageRef(image_id="fjp:0", origin="fjp", url=f"{BUCKET}/D_4_1r.jpg")
    c = cand("lit:work:p1:d4", "literature", cids=["Test_Lib_D_4"], images=[pool])
    world.resolve([c])
    assert by_origin(c) == Counter({"fjp_bucket": 1, "cudl_local": 1, "fjp_local": 1})
    local = next(r for r in c.images if r.origin == "cudl_local")
    assert local.label == "1r" and local.status == "present" and local.sha256 == sha(world.data["01_1r.jpg"])
    legacy = next(r for r in c.images if r.origin == "fjp_local")
    assert legacy.label == "2r" and legacy.path == str(world.files["D_4_1v.jpg"])   # the slip side wins
    assert "conflict" in legacy.note
    assert c.notes[im.NOTES_KEY]["misses"] == ["merged_record_absent Test_Lib_D_4"]


def test_cambridge_manifest_derived_when_nothing_is_listed(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A literature candidate with no listed image gets the library manifest of its Cambridge id (via the facade)."""
    monkeypatch.setattr(shelfmarks, "CUDL_BASE", "https://cudl.example.invalid/iiif/")
    derived = "https://cudl.example.invalid/iiif/MS-TS-NS-00009-00012"
    c = cand("lit:work:p2:ns9", "literature", cids=["Cambridge_CUL_T_S_NS_9_12"])
    http = Http(manifests={derived: manifest_v2()})
    world.resolve([c], fetch_manifest=http.manifest)
    assert by_origin(c) == Counter({"iiif_cudl": 2}) and http.calls[("manifest", derived)] == 1
    assert "no_listed_image cudl_manifest_derived" in c.notes[im.NOTES_KEY]["misses"]
    listed = cand("lit:work:p1:d4", "literature", cids=["Test_Lib_D_4", "Cambridge_CUL_T_S_NS_9_12"])
    world.resolve([listed], fetch_manifest=http.manifest)
    assert "iiif_cudl" not in by_origin(listed)                  # something was listed: no derived manifest


def test_resolve_pass_through_verifies_given_image(world: World) -> None:
    """Built-row sources keep their given image; a missing file becomes ``missing``."""
    given = ImageRef(image_id="apd_local:0", origin="apd_local", path=str(world.files["page.jpg"]), status="present")
    absent = ImageRef(image_id="apd_local:0", origin="apd_local", path=str(world.root / "apd/none.jpg"),
                      status="present")
    ok, bad = cand("apd_v3:x", "apd_v3", images=[given]), cand("apd_v3:y", "apd_v3", images=[absent])
    world.resolve([ok, bad])
    assert ok.images[0].status == "present" and ok.images[0].sha256 == sha(world.data["page.jpg"])
    assert bad.images[0].status == "missing"


# ---------------------------------------------------------------------------------------------------- fetching


def url_cand(row_id: str, urls: Sequence[str], cid: str = "Test_Lib_F_6") -> Candidate:
    """A candidate whose images are URLs only (as after resolution).

    :param row_id: Row id.
    :param urls: Image URLs.
    :param cid: Canonical id (image directory).
    :return: Candidate.
    """
    refs = [ImageRef(image_id=f"{im.origin_of_url(u)}:{i}", origin=im.origin_of_url(u), url=u, status="listed")
            for i, u in enumerate(urls)]
    return cand(row_id, "pgp", cids=[cid], images=refs)


def test_fetch_head_404_skips_get(tmp_path: Path) -> None:
    """A bucket URL answering 404 to HEAD is ``head_404`` and never fetched; other URLs get no HEAD."""
    gone, other = f"{BUCKET}/gone_1r.jpg", "https://img-cudl.example.invalid/iiif/x/full/3245,/0/default.jpg"
    http = Http(bodies={other: encode(pattern_image(1000, 900, 1))}, heads={gone: 404})
    c = url_cand("pgp:1", [gone, other])
    counts = fetch([c], tmp_path, http)
    assert c.images[0].status == "head_404" and http.calls[("get", gone)] == 0
    assert c.images[1].status == "fetched" and http.calls[("head", other)] == 0
    assert counts["head_404"] == 1 and counts["fetched"] == 1


def test_fetch_exif_rotation_upright_file_source_keys(tmp_path: Path) -> None:
    """An EXIF-rotated JPEG is stored upright (re-encoded); sha256 is the written file's, phash the source's."""
    url = "https://iiif.other.example.invalid/img/rot.jpg"
    source = pattern_image(1000, 900, 4)
    body = encode(source, orientation=6)
    http = Http(bodies={url: body})
    c = url_cand("pgp:2", [url])
    fetch([c], tmp_path, http)
    (ref,) = c.images
    assert ref.status == "fetched" and (ref.width, ref.height) == (900, 1000)
    with Image.open(ref.path) as saved:
        assert saved.size == (900, 1000) and int(saved.getexif().get(0x0112, 1)) == 1
    assert ref.sha256 == sha(Path(ref.path).read_bytes()) and ref.sha256 != sha(body)
    with Image.open(io.BytesIO(body)) as raw:
        assert ref.phash == im.phash_image(raw)
    assert "exif_orientation 6" in ref.note and "reencoded" in ref.note
    assert Path(ref.path) == tmp_path / "images" / "Test_Lib_F_6" / "0.jpg"


def test_fetch_same_bytes_stored_once(tmp_path: Path) -> None:
    """Identical bytes under two URLs (two candidates, then twice in one candidate) are stored once; the later
    reference of one candidate is a duplicate."""
    body = encode(pattern_image(1200, 1000, 5))
    u1, u2, u3 = (f"https://iiif.other.example.invalid/img/{n}.jpg" for n in ("a", "b", "c"))
    http = Http(bodies={u1: body, u2: body, u3: body})
    first, second = url_cand("pgp:3", [u1], cid="Test_Lib_G_7"), url_cand("pgp:4", [u2, u3], cid="Test_Lib_H_8")
    fetch([first, second], tmp_path, http)
    paths_ = {r.path for r in first.images + second.images}
    assert len(paths_) == 1 and len(list((tmp_path / "images").rglob("*.jpg"))) == 1
    assert second.images[0].status == "fetched" and "same bytes as" in second.images[0].note
    assert second.images[1].status == "duplicate" and second.images[1].sha256 == second.images[0].sha256


def test_fetch_reuses_local_file_with_same_bytes(world: World) -> None:
    """A URL whose bytes equal a candidate's local file reuses that file instead of storing a copy."""
    url = "https://iiif.other.example.invalid/img/copy.jpg"
    local = ImageRef(image_id="fjp_local:0", origin="fjp_local", path=str(world.files["A_1_1r.jpg"]),
                     sha256=sha(world.data["A_1_1r.jpg"]), status="present")
    c = cand("pgp:5", "pgp", cids=["Test_Lib_A_1"],
             images=[local, ImageRef(image_id="iiif_other:1", origin="iiif_other", url=url, status="listed")])
    fetch([c], world.work, Http(bodies={url: world.data["A_1_1r.jpg"]}))
    assert c.images[1].path == str(world.files["A_1_1r.jpg"]) and c.images[1].status == "duplicate"
    assert not (world.work / "images" / "Test_Lib_A_1").exists()


def test_fetch_too_small_kept(tmp_path: Path) -> None:
    """A long side under the minimum gives ``too_small``; the file is kept."""
    url = "https://iiif.other.example.invalid/img/small.jpg"
    c = url_cand("pgp:6", [url])
    fetch([c], tmp_path, Http(bodies={url: encode(pattern_image(600, 400, 6))}))
    (ref,) = c.images
    assert ref.status == "too_small" and Path(ref.path).is_file() and (ref.width, ref.height) == (600, 400)


def analytic(width: int, height: int) -> Image.Image:
    """The deterministic analytic image the pinned hashes were computed on with the ledger's code.

    :param width: Pixels.
    :param height: Pixels.
    :return: Greyscale image.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    a = 0.5 * xx / width + 0.3 * np.sin(yy / 9.0) + 0.2 * np.cos(xx / 13.0)
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8), "L")


def test_phash_matches_ledger_values_and_is_recorded_on_fetch(tmp_path: Path) -> None:
    """The phash equals the ledger's (pinned values from its implementation); a fetched PNG is stored as JPEG while
    its phash is the source's."""
    assert im.phash_image(analytic(320, 240)) == "a5878f8f87838783"
    url = "https://iiif.other.example.invalid/img/analytic.png"
    c = url_cand("pgp:7", [url])
    fetch([c], tmp_path, Http(bodies={url: encode(analytic(1200, 900).convert("RGB"), fmt="PNG")}))
    (ref,) = c.images
    assert ref.phash == "a0849f859f8f9f84" and ref.status == "fetched" and "reencoded" in ref.note
    with Image.open(ref.path) as saved:
        assert saved.format == "JPEG"
    assert im.phash_distance(ref.phash, im.phash_image(Image.open(ref.path))) <= 4


@pytest.mark.skipif(not LEDGER_LIB.is_file(), reason="the ledger's local library is not in this checkout")
def test_phash_identical_to_ledger_library() -> None:
    """Direct parity with the ledger's own ``phash_image`` when its library file is available."""
    spec = importlib.util.spec_from_file_location("ledger_registry_lib", LEDGER_LIB)
    assert spec is not None and spec.loader is not None
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    for seed in range(12):
        img = pattern_image(200 + 37 * seed, 150 + 23 * seed, seed)
        assert im.phash_image(img) == lib.phash_image(img)


def test_fetch_is_resumable(tmp_path: Path) -> None:
    """A second run requests nothing: fetched files and 404s come from the state log (a ``missing`` is retried)."""
    good, gone, flaky = ("https://iiif.other.example.invalid/img/good.jpg", f"{BUCKET}/gone_1v.jpg",
                         "https://iiif.other.example.invalid/img/flaky.jpg")
    http = Http(bodies={good: encode(pattern_image(1000, 1000, 7))}, heads={gone: 404})
    fetch([url_cand("pgp:8", [good, gone, flaky])], tmp_path, http, retries=0)
    first_calls = http.total()

    rerun = url_cand("pgp:8", [good, gone, flaky])
    http.bodies[flaky] = encode(pattern_image(1000, 1000, 8))
    counts = fetch([rerun], tmp_path, http, retries=0)
    assert http.total() == first_calls + 1 and http.calls[("get", flaky)] == 2   # only the failed URL again
    assert [r.status for r in rerun.images] == ["fetched", "head_404", "fetched"]
    assert counts["reused_fetched"] == 1 and counts["reused_head_404"] == 1

    third = url_cand("pgp:8", [good, gone, flaky])
    before = http.total()
    fetch([third], tmp_path, http)
    assert http.total() == before
    assert [r.path for r in third.images] == [r.path for r in rerun.images]


def test_fetch_retries_and_permanent_errors(tmp_path: Path) -> None:
    """Transient errors are retried up to ``retries``; a permanent error stops at once; both are recorded."""
    flaky, dead = "https://iiif.other.example.invalid/img/f.jpg", "https://iiif.other.example.invalid/img/d.jpg"
    body = encode(pattern_image(1000, 1000, 9))
    attempts: Counter = Counter()

    def get(url: str) -> bytes:
        """GET stub: two transient failures for ``flaky``, a permanent one for ``dead``.

        :param url: URL.
        :return: Body.
        """
        attempts[url] += 1
        if url == dead:
            raise im.PermanentFetchError("HTTP 403")
        if attempts[url] <= 2:
            raise RuntimeError("timeout")
        return body

    c = url_cand("pgp:9", [flaky, dead])
    im.fetch_missing([c], tmp_path, head=lambda u: 200, get=get, delay=0, retries=2, sleep=lambda s: None)
    assert c.images[0].status == "fetched" and attempts[flaky] == 3
    assert c.images[1].status == "missing" and attempts[dead] == 1 and "PermanentFetchError" in c.images[1].note


def test_host_pacer_spaces_requests_per_host() -> None:
    """Two requests to one host wait ``delay``; another host does not wait."""
    now = [100.0]
    slept: List[float] = []
    pacer = im.HostPacer(1.0, sleep=slept.append, clock=lambda: now[0])
    pacer.wait("https://a.example.invalid/1")
    pacer.wait("https://a.example.invalid/2")
    pacer.wait("https://b.example.invalid/1")
    assert slept == [1.0]


# ---------------------------------------------------------------------------------------------------- fetch cap


def box_candidate(names: Sequence[str], cid: str = "Cambridge_CUL_T_S_NS_10_1", shelfmark: str = "T-S NS 10.1",
                  source: str = "fjp_scrape") -> Candidate:
    """A legacy-store candidate listing bucket photographs by file name only.

    :param names: File names.
    :param cid: Canonical id.
    :param shelfmark: Shelf mark.
    :param source: Ledger source.
    :return: Candidate (references ``listed``, ids assigned).
    """
    refs = [ImageRef(image_id=f"fjp_bucket:{i}", origin="fjp_bucket", url=f"{BUCKET}/{n}", status="listed")
            for i, n in enumerate(names)]
    c = cand(f"{cid}:s0", source, cids=[cid], images=refs)
    c.shelfmark = shelfmark
    return c


def test_stem_reading_through_the_facade() -> None:
    """A stem names the mark (tier 1), a folio-1 side of it (tier 2), or something else (0, a child leaf included)."""
    keys, insts = im.own_mark_keys(box_candidate([]))
    assert im.stem_mark("T_S_NS_10_1_1r") == ("T S NS 10 1", "1r")
    assert im.stem_mark("T_S_NS_10_1_2v") == ("T S NS 10 1 2", "2v")
    assert im.stem_tier("T_S_NS_10_1.jpg", keys, insts) == 1
    assert im.stem_tier("T-S_NS_10_1r.jpg", keys, insts) == 1
    assert im.stem_tier("T_S_NS_10_1_1v.jpg", keys, insts) == 2
    assert im.stem_tier("T_S_NS_10_1_2v.jpg", keys, insts) == 0
    assert im.stem_tier("T_S_NS_10_11_1r.jpg", keys, insts) == 0
    assert im.stem_tier("T_S_NS_101_1r.jpg", keys, insts) == 0


def test_fetch_cap_keeps_stems_naming_the_mark(tmp_path: Path) -> None:
    """12 listed photographs, 3 named by the candidate's own mark: those 3 are kept, 9 become ``box_listing``."""
    others = [f"T_S_NS_10_{n}_1r.jpg" for n in range(2, 11)]
    c = box_candidate(["T_S_NS_10_1.jpg", "T_S_NS_10_1r.jpg", "T-S_NS_10_1v.jpg"] + others)
    counts = im.cap_fetch_selection(c, 8)
    assert [r.status for r in c.images] == ["listed"] * 3 + ["box_listing"] * 9
    assert counts == {"listed": 12, "selected": 3, "box_listing": 9}
    assert "file name names another mark" in c.images[5].note
    http = Http(heads={r.url: 404 for r in c.images if r.url})
    fetch([c], tmp_path, http)
    assert sum(n for (kind, _), n in http.calls.items() if kind == "head") == 3   # only the selected are requested


def test_fetch_cap_falls_back_to_sides_of_the_mark() -> None:
    """With fewer than N exact names, the side-named photographs of the same mark fill up (never other marks)."""
    names = ["T_S_NS_10_1.jpg", "T_S_NS_10_1_1r.jpg", "T_S_NS_10_1_1v.jpg"]
    names += [f"T_S_NS_10_{n}_1r.jpg" for n in range(2, 12)]
    c = box_candidate(names)
    counts = im.cap_fetch_selection(c, 2)
    assert [r.status for r in c.images[:3]] == ["listed", "listed", "box_listing"]   # exact first, then a side
    assert "over the cap" in c.images[2].note
    assert all(r.status == "box_listing" for r in c.images[3:])
    assert counts == {"listed": 13, "selected": 2, "box_listing": 11}
    wide = box_candidate(names)
    assert im.cap_fetch_selection(wide, 8)["selected"] == 3                          # unrelated never fill


def test_fetch_cap_unmatched_box_is_a_review_case() -> None:
    """No file name names the candidate's mark: every URL becomes ``box_listing`` and the candidate is flagged."""
    c = box_candidate([f"T_S_AS_15_{n}_1r.jpg" for n in range(100, 112)], cid="Cambridge_CUL_T_S_AS_15",
                      shelfmark="T-S AS 15")
    counts = im.cap_fetch_selection(c, 8)
    assert all(r.status == "box_listing" for r in c.images)
    assert counts == {"listed": 12, "selected": 0, "box_listing": 12, "box_listing_unmatched": True}
    small = box_candidate([f"T_S_AS_15_{n}_1r.jpg" for n in range(100, 104)], cid="Cambridge_CUL_T_S_AS_15")
    assert im.cap_fetch_selection(small, 8) == {"listed": 4, "selected": 4, "box_listing": 0}   # under the cap


def test_fetch_cap_keeps_manifest_canvases_and_local_files(world: World) -> None:
    """Canvases of the candidate's own manifests count as naming its mark; local files are never capped; the
    counts land in the notes and the inventory."""
    canvases = [ImageRef(image_id=f"iiif_cudl:{i}", origin="iiif_cudl", canvas_index=i, status="listed",
                         url=f"https://img-cudl.example.invalid/iiif/x{i}/full/3245,/0/default.jpg") for i in range(3)]
    local = ImageRef(image_id="fjp_local:3", origin="fjp_local", path=str(world.files["A_1_1r.jpg"]), status="present")
    codex = [ImageRef(image_id=f"iiif_nli:{4 + i}", origin="iiif_nli", status="listed",
                      url=f"https://bucket.example.invalid/store/KTIV/990/{i:03d}_Cover.jpg") for i in range(8)]
    c = cand("pgp:7", "pgp", cids=["Cambridge_CUL_T_S_13J35_3"], images=canvases + [local] + codex)
    assert im.cap_fetch_selection(c, 8) == {"listed": 12, "selected": 4, "box_listing": 8}
    assert [r.status for r in c.images[:4]] == ["listed", "listed", "listed", "present"]
    c.notes[im.NOTES_KEY] = {"box_listing": 8}
    assert "| pgp | 1 | 0 | 12 | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 8 |" in im.inventory([c])


# --------------------------------------------------------------------------------------------- inventory + CLI


def test_inventory_counts_per_source() -> None:
    """The inventory row of a source counts candidates, images by status and distinct sha256."""
    pgp1 = cand("pgp:1", "pgp", images=[
        ImageRef(image_id="fjp_local:0", origin="fjp_local", path="/p/a.jpg", sha256="s1", status="present"),
        ImageRef(image_id="iiif_cudl:1", origin="iiif_cudl", url="https://x.example.invalid/1.jpg"),
        ImageRef(image_id="fjp_bucket:2", origin="fjp_bucket", url="https://x.example.invalid/2.jpg",
                 status="head_404")])
    pgp2 = cand("pgp:2", "pgp")
    pgp2.notes[im.NOTES_KEY] = {"manifests_unexpanded": [{"url": "https://x.example.invalid/m", "reason": "r"}]}
    ktiv = cand("1:FL1", "ktiv", images=[
        ImageRef(image_id="ktiv_zip:0", origin="ktiv_zip", path="/k/1.jpg", sha256="s2", status="too_small"),
        ImageRef(image_id="iiif_nli:1", origin="iiif_nli", url="https://x.example.invalid/3.jpg", path="/w/1.jpg",
                 sha256="s3", status="fetched"),
        ImageRef(image_id="iiif_nli:2", origin="iiif_nli", url="https://x.example.invalid/4.jpg", path="/w/1.jpg",
                 sha256="s3", status="duplicate")])
    text = im.inventory([pgp1, pgp2, ktiv])
    assert text.startswith("# Stage 1 image inventory")
    assert ("| source | candidates | none | listed | present | fetched | url-only | missing | too_small | head_404 | "
            "duplicate | box_listing | distinct sha256 | usable cands | box unmatched cands |") in text
    assert "| pgp | 2 | 1 | 3 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 3 |" in text
    assert "| ktiv | 1 | 0 | 3 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 2 | 1 | 0 | 0 | 3 |" in text
    assert "| **total** | 3 | 1 | 6 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 3 | 2 | 0 | 1 | 3 |" in text
    assert "| pgp | iiif_cudl | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |" in text
    assert "| pgp | manifest:r | 1 |" in text


def test_main_inventory_is_offline_and_writes_stage1(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--inventory`` resolves without any request, writes stage 1 and the inventory; a limited run keeps earlier
    stage-1 images of the other candidates."""
    def no_network(*args: Any, **kwargs: Any) -> Any:
        """Fail on any request.

        :param args: Ignored.
        :param kwargs: Ignored.
        :raises AssertionError: Always.
        """
        raise AssertionError("network used in inventory mode")

    for name in ("http_get_json", "http_head", "http_get"):
        monkeypatch.setattr(im, name, no_network)
    stage0 = [
        cand("pgp:101", "pgp", cids=["Test_Lib_A_1"], pgpids=["101"]),
        cand("990001:FL123", "ktiv", ktiv_id="990001", images=[ImageRef(
            image_id="ktiv:0", origin="ktiv", url="https://nli.example.invalid/IIIF/FL123/full/2800,/0/default.jpg")]),
        cand("apd_v3:x", "apd_v3", images=[ImageRef(image_id="apd_local:0", origin="apd_local",
                                                    path=str(world.files["page.jpg"]), status="present")]),
    ]
    write_jsonl(paths.stage_path(world.work, "candidates"), stage0)
    im.main(["--inventory"] + world.cli_args())
    stage1 = list(read_jsonl(paths.stage_path(world.work, "images"), Candidate))
    assert [c.row_id for c in stage1] == ["pgp:101", "990001:FL123", "apd_v3:x"]
    assert by_origin(stage1[1]) == Counter({"ktiv_zip": 1})
    report = (world.work / im.INVENTORY_FILE).read_text(encoding="utf-8")
    assert "| apd_v3 | 1 | 0 | 1 | 1 |" in report and "| ktiv | 1 | 0 | 1 | 1 |" in report

    im.main(["--inventory", "--sources", "ktiv"] + world.cli_args())
    kept = list(read_jsonl(paths.stage_path(world.work, "images"), Candidate))
    assert [c.row_id for c in kept] == ["pgp:101", "990001:FL123", "apd_v3:x"]
    assert [len(c.images) for c in kept] == [len(c.images) for c in stage1]


def test_main_limit_is_per_source(world: World, capsys: pytest.CaptureFixture) -> None:
    """``--limit N`` takes at most N candidates of each source, in stage-0 order."""
    stage0 = [cand(f"apd_v3:{i}", "apd_v3", images=[ImageRef(image_id="apd_local:0", origin="apd_local",
                                                             path=str(world.files["page.jpg"]))]) for i in range(3)]
    stage0 += [cand(f"Test_Lib_A_1:{i}", "fjp", cids=["Test_Lib_A_1"]) for i in range(3)]
    write_jsonl(paths.stage_path(world.work, "candidates"), stage0)
    im.main(["--inventory", "--limit", "2"] + world.cli_args())
    stage1 = list(read_jsonl(paths.stage_path(world.work, "images"), Candidate))
    assert [c.row_id for c in stage1] == ["apd_v3:0", "apd_v3:1", "Test_Lib_A_1:0", "Test_Lib_A_1:1"]
    assert "resolved 4/6 candidates" in capsys.readouterr().out
