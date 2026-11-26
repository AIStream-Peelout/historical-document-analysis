"""Build a fresh private edition export and verify its photograph and answer through the trainer's row loader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import build_rows as br, release_audit as audit
from .edition_candidate import index_unique
from .edition_lines import digest


def verify_export(manifest: list[dict], export: Path) -> int:
    """Compare the actual distinct parquet rows and JPEGs with their reviewed targets and split.

    :param manifest: Selected builder manifest.
    :param export: Images-once or trainer export.
    :return: Number of distinct verified rows.
    """
    from ..images_once import image_sha1
    rows = audit.load_export(export)
    expected = index_unique(manifest, "stem")
    if {r["stem"] for r in rows} != set(expected):
        raise ValueError("Exported and selected stems differ")
    groups = {}
    seen_views = set()
    seen_lines = set()
    for row in rows:
        m = expected[row["stem"]]
        edition = m["edition"]
        line_refs = {(edition["pdf_sha256"], line_id) for line_id in edition["line_ids"]}
        if (edition["view_sha256"] in seen_views or seen_lines.intersection(line_refs)
                or len(line_refs) != len(edition["line_ids"])):
            raise ValueError(f"Repeated photographed view or printed line for {row['stem']}")
        seen_views.add(edition["view_sha256"])
        seen_lines.update(line_refs)
        photo = Path(row["image_path"])
        if (digest(row["answer"]) != m["approval"]["answer_sha256"]
                or br.file_sha256(str(photo)) != edition["view_sha256"]
                or image_sha1(photo.read_bytes()) != row["image_sha1"]):
            raise ValueError(f"Photo or answer changed for {row['stem']}")
        split = audit.SPLITS[m["split"]]
        if (row["split"] != split or groups.setdefault(m["document_group"], split) != split
                or (split == "val" and m["rnl"])):
            raise ValueError(f"Document split mismatch for {row['stem']}")
    return len(rows)


def main() -> None:
    """Build both ordinary and trainer parquet layouts without staging or releasing either."""
    from .. import images_once
    from ..colab import derive_novel_v2 as trainer
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-only", action="store_true", help="Write only the dry-run selection")
    args = ap.parse_args()
    bundle, out = args.bundle.resolve(), args.out.resolve()
    run_path = bundle / "run.json"
    run = json.loads(run_path.read_text())
    names = {"candidates.jsonl", "pairs.jsonl", "verdicts.jsonl", "review_queue.jsonl"}
    if run["kind"] != "edition_candidate" or set(run["files"]) != names:
        raise ValueError("Incomplete edition bundle")
    for name, sha in run["files"].items():
        if br.file_sha256(str(bundle / name)) != sha:
            raise ValueError(f"Changed bundle member {name}")
    for item in run["inputs"].values():
        if br.file_sha256(item["path"]) != item["sha256"]:
            raise ValueError("An edition input changed after preparation")
    dest = {suffix: out.with_name(out.name + suffix) for suffix in
            ("", "_dryrun", "_build", "_derived", "_images_once", "_train", ".building", ".previous")}
    if any(path.exists() for path in dest.values()):
        raise FileExistsError(f"Use a new build name for {out}")
    trainer.refuse_boot_disk(dest["_train"], "private candidate training rows")
    report = dest["_build"]
    dest["_dryrun"].mkdir()  # reserve the name across simultaneous dry and actual builds
    common = ["--pairs", str(bundle / "pairs.jsonl"), "--candidates", str(bundle / "candidates.jsonl"),
              "--verdicts", str(bundle / "verdicts.jsonl"), "--out", str(out), "--tiers", "novel_strict",
              "--remedy-classes", "assemble,assemble_crop"]
    br.main(common + ["--dry-run"])
    if args.dry_only:
        return
    dry = br.load_jsonl(dest["_dryrun"] / "manifest.jsonl")
    if not dry:
        raise ValueError("No source-reviewed rows cleared the ordinary row builder")
    report.mkdir()
    br.main(common)
    manifest_path = report / "manifest.jsonl"
    manifest = br.load_jsonl(manifest_path)
    if [r["pair_id"] for r in dry] != [r["pair_id"] for r in manifest]:
        raise ValueError("Dry and final build decisions differ")
    images = dest["_images_once"]
    images_once.export_images_once(out, images)
    verify_export(manifest, images)
    plan = trainer.prepare_rows(images, keep_rnl=True, manifest_path=manifest_path)
    trainer.write_rows(plan, images, dest["_train"], manifest_path, trainer.AUTO, trainer.SEED,
                       workers=1, mixture_name=dest["_train"].name)
    count = verify_export(manifest, dest["_train"])
    input_hashes = {"dataset": out.name, "run": str(bundle), "bundle_sha256": br.file_sha256(str(run_path)),
                    "manifest_sha256": br.file_sha256(str(manifest_path)), "files": run["files"],
                    "inputs": run["inputs"], "training_export": str(dest["_train"]), "verified_export_rows": count,
                    "parquet_sha256": {p.name: br.file_sha256(str(p)) for p in audit.export_row_files(dest["_train"])},
                    "release_audit": "not run"}
    (report / "INPUTS.json").write_text(json.dumps(input_hashes, indent=2) + "\n")
    print(f"{count} distinct source-reviewed rows verified in {dest['_train']}")


if __name__ == "__main__":
    main()
