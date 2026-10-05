# File name: test_reserve_test_documents.py
# Date: 10/4/26
# Author: Isaac Godfried. Coded originally by Claude Fable 5.1.
"""Tests for reserving test documents ahead of a training-set build."""
import json

from src.datasets.evaluations.helper_eval_scripts import reserve_test_documents as rtd

CANONICAL = {"10": ["Cambridge_CUL_T_S_8J1_1"], "11": ["Cambridge_CUL_T_S_8J1_2", "New_York_JTS_ENA_1_2"], "12": []}


def test_never_trained_checks_the_pgp_key_and_every_canonical_id():
    exposure = {"pgp:10", "New_York_JTS_ENA_1_2", "Oxford_Bodleian_MS_heb_a_1_1"}
    assert rtd.never_trained(["10", "11", "12", "13"], CANONICAL, exposure) == {"12", "13"}
    assert rtd.never_trained(["10", "11"], CANONICAL, set()) == {"10", "11"}


def test_with_canonical_ids_keeps_documents_missing_from_the_index_by_pgp_id():
    ids, pgpids = rtd.with_canonical_ids(["11", 12], CANONICAL)
    assert ids == {"Cambridge_CUL_T_S_8J1_2", "New_York_JTS_ENA_1_2"} and pgpids == {"11", "12"}


def test_manifest_documents_reads_one_split(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [{"pgpid": "10", "canonical_id": "A_1", "split": "train"}, {"pgpid": 11, "canonical_id": "B_2", "split": "val"},
            {"pgpid": "11", "canonical_id": "B_2", "split": "val"}, {"pgpid": "12", "canonical_id": "C_3", "split": "val"}]
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert rtd.manifest_documents(manifest) == ({"B_2", "C_3"}, {"11", "12"})
    assert rtd.manifest_documents(manifest, "train") == ({"A_1"}, {"10"})


def test_sample_documents_collects_every_job(tmp_path):
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps([{"doc_id": "A_1", "image_index": 0, "pgpid": "10", "set": "unseen"},
                                  {"doc_id": "A_1", "image_index": 1, "pgpid": "10", "set": "unseen"},
                                  {"doc_id": "D_4", "image_index": 0, "pgpid": 14, "set": "ts_as"}]), encoding="utf-8")
    assert rtd.sample_documents(sample) == ({"A_1", "D_4"}, {"10", "14"})
