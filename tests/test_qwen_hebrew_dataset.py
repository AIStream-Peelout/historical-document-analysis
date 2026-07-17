"""Dataset integrity tests for the Qwen3-VL Hebrew fine-tuning pipeline.

Two layers:

- Pure-function tests (always run): normalization, split determinism,
  benchmark-holdout discovery.
- Artifact tests (run once the dataset has been built): schema, benchmark
  leakage, non-empty targets, image files resolvable, resolution floor.
"""

from pathlib import Path

import pytest

from src.finetuning.qwen_hebrew.build_dataset import (
    DEFAULT_OUTPUT_DIR,
    MAX_TARGET_TOKENS,
    MIN_CROP_WIDTH_PX,
    SMOKE_SIZE,
    benchmark_holdout_stems,
    normalize_target,
    pick_val_stems,
)

_ARTIFACT_EXISTS = (DEFAULT_OUTPUT_DIR / "dataset_dict.json").exists()

requires_artifact = pytest.mark.skipif(
    not _ARTIFACT_EXISTS,
    reason="dataset not built yet — run build_dataset.py first",
)


# ── Pure-function tests ───────────────────────────────────────────────────────

def test_normalize_target_nfc_and_strip() -> None:
    """NFC normalization must preserve internal line breaks, strip outer space."""
    raw = "  שלום\nעולם  "
    out = normalize_target(raw)
    assert out == "שלום\nעולם"
    # decomposed alef + combining qamats must compose to a stable NFC form
    import unicodedata

    decomposed = "אָ"
    assert normalize_target(decomposed) == unicodedata.normalize("NFC", decomposed)


def test_benchmark_holdout_stems_found() -> None:
    """The 66 benchmark stems must be discoverable."""
    stems = benchmark_holdout_stems()
    assert len(stems) >= 60
    assert all("_page_" not in s for s in stems), "holdout stems must be GT stems"


def test_pick_val_stems_deterministic_and_stratified() -> None:
    """Same seed → same validation pages; volumes get spread coverage."""
    stems = [f"{vol:02d}_{i}" for vol in range(1, 11) for i in range(2, 30)]
    a = pick_val_stems(stems, 20, seed=123)
    b = pick_val_stems(stems, 20, seed=123)
    assert a == b
    assert len(a) == 20
    volumes = {s.split("_")[0] for s in a}
    assert len(volumes) == 10, "expected every volume represented"


def test_pick_val_stems_terminates_when_overasked() -> None:
    """Asking for more pages than exist must return everything, not hang."""
    stems = [f"01_{i}" for i in range(2, 12)]
    picked = pick_val_stems(stems, n_pages=10**6, seed=7)
    assert picked == set(stems)


# ── Artifact tests ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    """Load the built dataset artifact once per module."""
    from datasets import load_from_disk

    return load_from_disk(str(DEFAULT_OUTPUT_DIR))


@requires_artifact
def test_no_benchmark_leakage(dataset) -> None:
    """No benchmark stem may appear in ANY split — the cardinal rule."""
    holdout = benchmark_holdout_stems()
    for split_name, split in dataset.items():
        leaked = set(split["stem"]) & holdout
        assert not leaked, f"benchmark stems leaked into {split_name}: {sorted(leaked)[:5]}"


@requires_artifact
def test_schema_and_targets(dataset) -> None:
    """Required columns exist; targets are non-empty, Hebrew, within budget."""
    for split_name, split in dataset.items():
        for col in ("image", "question", "answer", "task", "section",
                    "stem", "label_source", "target_tokens"):
            assert col in split.column_names, f"{split_name} missing {col}"
        assert all(t <= MAX_TARGET_TOKENS for t in split["target_tokens"])
        answers = split["answer"]
        assert all(a and a.strip() for a in answers), f"empty answer in {split_name}"
        sampled = answers[:50]
        hebrew = sum(1 for a in sampled if any("א" <= c <= "ת" for c in a))
        assert hebrew >= 0.9 * len(sampled), f"{split_name}: answers don't look like Hebrew"


@requires_artifact
def test_smoke_split_fixed_size(dataset) -> None:
    """Smoke split must hold the fixed overfit set."""
    assert len(dataset["smoke"]) == SMOKE_SIZE
    assert set(dataset["smoke"]["task"]) == {"crop_transcribe"}


@requires_artifact
def test_images_resolvable_and_wide_enough(dataset) -> None:
    """Sampled images must exist on disk and respect the resolution floor."""
    from datasets import Image as HFImage

    val = dataset["val"].cast_column("image", HFImage(decode=False))
    for i in range(0, len(val), max(len(val) // 10, 1)):
        row = val[i]
        img = row["image"]
        path = img.get("path")
        assert (path and Path(path).exists()) or img.get("bytes"), (
            f"image for {row['stem']}/{row['section']} unresolvable"
        )
        assert row["image_width"] >= MIN_CROP_WIDTH_PX


@requires_artifact
def test_label_source_is_gold(dataset) -> None:
    """Phase 1 data is all gold-labelled (schema ready for Phase 2/3 sources)."""
    for split in dataset.values():
        assert set(split["label_source"]) <= {"gold"}
