"""Suite-wide fixtures.

Every test runs with the LM Studio yield gate switched off and pointed at a temporary directory, so tests that
reach hooked code (``lms_transcriber``, ``coreference_resolver``, ...) never shell out to ``lms ps`` and never
append to the real ``logs/lms_yield/*.jsonl`` dry-run data. ``tests/test_lms_yield_gate.py`` turns the gate back on
per test (``monkeypatch.setenv`` or an explicit ``YieldConfig``).
"""

from pathlib import Path
from typing import Iterator

import pytest

from src.models.ocr import lms_yield_gate


@pytest.fixture(autouse=True)
def isolate_lms_yield_gate(tmp_path_factory: pytest.TempPathFactory,
                           monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Pin ``LMS_YIELD_MODE=off``, a temporary ``LMS_YIELD_DIR`` and no ``.env``, with fresh gate caches.

    :param tmp_path_factory: Pytest temp-dir factory (keeps each test's own ``tmp_path`` untouched).
    :param monkeypatch: Pytest monkeypatch fixture.
    :returns: The temporary gate state directory (created only if a test writes to it).
    :rtype: Iterator[Path]
    """
    state_dir = tmp_path_factory.mktemp("lms_yield_state") / "lms_yield"
    monkeypatch.setenv("LMS_YIELD_MODE", "off")
    monkeypatch.setenv("LMS_YIELD_DIR", str(state_dir))
    monkeypatch.setattr(lms_yield_gate, "DEFAULT_ENV_FILE", state_dir.parent / "no-such.env")
    lms_yield_gate.reset_caches()
    yield state_dir
    lms_yield_gate.reset_caches()
