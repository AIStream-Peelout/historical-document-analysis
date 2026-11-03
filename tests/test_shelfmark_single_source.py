"""T5 single-source lint: shelf-mark normalisation lives in ``src.datasets.shelfmarks`` and its engines only.

An AST scan of the tracked and untracked-not-ignored ``*.py`` files under ``src/`` (``git ls-files --cached --others
--exclude-standard``; not ``logs/``, not ``tests/``;
``src/datasets/raw_data`` is skipped) fails on any of:

* a ``[^a-z0-9]`` squash: ``re.sub(<negated alphanumeric class>, "", …)`` or ``<compiled>.sub("", …)``;
* a new function named ``*shelf*key*`` (which covers ``*shelfmark_key*``), ``loose_key``, ``canonical_id`` or
  ``shelfmark_core_id``;
* a direct import of ``ShelfmarkNormalizer`` / the ``genizah_normalizer`` module (any name imported from it) or of
  ``institution_tokens``, including the bypasses: an ``institution_tokens`` public name (:data:`TOKEN_NAMES`)
  imported from ``merge_shelfmarks`` or from any module other than the facade and the engines; the facade's private
  ``_engine`` imported from ``src.datasets.shelfmarks``; ``ShelfmarkNormalizer`` / ``_engine`` read as an attribute
  of a module alias of the facade or of ``merge_shelfmarks``; a :data:`TOKEN_NAMES` name read as an attribute of a
  ``merge_shelfmarks`` alias (``ms.combine``); ``importlib.import_module("<…engine module>")`` with a literal name,

outside the facade, the two engines, the merge recipe (``merging/merge_shelfmarks.py``) and the allowlist. The
public :data:`ALLOWLIST` names today's git-tracked callers (2026-11-03, from the audit catalogue, each confirmed by
this scan) with the exact violations they may keep. It only shrinks: a stale entry fails
:func:`test_allowlist_has_no_stale_entries`, so a migrated caller must leave the list in the same PR.

Files that are not git-tracked are not scanned, so they never appear in the public list. A clone that keeps such
files can list them in the optional, git-ignored :data:`LOCAL_ALLOWLIST_FILE` (``{"allowlist": {path: [violation,
…]}}``); it is merged into the allowlist and matters only when git is unavailable and the scan falls back to every
file on disk.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
#: Directories never scanned (data, caches).
SKIP_DIRS = frozenset({"raw_data", "__pycache__", "node_modules", ".git"})
#: The facade, the two engines and the merge recipe (``merge_shelfmarks``, "the merge" of the plan) may use the
#: engines freely. Every other file, including the rest of ``src/datasets/merging/``, is scanned.
EXEMPT_FILES = frozenset({
    "src/datasets/shelfmarks.py",
    "src/datasets/document_models/genizah_normalizer.py",
    "src/datasets/merging/institution_tokens.py",
    "src/datasets/merging/merge_shelfmarks.py",
})

#: The facade, imported by its absolute name, and the merge recipe module.
FACADE_MODULE = "src.datasets.shelfmarks"
MERGE_MODULE = "src.datasets.merging.merge_shelfmarks"
#: Public names of ``institution_tokens`` (plus the registry token tuple): importing one from anywhere but the facade
#: or an engine is an ``institution_tokens`` import (e.g. ``from …merge_shelfmarks import combine``).
TOKEN_NAMES = frozenset({"combine", "institution_token", "resolve_token", "canonical_id", "split_id", "BL_TOKEN",
                         "OXFORD_TOKEN", "SERIES_TOKENS", "_REGISTRY_TOKENS"})
#: Facade / merge attributes that hand out an engine: name -> violation.
_ENGINE_ATTRS = {"ShelfmarkNormalizer": "import:ShelfmarkNormalizer", "_engine": "import:institution_tokens"}
#: Engine module name suffix -> violation (for ``importlib.import_module`` with a literal name).
_ENGINE_MODULES = {"genizah_normalizer": "import:ShelfmarkNormalizer", "institution_tokens": "import:institution_tokens"}
#: Optional, git-ignored supplement of :data:`ALLOWLIST` for files a local clone keeps outside git.
LOCAL_ALLOWLIST_FILE = REPO / "tests/fixtures/shelfmark_allowlist.local.json"

_SQUASH_CLASS = re.compile(r"\[\^(?:a-z0-9|0-9a-z|A-Za-z0-9|a-zA-Z0-9|0-9A-Za-z|0-9a-zA-Z|A-Z0-9|0-9A-Z)\]")
_KEY_NAMES = ("*shelf*key*", "loose_key", "canonical_id", "shelfmark_core_id")

#: Git-tracked path -> the violations it may keep (PR-1: every tracked caller found today). Shrinks in PRs 3, 4, 6
#: and 7. Untracked files go in :data:`LOCAL_ALLOWLIST_FILE`, never here.
ALLOWLIST: Dict[str, FrozenSet[str]] = {
    # Group C key wrappers and validators (PR-4)
    "src/datasets/cleaning/screen_training_misalignment.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/cleaning/verify_decontamination.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/evaluations/helper_eval_scripts/build_script_tags.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/evaluations/helper_eval_scripts/decontam_gate.py": frozenset(
        {"def:loose_key", "import:ShelfmarkNormalizer"}),
    "src/datasets/indexing/arabic_scrape_queue.py": frozenset({"def:loose_key", "import:ShelfmarkNormalizer"}),
    "src/datasets/indexing/es_image_gap_report.py": frozenset({"import:institution_tokens"}),
    "src/datasets/indexing/legacy_store/pdf_transcriptions.py": frozenset(
        {"def:shelfmark_core_id", "import:ShelfmarkNormalizer"}),
    "src/datasets/indexing/neo4j/enrich_fragment_people.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/finetuning/qwen_hebrew/build_documentary_grounding.py": frozenset(
        {"def:loose_key", "import:institution_tokens"}),
    # merging/ engine users outside the merge recipe (PR-4: legacy-store trio, normalizer_report, oxford_id_map;
    # bodleian_images / ktiv_transcription_gap use classify / institution lookups until the facade covers them)
    "src/datasets/merging/bodleian_images.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/merging/ktiv_transcription_gap.py": frozenset({"import:institution_tokens"}),
    "src/datasets/merging/normalizer_report.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/merging/oxford_id_map.py": frozenset({"import:ShelfmarkNormalizer", "import:institution_tokens"}),
    # squash family (PR-3)
    "src/finetuning/text27b/exposure.py": frozenset({"squash:id_key"}),
    # not a shelf mark: case-folded book / metadata-file names
    "src/datasets/indexing/bibliography/index_all_bibliography.py": frozenset({"squash:_name_key"}),
    # FROZEN key grammars (PR-7 marks them; they stay)
    "src/finetuning/text27b/shelfmarks.py": frozenset({"import:ShelfmarkNormalizer", "import:institution_tokens"}),
    # engine users for validation / KG cores (classify_shelfmark, to_canonical_id, get_institution_info): PR-4
    "src/datasets/document_models/genizah_document.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/indexing/neo4j/academic_kg_import.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/indexing/neo4j/backfill_es_doc_id.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/indexing/neo4j/biblio_import.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/indexing/neo4j/geocode_places.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/indexing/neo4j/knowlege_graph_poc.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/indexing/neo4j/ktiv_biblio.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/datasets/indexing/neo4j/merged_shelfmarks_import.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/models/llm/academic/coreference_resolver.py": frozenset({"import:ShelfmarkNormalizer"}),
    "src/models/llm/academic/relationship_extractor.py": frozenset({"import:ShelfmarkNormalizer"}),
}


def _is_squash_pattern(node: ast.AST) -> bool:
    """Whether *node* is a string constant holding a negated alphanumeric character class.

    :param node: AST node.
    :type node: ast.AST
    :return: True for e.g. ``r"[^a-z0-9]"``.
    :rtype: bool
    """
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and bool(_SQUASH_CLASS.search(node.value))


def _is_empty_string(node: ast.AST) -> bool:
    """Whether *node* is the constant ``""``.

    :param node: AST node.
    :type node: ast.AST
    :return: True for an empty string literal.
    :rtype: bool
    """
    return isinstance(node, ast.Constant) and node.value == ""


def _dotted(node: ast.AST) -> Optional[str]:
    """The dotted name of a ``Name`` / ``Attribute`` chain (``src.datasets.shelfmarks``), else ``None``.

    :param node: AST node.
    :type node: ast.AST
    :return: The dotted name, or ``None`` for any other expression.
    :rtype: Optional[str]
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


class _Scanner(ast.NodeVisitor):
    """Collect the violations of one module, naming each squash by its enclosing function."""

    def __init__(self) -> None:
        """Start with no violations and no enclosing function.

        :return: None.
        :rtype: None
        """
        self.found: Set[str] = set()
        self._stack: List[str] = []
        self._compiled: Set[str] = set()
        #: Dotted names bound to the facade module (``shelfmarks``, ``sm``, ``src.datasets.shelfmarks``).
        self._facades: Set[str] = set()
        #: Dotted names bound to the merge recipe module (``ms``, ``merge_shelfmarks``, ``src.datasets.…``).
        self._merges: Set[str] = set()

    def _func(self, node: ast.AST) -> None:
        """Record a key-like function name and descend with it on the stack.

        :param node: A function definition.
        :type node: ast.AST
        :return: None.
        :rtype: None
        """
        name = node.name  # type: ignore[attr-defined]
        if any(fnmatch.fnmatchcase(name.lower(), pat) for pat in _KEY_NAMES):
            self.found.add(f"def:{name}")
        self._stack.append(name)
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _func
    visit_AsyncFunctionDef = _func

    def visit_Assign(self, node: ast.Assign) -> None:
        """Remember names bound to ``re.compile(<squash class>)``.

        :param node: Assignment.
        :type node: ast.Assign
        :return: None.
        :rtype: None
        """
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "compile"
                and value.args and _is_squash_pattern(value.args[0])):
            self._compiled.update(t.id for t in node.targets if isinstance(t, ast.Name))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Flag ``re.sub(<squash class>, "", …)``, ``<compiled squash>.sub("", …)`` and
        ``importlib.import_module("<…engine module>")`` / ``__import__("<…engine module>")`` with a literal name.

        :param node: Call.
        :type node: ast.Call
        :return: None.
        :rtype: None
        """
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "sub":
            direct = len(node.args) >= 2 and _is_squash_pattern(node.args[0]) and _is_empty_string(node.args[1])
            compiled = (isinstance(func.value, ast.Name) and func.value.id in self._compiled
                        and node.args and _is_empty_string(node.args[0]))
            if direct or compiled:
                self.found.add(f"squash:{self._stack[-1] if self._stack else '<module>'}")
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        if (name in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            for suffix, violation in _ENGINE_MODULES.items():
                if node.args[0].value.endswith(suffix):
                    self.found.add(violation)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Flag imports of the engines, whatever module re-exports them.

        ``ShelfmarkNormalizer`` is flagged by name (so ``from …merge_shelfmarks import ShelfmarkNormalizer`` counts),
        any name imported from a ``genizah_normalizer`` module (``OxfordShelfmark`` too), and the
        ``genizah_normalizer`` / ``institution_tokens`` modules by name (absolute, package and relative forms).

        :param node: ``from … import …``.
        :type node: ast.ImportFrom
        :return: None.
        :rtype: None
        """
        module = node.module or ""
        names = {alias.name for alias in node.names}
        is_facade = node.level == 0 and module == FACADE_MODULE
        is_engine = module.endswith(("genizah_normalizer", "institution_tokens"))
        if names & {"ShelfmarkNormalizer", "genizah_normalizer"} or module.endswith("genizah_normalizer"):
            self.found.add("import:ShelfmarkNormalizer")
        if module.endswith("institution_tokens") or "institution_tokens" in names:
            self.found.add("import:institution_tokens")
        if is_facade and "_engine" in names:
            self.found.add("import:institution_tokens")
        if not is_facade and not is_engine and (names & TOKEN_NAMES or (module.endswith("merge_shelfmarks")
                                                                        and "*" in names)):
            self.found.add("import:institution_tokens")
        for alias in node.names:
            full = f"{module}.{alias.name}" if node.level == 0 else ""
            if full == FACADE_MODULE:
                self._facades.add(alias.asname or alias.name)
            if full == MERGE_MODULE or (node.level and alias.name == "merge_shelfmarks"):
                self._merges.add(alias.asname or alias.name)

    def visit_Import(self, node: ast.Import) -> None:
        """Flag ``import …genizah_normalizer`` / ``import …institution_tokens``.

        :param node: ``import …``.
        :type node: ast.Import
        :return: None.
        :rtype: None
        """
        for alias in node.names:
            if alias.name.endswith("genizah_normalizer"):
                self.found.add("import:ShelfmarkNormalizer")
            if alias.name.endswith("institution_tokens"):
                self.found.add("import:institution_tokens")
            if alias.name == FACADE_MODULE:
                self._facades.add(alias.asname or alias.name)
            if alias.name == MERGE_MODULE:
                self._merges.add(alias.asname or alias.name)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Flag ``<facade or merge module alias>.ShelfmarkNormalizer`` / ``._engine`` and
        ``<merge module alias>.<TOKEN_NAMES name>`` (the facade's own ``canonical_id`` / ``split_id`` are public).

        :param node: Attribute access.
        :type node: ast.Attribute
        :return: None.
        :rtype: None
        """
        base = _dotted(node.value)
        if node.attr in _ENGINE_ATTRS and base in self._facades | self._merges:
            self.found.add(_ENGINE_ATTRS[node.attr])
        if node.attr in TOKEN_NAMES and base in self._merges:
            self.found.add("import:institution_tokens")
        self.generic_visit(node)


def scan_source(source: str) -> Set[str]:
    """Violations in one module's source.

    :param source: Python source text.
    :type source: str
    :return: Violation descriptors (``def:<name>``, ``squash:<function>``, ``import:<engine>``).
    :rtype: Set[str]
    """
    scanner = _Scanner()
    scanner.visit(ast.parse(source))
    return scanner.found


def _exempt(rel: str) -> bool:
    """Whether a repo-relative path may use the engines freely.

    :param rel: Repo-relative POSIX path.
    :type rel: str
    :return: True for the facade, the two engines and the merge recipe.
    :rtype: bool
    """
    return rel in EXEMPT_FILES


def tracked_python_files(root: Path = SRC) -> Optional[List[str]]:
    """Repo-relative paths of the ``*.py`` files under *root* that git tracks or would add (untracked but not ignored).

    Untracked modules are scanned so that a new script written before its first commit is caught; ignored and
    privately excluded files are not listed (``--exclude-standard`` honours ``.gitignore`` and ``.git/info/exclude``).

    :param root: Directory to list.
    :type root: Path
    :return: The paths that exist on disk (an entry deleted but not yet ``git rm``-ed is left out), or ``None``
        when git is unavailable (no ``git`` binary or not a work tree).
    :rtype: Optional[List[str]]
    """
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
                              root.relative_to(REPO).as_posix()], cwd=REPO,
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted(p for p in out.decode("utf-8").split("\0") if p.endswith(".py") and (REPO / p).is_file())


def _all_python_files(root: Path = SRC) -> List[str]:
    """Repo-relative paths of every ``*.py`` file on disk under *root* (the fallback without git).

    :param root: Directory to walk.
    :type root: Path
    :return: The paths.
    :rtype: List[str]
    """
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        out.extend((Path(dirpath) / f).relative_to(REPO).as_posix() for f in filenames if f.endswith(".py"))
    return sorted(out)


def scan_tree(files: List[str]) -> Dict[str, Set[str]]:
    """Violations of every non-exempt module among *files*.

    :param files: Repo-relative paths (from :func:`tracked_python_files` or :func:`_all_python_files`).
    :type files: List[str]
    :return: Repo-relative path -> violations (paths without violations are left out).
    :rtype: Dict[str, Set[str]]
    """
    out: Dict[str, Set[str]] = {}
    for rel in files:
        if _exempt(rel) or SKIP_DIRS.intersection(Path(rel).parts):
            continue
        found = scan_source((REPO / rel).read_text(encoding="utf-8"))
        if found:
            out[rel] = found
    return out


def local_allowlist() -> Dict[str, FrozenSet[str]]:
    """The optional git-ignored supplement (:data:`LOCAL_ALLOWLIST_FILE`), or ``{}`` when it does not exist.

    :return: Path -> allowed violations.
    :rtype: Dict[str, FrozenSet[str]]
    """
    if not LOCAL_ALLOWLIST_FILE.exists():
        return {}
    data = json.loads(LOCAL_ALLOWLIST_FILE.read_text(encoding="utf-8"))
    return {rel: frozenset(found) for rel, found in data["allowlist"].items()}


def effective_allowlist() -> Dict[str, FrozenSet[str]]:
    """:data:`ALLOWLIST` merged with :func:`local_allowlist` (union per path).

    :return: Path -> allowed violations.
    :rtype: Dict[str, FrozenSet[str]]
    """
    merged = dict(ALLOWLIST)
    for rel, found in local_allowlist().items():
        merged[rel] = merged.get(rel, frozenset()) | found
    return merged


_TRACKED: Optional[List[str]] = None
_SCAN: Optional[Dict[str, Set[str]]] = None


def _scan() -> Dict[str, Set[str]]:
    """The scan of ``src/`` (git-tracked files; every file only without git), computed once per test session.

    :return: See :func:`scan_tree`.
    :rtype: Dict[str, Set[str]]
    """
    global _SCAN, _TRACKED
    if _SCAN is None:
        _TRACKED = tracked_python_files()
        _SCAN = scan_tree(_TRACKED if _TRACKED is not None else _all_python_files())
    return _SCAN


def test_no_new_shelfmark_normalisation_outside_the_facade() -> None:
    """Every violation found under ``src/`` is allowlisted for its own file.

    :return: None.
    :rtype: None
    """
    allow = effective_allowlist()
    new = {rel: sorted(found - allow.get(rel, frozenset())) for rel, found in _scan().items()}
    new = {rel: found for rel, found in new.items() if found}
    assert not new, ("shelf-mark normalisation outside src/datasets/shelfmarks.py; use the facade instead: "
                     f"{new}")


def test_allowlist_has_no_stale_entries() -> None:
    """Every public allowlisted violation still exists, so a migrated caller must leave the list (it only shrinks).

    :return: None.
    :rtype: None
    """
    scan = _scan()
    stale = {rel: sorted(allowed - scan.get(rel, set())) for rel, allowed in ALLOWLIST.items()}
    stale = {rel: found for rel, found in stale.items() if found}
    assert not stale, f"remove these entries from ALLOWLIST: {stale}"


def test_allowlist_paths_are_not_exempt() -> None:
    """The allowlist (public and local) never names an exempt file (those need no entry).

    :return: None.
    :rtype: None
    """
    assert not [rel for rel in effective_allowlist() if _exempt(rel)]


def test_scan_covers_git_tracked_files_and_public_allowlist_names_only_them() -> None:
    """The scan reads ``git ls-files`` (tracked plus untracked-not-ignored); the public allowlist names git-tracked files only.

    :return: None.
    :rtype: None
    """
    _scan()
    if _TRACKED is None:
        pytest.skip("git unavailable: the lint fell back to scanning every .py file under src/ (untracked files "
                    "included, allowed only through tests/fixtures/shelfmark_allowlist.local.json)")
    tracked = set(_TRACKED)
    assert not [rel for rel in ALLOWLIST if rel not in tracked]
    assert set(_scan()) <= tracked


def test_local_allowlist_is_git_ignored() -> None:
    """The local supplement can never be committed by accident.

    :return: None.
    :rtype: None
    """
    rel = LOCAL_ALLOWLIST_FILE.relative_to(REPO).as_posix()
    try:
        result = subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=REPO, capture_output=True)
    except OSError:
        pytest.skip("git unavailable (no git binary): cannot check that the local allowlist is ignored")
    if result.returncode == 128:
        pytest.skip("git unavailable (not a work tree): cannot check that the local allowlist is ignored")
    assert result.returncode == 0


def test_scanner_flags_each_rule() -> None:
    """The scanner sees a squash, a key-like definition and each engine import, and ignores a slug.

    :return: None.
    :rtype: None
    """
    source = (
        "import re\n"
        "from src.datasets.document_models.genizah_normalizer import ShelfmarkNormalizer\n"
        "from src.datasets.merging import institution_tokens\n"
        "_SQ = re.compile(r'[^a-z0-9]+')\n"
        "def my_shelf_key(s):\n    return re.sub(r'[^0-9a-z]', '', s.lower())\n"
        "def other(s):\n    return _SQ.sub('', s)\n"
        "def slug(s):\n    return re.sub(r'[^A-Za-z0-9]+', '_', s)\n"
        "class G:\n    def loose_key(self, s):\n        return s\n"
    )
    assert scan_source(source) == {"import:ShelfmarkNormalizer", "import:institution_tokens", "def:my_shelf_key",
                                   "squash:my_shelf_key", "squash:other", "def:loose_key"}
    assert scan_source("from src.datasets.shelfmarks import loose_key, canonical_id\n") == set()
    # re-exports and package / relative module imports are engine imports too
    sn = {"import:ShelfmarkNormalizer"}
    assert scan_source("from src.datasets.merging.merge_shelfmarks import ShelfmarkNormalizer\n") == sn
    assert scan_source("from src.datasets.document_models import genizah_normalizer\n") == sn
    assert scan_source("from . import genizah_normalizer\n") == sn
    assert scan_source("from .genizah_normalizer import *\n") == sn
    assert scan_source("import src.datasets.document_models.genizah_normalizer as gn\n") == sn
    it = {"import:institution_tokens"}
    assert scan_source("from .merging import institution_tokens\n") == it
    assert scan_source("from ..merging.institution_tokens import canonical_id as cid\n") == it
    assert scan_source("import src.datasets.merging.institution_tokens\n") == it


def test_scanner_flags_engine_bypasses() -> None:
    """Engine names re-exported by the merge recipe or the facade, and engine attributes of their aliases, count.

    :return: None.
    :rtype: None
    """
    it, sn = {"import:institution_tokens"}, {"import:ShelfmarkNormalizer"}
    # institution_tokens names imported through merge_shelfmarks or any other non-facade module
    for name in sorted(TOKEN_NAMES):
        assert scan_source(f"from src.datasets.merging.merge_shelfmarks import {name}\n") == it, name
        assert scan_source(f"from src.datasets.some_module import {name} as x\n") == it, name
    assert scan_source("from .merge_shelfmarks import combine\n") == it
    assert scan_source("from src.datasets.merging.merge_shelfmarks import *\n") == it
    # the facade's own public names are fine, its private engine handle is not
    assert scan_source("from src.datasets.shelfmarks import canonical_id, split_id, loose_key\n") == set()
    assert scan_source("from src.datasets.shelfmarks import _engine\n") == it
    assert scan_source("from src.datasets.shelfmarks import ShelfmarkNormalizer\n") == sn
    # engine attributes read off a module alias of the facade or of merge_shelfmarks
    assert scan_source("from src.datasets import shelfmarks\nshelfmarks.ShelfmarkNormalizer.loose_key('x')\n") == sn
    assert scan_source("from src.datasets import shelfmarks as sm\nsm._engine.combine('a', 'b')\n") == it
    assert scan_source("import src.datasets.shelfmarks as sm\nsm.ShelfmarkNormalizer\n") == sn
    assert scan_source("import src.datasets.shelfmarks\nsrc.datasets.shelfmarks._engine\n") == it
    assert scan_source("from src.datasets.merging import merge_shelfmarks as ms\nms.ShelfmarkNormalizer\n") == sn
    assert scan_source("import src.datasets.merging.merge_shelfmarks as ms\nms._engine\n") == it
    assert scan_source("from . import merge_shelfmarks\nmerge_shelfmarks.ShelfmarkNormalizer\n") == sn
    # institution_tokens names read off a merge_shelfmarks alias
    assert scan_source("from src.datasets.merging import merge_shelfmarks as ms\nms.combine('a', 'b')\n") == it
    assert scan_source("import src.datasets.merging.merge_shelfmarks as ms\nms.institution_token('x')\n") == it
    assert scan_source("import src.datasets.merging.merge_shelfmarks\n"
                       "src.datasets.merging.merge_shelfmarks.BL_TOKEN\n") == it
    assert scan_source("from . import merge_shelfmarks\nmerge_shelfmarks.resolve_token('x')\n") == it
    # ... but the facade's public canonical_id / split_id read as attributes are fine
    assert scan_source("import src.datasets.shelfmarks as sm\nsm.canonical_id('x')\nsm.split_id('y')\n") == set()
    # any name imported from the genizah_normalizer module is an engine import
    assert scan_source("from src.datasets.document_models.genizah_normalizer import OxfordShelfmark\n") == sn
    assert scan_source("from .genizah_normalizer import anything_else as x\n") == sn
    # importlib with a literal engine module name
    assert scan_source("import importlib\n"
                       "importlib.import_module('src.datasets.merging.institution_tokens')\n") == it
    assert scan_source("from importlib import import_module\n"
                       "import_module('src.datasets.document_models.genizah_normalizer')\n") == sn
    assert scan_source("__import__('src.datasets.merging.institution_tokens')\n") == it
    assert scan_source("import importlib\nimportlib.import_module('src.datasets.shelfmarks')\n") == set()
    # the facade's public API read as attributes is fine; an unrelated object's attribute is not a bypass
    assert scan_source("from src.datasets import shelfmarks\nshelfmarks.canonical_id('T-S 12.301')\n") == set()
    assert scan_source("other = object()\nother.ShelfmarkNormalizer\n") == set()
