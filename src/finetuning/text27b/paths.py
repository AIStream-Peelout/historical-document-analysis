# File name: paths.py
# Date: 10/22/26
# Author: Isaac Godfried. Coded originally by Claude Opus 5.5.
"""Machine-local configuration for the text27b dataset build.

Every location the build reads or writes comes from a git-ignored JSON file
(``logs/text_finetune_27b/paths.json`` under the repository root, or the file named by the
``TEXT27B_PATHS`` environment variable). Committed code never names absolute machine paths,
hosts, index names or holdings; it asks :func:`load_paths` instead.

Public API (other text27b modules import these names):

* :class:`Paths` -- frozen dataclass of resolved :class:`pathlib.Path` entries, with
  :meth:`Paths.out` (a path under ``out_dir``) and :meth:`Paths.get` (any entry, including keys
  this module does not know about).
* :func:`load_paths` -- read and resolve the JSON file (cached per file).
* :func:`repo_root` -- the repository root this package lives in.
* :data:`DEFAULT_PATHS_FILE` -- the default config location relative to the repository root.
* :data:`ENV_VAR` -- the environment variable that overrides the config location.

Resolution rules: a relative entry is resolved against the ``repo`` entry (which itself defaults
to :func:`repo_root`); absolute entries are kept as they are; ``~`` is expanded. Keys missing from
the JSON file fall back to :data:`OPTIONAL_DEFAULTS` (repository-relative), so an older
``paths.json`` keeps working when a module needs a new entry.
"""
from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

#: Config file location relative to the repository root.
DEFAULT_PATHS_FILE = Path("logs") / "text_finetune_27b" / "paths.json"
#: Environment variable that points at an alternative config file (tests, other machines).
ENV_VAR = "TEXT27B_PATHS"
#: Repository-relative defaults for entries an older ``paths.json`` may not carry.
OPTIONAL_DEFAULTS: Dict[str, str] = {
    "merged_shelfmarks": "src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl",
    "pgp_raw": "src/datasets/raw_data/cairo_genizah/pgp_raw/data",
}
#: Entries that are not filesystem paths (kept as strings).
_NON_PATH_KEYS: frozenset = frozenset()


def repo_root() -> Path:
    """Return the repository root that contains this package.

    :returns: Absolute path of the checkout (``src/finetuning/text27b`` is three levels below it).
    """
    return Path(__file__).resolve().parents[3]


def _resolve(value: Union[str, Path], repo: Path) -> Path:
    """Resolve one config entry to an absolute path.

    :param value: The raw entry (absolute, ``~``-prefixed or repository-relative).
    :param repo: Repository root used for relative entries.
    :returns: The absolute path.
    """
    p = Path(os.path.expanduser(str(value)))
    return p if p.is_absolute() else (repo / p)


@dataclass(frozen=True)
class Paths:
    """Resolved build locations (all absolute).

    Field names match the keys of ``paths.json``. Unknown keys are kept in :attr:`extra`.
    """

    repo: Path
    probe_dir: Path
    text_ft_v0_1: Path
    pgp_parallel: Path
    pgp_export: Path
    merged_pgpid_map: Path
    document_treatments: Path
    analysis_pages_index: Path
    exclusions_v1: Path
    samples: Path
    a50_overlap: Path
    benchmark_ids: Path
    academic_literature: Path
    ai_reads: Path
    book_editions_nas: Path
    tokenizer_json: Path
    chat_template: Path
    out_dir: Path
    scratch: Path
    merged_shelfmarks: Path
    pgp_raw: Path
    source_file: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def out(self, *parts: Union[str, Path], mkdir: bool = False) -> Path:
        """Return a path under :attr:`out_dir`.

        :param parts: Path components below ``out_dir`` (e.g. ``"families", "translate_en.jsonl"``).
        :param mkdir: Create the parent directory of the returned path when True.
        :returns: ``out_dir / parts...``.
        """
        p = self.out_dir.joinpath(*[str(x) for x in parts])
        if mkdir:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get(self, key: str, default: Any = None) -> Any:
        """Return any entry by name, known field or extra key.

        :param key: Entry name as written in ``paths.json``.
        :param default: Returned when the entry is absent.
        :returns: A resolved :class:`Path` for path entries, the raw value otherwise.
        """
        if key in {f.name for f in fields(self)} and key not in ("extra", "source_file"):
            return getattr(self, key)
        return self.extra.get(key, default)


_REQUIRED = tuple(f.name for f in fields(Paths) if f.name not in ("source_file", "extra"))


def _config_file(path: Optional[Union[str, Path]]) -> Path:
    """Pick the config file: explicit argument, then ``$TEXT27B_PATHS``, then the default.

    :param path: Explicit path or None.
    :returns: The config file location.
    """
    if path is not None:
        return Path(path).expanduser().resolve()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / DEFAULT_PATHS_FILE


@functools.lru_cache(maxsize=8)
def _load_cached(config: Path, mtime_ns: int) -> Paths:
    """Parse and resolve one config file (cached on path and modification time).

    :param config: Absolute config path.
    :param mtime_ns: Modification time, part of the cache key so edits are picked up.
    :returns: The resolved :class:`Paths`.
    :raises KeyError: When a required entry is missing and has no default.
    """
    with open(config, encoding="utf-8") as fh:
        raw: Dict[str, Any] = json.load(fh)
    repo = _resolve(raw.get("repo") or repo_root(), repo_root())
    merged = {**OPTIONAL_DEFAULTS, **raw}
    missing = [k for k in _REQUIRED if k != "repo" and k not in merged]
    if missing:
        raise KeyError(f"{config}: missing entries {missing}")
    kwargs: Dict[str, Any] = {"repo": repo, "source_file": config}
    for name in _REQUIRED:
        if name == "repo":
            continue
        kwargs[name] = _resolve(merged[name], repo)
    extra: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in kwargs or k == "repo":
            continue
        looks_like_path = isinstance(v, str) and ("/" in v or v.startswith("~"))
        extra[k] = _resolve(v, repo) if looks_like_path and k not in _NON_PATH_KEYS else v
    kwargs["extra"] = extra
    return Paths(**kwargs)


def load_paths(path: Optional[Union[str, Path]] = None) -> Paths:
    """Load the build configuration.

    :param path: Config file; defaults to ``$TEXT27B_PATHS`` or ``<repo>/logs/text_finetune_27b/paths.json``.
    :returns: Resolved :class:`Paths` (cached until the file changes).
    :raises FileNotFoundError: When the config file does not exist.
    """
    config = _config_file(path)
    return _load_cached(config, config.stat().st_mtime_ns)
