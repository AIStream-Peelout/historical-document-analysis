#!/usr/bin/env python3
"""Check an MLX model directory for completeness; write or compare a size/sha256 manifest.

Standard library only, so it runs on any Mac without the MLX environment (for example on the machine that receives
a copy made elsewhere).

``check`` (the default): every required file is present; every shard named in ``model.safetensors.index.json``
exists, its safetensors header parses, the header lists exactly the tensors the index assigns to that file, and the
file is exactly as long as the header's data offsets say. A shard truncated by a crash or a network drop, or one that
is still being written, fails the length check. The ``quantization`` block of ``config.json`` is reported.

``--manifest PATH`` records every file's size and sha256 (``--no-hash`` skips the hashes) so that a copy made over
the network can be checked on the other side with ``--compare PATH``.

Finder and SMB dotfiles (``.DS_Store``, ``._*``) are ignored everywhere, so a copy opened in Finder still verifies.

Exit status 0 when the directory is complete (and matches the manifest when one is compared), 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REQUIRED_FILES: Tuple[str, ...] = ('config.json', 'model.safetensors.index.json', 'tokenizer.json',
                                   'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json')
INDEX_NAME = 'model.safetensors.index.json'
MAX_HEADER_BYTES = 100 * 2**20
HASH_CHUNK = 16 * 2**20


def model_files(path: Path) -> List[Path]:
    """Regular files of a model directory, without Finder/SMB dotfiles (``.DS_Store``, ``._*``).

    :param path: Model directory.
    :returns: Sorted paths.
    """
    return sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith('.'))


def read_safetensors_header(path: Path) -> Tuple[Dict[str, Dict[str, Any]], int, Optional[Dict[str, str]]]:
    """Parse a safetensors file's JSON header without reading any tensor data.

    :param path: Safetensors file.
    :returns: ``(tensors, expected_file_length, metadata)``: ``tensors`` maps each tensor name to its header entry
        (``dtype``, ``shape``, ``data_offsets``); ``expected_file_length`` is the length the offsets imply.
    :raises ValueError: When the file is too short, the header length is implausible or the header is not JSON.
    """
    with path.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise ValueError(f'{path.name}: shorter than the 8-byte header length')
        (header_length,) = struct.unpack('<Q', prefix)
        if header_length > MAX_HEADER_BYTES:
            raise ValueError(f'{path.name}: implausible header length {header_length}')
        header = json.loads(handle.read(header_length).decode('utf-8'))
    metadata = header.pop('__metadata__', None)
    data_end = max((entry['data_offsets'][1] for entry in header.values()), default=0)
    return header, 8 + header_length + data_end, metadata


def check_shard(shard: Path, tensors: Sequence[str]) -> List[str]:
    """Problems with one shard: missing, unreadable header, wrong length, or a tensor set that differs from the index.

    :param shard: Shard file.
    :param tensors: Tensor names the index assigns to this shard.
    :returns: Problem descriptions (empty when the shard is sound).
    """
    if not shard.is_file():
        return [f'missing shard {shard.name}']
    try:
        header, expected_length, _ = read_safetensors_header(shard)
    except ValueError as error:  # truncated or corrupt header
        return [f'{shard.name}: unreadable header ({error})']
    problems = []
    size = shard.stat().st_size
    if size != expected_length:
        problems.append(f'{shard.name}: {size} bytes on disk, header implies {expected_length}')
    if set(header) != set(tensors):
        only_file, only_index = len(set(header) - set(tensors)), len(set(tensors) - set(header))
        problems.append(f'{shard.name}: tensor set differs from the index '
                        f'({only_file} only in the file, {only_index} only in the index)')
    return problems


def check_dir(path: Path, required: Sequence[str] = REQUIRED_FILES) -> Dict[str, Any]:
    """Check that ``path`` holds a complete model directory.

    :param path: Model directory.
    :param required: Files that must be present.
    :returns: Report with ``complete``, ``problems``, shard and tensor counts, total bytes, file list and the
        ``quantization`` block of the config.
    """
    if not path.is_dir():
        return {'dir': str(path), 'complete': False, 'problems': ['not a directory'], 'shards': 0, 'tensors': 0,
                'bytes': 0, 'quantization': None, 'files': []}
    problems: List[str] = []
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        problems.append(f'missing files: {missing}')
    shards: Dict[str, List[str]] = {}
    if (path / INDEX_NAME).is_file():
        weight_map = json.loads((path / INDEX_NAME).read_text(encoding='utf-8'))['weight_map']
        for tensor, shard in weight_map.items():
            shards.setdefault(shard, []).append(tensor)
        for shard, tensors in sorted(shards.items()):
            problems.extend(check_shard(path / shard, tensors))
    files = [p.name for p in model_files(path)]
    quantization = None
    if (path / 'config.json').is_file():
        quantization = json.loads((path / 'config.json').read_text(encoding='utf-8')).get('quantization')
    return {'dir': str(path), 'complete': not problems, 'problems': problems, 'shards': len(shards),
            'tensors': sum(len(t) for t in shards.values()),
            'bytes': sum((path / name).stat().st_size for name in files), 'quantization': quantization, 'files': files}


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in 16 MiB chunks.

    :param path: File to hash.
    :returns: Hex digest.
    """
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(path: Path, hashes: bool = True, progress: bool = False) -> Dict[str, Any]:
    """Record the size (and sha256) of every file in a directory.

    :param path: Directory.
    :param hashes: Whether to compute sha256 digests.
    :param progress: Print one line per file to stderr.
    :returns: Manifest ``{dir_name, created_utc, hashed, files: {name: {bytes, sha256}}}``.
    """
    files: Dict[str, Dict[str, Any]] = {}
    for file in model_files(path):
        started = time.time()
        entry: Dict[str, Any] = {'bytes': file.stat().st_size, 'sha256': sha256_file(file) if hashes else None}
        files[file.name] = entry
        if progress:
            took = f', hashed in {time.time() - started:.0f} s' if hashes else ''
            print(f'[manifest] {file.name}: {entry["bytes"]} bytes{took}', file=sys.stderr, flush=True)
    return {'dir_name': path.name, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'hashed': hashes, 'files': files}


def compare_manifest(path: Path, manifest: Dict[str, Any], hashes: bool = True, progress: bool = False) -> List[str]:
    """Differences between a directory and a manifest.

    :param path: Directory to check.
    :param manifest: Manifest from :func:`build_manifest`.
    :param hashes: Compare sha256 digests where the manifest has them.
    :param progress: Print one line per hashed file to stderr.
    :returns: Problems: missing files, size or digest mismatches, and files absent from the manifest.
    """
    problems = []
    for name, entry in manifest['files'].items():
        if name.startswith('.'):  # a dotfile recorded by an older manifest
            continue
        file = path / name
        if not file.is_file():
            problems.append(f'missing {name}')
            continue
        size = file.stat().st_size
        if size != entry['bytes']:
            problems.append(f'{name}: {size} bytes, manifest says {entry["bytes"]}')
            continue
        if hashes and entry.get('sha256'):
            started = time.time()
            if sha256_file(file) != entry['sha256']:
                problems.append(f'{name}: sha256 differs from the manifest')
            if progress:
                print(f'[compare] {name}: hashed in {time.time() - started:.0f} s', file=sys.stderr, flush=True)
    extra = [p.name for p in model_files(path) if p.name not in manifest['files']]
    if extra:
        problems.append(f'files not in the manifest: {extra}')
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :returns: Exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dir', type=Path, help='model directory to check')
    parser.add_argument('--manifest', type=Path, help='write a size/sha256 manifest of the directory here')
    parser.add_argument('--no-hash', action='store_true', help='skip sha256 digests (manifest and compare)')
    parser.add_argument('--compare', type=Path, help='compare the directory with this manifest')
    parser.add_argument('--json', type=Path, help='also write the report here')
    args = parser.parse_args(argv)
    report = check_dir(args.dir)
    if args.compare:
        manifest = json.loads(args.compare.read_text(encoding='utf-8'))
        report['compare'] = {'manifest': str(args.compare),
                             'problems': compare_manifest(args.dir, manifest, hashes=not args.no_hash, progress=True)}
    if args.manifest and report['complete']:
        manifest = build_manifest(args.dir, hashes=not args.no_hash, progress=True)
        args.manifest.write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
        report['manifest'] = str(args.manifest)
    ok = report['complete'] and not report.get('compare', {}).get('problems')
    report['ok'] = ok
    text = json.dumps(report, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text + '\n', encoding='utf-8')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
