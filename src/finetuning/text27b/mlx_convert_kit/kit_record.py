#!/usr/bin/env python3
"""Write the record of a finished conversion (``<mlx-path>.CONVERTED.json``) and check library pins.

The record holds the bits-per-weight figure from the conversion log, the machine (chip, memory, macOS), the library
versions and any pins of ``requirements.txt`` the interpreter does not satisfy, the device, the duration and, when
given, the identity-test verdict and the manifest path. ``--check-pins REQ`` only compares the installed libraries
with the pins and exits 1 on a mismatch (``run_convert.sh`` runs it before converting). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

LIBRARIES: Sequence[str] = ('mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'tokenizers', 'safetensors', 'numpy',
                            'huggingface_hub')
DEFAULT_REQUIREMENTS = Path(__file__).resolve().parent / 'requirements.txt'


def bits_per_weight(log_text: str) -> Optional[float]:
    """The last ``Quantized model with X bits per weight`` figure of an ``mlx_lm`` conversion log.

    :param log_text: Log contents.
    :returns: Bits per weight, or None when the log has no such line.
    """
    found = re.findall(r'Quantized model with ([\d.]+) bits per weight', log_text)
    return float(found[-1]) if found else None


def sysctl(key: str) -> Optional[str]:
    """One ``sysctl -n`` value.

    :param key: sysctl key such as ``machdep.cpu.brand_string``.
    :returns: The value, or None when unavailable (no sysctl, unknown key).
    """
    if shutil.which('sysctl') is None:
        return None
    result = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def machine_info() -> Dict[str, Any]:
    """Chip, memory, macOS and Python of this machine (no host name).

    :returns: Mapping with ``chip``, ``memory_gib``, ``macos``, ``python`` and ``arch``.
    """
    memory = sysctl('hw.memsize')
    return {'chip': sysctl('machdep.cpu.brand_string'),
            'memory_gib': round(int(memory) / 2**30, 1) if memory and memory.isdigit() else None,
            'macos': platform.mac_ver()[0] or platform.platform(), 'python': platform.python_version(),
            'arch': platform.machine()}


def installed_version(name: str) -> Optional[str]:
    """Installed version of a distribution, accepting ``-`` or ``_`` spellings of its name.

    :param name: Distribution name.
    :returns: Version string, or None when not installed.
    """
    for candidate in (name, name.replace('-', '_'), name.replace('_', '-')):
        try:
            return version(candidate)
        except PackageNotFoundError:
            continue
    return None


def library_versions(names: Sequence[str] = LIBRARIES) -> Dict[str, Optional[str]]:
    """Installed versions of the libraries the conversion depends on.

    :param names: Distribution names.
    :returns: Name -> version (None when not installed).
    """
    return {name: installed_version(name) for name in names}


def parse_requirements(path: Path) -> Dict[str, str]:
    """``name==version`` pins of a requirements file (comments and blank lines ignored).

    :param path: requirements.txt.
    :returns: Name -> pinned version.
    """
    pins: Dict[str, str] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        text = line.split('#', 1)[0].strip()
        if text and '==' in text:
            name, pinned = text.split('==', 1)
            pins[name.strip()] = pinned.strip()
    return pins


def pin_mismatches(requirements: Path) -> Dict[str, Tuple[str, Optional[str]]]:
    """Pins of ``requirements`` that the running interpreter does not satisfy exactly.

    :param requirements: requirements.txt.
    :returns: Name -> ``(pinned, installed or None)`` for every mismatch; empty when all pins match.
    """
    mismatches: Dict[str, Tuple[str, Optional[str]]] = {}
    for name, pinned in parse_requirements(requirements).items():
        installed = installed_version(name)
        if installed != pinned:
            mismatches[name] = (pinned, installed)
    return mismatches


def build_record(log_text: str, hf_path: str, mlx_path: str, device: str, seconds: Optional[float],
                 identity: Optional[Dict[str, Any]] = None, manifest: Optional[str] = None,
                 requirements: Optional[Path] = None) -> Dict[str, Any]:
    """Assemble the conversion record.

    :param log_text: Conversion log contents.
    :param hf_path: Source checkpoint directory.
    :param mlx_path: Output MLX directory.
    :param device: ``gpu`` or ``cpu``.
    :param seconds: Wall-clock duration of the conversion.
    :param identity: Parsed identity-test report, when one was run on this machine.
    :param manifest: Path of the size/sha256 manifest, when one was written.
    :param requirements: requirements.txt to compare the installed libraries with.
    :returns: The record.
    """
    record: Dict[str, Any] = {'hf_path': hf_path, 'mlx_path': mlx_path, 'device': device, 'bits': 8, 'group_size': 64,
                              'bits_per_weight': bits_per_weight(log_text), 'convert_seconds': seconds,
                              'machine': machine_info(), 'libraries': library_versions(), 'manifest': manifest,
                              'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if requirements is not None and requirements.is_file():
        record['pins'] = {'requirements': str(requirements),
                          'mismatches': {name: {'pinned': pinned, 'installed': installed}
                                         for name, (pinned, installed) in pin_mismatches(requirements).items()}}
    if identity is not None:
        record['identity_test'] = {'identical': identity.get('identical'), 'device': identity.get('device'),
                                   'reference_dir': identity.get('reference_dir'), 'time_utc': identity.get('time_utc')}
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :returns: Exit status (``--check-pins``: 0 all pins match, 1 otherwise).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check-pins', type=Path, metavar='REQ', help='only compare installed libraries with REQ')
    parser.add_argument('--log', type=Path, help='conversion log')
    parser.add_argument('--out', type=Path, help='record to write')
    parser.add_argument('--hf-path')
    parser.add_argument('--mlx-path')
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--seconds', type=float, help='conversion wall-clock seconds')
    parser.add_argument('--identity-json', type=Path, help='report written by gpu_quant_identity.py')
    parser.add_argument('--manifest', help='manifest path written by verify_mlx_dir.py')
    parser.add_argument('--requirements', type=Path, default=DEFAULT_REQUIREMENTS if DEFAULT_REQUIREMENTS.is_file() else None,
                        help='requirements.txt to compare with (default: the kit\'s)')
    args = parser.parse_args(argv)
    if args.check_pins:
        mismatches = pin_mismatches(args.check_pins)
        for name, (pinned, installed) in sorted(mismatches.items()):
            print(f'{name}: pinned {pinned}, installed {installed or "nothing"}', file=sys.stderr)
        print('pins ok' if not mismatches else f'{len(mismatches)} pin mismatch(es)')
        return 0 if not mismatches else 1
    missing = [flag for flag, value in (('--log', args.log), ('--out', args.out), ('--hf-path', args.hf_path),
                                        ('--mlx-path', args.mlx_path)) if value is None]
    if missing:
        parser.error(f'{", ".join(missing)} required unless --check-pins is used')
    identity = json.loads(args.identity_json.read_text(encoding='utf-8')) if args.identity_json else None
    record = build_record(args.log.read_text(encoding='utf-8', errors='replace'), args.hf_path, args.mlx_path,
                          args.device, args.seconds, identity, args.manifest, args.requirements)
    args.out.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
