#!/usr/bin/env python3
"""Does this Mac's quantiser reproduce a reference 8-bit MLX copy bit for bit?

Quantises chosen tensors from the bf16 shards (by default the token embedding and the output head, the two largest)
with the conversion's recipe (affine, 8 bits, group 64, bf16 input) on the chosen device, and compares the packed
weights, scales and biases with the same tensors in the reference MLX copy using ``mx.array_equal``.

Reading the result: ``identical: true`` means conversions made on this machine are bit-paired with the reference
machine's, so a model converted here can be compared with one converted there as if one machine had made both.
``identical: false`` still means a valid 8-bit model, just not a bit-identical one; expect one-ulp differences in a
fraction of a percent of the scales (the reference machine's CPU against its own GPU: 0.46 % of the 8-bit weights,
1.3 % of the packed 32-bit words, 0.17 % of the scales, biases identical).

Inputs: ``--shards-dir`` holds the bf16 safetensors shards (a whole checkpoint directory, or only the shards that
hold the chosen tensors; an index file is used when present, otherwise the shard headers are scanned) and
``--mlx-dir`` is the reference MLX copy. Nothing is written except the JSON report (``--out``, replaced on every run
so that a crash can never leave an old verdict behind).

Exit status: 0 identical, 10 differences found, 11 a requested tensor is missing. The codes are above 2 so that an
uncaught error (1) or an argument error (2) cannot be mistaken for a verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit_record import DEFAULT_REQUIREMENTS, library_versions, machine_info, pin_mismatches  # noqa: E402
from verify_mlx_dir import INDEX_NAME, read_safetensors_header  # noqa: E402

DEFAULT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ('model.language_model.embed_tokens.weight', 'language_model.model.embed_tokens'),
    ('lm_head.weight', 'language_model.lm_head'),
)
PARTS: Tuple[str, ...] = ('weight', 'scales', 'biases')
EXIT_IDENTICAL, EXIT_DIFFERENT, EXIT_MISSING = 0, 10, 11


def parse_pair(text: str) -> Tuple[str, str]:
    """``HF_KEY=MLX_BASE`` -> ``(hf_key, mlx_base)``.

    :param text: Pair as written on the command line.
    :returns: The two names, stripped.
    :raises argparse.ArgumentTypeError: Without an ``=``.
    """
    if '=' not in text:
        raise argparse.ArgumentTypeError(f'expected HF_KEY=MLX_BASE, got {text!r}')
    hf_key, mlx_base = text.split('=', 1)
    return hf_key.strip(), mlx_base.strip()


def locate_tensors(directory: Path, keys: Sequence[str]) -> Dict[str, Path]:
    """Which safetensors file holds each key: from the index when present, else by scanning shard headers.

    :param directory: Checkpoint directory or a directory of shards.
    :param keys: Tensor names to find.
    :returns: Key -> file, for the keys that were found.
    """
    index = directory / INDEX_NAME
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding='utf-8'))['weight_map']
        return {key: directory / weight_map[key] for key in keys if key in weight_map}
    found: Dict[str, Path] = {}
    for file in sorted(directory.glob('*.safetensors')):
        header, _, _ = read_safetensors_header(file)
        for key in keys:
            if key in header:
                found[key] = file
    return found


def compare_parts(mx: Any, made: Sequence[Any], reference: Sequence[Any]) -> Dict[str, Any]:
    """Compare ``(weight, scales, biases)`` made here with the reference's.

    :param mx: The ``mlx.core`` module (or any module with ``array_equal``, ``abs``, ``float32`` and, optionally,
        ``view`` and ``uint8`` for the byte-level count).
    :param made: Arrays quantised on this machine.
    :param reference: The reference copy's arrays, same order.
    :returns: Per-part equality and, when unequal, difference counts and fractions: for ``weight`` over the packed
        32-bit words (``n_diff``) and over their bytes (``n_diff_bytes``; for 8-bit quantisation one byte is one
        weight), for ``scales`` and ``biases`` over elements with the largest absolute difference; plus ``identical``.
    """
    result: Dict[str, Any] = {}
    identical = True
    for part, a, b in zip(PARTS, made, reference):
        entry: Dict[str, Any] = {'shape': list(a.shape), 'reference_shape': list(b.shape), 'dtype': str(a.dtype),
                                 'reference_dtype': str(b.dtype)}
        if tuple(a.shape) != tuple(b.shape) or str(a.dtype) != str(b.dtype):
            entry['equal'] = False
            identical = False
        else:
            equal = bool(mx.array_equal(a, b))
            entry['equal'] = equal
            if not equal:
                n_diff = int((a != b).sum())
                entry['n_diff'] = n_diff
                entry['fraction_diff'] = n_diff / int(a.size)
                entry['unit'] = 'packed_uint32_words' if part == 'weight' else 'elements'
                if part == 'weight' and hasattr(mx, 'view') and hasattr(mx, 'uint8'):
                    n_bytes = int((mx.view(a, mx.uint8) != mx.view(b, mx.uint8)).sum())
                    entry['n_diff_bytes'] = n_bytes
                    entry['fraction_diff_bytes'] = n_bytes / (int(a.size) * 4)
                if part != 'weight':
                    entry['max_abs_diff'] = float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())
                identical = False
        result[part] = entry
    result['identical'] = identical
    return result


def overall_verdict(results: Sequence[Dict[str, Any]]) -> bool:
    """Whether every compared tensor came out identical.

    :param results: Per-tensor results holding a ``comparison`` mapping.
    :returns: True only when there is at least one result and all are identical.
    """
    return bool(results) and all(result['comparison']['identical'] for result in results)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :returns: Exit status (see the module docstring).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shards-dir', type=Path, required=True, help='bf16 shards (checkpoint dir or shard subset)')
    parser.add_argument('--mlx-dir', type=Path, required=True, help='reference 8-bit MLX copy')
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--pair', type=parse_pair, action='append',
                        help='HF_KEY=MLX_BASE to compare (repeatable; default: token embedding and output head)')
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--group-size', type=int, default=64)
    parser.add_argument('--out', type=Path, default=Path('quant_identity.json'), help='JSON report (replaced)')
    args = parser.parse_args(argv)
    pairs: List[Tuple[str, str]] = args.pair or list(DEFAULT_PAIRS)
    if args.out.exists():
        args.out.unlink()

    reference_map = json.loads((args.mlx_dir / INDEX_NAME).read_text(encoding='utf-8'))['weight_map']
    located = locate_tensors(args.shards_dir, [hf_key for hf_key, _ in pairs])
    missing = [hf_key for hf_key, _ in pairs if hf_key not in located]
    if missing:
        print(f'not found in {args.shards_dir}: {missing}', file=sys.stderr)
        return EXIT_MISSING
    absent = [f'{base}.{part}' for _, base in pairs for part in PARTS if f'{base}.{part}' not in reference_map]
    if absent:
        print(f'not found in the reference index: {absent}', file=sys.stderr)
        return EXIT_MISSING
    pins = {name: {'pinned': pinned, 'installed': installed}
            for name, (pinned, installed) in pin_mismatches(DEFAULT_REQUIREMENTS).items()} if DEFAULT_REQUIREMENTS.is_file() else {}
    if pins:
        print(f'warning: installed libraries differ from requirements.txt: {pins}', file=sys.stderr)

    import mlx.core as mx  # noqa: E402  (after the cheap checks, so --help and exit 11 need no MLX)
    mx.set_default_device(mx.gpu if args.device == 'gpu' else mx.cpu)
    results: List[Dict[str, Any]] = []
    reference_cache: Dict[Path, Dict[str, Any]] = {}
    for hf_key, mlx_base in pairs:
        raw = mx.load(str(located[hf_key]))[hf_key]
        source = raw.astype(mx.bfloat16)
        mx.eval(source)
        started = time.time()
        made = mx.quantize(source, group_size=args.group_size, bits=args.bits)
        mx.eval(*made)
        quantize_seconds = time.time() - started
        reference = []
        for part in PARTS:
            file = args.mlx_dir / reference_map[f'{mlx_base}.{part}']
            if file not in reference_cache:
                reference_cache.clear()
                reference_cache[file] = mx.load(str(file))
            reference.append(reference_cache[file][f'{mlx_base}.{part}'])
        mx.eval(*reference)
        comparison = compare_parts(mx, made, reference)
        results.append({'hf_key': hf_key, 'mlx_base': mlx_base, 'shard': located[hf_key].name,
                        'shape': list(source.shape), 'elements': int(source.size), 'source_dtype': str(raw.dtype),
                        'quantize_seconds': round(quantize_seconds, 2),
                        'gb_per_s': round(int(source.size) * 2 / max(quantize_seconds, 1e-9) / 1e9, 2),
                        'comparison': comparison})
        print(json.dumps(results[-1]), flush=True)
        del raw, source, made, reference
        mx.clear_cache()

    identical = overall_verdict(results)
    report = {'identical': identical, 'device': args.device, 'bits': args.bits, 'group_size': args.group_size,
              'machine': machine_info(), 'libraries': library_versions(), 'pin_mismatches': pins,
              'shards_dir': str(args.shards_dir), 'reference_dir': str(args.mlx_dir), 'tensors': results,
              'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    args.out.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print('IDENTICAL' if identical else 'DIFFERENCES_FOUND', f'(report: {args.out})')
    return EXIT_IDENTICAL if identical else EXIT_DIFFERENT


if __name__ == '__main__':
    sys.exit(main())
