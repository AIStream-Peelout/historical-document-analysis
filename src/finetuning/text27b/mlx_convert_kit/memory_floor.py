#!/usr/bin/env python3
"""Free memory the eager quantisation order needs for a source checkpoint, from its shard headers (stdlib only).

``convert_lowmem.py --quantize-order eager`` holds the packed model until it is written, plus one source tensor while
that tensor is being quantised. This estimates the floor ``run_convert.sh`` checks before an eager conversion:

    packed model + largest source tensor + margin

* packed model: every tensor with at least two dimensions whose last dimension is a multiple of the group size counts
  as quantised (``bits`` per weight plus one scale and one bias per group in the source dtype, ``bits + 32 / group``
  bits per weight for a bf16 source); every other tensor (norms, biases, convolutions) counts at its source size;
* largest source tensor: usually the token embedding or the output head;
* margin: the process itself and the kernels' temporaries (default 4 GiB).

Only the JSON headers are read, so the estimate costs a few kilobytes per shard even over a network volume. It is
deliberately on the high side: tensors the model drops on load (for example extra prediction heads) are counted too.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_mlx_dir import INDEX_NAME, read_safetensors_header  # noqa: E402

GIB = 2**30
DEFAULT_MARGIN_GIB = 4.0


def source_shards(source: Path) -> List[Path]:
    """The safetensors files of a source checkpoint.

    :param source: Checkpoint directory.
    :returns: The shards named in the index (sorted, each once), or every ``*.safetensors`` file when there is no
        index.
    :raises FileNotFoundError: When the index names a shard that is absent, or the directory holds no shard at all.
    """
    index = source / INDEX_NAME
    if index.is_file():
        names = sorted(set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values()))
        shards = [source / name for name in names]
    else:
        shards = sorted(source.glob('*.safetensors'))
    missing = [shard.name for shard in shards if not shard.is_file()]
    if missing or not shards:
        raise FileNotFoundError(f'{source}: missing shards {missing}' if missing else f'{source}: no safetensors files')
    return shards


def packed_bytes(entry: Dict[str, Any], bits: int, group_size: int) -> float:
    """Bytes one source tensor occupies after conversion.

    :param entry: The tensor's safetensors header entry (``shape``, ``data_offsets``).
    :param bits: Quantisation bits.
    :param group_size: Quantisation group size.
    :returns: Packed size (weights plus per-group scale and bias in the source dtype) for a quantisable tensor, the
        source size otherwise.
    """
    shape = entry['shape']
    nbytes = entry['data_offsets'][1] - entry['data_offsets'][0]
    elements = math.prod(shape)
    if len(shape) < 2 or shape[-1] % group_size or elements == 0:
        return float(nbytes)
    element_bytes = nbytes / elements
    return elements * bits / 8 + 2 * (elements / group_size) * element_bytes


def estimate(source: Path, bits: int = 8, group_size: int = 64,
             margin_gib: float = DEFAULT_MARGIN_GIB) -> Dict[str, Any]:
    """Estimate the eager order's free-memory floor for a source checkpoint.

    :param source: Checkpoint directory (safetensors shards, normally with an index).
    :param bits: Quantisation bits.
    :param group_size: Quantisation group size.
    :param margin_gib: Allowance for the process and temporaries, in GiB.
    :returns: ``source_gib``, ``packed_gib``, ``largest_tensor`` (name), ``largest_tensor_gib``, ``margin_gib`` and
        ``floor_gib`` (rounded up to 0.1 GiB).
    """
    source_bytes, packed, largest_name, largest = 0, 0.0, None, 0
    for shard in source_shards(source):
        tensors, _, _ = read_safetensors_header(shard)
        for name, entry in tensors.items():
            nbytes = entry['data_offsets'][1] - entry['data_offsets'][0]
            source_bytes += nbytes
            packed += packed_bytes(entry, bits, group_size)
            if nbytes > largest:
                largest_name, largest = name, nbytes
    floor = packed / GIB + largest / GIB + margin_gib
    return {'source_gib': round(source_bytes / GIB, 2), 'packed_gib': round(packed / GIB, 2),
            'largest_tensor': largest_name, 'largest_tensor_gib': round(largest / GIB, 2), 'margin_gib': margin_gib,
            'floor_gib': math.ceil(floor * 10) / 10}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the floor in GiB (one number, for ``run_convert.sh``) or the whole estimate as JSON.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :returns: 0 on success, 1 when the shards cannot be read.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='source checkpoint directory (safetensors shards)')
    parser.add_argument('--q-bits', type=int, default=8)
    parser.add_argument('--q-group-size', type=int, default=64)
    parser.add_argument('--margin-gib', type=float, default=DEFAULT_MARGIN_GIB)
    parser.add_argument('--json', action='store_true', help='print the whole estimate instead of the floor alone')
    args = parser.parse_args(argv)
    try:
        result = estimate(args.source, args.q_bits, args.q_group_size, args.margin_gib)
    except (OSError, ValueError, KeyError) as error:  # unreadable index or shard header
        print(f'memory_floor: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1) if args.json else result['floor_gib'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
