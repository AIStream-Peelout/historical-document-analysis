#!/usr/bin/env python3
"""``mlx_lm convert -q --q-bits 8 --q-group-size 64`` with a bounded Metal working set.

Same recipe and code path as the stock command (``mlx_lm.convert.convert``: lazy load, cast to the config dtype,
``nn.quantize``, ``save``), so the quantised tensors are exactly what the stock conversion produces for the same
inputs on the same device. The one change: before each shard is written, its tensors are evaluated one at a time and
the MLX buffer cache is released after each, so the GPU never holds a whole shard's quantisation graph at once. The
footprint stays around 5 GiB for a 27B model instead of tens of GiB.

What it does not protect against: other GPU work on the same machine. Metal's watchdog kills a long command buffer
when another process (LM Studio, a model server, a game) monopolises the GPU, and a killed buffer poisons this
process's Metal context, so the conversion must have the GPU to itself. ``--device cpu`` exists for self-tests only:
CPU quantisation is not bit-identical to GPU quantisation.

The output directory must not exist (``mlx_lm`` refuses to write into one). A conversion is not resumable: after a
crash delete the partial directory and start again.
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import mlx.core as mx

_original_save = mx.save_safetensors


def _eval_one_at_a_time(path: str, shard: Dict[str, Any], **kwargs: Any) -> None:
    """Evaluate ``shard``'s arrays sequentially, releasing cached buffers between them, then save.

    :param path: Shard file path.
    :param shard: Tensor name -> lazily computed array.
    :param kwargs: Passed through to :func:`mlx.core.save_safetensors` (``metadata``).
    :returns: None.
    """
    started = time.time()
    for value in shard.values():
        mx.eval(value)
        mx.clear_cache()
    print(f'[lowmem] {str(path).rsplit("/", 1)[-1]}: {len(shard)} tensors evaluated in {time.time() - started:.1f} s '
          f'(active {mx.get_active_memory() / 2**30:.2f} GiB, peak {mx.get_peak_memory() / 2**30:.2f} GiB)', flush=True)
    _original_save(path, shard, **kwargs)
    mx.clear_cache()


def main() -> None:
    """Parse arguments, pick the device, bound the allocator, patch the save hook and run the stock conversion.

    :returns: None.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hf-path', required=True, help='source checkpoint directory (bf16 safetensors)')
    parser.add_argument('--mlx-path', required=True, help='output directory; must not exist')
    parser.add_argument('--q-bits', type=int, default=8)
    parser.add_argument('--q-group-size', type=int, default=64)
    parser.add_argument('--memory-limit-gib', type=float, default=12.0,
                        help='MLX allocator limit; the scheduler waits for in-flight work above it')
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu',
                        help='cpu is for self-tests only: CPU quantisation is not bit-identical to GPU quantisation')
    args = parser.parse_args()
    mx.set_default_device(mx.gpu if args.device == 'gpu' else mx.cpu)
    mx.set_cache_limit(0)  # freed buffers go straight back to the OS
    mx.set_memory_limit(int(args.memory_limit_gib * 2**30))
    import mlx_lm  # noqa: E402
    import mlx_lm.utils as utils  # noqa: E402  (patched before convert() imports it)
    utils.mx.save_safetensors = _eval_one_at_a_time
    from mlx_lm.convert import convert  # noqa: E402
    print(f'[lowmem] mlx {mx.__version__}, mlx_lm {mlx_lm.__version__}, device {mx.default_device()}, '
          f'cache limit 0, memory limit {args.memory_limit_gib} GiB', flush=True)
    convert(hf_path=args.hf_path, mlx_path=args.mlx_path, quantize=True, q_bits=args.q_bits,
            q_group_size=args.q_group_size)
    print(f'[lowmem] done; peak memory {mx.get_peak_memory() / 2**30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
