#!/usr/bin/env python3
"""``mlx_lm convert -q --q-bits 8 --q-group-size 64`` that a slow source cannot kill, with bounded memory.

Same recipe and code path as the stock command (``mlx_lm.convert.convert``: lazy load, cast to the config dtype,
``nn.quantize``, ``save``). The quantisation kernel, the device it runs on and its arguments are unchanged; what changes
is the order of evaluation, and, in the eager order, the device of the ops before the kernel: loading, the dtype cast
and the model's ``sanitize`` fix-ups run on the CPU instead of the GPU. For a source already stored in the config dtype
the kernel's input is the bytes read from the file, at most renamed or transposed by ``sanitize`` (a same-dtype cast
leaves them untouched), so each quantised tensor is bit for bit what the stock conversion produces on the same device.
A tensor that is changed by fix-up arithmetic (for example a norm weight shifted by 1.0, which is not quantised) comes
from the CPU's arithmetic in the eager order; ``selftest.py --device gpu`` checks that the CPU and the GPU compute such
fix-ups identically.

Why the order matters. In the stock (lazy) order each quantised tensor is the graph ``Load`` (a CPU read from the
source file) -> ``quantize`` (a GPU kernel), and it is evaluated only when its shard is saved. The Metal command buffer
that runs the kernel then waits for the read. When the source sits on a network volume (SMB), a multi-GB tensor can
take longer to arrive than Metal's GPU watchdog allows, and the conversion dies a few minutes in with
``[METAL] Command buffer execution failed: Caused GPU Timeout Error`` inside ``mx.eval``, even with the GPU otherwise
idle. From a fast local SSD the same lazy order works.

``--quantize-order eager`` (the default):

* the default device is the CPU while ``convert()`` runs, so loading, the dtype cast and the model's ``sanitize`` ops
  (renames, transposes, shifts) are CPU work; nothing touches the GPU except the quantisation kernel;
* ``mx.quantize`` is wrapped: the module's weight is evaluated first (the read completes on the CPU and nothing is
  encoded on the GPU yet), then the kernel runs on the ``--device`` stream on data already in memory, and its outputs
  are evaluated at once. The stock layer constructors also call ``mx.quantize`` on a random initialisation, which
  ``to_quantized`` overwrites at once (mlx 0.32 ``nn/layers/quantized.py`` l. 133-136, 239-247; ``mlx_lm``'s
  ``QuantizedSwitchLinear``); that call stays lazy and is never evaluated, as in the stock path;
* ``to_quantized`` is wrapped so that the source module's weight is replaced by an empty placeholder as soon as its
  quantised copy exists: on ``nn.Linear`` and ``nn.Embedding`` for the whole conversion, and, for the duration of
  ``nn.quantize`` (which ``mlx_lm`` calls as ``nn.quantize``), on every other class in the model that defines its own
  ``to_quantized``, such as ``mlx_lm``'s ``SwitchLinear``, which holds a mixture-of-experts model's expert weights.
  ``nn.quantize`` otherwise keeps every original module, and with it every bf16 weight, until its final
  ``update_modules`` (l. 93-95), which is how the first eager attempt peaked at 79 GiB for a 27B model. Nothing reads a
  source module after its ``to_quantized`` returned; the bias reference that ``from_linear`` (l. 299-300) and
  ``SwitchLinear.to_quantized`` copy is left intact. The closing ``[lowmem] eager:`` line counts quantised and released
  weights, and a warning follows when the two differ (then the memory bound below does not hold).

Memory in eager mode: the packed model plus the source tensor being quantised. The packed model is held until it is
written; the peak comes while the last large tensor is quantised (often the output head). For a dense 27B model at
8 bits and group size 64 the packed model is about 26.6 GiB (28.6 GB) and the peak about 29 GiB (31 GB). Sizing rule:
free memory >= packed size (the bf16 checkpoint x (bits + 32 / group size) / 16, about 0.53 of it at 8 bits) + the
largest bf16 tensor (usually the token embedding or the output head) + a few GB; ``run_convert.sh`` computes this floor
from the source's shard headers with ``memory_floor.py``. ``--memory-limit-gib`` does not cap it: MLX's limit is a
guideline (above it the scheduler waits for in-flight work, then the allocation proceeds), and the packed model is
simply held until it is written.

Verification. A prototype of the eager order (the same wrapped ``mx.quantize``, but with the default device left on
the GPU, no weight release and the constructors' random initialisations evaluated too) converted a 27B model on the
GPU, and the tensors that model shares verbatim with an earlier checkpoint (token embedding and output head) matched
that checkpoint's lazily converted copy bit for bit. The implementation in this file is checked by ``selftest.py``,
which converts a tiny model in both orders and compares the outputs byte for byte; run it with ``--device gpu`` before
the first real conversion on a machine.

``--quantize-order lazy``: the stock order. Before each shard is written its tensors are evaluated one at a time and
the buffer cache is released after each, so the footprint stays around one shard (about 5 GiB) plus one bf16 tensor.
Use it only when the source is on a fast local SSD, for example on a Mac with too little memory for eager after copying
the bf16 checkpoint to its internal disk.

The save hook (each shard's tensors evaluated one at a time, then written) runs in both orders; in eager mode it only
bounds the non-quantised tensors (norms, biases), which are read on the CPU.

What neither order protects against: other GPU work on the same machine. Metal's watchdog kills a long command buffer
when another process (LM Studio, a model server, a game) monopolises the GPU, and a killed buffer poisons this
process's Metal context, so the conversion must have the GPU to itself. ``--device cpu`` exists for self-tests only:
CPU quantisation is not bit-identical to GPU quantisation.

The output directory must not exist (``mlx_lm`` refuses to write into one). A conversion is not resumable: after a
crash delete the partial directory and start again.
"""
from __future__ import annotations

import argparse
import contextlib
import time
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

_original_save = mx.save_safetensors
QUANTIZE_ORDERS = ('eager', 'lazy')
RELEASED_MARK = '_lowmem_weight_released'  # set on a source module whose weight was replaced by a placeholder
RELEASING_MARK = '_lowmem_releasing'  # set on a to_quantized function that already releases its source weight


def _eval_one_at_a_time(path: str, shard: Dict[str, Any], **kwargs: Any) -> None:
    """Evaluate ``shard``'s arrays sequentially, releasing cached buffers between them, then save.

    :param path: Shard file path.
    :param shard: Tensor name -> lazily computed (or, in eager mode, already evaluated) array.
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


class EagerScope:
    """State shared by the eager wrappers while one conversion runs.

    ``source`` is the weight that the enclosing wrapped ``to_quantized`` call is converting (``None`` outside such a
    call); the counters feed the log line that ``selftest.py`` checks; ``covered`` names the layer classes besides
    ``nn.Linear`` and ``nn.Embedding`` whose ``to_quantized`` was wrapped during ``nn.quantize``.
    """

    def __init__(self) -> None:
        """Start with no enclosing ``to_quantized`` call and zero counters.

        :returns: None.
        """
        self.source: Optional[Any] = None
        self.quantised = 0
        self.released = 0
        self.left_lazy = 0
        self.covered: List[str] = []

    def summary(self, device_name: str) -> str:
        """One log line with the counters.

        :param device_name: Name of the device the quantisation kernels ran on (``gpu`` or ``cpu``).
        :returns: The line, prefixed ``[lowmem] eager:``.
        """
        extra = f'; other layer types: {", ".join(sorted(set(self.covered)))}' if self.covered else ''
        return (f'[lowmem] eager: quantised {self.quantised}, released {self.released}, '
                f'init left lazy {self.left_lazy} (kernel device {device_name}){extra}')

    def warning(self) -> Optional[str]:
        """A warning line when the counters show weights that were quantised but not released (or the reverse).

        :returns: The line, prefixed ``[lowmem] warning:``, or None when every quantised weight was released.
        """
        if self.quantised == self.released:
            return None
        return (f'[lowmem] warning: {self.quantised} weights quantised eagerly but {self.released} source weights '
                'released; some quantisation ran outside a wrapped to_quantized, so peak memory can exceed the '
                'packed model plus one tensor')


def make_eager_quantize(original: Callable[..., Any], quant_device: Any, scope: EagerScope) -> Callable[..., Any]:
    """Wrap :func:`mlx.core.quantize` so that its input is read before the kernel is encoded.

    :param original: The unwrapped :func:`mlx.core.quantize`.
    :param quant_device: Device (or stream) the kernel runs on: ``mx.gpu`` for ``--device gpu``, ``mx.cpu`` otherwise.
    :param scope: Shared state; ``scope.source`` marks which input is the real weight inside a wrapped
        ``to_quantized``.
    :returns: A drop-in replacement for :func:`mlx.core.quantize`.
    """

    def quantize_eagerly(w: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate ``w``, quantise it on ``quant_device``, evaluate the result and release cached buffers.

        Inside a wrapped ``to_quantized``, a call whose input is not the module's own weight (the constructor's random
        initialisation, which ``from_linear`` overwrites) is passed through lazily, exactly as the stock path leaves it.

        :param w: The weight, normally a lazy ``Load`` of a safetensors slice.
        :param args: Positional arguments of :func:`mlx.core.quantize` (group size, bits).
        :param kwargs: Keyword arguments of :func:`mlx.core.quantize` (``mode``).
        :returns: Whatever ``original`` returns (weight, scales[, biases]), evaluated unless passed through.
        """
        if scope.source is not None and w is not scope.source:
            scope.left_lazy += 1
            return original(w, *args, **kwargs)
        mx.eval(w)  # the read completes here, before any GPU command buffer depends on it
        with mx.stream(quant_device):
            out = original(w, *args, **kwargs)
        mx.eval(out)
        mx.clear_cache()
        scope.quantised += 1
        return out

    return quantize_eagerly


def make_releasing_to_quantized(original: Callable[..., Any], scope: EagerScope) -> Callable[..., Any]:
    """Wrap a layer class's ``to_quantized`` so the source weight is dropped as soon as the quantised layer exists.

    :param original: The unwrapped ``to_quantized`` function (``nn.Linear.to_quantized``,
        ``nn.Embedding.to_quantized`` or another layer class's own, such as ``SwitchLinear.to_quantized``).
    :param scope: Shared state; ``scope.source`` is set to the module's weight while ``original`` runs.
    :returns: A replacement method with the same signature, marked with ``RELEASING_MARK``.
    """

    def to_quantized_releasing(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Quantise like ``original``, then replace this (discarded) module's weight by an empty placeholder.

        :param self: The source layer, which ``nn.quantize`` replaces by the returned layer.
        :param args: Positional arguments of ``to_quantized``.
        :param kwargs: Keyword arguments of ``to_quantized`` (``group_size``, ``bits``, ``mode``).
        :returns: The quantised layer from ``original``.
        :raises RuntimeError: When the same module is quantised a second time (a module shared by two paths), since
            its weight is already gone.
        """
        if getattr(self, RELEASED_MARK, False):
            raise RuntimeError('a module whose weight was already released is being quantised again (a module shared '
                               'by two paths?); run with --quantize-order lazy')
        source = self.weight
        previous, scope.source = scope.source, source
        try:
            quantised = original(self, *args, **kwargs)
        finally:
            scope.source = previous
        self.weight = mx.zeros((0,), dtype=source.dtype)  # placeholder; the bias reference is untouched
        setattr(self, RELEASED_MARK, True)  # a plain attribute on an mlx Module, not a parameter
        del source
        mx.clear_cache()
        scope.released += 1
        return quantised

    setattr(to_quantized_releasing, RELEASING_MARK, True)
    to_quantized_releasing.__wrapped__ = original  # type: ignore[attr-defined]
    return to_quantized_releasing


def install_releasing_wrappers(model: Any, scope: EagerScope) -> List[Tuple[type, Optional[Any]]]:
    """Wrap ``to_quantized`` on every class in ``model`` whose method does not release its source weight yet.

    This covers the layer types that define their own ``to_quantized`` instead of inheriting ``nn.Linear``'s, for
    example ``mlx_lm``'s ``SwitchLinear`` (mixture-of-experts weights) and ``MultiLinear``. Each class is checked when
    it is reached, so a subclass that inherits an already wrapped method is left alone and nothing is wrapped twice.

    :param model: The module ``nn.quantize`` is about to quantise (anything with ``modules()``).
    :param scope: Shared state; the wrapped class names are appended to ``scope.covered``.
    :returns: ``(class, the class's own to_quantized or None when it was inherited)`` per wrapped class, for
        :func:`restore_methods`.
    """
    patched: List[Tuple[type, Optional[Any]]] = []
    for module in model.modules():
        cls = type(module)
        method = getattr(cls, 'to_quantized', None)
        if not callable(method) or getattr(method, RELEASING_MARK, False):
            continue
        patched.append((cls, cls.__dict__.get('to_quantized')))
        cls.to_quantized = make_releasing_to_quantized(method, scope)
        scope.covered.append(cls.__name__)
    return patched


def restore_methods(patched: Sequence[Tuple[type, Optional[Any]]]) -> None:
    """Undo :func:`install_releasing_wrappers`.

    :param patched: Its return value.
    :returns: None.
    """
    for cls, own in reversed(patched):
        if own is None:
            delattr(cls, 'to_quantized')  # the class inherited the method; let it inherit again
        else:
            cls.to_quantized = own


def make_covering_quantize(original: Callable[..., Any], scope: EagerScope) -> Callable[..., Any]:
    """Wrap :func:`mlx.nn.quantize` so that every layer type in the model releases its source weight.

    :param original: The unwrapped :func:`mlx.nn.quantize`.
    :param scope: Shared state.
    :returns: A drop-in replacement for :func:`mlx.nn.quantize`.
    """

    def quantize_covering(model: Any, *args: Any, **kwargs: Any) -> Any:
        """Wrap the model's remaining ``to_quantized`` methods, quantise like ``original``, then unwrap them.

        :param model: The module to quantise in place.
        :param args: Positional arguments of :func:`mlx.nn.quantize` (group size, bits).
        :param kwargs: Keyword arguments of :func:`mlx.nn.quantize` (``mode``, ``class_predicate``).
        :returns: Whatever ``original`` returns.
        """
        patched = install_releasing_wrappers(model, scope)
        try:
            return original(model, *args, **kwargs)
        finally:
            restore_methods(patched)

    return quantize_covering


@contextlib.contextmanager
def eager_quantization(quant_device: Any) -> Iterator[EagerScope]:
    """Install the eager wrappers for the duration of the block, then restore the originals.

    :param quant_device: Device the quantisation kernels run on.
    :returns: A context manager yielding the shared :class:`EagerScope`.
    """
    scope = EagerScope()
    originals = ((mx, 'quantize', mx.quantize), (nn, 'quantize', nn.quantize),
                 (nn.Linear, 'to_quantized', nn.Linear.to_quantized),
                 (nn.Embedding, 'to_quantized', nn.Embedding.to_quantized))
    mx.quantize = make_eager_quantize(mx.quantize, quant_device, scope)  # looked up as mx.quantize at call time
    nn.quantize = make_covering_quantize(nn.quantize, scope)  # mlx_lm.utils calls it as nn.quantize
    nn.Linear.to_quantized = make_releasing_to_quantized(nn.Linear.to_quantized, scope)
    nn.Embedding.to_quantized = make_releasing_to_quantized(nn.Embedding.to_quantized, scope)
    try:
        yield scope
    finally:
        for owner, name, value in originals:
            setattr(owner, name, value)


def quantization_hooks(order: str, quant_device: Any) -> ContextManager[Optional[EagerScope]]:
    """The hooks for one quantisation order.

    :param order: ``eager`` or ``lazy``.
    :param quant_device: Device the quantisation kernels run on.
    :returns: :func:`eager_quantization` for ``eager``; for ``lazy`` a no-op context yielding ``None``.
    """
    if order == 'eager':
        return eager_quantization(quant_device)
    return contextlib.nullcontext(None)


def configure_devices(order: str, device: str) -> Any:
    """Set MLX's default device for the conversion and return the device the quantisation kernels run on.

    :param order: ``eager`` (everything but the kernel on the CPU) or ``lazy`` (everything on ``device``).
    :param device: ``gpu`` or ``cpu``.
    :returns: ``mx.gpu`` or ``mx.cpu``.
    """
    quant_device = mx.gpu if device == 'gpu' else mx.cpu
    mx.set_default_device(mx.cpu if order == 'eager' else quant_device)
    return quant_device


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser.

    :returns: The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hf-path', required=True, help='source checkpoint directory (bf16 safetensors)')
    parser.add_argument('--mlx-path', required=True, help='output directory; must not exist')
    parser.add_argument('--q-bits', type=int, default=8)
    parser.add_argument('--q-group-size', type=int, default=64)
    parser.add_argument('--memory-limit-gib', type=float, default=12.0,
                        help='MLX allocator guideline; the scheduler waits for in-flight work above it (does not cap '
                             "eager mode's packed model)")
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu',
                        help='cpu is for self-tests only: CPU quantisation is not bit-identical to GPU quantisation')
    parser.add_argument('--quantize-order', choices=QUANTIZE_ORDERS, default='eager',
                        help='eager (default): read each weight on the CPU, then quantise it at once; holds the packed '
                             'model in memory. lazy: the stock order, only safe from a fast local SSD')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse arguments, pick the devices, bound the allocator, patch the save hook and run the stock conversion.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :returns: None.
    """
    args = build_parser().parse_args(argv)
    quant_device = configure_devices(args.quantize_order, args.device)
    mx.set_cache_limit(0)  # freed buffers go straight back to the OS
    mx.set_memory_limit(int(args.memory_limit_gib * 2**30))
    import mlx_lm  # noqa: E402
    import mlx_lm.utils as utils  # noqa: E402  (patched before convert() imports it)
    utils.mx.save_safetensors = _eval_one_at_a_time
    from mlx_lm.convert import convert  # noqa: E402
    print(f'[lowmem] mlx {mx.__version__}, mlx_lm {mlx_lm.__version__}, quantize order {args.quantize_order}, '
          f'default device {mx.default_device()}, kernel device {args.device}, cache limit 0, '
          f'memory limit {args.memory_limit_gib} GiB', flush=True)
    with quantization_hooks(args.quantize_order, quant_device) as scope:
        convert(hf_path=args.hf_path, mlx_path=args.mlx_path, quantize=True, q_bits=args.q_bits,
                q_group_size=args.q_group_size)
    if scope is not None:
        print(scope.summary(args.device), flush=True)
        if scope.warning() is not None:
            print(scope.warning(), flush=True)
    print(f'[lowmem] done; peak memory {mx.get_peak_memory() / 2**30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
