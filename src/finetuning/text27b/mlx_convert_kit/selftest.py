#!/usr/bin/env python3
"""Kit self-test: a one-minute end-to-end run on a tiny random model, with no network and no shared storage.

Builds a two-layer llama checkpoint (random bf16 weights, a 512-token word-level tokenizer, a chat template and a
generation config), converts it with ``convert_lowmem.py`` on the chosen device in the chosen quantisation order
(``--quantize-order``, default ``eager`` like the converter), converts it again in the other order and compares the two
outputs' weight files byte for byte, checks from the eager conversion's log that every quantised weight was read first
and released, checks the output directory with ``verify_mlx_dir.py``, records it with ``kit_record.py``, checks the
installed libraries against ``requirements.txt`` and loads the result with ``mlx_lm`` for one forward pass. With
``--device gpu`` it also checks that the CPU and the GPU compute the model fix-ups that the eager order moves to the CPU
(a bf16 norm shift ``x + 1.0`` and a float16 -> bf16 cast, over every 16-bit pattern) bit for bit. Run it once with
``--device cpu`` and once with ``--device gpu`` on a new machine before a real conversion; a failure here is an
environment problem (Python, package versions, Metal), not a model problem. It exercises the conversion machinery on a
small llama, not the exact architecture of the real checkpoint.

Exit status 0 when every check passes; the work directory is kept on failure (and with ``--keep``).
"""
from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from kit_record import DEFAULT_REQUIREMENTS, bits_per_weight, pin_mismatches  # noqa: E402
from verify_mlx_dir import check_dir  # noqa: E402

MIN_PYTHON = (3, 12)
CONFIG: Dict[str, Any] = {'architectures': ['LlamaForCausalLM'], 'model_type': 'llama', 'hidden_size': 128,
                          'intermediate_size': 256, 'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 32,
                          'num_hidden_layers': 2, 'rms_norm_eps': 1e-5, 'vocab_size': 512, 'rope_theta': 10000.0,
                          'max_position_embeddings': 2048, 'tie_word_embeddings': False, 'torch_dtype': 'bfloat16'}
CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
ORDERS = ('eager', 'lazy')
EAGER_LINE = re.compile(r'\[lowmem\] eager: quantised (\d+), released (\d+), init left lazy (\d+)')


def random_bf16(mx: Any, *shape: int) -> Any:
    """Small random bf16 tensor.

    :param mx: The ``mlx.core`` module.
    :param shape: Tensor shape.
    :returns: Normal(0, 0.02) values in bf16.
    """
    return (mx.random.normal(shape) * 0.02).astype(mx.bfloat16)


def ones_bf16(mx: Any, n: int) -> Any:
    """A bf16 vector of ones (a norm weight).

    :param mx: The ``mlx.core`` module.
    :param n: Length.
    :returns: Ones in bf16.
    """
    return mx.ones((n,), dtype=mx.bfloat16)


def write_tiny_checkpoint(directory: Path) -> int:
    """Random bf16 llama weights in Hugging Face layout plus tokenizer, chat template and generation config.

    :param directory: Directory to create.
    :returns: Number of tensors written.
    """
    import mlx.core as mx
    from tokenizers import Tokenizer, models, pre_tokenizers

    hidden, inter, vocab = CONFIG['hidden_size'], CONFIG['intermediate_size'], CONFIG['vocab_size']
    kv = CONFIG['num_key_value_heads'] * CONFIG['head_dim']
    tensors = {'model.embed_tokens.weight': random_bf16(mx, vocab, hidden), 'model.norm.weight': ones_bf16(mx, hidden),
               'lm_head.weight': random_bf16(mx, vocab, hidden)}
    for i in range(CONFIG['num_hidden_layers']):
        prefix = f'model.layers.{i}.'
        tensors.update({prefix + 'self_attn.q_proj.weight': random_bf16(mx, hidden, hidden),
                        prefix + 'self_attn.k_proj.weight': random_bf16(mx, kv, hidden),
                        prefix + 'self_attn.v_proj.weight': random_bf16(mx, kv, hidden),
                        prefix + 'self_attn.o_proj.weight': random_bf16(mx, hidden, hidden),
                        prefix + 'mlp.gate_proj.weight': random_bf16(mx, inter, hidden),
                        prefix + 'mlp.up_proj.weight': random_bf16(mx, inter, hidden),
                        prefix + 'mlp.down_proj.weight': random_bf16(mx, hidden, inter),
                        prefix + 'input_layernorm.weight': ones_bf16(mx, hidden),
                        prefix + 'post_attention_layernorm.weight': ones_bf16(mx, hidden)})
    mx.eval(*tensors.values())
    directory.mkdir(parents=True)
    mx.save_safetensors(str(directory / 'model.safetensors'), tensors, metadata={'format': 'pt'})
    (directory / 'config.json').write_text(json.dumps(CONFIG, indent=1), encoding='utf-8')
    (directory / 'generation_config.json').write_text(
        json.dumps({'bos_token_id': 1, 'eos_token_id': 2, 'max_length': 64}), encoding='utf-8')
    words = {'<unk>': 0, '<s>': 1, '</s>': 2, **{f'tok{i}': i for i in range(3, vocab)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=words, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / 'tokenizer.json'))
    (directory / 'tokenizer_config.json').write_text(json.dumps(
        {'tokenizer_class': 'PreTrainedTokenizerFast', 'unk_token': '<unk>', 'bos_token': '<s>', 'eos_token': '</s>',
         'model_max_length': 2048, 'chat_template': CHAT_TEMPLATE}, indent=1), encoding='utf-8')
    return len(tensors)


def convert_tiny(source: Path, output: Path, log: Path, device: str, order: str) -> int:
    """Run ``convert_lowmem.py`` on the tiny checkpoint, logging stdout and stderr to ``log``.

    :param source: The tiny bf16 checkpoint.
    :param output: Output directory (must not exist).
    :param log: Log file to write.
    :param device: ``cpu`` or ``gpu``.
    :param order: ``eager`` or ``lazy``.
    :returns: The converter's exit status.
    """
    with log.open('w', encoding='utf-8') as handle:
        proc = subprocess.run([sys.executable, '-u', str(KIT / 'convert_lowmem.py'), '--hf-path', str(source),
                               '--mlx-path', str(output), '--q-bits', '8', '--q-group-size', '64', '--device', device,
                               '--quantize-order', order], stdout=handle, stderr=subprocess.STDOUT)
    return proc.returncode


def eager_counts(log_text: str) -> Optional[Dict[str, int]]:
    """The counters of the converter's ``[lowmem] eager:`` line.

    :param log_text: Conversion log.
    :returns: ``quantised``, ``released`` and ``init_left_lazy``, or None when the line is absent.
    """
    found = EAGER_LINE.search(log_text)
    if found is None:
        return None
    return dict(zip(('quantised', 'released', 'init_left_lazy'), map(int, found.groups())))


def eager_hooks_ran(counts: Optional[Dict[str, int]]) -> bool:
    """Whether the eager conversion quantised at least one weight and released every weight it quantised.

    :param counts: :func:`eager_counts` of the eager conversion's log.
    :returns: True when the counters are present, non-zero and balanced.
    """
    return counts is not None and counts['quantised'] > 0 and counts['released'] == counts['quantised']


def fixups_identical(mx: Any, first: Any, second: Any) -> Dict[str, bool]:
    """Whether two devices compute the fix-ups that the eager order moves to the CPU bit for bit.

    Exhaustive over the 65,536 16-bit patterns: ``x + 1.0`` in bf16 (the norm-weight shift some models' ``sanitize``
    applies) and a float16 -> bf16 cast (the dtype cast for a float16 source). Two NaNs count as equal whatever their
    payloads.

    :param mx: The ``mlx.core`` module.
    :param first: One device (``mx.cpu``).
    :param second: The other device (``mx.gpu``).
    :returns: Operation name -> whether both devices produced the same bits.
    """
    bits = mx.arange(1 << 16, dtype=mx.uint32).astype(mx.uint16)
    mx.eval(bits)
    cases = {'bf16 + 1.0': lambda: mx.view(bits, mx.bfloat16) + 1.0,
             'float16 -> bf16': lambda: mx.view(bits, mx.float16).astype(mx.bfloat16)}
    result: Dict[str, bool] = {}
    for name, operation in cases.items():
        outputs = []
        for device in (first, second):
            with mx.stream(device):
                outputs.append(operation())
            mx.eval(outputs[-1])
        a, b = outputs
        same = (mx.view(a, mx.uint16) == mx.view(b, mx.uint16)) | (mx.isnan(a) & mx.isnan(b))
        result[name] = bool(same.all())
    return result


def same_weights(first: Path, second: Path) -> bool:
    """Whether two converted directories hold the same safetensors files with identical bytes.

    :param first: One model directory.
    :param second: The other model directory.
    :returns: True when both have the same non-empty set of ``*.safetensors`` names and every pair is byte-equal.
    """
    names = sorted(p.name for p in first.glob('*.safetensors'))
    return (bool(names) and names == sorted(p.name for p in second.glob('*.safetensors'))
            and all(filecmp.cmp(first / name, second / name, shallow=False) for name in names))


def run_selftest(device: str, work: Path, quantize_order: str = 'eager') -> Dict[str, Any]:
    """Build, convert, compare the two quantisation orders, verify, record and check pins, then run a forward pass.

    :param device: ``cpu`` or ``gpu``.
    :param work: Empty work directory.
    :param quantize_order: Order of the recorded conversion (``eager`` or ``lazy``); the other order is converted
        too and compared byte for byte.
    :returns: Report with ``ok`` and the individual checks.
    """
    import mlx.core as mx
    mx.set_default_device(mx.gpu if device == 'gpu' else mx.cpu)
    source, output, log = work / 'tiny-bf16', work / 'tiny-MLX-8bit', work / 'tiny.convert.log'
    other = ORDERS[1] if quantize_order == ORDERS[0] else ORDERS[0]
    n_tensors = write_tiny_checkpoint(source)
    started = time.time()
    returncode = convert_tiny(source, output, log, device, quantize_order)
    seconds = time.time() - started
    text = log.read_text(encoding='utf-8', errors='replace')
    checks: Dict[str, bool] = {'convert_exit_0': returncode == 0,
                               'lowmem_hook_ran': '[lowmem]' in text and 'tensors evaluated' in text,
                               'bits_per_weight_logged': 'bits per weight' in text}
    other_output, other_log = work / f'tiny-MLX-8bit-{other}', work / f'tiny.{other}.convert.log'
    checks['orders_bit_identical'] = (convert_tiny(source, other_output, other_log, device, other) == 0
                                      and same_weights(output, other_output))
    logs = {quantize_order: text, other: other_log.read_text(encoding='utf-8', errors='replace')}
    eager = eager_counts(logs['eager'])  # whichever conversion ran in the eager order
    checks['eager_hooks_ran'] = eager_hooks_ran(eager)
    fixups: Optional[Dict[str, bool]] = None
    if device == 'gpu':  # the eager order computes these on the CPU; the lazy order on the GPU
        fixups = fixups_identical(mx, mx.cpu, mx.gpu)
        checks['fixups_cpu_gpu_identical'] = all(fixups.values())
    report = check_dir(output)
    checks['output_complete'] = report['complete']
    record = subprocess.run([sys.executable, str(KIT / 'kit_record.py'), '--log', str(log), '--out',
                             str(work / 'tiny.CONVERTED.json'), '--hf-path', str(source), '--mlx-path', str(output),
                             '--device', device, '--seconds', f'{seconds:.1f}'], capture_output=True, text=True)
    checks['record_written'] = record.returncode == 0 and (work / 'tiny.CONVERTED.json').is_file()
    mismatches = pin_mismatches(DEFAULT_REQUIREMENTS) if DEFAULT_REQUIREMENTS.is_file() else {}
    checks['pins_match'] = not mismatches
    logits_shape: Optional[List[int]] = None
    if all(checks.values()):  # load the converted model and run one forward pass (works on both devices)
        from mlx_lm import load
        model, tokenizer = load(str(output))
        tokens = tokenizer.encode('tok5 tok6 tok7')
        logits = model(mx.array(tokens)[None])
        mx.eval(logits)
        logits_shape = list(logits.shape)
        checks['forward_pass'] = (logits_shape == [1, len(tokens), CONFIG['vocab_size']]
                                  and bool(mx.isfinite(logits).all()))
    return {'ok': all(checks.values()), 'device': device, 'quantize_order': quantize_order, 'checks': checks,
            'convert_seconds': round(seconds, 1), 'tensors': n_tensors, 'bits_per_weight': bits_per_weight(text),
            'eager': eager, 'fixups': fixups, 'quantization': report['quantization'],
            'output_problems': report['problems'],
            'pin_mismatches': {name: {'pinned': p, 'installed': i} for name, (p, i) in mismatches.items()},
            'logits_shape': logits_shape, 'workdir': str(work), 'log_tail': text.splitlines()[-6:]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point.

    :param argv: Arguments (defaults to ``sys.argv[1:]``).
    :returns: Exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--quantize-order', choices=ORDERS, default='eager',
                        help="order of the recorded conversion (the other order is converted too and compared)")
    parser.add_argument('--workdir', type=Path, help='work directory (default: a fresh temporary directory)')
    parser.add_argument('--keep', action='store_true', help='keep the work directory on success')
    args = parser.parse_args(argv)
    if sys.version_info < MIN_PYTHON:
        print(f'Python {".".join(map(str, MIN_PYTHON))} or newer is required (the pinned numpy needs it); '
              f'this is {sys.version.split()[0]}', file=sys.stderr)
        return 1
    work = args.workdir or Path(tempfile.mkdtemp(prefix='mlx_kit_selftest_'))
    work.mkdir(parents=True, exist_ok=True)
    result = run_selftest(args.device, work, args.quantize_order)
    print(json.dumps(result, indent=1))
    if result['ok'] and not args.keep:
        shutil.rmtree(work)
    elif not result['ok']:
        print(f'FAILED; work directory kept at {work}', file=sys.stderr)
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
