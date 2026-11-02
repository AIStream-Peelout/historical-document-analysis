"""Unit tests for the MLX conversion kit's stdlib parts, the converter's eager hooks (with a recording fake
``mlx.core`` / ``mlx.nn``) and the runner's refusal paths (no MLX needed)."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pytest

from src.finetuning.text27b.mlx_convert_kit import gpu_quant_identity, kit_record, memory_floor, verify_mlx_dir

KIT = Path(__file__).resolve().parents[1] / 'src/finetuning/text27b/mlx_convert_kit'
TensorSpec = Tuple[str, List[int], int]
FAKE_MX = SimpleNamespace(array_equal=np.array_equal, abs=np.abs, float32=np.float32, uint8=np.uint8,
                          view=lambda a, dtype: a.view(dtype))


def write_safetensors(path: Path, tensors: Dict[str, TensorSpec], metadata: Optional[Dict[str, str]] = None) -> None:
    """Write a structurally valid safetensors file with zeroed data.

    :param path: Destination.
    :param tensors: Name -> (dtype, shape, byte length).
    :param metadata: Optional ``__metadata__`` block.
    """
    header: Dict[str, object] = {}
    offset = 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + nbytes]}
        offset += nbytes
    if metadata:
        header['__metadata__'] = metadata
    blob = json.dumps(header).encode('utf-8')
    path.write_bytes(struct.pack('<Q', len(blob)) + blob + b'\0' * offset)


def write_header_only(path: Path, tensors: Dict[str, Tuple[str, List[int]]]) -> None:
    """Write only a safetensors header (no tensor data), for tensors too large to write.

    :param path: Destination.
    :param tensors: Name -> (dtype, shape); the offsets follow from the dtype's element size.
    """
    element_bytes = {'BF16': 2, 'F16': 2, 'F32': 4}
    header: Dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = element_bytes[dtype] * int(np.prod(shape, dtype=np.int64))
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode('utf-8')
    path.write_bytes(struct.pack('<Q', len(blob)) + blob)


def make_model_dir(root: Path) -> Path:
    """A complete fake MLX model directory with two shards.

    :param root: Parent directory.
    :returns: The model directory.
    """
    model = root / 'tiny-MLX-8bit'
    model.mkdir()
    weight_map = {'a.weight': 'model-00001-of-00002.safetensors', 'a.scales': 'model-00001-of-00002.safetensors',
                  'b.weight': 'model-00002-of-00002.safetensors'}
    write_safetensors(model / 'model-00001-of-00002.safetensors',
                      {'a.weight': ('U32', [4, 2], 32), 'a.scales': ('BF16', [4, 1], 8)}, {'format': 'mlx'})
    write_safetensors(model / 'model-00002-of-00002.safetensors', {'b.weight': ('U32', [2, 2], 16)}, {'format': 'mlx'})
    (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
    (model / 'config.json').write_text(json.dumps({'quantization': {'bits': 8, 'group_size': 64}}))
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json'):
        (model / name).write_text('{}')
    return model


def make_source_dir(root: Path, complete: bool = True) -> Path:
    """A fake bf16 source checkpoint directory for the runner's preflight checks.

    :param root: Parent directory.
    :param complete: Whether to include generation_config.json and the chat template.
    :returns: The source directory.
    """
    source = root / 'fake-bf16'
    source.mkdir(parents=True)
    (source / 'config.json').write_text('{}')
    (source / 'model.safetensors.index.json').write_text('{"weight_map": {}}')
    if complete:
        (source / 'generation_config.json').write_text('{}')
        (source / 'tokenizer_config.json').write_text('{"chat_template": "x"}')
    return source


def run_runner(*args: str) -> subprocess.CompletedProcess:
    """Run run_convert.sh with the main interpreter as PYTHON.

    :param args: Script arguments.
    :returns: The completed process (stdout and stderr captured).
    """
    env = {**os.environ, 'PYTHON': sys.executable}
    return subprocess.run(['zsh', str(KIT / 'run_convert.sh'), *args], capture_output=True, text=True, env=env)


def test_read_safetensors_header_reports_length_and_metadata(tmp_path: Path) -> None:
    """The header parser returns tensors, the implied file length and the metadata, and rejects short files."""
    file = tmp_path / 'x.safetensors'
    write_safetensors(file, {'w': ('BF16', [2, 3], 12)}, {'format': 'mlx'})
    header, expected, metadata = verify_mlx_dir.read_safetensors_header(file)
    assert set(header) == {'w'} and header['w']['shape'] == [2, 3]
    assert expected == file.stat().st_size and metadata == {'format': 'mlx'}
    short = tmp_path / 'short.safetensors'
    short.write_bytes(b'\x01\x02')
    with pytest.raises(ValueError):
        verify_mlx_dir.read_safetensors_header(short)


def test_check_dir_complete_then_truncated_then_missing(tmp_path: Path) -> None:
    """A complete directory passes; a truncated shard and a missing shard are each reported."""
    model = make_model_dir(tmp_path)
    report = verify_mlx_dir.check_dir(model)
    assert report['complete'] and report['shards'] == 2 and report['tensors'] == 3
    assert report['quantization'] == {'bits': 8, 'group_size': 64}
    shard = model / 'model-00002-of-00002.safetensors'
    shard.write_bytes(shard.read_bytes()[:-4])
    problems = verify_mlx_dir.check_dir(model)['problems']
    assert len(problems) == 1 and 'bytes on disk' in problems[0]
    shard.unlink()
    problems = verify_mlx_dir.check_dir(model)['problems']
    assert problems == ['missing shard model-00002-of-00002.safetensors']


def test_check_dir_detects_index_mismatch_and_missing_files(tmp_path: Path) -> None:
    """A tensor listed in the index but absent from its shard, and a missing required file, are reported."""
    model = make_model_dir(tmp_path)
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    index['weight_map']['c.weight'] = 'model-00002-of-00002.safetensors'
    (model / 'model.safetensors.index.json').write_text(json.dumps(index))
    (model / 'chat_template.jinja').unlink()
    problems = verify_mlx_dir.check_dir(model)['problems']
    assert any('tensor set differs' in p and '1 only in the index' in p for p in problems)
    assert any("missing files: ['chat_template.jinja']" in p for p in problems)
    assert verify_mlx_dir.check_dir(tmp_path / 'nowhere')['problems'] == ['not a directory']


def test_manifest_roundtrip_and_compare(tmp_path: Path) -> None:
    """A manifest matches its directory; content, size and extra-file differences are each reported."""
    model = make_model_dir(tmp_path)
    manifest = verify_mlx_dir.build_manifest(model, hashes=True)
    assert set(manifest['files']) == set(verify_mlx_dir.check_dir(model)['files'])
    assert verify_mlx_dir.compare_manifest(model, manifest) == []
    shard = model / 'model-00002-of-00002.safetensors'
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF  # same length, different content
    shard.write_bytes(bytes(data))
    assert verify_mlx_dir.compare_manifest(model, manifest) == [f'{shard.name}: sha256 differs from the manifest']
    assert verify_mlx_dir.compare_manifest(model, manifest, hashes=False) == []
    (model / 'extra.txt').write_text('x')
    (model / 'config.json').write_text('{"quantization": null, "longer": true}')
    problems = verify_mlx_dir.compare_manifest(model, manifest, hashes=False)
    assert any('config.json' in p and 'manifest says' in p for p in problems)
    assert "files not in the manifest: ['extra.txt']" in problems


def test_dotfiles_are_ignored_everywhere(tmp_path: Path) -> None:
    """Finder and SMB dotfiles affect neither the check, the manifest nor the comparison."""
    model = make_model_dir(tmp_path)
    clean = verify_mlx_dir.check_dir(model)
    manifest = verify_mlx_dir.build_manifest(model, hashes=False)
    (model / '.DS_Store').write_bytes(b'\0' * 6148)
    (model / '._config.json').write_bytes(b'\0' * 4096)
    dirty = verify_mlx_dir.check_dir(model)
    assert dirty['complete'] and dirty['files'] == clean['files'] and dirty['bytes'] == clean['bytes']
    assert set(verify_mlx_dir.build_manifest(model, hashes=False)['files']) == set(manifest['files'])
    assert verify_mlx_dir.compare_manifest(model, manifest, hashes=False) == []
    manifest['files']['.DS_Store'] = {'bytes': 1, 'sha256': None}  # recorded by an older manifest
    assert verify_mlx_dir.compare_manifest(model, manifest, hashes=False) == []


def test_verify_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI returns 0 for a complete, matching directory and 1 once a shard is missing."""
    model = make_model_dir(tmp_path)
    manifest_path = tmp_path / 'm.json'
    assert verify_mlx_dir.main([str(model), '--manifest', str(manifest_path), '--no-hash']) == 0
    assert json.loads(manifest_path.read_text())['hashed'] is False
    assert verify_mlx_dir.main([str(model), '--compare', str(manifest_path)]) == 0
    (model / 'model-00001-of-00002.safetensors').unlink()
    assert verify_mlx_dir.main([str(model), '--compare', str(manifest_path)]) == 1
    capsys.readouterr()


def test_locate_tensors_with_and_without_index(tmp_path: Path) -> None:
    """Tensors are found by scanning shard headers, or through the index when one exists."""
    shards = tmp_path / 'shards'
    shards.mkdir()
    write_safetensors(shards / 'model-00003-of-00018.safetensors',
                      {'model.language_model.embed_tokens.weight': ('BF16', [8, 4], 64)})
    write_safetensors(shards / 'model-00018-of-00018.safetensors',
                      {'lm_head.weight': ('BF16', [8, 4], 64), 'mtp.x': ('BF16', [2], 4)})
    keys = [hf for hf, _ in gpu_quant_identity.DEFAULT_PAIRS] + ['absent']
    found = gpu_quant_identity.locate_tensors(shards, keys)
    assert found == {'model.language_model.embed_tokens.weight': shards / 'model-00003-of-00018.safetensors',
                     'lm_head.weight': shards / 'model-00018-of-00018.safetensors'}
    (shards / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'lm_head.weight': 'model-00018-of-00018.safetensors'}}))
    assert gpu_quant_identity.locate_tensors(shards, keys) == {'lm_head.weight': shards / 'model-00018-of-00018.safetensors'}


def test_compare_parts_and_verdict_with_numpy_as_mx() -> None:
    """Equality, word and byte difference counts, shape mismatches and the overall verdict."""
    made = [np.array([[1, 2], [3, 4]], dtype=np.uint32), np.array([1.0, 2.0], dtype=np.float32),
            np.array([0.0, 0.5], dtype=np.float32)]
    same = [a.copy() for a in made]
    result = gpu_quant_identity.compare_parts(FAKE_MX, made, same)
    assert result['identical'] and all(result[p]['equal'] for p in gpu_quant_identity.PARTS)
    different = [made[0].copy(), np.array([1.0, 2.5], dtype=np.float32), made[2].copy()]
    result = gpu_quant_identity.compare_parts(FAKE_MX, made, different)
    assert not result['identical'] and result['weight']['equal'] and not result['scales']['equal']
    assert result['scales']['n_diff'] == 1 and result['scales']['fraction_diff'] == 0.5
    assert result['scales']['max_abs_diff'] == 0.5 and result['scales']['unit'] == 'elements'
    word_diff = [np.array([[1, 2], [3, 0x01010104]], dtype=np.uint32), made[1].copy(), made[2].copy()]
    result = gpu_quant_identity.compare_parts(FAKE_MX, made, word_diff)
    assert result['weight']['n_diff'] == 1 and result['weight']['fraction_diff'] == 0.25
    assert result['weight']['unit'] == 'packed_uint32_words'
    assert result['weight']['n_diff_bytes'] == 3 and result['weight']['fraction_diff_bytes'] == 3 / 16
    shape_mismatch = [made[0][:1], made[1], made[2]]
    assert gpu_quant_identity.compare_parts(FAKE_MX, made, shape_mismatch)['weight']['equal'] is False
    assert gpu_quant_identity.overall_verdict([{'comparison': {'identical': True}}, {'comparison': {'identical': True}}])
    assert not gpu_quant_identity.overall_verdict([{'comparison': {'identical': True}}, {'comparison': {'identical': False}}])
    assert not gpu_quant_identity.overall_verdict([])
    assert (gpu_quant_identity.EXIT_IDENTICAL, gpu_quant_identity.EXIT_DIFFERENT, gpu_quant_identity.EXIT_MISSING) == (0, 10, 11)


def test_parse_pair() -> None:
    """Pairs are split on the first ``=`` and stripped; a value without ``=`` is rejected."""
    assert gpu_quant_identity.parse_pair(' a.weight = b ') == ('a.weight', 'b')
    with pytest.raises(Exception):
        gpu_quant_identity.parse_pair('nopair')


def test_bits_per_weight_and_record(tmp_path: Path) -> None:
    """The last bits-per-weight figure is taken; the record carries identity, manifest, pins and libraries."""
    text = '[INFO] Quantized model with 8.501 bits per weight.\nx\n[INFO] Quantized model with 8.502 bits per weight.\n'
    assert kit_record.bits_per_weight(text) == 8.502
    assert kit_record.bits_per_weight('nothing') is None
    requirements = tmp_path / 'req.txt'
    requirements.write_text('# pins\nnumpy==0.0.1\npytest==%s\n\nnot-installed-thing==1.0\n' % kit_record.installed_version('pytest'))
    identity = {'identical': False, 'device': 'gpu', 'reference_dir': '/ref', 'time_utc': 't'}
    record = kit_record.build_record(text, '/src', '/dst', 'gpu', 123.4, identity, '/dst.MANIFEST.json', requirements)
    assert record['bits_per_weight'] == 8.502 and record['convert_seconds'] == 123.4
    assert record['identity_test'] == identity and record['manifest'] == '/dst.MANIFEST.json'
    assert set(record['libraries']) == set(kit_record.LIBRARIES) and 'python' in record['machine']
    assert set(record['pins']['mismatches']) == {'numpy', 'not-installed-thing'}
    assert record['pins']['mismatches']['not-installed-thing'] == {'pinned': '1.0', 'installed': None}
    log = tmp_path / 'c.log'
    log.write_text(text)
    out = tmp_path / 'r.json'
    assert kit_record.main(['--log', str(log), '--out', str(out), '--hf-path', 'a', '--mlx-path', 'b', '--seconds', '1']) == 0
    assert json.loads(out.read_text())['bits_per_weight'] == 8.502


def test_requirements_parsing_and_check_pins(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Pins parse past comments; --check-pins exits 0 only when every pin matches the running interpreter."""
    requirements = tmp_path / 'req.txt'
    requirements.write_text('# comment\npytest == %s  # trailing\n\nnumpy==0.0.1\n' % kit_record.installed_version('pytest'))
    assert kit_record.parse_requirements(requirements) == {'pytest': kit_record.installed_version('pytest'), 'numpy': '0.0.1'}
    assert set(kit_record.pin_mismatches(requirements)) == {'numpy'}
    assert kit_record.main(['--check-pins', str(requirements)]) == 1
    requirements.write_text('pytest==%s\n' % kit_record.installed_version('pytest'))
    assert kit_record.main(['--check-pins', str(requirements)]) == 0
    assert kit_record.installed_version('definitely-not-installed-xyz') is None
    with pytest.raises(SystemExit):
        kit_record.main(['--log', 'x'])  # the record mode needs its four arguments
    capsys.readouterr()


@pytest.mark.skipif(shutil.which('zsh') is None, reason='the runner is a zsh script')
def test_runner_refuses_before_touching_anything(tmp_path: Path) -> None:
    """Every preflight refusal exits 1 with its message and leaves no staging directory or lock behind."""
    assert 'required' in run_runner().stderr
    bad = run_runner('--hf-path', str(tmp_path / 'nope'), '--mlx-path', str(tmp_path / 'out'))
    assert bad.returncode == 1 and 'not a directory' in bad.stderr
    incomplete = make_source_dir(tmp_path / 'inc', complete=False)
    assert 'generation_config.json' in run_runner('--hf-path', str(incomplete), '--mlx-path', str(tmp_path / 'out')).stderr
    source = make_source_dir(tmp_path)
    missing_identity = run_runner('--hf-path', str(source), '--mlx-path', str(tmp_path / 'out'),
                                  '--identity-json', str(tmp_path / 'no.json'))
    assert missing_identity.returncode == 1 and 'identity report' in missing_identity.stderr
    (tmp_path / 'exists').mkdir()
    assert 'exists' in run_runner('--hf-path', str(source), '--mlx-path', str(tmp_path / 'exists')).stderr
    (tmp_path / 'out.partial').mkdir()
    foreign = run_runner('--hf-path', str(source), '--mlx-path', str(tmp_path / 'out'))
    assert foreign.returncode == 1 and 'another converter' in foreign.stderr and (tmp_path / 'out.partial').is_dir()
    (tmp_path / 'out.partial').rmdir()
    (tmp_path / 'out.lock').mkdir()
    (tmp_path / 'out.lock' / 'owner').write_text('other-mac pid 1 started earlier')
    locked = run_runner('--hf-path', str(source), '--mlx-path', str(tmp_path / 'out'))
    assert locked.returncode == 1 and 'other-mac pid 1' in locked.stderr and (tmp_path / 'out.lock').is_dir()
    (tmp_path / 'out.lock' / 'owner').unlink()
    (tmp_path / 'out.lock').rmdir()
    no_mlx = run_runner('--hf-path', str(source), '--mlx-path', str(tmp_path / 'out'))  # main venv has no mlx
    assert no_mlx.returncode == 1 and ('no mlx' in no_mlx.stderr or 'requirements.txt' in no_mlx.stderr)
    assert not (tmp_path / 'out.lock').exists() and not (tmp_path / 'out.kit-partial').exists()
    assert not (tmp_path / 'out').exists()


class FakeArray:
    """A named stand-in for an mlx array (only ``dtype`` and ``size`` are read by the converter)."""

    def __init__(self, name: str, dtype: str = 'bfloat16', size: int = 8) -> None:
        """Remember the name, dtype and size.

        :param name: Label used in the recorded events.
        :param dtype: Dtype label.
        :param size: Element count.
        """
        self.name, self.dtype, self.size = name, dtype, size


def _names(arrays: Tuple[Any, ...]) -> List[str]:
    """Flatten (possibly nested) tuples of fake arrays into their names.

    :param arrays: Arguments given to the fake ``eval``.
    :returns: Names in order.
    """
    names: List[str] = []
    for item in arrays:
        names.extend(_names(tuple(item)) if isinstance(item, (tuple, list)) else [item.name])
    return names


class RecordingMx(types.ModuleType):
    """A fake ``mlx.core`` that records evaluations, stream switches, kernels and cache releases in call order."""

    def __init__(self) -> None:
        """Start with an empty event log on the default stream.

        :returns: None.
        """
        super().__init__('mlx.core')
        self.events: List[Tuple[Any, ...]] = []
        self.gpu, self.cpu, self.current, self.default = 'gpu', 'cpu', 'default-stream', None
        self.last_out: Optional[Tuple[FakeArray, ...]] = None
        self.__version__ = 'fake'

    def eval(self, *arrays: Any) -> None:
        """Record which arrays were evaluated.

        :param arrays: Arrays or tuples of arrays.
        :returns: None.
        """
        self.events.append(('eval', _names(arrays)))

    @contextlib.contextmanager
    def stream(self, device: str) -> Iterator[None]:
        """Make ``device`` the current stream for the block, recording entry and exit.

        :param device: Device label.
        :returns: A context manager.
        """
        previous, self.current = self.current, device
        self.events.append(('stream', device))
        try:
            yield
        finally:
            self.current = previous
            self.events.append(('stream_exit', device))

    def quantize(self, w: FakeArray, *args: Any, **kwargs: Any) -> Tuple[FakeArray, ...]:
        """Record the kernel with the stream it was created on and return packed stand-ins.

        :param w: Input array.
        :param args: Positional arguments (recorded).
        :param kwargs: Keyword arguments (recorded).
        :returns: (weight, scales, biases) stand-ins, also kept as ``last_out``.
        """
        self.events.append(('quantize', w.name, self.current, args, kwargs))
        self.last_out = (FakeArray(w.name + '.q', 'uint32'), FakeArray(w.name + '.scales'),
                         FakeArray(w.name + '.biases'))
        return self.last_out

    def clear_cache(self) -> None:
        """Record a cache release.

        :returns: None.
        """
        self.events.append(('clear_cache',))

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> FakeArray:
        """An empty placeholder.

        :param shape: Shape (only ``(0,)`` is used).
        :param dtype: Dtype label.
        :returns: A size-0 fake array.
        """
        return FakeArray('placeholder', dtype or 'float32', size=0)

    def set_default_device(self, device: str) -> None:
        """Remember the default device.

        :param device: Device label.
        :returns: None.
        """
        self.default = device

    def save_safetensors(self, *args: Any, **kwargs: Any) -> None:
        """Do nothing (captured at import as the original save).

        :param args: Ignored.
        :param kwargs: Ignored.
        :returns: None.
        """


def stock_like_to_quantized(self: Any, group_size: Optional[int] = None, bits: Optional[int] = None,
                            mode: str = 'affine') -> SimpleNamespace:
    """Mimic ``QuantizedLinear.from_linear``: quantise a random init, then the layer's weight; copy the bias reference.

    ``mx.quantize`` is looked up on the ``mlx.core`` module at call time, as ``mlx.nn.layers.quantized`` does.

    :param self: The source layer.
    :param group_size: Group size.
    :param bits: Bits.
    :param mode: Quantisation mode.
    :returns: A stand-in quantised layer.
    """
    core = sys.modules['mlx.core']
    core.quantize(FakeArray('random-init', 'float32'), group_size, bits, mode=mode)  # the constructor's throwaway
    weight, scales, biases = core.quantize(self.weight, group_size, bits, mode=mode)
    return SimpleNamespace(weight=weight, scales=scales, biases=biases, bias=getattr(self, 'bias', None))


class FakeModel:
    """A stand-in model: named leaf layers and a ``modules()`` listing like ``mlx.nn.Module``'s."""

    def __init__(self, **leaves: Any) -> None:
        """Keep the leaves.

        :param leaves: Name -> layer.
        """
        self.leaves: Dict[str, Any] = dict(leaves)

    def modules(self) -> List[Any]:
        """The model itself and its leaves.

        :returns: All modules.
        """
        return [self, *self.leaves.values()]


@pytest.fixture()
def lowmem(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """``convert_lowmem.py`` imported against a recording fake ``mlx.core`` and a fake ``mlx.nn``.

    :param monkeypatch: Restores ``sys.modules`` afterwards.
    :returns: Namespace with ``module`` (the converter), ``mx`` and ``nn`` (the fakes).
    """
    fake_mx = RecordingMx()
    fake_nn = types.ModuleType('mlx.nn')

    class Linear:
        """Fake ``nn.Linear``."""

        def __init__(self, weight: FakeArray, bias: Optional[FakeArray] = None) -> None:
            """Store the parameters.

            :param weight: Weight.
            :param bias: Optional bias.
            """
            self.weight = weight
            if bias is not None:
                self.bias = bias

        to_quantized = stock_like_to_quantized

    class Embedding(Linear):
        """Fake ``nn.Embedding`` (own ``to_quantized``, like the real class)."""

        to_quantized = stock_like_to_quantized

    def quantize(model: 'FakeModel', group_size: Optional[int] = None, bits: Optional[int] = None, *,
                 mode: str = 'affine', class_predicate: Any = None) -> None:
        """Fake ``nn.quantize``: replace every leaf that has ``to_quantized`` by its quantised version.

        :param model: The fake model.
        :param group_size: Group size.
        :param bits: Bits.
        :param mode: Quantisation mode.
        :param class_predicate: Ignored.
        :returns: None.
        """
        for name, module in list(model.leaves.items()):
            if hasattr(module, 'to_quantized'):
                model.leaves[name] = module.to_quantized(group_size=group_size, bits=bits, mode=mode)

    fake_nn.Linear, fake_nn.Embedding, fake_nn.quantize = Linear, Embedding, quantize
    package = types.ModuleType('mlx')
    package.core, package.nn, package.__path__ = fake_mx, fake_nn, []
    for name, module in (('mlx', package), ('mlx.core', fake_mx), ('mlx.nn', fake_nn)):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location('kit_convert_lowmem_under_test', KIT / 'convert_lowmem.py')
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    return SimpleNamespace(module=converter, mx=fake_mx, nn=fake_nn)


@pytest.mark.parametrize('device', ['gpu', 'cpu'])
def test_eager_quantize_reads_first_then_kernel_on_stream_then_evaluates(lowmem: SimpleNamespace, device: str) -> None:
    """The input is evaluated before the kernel exists, the kernel runs on the configured stream, its outputs are
    evaluated at once, the cache is released, and the original's return value comes back unchanged."""
    scope = lowmem.module.EagerScope()
    wrapped = lowmem.module.make_eager_quantize(lowmem.mx.quantize, device, scope)
    w = FakeArray('w')
    out = wrapped(w, 64, 8, mode='affine')
    assert lowmem.mx.events == [('eval', ['w']), ('stream', device),
                                ('quantize', 'w', device, (64, 8), {'mode': 'affine'}), ('stream_exit', device),
                                ('eval', ['w.q', 'w.scales', 'w.biases']), ('clear_cache',)]
    assert out is lowmem.mx.last_out and lowmem.mx.current == 'default-stream'
    assert (scope.quantised, scope.left_lazy) == (1, 0)


def test_eager_quantize_leaves_the_constructor_init_lazy(lowmem: SimpleNamespace) -> None:
    """Inside a wrapped to_quantized only the module's own weight is quantised eagerly; any other input passes
    through lazily (no evaluation, default stream), as in the stock path."""
    scope = lowmem.module.EagerScope()
    wrapped = lowmem.module.make_eager_quantize(lowmem.mx.quantize, 'gpu', scope)
    scope.source = FakeArray('weight')
    init_out = wrapped(FakeArray('random-init'), 64, 8)
    assert lowmem.mx.events == [('quantize', 'random-init', 'default-stream', (64, 8), {})]
    assert init_out is lowmem.mx.last_out and (scope.quantised, scope.left_lazy) == (0, 1)
    lowmem.mx.events.clear()
    wrapped(scope.source, 64, 8)
    assert lowmem.mx.events[0] == ('eval', ['weight']) and lowmem.mx.events[2][:3] == ('quantize', 'weight', 'gpu')
    assert (scope.quantised, scope.left_lazy) == (1, 1)


def test_to_quantized_wrapper_releases_weight_only_after_original_and_keeps_bias(lowmem: SimpleNamespace) -> None:
    """The source weight is visible (and marked as the scope's source) while the original runs, replaced by an empty
    placeholder of the same dtype only after it returned; the bias is untouched; a second quantisation is refused."""
    scope = lowmem.module.EagerScope()
    seen: Dict[str, Any] = {}

    def original(self: Any, *args: Any, **kwargs: Any) -> SimpleNamespace:
        """Record what the wrapped call sees.

        :param self: The layer.
        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :returns: A stand-in quantised layer sharing the bias reference.
        """
        seen.update(weight=self.weight, source=scope.source, args=args, kwargs=kwargs)
        return SimpleNamespace(bias=self.bias)

    weight, bias = FakeArray('w', 'bfloat16'), FakeArray('b')
    layer = SimpleNamespace(weight=weight, bias=bias)
    wrapped = lowmem.module.make_releasing_to_quantized(original, scope)
    quantised = wrapped(layer, group_size=64, bits=8, mode='affine')
    assert seen == {'weight': weight, 'source': weight, 'args': (),
                    'kwargs': {'group_size': 64, 'bits': 8, 'mode': 'affine'}}
    assert layer.weight is not weight and layer.weight.size == 0 and layer.weight.dtype == 'bfloat16'
    assert layer.bias is bias and quantised.bias is bias
    assert scope.source is None and scope.released == 1 and lowmem.mx.events[-1] == ('clear_cache',)
    with pytest.raises(RuntimeError, match='quantize-order lazy'):
        wrapped(layer)

    def failing(self: Any, *args: Any, **kwargs: Any) -> None:
        """Fail like a broken quantisation.

        :param self: The layer.
        :param args: Ignored.
        :param kwargs: Ignored.
        :raises ValueError: Always.
        """
        raise ValueError('boom')

    untouched = SimpleNamespace(weight=FakeArray('x'))
    with pytest.raises(ValueError):
        lowmem.module.make_releasing_to_quantized(failing, scope)(untouched)
    assert untouched.weight.name == 'x' and scope.source is None and scope.released == 1


def test_eager_hooks_install_and_restore_and_lazy_installs_none(lowmem: SimpleNamespace) -> None:
    """Lazy mode leaves mlx untouched; eager mode wraps quantize and both to_quantized methods for the block only,
    and through them a stock-like layer is quantised with its init left lazy and its weight released."""
    nn = lowmem.nn
    before = (lowmem.mx.quantize, nn.quantize, nn.Linear.to_quantized, nn.Embedding.to_quantized)
    with lowmem.module.quantization_hooks('lazy', 'gpu') as scope:
        assert scope is None
        assert (lowmem.mx.quantize, nn.quantize, nn.Linear.to_quantized, nn.Embedding.to_quantized) == before
    with lowmem.module.quantization_hooks('eager', 'gpu') as scope:
        assert isinstance(scope, lowmem.module.EagerScope)
        assert lowmem.mx.quantize != before[0] and nn.quantize is not before[1]
        assert nn.Linear.to_quantized is not before[2] and nn.Embedding.to_quantized is not before[3]
        bias = FakeArray('b')
        layer = nn.Linear(FakeArray('w'), bias)
        quantised = layer.to_quantized(group_size=64, bits=8, mode='affine')
        embedding = nn.Embedding(FakeArray('e'))
        embedding.to_quantized(group_size=64, bits=8, mode='affine')
    assert (lowmem.mx.quantize, nn.quantize, nn.Linear.to_quantized, nn.Embedding.to_quantized) == before
    kernels = [event[1:3] for event in lowmem.mx.events if event[0] == 'quantize']
    assert kernels == [('random-init', 'default-stream'), ('w', 'gpu'), ('random-init', 'default-stream'), ('e', 'gpu')]
    evaluated = [name for event in lowmem.mx.events if event[0] == 'eval' for name in event[1]]
    assert 'random-init' not in evaluated and evaluated[0] == 'w'
    assert layer.weight.size == 0 and embedding.weight.size == 0 and quantised.bias is bias and layer.bias is bias
    assert (scope.quantised, scope.released, scope.left_lazy) == (2, 2, 2)


def test_configure_devices(lowmem: SimpleNamespace) -> None:
    """Eager mode puts everything but the kernel on the CPU; lazy mode puts everything on the chosen device."""
    configure = lowmem.module.configure_devices
    assert configure('eager', 'gpu') == 'gpu' and lowmem.mx.default == 'cpu'
    assert configure('lazy', 'gpu') == 'gpu' and lowmem.mx.default == 'gpu'
    assert configure('eager', 'cpu') == 'cpu' and lowmem.mx.default == 'cpu'
    assert configure('lazy', 'cpu') == 'cpu' and lowmem.mx.default == 'cpu'


def test_cli_quantize_order_flag(lowmem: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    """--quantize-order defaults to eager, accepts lazy and rejects anything else; the other defaults are unchanged."""
    parser = lowmem.module.build_parser()
    required = ['--hf-path', 'src', '--mlx-path', 'dst']
    args = parser.parse_args(required)
    assert (args.quantize_order, args.device, args.q_bits, args.q_group_size) == ('eager', 'gpu', 8, 64)
    assert parser.parse_args(required + ['--quantize-order', 'lazy']).quantize_order == 'lazy'
    with pytest.raises(SystemExit):
        parser.parse_args(required + ['--quantize-order', 'sometimes'])
    capsys.readouterr()


def test_selftest_parses_the_eager_line_and_compares_weights(lowmem: SimpleNamespace, tmp_path: Path) -> None:
    """The self-test's parser reads the converter's own summary line; weight comparison is byte-exact."""
    from src.finetuning.text27b.mlx_convert_kit import selftest
    scope = lowmem.module.EagerScope()
    scope.quantised, scope.released, scope.left_lazy = 16, 16, 16
    assert selftest.eager_counts('x\n' + scope.summary('cpu') + '\n') == {'quantised': 16, 'released': 16,
                                                                          'init_left_lazy': 16}
    assert selftest.eager_counts('[lowmem] done') is None
    first, second = tmp_path / 'a', tmp_path / 'b'
    first.mkdir()
    second.mkdir()
    assert not selftest.same_weights(first, second)  # nothing to compare is not a match
    (first / 'model.safetensors').write_bytes(b'abc')
    (second / 'model.safetensors').write_bytes(b'abc')
    assert selftest.same_weights(first, second)
    (second / 'model.safetensors').write_bytes(b'abd')
    assert not selftest.same_weights(first, second)


@pytest.mark.skipif(shutil.which('zsh') is None, reason='the runner is a zsh script')
def test_runner_rejects_unknown_quantize_order(tmp_path: Path) -> None:
    """An unknown --quantize-order is refused before the lock or any staging directory is created."""
    source = make_source_dir(tmp_path)
    result = run_runner('--hf-path', str(source), '--mlx-path', str(tmp_path / 'out'), '--quantize-order', 'sometimes')
    assert result.returncode == 1 and 'must be eager or lazy' in result.stderr
    assert not (tmp_path / 'out.lock').exists() and not (tmp_path / 'out.kit-partial').exists()


def test_covering_quantize_wraps_other_layer_types_for_the_call_only(lowmem: SimpleNamespace) -> None:
    """Inside eager mode, nn.quantize also wraps layer classes with their own to_quantized (like mlx_lm's
    SwitchLinear) for the duration of the call: their weights are quantised eagerly and released, their constructor
    init stays lazy, nothing is wrapped twice, and the classes are restored afterwards."""
    nn = lowmem.nn

    class SwitchLike:
        """A layer with its own ``to_quantized``, like ``mlx_lm``'s ``SwitchLinear``."""

        def __init__(self, weight: FakeArray, bias: Optional[FakeArray] = None) -> None:
            """Store the parameters.

            :param weight: Weight.
            :param bias: Optional bias.
            """
            self.weight = weight
            if bias is not None:
                self.bias = bias

        to_quantized = stock_like_to_quantized

    class SwitchChild(SwitchLike):
        """Inherits ``SwitchLike.to_quantized``."""

    class LinearChild(nn.Linear):
        """Inherits ``nn.Linear.to_quantized`` (wrapped already)."""

    model = FakeModel(child=SwitchChild(FakeArray('c')), switch=SwitchLike(FakeArray('s'), FakeArray('sb')),
                      linear=LinearChild(FakeArray('l')), embed=nn.Embedding(FakeArray('e')))
    sources = dict(model.leaves)
    original_quantize = nn.quantize
    with lowmem.module.quantization_hooks('eager', 'gpu') as scope:
        nn.quantize(model, 64, 8, mode='affine')
        restored_during_block = (SwitchLike.__dict__['to_quantized'], 'to_quantized' in SwitchChild.__dict__)
    assert restored_during_block == (stock_like_to_quantized, False)
    assert nn.quantize is original_quantize and scope.covered == ['SwitchChild', 'SwitchLike']
    assert all(layer.weight.size == 0 for layer in sources.values())
    assert model.leaves['switch'].bias is sources['switch'].bias
    assert (scope.quantised, scope.released, scope.left_lazy) == (4, 4, 4) and scope.warning() is None
    kernels = [event[1:3] for event in lowmem.mx.events if event[0] == 'quantize' and event[1] != 'random-init']
    assert kernels == [('c', 'gpu'), ('s', 'gpu'), ('l', 'gpu'), ('e', 'gpu')]
    evaluated = [name for event in lowmem.mx.events if event[0] == 'eval' for name in event[1]]
    assert 'random-init' not in evaluated
    assert 'other layer types: SwitchChild, SwitchLike' in scope.summary('gpu')


def test_covering_quantize_restores_classes_when_a_layer_fails(lowmem: SimpleNamespace) -> None:
    """A layer whose to_quantized rejects the stock arguments (like afm7's FusedLinear, which takes no mode) fails as
    in the stock path, and its class, the scope and the weight are left as they were."""

    class NoMode:
        """A layer whose ``to_quantized`` takes no ``mode``."""

        def __init__(self) -> None:
            """Give it a weight."""
            self.weight = FakeArray('x')

        def to_quantized(self, group_size: Optional[int] = None, bits: Optional[int] = None) -> None:
            """Never reached with the stock keyword arguments.

            :param group_size: Group size.
            :param bits: Bits.
            :returns: None.
            """

    own = NoMode.__dict__['to_quantized']
    model = FakeModel(layer=NoMode())
    with lowmem.module.quantization_hooks('eager', 'gpu') as scope:
        with pytest.raises(TypeError):
            lowmem.nn.quantize(model, 64, 8, mode='affine')
    assert NoMode.__dict__['to_quantized'] is own and scope.source is None
    assert model.leaves['layer'].weight.name == 'x' and scope.released == 0


def test_scope_warning_when_counters_differ(lowmem: SimpleNamespace) -> None:
    """Balanced counters give no warning; unbalanced ones give a warning line; the summary stays parseable."""
    from src.finetuning.text27b.mlx_convert_kit import selftest
    scope = lowmem.module.EagerScope()
    scope.quantised = scope.released = 3
    assert scope.warning() is None
    scope.quantised = 5
    assert scope.warning().startswith('[lowmem] warning: 5 weights quantised eagerly but 3 source weights released')
    scope.covered = ['SwitchLinear']
    assert selftest.eager_counts(scope.summary('gpu'))['quantised'] == 5
    assert not selftest.eager_hooks_ran(selftest.eager_counts(scope.summary('gpu')))
    assert selftest.eager_hooks_ran({'quantised': 3, 'released': 3, 'init_left_lazy': 3})
    assert not selftest.eager_hooks_ran(None)
    assert not selftest.eager_hooks_ran({'quantised': 0, 'released': 0, 'init_left_lazy': 0})


def test_memory_floor_packed_bytes() -> None:
    """Quantisable tensors count bits plus a scale and a bias per group in the source dtype; others their own size."""
    assert memory_floor.packed_bytes({'shape': [256, 128], 'data_offsets': [0, 65536]}, 8, 64) == 32768 + 2048
    assert memory_floor.packed_bytes({'shape': [128, 128], 'data_offsets': [0, 65536]}, 8, 64) == 16384 + 2048
    assert memory_floor.packed_bytes({'shape': [64, 100], 'data_offsets': [0, 12800]}, 8, 64) == 12800
    assert memory_floor.packed_bytes({'shape': [128], 'data_offsets': [0, 256]}, 8, 64) == 256


def test_memory_floor_estimate_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The floor is packed model + largest tensor + margin, from the shards the index names; the CLI prints it or the
    whole estimate, and exits 1 when a shard is missing."""
    source = tmp_path / 'src'
    source.mkdir()
    write_header_only(source / 'model-00001-of-00002.safetensors',
                      {'embed.weight': ('BF16', [262144, 4096]), 'norm.weight': ('BF16', [4096])})
    write_header_only(source / 'model-00002-of-00002.safetensors', {'lm_head.weight': ('BF16', [262144, 4096])})
    weight_map = {'embed.weight': 'model-00001-of-00002.safetensors', 'norm.weight': 'model-00001-of-00002.safetensors',
                  'lm_head.weight': 'model-00002-of-00002.safetensors'}
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
    result = memory_floor.estimate(source, 8, 64, margin_gib=4.0)
    assert result['source_gib'] == 4.0 and result['largest_tensor'] == 'embed.weight'
    assert result['largest_tensor_gib'] == 2.0 and result['packed_gib'] == pytest.approx(2.125, abs=0.01)
    assert result['floor_gib'] == 8.2  # 2.125 GiB packed + 8 KiB norm + 2 GiB + 4 GiB, rounded up to 0.1
    assert memory_floor.main([str(source)]) == 0 and capsys.readouterr().out.strip() == '8.2'
    assert memory_floor.main([str(source), '--json', '--margin-gib', '0']) == 0
    assert json.loads(capsys.readouterr().out)['floor_gib'] == 4.2
    (source / 'model-00002-of-00002.safetensors').unlink()
    assert memory_floor.main([str(source)]) == 1 and 'missing shards' in capsys.readouterr().err
    (source / 'model.safetensors.index.json').unlink()  # without an index every shard file counts
    assert memory_floor.estimate(source, 8, 64, margin_gib=0.0)['largest_tensor'] == 'embed.weight'


FAKE_INTERPRETER = """#!/bin/zsh
# Test stand-in for the kit's interpreter: passes the mlx and pin checks, never converts, runs anything else for real.
case "$*" in
  *convert_lowmem.py*) print -u2 -- 'fake interpreter: no conversion'; exit 3 ;;
  *--check-pins*) exit 0 ;;
  '-c import mlx.core, mlx_lm') exit 0 ;;
esac
exec REAL_PYTHON "$@"
"""


@pytest.mark.skipif(sys.platform != 'darwin' or shutil.which('zsh') is None,
                    reason='the runner is a zsh script and its memory check reads vm_stat (macOS)')
def test_runner_memory_floor_depends_on_the_order(tmp_path: Path) -> None:
    """Without --min-free-gib the eager order's floor is estimated from the source and the lazy order's is 12 GiB; an
    explicit --min-free-gib wins. A fake interpreter stands in for the kit's, so nothing is ever converted."""
    source = make_source_dir(tmp_path)
    write_header_only(source / 'model.safetensors', {'huge.weight': ('BF16', [2**21, 2**20])})  # 4 TiB of bf16
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'huge.weight': 'model.safetensors'}}))
    fake = tmp_path / 'fake-python'
    fake.write_text(FAKE_INTERPRETER.replace('REAL_PYTHON', sys.executable))
    fake.chmod(0o755)
    env = {**os.environ, 'PYTHON': str(fake)}

    def run(*extra: str) -> subprocess.CompletedProcess:
        """Run the runner against the fake source (``--force``: the fake never touches the GPU).

        :param extra: Extra arguments.
        :returns: The completed process.
        """
        return subprocess.run(['zsh', str(KIT / 'run_convert.sh'), '--hf-path', str(source), '--mlx-path',
                               str(tmp_path / 'out'), '--force', *extra], capture_output=True, text=True, env=env)

    floor = memory_floor.estimate(source)['floor_gib']
    eager = run()
    assert eager.returncode == 1 and f'need {floor} GiB (eager order estimate' in eager.stderr
    explicit = run('--min-free-gib', '100000')
    assert explicit.returncode == 1 and 'need 100000 GiB (--min-free-gib)' in explicit.stderr
    lazy = run('--quantize-order', 'lazy')  # either refused at 12 GiB or stopped by the fake conversion
    assert lazy.returncode == 1 and 'lazy order default' in lazy.stdout + lazy.stderr
    assert not (tmp_path / 'out.lock').exists() and not (tmp_path / 'out').exists()
    assert not (tmp_path / 'out.kit-partial').exists()
