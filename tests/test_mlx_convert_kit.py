"""Unit tests for the MLX conversion kit's stdlib parts and the runner's refusal paths (no MLX needed)."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from src.finetuning.text27b.mlx_convert_kit import gpu_quant_identity, kit_record, verify_mlx_dir

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
