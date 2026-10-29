"""CPU-only checks at the uint16 tile boundary, using the actual host policy."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

KERNEL = Path(__file__).resolve().parents[3] / 'python/fastvideo_kernel/triton_kernels/attn_qat_train.py'


@pytest.mark.parametrize('tiles,enabled', [(65534, True), (65535, True), (65536, False), (65537, False)])
@pytest.mark.parametrize('force_short', [False, True])
def test_uint16_boundary(tiles, enabled, force_short):
    tree = ast.parse(KERNEL.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sm121_use_change_points')
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(SM121_BLOCK=32, triton=SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b))
    exec(compile(module, str(KERNEL), 'exec'), namespace)
    dispatch = namespace['_sm121_use_change_points']
    assert dispatch(True, 'packed_slots', 8192, tiles*32, force_short) is enabled
    assert dispatch(True, 'packed_slots', 8192, (tiles-1)*32+1, force_short) is enabled


def load_measure():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'code/spark_measure.py'
    spec = importlib.util.spec_from_file_location('spark_measure_cpu_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failure', ['existing', 'same-lock', 'zero-budget'])
def test_measure_rejects_unsafe_attempt(tmp_path, monkeypatch, failure):
    module = load_measure()
    output = tmp_path / 'output'
    if failure == 'existing':
        output.mkdir()
    argv = ['measure', '--output', str(output), '--lock', str(tmp_path / 'a'), '--lock',
            str(tmp_path / ('a' if failure == 'same-lock' else 'b')), '--policy', 'dense', '--tokens', '8192']
    if failure == 'zero-budget':
        argv += ['--measure', '0']
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(SystemExit):
        module.parse_args()


def test_streaming_source_hash(tmp_path):
    import hashlib
    p = tmp_path / 'blob'
    data = b'known-source' * 200000
    p.write_bytes(data)
    assert load_measure().digest(p) == hashlib.sha256(data).hexdigest()
