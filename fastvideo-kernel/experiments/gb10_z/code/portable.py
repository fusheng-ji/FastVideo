"""Explicit paths and locked execution for private, relocatable GB10 diagnostics."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any

_ACTIVE: Any = None
CODE = Path(__file__).resolve().parent
REPO = CODE.parents[3]
SCRIPTS = ('control_train.py', 'probe_model.py', 'fa2_replay.py', 'fa2_extract.py')


def settings() -> Any:
    if _ACTIVE is None:
        raise RuntimeError('Run model diagnostics through portable.py with explicit inputs')
    return _ACTIVE


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_sources(manifest: Path) -> dict[str, Any]:
    doc = json.loads(manifest.read_text())
    if doc.get('acceptance_scope') != 'gb10-z-portable-calibration-v1' or not doc.get('sources'):
        raise ValueError('Incorrect portable source manifest')
    for path, expected in {**doc['sources'], **doc['inputs']}.items():
        if not Path(path).is_absolute() or digest(Path(path)) != expected:
            raise ValueError('Source or input fingerprint mismatch: ' + path)
    return doc


def parse_args(argv: Any = None) -> Any:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'check', 'run'))
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--validation-manifest', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--train-data', type=Path)
    p.add_argument('--validation-data', type=Path)
    p.add_argument('--lock', type=Path, action='append', default=[])
    p.add_argument('--script', choices=SCRIPTS)
    # Split downstream arguments at the explicit separator before parsing.
    raw = list(sys.argv[1:] if argv is None else argv)
    cut = raw.index('--') if '--' in raw else len(raw)
    a = p.parse_args(raw[:cut])
    a.arguments = raw[cut + 1:]
    if a.action == 'prepare':
        for name in ('model', 'config', 'train_data', 'validation_data', 'validation_manifest'):
            if getattr(a, name) is None:
                p.error('prepare requires --' + name.replace('_', '-'))
        if not a.model.is_dir():
            p.error('--model must be a local model directory')
        for path in (a.config, a.train_data, a.validation_data):
            if not path.is_file():
                p.error('Missing input file: ' + str(path))
    if a.action == 'run' and (not a.script or len(set(a.lock)) != 2):
        p.error('run requires --script and two distinct --lock paths')
    return a


def prepare(a: Any) -> None:
    if a.source_manifest.exists() or a.validation_manifest.exists():
        raise FileExistsError('Refuse to overwrite an existing input/source manifest')
    import pyarrow.parquet as pq
    ids = pq.read_table(a.validation_data, columns=['id'])['id'].to_pylist()
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('Expected exactly 32 unique validation IDs')
    sources = {}
    for directory in (REPO / 'fastvideo', REPO / 'fastvideo-kernel/python', CODE):
        for path in directory.rglob('*.py'):
            sources[str(path.resolve())] = digest(path)
    spec = importlib.util.find_spec('fastvideo_kernel')
    if spec is None or not spec.submodule_search_locations:
        raise ImportError('Install fastvideo-kernel before preparing a runtime manifest')
    for root in spec.submodule_search_locations:
        for path in Path(root).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.so'):
                sources[str(path.resolve())] = digest(path)
    a.validation_manifest.parent.mkdir(parents=True, exist_ok=True)
    a.validation_manifest.write_text(json.dumps(dict(rows=[dict(id=value, noise_seed=3100 + i, collate_seed=1000)
                                                          for i, value in enumerate(ids)]), indent=2) + '\n')
    inputs = {str(path.resolve()): digest(path) for path in (a.config, a.train_data, a.validation_data,
                                                           a.validation_manifest)}
    for path in a.model.rglob('*'):
        if path.is_file():
            inputs[str(path.resolve())] = digest(path)
    paths = {name: str(getattr(a, name).resolve()) for name in ('model', 'config', 'train_data', 'validation_data',
                                                              'validation_manifest')}
    a.source_manifest.parent.mkdir(parents=True, exist_ok=True)
    a.source_manifest.write_text(json.dumps(dict(acceptance_scope='gb10-z-portable-calibration-v1', sources=sources,
                                                 inputs=inputs, paths=paths), indent=2) + '\n')


def configure() -> None:
    sys.path.insert(0, str(REPO))
    kernels = importlib.import_module('fastvideo_kernel.triton_kernels')
    kernels.__path__ = [str(REPO / 'fastvideo-kernel/python/fastvideo_kernel/triton_kernels'), *kernels.__path__]
    name = 'fastvideo_kernel.triton_kernels.attn_qat_train'
    path = Path(kernels.__path__[0]) / 'attn_qat_train.py'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    kernels.attn_qat_train = module


def main() -> None:
    global _ACTIVE
    a = parse_args()
    if a.action == 'prepare':
        prepare(a)
        return
    doc = verify_sources(a.source_manifest)
    if a.action == 'check':
        print('Source and input fingerprints match; no GPU execution performed')
        return
    for name, path in doc['paths'].items():
        setattr(a, name, Path(path))
    _ACTIVE = a
    with ExitStack() as stack:
        for path in a.lock:
            handle = stack.enter_context(path.open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX)
        verify_sources(a.source_manifest)
        if a.script not in ('fa2_replay.py', 'fa2_extract.py'):
            configure()
        arguments = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
        sys.argv = [str(CODE / a.script), *arguments]
        runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    sys.modules['portable'] = sys.modules[__name__]
    main()
