"""Private evaluation isolation and bounded-memory checkpoint evidence."""
from __future__ import annotations
from functools import wraps
import json
from pathlib import Path
import time
from typing import Any
from common import atomic, digest, tensor_hash
from probe_helpers import capture_rng, restore_rng, rng_identity

STATE: dict[str, Any] = {}
OUTPUT: Path | None = None


def save_named(directory: Path, named: Any, torch: Any) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (name, tensor) in enumerate(named):
        if tensor is None:
            raise RuntimeError('Missing final tensor: ' + name)
        value = tensor.detach()
        if hasattr(value, 'to_local'):
            value = value.to_local()
        value = value.to('cpu', copy=True).contiguous()
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise RuntimeError('Nonfinite final tensor: ' + name)
        path = directory / f'{index:04}.pt'
        torch.save(value, path)
        rows.append(
            dict(name=name,
                 file=path.name,
                 sha256=digest(path),
                 raw_sha256=tensor_hash(value),
                 shape=list(value.shape),
                 dtype=str(value.dtype)))
        del value
    atomic(directory / 'manifest.json', rows)


def save_tree(directory: Path, tree: Any, torch: Any) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    count = 0

    def visit(value: Any) -> Any:
        nonlocal count
        if isinstance(value, torch.Tensor):
            name = f'tensor-{count:05}'
            count += 1
            save_named(directory / name, [(name, value)], torch)
            return dict(kind='tensor', directory=name)
        if isinstance(value, dict):
            return dict(kind='dict', items=[[visit(k), visit(v)] for k, v in value.items()])
        if isinstance(value, tuple | list):
            return dict(kind=type(value).__name__, items=[visit(v) for v in value])
        if value is None or isinstance(value, str | int | float | bool):
            return value
        raise TypeError('Unsupported checkpoint state: ' + str(type(value)))

    atomic(directory / 'structure.json', visit(tree))


def evaluation_guard(method: Any, transformer: Any) -> Any:

    def decorate(function: Any) -> Any:

        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            import torch
            import numpy as np
            if STATE.get('phase') != 'idle' or STATE.get('rep') is not None:
                raise RuntimeError('Evaluation entered inside a training microbatch')
            before = capture_rng(torch, np, method.cuda_generator)
            signature = rng_identity(before, tensor_hash)
            sampler = method.student._sample_timesteps
            modes = [(module, module.training) for module in transformer.modules()]
            old_phase = STATE['phase']
            STATE['phase'] = 'evaluation'
            start = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                method.student._sample_timesteps = sampler
                for module, training in modes:
                    module.training = training
                restore_rng(before, torch, np, method.cuda_generator)
                after = rng_identity(capture_rng(torch, np, method.cuda_generator), tensor_hash)
                STATE['phase'] = old_phase
                if after != signature:
                    raise RuntimeError('Evaluation changed training RNG state')
                if OUTPUT is not None:
                    with (OUTPUT / 'evaluation-isolation.jsonl').open('a') as stream:
                        stream.write(
                            json.dumps(
                                dict(step=args[0],
                                     rng_restored=True,
                                     rng_before=signature,
                                     rng_after=after,
                                     seconds=time.monotonic() - start)) + '\n')

        return wrapped

    return decorate


def validate_evaluations(directory: Path) -> None:
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    samples = [row for row in rows if row['event'] == 'eval-sample']
    if len(samples) != 96:
        raise RuntimeError('Expected exactly 96 validation forwards per run')
    declared = json.loads(__import__('portable').settings().validation_manifest.read_text())['rows']
    for step in (0, 10, 20):
        selected = [row for row in samples if row['step'] == step]
        if len(selected) != 32 or any(row['split'] != 'heldout' or row['draw'] != 'fixed' for row in selected):
            raise RuntimeError('Validation budget or phase mismatch')
        if [(row['id'], row['noise_seed']) for row in selected] != [(row['id'], row['noise_seed']) for row in declared]:
            raise RuntimeError('Validation input identity differs from frozen manifest')
        if step:
            initial = [row for row in samples if row['step'] == 0]
            if [row['timestep'] for row in selected] != [row['timestep'] for row in initial]:
                raise RuntimeError('Validation timestep changed across updates')
    controls = [json.loads(line) for line in (directory / 'evaluation-isolation.jsonl').read_text().splitlines()]
    if [row['step'] for row in controls] != [0, 10, 20] or not all(row['rng_restored'] for row in controls):
        raise RuntimeError('Missing evaluation RNG isolation evidence')
