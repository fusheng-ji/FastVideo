"""CPU-safe state and coverage helpers for isolated GB10 probes.

Importing this module never imports torch or initializes CUDA. Tensor/runtime
operations receive their implementation explicitly, which also permits CPU tests.
"""
from __future__ import annotations
from typing import Any
import copy
import hashlib
import json
import os
import pickle
import random
from collections import Counter
from pathlib import Path


def digest(value: Any) -> Any:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def fingerprint(value: Any) -> Any:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path: Any, value: Any) -> Any:
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def clone_tree(value: Any, is_tensor: Any, clone_tensor: Any) -> Any:
    if is_tensor(value):
        return clone_tensor(value)
    if isinstance(value, dict):
        return {key: clone_tree(item, is_tensor, clone_tensor) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item, is_tensor, clone_tensor) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item, is_tensor, clone_tensor) for item in value)
    return copy.deepcopy(value)


def capture_rng(torch: Any, numpy: Any, method_generator: Any) -> Any:
    return {
        'python': copy.deepcopy(random.getstate()),
        'numpy': copy.deepcopy(numpy.random.get_state()),
        'torch_cpu': torch.get_rng_state().clone(),
        'cuda_all': [state.clone() for state in torch.cuda.get_rng_state_all()],
        'method': method_generator.get_state().clone()
    }


def restore_rng(state: Any, torch: Any, numpy: Any, method_generator: Any) -> Any:
    random.setstate(copy.deepcopy(state['python']))
    numpy.random.set_state(copy.deepcopy(state['numpy']))
    torch.set_rng_state(state['torch_cpu'].clone())
    torch.cuda.set_rng_state_all([item.clone() for item in state['cuda_all']])
    method_generator.set_state(state['method'].clone())


def rng_identity(state: Any, tensor_digest: Any) -> Any:
    return {
        'python': digest(state['python']),
        'numpy': digest(state['numpy']),
        'torch_cpu': tensor_digest(state['torch_cpu']),
        'cuda_all': [tensor_digest(item) for item in state['cuda_all']],
        'method': tensor_digest(state['method'])
    }


class EventLedger:
    """Record actual callback order and allocate phase-local call identities."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.calls: Counter[Any] = Counter()

    def next_call(self, phase: Any, module: Any) -> Any:
        key = (phase, module)
        index = self.calls[key]
        self.calls[key] += 1
        return f'{module}/{phase}/{index}'

    def add(self, phase: Any, kind: Any, module: Any, call_id: Any, tensors: Any = None, **extra: Any) -> Any:
        event = {
            'order': len(self.events),
            'phase': phase,
            'kind': kind,
            'module': module,
            'call_id': call_id,
            'tensors': tensors or {},
            **extra
        }
        self.events.append(event)
        return event


def event_key(event: Any) -> Any:
    return [event[key] for key in ('phase', 'kind', 'module', 'call_id')]


def coverage_errors(events: Any, expected_keys: Any, expected_parameters: Any, gradients: Any) -> Any:
    """Absence on both sides is never an equality success."""
    observed = Counter(tuple(event_key(event)) for event in events)
    errors = []
    for key in expected_keys:
        count = observed[tuple(key)]
        if count != 1:
            errors.append(f'Expected exactly one event {key!r}, observed {count}')
    if not expected_parameters:
        errors.append('No expected trainable parameter gradients')
    for name in expected_parameters:
        record = gradients.get(name)
        if not record or record.get('missing') or (not record.get('sha256')):
            errors.append(f'Missing trainable parameter gradient: {name}')
        elif record.get('finite') is not True:
            errors.append(f'Nonfinite trainable parameter gradient: {name}')
    for event in events:
        for name, record in event.get('tensors', {}).items():
            if record.get('missing') or record.get('finite') is not True:
                errors.append(f'Missing/nonfinite event tensor {event_key(event)!r}/{name}')
    return errors
