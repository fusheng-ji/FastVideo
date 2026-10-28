"""Record-only FA2 observer for ordinary PyTorch mode diagnostics."""
from typing import Any
from local_policy import SCOPE as SCOPE
import contextlib
import importlib
import hashlib
from pathlib import Path
import sys


def file_record(path: Any) -> Any:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_calls(records: Any, expected: Any = 30) -> Any:
    if len(records) != expected or any(
            r.get("phase") != "backward" or r.get("supplied_deterministic") is not False
            or r.get("effective_deterministic") is not False or r.get("observer_mode") != "record-only"
            or r.get("torch_deterministic") is not False or r.get("torch_warn_only") is not False for r in records):
        raise RuntimeError(f"Nonstrict FA2 policy/coverage failed: expected {expected}, observed {len(records)}")
    return {"expected": expected, "observed": len(records), "all_deterministic": False, "all_nondeterministic": True}


@contextlib.contextmanager
def flash_observer(records: Any, phase: Any, capture: Any = None) -> Any:
    """Observe unchanged kwargs; a policy violation aborts before kernel launch."""
    patches = []
    modules = [
        importlib.import_module('fastvideo.attention.utils.' + name)
        for name in ('flash_attn_default', 'flash_attn_no_pad')
    ]
    try:
        for module in modules:
            for name in ('_fa2_backward', '_fa2_varlen_backward'):
                original = getattr(module, name, None)
                if original is None:
                    continue

                def wrapped(*args: Any,
                            _original: Any = original,
                            _module: Any = module.__name__,
                            _name: Any = name,
                            **kwargs: Any) -> Any:
                    torch = sys.modules.get('torch')
                    record = {
                        'phase': phase(),
                        'module': _module,
                        'function': _name,
                        'supplied_deterministic': kwargs.get('deterministic'),
                        'effective_deterministic': kwargs.get('deterministic'),
                        'observer_mode': 'record-only',
                        'torch_deterministic': torch.are_deterministic_algorithms_enabled() if torch else None,
                        'torch_warn_only': torch.is_deterministic_algorithms_warn_only_enabled() if torch else None,
                        'operator_module': getattr(_original, '__module__', None),
                        'shapes': [list(value.shape) for value in args[:4] if hasattr(value, 'shape')]
                    }
                    records.append(record)
                    validate_calls([record], 1)
                    if capture is not None:
                        return capture(_name, args, kwargs, _original)
                    return _original(*args, **kwargs)

                setattr(module, name, wrapped)
                patches.append((module, name, original))
        if not patches:
            raise RuntimeError('Production acceptance has no FA2 backward binding')
        yield [{
            'module': module.__name__,
            'version': getattr(module, 'fa_version', None),
            **file_record(module.__file__)
        } for module in modules]
    finally:
        for module, name, original in reversed(patches):
            setattr(module, name, original)
