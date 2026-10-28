"""Content-addressed CPU evidence and bounded-memory, byte-sensitive differences."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import shutil
from typing import Any


class TensorStore:
    """Write one CPU tensor at a time; never retain a full gradient snapshot."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def write(self, tensor: Any, expected: str) -> dict[str, Any]:
        import torch
        if tensor.device.type != 'cpu':
            raise ValueError('Evidence writer requires a CPU tensor')
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        actual = hashlib.sha256(memoryview(raw)).hexdigest()
        if actual != expected:
            raise ValueError('Tensor hash does not match the observed bytes')
        path = self.root / (expected + '.bin')
        if not path.exists():
            if shutil.disk_usage(self.root).free < 100 * 1024**3:
                raise RuntimeError('Evidence disk has less than 100 GiB available; stop without deleting evidence')
            with path.open('xb') as stream:
                stream.write(memoryview(raw))
        if path.stat().st_size != raw.nbytes:
            raise RuntimeError('Truncated evidence file')
        return dict(path=str(path.resolve()), sha256=actual, bytes=raw.nbytes)


def load(record: dict[str, Any]) -> Any:
    import torch
    item = record['evidence']
    path = Path(item['path'])
    if path.stat().st_size != item['bytes']:
        raise RuntimeError('Evidence file size mismatch')
    dtype = getattr(torch, record['dtype'].removeprefix('torch.'))
    raw = torch.from_file(str(path), shared=False, size=item['bytes'], dtype=torch.uint8)
    return raw.view(dtype).reshape(record['shape'])


def metrics(a: Any, b: Any, chunk: int = 262144) -> dict[str, Any]:
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('Tensor dtype or shape mismatch')
    if a.device.type != 'cpu' or b.device.type != 'cpu':
        raise ValueError('Difference metrics must use CPU evidence')
    left, right = a.reshape(-1), b.reshape(-1)
    changed, squared, norm, maximum, finite = 0, 0., 0., 0., True
    for start in range(0, a.numel(), chunk):
        x, y = left[start:start + chunk], right[start:start + chunk]
        rx = x.contiguous().view(torch.uint8).reshape(-1, a.element_size())
        ry = y.contiguous().view(torch.uint8).reshape(-1, b.element_size())
        changed += int((rx != ry).any(dim=1).sum())
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            finite = False
            continue
        xd, yd = x.double(), y.double()
        diff = xd - yd
        squared += float(diff.square().sum(dtype=torch.float64))
        norm += float(yd.square().sum(dtype=torch.float64))
        maximum = max(maximum, float(diff.abs().max()))
    relative = math.sqrt(squared / norm) if norm else 0. if squared == 0 else None
    return dict(byte_equal=changed == 0,
                finite=finite,
                changed_elements=changed,
                elements=a.numel(),
                max_absolute=maximum if finite else None,
                absolute_l2=math.sqrt(squared) if finite else None,
                reference_l2=math.sqrt(norm) if finite else None,
                relative_l2=relative if finite else None,
                relative_l2_status='finite' if finite and
                (norm or not squared) else 'zero_reference_nonzero_difference' if finite else 'nonfinite')


def record_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if a['shape'] != b['shape'] or a['dtype'] != b['dtype']:
        raise ValueError('Record shape/dtype mismatch')
    if a['finite'] is not True or b['finite'] is not True:
        raise ValueError('Nonfinite report cannot be accepted as repeatability noise')
    if a['sha256'] == b['sha256']:
        return dict(byte_equal=True,
                    finite=True,
                    changed_elements=0,
                    elements=math.prod(a['shape']),
                    max_absolute=0.,
                    relative_l2=0.,
                    relative_l2_status='finite')
    if 'evidence' not in a or 'evidence' not in b:
        return dict(byte_equal=False, finite=True, metric_available=False, reason='hash-only boundary; needs capture')
    return metrics(load(a), load(b))
