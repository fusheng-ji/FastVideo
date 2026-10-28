"""CPU-safe contracts and streaming fingerprints for isolated native FA2 replay."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

INPUTS = ('dout', 'q', 'k', 'v', 'out', 'lse')
GRADS = ('dq', 'dk', 'dv')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def span(shape: list[int], stride: list[int], itemsize: int) -> int:
    if len(shape) != len(stride) or type(itemsize) is not int or itemsize <= 0:
        raise ValueError('Invalid layout rank/itemsize')
    if any(type(x) is not int or x < 0 for x in [*shape, *stride]):
        raise ValueError('Invalid layout extent/stride')
    if 0 in shape:
        return 0
    extent = 1
    for step, size in sorted((s, n) for n, s in zip(shape, stride, strict=True) if n > 1):
        if step < extent:
            raise ValueError('Overlapping or unsupported layout')
        extent += (size - 1) * step
    size = (1 + sum((n - 1) * s for n, s in zip(shape, stride, strict=True))) * itemsize
    if size > 2**63 - 1:
        raise ValueError('Layout exceeds address range')
    return size


def tensor_hash(tensor: Any) -> str:
    import torch
    value = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


def memory(label: str, torch: Any = None) -> dict[str, Any]:
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    stat = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    row = dict(label=label,
               timestamp=time.time(),
               rss_bytes=int(stat['VmRSS'].split()[0]) * 1024,
               rss_high_water_bytes=int(stat['VmHWM'].split()[0]) * 1024,
               host_available_bytes=int(info['MemAvailable'].split()[0]) * 1024)
    if torch is not None:
        free, total = torch.cuda.mem_get_info()
        row.update(cuda_free=free,
                   cuda_total=total,
                   allocated=torch.cuda.memory_allocated(),
                   reserved=torch.cuda.memory_reserved(),
                   peak_allocated=torch.cuda.max_memory_allocated(),
                   peak_reserved=torch.cuda.max_memory_reserved())
    return row
