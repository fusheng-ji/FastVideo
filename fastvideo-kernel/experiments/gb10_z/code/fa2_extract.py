"""Extract only native FA2 arguments in a CPU process that exits before GPU work."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
from common import INPUTS, GRADS, atomic, digest, memory, span, tensor_hash


def extract(source: Path, output: Path) -> dict[str, Any]:
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Extraction must not initialize CUDA')
    output.mkdir(parents=True, exist_ok=False)
    samples = [memory('before-load')]
    capture = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    if capture.get('schema_version') != 1 or len(capture.get('flash', [])) != 1:
        raise ValueError('Expected one captured native FA2 call')
    row = capture['flash'][0]
    if row['function'] != '_fa2_backward' or set(row['inputs']) != set(INPUTS) or set(row['gradients']) != set(GRADS):
        raise ValueError('Missing or unexpected native arguments')
    if row['kwargs'].get('deterministic') is not False or row['kwargs'].get('dropout_p') != 0.:
        raise ValueError('Expected original ordinary-mode, zero-dropout call')
    result = dict(schema_version=1,
                  source=str(source),
                  source_sha256=digest(source),
                  kwargs=row['kwargs'],
                  function=row['function'],
                  layer=row['layer'],
                  tensors={})
    for name in (*INPUTS, *GRADS):
        original = row['inputs' if name in INPUTS else 'gradients'][name]
        meta = row['metadata'][name] if name in INPUTS else dict(
            shape=list(original.shape), stride=list(original.stride()), dtype=str(original.dtype))
        if list(original.shape) != meta['shape'] or str(original.dtype) != meta['dtype']:
            raise ValueError('Capture metadata mismatch')
        required = span(meta['shape'], meta['stride'], original.element_size())
        value = original.detach().clone(memory_format=torch.contiguous_format)
        logical = value.numel() * value.element_size()
        if value.untyped_storage().nbytes() != logical or value.untyped_storage().data_ptr(
        ) == original.untyped_storage().data_ptr():
            raise RuntimeError('Extracted tensor is not compact independent storage')
        expected = tensor_hash(original)
        if tensor_hash(value) != expected:
            raise RuntimeError('Extraction changed tensor bytes')
        path = output / (name + '.pt')
        torch.save(value, path)
        restored = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if tensor_hash(restored) != expected or restored.untyped_storage().nbytes() != logical:
            raise RuntimeError('Serialized minimal tensor mismatch')
        result['tensors'][name] = dict(**meta,
                                       logical_bytes=logical,
                                       storage_bytes=logical,
                                       span_bytes=required,
                                       original_storage_bytes=original.untyped_storage().nbytes(),
                                       raw_sha256=expected,
                                       file=path.name,
                                       file_sha256=digest(path))
        del restored, value
        samples.append(memory('extracted-' + name))
    del row, capture
    result.update(execution_complete=True, accepted=True, memory=samples, cuda_initialized=torch.cuda.is_initialized())
    atomic(output / 'manifest.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    extract(a.source, a.output)
