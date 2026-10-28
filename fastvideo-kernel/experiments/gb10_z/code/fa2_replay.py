"""Independent native FA2 process; bounded CPU state and new output buffers per repetition."""
from __future__ import annotations
import argparse
import gc
import weakref
import json
from pathlib import Path
from typing import Any
from common import INPUTS, GRADS, atomic, digest, memory, span, tensor_hash


def run(root: Path, output: Path, report: dict[str, Any]) -> None:
    import torch
    from flash_attn.flash_attn_interface import _flash_attn_backward
    from tensor_evidence import metrics
    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.set_num_threads(4)
    manifest = json.loads((root / 'manifest.json').read_text())
    report['input_manifest_sha256'] = digest(root / 'manifest.json')
    report['configuration'] = manifest['kwargs']
    report['torch_settings'] = dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                                    warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    report['gpu_layouts'] = {}
    inputs = {}
    gradient_refs: list[Any] = []

    def sample(label: str) -> None:
        record = memory(label, torch)
        if label.startswith('after-release'):
            record['live_gradient_buffers'] = sum(ref() is not None for ref in gradient_refs)
            record['gradient_buffers_created'] = len(gradient_refs)
            if len(gradient_refs) != 3 or record['live_gradient_buffers'] != 0:
                raise RuntimeError('Gradient lifetime check failed')
            gradient_refs.clear()
        report['memory'].append(record)
        atomic(output / 'result.json', report)

    try:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        sample('before-input-load')
        for name in INPUTS:
            meta = manifest['tensors'][name]
            path = root / meta['file']
            if digest(path) != meta['file_sha256']:
                raise ValueError('Minimal capture changed')
            value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite captured input: ' + name)
            if tensor_hash(value) != meta['raw_sha256']:
                raise ValueError('Minimal tensor bytes changed')
            if span(meta['shape'], meta['stride'], value.element_size()) != meta['span_bytes']:
                raise ValueError('Allocation span mismatch')
            sample('before-allocate-' + name)
            restored = torch.empty_strided(meta['shape'], meta['stride'], dtype=value.dtype, device='cuda')
            restored.copy_(value)
            torch.cuda.synchronize()
            if tensor_hash(restored.detach().cpu()) != meta['raw_sha256']:
                raise RuntimeError('GPU layout restore changed logical values')
            report['gpu_layouts'][name] = dict(shape=list(restored.shape),
                                               stride=list(restored.stride()),
                                               dtype=str(restored.dtype),
                                               storage_bytes=restored.untyped_storage().nbytes())
            if list(restored.stride()) != meta['stride']:
                raise RuntimeError('GPU input stride mismatch')
            inputs[name] = restored
            del restored, value
            sample('after-allocate-' + name)
        for deterministic in (False, True):
            runs: list[dict[str, Any]] = []
            kwargs = dict(manifest['kwargs'], deterministic=deterministic)
            for index in range(10):
                sample(f'before-backward-{deterministic}-{index}')
                grads = [torch.empty_like(inputs[name]) for name in ('q', 'k', 'v')]
                gradient_refs.extend(weakref.ref(value) for value in grads)
                gradient: Any = None
                try:
                    _flash_attn_backward(*[inputs[name] for name in INPUTS], *grads, **kwargs)
                    torch.cuda.synchronize()
                    sample(f'after-backward-{deterministic}-{index}')
                    comparisons = {}
                    for name, gradient in zip(GRADS, grads, strict=True):
                        value = gradient.detach().cpu()
                        if not bool(torch.isfinite(value).all()):
                            raise RuntimeError('Nonfinite native gradient')
                        path = output / f'{deterministic}-{index}-{name}.pt'
                        torch.save(value, path)
                        first = torch.load(output / f'{deterministic}-0-{name}.pt',
                                           map_location='cpu',
                                           weights_only=True,
                                           mmap=True)
                        comparisons[name] = metrics(value, first)
                        comparisons[name]['artifact_sha256'] = digest(path)
                        del first, value
                    for name in INPUTS:
                        value = inputs[name].detach().cpu()
                        if tensor_hash(value) != manifest['tensors'][name]['raw_sha256']:
                            raise RuntimeError('Native FA2 modified input: ' + name)
                        del value
                    record = dict(index=index, inputs_unchanged=True, gradients=comparisons)
                    runs.append(record)
                    atomic(output / f'mode-{deterministic}.json', runs)
                    if deterministic and not all(v['byte_equal'] for v in comparisons.values()):
                        raise RuntimeError('Deterministic native FA2 control is not byte-repeatable')
                finally:
                    del grads
                    # Avoid locals(): its frame snapshot can keep the last gradient alive.
                    gradient = None
                    gc.collect()
                sample(f'after-release-{deterministic}-{index}')
            report['modes'].append(
                dict(deterministic=deterministic,
                     runs=runs,
                     repeatable=all(v['byte_equal'] for x in runs for v in x['gradients'].values())))
        report.update(execution_complete=True, accepted=True, evidence_valid=True)
    finally:
        inputs.clear()
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            sample('cleanup')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(execution_complete=False,
                  accepted=False,
                  evidence_valid=False,
                  memory=[],
                  modes=[],
                  default_model_acceptance=False,
                  statistical_equivalence=None)
    try:
        run(a.input, a.output, report)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        atomic(a.output / 'result.json', report)
