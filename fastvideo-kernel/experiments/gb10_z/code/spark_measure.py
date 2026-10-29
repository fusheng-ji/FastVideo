"""Bounded, fresh-directory Spark replication; no production defaults are changed."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[3]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path, action='append', required=True)
    p.add_argument('--kind', choices=['kernel', 'model', 'reference', 'overflow'], default='kernel')
    p.add_argument('--policy', choices=['dense', 'recompute', 'Z'], required=True)
    p.add_argument('--tokens', type=int, required=True)
    p.add_argument('--heads', type=int, default=12)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--measure', type=int, default=5)
    p.add_argument('--model', type=Path)
    p.add_argument('--data', type=Path)
    a = p.parse_args()
    if len(set(a.lock)) != 2 or a.output.exists():
        p.error('Require two distinct locks and a new, nonexistent output directory')
    if a.warmup < 1 or a.measure < 1 or a.tokens < 1:
        p.error('Positive budget and length required')
    return a


def main():
    a = parse_args()
    a.output.mkdir(parents=True)
    report = dict(domain='spark-new-replication-v1', replication_label='new replication on new hardware',
                  kind=a.kind, policy=a.policy, tokens=a.tokens, heads=a.heads,
                  warmup=a.warmup, measurements=a.measure, status='started',
                  kernel_sha256=digest(REPO / 'fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py'),
                  harness_sha256=digest(__file__), seed=1000)
    def save():
        (a.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    try:
        with ExitStack() as stack:
            for path in a.lock:
                handle = stack.enter_context(path.open('a+'))
                fcntl.flock(handle, fcntl.LOCK_EX)
            import torch
            import portable
            portable.configure()
            from fastvideo_kernel.triton_kernels import attn_qat_train as kernel
            from local_policy import kernel_policy, statistics_metadata
            torch.cuda.set_device(0)
            assert torch.cuda.get_device_capability() == (12, 1), 'Expected SM121'
            torch.manual_seed(1000)
            torch.use_deterministic_algorithms(True, warn_only=False)
            report['environment'] = {name: importlib.metadata.version(name) for name in ['torch', 'triton', 'flash-attn', 'numpy']}
            report['environment'].update(device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()), cuda=torch.version.cuda)
            report['strict_determinism'] = True
            report['historical_acceptance_inherited'] = False
            assert digest(kernel.__file__) == report['kernel_sha256'], 'Wrong loaded kernel'
            policy = kernel_policy(kernel, 'Z') if a.policy == 'Z' else nullcontext()
            with policy:
                assert kernel._sm121_dv_stats_mode() == ('recompute' if a.policy == 'recompute' else 'save'), 'Wrong effective stats mode'
                if a.kind == 'model':
                    measure_model(a, report, kernel, torch, statistics_metadata, save)
                else:
                    measure_kernel(a, report, kernel, torch, statistics_metadata, save)
            report['status'] = 'completed'
    except BaseException as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        # Keep detailed tracebacks local; do not publish machine paths from exception text.
        report['error'] = str(error).replace(str(REPO), '<REPO>').replace(str(a.output), '<OUTPUT>')
        save()
        raise
    save()
    print(json.dumps(dict(status=report['status'], kind=a.kind, policy=a.policy, tokens=a.tokens)))


def qat(kernel, q, k, v):
    return kernel.attention(q, k, v, False, 128**-0.5, True, False, False, True,
                            False, True, True, False, False, False)


def memory(torch):
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())


def sha_tensor(t, torch):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def measure_kernel(a, report, kernel, torch, metadata, save):
    shape = (1, a.heads, a.tokens, 128)
    q, k, v = [torch.randn(shape, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    do = torch.randn_like(q)
    if a.kind == 'overflow':
        # Strictly increasing key-tile logits force >64 running-maximum changes.
        with torch.no_grad():
            q.fill_(1)
            tile = torch.arange(a.tokens, device='cuda') // 32
            k.copy_((tile.float() / max(1, tile.max().item()) * 4 + .125)[None, None, :, None])
    rows = []
    for step in range(a.warmup + a.measure):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = memory(torch)
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        tick = time.perf_counter()
        start.record()
        out = qat(kernel, q, k, v)
        meta = metadata(out.grad_fn)
        after_forward = memory(torch)
        saved_reference = None
        if a.kind == 'reference':
            assert a.policy == 'dense'
            saved_reference = tuple(t.detach() for t in out.grad_fn.saved_tensors)
        grads = torch.autograd.grad(out, (q, k, v), do)
        end.record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - tick
        row = dict(step=step, warmup=step < a.warmup, wall_seconds=wall,
                   cuda_ms=start.elapsed_time(end), before=before, after_forward=after_forward,
                   after_backward=memory(torch), statistics=meta)
        assert all(torch.isfinite(t).all().item() for t in (out, *grads)), 'Nonfinite output or gradient'
        if step == a.warmup:
            report['tensor_sha256'] = {n: sha_tensor(t, torch) for n, t in zip(['O', 'dQ', 'dK', 'dV'], [out, *grads])}
            if a.kind == 'reference':
                report['reference'] = dv_reference(saved_reference, do, grads[2], torch)
            if a.kind == 'overflow':
                assert meta['attempted_spill_tile_count'] > 0, 'Overflow case failed to overflow'
                assert meta['local_replay'] or meta['overflow_fallback'], 'Unsafe overflow routing'
            if meta['local_replay']:
                computed = kernel._sm121_local_cache_bytes(1, a.heads, a.tokens, a.tokens,
                                                          meta['spill_tile_count'], 64, True, False)
                assert computed == meta['storage_bytes'], 'Cache accounting mismatch'
                report['local_cache_bytes_formula'] = computed
        del out, grads, saved_reference
        rows.append(row)
        report['steps'] = rows
        save()
    samples = rows[a.warmup:]
    report['median_wall_seconds'] = statistics.median(r['wall_seconds'] for r in samples)
    report['median_cuda_ms'] = statistics.median(r['cuda_ms'] for r in samples)
    report['peak_allocated'] = max(r['after_backward']['peak_allocated'] for r in samples)
    report['peak_reserved'] = max(r['after_backward']['peak_reserved'] for r in samples)
    report['timing_scope'] = 'attention forward + backward, allocations and synchronization; no optimizer'


def dv_reference(saved, do, actual, torch):
    # Independent Torch quantization; use saved quantized Q/K and online maxima/denominator.
    sys.path.insert(0, str(REPO / 'fastvideo-kernel/tests'))
    from qat_forward_reference import fp4_fake_quantize
    q, k, _, _, _, maxima, denominator = saved
    n = q.shape[2]
    result = torch.empty_like(actual)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for tile, lo in enumerate(range(0, n, 32)):
            hi = min(lo + 32, n)
            key = torch.nn.functional.pad(k[:, :, lo:hi].float(), (0, 0, 0, 32 - (hi-lo)))
            acc = torch.zeros_like(key)
            for start in range(0, n, 1024):
                stop = min(start + 1024, n)
                logits = q[:, :, start:stop].float() @ key.transpose(-1, -2)
                probability = torch.exp2(logits * (128**-.5 * 1.44269504) - maxima[:, :, tile, start:stop, None])
                mask = torch.arange(32, device=q.device) < hi-lo
                quantized = fp4_fake_quantize(probability, mask.expand_as(probability)).float()
                correction = torch.exp2(maxima[:, :, tile, start:stop] - maxima[:, :, -1, start:stop]) / denominator[:, :, start:stop]
                weights = quantized * correction[..., None]
                acc += weights.transpose(-1, -2) @ do[:, :, start:stop].float()
            result[:, :, lo:hi] = acc[:, :, :hi-lo]
        diff = (actual.float() - result.float()).double()
        norm = torch.linalg.vector_norm(result.double(), dim=(-2, -1))
        rel = torch.linalg.vector_norm(diff, dim=(-2, -1)) / norm
        assert torch.isfinite(rel).all() and (rel <= 1e-3).all(), 'dV reference relative L2 exceeded 1e-3'
        return dict(scope='blocked dV only, conditioned on saved quantized Q/K and online maxima/denominator',
                    relative_l2_per_head=rel.cpu().tolist(), threshold=1e-3, passed=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def measure_model(a, report, kernel, torch, metadata, save):
    import random
    import numpy as np
    import pyarrow.parquet as pq
    from fastvideo.attention.backends import attn_qat_train as backend
    from fastvideo.distributed import init_distributed_environment, initialize_model_parallel
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.instantiate import instantiate
    from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
    from fastvideo.train.utils.optimizer import clip_grad_norm_if_needed
    from fastvideo.dataset.utils import collate_rows_from_parquet_schema
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
    assert a.model and a.data, 'Explicit model and parquet paths are required'
    import yaml
    config = yaml.safe_load((CODE.parent / 'configs/calibration-8192.yaml').read_text())
    config['models']['student']['init_from'] = str(a.model)
    config['training']['data']['data_path'] = str(a.data.parent)
    config['training']['checkpoint']['output_dir'] = str(a.output)
    config_path = a.output / 'resolved-config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    cfg = load_run_config(str(config_path))
    cfg.models['student']['init_from'] = str(a.model)
    cfg.training.data.data_path = str(a.data.parent)
    cfg.training.checkpoint.output_dir = str(a.output)
    cfg.training.loop.max_train_steps = a.warmup + a.measure
    geometry = {8192: (16, 256, 512, 61), 16384: (16, 512, 512, 61), 31200: (20, 480, 832, 77)}
    for key, value in zip(('num_latent_t', 'num_height', 'num_width', 'num_frames'), geometry[a.tokens]):
        setattr(cfg.training.data, key, value)
    random.seed(1000)
    np.random.seed(1000)
    torch.manual_seed(1000)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    init_distributed_environment(distributed_init_method='file://' + str(a.output / 'rendezvous'))
    initialize_model_parallel()
    phase = {'name': 'idle'}
    statistics_rows = []
    def attention(*args):
        out = kernel.attention(*args)
        row = metadata(out.grad_fn)
        row['phase'] = phase['name']
        row['nq'] = args[0].shape[2]
        assert row['nq'] == a.tokens, 'Unexpected QAT sequence length'
        assert row['forward_consistent_dv'], 'Wrong QAT route'
        statistics_rows.append(row)
        return out
    backend._attn_qat_train_attention = attention
    backend._attn_qat_train_import_attempted = True
    backend._attn_qat_train_import_error = None
    student = instantiate(cfg.models['student'], training_config=cfg.training)
    student.set_requires_negative_conditioning(False)
    method = FineTuneMethod(cfg=cfg, role_models={'student': student})
    method.on_train_start()
    parameters = list(student.transformer.parameters())
    def local(t):
        return t.to_local() if hasattr(t, 'to_local') else t
    row = pq.read_table(a.data).slice(0, 1).to_pylist()[0]
    batch = collate_rows_from_parquet_schema([row], pyarrow_schema_t2v, 512, cfg_rate=0., seed=1000)
    if a.tokens == 16384:
        batch['vae_latent'] = torch.nn.functional.interpolate(batch['vae_latent'].float(), size=(16, 64, 64), mode='trilinear', align_corners=False).to(batch['vae_latent'].dtype)
    report['input'] = dict(parquet_sha256=digest(a.data), latent_sha256=sha_tensor(batch['vae_latent'], torch),
                           latent_shape=list(batch['vae_latent'].shape),
                           preprocessing='trilinear resize of 8192 latent to 16384 geometry' if a.tokens == 16384 else 'original cached latent',
                           repeated_batch=True)
    report['model'] = dict(name='Wan2.1-T2V-1.3B', checkpoint='full', G=1, lr=1e-6, clip=1.0,
                          parameter_count=sum(p.numel() for p in parameters), trainable_tensors=sum(p.requires_grad for p in parameters))
    counts = {}
    handles = []
    for i, p in enumerate(parameters):
        if p.requires_grad:
            def hook(parameter, i=i):
                counts[i] = counts.get(i, 0) + 1
            handles.append(p.register_post_accumulate_grad_hook(hook))
    rows = []
    try:
        for step in range(a.warmup + a.measure):
            statistics_rows.clear()
            counts.clear()
            method.optimizers_zero_grad(step)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = memory(torch)
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            tick = time.perf_counter()
            start.record()
            phase['name'] = 'forward'
            losses, outputs, _ = method.single_train_step(batch, step + 1)
            forward = memory(torch)
            loss = float(losses['total_loss'].detach())
            assert np.isfinite(loss), 'Nonfinite loss'
            phase['name'] = 'backward'
            method.backward(losses, outputs, grad_accum_rounds=1)
            backward = memory(torch)
            assert len(counts) == 825 and all(n == 1 for n in counts.values()), 'Parameter event coverage failure'
            assert all(p.grad is not None and torch.isfinite(local(p.grad)).all().item() for p in parameters if p.requires_grad), 'Missing/nonfinite gradient'
            del losses, outputs
            norm = float(clip_grad_norm_if_needed(student.transformer, 1.0))
            assert np.isfinite(norm), 'Nonfinite norm'
            method.optimizers_schedulers_step(step + 1)
            method.optimizers_zero_grad(step + 1)
            end.record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            record = dict(step=step, warmup=step < a.warmup, loss=loss, gradient_norm=norm,
                          wall_seconds=elapsed, cuda_ms=start.elapsed_time(end), before=before,
                          after_forward=forward, after_backward=backward, after_update=memory(torch),
                          parameter_events=len(counts), statistics=list(statistics_rows))
            assert sum(r['phase'] == 'forward' for r in statistics_rows) == 30, 'Missing forward layers'
            rows.append(record)
            report['steps'] = rows
            save()
            print(json.dumps(dict(step=step, policy=a.policy, tokens=a.tokens, seconds=elapsed)), flush=True)
        measured = rows[a.warmup:]
        report['median_wall_seconds'] = statistics.median(r['wall_seconds'] for r in measured)
        report['median_cuda_ms'] = statistics.median(r['cuda_ms'] for r in measured)
        report['peak_allocated'] = max(r['after_update']['peak_allocated'] for r in measured)
        report['peak_reserved'] = max(r['after_update']['peak_reserved'] for r in measured)
        report['timing_scope'] = 'Wan forward, checkpoint backward, gradient finite checks, clipping, AdamW update, zeroing and synchronize; batch already loaded'
        report['performance_certification'] = False
        report['final_parameter_sha256'] = {str(i): sha_tensor(local(p), torch) for i, p in enumerate(parameters)}
    finally:
        for handle in handles:
            handle.remove()
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
