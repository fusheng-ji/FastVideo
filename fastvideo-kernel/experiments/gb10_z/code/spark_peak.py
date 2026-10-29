"""Isolated 8192-token dense peak attribution; raw snapshots remain local."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
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
    p.add_argument('--kind', choices=['model'], default='model')
    p.add_argument('--variant', choices=['baseline', 'clip-loop', 'clip-foreach', 'optimizer-loop', 'optimizer-fused', 'grad-bf16', 'release-grad'], default='baseline')
    p.add_argument('--history', action='store_true')
    p.add_argument('--policy', choices=['dense'], required=True)
    p.add_argument('--tokens', type=int, required=True)
    p.add_argument('--heads', type=int, default=12)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--measure', type=int, default=5)
    p.add_argument('--model', type=Path)
    p.add_argument('--data', type=Path)
    a = p.parse_args()
    if len(set(a.lock)) != 2 or a.output.exists():
        p.error('Require two distinct locks and a new, nonexistent output directory')
    if a.warmup < 1 or a.measure < 1 or a.tokens != 8192:
        p.error('Positive budget and length required')
    return a


def main():
    a = parse_args()
    a.output.mkdir(parents=True)
    report = dict(domain='spark-peak-profile-v1', variant=a.variant, history=a.history, replication_label='new replication on new hardware',
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
            from local_policy import statistics_metadata
            torch.cuda.set_device(0)
            assert torch.cuda.get_device_capability() == (12, 1), 'Expected SM121'
            torch.manual_seed(1000)
            torch.use_deterministic_algorithms(True, warn_only=False)
            report['environment'] = {name: importlib.metadata.version(name) for name in ['torch', 'triton', 'flash-attn', 'numpy']}
            report['environment'].update(device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()), cuda=torch.version.cuda)
            report['strict_determinism'] = True
            report['historical_acceptance_inherited'] = False
            assert digest(kernel.__file__) == report['kernel_sha256'], 'Wrong loaded kernel'
            assert kernel._sm121_dv_stats_mode() == 'save', 'Wrong effective stats mode'
            measure_model(a, report, kernel, torch, statistics_metadata, save)
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


def memory(torch):
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())


def sha_tensor(t, torch):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


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
    optimizer = method.get_optimizers(0)[0]
    if a.variant in ('optimizer-loop', 'optimizer-fused'):
        for group in optimizer.param_groups:
            group['foreach'] = False
            group['fused'] = a.variant == 'optimizer-fused'
    report['optimizer_flags'] = [{k: group.get(k) for k in ('foreach', 'fused', 'capturable')} for group in optimizer.param_groups]
    from peak_memory import PhaseRecorder
    recorder = PhaseRecorder(torch, parameters, optimizer, a.output, a.history)

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
            start_unix = time.time()
            tick = time.perf_counter()
            start.record()
            recorder.begin(step)
            phase['name'] = 'forward'
            losses, outputs, _ = method.single_train_step(batch, step + 1)
            forward = recorder.mark('forward')
            loss = float(losses['total_loss'].detach())
            assert np.isfinite(loss), 'Nonfinite loss'
            phase['name'] = 'backward'
            method.backward(losses, outputs, grad_accum_rounds=1)
            backward = recorder.mark('backward')
            assert len(counts) == 825 and all(n == 1 for n in counts.values()), 'Parameter event coverage failure'
            if a.variant == 'release-grad':
                method.optimizers_zero_grad(step)
                recorder.mark('released_before_finite')
            if a.variant == 'grad-bf16':
                # Public assignment must enforce parameter/gradient dtype compatibility.
                # A rejection is an experimental failure, not a reason to bypass it via .data.
                for p in parameters:
                    if p.grad is not None:
                        p.grad = p.grad.to(torch.bfloat16)
            assert all(p.grad is not None and torch.isfinite(local(p.grad)).all().item() for p in parameters if p.requires_grad), 'Missing/nonfinite gradient'
            finite = recorder.mark('finite')
            del losses, outputs
            if a.variant in ('clip-loop', 'clip-foreach'):
                from fastvideo.training.training_utils import clip_grad_norm_
                norm = float(clip_grad_norm_(parameters, 1.0, foreach=a.variant == 'clip-foreach').item())
            else:
                norm = float(clip_grad_norm_if_needed(student.transformer, 1.0))
            clipped = recorder.mark('clip')
            assert np.isfinite(norm), 'Nonfinite norm'
            method.optimizers_schedulers_step(step + 1)
            updated = recorder.mark('optimizer')
            method.optimizers_zero_grad(step + 1)
            zeroed = recorder.mark('zero_grad')
            end.record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            profile = recorder.finish()
            record = dict(start_unix=start_unix, end_unix=time.time(), phase_profile=profile, after_finite=finite, after_clip=clipped, after_optimizer=updated, after_zero_grad=zeroed, step=step, warmup=step < a.warmup, loss=loss, gradient_norm=norm,
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
        recorder.close()
        for handle in handles:
            handle.remove()
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
