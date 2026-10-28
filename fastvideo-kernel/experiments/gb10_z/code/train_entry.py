"""GB10 Wan QAT accumulation experiment using a fixed sample budget.

Runtime weights/data/results live below --artifact-root. These are local
experiment controls; the canonical training API and installed kernel stay unchanged.
"""

from typing import Any
import argparse
import hashlib
import json
import math
import pathlib
import time

GEOMETRY = {
    8192: dict(num_latent_t=16, num_height=256, num_width=512, num_frames=61),
    31200: dict(num_latent_t=20, num_height=480, num_width=832, num_frames=77)
}
STRATA = (0.1, 0.35, 0.6, 0.85)


def parse_args(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['save'], required=True)
    parser.add_argument('--tokens', type=int, choices=sorted(GEOMETRY), default=8192)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--samples', '--max-train-samples', type=int, default=800, dest='samples')
    parser.add_argument('--steps', type=int, default=None, help='smoke-only optimizer update budget')
    parser.add_argument('--grad-accum-steps', type=int, choices=[1, 4], default=4)
    parser.add_argument('--eval-every-samples', type=int, default=200)
    parser.add_argument('--eval-train', type=int, default=8)
    parser.add_argument('--eval-heldout', type=int, default=None, help='smoke tests only: limit validation rows')
    parser.add_argument('--no-eval', action='store_true')
    parser.add_argument('--step0-strata', action='store_true')
    parser.add_argument('--mem-history', action='store_true')
    parser.add_argument('--capture-dir', default=None, help='one forward/backward without optimizer update')
    parser.add_argument('--capture-layers', default='0,15,29')
    parser.add_argument('--capture-at-end', default=None)
    parser.add_argument('--abort-mem-gib', type=float, default=12.0)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--cooldown-temp', type=float, default=None)
    parser.add_argument('--step-telemetry', action='store_true')
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--store-mode', choices=['tma', 'ordinary'], default='tma')
    parser.add_argument('--stats-format', choices=['dense', 'packed_slots'], default='dense')
    parser.add_argument('--event-capacity', type=int, choices=[32, 64], default=32)
    parser.add_argument('--force-short', action='store_true')
    parser.add_argument('--verify-first-window',
                        action='store_true',
                        help='hash-check checkpoint replay and gradient hooks on the first four samples')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args(argv)
    if args.steps is not None:
        args.samples = args.steps * args.grad_accum_steps
    if args.samples <= 0 or args.samples % args.grad_accum_steps:
        parser.error('sample budget must be positive and divisible by gradient accumulation')
    if args.eval_every_samples <= 0 or args.eval_every_samples % args.grad_accum_steps:
        parser.error('evaluation sample interval must be positive and divisible by accumulation')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error('learning rate and max gradient norm must be finite and positive')
    return args


def _run(args: Any) -> Any:
    import pyarrow.parquet as pq
    import torch
    import fastvideo
    from fastvideo_kernel.triton_kernels import attn_qat_train as prototype
    baseline = prototype  # This portable recipe supports save only.
    from fastvideo.attention.backends import attn_qat_train as backend
    from fastvideo.distributed import init_distributed_environment, initialize_model_parallel
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.instantiate import instantiate
    from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
    from fastvideo.train.utils.optimizer import clip_grad_norm_if_needed
    from fastvideo.dataset.utils import collate_rows_from_parquet_schema
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v

    from accumulation import (NonFiniteWindow, accumulation_window, combine_cache_metrics, empty_cache_metrics,
                                       record_saved_cache)

    ROOT = pathlib.Path(args.artifact_root).resolve()
    RUN = pathlib.Path(args.run_dir)
    RUN.mkdir(parents=True, exist_ok=True)
    assert not (RUN / 'metrics.jsonl').exists(), 'Refuse to mix separate training runs'

    def emit(row: Any) -> Any:
        with (RUN / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    def local(x: Any) -> Any:
        return x.to_local() if hasattr(x, 'to_local') else x

    def mem_available_gib() -> Any:
        for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20

    def tensor_sha(t: Any) -> Any:
        return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

    def memory(tag: Any) -> Any:
        free, total = torch.cuda.mem_get_info()
        return {
            f'{tag}_max_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
            f'{tag}_max_reserved_MiB': torch.cuda.max_memory_reserved() / 2**20,
            f'{tag}_allocated_MiB': torch.cuda.memory_allocated() / 2**20,
            f'{tag}_device_free_MiB': free / 2**20,
            f'{tag}_host_available_GiB': mem_available_gib()
        }

    # ---------------------------------------------------------------- kernel binding / capture
    shape_counts: dict[str, int] = {}
    selected = baseline if args.variant == 'baseline' else prototype
    capture_layers = [int(x) for x in args.capture_layers.split(',')]
    phase: dict[str, Any] = {'name': 'idle', 'forward_calls': 0, 'recompute_calls': 0}
    captured: dict[Any, Any] = {}
    hook_fires: dict[Any, Any] = {}
    recompute_hashes: dict[Any, Any] = {}

    def attention(*values: Any) -> Any:
        q, k, v = values[:3]
        key = str((q.shape[1], q.shape[2], k.shape[2], q.shape[3]))
        shape_counts[key] = shape_counts.get(key, 0) + 1
        out = selected.attention(*values)
        if phase['name'] == 'forward' and phase.get('cache_metrics') is not None:
            record_saved_cache(phase['cache_metrics'], out.grad_fn, q.shape, k.shape)
        if not phase.get('capture') or q.shape[2] != k.shape[2]:
            return out
        if phase['name'] == 'forward':
            layer = phase['forward_calls']
            phase['forward_calls'] += 1
            if layer in capture_layers:
                slot = captured.setdefault(layer, {})
                if phase.get('verify_only'):
                    slot['forward_sha256'] = {
                        name: tensor_sha(t)
                        for name, t in (('q', q), ('k', k), ('v', v), ('out', out))
                    }
                else:
                    for name, t in (('q', q), ('k', k), ('v', v), ('out', out)):
                        slot[name] = t.detach().to('cpu', copy=True)

                def hook(name: Any) -> Any:

                    def record(grad: Any) -> Any:
                        if phase.get('verify_only'):
                            slot.setdefault('gradient_finite', {})[name] = bool(torch.isfinite(grad).all())
                        else:
                            slot[name] = grad.detach().to('cpu', copy=True)
                        hook_fires[(layer, name)] = hook_fires.get((layer, name), 0) + 1

                    return record

                out.register_hook(hook('dout'))
                for name, t in (('dq', q), ('dk', k), ('dv', v)):
                    t.register_hook(hook(name))
        elif phase['name'] == 'backward':
            layer = 29 - phase['recompute_calls']
            phase['recompute_calls'] += 1
            if layer in capture_layers:
                recompute_hashes[layer] = {n: tensor_sha(t) for n, t in (('q', q), ('k', k), ('v', v), ('out', out))}
        return out

    backend._attn_qat_train_attention = attention
    backend._attn_qat_train_import_attempted = True
    backend._attn_qat_train_import_error = None

    # ---------------------------------------------------------------- model / config
    cfg = load_run_config(str(__import__('portable').settings().config))
    cfg.training.loop.max_train_steps = args.samples // args.grad_accum_steps
    cfg.training.loop.gradient_accumulation_steps = args.grad_accum_steps
    cfg.models['student']['init_from'] = str(__import__('portable').settings().model)
    cfg.training.data.data_path = str(__import__('portable').settings().train_data.parent)
    cfg.training.checkpoint.output_dir = str(RUN)
    cfg.training.data.seed = args.seed
    if args.lr is not None:
        cfg.training.optimizer.learning_rate = args.lr
    cfg.training.optimizer.lr_scheduler = 'constant'
    cfg.training.optimizer.lr_warmup_steps = 0
    for key, value in GEOMETRY[args.tokens].items():
        assert getattr(cfg.training.data, key) == value, key
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    init_distributed_environment(distributed_init_method='file://' + str(RUN / 'distributed-rendezvous'))
    initialize_model_parallel()
    if args.mem_history:
        torch.cuda.memory._record_memory_history(max_entries=200000)
    student = instantiate(cfg.models['student'], training_config=cfg.training)
    student.set_requires_negative_conditioning(False)
    method = FineTuneMethod(cfg=cfg, role_models={'student': student})
    method.on_train_start()
    transformer = student.transformer
    parameters = [p for p in transformer.parameters() if p.requires_grad]
    v_weights = [(n, p) for n, p in transformer.named_parameters() if '.to_v.weight' in n and '.attn2.' not in n]
    assert len(v_weights) == 30, len(v_weights)
    monitored = [v_weights[i] for i in (0, 15, 29)]
    initial_weights = {n: local(p.detach()).float().cpu().clone() for n, p in monitored}
    full_hash = hashlib.sha256()
    for name, p in transformer.named_parameters():
        full_hash.update(name.encode())
        full_hash.update(local(p.detach()).cpu().contiguous().view(torch.uint8).numpy().tobytes())
    emit({
        'event': 'loaded',
        'variant': args.variant,
        'tokens': args.tokens,
        'seed': args.seed,
        'trainable_parameters': sum(p.numel() for p in parameters),
        'grad_accum_steps': args.grad_accum_steps,
        'samples': args.samples,
        'optimizer_steps': args.samples // args.grad_accum_steps,
        'store_mode': args.store_mode,
        'stats_format': args.stats_format,
        'event_capacity': args.event_capacity,
        'evaluation_role': 'reused validation split; not an independent heldout test',
        'master_dtypes': sorted({str(p.dtype)
                                 for p in parameters}),
        'attention_backend': student.attention_backend_name,
        'checkpointing': 'full',
        'fastvideo_path': fastvideo.__file__,
        'prototype_path': prototype.__file__,
        'baseline_path': baseline.__file__,
        'initial_full_parameter_sha256': full_hash.hexdigest(),
        'initial_monitored_sha256': {
            n: tensor_sha(t)
            for n, t in initial_weights.items()
        },
        **memory('loaded')
    })

    def batch_identity(batch: Any) -> Any:
        infos = batch.get('info_list') or [{}]
        return {
            'id': infos[0].get('id') or infos[0].get('file_name'),
            'latent_sha256': tensor_sha(batch['vae_latent'])[:16]
        }

    def dump_failure(reason: Any, step: Any, batch: Any = None, extra: Any = None) -> Any:
        payload = {
            'reason': reason,
            'step': step,
            'rng_state': method.cuda_generator.get_state(),
            'monitored_sha256': {
                n: tensor_sha(p)
                for n, p in monitored
            },
            'extra': extra
        }
        if batch is not None:
            payload['batch'] = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
            payload['batch_identity'] = batch_identity(batch)
        torch.save(payload, RUN / 'failure.pt')
        if args.mem_history:
            torch.cuda.memory._dump_snapshot(str(RUN / 'failure-memory-snapshot.pickle'))
        emit({'event': 'failure', 'reason': reason, 'step': step, **memory('failure')})

    # ---------------------------------------------------------------- evaluation
    eval_rows = []
    if not args.no_eval:
        train_table = pq.read_table(__import__('portable').settings().train_data)
        heldout_table = pq.read_table(__import__('portable').settings().validation_data)
        eval_rows = [('train', i, r) for i, r in enumerate(train_table.slice(0, args.eval_train).to_pylist())]
        heldout_rows = heldout_table.to_pylist()
        if len(heldout_rows) != 32 or args.eval_train != 0:
            raise RuntimeError('Calibration requires exactly 32 reused validation rows and no train evaluation')
        eval_rows += [('heldout', i, r) for i, r in enumerate(heldout_rows)]
        del train_table, heldout_table
    scheduler_timesteps = student.noise_scheduler.timesteps
    num_train_timesteps = student.noise_scheduler.config.num_train_timesteps

    from calibration_runtime import evaluation_guard

    @evaluation_guard(method, transformer)
    def evaluate(step: Any, strata: Any = False) -> Any:
        old_state = method.cuda_generator.get_state()
        was_training = transformer.training
        original_sampler = student._sample_timesteps
        transformer.eval()
        results: dict[Any, Any] = {}
        draws = [(None, 3000)]
        if strata:
            raise RuntimeError('Stratified evaluation is outside this calibration budget')
        tick = time.time()
        try:
            with torch.no_grad():
                for quantile, base in draws:
                    if quantile is not None:
                        index = min(int(quantile * num_train_timesteps), len(scheduler_timesteps) - 1)
                        fixed = scheduler_timesteps[index:index + 1]
                        student._sample_timesteps = lambda batch_size, device, generator, fixed=fixed: fixed.to(device)
                    for split, index_row, row in eval_rows:
                        batch = collate_rows_from_parquet_schema([row], pyarrow_schema_t2v, 512, cfg_rate=0., seed=1000)
                        seed = base + (0 if split == 'train' else 100) + index_row
                        method.cuda_generator.manual_seed(seed)
                        losses, outputs, _ = method.single_train_step(batch, step)
                        value = losses['total_loss'].item()
                        assert math.isfinite(value)
                        tag = 'fixed' if quantile is None else f'q{quantile}'
                        results.setdefault((tag, split), []).append(value)
                        emit({
                            'event': 'eval-sample',
                            'step': step,
                            'draw': tag,
                            'split': split,
                            'index': index_row,
                            'id': row['id'],
                            'loss': value,
                            'noise_seed': seed,
                            'timestep': outputs['_fv_backward'][0].detach().float().cpu().tolist()
                        })
                        del losses, outputs
                    student._sample_timesteps = original_sampler
        finally:
            student._sample_timesteps = original_sampler
            method.cuda_generator.set_state(old_state)
            transformer.train(was_training)
        means = {f'{tag}_{split}_mean': sum(v) / len(v) for (tag, split), v in results.items()}
        emit({
            'event': 'evaluation',
            'step': step,
            'strata': strata,
            'eval_seconds': time.time() - tick,
            'train_samples': sum(1 for s, _, _ in eval_rows if s == 'train'),
            'heldout_samples': sum(1 for s, _, _ in eval_rows if s == 'heldout'),
            'samples_seen': step,
            'evaluation_role': 'reused validation split',
            **means
        })
        return means

    # ---------------------------------------------------------------- capture mode
    def gpu_state() -> Any:
        import subprocess
        response = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,power.draw', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True)
        out = response.stdout.strip().split(',')
        try:
            return {'gpu_temp_C': float(out[0]), 'gpu_sm_MHz': float(out[1]), 'gpu_power_W': float(out[2])}
        except (ValueError, IndexError):
            return {'gpu_state_raw': ','.join(out)}

    def run_capture(capture_dir: Any, batch: Any, stage: Any) -> Any:
        out_dir = pathlib.Path(capture_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        phase.update(name='idle',
                     forward_calls=0,
                     recompute_calls=0,
                     capture=True,
                     verify_only=False,
                     cache_metrics=None)
        captured.clear()
        hook_fires.clear()
        recompute_hashes.clear()
        method.optimizers_zero_grad(0)
        phase['name'] = 'forward'
        losses, outputs, _ = method.single_train_step(batch, 1)
        phase['name'] = 'backward'
        method.backward(losses, outputs, grad_accum_rounds=1)
        torch.cuda.synchronize()
        phase['name'] = 'idle'
        assert phase['forward_calls'] == 30 and phase['recompute_calls'] == 30, phase
        manifest = {
            'variant': args.variant,
            'tokens': args.tokens,
            'seed': args.seed,
            'stage': stage,
            'lr': args.lr,
            'batch_identity': batch_identity(batch),
            'loss': losses['total_loss'].item(),
            'hook_fires': {
                f'{layer_id}:{n}': c
                for (layer_id, n), c in sorted(hook_fires.items())
            },
            'layers': {}
        }
        for layer, slot in captured.items():
            assert all(hook_fires.get((layer, n)) == 1 for n in ('dout', 'dq', 'dk', 'dv')), (layer, hook_fires)
            path = out_dir / f'layer{layer:02d}.pt'
            torch.save(slot, path)
            forward_hashes = {n: tensor_sha(slot[n]) for n in ('q', 'k', 'v', 'out')}
            manifest['layers'][layer] = {
                'file': path.name,
                'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'shapes': {
                    n: list(t.shape)
                    for n, t in slot.items()
                },
                'forward_sha256': forward_hashes,
                'recompute_sha256': recompute_hashes.get(layer),
                'recompute_bitwise_equal': forward_hashes == recompute_hashes.get(layer)
            }
        (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        emit({
            'event': 'captured',
            **{
                k: v
                for k, v in manifest.items() if k != 'layers'
            }, 'recompute_bitwise_equal': {
                layer_id: m['recompute_bitwise_equal']
                for layer_id, m in manifest['layers'].items()
            },
            **memory('capture')
        })
        phase['capture'] = False
        method.optimizers_zero_grad(0)

    if args.capture_dir is not None:
        run_capture(args.capture_dir, next(iter(student.dataloader)), 'initial-weights')
        torch.distributed.destroy_process_group()
        raise SystemExit(0)

    emit({'event': 'precompile-skipped', 'reason': 'diagnostic timing includes first-call compilation'})

    # ---------------------------------------------------------------- training
    evaluations = {}
    if not args.no_eval:
        evaluations[0] = evaluate(0, strata=args.step0_strata)
    if args.cooldown_temp is not None:
        tick = time.time()
        state = gpu_state()
        while state.get('gpu_temp_C', 0) > args.cooldown_temp and time.time() - tick < 1800:
            time.sleep(5)
            state = gpu_state()
        emit({'event': 'cooldown', 'target_C': args.cooldown_temp, 'waited_sec': time.time() - tick, **state})
    stream = iter(student.dataloader)
    started = time.time()
    method.optimizers_zero_grad(0)
    grad_history = []
    cache_history = []
    active_training_seconds = 0.0
    samples_seen = 0
    optimizer_steps = args.samples // args.grad_accum_steps
    for step in range(1, optimizer_steps + 1):
        window_batches = []
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(stream)
            except StopIteration:
                stream = iter(student.dataloader)
                batch = next(stream)
            window_batches.append(batch)
        identities = [batch_identity(batch) for batch in window_batches]
        before = shape_counts.copy()
        torch.cuda.reset_peak_memory_stats()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.time()
        start_event.record()
        phase_memory: dict[str, Any] = {}
        current_micro = {'index': 0}
        micro_states: list[Any] = []
        lr_used = [float(group['lr']) for group in method._student_optimizer.param_groups]

        def before_microstep(index: Any,
                             batch: Any,
                             *,
                             current_micro: Any = current_micro,
                             samples_seen: Any = samples_seen,
                             micro_states: Any = micro_states,
                             identities: Any = identities) -> Any:
            current_micro['index'] = index
            phase.update(name='forward', cache_metrics=empty_cache_metrics())
            if args.verify_first_window and samples_seen + index < 4:
                captured.clear()
                hook_fires.clear()
                recompute_hashes.clear()
                phase.update(name='forward', forward_calls=0, recompute_calls=0, capture=True, verify_only=True)
            micro_states.append({
                'rng_before_sha256':
                hashlib.sha256(method.cuda_generator.get_state().numpy().tobytes()).hexdigest(),
                'identity':
                identities[index],
            })

        def before_backward(index: Any, batch: Any, losses: Any, outputs: Any) -> Any:
            phase['name'] = 'backward'

        def after_microstep(index: Any,
                            batch: Any,
                            losses: Any,
                            outputs: Any,
                            *,
                            micro_states: Any = micro_states,
                            step: Any = step,
                            samples_seen: Any = samples_seen,
                            phase_memory: Any = phase_memory) -> Any:
            context = outputs.get('_fv_backward')
            timesteps = context[0].detach().float().cpu().tolist() if context is not None else None
            cache = phase['cache_metrics']
            micro_states[index]['cache_metrics'] = cache
            emit({
                'event': 'train-microstep',
                'optimizer_step': step,
                'microstep': samples_seen + index + 1,
                'samples_seen': samples_seen + index + 1,
                'accumulation_index': index,
                'grad_accum_steps': args.grad_accum_steps,
                'loss': losses['total_loss'].item(),
                'timesteps': timesteps,
                'batch': micro_states[index]['identity'],
                'saved_statistics_cache': cache,
                'rng_before_sha256': micro_states[index]['rng_before_sha256'],
                'rng_after_sha256': hashlib.sha256(method.cuda_generator.get_state().numpy().tobytes()).hexdigest()
            })
            if phase.get('verify_only') and phase.get('capture'):
                verification = {}
                for layer in capture_layers:
                    slot = captured.get(layer, {})
                    parity = slot.get('forward_sha256') == recompute_hashes.get(layer)
                    hooks_once = all(hook_fires.get((layer, name)) == 1 for name in ('dout', 'dq', 'dk', 'dv'))
                    gradient_finite = slot.get('gradient_finite', {})
                    finite = len(gradient_finite) == 4 and all(gradient_finite.values())
                    verification[str(layer)] = {
                        'forward_sha256': slot.get('forward_sha256'),
                        'recompute_sha256': recompute_hashes.get(layer),
                        'checkpoint_bitwise_equal': parity,
                        'hooks_once': hooks_once,
                        'gradient_finite': gradient_finite,
                        'accepted': bool(parity and hooks_once and finite),
                    }
                counts_valid = phase['forward_calls'] == phase['recompute_calls'] == 30
                accepted = counts_valid and all(row['accepted'] for row in verification.values())
                emit({
                    'event': 'first-window-verification',
                    'microstep': samples_seen + index + 1,
                    'optimizer_step': step,
                    'forward_calls': phase['forward_calls'],
                    'checkpoint_recompute_calls': phase['recompute_calls'],
                    'layers': verification,
                    'accepted': accepted
                })
                phase.update(name='idle', capture=False, verify_only=False)
                if not accepted:
                    raise RuntimeError('First-window checkpoint hash, hook or finite verification failed')
            phase['name'] = 'idle'
            if step == 1 and index == args.grad_accum_steps - 1:
                torch.cuda.synchronize()
                phase_memory.update(memory('after_backward'))
                if args.mem_history:
                    torch.cuda.memory._dump_snapshot(str(RUN / 'memory-after-backward.pickle'))

        def inspect_gradients() -> Any:
            v_grad = [local(p.grad).float().norm().item() if p.grad is not None else 0. for _, p in v_weights]
            if not all(math.isfinite(value) for value in v_grad):
                raise NonFiniteWindow('Nonfinite V projection gradient')
            return {'v_grad_norm': v_grad, 'v_grad_zero_layers': sum(value == 0. for value in v_grad)}

        try:
            result = accumulation_window(
                method,
                window_batches,
                iteration=step,
                rounds=args.grad_accum_steps,
                clip_gradients=lambda: float(clip_grad_norm_if_needed(transformer, args.max_grad_norm)),
                inspect_gradients=inspect_gradients,
                before_microstep=before_microstep,
                before_backward=before_backward,
                after_microstep=after_microstep)
        except BaseException as error:
            dump_failure('accumulation-window', step, window_batches[current_micro['index']], {
                'microstep': samples_seen + current_micro['index'] + 1,
                'error': repr(error)
            })
            raise
        end_event.record()
        torch.cuda.synchronize()
        wall_end = time.time()
        samples_seen += args.grad_accum_steps
        if step == 1:
            phase_memory.update(memory('after_optimizer_step'))
            if args.mem_history:
                torch.cuda.memory._dump_snapshot(str(RUN / 'memory-after-optimizer.pickle'))
        grad_history.append(result.gradient_norm)
        cache = combine_cache_metrics([row['cache_metrics'] for row in micro_states])
        cache_history.append(cache)
        active_training_seconds += wall_end - wall_start
        host = mem_available_gib()
        clip_coefficient = min(1.0, args.max_grad_norm / (result.gradient_norm + 1e-6))
        emit({
            'event':
            'train-step',
            'step':
            step,
            'optimizer_step':
            step,
            'microstep':
            samples_seen,
            'samples_seen':
            samples_seen,
            'grad_accum_steps':
            args.grad_accum_steps,
            'loss':
            result.losses['total_loss'],
            'grad_norm_before_clip':
            result.gradient_norm,
            'clipped':
            result.gradient_norm > args.max_grad_norm,
            'clip_coefficient':
            clip_coefficient,
            'lr':
            lr_used[0],
            'lr_used':
            lr_used,
            'lr_next':
            method._student_lr_scheduler.get_last_lr()[0],
            **result.diagnostics, 'batches':
            identities,
            'saved_statistics_cache':
            cache,
            'cuda_event_ms':
            start_event.elapsed_time(end_event),
            'wall_start':
            wall_start,
            'wall_end':
            wall_end,
            'timing_exclude_reason':
            'diagnostic first update includes compilation and optimizer allocation' if step == 1 else None,
            'active_training_seconds_raw':
            active_training_seconds,
            'elapsed_training_sec':
            wall_end - started,
            'step_max_allocated_MiB':
            torch.cuda.max_memory_allocated() / 2**20,
            'step_max_reserved_MiB':
            torch.cuda.max_memory_reserved() / 2**20,
            'host_available_GiB':
            host,
            'qat_launch_shapes': {
                key: shape_counts[key] - before.get(key, 0)
                for key in shape_counts if shape_counts[key] != before.get(key, 0)
            },
            **phase_memory,
            **(gpu_state() if args.step_telemetry else {})
        })
        del window_batches
        if host < args.abort_mem_gib:
            dump_failure('host-memory', step, extra={'host_available_GiB': host})
            raise SystemExit(4)
        if not args.no_eval and (samples_seen % args.eval_every_samples == 0 or samples_seen == args.samples):
            evaluations[samples_seen] = evaluate(samples_seen, strata=False)

    final_weights = {n: local(p.detach()).float().cpu().clone() for n, p in monitored}
    updates = {
        n: {
            'relative_l2': ((final_weights[n] - initial_weights[n]).norm() / initial_weights[n].norm()).item(),
            'initial_sha256': tensor_sha(initial_weights[n]),
            'final_sha256': tensor_sha(final_weights[n])
        }
        for n in initial_weights
    }
    torch.save({
        'monitored_initial_weights': initial_weights,
        'monitored_final_weights': final_weights
    }, RUN / 'monitored-v-weights.pt')
    summary = {
        'variant': args.variant,
        'tokens': args.tokens,
        'seed': args.seed,
        'steps': optimizer_steps,
        'optimizer_steps': optimizer_steps,
        'samples_seen': samples_seen,
        'grad_accum_steps': args.grad_accum_steps,
        'lr': args.lr,
        'real_pretrained_model': 'Wan2.1-T2V-1.3B',
        'initial_full_parameter_sha256': full_hash.hexdigest(),
        'evaluations': {
            str(k): v
            for k, v in evaluations.items()
        },
        'updates': updates,
        'launch_shapes': shape_counts,
        'elapsed_training_sec': time.time() - started,
        'grad_norm_median': sorted(grad_history)[len(grad_history) // 2],
        'store_mode': args.store_mode,
        'stats_format': args.stats_format,
        'event_capacity': args.event_capacity,
        'evaluation_role': 'reused validation split; not an independent heldout test',
        'active_training_seconds_raw': active_training_seconds,
        'saved_statistics_cache_totals': combine_cache_metrics(cache_history),
        'scope': ('early full-parameter training on real MixKit latents; '
                  'not long-term convergence or video quality')
    }
    if args.capture_at_end is not None:
        run_capture(args.capture_at_end, next(iter(student.dataloader)),
                    f'after-{samples_seen}-samples-{optimizer_steps}-updates')
    (RUN / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    emit({'event': 'finished', **summary})
    torch.distributed.destroy_process_group()


def main() -> Any:
    args = parse_args()
    from fastvideo_kernel.triton_kernels import attn_qat_train as prototype
    baseline = prototype  # This portable recipe supports save only.
    from train_policy import kernel_options
    selected = baseline if args.variant == 'baseline' else prototype
    with kernel_options(selected,
                        store_mode=args.store_mode,
                        stats_format=args.stats_format,
                        event_capacity=args.event_capacity,
                        force_short=args.force_short):
        _run(args)


if __name__ == '__main__':
    main()
