"""Private full Wan gradient probe; execute only through the diagnostic GPU scheduler.

No production files are edited. Source policy covers forward AND backward.
Imports at module scope are CPU-safe so the harness can be inspected without CUDA.
"""
from __future__ import annotations
from typing import Any

import argparse
import hashlib
import inspect
import json
import os
import random
from pathlib import Path
import sys
import time
import traceback
import uuid

from probe_helpers import (EventLedger, atomic_json, capture_rng, clone_tree, coverage_errors, event_key, fingerprint,
                           restore_rng, rng_identity)

from flash_control import flash_observer, validate_calls, SCOPE
from parameter_hooks import HOOK_SEMANTICS, ParameterHookRegistry
from upstream_capture import UpstreamCapture
from local_policy import (compact_record, kernel_policy, observe_statistics, statistics_metadata, verify_loaded)

GEOMETRY = {
    8192: {
        'num_latent_t': 16,
        'num_height': 256,
        'num_width': 512,
        'num_frames': 61
    },
    31200: {
        'num_latent_t': 20,
        'num_height': 480,
        'num_width': 832,
        'num_frames': 77
    }
}
EXPECTED_INITIAL_WEIGHTS = '72fb71f0b57c3badbdf4d3466738fcfe21ba4d78788270862923b5205318bc31'


def parse_args(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--original-first-step', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--variant', choices=('dense', 'Z'), required=True)
    parser.add_argument('--repetitions', type=int, choices=(1, 2), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tokens', type=int, choices=tuple(GEOMETRY), default=8192)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--production-determinism', action='store_true')
    parser.add_argument('--capture-upstream', action='store_true')
    parser.add_argument('--capture-layer', type=int, action='append', default=[])
    parser.add_argument('--capture-parameter', action='append', default=[])
    args = parser.parse_args(argv)
    if args.capture_upstream and 29 not in args.capture_layer:
        args.capture_layer.append(29)
    if any(layer < 0 or layer >= 30 for layer in args.capture_layer):
        parser.error('--capture-layer must be in [0, 29]')
    return args


def file_record(path: Any) -> Any:
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def run(args: Any, report: Any) -> Any:
    import numpy as np
    import torch
    import fastvideo
    from fastvideo_kernel.triton_kernels import attn_qat_train as kernel
    from fastvideo.attention.backends import attn_qat_train as backend
    from fastvideo.distributed import init_distributed_environment, initialize_model_parallel
    from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.instantiate import instantiate

    output = args.output_dir.resolve()
    snapshot = args.snapshot.resolve()
    report['loaded_modules'] = verify_loaded((fastvideo, kernel, backend))

    def local(tensor: Any) -> Any:
        return tensor.to_local() if hasattr(tensor, 'to_local') else tensor

    def cpu_tensor(tensor: Any) -> Any:
        return local(tensor.detach()).contiguous().to('cpu', copy=True)

    def raw_bytes(tensor: Any) -> Any:
        return cpu_tensor(tensor).reshape(-1).view(torch.uint8).numpy().tobytes()

    def tensor_hash(tensor: Any) -> Any:
        return hashlib.sha256(raw_bytes(tensor)).hexdigest()

    from tensor_evidence import TensorStore
    evidence = TensorStore(output / 'tensors')

    def tensor_record(tensor: Any, persist: bool = False) -> Any:
        if tensor is None:
            return {'missing': True}
        value = cpu_tensor(tensor)
        floating = value.is_floating_point() or value.is_complex()
        finite = bool(torch.isfinite(value).all()) if floating else True
        record: dict[str, Any] = {
            'shape': list(value.shape),
            'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
            'finite': finite,
            'norm': float(value.float().norm()) if finite else None
        }

        if persist:
            record['evidence'] = evidence.write(value, record['sha256'])
        return record

    def named_hash(named: Any) -> Any:
        sha = hashlib.sha256()
        for name, tensor in named:
            sha.update(name.encode())
            sha.update(raw_bytes(tensor))
        return sha.hexdigest()

    def tree_identity(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {'tensor': tensor_record(value)}
        if isinstance(value, dict):
            return {str(key): tree_identity(item) for key, item in sorted(value.items())}
        if isinstance(value, tuple | list):
            return [tree_identity(item) for item in value]
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return {'type': type(value).__qualname__, 'repr': repr(value)}

    def tensor_leaves(value: Any, prefix: Any = 'value') -> Any:
        if isinstance(value, torch.Tensor):
            return {prefix: value}
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                result.update(tensor_leaves(item, f'{prefix}.{key}'))
            return result
        if isinstance(value, list | tuple):
            result = {}
            for index, item in enumerate(value):
                result.update(tensor_leaves(item, f'{prefix}.{index}'))
            return result
        return {}

    cfg = load_run_config(str(__import__('portable').settings().config))
    cfg.training.loop.max_train_steps = 1
    cfg.training.loop.gradient_accumulation_steps = 1
    cfg.models['student']['init_from'] = str(__import__('portable').settings().model)
    cfg.training.data.data_path = str(__import__('portable').settings().train_data.parent)
    cfg.training.checkpoint.output_dir = str(output)
    cfg.training.data.seed = args.seed
    cfg.training.optimizer.learning_rate = 1e-6
    cfg.training.optimizer.lr_scheduler = 'constant'
    cfg.training.optimizer.lr_warmup_steps = 0
    for name, expected in GEOMETRY[args.tokens].items():
        if getattr(cfg.training.data, name) != expected:
            raise RuntimeError(f'Frozen geometry mismatch: {name}')
    if cfg.training.model.enable_gradient_checkpointing_type != 'full' or cfg.training.dit_precision != 'fp32':
        raise RuntimeError('Probe requires full checkpoint and FP32 master parameters')
    torch.cuda.set_device(0)
    capability = torch.cuda.get_device_capability(0)
    report['hardware'] = {
        'name': torch.cuda.get_device_name(0),
        'capability': list(capability),
        'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory
    }
    if capability != (12, 1):
        raise RuntimeError(f'This diagnostic batch requires the validated GB10 capability (12, 1), got {capability}')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False, warn_only=False)
    report['backend_settings'] = {
        'torch_deterministic': torch.are_deterministic_algorithms_enabled(),
        'torch_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    init_distributed_environment(distributed_init_method='file://' + str(output / 'distributed-rendezvous'))
    initialize_model_parallel()
    student = instantiate(cfg.models['student'], training_config=cfg.training)
    student.set_requires_negative_conditioning(False)
    method = FineTuneMethod(cfg=cfg, role_models={'student': student})
    method.on_train_start()
    transformer = student.transformer
    blocks = transformer.blocks
    if len(blocks) != 30:
        raise RuntimeError(f'Expected 30 Wan blocks, got {len(blocks)}')
    parameters = list(transformer.named_parameters())
    expected_parameters = [name for name, value in parameters if value.requires_grad]
    if len(expected_parameters) != 825:
        raise RuntimeError(f'Expected 825 Wan master parameters, got {len(expected_parameters)}')
    unknown_captures = set(args.capture_parameter) - set(expected_parameters)
    if unknown_captures:
        raise RuntimeError(f'Unknown capture parameters: {sorted(unknown_captures)}')
    initial_parameter_hash = named_hash(parameters)
    initial_buffers = named_hash(transformer.named_buffers())
    if initial_parameter_hash != EXPECTED_INITIAL_WEIGHTS:
        raise RuntimeError(f'Weights differ from R5 performance experiment: {initial_parameter_hash}')
    if {str(value.dtype) for _, value in parameters if value.requires_grad} != {'torch.float32'}:
        raise RuntimeError('Unexpected trainable master dtype')
    report['model'] = {
        'initial_full_parameter_sha256': initial_parameter_hash,
        'initial_buffers_sha256': initial_buffers,
        'trainable_parameter_names': expected_parameters,
        'checkpointing': 'full',
        'preserve_rng_state': False,
        'attention_backend': student.attention_backend_name,
        'master_dtypes': sorted({str(value.dtype)
                                 for _, value in parameters}),
        'optimizer_updates': 0,
        'gradient_clipping_calls': 0
    }
    frozen_batch = clone_tree(next(iter(student.dataloader)), lambda value: isinstance(value, torch.Tensor),
                              lambda value: value.detach().clone())
    batch_identity = tree_identity(frozen_batch)
    frozen_rng = capture_rng(torch, np, method.cuda_generator)
    original_path = args.original_first_step
    original = json.loads(original_path.read_text())
    info = (frozen_batch.get('info_list') or [{}])[0]
    identity = {
        'id': info.get('id') or info.get('file_name'),
        'latent_sha256': tensor_hash(frozen_batch['vae_latent'])[:16]
    }
    paired = original['first_microstep']
    if args.seed == 1000 and (identity != paired['batch']
                              or tensor_hash(frozen_rng['method']) != paired['rng_before_sha256']):
        raise RuntimeError('Probe first batch or method RNG differs from the original failed R5 timing run')
    report['original_run_pairing'] = {
        'source': file_record(original_path),
        'batch': identity,
        'method_rng_before_equal': args.seed == 1000,
        'original_seed_pairing_applicable': args.seed == 1000,
        'initial_parameters_equal': True
    }
    report['batch_identity'] = batch_identity
    report['recipe'] = {
        'seed': args.seed,
        'tokens': args.tokens,
        'grad_accum_steps': 1,
        'learning_rate': 1e-6,
        'warmup_steps': 0,
        'store_mode': 'tma',
        'variant_semantics': 'save forward-consistent dV',
        'checkpointing': 'full',
        'no_precompile': 'not timed; kernels JIT on first invocation; no training RNG consumed'
    }
    original_predict = student.predict_noise
    original_attention = backend._attn_qat_train_attention
    original_attempted = backend._attn_qat_train_import_attempted
    original_error = backend._attn_qat_train_import_error
    state = {'phase': 'idle'}
    hook_handles = []
    active: dict[str, Any] = {}
    report['cross_attention_backends'] = []

    def phase() -> Any:
        return state['phase']

    parameter_registry = ParameterHookRegistry(transformer.named_parameters(remove_duplicate=False), phase,
                                               lambda: active['ledger'])
    if parameter_registry.names != expected_parameters:
        raise RuntimeError('Canonical master parameter mapping disagrees with named_parameters')
    report['parameter_registry'] = parameter_registry.declaration()

    def record_prediction(*values: Any, **kwargs: Any) -> Any:
        result = original_predict(*values, **kwargs)
        active['rep']['prediction'] = {
            name: tensor_record(value, persist=True)
            for name, value in tensor_leaves(result, 'prediction').items()
        }
        if not active['rep']['prediction']:
            raise RuntimeError('Model returned no prediction tensor')
        return result

    def add_gradient_hook(tensor: Any,
                          name: Any,
                          module: Any,
                          call_id: Any,
                          slot: Any = None,
                          capture_name: Any = None) -> Any:
        if not tensor.requires_grad:
            raise RuntimeError(f'Required gradient tensor has requires_grad=False: {module}:{name}')
        rep = active['rep']
        ledger = active['ledger']
        key = ['backward', name, module, call_id]
        rep['coverage']['expected_event_keys'].append(key)

        def hook(grad: Any) -> Any:
            ledger.add('backward', name, module, call_id, {'gradient': tensor_record(grad, persist=True)})
            if name == 'block_output_grad':
                rep['coverage']['block_backward_calls'][module] += 1
            if slot is not None:
                slot[capture_name] = cpu_tensor(grad)
            return None

        active['tensor_handles'].append(tensor.register_hook(hook))

    upstream = UpstreamCapture(phase, active, tensor_record, cpu_tensor,
                               tensor_leaves) if args.capture_upstream else None

    def block_pre(name: Any) -> Any:

        def pre(module: Any, values: Any, kwargs: Any) -> Any:
            current_phase = 'forward' if phase() == 'forward' else 'recompute'
            ledger = active['ledger']
            if current_phase == 'recompute':
                active['recompute_block'] = name
            call_id = ledger.next_call(current_phase, name)
            active['stack'].append((name, current_phase, call_id))
            tensors = tensor_leaves((values, kwargs), 'input')
            ledger.add(current_phase, 'block_input', name, call_id, {
                key: tensor_record(value)
                for key, value in tensors.items()
            })
            if current_phase == 'forward':
                active['rep']['coverage']['block_forward_calls'][name] += 1
                active['rep']['coverage']['expected_event_keys'].append([current_phase, 'block_input', name, call_id])
                # Record the first hidden-state input gradient; auxiliary inputs
                # can legitimately have requires_grad=False and remain hashes only.
                first = next(iter(tensors.values()), None)
                if first is None:
                    raise RuntimeError(f'No input tensor for {name}')
                add_gradient_hook(first, 'block_input_grad', name, call_id)

        return pre

    def block_post(name: Any) -> Any:

        def post(module: Any, values: Any, kwargs: Any, result: Any) -> Any:
            if not active['stack'] or active['stack'][-1][0] != name:
                raise RuntimeError(f'Block hook stack mismatch: {name}')
            _, current_phase, call_id = active['stack'].pop()
            if result is None and current_phase == 'recompute':
                active['ledger'].add(current_phase, 'checkpoint_early_stop', name, call_id, marker=True)
                return
            tensors = tensor_leaves(result, 'output')
            active['ledger'].add(current_phase, 'block_output', name, call_id, {
                key: tensor_record(value)
                for key, value in tensors.items()
            })
            if current_phase == 'forward':
                active['rep']['coverage']['expected_event_keys'].append([current_phase, 'block_output', name, call_id])
                if len(tensors) != 1:
                    raise RuntimeError(f'Expected one block output tensor: {name}: {list(tensors)}')
                add_gradient_hook(next(iter(tensors.values())), 'block_output_grad', name, call_id)

        return post

    def observe_counts(metadata: Any) -> Any:
        call = active.get('qat_pending')
        if call is None:
            raise RuntimeError('Statistics observer called outside recorded QAT invocation')
        counts = metadata['counts'].detach().cpu()
        active['count_observations'][call] = {
            'event_count_min':
            int(counts.min()),
            'event_count_max':
            int(counts.max()),
            'event_count_mean':
            float(counts.float().mean()),
            'event_count_histogram':
            torch.bincount(counts.reshape(-1).long()).tolist(),
            'overflow_rows':
            int(metadata['overflow'].sum().item()),
            'requested_capacity':
            int(metadata['capacity']),
            'per_head': [
                dict(batch=b,
                     head=h,
                     minimum=int(counts[b, h].min()),
                     maximum=int(counts[b, h].max()),
                     histogram=torch.bincount(counts[b, h].long()).tolist(),
                     overflow_rows=int((counts[b, h] > int(metadata['capacity'])).sum()))
                for b in range(counts.shape[0]) for h in range(counts.shape[1])
            ],
        }

    def attention(*values: Any, **kwargs: Any) -> Any:
        if not active['stack']:
            raise RuntimeError('QAT invoked outside a known Wan block')
        name, current_phase, block_call = active['stack'][-1]
        qat_name = name + '.qat'
        call_id = active['ledger'].next_call(current_phase, qat_name)
        q, k, v = values[:3]
        if q.dtype != torch.bfloat16 or q.shape[1:] != (12, args.tokens, 128):
            raise RuntimeError(f'Unexpected QAT input shape/dtype: {q.shape}/{q.dtype}')
        active['ledger'].add(current_phase, 'qat_input', qat_name, call_id, {
            key: tensor_record(value)
            for key, value in zip(('q', 'k', 'v'), (q, k, v), strict=False)
        })
        active['qat_pending'] = call_id
        try:
            result = kernel.attention(*values, **kwargs)
        finally:
            active['qat_pending'] = None
        active['ledger'].add(current_phase, 'qat_output', qat_name, call_id, {'out': tensor_record(result)})
        ctx = result.grad_fn
        # Do not access saved_tensors: unpacking them invokes checkpoint hooks.
        if ctx is None or not hasattr(ctx, 'dv_stats_format'):
            raise RuntimeError('Missing R5 QAT context metadata')
        metadata = {
            'phase': current_phase,
            'module': qat_name,
            'call_id': call_id,
            **statistics_metadata(ctx),
            **active['count_observations'].pop(call_id, {})
        }
        active['rep']['statistics'].append(metadata)
        if not ctx.forward_consistent_dv or ctx.dv_stats_mode != 'save':
            raise RuntimeError(f'QAT did not use forward-consistent save statistics: {metadata}')
        if current_phase == 'recompute' and int(name.split('.')[-1]) in args.capture_layer:
            capture_path = output / f'rep{active["rep"]["index"]}-layer{int(name.split(".")[-1]):02d}-recompute.pt'
            if capture_path.exists():
                raise RuntimeError('Duplicate checkpoint QAT capture')
            torch.save(
                {
                    key: cpu_tensor(value)
                    for key, value in zip(('q', 'k', 'v', 'out'), (q, k, v, result), strict=True)
                }, capture_path)
            active['rep'].setdefault('recompute_captures', []).append(
                dict(layer=int(name.split('.')[-1]), phase=current_phase, **file_record(capture_path)))
        if current_phase == 'forward':
            for kind in ('qat_input', 'qat_output'):
                active['rep']['coverage']['expected_event_keys'].append([current_phase, kind, qat_name, call_id])
            layer = int(name.split('.')[-1])
            slot = None
            if layer in args.capture_layer:
                slot = {
                    key: cpu_tensor(value)
                    for key, value in zip(('q', 'k', 'v', 'out'), (q, k, v, result), strict=False)
                }
                active['captures'][call_id] = {
                    'layer': layer,
                    'phase': current_phase,
                    'tensor_metadata': {
                        key: {
                            'shape': list(value.shape),
                            'dtype': str(value.dtype),
                            'stride': list(value.stride())
                        }
                        for key, value in zip(('q', 'k', 'v', 'out'), (q, k, v, result), strict=True)
                    },
                    'slot': slot,
                    'module': qat_name,
                    'call_id': call_id
                }
            for tensor, grad_name in ((q, 'dq'), (k, 'dk'), (v, 'dv'), (result, 'dout')):
                add_gradient_hook(tensor, 'qat_' + grad_name, qat_name, call_id, slot, grad_name)
        return result

    try:
        student.predict_noise = record_prediction
        backend._attn_qat_train_attention = attention
        backend._attn_qat_train_import_attempted = True
        backend._attn_qat_train_import_error = None
        for index, wrapper in enumerate(blocks):
            module = wrapper
            while hasattr(module, '_checkpoint_wrapped_module'):
                module = module._checkpoint_wrapped_module
            name = f'blocks.{index}'
            cross = module.attn2.attn
            implementation = cross.attn_impl
            report['cross_attention_backends'].append({
                'module':
                name + '.attn2.attn',
                'backend':
                str(cross.backend),
                'implementation':
                type(implementation).__module__ + '.' + type(implementation).__qualname__,
                'implementation_source':
                file_record(inspect.getfile(type(implementation))),
            })
            hook_handles.append(module.register_forward_pre_hook(block_pre(name), with_kwargs=True))
            hook_handles.append(module.register_forward_hook(block_post(name), with_kwargs=True, always_call=True))
        if upstream is not None:
            block = blocks[29]
            while hasattr(block, '_checkpoint_wrapped_module'):
                block = block._checkpoint_wrapped_module
            upstream.mount(block)
            report['upstream_capture_source'] = file_record(inspect.getfile(UpstreamCapture))
        parameter_registry.mount()
        with flash_observer(report['flash_calls'], phase,
                            upstream.flash_call if upstream is not None else None) as flash_modules:
            report['flash_modules'] = flash_modules
            for repetition in range(args.repetitions):
                method.optimizers_zero_grad(0)
                batch = clone_tree(frozen_batch, lambda value: isinstance(value, torch.Tensor),
                                   lambda value: value.detach().clone())
                restore_rng(frozen_rng, torch, np, method.cuda_generator)
                before_rng = capture_rng(torch, np, method.cuda_generator)
                before_parameters = named_hash(parameters)
                before_buffers = named_hash(transformer.named_buffers())
                if before_parameters != initial_parameter_hash or before_buffers != initial_buffers:
                    raise RuntimeError('Parameter or buffer mutation between repetitions')
                rep: dict[str, Any] = {
                    'index': repetition,
                    'events': [],
                    'statistics': [],
                    'captures': [],
                    'parameter_gradients': {},
                    'coverage': {
                        'expected_parameter_gradients':
                        expected_parameters,
                        'expected_block_modules': [f'blocks.{index}' for index in range(30)],
                        'expected_event_keys':
                        [['backward', 'parameter_gradient_arrival', name, name] for name in expected_parameters],
                        'block_forward_calls': {
                            f'blocks.{index}': 0
                            for index in range(30)
                        },
                        'block_backward_calls': {
                            f'blocks.{index}': 0
                            for index in range(30)
                        }
                    },
                    'input_state': {
                        'batch_sha256': fingerprint(tree_identity(batch)),
                        'parameters_sha256': before_parameters,
                        'buffers_sha256': before_buffers,
                        'rng_before': rng_identity(before_rng, tensor_hash)
                    }
                }
                report['repetitions'].append(rep)
                ledger = EventLedger()
                rep['events'] = ledger.events
                active.clear()
                active.update(rep=rep, ledger=ledger, stack=[], captures={}, count_observations={}, tensor_handles=[])
                if upstream is not None:
                    upstream.begin()
                parameter_registry.begin()
                flash_start = len(report['flash_calls'])
                started = time.monotonic()
                try:
                    with kernel_policy(kernel, args.variant), \
                            observe_statistics(kernel, observe_counts):
                        state['phase'] = 'forward'
                        losses, outputs, _ = method.single_train_step(batch, 1)
                        rep['loss'] = tensor_record(losses['total_loss'], persist=True)
                        rep['loss']['value'] = float(losses['total_loss'].detach().item())
                        backward_context = outputs.get('_fv_backward')
                        rep['timesteps'] = tensor_record(backward_context[0]) if backward_context else {'missing': True}
                        if backward_context is None or (args.seed == 1000
                                                        and backward_context[0].detach().float().cpu().tolist()
                                                        != paired['timesteps']):
                            raise RuntimeError('Probe timestep differs from the original failed R5 timing run')
                        state['phase'] = 'backward'
                        method.backward(losses, outputs, grad_accum_rounds=1)
                        torch.cuda.synchronize()
                    rep['flash_control'] = validate_calls(report['flash_calls'][flash_start:])
                    rep['parameter_hooks'] = parameter_registry.finish(
                        transformer.named_parameters(remove_duplicate=False))
                    state['phase'] = 'idle'
                    rep['elapsed_seconds_instrumented_not_performance'] = time.monotonic() - started
                    rep['parameter_gradients'] = {
                        name: tensor_record(parameter.grad, persist=True)
                        for name, parameter in parameters if parameter.requires_grad
                    }
                    rep['rng_after'] = rng_identity(capture_rng(torch, np, method.cuda_generator), tensor_hash)
                    rep['parameters_after_sha256'] = named_hash(parameters)
                    rep['buffers_after_sha256'] = named_hash(transformer.named_buffers())
                    rep['frozen_batch_unchanged'] = fingerprint(
                        tree_identity(frozen_batch)) == fingerprint(batch_identity)
                    rep['batch_after_sha256'] = fingerprint(tree_identity(batch))
                    for call_id, capture in active['captures'].items():
                        required = {'q', 'k', 'v', 'out', 'dout', 'dq', 'dk', 'dv'}
                        if set(capture['slot']) != required:
                            raise RuntimeError(f'Incomplete capture {call_id}: {set(capture["slot"])}')
                        path = output / f'rep{repetition}-layer{capture["layer"]:02d}.pt'
                        torch.save(capture['slot'], path)
                        rep['captures'].append({
                            key: value
                            for key, value in capture.items() if key != 'slot'
                        }
                                               | file_record(path))
                    if upstream is not None:
                        path = output / f'rep{repetition}-upstream29.pt'
                        rep['upstream_coverage'] = upstream.finish(path, torch)
                        rep['upstream_capture'] = file_record(path)
                    if args.capture_parameter:
                        path = output / f'rep{repetition}-parameter-gradients.pt'
                        torch.save(
                            {
                                name: cpu_tensor(parameter.grad)
                                for name, parameter in parameters if name in args.capture_parameter
                            }, path)
                        rep['parameter_capture'] = file_record(path)
                    missing = [name for name, value in rep['parameter_gradients'].items() if value.get('missing')]
                    rep['coverage']['missing_parameter_gradients'] = missing
                    rep['coverage']['observed_parameter_gradients'] = [
                        name for name in expected_parameters if name not in missing
                    ]
                    errors = coverage_errors(rep['events'], rep['coverage']['expected_event_keys'], expected_parameters,
                                             rep['parameter_gradients'])
                    errors.extend(rep['parameter_hooks']['errors'])
                    if upstream is not None:
                        errors.extend(rep['upstream_coverage']['errors'])
                    for name in rep['coverage']['expected_block_modules']:
                        for key in ('block_forward_calls', 'block_backward_calls'):
                            if rep['coverage'][key][name] != 1:
                                errors.append(f'{name} {key} expected 1, got {rep["coverage"][key][name]}')
                        recomputes = ledger.calls[('recompute', name)]
                        if recomputes != 1:
                            errors.append(f'{name} checkpoint recomputes expected 1, got {recomputes}')
                    if len([entry for entry in rep['statistics'] if entry['phase'] == 'forward']) != 30:
                        errors.append('Missing QAT forward statistics metadata')
                    if active['stack']:
                        errors.append('Unbalanced block hook stack after backward')
                    if rep['parameters_after_sha256'] != before_parameters or rep[
                            'buffers_after_sha256'] != before_buffers:
                        errors.append('Model parameters/buffers changed during backward-only probe')
                    if not rep['frozen_batch_unchanged'] or rep['batch_after_sha256'] != rep['input_state'][
                            'batch_sha256']:
                        errors.append('Frozen batch or per-repetition batch mutated')
                    if not rep['loss']['finite'] or not all(record['finite'] for record in rep['prediction'].values()):
                        errors.append('Nonfinite loss or prediction')
                    rep['coverage']['errors'] = errors
                    rep['coverage']['passed'] = not errors
                    # Mandatory first-forward/gradient callbacks are validated
                    # independently above. Record the entire observed sequence,
                    # including legitimate checkpoint early-stop markers, so
                    # cross-run comparison cannot drop unexpected extra events.
                    rep['coverage']['required_event_keys'] = rep['coverage']['expected_event_keys']
                    rep['coverage']['expected_event_keys'] = [event_key(event) for event in rep['events']]
                    rep['compact_path_exercised'] = any(compact_record(entry) for entry in rep['statistics'])
                    atomic_json(output / 'report.json', report)
                    from compare_probes_v2 import repetition_errors
                    errors.extend(repetition_errors(rep, f'repetition {repetition}'))
                    if errors:
                        raise RuntimeError('Probe coverage failure: ' + '; '.join(errors[:8]))
                    del losses, outputs, batch
                finally:
                    if upstream is not None:
                        upstream.clear()
                    parameter_registry.abort()
                    state['phase'] = 'idle'
                    for handle in active.get('tensor_handles', []):
                        handle.remove()
                    active.get('captures', {}).clear()
                    method.optimizers_zero_grad(0)
            if args.production_determinism and not report['flash_calls']:
                raise RuntimeError(
                    'Flash deterministic control was requested, but no instrumented backward call occurred')
    finally:
        if upstream is not None:
            upstream.close()
        parameter_registry.close()
        for handle in reversed(hook_handles):
            handle.remove()
        student.predict_noise = original_predict
        backend._attn_qat_train_attention = original_attention
        backend._attn_qat_train_import_attempted = original_attempted
        backend._attn_qat_train_import_error = original_error
        report['loaded_modules'].update({
            name: file_record(module.__file__)
            for name, module in list(sys.modules.items())
            if name.startswith(('fastvideo.', 'fastvideo_kernel.', 'gb10_qat.')) and getattr(module, '__file__', None)
            and Path(str(module.__file__)).is_file()
        })
        report['loaded_modules'].update(verify_loaded())
        report['runtime_versions'] = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': np.__version__}
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main(argv: Any = None) -> Any:
    args = parse_args(argv)
    if args.production_determinism or args.strict or args.seed != 1000:
        raise RuntimeError('This batch requires seed=1000 and ordinary production FA2 mode')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / 'report.json'
    if report_path.exists():
        raise RuntimeError(f'Refuse to overwrite an existing probe: {report_path}')
    manifest = Path(__file__).resolve().parent.parent / 'model-manifest.json'
    if not manifest.exists():
        raise RuntimeError('Missing new model-stage source manifest')
    config = {
        'tokens': args.tokens,
        'seed': args.seed,
        'strict': args.strict,
        'flash_deterministic': args.production_determinism,
        'capture_upstream': args.capture_upstream,
        'grad_accum_steps': 1,
        'store_mode': 'tma',
        'checkpointing': 'full',
        'capture_layers': sorted(set(args.capture_layer)),
        'capture_parameters': sorted(set(args.capture_parameter)),
        'config_sha256': file_record(__import__('portable').settings().config)['sha256']
    }
    report = {
        'acceptance_scope': SCOPE,
        'default_model_acceptance': False,
        'observer_mode': 'record-only',
        'schema_version': 2,
        'parameter_hook_semantics': HOOK_SEMANTICS,
        'status': 'running',
        'variant': args.variant,
        'tokens': args.tokens,
        'seed': args.seed,
        'process_id': os.getpid(),
        'process_nonce': uuid.uuid4().hex,
        'source_fingerprint': file_record(manifest)['sha256'],
        'config_fingerprint': fingerprint(config),
        'arguments': {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'configuration': config,
        'repetitions': [],
        'flash_calls': [],
        'artifact_sha256': {},
        'scope': 'Backward-only full-model repeatability; instrumented times are not performance data'
    }
    atomic_json(report_path, report)
    try:
        run(args, report)
        for repetition in report['repetitions']:
            for capture in repetition.get('captures', []):
                report['artifact_sha256'][capture['path']] = capture['sha256']
            capture = repetition.get('upstream_capture')
            if capture:
                report['artifact_sha256'][capture['path']] = capture['sha256']
            capture = repetition.get('parameter_capture')
            if capture:
                report['artifact_sha256'][capture['path']] = capture['sha256']
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        atomic_json(report_path, report)
        raise
    atomic_json(report_path, report)
    print(json.dumps({'status': report['status'], 'report': str(report_path), 'variant': args.variant}, sort_keys=True))


if __name__ == '__main__':
    main()
