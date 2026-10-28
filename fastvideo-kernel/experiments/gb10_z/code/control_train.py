"""Audit real Wan windows and updates under the private FA2 control."""
from typing import Any
import argparse
import math
import contextlib
import hashlib
import json
from pathlib import Path
import random
import runpy
import sys
import time

from compare_probes_v2 import compare_repetitions, repetition_errors
from flash_control import SCOPE, flash_observer, validate_calls
from model_trace import ModelTrace
import calibration_runtime
from local_policy import compact_record, kernel_policy, observe_statistics, verify_loaded, require_report_domain
from parameter_hooks import HOOK_SEMANTICS, ParameterHookRegistry
from probe_helpers import (atomic_json, capture_rng, fingerprint, rng_identity)

EXPECTED_INITIAL_WEIGHTS = '72fb71f0b57c3badbdf4d3466738fcfe21ba4d78788270862923b5205318bc31'


def require_recipe(args: Any) -> Any:
    if args.seed not in (41001, 41002, 41003) or args.factor != 1 or args.samples != 20 or args.tokens != 8192:
        raise ValueError('Invalid frozen seed or accumulation budget')


def require_finite_tree(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get('finite') is False:
            raise RuntimeError('Nonfinite optimizer/state tensor')
        for child in value.values():
            require_finite_tree(child)
    elif isinstance(value, list | tuple):
        for child in value:
            require_finite_tree(child)


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--tokens', type=int, choices=(8192, 31200), required=True)
    parser.add_argument('--policy', choices=('dense', 'Z'), required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--factor', type=int, choices=(1, 4), required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--repeat-id', choices=('A', 'B', 'Z'), required=True)
    args = parser.parse_args()
    require_recipe(args)
    if args.reference is not None or (args.policy == 'Z') != (args.repeat_id == 'Z'):
        raise ValueError('Calibration uses paired descriptive analysis, not strict trajectory references')
    from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
    from fastvideo.attention.backends import attn_qat_train as backend
    from fastvideo.train.utils import optimizer as optimizer_module
    from fastvideo_kernel.triton_kernels import attn_qat_train as kernel
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False, warn_only=False)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.run_dir / 'audit.json'
    if report_path.exists():
        raise RuntimeError('Refuse to overwrite an audit attempt')
    from portable import settings
    manifest = settings().source_manifest
    paired_config = dict(seed=args.seed,
                         tokens=args.tokens,
                         grad_accum_steps=args.factor,
                         samples=args.samples,
                         store_mode='tma',
                         checkpointing='full',
                         lr=1e-6,
                         warmup=0,
                         clip=1.0,
                         strict=False,
                         flash_deterministic=False,
                         acceptance_scope=SCOPE)
    loaded_modules = verify_loaded()
    report: dict[str, Any] = dict(loaded_modules=loaded_modules,
                                  schema_version=2,
                                  semantics=HOOK_SEMANTICS,
                                  acceptance_scope=SCOPE,
                                  default_model_acceptance=False,
                                  observer_mode='record-only',
                                  status='running',
                                  accepted=False,
                                  variant=args.policy,
                                  repeat_id=args.repeat_id,
                                  configuration=paired_config,
                                  config_fingerprint=fingerprint(paired_config),
                                  source_fingerprint=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                  microbatches=0,
                                  optimizer_updates=0,
                                  compact_calls=0,
                                  fallback_calls=0,
                                  artifacts={})
    micro_path, update_path = (args.run_dir / name for name in ('audit-microsteps.jsonl', 'audit-updates.jsonl'))
    micro_stream = micro_path.open('x')
    update_stream = update_path.open('x')
    reference_micro = reference_updates = None
    if args.reference:
        reference = json.loads((args.reference / 'audit.json').read_text())
        require_report_domain(reference)
        if (reference.get('accepted') is not True or reference.get('variant') != 'dense' or any(
                reference.get(k) != report[k]
                for k in ('acceptance_scope', 'source_fingerprint', 'config_fingerprint'))):
            raise RuntimeError('Reference audit/configuration mismatch')
        reference_micro = (args.reference / micro_path.name).open()
        reference_updates = (args.reference / update_path.name).open()

    state: dict[str, Any] = dict(phase='idle',
                                 method=None,
                                 trace=None,
                                 registry=None,
                                 rep=None,
                                 flash=[],
                                 window_micros=0,
                                 clip_calls=0,
                                 handles=[],
                                 original_predict=None,
                                 original_attention=None)
    originals = {
        name: getattr(FineTuneMethod, name)
        for name in ('on_train_start', 'single_train_step', 'backward', 'optimizers_schedulers_step')
    }
    calibration_runtime.STATE = state
    calibration_runtime.OUTPUT = args.run_dir
    original_clip = optimizer_module.clip_grad_norm_if_needed

    def local(value: Any) -> Any:
        return value.to_local() if hasattr(value, 'to_local') else value

    def cpu_tensor(value: Any) -> Any:
        return local(value.detach()).contiguous().to('cpu', copy=True)

    def tensor_record(value: Any) -> Any:
        if value is None:
            return {'missing': True}
        started_hash = time.monotonic()
        value = cpu_tensor(value)
        finite = not value.is_floating_point() or bool(torch.isfinite(value).all())
        record = dict(shape=list(value.shape),
                      dtype=str(value.dtype),
                      finite=finite,
                      sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                      norm=float(value.float().norm()) if finite else None)
        report['hash_seconds'] = report.get('hash_seconds', 0.0) + time.monotonic() - started_hash
        return record

    def tensor_hash(value: Any) -> Any:
        return tensor_record(value)['sha256']

    def tensor_leaves(value: Any, prefix: Any = 'value') -> Any:
        if isinstance(value, torch.Tensor):
            return {prefix: value}
        if isinstance(value, dict):
            return {k: v for key, child in value.items() for k, v in tensor_leaves(child, f'{prefix}.{key}').items()}
        if isinstance(value, tuple | list):
            return {k: v for i, child in enumerate(value) for k, v in tensor_leaves(child, f'{prefix}.{i}').items()}
        return {}

    def identity(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {'tensor': tensor_record(value)}
        if isinstance(value, dict):
            return {str(k): identity(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [identity(v) for v in value]
        if value is None or isinstance(value, str | bool | int | float):
            return value
        raise TypeError(f'Unrecognized state type: {type(value)}')

    def named_hash(named: Any) -> Any:
        started_hash = time.monotonic()
        digest = hashlib.sha256()
        for name, value in named:
            digest.update(name.encode())
            value = cpu_tensor(value)
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        report['hash_seconds'] = report.get('hash_seconds', 0.0) + time.monotonic() - started_hash
        return digest.hexdigest()

    def parameters() -> Any:
        return state['method'].student.transformer.named_parameters()

    def parameter_records() -> Any:
        return {name: tensor_record(value) for name, value in parameters() if value.requires_grad}

    def buffers_hash() -> Any:
        return named_hash(state['method'].student.transformer.named_buffers())

    def rng() -> Any:
        return rng_identity(capture_rng(torch, np, state['method'].cuda_generator), tensor_hash)

    def append(stream: Any, value: Any) -> Any:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()

    def fail_tensors(reason: Any) -> Any:
        failure = args.run_dir / 'failure'
        failure.mkdir(exist_ok=True)
        atomic_json(failure / 'reason.json', {'reason': reason, 'microstep': report['microbatches'] + 1})
        if state['method'] is not None:
            for index, (name, value) in enumerate(parameters()):
                if value.grad is not None:
                    torch.save({'name': name, 'gradient': cpu_tensor(value.grad)}, failure / f'gradient-{index:04}.pt')
            if state.get('batch') is not None:
                torch.save({k: cpu_tensor(v) for k, v in tensor_leaves(state['batch']).items()}, failure / 'batch.pt')
            torch.save(capture_rng(torch, np, state['method'].cuda_generator), failure / 'rng.pt')

    def started(method: Any, *values: Any, **kwargs: Any) -> Any:
        result = originals['on_train_start'](method, *values, **kwargs)
        if state['method'] is not None:
            raise RuntimeError('Duplicate model initialization')
        state['method'] = method
        method.cuda_generator.manual_seed(args.seed)
        transformer = method.student.transformer
        state['original_predict'] = method.student.predict_noise
        state['original_attention'] = backend._attn_qat_train_attention
        trace = state['trace'] = ModelTrace(transformer, args.tokens, tensor_record, cpu_tensor, tensor_leaves,
                                            lambda: state['phase'], state['original_attention'])
        registry = state['registry'] = ParameterHookRegistry(transformer.named_parameters(remove_duplicate=False),
                                                             lambda: state['phase'], lambda: trace.active['ledger'])
        if len(registry.names) != 825 or any(
                str(entry['parameter'].dtype) != 'torch.float32' for entry in registry.entries):
            raise RuntimeError('Expected 825 FP32 master parameters')
        registry.mount()
        if named_hash(parameters()) != EXPECTED_INITIAL_WEIGHTS:
            raise RuntimeError('Initial weights differ from frozen R5 input')
        report['parameter_registry'] = registry.declaration()
        report['initial_parameters'] = parameter_records()
        report['initial_buffers_sha256'] = buffers_hash()
        report['initial_rng'] = rng()
        report['initial_optimizer'] = identity(method._student_optimizer.state_dict())
        report['initial_scheduler'] = identity(method._student_lr_scheduler.state_dict())
        backend._attn_qat_train_attention = trace.attention

        def predict(*values: Any, **kwargs: Any) -> Any:
            result = state['original_predict'](*values, **kwargs)
            if state['phase'] == 'evaluation':
                return result
            state['rep']['prediction'] = {k: tensor_record(v) for k, v in tensor_leaves(result, 'prediction').items()}
            return result

        method.student.predict_noise = predict
        return result

    def single(method: Any, batch: Any, *values: Any, **kwargs: Any) -> Any:
        if state['phase'] == 'evaluation':
            return originals['single_train_step'](method, batch, *values, **kwargs)
        if state['rep'] is not None:
            raise RuntimeError('Previous microbatch has not completed')
        rep: dict[str, Any] = dict(index=report['microbatches'],
                                   input_state=dict(batch_sha256=fingerprint(identity(batch)),
                                                    parameters_sha256=named_hash(parameters()),
                                                    buffers_sha256=buffers_hash(),
                                                    rng_before=rng()))
        state['rep'] = rep
        state['batch'] = batch
        state['flash_start'] = len(state['flash'])
        state['trace'].begin(rep, state['registry'].names)
        state['registry'].begin()
        state['phase'] = 'forward'
        losses, outputs, other = originals['single_train_step'](method, batch, *values, **kwargs)
        rep['loss'] = tensor_record(losses['total_loss'])
        rep['loss']['value'] = float(losses['total_loss'].item())
        context = outputs.get('_fv_backward')
        rep['timesteps'] = tensor_record(context[0]) if context else {'missing': True}
        return losses, outputs, other

    def backward(method: Any, *values: Any, **kwargs: Any) -> Any:
        if kwargs.get('grad_accum_rounds') != args.factor or state['rep'] is None:
            raise RuntimeError('Unexpected backward accumulation or lifecycle')
        state['phase'] = 'backward'
        try:
            result = originals['backward'](method, *values, **kwargs)
            torch.cuda.synchronize()
            rep = state['rep']
            rep['parameter_hooks'] = state['registry'].finish(
                method.student.transformer.named_parameters(remove_duplicate=False))
            rep['flash_control'] = validate_calls(state['flash'][state['flash_start']:])
            rep['parameter_gradients'] = {
                name: tensor_record(value.grad)
                for name, value in parameters() if value.requires_grad
            }
            rep.update(rng_after=rng(),
                       parameters_after_sha256=named_hash(parameters()),
                       buffers_after_sha256=buffers_hash(),
                       frozen_batch_unchanged=True,
                       batch_after_sha256=fingerprint(identity(state['batch'])))
            if any(rep[k] != rep['input_state'][v]
                   for k, v in [('parameters_after_sha256',
                                 'parameters_sha256'), ('buffers_after_sha256',
                                                        'buffers_sha256'), ('batch_after_sha256', 'batch_sha256')]):
                raise RuntimeError('Input/model state mutated during microbatch')
            state['trace'].finish(rep, state['registry'].names)
            errors = repetition_errors(rep, f'microstep {report["microbatches"] + 1}')
            if errors:
                raise RuntimeError(repr(errors[:8]))
            if reference_micro is not None:
                line = reference_micro.readline()
                if not line:
                    raise RuntimeError('Missing reference microbatch')
                comparison = compare_repetitions(json.loads(line), rep)
                if not comparison['accepted']:
                    atomic_json(args.run_dir / 'first-difference.json', comparison)
                    raise RuntimeError('Strict microbatch comparison failed')
            if report['microbatches'] == 19:
                calibration_runtime.save_named(args.run_dir / 'final-gradients',
                                               ((name, value.grad)
                                                for name, value in parameters() if value.requires_grad), torch)
            append(micro_stream, rep)
            report['microbatches'] += 1
            report['compact_calls'] += sum(compact_record(x) for x in rep['statistics'])
            report['fallback_calls'] += sum(bool(x['overflow_fallback']) for x in rep['statistics'])
            state['window_micros'] += 1
            atomic_json(report_path, report)
            return result
        except BaseException as error:
            fail_tensors(repr(error))
            raise
        finally:
            state['registry'].abort()
            state['trace'].clear()
            state['rep'] = None
            state['phase'] = 'idle'

    def clip(*values: Any, **kwargs: Any) -> Any:
        if state['window_micros'] != args.factor or state['clip_calls']:
            raise RuntimeError('Unexpected window clipping')
        result = original_clip(*values, **kwargs)
        state['clip_calls'] += 1
        norm = float(result)
        if not math.isfinite(norm):
            raise RuntimeError('Nonfinite accumulated norm; optimizer must not update')
        state['clip_evidence'] = dict(gradient_norm_before_clip=norm,
                                      clip_coefficient=min(1.0, 1.0 / (norm + 1e-6)),
                                      clip_calls=1)
        if reference_updates is not None:
            line = reference_updates.readline()
            if not line:
                raise RuntimeError('Missing reference update; optimizer must not update')
            state['expected_update'] = json.loads(line)
            rates = [float(group['lr']) for group in state['method']._student_optimizer.param_groups]
            if (state['expected_update']['clipping'] != state['clip_evidence']
                    or rates != state['expected_update']['learning_rates']):
                fail_tensors('Accumulated norm, clipping or LR differs before optimizer')
                raise RuntimeError('Window comparison failed; optimizer must not update')
        return result

    def update(method: Any, *values: Any, **kwargs: Any) -> Any:
        if state['window_micros'] != args.factor or state['clip_calls'] != 1:
            raise RuntimeError('Incomplete window or clipping before optimizer')
        result = originals['optimizers_schedulers_step'](method, *values, **kwargs)
        torch.cuda.synchronize()
        parameter_ids = {id(e['parameter']): e['name'] for e in state['registry'].entries}
        opt = method._student_optimizer
        optimizer_state = {parameter_ids[id(p)]: identity(v) for p, v in opt.state.items()}
        if set(optimizer_state) != set(parameter_ids.values()):
            raise RuntimeError('Missing optimizer state')
        row = dict(optimizer_step=report['optimizer_updates'] + 1,
                   clipping=state['clip_evidence'],
                   samples_seen=report['microbatches'],
                   parameters=parameter_records(),
                   buffers_sha256=buffers_hash(),
                   rng=rng(),
                   optimizer_state=optimizer_state,
                   scheduler_state=identity(method._student_lr_scheduler.state_dict()),
                   learning_rates=[float(g['lr']) for g in opt.param_groups])
        require_finite_tree(row)
        if any(v['finite'] is not True for v in row['parameters'].values()):
            raise RuntimeError('Nonfinite updated parameter')
        if reference_updates is not None and state.pop('expected_update') != row:
            atomic_json(args.run_dir / 'first-update-difference.json', row)
            fail_tensors('Strict optimizer/parameter update comparison failed')
            raise RuntimeError('Strict update comparison failed')
        if report['optimizer_updates'] == 19:
            calibration_runtime.save_named(args.run_dir / 'final-parameters', parameters(), torch)
            calibration_runtime.save_tree(args.run_dir / 'final-optimizer', {
                parameter_ids[id(p)]: v
                for p, v in opt.state.items()
            }, torch)
            calibration_runtime.save_tree(args.run_dir / 'final-scheduler', method._student_lr_scheduler.state_dict(),
                                          torch)
        append(update_stream, row)
        report['optimizer_updates'] += 1
        state.update(window_micros=0, clip_calls=0, clip_evidence=None)
        atomic_json(report_path, report)
        return result

    FineTuneMethod.on_train_start = started
    FineTuneMethod.single_train_step = single
    FineTuneMethod.backward = backward
    FineTuneMethod.optimizers_schedulers_step = update
    optimizer_module.clip_grad_norm_if_needed = clip
    try:
        with contextlib.ExitStack() as contexts:
            contexts.enter_context(kernel_policy(kernel, args.policy))
            report['flash_modules'] = contexts.enter_context(flash_observer(state['flash'], lambda: state['phase']))
            contexts.enter_context(
                observe_statistics(kernel, lambda m: state['trace'].observe_counts(m) if state['trace'] else None))
            sys.argv = [
                str(args.script), '--variant', 'save', '--tokens',
                str(args.tokens), '--seed',
                str(args.seed), '--grad-accum-steps',
                str(args.factor), '--samples',
                str(args.samples), '--eval-every-samples', '10', '--eval-train', '0', '--eval-heldout', '32',
                '--step-telemetry', '--store-mode', 'tma', '--lr', '1e-6', '--max-grad-norm', '1.0', '--cooldown-temp',
                '50', '--artifact-root',
                str(args.artifact_root), '--run-dir',
                str(args.run_dir)
            ]
            if args.policy != 'dense':
                sys.argv += ['--stats-format', 'packed_slots', '--event-capacity', '64']
            runpy.run_path(str(args.script), run_name='__main__')
        calibration_runtime.validate_evaluations(args.run_dir)
        if report['microbatches'] != args.samples or report['optimizer_updates'] != args.samples // args.factor:
            raise RuntimeError('Incomplete budget')
        if args.policy != 'dense' and not report['compact_calls']:
            raise RuntimeError('Only dense fallback observed; compact path not certified')
        if any(s is not None and s.readline() for s in (reference_micro, reference_updates)):
            raise RuntimeError('Extra reference rows')
        report.update(status='complete', accepted=True, final_parameters=parameter_records())
    except BaseException as error:
        report.update(status='failed', accepted=False, error=repr(error))
        raise
    finally:
        for name, original in originals.items():
            setattr(FineTuneMethod, name, original)
        optimizer_module.clip_grad_norm_if_needed = original_clip
        if state['registry']:
            state['registry'].close()
        if state['trace']:
            state['trace'].close()
        if state['method'] is not None:
            state['method'].student.predict_noise = state['original_predict']
            backend._attn_qat_train_attention = state['original_attention']
        for stream in (micro_stream, update_stream, reference_micro, reference_updates):
            if stream is not None:
                stream.close()
        report['flash_calls'] = state['flash']
        report['loaded_modules'].update(verify_loaded())
        for path in args.run_dir.rglob('*'):
            if path.is_file() and path != report_path:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                report['artifacts'][str(path)] = digest.hexdigest()
        report['artifact_sha256'] = report['artifacts']
        atomic_json(report_path, report)


if __name__ == '__main__':
    main()
