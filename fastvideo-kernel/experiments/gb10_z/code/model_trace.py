"""Private full-model observer derived from the frozen V6 coverage implementation."""
from typing import Any
from types import SimpleNamespace
from local_policy import compact_record, statistics_metadata
from probe_helpers import EventLedger, coverage_errors, event_key


class ModelTrace:

    def __init__(self, transformer: Any, tokens: Any, record: Any, cpu_tensor: Any, tensor_leaves: Any, phase: Any,
                 original_attention: Any) -> None:
        args = SimpleNamespace(tokens=tokens, capture_layer=[])
        active: dict[str, Any] = {}
        self.active = active
        hook_handles: list[Any] = []
        self.handles = hook_handles
        tensor_record = record

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
                ledger.add('backward', name, module, call_id, {'gradient': tensor_record(grad)})
                if name == 'block_output_grad':
                    rep['coverage']['block_backward_calls'][module] += 1
                if slot is not None:
                    slot[capture_name] = cpu_tensor(grad)
                return None

            active['tensor_handles'].append(tensor.register_hook(hook))

        def block_pre(name: Any) -> Any:

            def pre(module: Any, values: Any, kwargs: Any) -> Any:
                if phase() == 'evaluation':
                    return
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
                    active['rep']['coverage']['expected_event_keys'].append(
                        [current_phase, 'block_input', name, call_id])
                    # Record the first hidden-state input gradient; auxiliary inputs
                    # can legitimately have requires_grad=False and remain hashes only.
                    first = next(iter(tensors.values()), None)
                    if first is None:
                        raise RuntimeError(f'No input tensor for {name}')
                    add_gradient_hook(first, 'block_input_grad', name, call_id)

            return pre

        def block_post(name: Any) -> Any:

            def post(module: Any, values: Any, kwargs: Any, result: Any) -> Any:
                if phase() == 'evaluation':
                    return
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
                    active['rep']['coverage']['expected_event_keys'].append(
                        [current_phase, 'block_output', name, call_id])
                    if len(tensors) != 1:
                        raise RuntimeError(f'Expected one block output tensor: {name}: {list(tensors)}')
                    add_gradient_hook(next(iter(tensors.values())), 'block_output_grad', name, call_id)

            return post

        def observe_counts(metadata: Any) -> Any:
            import torch
            call = active.get('qat_pending')
            if call is None:
                return
            counts = metadata['counts'].detach().cpu()
            active['count_observations'][call] = {
                'event_count_min': int(counts.min()),
                'event_count_max': int(counts.max()),
                'event_count_mean': float(counts.float().mean()),
                'event_count_histogram': torch.bincount(counts.reshape(-1).long()).tolist(),
                'overflow_rows': int(metadata['overflow'].sum().item()),
                'requested_capacity': int(metadata['capacity']),
            }

        def attention(*values: Any, **kwargs: Any) -> Any:
            if phase() == 'evaluation':
                return original_attention(*values, **kwargs)
            if not active['stack']:
                raise RuntimeError('QAT invoked outside a known Wan block')
            name, current_phase, block_call = active['stack'][-1]
            qat_name = name + '.qat'
            call_id = active['ledger'].next_call(current_phase, qat_name)
            q, k, v = values[:3]
            if str(q.dtype) != 'torch.bfloat16' or q.shape[1:] != (12, args.tokens, 128):
                raise RuntimeError(f'Unexpected QAT input shape/dtype: {q.shape}/{q.dtype}')
            active['ledger'].add(current_phase, 'qat_input', qat_name, call_id, {
                key: tensor_record(value)
                for key, value in zip(('q', 'k', 'v'), (q, k, v), strict=False)
            })
            active['qat_pending'] = call_id
            try:
                result = original_attention(*values, **kwargs)
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
                        'slot': slot,
                        'module': qat_name,
                        'call_id': call_id
                    }
                for tensor, grad_name in ((q, 'dq'), (k, 'dk'), (v, 'dv'), (result, 'dout')):
                    add_gradient_hook(tensor, 'qat_' + grad_name, qat_name, call_id, slot, grad_name)
            return result

        self.observe_counts = observe_counts
        self.attention = attention
        self.block_pre = block_pre
        self.block_post = block_post
        try:
            for i, module in enumerate(transformer.blocks):
                while hasattr(module, '_checkpoint_wrapped_module'):
                    module = module._checkpoint_wrapped_module
                name = f'blocks.{i}'
                hook_handles.append(module.register_forward_pre_hook(block_pre(name), with_kwargs=True))
                hook_handles.append(module.register_forward_hook(block_post(name), with_kwargs=True, always_call=True))
            if len(transformer.blocks) != 30:
                raise RuntimeError('Expected thirty Wan blocks')

        except BaseException:
            self.close()
            raise

    def begin(self, rep: Any, names: Any) -> Any:
        self.clear()
        rep.update(events=[],
                   statistics=[],
                   captures=[],
                   parameter_gradients={},
                   coverage={
                       'expected_parameter_gradients': names,
                       'expected_block_modules': [f'blocks.{i}' for i in range(30)],
                       'expected_event_keys':
                       [['backward', 'parameter_gradient_arrival', name, name] for name in names],
                       'block_forward_calls': {
                           f'blocks.{i}': 0
                           for i in range(30)
                       },
                       'block_backward_calls': {
                           f'blocks.{i}': 0
                           for i in range(30)
                       }
                   })
        self.active.update(rep=rep,
                           ledger=EventLedger(),
                           stack=[],
                           captures={},
                           count_observations={},
                           tensor_handles=[])
        rep['events'] = self.active['ledger'].events

    def finish(self, rep: Any, names: Any) -> Any:
        coverage = rep['coverage']
        ledger = self.active['ledger']
        errors = coverage_errors(rep['events'], coverage['expected_event_keys'], names, rep['parameter_gradients'])
        errors.extend(rep['parameter_hooks']['errors'])
        for name in coverage['expected_block_modules']:
            if any(coverage[k][name] != 1
                   for k in ('block_forward_calls', 'block_backward_calls')) or ledger.calls['recompute', name] != 1:
                errors.append(f'Missing forward/gradient/checkpoint coverage: {name}')
        if len([x for x in rep['statistics'] if x['phase'] == 'forward']) != 30 or self.active['stack']:
            errors.append('Incomplete statistics or block stack')
        missing = [name for name in names if rep['parameter_gradients'][name].get('missing')]
        coverage.update(missing_parameter_gradients=missing,
                        observed_parameter_gradients=[n for n in names if n not in missing],
                        errors=errors,
                        passed=not errors)
        coverage['required_event_keys'] = coverage['expected_event_keys']
        coverage['expected_event_keys'] = [event_key(x) for x in rep['events']]
        rep['compact_path_exercised'] = any(compact_record(x) for x in rep['statistics'])
        if errors:
            raise RuntimeError('Trace coverage failed: ' + repr(errors[:8]))

    def clear(self) -> Any:
        for h in self.active.get('tensor_handles', []):
            h.remove()
        self.active.clear()

    def close(self) -> Any:
        self.clear()
        for h in reversed(self.handles):
            h.remove()
        self.handles.clear()
