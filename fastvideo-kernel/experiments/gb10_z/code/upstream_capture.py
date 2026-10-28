"""Private tensor-boundary observer. Master parameter callbacks are untouched."""
from collections import Counter
from pathlib import Path
from typing import Any

SUFFIXES: tuple[str, ...] = ('to_out', 'self_attn_residual_norm', 'self_attn_residual_norm.norm', 'attn2', 'attn2.to_q',
                             'attn2.norm_q', 'attn2.to_k', 'attn2.norm_k', 'attn2.to_v', 'attn2.attn', 'attn2.to_out',
                             'cross_attn_residual_norm', 'cross_attn_residual_norm.norm', 'ffn', 'ffn.fc_in', 'ffn.act',
                             'ffn.fc_out', 'mlp_residual')


class UpstreamCapture:

    def __init__(self, phase: Any, active: Any, record: Any, cpu: Any, leaves: Any) -> None:
        self.phase, self.active = phase, active
        self.record, self.cpu, self.leaves = record, cpu, leaves
        self.handles: list[Any] = []
        self.tensor_handles: list[Any] = []
        self.records: list[dict[str, Any]] = []
        self.flash: list[dict[str, Any]] = []
        self.pending: dict[str, list[Any]] = {}
        self.counts: Counter[Any] = Counter()
        self.flash_keys: set[Any] = set()
        self.errors: list[str] = []

    @staticmethod
    def key(values: Any) -> Any:
        return tuple((value.data_ptr(), tuple(value.shape), tuple(value.stride())) for value in values)

    def begin(self) -> None:
        self.clear()
        self.records, self.flash, self.pending = [], [], {}
        self.counts, self.flash_keys, self.errors = Counter(), set(), []

    def mount(self, block: Any) -> None:
        modules = dict(block.named_modules())
        if any(name not in modules for name in SUFFIXES):
            raise RuntimeError('Missing predefined block-29 boundary module')
        try:
            for suffix in SUFFIXES:
                module = modules[suffix]
                name = 'blocks.29' + ('.' + suffix if suffix else '')
                self.handles.append(module.register_forward_pre_hook(self.pre(name), with_kwargs=True))
                self.handles.append(module.register_forward_hook(self.post(name), with_kwargs=True, always_call=True))
        except BaseException:
            self.close()
            raise

    def add(self, phase: str, kind: str, name: str, call_id: str, values: Any) -> None:
        ledger = self.active['ledger']
        ledger.add(phase, kind, name, call_id, {key: self.record(value) for key, value in values.items()})
        if phase == 'forward' or phase == 'backward':
            self.active['rep']['coverage']['expected_event_keys'].append([phase, kind, name, call_id])

    def gradient_hooks(self, values: Any, row: Any, side: str) -> None:
        # Alias slots in a module share one real tensor hook, without double-counting.
        groups: dict[int, list[str]] = {}
        for key, value in values.items():
            if value.requires_grad:
                groups.setdefault(id(value), []).append(key)
        row['expected_' + side + '_gradients'] = [keys for keys in groups.values()]
        row[side + '_gradients'] = {}
        for keys in groups.values():
            value = values[keys[0]]

            def hook(grad: Any, keys: Any = keys) -> None:
                for key in keys:
                    if key in row[side + '_gradients']:
                        self.errors.append('Repeated boundary gradient: ' + row['name'] + ':' + key)
                    row[side + '_gradients'][key] = self.cpu(grad)
                self.add('backward', 'boundary_' + side + '_grad', row['name'], row['call_id'] + '/' + keys[0],
                         {'gradient': grad})

            self.tensor_handles.append(value.register_hook(hook))

    def clone_arguments(self, value: Any) -> Any:
        if hasattr(value, 'shape'):
            return self.cpu(value)
        if isinstance(value, tuple):
            return tuple(self.clone_arguments(item) for item in value)
        if isinstance(value, list):
            return [self.clone_arguments(item) for item in value]
        if isinstance(value, dict):
            return {key: self.clone_arguments(item) for key, item in value.items()}
        if isinstance(value, int | float | str | bool) or value is None:
            return value
        raise TypeError('Unsupported captured argument type: ' + str(type(value)))

    def pre(self, name: str) -> Any:

        def callback(module: Any, args: Any, kwargs: Any) -> None:
            phase = 'forward' if self.phase() == 'forward' else 'recompute'
            index = self.counts[phase, name]
            self.counts[phase, name] += 1
            call_id = f'{name}/{phase}/{index}'
            values = self.leaves((args, kwargs), 'input')
            row: dict[str, Any] = {
                'name': name,
                'phase': phase,
                'call_id': call_id,
                'type': type(module).__module__ + '.' + type(module).__qualname__,
                'inputs': {
                    key: self.cpu(value)
                    for key, value in values.items()
                },
                'input_metadata': {
                    key: {
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'stride': list(value.stride()),
                        'requires_grad': value.requires_grad,
                        'object_id': id(value)
                    }
                    for key, value in values.items()
                },
                'weights': {
                    key: self.cpu(value)
                    for key, value in module.named_parameters()
                },
                'buffers': {
                    key: self.cpu(value)
                    for key, value in module.named_buffers()
                },
                'scalars': {
                    key: value
                    for key, value in vars(module).items() if isinstance(value, int | float | str
                                                                         | bool) or value is None
                },
                'positional_scalars': {
                    str(index): value
                    for index, value in enumerate(args) if isinstance(value, int | float | str
                                                                      | bool) or value is None
                }
            }
            row['argument_tree'] = self.clone_arguments((args, kwargs))
            for key, value in values.items():
                row['input_metadata'][key]['requires_grad'] = value.requires_grad
            if name == 'blocks.29':
                row['block_recipe'] = {
                    'dim': module.hidden_dim,
                    'ffn_dim': module.ffn.fc_in.weight.shape[0],
                    'num_heads': module.num_attention_heads,
                    'qk_norm':
                    'rms_norm_across_heads' if module.norm_q.hidden_size == module.hidden_dim else 'rms_norm',
                    'cross_attn_norm': True,
                    'eps': module.norm1.eps
                }
            self.records.append(row)
            self.pending.setdefault(name, []).append(row)
            self.add(phase, 'boundary_input', name, call_id, values)
            if name == 'blocks.29.attn2.attn':
                self.flash_keys.add(self.key(args[:3]))
            if phase == 'forward':
                self.gradient_hooks(values, row, 'input')

        return callback

    def post(self, name: str) -> Any:

        def callback(module: Any, args: Any, kwargs: Any, output: Any) -> None:
            row = self.pending[name].pop()
            if output is None:
                row['checkpoint_early_stop'] = True
                return
            values = self.leaves(output, 'output')
            row['outputs'] = {key: self.cpu(value) for key, value in values.items()}
            self.add(row['phase'], 'boundary_output', name, row['call_id'], values)
            if row['phase'] == 'forward':
                self.gradient_hooks(values, row, 'output')

        return callback

    def flash_call(self, function: str, args: Any, kwargs: Any, original: Any) -> Any:
        if (function != '_fa2_backward' or self.active.get('recompute_block') != 'blocks.29'
                or self.key(args[1:4]) not in self.flash_keys):
            return original(*args, **kwargs)
        names = ('dout', 'q', 'k', 'v', 'out', 'lse')
        row = {
            'function': function,
            'layer': 29,
            'kwargs': dict(kwargs),
            'inputs': {
                key: self.cpu(value)
                for key, value in zip(names, args[:6], strict=True)
            },
            'metadata': {
                key: {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'stride': list(value.stride())
                }
                for key, value in zip(names, args[:6], strict=True)
            }
        }
        if any(hasattr(value, 'shape') for value in kwargs.values()):
            raise RuntimeError('Unexpected FA2 tensor keyword; capture protocol must be extended')
        result = original(*args, **kwargs)
        row['gradients'] = {key: self.cpu(value) for key, value in zip(('dq', 'dk', 'dv'), args[6:9], strict=True)}
        self.flash.append(row)
        return result

    def finish(self, output: Path, torch: Any) -> dict[str, Any]:
        errors = list(self.errors)
        for suffix in SUFFIXES:
            name = 'blocks.29' + ('.' + suffix if suffix else '')
            rows = [row for row in self.records if row['phase'] == 'forward' and row['name'] == name]
            if len(rows) != 1:
                errors.append('Expected one forward boundary: ' + name)
            for row in rows:
                if 'outputs' not in row:
                    errors.append('Missing output boundary: ' + name)
                for side in ('input', 'output'):
                    expected = {key for group in row.get('expected_' + side + '_gradients', []) for key in group}
                    if set(row.get(side + '_gradients', {})) != expected:
                        errors.append('Missing boundary gradients: ' + name + ':' + side)
        if len(self.flash) != 1:
            errors.append(f'Expected exactly one mapped block-29 FA2 backward, got {len(self.flash)}')
        if any(self.pending.values()):
            errors.append('Unbalanced boundary hook stack')
        torch.save(
            {
                'schema_version': 1,
                'records': self.records,
                'flash': self.flash,
                'gradient_scope': 'tensor gradients include contributions from all consumers'
            }, output)
        return {
            'schema_version': 1,
            'passed': not errors,
            'errors': errors,
            'predeclared_modules': list(SUFFIXES),
            'record_count': len(self.records),
            'flash_calls': len(self.flash)
        }

    def clear(self) -> None:
        for handle in reversed(self.tensor_handles):
            handle.remove()
        self.tensor_handles.clear()

    def close(self) -> None:
        self.clear()
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
