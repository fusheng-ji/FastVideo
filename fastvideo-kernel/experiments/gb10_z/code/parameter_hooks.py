"""Private, CPU-import-safe registry for master gradient accumulation events."""
from typing import Any
from collections import Counter

HOOK_SEMANTICS = 'master_parameter_gradient_accumulation_complete'


class ParameterHookRegistry:

    def __init__(self, named_parameters: Any, phase: Any, ledger: Any) -> None:
        self.phase = phase
        self.ledger = ledger
        self.entries = []
        self.handles: list[Any] = []
        self.counts: Counter[str] | None = None
        self.callback_errors: list[str] = []
        self.mounted = False
        seen: dict[int, dict[str, Any]] = {}
        names: set[str] = set()
        for name, parameter in named_parameters:
            if name in names:
                raise RuntimeError(f'Duplicate parameter name: {name}')
            names.add(name)
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                seen[id(parameter)]['aliases'].append(name)
                continue
            entry = {
                'name': name,
                'aliases': [name],
                'parameter': parameter,
                'type': type(parameter).__module__ + '.' + type(parameter).__qualname__
            }
            seen[id(parameter)] = entry
            self.entries.append(entry)
        if not self.entries:
            raise RuntimeError('No trainable master parameters')

    @property
    def names(self) -> Any:
        return [entry['name'] for entry in self.entries]

    def declaration(self) -> Any:
        return [{
            'name': entry['name'],
            'aliases': list(entry['aliases']),
            'type': entry['type'],
            'object_id': id(entry['parameter'])
        } for entry in self.entries]

    def mount(self) -> Any:
        if self.mounted:
            raise RuntimeError('Parameter hooks already mounted')
        try:
            for entry in self.entries:

                def callback(parameter: Any, entry: Any = entry) -> Any:
                    # Python metadata only: no tensor arithmetic, transfer, or synchronization.
                    error = None
                    if parameter is not entry['parameter']:
                        error = 'Callback master identity changed'
                    elif self.counts is None or self.phase() != 'backward':
                        error = 'Callback outside active backward'
                    elif parameter.grad is None:
                        error = 'Accumulation callback has no master gradient'
                    if error:
                        self.callback_errors.append(error + ': ' + entry['name'])
                        raise RuntimeError(self.callback_errors[-1])
                    assert self.counts is not None
                    self.counts[entry['name']] += 1
                    self.ledger().add('backward',
                                      'parameter_gradient_arrival',
                                      entry['name'],
                                      entry['name'],
                                      marker=True)

                self.handles.append(entry['parameter'].register_post_accumulate_grad_hook(callback))
            self.mounted = True
        except BaseException:
            self.close()
            raise

    def begin(self) -> Any:
        if not self.mounted or self.counts is not None:
            raise RuntimeError('Cannot begin an unmounted or active parameter registry')
        self.counts = Counter({name: 0 for name in self.names})
        self.callback_errors = []

    def finish(self, named_parameters: Any) -> Any:
        if self.counts is None:
            raise RuntimeError('No active backward to finish')
        errors = list(self.callback_errors)
        current = {name: parameter for name, parameter in named_parameters if parameter.requires_grad}
        expected = {alias: entry['parameter'] for entry in self.entries for alias in entry['aliases']}
        identity_verified = (set(current) == set(expected)
                             and all(current[name] is parameter for name, parameter in expected.items()))
        if not identity_verified:
            errors.append('Master parameter identity or alias mapping changed after backward')
        for entry in self.entries:
            name = entry['name']
            if self.counts[name] != 1:
                errors.append(f'{name}: expected 1 accumulation event, got {self.counts[name]}')
            if entry['parameter'].grad is None:
                errors.append(f'{name}: missing master gradient')
        result = {
            'semantics': HOOK_SEMANTICS,
            'identity_verified': identity_verified,
            'counts': dict(self.counts),
            'errors': errors,
            'passed': not errors
        }
        self.counts = None
        return result

    def abort(self) -> Any:
        self.counts = None

    def close(self) -> Any:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.mounted = False
        self.abort()

    def __enter__(self) -> Any:
        self.mount()
        return self

    def __exit__(self, *exc: Any) -> Any:
        self.close()
