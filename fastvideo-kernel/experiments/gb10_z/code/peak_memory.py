"""Storage-deduplicated inventory and local-only allocator trace recording."""
from pathlib import Path
import json


def local(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


def inventory(parameters, optimizer):
    addresses = {}
    for category, tensors in (
        ('parameters_master_fp32', parameters),
        ('gradients', [p.grad for p in parameters if p.grad is not None]),
        ('optimizer_state', [v for state in optimizer.state.values() for v in state.values() if hasattr(v, 'untyped_storage')]),
    ):
        for tensor in tensors:
            tensor = local(tensor)
            if tensor.device.type != 'cuda':
                continue
            storage = tensor.untyped_storage()
            addresses.setdefault(storage.data_ptr(), {'category': category, 'bytes': storage.nbytes(), 'dtype': str(tensor.dtype)})
    return addresses


class PhaseRecorder:
    def __init__(self, torch, parameters, optimizer, output, enabled):
        self.torch, self.parameters, self.optimizer = torch, parameters, optimizer
        self.output, self.enabled = Path(output), enabled
        self.addresses = {}
        self.rows = []

    def begin(self, step):
        self.step, self.rows, self.addresses = step, [], {}
        if self.enabled:
            self.torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='python', max_entries=200000, clear_history=True)
            self.torch.cuda.memory._dump_snapshot(str(self.output / f'step-{step}-start.pickle'))

    def mark(self, phase):
        # No synchronizations added to ordinary paired timing. These are allocator counters.
        cuda = self.torch.cuda
        result = dict(allocated=cuda.memory_allocated(), reserved=cuda.memory_reserved(), peak_allocated=cuda.max_memory_allocated(), peak_reserved=cuda.max_memory_reserved())
        if self.enabled:
            current = inventory(self.parameters, self.optimizer)
            self.addresses.update(current)
            categories = {}
            for item in current.values():
                categories[item['category']] = categories.get(item['category'], 0) + item['bytes']
            self.rows.append(dict(phase=phase, **result, live_storage_bytes=categories,
                                 other_allocated_bytes=result['allocated']-sum(categories.values())))
            self.torch.cuda.memory._dump_snapshot(str(self.output / f'step-{self.step}-{phase}.pickle'))
        return result

    def finish(self):
        if self.enabled:
            (self.output / f'step-{self.step}-addresses.json').write_text(json.dumps(self.addresses))
            self.torch.cuda.memory._record_memory_history(enabled=None)
        return self.rows

    def close(self):
        if self.enabled:
            self.torch.cuda.memory._record_memory_history(enabled=None)
            (self.output / 'last-phase-inventory.json').write_text(json.dumps(self.rows, indent=2))
