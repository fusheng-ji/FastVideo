"""Measure optimizer step peaks and elapsed time via public optimizer hooks."""
import json
import time
from unittest.mock import patch

import torch
from chunk_v5_memory import MemoryObserver as BaseObserver


class MemoryObserver(BaseObserver):
    def __enter__(self):
        super().__enter__()
        from fastvideo.train.methods.base import TrainingMethod
        self.optimizer_measurements = []
        original = TrainingMethod.optimizers_schedulers_step

        def measure(method, *args, **kwargs):
            handles = []
            current = {}
            optimizers = list(method.get_optimizers(self.step))
            assert len(optimizers) == 1, 'Expected exactly one optimizer'

            def before(optimizer, hook_args, hook_kwargs):
                torch.cuda.synchronize()
                current.update(step=self.step, pre_optimizer_allocated=torch.cuda.memory_allocated(),
                               pre_segment_peak_allocated=torch.cuda.max_memory_allocated(),
                               pre_segment_peak_reserved=torch.cuda.max_memory_reserved(), status='started')
                torch.cuda.reset_peak_memory_stats()
                current['started_at'] = time.perf_counter()

            def after(optimizer, hook_args, hook_kwargs):
                torch.cuda.synchronize()
                current.update(status='completed', optimizer_wall_sec=time.perf_counter() - current.pop('started_at'),
                               optimizer_peak_allocated=torch.cuda.max_memory_allocated(),
                               optimizer_peak_reserved=torch.cuda.max_memory_reserved())
                self.optimizer_measurements.append(dict(current))
                (self.output / 'optimizer_measurements.json').write_text(json.dumps(self.optimizer_measurements, indent=2) + '\n')
                current.clear()

            try:
                for optimizer in optimizers:
                    handles.extend((optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)))
                return original(method, *args, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()
                if current:
                    current.pop('started_at', None)
                    current['status'] = 'failed'
                    self.optimizer_measurements.append(dict(current))
                    (self.output / 'optimizer_measurements.json').write_text(json.dumps(self.optimizer_measurements, indent=2) + '\n')

        self.stack.enter_context(patch.object(TrainingMethod, 'optimizers_schedulers_step', measure))
        return self
