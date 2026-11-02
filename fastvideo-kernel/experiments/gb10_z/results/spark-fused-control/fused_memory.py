"""Private observer for per-optimizer peaks and wall time; training arithmetic stays unchanged."""
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
            torch.cuda.synchronize()
            before = {'step': self.step, 'pre_optimizer_allocated': torch.cuda.memory_allocated(),
                      'pre_segment_peak_allocated': torch.cuda.max_memory_allocated(),
                      'pre_segment_peak_reserved': torch.cuda.max_memory_reserved()}
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                result = original(method, *args, **kwargs)
                torch.cuda.synchronize()
                before['status'] = 'completed'
                return result
            except BaseException:
                before['status'] = 'failed'
                raise
            finally:
                before.update(optimizer_wall_sec=time.perf_counter() - start,
                              optimizer_peak_allocated=torch.cuda.max_memory_allocated(),
                              optimizer_peak_reserved=torch.cuda.max_memory_reserved())
                self.optimizer_measurements.append(before)
                (self.output / 'optimizer_measurements.json').write_text(json.dumps(self.optimizer_measurements, indent=2) + '\n')

        self.stack.enter_context(patch.object(TrainingMethod, 'optimizers_schedulers_step', measure))
        return self
