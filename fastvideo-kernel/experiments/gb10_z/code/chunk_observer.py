"""Experiment-only observation for the unchanged YAML Trainer entrypoint."""
from pathlib import Path
import hashlib
import json
import math

import torch
from fastvideo.train.callbacks.callback import Callback


def parameter_hashes(parameters):
    result = {}
    for index, parameter in enumerate(parameters):
        value = parameter.detach()
        if hasattr(value, 'to_local'):
            value = value.to_local()
        raw = value.contiguous().cpu().view(torch.uint8).numpy().tobytes()
        result[str(index)] = hashlib.sha256(raw).hexdigest()
    return result


class SmokeObserver(Callback):
    def __init__(self):
        self.report = {'status': 'started', 'updates': [], 'tracker_metrics': {}}

    def save(self):
        self.path.write_text(json.dumps(self.report, indent=2, allow_nan=False) + '\n')

    def on_train_start(self, method, iteration=0):
        self.path = Path(self.training_config.checkpoint.output_dir) / 'observation.json'
        self.parameters = list(method.student.transformer.parameters())
        self.report['parameter_tensors'] = len(self.parameters)
        assert len(self.parameters) == 825, 'Expected 825 parameter tensors'
        self.report['initial_parameter_sha256'] = parameter_hashes(self.parameters)
        self.report['initial_method_rng_sha256'] = parameter_hashes([method.cuda_generator.get_state()])
        self.report['initial_cpu_rng_sha256'] = parameter_hashes([torch.get_rng_state()])
        self.report['initial_cuda_rng_sha256'] = parameter_hashes([torch.cuda.get_rng_state()])
        optimizer = method.get_optimizers(iteration)[0]
        self.report['optimizer_class'] = type(optimizer).__name__
        self.report['optimizer_chunk_size'] = getattr(optimizer, '_foreach_chunk_size', 0)
        self.report['optimizer_flags'] = [{key: group.get(key) for key in ('foreach', 'fused', 'capturable')}
                                          for group in optimizer.param_groups]
        original_log = method.tracker.log
        def observe_log(metrics, step, *args, **kwargs):
            for key, value in metrics.items():
                if key.startswith('grad_norm/'):
                    self.report['tracker_metrics'].setdefault(str(step), {})[key] = float(value)
            return original_log(metrics, step, *args, **kwargs)
        method.tracker.log = observe_log
        self.save()
        torch.cuda.reset_peak_memory_stats()

    def on_training_step_end(self, method, loss_dict, iteration=0):
        torch.cuda.synchronize()
        hashes = parameter_hashes(self.parameters)
        metrics = {key: float(value) for key, value in loss_dict.items()}
        assert all(math.isfinite(value) for value in metrics.values()), 'Nonfinite metrics'
        changed = sum(hashes[key] != value for key, value in self.report['initial_parameter_sha256'].items())
        row = {'step': iteration, 'metrics': metrics, 'parameter_sha256': hashes,
               'parameters_changed_from_initial': changed,
               'method_rng_sha256': parameter_hashes([method.cuda_generator.get_state()]),
               'peak_allocated': torch.cuda.max_memory_allocated(),
               'peak_reserved': torch.cuda.max_memory_reserved()}
        self.report['updates'].append(row)
        self.save()
        assert changed > 0, 'No parameter update observed'
        print(json.dumps({'observed_update': iteration, 'loss': metrics.get('total_loss'),
                          'parameter_tensors': len(hashes), 'changed_parameters': changed}), flush=True)

    def on_train_end(self, method, iteration=0):
        self.report['status'] = 'completed'
        self.save()
