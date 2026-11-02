"""Measure when the negative-prompt text encoder dies, inside the unmodified formal training entrypoint.

The run is the 8192-token calibration recipe through ``run_training_from_config``. Only three names are replaced:
``negative_prompt.TextEncoderLoader`` (to take weak references to the encoder), ``negative_prompt._release_encoder_memory``
(to record the state right after ``del``) and ``wan.encode_negative_prompt`` (to record the state after the function returns).
The run stops with ``ProbeDone`` before the first training step.

Modes:
  fixed        the release helper from the PR runs unchanged; checkpoints are only observed
  natural      the release helper is replaced by a no-op, reproducing the code before the PR
  gc-disabled  as natural, with the cyclic collector disabled from encoder load to the explicit collection
  cycle        as natural, and the explicit collection runs with DEBUG_SAVEALL to list what the collector found
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import traceback
import types
from typing import Any
import weakref

MODES = ('fixed', 'natural', 'gc-disabled', 'cycle')


class ProbeDone(Exception):
    pass


def _sanitize(text: str, replacements: list[tuple[str, str]]) -> str:
    for source, replacement in replacements:
        if source:
            text = text.replace(source, replacement)
    return text


class LifetimeProbe:
    """Tracks the encoder through weak references only; never holds a strong reference between calls."""

    def __init__(self, cuda: Any, replacements: list[tuple[str, str]]):
        self.cuda = cuda
        self.replacements = replacements
        self.tensor_refs: list[weakref.ref] = []
        self.module_ref: weakref.ref | None = None
        self.module_id: int | None = None
        self.module_class: str | None = None
        self.checkpoints: list[dict[str, Any]] = []
        self.referrers: dict[str, Any] = {}
        self.cycle: dict[str, Any] = {}
        self.gc_collected: int | None = None

    def track(self, module: Any) -> None:
        tensors = list(module.parameters()) + list(module.buffers())
        self.tensor_refs = [weakref.ref(t) for t in tensors]
        self.module_ref = weakref.ref(module)
        self.module_id = id(module)
        self.module_class = type(module).__name__
        self.tracked_bytes = sum(t.numel() * t.element_size() for t in tensors)

    def live_tensors(self) -> int:
        return sum(ref() is not None for ref in self.tensor_refs)

    def module_live(self) -> bool:
        return self.module_ref is not None and self.module_ref() is not None

    def checkpoint(self, name: str) -> dict[str, Any]:
        if self.cuda is not None:
            self.cuda.synchronize()
        record = {
            'name': name,
            'tracked_tensors': len(self.tensor_refs),
            'live_tensors': self.live_tensors(),
            'encoder_module_live': self.module_live(),
            'gc_enabled': gc.isenabled(),
            'allocated_bytes': self.cuda.memory_allocated() if self.cuda is not None else None,
            'reserved_bytes': self.cuda.memory_reserved() if self.cuda is not None else None,
        }
        self.checkpoints.append(record)
        return record

    def _describe(self, obj: Any, child: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {'type': type(obj).__name__}
        if isinstance(obj, types.FrameType):
            entry['function'] = obj.f_code.co_name
            entry['file'] = _sanitize(obj.f_code.co_filename, self.replacements)
            entry['line'] = obj.f_lineno
            entry['locals'] = sorted(k for k, v in obj.f_locals.items() if v is child)[:5]
        elif isinstance(obj, dict):
            entry['keys'] = sorted(str(k) for k, v in obj.items() if v is child)[:5]
            owners = [o for o in gc.get_referrers(obj) if getattr(o, '__dict__', None) is obj]
            if owners:
                entry['dict_of'] = type(owners[0]).__name__
        elif isinstance(obj, (types.FunctionType, types.MethodType)):
            entry['qualname'] = getattr(obj, '__qualname__', '')
        elif isinstance(obj, (list, tuple, set)):
            entry['length'] = len(obj)
        elif hasattr(obj, '__class__') and hasattr(obj, '__module__'):
            entry['class_module'] = type(obj).__module__
        return entry

    def referrer_tree(self, label: str, depth: int = 3, fanout: int = 8) -> None:
        """Record who refers to the encoder module (or to a live tensor when the module is gone)."""
        target = self.module_ref() if self.module_ref is not None else None
        start = 'encoder_module'
        if target is None:
            target = next((ref() for ref in self.tensor_refs if ref() is not None), None)
            start = 'encoder_tensor'
        if target is None:
            self.referrers[label] = {'start': None}
            return
        ignore = {id(self), id(self.checkpoints), id(self.tensor_refs)}
        self.referrers[label] = {'start': start, 'tree': self._walk(target, 0, {id(target)}, ignore, depth, fanout)}
        del target

    def _walk(self, obj: Any, level: int, seen: set[int], ignore: set[int], depth: int,
              fanout: int) -> list[dict[str, Any]]:
        if level == depth:
            return []
        nodes: list[dict[str, Any]] = []
        ignore.add(id(nodes))
        refs = gc.get_referrers(obj)
        ignore.add(id(refs))
        for index in range(len(refs)):
            ref = refs[index]
            if id(ref) in ignore or id(ref) in seen:
                continue
            if isinstance(ref, types.FrameType) and ref.f_code.co_filename == __file__:
                continue
            entry = self._describe(ref, obj)
            if not isinstance(ref, (types.FrameType, types.ModuleType)):
                entry['referred_by'] = self._walk(ref, level + 1, seen | {id(ref)}, ignore, depth, fanout)
            nodes.append(entry)
            if len(nodes) == fanout:
                break
        return nodes

    def collect(self, save_all: bool) -> None:
        if not save_all:
            self.gc_collected = gc.collect()
            return
        gc.set_debug(gc.DEBUG_SAVEALL)
        try:
            self.gc_collected = gc.collect()
            # The collector clears weak references before it saves garbage, so find the module by identity.
            garbage_ids = {id(o) for o in gc.garbage}
            module = next((o for o in gc.garbage
                           if id(o) == self.module_id and type(o).__name__ == self.module_class), None)
            summary: dict[str, Any] = {
                'garbage_objects': len(gc.garbage),
                'encoder_module_in_garbage': module is not None,
                'garbage_type_counts': dict(Counter(type(o).__name__ for o in gc.garbage).most_common(15)),
            }
            if module is not None:
                summary['garbage_referrers_of_module'] = [
                    self._describe(o, module) for o in gc.get_referrers(module) if id(o) in garbage_ids
                ][:10]
            del module
            self.cycle = summary
        finally:
            gc.set_debug(0)
            del gc.garbage[:]
            gc.collect()


def _install(probe: LifetimeProbe, mode: str, report: dict[str, Any]) -> None:
    import fastvideo.train.models.wan.wan as wan_module
    import fastvideo.train.utils.negative_prompt as negative_prompt

    original_loader = negative_prompt.TextEncoderLoader
    original_release = negative_prompt._release_encoder_memory
    original_encode = wan_module.encode_negative_prompt
    if original_encode is not negative_prompt.encode_negative_prompt:
        raise RuntimeError('wan.encode_negative_prompt is not the shared helper; the probe would not observe it')

    class ProbeLoader(original_loader):  # type: ignore[misc, valid-type]

        def load(self, *args: Any, **kwargs: Any) -> Any:
            if mode == 'gc-disabled':
                gc.disable()
            module = super().load(*args, **kwargs)
            probe.track(module)

            def after_forward(_module: Any, _inputs: Any, _outputs: Any) -> None:
                probe.checkpoint('after_forward')

            module.register_forward_hook(after_forward)
            probe.checkpoint('after_load')
            return module

    def release_probe() -> None:
        probe.checkpoint('after_del_inside_function')
        if mode == 'fixed':
            original_release()
            probe.checkpoint('after_release_helper_inside_function')

    def encode_probe(*args: Any, **kwargs: Any) -> Any:
        result = original_encode(*args, **kwargs)
        probe.checkpoint('after_return')
        if probe.live_tensors():
            probe.referrer_tree('after_return')
        if mode == 'gc-disabled':
            gc.enable()
        probe.collect(save_all=mode == 'cycle')
        probe.checkpoint('after_gc_collect')
        if probe.live_tensors():
            probe.referrer_tree('after_gc_collect')
        if probe.cuda is not None:
            probe.cuda.empty_cache()
        probe.checkpoint('after_empty_cache')
        report['negative_embeds_shape'] = list(result[0].shape)
        raise ProbeDone

    negative_prompt.TextEncoderLoader = ProbeLoader
    negative_prompt._release_encoder_memory = release_probe
    wan_module.encode_negative_prompt = encode_probe
    report['patched'] = [
        'negative_prompt.TextEncoderLoader', 'negative_prompt._release_encoder_memory', 'wan.encode_negative_prompt'
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path, action='append', required=True)
    a = p.parse_args()
    if a.output.exists() or len(set(a.lock)) != 2:
        p.error('Require a fresh output directory and two distinct locks')
    a.output.mkdir(parents=True)

    from portable import CODE, REPO, configure

    replacements = [(str(a.model), '<MODEL>'), (str(a.data), '<DATA>'), (str(a.output), '<OUTPUT>'),
                    (str(REPO), '<REPO>'), (str(Path.home()), '<HOME>')]
    report: dict[str, Any] = {
        'domain': 'spark-encoder-root-cause',
        'mode': a.mode,
        'status': 'started',
        'commit': subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                 check=False).stdout.strip(),
        'historical_acceptance_inherited': False,
    }
    probe = None
    try:
        with ExitStack() as stack:
            for path in a.lock:
                handle = stack.enter_context(path.open('a+'))
                fcntl.flock(handle, fcntl.LOCK_EX)
            import numpy as np
            import torch
            import yaml
            configure()
            from fastvideo.distributed import init_distributed_environment, initialize_model_parallel
            from fastvideo.train.entrypoint.train import run_training_from_config
            report['environment'] = {n: importlib.metadata.version(n) for n in ('torch', 'triton', 'flash-attn')}
            torch.cuda.set_device(0)
            report['capability'] = list(torch.cuda.get_device_capability())
            if tuple(report['capability']) != (12, 1):
                raise RuntimeError(f'expected GB10 capability (12, 1), got {report["capability"]}')
            random.seed(1000)
            np.random.seed(1000)
            torch.manual_seed(1000)
            torch.use_deterministic_algorithms(True, warn_only=False)

            config = yaml.safe_load((CODE.parent / 'configs/calibration-8192.yaml').read_text())
            config['models']['student']['init_from'] = str(a.model)
            config['training']['data']['data_path'] = str(a.data)
            config['training']['loop']['max_train_steps'] = 1
            config['training']['checkpoint']['output_dir'] = str(a.output)
            config['training']['tracker'] = {'project_name': 'gb10-encoder-root-cause', 'trackers': ['none']}
            run_yaml = a.output / 'run.yaml'
            run_yaml.write_text(yaml.safe_dump(config))

            probe = LifetimeProbe(torch.cuda, replacements)
            _install(probe, a.mode, report)
            init_distributed_environment(distributed_init_method='file://' + str(a.output / 'rendezvous'))
            initialize_model_parallel()
            try:
                run_training_from_config(str(run_yaml))
                raise RuntimeError('training finished without encoding the negative prompt')
            except ProbeDone:
                report['status'] = 'completed'
            finally:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
    except BaseException as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        report['error'] = _sanitize(str(error), replacements)
        (a.output / 'traceback.local.txt').write_text(traceback.format_exc())
        raise
    finally:
        if probe is not None:
            report['encoder_class'] = probe.module_class
            report['tracked_tensors'] = len(probe.tensor_refs)
            report['tracked_bytes'] = getattr(probe, 'tracked_bytes', None)
            report['checkpoints'] = probe.checkpoints
            report['gc_collect_return'] = probe.gc_collected
            report['referrers'] = probe.referrers
            report['cycle'] = probe.cycle
        (a.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
