"""Run the unmodified YAML training entrypoint once before chunk experiments."""
from contextlib import ExitStack
from pathlib import Path
import argparse
import fcntl
import importlib.metadata
import json
import random
import traceback

from portable import CODE, REPO, configure, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', action='store_true')
    p.add_argument('--tokens', type=int, choices=(8192, 31200), default=8192)
    p.add_argument('--hash-every', type=int, choices=(1, 20), default=1)
    p.add_argument('--chunk', type=int, choices=(0, 16, 32, 64, 128), default=0)
    p.add_argument('--steps', type=int, choices=(1, 2, 4, 200), default=1)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path, action='append', required=True)
    a = p.parse_args()
    if a.output.exists() or len(set(a.lock)) != 2:
        p.error('Require a fresh output directory and two distinct locks')
    a.output.mkdir(parents=True)
    report = {'domain': 'spark-chunk-validation-v5', 'experiment': 'formal-entrypoint-preflight',
              'status': 'started', 'tokens': a.tokens, 'chunk_size': a.chunk, 'steps_requested': a.steps,
              'kernel_sha256': digest(REPO / 'fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py'),
              'harness_sha256': digest(Path(__file__)), 'historical_acceptance_inherited': False}
    try:
        with ExitStack() as stack:
            for path in a.lock:
                handle = stack.enter_context(path.open('a+'))
                fcntl.flock(handle, fcntl.LOCK_EX)
            import torch
            import numpy as np
            import yaml
            configure()
            from fastvideo_kernel.triton_kernels import attn_qat_train as kernel
            from fastvideo.distributed import init_distributed_environment, initialize_model_parallel
            from fastvideo.train.entrypoint.train import run_training_from_config
            assert digest(Path(kernel.__file__)) == report['kernel_sha256']
            assert kernel._sm121_dv_stats_mode() == 'save'
            report['environment'] = {n: importlib.metadata.version(n) for n in ('torch', 'triton', 'flash-attn')}
            torch.cuda.set_device(0)
            assert torch.cuda.get_device_capability() == (12, 1)
            random.seed(1000)
            np.random.seed(1000)
            torch.manual_seed(1000)
            torch.use_deterministic_algorithms(True, warn_only=False)
            config = yaml.safe_load((CODE.parent / 'configs/calibration-8192.yaml').read_text())
            config['models']['student']['init_from'] = str(a.model)
            config['training']['data']['data_path'] = str(a.data)
            if a.tokens == 31200:
                config['training']['data'].update(num_latent_t=20, num_height=480, num_width=832, num_frames=77)
            config['training']['optimizer']['foreach_chunk_size'] = a.chunk
            config['training']['loop']['max_train_steps'] = a.steps
            config['training']['checkpoint']['output_dir'] = str(a.output)
            config['training']['tracker'] = {'project_name': 'gb10-chunk-validation', 'trackers': ['none']}
            config['callbacks']['smoke_observer'] = {'_target_': 'chunk_v4_observer.Observer', 'hash_every': a.hash_every}
            report['external_trackers'] = ['none']
            path = a.output / 'run.yaml'
            path.write_text(yaml.safe_dump(config))
            from fastvideo.train.utils.config import load_run_config
            resolved = load_run_config(str(path))
            report['resolved_chunk_size'] = resolved.training.optimizer.foreach_chunk_size
            assert report['resolved_chunk_size'] == a.chunk, 'YAML chunk size was not preserved by config loader'
            init_distributed_environment(distributed_init_method='file://' + str(a.output / 'rendezvous'))
            initialize_model_parallel()
            try:
                from chunk_v5_memory import MemoryObserver
                with MemoryObserver(a.output, history=a.history) as memory:
                    run_training_from_config(str(path))
                observation = json.loads((a.output / 'observation.json').read_text())
                assert observation['status'] == 'completed' and len(observation['updates']) == a.steps
                report['updates_completed'] = a.steps
                report['status'] = 'completed'
            finally:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
    except BaseException as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        if 'memory' in locals():
            report['failure_phase'] = memory.phase
            report['failure_step'] = memory.step
        report['error'] = str(error)
        for source, replacement in ((str(a.model), '<MODEL>'), (str(a.data), '<DATA>'),
                                    (str(a.output), '<OUTPUT>'), (str(REPO), '<REPO>')):
            report['error'] = report['error'].replace(source, replacement)
        (a.output / 'traceback.local.txt').write_text(traceback.format_exc())
        raise
    finally:
        (a.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
