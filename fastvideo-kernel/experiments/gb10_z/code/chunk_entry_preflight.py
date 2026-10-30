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
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path, action='append', required=True)
    a = p.parse_args()
    if a.output.exists() or len(set(a.lock)) != 2:
        p.error('Require a fresh output directory and two distinct locks')
    a.output.mkdir(parents=True)
    report = {'domain': 'spark-chunk-validation-v3', 'experiment': 'formal-entrypoint-preflight',
              'status': 'started', 'tokens': 8192, 'chunk_size': 0, 'steps_requested': 1,
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
            config['training']['optimizer']['foreach_chunk_size'] = 0
            config['training']['loop']['max_train_steps'] = 1
            config['training']['checkpoint']['output_dir'] = str(a.output)
            config['training']['tracker'] = {'project_name': 'gb10-chunk-validation', 'trackers': ['none']}
            config['callbacks']['smoke_observer'] = {'_target_': 'chunk_observer.SmokeObserver'}
            report['external_trackers'] = ['none']
            path = a.output / 'run.yaml'
            path.write_text(yaml.safe_dump(config))
            init_distributed_environment(distributed_init_method='file://' + str(a.output / 'rendezvous'))
            initialize_model_parallel()
            try:
                run_training_from_config(str(path))
                observation = json.loads((a.output / 'observation.json').read_text())
                assert observation['status'] == 'completed' and len(observation['updates']) == 1
                report['updates_completed'] = 1
                report['status'] = 'completed'
            finally:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
    except BaseException as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
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
