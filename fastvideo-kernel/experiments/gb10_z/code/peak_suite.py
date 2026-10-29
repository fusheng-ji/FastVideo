"""Bounded randomized fresh-process comparisons; service lifecycle is external."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('python', 'output', 'model', 'data'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--lock', type=Path, action='append', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).with_name('spark_peak.py')
    variants = ['baseline', 'clip-loop', 'clip-foreach', 'optimizer-loop', 'optimizer-fused']
    rng = random.Random(20261029)
    plan = []
    for round_index in range(3):
        order = variants.copy()
        rng.shuffle(order)
        plan.extend(dict(round=round_index, variant=v, warmup=1, measure=3) for v in order)
    plan.extend(dict(round='safety', variant=v, warmup=1, measure=1) for v in ('grad-bf16', 'release-grad'))
    source = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in (code, code.with_name('peak_memory.py'))}
    (a.output / 'plan.json').write_text(json.dumps(dict(seed=20261029, runs=plan, source=source), indent=2)+'\n')
    failed_variants = set()
    for job in plan:
        if job['variant'] in failed_variants:
            continue
        name = '{}-{}'.format(job['round'], job['variant'])
        command = [str(a.python), str(code), '--policy', 'dense', '--tokens', '8192', '--variant', job['variant'], '--warmup', str(job['warmup']), '--measure', str(job['measure']), '--output', str(a.output/name), '--model', str(a.model), '--data', str(a.data)]
        for lock in a.lock:
            command.extend(['--lock', str(lock)])
        for filename, expected in source.items():
            assert hashlib.sha256(code.with_name(filename).read_bytes()).hexdigest() == expected
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('Competing GPU process detected; no automatic retry')
        with (a.output/(name+'.log')).open('w') as log, (a.output/(name+'-telemetry.csv')).open('w') as telemetry:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,power.draw', '--format=csv,nounits', '-lms', '200'], stdout=telemetry, stderr=subprocess.STDOUT)
            try:
                started = time.time()
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            finally:
                monitor.terminate()
                monitor.wait()
        receipt = dict(**job, returncode=result.returncode, started_unix=started, ended_unix=time.time())
        (a.output/(name+'-receipt.json')).write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt), flush=True)
        if result.returncode:
            failed_variants.add(job['variant'])
        if result.returncode and (a.output/name/'result.json').exists():
            failure = json.loads((a.output/name/'result.json').read_text())
            if failure.get('error_type') == 'OutOfMemoryError':
                raise RuntimeError('OOM: stop suite')


if __name__ == '__main__':
    main()
