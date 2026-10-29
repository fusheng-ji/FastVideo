"""Three independently paired tests of the bounded-foreach candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('python', 'output', 'model', 'data'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--lock', type=Path, action='append', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).parent
    sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [code/'spark_peak.py',code/'peak_memory.py',code/'peak_chunked.py']}
    rng=random.Random(20261030)
    plan=[]
    for number in range(3):
        pair=['baseline','optimizer-chunk64'];rng.shuffle(pair)
        plan.extend(dict(round=number,variant=v) for v in pair)
    (args.output/'plan.json').write_text(json.dumps(dict(seed=20261030,runs=plan,source=sources),indent=2)+'\n')
    for job in plan:
        name='{}-{}'.format(job['round'],job['variant'])
        script='spark_peak.py' if job['variant']=='baseline' else 'peak_chunked.py'
        command=[str(args.python),str(code/script),'--policy','dense','--tokens','8192','--variant','baseline','--warmup','1','--measure','3','--output',str(args.output/name),'--model',str(args.model),'--data',str(args.data)]
        for lock in args.lock:command.extend(['--lock',str(lock)])
        for filename,expected in sources.items():
            assert hashlib.sha256((code/filename).read_bytes()).hexdigest()==expected
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'Competing GPU process'
        cooling_start = time.time()
        while float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip()) > 40:
            if time.time()-cooling_start > 300:
                raise RuntimeError('Cooling timeout; no measurement started')
            time.sleep(3)
        cooling_seconds = time.time()-cooling_start
        with (args.output/(name+'.log')).open('w') as log,(args.output/(name+'-telemetry.csv')).open('w') as telemetry:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,power.draw','--format=csv,nounits','-lms','200'],stdout=telemetry,stderr=subprocess.STDOUT)
            try:
                start=time.time();result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            finally:
                monitor.terminate();monitor.wait()
        receipt=dict(**job,cooling_seconds=cooling_seconds,returncode=result.returncode,started_unix=start,ended_unix=time.time(),warmup=1,measure=3)
        (args.output/(name+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
        assert result.returncode==0, 'Failed attempt: no retry'
        paths=[args.output/f"{job['round']}-{v}"/'result.json' for v in ('baseline','optimizer-chunk64')]
        if all(p.exists() for p in paths):
            left,right=[json.loads(p.read_text()) for p in paths]
            assert left['final_parameter_sha256']==right['final_parameter_sha256'], 'Candidate changes parameters; stop'
            assert [s['loss'] for s in left['steps']]==[s['loss'] for s in right['steps']], 'Candidate changes losses'
            assert [s['gradient_norm'] for s in left['steps']]==[s['gradient_norm'] for s in right['steps']], 'Candidate changes norms'


if __name__=='__main__':
    main()
