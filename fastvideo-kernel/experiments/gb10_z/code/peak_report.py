"""Export scalar-only peak results, paired clock evidence, and scientific plots."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clocks(path, steps, started_unix=None):
    import numpy as np
    samples = []
    utc_offset = None
    with path.open() as stream:
        for row in csv.DictReader(stream):
            row = {k.strip(): v.strip() for k, v in row.items()}
            wall = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
            if utc_offset is None:
                if started_unix is None:
                    utc_offset = datetime.now().astimezone().utcoffset().total_seconds()
                else:
                    utc_offset = round((wall-started_unix)/900)*900
            stamp = wall-utc_offset
            if any(s['start_unix'] <= stamp <= s['end_unix'] for s in steps):
                samples.append(dict(unix_time=stamp, temperature=float(row['temperature.gpu']), clock_mhz=float(row['clocks.current.sm [MHz]']), utilization=float(row['utilization.gpu [%]']), power_watts=float(row['power.draw [W]'])))
    busy = [s['clock_mhz'] for s in samples if s['utilization'] >= 20]
    return dict(samples=samples, telemetry_utc_offset_seconds=utc_offset, busy_threshold_percent=20,
                clock_p10_p50_p90=np.percentile(busy, [10,50,90]).tolist() if busy else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suite-folder', choices=['paired-v1', 'followup-v1'], default='paired-v1')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    pair = a.input/a.suite_folder
    summaries = []
    plan = json.loads((pair/'plan.json').read_text())
    (a.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    for receipt in sorted(pair.glob('*-receipt.json')):
        name = receipt.name.removesuffix('-receipt.json')
        r = json.loads((pair/name/'result.json').read_text())
        r['raw_result_sha256'] = digest(pair/name/'result.json')
        r['receipt'] = json.loads(receipt.read_text())
        if r['status'] == 'completed':
            steps = [s for s in r['steps'] if not s['warmup']]
            telemetry = clocks(pair/(name+'-telemetry.csv'), steps, r['receipt']['started_unix'])
            r['telemetry'] = telemetry
            baseline = json.loads((pair/(str(r['receipt']['round'])+'-baseline')/'result.json').read_text())
            r['final_parameters_identical_to_paired_dense'] = r['final_parameter_sha256'] == baseline['final_parameter_sha256']
            r['all_825_parameter_events'] = all(s['parameter_events'] == 825 for s in r['steps'])
            summaries.append(dict(run=name, variant=r['variant'], round=r['receipt']['round'], status=r['status'], wall_seconds=r['median_wall_seconds'], cuda_ms=r['median_cuda_ms'], peak_allocated=r['peak_allocated'], peak_reserved=r['peak_reserved'], parameters_identical=r['final_parameters_identical_to_paired_dense'], clock_p10_p50_p90=telemetry['clock_p10_p50_p90']))
        else:
            summaries.append(dict(run=name,variant=r['variant'],status=r['status'],error_type=r['error_type'],error=r['error'], completed_step_timing_available=False,parameters_identical=None))
        (a.output/(name+'.json')).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    successful = [r for r in summaries if r['status']=='completed']
    for r in successful:
        b = next(x for x in successful if x['round']==r['round'] and x['variant']=='baseline')
        r['paired_time_change_percent'] = 100*(r['wall_seconds']/b['wall_seconds']-1)
        r['paired_peak_reduction_percent'] = 100*(1-r['peak_allocated']/b['peak_allocated'])
        r['paired_clock_relative_differences'] = [abs(x-y)/min(x,y) for x,y in zip(r['clock_p10_p50_p90'],b['clock_p10_p50_p90'])]
    (a.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    with (a.output/'summary.csv').open('w') as stream:
        fields = ['run','variant','round','wall_seconds','cuda_ms','peak_allocated','peak_reserved','parameters_identical','paired_time_change_percent','paired_peak_reduction_percent']
        writer=csv.DictWriter(stream,fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(successful)
    profile=a.input/'profile-exact'
    if profile.exists():
        for f in profile.glob('*-peak-attribution.json'):
            (a.output/f.name).write_text(f.read_text())
        r=json.loads((profile/'result.json').read_text())
        (a.output/'profile-baseline.json').write_text(json.dumps(r,indent=2)+'\n')
        receipt={p.name:digest(p) for p in profile.glob('*.pickle')}
        (a.output/'local-snapshot-hashes.json').write_text(json.dumps(receipt,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants=list(dict.fromkeys(r['variant'] for r in successful))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,label,scale in [(axes[0],'peak_allocated','Peak allocated (GiB)',2**30),(axes[1],'wall_seconds','Median step time (s)',1)]:
        groups=[[r[key]/scale for r in successful if r['variant']==v] for v in variants]
        colors=['#4078ad' if all(r['parameters_identical'] for r in successful if r['variant']==v) else '#b95836' for v in variants]
        ax.bar(range(len(groups)),[statistics.median(g) for g in groups],color=colors)
        for i,g in enumerate(groups):ax.scatter([i]*len(g),g,color='black',s=15,zorder=3)
        ax.set_xticks(range(len(groups)),variants,rotation=25,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('GB10 / Wan 1.3B / 8192 / dense — blue: exact parameters; orange: mismatch')
    fig.tight_layout();fig.savefig(a.output/'peak-and-time.png',dpi=160);plt.close(fig)
    if (profile/'step-1-peak-attribution.json').exists():
        attribution=json.loads((profile/'step-1-peak-attribution.json').read_text())
        data=json.loads((profile/'result.json').read_text())['steps'][1]['phase_profile']
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        names=[d['phase'] for d in data]
        axes[0].plot(names,[d['allocated']/2**30 for d in data],'o-',label='live allocated')
        axes[0].plot(names,[d['peak_allocated']/2**30 for d in data],'o--',label='cumulative peak')
        axes[0].set_ylabel('GiB');axes[0].tick_params(axis='x',rotation=25);axes[0].legend();axes[0].grid(alpha=.2)
        categories=attribution['category_bytes']
        axes[1].barh(list(categories),[v/2**30 for v in categories.values()],color='#4078ad')
        axes[1].set_xlabel('GiB at actual allocator peak');axes[1].grid(axis='x',alpha=.2)
        fig.suptitle('Dense 8192: peak occurs inside AdamW foreach sqrt')
        fig.tight_layout();fig.savefig(a.output/'phase-attribution.png',dpi=160);plt.close(fig)



if __name__=='__main__':
    main()
