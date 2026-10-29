"""Export only scalar/hash evidence from an explicitly selected Spark batch."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    docs = {}
    names = [f'{kind}-{n}-{p}' for kind in ['kernel', 'model-v2'] for n in [8192, 16384, 31200] for p in ['dense', 'recompute', 'Z']]
    names += [f'reference-{n}' for n in [8192, 16384, 31200]] + ['overflow-one-tile', 'overflow-all-tiles', 'model-8192-dense']
    for name in names:
        source = a.runs / name / 'result.json'
        if not source.is_file():
            docs[name] = dict(status='not_executed', reason='No result exists in selected attempt')
            continue
        raw = source.read_text()
        doc = json.loads(raw)
        if 'error' in doc:
            doc['error'] = re.sub(r'/(?:home|tmp|run|usr)/[^\s\"\']+', '<LOCAL_PATH>', doc['error'])
        doc['original_result_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
        doc['physical_hardware_distinct_from_history_verified'] = False
        doc['historical_acceptance_inherited'] = False
        public = json.dumps(doc, indent=2, allow_nan=False) + '\n'
        if re.search(r'/home/|/tmp/|gh[pousr]_[A-Za-z0-9]{20,}', public):
            raise ValueError('Publication privacy check failed: ' + name)
        (a.output / (name + '.json')).write_text(public)
        docs[name] = doc
    summary = dict(domain='spark-new-replication-v1', kernel=[], model=[], references=[], overflow=[], numerical=[],
                   production_defaults_changed=False, statistical_equivalence_certified=False, performance_certified=False)
    for kind in ['kernel', 'model']:
        prefix = 'model-v2' if kind == 'model' else kind
        for n in [8192, 16384, 31200]:
            base = docs[f'{prefix}-{n}-dense']
            compared = [docs[f'{prefix}-{n}-{p}'] for p in ['dense', 'recompute', 'Z']]
            if not all(d['status'] == 'completed' for d in compared):
                summary['numerical'].append(dict(kind=kind,tokens=n,status='incomplete'))
                continue
            key = 'final_parameter_sha256' if kind == 'model' else 'tensor_sha256'
            equal = all(d[key] == base[key] for d in compared)
            if kind == 'model':
                equal = equal and all(d['input'] == base['input'] for d in compared)
                equal = equal and all([(x['loss'], x['gradient_norm'], x['parameter_events']) for x in d['steps']] ==
                                      [(x['loss'], x['gradient_norm'], x['parameter_events']) for x in base['steps']]
                                      for d in compared)
            summary['numerical'].append(dict(kind=kind,tokens=n,compared=key,bitwise_equal=equal))
            if not equal:
                raise ValueError('Cross-policy hashes differ; do not claim numerical parity')
            for d in compared:
                row = dict(tokens=n,policy=d['policy'],median_seconds=d['median_wall_seconds'],
                           overhead_percent=100*(d['median_wall_seconds']/base['median_wall_seconds']-1),
                           peak_allocated_GiB=d['peak_allocated']/2**30,peak_reserved_GiB=d['peak_reserved']/2**30,
                           peak_saved_bytes=base['peak_allocated']-d['peak_allocated'])
                if kind=='kernel':
                    stats=d['steps'][-1]['statistics']
                    row.update(saved_statistics_bytes=stats['storage_bytes'],dense_storage_bytes=stats['dense_storage_bytes'],
                               reduction_percent=100*(1-stats['storage_bytes']/stats['dense_storage_bytes']))
                else:
                    forward=[r for r in d['steps'][-1]['statistics'] if r['phase']=='forward']
                    row['maximum_cache_bytes_per_call']=max(r['storage_bytes'] for r in forward)
                    row['aggregate_cache_bytes_not_simultaneously_resident']=sum(r['storage_bytes'] for r in forward)
                    row['spill_calls']=sum(r['spill_tile_count']>0 for r in forward)
                    row['fallback_calls']=sum(r['overflow_fallback'] for r in forward)
                    row['phase_peak_GiB']={phase:d['steps'][-1][phase]['peak_allocated']/2**30 for phase in ['after_forward','after_backward','after_update']}
                summary[kind].append(row)
    for n in [8192,16384,31200]:
        d=docs[f'reference-{n}']
        summary['references'].append(dict(tokens=n,status=d['status'],reference=d.get('reference')))
    for name in ['overflow-one-tile','overflow-all-tiles']:
        d=docs[name]
        summary['overflow'].append(dict(case=name,status=d['status'],bitwise_equal=d.get('bitwise_equal')))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for kind in ['kernel','model']:
        with (a.output/(kind+'.csv')).open('w') as stream:
            fields=['tokens','policy','median_seconds','overhead_percent','peak_allocated_GiB','peak_reserved_GiB','peak_saved_bytes']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
            writer.writeheader();writer.writerows(summary[kind])
    telemetry=a.runs/'model-v2-telemetry.csv'
    if telemetry.exists():
        (a.output/'model-telemetry.csv').write_bytes(telemetry.read_bytes())
    windows = json.loads((a.runs / 'model-v2-queue.json').read_text())
    (a.output / 'model-job-windows.json').write_text(json.dumps(windows, indent=2) + '\n')
    with telemetry.open() as stream:
        samples = list(csv.DictReader(stream))
    def quantile(values, fraction):
        values = sorted(values)
        at = (len(values) - 1) * fraction
        low = int(at)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (at - low)
    clocks = []
    for job in windows:
        busy = [x for x in samples if job['started'] <= float(x['unix_seconds']) <= job['ended']
                and float(x['utilization_percent']) >= 80]
        values = [float(x['sm_MHz']) for x in busy]
        clocks.append(dict(name=job['name'], busy_samples=len(values),
                           sm_MHz_quantiles={str(q): quantile(values, q) for q in [.1,.5,.9]} if values else {},
                           scope='whole process, includes warmup; descriptive, not a paired-clock acceptance'))
    (a.output / 'model-clock-summary.json').write_text(json.dumps(clocks, indent=2) + '\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
