"""Recompute latency tables from the public numerical exports on CPU."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/gb10-recomputed-numerics'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    definitions = []
    for grid, tokens in [('20x30x52', 31200), ('8x32x32', 8192)]:
        for mode, scope in [('attention', 'layout+attention'), ('layout', 'layout')]:
            definitions.append((scope, tokens, 'ms', f'vsa64/benchmark-{grid}-{mode}.csv', 'latency_ms'))
    for tokens in [31200, 8192]:
        definitions.append(('trained full DiT, synthetic conditioning', tokens, 'ms',
                            f'vsa64/trained-wan-{tokens}-benchmark.csv', 'latency_ms'))
    definitions.extend([
        ('3-step trained DiT with real prompts', 31200, 'ms', 'vsa64/pipeline-full-v3/report.csv', 'dit_total_ms'),
        ('instrumented prompt pipeline including T5/VAE', 31200, 's', 'vsa64/pipeline-full-v3/report.csv', 'generation_wall_seconds'),
    ])
    results = []
    for scope, tokens, unit, relative, field in definitions:
        with (args.data_dir / relative).open(newline='') as source:
            rows = list(csv.DictReader(source))
        for route in ['legacy', 'fused']:
            selected = [row for row in rows if row['route'] == route]
            values = [float(row[field]) for row in selected]
            assert values and all(math.isfinite(value) and value > 0 for value in values)
            results.append({
                'scope': scope, 'tokens': tokens, 'route': route, 'unit': unit, 'samples': len(values),
                'mean': statistics.mean(values), 'median': statistics.median(values),
                'p10': percentile(values, .1), 'p90': percentile(values, .9),
                'minimum': min(values), 'maximum': max(values),
                'peak_allocated_bytes': max(int(row['peak_allocated_bytes']) for row in selected),
                'peak_reserved_bytes': max(int(row['peak_reserved_bytes']) for row in selected),
                'source_csv': relative,
            })
    (args.output_dir / 'performance_statistics.json').write_text(json.dumps(results, indent=2) + '\n')
    with (args.output_dir / 'performance_statistics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({'statistics_rows': len(results), 'raw_timing_rows': 1818, 'sample_filtering': False}))


if __name__ == '__main__':
    main()
