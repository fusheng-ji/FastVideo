"""Descriptive three-seed report; never interprets absence of significance as equivalence."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import statistics
from common import atomic, digest

from typing import Any
ROOT: Any = None


def main() -> None:
    results = [
        json.loads((ROOT / 'comparisons' / str(seed) / 'result.json').read_text()) for seed in (41001, 41002, 41003)
    ]
    if not all(r['execution_complete'] and r['paired_checks_passed'] for r in results):
        raise RuntimeError('Calibration matrix incomplete')
    ratios = {name: [r[name] for r in results] for name in ('dense_repeat_log_ratio', 'Z_dense_log_ratio')}
    report = dict(
        execution_complete=True,
        statistical_equivalence=None,
        independent_seeds=3,
        runs=9,
        updates=180,
        validation_forwards=864,
        results=results,
        descriptive_summary={
            name:
            dict(mean=statistics.mean(values),
                 sample_sd=statistics.stdev(values),
                 minimum=min(values),
                 maximum=max(values))
            for name, values in ratios.items()
        },
        scope='8192/G1/20 updates, reused validation videos; no performance or video-quality claim',
        next_step=
        'Choose a practical equivalence margin independently of Z observations, then size a new independent-seed confirmation experiment'
    )
    atomic(ROOT / 'result.json', report)
    with (ROOT / 'validation-loss.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'repeat', 'update', 'mean_validation_loss'])
        for result in results:
            for label, values in result['means'].items():
                for step, loss in values.items():
                    writer.writerow([result['seed'], label, step, loss])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, result in zip(axes, results, strict=True):
        for label, values in result['means'].items():
            axis.plot([0, 10, 20], [values[str(step)] for step in (0, 10, 20)], marker='o', label=label)
        axis.set_title('Seed ' + str(result['seed']))
        axis.set_xlabel('Optimizer updates')
        axis.set_ylabel('Mean validation loss')
        axis.legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'validation-loss.png', dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    for name, values in ratios.items():
        axis.plot([41001, 41002, 41003], values, marker='o', label=name)
    axis.axhline(0, color='grey', linewidth=0.8)
    axis.set_xlabel('Independent seed')
    axis.set_ylabel('Log loss ratio at update 20')
    axis.legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'paired-differences.png', dpi=160)
    plt.close(fig)
    (ROOT / 'REPORT.txt').write_text(json.dumps(report, indent=2) + '\n')
    atomic(
        ROOT / 'SHA256.json', {
            str(p.relative_to(ROOT)): digest(p)
            for p in ROOT.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ('SHA256.json', 'state.json',
                                                                               'launcher.log') and p.suffix != '.lock'
        })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    ROOT = parser.parse_args().root
    main()
