"""Summarize per-ref kernel-lane runs (JUnit XML) and list failures that the baseline ref does not have."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

MESSAGE_CHARS = 600


def read_run(run_dir: Path) -> dict:
    run: dict = {'name': run_dir.name, 'failed_stage': None, 'lane_exit_code': None, 'junit': False,
                 'counts': {'passed': 0, 'failed': 0, 'error': 0, 'skipped': 0, 'xfailed': 0}, 'failures': {}}
    stage = run_dir / 'failed_stage'
    if stage.exists():
        run['failed_stage'] = stage.read_text().strip()
    code = run_dir / 'lane_exit_code'
    if code.exists():
        run['lane_exit_code'] = int(code.read_text().strip())
    junit = run_dir / 'junit.xml'
    if not junit.exists():
        return run
    run['junit'] = True
    for case in ET.parse(junit).getroot().iter('testcase'):
        test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
        outcome = 'passed'
        detail = None
        for child in case:
            if child.tag in ('failure', 'error'):
                outcome = 'failed' if child.tag == 'failure' else 'error'
                detail = (child.get('message') or '') + '\n' + (child.text or '')
                break
            if child.tag == 'skipped':
                outcome = 'xfailed' if child.get('type') == 'pytest.xfail' else 'skipped'
        run['counts'][outcome] += 1
        if outcome in ('failed', 'error'):
            run['failures'][test_id] = {'outcome': outcome, 'message': (detail or '').strip()[:MESSAGE_CHARS]}
    return run


def summarize(runs_root: Path, baseline: str) -> dict:
    runs = {d.name: read_run(d) for d in sorted(runs_root.iterdir()) if d.is_dir()}
    if baseline not in runs:
        raise SystemExit(f'baseline run {baseline!r} not found in {runs_root}')
    base_failures = set(runs[baseline]['failures'])
    comparison = {}
    for name, run in runs.items():
        if name == baseline:
            continue
        if not run['junit'] or not runs[baseline]['junit']:
            comparison[name] = {'not_comparable': f"no test results (stage failure: {run['failed_stage']}, "
                                                  f"baseline stage failure: {runs[baseline]['failed_stage']})"}
            continue
        failures = set(run['failures'])
        comparison[name] = {'only_here': sorted(failures - base_failures),
                            'only_in_baseline': sorted(base_failures - failures),
                            'shared': sorted(failures & base_failures)}
    return {'baseline': baseline, 'runs': runs, 'comparison_to_baseline': comparison}


def to_markdown(summary: dict) -> str:
    lines = ['| ref | stage failure | lane exit | passed | failed | error | skipped | xfailed |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, run in summary['runs'].items():
        c = run['counts']
        lines.append(f"| {name} | {run['failed_stage'] or '-'} | {run['lane_exit_code']} | {c['passed']} | "
                     f"{c['failed']} | {c['error']} | {c['skipped']} | {c['xfailed']} |")
    for name, diff in summary['comparison_to_baseline'].items():
        lines += ['', f"## {name}: failures not on {summary['baseline']}", '']
        if 'not_comparable' in diff:
            lines.append(diff['not_comparable'])
            continue
        if not diff['only_here']:
            lines.append('none')
        for test_id in diff['only_here']:
            message = summary['runs'][name]['failures'][test_id]['message'].splitlines()
            lines.append(f"- `{test_id}`: {message[0] if message else ''}")
        if diff['only_in_baseline']:
            lines.append(f"- fixed relative to {summary['baseline']}: " +
                         ', '.join(f'`{t}`' for t in diff['only_in_baseline']))
    return '\n'.join(lines) + '\n'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs_root', type=Path)
    p.add_argument('--baseline', default='main')
    p.add_argument('--output', type=Path, required=True, help='writes OUTPUT.json and OUTPUT.md')
    a = p.parse_args()
    summary = summarize(a.runs_root, a.baseline)
    a.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    a.output.with_suffix('.md').write_text(to_markdown(summary))
    print(to_markdown(summary))


if __name__ == '__main__':
    main()
