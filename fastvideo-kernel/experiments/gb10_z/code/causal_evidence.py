"""Keep default acceptance blocked even when independent FA2 controls pass."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from probe_helpers import atomic_json


def validate_local(local: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    if local.get('evidence_valid') is not True or replay.get('evidence_valid') is not True:
        raise ValueError('Incomplete local evidence')
    modes = replay.get('modes', [])
    if len(modes) != 2 or [mode['deterministic'] for mode in modes] != [False, True]:
        raise ValueError('Missing FA2 False/True control coverage')
    for mode in modes:
        if [run['repetition'] for run in mode['runs']] != list(range(10)):
            raise ValueError('Expected ten real FA2 repetitions per mode')
    contrast = (local['flash']['operator_local_divergence'] and modes[0]['repeatable'] is False
                and modes[1]['repeatable'] is True)
    return {
        'status': 'complete',
        'evidence_valid': True,
        'local_fa2_contrast': contrast,
        'root_cause_proven': False,
        'default_model_acceptance': False
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--kind', choices=('local', 'final'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = [args.root / 'results/localize.json', args.root / 'results/fa2-replay.json']
    local, replay = [json.loads(path.read_text()) for path in files]
    result = validate_local(local, replay)
    if args.kind == 'local':
        if not result['local_fa2_contrast']:
            result['next_step'] = 'Adjacent-operator and checkpoint isolation required; full FA2 causal claim blocked'
    else:
        files.append(args.root / 'results/control-process-parity.json')
        full = json.loads(files[-1].read_text())
        if full.get('accepted') is not True:
            raise ValueError('Full-model control has not passed strict across-process byte/order equality')
        result['root_cause_proven'] = result['local_fa2_contrast']
        result['full_model_deterministic_control_passed'] = True
        result['scope'] = 'Validated GB10 8192-token seed=1000 dense/G1 only; default acceptance remains blocked'
        for name in ('baseline-regression', 'prototype-regression'):
            files.append(args.root / f'results/{name}.json')
            if json.loads(files[-1].read_text()).get('accepted') is not True:
                raise ValueError('Regression failed: ' + name)
    result['inputs'] = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()
