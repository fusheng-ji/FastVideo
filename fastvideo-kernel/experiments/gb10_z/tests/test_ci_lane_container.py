"""CPU checks for ci_lane_container.sh (stage handling, with stub uv, build.sh, python and lane script) and for the
provenance comparison in ci_lane_environment.py. The real build and provenance check are exercised by running the
container script outside Docker against a real checkout; see SPARK_CI_KERNEL_LANE.md."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
from ci_lane_environment import compare_tree


def write_exec(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\n' + body)
    path.chmod(0o755)


@pytest.fixture
def lane(tmp_path):
    src, out, venv, stubs = tmp_path / 'src', tmp_path / 'out', tmp_path / 'venv', tmp_path / 'stubs'
    out.mkdir()
    (venv / 'bin').mkdir(parents=True)
    (venv / 'bin' / 'activate').write_text('')
    write_exec(stubs / 'uv', 'echo "uv $*" >>"$STUB_LOG"; exit "${STUB_UV_RC:-0}"\n')
    write_exec(stubs / 'python', 'echo "python $*" >>"$STUB_LOG"; echo "{}"; echo env-problem >&2; '
               'exit "${STUB_ENV_RC:-0}"\n')
    write_exec(src / 'fastvideo-kernel' / 'build.sh', 'echo "build.sh $*" >>"$STUB_LOG"; exit "${STUB_BUILD_RC:-0}"\n')
    write_exec(src / 'lane.sh', 'sleep "${STUB_LANE_SLEEP:-0}"; echo lane ran; exit "${STUB_LANE_RC:-0}"\n')
    env = {**os.environ, 'PATH': f"{stubs}{os.pathsep}{os.environ['PATH']}", 'HOME': str(tmp_path),
           'STUB_LOG': str(tmp_path / 'calls.log'), 'REPRO_SRC': str(src), 'REPRO_OUT': str(out),
           'REPRO_VENV': str(venv), 'REPRO_KERNEL_WHEEL_DIR': str(tmp_path / 'kernel-wheel'),
           'REPRO_LANE_SCRIPT': 'lane.sh', 'HOST_UID': str(os.getuid()), 'HOST_GID': str(os.getgid())}

    def run(**overrides):
        result = subprocess.run(['bash', str(CODE / 'ci_lane_container.sh')], env={**env, **overrides},
                                capture_output=True, text=True)
        calls = (tmp_path / 'calls.log').read_text().splitlines() if (tmp_path / 'calls.log').exists() else []
        return result, out, calls

    return run, tmp_path


def test_all_stages_then_lane_status_is_recorded(lane):
    run, tmp_path = lane
    result, out, calls = run(STUB_LANE_RC='3')
    assert result.returncode == 0, result.stderr
    assert not (out / 'failed_stage').exists()
    assert (out / 'lane_exit_code').read_text().strip() == '3'
    assert (out / 'lane.log').read_text().strip() == 'lane ran'
    assert calls[0] == 'uv pip install --no-deps -e .'
    assert calls[1] == f"build.sh --wheel-dir {tmp_path / 'kernel-wheel'}"
    assert calls[2] == f"python -P {CODE / 'ci_lane_environment.py'}"
    assert (out / 'environment.json').read_text().strip() == '{}'


@pytest.mark.parametrize('overrides,stage,code', [
    ({'STUB_UV_RC': '1'}, 'install', 10),
    ({'STUB_BUILD_RC': '1'}, 'kernel_build', 11),
    ({'STUB_ENV_RC': '1'}, 'environment', 12),
])
def test_a_failed_setup_stage_stops_before_the_lane(lane, overrides, stage, code):
    run, _ = lane
    result, out, _ = run(**overrides)
    assert result.returncode == code
    assert (out / 'failed_stage').read_text().strip() == stage
    assert not (out / 'lane_exit_code').exists() and not (out / 'lane.log').exists()
    if stage == 'environment':
        assert (out / 'environment.err').read_text().strip() == 'env-problem'


def test_lane_timeout_is_recorded_as_124(lane):
    run, _ = lane
    result, out, _ = run(STUB_LANE_SLEEP='5', REPRO_LANE_TIMEOUT='1')
    assert result.returncode == 0
    assert (out / 'lane_exit_code').read_text().strip() == '124'


def test_environment_check_writes_json_and_fails_outside_the_checkout(tmp_path):
    # Nothing under tmp_path is installed, so every provenance check fails; the JSON must still be complete.
    env = {**os.environ, 'REPRO_SRC': str(tmp_path / 'src'), 'REPRO_KERNEL_WHEEL_DIR': str(tmp_path / 'wheel')}
    result = subprocess.run([sys.executable, '-P', str(CODE / 'ci_lane_environment.py')], env=env, capture_output=True,
                            text=True)
    assert result.returncode == 1
    record = json.loads(result.stdout)
    assert record['checks_passed'] is False and record['problems']
    assert any('not the checkout' in p for p in record['problems'])
    assert any('not from a wheel built in' in p for p in record['problems'])
    assert result.stderr.strip().splitlines()[-len(record['problems']):] == record['problems']


def test_compare_tree_reports_missing_and_different_files(tmp_path):
    source, installed = tmp_path / 'source', tmp_path / 'installed'
    for root in (source, installed):
        (root / 'pkg').mkdir(parents=True)
        (root / 'same.py').write_text('a = 1\n')
    (source / 'pkg' / 'changed.py').write_text('b = 2\n')
    (installed / 'pkg' / 'changed.py').write_text('b = 3\n')
    (source / 'only_in_source.py').write_text('c = 4\n')
    (source / 'data.cu').write_text('not python\n')
    result = compare_tree(source, installed)
    assert result['files'] == 3
    assert result['missing'] == ['only_in_source.py']
    assert result['differ'] == ['pkg/changed.py']
