"""CPU checks for the kernel-lane reproduction summary."""

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from ci_lane_summary import summarize, to_markdown

CODE = Path(__file__).resolve().parents[1] / 'code'
JUNIT = '''<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
<testcase classname="tests.test_k" name="test_ok"/>
<testcase classname="tests.test_k" name="test_shared"><failure message="shared">trace</failure></testcase>
{extra}
<testcase classname="tests.test_k" name="test_skip"><skipped type="pytest.skip" message="skip"/></testcase>
<testcase classname="tests.test_k" name="test_xfail"><skipped type="pytest.xfail" message="known"/></testcase>
</testsuite></testsuites>
'''


def write_run(root, name, extra='', stage=None, exit_code=1):
    run = root / name
    run.mkdir(parents=True)
    if stage:
        (run / 'failed_stage').write_text(stage + '\n')
        return
    (run / 'junit.xml').write_text(JUNIT.format(extra=extra))
    (run / 'lane_exit_code').write_text(f'{exit_code}\n')


def test_failures_only_on_the_pr_are_listed(tmp_path):
    write_run(tmp_path, 'main')
    write_run(tmp_path, 'pr1934',
              extra='<testcase classname="tests.test_k" name="test_new"><error message="boom">trace</error></testcase>')
    summary = summarize(tmp_path, 'main')
    assert summary['runs']['main']['counts'] == {'passed': 1, 'failed': 1, 'error': 0, 'skipped': 1, 'xfailed': 1}
    assert summary['comparison_to_baseline']['pr1934']['only_here'] == ['tests.test_k::test_new']
    assert summary['comparison_to_baseline']['pr1934']['shared'] == ['tests.test_k::test_shared']
    assert '`tests.test_k::test_new`: boom' in to_markdown(summary)


def test_a_run_without_results_is_not_compared(tmp_path):
    write_run(tmp_path, 'main')
    write_run(tmp_path, 'pr1938', stage='kernel_build')
    summary = summarize(tmp_path, 'main')
    assert 'not_comparable' in summary['comparison_to_baseline']['pr1938']
    assert 'kernel_build' in to_markdown(summary)


def test_repro_script_dry_run_lists_every_step(tmp_path):
    main_sha = 'de873eae2dab2d1c72c33d465663f4d77c4f819b'
    if subprocess.run(['git', '-C', str(CODE), 'cat-file', '-e', f'{main_sha}^{{commit}}']).returncode:
        pytest.skip('main commit not fetched yet; the script fetches it before reading its Dockerfile')
    out = tmp_path / 'out'
    result = subprocess.run(['bash', str(CODE / 'ci_kernel_lane_repro.sh'), '--out', str(out), '--lock',
                             str(tmp_path / 'a'), '--lock', str(tmp_path / 'b'), '--dry-run'],
                            capture_output=True, text=True, check=True)
    lines = [line for line in result.stdout.splitlines() if line.startswith('DRY-RUN:')]
    assert 'fetch https://github.com/hao-ai-lab/FastVideo.git main' in lines[0]
    assert any('docker build' in line for line in lines)
    runs = [line for line in lines if 'docker run' in line]
    assert len(runs) == 4 and all('FASTVIDEO_CI_BACKEND=spark4' in line for line in runs)
    assert (out / 'image-context' / 'Dockerfile.spark-ci').read_text().startswith('# syntax=docker')
