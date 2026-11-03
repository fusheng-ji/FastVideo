"""CPU checks for ci_kernel_lane_repro.sh.

The end-to-end tests run the real script against a throwaway repository (real git, real worktrees, real submodules) with
fake `docker` and `nvidia-smi` on PATH. The fake lane container writes JUnit results only when the kernel headers are
present in the mounted checkout and the container is on the host network, so the tests cover the parts that broke or
could break on the Spark: networking, submodules, per-ref flow and the summary.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

CODE = Path(__file__).resolve().parents[1] / 'code'
SCRIPT_FILES = ('ci_kernel_lane_repro.sh', 'ci_lane_summary.py', 'container_network_probe.py')
GIT_ENV = {
    'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t', 'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t',
    'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'protocol.file.allow', 'GIT_CONFIG_VALUE_0': 'always',
}

FAKE_DOCKER = r'''#!/usr/bin/env bash
echo "$*" >>"$FAKE_STATE/docker.log"
args=" $* "
case "$1" in
    info) exit 0 ;;
    pull) touch "$FAKE_STATE/base"; exit 0 ;;
    build)
        [[ "$args" == *" --network=host "* ]] || { echo "build without host network" >&2; exit 97; }
        [[ -z "${FAKE_BUILD_FAIL:-}" ]] || { echo "simulated build failure"; exit 1; }
        touch "$FAKE_STATE/image"; exit 0 ;;
    image)
        if [[ "$3" == ghcr.io/* ]]; then state=base; else state=image; fi
        [[ -e "$FAKE_STATE/$state" ]] || exit 1
        echo '{"fake": true}'; exit 0 ;;
    network) echo 1500; exit 0 ;;
    run)
        if [[ "$args" == *container_network_probe.py* ]]; then
            if [[ "$args" == *" --network=bridge "* || -n "${FAKE_HOST_NET_FAIL:-}" ]]; then
                echo '{"all_ok": false}'; exit 1
            fi
            echo '{"all_ok": true}'; exit 0
        fi
        [[ "$args" == *" --network=host "* ]] || exit 98
        prev=""
        for a in "$@"; do
            if [[ "$prev" == -v ]]; then
                case "$a" in *:/src) src="${a%:/src}" ;; *:/out) out="${a%:/out}" ;; esac
            fi
            prev="$a"
        done
        [[ -f "$src/fastvideo-kernel/include/cutlass/include/cutlass.h" ]] || { echo kernel_build >"$out/failed_stage"; exit 11; }
        extra=""
        if grep -q gb10_only "$src/marker"; then
            extra='<testcase classname="t" name="test_gb10"><failure message="mismatch">tb</failure></testcase>'
        fi
        printf '<testsuites><testsuite><testcase classname="t" name="test_ok"/>%s</testsuite></testsuites>' "$extra" \
            >"$out/junit.xml"
        echo 0 >"$out/lane_exit_code"
        exit 0 ;;
esac
exit 0
'''


def git(cwd, *args):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env={**os.environ, **GIT_ENV}).stdout.strip()


def make_header_repo(path, header):
    (path / 'include').mkdir(parents=True)
    (path / 'include' / header).write_text('// header\n')
    git(path, 'init', '-q', '-b', 'main')
    git(path, 'add', '.')
    git(path, 'commit', '-q', '-m', 'headers')


@pytest.fixture
def repo(tmp_path):
    make_header_repo(tmp_path / 'cutlass', 'cutlass.h')
    make_header_repo(tmp_path / 'tk', 'tk.h')
    repo = tmp_path / 'repo'
    (repo / 'docker').mkdir(parents=True)
    (repo / 'docker' / 'Dockerfile.spark-ci').write_text(
        '# syntax=docker/dockerfile:1.7\nARG FLASH_ATTN_WHEEL=https://example.invalid/flash_attn.whl\n')
    (repo / 'marker').write_text('base\n')
    git(repo, 'init', '-q', '-b', 'main')
    git(repo, 'submodule', 'add', '-q', str(tmp_path / 'cutlass'), 'fastvideo-kernel/include/cutlass')
    git(repo, 'submodule', 'add', '-q', str(tmp_path / 'tk'), 'fastvideo-kernel/include/tk')
    git(repo, 'add', '.')
    git(repo, 'commit', '-q', '-m', 'base')
    base = git(repo, 'rev-parse', 'HEAD')
    (repo / 'marker').write_text('gb10_only\n')
    git(repo, 'commit', '-q', '-am', 'pr')
    pr = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'reset', '-q', '--hard', base)
    for number, sha in (('1934', pr), ('1938', pr), ('1948', base)):
        git(repo, 'update-ref', f'refs/pull/{number}/head', sha)
    code = repo / 'fastvideo-kernel' / 'experiments' / 'code'
    code.mkdir(parents=True)
    for name in SCRIPT_FILES:
        shutil.copy2(CODE / name, code / name)
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    (fake_bin / 'docker').write_text(FAKE_DOCKER)
    (fake_bin / 'nvidia-smi').write_text('#!/bin/sh\necho "NVIDIA GB10"\n')
    for tool in fake_bin.iterdir():
        tool.chmod(0o755)
    state = tmp_path / 'state'
    state.mkdir()
    return {'root': tmp_path, 'repo': repo, 'script': code / 'ci_kernel_lane_repro.sh', 'base': base, 'pr': pr,
            'state': state,
            'env': {**os.environ, **GIT_ENV, 'PATH': f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                    'FAKE_STATE': str(state), 'CI_REPRO_REMOTE': str(repo)}}


def run_script(repo, *extra, env=None, locks=True):
    out = repo['root'] / 'out'
    args = ['bash', str(repo['script']), '--out', str(out)]
    if locks:
        args += ['--lock', str(repo['root'] / 'a.lock'), '--lock', str(repo['root'] / 'b.lock')]
    args += [*extra, f"main={repo['base']}", f"pr1948={repo['base']}", f"pr1934={repo['pr']}", f"pr1938={repo['pr']}"]
    result = subprocess.run(args, env={**repo['env'], **(env or {})}, capture_output=True, text=True)
    docker_log = repo['state'] / 'docker.log'
    return result, out, docker_log.read_text().splitlines() if docker_log.exists() else []


def test_full_run_reports_pr_only_failures(repo):
    result, out, docker = run_script(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / 'summary.json').read_text())
    comparison = summary['comparison_to_baseline']
    assert comparison['pr1948']['only_here'] == []
    assert comparison['pr1934']['only_here'] == comparison['pr1938']['only_here'] == ['t::test_gb10']
    for name, sha in (('main', repo['base']), ('pr1934', repo['pr'])):
        lines = (out / 'runs' / name / 'source.txt').read_text().splitlines()
        assert lines[0] == f'head {sha}'
        assert len(lines) == 3 and all(line.startswith(' ') for line in lines[1:])
    # Later worktrees borrow submodule objects from the first instead of cloning again.
    alternates = git(out / 'src' / 'pr1934' / 'fastvideo-kernel/include/cutlass', 'rev-parse', '--git-path',
                     'objects/info/alternates')
    assert (out / 'src' / 'pr1934' / 'fastvideo-kernel/include/cutlass' / alternates).exists()
    assert json.loads((out / 'network-host.json').read_text())['all_ok']
    assert not json.loads((out / 'network-bridge.json').read_text())['all_ok']
    assert (out / 'ci-image.json').exists() and (out / 'base-image.json').exists()
    builds = [line for line in docker if line.startswith('build ')]
    assert len(builds) == 1 and '--progress=plain' in builds[0]
    lanes = [line for line in docker if line.startswith('run ') and '--gpus all' in line]
    assert len(lanes) == 4 and all('FASTVIDEO_CI_BACKEND=spark4' in line for line in lanes)


def test_host_network_failure_stops_before_the_build(repo):
    result, out, docker = run_script(repo, env={'FAKE_HOST_NET_FAIL': '1'})
    assert result.returncode == 5
    assert (out / 'failed_stage').read_text().strip() == 'network'
    assert not any(line.startswith('build ') for line in docker)


def test_check_only_needs_no_locks_and_builds_nothing(repo):
    result, out, docker = run_script(repo, '--check-only', locks=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((out / 'network-host.json').read_text())['all_ok']
    assert not any(line.startswith('build ') for line in docker)
    assert not (out / 'src').exists()


def test_failed_build_is_recorded(repo):
    result, out, _ = run_script(repo, env={'FAKE_BUILD_FAIL': '1'})
    assert result.returncode == 6
    assert (out / 'failed_stage').read_text().strip() == 'image_build'
    assert 'simulated build failure' in (out / 'image-build.log').read_text()


def test_dry_run_of_the_real_refs_lists_every_step(tmp_path):
    main_sha = 'de873eae2dab2d1c72c33d465663f4d77c4f819b'
    if subprocess.run(['git', '-C', str(CODE), 'cat-file', '-e', f'{main_sha}^{{commit}}']).returncode:
        pytest.skip('main commit not fetched yet; the script fetches it before reading its Dockerfile')
    out = tmp_path / 'out'
    result = subprocess.run(['bash', str(CODE / 'ci_kernel_lane_repro.sh'), '--out', str(out), '--lock',
                             str(tmp_path / 'a'), '--lock', str(tmp_path / 'b'), '--dry-run'],
                            capture_output=True, text=True, check=True)
    lines = [line for line in result.stdout.splitlines() if line.startswith('DRY-RUN:')]
    assert 'fetch https://github.com/hao-ai-lab/FastVideo.git main' in lines[0]
    assert any('--network=host' in line and 'container_network_probe.py' in line for line in lines)
    assert any('docker build --network=host' in line for line in lines)
    assert sum('submodule update --init' in line for line in lines) == 8
    runs = [line for line in lines if 'docker run' in line and '--gpus all' in line]
    assert len(runs) == 4 and all('--network=host' in line and 'FASTVIDEO_CI_BACKEND=spark4' in line for line in runs)
    assert (out / 'image-context' / 'Dockerfile.spark-ci').read_text().startswith('# syntax=docker')
