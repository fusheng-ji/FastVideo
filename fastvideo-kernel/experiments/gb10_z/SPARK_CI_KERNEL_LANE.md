# Spark task: reproduce the CI kernel lane failure of #1934 and #1938

You are the agent on the DGX Spark (GB10, SM121). The reproduction is scripted and CPU-tested; run it as below and report.
Do not edit production code, the scripts, or any PR branch. Locks and services follow the Rules in `SPARK_RETEST_PRS.md`.

## Question

Buildkite's `microscope-kernel-tests` fails on #1934 (split P@V on GB10) and #1938 (forward-consistent dV, stacked on #1934)
and passes on `main` and on #1948. The logs are not readable from outside Buildkite. The lane runs
`pytest fastvideo-kernel/tests/ -vs` (`.buildkite/scripts/lanes/kernel_tests.sh`), on the GB10 runner `spark4` when it is
connected, otherwise on GB200. #1934 only changes routing on 12.x devices, so the failure most likely happens on the GB10 runner,
where these PRs add GB10-only tests. The same tests passed in our earlier Spark runs, which used our own environment, not the
CI image (`docker/Dockerfile.spark-ci` on `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest`).
This run repeats the lane in that image to find the failing tests.

## What changed since the earlier attempts (`spark/gb10-ci-kernel-lane` @ `735f1e032`, `c49633178`)

Attempt 1 stalled in the image build at the Dockerfile's download of the 242 MB FlashAttention wheel from GitHub.
Attempt 2's network check showed that HTTPS from the Spark is intermittent rather than broken on Docker's bridge network:
TLS handshakes timed out on both the bridge and the host network, to GitHub and to PyPI, with no consistent pattern, and
both MTUs are 1500. The script now:

- downloads the wheel on the host with `curl`, resuming across up to 30 attempts (restarting from byte 0 if the server does
  not resume), into a cache directory (`--wheel-dir`, default `~/.cache/fastvideo-ci-repro`), and checks it against the size
  and sha256 GitHub publishes for that release asset (`242333202` bytes,
  `3274144f92481df3d7deefae087ed5449b06389507de1f97d9b8109e053c6ff3`). A verified copy is reused by later runs;
- serves the verified wheel on `127.0.0.1` during the image build and passes it as the Dockerfile's `FLASH_ATTN_WHEEL`
  build argument, so the build needs no GitHub access; the installed file is byte-identical to CI's;
- checks PyPI (index and file host) from a container with three attempts per URL before downloading or building, since the
  lane setup installs build dependencies from PyPI (`network-pypi.json`, now plain JSON: the image's CUDA banner had been
  mixed into the earlier files);
- gives `uv` in the lane containers `UV_HTTP_TIMEOUT=300` and `UV_HTTP_RETRIES=10`, and records the `uv` version;
- has `--check-only`, which stops after the PyPI check and the wheel download, needs no locks and no GPU work;
- bounds the image build at 45 minutes with `--progress=plain` output in `image-build.log`, recording
  `failed_stage=image_build` instead of hanging.

A defect was also fixed before it was hit: a fresh worktree has no kernel header submodules, and inside the container git
cannot see the worktree (its `.git` points to a host path), so `build.sh` would have failed on missing CUTLASS headers for
every ref. The host now checks out `fastvideo-kernel/include/cutlass` and `include/tk` in each worktree at the pinned commits
(later worktrees borrow objects from the first) and records them in `source.txt`.

## What the script does

`fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh`:

1. Checks Docker and that the host GPU is a GB10.
2. Fetches `main` and `pull/1934|1938|1948/head` from `https://github.com/hao-ai-lab/FastVideo.git` and checks each SHA exists.
3. Pulls the base image if absent and records its digest (`base-image.json`). Checks PyPI from a container
   (`network-pypi.json`); failure stops with `failed_stage=network`, exit 5. Downloads and verifies the wheel
   (`wheel-download.log`, `wheel.json`); failure stops with `failed_stage=wheel`, exit 7.
4. Builds `fastvideo-ci-gb10:repro` from `main`'s `docker/Dockerfile.spark-ci` unless that tag exists, with the wheel served
   locally (`image-build.log`, `ci-image.json`). A failed or timed-out build stops with `failed_stage=image_build`, exit 6.
5. Creates one detached worktree per ref at the exact SHA, checks out the two header submodules on the host:

   | name | SHA | role |
   |---|---|---|
   | `main` | `de873eae2dab2d1c72c33d465663f4d77c4f819b` | baseline; passes in CI |
   | `pr1948` | `ad8fc52b527f534fa12b0b0b89e63c1346289324` | control; changes the kernel package and passes in CI |
   | `pr1934` | `628fa0860cd573ca3050a4f38cb221f879b619b7` | fails in CI |
   | `pr1938` | `3e5ba79a13b1a1306262beb5b1bbdfcf33496e86` | fails in CI |

6. For each ref, under both locks, in a fresh container (`--gpus all`, `--network=host`, `FASTVIDEO_CI_BACKEND=spark4` as on
   the CI runner): installs the checkout (`uv pip install --no-deps -e .`), builds the kernel from that checkout
   (`fastvideo-kernel/build.sh`, as CI does for PRs that change the kernel), verifies that `fastvideo_kernel` imports from the
   checkout, then runs the lane script with a 90-minute timeout and JUnit output. Files per ref: `source.txt`, `install.log`,
   `kernel_build.log`, `environment.json`, `lane.log`, `junit.xml`, `lane_exit_code`, and `failed_stage` if a setup stage
   failed.
7. Writes `summary.json` and `summary.md`: per-ref counts (passed, failed, error, skipped, xfailed) and, for each ref, the
   failures that `main` does not have, with the first line of each assertion message.

## Step 0: preconditions (services can stay up)

1. Check out `validate/ci-kernel-lane-repro` at the commit in the instruction; working tree clean.
2. Run the CPU checks (do not set `CUDA_VISIBLE_DEVICES=''`); all 14 must pass (the dry-run test skips only if `main` has not
   been fetched yet; fetch it first so it runs):

```bash
git fetch https://github.com/hao-ai-lab/FastVideo.git main
python -m pytest fastvideo-kernel/experiments/gb10_z/tests/test_ci_lane_summary.py fastvideo-kernel/experiments/gb10_z/tests/test_container_network_probe.py fastvideo-kernel/experiments/gb10_z/tests/test_ci_kernel_lane_repro.py -q --confcutdir=fastvideo-kernel/experiments/gb10_z
```

3. Confirm `docker run --rm --gpus all ubuntu:22.04 true` works. If Docker or GPU access is unavailable, stop and report that;
   do not fall back to running outside a container, because the point is the CI image.
4. PyPI check and wheel download, with a fresh directory (the download can take a while on an intermittent connection):

```bash
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT-check" --check-only
```

   It must end with `check-only: PyPI check passed and wheel verified`. Do not change Docker or network settings.
   - Exit 5 (PyPI unreachable after 3 attempts per URL): stop and report `network-pypi.json`.
   - Exit 7 (wheel still incomplete after 30 attempts, or complete with the wrong digest, which the script deletes):
     report the message and `wheel-download.log` (last 30 lines, after `tr '\r' '\n'`). You may run the same
     `--check-only` command once more with a fresh `--out`; an incomplete download resumes. If it still fails, stop: the
     user can copy the wheel into the wheel directory from another machine, and the script verifies it.
   - The wheel stays in the wheel directory, so Step 1 does not download it again.
5. Dry run, and read the printed commands:

```bash
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT-dry" --lock "$LOCK_A" --lock "$LOCK_B" --dry-run
```

## Step 1: run

Stop competing services first and restore them afterwards, as in earlier batches. Use a fresh `OUT`. Expect up to 45 minutes
for the image build, then four kernel builds and four lane runs; each lane can take up to 90 minutes.

```bash
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT" --lock "$LOCK_A" --lock "$LOCK_B"
```

The script does not stop when one ref's lane fails; a failed setup stage for one ref is recorded in its `failed_stage`.
Do not retry and do not change anything between refs. If the script exits 5, 6 or 7, stop and report `failed_stage` and
`network-pypi.json` (exit 5), the last 50 lines of `image-build.log` (exit 6) or of `wheel-download.log` (exit 7). If a ref
fails at `install` or `kernel_build`, include the last 50 lines of that log: a PyPI timeout there is a network failure, not
a result. If no output appears for more than 60 minutes during a lane, record the last `lane.log` lines before stopping.

## Step 2: read the result (stated before the run)

Validity first:

- `main` and `pr1948` must have `junit.xml`. If `main` itself shows failures, the reproduction environment differs from the CI
  runner; report `main`'s failures and still report the PR-only failures, labelled as such.
- `environment.json` must show capability `[12, 1]`; `source.txt` must show the ref's SHA and both submodules with a leading
  space (checked out at the pinned commit).

Then:

- `pr1934` and/or `pr1938` have failures that `main` lacks: reproduced. Report each test id, the assertion message, and the
  relevant `lane.log` excerpt (traceback of that test only). State whether the `pr1938` failures include the `pr1934` ones.
- No PR-only failures: not reproduced. Report that, plus the image digest, so the difference from the CI runner can be checked.
- A setup stage fails for a PR but not for `main`: report the stage and the last 50 lines of its log.

## Report

Under `fastvideo-kernel/experiments/gb10_z/results/spark-ci-kernel-lane/`: move attempt 2's files (`README.md`,
`failed_stage`, `network-host.json`, `network-bridge.json`, `network-mtu.txt`) into `attempt-2/` unchanged, keep
`attempt-1.json`, and add this run's `summary.md`, `summary.json`, `base-image.json`, `ci-image.json`, `network-pypi.json`,
`wheel.json`, and per ref `source.txt`, `environment.json`, `junit.xml`, `lane_exit_code`, `failed_stage` (if present), and
the failing tests' traceback excerpts from `lane.log`. Do not commit the wheel, the full logs, the worktrees, local paths or
hostnames. A `README.md` with the host versions, the exact commands, how many wheel download attempts were needed, and
failures as failures. Commit to `spark/gb10-ci-kernel-lane` on top of `c49633178`, merging `validate/ci-kernel-lane-repro`
first so the branch carries the scripts that produced the results, and push. Answer:

1. Did the reproduction match CI on `main` and `pr1948` (both pass)?
2. Which tests fail on `pr1934` and on `pr1938` but not on `main`, with their assertion messages?
3. If nothing fails: the base image digest used.
4. How many attempts did the wheel download need, and did any lane setup (`install`, `kernel_build`) fail on the network?
