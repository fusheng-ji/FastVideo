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

## What changed since the first attempt (`spark/gb10-ci-kernel-lane` @ `735f1e032`)

The first attempt stalled in the image build at the FlashAttention wheel download. Containers on Docker's default bridge
network opened TCP connections to GitHub but timed out in the TLS handshake, while the host itself downloaded fine. The
script now:

- runs every container, including the image build, on the host network (`--network=host`);
- checks HTTPS from a container before any build (`code/container_network_probe.py`: the wheel URL from the Dockerfile,
  PyPI index and file host) on the host network (must pass) and on the bridge network (recorded only), and records the
  host and `docker0` MTUs in `network-mtu.txt`;
- has `--check-only`, which stops after that check, needs no locks and no GPU work;
- bounds the image build at 45 minutes with `--progress=plain` output in `image-build.log`, recording
  `failed_stage=image_build` instead of hanging.

A second defect was found and fixed before it was hit: a fresh worktree has no kernel header submodules, and inside the
container git cannot see the worktree (its `.git` points to a host path), so `build.sh` would have failed on missing CUTLASS
headers for every ref. The host now checks out `fastvideo-kernel/include/cutlass` and `include/tk` in each worktree at the
pinned commits (later worktrees borrow objects from the first) and records them in `source.txt`.

## What the script does

`fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh`:

1. Checks Docker and that the host GPU is a GB10.
2. Fetches `main` and `pull/1934|1938|1948/head` from `https://github.com/hao-ai-lab/FastVideo.git` and checks each SHA exists.
3. Pulls the base image if absent, records its digest (`base-image.json`), records MTUs, runs the network check
   (`network-host.json`, `network-bridge.json`). A failed host-network check stops with `failed_stage=network`, exit 5.
4. Builds `fastvideo-ci-gb10:repro` from `main`'s `docker/Dockerfile.spark-ci` unless that tag exists (`image-build.log`,
   `ci-image.json`). A failed or timed-out build stops with `failed_stage=image_build`, exit 6.
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
2. Run the CPU checks (do not set `CUDA_VISIBLE_DEVICES=''`); all 10 must pass (the dry-run test skips only if `main` has not
   been fetched yet; fetch it first so it runs):

```bash
git fetch https://github.com/hao-ai-lab/FastVideo.git main
python -m pytest fastvideo-kernel/experiments/gb10_z/tests/test_ci_lane_summary.py fastvideo-kernel/experiments/gb10_z/tests/test_container_network_probe.py fastvideo-kernel/experiments/gb10_z/tests/test_ci_kernel_lane_repro.py -q --confcutdir=fastvideo-kernel/experiments/gb10_z
```

3. Confirm `docker run --rm --gpus all ubuntu:22.04 true` works. If Docker or GPU access is unavailable, stop and report that;
   do not fall back to running outside a container, because the point is the CI image.
4. Network check, with a fresh directory:

```bash
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT-check" --check-only
```

   It must end with `check-only: network check passed`. If it exits 5, stop and report `network-host.json`,
   `network-bridge.json` and `network-mtu.txt`; do not change Docker or network settings.
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
Do not retry and do not change anything between refs. If the script exits 5 or 6, stop and report `failed_stage` and the
last 50 lines of `image-build.log` (exit 6) or the network files (exit 5). If no output appears for more than 60 minutes
during a lane, record the last `lane.log` lines before stopping.

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

Under `fastvideo-kernel/experiments/gb10_z/results/spark-ci-kernel-lane/`, replacing the first attempt's files and keeping
`attempt.json` renamed to `attempt-1.json`: `summary.md`, `summary.json`, `base-image.json`, `ci-image.json`,
`network-host.json`, `network-bridge.json`, `network-mtu.txt`, and per ref `source.txt`, `environment.json`, `junit.xml`,
`lane_exit_code`, `failed_stage` (if present), and the failing tests' traceback excerpts from `lane.log`. Do not commit the
full logs, the worktrees, local paths or hostnames (remove them from `network-mtu.txt` if present; interface names may stay).
A `README.md` with the host versions, the exact commands, and failures as failures. Commit to `spark/gb10-ci-kernel-lane` on
top of `735f1e032`, merging `validate/ci-kernel-lane-repro` first so the branch carries the scripts that produced the results,
and push. Answer:

1. Did the reproduction match CI on `main` and `pr1948` (both pass)?
2. Which tests fail on `pr1934` and on `pr1938` but not on `main`, with their assertion messages?
3. If nothing fails: the base image digest used.
4. Did the bridge-network check fail while the host-network check passed, and what are the two MTUs?
