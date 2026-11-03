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

## What the script does

`fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh`:

1. Checks Docker and that the host GPU is a GB10.
2. Fetches `main` and `pull/1934|1938|1948/head` from `https://github.com/hao-ai-lab/FastVideo.git` and checks each SHA exists.
3. Builds `fastvideo-ci-gb10:repro` from `main`'s `docker/Dockerfile.spark-ci` after pulling the base image, unless that tag
   already exists. Records the base image digest (`base-image.json`) and the built image id (`ci-image.json`).
4. Creates one detached worktree per ref at the exact SHA:

   | name | SHA | role |
   |---|---|---|
   | `main` | `de873eae2dab2d1c72c33d465663f4d77c4f819b` | baseline; passes in CI |
   | `pr1948` | `ad8fc52b527f534fa12b0b0b89e63c1346289324` | control; changes the kernel package and passes in CI |
   | `pr1934` | `628fa0860cd573ca3050a4f38cb221f879b619b7` | fails in CI |
   | `pr1938` | `3e5ba79a13b1a1306262beb5b1bbdfcf33496e86` | fails in CI |

5. For each ref, under both locks, in a fresh container (`--gpus all`, `FASTVIDEO_CI_BACKEND=spark4` as on the CI runner):
   installs the checkout (`uv pip install --no-deps -e .`), builds the kernel from that checkout (`fastvideo-kernel/build.sh`,
   as CI does for PRs that change the kernel), verifies that `fastvideo_kernel` imports from the checkout, then runs the lane
   script with a 90-minute timeout and JUnit output. Files: `install.log`, `kernel_build.log`, `environment.json`, `lane.log`,
   `junit.xml`, `lane_exit_code`, and `failed_stage` if a setup stage failed.
6. Writes `summary.json` and `summary.md`: per-ref counts (passed, failed, error, skipped, xfailed) and, for each ref, the
   failures that `main` does not have, with the first line of each assertion message.

## Step 0: preconditions

1. Check out `validate/ci-kernel-lane-repro` at the commit in the instruction; working tree clean.
2. Run the CPU checks (do not set `CUDA_VISIBLE_DEVICES=''`); all must pass:

```bash
python -m pytest fastvideo-kernel/experiments/gb10_z/tests/test_ci_lane_summary.py -q --confcutdir=fastvideo-kernel/experiments/gb10_z
```

3. Confirm `docker run --rm --gpus all ubuntu:22.04 true` works. If Docker or GPU access is unavailable, stop and report that;
   do not fall back to running outside a container, because the point is the CI image.
4. Dry run, and read the printed commands:

```bash
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT-dry" --lock "$LOCK_A" --lock "$LOCK_B" --dry-run
```

## Step 1: run

Stop competing services first and restore them afterwards, as in earlier batches. Use a fresh `OUT`. Expect four kernel builds
and four lane runs; each lane can take up to 90 minutes.

```bash
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT" --lock "$LOCK_A" --lock "$LOCK_B"
```

The script does not stop when one ref's lane fails; a failed setup stage for one ref is recorded in its `failed_stage`.
Do not retry and do not change anything between refs. If the image pull or build fails, stop and report the error.

## Step 2: read the result (stated before the run)

Validity first:

- `main` and `pr1948` must have `junit.xml`. If `main` itself shows failures, the reproduction environment differs from the CI
  runner; report `main`'s failures and still report the PR-only failures, labelled as such.
- `environment.json` must show capability `[12, 1]` and the head SHA of that ref.

Then:

- `pr1934` and/or `pr1938` have failures that `main` lacks: reproduced. Report each test id, the assertion message, and the
  relevant `lane.log` excerpt (traceback of that test only). State whether the `pr1938` failures include the `pr1934` ones.
- No PR-only failures: not reproduced. Report that, plus the image digest, so the difference from the CI runner can be checked.
- A setup stage fails for a PR but not for `main`: report the stage and the last 50 lines of its log.

## Report

Under `fastvideo-kernel/experiments/gb10_z/results/spark-ci-kernel-lane/`: `summary.md`, `summary.json`, `base-image.json`,
`ci-image.json`, and per ref `environment.json`, `junit.xml`, `lane_exit_code`, `failed_stage` (if present), and the failing
tests' traceback excerpts from `lane.log`. Do not commit the full logs, the worktrees, local paths or hostnames. A `README.md`
with the host versions, the exact commands, and failures as failures. Commit to `spark/gb10-ci-kernel-lane` (based on
`validate/ci-kernel-lane-repro`) and push. Answer:

1. Did the reproduction match CI on `main` and `pr1948` (both pass)?
2. Which tests fail on `pr1934` and on `pr1938` but not on `main`, with their assertion messages?
3. If nothing fails: the base image digest used.
