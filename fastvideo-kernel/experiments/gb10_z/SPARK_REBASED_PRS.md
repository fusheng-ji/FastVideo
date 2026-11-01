# DGX Spark task: validate three GB10 PRs after rebase onto current main

Audience: the agent on the DGX Spark (GB10, SM121). Read-only task: do not change production code.
If a test fails, record the failure and stop that item; do not fix it.

## What changed and why this run exists

PRs #1934, #1938 and #1948 (hao-ai-lab/FastVideo) were validated on `main@2b164405`. Their branches are
40 commits behind `main`, and `microscope-transformer-tests` failed on all three in CI. They have been
rebased locally onto the current `upstream/main` with no conflicts, but nothing has run on a GB10 since.
Local checks (RTX PRO 6000, SM120, torch 2.14, triton 3.8) cannot cover SM121.

| PR | Branch to test | Content |
|---|---|---|
| #1934 | `rebase/gb10-qat-split-pv` | GB10 keeps the split P@V path in Attn-QAT training |
| #1938 | `rebase/gb10-dv` | Forward-consistent dV on GB10; stacked on #1934 (contains its commits) |
| #1948 | `rebase/vsa64-gb10` | Fused VSA64 layout enabled on `(12, 1)` |
| baseline | `upstream/main` | Control |

Fetch: `git fetch origin rebase/gb10-qat-split-pv rebase/gb10-dv rebase/vsa64-gb10` and
`git fetch upstream main`. Use one `git worktree` per ref so installs and results do not mix.

## Environment (record it in the report)

Torch / triton / FA2 versions, driver, CUDA, `git rev-parse HEAD` for every ref, and whether
`fastvideo-kernel` is installed editable from that worktree (it must be; otherwise you test another tree).
Original validation used torch 2.12.0+cu130, triton 3.7.0, driver 580.173.02. A different environment is a
new replication; say so.

## Step 1: control on `upstream/main`

Run the same commands as below on `upstream/main` first. Anything that fails there is a main-wide
failure and not caused by the PRs. Report it separately.

## Step 2: #1934 on `rebase/gb10-qat-split-pv`

```bash
pytest fastvideo-kernel/tests/test_attn_qat_train_routing.py fastvideo-kernel/tests/test_attn_qat_train.py -k "not fused" -q
```

Earlier result on the old base: 47 passed, 4 skipped, 12 deselected. Do not tune toward it; report what you see.
Expected on `upstream/main`: the two `test_sm120_joined_pv_forward_backward_matches_split_path` cases
fail on GB10 (this is the bug the PR fixes). Confirm that, and confirm they pass on the PR branch.

## Step 3: #1938 on `rebase/gb10-dv`

```bash
pytest fastvideo-kernel/tests -q
```

Earlier result on the old base: 124 passed, 23 skipped (SM100-only). Also run
`fastvideo-kernel/tests/test_attn_qat_sm121_dv.py` alone and confirm it runs in both
`FASTVIDEO_ATTN_QAT_SM121_DV_STATS=save` and `recompute`. Confirm that the legacy path is selected with
`FASTVIDEO_ATTN_QAT_SM121_FWD_DV=0`.

Numerical regression, same input on this branch and on `upstream/main`: the PR claims O, M, dQ, dK and
the STE output are bit-identical to the legacy kernel, and only dV changes. Check those five bitwise at
2112x2112 and 31200 tokens (3 heads) and report which differ.

## Step 4: #1948 on `rebase/vsa64-gb10`

```bash
pytest fastvideo-kernel/tests/test_vsa_tile_layout.py fastvideo/tests/attention/test_vsa64_fused_layout.py -q
```

Earlier result: 16 passed, including the two real-device parity cases. The routing test is parameterized
over `(10, 0)` and `(12, 1)`; confirm both IDs ran.

## Step 5: the CI failure

CI job `microscope-transformer-tests` failed on all three PRs (Buildkite builds 1699, 1695, 1701). Its logs
are not readable from here. Run `pytest fastvideo/tests/transformers -q` on `upstream/main` and on each branch
and report which tests fail and where the sets differ. If the sets are identical, the failure is not
caused by the PRs.

## Report

Push results to branch `spark/gb10-rebased-prs-results`, based on `handoff/gb10-rebased-prs`, under
`fastvideo-kernel/experiments/gb10_z/results/spark-rebased-prs/`: one JSON or text log per step plus a short
`README.md` with the environment, exact commands and the pass/fail table for each ref. Failures stay as
failures. Do not commit weights, paths, hostnames or video IDs. Report the branch and commit hash back, and
answer: (1) does each PR pass on its own after the rebase, (2) which failures also occur on `upstream/main`,
(3) did any bitwise check in Step 3 differ.
