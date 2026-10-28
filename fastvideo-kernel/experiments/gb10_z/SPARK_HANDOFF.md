# DGX Spark task brief: GB10 Attn-QAT saved-statistics memory

Audience: the agent running on the DGX Spark (GB10, SM121). Read `README.md` in this directory first.

## Problem
With the GB10 forward-consistent dV, the forward saves per-KV-tile online maxima (`dv_maxima`) of shape
`(B, H, ceil(N_KV/32), N_Q)` in fp32. Memory therefore grows with `N_Q * N_KV`.
`41ef03d7e` adds the private "Z" format (P64 packed event slots, no spill-map lookup when nothing spills)
in `fastvideo_kernel/triton_kernels/attn_qat_train.py` to compress it.

## Baseline
- Code: branch `perf/attn-qat-gb10-stats-overflow`, commit `41ef03d7e7453dedde129439a08109149abbe515`.
- Evidence (not in this branch): `origin/pr_asset` at `94eb140ffd359f8546868c938853971167f7f6ff`,
  directory `gb10-qat/z-handoff-2026-10-28/`:

```bash
git fetch origin pr_asset
git worktree add --detach ../FastVideo-Z-assets origin/pr_asset
cd ../FastVideo-Z-assets/gb10-qat/z-handoff-2026-10-28 && sha256sum -c SHA256SUMS
```

## Findings from review (verify, do not assume)
1. Z is not wired into the product path. `_sm121_dv_stats_format()`, `_sm121_local_replay_enabled()` and
   `_sm121_packed_tuning()` are constant stubs returning dense/off; Z is enabled only by monkeypatching in
   `code/local_policy.py::kernel_policy`.
2. No memory win is shown in the evidence. In `historical-L64/performance-failed.json` (8192 tokens, G1)
   dense and candidate both peak at about 29.54 GB allocated. The `strict-Z` 31200-token results only show
   clock equality. Measure whether the saved statistics are even the peak at 8192.
3. Old L64 overhead was 19.7% to 41.1% (failed). Z overhead on the Spark is unmeasured.
4. Ordinary-mode statistical equivalence is NOT established (3 seeds, 20 updates, reused validation set).
5. Portable entrypoints were not GPU-revalidated; a different environment is a new replication.

## Experiments, in priority order
1. **Memory and time.** For dense, `FASTVIDEO_ATTN_QAT_SM121_DV_STATS=recompute`, and Z, at 8192, 16384 and 31200
   tokens: peak allocated/reserved, step time, and statistics bytes (`_sm121_local_cache_bytes` vs
   `dv_dense_storage_bytes`). Report whether Z reduces peak memory and by how much.
2. **dV accuracy.** Relative L2 of dV against the independent Torch reference (existing SM121 kernel tests are
   the template; dense baseline is about 3.5e-4 on real Wan2.1 activations).
3. **Edge behavior.** Overflow to spill-history fallback, and `ceil(N_KV/32) > 65535` (must fall back or fail loudly).
4. **Candidates, only if 1 shows Z or dense is a real bottleneck:** a documented env switch replacing the stubs;
   lower-precision or coarser-granularity statistics. Compare against `recompute` first.

## Rules
- Do not change kernel defaults until experiment 1 is reported.
- Use an empty output directory per attempt; no automatic retries; honor the GPU lock convention in `README.md`.
- Do not commit model weights, videos, captures, local paths, hostnames or video IDs.
- Label everything as a new replication on new hardware.

## Report back
Put results under `fastvideo-kernel/experiments/gb10_z/results/spark/`: one JSON per experiment plus a short
`README.md` stating the environment (torch/triton/FA2 versions), the exact commands, and failures as failures.

## CPU self-check
```bash
python -m pytest fastvideo-kernel/experiments/gb10_z/tests -q --confcutdir=fastvideo-kernel/experiments/gb10_z
python -m pytest fastvideo/tests/attention/test_fa2_determinism_policy.py -q --confcutdir=fastvideo/tests/attention
```
