# Spark task: fused AdamW control for the chunked-foreach PR

You are the agent on the DGX Spark (GB10, SM121). Read `SPARK_RETEST_PRS.md` first: the harness, preconditions, pitfalls
and rules there apply unchanged. This file only adds the missing comparison. Everything you measure is a new replication.

## Why this experiment exists

`pr/foreach-chunk-adamw` lowers the optimizer-step memory peak by running foreach AdamW on chunks of at most N tensors.
Upstream already has `optimizer.fused`, which runs AdamW in one fused CUDA kernel without the foreach temporaries.
We refused `fused` + chunking but never measured plain `fused`. A reviewer will ask: why not just set `fused: true`?
The answer decides whether the chunking PR is worth opening. Chunking is only justified if `fused` is unavailable,
broken on GB10, or does not reduce memory comparably.

## Branch

Test `validate/fused-adamw-control` (this file on top of `validate/pr-pair-on-gb10` `21b2fdac3`). Do not edit production code.
Confirm from the YAML-loaded config that `training.optimizer.fused: true` reaches `torch.optim.AdamW(fused=True)` and record the
actual optimizer class, `fused` flag and `foreach` flag of every run. Run Step 0 of `SPARK_RETEST_PRS.md` first (CPU self-check,
no `CUDA_VISIBLE_DEVICES=''`).

## Arms (YAML only, no monkeypatching)

| arm | `foreach_chunk_size` | `fused` |
|---|---|---|
| F0 foreach default | 0 | false |
| F64 chunked foreach | 64 | false |
| FU fused | 0 | true |

All arms: same seed, same single repeated parquet sample (state it), `training.tracker.trackers: [none]`, encoder-release code as on the branch.

## Measurements

1. Feasibility. Does FU start, take an optimizer step and finish at 8192 and 31200 tokens on GB10? Record any exception verbatim
   (no paths or hostnames). If FU fails, stop the remaining FU steps and report that as the result.
2. Memory. 8192 tokens, 4 updates, and 31200 tokens, 2 updates; 3 randomized paired rounds with all three arms per round, each arm in a
   fresh process. Record `memory_allocated` per phase as in `SPARK_RETEST_PRS.md` Step 1, the peak allocated and peak reserved,
   the limiting phase, and the optimizer-step peak separately (reset the peak counters before the step or state that they are cumulative).
3. Numerics. FU is not expected to match foreach bitwise (different kernel). Over 200 updates at 8192 tokens, record per-update loss and
   gradient norm for F0 and FU (and F64 for reference, which must still equal F0 bitwise). Report the maximum relative difference of
   loss and of the gradient norm between FU and F0, and the first update at which they differ. Do not call this convergence.
   Optimizer state and parameter hashes are not comparable between FU and F0; do not report them as a failure.
4. Time. Record step time and optimizer-step time per arm with clock telemetry. Report as descriptive; do not claim a speedup unless
   clocks are fixed and the difference is larger than the pair-to-pair spread.

## Pre-stated reading of the result (do not tune toward it)

- FU runs, and its peak allocated at 8192 is within 0.2 GiB of F64 or lower: the chunking PR is not justified; say so.
- FU runs but its peak allocated is more than 0.2 GiB above F64: chunking still reduces memory beyond `fused`; report the gap.
- FU fails or is unsupported on GB10: chunking is the only memory option; report the exact failure.
- At 31200 tokens the peak is set by backward for F0 and F64; if FU gives the same allocated peak, say that no arm helps there.

## Rules

Everything in `SPARK_RETEST_PRS.md` Rules applies: fresh empty directory per run, no automatic retries, both GPU locks held, services
stopped and restored, randomized pairing, failures recorded as failures. Do not open PRs, change defaults, or commit weights, videos,
captures, paths, hostnames or video IDs.

## Report

Under `fastvideo-kernel/experiments/gb10_z/results/spark-fused-control/`: one JSON per experiment and a README with versions,
exact commands, a table of the three arms for allocated peak, reserved peak, limiting phase and optimizer-step peak at both lengths,
and direct answers to: (1) does `fused` work on GB10 through the formal entrypoint, (2) how does its memory compare with F0 and F64
at 8192 and 31200, (3) how far do loss and gradient norm diverge from F0 over 200 updates, (4) which pre-stated reading applies.
Commit to `spark/gb10-fused-control` (based on `validate/fused-adamw-control`) and push.
