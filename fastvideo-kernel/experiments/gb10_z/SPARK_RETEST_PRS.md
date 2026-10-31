# Spark task: retest the two memory PRs on GB10

You are the agent on the DGX Spark (GB10, SM121). Read this file first, then `SPARK_FINDINGS.md`.
Everything you measure is a new replication; earlier acceptance is not inherited.

## What is being retested
Two independent PR branches (based on `upstream/main` `4d0a10d98`, pushed to the `origin` fork):

| PR branch | Change | Claimed effect on GB10 |
|---|---|---|
| `pr/release-neg-prompt-encoder` (`8b7c743c4`) | `encode_negative_prompt` collects garbage and empties the CUDA cache after deleting the text encoder | 194 encoder tensors (21.163 GiB) are released right after encoding |
| `pr/foreach-chunk-adamw` (`5159cafa4`, `cdd652dfd`) | `optimizer.foreach_chunk_size` (default 0 = off) runs foreach AdamW on lists of at most N tensors; cannot combine with `optimizer.fused`; YAML value is read by `load_run_config` | 8192 tokens: allocated peak -3.2 GiB; 31200 tokens: reserved peak -5.1 GiB; bitwise-identical results |

`upstream/main` does not contain the GB10 kernel (`dv_maxima`, forward-consistent dV) or the FA2 determinism policy,
so the PR branches alone cannot run this recipe. Test the **integration branch**:

- `validate/pr-pair-on-gb10` = `spark/gb10-chunk-validation-v5` (`06de11c15`) merged with `upstream/main`.
- The PR source files (`optimizer.py`, `training_config.py`, `config.py`, `negative_prompt.py` and their new tests)
  were verified identical to the two PR branches. `tests/train/utils/test_config.py` additionally has Spark's three YAML cases.
- Do not edit production code. If something needs a change, record it and stop.

## Step 0: preconditions and CPU self-check
- Confirm the kernel and FA2 policy are present (`dv_maxima`, `_resolve_fa2_deterministic`), the base is `376a1f647`, and the
  working tree is clean. Record torch, triton, flash-attn, driver versions.
- Run on an idle GPU **without** setting `CUDA_VISIBLE_DEVICES=''` (importing FastVideo loads Triton, which probes the driver):
  `fastvideo/tests/train/utils/test_negative_prompt_release.py`, `test_config.py`, `test_optimizer_foreach_chunk.py`,
  the `gb10_z` tests and the FA2 policy tests. Any failure: record the full error (no paths or hostnames) and stop.
- Smoke: one chunk=0 update at 8192 tokens through the formal entrypoint with `training.tracker.trackers: [none]`.

## Step 1: encoder release (PR 1)
8192 tokens, chunk=0, one process, 2 updates. Record, after negative encoding, whether the 194 encoder tensors still resolve to GPU
storage (weak references), `memory_allocated` per phase (after model load, after encoding, before first forward, end of forward,
end of backward, after clip, after optimizer step, after zero_grad).
Control: the same measurements at `d1235c66a` (v4, no release) in a separate fresh process on the same day.
Expected: weak references dead; allocated after encoding lower by about 21.16 GiB than the control.
If not released, list the referrer types of the encoder parameters (`gc.get_referrers`, types only) and stop this step.

## Step 2: chunked AdamW (PR 2)
Config comes from YAML (`training.optimizer.foreach_chunk_size`), not from monkeypatching. Confirm each run's actual optimizer class.
- 8192 tokens: chunk 0 and 64, 3 randomized paired rounds, 4 updates each.
- 31200 tokens: chunk 0 and 64, 3 randomized paired rounds, 2 updates each; record the limiting phase.
- 200 updates at 8192 tokens, chunk 0 and 64, fixed seed, one repeated sample (state this in the report); loss, gradient norm and
  parameter hash every 20 updates; report the first differing update, if any.
- Optional trend: 8192 tokens, chunk 16/32/64/128, one run each, no conclusions.
Pass criteria: loss, gradient norm, all 825 parameter hashes bitwise equal between chunk 0 and 64 in every pair.
Reference numbers from the earlier batch (do not tune to them): 8192 allocated 27.514 -> 24.281 GiB; 31200 allocated 32.876 GiB
for both, reserved 42.797 -> 37.705 GiB. Report differences from these without explaining them away.

## Known pitfalls from earlier batches
- Use `training.tracker.trackers: [none]`; no W&B credentials.
- Do not assert that snapshot categories sum to `memory_allocated`; record the difference instead (that assertion aborted an earlier run).
- `foreach_chunk_size` was once ignored because `load_run_config` dropped it; always verify the optimizer class.
- Phase peak counters are cumulative; reset or note them.
- Do not read timings as speedups unless clocks are fixed (record clock telemetry; earlier batches showed throttling).

## Rules
Fresh empty output directory per run; no automatic retries; hold both GPU locks; stop Duty, Ollama and compatibility services
before each run and restore them after; randomize and pair; record failures as failures and stop conclusions on any hash mismatch.
Do not change kernel defaults, open PRs, or commit weights, videos, captures, local paths, hostnames or video IDs.

## Report
Under `fastvideo-kernel/experiments/gb10_z/results/spark-pr-retest/`: one JSON per experiment plus a README with versions, exact commands,
failures, and direct answers to: was the encoder released and by how much; chunk=64 saving at 8192 and 31200 (allocated and reserved);
bitwise equality in all pairs and over 200 updates. Commit to `spark/gb10-pr-retest` (based on `validate/pr-pair-on-gb10`) and push.
