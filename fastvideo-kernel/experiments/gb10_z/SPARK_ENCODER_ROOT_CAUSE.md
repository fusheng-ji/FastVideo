# Spark task: why does the negative-prompt text encoder survive `del`?

You are the agent on the DGX Spark (GB10, SM121). Read `SPARK_RETEST_PRS.md` for rules and pitfalls (they apply unchanged;
in particular do not set `CUDA_VISIBLE_DEVICES=''`). Everything you measure is a new replication. Do not edit production code.

## Question

The PR `pr/release-neg-prompt-encoder` adds `gc.collect()` and `torch.cuda.empty_cache()` after `del text_encoder, ...` in
`encode_negative_prompt`. Earlier batches showed that without it 194 encoder tensors (21.163 GiB) stay alive after encoding, and with it they die.
We do not know **why** they stayed alive. The PR text currently says "encoder modules can sit in reference cycles", which is a hypothesis.
Three explanations fit the observation, and they need different wording (and maybe a different fix):

| H | Explanation | What it predicts |
|---|---|---|
| H1 | Reference cycle: objects are only freed by the cyclic collector | tensors alive after `del`; dead after `gc.collect()` alone |
| H2 | A live external reference (a global, a cache, a returned tensor, a hook) | tensors still alive after `gc.collect()`; they die only after that reference is removed |
| H3 | Not Python lifetime: the CUDA caching allocator holds freed blocks | `memory_allocated` drops after `del` already; only `memory_reserved` stays high |

`memory_allocated` counts live tensors only. `torch.cuda.empty_cache()` never changes it; if only that call "helped", the earlier
effect was on `reserved`, not allocated. Check which counter the earlier 21.163 GiB figure used (see `results/spark-pr-retest/`).

## Procedure

Write the probe under `fastvideo-kernel/experiments/gb10_z/results/spark-encoder-root-cause/`. It must follow the body of
`fastvideo/train/utils/negative_prompt.py::encode_negative_prompt` on branch `pr/release-neg-prompt-encoder` step by step,
but without the release helper: same loader, same `inference_args` (with `text_encoder_cpu_offload = False`), same tokenizer,
same forward under `set_forward_context`, same Wan2.1 model and prompt, bf16 as in the formal run. One fresh process per trial.
Take weak references to every parameter and buffer of the encoder before the forward. At each checkpoint record: the number of live
weak references, `memory_allocated`, `memory_reserved`.

Checkpoints, in order:
1. after the forward, before `del`;
2. after `del text_encoder, tokenizer, outputs, text_inputs` and nothing else;
3. after `gc.collect()` only (do NOT call `empty_cache` yet);
4. after `torch.cuda.empty_cache()`.
Run the trial 3 times, and also once with `gc.disable()` before step 1 so a background collection cannot interfere.

If tensors are still alive at checkpoint 3 (H2), do not guess: list the referrers with `gc.get_referrers` for one parameter, recursively
two levels, printing only types and, for dicts and modules, the key or class name (no paths or hostnames). Also try, one at a time,
dropping `embeds`/`mask`, clearing `outputs.attention_mask`, and leaving the `set_forward_context` block, and report which (if any)
frees the tensors.

For H1, show the cycle: with `gc.set_debug(gc.DEBUG_SAVEALL)` after `gc.collect()`, report the types of garbage objects that reference
encoder modules (a short summary, not a dump).

## Decision reading (do not tune toward it)

- Alive at 2, dead at 3: H1. The PR wording is right; `gc.collect()` is the necessary part; `empty_cache` only trims `reserved`.
- Alive at 3, dead after removing a specific reference: H2. Report the reference; the fix should remove it instead of calling `gc.collect()`.
- `memory_allocated` already low at 2: H3. The earlier 194-tensor count disagrees, so state the discrepancy and stop.
- Dead only after step 4: report it; this contradicts the usual PyTorch behaviour, so rerun twice before reporting.

## Report

`results/spark-encoder-root-cause/`: one JSON per trial and a README with versions, exact commands, the checkpoint table for every trial,
which hypothesis the data supports, and failures as failures. No weights, paths, hostnames or video IDs. Commit to
`spark/gb10-encoder-root-cause` (based on `validate/encoder-root-cause`) and push. Answer: which of H1/H2/H3 holds, and does the claim
"empty_cache releases the encoder" hold for allocated or only for reserved memory?
