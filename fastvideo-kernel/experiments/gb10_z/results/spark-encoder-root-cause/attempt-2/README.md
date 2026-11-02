# GB10 negative encoder lifetime: attempt 2

Baseline merged from `5c4ecc92399ba13478310d6039c27e770dddaa23`; measured source commit `2289b992cce22eff3997921de4a166f395105676`. This is a new GB10 replication. Torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3+cu130torch2.12, NVIDIA driver 580.173.02. Prior failed attempt remains unchanged. No production code or probe was modified.

CPU checks: probe tests 12 passed, gb10_z suite 142 passed, negative-prompt release test 1 passed. CUDA remained visible. The probe uses the unchanged formal training entrypoint and stops before training. Each of the six processes used a fresh output directory, two GPU locks, and had competing services stopped before the trial and restored after it; temperature guard remained active. All six completed; there were no failed trials. One-row 8192-token data and the Wan2.1-T2V-1.3B Diffusers model were supplied locally.

Commands (local paths represented by placeholders):

```bash
python -m pytest fastvideo-kernel/experiments/gb10_z/tests/test_encoder_lifetime_probe.py -q --confcutdir=fastvideo-kernel/experiments/gb10_z
python -m pytest fastvideo-kernel/experiments/gb10_z/tests -q --confcutdir=fastvideo-kernel/experiments/gb10_z
python -m pytest fastvideo/tests/train/utils/test_negative_prompt_release.py -q --confcutdir=fastvideo/tests/train
P=fastvideo-kernel/experiments/gb10_z/code/encoder_lifetime_probe.py
n=0
for trial in fixed natural natural natural gc-disabled cycle; do
  n=$((n + 1))
  python "$P" --mode "$trial" --model "$MODEL" --data "$ONE_ROW_8192" \
    --output "$OUT/$n-$trial" --lock "$LOCK_A" --lock "$LOCK_B" || break
done
```

In the actual execution a service-management wrapper stopped and restored competing services around each invocation; the probe held both locks. `trackers: [none]` is set by the probe.

| Trial | Checkpoint | Live tensors / 194 | allocated bytes | reserved bytes |
| --- | --- | ---: | ---: | ---: |
| 1-fixed | after_load | 194 | 28701702656 | 33728495616 |
| 1-fixed | after_forward | 194 | 28737356288 | 33728495616 |
| 1-fixed | after_del_inside_function | 194 | 28739451904 | 33728495616 |
| 1-fixed | after_release_helper_inside_function | 0 | 6015810560 | 6400507904 |
| 1-fixed | after_return | 0 | 6015810560 | 6400507904 |
| 1-fixed | after_gc_collect | 0 | 6015810560 | 6400507904 |
| 1-fixed | after_empty_cache | 0 | 6015810560 | 6400507904 |
| 2-natural | after_load | 194 | 28701702656 | 33728495616 |
| 2-natural | after_forward | 194 | 28737356288 | 33728495616 |
| 2-natural | after_del_inside_function | 194 | 28739451904 | 33728495616 |
| 2-natural | after_return | 194 | 28739451904 | 33728495616 |
| 2-natural | after_gc_collect | 0 | 6015810560 | 33728495616 |
| 2-natural | after_empty_cache | 0 | 6015810560 | 6400507904 |
| 3-natural | after_load | 194 | 28701702656 | 33728495616 |
| 3-natural | after_forward | 194 | 28737356288 | 33728495616 |
| 3-natural | after_del_inside_function | 194 | 28739451904 | 33728495616 |
| 3-natural | after_return | 194 | 28739451904 | 33728495616 |
| 3-natural | after_gc_collect | 0 | 6015810560 | 33728495616 |
| 3-natural | after_empty_cache | 0 | 6015810560 | 6400507904 |
| 4-natural | after_load | 194 | 28701702656 | 33728495616 |
| 4-natural | after_forward | 194 | 28737356288 | 33728495616 |
| 4-natural | after_del_inside_function | 194 | 28739451904 | 33728495616 |
| 4-natural | after_return | 194 | 28739451904 | 33728495616 |
| 4-natural | after_gc_collect | 0 | 6015810560 | 33728495616 |
| 4-natural | after_empty_cache | 0 | 6015810560 | 6400507904 |
| 5-gc-disabled | after_load | 194 | 28701702656 | 33728495616 |
| 5-gc-disabled | after_forward | 194 | 28737356288 | 33728495616 |
| 5-gc-disabled | after_del_inside_function | 194 | 28739451904 | 33728495616 |
| 5-gc-disabled | after_return | 194 | 28739451904 | 33728495616 |
| 5-gc-disabled | after_gc_collect | 0 | 6015810560 | 33728495616 |
| 5-gc-disabled | after_empty_cache | 0 | 6015810560 | 6400507904 |
| 6-cycle | after_load | 194 | 28701702656 | 33728495616 |
| 6-cycle | after_forward | 194 | 28737356288 | 33728495616 |
| 6-cycle | after_del_inside_function | 194 | 28739451904 | 33728495616 |
| 6-cycle | after_return | 194 | 28739451904 | 33728495616 |
| 6-cycle | after_gc_collect | 0 | 6015810560 | 33728495616 |
| 6-cycle | after_empty_cache | 0 | 6015810560 | 6400507904 |

All three natural trials and the GC-disabled control agree: 194 tensors survive the function return and none survive explicit `gc.collect()`, supporting **H1**, a Python cyclic-GC lifetime effect. For a live tensor, the recorded referrer chain is `weight` in a parameters dictionary, itself under `_parameters` of `VocabParallelEmbedding`. The cycle-mode `DEBUG_SAVEALL` collection recorded 4,933 garbage objects, including 194 `Parameter`, 25 `VocabParallelEmbedding`, and 2,398 `OrderedDict` objects. The top-level encoder module was already dead and was not in garbage, so the probe could not report `garbage_referrers_of_module` or establish the complete cycle edge; the retained submodule/tensor chain and collection behavior are the available evidence.

Tracked tensors total 22,723,641,344 bytes (21.163 GiB). In natural trial 2, `after_return` -> `after_gc_collect` allocated changes 28,739,451,904 -> 6,015,810,560 bytes (−21.163 GiB), while reserved stays 33,728,495,616 bytes. `after_gc_collect` -> `after_empty_cache` allocated remains 6,015,810,560 bytes, while reserved changes 33,728,495,616 -> 6,400,507,904 bytes (−25.451 GiB). Thus `gc.collect()` releases the live allocation; `empty_cache()` trims reserved cached memory in this experiment.

In the fixed trial, 194 tensors were live just after `del`, and 0 were live after the PR helper, still inside the function; allocated fell by 21.163 GiB. The helper works here, although the probe does not separately isolate the two helper calls within that checkpoint.
