# GB10 negative encoder lifetime: failed first attempt

Baseline: `9375944ea07ea100250af32750a83ea71b7212bd`. New GB10 replication; no historical acceptance is inherited. Torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3, driver 580.173.02.

The first fresh-process trial failed while constructing the text encoder: `AssertionError: tensor model parallel group is not initialized`. The standalone probe did not initialize FastVideo model parallel state as the formal training entrypoint does. No encoder forward or checkpoint measurement took place. Under the no-automatic-retry rule, the other trials were not run. This is a diagnostic setup failure; it is not evidence of a production regression.

Both GPU locks were held; competing services were stopped and their prior active state was restored. No production code was modified.

Command, with local paths replaced by placeholders:

```bash
flock --exclusive "$LOCK_A" flock --exclusive "$LOCK_B" python fastvideo-kernel/experiments/gb10_z/results/spark-encoder-root-cause/probe.py --config "$FORMAL_TRAINING_YAML" --output "$FRESH_OUTPUT/trial-1"
```

The YAML uses the Wan2.1-1.3B formal training recipe; no tracking credentials are needed. The first trial failed before the model was loaded.

**H1/H2/H3:** undetermined. **Does empty_cache release allocated or only reserved memory in this workload?** Not measured by this attempt; no conclusion can be drawn from its failed setup. The baseline task documentation reports prior allocated-memory observations, but this run did not replicate them. A separately authorized attempt must first initialize the same distributed state as the formal entrypoint, preserve this failed trial, and run all required fresh-process controls.
