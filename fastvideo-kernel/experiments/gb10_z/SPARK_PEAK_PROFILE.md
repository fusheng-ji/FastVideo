# Spark task: find what sets the post-backward memory peak

Goal: at 8192 tokens (dense policy, same Wan2.1-1.3B recipe as `results/spark/`), identify which allocations
form the 26.8 GiB peak after backward, and whether any can be reduced without changing results.

1. Break the peak down by phase (end of forward, end of backward, after grad finite check, after clip,
   after optimizer step, after zero_grad): `torch.cuda.memory_allocated` plus a `torch.cuda.memory._record_memory_history`
   snapshot at the peak. Attribute bytes to parameters, gradients, optimizer state, master weights, temporaries.
2. Test each suspect in isolation, one change per run, fresh output directory: foreach vs for-loop grad norm/clip,
   fused vs unfused optimizer, gradient dtype, releasing grads before the finite check.
3. For every change report peak allocated/reserved, step time and parameter-hash equality vs unchanged dense.
4. Fix the GPU clock or report clock telemetry, stop competing services (Duty, Ollama) for the run, randomize order.

Rules: do not change kernel defaults; failures recorded as failures; no paths, hostnames or video IDs in commits.
Report under `fastvideo-kernel/experiments/gb10_z/results/spark-peak/` on a new branch `spark/gb10-peak-profile`.
