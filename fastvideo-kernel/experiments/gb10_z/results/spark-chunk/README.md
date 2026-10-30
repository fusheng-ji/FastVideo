# GB10 production foreach chunk validation: blocked at source preflight

Target: `perf/train-adamw-foreach-chunk`, commit
`ce33886087f5aa3db0283c90e993adcba29259e3`.
Result branch: `spark/gb10-chunk-validation`, based directly on that commit.
Inspection date: 2026-10-30 (Europe/Berlin).

**No full-model GPU training or performance experiment was started.** The requested
historical Wan/Attn-QAT GB10 dense/save baseline is absent from this branch.
This is a source compatibility failure, **not** an observed training crash or a
measured failure of the new optimizer option. No production code was repaired,
no old kernel was transplanted, and no defaults or services were changed.

## Findings

The production optimizer option is present. `OptimizerConfig.foreach_chunk_size`
defaults to zero; a positive value selects `_ChunkedForeachAdamW`. Its `step`
scopes and restores the installed Torch foreach function. The new CPU tests pass:
**6 passed**, with 14 Torch `script_method` deprecation warnings. That verifies
CPU unit behavior, not the requested full-model performance or 200-step equality.

The target's QAT kernel differs materially from the kernel used for the previous
26.805 → 23.571 GiB result:

| Evidence | Target branch | Historical `spark/gb10-peak-profile` |
|---|---|---|
| Kernel SHA256 | `1481f97d2d4a7f74a87e1532e89c1af1a99d33a56c8f6cb65ff5d8d61b2c437e` | `477cf40132176b4e430295ca830e3e35a9928698b68ba99ba5f52dbe7ae03407` |
| SM121 forward-consistent dV gate | Absent | Present |
| SM121 saved-statistics mode and shape helpers | Absent | Present |
| `dv_maxima` history | Absent | Present |
| `_attn_bwd_dv_forward_weights` | Absent | Present |
| Autograd saved tensors | `q, k, v, o_for_bwd, M` (line 1324) | Includes the saved online-statistics path |
| Historical private experiment tools | Absent | Present |

The target does have an `ATTN_QAT_TRAIN` backend and other consumer-Blackwell
logic. Their existence does not provide the missing forward-consistent GB10
saved-statistics route. Running the legacy route would change the numerical and
memory baseline being tested. This report does not claim that every Wan or QAT
configuration on the target is unable to run.

There is also a separate reproducibility prerequisite missing:
`fastvideo/attention/utils/flash_attn_default.py` passes
`deterministic=False` to `_fa2_backward` at line 173 and has no policy that combines
it with `torch.are_deterministic_algorithms_enabled()`. Thus simply enabling
strict PyTorch mode cannot reproduce the previous FA2 determinism policy.
We did not force or monkeypatch that argument.

An additional import-source probe confirmed that `fastvideo` and its optimizer
resolved to this checkout, but its assertion that CUDA remained uninitialized
failed: importing the package initialized CUDA. That failed probe is recorded in
`import-probe.json`; no model was loaded or trained. The subsequent compatibility
check uses only AST, Git and package metadata, with no Torch import or GPU calls.

## Requested outcomes

| Experiment | Outcome |
|---|---|
| 8192, YAML chunk 0 versus 64 through the production entrypoint | Not run; prerequisite blocked. Historical memory reduction is not re-certified. |
| 31200, phase attribution | Not run; optimizer versus backward peak is unknown on this branch. |
| 8192, 200 updates per configuration | Not run; equality and first differing step are unknown. |
| Optional 16/32/64/128 scan | Not run; no best size established on this branch. |

Each has a separate JSON with null measurements, not zero-valued or fabricated
results. No retries, training checkpoint, weight, tensor, capture, video, or
private input identifier is included. Duty/Ollama/thermal services and the GPU
hold were left unchanged because no training measurement was admitted.

To unblock, an explicitly approved baseline needs both the GB10 dense/save QAT
implementation and the FA2 determinism policy used by the historical recipe,
alongside this optimizer change. Per the stop instruction, this branch does not
cherry-pick or implement those dependencies automatically.

## Environment and commands

Installed environment observed through distribution metadata: Torch
`2.12.0+cu130`, Triton `3.7.0`, FA2 `2.8.3+cu130torch2.12`, NumPy `2.5.3`;
DGX Spark / NVIDIA GB10. No new GPU-runtime performance certification is made.

From the target checkout, with `PY` set to the installed project interpreter:

```bash
git fetch origin perf/train-adamw-foreach-chunk
# Target fetched and inspected: ce33886087f5aa3db0283c90e993adcba29259e3
"$PY" -m pytest fastvideo/tests/train/utils/test_optimizer_foreach_chunk.py -q \
  --confcutdir=fastvideo/tests/train/utils

# Historical reference is needed for the source comparison, not transplanted.
git fetch origin spark/gb10-peak-profile
TOOL=fastvideo-kernel/experiments/gb10_z/results/spark-chunk
"$PY" "$TOOL/check_preflight.py" --repo . --output "$FRESH_OUTPUT/preflight.json"
```

`FRESH_OUTPUT/preflight.json` must not exist. The source checker pins both commits,
records exact helper/saved-tensor evidence and hashes, and never imports the GPU
kernel. Its blocked status is written to JSON; printing that status is not a
successful GPU acceptance result. The import probe's exact command is retained in
its JSON. No production-entrypoint command was executed or represented as passed.

`SHA256SUMS` covers these published artifacts. Historical reference commit:
`cf6327b7f4d8f1326717f5be9ef9da74fb067560`. No PR was opened.
