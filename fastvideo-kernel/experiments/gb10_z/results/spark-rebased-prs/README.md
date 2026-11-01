# GB10 rebased PR validation

Handoff commit `a197523f015084237cfab0760892c9a64d3a7100`; this is a new GB10 SM121 replication. No production code or kernel defaults were modified. Each ref used a separate worktree and editable `fastvideo-kernel` installation; imported module paths and editable metadata were checked before each test. Versions: Torch 2.12.0+cu130, Triton 3.7.0, FlashAttention 2.8.3+cu130torch2.12, CUDA 13.0, driver 580.173.02. Commits and source fingerprints are in summary.json.

Initial setup failures are preserved in preflight.json: no pytest ran before the missing temporary lock directory was restored. Two GPU locks were held for each test, competing services were stopped and restored, and the temperature guard was kept active.

| Ref | Test item | Exit | Summary |
|---|---|---:|---|
| main | main-qat | 0 | 28 passed, 2 skipped, 12 deselected, 2 xfailed in 11.28s |
| main | main-kernel | 0 | 416 passed, 85 skipped, 2 xfailed, 140 warnings in 250.98s (0:04:10) |
| main | main-dv-single | not_present | — |
| main | main-vsa | 0 | ================== 11 passed, 2 skipped, 14 warnings in 0.79s ================== |
| main | main-transformers | 2 | 14 warnings, 3 errors in 2.27s |
| main | main-bitwise-2112 | 0 | — |
| main | main-bitwise-31200 | 0 | — |
| split | split-qat | 0 | 47 passed, 4 skipped, 12 deselected in 6.73s |
| dv | dv-kernel | 0 | 502 passed, 87 skipped, 130 warnings in 176.97s (0:02:56) |
| dv | dv-single | 0 | ============================== 42 passed in 6.21s ============================== |
| dv | dv-legacy-route | 0 | — |
| dv | dv-bitwise-2112 | 0 | — |
| dv | dv-bitwise-31200 | 0 | — |
| vsa | vsa-layout | 0 | ======================= 20 passed, 14 warnings in 3.97s ======================== |
| split | split-transformers | 2 | 14 warnings, 3 errors in 0.84s |
| dv | dv-transformers | 2 | 14 warnings, 3 errors in 1.47s |
| vsa | vsa-transformers | 2 | 14 warnings, 3 errors in 1.46s |

## Required findings

- #1934: exit 0; 47 passed, 4 skipped, 12 deselected in 6.73s.
- #1938: exit 0; 502 passed, 87 skipped, 130 warnings in 176.97s (0:02:56).
- #1938 standalone save/recompute: exit 0; ============================== 42 passed in 6.21s ==============================.
- #1938 legacy routing: exit 0; see per-item evidence.
- #1948: exit 0; ======================= 20 passed, 14 warnings in 3.97s ========================.

The upstream test marks the two joined-P@V cases xfail on GB10. The split PR changes those SM120-only cases to skipped on GB10 and adds separate SM121 split-path cases; skipped cases are not counted as passes. Per-case IDs from verbose runs are preserved where available.

Transformer failure-ID sets identical across four refs: True; per-ref outcomes and limitations are in step5-comparison.json. Equality alone does not establish identical collection or coverage.

Bitwise comparison 2112 tokens: input hashes equal: True; five-preserved-tensor check: failed; output byte-hash equality: {'O': False, 'M': False, 'dQ': False, 'dK': False, 'STE': False, 'dV': False}.

Bitwise comparison 31200 tokens: input hashes equal: True; five-preserved-tensor check: failed; output byte-hash equality: {'O': False, 'M': False, 'dQ': False, 'dK': False, 'STE': False, 'dV': False}.

## Commands and limitations

Use the branch's own isolated Python environment and editable kernel install. Replace `$WORKTREE`, `$PYTHON`, `$GPU_LOCK_A`, and `$GPU_LOCK_B` with the local paths; hold both locks and ensure the GPU is idle before each command. Stop Duty/Ollama/compatibility services before each run and restore their prior state afterward. Each command has one retained attempt; input and output tensors are not shipped.

```bash
$PYTHON -m pytest fastvideo-kernel/tests/test_attn_qat_train_routing.py fastvideo-kernel/tests/test_attn_qat_train.py -k "not fused" -q
$PYTHON -m pytest fastvideo-kernel/tests -q
$PYTHON -m pytest fastvideo-kernel/tests/test_attn_qat_sm121_dv.py -q -v
$PYTHON check-route.py
$PYTHON -m pytest fastvideo-kernel/tests/test_vsa_tile_layout.py fastvideo/tests/attention/test_vsa64_fused_layout.py -q -v
$PYTHON -m pytest fastvideo/tests/transformers -q
$PYTHON bitwise.py 2112 "$OUTPUT_JSON"
$PYTHON bitwise.py 31200 "$OUTPUT_JSON"
```

Verbose output was enabled for the standalone dV and VSA tests to retain parameter IDs. The dV-only test file does not exist on upstream main. Bitwise evidence uses SHA256 of raw tensor bytes from separately executed, fixed-seed inputs; raw tensors, weights, videos and host-specific paths are excluded. The main and dV scripts use their default routes. `FASTVIDEO_ATTN_QAT_SM121_FWD_DV=0` is checked by check-route.py on the dV branch.

Per-item JSON files preserve status, commands, test summaries, observed IDs and failure identifiers. Raw full logs remain local. Verify exported artifacts with `sha256sum -c SHA256SUMS`.
