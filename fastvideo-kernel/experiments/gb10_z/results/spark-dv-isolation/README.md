# GB10 dV isolation: split baseline versus forward-consistent dV

Six cases share identical Q/K/V/dO bytes between modes: BF16, batch 1, three heads, head dimension 128, non-causal QAT and the SM121 split-path test flags. Lengths are 2112 and 31200; seeds are 0, 11 and 42. `FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV` was unset and the joined route was checked false in both worktrees. All outputs were finite. Raw inputs and outputs remain local and are not committed.

Environment: GB10 SM121; Torch 2.12.0+cu130; Triton 3.7.0; FlashAttention 2.8.3+cu130torch2.12; CUDA toolkit 13.0; driver 580.173.02. Both installations were verified editable from their respective worktrees:

- S `08e5dd90fa351bc7f2716858ac78549a96a9a383`: imported `/home/yiyispark/Desktop/FastVideo-GB10-Rebased-Split/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py`, SHA256 `2c745adf56be3be4bb4ccfe8e4f1aa47fbe47a7440d3ca194b9cdd07c697fbc8`.
- D `dcfd77fd4cd0469226a72e729bfcef40fb38e1b3`: imported `/home/yiyispark/Desktop/FastVideo-GB10-Rebased-DV/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py`, SHA256 `3c533ec41f1844287653e63080f8fbfb6abe629b58a139057a002667bbcbc678`.

## Results

- Control 0: all six outputs in all six cases were bitwise equal across consecutive S repeats; this gate passed before D was run.
- Control 1: D with `FASTVIDEO_ATTN_QAT_SM121_FWD_DV=0` matched S bitwise for all six outputs in all cases.
- Main: D with both dV variables unset matched S bitwise for O, M, dQ, dK and STE in all cases. dV differed in all cases; relative L2 uses S dV as denominator.
- D `save` versus `recompute`: dV was bitwise equal for both lengths and all three seeds.

| Tokens | Seed | dV unequal elements | dV max absolute difference | dV relative L2 vs S |
|---:|---:|---:|---:|---:|
| 2112 | 0 | 810414 | 0.172973633 | 2.14370532 |
| 2112 | 11 | 810432 | 0.177642822 | 2.19384317 |
| 2112 | 42 | 810429 | 0.182067871 | 2.21586574 |
| 31200 | 0 | 11980795 | 0.0659179688 | 55.7870121 |
| 31200 | 11 | 11980797 | 0.0578613281 | 66.8220478 |
| 31200 | 42 | 11980797 | 0.0625 | 51.5548783 |

The four comparison JSON files give bitwise equality, unequal logical-element count and maximum absolute difference for every tensor and case, plus dV relative L2. Equality compares raw representation, including signed zero. Relative L2 uses float64 accumulation; a small S norm may produce a large ratio. No tolerance was used.

Control 0's original comparison preceded an added input-fingerprint check in `compare.py`; its numerical comparison is unchanged. `input-fingerprints.json` is the later independent verification that all saved inputs still match the fingerprints recorded by every mode. The original control0 JSON is preserved.

## Reproduction commands

Each GPU run used the existing service/lock wrapper to stop compatibility and vLLM services, hold both GPU locks, verify the editable import, and restore prior service states. The thermal guard remained active. Routing variables were removed from the child environment before each run; `run_case.py` sets only the requested D dV variable internally.

```bash
RESULTS=/home/yiyispark/Desktop/FastVideo-GB10-DV-Isolation/fastvideo-kernel/experiments/gb10_z/results/spark-dv-isolation
RAW=/home/yiyispark/Desktop/gb10-dv-isolation-2026-11-01
WRAPPER=/home/yiyispark/Desktop/gb10-rebased-prs-2026-11-01/attempt-2/run-test.sh
S=/home/yiyispark/Desktop/FastVideo-GB10-Rebased-Split
D=/home/yiyispark/Desktop/FastVideo-GB10-Rebased-DV
ES=/home/yiyispark/Desktop/gb10-rebased-prs-2026-11-01/env-split
ED=/home/yiyispark/Desktop/gb10-rebased-prs-2026-11-01/env-dv
env -u FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV -u FASTVIDEO_ATTN_QAT_SM121_FWD_DV -u FASTVIDEO_ATTN_QAT_SM121_DV_STATS "$WRAPPER" "$S" "$ES" "$RAW/split.local.log" "$RESULTS/run_case.py" --mode split --artifact-root "$RAW/artifacts" --repo "$S"
"$ES/bin/python" "$RESULTS/compare.py" --artifact-root "$RAW/artifacts" --comparison control0
# Continue only if control0 succeeds.
for MODE in legacy default save recompute; do
  env -u FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV -u FASTVIDEO_ATTN_QAT_SM121_FWD_DV -u FASTVIDEO_ATTN_QAT_SM121_DV_STATS "$WRAPPER" "$D" "$ED" "$RAW/$MODE.local.log" "$RESULTS/run_case.py" --mode "$MODE" --artifact-root "$RAW/artifacts" --repo "$D"
done
for CHECK in control1 main save-recompute; do
  "$ES/bin/python" "$RESULTS/compare.py" --artifact-root "$RAW/artifacts" --comparison "$CHECK"
done
```

Actual execution ran control1 immediately after legacy, and main immediately after default. Use a fresh raw directory to reproduce. No production source was changed. Verify committed artifacts with `sha256sum -c SHA256SUMS`.
