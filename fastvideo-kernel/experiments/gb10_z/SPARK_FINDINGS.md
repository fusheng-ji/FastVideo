# Spark findings: saved dV statistics are not the whole-model memory bottleneck

Source: branch `spark/gb10-stats-results` at `e30fc1214` (GB10, Torch 2.12.0+cu130, Triton 3.7.0, FA2 2.8.3).
Wan2.1-T2V-1.3B, full checkpointing, G1; two measured updates per run. Reported by the Spark agent, not reproduced elsewhere.

| tokens | dense peak GiB | recompute | Z | Z step delta | recompute step delta |
|---:|---:|---:|---:|---:|---:|
| 8192 | 26.81 | 26.81 | 26.81 | +1.9% | +4.8% |
| 16384 | 26.80 | 26.80 | 26.80 | +16.6% | +19.2% |
| 31200 | 30.19 | 28.83 | 29.20 | +13.5% | +29.8% |

- Z cuts the per-layer statistics cache by about 54% at 8192 (96.4 to 44.7 MiB) but saves nothing at the model peak.
  The 8192 and 16384 peaks occur after backward (gradient checks, clipping, optimizer), not in attention.
- At 31200 Z saves 3.3% and recompute 4.5% of peak allocated; Z reserved memory is slightly above dense.
- Step times are confounded: GPU clock fell from 1391 MHz (dense) to 981 MHz (recompute, Z) at 31200 and run order was not randomized.
- Numerics: parameter hashes match across policies; dV relative L2 vs Torch reference 3.1e-4 / 4.4e-4 / 6.2e-4.
- Overflow fallback and the 65535-KV-tile guard behave as designed (the latter checked on CPU routing only).

## Decision
Keep dense as the default; do not wire Z into the product path. Next target is the post-backward peak,
see `SPARK_PEAK_PROFILE.md`.
