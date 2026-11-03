# GB10 CI kernel lane: check-only stopped at network precondition

Source update: `8b1f6599729dcb374165aaa4ed187a53c38420a2`, merged onto results commit `735f1e0322328057a3dddca26c53f18404a4edfb`. The prior `attempt.json` is preserved unchanged as `attempt-1.json`. Supplied scripts, summarizer, tests, production code and PR branches were not edited.

Host: NVIDIA GB10, driver 580.173.02, Docker client/server 29.6.2. CPU checks: **10 passed**. `docker run --rm --gpus all ubuntu:22.04 true` succeeded. No empty CUDA visibility setting was used.

The supplied `--check-only` invocation exited **5**, with `failed_stage` equal to `network`. It did not print `check-only: network check passed`. As required, processing stopped: no services were stopped, no Docker or network settings were changed, no dry run, image build or kernel lanes were started, and no retry was performed. Inference/compatibility services and the thermal guard remained active. No lane summaries, CI image metadata or per-ref results exist for this attempt.

Commands, using placeholders for local paths:

```bash
git fetch https://github.com/hao-ai-lab/FastVideo.git main
python -m pytest fastvideo-kernel/experiments/gb10_z/tests/test_ci_lane_summary.py fastvideo-kernel/experiments/gb10_z/tests/test_container_network_probe.py fastvideo-kernel/experiments/gb10_z/tests/test_ci_kernel_lane_repro.py -q --confcutdir=fastvideo-kernel/experiments/gb10_z
docker run --rm --gpus all ubuntu:22.04 true
fastvideo-kernel/experiments/gb10_z/code/ci_kernel_lane_repro.sh --out "$OUT-check" --check-only
```

The three supplied network output files are copied verbatim. The two files named `.json` include the base container's CUDA entrypoint banner before their JSON payload; they are therefore not standalone JSON documents. The banner warns about absent GPU support because this network-only container invocation does not request GPUs. The separate prescribed GPU preflight succeeded.

| Endpoint | Host network | Bridge network |
| --- | --- | --- |
| Pinned FlashAttention wheel | TLS handshake timeout; 79.82 s; no HTTP response | Read timeout; 30.19 s; no HTTP response |
| PyPI setuptools index | HTTP 206; 1024 bytes; 42.11 s | HTTP 206; 1024 bytes; 1.65 s |
| files.pythonhosted.org root | HTTP 404; 1.98 s; accepted as reachable by the supplied probe | TLS handshake timeout; 33.43 s; no HTTP response |

## Answers

1. **Did main and PR #1948 both pass?** Unknown: neither lane ran because Step 0 failed.
2. **Which PR-only tests fail on #1934 and #1938?** Unknown: neither PR lane ran; no test assertions are available.
3. **Base image digest:** `sha256:caf466e02a916e4f29238a69399674a60dbe3f63c430e66cb3b46e75e1489df5`, recorded by the supplied script in `base-image.json`. This is not evidence that the lanes pass or that the CI failure was not reproduced.
4. **Bridge fails while host passes?** No: both checks failed overall. Host default-route interface `enP7s7` MTU is **1500**; `docker0` MTU is **1500**, and the default-bridge MTU option is also **1500**. These observations do not establish an MTU cause.
