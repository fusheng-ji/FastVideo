#!/usr/bin/env bash
# Reproduce the Buildkite kernel lane (`.buildkite/scripts/lanes/kernel_tests.sh`) on a GB10 inside the CI image
# (`docker/Dockerfile.spark-ci`), once per git ref, and summarize which tests fail where.
#
# Usage:
#   ci_kernel_lane_repro.sh --out DIR --lock LOCK_A --lock LOCK_B [--image TAG] [--dry-run] [NAME=SHA ...]
# Without NAME=SHA arguments the four refs below are run in this order.
set -euo pipefail

DEFAULT_REFS=(
    "main=de873eae2dab2d1c72c33d465663f4d77c4f819b"
    "pr1948=ad8fc52b527f534fa12b0b0b89e63c1346289324"
    "pr1934=628fa0860cd573ca3050a4f38cb221f879b619b7"
    "pr1938=3e5ba79a13b1a1306262beb5b1bbdfcf33496e86"
)
IMAGE="fastvideo-ci-gb10:repro"
OUT=""
LOCKS=()
DRY_RUN=0
REFS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --lock) LOCKS+=("$2"); shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *=*) REFS+=("$1"); shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ ${#REFS[@]} -gt 0 ]] || REFS=("${DEFAULT_REFS[@]}")
[[ -n "$OUT" ]] || { echo "--out is required" >&2; exit 2; }
[[ ${#LOCKS[@]} -eq 2 && "${LOCKS[0]}" != "${LOCKS[1]}" ]] || { echo "two distinct --lock paths are required" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "$OUT exists; use a fresh output directory" >&2; exit 2; }

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$CODE" rev-parse --show-toplevel)"
MAIN_SHA="${DEFAULT_REFS[0]#main=}"

run() {
    if (( DRY_RUN )); then printf 'DRY-RUN:'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
exec > >(tee -a "$OUT/driver.log") 2>&1
echo "repro start $(date -u +%FT%TZ) image=$IMAGE refs=${REFS[*]}"

# 1. Preconditions: Docker with GPU access on a GB10.
if (( ! DRY_RUN )); then
    docker info >/dev/null
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "host GPU: $gpu"
    [[ "$gpu" == *GB10* ]] || { echo "expected a GB10 host, got: $gpu" >&2; exit 3; }
fi

# 2. Sources from hao-ai-lab/FastVideo by URL, so no particular remote name is needed.
run git -C "$REPO" fetch https://github.com/hao-ai-lab/FastVideo.git main pull/1934/head pull/1938/head pull/1948/head
for ref in "${REFS[@]}"; do
    sha="${ref#*=}"
    (( DRY_RUN )) || git -C "$REPO" cat-file -e "$sha^{commit}" || { echo "commit $sha not fetched" >&2; exit 4; }
done

# 3. The CI image: Dockerfile.spark-ci from main on top of the published CUDA 13 dev image.
if (( DRY_RUN )) || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    mkdir -p "$OUT/image-context"
    git -C "$REPO" show "$MAIN_SHA:docker/Dockerfile.spark-ci" >"$OUT/image-context/Dockerfile.spark-ci"
    run docker pull ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest
    run docker build -f "$OUT/image-context/Dockerfile.spark-ci" -t "$IMAGE" "$OUT/image-context"
fi
if (( ! DRY_RUN )); then
    docker image inspect ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest \
        --format '{"base_repo_digests": {{json .RepoDigests}}, "base_created": {{json .Created}}}' >"$OUT/base-image.json"
    docker image inspect "$IMAGE" --format '{"image_id": {{json .Id}}, "created": {{json .Created}}}' >"$OUT/ci-image.json"
fi

# 4. One detached worktree per ref, at the exact SHA.
for ref in "${REFS[@]}"; do
    name="${ref%%=*}"; sha="${ref#*=}"
    run git -C "$REPO" worktree add --detach "$OUT/src/$name" "$sha"
    if (( ! DRY_RUN )); then
        [[ "$(git -C "$OUT/src/$name" rev-parse HEAD)" == "$sha" ]] || { echo "$name is not at $sha" >&2; exit 4; }
    fi
done

# 5. The lane, per ref, in a fresh container: install the checkout, build its kernel from source, run the lane script.
INNER='
set -uo pipefail
source /opt/venv/bin/activate
export PATH=/root/.local/bin:$PATH
trap "chown -R $HOST_UID:$HOST_GID /src /out" EXIT
uv pip install --no-deps -e . >/out/install.log 2>&1 || { echo install >/out/failed_stage; exit 10; }
(cd fastvideo-kernel && ./build.sh) >/out/kernel_build.log 2>&1 || { echo kernel_build >/out/failed_stage; exit 11; }
python - >/out/environment.json <<"PY" || { echo environment >/out/failed_stage; exit 12; }
import importlib.metadata as md, json, subprocess, torch, triton, fastvideo_kernel
from fastvideo_kernel.triton_kernels import attn_qat_train
assert fastvideo_kernel.__file__.startswith("/src/"), fastvideo_kernel.__file__
assert attn_qat_train.__file__.startswith("/src/"), attn_qat_train.__file__
versions = {}
for name in ("torch", "triton", "flash-attn", "flashinfer-python", "fastvideo-kernel"):
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"versions": versions, "capability": list(torch.cuda.get_device_capability()),
                  "device": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                  "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "kernel_imported_from_checkout": True}, indent=2))
PY
timeout 90m bash .buildkite/scripts/lanes/kernel_tests.sh >/out/lane.log 2>&1
status=$?
echo $status >/out/lane_exit_code
exit 0
'
for ref in "${REFS[@]}"; do
    name="${ref%%=*}"
    mkdir -p "$OUT/runs/$name"
    echo "run $name start $(date -u +%FT%TZ)"
    run flock --exclusive "${LOCKS[0]}" flock --exclusive "${LOCKS[1]}" \
        docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --shm-size=16g \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        -e FASTVIDEO_CI_BACKEND=spark4 -e MASTER_PORT=29650 \
        -e PYTEST_ADDOPTS="-p no:cacheprovider --junitxml=/out/junit.xml" \
        -v "$OUT/src/$name:/src" -v "$OUT/runs/$name:/out" -w /src \
        "$IMAGE" bash -c "$INNER" || echo "container for $name exited $?"
    echo "run $name end $(date -u +%FT%TZ)"
done

# 6. Summary: per-ref counts and the failures that do not occur on main.
run python3 "$CODE/ci_lane_summary.py" "$OUT/runs" --baseline main --output "$OUT/summary"
echo "repro end $(date -u +%FT%TZ)"
