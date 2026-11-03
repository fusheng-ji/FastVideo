#!/usr/bin/env bash
# Reproduce the Buildkite kernel lane (`.buildkite/scripts/lanes/kernel_tests.sh`) on a GB10 inside the CI image
# (`docker/Dockerfile.spark-ci`), once per git ref, and summarize which tests fail where.
#
# Usage:
#   ci_kernel_lane_repro.sh --out DIR --check-only [--wheel-dir DIR] [--dry-run]
#   ci_kernel_lane_repro.sh --out DIR --lock LOCK_A --lock LOCK_B [--wheel-dir DIR] [--image TAG] [--dry-run] [NAME=SHA ...]
# --check-only stops after the PyPI check and the FlashAttention wheel download (no GPU work, no locks needed).
# Without NAME=SHA arguments the four refs below are run in this order.
#
# HTTPS from the Spark is intermittent (TLS handshakes to GitHub and PyPI time out on both the bridge and the host
# network), and the Dockerfile's single download of the 242 MB FlashAttention wheel from GitHub stalled the image build.
# The host therefore downloads that wheel with retries and resume into --wheel-dir, checks its published sha256, and
# serves it to the build on 127.0.0.1 (FLASH_ATTN_WHEEL build argument); the installed file is byte-identical. uv in the
# lane containers gets long timeouts and retries for its PyPI downloads.
set -euo pipefail

# The wheel named in main's Dockerfile.spark-ci (GitHub release asset digest). CI_REPRO_* overrides are for tests.
WHEEL_SHA256="${CI_REPRO_WHEEL_SHA256:-3274144f92481df3d7deefae087ed5449b06389507de1f97d9b8109e053c6ff3}"
WHEEL_SIZE="${CI_REPRO_WHEEL_SIZE:-242333202}"
RETRY_DELAY="${CI_REPRO_RETRY_DELAY:-15}"
PYPI_URLS=(https://pypi.org/simple/setuptools/ https://files.pythonhosted.org/)

DEFAULT_REFS=(
    "main=de873eae2dab2d1c72c33d465663f4d77c4f819b"
    "pr1948=ad8fc52b527f534fa12b0b0b89e63c1346289324"
    "pr1934=628fa0860cd573ca3050a4f38cb221f879b619b7"
    "pr1938=3e5ba79a13b1a1306262beb5b1bbdfcf33496e86"
)
BASE_IMAGE="ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest"
SUBMODULES=(fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk)
IMAGE="fastvideo-ci-gb10:repro"
WHEEL_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/fastvideo-ci-repro"
OUT=""
LOCKS=()
DRY_RUN=0
CHECK_ONLY=0
REFS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --lock) LOCKS+=("$2"); shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --wheel-dir) WHEEL_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --check-only) CHECK_ONLY=1; shift ;;
        *=*) REFS+=("$1"); shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ ${#REFS[@]} -gt 0 ]] || REFS=("${DEFAULT_REFS[@]}")
[[ -n "$OUT" ]] || { echo "--out is required" >&2; exit 2; }
if (( ! CHECK_ONLY )); then
    [[ ${#LOCKS[@]} -eq 2 && "${LOCKS[0]}" != "${LOCKS[1]}" ]] || { echo "two distinct --lock paths are required" >&2; exit 2; }
fi
[[ ! -e "$OUT" ]] || { echo "$OUT exists; use a fresh output directory" >&2; exit 2; }

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$CODE" rev-parse --show-toplevel)"
REMOTE="${CI_REPRO_REMOTE:-https://github.com/hao-ai-lab/FastVideo.git}"
MAIN_SHA=""
for ref in "${REFS[@]}"; do [[ "${ref%%=*}" != main ]] || MAIN_SHA="${ref#*=}"; done
[[ -n "$MAIN_SHA" ]] || { echo "a main=SHA ref is required (the baseline and the Dockerfile source)" >&2; exit 2; }

run() {
    if (( DRY_RUN )); then printf 'DRY-RUN:'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}
# run_to FILE CMD...: stdout to FILE.  run_log FILE CMD...: stdout and stderr to FILE.
run_to() {
    local file="$1"; shift
    if (( DRY_RUN )); then printf 'DRY-RUN:'; printf ' %q' "$@"; printf ' > %q\n' "$file"; else "$@" >"$file"; fi
}
run_log() {
    local file="$1"; shift
    if (( DRY_RUN )); then printf 'DRY-RUN:'; printf ' %q' "$@"; printf ' &> %q\n' "$file"; else "$@" >"$file" 2>&1; fi
}

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
exec > >(tee -a "$OUT/driver.log") 2>&1
echo "repro start $(date -u +%FT%TZ) image=$IMAGE refs=${REFS[*]} check_only=$CHECK_ONLY"

# 1. Preconditions: Docker with GPU access on a GB10.
if (( ! DRY_RUN )); then
    docker info >/dev/null
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "host GPU: $gpu"
    [[ "$gpu" == *GB10* ]] || { echo "expected a GB10 host, got: $gpu" >&2; exit 3; }
fi

# 2. Sources from hao-ai-lab/FastVideo by URL, so no particular remote name is needed (CI_REPRO_REMOTE is for tests).
for attempt in 1 2 3; do
    run git -C "$REPO" fetch "$REMOTE" main pull/1934/head pull/1938/head pull/1948/head && break
    (( attempt < 3 )) || { echo "git fetch failed 3 times" >&2; exit 4; }
    sleep "$RETRY_DELAY"
done
for ref in "${REFS[@]}"; do
    sha="${ref#*=}"
    (( DRY_RUN )) || git -C "$REPO" cat-file -e "$sha^{commit}" || { echo "commit $sha not fetched" >&2; exit 4; }
done
mkdir -p "$OUT/image-context"
git -C "$REPO" show "$MAIN_SHA:docker/Dockerfile.spark-ci" >"$OUT/image-context/Dockerfile.spark-ci"
WHEEL_URL="$(sed -n 's/^ARG FLASH_ATTN_WHEEL=//p' "$OUT/image-context/Dockerfile.spark-ci")"
[[ -n "$WHEEL_URL" ]] || { echo "no FLASH_ATTN_WHEEL in Dockerfile.spark-ci" >&2; exit 4; }
WHEEL_NAME="${WHEEL_URL##*/}"
WHEEL="$WHEEL_DIR/$WHEEL_NAME"

# 3. Base image, PyPI reachability from a container (the lane setup installs build dependencies from PyPI), and the
# FlashAttention wheel on the host.
if (( DRY_RUN )) || ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    run docker pull "$BASE_IMAGE"
fi
if (( ! DRY_RUN )); then
    docker image inspect "$BASE_IMAGE" \
        --format '{"base_repo_digests": {{json .RepoDigests}}, "base_created": {{json .Created}}}' >"$OUT/base-image.json"
fi
probe_urls=()
for url in "${PYPI_URLS[@]}"; do probe_urls+=(--url "$url"); done
# --entrypoint skips the image's CUDA banner, which would otherwise precede the JSON on stdout.
if ! run_to "$OUT/network-pypi.json" docker run --rm --network=host --entrypoint /opt/venv/bin/python \
        -v "$CODE:/probe:ro" "$BASE_IMAGE" /probe/container_network_probe.py "${probe_urls[@]}" --timeout 60 --attempts 3; then
    echo network >"$OUT/failed_stage"
    echo "PyPI check failed after 3 attempts per URL; see $OUT/network-pypi.json" >&2
    exit 5
fi

wheel_ok() {
    [[ -f "$1" && "$(stat -c %s "$1")" == "$WHEEL_SIZE" && "$(sha256sum "$1" | cut -d' ' -f1)" == "$WHEEL_SHA256" ]]
}
if (( DRY_RUN )); then
    run curl -fL --connect-timeout 60 --speed-time 120 --speed-limit 1024 -C - -o "$WHEEL.part" "$WHEEL_URL"
elif wheel_ok "$WHEEL"; then
    echo "wheel: verified cached copy $WHEEL"
else
    mkdir -p "$WHEEL_DIR"
    for attempt in $(seq 1 30); do
        have="$(stat -c %s "$WHEEL.part" 2>/dev/null || echo 0)"
        (( have < WHEEL_SIZE )) || break
        echo "wheel: attempt $attempt, have $have of $WHEEL_SIZE bytes"
        status=0
        curl -fL --connect-timeout 60 --speed-time 120 --speed-limit 1024 -C - -o "$WHEEL.part" "$WHEEL_URL" \
            >>"$OUT/wheel-download.log" 2>&1 || status=$?
        now="$(stat -c %s "$WHEEL.part" 2>/dev/null || echo 0)"
        if (( status == 33 || (status == 0 && now == have) )); then
            echo "wheel: server did not resume (curl exit $status); restarting from byte 0"
            rm -f "$WHEEL.part"
        elif (( status != 0 )); then
            echo "wheel: curl exited $status at $now bytes"
            sleep "$RETRY_DELAY"
        fi
    done
    if ! wheel_ok "$WHEEL.part"; then
        echo wheel >"$OUT/failed_stage"
        if (( $(stat -c %s "$WHEEL.part" 2>/dev/null || echo 0) < WHEEL_SIZE )); then
            echo "wheel: still incomplete after 30 attempts; $WHEEL.part is kept and a rerun resumes it" >&2
        else
            rm -f "$WHEEL.part"
            echo "wheel: downloaded file does not match sha256 $WHEEL_SHA256; deleted it" >&2
        fi
        exit 7
    fi
    mv "$WHEEL.part" "$WHEEL"
    echo "wheel: downloaded and verified $WHEEL"
fi
(( DRY_RUN )) || printf '{"wheel": "%s", "sha256": "%s", "size": %s}\n' "$WHEEL_NAME" "$WHEEL_SHA256" "$WHEEL_SIZE" \
    >"$OUT/wheel.json"
if (( CHECK_ONLY )); then
    echo "check-only: PyPI check passed and wheel verified; stopping before the image build"
    exit 0
fi

# 4. The CI image: Dockerfile.spark-ci from main on top of the base image. The build fetches the verified wheel from a
# server on 127.0.0.1 (hence --network=host) instead of GitHub; a stalled build is bounded.
if (( DRY_RUN )) || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
    if (( ! DRY_RUN )); then
        python3 -m http.server --bind 127.0.0.1 --directory "$WHEEL_DIR" "$port" >"$OUT/wheel-server.log" 2>&1 &
        server_pid=$!
        trap 'kill "$server_pid" 2>/dev/null || true' EXIT
        for _ in $(seq 1 50); do
            curl -sf -r 0-0 -o /dev/null "http://127.0.0.1:$port/$WHEEL_NAME" && break
            sleep 0.2
        done
    fi
    if ! run_log "$OUT/image-build.log" timeout 45m docker build --network=host --progress=plain \
            --build-arg "FLASH_ATTN_WHEEL=http://127.0.0.1:$port/$WHEEL_NAME" \
            -f "$OUT/image-context/Dockerfile.spark-ci" -t "$IMAGE" "$OUT/image-context"; then
        echo image_build >"$OUT/failed_stage"
        tail -50 "$OUT/image-build.log" >&2
        exit 6
    fi
    if (( ! DRY_RUN )); then
        kill "$server_pid" 2>/dev/null || true
        trap - EXIT
    fi
fi
if (( ! DRY_RUN )); then
    docker image inspect "$IMAGE" --format '{"image_id": {{json .Id}}, "created": {{json .Created}}}' >"$OUT/ci-image.json"
fi

# 5. One detached worktree per ref at the exact SHA, with the kernel's header submodules checked out on the host:
# inside the container the worktree's .git points to a host path, so build.sh finds no git checkout and uses the
# headers already present (the same pinned commits CI's checkout gets). Later worktrees borrow objects from the first.
first=""
for ref in "${REFS[@]}"; do
    name="${ref%%=*}"; sha="${ref#*=}"
    src="$OUT/src/$name"
    mkdir -p "$OUT/runs/$name"
    run git -C "$REPO" worktree add --detach "$src" "$sha"
    for sm in "${SUBMODULES[@]}"; do
        donor=""
        if [[ -n "$first" ]]; then
            donor="$OUT/src/$first/$sm"
        elif [[ "$(git -C "$REPO/$sm" rev-parse --show-toplevel 2>/dev/null)" == "$REPO/$sm" ]]; then
            donor="$REPO/$sm"
        fi
        reference=()
        if [[ -n "$donor" ]]; then
            (( DRY_RUN )) || donor="$(git -C "$donor" rev-parse --absolute-git-dir)"
            reference=(--reference "$donor")
        fi
        run git -C "$src" submodule update --init --recursive "${reference[@]}" -- "$sm"
    done
    if (( ! DRY_RUN )); then
        [[ "$(git -C "$src" rev-parse HEAD)" == "$sha" ]] || { echo "$name is not at $sha" >&2; exit 4; }
        status="$(git -C "$src" submodule status -- "${SUBMODULES[@]}")"
        ! grep -q '^[-+U]' <<<"$status" || { echo "$name submodules not at their pinned commits: $status" >&2; exit 4; }
        { echo "head $sha"; echo "$status"; } >"$OUT/runs/$name/source.txt"
    fi
    first="${first:-$name}"
done

# 6. The lane, per ref, in a fresh container: install the checkout, build its kernel from source, run the lane script.
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
                  "uv": subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip(),
                  "kernel_imported_from_checkout": True}, indent=2))
PY
timeout 90m bash .buildkite/scripts/lanes/kernel_tests.sh >/out/lane.log 2>&1
status=$?
echo $status >/out/lane_exit_code
exit 0
'
for ref in "${REFS[@]}"; do
    name="${ref%%=*}"
    echo "run $name start $(date -u +%FT%TZ)"
    run flock --exclusive "${LOCKS[0]}" flock --exclusive "${LOCKS[1]}" \
        docker run --rm --gpus all --network=host --ipc=host --ulimit memlock=-1 --shm-size=16g \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        -e FASTVIDEO_CI_BACKEND=spark4 -e MASTER_PORT=29650 -e UV_HTTP_TIMEOUT=300 -e UV_HTTP_RETRIES=10 \
        -e PYTEST_ADDOPTS="-p no:cacheprovider --junitxml=/out/junit.xml" \
        -v "$OUT/src/$name:/src" -v "$OUT/runs/$name:/out" -w /src \
        "$IMAGE" bash -c "$INNER" || echo "container for $name exited $?"
    echo "run $name end $(date -u +%FT%TZ)"
done

# 7. Summary: per-ref counts and the failures that do not occur on main.
run python3 "$CODE/ci_lane_summary.py" "$OUT/runs" --baseline main --output "$OUT/summary"
echo "repro end $(date -u +%FT%TZ)"
