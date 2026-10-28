"""Private P64 policy and source identity; importing this module never initializes CUDA."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

SCOPE = "gb10-z-portable-calibration-v1"
POLICIES = ("dense", "Z")
CANDIDATE = "Z"
CANDIDATE_FLAGS = {"Z": (True, False), "T": (False, True)}
LOCAL_FORMAT = "packed_slots_local_replay"
KERNEL_SHA256 = "477cf40132176b4e430295ca830e3e35a9928698b68ba99ba5f52dbe7ae03407"


def digest(path: Any) -> str:
    with Path(path).open("rb") as stream:
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
        return hasher.hexdigest()


def default_manifest() -> Path:
    from portable import settings
    return settings().source_manifest


def manifest_sources(path: Any = None) -> dict[str, str]:
    document = json.loads(Path(path or default_manifest()).read_text())
    if document.get("acceptance_scope") != SCOPE or not isinstance(document.get("sources"), dict):
        raise RuntimeError("Missing local-replay source allowlist or incorrect acceptance domain")
    sources = document["sources"]
    if not sources or any(not Path(p).is_absolute() or not isinstance(h, str) or len(h) != 64
                          for p, h in sources.items()):
        raise RuntimeError("Source allowlist must contain absolute filenames and SHA256 values")
    return sources


def verify_loaded(modules: Any = None, manifest: Any = None) -> dict[str, Any]:
    sources = manifest_sources(manifest)
    if modules is None:
        modules = [
            m for name, m in list(sys.modules.items())
            if name.startswith(("fastvideo.", "fastvideo_kernel.", "gb10_qat."))
        ]
    records = {}
    for module in modules:
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = str(Path(filename).resolve())
        expected = sources.get(path)
        if expected is None or digest(path) != expected:
            raise RuntimeError(f"Loaded source outside exact frozen allowlist: {path}")
        records[module.__name__] = {"path": path, "sha256": expected}
    return records


@contextmanager
def kernel_policy(module: Any, policy: str) -> Any:
    if policy not in POLICIES:
        raise ValueError("Only dense and private C64 local replay (P64) are supported")
    names = ("_sm121_dv_stats_mode", "_sm121_dv_stats_format", "_sm121_local_replay_enabled", "_sm121_packed_tuning")
    original = {name: getattr(module, name) for name in names}
    module._sm121_dv_stats_mode = lambda: "save"
    module._sm121_dv_stats_format = lambda: ("packed_slots", 64, False) if policy == "Z" else ("dense", 0, False)
    module._sm121_local_replay_enabled = lambda: policy == "Z"
    module._sm121_packed_tuning = lambda: CANDIDATE_FLAGS[CANDIDATE] if policy == "Z" else (False, False)
    try:
        yield {
            "policy": policy,
            "capacity": 64 if policy == "Z" else 0,
            "candidate": CANDIDATE,
            "candidate_flags": module._sm121_packed_tuning()
        }
    finally:
        for name, value in original.items():
            setattr(module, name, value)


@contextmanager
def observe_statistics(module: Any, callback: Any) -> Any:
    """Observe attempted counts without replacing the final-format observer."""
    name = "_sm121_dv_stats_observer"
    original = getattr(module, name)

    def observe(metadata: Any) -> None:
        if original is not None:
            original(metadata)
        callback(metadata)

    setattr(module, name, observe)
    try:
        yield
    finally:
        setattr(module, name, original)


def compact_record(row: Any) -> bool:
    return row.get("format") == LOCAL_FORMAT and row.get("overflow_fallback") is False


def statistics_metadata(ctx: Any) -> dict[str, Any]:
    """Read scalar context attributes only; saved_tensors would trigger checkpoint replay."""
    return dict(format=ctx.dv_stats_format,
                capacity=ctx.dv_event_capacity,
                overflow_fallback=ctx.dv_overflow_fallback,
                storage_bytes=ctx.dv_stats_storage_bytes,
                forward_consistent_dv=ctx.forward_consistent_dv,
                stats_mode=ctx.dv_stats_mode,
                local_replay=ctx.dv_local_replay,
                spill_tile_count=ctx.dv_spill_tile_count,
                attempted_spill_tile_count=ctx.dv_attempted_spill_tile_count,
                fallback_reason=ctx.dv_fallback_reason,
                replay_scratch_bytes=ctx.dv_replay_scratch_bytes,
                dense_storage_bytes=ctx.dv_dense_storage_bytes,
                candidate=CANDIDATE,
                tiled_values=ctx.dv_packed_tiled_values,
                zero_spill_fastpath=ctx.dv_zero_spill_fastpath,
                has_spill_launch=ctx.dv_has_spill_launch)


def statistics_errors(row: Any) -> list[str]:
    errors = []
    if row.get("format") not in ("dense", LOCAL_FORMAT):
        errors.append("Unexpected format: fixed-capacity whole-layer fallback reports cannot certify P64")
    for name in ("local_replay", "overflow_fallback"):
        if type(row.get(name)) is not bool:
            errors.append(f"Missing boolean {name}")
    for name in ("storage_bytes", "dense_storage_bytes", "spill_tile_count", "attempted_spill_tile_count",
                 "replay_scratch_bytes"):
        if type(row.get(name)) is not int or row[name] < 0:
            errors.append(f"Missing nonnegative integer {name}")
    if row.get("candidate") != CANDIDATE:
        errors.append("Missing or mismatched selected tuning candidate")
    expected_zero, expected_tiled = CANDIDATE_FLAGS[CANDIDATE] if compact_record(row) else (False, False)
    expected_launch = compact_record(row) and (row.get("spill_tile_count", 0) > 0 or not expected_zero)
    for name, expected in (("zero_spill_fastpath", expected_zero), ("tiled_values", expected_tiled),
                           ("has_spill_launch", expected_launch)):
        if row.get(name) is not expected:
            errors.append("Actual tuning dispatch mismatch: " + name)
    if compact_record(row):
        if row.get("capacity") != 64 or row.get("local_replay") is not True:
            errors.append("P64 route/capacity mismatch")
        if row.get("spill_tile_count") != row.get("attempted_spill_tile_count"):
            errors.append("Accepted local route dropped attempted spill tiles")
        if row.get("fallback_reason") is not None:
            errors.append("Accepted local route carries a fallback reason")
    elif row.get("format") == "dense":
        if row.get("capacity") != 0 or row.get("local_replay") is not False:
            errors.append("Dense route has compressed state")
        if row.get("spill_tile_count") != 0:
            errors.append("Dense route retains local spill tiles")
    return errors


def require_report_domain(document: Any) -> None:
    if (document.get("acceptance_scope") != SCOPE or document.get("observer_mode") != "record-only"
            or document.get("configuration", {}).get("strict") is not False
            or document.get("configuration", {}).get("flash_deterministic") is not False):
        raise RuntimeError("Report is outside nonstrict diagnostic domain")
    if document.get("variant") not in POLICIES:
        raise RuntimeError("Old fixed-capacity/default report cannot certify local replay")
