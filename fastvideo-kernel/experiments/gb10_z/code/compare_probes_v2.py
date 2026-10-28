"""Strict, CPU-only comparison of pinned GB10 full-model gradient probes.

A complete report is not necessarily accepted. Missing tensor records, incomplete
process/repetition coverage, and source/configuration drift fail closed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from parameter_hooks import HOOK_SEMANTICS
from local_policy import compact_record, require_report_domain, statistics_errors
from flash_control import validate_calls

SCHEMA_VERSION = 2
TENSOR_FIELDS = ("shape", "dtype", "sha256", "finite")
EVENT_FIELDS = ("phase", "kind", "module", "call_id")
MARKER_KINDS = {"checkpoint_early_stop", "parameter_gradient_arrival"}
STATE_FIELDS = ("rng_after", "parameters_after_sha256", "buffers_after_sha256", "frozen_batch_unchanged",
                "batch_after_sha256")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_errors(value: Any, label: str) -> list[str]:
    if not isinstance(value, dict) or value.get("missing") is True:
        return [f"{label}: missing tensor"]
    errors = []
    shape = value.get("shape")
    if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
        errors.append(f"{label}: invalid or missing shape")
    if not isinstance(value.get("dtype"), str) or not value["dtype"]:
        errors.append(f"{label}: invalid or missing dtype")
    digest = value.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        errors.append(f"{label}: invalid or missing byte hash")
    if value.get("finite") is not True:
        errors.append(f"{label}: finite must be true")
    return errors


def tensor_equal(left: Any, right: Any) -> bool:
    if tensor_errors(left, "left") or tensor_errors(right, "right"):
        return False
    return all(left[field] == right[field] for field in TENSOR_FIELDS)


def event_key(event: dict) -> tuple:
    return tuple(event.get(field) for field in EVENT_FIELDS)


def normalized_event_key(value: Any) -> tuple | None:
    values = event_key(value) if isinstance(value, dict) else tuple(value) if isinstance(value, tuple | list) else ()
    if (len(values) != len(EVENT_FIELDS) or any(not isinstance(item, str) or not item for item in values[:3])
            or not (isinstance(values[3], str) and values[3] or type(values[3]) is int)):
        return None
    return values


def _expected_names(value: Any, label: str, errors: list[str]) -> set[str]:
    if not isinstance(value, list) or not value or any(not isinstance(name, str) or not name for name in value):
        errors.append(f"{label}: expected a nonempty list of names")
        return set()
    if len(set(value)) != len(value):
        errors.append(f"{label}: duplicate expected names")
    return set(value)


def repetition_errors(repetition: Any, label: str) -> list[str]:
    if not isinstance(repetition, dict):
        return [f"{label}: repetition is not an object"]
    errors = []
    if type(repetition.get("index")) is not int or repetition["index"] < 0:
        errors.append(f"{label}: invalid repetition index")
    state = repetition.get("input_state")
    if not isinstance(state, dict) or not state:
        errors.append(f"{label}: missing input state")
    else:
        for field in ("batch_sha256", "parameters_sha256", "buffers_sha256"):
            if not isinstance(state.get(field), str) or not state[field]:
                errors.append(f"{label}: missing input-state {field}")
        rng = state.get("rng_before")
        if not isinstance(rng, dict):
            errors.append(f"{label}: missing RNG state fingerprints")
        else:
            for name in ("python", "numpy", "torch_cpu", "cuda_all", "method"):
                if not rng.get(name):
                    errors.append(f"{label}: missing RNG state {name}")
    for name in ("loss", "timesteps"):
        errors.extend(tensor_errors(repetition.get(name), f"{label}.{name}"))
    prediction = repetition.get("prediction")
    if not isinstance(prediction, dict) or not prediction:
        errors.append(f"{label}: missing prediction tensor mapping")
    else:
        for name, value in prediction.items():
            errors.extend(tensor_errors(value, f"{label}.prediction.{name}"))
    rng_after = repetition.get("rng_after")
    if not isinstance(rng_after, dict) or set(rng_after) != {"python", "numpy", "torch_cpu", "cuda_all", "method"}:
        errors.append(f"{label}: missing RNG state after backward")
    elif any(not value for value in rng_after.values()):
        errors.append(f"{label}: empty RNG fingerprint after backward")
    for after, before in (("parameters_after_sha256", "parameters_sha256"), ("buffers_after_sha256", "buffers_sha256"),
                          ("batch_after_sha256", "batch_sha256")):
        if not repetition.get(after) or not isinstance(state, dict) or repetition[after] != state.get(before):
            errors.append(f"{label}: {after} missing or changed")
    if repetition.get("frozen_batch_unchanged") is not True:
        errors.append(f"{label}: frozen batch was not verified unchanged")
    flash = repetition.get("flash_control")
    if (not isinstance(flash, dict) or flash.get("expected") != 30 or flash.get("observed") != 30
            or flash.get("all_nondeterministic") is not True):
        errors.append(f"{label}: missing 30 nonstrict FA2 backward calls")
    coverage = repetition.get("coverage")
    if not isinstance(coverage, dict):
        coverage = {}
        errors.append(f"{label}: missing coverage")
    if coverage.get("passed") is not True or coverage.get("errors") != []:
        errors.append(f"{label}: producer coverage did not explicitly pass")
    expected = _expected_names(coverage.get("expected_parameter_gradients"), f"{label}.expected_parameter_gradients",
                               errors)
    hooks = repetition.get("parameter_hooks")
    if (not isinstance(hooks, dict) or hooks.get("semantics") != HOOK_SEMANTICS
            or hooks.get("identity_verified") is not True or hooks.get("passed") is not True
            or hooks.get("errors") != [] or not isinstance(hooks.get("counts"), dict)
            or set(hooks["counts"]) != expected
            or any(type(value) is not int or value != 1 for value in hooks["counts"].values())):
        errors.append(f"{label}: master accumulation hook validation failed")
    gradients = repetition.get("parameter_gradients")
    if not isinstance(gradients, dict):
        gradients = {}
        errors.append(f"{label}: missing parameter_gradients")
    if set(gradients) != expected:
        errors.append(f"{label}: parameter gradient names do not match expected names")
    if coverage.get("missing_parameter_gradients") != []:
        errors.append(f"{label}: missing_parameter_gradients must be an explicit empty list")
    observed = coverage.get("observed_parameter_gradients")
    if isinstance(observed, list):
        if any(not isinstance(name, str)
               for name in observed) or len(observed) != len(set(observed)) or set(observed) != expected:
            errors.append(f"{label}: observed parameter coverage differs")
    elif type(observed) is not int or observed != len(expected):
        errors.append(f"{label}: invalid observed parameter coverage")
    for name, value in gradients.items():
        errors.extend(tensor_errors(value, f"{label}.gradient.{name}"))
    events = repetition.get("events")
    if not isinstance(events, list) or not events:
        events = []
        errors.append(f"{label}: missing hook events")
    keys = []
    for index, event in enumerate(events):
        prefix = f"{label}.event[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{prefix}: invalid event")
            continue
        order = event.get("order")
        if type(order) is not int or order != index:
            errors.append(f"{prefix}: event order is not contiguous in recorded order")
        identity = normalized_event_key(event)
        if identity is None:
            errors.append(f"{prefix}: missing or malformed event identity")
        else:
            keys.append(identity)
        tensors = event.get("tensors")
        marker = event.get("marker") is True and event.get("kind") in MARKER_KINDS
        if event.get("marker") is True and not marker:
            errors.append(f"{prefix}: unsupported marker kind")
        if not isinstance(tensors, dict) or (not tensors and not marker):
            errors.append(f"{prefix}: missing tensors")
        elif marker and tensors:
            errors.append(f"{prefix}: pure marker must not contain tensors")
        else:
            for name, value in tensors.items():
                errors.extend(tensor_errors(value, f"{prefix}.{name}"))
    key_counts = Counter(keys)
    if len(keys) != len(set(keys)):
        errors.append(f"{label}: duplicate hook identities")
    expected_keys = coverage.get("expected_event_keys")
    if not isinstance(expected_keys, list) or not expected_keys:
        errors.append(f"{label}: expected_event_keys must be nonempty")
    else:
        normalized = []
        for key in expected_keys:
            values = normalized_event_key(key)
            if values is None:
                errors.append(f"{label}: malformed expected hook identity")
            else:
                normalized.append(values)
        if key_counts != Counter(normalized):
            errors.append(f"{label}: hook identities do not match expected coverage")
    required_keys = coverage.get("required_event_keys")
    if not isinstance(required_keys, list) or not required_keys:
        errors.append(f"{label}: required_event_keys must be a nonempty list")
    else:
        required = [normalized_event_key(key) for key in required_keys]
        if len(required) != len(set(required)):
            errors.append(f"{label}: duplicate mandatory hook declarations")
        for values in required:
            if values is None or key_counts[values] != 1:
                errors.append(f"{label}: mandatory hook is absent or duplicated")
    declared_required = {normalized_event_key(item) for item in (required_keys or [])}
    for name in expected:
        key = ("backward", "parameter_gradient_arrival", name, name)
        if key_counts[key] != 1 or key not in declared_required:
            errors.append(f"{label}: master accumulation event absent, duplicated, or undeclared: {name}")
    modules = _expected_names(coverage.get("expected_block_modules"), f"{label}.expected_block_modules", errors)
    for module in sorted(modules):
        local = [event for event in events if isinstance(event, dict) and event.get("module") == module]
        for phase, kinds in (("forward", ("block_input", "block_output")), ("backward", ("block_input_grad",
                                                                                         "block_output_grad"))):
            for kind in kinds:
                if sum(event.get("phase") == phase and event.get("kind") == kind for event in local) != 1:
                    errors.append(f"{label}: expected exactly one {phase} {kind} hook for {module}")
        if not any(event.get("phase") == "recompute" for event in local):
            errors.append(f"{label}: missing checkpoint recomputation for {module}")
    statistics = repetition.get("statistics")
    if not isinstance(statistics, list) or not statistics:
        errors.append(f"{label}: missing observed statistics routes")
    else:
        statistic_keys = []
        for index, statistic in enumerate(statistics):
            if not isinstance(statistic, dict):
                errors.append(f"{label}.statistics[{index}]: missing actual statistics route")
                continue
            key = normalized_event_key(dict(statistic, kind="qat_output"))
            if key is None:
                errors.append(f"{label}.statistics[{index}]: missing QAT call identity")
            else:
                statistic_keys.append(key)
            if (statistic.get("format") not in ("dense", "packed_slots_local_replay")
                    or type(statistic.get("overflow_fallback")) is not bool
                    or statistic.get("forward_consistent_dv") is not True or statistic.get("stats_mode") != "save"
                    or type(statistic.get("storage_bytes")) is not int or statistic["storage_bytes"] <= 0
                    or type(statistic.get("capacity")) is not int
                    or (statistic["format"] == "dense" and statistic["capacity"] != 0)
                    or (statistic["format"] == "packed_slots_local_replay" and
                        (not 1 <= statistic["capacity"] <= 128 or statistic["overflow_fallback"]))):
                errors.append(f"{label}.statistics[{index}]: unrecorded or invalid actual statistics route")
            errors.extend(f"{label}.statistics[{index}]: {e}" for e in statistics_errors(statistic))
        qat_keys = [key for key in keys if key[1] == "qat_output"]
        if Counter(statistic_keys) != Counter(qat_keys):
            errors.append(f"{label}: statistics routes do not match every QAT output")
    return errors


def _layer(module: str | None) -> int | None:
    match = re.search(r"(?:blocks|layers)\.(\d+)(?:\.|$)", module or "")
    return int(match.group(1)) if match else None


def route_for(difference: dict | None) -> dict:
    if difference is None:
        return {"category": "none", "layer": None}
    module = difference.get("module")
    location = difference.get("location", "")
    phase = str(difference.get("phase", ""))
    kind = str(difference.get("kind", ""))
    tensor = str(difference.get("tensor", ""))
    if location == "input_state":
        category = "input_or_rng_pairing"
    elif location in ("event_order", "event_coverage"):
        category = "hook_coverage_or_execution_order"
    elif "recompute" in phase or "checkpoint" in phase:
        category = "checkpoint_recomputation"
    elif tensor.lower() in ("do", "dout", "output_gradient", "grad_output") or kind == "qat_dout":
        category = "upstream_dout"
    elif phase == "backward" and location == "event":
        category = "attention_backward" if "attn" in str(module) or "attention" in str(module) else "block_backward"
    elif location == "parameter_gradients":
        category = "parameter_gradient_reduction"
    elif location in ("prediction", "loss", "event"):
        category = "forward"
    else:
        category = "configuration_or_coverage"
    return {"category": category, "layer": _layer(module), "module": module, "first_difference": difference}


def compare_repetitions(left: dict, right: dict) -> dict:
    differences: list[dict[str, Any]] = []
    if left["input_state"] != right["input_state"]:
        differences.append({"location": "input_state", "reason": "input, RNG, or model-state fingerprint differs"})
    left_events, right_events = left["events"], right["events"]
    for index, (a, b) in enumerate(zip(left_events, right_events, strict=False)):
        identity = {field: a[field] for field in EVENT_FIELDS}
        identity["event_index"] = index
        if event_key(a) != event_key(b) or a["order"] != b["order"]:
            differences.append(dict(identity, location="event_order", left=event_key(a), right=event_key(b)))
            break
        if set(a["tensors"]) != set(b["tensors"]):
            differences.append(dict(identity, location="event_coverage", reason="event tensor names differ"))
            continue
        for name in a["tensors"]:
            if not tensor_equal(a["tensors"][name], b["tensors"][name]):
                differences.append(
                    dict(identity, location="event", tensor=name, left=a["tensors"][name], right=b["tensors"][name]))
    if len(left_events) != len(right_events):
        differences.append({
            "location": "event_coverage",
            "event_index": min(len(left_events), len(right_events)),
            "left": len(left_events),
            "right": len(right_events)
        })
    for name in ("loss", "timesteps"):
        if not tensor_equal(left[name], right[name]):
            differences.append({"location": name, "left": left[name], "right": right[name]})
    left_prediction, right_prediction = left["prediction"], right["prediction"]
    if set(left_prediction) != set(right_prediction):
        differences.append({"location": "prediction", "reason": "prediction tensor names differ"})
    for name in sorted(set(left_prediction) & set(right_prediction)):
        if not tensor_equal(left_prediction[name], right_prediction[name]):
            differences.append({
                "location": "prediction",
                "tensor": name,
                "left": left_prediction[name],
                "right": right_prediction[name]
            })
    for name in STATE_FIELDS:
        if left[name] != right[name]:
            differences.append({
                "location": "input_state",
                "field": name,
                "reason": "state after backward differs",
                "left": left[name],
                "right": right[name]
            })
    left_gradients, right_gradients = left["parameter_gradients"], right["parameter_gradients"]
    if set(left_gradients) != set(right_gradients):
        differences.append({"location": "parameter_gradients", "reason": "parameter names differ"})
    for name in sorted(set(left_gradients) & set(right_gradients)):
        if not tensor_equal(left_gradients[name], right_gradients[name]):
            differences.append({
                "location": "parameter_gradients",
                "module": name,
                "left": left_gradients[name],
                "right": right_gradients[name]
            })
    return {
        "accepted": not differences,
        "difference_count": len(differences),
        "differences": differences,
        "route": route_for(differences[0] if differences else None)
    }


def compare_documents(documents: list[dict], variants: list[str], processes: int,
                      repeats_per_process: list[int]) -> dict:
    errors = []
    comparisons = []
    by_variant = defaultdict(list)
    if not variants or len(set(variants)) != len(variants):
        errors.append("required variants must be nonempty and unique")
    if processes < 1 or len(repeats_per_process) != processes or any(count < 1 for count in repeats_per_process):
        errors.append("repeat counts must specify one positive count per required process")
    seen_processes = set()
    for document_index, document in enumerate(documents):
        prefix = f"probe[{document_index}]"
        if not isinstance(document, dict):
            errors.append(f"{prefix}: probe is not an object")
            continue
        try:
            require_report_domain(document)
        except RuntimeError as error:
            errors.append(f"{prefix}: {error}")
        if len(document.get("parameter_registry") or []) != 825:
            errors.append(f"{prefix}: expected 825 declared Wan master parameters")
        if document.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{prefix}: unsupported schema_version")
        if document.get("parameter_hook_semantics") != HOOK_SEMANTICS:
            errors.append(f"{prefix}: unsupported parameter hook semantics")
        registry = document.get("parameter_registry")
        if (not isinstance(registry, list) or not registry
                or any(not isinstance(entry, dict) or not isinstance(entry.get("name"), str)
                       or not isinstance(entry.get("aliases"), list) or not entry["aliases"]
                       or entry["aliases"][0] != entry["name"] or not isinstance(entry.get("type"), str)
                       for entry in (registry or []))):
            errors.append(f"{prefix}: malformed master registry")
        else:
            names = [entry["name"] for entry in registry]
            aliases = [alias for entry in registry for alias in entry["aliases"]]
            if len(set(names)) != len(names) or len(set(aliases)) != len(aliases):
                errors.append(f"{prefix}: duplicate master registry names or aliases")
            for rep in document.get("repetitions") or []:
                if names != rep.get("coverage", {}).get("expected_parameter_gradients"):
                    errors.append(f"{prefix}: registry differs from declared parameter coverage")
        if document.get("status") != "complete":
            errors.append(f"{prefix}: probe did not complete")
        variant = document.get("variant")
        if variant not in variants:
            errors.append(f"{prefix}: unexpected or missing variant {variant!r}")
            continue
        process = document.get("process_nonce")
        if not isinstance(process, str) or not process:
            errors.append(f"{prefix}: missing process_nonce")
        elif process in seen_processes:
            errors.append(f"{prefix}: duplicate process identity")
        if isinstance(process, str):
            seen_processes.add(process)
        for field in ("source_fingerprint", "config_fingerprint"):
            if not isinstance(document.get(field), str) or not document[field]:
                errors.append(f"{prefix}: missing {field}")
        repetitions = document.get("repetitions")
        if not isinstance(repetitions, list) or not repetitions:
            repetitions = []
            errors.append(f"{prefix}: missing repetitions")
        try:
            validate_calls(document.get("flash_calls", []), 30 * len(repetitions))
        except RuntimeError as error:
            errors.append(f"{prefix}: {error}")
        for index, repetition in enumerate(repetitions):
            errors.extend(repetition_errors(repetition, f"{prefix}.repetition[{index}]"))
        if [item.get("index") for item in repetitions if isinstance(item, dict)] != list(range(len(repetitions))):
            errors.append(f"{prefix}: repetition indices must be contiguous from zero")
        by_variant[variant].append(document)
    for variant in variants:
        group = by_variant[variant]
        if len(group) != processes:
            errors.append(f"{variant}: expected {processes} processes, observed {len(group)}")
        observed_counts = sorted(len(doc.get("repetitions") or []) for doc in group)
        if observed_counts != sorted(repeats_per_process):
            errors.append(f"{variant}: expected repetition counts {repeats_per_process}, observed {observed_counts}")
    if documents:
        for field in ("source_fingerprint", "config_fingerprint", "tokens", "seed"):
            values = {json.dumps(doc.get(field), sort_keys=True) for doc in documents if isinstance(doc, dict)}
            if len(values) != 1 or values == {"null"}:
                errors.append(f"{field}: probes are not paired")
    declarations = [[{
        key: entry.get(key)
        for key in ("name", "aliases", "type")
    } for entry in doc.get("parameter_registry", []) if isinstance(entry, dict)] for doc in documents
                    if isinstance(doc, dict)]
    if declarations and any(value != declarations[0] for value in declarations[1:]):
        errors.append("master parameter registries are not paired")
    compact_coverage = {}
    for variant, group in by_variant.items():
        observations = [
            item for doc in group for rep in (doc.get("repetitions") or []) if isinstance(rep, dict)
            for item in (rep.get("statistics") or []) if isinstance(item, dict)
        ]
        compact = [item for item in observations if compact_record(item)]
        compact_coverage[variant] = {
            "observed_calls": len(observations),
            "compact_calls": len(compact),
            "fallback_calls": sum(item.get("overflow_fallback") is True for item in observations),
            "compact_path_exercised": bool(compact)
        }
    if not errors:
        for variant in variants:
            group = by_variant[variant]
            for process_index, document in enumerate(group):
                first = document["repetitions"][0]
                for repetition in document["repetitions"][1:]:
                    comparisons.append(
                        dict(compare_repetitions(first, repetition),
                             kind="within_process",
                             variant=variant,
                             process_index=process_index,
                             repetitions=[0, repetition["index"]]))
            for process_index in range(1, len(group)):
                comparisons.append(
                    dict(compare_repetitions(group[0]["repetitions"][0], group[process_index]["repetitions"][0]),
                         kind="across_process",
                         variant=variant,
                         process_index=process_index))
        reference_variant = "dense" if "dense" in variants else variants[0]
        reference = by_variant[reference_variant][0]["repetitions"][0]
        for variant in variants:
            if variant != reference_variant:
                comparisons.append(
                    dict(compare_repetitions(reference, by_variant[variant][0]["repetitions"][0]),
                         kind="across_variant",
                         reference_variant=reference_variant,
                         variant=variant))
    first_failed = next((item for item in comparisons if not item["accepted"]), None)
    route = route_for({
        "location": "coverage",
        "reason": errors[0]
    }) if errors else (first_failed["route"] if first_failed else route_for(None))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "accepted": not errors and first_failed is None,
        "required_variants": variants,
        "required_processes": processes,
        "repeats_per_process": repeats_per_process,
        "errors": errors,
        "comparisons": comparisons,
        "route": route,
        "compact_coverage": compact_coverage,
        "compact_certification": {
            variant: item["compact_path_exercised"] and not errors and first_failed is None
            for variant, item in compact_coverage.items() if variant != "dense"
        },
        "scope": "Byte equality and finite values; scalar norm differences are not tensor relative L2."
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, action="append", required=True)
    parser.add_argument("--require-variants", default="dense,P64")
    parser.add_argument("--require-processes", type=int, default=2)
    parser.add_argument("--repeats-per-process", default="2,1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, errors, documents = [], [], []
    paths = set()
    for path in args.probe:
        try:
            resolved = path.resolve(strict=True)
            if resolved in paths:
                raise ValueError("duplicate probe path")
            paths.add(resolved)
            raw = path.read_bytes()
            documents.append(json.loads(raw))
            inputs.append({"path": str(resolved), "sha256": hashlib.sha256(raw).hexdigest()})
        except (OSError, ValueError) as exception:
            errors.append(f"{path}: {type(exception).__name__}: {exception}")
    try:
        repeats = [int(value) for value in args.repeats_per_process.split(",")]
        result = compare_documents(documents, args.require_variants.split(","), args.require_processes, repeats)
    except (TypeError, ValueError, KeyError) as exception:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "accepted": False,
            "errors": [f"Malformed probe: {type(exception).__name__}: {exception}"],
            "route": {
                "category": "configuration_or_coverage",
                "layer": None
            }
        }
    result["inputs"] = inputs
    result["errors"].extend(errors)
    result["accepted"] = result["accepted"] and not errors
    if errors:
        result["route"] = {"category": "configuration_or_coverage", "layer": None, "reason": errors[0]}
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "accepted", "route")}))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
