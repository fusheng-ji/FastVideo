#!/usr/bin/env python3
"""Generate static GB10 figures from published numerical exports.

Use --source-root for the numerical export directory, which retains
the source paths listed in figure-manifest.json. No CUDA libraries are imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

LEGACY = "#5D6875"
FUSED = "#008F8C"
ACCENT = "#D6682A"
BLUE = "#3066A8"
SOURCES: dict[str, dict[str, str]] = {}
FIGURES: list[dict[str, object]] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_file(root: Path, relative: str) -> Path:
    path = root / relative
    SOURCES[relative] = {"sha256": sha256(path)}
    return path


def read_csv(root: Path, relative: str) -> list[dict[str, str]]:
    with load_file(root, relative).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(root: Path, relative: str) -> dict:
    return json.loads(load_file(root, relative).read_text(encoding="utf-8"))


def reduction(legacy: float, fused: float) -> float:
    return 100.0 * (legacy - fused) / legacy


def finish(fig: plt.Figure, out: Path, name: str, caption: str) -> None:
    filenames = []
    for extension in ("png", "svg"):
        path = out / f"{name}.{extension}"
        metadata = {"Creator" if extension == "svg" else "Software":
                    f"Matplotlib {matplotlib.__version__}; generate_figures.py"}
        if extension == "svg":
            metadata["Date"] = None  # Avoid non-reproducible export timestamps.
        fig.savefig(path, dpi=200, facecolor="white", metadata=metadata)
        filenames.append({"file": path.name, "sha256": sha256(path)})
    plt.close(fig)
    FIGURES.append({"name": name, "caption": caption, "files": filenames})


def title(fig: plt.Figure, heading: str, subheading: str) -> None:
    fig.text(0.055, 0.965, heading, fontsize=17, weight="bold", va="top")
    fig.text(0.055, 0.914, subheading, fontsize=10, color="#45515E", va="top")


def footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.055, 0.05, text, fontsize=9, color="#45515E", va="bottom")


def int8_figure(root: Path, out: Path) -> None:
    metric = read_json(root, "int8/metrics.json")
    counts = [metric["memcheck"]["baseline_reported_errors"],
              metric["memcheck"]["fixed_reported_errors"]]
    gemm = metric["gemm"]
    linear = metric["linear"]
    errors = [gemm["output"]["relative_l2"], linear["output"]["relative_l2"]]
    if counts != [84, 0] or not all(record["output"]["finite"]
                                  for record in (gemm, linear)):
        raise ValueError("Unexpected INT8 evidence or non-finite result")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.25, wspace=0.38)
    title(fig, "DGX Spark GB10 · INT8 tail-write fix",
          "Matched sm_121a CMake flags; private native artifacts; installed wheel unchanged")
    axes[0].bar([0, 1], counts, width=0.58, color=[LEGACY, FUSED])
    axes[0].set_xticks([0, 1], ["Original Saver", "Corrected row stride"])
    axes[0].set_ylim(0, 103)
    axes[0].set_ylabel("Reported memcheck errors")
    axes[0].set_title("Tail quantization: 129 × 129", fontsize=12, pad=12)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    for x, value in enumerate(counts):
        axes[0].text(x, value + 4, str(value), ha="center", weight="bold", fontsize=16)
    axes[1].bar([0, 1], errors, width=0.58, color=[BLUE, FUSED])
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-9, 5e-3)
    axes[1].set_xticks([0, 1], ["GEMM", "Int8Linear"])
    axes[1].set_ylabel("Relative L2 vs quantized reference")
    axes[1].set_title("BF16 · M=128, N=256, K=512", fontsize=12, pad=12)
    axes[1].axhline(1e-3, color=ACCENT, linestyle="--", linewidth=1.5)
    axes[1].text(0.98, 1e-3, "Acceptance: ≤ 1e−3", ha="right", va="bottom",
                 transform=axes[1].get_yaxis_transform(), color=ACCENT, fontsize=9)
    for x, value in enumerate(errors):
        axes[1].text(x, value * 1.45, f"{value:.3e}", ha="center", fontsize=10, weight="bold")
    footnote(fig, "Memcheck and numerical agreement are separate results; no INT8 speedup is claimed.\n"
             "Unquantized Int8Linear relative L2 = 0.0136449 (not plotted; a different reference).")
    finish(fig, out, "int8-tail-memory-and-reference", 
           "Matched production-CMake flags at sm_121a: memcheck reports 84 → 0 errors for "
           "129×129 tail quantization. Separately, fixed BF16 GEMM and Int8Linear agree with "
           "an independent integer-dot/block-scale quantized reference at relative L2 "
           "1.486e−8 and 2.317e−4. Both outputs are finite; their comparison is numerical, "
           "not byte equality. Unquantized Int8Linear error 0.0136449 is a different "
           "reference and is not a before/after fix metric. Installed wheel unchanged; "
           "the historical aligned-512 crash was not reproduced.")


def formal_rows(root: Path) -> dict[int, list[dict[str, str]]]:
    return {tokens: read_csv(root, f"vsa64/trained-wan-{tokens}-benchmark.csv")
            for tokens in (8192, 31200)}


def route_values(rows: list[dict[str, str]], field: str, factor: float = 1.0) -> dict[str, list[float]]:
    return {route: [float(row[field]) * factor for row in rows if row["route"] == route]
            for route in ("legacy", "fused")}


def latency_figure(root: Path, out: Path) -> None:
    formal = formal_rows(root)
    pipeline = read_csv(root, "vsa64/pipeline-full-v3/report.csv")
    if len(pipeline) != 18 or any(row["bitwise_passed"] != "True" for row in pipeline):
        raise ValueError("Unexpected full-pipeline numerical evidence")
    medians = [route_values(formal[8192], "latency_ms", 0.001),
               route_values(formal[31200], "latency_ms", 0.001),
               route_values(pipeline, "dit_total_ms", 0.001),
               route_values(pipeline, "generation_wall_seconds")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.3))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.76, bottom=0.25, wspace=0.28)
    title(fig, "DGX Spark GB10 · VSA64 full-model latency",
          "Public trained FastWan2.1-T2V-1.3B checkpoint · unchanged full-model peak allocation")
    labels = [["8,192 tokens", "31,200 tokens"], ["3-step DiT", "Generation incl. T5/VAE"]]
    for panel, ax in enumerate(axes):
        values = medians[panel * 2:panel * 2 + 2]
        for index, route in enumerate(("legacy", "fused")):
            bars = ax.bar([x + (-0.19 if index == 0 else 0.19) for x in range(2)],
                          [statistics.median(value[route]) for value in values],
                          width=0.35, color=LEGACY if index == 0 else FUSED,
                          label="Legacy layout" if index == 0 else "Fused layout")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.10 if panel == 0 else 0.5),
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        for x, value in enumerate(values):
            legacy, fused = (statistics.median(value[route]) for route in ("legacy", "fused"))
            ax.text(x, max(legacy, fused) + (0.50 if panel == 0 else 2.6),
                    f"−{reduction(legacy, fused):.2f}% latency", ha="center",
                    fontsize=10, weight="bold", color=FUSED)
        ax.set_xticks([0, 1], labels[panel], fontsize=9)
        ax.set_ylim(0, 8.2 if panel == 0 else 47)
        ax.set_ylabel("Median latency (seconds)")
        ax.set_title("Single DiT forward · synthetic conditioning\n150 samples/route (5 × 30)" if panel == 0
                     else "Actual prompts · 77 frames, 480 × 832\n9 generations/route (3 prompts × 3 repeats)",
                     fontsize=11, pad=12)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    footnote(fig, "Generation is instrumented: includes T5/VAE and progress metadata; no MP4 encoding.\n"
             "Loading, initial compile, RNG setup and deferred checks excluded. Dynamic clocks; all samples retained.")
    finish(fig, out, "vsa64-full-model-latency", 
           "Pooled route medians, with scopes shown separately. Formal single full-DiT "
           "forwards use synthetic conditioning (5 warmups, 30 samples, 5 rounds/route): "
           "8192 tokens 2.607784 → 2.521559 s (3.306% latency reduction); 31200 tokens "
           "6.485170 → 6.236700 s (3.831%). Actual-prompt, three-step 31200-token pipeline "
           "uses three prompts/seeds × three repeats/route: DiT 17.467208 → 16.845215 s "
           "(3.561%); instrumented generation including T5/VAE 37.821364 → 37.197883 s "
           "(1.648%). No MP4 encoding. Full-model peak allocation is equal between routes. "
           "Dynamic clocks and two adverse paired samples are retained; paused production "
           "route-validation timing is excluded.")


def paired_figure(root: Path, out: Path) -> None:
    rows = read_csv(root, "vsa64/pipeline-full-v3/report.csv")
    pairs = {}
    for row in rows:
        key = (int(row["pair"]), int(row["repeat"]))
        if row["route"] in pairs.setdefault(key, {}):
            raise ValueError(f"Duplicate paired route: {key}")
        pairs[key][row["route"]] = row
    if len(pairs) != 9 or any(set(pair) != {"legacy", "fused"} for pair in pairs.values()):
        raise ValueError("Incomplete nine-pair pipeline results")
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.29)
    title(fig, "DGX Spark GB10 · every actual-prompt timing pair",
          "31,200 tokens · 3 DMD steps · 18 generations · no paused-production timing")
    reductions = {}
    for field, offset, color, marker, label in (
        ("dit_total_ms", -0.12, BLUE, "o", "3-step DiT (CUDA events)"),
        ("generation_wall_seconds", 0.12, ACCENT, "D", "Generation incl. T5/VAE (wall time)")):
        values = [reduction(float(pairs[key]["legacy"][field]), float(pairs[key]["fused"][field]))
                  for key in sorted(pairs)]
        reductions[field] = values
        ax.scatter([x + offset for x in range(9)], values, color=color, marker=marker,
                   s=58, label=label, zorder=4)
        for x, value in enumerate(values):
            if value < 0:
                ax.annotate(f"{value:.2f}%", (x + offset, value),
                            xytext=(-25 if offset < 0 else 28, 13 if offset < 0 else -15),
                            textcoords="offset points", fontsize=9, ha="center", color=color,
                            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7})
    if sum(value < 0 for value in reductions["dit_total_ms"]) != 2:
        raise ValueError("Expected two adverse DiT pairs in the retained raw samples")
    ax.axhline(0, color=LEGACY, lw=1.1)
    ax.axhspan(-4.1, 0, color="#F7EBE6", zorder=0)
    ax.set_ylim(-4.1, 5.5)
    ax.set_xlim(-0.65, 8.65)
    ax.set_ylabel("Paired latency reduction: (legacy − fused) / legacy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xticks(range(9), [f"s{pairs[key]['legacy']['seed']} / r{key[1]}" for key in sorted(pairs)], fontsize=9)
    ax.set_xlabel("Prompt/seed and repeat (zero-based); every pair shown", labelpad=10)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    for boundary in (2.5, 5.5):
        ax.axvline(boundary, color="#CDD4DC", linestyle=":", lw=1)
    ax.text(0.012, 0.03, "Below zero = fused slower in that timing pair", transform=ax.transAxes,
            fontsize=9, color="#805345")
    footnote(fig, "Dynamic-clock transitions occurred. Paired reductions and pooled-median reductions are different statistics.\n"
             "All tensors pass strict byte comparisons; this chart is timing evidence, not a video-quality evaluation.")
    finish(fig, out, "vsa64-all-paired-pipeline-samples", 
           "Every one of nine actual-prompt timing pairs is shown; positive values mean "
           "lower fused-route latency. Seed 11 repeat 2 and seed 42 repeat 0 are slower "
           "on both metrics and remain visible. Three-step DiT adverse reductions are "
           "−0.796% and −2.300%; instrumented generation adverse reductions are −0.631% "
           "and −2.617%. Dynamic clock transitions occurred. No pair is discarded and "
           "no fixed-clock or every-sample improvement is claimed. No MP4 encoding; "
           "the separately paused production diagnostic is not performance evidence.")


def round_figure(root: Path, out: Path) -> None:
    data = formal_rows(root)
    rounds = {}
    for tokens, rows in data.items():
        rounds[tokens] = {}
        for index in range(5):
            values = route_values([row for row in rows if int(row["round"]) == index], "latency_ms", 0.001)
            if any(len(samples) != 30 for samples in values.values()):
                raise ValueError("Expected exactly 30 samples/route/round")
            rounds[tokens][index] = {route: statistics.median(samples) for route, samples in values.items()}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.1))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.24, wspace=0.28)
    title(fig, "DGX Spark GB10 · five-round full-DiT results",
          "Single full-model forward · synthetic conditioning · 30 serial samples/route/round")
    for tokens, color, marker in ((8192, BLUE, "o"), (31200, FUSED, "s")):
        for route, linestyle in (("legacy", "--"), ("fused", "-")):
            axes[0].plot(range(5), [rounds[tokens][index][route] for index in range(5)],
                         color=color, linestyle=linestyle, marker=marker, lw=1.6,
                         label=f"{tokens:,} · {route}")
        changes = [reduction(rounds[tokens][index]["legacy"], rounds[tokens][index]["fused"])
                   for index in range(5)]
        axes[1].plot(range(5), changes, color=color, marker=marker, lw=1.6, label=f"{tokens:,} tokens")
        for x, value in enumerate(changes):
            axes[1].text(x, value + (0.35 if tokens == 31200 else -0.6), f"{value:.2f}%",
                         color=color, fontsize=8.5, ha="center")
    axes[0].set_ylim(0, 7.5)
    axes[0].set_ylabel("Round median latency (seconds)")
    axes[0].legend(loc="center left", frameon=False, fontsize=9)
    axes[1].set_ylim(0, 13)
    axes[1].set_ylabel("Reduction of round medians (%)")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xticks(range(5))
        ax.set_xlabel("Round (zero-based)", labelpad=8)
    axes[1].annotate("Clock-regime transition\nretained, not filtered", (0, 11.2207061475591),
                     xytext=(0.7, 9.2), fontsize=9, color="#45515E",
                     arrowprops={"arrowstyle": "-", "color": "#45515E", "lw": 0.8})
    footnote(fig, "5 untimed warmups precede each 30-sample round. All five rounds are retained.\n"
             "Dynamic clocks: the 8,192-token first round differs; these are not fixed-clock measurements.")
    finish(fig, out, "vsa64-five-round-full-dit", 
           "Five serial rounds, 30 timed samples per route after five untimed warmups. "
           "All rounds are shown. The 8192-token first round samples a different dynamic "
           "clock regime (round-median reduction 11.221%) and is not removed; subsequent "
           "rounds show 3.248–3.377%. The 31200-token round reductions are 3.784–3.857%. "
           "Round-median reduction is distinct from the pooled-route-median summary. "
           "No fixed-clock claim or confidence interval inferred from independent rounds.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "axes.grid": True,
                         "grid.alpha": 0.18, "grid.linestyle": "-",
                         "svg.fonttype": "none", "svg.hashsalt": "gb10-int8-vsa64-20261009"})
    int8_figure(args.source_root, args.output_dir)
    latency_figure(args.source_root, args.output_dir)
    paired_figure(args.source_root, args.output_dir)
    round_figure(args.source_root, args.output_dir)
    manifest = {"schema_version": 1, "script_sha256": sha256(Path(__file__)),
                "matplotlib_version": matplotlib.__version__,
                "sources_relative_to_source_root": SOURCES, "figures": FIGURES,
                "reproduction": "python scripts/generate_figures.py --source-root data --output-dir figures",
                "scope": "Static plots from selected numerical JSON and all timing CSV rows; no GPU execution; no sample filtering"}
    (args.output_dir / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "CAPTIONS.txt").write_text("\n\n".join(
        f"{figure['name']}:\n{figure['caption']}" for figure in FIGURES) + "\n")
    print(json.dumps({"figures": len(FIGURES), "source_files": len(SOURCES),
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
