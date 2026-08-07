#!/usr/bin/env python3
"""
plot_paper.py — publication-ready figures for the PauLIB paper.

Outputs PDF + PNG to doc/img/:
  fig_mul_throughput   Benchmark A: per-op multiplication time vs qubit count
  fig_sum_mul_scaling  Benchmark B: PauliSum × PauliSum scaling
  fig_gpu_mul          Benchmark G: GPU vs CPU outer-product throughput
  fig_thread_scaling   Benchmark H: outer-product speedup vs. OMP threads

Run from the repo root:
    python bench/plot_paper.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("bench/results")
OUT = Path("doc/img")
OUT.mkdir(exist_ok=True)

# IEEE single-column width
FIG_W = 3.5
FIG_H = 2.5

# Colours — distinguishable on screen and when printed grey-scale
C = {
    "paulib_soa":   "#1a6faf",
    "paulib_light": "#56a0d3",
    "pe":          "#e07b00",
    "qiskit":      "#c0392b",
    "pl":          "#c0392b",
    "scalar":      "#56a0d3",
    "simd":        "#1a6faf",
}


def read_csv(name: str) -> list[dict]:
    rows = []
    with open(RESULTS / name) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = None
            rows.append(parsed)
    return rows


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  → doc/img/{stem}.pdf/.png")


def apply_style(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, which="both", linestyle="--", linewidth=0.4,
            alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(width=0.6)


# ---------------------------------------------------------------------------
# A — single PauliString multiplication throughput
# ---------------------------------------------------------------------------

def plot_A():
    rows = read_csv("bench_A_ps_mul.csv")

    n_pairs = [int(r["n_terms"]) for r in rows]
    t_soa   = [r["paulib SoA per-op"] * 1e9 for r in rows]   # ns
    t_pe    = [r["PauliEngine per-op"] * 1e9 for r in rows]
    t_qs    = [r["Qiskit per-op"] * 1e9 for r in rows]

    x = np.arange(len(n_pairs))
    w = 0.25

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    ax.bar(x - w, t_soa, w, label="PauLIB",   color=C["paulib_soa"], zorder=3)
    ax.bar(x,     t_pe,  w, label="PauliEngine",   color=C["pe"],         zorder=3)
    ax.bar(x + w, t_qs,  w, label="Qiskit",        color=C["qiskit"],     zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_pairs])
    ax.set_xlabel("Number of pairs (500 qubits)")
    ax.set_ylabel("Time per operation (ns)")
    ax.legend(loc="upper right")
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    save(fig, "fig_mul_throughput")
    plt.close(fig)


# ---------------------------------------------------------------------------
# B — PauliSum × PauliSum outer product scaling
# ---------------------------------------------------------------------------

def plot_B():
    rows = read_csv("bench_B_sum_mul.csv")

    # x-axis: number of output terms (N²)
    libs = [
        ("paulib SoA",  "PauLIB",  C["paulib_soa"], "o", "-"),
        ("PauliEngine", "PauliEngine",  C["pe"],         "^", "--"),
        ("Qiskit",      "Qiskit",       C["qiskit"],     "D", "--"),
    ]

    # Collect (output_terms, time_ms) per library; sort by output_terms
    data = defaultdict(list)
    for r in rows:
        n2 = int(r["n_terms"]) ** 2
        for key, *_ in libs:
            if r.get(key) is not None:
                data[key].append((n2, r[key] * 1e3))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for key, label, color, marker, ls in libs:
        pts = sorted(data[key])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=marker, color=color, label=label,
                linestyle=ls, markersize=4, linewidth=1.2)

    all_n = sorted(set(int(r["n_terms"]) for r in rows))
    tick_xs = [n ** 2 for n in all_n]
    tick_labels = [f"{n**2:,}\n(N={n})" for n in all_n]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Output terms N² (input Pauli terms N)")
    ax.set_ylabel("Time (ms)")
    ax.legend()
    apply_style(ax)
    fig.tight_layout()
    save(fig, "fig_sum_mul_scaling")
    plt.close(fig)


# ---------------------------------------------------------------------------
# G — GPU vs CPU outer-product throughput (GH200: H100 + Grace)
# ---------------------------------------------------------------------------

def plot_G():
    rows = read_csv("bench_G_gpu_mul.csv")

    output_size = [r["n_lhs"] * r["n_rhs"] for r in rows]
    tput_soa    = [r["n_lhs"] * r["n_rhs"] / r["cpu_soa_kj_s"] / 1e9 for r in rows]  # G pairs/s
    tput_gpu    = [r["n_lhs"] * r["n_rhs"] / r["gpu_kernel_s"] / 1e9 for r in rows]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    ax.plot(output_size, tput_gpu, "o-",  color=C["paulib_soa"],  label="PauLIB GPU (H100)",
            markersize=4, linewidth=1.4)
    ax.plot(output_size, tput_soa, "s--", color=C["paulib_light"],  label="PauLIB CPU SoA (72 cores)",
            markersize=4, linewidth=1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=200)  # headroom so the legend (top-left) clears the GPU line
    ax.set_xlabel(r"Output pairs $N \times M$ (500 qubits)")
    ax.set_ylabel("Throughput (G pairs/s)")
    ax.legend(loc="upper left")
    apply_style(ax)
    fig.tight_layout()
    save(fig, "fig_gpu_mul")
    plt.close(fig)


# ---------------------------------------------------------------------------
# H — thread-level parallelism scaling (outer product vs. OMP threads)
# ---------------------------------------------------------------------------

def plot_H():
    m4  = read_csv("thread_scaling_LMMIMNM-4F0JWQW.csv")
    gh4 = read_csv("thread_scaling_gh4.csv")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    max_threads = max(int(r["threads"]) for r in gh4)
    ideal = [1, max_threads]
    ax.plot(ideal, ideal, ":", color="grey", linewidth=1.0, label="Ideal", zorder=1)

    ax.plot([int(r["threads"]) for r in m4], [r["speedup"] for r in m4],
            "s--", color=C["paulib_light"], label="Apple M4 Pro (14C)",
            markersize=4, linewidth=1.2)
    ax.plot([int(r["threads"]) for r in gh4], [r["speedup"] for r in gh4],
            "o-", color=C["paulib_soa"], label="GH200 Grace (72C)",
            markersize=4, linewidth=1.4)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("OMP threads")
    ax.set_ylabel("Speedup vs. 1 thread")
    ax.legend(loc="upper left")
    apply_style(ax)
    fig.tight_layout()
    save(fig, "fig_thread_scaling")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       8,
        "axes.labelsize":  8,
        "axes.titlesize":  8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.framealpha": 0.8,
        "figure.dpi":      150,
        "pdf.fonttype":    42,   # embed fonts — required by IEEE
        "ps.fonttype":     42,
    })

    print(f"Writing figures to {OUT}/")
    plot_A()
    plot_B()
    plot_G()
    plot_H()
    print("Done.")
