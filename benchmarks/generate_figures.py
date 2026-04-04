"""Generate publication-quality figures for the ClustKIT paper.

Main Figure 1: Clustering Quality (ARI, Precision-Recall, Clan-level)
Main Figure 2: Speed-Quality Tradeoff (Runtime vs ARI, Memory)
Main Figure 3: Ablation & Generalization

All from existing benchmark data.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Color palette (colorblind-friendly)
COLORS = {
    "ClustKIT": "#E64B35",
    "MMseqs2": "#4DBBD5",
    "DeepClust": "#00A087",
    "CD-HIT": "#3C5488",
    "Linclust": "#F39B7F",
    "VSEARCH": "#8491B4",
}
MARKERS = {
    "ClustKIT": "o",
    "MMseqs2": "s",
    "DeepClust": "D",
    "CD-HIT": "^",
    "Linclust": "v",
    "VSEARCH": "X",
}


def load_large_pfam():
    with open(DATA_DIR / "pfam_large_results" / "pfam_large_results.json") as f:
        return json.load(f)


def load_small_pfam():
    with open(DATA_DIR / "pfam_benchmark_results" / "pfam_concordance_results_4threads.json") as f:
        return json.load(f)


def load_paper_metrics():
    with open(DATA_DIR / "paper_analysis" / "paper_metrics.json") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Clustering Quality
# ══════════════════════════════════════════════════════════════════════

def figure1_clustering_quality():
    """3-panel figure: (A) Family ARI vs threshold, (B) Precision-Recall, (C) Clan ARI."""
    data = load_large_pfam()
    thresholds = [0.3, 0.5, 0.7, 0.9]
    tools = ["ClustKIT", "MMseqs2", "DeepClust", "CD-HIT", "Linclust"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel A: Family ARI vs threshold
    ax = axes[0]
    for tool in tools:
        aris = []
        ts = []
        for t in thresholds:
            r = data[str(t)].get(tool, {})
            if "error" not in r and r:
                aris.append(r["family_ARI"])
                ts.append(t)
        if ts:
            ax.plot(ts, aris, marker=MARKERS[tool], color=COLORS[tool],
                    label=tool, linewidth=2, markersize=7)
    ax.set_xlabel("Sequence Identity Threshold")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("A. Family-Level Clustering Quality")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, 0.7)
    ax.grid(True, alpha=0.3)

    # Panel B: Precision vs Recall scatter
    ax = axes[1]
    for tool in tools:
        precs, recs = [], []
        for t in thresholds:
            r = data[str(t)].get(tool, {})
            if "error" not in r and r:
                precs.append(r["family_precision"])
                recs.append(r["family_recall"])
        if precs:
            ax.plot(precs, recs, marker=MARKERS[tool], color=COLORS[tool],
                    label=tool, linewidth=1.5, markersize=7)
            # Label the t=0.3 point
            if len(precs) > 0:
                ax.annotate(f"t={thresholds[0]}", (precs[0], recs[0]),
                           textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Pairwise Precision")
    ax.set_ylabel("Pairwise Recall")
    ax.set_title("B. Precision-Recall Tradeoff")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 0.75)
    ax.grid(True, alpha=0.3)
    # Add iso-F1 curves
    for f1_val in [0.1, 0.2, 0.3, 0.5]:
        p_range = np.linspace(0.01, 1.0, 100)
        r_range = f1_val * p_range / (2 * p_range - f1_val)
        valid = (r_range > 0) & (r_range <= 1)
        ax.plot(p_range[valid], r_range[valid], "--", color="gray", alpha=0.2, linewidth=0.8)

    # Panel C: Clan-level ARI
    ax = axes[2]
    for tool in tools:
        aris = []
        ts = []
        for t in thresholds:
            r = data[str(t)].get(tool, {})
            if "error" not in r and r and "clan_ARI" in r:
                aris.append(r["clan_ARI"])
                ts.append(t)
        if ts:
            ax.plot(ts, aris, marker=MARKERS[tool], color=COLORS[tool],
                    label=tool, linewidth=2, markersize=7)
    ax.set_xlabel("Sequence Identity Threshold")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("C. Clan-Level Clustering Quality")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0.2, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "figure1_clustering_quality.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "figure1_clustering_quality.png")
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Speed-Quality Tradeoff
# ══════════════════════════════════════════════════════════════════════

def figure2_speed_quality():
    """2-panel figure: (A) Runtime vs ARI at t=0.3 and t=0.5, (B) Runtime bars annotated with ARI."""
    data = load_large_pfam()
    tools = ["ClustKIT", "MMseqs2", "DeepClust", "CD-HIT", "Linclust"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Speed vs ARI scatter (t=0.3 and t=0.5)
    ax = axes[0]
    for tool in tools:
        for t, marker_fill in [(0.3, "full"), (0.5, "none")]:
            r = data[str(t)].get(tool, {})
            if "error" in r or not r:
                continue
            ari = r["family_ARI"]
            runtime = r["runtime"]
            fillstyle = marker_fill
            ms = 10 if t == 0.3 else 8
            edge = COLORS[tool]
            face = COLORS[tool] if fillstyle == "full" else "white"
            label = f"{tool}" if t == 0.3 else None
            ax.scatter(runtime, ari, marker=MARKERS[tool], c=face,
                      edgecolors=edge, s=ms**2, label=label, linewidths=1.5, zorder=5)
            ax.annotate(f"t={t}", (runtime, ari),
                       textcoords="offset points", xytext=(8, -3), fontsize=7)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Family-Level ARI")
    ax.set_xscale("log")
    ax.set_title("A. Speed vs Quality Tradeoff")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel B: Grouped bar chart at t=0.3
    ax = axes[1]
    t = "0.3"
    bar_tools = [tool for tool in tools if tool in data[t] and "error" not in data[t][tool]]
    x = np.arange(len(bar_tools))
    aris = [data[t][tool]["family_ARI"] for tool in bar_tools]
    runtimes = [data[t][tool]["runtime"] for tool in bar_tools]
    colors = [COLORS[tool] for tool in bar_tools]

    bars = ax.bar(x, aris, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
    for i, (bar, rt) in enumerate(zip(bars, runtimes)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rt:.0f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_tools, rotation=30, ha="right")
    ax.set_ylabel("Family-Level ARI")
    ax.set_title("B. Clustering Quality at t=0.3 (runtime annotated)")
    ax.set_ylim(0, 0.75)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "figure2_speed_quality.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "figure2_speed_quality.png")
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Ablation & Additional Analysis
# ══════════════════════════════════════════════════════════════════════

def figure3_ablation():
    """3-panel: (A) Per-family-size ARI, (B) Bootstrap CIs, (C) Cluster count vs threshold."""
    data = load_large_pfam()
    metrics = load_paper_metrics()
    thresholds = [0.3, 0.5, 0.7, 0.9]
    tools = ["ClustKIT", "MMseqs2", "DeepClust"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel A: Per-family-size ARI at t=0.3
    ax = axes[0]
    size_bins = ["small (10-30)", "medium (31-100)", "large (101-200)"]
    x = np.arange(len(size_bins))
    width = 0.25
    for i, tool in enumerate(tools):
        vals = []
        for bin_name in size_bins:
            key = f"{tool}_t0.3_{bin_name}"
            r = metrics.get("per_family_size", {}).get(key, {})
            vals.append(r.get("ARI", 0))
        ax.bar(x + i * width, vals, width, label=tool, color=COLORS[tool],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Small\n(10-30)", "Medium\n(31-100)", "Large\n(101-200)"])
    ax.set_ylabel("ARI at t=0.3")
    ax.set_title("A. Quality by Family Size")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Bootstrap CIs
    ax = axes[1]
    ci_data = metrics.get("bootstrap_ci", {})
    ci_tools_t03 = []
    for tool in ["ClustKIT", "MMseqs2", "DeepClust", "Linclust"]:
        key = f"{tool}_t0.3"
        if key in ci_data:
            ci_tools_t03.append((tool, ci_data[key]))
    if ci_tools_t03:
        y_pos = np.arange(len(ci_tools_t03))
        for i, (tool, ci) in enumerate(ci_tools_t03):
            ari = ci["ARI"]
            lo = ci["CI_lo"]
            hi = ci["CI_hi"]
            ax.barh(i, ari, color=COLORS[tool], height=0.5,
                    edgecolor="black", linewidth=0.5)
            ax.errorbar(ari, i, xerr=[[ari - lo], [hi - ari]],
                       fmt="none", color="black", capsize=4, linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([t[0] for t in ci_tools_t03])
        ax.set_xlabel("ARI (95% Bootstrap CI)")
        ax.set_title("B. Statistical Significance (t=0.3)")
        ax.grid(True, alpha=0.3, axis="x")

    # Panel C: Number of clusters vs threshold
    ax = axes[2]
    for tool in ["ClustKIT", "MMseqs2", "DeepClust", "Linclust"]:
        ns = []
        ts = []
        for t in thresholds:
            r = data[str(t)].get(tool, {})
            if "error" not in r and r:
                ns.append(r["n_predicted_clusters"])
                ts.append(t)
        if ts:
            ax.plot(ts, ns, marker=MARKERS[tool], color=COLORS[tool],
                    label=tool, linewidth=2, markersize=7)
    # Add ground truth line
    ax.axhline(y=1642, color="black", linestyle="--", linewidth=1, alpha=0.7, label="True families (1642)")
    ax.set_xlabel("Sequence Identity Threshold")
    ax.set_ylabel("Number of Clusters")
    ax.set_title("C. Cluster Count vs Threshold")
    ax.set_yscale("log")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7.5)
    ax.set_xlim(0.2, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "figure3_ablation.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "figure3_ablation.png")
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES
# ══════════════════════════════════════════════════════════════════════

def supp_small_pfam():
    """Small Pfam benchmark with fine-grained thresholds (7 thresholds, 6 tools)."""
    data = load_small_pfam()
    thresholds = sorted([float(t) for t in data.keys()])
    tools_map = {"clustkit": "ClustKIT", "mmseqs2": "MMseqs2", "deepclust": "DeepClust",
                 "cdhit": "CD-HIT", "linclust": "Linclust", "vsearch": "VSEARCH"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, tool in tools_map.items():
        aris = []
        ts = []
        for t in thresholds:
            r = data[str(t)].get(key, {})
            if "error" not in r and r:
                aris.append(r["ARI"])
                ts.append(t)
        if ts:
            ax.plot(ts, aris, marker=MARKERS.get(tool, "o"), color=COLORS.get(tool, "gray"),
                    label=tool, linewidth=2, markersize=6)
    ax.set_xlabel("Sequence Identity Threshold")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("Pfam Concordance (22K sequences, 56 families, Leiden + SW)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.25, 0.95)
    plt.tight_layout()
    out = FIG_DIR / "supp_small_pfam.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "supp_small_pfam.png")
    plt.close()
    print(f"  Saved {out}")


def supp_coverage():
    """Coverage and cluster size analysis."""
    metrics = load_paper_metrics()
    cov = metrics.get("coverage", {})
    sizes = metrics.get("cluster_sizes", {})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Coverage bar chart
    ax = axes[0]
    tools = ["ClustKIT", "MMseqs2", "DeepClust", "Linclust"]
    for i_t, t in enumerate(["0.3", "0.5"]):
        x = np.arange(len(tools))
        vals = []
        for tool in tools:
            key = f"{tool}_t{t}"
            vals.append(cov.get(key, {}).get("pct_covered", 0))
        offset = (i_t - 0.5) * 0.35
        colors_t = [COLORS[tool] for tool in tools]
        alpha = 1.0 if t == "0.3" else 0.6
        bars = ax.bar(x + offset, vals, 0.3, label=f"t={t}",
                     color=colors_t, alpha=alpha, edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(tools)))
    ax.set_xticklabels(tools)
    ax.set_ylabel("Sequences in Non-Singleton Clusters (%)")
    ax.set_title("A. Clustering Coverage")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(60, 100)

    # Panel B: Cluster size distribution at t=0.3
    ax = axes[1]
    size_cats = ["1", "2-5", "6-20", "21-100", "101-500", ">500"]
    for tool in ["ClustKIT", "MMseqs2", "DeepClust"]:
        key = f"{tool}_t0.3"
        if key in sizes:
            dist = sizes[key].get("size_distribution", {})
            vals = [dist.get(cat, 0) for cat in size_cats]
            ax.plot(range(len(size_cats)), vals, marker=MARKERS[tool],
                   color=COLORS[tool], label=tool, linewidth=2, markersize=6)
    ax.set_xticks(range(len(size_cats)))
    ax.set_xticklabels(size_cats)
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Number of Clusters")
    ax.set_yscale("log")
    ax.set_title("B. Cluster Size Distribution (t=0.3)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "supp_coverage_sizes.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "supp_coverage_sizes.png")
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating publication figures...\n")

    print("Main figures:")
    figure1_clustering_quality()
    figure2_speed_quality()
    figure3_ablation()

    print("\nSupplementary figures:")
    supp_small_pfam()
    supp_coverage()

    print(f"\nAll figures saved to {FIG_DIR}/")
