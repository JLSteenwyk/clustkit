"""Generate cluster size distribution plot."""

from collections import Counter
from pathlib import Path

import numpy as np


def plot_cluster_sizes(clusters_tsv: Path, output_dir: Path):
    """Read clusters.tsv and generate a cluster size distribution figure.

    Produces output_dir/cluster_size_distribution.png with two panels:
      (a) Histogram of cluster sizes (log-scaled x-axis)
      (b) Cumulative sequence coverage vs. minimum cluster size

    Args:
        clusters_tsv: Path to clusters.tsv (sequence_id, cluster_id, is_representative).
        output_dir: Directory to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pypubfigs import friendly_pal, theme_simple

    # Apply pypubfigs theme
    theme_simple()
    palette = friendly_pal("contrast_three")

    # Parse cluster assignments
    cluster_ids = []
    with open(clusters_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] != "sequence_id":
                cluster_ids.append(int(parts[1]))

    if not cluster_ids:
        return

    # Compute cluster sizes
    size_counts = Counter(Counter(cluster_ids).values())
    total_seqs = len(cluster_ids)
    cluster_sizes = sorted(Counter(cluster_ids).values(), reverse=True)
    n_clusters = len(cluster_sizes)

    fig, (ax_hist, ax_cum) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) Cluster size histogram
    max_size = max(cluster_sizes)
    if max_size > 1:
        bins = np.logspace(0, np.log10(max_size + 1), 30)
    else:
        bins = np.arange(0.5, 3.5)

    ax_hist.hist(cluster_sizes, bins=bins, color=palette[0],
                 edgecolor="white", linewidth=0.5)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Cluster size")
    ax_hist.set_ylabel("Number of clusters")
    ax_hist.set_title(f"{n_clusters:,} clusters, {total_seqs:,} sequences",
                      fontsize=9)

    # Annotate singleton count
    n_singletons = size_counts.get(1, 0)
    if n_singletons > 0:
        pct = n_singletons / n_clusters * 100
        ax_hist.text(0.95, 0.95, f"Singletons: {n_singletons:,} ({pct:.0f}%)",
                     transform=ax_hist.transAxes, ha="right", va="top",
                     fontsize=8, color="#555555")

    # (b) Cumulative coverage
    sorted_sizes = np.array(sorted(cluster_sizes))
    cum_seqs = np.cumsum(sorted_sizes)
    # Fraction of sequences in clusters of size >= threshold
    thresholds = np.arange(1, max_size + 1)
    coverage = []
    for t in thresholds:
        n_in = sum(s for s in cluster_sizes if s >= t)
        coverage.append(n_in / total_seqs * 100)

    ax_cum.plot(thresholds, coverage, color=palette[1], linewidth=1.5)
    ax_cum.set_xscale("log")
    ax_cum.set_xlabel("Minimum cluster size")
    ax_cum.set_ylabel("Sequences covered (%)")
    ax_cum.set_ylim(0, 105)
    ax_cum.axhline(50, color="#cccccc", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cluster_size_distribution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path
