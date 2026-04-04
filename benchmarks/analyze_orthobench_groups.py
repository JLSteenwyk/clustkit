"""Analyze OrthoBench results at the orthogroup level (not just pairwise).

For each tool, compute:
1. How many of the 70 RefOGs are exactly recovered
2. How many are "mostly correct" (>=80% of members in one cluster)
3. Fission rate: how many clusters does each RefOG get split into
4. Fusion rate: how many RefOGs are merged into a single cluster
5. Per-RefOG Jaccard similarity (best matching cluster)
"""

import json
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

OB_DIR = Path(__file__).resolve().parent / "data" / "open_orthobench" / "BENCHMARKS"
RESULTS_DIR = Path(__file__).resolve().parent / "data" / "orthobench_results"


def load_refogs():
    """Load the 70 reference orthogroups."""
    refogs = []
    for i in range(1, 71):
        fn = OB_DIR / "RefOGs" / f"RefOG{i:03d}.txt"
        with open(fn) as f:
            genes = set(line.strip() for line in f if line.strip())
        refogs.append(genes)
    return refogs


def load_predicted_clusters(orthogroups_file):
    """Load predicted orthogroups from the benchmark output file."""
    gene_pat = re.compile(
        r"WBGene00\d+\.1|ENSCAFP\d+|ENSCINP\d+|ENSDARP\d+|FBpp0\d+|"
        r"ENSGALP\d+|ENSP000\d+|ENSMODP\d+|ENSMUSP\d+|ENSPTRP\d+|"
        r"ENSRNOP\d+|ENSTNIP\d+"
    )
    clusters = []
    with open(orthogroups_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            genes = set(re.findall(gene_pat, line))
            if genes:
                clusters.append(genes)
    return clusters


def analyze_tool(refogs, clusters, tool_name):
    """Analyze how well predicted clusters recover reference orthogroups."""
    n_refogs = len(refogs)

    # Build gene → cluster_id mapping
    gene_to_cluster = {}
    for ci, cluster in enumerate(clusters):
        for gene in cluster:
            gene_to_cluster[gene] = ci

    exact_correct = 0
    mostly_correct = 0  # >=80% in one cluster
    half_correct = 0    # >=50% in one cluster
    fission_counts = []  # how many clusters each RefOG is split into
    best_jaccards = []   # best Jaccard for each RefOG
    best_precisions = []
    best_recalls = []

    for ri, refog in enumerate(refogs):
        # Find which clusters contain RefOG members
        cluster_overlaps = Counter()
        for gene in refog:
            ci = gene_to_cluster.get(gene)
            if ci is not None:
                cluster_overlaps[ci] += 1

        n_ref = len(refog)
        n_clusters_hit = len(cluster_overlaps)
        fission_counts.append(n_clusters_hit)

        if n_clusters_hit == 0:
            best_jaccards.append(0.0)
            best_precisions.append(0.0)
            best_recalls.append(0.0)
            continue

        # Find the best matching cluster (highest Jaccard)
        best_jaccard = 0.0
        best_p = 0.0
        best_r = 0.0
        for ci, overlap in cluster_overlaps.items():
            cluster_size = len(clusters[ci])
            jaccard = overlap / (n_ref + cluster_size - overlap)
            precision = overlap / cluster_size  # what fraction of cluster is RefOG
            recall = overlap / n_ref  # what fraction of RefOG is in this cluster
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_p = precision
                best_r = recall

        best_jaccards.append(best_jaccard)
        best_precisions.append(best_p)
        best_recalls.append(best_r)

        # Check exact match: one cluster contains all RefOG members and nothing else
        for ci, overlap in cluster_overlaps.items():
            if overlap == n_ref and len(clusters[ci]) == n_ref:
                exact_correct += 1
                break

        # Check mostly correct: >=80% in best cluster
        best_overlap = max(cluster_overlaps.values())
        if best_overlap >= 0.8 * n_ref:
            mostly_correct += 1
        if best_overlap >= 0.5 * n_ref:
            half_correct += 1

    # Fusion: how many RefOGs share the same best cluster
    refog_best_cluster = []
    for ri, refog in enumerate(refogs):
        cluster_overlaps = Counter()
        for gene in refog:
            ci = gene_to_cluster.get(gene)
            if ci is not None:
                cluster_overlaps[ci] += 1
        if cluster_overlaps:
            best_ci = cluster_overlaps.most_common(1)[0][0]
            refog_best_cluster.append(best_ci)
        else:
            refog_best_cluster.append(-1)

    # Count fusions: multiple RefOGs mapping to same cluster
    cluster_to_refogs = defaultdict(list)
    for ri, ci in enumerate(refog_best_cluster):
        if ci >= 0:
            cluster_to_refogs[ci].append(ri)
    n_fused = sum(1 for refogs_in_cluster in cluster_to_refogs.values() if len(refogs_in_cluster) > 1)

    return {
        "exact_correct": exact_correct,
        "mostly_correct_80pct": mostly_correct,
        "half_correct_50pct": half_correct,
        "mean_fission": round(float(np.mean(fission_counts)), 2),
        "median_fission": int(np.median(fission_counts)),
        "n_fused_clusters": n_fused,
        "mean_best_jaccard": round(float(np.mean(best_jaccards)), 4),
        "mean_best_precision": round(float(np.mean(best_precisions)), 4),
        "mean_best_recall": round(float(np.mean(best_recalls)), 4),
    }


if __name__ == "__main__":
    refogs = load_refogs()
    print(f"Loaded {len(refogs)} reference orthogroups "
          f"({sum(len(r) for r in refogs)} total genes)\n")

    # Find all orthogroup result files
    og_files = sorted(RESULTS_DIR.glob("*_orthogroups.txt"))
    if not og_files:
        print("No orthogroup files found!")
        sys.exit(1)

    print(f"{'Tool':<25} {'t':<5} {'Exact':>6} {'>=80%':>6} {'>=50%':>6} "
          f"{'Fission':>8} {'Fusion':>7} {'Jaccard':>8} {'BestP':>7} {'BestR':>7}")
    print("-" * 95)

    all_results = {}
    for og_file in og_files:
        name = og_file.stem.replace("_orthogroups", "")
        # Parse tool and threshold from filename
        clusters = load_predicted_clusters(og_file)
        if not clusters:
            continue
        result = analyze_tool(refogs, clusters, name)
        all_results[name] = result

        print(f"{name:<25} {'':>5} {result['exact_correct']:>6} "
              f"{result['mostly_correct_80pct']:>6} {result['half_correct_50pct']:>6} "
              f"{result['mean_fission']:>8.1f} {result['n_fused_clusters']:>7} "
              f"{result['mean_best_jaccard']:>8.4f} "
              f"{result['mean_best_precision']:>7.4f} {result['mean_best_recall']:>7.4f}")

    # Save
    out_file = RESULTS_DIR / "orthobench_group_analysis.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")
