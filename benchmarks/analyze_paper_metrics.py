"""Compute additional paper metrics from existing benchmark results.

1. Cluster size distributions (all tools, t=0.3 and t=0.5)
2. Per-family-size sensitivity analysis
3. Bootstrap CIs on ARI
4. Coverage analysis (singletons, cluster sizes)
5. Runtime breakdown from logs
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PFAM_LARGE_DIR = Path(__file__).resolve().parent / "data" / "pfam_large_results"
PFAM_FULL_DIR = Path(__file__).resolve().parent / "data" / "pfam_full"
OUT_DIR = Path(__file__).resolve().parent / "data" / "paper_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth():
    with open(PFAM_FULL_DIR / "ground_truth.json") as f:
        gt = json.load(f)
    return gt["protein_family"], gt["family_clan"]


def normalize_id(seq_id):
    parts = seq_id.split("|")
    return parts[1] if len(parts) >= 2 else seq_id


def load_clusters(tool_name, threshold):
    """Load cluster assignments for a tool from the large Pfam benchmark."""
    if tool_name == "ClustKIT":
        tsv = PFAM_LARGE_DIR / f"clustkit_t{threshold}" / "clusters.tsv"
        if not tsv.exists():
            return None
        clusters = {}
        with open(tsv) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                clusters[normalize_id(parts[0])] = int(parts[1])
        return clusters

    # MMseqs2, Linclust, DeepClust — 2-col TSV
    name_map = {"MMseqs2": "mmseqs", "Linclust": "linclust", "DeepClust": "deepclust"}
    prefix = name_map.get(tool_name, tool_name.lower())
    tsv = PFAM_LARGE_DIR / f"{prefix}_t{threshold}_cluster.tsv"
    if not tsv.exists():
        tsv = PFAM_LARGE_DIR / f"{prefix}_t{threshold}.tsv"
    if not tsv.exists():
        return None
    clusters = {}
    rep_to_id = {}
    next_id = 0
    with open(tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            rep, member = normalize_id(parts[0]), normalize_id(parts[1])
            if rep not in rep_to_id:
                rep_to_id[rep] = next_id
                next_id += 1
            clusters[member] = rep_to_id[rep]
    return clusters


# ──────────────────────────────────────────────────────────────────────
# 1. Cluster size distributions
# ──────────────────────────────────────────────────────────────────────

def analyze_cluster_sizes(protein_family):
    """Compute cluster size distributions for all tools."""
    print("\n" + "=" * 80)
    print("1. CLUSTER SIZE DISTRIBUTIONS")
    print("=" * 80)

    results = {}
    for threshold in [0.3, 0.5]:
        for tool in ["ClustKIT", "MMseqs2", "DeepClust", "Linclust"]:
            clusters = load_clusters(tool, threshold)
            if clusters is None:
                continue

            # Count cluster sizes
            cluster_sizes = Counter(clusters.values())
            sizes = sorted(cluster_sizes.values())
            n_clusters = len(cluster_sizes)
            n_singletons = sum(1 for s in sizes if s == 1)
            n_seqs = sum(sizes)
            max_size = max(sizes) if sizes else 0
            median_size = np.median(sizes) if sizes else 0
            mean_size = np.mean(sizes) if sizes else 0

            key = f"{tool}_t{threshold}"
            results[key] = {
                "n_clusters": n_clusters,
                "n_singletons": n_singletons,
                "pct_singletons": round(100 * n_singletons / n_clusters, 1) if n_clusters else 0,
                "n_sequences": n_seqs,
                "max_cluster_size": max_size,
                "median_cluster_size": round(float(median_size), 1),
                "mean_cluster_size": round(float(mean_size), 1),
                "size_distribution": {
                    "1": sum(1 for s in sizes if s == 1),
                    "2-5": sum(1 for s in sizes if 2 <= s <= 5),
                    "6-20": sum(1 for s in sizes if 6 <= s <= 20),
                    "21-100": sum(1 for s in sizes if 21 <= s <= 100),
                    "101-500": sum(1 for s in sizes if 101 <= s <= 500),
                    ">500": sum(1 for s in sizes if s > 500),
                },
            }
            print(f"  {key}: {n_clusters} clusters, {n_singletons} singletons "
                  f"({results[key]['pct_singletons']}%), max={max_size}, median={median_size:.0f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# 2. Per-family-size sensitivity analysis
# ──────────────────────────────────────────────────────────────────────

def analyze_per_family_size(protein_family, family_clan):
    """Analyze clustering quality as a function of true family size."""
    print("\n" + "=" * 80)
    print("2. PER-FAMILY-SIZE SENSITIVITY")
    print("=" * 80)

    gt_norm = {normalize_id(k): v for k, v in protein_family.items()}
    family_sizes = Counter(gt_norm.values())

    # Bin families by size
    size_bins = {
        "small (10-30)": [f for f, s in family_sizes.items() if 10 <= s <= 30],
        "medium (31-100)": [f for f, s in family_sizes.items() if 31 <= s <= 100],
        "large (101-200)": [f for f, s in family_sizes.items() if 101 <= s <= 200],
    }

    results = {}
    for threshold in [0.3, 0.5]:
        for tool in ["ClustKIT", "MMseqs2", "DeepClust"]:
            clusters = load_clusters(tool, threshold)
            if clusters is None:
                continue

            common = sorted(set(gt_norm.keys()) & set(clusters.keys()))
            if not common:
                continue

            for bin_name, families in size_bins.items():
                # Filter to sequences in this bin
                bin_seqs = [s for s in common if gt_norm[s] in families]
                if len(bin_seqs) < 10:
                    continue

                true_l = [gt_norm[s] for s in bin_seqs]
                pred_l = [clusters[s] for s in bin_seqs]
                ari = adjusted_rand_score(true_l, pred_l)

                key = f"{tool}_t{threshold}_{bin_name}"
                results[key] = {"ARI": round(ari, 4), "n_seqs": len(bin_seqs),
                                "n_families": len(families)}
                print(f"  {tool} t={threshold} {bin_name}: ARI={ari:.4f} "
                      f"({len(bin_seqs)} seqs, {len(families)} families)")

    return results


# ──────────────────────────────────────────────────────────────────────
# 3. Bootstrap CIs on ARI
# ──────────────────────────────────────────────────────────────────────

def bootstrap_ari(protein_family, n_bootstrap=1000, seed=42):
    """Compute bootstrap 95% CIs on ARI for all tools."""
    print("\n" + "=" * 80)
    print("3. BOOTSTRAP 95% CIs ON ARI")
    print("=" * 80)

    gt_norm = {normalize_id(k): v for k, v in protein_family.items()}
    rng = np.random.RandomState(seed)

    results = {}
    for threshold in [0.3, 0.5]:
        for tool in ["ClustKIT", "MMseqs2", "DeepClust", "Linclust"]:
            clusters = load_clusters(tool, threshold)
            if clusters is None:
                continue

            common = sorted(set(gt_norm.keys()) & set(clusters.keys()))
            if not common:
                continue

            true_arr = np.array([gt_norm[s] for s in common])
            pred_arr = np.array([clusters[s] for s in common])
            n = len(common)

            # Point estimate
            point_ari = adjusted_rand_score(true_arr, pred_arr)

            # Bootstrap
            boot_aris = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                boot_ari = adjusted_rand_score(true_arr[idx], pred_arr[idx])
                boot_aris.append(boot_ari)

            boot_aris = np.array(boot_aris)
            ci_lo = np.percentile(boot_aris, 2.5)
            ci_hi = np.percentile(boot_aris, 97.5)

            key = f"{tool}_t{threshold}"
            results[key] = {
                "ARI": round(point_ari, 4),
                "CI_lo": round(float(ci_lo), 4),
                "CI_hi": round(float(ci_hi), 4),
                "CI_width": round(float(ci_hi - ci_lo), 4),
            }
            print(f"  {tool} t={threshold}: ARI={point_ari:.4f} "
                  f"[{ci_lo:.4f}, {ci_hi:.4f}] (width={ci_hi-ci_lo:.4f})")

    return results


# ──────────────────────────────────────────────────────────────────────
# 4. Coverage analysis
# ──────────────────────────────────────────────────────────────────────

def analyze_coverage(protein_family):
    """What fraction of true family members are covered by non-singleton clusters?"""
    print("\n" + "=" * 80)
    print("4. COVERAGE ANALYSIS")
    print("=" * 80)

    gt_norm = {normalize_id(k): v for k, v in protein_family.items()}

    results = {}
    for threshold in [0.3, 0.5]:
        for tool in ["ClustKIT", "MMseqs2", "DeepClust", "Linclust"]:
            clusters = load_clusters(tool, threshold)
            if clusters is None:
                continue

            # Find non-singleton clusters
            cluster_sizes = Counter(clusters.values())
            non_singleton_clusters = {c for c, s in cluster_sizes.items() if s >= 2}
            covered = sum(1 for seq, c in clusters.items()
                         if c in non_singleton_clusters and seq in gt_norm)
            total = sum(1 for seq in clusters if seq in gt_norm)

            key = f"{tool}_t{threshold}"
            results[key] = {
                "covered": covered,
                "total": total,
                "pct_covered": round(100 * covered / total, 1) if total else 0,
                "n_non_singleton_clusters": len(non_singleton_clusters),
            }
            print(f"  {tool} t={threshold}: {covered}/{total} seqs covered "
                  f"({results[key]['pct_covered']}%), "
                  f"{len(non_singleton_clusters)} non-singleton clusters")

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    protein_family, family_clan = load_ground_truth()
    print(f"Ground truth: {len(protein_family)} proteins, "
          f"{len(set(protein_family.values()))} families, "
          f"{len(set(family_clan.values()))} clans\n")

    all_results = {}
    all_results["cluster_sizes"] = analyze_cluster_sizes(protein_family)
    all_results["per_family_size"] = analyze_per_family_size(protein_family, family_clan)
    all_results["bootstrap_ci"] = bootstrap_ari(protein_family)
    all_results["coverage"] = analyze_coverage(protein_family)

    out_file = OUT_DIR / "paper_metrics.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_file}")
