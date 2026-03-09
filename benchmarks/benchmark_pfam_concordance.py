"""Pfam Concordance Benchmark

Downloads sequences from multiple Pfam families, mixes them into a single
dataset, clusters with ClustKIT at various thresholds, and measures how well
the output clusters recover the known Pfam family labels.

Metrics:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Homogeneity, Completeness, V-measure
  - Pairwise Precision, Recall, F1
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.pipeline import run_pipeline


DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "pfam_benchmark_results"


def load_and_mix_families(data_dir: Path, max_per_family: int = 500):
    """Load all Pfam family FASTAs and mix into one dataset with ground truth labels.

    Args:
        data_dir: Directory containing per-family FASTA files.
        max_per_family: Cap per family to keep dataset balanced. 0 = no cap.

    Returns:
        Tuple of (mixed_fasta_path, ground_truth dict mapping seq_id -> family).
    """
    fasta_files = sorted(data_dir.glob("PF*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No Pfam FASTA files found in {data_dir}")

    ground_truth = {}  # seq_id -> pfam_family
    mixed_fasta_path = data_dir.parent / "pfam_mixed.fasta"

    with open(mixed_fasta_path, "w") as out:
        for fasta_file in fasta_files:
            pfam_id = fasta_file.stem.split("_")[0]  # e.g. "PF00042"
            count = 0
            current_header = None
            current_seq_lines = []

            with open(fasta_file) as f:
                for line in f:
                    line = line.rstrip("\n\r")
                    if line.startswith(">"):
                        # Write previous record
                        if current_header is not None and (
                            max_per_family == 0 or count < max_per_family
                        ):
                            seq_id = current_header.split()[0][1:]
                            ground_truth[seq_id] = pfam_id
                            out.write(current_header + "\n")
                            out.write("\n".join(current_seq_lines) + "\n")
                            count += 1

                        current_header = line
                        current_seq_lines = []
                    elif current_header is not None:
                        current_seq_lines.append(line)

                # Last record
                if current_header is not None and (
                    max_per_family == 0 or count < max_per_family
                ):
                    seq_id = current_header.split()[0][1:]
                    ground_truth[seq_id] = pfam_id
                    out.write(current_header + "\n")
                    out.write("\n".join(current_seq_lines) + "\n")

    family_counts = Counter(ground_truth.values())
    print(f"Mixed dataset: {len(ground_truth)} sequences from {len(family_counts)} families")
    for fam, cnt in sorted(family_counts.items()):
        print(f"  {fam}: {cnt} sequences")

    return mixed_fasta_path, ground_truth


def pairwise_precision_recall_f1(true_labels, pred_labels):
    """Compute pairwise precision, recall, and F1.

    A pair is a "positive" if two sequences are in the same predicted cluster.
    A "true positive" means they are also in the same ground-truth family.

    Uses sampling for large datasets to avoid O(N^2) computation.
    """
    n = len(true_labels)

    # For large N, sample pairs instead of computing all N*(N-1)/2
    if n > 5000:
        num_samples = 2_000_000
        rng = np.random.RandomState(42)
        idx_a = rng.randint(0, n, size=num_samples)
        idx_b = rng.randint(0, n, size=num_samples)
        # Remove self-pairs
        valid = idx_a != idx_b
        idx_a = idx_a[valid]
        idx_b = idx_b[valid]
    else:
        # All pairs
        idx_a, idx_b = [], []
        for i in range(n):
            for j in range(i + 1, n):
                idx_a.append(i)
                idx_b.append(j)
        idx_a = np.array(idx_a)
        idx_b = np.array(idx_b)

    same_pred = pred_labels[idx_a] == pred_labels[idx_b]
    same_true = true_labels[idx_a] == true_labels[idx_b]

    tp = int(np.sum(same_pred & same_true))
    fp = int(np.sum(same_pred & ~same_true))
    fn = int(np.sum(~same_pred & same_true))
    tn = int(np.sum(~same_pred & ~same_true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "pairwise_TP": tp,
        "pairwise_FP": fp,
        "pairwise_FN": fn,
        "pairwise_TN": tn,
        "pairwise_precision": round(precision, 4),
        "pairwise_recall": round(recall, 4),
        "pairwise_F1": round(f1, 4),
        "pairwise_accuracy": round(accuracy, 4),
    }


def evaluate_clustering(ground_truth, cluster_tsv_path):
    """Evaluate a clustering result against Pfam ground truth.

    Returns dict of all metrics.
    """
    # Read cluster assignments
    pred_map = {}  # seq_id -> cluster_id
    with open(cluster_tsv_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            pred_map[parts[0]] = int(parts[1])

    # Align ground truth and predictions (only sequences present in both)
    common_ids = sorted(set(ground_truth.keys()) & set(pred_map.keys()))
    if not common_ids:
        raise ValueError("No common sequence IDs between ground truth and predictions")

    # Encode labels as integer arrays
    true_family_list = [ground_truth[sid] for sid in common_ids]
    pred_cluster_list = [pred_map[sid] for sid in common_ids]

    # Map string labels to ints for numpy operations
    family_to_int = {f: i for i, f in enumerate(sorted(set(true_family_list)))}
    true_labels = np.array([family_to_int[f] for f in true_family_list], dtype=np.int32)
    pred_labels = np.array(pred_cluster_list, dtype=np.int32)

    # Sklearn metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_labels, pred_labels
    )

    # Pairwise metrics
    pw = pairwise_precision_recall_f1(true_labels, pred_labels)

    # Cluster stats
    n_true_families = len(set(true_family_list))
    n_pred_clusters = len(set(pred_cluster_list))

    results = {
        "n_sequences": len(common_ids),
        "n_true_families": n_true_families,
        "n_predicted_clusters": n_pred_clusters,
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "homogeneity": round(homogeneity, 4),
        "completeness": round(completeness, 4),
        "V_measure": round(v_measure, 4),
        **pw,
    }

    return results


def run_benchmark(thresholds=None, max_per_family=500):
    """Run the full Pfam concordance benchmark."""
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 0.9]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and mix families
    print("=" * 70)
    print("PFAM CONCORDANCE BENCHMARK")
    print("=" * 70)
    print()

    mixed_fasta, ground_truth = load_and_mix_families(DATA_DIR, max_per_family)
    print()

    all_results = {}

    for threshold in thresholds:
        print("-" * 70)
        print(f"Clustering at threshold = {threshold}")
        print("-" * 70)

        out_dir = OUTPUT_DIR / f"threshold_{threshold}"
        out_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "input": mixed_fasta,
            "output": out_dir,
            "threshold": threshold,
            "mode": "protein",
            "alignment": "kmer",
            "sketch_size": 128,
            "kmer_size": 5,
            "sensitivity": "high",
            "cluster_method": "connected",
            "representative": "longest",
            "device": "cpu",
            "threads": 1,
            "format": "tsv",
        }

        start = time.perf_counter()
        run_pipeline(config)
        elapsed = time.perf_counter() - start

        # Evaluate
        cluster_tsv = out_dir / "clusters.tsv"
        results = evaluate_clustering(ground_truth, cluster_tsv)
        results["runtime_seconds"] = round(elapsed, 2)

        all_results[str(threshold)] = results

        print()
        print(f"  Results @ t={threshold}:")
        print(f"    Sequences:  {results['n_sequences']}")
        print(f"    Families:   {results['n_true_families']}")
        print(f"    Clusters:   {results['n_predicted_clusters']}")
        print(f"    ARI:        {results['ARI']}")
        print(f"    NMI:        {results['NMI']}")
        print(f"    Homogeneity:  {results['homogeneity']}")
        print(f"    Completeness: {results['completeness']}")
        print(f"    V-measure:    {results['V_measure']}")
        print(f"    Pairwise Precision: {results['pairwise_precision']}")
        print(f"    Pairwise Recall:    {results['pairwise_recall']}")
        print(f"    Pairwise F1:        {results['pairwise_F1']}")
        print(f"    Pairwise Accuracy:  {results['pairwise_accuracy']}")
        print(f"    Runtime:    {results['runtime_seconds']}s")
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Threshold':>10} {'Clusters':>10} {'ARI':>8} {'NMI':>8} {'Homog':>8} {'Compl':>8} {'P(pw)':>8} {'R(pw)':>8} {'F1(pw)':>8} {'Acc(pw)':>8}"
    print(header)
    print("-" * len(header))
    for t in thresholds:
        r = all_results[str(t)]
        print(
            f"{t:>10.1f} {r['n_predicted_clusters']:>10} "
            f"{r['ARI']:>8.4f} {r['NMI']:>8.4f} "
            f"{r['homogeneity']:>8.4f} {r['completeness']:>8.4f} "
            f"{r['pairwise_precision']:>8.4f} {r['pairwise_recall']:>8.4f} "
            f"{r['pairwise_F1']:>8.4f} {r['pairwise_accuracy']:>8.4f}"
        )

    # Save results
    results_file = OUTPUT_DIR / "pfam_concordance_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    run_benchmark()
