"""Quick accuracy validation for optimization testing.

Runs ClustKIT on the Pfam dataset at t=0.4, 0.5, 0.7 and reports ARI, F1,
number of clusters, and number of filtered pairs. Used to verify that
optimizations don't degrade accuracy.

Usage:
    python benchmarks/validate_accuracy.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.sketch import compute_sketches
from clustkit.lsh import lsh_candidates
from clustkit.pairwise import compute_pairwise_alignment
from clustkit.graph import build_similarity_graph
from clustkit.cluster import cluster_sequences
from clustkit.utils import auto_kmer_for_lsh, auto_lsh_params

DATA_DIR = Path(__file__).resolve().parent / "data"
FAMILIES_DIR = DATA_DIR / "pfam_families"
INPUT = DATA_DIR / "pfam_mixed.fasta"


def build_ground_truth(dataset):
    """Build ground truth labels from Pfam family names in sequence IDs."""
    family_map = {}
    labels = []
    for rec in dataset.records:
        # IDs are like "FAMILYNAME_SEQID" or contain family info in description
        # Check common formats
        family = rec.id.split("/")[0]  # e.g., "PF00001_sp|Q9Y5Y4|CLC7B_HUMAN/1-291"
        if "_" in family:
            family = family.rsplit("_", 1)[0]
        if family not in family_map:
            family_map[family] = len(family_map)
        labels.append(family_map[family])
    return np.array(labels, dtype=np.int32)


def build_ground_truth_from_files(dataset):
    """Build ground truth from the family FASTA files."""
    if not FAMILIES_DIR.exists():
        return None
    family_files = sorted(FAMILIES_DIR.glob("*.fasta"))
    if not family_files:
        return None

    # Map sequence ID -> family index
    id_to_family = {}
    for fam_idx, fam_file in enumerate(family_files):
        with open(fam_file) as f:
            for line in f:
                if line.startswith(">"):
                    seq_id = line[1:].strip().split()[0]
                    id_to_family[seq_id] = fam_idx

    labels = np.full(len(dataset.records), -1, dtype=np.int32)
    for i, rec in enumerate(dataset.records):
        if rec.id in id_to_family:
            labels[i] = id_to_family[rec.id]

    if np.any(labels == -1):
        n_missing = int(np.sum(labels == -1))
        print(f"  Warning: {n_missing} sequences not found in family files")
        return None

    return labels


def compute_ari(labels_true, labels_pred):
    """Compute Adjusted Rand Index."""
    from collections import Counter

    n = len(labels_true)
    if n < 2:
        return 0.0

    # Contingency table via pair counting
    pair_counts = Counter()
    for i in range(n):
        pair_counts[(labels_true[i], labels_pred[i])] += 1

    sum_comb_nij = sum(v * (v - 1) // 2 for v in pair_counts.values())

    # Row sums (true clusters)
    row_sums = Counter()
    for (t, _), v in pair_counts.items():
        row_sums[t] += v
    sum_comb_a = sum(v * (v - 1) // 2 for v in row_sums.values())

    # Column sums (predicted clusters)
    col_sums = Counter()
    for (_, p), v in pair_counts.items():
        col_sums[p] += v
    sum_comb_b = sum(v * (v - 1) // 2 for v in col_sums.values())

    total_comb = n * (n - 1) // 2
    expected = sum_comb_a * sum_comb_b / total_comb if total_comb > 0 else 0
    max_index = (sum_comb_a + sum_comb_b) / 2
    denom = max_index - expected

    if denom == 0:
        return 1.0 if sum_comb_nij == expected else 0.0

    return (sum_comb_nij - expected) / denom


def compute_pairwise_f1(labels_true, labels_pred):
    """Compute pairwise precision, recall, F1."""
    n = len(labels_true)
    tp = fp = fn = 0

    # Use label grouping for efficiency
    from collections import defaultdict
    true_groups = defaultdict(set)
    pred_groups = defaultdict(set)
    for i in range(n):
        true_groups[labels_true[i]].add(i)
        pred_groups[labels_pred[i]].add(i)

    # TP: pairs in same true cluster AND same predicted cluster
    for pg in pred_groups.values():
        for tg in true_groups.values():
            overlap = len(pg & tg)
            tp += overlap * (overlap - 1) // 2

    # Total same-true pairs
    total_true = sum(len(g) * (len(g) - 1) // 2 for g in true_groups.values())
    # Total same-pred pairs
    total_pred = sum(len(g) * (len(g) - 1) // 2 for g in pred_groups.values())

    fn = total_true - tp
    fp = total_pred - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def run_clustering(dataset, threshold, threads=192):
    """Run full ClustKIT pipeline and return labels + stats."""
    import numba
    numba.set_num_threads(threads)

    mode = "protein"
    k_lsh = auto_kmer_for_lsh(threshold, mode, 5)
    sketches = compute_sketches(
        dataset.encoded_sequences, dataset.lengths,
        k_lsh, 128, mode,
        flat_sequences=dataset.flat_sequences,
        offsets=dataset.offsets,
    )

    lsh_params = auto_lsh_params(threshold, "high", k=k_lsh)
    candidate_pairs = lsh_candidates(
        sketches,
        num_tables=lsh_params["num_tables"],
        num_bands=lsh_params["num_bands"],
    )

    p95_len = int(np.percentile(dataset.lengths, 95))
    band_width = max(20, int(p95_len * 0.3))

    t0 = time.perf_counter()
    filtered_pairs, similarities = compute_pairwise_alignment(
        candidate_pairs, dataset.encoded_sequences, dataset.lengths,
        threshold, band_width=band_width, mode=mode,
        sketches=sketches,
        flat_sequences=dataset.flat_sequences,
        offsets=dataset.offsets,
    )
    align_time = time.perf_counter() - t0

    n = dataset.num_sequences
    graph = build_similarity_graph(n, filtered_pairs, similarities)
    labels = cluster_sequences(graph, method="connected", lengths=dataset.lengths)

    return labels, {
        "candidates": len(candidate_pairs),
        "filtered": len(filtered_pairs),
        "clusters": len(np.unique(labels)),
        "align_time": align_time,
        "lsh_params": lsh_params,
        "band_width": band_width,
    }


def main():
    print("=" * 80)
    print("  ClustKIT Accuracy Validation")
    print("=" * 80)

    dataset = read_sequences(INPUT, "protein")
    print(f"  Dataset: {dataset.num_sequences:,} sequences")

    ground_truth = build_ground_truth_from_files(dataset)
    if ground_truth is None:
        print("  Could not build ground truth from family files, using ID-based")
        ground_truth = build_ground_truth(dataset)

    n_families = len(np.unique(ground_truth))
    print(f"  Ground truth: {n_families} families")
    print()

    thresholds = [0.4, 0.5, 0.7]
    print(f"  {'Thresh':>6} {'Clusters':>8} {'ARI':>8} {'P(pw)':>8} {'R(pw)':>8} {'F1(pw)':>8} "
          f"{'Cands':>12} {'Filtered':>10} {'AlignTime':>10} {'BW':>5}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*12} {'-'*10} {'-'*10} {'-'*5}")

    for t in thresholds:
        labels, stats = run_clustering(dataset, t)
        ari = compute_ari(ground_truth, labels)
        p, r, f1 = compute_pairwise_f1(ground_truth, labels)

        print(f"  {t:>6.1f} {stats['clusters']:>8} {ari:>8.4f} {p:>8.4f} {r:>8.4f} {f1:>8.4f} "
              f"{stats['candidates']:>12,} {stats['filtered']:>10,} "
              f"{stats['align_time']:>9.1f}s {stats['band_width']:>5}")

    print()
    print("=" * 80)
    print("  Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
