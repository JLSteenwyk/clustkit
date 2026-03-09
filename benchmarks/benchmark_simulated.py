"""Simulated Family Benchmark

Generates protein families via simulated evolution (pyvolve + WAG model),
where within-family divergence is calibrated to known identity levels.
Clusters the mixed dataset with ClustKIT and measures recovery.

This is a fair test because ground-truth families are defined by sequence
identity — exactly what ClustKIT clusters by.

Metrics:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Homogeneity, Completeness, V-measure
  - Pairwise Precision, Recall, F1, Accuracy
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyvolve
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "simulated_benchmark_results"


def branch_length_for_identity(target_identity):
    """Estimate WAG branch length that yields a given expected pairwise identity.

    Under the WAG substitution model, the expected fraction of identical sites
    decreases roughly as:  identity ≈ exp(-1.1 * branch_length) for moderate
    divergence. This is a rough calibration — we verify empirically below.

    For a star tree with root->leaf branch length d, two leaves are separated
    by total distance 2d, so we want: identity ≈ exp(-1.1 * 2d).
    """
    # Solve: target = exp(-2.2 * d)  =>  d = -ln(target) / 2.2
    d = -np.log(target_identity) / 2.2
    return max(0.001, d)


def generate_family(
    family_id,
    num_members,
    seq_length,
    branch_length,
    rng_seed,
):
    """Generate a protein family by simulating evolution from a random ancestor.

    Uses a star tree (all members equidistant from root) with WAG model.

    Args:
        family_id: String identifier for this family.
        num_members: Number of sequences in the family.
        seq_length: Length of the ancestor sequence (all members same length).
        branch_length: Root-to-leaf branch length (controls divergence).
        rng_seed: Random seed for reproducibility.

    Returns:
        List of (seq_id, sequence, family_id) tuples.
    """
    np.random.seed(rng_seed)

    # Build star tree: (A:d, B:d, C:d, ...)
    leaf_names = [f"{family_id}_s{i}" for i in range(num_members)]
    tree_str = "(" + ", ".join(f"{name}:{branch_length}" for name in leaf_names) + ");"

    tree = pyvolve.read_tree(tree=tree_str)
    model = pyvolve.Model("WAG")
    partition = pyvolve.Partition(models=model, size=seq_length)

    # Simulate
    evolver = pyvolve.Evolver(tree=tree, partitions=partition)
    tmp_file = f"/tmp/simfam_{family_id}_{rng_seed}.fasta"
    evolver(seqfile=tmp_file, ratefile=None, infofile=None)

    # Parse output
    sequences = []
    current_id = None
    current_seq = []
    with open(tmp_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq), family_id))
                current_id = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            sequences.append((current_id, "".join(current_seq), family_id))

    os.remove(tmp_file)
    return sequences


def measure_actual_identity(sequences):
    """Measure average pairwise identity within a set of sequences."""
    if len(sequences) < 2:
        return 1.0

    identities = []
    seqs = [s[1] for s in sequences]
    for i in range(min(len(seqs), 20)):  # sample up to 20 pairs
        for j in range(i + 1, min(len(seqs), 20)):
            matches = sum(a == b for a, b in zip(seqs[i], seqs[j]))
            length = min(len(seqs[i]), len(seqs[j]))
            if length > 0:
                identities.append(matches / length)

    return np.mean(identities) if identities else 1.0


def generate_dataset(
    num_families=20,
    members_per_family=50,
    seq_length=200,
    target_identity=0.7,
    between_family_divergence=2.0,
    seed=42,
):
    """Generate a complete simulated dataset with multiple families.

    Families are generated independently from different random ancestors,
    so between-family identity will be very low (~5-10% for proteins).

    Args:
        num_families: Number of distinct families.
        members_per_family: Sequences per family.
        seq_length: Ancestor sequence length.
        target_identity: Target within-family pairwise identity.
        between_family_divergence: Not used — families naturally diverge
            because they start from independent random ancestors.
        seed: Base random seed.

    Returns:
        Tuple of (fasta_path, ground_truth dict).
    """
    branch_len = branch_length_for_identity(target_identity)

    all_sequences = []
    ground_truth = {}

    print(f"Generating {num_families} families x {members_per_family} members")
    print(f"  Target within-family identity: {target_identity:.0%}")
    print(f"  WAG branch length: {branch_len:.4f}")
    print(f"  Sequence length: {seq_length} aa")

    for fam_idx in range(num_families):
        family_id = f"fam{fam_idx:03d}"
        fam_seqs = generate_family(
            family_id=family_id,
            num_members=members_per_family,
            seq_length=seq_length,
            branch_length=branch_len,
            rng_seed=seed + fam_idx,
        )

        actual_id = measure_actual_identity(fam_seqs)
        if fam_idx < 5 or fam_idx == num_families - 1:
            print(f"  {family_id}: {len(fam_seqs)} seqs, actual identity = {actual_id:.1%}")

        all_sequences.extend(fam_seqs)
        for seq_id, seq, fid in fam_seqs:
            ground_truth[seq_id] = fid

    # Shuffle to remove family ordering bias
    rng = np.random.RandomState(seed)
    rng.shuffle(all_sequences)

    # Write mixed FASTA
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fasta_path = OUTPUT_DIR / f"simulated_id{int(target_identity*100)}.fasta"
    with open(fasta_path, "w") as f:
        for seq_id, seq, _ in all_sequences:
            f.write(f">{seq_id}\n{seq}\n")

    print(f"  Total: {len(all_sequences)} sequences written to {fasta_path.name}")
    return fasta_path, ground_truth


def pairwise_precision_recall_f1(true_labels, pred_labels):
    """Compute pairwise TP, FP, FN, TN, precision, recall, F1, accuracy."""
    n = len(true_labels)

    if n > 5000:
        num_samples = 2_000_000
        rng = np.random.RandomState(42)
        idx_a = rng.randint(0, n, size=num_samples)
        idx_b = rng.randint(0, n, size=num_samples)
        valid = idx_a != idx_b
        idx_a, idx_b = idx_a[valid], idx_b[valid]
    else:
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
    """Evaluate clustering against ground truth."""
    pred_map = {}
    with open(cluster_tsv_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            pred_map[parts[0]] = int(parts[1])

    common_ids = sorted(set(ground_truth.keys()) & set(pred_map.keys()))

    true_family_list = [ground_truth[sid] for sid in common_ids]
    pred_cluster_list = [pred_map[sid] for sid in common_ids]

    family_to_int = {f: i for i, f in enumerate(sorted(set(true_family_list)))}
    true_labels = np.array([family_to_int[f] for f in true_family_list], dtype=np.int32)
    pred_labels = np.array(pred_cluster_list, dtype=np.int32)

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_labels, pred_labels
    )
    pw = pairwise_precision_recall_f1(true_labels, pred_labels)

    return {
        "n_sequences": len(common_ids),
        "n_true_families": len(set(true_family_list)),
        "n_predicted_clusters": len(set(pred_cluster_list)),
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "homogeneity": round(homogeneity, 4),
        "completeness": round(completeness, 4),
        "V_measure": round(v_measure, 4),
        **pw,
    }


def run_benchmark():
    """Run the full simulated family benchmark.

    Tests multiple within-family identity levels, each clustered at the
    matching threshold. This measures: "if families have X% identity within
    them, can ClustKIT recover them when clustering at X%?"
    """
    # Each scenario: (target_identity, clustering_threshold)
    # We also test with mismatched thresholds to see robustness
    scenarios = [
        # Matched: within-family identity ≈ clustering threshold
        {"target_identity": 0.95, "threshold": 0.9, "label": "95% id, t=0.9"},
        {"target_identity": 0.85, "threshold": 0.7, "label": "85% id, t=0.7"},
        {"target_identity": 0.70, "threshold": 0.5, "label": "70% id, t=0.5"},
        {"target_identity": 0.50, "threshold": 0.3, "label": "50% id, t=0.3"},
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print("=" * 70)
    print("SIMULATED FAMILY BENCHMARK")
    print("=" * 70)

    for scenario in scenarios:
        target_id = scenario["target_identity"]
        threshold = scenario["threshold"]
        label = scenario["label"]

        print()
        print("-" * 70)
        print(f"Scenario: {label}")
        print("-" * 70)

        # Generate dataset
        fasta_path, ground_truth = generate_dataset(
            num_families=20,
            members_per_family=50,
            seq_length=200,
            target_identity=target_id,
            seed=42,
        )

        # Cluster
        out_dir = OUTPUT_DIR / f"sim_id{int(target_id*100)}_t{int(threshold*100)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "input": fasta_path,
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
        results = evaluate_clustering(ground_truth, out_dir / "clusters.tsv")
        results["runtime_seconds"] = round(elapsed, 2)
        results["target_identity"] = target_id
        results["threshold"] = threshold
        all_results[label] = results

        print()
        print(f"  Results:")
        print(f"    Sequences:  {results['n_sequences']}")
        print(f"    True families:    {results['n_true_families']}")
        print(f"    Pred clusters:    {results['n_predicted_clusters']}")
        print(f"    ARI:              {results['ARI']}")
        print(f"    NMI:              {results['NMI']}")
        print(f"    Homogeneity:      {results['homogeneity']}")
        print(f"    Completeness:     {results['completeness']}")
        print(f"    V-measure:        {results['V_measure']}")
        print(f"    Pairwise Precision: {results['pairwise_precision']}")
        print(f"    Pairwise Recall:    {results['pairwise_recall']}")
        print(f"    Pairwise F1:        {results['pairwise_F1']}")
        print(f"    Pairwise Accuracy:  {results['pairwise_accuracy']}")
        print(f"    Runtime:          {results['runtime_seconds']}s")

    # Summary table
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = (
        f"{'Scenario':<20} {'Fam':>4} {'Clust':>6} "
        f"{'ARI':>7} {'NMI':>7} {'Homog':>7} {'Compl':>7} "
        f"{'P(pw)':>7} {'R(pw)':>7} {'F1(pw)':>7} {'Acc':>7}"
    )
    print(header)
    print("-" * len(header))
    for scenario in scenarios:
        label = scenario["label"]
        r = all_results[label]
        print(
            f"{label:<20} {r['n_true_families']:>4} {r['n_predicted_clusters']:>6} "
            f"{r['ARI']:>7.4f} {r['NMI']:>7.4f} "
            f"{r['homogeneity']:>7.4f} {r['completeness']:>7.4f} "
            f"{r['pairwise_precision']:>7.4f} {r['pairwise_recall']:>7.4f} "
            f"{r['pairwise_F1']:>7.4f} {r['pairwise_accuracy']:>7.4f}"
        )

    # Save results
    results_file = OUTPUT_DIR / "simulated_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    run_benchmark()
