"""CD-HIT vs MMseqs2 vs ClustKIT Head-to-Head Comparison

Runs all three tools on the same simulated datasets and compares accuracy metrics.
"""

import json
import os
import subprocess
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline

CDHIT_SIF = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/cd-hit.sif"
MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "cdhit_comparison_results"


def generate_families(num_families, members_per_family, seq_length, target_identity, seed=42):
    """Generate simulated protein families via pyvolve."""
    import pyvolve

    branch_len = -np.log(target_identity) / 2.2

    all_sequences = []
    ground_truth = {}

    for fam_idx in range(num_families):
        np.random.seed(seed + fam_idx)
        family_id = f"fam{fam_idx:03d}"

        leaf_names = [f"{family_id}_s{i}" for i in range(members_per_family)]
        tree_str = "(" + ", ".join(f"{name}:{branch_len}" for name in leaf_names) + ");"

        tree = pyvolve.read_tree(tree=tree_str)
        model = pyvolve.Model("WAG")
        partition = pyvolve.Partition(models=model, size=seq_length)
        evolver = pyvolve.Evolver(tree=tree, partitions=partition)

        tmp_file = f"/tmp/simfam_{family_id}_{seed}.fasta"
        evolver(seqfile=tmp_file, ratefile=None, infofile=None)

        with open(tmp_file) as f:
            current_id = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id is not None:
                        all_sequences.append((current_id, "".join(current_seq), family_id))
                        ground_truth[current_id] = family_id
                    current_id = line[1:].strip()
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id is not None:
                all_sequences.append((current_id, "".join(current_seq), family_id))
                ground_truth[current_id] = family_id

        os.remove(tmp_file)

    # Shuffle
    rng = np.random.RandomState(seed)
    rng.shuffle(all_sequences)

    return all_sequences, ground_truth


def write_fasta(sequences, path):
    """Write sequences to FASTA file."""
    with open(path, "w") as f:
        for seq_id, seq, _ in sequences:
            f.write(f">{seq_id}\n{seq}\n")


def parse_cdhit_clusters(clstr_path):
    """Parse CD-HIT .clstr file into a dict of seq_id -> cluster_id."""
    clusters = {}
    current_cluster = -1
    with open(clstr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
            elif line:
                # Format: 0	200aa, >fam000_s1... *
                # or: 0	200aa, >fam000_s1... at 95.00%
                seq_id = line.split(">")[1].split("...")[0]
                clusters[seq_id] = current_cluster
    return clusters


def run_cdhit(fasta_path, output_prefix, threshold, threads=1):
    """Run CD-HIT via Singularity and return cluster assignments."""
    cmd = [
        "singularity", "exec", CDHIT_SIF,
        "cd-hit",
        "-i", str(fasta_path),
        "-o", str(output_prefix),
        "-c", str(threshold),
        "-T", str(threads),
        "-M", "0",
        "-d", "0",  # full sequence name in output
    ]

    # CD-HIT requires -n (word size) to match threshold:
    #   0.7+ → -n 5, 0.6+ → -n 4, 0.5+ → -n 3, 0.4+ → -n 2
    if threshold >= 0.7:
        cmd.extend(["-n", "5"])
    elif threshold >= 0.6:
        cmd.extend(["-n", "4"])
    elif threshold >= 0.5:
        cmd.extend(["-n", "3"])
    else:
        cmd.extend(["-n", "2"])

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"CD-HIT failed: {result.stderr}")
        return None, elapsed

    clstr_path = str(output_prefix) + ".clstr"
    clusters = parse_cdhit_clusters(clstr_path)
    return clusters, elapsed


def parse_mmseqs_clusters(tsv_path):
    """Parse MMseqs2 cluster TSV into a dict of seq_id -> cluster_id.

    MMseqs2 easy-cluster outputs a TSV with two columns:
      representative_id\tmember_id
    We assign each unique representative a numeric cluster ID.
    """
    clusters = {}
    rep_to_id = {}
    next_id = 0
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            if rep not in rep_to_id:
                rep_to_id[rep] = next_id
                next_id += 1
            clusters[member] = rep_to_id[rep]
    return clusters


def run_mmseqs(fasta_path, output_prefix, threshold, threads=1):
    """Run MMseqs2 easy-cluster and return cluster assignments."""
    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_tmp_")

    cmd = [
        MMSEQS_BIN, "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        tmp_dir,
        "--min-seq-id", str(threshold),
        "--threads", str(threads),
        "-c", "0.8",          # coverage threshold
        "--cov-mode", "0",    # bidirectional coverage
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - start

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        print(f"MMseqs2 failed: {result.stderr}")
        return None, elapsed

    tsv_path = str(output_prefix) + "_cluster.tsv"
    clusters = parse_mmseqs_clusters(tsv_path)
    return clusters, elapsed


def run_clustkit(fasta_path, output_dir, threshold):
    """Run ClustKIT and return cluster assignments."""
    config = {
        "input": fasta_path,
        "output": output_dir,
        "threshold": threshold,
        "mode": "protein",
        "alignment": "align",
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

    clusters = {}
    with open(output_dir / "clusters.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            clusters[parts[0]] = int(parts[1])

    return clusters, elapsed


def pairwise_metrics(true_labels, pred_labels):
    """Compute pairwise precision, recall, F1, accuracy."""
    n = len(true_labels)
    if n > 5000:
        rng = np.random.RandomState(42)
        idx_a = rng.randint(0, n, size=2_000_000)
        idx_b = rng.randint(0, n, size=2_000_000)
        valid = idx_a != idx_b
        idx_a, idx_b = idx_a[valid], idx_b[valid]
    else:
        idx_a, idx_b = [], []
        for i in range(n):
            for j in range(i + 1, n):
                idx_a.append(i)
                idx_b.append(j)
        idx_a, idx_b = np.array(idx_a), np.array(idx_b)

    same_pred = pred_labels[idx_a] == pred_labels[idx_b]
    same_true = true_labels[idx_a] == true_labels[idx_b]

    tp = int(np.sum(same_pred & same_true))
    fp = int(np.sum(same_pred & ~same_true))
    fn = int(np.sum(~same_pred & same_true))
    tn = int(np.sum(~same_pred & ~same_true))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "F1": round(f1, 4),
        "accuracy": round(acc, 4),
    }


def evaluate(ground_truth, pred_clusters):
    """Evaluate predicted clusters against ground truth."""
    common = sorted(set(ground_truth.keys()) & set(pred_clusters.keys()))
    if not common:
        return None

    true_list = [ground_truth[s] for s in common]
    pred_list = [pred_clusters[s] for s in common]

    fam_to_int = {f: i for i, f in enumerate(sorted(set(true_list)))}
    true_arr = np.array([fam_to_int[f] for f in true_list], dtype=np.int32)
    pred_arr = np.array(pred_list, dtype=np.int32)

    ari = adjusted_rand_score(true_arr, pred_arr)
    nmi = normalized_mutual_info_score(true_arr, pred_arr)
    hom, comp, vm = homogeneity_completeness_v_measure(true_arr, pred_arr)
    pw = pairwise_metrics(true_arr, pred_arr)

    return {
        "n_sequences": len(common),
        "n_true_families": len(set(true_list)),
        "n_clusters": len(set(pred_list)),
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "homogeneity": round(hom, 4),
        "completeness": round(comp, 4),
        "V_measure": round(vm, 4),
        **{f"pw_{k}": v for k, v in pw.items()},
    }


def run_comparison():
    """Run head-to-head comparison."""
    scenarios = [
        {"target_identity": 0.95, "threshold": 0.9, "label": "95% id, t=0.9"},
        {"target_identity": 0.85, "threshold": 0.7, "label": "85% id, t=0.7"},
        {"target_identity": 0.70, "threshold": 0.5, "label": "70% id, t=0.5"},
        {"target_identity": 0.50, "threshold": 0.4, "label": "50% id, t=0.4"},
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print("=" * 90)
    print("CD-HIT vs MMseqs2 vs ClustKIT HEAD-TO-HEAD COMPARISON")
    print("=" * 90)

    for scenario in scenarios:
        target_id = scenario["target_identity"]
        threshold = scenario["threshold"]
        label = scenario["label"]

        print()
        print("-" * 90)
        print(f"Scenario: {label}")
        print("-" * 90)

        # Generate dataset
        print(f"Generating: 20 families x 50 members, {target_id:.0%} within-family identity")
        sequences, ground_truth = generate_families(
            num_families=20,
            members_per_family=50,
            seq_length=200,
            target_identity=target_id,
            seed=42,
        )

        fasta_path = OUTPUT_DIR / f"sim_{label.replace(' ', '_').replace(',', '')}.fasta"
        write_fasta(sequences, fasta_path)
        print(f"  {len(sequences)} sequences written")

        # --- CD-HIT ---
        cdhit_prefix = OUTPUT_DIR / f"cdhit_{label.replace(' ', '_').replace(',', '')}"
        print(f"\nRunning CD-HIT (t={threshold})...")
        cdhit_clusters, cdhit_time = run_cdhit(fasta_path, cdhit_prefix, threshold)

        if cdhit_clusters:
            cdhit_results = evaluate(ground_truth, cdhit_clusters)
            cdhit_results["runtime"] = round(cdhit_time, 2)
            print(f"  Clusters: {cdhit_results['n_clusters']}, "
                  f"ARI: {cdhit_results['ARI']}, "
                  f"F1(pw): {cdhit_results['pw_F1']}, "
                  f"Time: {cdhit_time:.2f}s")
        else:
            cdhit_results = {"error": "CD-HIT failed"}
            print("  CD-HIT failed!")

        # --- MMseqs2 ---
        mmseqs_prefix = OUTPUT_DIR / f"mmseqs_{label.replace(' ', '_').replace(',', '')}"
        print(f"\nRunning MMseqs2 (min-seq-id={threshold})...")
        mmseqs_clusters, mmseqs_time = run_mmseqs(fasta_path, mmseqs_prefix, threshold)

        if mmseqs_clusters:
            mmseqs_results = evaluate(ground_truth, mmseqs_clusters)
            mmseqs_results["runtime"] = round(mmseqs_time, 2)
            print(f"  Clusters: {mmseqs_results['n_clusters']}, "
                  f"ARI: {mmseqs_results['ARI']}, "
                  f"F1(pw): {mmseqs_results['pw_F1']}, "
                  f"Time: {mmseqs_time:.2f}s")
        else:
            mmseqs_results = {"error": "MMseqs2 failed"}
            print("  MMseqs2 failed!")

        # --- ClustKIT ---
        clustkit_dir = OUTPUT_DIR / f"clustkit_{label.replace(' ', '_').replace(',', '')}"
        clustkit_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nRunning ClustKIT (t={threshold})...")
        clustkit_clusters, clustkit_time = run_clustkit(fasta_path, clustkit_dir, threshold)
        clustkit_results = evaluate(ground_truth, clustkit_clusters)
        clustkit_results["runtime"] = round(clustkit_time, 2)
        print(f"  Clusters: {clustkit_results['n_clusters']}, "
              f"ARI: {clustkit_results['ARI']}, "
              f"F1(pw): {clustkit_results['pw_F1']}, "
              f"Time: {clustkit_time:.2f}s")

        all_results[label] = {
            "cdhit": cdhit_results,
            "mmseqs2": mmseqs_results,
            "clustkit": clustkit_results,
        }

    # Summary table
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print(f"{'Scenario':<20} {'Tool':<10} {'Clust':>6} {'ARI':>7} {'NMI':>7} "
          f"{'Homog':>7} {'Compl':>7} {'P(pw)':>7} {'R(pw)':>7} {'F1(pw)':>7} {'Time':>6}")
    print("-" * 104)

    for scenario in scenarios:
        label = scenario["label"]
        r = all_results[label]
        for tool, key in [("CD-HIT", "cdhit"), ("MMseqs2", "mmseqs2"), ("ClustKIT", "clustkit")]:
            res = r[key]
            if "error" in res:
                print(f"{label:<20} {tool:<10} {'FAILED':>6}")
            else:
                print(
                    f"{label:<20} {tool:<10} {res['n_clusters']:>6} "
                    f"{res['ARI']:>7.4f} {res['NMI']:>7.4f} "
                    f"{res['homogeneity']:>7.4f} {res['completeness']:>7.4f} "
                    f"{res['pw_precision']:>7.4f} {res['pw_recall']:>7.4f} "
                    f"{res['pw_F1']:>7.4f} {res['runtime']:>5.2f}s"
                )
        print()

    # Save
    results_file = OUTPUT_DIR / "comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    run_comparison()
