"""Large-Scale CD-HIT vs MMseqs2 vs ClustKIT Comparison

Runs all three tools at 10K, 50K, and 100K sequences to compare
accuracy and runtime at scale.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
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
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "scaling_comparison_results"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def generate_synthetic_fasta(path, num_sequences, num_families=100,
                             seq_length=200, target_identity=0.85, seed=42):
    """Generate synthetic protein FASTA by mutating family ancestors."""
    rng = np.random.RandomState(seed)
    members_per_family = num_sequences // num_families
    mutation_rate = 1.0 - target_identity

    ground_truth = {}

    with open(path, "w") as f:
        for fam in range(num_families):
            ancestor = rng.choice(len(AMINO_ACIDS), size=seq_length)
            for mem in range(members_per_family):
                seq = ancestor.copy()
                mask = rng.random(seq_length) < mutation_rate
                seq[mask] = rng.choice(len(AMINO_ACIDS), size=mask.sum())
                seq_str = "".join(AMINO_ACIDS[aa] for aa in seq)
                seq_id = f"fam{fam:04d}_s{mem}"
                f.write(f">{seq_id}\n{seq_str}\n")
                ground_truth[seq_id] = f"fam{fam:04d}"

        remainder = num_sequences - (num_families * members_per_family)
        for i in range(remainder):
            fam = rng.randint(0, num_families)
            ancestor = rng.choice(len(AMINO_ACIDS), size=seq_length)
            seq_str = "".join(AMINO_ACIDS[aa] for aa in ancestor)
            seq_id = f"fam{fam:04d}_extra{i}"
            f.write(f">{seq_id}\n{seq_str}\n")
            ground_truth[seq_id] = f"fam{fam:04d}"

    return ground_truth


def parse_cdhit_clusters(clstr_path):
    """Parse CD-HIT .clstr file."""
    clusters = {}
    current_cluster = -1
    with open(clstr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
            elif line:
                seq_id = line.split(">")[1].split("...")[0]
                clusters[seq_id] = current_cluster
    return clusters


def parse_mmseqs_clusters(tsv_path):
    """Parse MMseqs2 cluster TSV."""
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


def run_cdhit(fasta_path, output_prefix, threshold, threads=4):
    """Run CD-HIT."""
    cmd = [
        "singularity", "exec", CDHIT_SIF,
        "cd-hit",
        "-i", str(fasta_path),
        "-o", str(output_prefix),
        "-c", str(threshold),
        "-T", str(threads),
        "-M", "0",
        "-d", "0",
    ]
    if threshold >= 0.7:
        cmd.extend(["-n", "5"])
    elif threshold >= 0.6:
        cmd.extend(["-n", "4"])
    elif threshold >= 0.5:
        cmd.extend(["-n", "3"])
    else:
        cmd.extend(["-n", "2"])

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"  CD-HIT failed: {result.stderr[:200]}")
        return None, elapsed

    clusters = parse_cdhit_clusters(str(output_prefix) + ".clstr")
    return clusters, elapsed


def run_mmseqs(fasta_path, output_prefix, threshold, threads=4):
    """Run MMseqs2."""
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_tmp_")
    cmd = [
        MMSEQS_BIN, "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        tmp_dir,
        "--min-seq-id", str(threshold),
        "--threads", str(threads),
        "-c", "0.8",
        "--cov-mode", "0",
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - start

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        print(f"  MMseqs2 failed: {result.stderr[:200]}")
        return None, elapsed

    clusters = parse_mmseqs_clusters(str(output_prefix) + "_cluster.tsv")
    return clusters, elapsed


def run_clustkit(fasta_path, output_dir, threshold):
    """Run ClustKIT."""
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
    """Compute pairwise precision, recall, F1."""
    n = len(true_labels)
    rng = np.random.RandomState(42)
    sample_size = min(2_000_000, n * (n - 1) // 2)
    idx_a = rng.randint(0, n, size=sample_size)
    idx_b = rng.randint(0, n, size=sample_size)
    valid = idx_a != idx_b
    idx_a, idx_b = idx_a[valid], idx_b[valid]

    same_pred = pred_labels[idx_a] == pred_labels[idx_b]
    same_true = true_labels[idx_a] == true_labels[idx_b]

    tp = int(np.sum(same_pred & same_true))
    fp = int(np.sum(same_pred & ~same_true))
    fn = int(np.sum(~same_pred & same_true))
    tn = int(np.sum(~same_pred & ~same_true))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": round(prec, 4), "recall": round(rec, 4), "F1": round(f1, 4)}


def evaluate(ground_truth, pred_clusters):
    """Evaluate predicted clusters against ground truth."""
    common = sorted(set(ground_truth.keys()) & set(pred_clusters.keys()))
    if not common:
        return {"error": "no common sequences"}

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
        "n_clusters": len(set(pred_list)),
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "homogeneity": round(hom, 4),
        "completeness": round(comp, 4),
        "V_measure": round(vm, 4),
        "pw_F1": pw["F1"],
        "pw_precision": pw["precision"],
        "pw_recall": pw["recall"],
    }


def main():
    sizes = [10_000, 50_000, 100_000]
    threshold = 0.7
    num_families = 100

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print("=" * 110)
    print("LARGE-SCALE CD-HIT vs MMseqs2 vs ClustKIT COMPARISON")
    print("=" * 110)
    print(f"Config: threshold={threshold}, {num_families} families, 200aa, ~85% identity, 4 threads (CD-HIT/MMseqs2)")
    print()

    for n in sizes:
        label = f"{n // 1000}K"
        fasta_path = OUTPUT_DIR / f"synthetic_{label}.fasta"

        print("-" * 110)
        print(f"Dataset: {label} sequences ({num_families} families)")
        print("-" * 110)

        # Generate
        print(f"  Generating...", end=" ", flush=True)
        t0 = time.perf_counter()
        ground_truth = generate_synthetic_fasta(fasta_path, n, num_families=num_families)
        print(f"done ({time.perf_counter() - t0:.1f}s)")

        scenario_results = {}

        # --- CD-HIT ---
        print(f"  CD-HIT...", end=" ", flush=True)
        cdhit_prefix = OUTPUT_DIR / f"cdhit_{label}"
        cdhit_clusters, cdhit_time = run_cdhit(fasta_path, cdhit_prefix, threshold)
        if cdhit_clusters:
            cdhit_eval = evaluate(ground_truth, cdhit_clusters)
            cdhit_eval["runtime"] = round(cdhit_time, 2)
            scenario_results["cdhit"] = cdhit_eval
            print(f"{cdhit_eval['n_clusters']} clusters, ARI={cdhit_eval['ARI']}, "
                  f"F1={cdhit_eval['pw_F1']}, {cdhit_time:.2f}s")
        else:
            scenario_results["cdhit"] = {"error": "failed", "runtime": round(cdhit_time, 2)}
            print(f"FAILED ({cdhit_time:.2f}s)")

        # --- MMseqs2 ---
        print(f"  MMseqs2...", end=" ", flush=True)
        mmseqs_prefix = OUTPUT_DIR / f"mmseqs_{label}"
        mmseqs_clusters, mmseqs_time = run_mmseqs(fasta_path, mmseqs_prefix, threshold)
        if mmseqs_clusters:
            mmseqs_eval = evaluate(ground_truth, mmseqs_clusters)
            mmseqs_eval["runtime"] = round(mmseqs_time, 2)
            scenario_results["mmseqs2"] = mmseqs_eval
            print(f"{mmseqs_eval['n_clusters']} clusters, ARI={mmseqs_eval['ARI']}, "
                  f"F1={mmseqs_eval['pw_F1']}, {mmseqs_time:.2f}s")
        else:
            scenario_results["mmseqs2"] = {"error": "failed", "runtime": round(mmseqs_time, 2)}
            print(f"FAILED ({mmseqs_time:.2f}s)")

        # --- ClustKIT ---
        print(f"  ClustKIT...", end=" ", flush=True)
        clustkit_dir = OUTPUT_DIR / f"clustkit_{label}"
        clustkit_dir.mkdir(parents=True, exist_ok=True)
        clustkit_clusters, clustkit_time = run_clustkit(fasta_path, clustkit_dir, threshold)
        clustkit_eval = evaluate(ground_truth, clustkit_clusters)
        clustkit_eval["runtime"] = round(clustkit_time, 2)
        scenario_results["clustkit"] = clustkit_eval
        print(f"{clustkit_eval['n_clusters']} clusters, ARI={clustkit_eval['ARI']}, "
              f"F1={clustkit_eval['pw_F1']}, {clustkit_time:.2f}s")

        all_results[label] = scenario_results
        print()

    # Summary table
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print()
    print(f"{'Size':<8} {'Tool':<10} {'Clust':>6} {'ARI':>8} {'NMI':>8} "
          f"{'Homog':>8} {'Compl':>8} {'P(pw)':>8} {'R(pw)':>8} {'F1(pw)':>8} {'Time':>10}")
    print("-" * 110)

    for n in sizes:
        label = f"{n // 1000}K"
        r = all_results[label]
        for tool, key in [("CD-HIT", "cdhit"), ("MMseqs2", "mmseqs2"), ("ClustKIT", "clustkit")]:
            res = r[key]
            if "error" in res:
                rt = res.get("runtime", "?")
                print(f"{label:<8} {tool:<10} {'FAIL':>6} {'':>8} {'':>8} "
                      f"{'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {rt:>9}s")
            else:
                print(
                    f"{label:<8} {tool:<10} {res['n_clusters']:>6} "
                    f"{res['ARI']:>8.4f} {res['NMI']:>8.4f} "
                    f"{res['homogeneity']:>8.4f} {res['completeness']:>8.4f} "
                    f"{res['pw_precision']:>8.4f} {res['pw_recall']:>8.4f} "
                    f"{res['pw_F1']:>8.4f} {res['runtime']:>9.2f}s"
                )
        print()

    # Save
    results_file = OUTPUT_DIR / "scaling_comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
