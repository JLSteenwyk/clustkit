"""Pfam Concordance Benchmark

Loads sequences from multiple Pfam families, mixes them into a single
dataset, clusters with ClustKIT / CD-HIT / MMseqs2 at various thresholds,
and measures how well the output clusters recover the known Pfam family labels.

Metrics:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Homogeneity, Completeness, V-measure
  - Pairwise Precision, Recall, F1
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
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

CDHIT_SIF = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/cd-hit.sif"
MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
VSEARCH_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/VSEARCH/vsearch-2.30.5-linux-x86_64/bin/vsearch"


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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        return None, elapsed

    clusters = parse_mmseqs_clusters(str(output_prefix) + "_cluster.tsv")
    return clusters, elapsed


def run_mmseqs_linclust(fasta_path, output_prefix, threshold, threads=4):
    """Run MMseqs2 linclust (linear-time clustering)."""
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_linclust_tmp_")
    cmd = [
        MMSEQS_BIN, "easy-linclust",
        str(fasta_path),
        str(output_prefix),
        tmp_dir,
        "--min-seq-id", str(threshold),
        "--threads", str(threads),
        "-c", "0.8",
        "--cov-mode", "0",
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        return None, elapsed

    clusters = parse_mmseqs_clusters(str(output_prefix) + "_cluster.tsv")
    return clusters, elapsed


def parse_vsearch_clusters(uc_path):
    """Parse VSEARCH .uc file."""
    clusters = {}
    with open(uc_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            rec_type = parts[0]
            cluster_id = int(parts[1])
            seq_id = parts[8]
            if rec_type in ("S", "H"):
                clusters[seq_id] = cluster_id
    return clusters


def run_vsearch(fasta_path, output_prefix, threshold, threads=4):
    """Run VSEARCH cluster_fast."""
    uc_path = str(output_prefix) + ".uc"
    cmd = [
        VSEARCH_BIN, "--cluster_fast", str(fasta_path),
        "--id", str(threshold),
        "--uc", uc_path,
        "--threads", str(threads),
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        return None, elapsed

    clusters = parse_vsearch_clusters(uc_path)
    return clusters, elapsed


def _normalize_id(seq_id):
    """Extract UniProt accession from sp|ACC|NAME format, or return as-is."""
    parts = seq_id.split("|")
    if len(parts) >= 2:
        return parts[1]
    return seq_id


def evaluate_tool(ground_truth, pred_clusters):
    """Evaluate predicted clusters (dict: seq_id -> cluster_id) against ground truth."""
    # Try direct match first
    common_ids = sorted(set(ground_truth.keys()) & set(pred_clusters.keys()))
    if not common_ids:
        # Try normalized (accession-only) matching
        gt_norm = {_normalize_id(k): v for k, v in ground_truth.items()}
        pred_norm = {_normalize_id(k): v for k, v in pred_clusters.items()}
        common_ids = sorted(set(gt_norm.keys()) & set(pred_norm.keys()))
        if not common_ids:
            return {"error": "no common sequences"}
        ground_truth = gt_norm
        pred_clusters = pred_norm

    true_family_list = [ground_truth[sid] for sid in common_ids]
    pred_cluster_list = [pred_clusters[sid] for sid in common_ids]

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


def run_benchmark(thresholds=None, max_per_family=500, threads=4):
    """Run the full Pfam concordance benchmark with ClustKIT, CD-HIT, and MMseqs2.

    Args:
        thresholds: List of identity thresholds to test.
        max_per_family: Max sequences per Pfam family (0 = no cap).
        threads: Number of CPU threads for all tools (for fair comparison).
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and mix families
    print("=" * 120)
    print(f"PFAM CONCORDANCE BENCHMARK — ClustKIT vs CD-HIT vs MMseqs2 ({threads} threads)")
    print("=" * 120)
    print()

    mixed_fasta, ground_truth = load_and_mix_families(DATA_DIR, max_per_family)
    print()

    all_results = {}

    for threshold in thresholds:
        print("-" * 120)
        print(f"Threshold = {threshold}")
        print("-" * 120)

        scenario = {}

        # --- ClustKIT ---
        print(f"  ClustKIT (align)...", end=" ", flush=True)
        out_dir = OUTPUT_DIR / f"clustkit_t{threshold}"
        out_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "input": mixed_fasta,
            "output": out_dir,
            "threshold": threshold,
            "mode": "protein",
            "alignment": "align",
            "sketch_size": 128,
            "kmer_size": 5,
            "sensitivity": "high",
            "cluster_method": "connected",
            "representative": "longest",
            "device": "cpu",
            "threads": threads,
            "format": "tsv",
        }

        start = time.perf_counter()
        run_pipeline(config)
        elapsed = time.perf_counter() - start

        # Parse ClustKIT output
        clustkit_clusters = {}
        with open(out_dir / "clusters.tsv") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                clustkit_clusters[parts[0]] = int(parts[1])

        clustkit_eval = evaluate_tool(ground_truth, clustkit_clusters)
        clustkit_eval["runtime_seconds"] = round(elapsed, 2)
        scenario["clustkit"] = clustkit_eval
        if "error" not in clustkit_eval:
            print(f"{clustkit_eval['n_predicted_clusters']} clusters, "
                  f"ARI={clustkit_eval['ARI']}, "
                  f"P={clustkit_eval['pairwise_precision']}, R={clustkit_eval['pairwise_recall']}, F1={clustkit_eval['pairwise_F1']}, "
                  f"{elapsed:.2f}s")
        else:
            print(f"EVAL ERROR: {clustkit_eval['error']} ({elapsed:.2f}s)")

        # --- CD-HIT ---
        print(f"  CD-HIT...", end=" ", flush=True)
        cdhit_prefix = OUTPUT_DIR / f"cdhit_t{threshold}"
        cdhit_clusters, cdhit_time = run_cdhit(mixed_fasta, cdhit_prefix, threshold, threads=threads)
        if cdhit_clusters:
            cdhit_eval = evaluate_tool(ground_truth, cdhit_clusters)
            cdhit_eval["runtime_seconds"] = round(cdhit_time, 2)
            scenario["cdhit"] = cdhit_eval
            if "error" not in cdhit_eval:
                print(f"{cdhit_eval['n_predicted_clusters']} clusters, "
                      f"ARI={cdhit_eval['ARI']}, "
                      f"P={cdhit_eval['pairwise_precision']}, R={cdhit_eval['pairwise_recall']}, F1={cdhit_eval['pairwise_F1']}, "
                      f"{cdhit_time:.2f}s")
            else:
                print(f"EVAL ERROR: {cdhit_eval['error']} ({cdhit_time:.2f}s)")
        else:
            scenario["cdhit"] = {"error": "failed", "runtime_seconds": round(cdhit_time, 2)}
            print(f"FAILED ({cdhit_time:.2f}s)")

        # --- MMseqs2 ---
        print(f"  MMseqs2...", end=" ", flush=True)
        mmseqs_prefix = OUTPUT_DIR / f"mmseqs_t{threshold}"
        mmseqs_clusters, mmseqs_time = run_mmseqs(mixed_fasta, mmseqs_prefix, threshold, threads=threads)
        if mmseqs_clusters:
            mmseqs_eval = evaluate_tool(ground_truth, mmseqs_clusters)
            mmseqs_eval["runtime_seconds"] = round(mmseqs_time, 2)
            scenario["mmseqs2"] = mmseqs_eval
            if "error" not in mmseqs_eval:
                print(f"{mmseqs_eval['n_predicted_clusters']} clusters, "
                      f"ARI={mmseqs_eval['ARI']}, "
                      f"P={mmseqs_eval['pairwise_precision']}, R={mmseqs_eval['pairwise_recall']}, F1={mmseqs_eval['pairwise_F1']}, "
                      f"{mmseqs_time:.2f}s")
            else:
                print(f"EVAL ERROR: {mmseqs_eval['error']} ({mmseqs_time:.2f}s)")
        else:
            scenario["mmseqs2"] = {"error": "failed", "runtime_seconds": round(mmseqs_time, 2)}
            print(f"FAILED ({mmseqs_time:.2f}s)")

        # --- MMseqs2 linclust ---
        print(f"  MMseqs2 linclust...", end=" ", flush=True)
        linclust_prefix = OUTPUT_DIR / f"linclust_t{threshold}"
        linclust_clusters, linclust_time = run_mmseqs_linclust(mixed_fasta, linclust_prefix, threshold, threads=threads)
        if linclust_clusters:
            linclust_eval = evaluate_tool(ground_truth, linclust_clusters)
            linclust_eval["runtime_seconds"] = round(linclust_time, 2)
            scenario["linclust"] = linclust_eval
            if "error" not in linclust_eval:
                print(f"{linclust_eval['n_predicted_clusters']} clusters, "
                      f"ARI={linclust_eval['ARI']}, "
                      f"P={linclust_eval['pairwise_precision']}, R={linclust_eval['pairwise_recall']}, F1={linclust_eval['pairwise_F1']}, "
                      f"{linclust_time:.2f}s")
            else:
                print(f"EVAL ERROR: {linclust_eval['error']} ({linclust_time:.2f}s)")
        else:
            scenario["linclust"] = {"error": "failed", "runtime_seconds": round(linclust_time, 2)}
            print(f"FAILED ({linclust_time:.2f}s)")

        # --- VSEARCH ---
        print(f"  VSEARCH...", end=" ", flush=True)
        vsearch_prefix = OUTPUT_DIR / f"vsearch_t{threshold}"
        vsearch_clusters, vsearch_time = run_vsearch(mixed_fasta, vsearch_prefix, threshold, threads=threads)
        if vsearch_clusters:
            vsearch_eval = evaluate_tool(ground_truth, vsearch_clusters)
            vsearch_eval["runtime_seconds"] = round(vsearch_time, 2)
            scenario["vsearch"] = vsearch_eval
            if "error" not in vsearch_eval:
                print(f"{vsearch_eval['n_predicted_clusters']} clusters, "
                      f"ARI={vsearch_eval['ARI']}, "
                      f"P={vsearch_eval['pairwise_precision']}, R={vsearch_eval['pairwise_recall']}, F1={vsearch_eval['pairwise_F1']}, "
                      f"{vsearch_time:.2f}s")
            else:
                print(f"EVAL ERROR: {vsearch_eval['error']} ({vsearch_time:.2f}s)")
        else:
            scenario["vsearch"] = {"error": "failed", "runtime_seconds": round(vsearch_time, 2)}
            print(f"FAILED ({vsearch_time:.2f}s)")

        all_results[str(threshold)] = scenario
        print()

    # Summary table
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print()
    print(f"{'Thresh':<8} {'Tool':<10} {'Clust':>6} {'ARI':>8} {'NMI':>8} "
          f"{'Homog':>8} {'Compl':>8} {'P(pw)':>8} {'R(pw)':>8} {'F1(pw)':>8} {'Time':>10}")
    print("-" * 120)

    for t in thresholds:
        r = all_results[str(t)]
        for tool_name, key in [("ClustKIT", "clustkit"), ("CD-HIT", "cdhit"), ("MMseqs2", "mmseqs2"), ("linclust", "linclust"), ("VSEARCH", "vsearch")]:
            res = r.get(key, {})
            if "error" in res:
                rt = res.get("runtime_seconds", "?")
                print(f"{t:<8} {tool_name:<10} {'FAIL':>6} {'':>8} {'':>8} "
                      f"{'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {rt:>9}s")
            elif res:
                print(
                    f"{t:<8} {tool_name:<10} {res['n_predicted_clusters']:>6} "
                    f"{res['ARI']:>8.4f} {res['NMI']:>8.4f} "
                    f"{res['homogeneity']:>8.4f} {res['completeness']:>8.4f} "
                    f"{res['pairwise_precision']:>8.4f} {res['pairwise_recall']:>8.4f} "
                    f"{res['pairwise_F1']:>8.4f} {res['runtime_seconds']:>9.2f}s"
                )
        print()

    # Save results
    results_file = OUTPUT_DIR / f"pfam_concordance_results_{threads}threads.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pfam concordance benchmark")
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of CPU threads for all tools (default: 4)",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Identity thresholds to test",
    )
    parser.add_argument(
        "--max-per-family", type=int, default=500,
        help="Max sequences per Pfam family (default: 500)",
    )
    args = parser.parse_args()

    run_benchmark(
        thresholds=args.thresholds,
        max_per_family=args.max_per_family,
        threads=args.threads,
    )
