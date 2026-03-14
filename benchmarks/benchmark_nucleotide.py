"""Nucleotide (16S rRNA) clustering benchmark.

Compares ClustKIT vs VSEARCH vs CD-HIT-EST vs MMseqs2 on 16S sequences
at 97%, 95%, and 90% identity thresholds.

C3 from the publication plan.
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


# ──────────────────────────────────────────────────────────────────────
# Tool binary paths
# ──────────────────────────────────────────────────────────────────────

VSEARCH_BIN = (
    "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/VSEARCH/"
    "vsearch-2.30.5-linux-x86_64/bin/vsearch"
)
MMSEQS_BIN = (
    "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
)
CDHIT_SIF = (
    "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/cd-hit.sif"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "nucleotide_benchmark_results"


# ──────────────────────────────────────────────────────────────────────
# Evaluation helpers (reused from benchmark_pfam_concordance.py)
# ──────────────────────────────────────────────────────────────────────

def pairwise_precision_recall_f1(true_labels, pred_labels):
    """Compute pairwise precision, recall, and F1.

    A pair is a "positive" if two sequences are in the same predicted cluster.
    A "true positive" means they are also in the same ground-truth family.

    Uses sampling for large datasets to avoid O(N^2) computation.
    """
    n = len(true_labels)

    if n > 5000:
        num_samples = 2_000_000
        rng = np.random.RandomState(42)
        idx_a = rng.randint(0, n, size=num_samples)
        idx_b = rng.randint(0, n, size=num_samples)
        valid = idx_a != idx_b
        idx_a = idx_a[valid]
        idx_b = idx_b[valid]
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
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (
        (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    )

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


def evaluate_tool(ground_truth, pred_clusters):
    """Evaluate predicted clusters against ground truth taxonomy labels.

    Args:
        ground_truth: dict mapping seq_id -> taxon label.
        pred_clusters: dict mapping seq_id -> cluster_id.

    Returns:
        dict of evaluation metrics.
    """
    common_ids = sorted(set(ground_truth.keys()) & set(pred_clusters.keys()))
    if not common_ids:
        return {"error": "no common sequences"}

    true_label_list = [ground_truth[sid] for sid in common_ids]
    pred_cluster_list = [pred_clusters[sid] for sid in common_ids]

    label_to_int = {f: i for i, f in enumerate(sorted(set(true_label_list)))}
    true_labels = np.array(
        [label_to_int[f] for f in true_label_list], dtype=np.int32
    )
    pred_labels = np.array(pred_cluster_list, dtype=np.int32)

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_labels, pred_labels
    )
    pw = pairwise_precision_recall_f1(true_labels, pred_labels)

    return {
        "n_sequences": len(common_ids),
        "n_true_taxa": len(set(true_label_list)),
        "n_predicted_clusters": len(set(pred_cluster_list)),
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "homogeneity": round(homogeneity, 4),
        "completeness": round(completeness, 4),
        "V_measure": round(v_measure, 4),
        **pw,
    }


# ──────────────────────────────────────────────────────────────────────
# Taxonomy file parsing
# ──────────────────────────────────────────────────────────────────────

def load_taxonomy(taxonomy_path):
    """Load ground truth taxonomy file (TSV: seq_id<TAB>taxon).

    Returns:
        dict mapping seq_id -> taxon label.
    """
    taxonomy = {}
    with open(taxonomy_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                taxonomy[parts[0]] = parts[1]
    return taxonomy


# ──────────────────────────────────────────────────────────────────────
# External tool parsers
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Tool runners
# ──────────────────────────────────────────────────────────────────────

def run_clustkit(fasta_path, output_dir, threshold, threads=4):
    """Run ClustKIT in nucleotide mode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input": fasta_path,
        "output": output_dir,
        "threshold": threshold,
        "mode": "nucleotide",
        "alignment": "align",
        "sketch_size": 128,
        "kmer_size": 11,
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

    clusters = {}
    tsv_path = output_dir / "clusters.tsv"
    if tsv_path.exists():
        with open(tsv_path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    clusters[parts[0]] = int(parts[1])

    return clusters if clusters else None, elapsed


def run_vsearch(fasta_path, output_prefix, threshold, threads=4):
    """Run VSEARCH --cluster_fast on nucleotide sequences."""
    uc_path = str(output_prefix) + ".uc"
    cmd = [
        VSEARCH_BIN,
        "--cluster_fast", str(fasta_path),
        "--id", str(threshold),
        "--uc", uc_path,
        "--threads", str(threads),
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"    VSEARCH stderr: {result.stderr[:500]}")
        return None, elapsed

    clusters = parse_vsearch_clusters(uc_path)
    return clusters, elapsed


def _cdhit_est_word_size(threshold):
    """Choose word size (-n) for cd-hit-est based on threshold.

    From the CD-HIT documentation:
      -n 8,9,10,11 for t >= 0.90
      -n 6,7       for t >= 0.88
      -n 4,5       for t >= 0.80
      -n 2,3       for t <  0.80
    """
    if threshold >= 0.90:
        return 8
    elif threshold >= 0.88:
        return 6
    elif threshold >= 0.85:
        return 5
    elif threshold >= 0.80:
        return 4
    else:
        return 3


def run_cdhit_est(fasta_path, output_prefix, threshold, threads=4):
    """Run CD-HIT-EST on nucleotide sequences."""
    word_size = _cdhit_est_word_size(threshold)
    cmd = [
        "singularity", "exec", CDHIT_SIF,
        "cd-hit-est",
        "-i", str(fasta_path),
        "-o", str(output_prefix),
        "-c", str(threshold),
        "-n", str(word_size),
        "-T", str(threads),
        "-M", "0",
        "-d", "0",
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"    CD-HIT-EST stderr: {result.stderr[:500]}")
        return None, elapsed

    clusters = parse_cdhit_clusters(str(output_prefix) + ".clstr")
    return clusters, elapsed


def run_mmseqs(fasta_path, output_prefix, threshold, threads=4):
    """Run MMseqs2 easy-cluster on nucleotide sequences."""
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_nt_tmp_")
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
        print(f"    MMseqs2 stderr: {result.stderr[:500]}")
        return None, elapsed

    clusters = parse_mmseqs_clusters(str(output_prefix) + "_cluster.tsv")
    return clusters, elapsed


# ──────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────

def run_benchmark(input_fasta, taxonomy_path, thresholds, threads=4):
    """Run the 16S rRNA nucleotide clustering benchmark.

    Args:
        input_fasta: Path to the 16S rRNA FASTA file.
        taxonomy_path: Path to the ground truth taxonomy TSV.
        thresholds: List of identity thresholds to test.
        threads: Number of CPU threads for all tools.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print(f"NUCLEOTIDE 16S rRNA BENCHMARK — ClustKIT vs VSEARCH vs CD-HIT-EST vs MMseqs2 ({threads} threads)")
    print("=" * 120)
    print()

    # Load taxonomy
    print("Loading taxonomy...")
    ground_truth = load_taxonomy(taxonomy_path)
    taxon_counts = Counter(ground_truth.values())
    print(f"  {len(ground_truth)} sequences, {len(taxon_counts)} taxa")
    for taxon, cnt in sorted(taxon_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {taxon}: {cnt} sequences")
    if len(taxon_counts) > 10:
        print(f"    ... and {len(taxon_counts) - 10} more taxa")
    print()

    # Verify FASTA exists
    input_fasta = Path(input_fasta)
    if not input_fasta.exists():
        print(f"ERROR: Input FASTA not found: {input_fasta}")
        sys.exit(1)

    all_results = {}

    for threshold in thresholds:
        print("-" * 120)
        print(f"Threshold = {threshold}")
        print("-" * 120)

        scenario = {}

        # --- ClustKIT ---
        print(f"  ClustKIT (nucleotide)...", end=" ", flush=True)
        out_dir = OUTPUT_DIR / f"clustkit_t{threshold}"
        clustkit_clusters, clustkit_time = run_clustkit(
            input_fasta, out_dir, threshold, threads=threads
        )
        if clustkit_clusters:
            clustkit_eval = evaluate_tool(ground_truth, clustkit_clusters)
            clustkit_eval["runtime_seconds"] = round(clustkit_time, 2)
            scenario["clustkit"] = clustkit_eval
            if "error" not in clustkit_eval:
                print(
                    f"{clustkit_eval['n_predicted_clusters']} clusters, "
                    f"ARI={clustkit_eval['ARI']}, "
                    f"P={clustkit_eval['pairwise_precision']}, "
                    f"R={clustkit_eval['pairwise_recall']}, "
                    f"F1={clustkit_eval['pairwise_F1']}, "
                    f"{clustkit_time:.2f}s"
                )
            else:
                print(f"EVAL ERROR: {clustkit_eval['error']} ({clustkit_time:.2f}s)")
        else:
            scenario["clustkit"] = {
                "error": "failed",
                "runtime_seconds": round(clustkit_time, 2),
            }
            print(f"FAILED ({clustkit_time:.2f}s)")

        # --- VSEARCH ---
        print(f"  VSEARCH...", end=" ", flush=True)
        vsearch_prefix = OUTPUT_DIR / f"vsearch_t{threshold}"
        vsearch_clusters, vsearch_time = run_vsearch(
            input_fasta, vsearch_prefix, threshold, threads=threads
        )
        if vsearch_clusters:
            vsearch_eval = evaluate_tool(ground_truth, vsearch_clusters)
            vsearch_eval["runtime_seconds"] = round(vsearch_time, 2)
            scenario["vsearch"] = vsearch_eval
            if "error" not in vsearch_eval:
                print(
                    f"{vsearch_eval['n_predicted_clusters']} clusters, "
                    f"ARI={vsearch_eval['ARI']}, "
                    f"P={vsearch_eval['pairwise_precision']}, "
                    f"R={vsearch_eval['pairwise_recall']}, "
                    f"F1={vsearch_eval['pairwise_F1']}, "
                    f"{vsearch_time:.2f}s"
                )
            else:
                print(f"EVAL ERROR: {vsearch_eval['error']} ({vsearch_time:.2f}s)")
        else:
            scenario["vsearch"] = {
                "error": "failed",
                "runtime_seconds": round(vsearch_time, 2),
            }
            print(f"FAILED ({vsearch_time:.2f}s)")

        # --- CD-HIT-EST ---
        print(f"  CD-HIT-EST...", end=" ", flush=True)
        cdhit_prefix = OUTPUT_DIR / f"cdhitest_t{threshold}"
        cdhit_clusters, cdhit_time = run_cdhit_est(
            input_fasta, cdhit_prefix, threshold, threads=threads
        )
        if cdhit_clusters:
            cdhit_eval = evaluate_tool(ground_truth, cdhit_clusters)
            cdhit_eval["runtime_seconds"] = round(cdhit_time, 2)
            scenario["cdhit_est"] = cdhit_eval
            if "error" not in cdhit_eval:
                print(
                    f"{cdhit_eval['n_predicted_clusters']} clusters, "
                    f"ARI={cdhit_eval['ARI']}, "
                    f"P={cdhit_eval['pairwise_precision']}, "
                    f"R={cdhit_eval['pairwise_recall']}, "
                    f"F1={cdhit_eval['pairwise_F1']}, "
                    f"{cdhit_time:.2f}s"
                )
            else:
                print(f"EVAL ERROR: {cdhit_eval['error']} ({cdhit_time:.2f}s)")
        else:
            scenario["cdhit_est"] = {
                "error": "failed",
                "runtime_seconds": round(cdhit_time, 2),
            }
            print(f"FAILED ({cdhit_time:.2f}s)")

        # --- MMseqs2 ---
        print(f"  MMseqs2...", end=" ", flush=True)
        mmseqs_prefix = OUTPUT_DIR / f"mmseqs_t{threshold}"
        mmseqs_clusters, mmseqs_time = run_mmseqs(
            input_fasta, mmseqs_prefix, threshold, threads=threads
        )
        if mmseqs_clusters:
            mmseqs_eval = evaluate_tool(ground_truth, mmseqs_clusters)
            mmseqs_eval["runtime_seconds"] = round(mmseqs_time, 2)
            scenario["mmseqs2"] = mmseqs_eval
            if "error" not in mmseqs_eval:
                print(
                    f"{mmseqs_eval['n_predicted_clusters']} clusters, "
                    f"ARI={mmseqs_eval['ARI']}, "
                    f"P={mmseqs_eval['pairwise_precision']}, "
                    f"R={mmseqs_eval['pairwise_recall']}, "
                    f"F1={mmseqs_eval['pairwise_F1']}, "
                    f"{mmseqs_time:.2f}s"
                )
            else:
                print(f"EVAL ERROR: {mmseqs_eval['error']} ({mmseqs_time:.2f}s)")
        else:
            scenario["mmseqs2"] = {
                "error": "failed",
                "runtime_seconds": round(mmseqs_time, 2),
            }
            print(f"FAILED ({mmseqs_time:.2f}s)")

        all_results[str(threshold)] = scenario
        print()

    # ── Summary table ────────────────────────────────────────────────
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print()
    print(
        f"{'Thresh':<8} {'Tool':<12} {'Clust':>6} {'ARI':>8} {'NMI':>8} "
        f"{'Homog':>8} {'Compl':>8} {'P(pw)':>8} {'R(pw)':>8} "
        f"{'F1(pw)':>8} {'Time':>10}"
    )
    print("-" * 120)

    tool_order = [
        ("ClustKIT", "clustkit"),
        ("VSEARCH", "vsearch"),
        ("CD-HIT-EST", "cdhit_est"),
        ("MMseqs2", "mmseqs2"),
    ]

    for t in thresholds:
        r = all_results[str(t)]
        for tool_name, key in tool_order:
            res = r.get(key, {})
            if "error" in res:
                rt = res.get("runtime_seconds", "?")
                print(
                    f"{t:<8} {tool_name:<12} {'FAIL':>6} {'':>8} {'':>8} "
                    f"{'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {rt:>9}s"
                )
            elif res:
                print(
                    f"{t:<8} {tool_name:<12} "
                    f"{res['n_predicted_clusters']:>6} "
                    f"{res['ARI']:>8.4f} {res['NMI']:>8.4f} "
                    f"{res['homogeneity']:>8.4f} "
                    f"{res['completeness']:>8.4f} "
                    f"{res['pairwise_precision']:>8.4f} "
                    f"{res['pairwise_recall']:>8.4f} "
                    f"{res['pairwise_F1']:>8.4f} "
                    f"{res['runtime_seconds']:>9.2f}s"
                )
        print()

    # Save results
    results_file = OUTPUT_DIR / "nucleotide_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Nucleotide 16S rRNA clustering benchmark (C3)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input 16S rRNA FASTA file.",
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        required=True,
        help="Ground truth taxonomy file (TSV: seq_id<TAB>taxon).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for all tools (default: 4).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.97],
        help="Identity thresholds to test (default: 0.90 0.95 0.97).",
    )
    args = parser.parse_args()

    run_benchmark(
        input_fasta=args.input,
        taxonomy_path=args.taxonomy,
        thresholds=args.thresholds,
        threads=args.threads,
    )
