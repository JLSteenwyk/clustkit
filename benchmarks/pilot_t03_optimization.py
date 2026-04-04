"""Pilot: optimize ClustKIT clustering at t=0.3.

Tests greedy vs connected components, and coverage filtering.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.pipeline import run_pipeline

DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "pfam_benchmark_results"


def load_ground_truth(data_dir, max_per_family=500):
    gt = {}
    for f in sorted(data_dir.glob("PF*.fasta")):
        pfam_id = f.stem.split("_")[0]
        count = 0
        for line in open(f):
            if line.startswith(">"):
                sid = line.strip().split()[0][1:]
                if count < max_per_family:
                    gt[sid] = pfam_id
                    count += 1
    return gt


def evaluate(gt, cluster_file):
    pred = {}
    with open(cluster_file) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            pred[parts[0]] = int(parts[1])
    gt_n = {k.split("|")[1] if "|" in k else k: v for k, v in gt.items()}
    pred_n = {k.split("|")[1] if "|" in k else k: v for k, v in pred.items()}
    common = sorted(set(gt_n.keys()) & set(pred_n.keys()))
    if not common:
        return {"error": "no common seqs"}
    true_l = [gt_n[s] for s in common]
    pred_l = [pred_n[s] for s in common]
    ari = adjusted_rand_score(true_l, pred_l)
    n_clusters = len(set(pred_l))
    # Pairwise metrics
    n = len(common)
    rng = np.random.RandomState(42)
    ia = rng.randint(0, n, size=1_000_000)
    ib = rng.randint(0, n, size=1_000_000)
    v = ia != ib; ia, ib = ia[v], ib[v]
    ta = np.array([hash(t) for t in true_l], dtype=np.int64)
    pa = np.array(pred_l, dtype=np.int64)
    sp = pa[ia] == pa[ib]
    st = ta[ia] == ta[ib]
    tp = int(np.sum(sp & st)); fp = int(np.sum(sp & ~st))
    fn = int(np.sum(~sp & st))
    pr = tp / (tp + fp) if (tp + fp) else 0
    rc = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
    return {
        "n_clusters": n_clusters,
        "ARI": round(ari, 4),
        "precision": round(pr, 4),
        "recall": round(rc, 4),
        "F1": round(f1, 4),
    }


def run_config(label, threshold, cluster_method, threads=8):
    out_dir = OUTPUT_DIR / f"t03_pilot_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "input": DATA_DIR.parent / "pfam_mixed.fasta",
        "output": out_dir,
        "threshold": threshold,
        "mode": "protein",
        "alignment": "align",
        "sketch_size": 128,
        "kmer_size": 5,
        "sensitivity": "high",
        "cluster_method": cluster_method,
        "representative": "longest",
        "device": "cpu",
        "threads": threads,
        "format": "tsv",
        "use_c_ext": True,
    }
    start = time.perf_counter()
    run_pipeline(config)
    elapsed = time.perf_counter() - start
    gt = load_ground_truth(DATA_DIR)
    results = evaluate(gt, out_dir / "clusters.tsv")
    results["time"] = round(elapsed, 2)
    return results


if __name__ == "__main__":
    print("=" * 90)
    print("T=0.3 CLUSTERING OPTIMIZATION PILOT")
    print("=" * 90)

    configs = [
        # (label, threshold, cluster_method)
        ("A_connected_t03", 0.3, "connected"),
        ("B_greedy_t03", 0.3, "greedy"),
        ("C_greedy_t035", 0.35, "greedy"),
        ("D_greedy_t04", 0.4, "greedy"),
        ("E_connected_t04", 0.4, "connected"),
    ]

    print(f"\n{'Config':<25} {'Clust':>6} {'ARI':>8} {'P':>8} {'R':>8} {'F1':>8} {'Time':>8}")
    print("-" * 80)

    for label, threshold, method in configs:
        print(f"Running {label}...", flush=True)
        r = run_config(label, threshold, method)
        print(f"{label:<25} {r['n_clusters']:>6} {r['ARI']:>8.4f} {r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} {r['F1']:>8.4f} {r['time']:>7.1f}s", flush=True)

    print("\nReference: MMseqs2 t=0.3: ARI=0.270, P=0.998, R=0.158, 4070 clusters, 39s")
    print("Reference: DeepClust t=0.3: ARI=0.261, P=0.980, R=0.153, 3838 clusters, 77s")
