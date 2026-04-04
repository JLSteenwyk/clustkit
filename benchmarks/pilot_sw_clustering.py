"""Quick pilot: test SW local alignment in clustering pipeline.

Compares C SW vs old NW at t=0.3 and t=0.5 on Pfam data.
"""

import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.pipeline import run_pipeline


DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "pfam_benchmark_results"


def load_ground_truth(data_dir, max_per_family=500):
    """Load ground truth labels from Pfam families."""
    fasta_files = sorted(data_dir.glob("PF*.fasta"))
    ground_truth = {}
    for fasta_file in fasta_files:
        pfam_id = fasta_file.stem.split("_")[0]
        count = 0
        with open(fasta_file) as f:
            for line in f:
                if line.startswith(">"):
                    seq_id = line.strip().split()[0][1:]
                    if max_per_family == 0 or count < max_per_family:
                        ground_truth[seq_id] = pfam_id
                        count += 1
    return ground_truth


def evaluate(ground_truth, cluster_file):
    """Evaluate clustering output against ground truth."""
    pred = {}
    with open(cluster_file) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            pred[parts[0]] = int(parts[1])

    # Try normalized matching
    if not set(ground_truth.keys()) & set(pred.keys()):
        gt_norm = {}
        for k, v in ground_truth.items():
            parts = k.split("|")
            key = parts[1] if len(parts) >= 2 else k
            gt_norm[key] = v
        pred_norm = {}
        for k, v in pred.items():
            parts = k.split("|")
            key = parts[1] if len(parts) >= 2 else k
            pred_norm[key] = v
        ground_truth = gt_norm
        pred = pred_norm

    common = sorted(set(ground_truth.keys()) & set(pred.keys()))
    if not common:
        return {"error": "no common sequences"}

    true_labels = [ground_truth[s] for s in common]
    pred_labels = [pred[s] for s in common]
    ari = adjusted_rand_score(true_labels, pred_labels)
    n_clusters = len(set(pred_labels))

    # Pairwise precision/recall
    n = len(common)
    rng = np.random.RandomState(42)
    idx_a = rng.randint(0, n, size=500000)
    idx_b = rng.randint(0, n, size=500000)
    valid = idx_a != idx_b
    idx_a, idx_b = idx_a[valid], idx_b[valid]
    true_arr = np.array([hash(t) for t in true_labels], dtype=np.int64)
    pred_arr = np.array(pred_labels, dtype=np.int64)
    same_pred = pred_arr[idx_a] == pred_arr[idx_b]
    same_true = true_arr[idx_a] == true_arr[idx_b]
    tp = int(np.sum(same_pred & same_true))
    fp = int(np.sum(same_pred & ~same_true))
    fn = int(np.sum(~same_pred & same_true))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "n_clusters": n_clusters,
        "ARI": round(ari, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "F1": round(f1, 4),
    }


def run_test(threshold, use_c_ext, label):
    """Run clustering at given threshold and evaluate."""
    mixed_fasta = DATA_DIR.parent / "pfam_mixed.fasta"
    out_dir = OUTPUT_DIR / f"sw_pilot_{label}_t{threshold}"
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
        "threads": 8,
        "format": "tsv",
        "use_c_ext": use_c_ext,
    }

    start = time.perf_counter()
    run_pipeline(config)
    elapsed = time.perf_counter() - start

    gt = load_ground_truth(DATA_DIR)
    results = evaluate(gt, out_dir / "clusters.tsv")
    results["time"] = round(elapsed, 2)
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("SW CLUSTERING PILOT TEST")
    print("=" * 80)

    results = {}

    for threshold in [0.3, 0.5]:
        print(f"\n--- Threshold = {threshold} ---")

        # C SW (new)
        print(f"  C SW (new)...", end=" ", flush=True)
        r = run_test(threshold, use_c_ext=True, label="csw")
        print(f"clusters={r['n_clusters']}, ARI={r['ARI']}, P={r['precision']}, "
              f"R={r['recall']}, F1={r['F1']}, time={r['time']}s")
        results[f"c_sw_t{threshold}"] = r

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for k, v in results.items():
        print(f"  {k}: {v}")
