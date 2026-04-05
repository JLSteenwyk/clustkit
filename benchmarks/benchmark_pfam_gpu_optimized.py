"""Pfam GPU Benchmark: optimized GPU on real Pfam data.

GPU-only — reuses existing CPU results from pfam_concordance_results_8threads.json.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline

DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "pfam_benchmark_results"
THREADS = 8
GPU_DEVICE = "1"


def load_and_mix_families(data_dir, max_per_family=500):
    fasta_files = sorted(data_dir.glob("PF*.fasta"))
    ground_truth = {}
    mixed_fasta_path = data_dir.parent / "pfam_mixed.fasta"

    with open(mixed_fasta_path, "w") as out:
        for fasta_file in fasta_files:
            pfam_id = fasta_file.stem.split("_")[0]
            count = 0
            current_header = None
            current_seq_lines = []

            with open(fasta_file) as f:
                for line in f:
                    line = line.rstrip("\n\r")
                    if line.startswith(">"):
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

                if current_header is not None and (
                    max_per_family == 0 or count < max_per_family
                ):
                    seq_id = current_header.split()[0][1:]
                    ground_truth[seq_id] = pfam_id
                    out.write(current_header + "\n")
                    out.write("\n".join(current_seq_lines) + "\n")

    family_counts = Counter(ground_truth.values())
    print(f"Mixed dataset: {len(ground_truth)} sequences from {len(family_counts)} families")
    return mixed_fasta_path, ground_truth


def _normalize_id(seq_id):
    parts = seq_id.split("|")
    return parts[1] if len(parts) >= 2 else seq_id


def evaluate(ground_truth, pred_clusters):
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import LabelEncoder

    common_ids = sorted(set(ground_truth.keys()) & set(pred_clusters.keys()))
    if not common_ids:
        gt_norm = {_normalize_id(k): v for k, v in ground_truth.items()}
        pred_norm = {_normalize_id(k): v for k, v in pred_clusters.items()}
        common_ids = sorted(set(gt_norm.keys()) & set(pred_norm.keys()))
        if not common_ids:
            return {"error": "no common sequences"}
        ground_truth, pred_clusters = gt_norm, pred_norm

    true_labels = np.array([ground_truth[s] for s in common_ids])
    pred_labels = np.array([pred_clusters[s] for s in common_ids])

    le_true = LabelEncoder().fit(true_labels)
    le_pred = LabelEncoder().fit(pred_labels)

    ari = adjusted_rand_score(le_true.transform(true_labels), le_pred.transform(pred_labels))
    n_clusters = len(set(pred_labels))
    return {"ARI": round(ari, 4), "n_clusters": n_clusters, "n_seqs": len(common_ids)}


def run_gpu(fasta, threshold, device, threads):
    import numba
    numba.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    out_dir = OUTPUT_DIR / f"gpu_opt_{device}_t{threshold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input": str(fasta),
        "output": str(out_dir),
        "threshold": threshold,
        "mode": "protein",
        "alignment": "align",
        "sketch_size": 128,
        "kmer_size": 5,
        "sensitivity": "high",
        "cluster_method": "leiden",
        "representative": "longest",
        "device": device,
        "threads": threads,
        "format": "tsv",
        "use_c_ext": True,
        "band_width": 100,
        "block": "off",
        "cascade": "off",
    }

    start = time.perf_counter()
    run_pipeline(config)
    elapsed = time.perf_counter() - start

    clusters = {}
    with open(out_dir / "clusters.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            clusters[parts[0]] = int(parts[1])
    return clusters, elapsed


def main():
    thresholds = [0.3, 0.5, 0.7, 0.9]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing CPU results
    cpu_file = OUTPUT_DIR / "pfam_concordance_results_8threads.json"
    with open(cpu_file) as f:
        cpu_data = json.load(f)
    print(f"Loaded CPU baselines from {cpu_file.name}")

    mixed_fasta, ground_truth = load_and_mix_families(DATA_DIR, max_per_family=500)

    # Warm up GPU
    print("\nWarming up GPU...", flush=True)
    warmup_dir = OUTPUT_DIR / "gpu_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline({
        "input": str(mixed_fasta), "output": str(warmup_dir),
        "threshold": 0.9, "mode": "protein", "alignment": "align",
        "sketch_size": 128, "kmer_size": 5, "sensitivity": "medium",
        "cluster_method": "greedy", "representative": "longest",
        "device": GPU_DEVICE, "threads": THREADS, "format": "tsv",
        "use_c_ext": True, "band_width": 50, "block": "off", "cascade": "off",
    })
    print("Done.\n")

    print("=" * 80)
    print(f"PFAM BENCHMARK — Optimized GPU:{GPU_DEVICE} vs CPU ({THREADS} threads)")
    print(f"22K Pfam sequences, 56 families, max_len=35213")
    print("=" * 80)

    results = {}

    for threshold in thresholds:
        print(f"\n{'─'*80}")
        print(f"Threshold = {threshold}")
        print(f"{'─'*80}")

        # CPU from saved results
        cpu_res = cpu_data.get(str(threshold), {}).get("clustkit", {})
        cpu_time = cpu_res.get("runtime_seconds", 0)
        cpu_ari = cpu_res.get("ARI", 0)
        cpu_clusters = cpu_res.get("n_predicted_clusters", 0)
        print(f"  CPU (saved): {cpu_clusters} clusters, ARI={cpu_ari}, {cpu_time}s")

        # GPU
        print(f"  GPU:{GPU_DEVICE}...", end=" ", flush=True)
        gpu_clusters_dict, gpu_time = run_gpu(mixed_fasta, threshold, GPU_DEVICE, THREADS)
        gpu_ev = evaluate(ground_truth, gpu_clusters_dict)
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"{gpu_ev['n_clusters']} clusters, ARI={gpu_ev['ARI']}, {gpu_time:.1f}s  ({speedup:.1f}x speedup)")

        results[threshold] = {
            "cpu_clusters": cpu_clusters,
            "cpu_ari": cpu_ari,
            "cpu_time": cpu_time,
            "gpu_clusters": gpu_ev["n_clusters"],
            "gpu_ari": gpu_ev["ARI"],
            "gpu_time": round(gpu_time, 2),
            "speedup": round(speedup, 1),
        }

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY — GPU Optimized vs CPU (8 threads)")
    print(f"{'='*80}")
    print(f"{'Thresh':<8} {'CPU clust':>10} {'GPU clust':>10} {'CPU ARI':>10} {'GPU ARI':>10} {'CPU time':>10} {'GPU time':>10} {'Speedup':>10}")
    print("-" * 78)
    for t in thresholds:
        r = results[t]
        print(
            f"{t:<8} {r['cpu_clusters']:>10} {r['gpu_clusters']:>10} "
            f"{r['cpu_ari']:>10.4f} {r['gpu_ari']:>10.4f} "
            f"{r['cpu_time']:>9.1f}s {r['gpu_time']:>9.1f}s {r['speedup']:>9.1f}x"
        )

    # Compare with old GPU results
    old_gpu_file = OUTPUT_DIR / "pfam_gpu_benchmark_results.json"
    if old_gpu_file.exists():
        with open(old_gpu_file) as f:
            old_data = json.load(f)
        print(f"\n{'='*80}")
        print("COMPARISON — New GPU vs Old GPU")
        print(f"{'='*80}")
        print(f"{'Thresh':<8} {'Old GPU time':>12} {'New GPU time':>12} {'Improvement':>12}")
        print("-" * 48)
        for t in thresholds:
            old_t = old_data.get(str(t), {})
            old_gpu = None
            for key, val in old_t.items():
                if "gpu:0" in key and "error" not in val:
                    old_gpu = val.get("runtime_seconds")
                    break
            if old_gpu:
                new_gpu = results[t]["gpu_time"]
                improvement = old_gpu / new_gpu if new_gpu > 0 else 0
                print(f"{t:<8} {old_gpu:>11.1f}s {new_gpu:>11.1f}s {improvement:>11.1f}x")

    out_file = OUTPUT_DIR / "pfam_gpu_optimized_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
