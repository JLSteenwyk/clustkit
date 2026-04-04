"""Pfam GPU Benchmark: ClustKIT GPU vs CPU vs competitors.

Runs ClustKIT in CPU (8 threads), GPU:0, GPU:1, and multi-GPU (0,1) modes
on the Pfam concordance dataset. Loads existing competitor results for
side-by-side comparison.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.pipeline import run_pipeline

DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "pfam_benchmark_results"
THREADS = 8


def load_and_mix_families(data_dir: Path, max_per_family: int = 500):
    """Load all Pfam family FASTAs and mix into one dataset with ground truth labels."""
    fasta_files = sorted(data_dir.glob("PF*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No Pfam FASTA files found in {data_dir}")

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


def pairwise_precision_recall_f1(true_labels, pred_labels):
    """Compute pairwise precision, recall, and F1."""
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
        idx_a, idx_b = np.array(idx_a), np.array(idx_b)

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
        "pairwise_TP": tp, "pairwise_FP": fp,
        "pairwise_FN": fn, "pairwise_TN": tn,
        "pairwise_precision": round(precision, 4),
        "pairwise_recall": round(recall, 4),
        "pairwise_F1": round(f1, 4),
        "pairwise_accuracy": round(accuracy, 4),
    }


def _normalize_id(seq_id):
    parts = seq_id.split("|")
    return parts[1] if len(parts) >= 2 else seq_id


def evaluate_tool(ground_truth, pred_clusters):
    """Evaluate predicted clusters against ground truth."""
    common_ids = sorted(set(ground_truth.keys()) & set(pred_clusters.keys()))
    if not common_ids:
        gt_norm = {_normalize_id(k): v for k, v in ground_truth.items()}
        pred_norm = {_normalize_id(k): v for k, v in pred_clusters.items()}
        common_ids = sorted(set(gt_norm.keys()) & set(pred_norm.keys()))
        if not common_ids:
            return {"error": "no common sequences"}
        ground_truth, pred_clusters = gt_norm, pred_norm

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


def run_clustkit(fasta, threshold, device, mode_name, mode_cfg, threads):
    """Run ClustKIT and return (clusters_dict, elapsed_seconds)."""
    import numba
    numba.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    out_dir = OUTPUT_DIR / f"gpu_bench_clustkit_{mode_name}_{device.replace(',','_')}_t{threshold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input": str(fasta),
        "output": str(out_dir),
        "threshold": threshold,
        "mode": "protein",
        "alignment": "align",
        "sketch_size": 128,
        "kmer_size": 5,
        "sensitivity": mode_cfg["sensitivity"],
        "cluster_method": mode_cfg["cluster_method"],
        "representative": "longest",
        "device": device,
        "threads": threads,
        "format": "tsv",
        "use_c_ext": True,
        "band_width": mode_cfg["band_width"],
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
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # GPU-only device configs — CPU results are reused from existing file
    # Both GPUs are identical RTX 6000 Ada, so only test one GPU + multi-GPU
    device_configs = [
        ("gpu:0", "0"),
        ("gpu:0,1", "0,1"),
    ]

    # ClustKIT modes — only test default for clean comparison
    clustkit_modes = {
        "default": {"cluster_method": "leiden", "band_width": 100, "sensitivity": "high"},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing competitor results
    competitor_file = OUTPUT_DIR / "pfam_concordance_results_8threads.json"
    competitor_results = {}
    if competitor_file.exists():
        with open(competitor_file) as f:
            competitor_results = json.load(f)
        print(f"Loaded existing competitor results from {competitor_file.name}")
    else:
        print("WARNING: No existing competitor results found. Only ClustKIT will be shown.")

    print()
    print("=" * 120)
    print(f"PFAM GPU BENCHMARK — ClustKIT (CPU vs GPU:0 vs GPU:1 vs GPU:0,1) + competitors")
    print("=" * 120)

    mixed_fasta, ground_truth = load_and_mix_families(DATA_DIR, max_per_family=500)
    print()

    # Warm up GPU kernels with a tiny run
    print("Warming up GPU kernels...", flush=True)
    warmup_dir = OUTPUT_DIR / "gpu_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_pipeline({
            "input": str(mixed_fasta), "output": str(warmup_dir),
            "threshold": 0.9, "mode": "protein", "alignment": "align",
            "sketch_size": 128, "kmer_size": 5, "sensitivity": "medium",
            "cluster_method": "greedy", "representative": "longest",
            "device": "0", "threads": THREADS, "format": "tsv",
            "use_c_ext": True, "band_width": 50, "block": "off", "cascade": "off",
        })
        print("GPU warmup done.\n")
    except Exception as e:
        print(f"GPU warmup failed: {e}\n")

    all_results = {}

    for threshold in thresholds:
        print("-" * 120)
        print(f"Threshold = {threshold}")
        print("-" * 120)

        scenario = {}

        # Run ClustKIT with each device config
        for mode_name, mode_cfg in clustkit_modes.items():
            for dev_label, dev_str in device_configs:
                label = f"clustkit_{mode_name}_{dev_label}"
                print(f"  ClustKIT {mode_name} [{dev_label}]...", end=" ", flush=True)

                try:
                    clusters, elapsed = run_clustkit(
                        mixed_fasta, threshold, dev_str, mode_name, mode_cfg, THREADS,
                    )
                    ev = evaluate_tool(ground_truth, clusters)
                    ev["runtime_seconds"] = round(elapsed, 2)
                    ev["device"] = dev_label
                    scenario[label] = ev
                    print(
                        f"{ev['n_predicted_clusters']} clusters, "
                        f"ARI={ev['ARI']}, F1={ev['pairwise_F1']}, "
                        f"{elapsed:.2f}s"
                    )
                except Exception as e:
                    scenario[label] = {"error": str(e), "device": dev_label}
                    print(f"ERROR: {e}")

        # Include existing CPU + competitor results from prior benchmark
        comp = competitor_results.get(str(threshold), {})
        if "clustkit" in comp:
            cpu_res = comp["clustkit"].copy()
            cpu_res["device"] = "cpu"
            scenario["clustkit_default_cpu"] = cpu_res
            print(
                f"  ClustKIT default [cpu]... (reused) "
                f"{cpu_res.get('n_predicted_clusters')} clusters, "
                f"ARI={cpu_res.get('ARI')}, F1={cpu_res.get('pairwise_F1')}, "
                f"{cpu_res.get('runtime_seconds')}s"
            )
        for tool_key in ["cdhit", "mmseqs2", "linclust", "vsearch", "deepclust"]:
            if tool_key in comp:
                scenario[tool_key] = comp[tool_key]

        all_results[str(threshold)] = scenario
        print()

    # Summary table
    print("=" * 120)
    print("SUMMARY — Pfam Concordance (22K sequences, 56 families)")
    print("=" * 120)
    print()

    tool_order = [
        ("CK-cpu", "clustkit_default_cpu"),
        ("CK-gpu:0", "clustkit_default_gpu:0"),
        ("CK-gpu:0,1", "clustkit_default_gpu:0,1"),
        ("CD-HIT", "cdhit"),
        ("MMseqs2", "mmseqs2"),
        ("Linclust", "linclust"),
        ("VSEARCH", "vsearch"),
        ("DeepClust", "deepclust"),
    ]

    header = f"{'Thresh':<8}"
    for display_name, _ in tool_order:
        header += f" {display_name:>12}"
    print(header)

    # ARI table
    print(f"\n{'--- ARI ---':^120}")
    for t in thresholds:
        row = f"{t:<8}"
        r = all_results.get(str(t), {})
        for _, key in tool_order:
            res = r.get(key, {})
            if "error" in res:
                row += f" {'FAIL':>12}"
            elif "ARI" in res:
                row += f" {res['ARI']:>12.4f}"
            else:
                row += f" {'—':>12}"
        print(row)

    # F1 table
    print(f"\n{'--- Pairwise F1 ---':^120}")
    for t in thresholds:
        row = f"{t:<8}"
        r = all_results.get(str(t), {})
        for _, key in tool_order:
            res = r.get(key, {})
            if "error" in res:
                row += f" {'FAIL':>12}"
            elif "pairwise_F1" in res:
                row += f" {res['pairwise_F1']:>12.4f}"
            else:
                row += f" {'—':>12}"
        print(row)

    # Runtime table
    print(f"\n{'--- Runtime (seconds) ---':^120}")
    for t in thresholds:
        row = f"{t:<8}"
        r = all_results.get(str(t), {})
        for _, key in tool_order:
            res = r.get(key, {})
            if "error" in res:
                row += f" {'FAIL':>12}"
            elif "runtime_seconds" in res:
                row += f" {res['runtime_seconds']:>12.2f}"
            else:
                row += f" {'—':>12}"
        print(row)

    # GPU speedup table
    print(f"\n{'--- GPU Speedup vs CPU ---':^120}")
    for t in thresholds:
        r = all_results.get(str(t), {})
        cpu_res = r.get("clustkit_default_cpu", {})
        cpu_time = cpu_res.get("runtime_seconds")
        row = f"{t:<8}"
        for _, key in tool_order:
            res = r.get(key, {})
            rt = res.get("runtime_seconds")
            if key.startswith("clustkit") and cpu_time and rt and rt > 0:
                speedup = cpu_time / rt
                row += f" {speedup:>11.1f}x"
            else:
                row += f" {'—':>12}"
        print(row)

    # Save results
    results_file = OUTPUT_DIR / "pfam_gpu_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
