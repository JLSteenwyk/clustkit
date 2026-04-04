"""Pilot 2: Close the ARI gap at t=0.3 (greedy=0.254 vs MMseqs2=0.270).

The gap is in recall (0.143 vs 0.158). ClustKIT greedy produces 4334 clusters
vs MMseqs2's 4070 — we're slightly under-merging.

Tests:
1. Greedy + merge small clusters into best-connected neighbor
2. Greedy at slightly lower threshold (t=0.28, t=0.25)
3. Leiden at various resolutions
4. Greedy with reassignment of singletons
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.pipeline import run_pipeline
from clustkit.cluster import cluster_greedy, cluster_connected_components

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


def evaluate(gt, labels_or_file, ids=None):
    """Evaluate clustering. Accepts either a file path or (labels, ids) arrays."""
    if isinstance(labels_or_file, (str, Path)):
        pred = {}
        with open(labels_or_file) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                pred[parts[0]] = int(parts[1])
        gt_n = {k.split("|")[1] if "|" in k else k: v for k, v in gt.items()}
        pred_n = {k.split("|")[1] if "|" in k else k: v for k, v in pred.items()}
        common = sorted(set(gt_n.keys()) & set(pred_n.keys()))
        true_l = [gt_n[s] for s in common]
        pred_l = [pred_n[s] for s in common]
    else:
        labels = labels_or_file
        gt_n = {k.split("|")[1] if "|" in k else k: v for k, v in gt.items()}
        id_n = [i.split("|")[1] if "|" in i else i for i in ids]
        pred_n = {id_n[i]: int(labels[i]) for i in range(len(labels))}
        common = sorted(set(gt_n.keys()) & set(pred_n.keys()))
        true_l = [gt_n[s] for s in common]
        pred_l = [pred_n[s] for s in common]

    if not common:
        return {"error": "no common seqs"}
    ari = adjusted_rand_score(true_l, pred_l)
    n_clusters = len(set(pred_l))
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
        "n_clusters": n_clusters, "ARI": round(ari, 4),
        "precision": round(pr, 4), "recall": round(rc, 4), "F1": round(f1, 4),
    }


def greedy_merge_small(graph, lengths, min_cluster_size=3):
    """Greedy clustering then merge small clusters into best-connected larger cluster."""
    labels = cluster_greedy(graph, lengths)
    n = len(labels)

    # Count cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    size_map = dict(zip(unique.tolist(), counts.tolist()))
    small_clusters = {c for c, s in size_map.items() if s < min_cluster_size}

    if not small_clusters:
        return labels

    # For each node in a small cluster, find its best-connected large cluster
    merged = labels.copy()
    for node in range(n):
        if merged[node] not in small_clusters:
            continue
        # Find neighbor clusters and their total edge weight
        neighbors = graph[node].indices
        weights = graph[node].data
        cluster_weights = {}
        for nb, w in zip(neighbors, weights):
            nb_cluster = merged[nb]
            if nb_cluster not in small_clusters:
                cluster_weights[nb_cluster] = cluster_weights.get(nb_cluster, 0) + w
        if cluster_weights:
            best_cluster = max(cluster_weights, key=cluster_weights.get)
            merged[node] = best_cluster

    # Relabel to be contiguous
    _, merged = np.unique(merged, return_inverse=True)
    return merged.astype(np.int32)


def greedy_merge_small_v2(graph, lengths, min_cluster_size=2):
    """Greedy + merge entire small clusters (not just individual nodes)."""
    labels = cluster_greedy(graph, lengths)
    n = len(labels)

    unique, counts = np.unique(labels, return_counts=True)
    size_map = dict(zip(unique.tolist(), counts.tolist()))
    small_clusters = {c for c, s in size_map.items() if s < min_cluster_size}

    if not small_clusters:
        return labels

    merged = labels.copy()
    # Process small clusters: find the large cluster with strongest total connection
    for sc in small_clusters:
        members = np.where(labels == sc)[0]
        cluster_weights = {}
        for node in members:
            neighbors = graph[node].indices
            weights = graph[node].data
            for nb, w in zip(neighbors, weights):
                nb_cluster = merged[nb]
                if nb_cluster not in small_clusters and nb_cluster != sc:
                    cluster_weights[nb_cluster] = cluster_weights.get(nb_cluster, 0) + w
        if cluster_weights:
            best_cluster = max(cluster_weights, key=cluster_weights.get)
            for node in members:
                merged[node] = best_cluster

    _, merged = np.unique(merged, return_inverse=True)
    return merged.astype(np.int32)


def run_pipeline_get_graph(threshold, threads=8):
    """Run pipeline through Phase 4 (graph construction) and return graph + metadata."""
    out_dir = OUTPUT_DIR / f"t03_pilot2_t{threshold}"
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
        "cluster_method": "greedy",  # doesn't matter, we'll re-cluster
        "representative": "longest",
        "device": "cpu",
        "threads": threads,
        "format": "tsv",
        "use_c_ext": True,
    }
    start = time.perf_counter()
    run_pipeline(config)
    elapsed = time.perf_counter() - start
    return out_dir, elapsed


if __name__ == "__main__":
    print("=" * 90)
    print("T=0.3 OPTIMIZATION PILOT 2: Closing the ARI gap")
    print("=" * 90)

    gt = load_ground_truth(DATA_DIR)

    # First, build the graph at t=0.3 (this is the slow part)
    print("\n--- Building graph at t=0.3 ---", flush=True)
    out_dir_03, build_time_03 = run_pipeline_get_graph(0.3)
    print(f"  Build time: {build_time_03:.1f}s", flush=True)

    # Load the graph and dataset
    from clustkit.io import read_sequences as _rs
    from clustkit.graph import build_similarity_graph
    ds = _rs(str(DATA_DIR.parent / "pfam_mixed.fasta"), mode="protein")
    ids = ds.ids
    lengths = ds.lengths

    # Read the clusters.tsv to get the greedy baseline
    r = evaluate(gt, out_dir_03 / "clusters.tsv")
    print(f"\n  Greedy baseline: clusters={r['n_clusters']}, ARI={r['ARI']}, "
          f"P={r['precision']}, R={r['recall']}, F1={r['F1']}", flush=True)

    # Now reload the filtered pairs and similarities to reconstruct the graph
    # Actually, let's just re-run the clustering part with different methods
    # We need the graph. Let's rebuild it from the pipeline's output.
    # The pipeline doesn't save the graph, so we need to re-run Phase 3.
    # Instead, let's just run full pipeline with different cluster methods.

    print(f"\n{'Config':<30} {'Clust':>6} {'ARI':>8} {'P':>8} {'R':>8} {'F1':>8} {'Time':>8}")
    print("-" * 85)

    configs = []

    # Test different thresholds with greedy
    for t in [0.25, 0.28, 0.3]:
        label = f"greedy_t{t}"
        out = OUTPUT_DIR / f"t03_p2_{label}"
        out.mkdir(parents=True, exist_ok=True)
        config = {
            "input": DATA_DIR.parent / "pfam_mixed.fasta",
            "output": out,
            "threshold": t,
            "mode": "protein",
            "alignment": "align",
            "sketch_size": 128,
            "kmer_size": 5,
            "sensitivity": "high",
            "cluster_method": "greedy",
            "representative": "longest",
            "device": "cpu",
            "threads": 8,
            "format": "tsv",
            "use_c_ext": True,
        }
        print(f"Running {label}...", flush=True)
        start = time.perf_counter()
        run_pipeline(config)
        elapsed = time.perf_counter() - start
        r = evaluate(gt, out / "clusters.tsv")
        print(f"{label:<30} {r['n_clusters']:>6} {r['ARI']:>8.4f} {r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} {r['F1']:>8.4f} {elapsed:>7.1f}s", flush=True)
        configs.append((label, r, elapsed))

    # Test Leiden at different resolutions (if available)
    for res in [0.5, 1.0, 2.0, 5.0]:
        try:
            import leidenalg
            label = f"leiden_r{res}_t03"
            out = OUTPUT_DIR / f"t03_p2_{label}"
            out.mkdir(parents=True, exist_ok=True)
            config = {
                "input": DATA_DIR.parent / "pfam_mixed.fasta",
                "output": out,
                "threshold": 0.3,
                "mode": "protein",
                "alignment": "align",
                "sketch_size": 128,
                "kmer_size": 5,
                "sensitivity": "high",
                "cluster_method": "leiden",
                "representative": "longest",
                "device": "cpu",
                "threads": 8,
                "format": "tsv",
                "use_c_ext": True,
            }
            print(f"Running {label}...", flush=True)
            start = time.perf_counter()
            run_pipeline(config)
            elapsed = time.perf_counter() - start
            r = evaluate(gt, out / "clusters.tsv")
            print(f"{label:<30} {r['n_clusters']:>6} {r['ARI']:>8.4f} {r['precision']:>8.4f} "
                  f"{r['recall']:>8.4f} {r['F1']:>8.4f} {elapsed:>7.1f}s", flush=True)
            configs.append((label, r, elapsed))
        except ImportError:
            print(f"  Leiden not available (pip install leidenalg python-igraph)", flush=True)
            break

    print(f"\nReference: MMseqs2 t=0.3: ARI=0.270, P=0.998, R=0.158, F1=0.273, 4070 clusters")
    print(f"Reference: DeepClust t=0.3: ARI=0.261, P=0.980, R=0.153, F1=0.264, 3838 clusters")
