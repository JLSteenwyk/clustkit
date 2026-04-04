"""Publication-quality Pfam concordance benchmark.

133K sequences, 1642 families, 418 clans from SwissProt × Pfam-A.
Evaluates both family-level and clan-level clustering quality.

Tools: ClustKIT (Leiden), MMseqs2, DeepClust, CD-HIT, Linclust, VSEARCH
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.pipeline import run_pipeline

DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_full"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "pfam_large_results"

CDHIT_SIF = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/cd-hit.sif"
MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
VSEARCH_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/VSEARCH/vsearch-2.30.5-linux-x86_64/bin/vsearch"
DIAMOND_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/diamond-linux64/diamond"


# ── Ground truth loading ─────────────────────────────────────────────

def load_ground_truth():
    """Load ground truth: protein→family and family→clan mappings."""
    with open(DATA_DIR / "ground_truth.json") as f:
        gt = json.load(f)
    return gt["protein_family"], gt["family_clan"]


def _normalize_id(seq_id):
    """Extract UniProt accession from sp|ACC|NAME format."""
    parts = seq_id.split("|")
    return parts[1] if len(parts) >= 2 else seq_id


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_clustering(protein_family, family_clan, pred_clusters):
    """Evaluate predicted clusters at both family and clan level.

    Args:
        protein_family: dict mapping protein_id → pfam_family
        family_clan: dict mapping pfam_family → clan_id
        pred_clusters: dict mapping protein_id → cluster_id

    Returns:
        dict with family-level and clan-level metrics.
    """
    # Normalize IDs
    gt_norm = {_normalize_id(k): v for k, v in protein_family.items()}
    pred_norm = {_normalize_id(k): v for k, v in pred_clusters.items()}
    common = sorted(set(gt_norm.keys()) & set(pred_norm.keys()))

    if not common:
        return {"error": "no common sequences", "n_common": 0}

    true_families = [gt_norm[s] for s in common]
    pred_labels = [pred_norm[s] for s in common]

    # Map families to clans (families without clan get their own "clan")
    true_clans = [family_clan.get(f, f"noclan_{f}") for f in true_families]

    # Integer-encode for sklearn
    fam_to_int = {f: i for i, f in enumerate(sorted(set(true_families)))}
    clan_to_int = {c: i for i, c in enumerate(sorted(set(true_clans)))}

    true_fam_arr = np.array([fam_to_int[f] for f in true_families], dtype=np.int32)
    true_clan_arr = np.array([clan_to_int[c] for c in true_clans], dtype=np.int32)
    pred_arr = np.array(pred_labels, dtype=np.int32)

    # Family-level metrics
    fam_ari = adjusted_rand_score(true_fam_arr, pred_arr)
    fam_nmi = normalized_mutual_info_score(true_fam_arr, pred_arr)

    # Clan-level metrics (are sequences in the same clan clustered together?)
    clan_ari = adjusted_rand_score(true_clan_arr, pred_arr)
    clan_nmi = normalized_mutual_info_score(true_clan_arr, pred_arr)

    # Pairwise precision/recall (sampled)
    n = len(common)
    rng = np.random.RandomState(42)
    num_samples = min(2_000_000, n * (n - 1) // 2)
    ia = rng.randint(0, n, size=num_samples)
    ib = rng.randint(0, n, size=num_samples)
    valid = ia != ib
    ia, ib = ia[valid], ib[valid]

    # Family-level pairwise
    same_pred = pred_arr[ia] == pred_arr[ib]
    same_fam = true_fam_arr[ia] == true_fam_arr[ib]
    fam_tp = int(np.sum(same_pred & same_fam))
    fam_fp = int(np.sum(same_pred & ~same_fam))
    fam_fn = int(np.sum(~same_pred & same_fam))
    fam_prec = fam_tp / (fam_tp + fam_fp) if (fam_tp + fam_fp) else 0
    fam_rec = fam_tp / (fam_tp + fam_fn) if (fam_tp + fam_fn) else 0
    fam_f1 = 2 * fam_prec * fam_rec / (fam_prec + fam_rec) if (fam_prec + fam_rec) else 0

    # Clan-level pairwise
    same_clan = true_clan_arr[ia] == true_clan_arr[ib]
    clan_tp = int(np.sum(same_pred & same_clan))
    clan_fp = int(np.sum(same_pred & ~same_clan))
    clan_fn = int(np.sum(~same_pred & same_clan))
    clan_prec = clan_tp / (clan_tp + clan_fp) if (clan_tp + clan_fp) else 0
    clan_rec = clan_tp / (clan_tp + clan_fn) if (clan_tp + clan_fn) else 0
    clan_f1 = 2 * clan_prec * clan_rec / (clan_prec + clan_rec) if (clan_prec + clan_rec) else 0

    return {
        "n_common": len(common),
        "n_predicted_clusters": len(set(pred_labels)),
        "n_true_families": len(set(true_families)),
        "n_true_clans": len(set(true_clans)),
        # Family-level
        "family_ARI": round(fam_ari, 4),
        "family_NMI": round(fam_nmi, 4),
        "family_precision": round(fam_prec, 4),
        "family_recall": round(fam_rec, 4),
        "family_F1": round(fam_f1, 4),
        # Clan-level
        "clan_ARI": round(clan_ari, 4),
        "clan_NMI": round(clan_nmi, 4),
        "clan_precision": round(clan_prec, 4),
        "clan_recall": round(clan_rec, 4),
        "clan_F1": round(clan_f1, 4),
    }


# ── Tool runners ─────────────────────────────────────────────────────

def parse_mmseqs_clusters(tsv_path):
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


def parse_cdhit_clusters(clstr_path):
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


def parse_vsearch_clusters(uc_path):
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


CLUSTKIT_MODES = {
    "fast": {"cluster_method": "greedy", "band_width": 50, "sensitivity": "medium"},
    "default": {"cluster_method": "leiden", "band_width": 100, "sensitivity": "high"},
    "sensitive": {"cluster_method": "leiden", "band_width": 200, "sensitivity": "high"},
}


def run_clustkit_mode(fasta_path, threshold, threads, mode_name="default"):
    mode_cfg = CLUSTKIT_MODES[mode_name]
    out_dir = OUTPUT_DIR / f"clustkit_{mode_name}_t{threshold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "input": fasta_path,
        "output": out_dir,
        "threshold": threshold,
        "mode": "protein",
        "alignment": "align",
        "sketch_size": 128,
        "kmer_size": 5,
        "sensitivity": mode_cfg["sensitivity"],
        "cluster_method": mode_cfg["cluster_method"],
        "representative": "longest",
        "device": "cpu",
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


def run_mmseqs(fasta_path, threshold, threads):
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_tmp_")
    prefix = OUTPUT_DIR / f"mmseqs_t{threshold}"
    cmd = [
        MMSEQS_BIN, "easy-cluster", str(fasta_path), str(prefix), tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(threads),
        "-c", "0.8", "--cov-mode", "0",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if result.returncode != 0:
        return None, elapsed
    return parse_mmseqs_clusters(str(prefix) + "_cluster.tsv"), elapsed


def run_linclust(fasta_path, threshold, threads):
    tmp_dir = tempfile.mkdtemp(prefix="linclust_tmp_")
    prefix = OUTPUT_DIR / f"linclust_t{threshold}"
    cmd = [
        MMSEQS_BIN, "easy-linclust", str(fasta_path), str(prefix), tmp_dir,
        "--min-seq-id", str(threshold), "--threads", str(threads),
        "-c", "0.8", "--cov-mode", "0",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if result.returncode != 0:
        return None, elapsed
    return parse_mmseqs_clusters(str(prefix) + "_cluster.tsv"), elapsed


def run_deepclust(fasta_path, threshold, threads):
    out_file = str(OUTPUT_DIR / f"deepclust_t{threshold}.tsv")
    approx_id = int(threshold * 100)
    cmd = [
        DIAMOND_BIN, "deepclust", "-d", str(fasta_path), "-o", out_file,
        "--approx-id", str(approx_id), "--member-cover", "80",
        "--threads", str(threads), "-M", "64G",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        return None, elapsed
    return parse_mmseqs_clusters(out_file), elapsed


def run_cdhit(fasta_path, threshold, threads):
    prefix = OUTPUT_DIR / f"cdhit_t{threshold}"
    n = 5 if threshold >= 0.7 else (4 if threshold >= 0.6 else (3 if threshold >= 0.5 else 2))
    cmd = [
        "singularity", "exec", CDHIT_SIF, "cd-hit",
        "-i", str(fasta_path), "-o", str(prefix),
        "-c", str(threshold), "-T", str(threads), "-M", "0", "-d", "0", "-n", str(n),
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        return None, elapsed
    return parse_cdhit_clusters(str(prefix) + ".clstr"), elapsed


def run_vsearch(fasta_path, threshold, threads):
    uc_path = str(OUTPUT_DIR / f"vsearch_t{threshold}.uc")
    cmd = [
        VSEARCH_BIN, "--cluster_fast", str(fasta_path),
        "--id", str(threshold), "--uc", uc_path, "--threads", str(threads),
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        return None, elapsed
    return parse_vsearch_clusters(uc_path), elapsed


# ── Main benchmark ────────────────────────────────────────────────────

def run_benchmark(thresholds=None, threads=8):
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 0.9]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fasta = DATA_DIR / "pfam_benchmark_large.fasta"
    protein_family, family_clan = load_ground_truth()

    print("=" * 130)
    print(f"PUBLICATION-QUALITY PFAM BENCHMARK — 133K seqs, 1642 families, 418 clans ({threads} threads)")
    print("=" * 130)
    print(f"Dataset: {fasta}")
    print(f"Ground truth: {len(protein_family)} proteins, {len(set(protein_family.values()))} families, "
          f"{len(set(family_clan.values()))} clans\n")

    tools = [
        ("ClustKIT (fast)", lambda f, t, th: run_clustkit_mode(f, t, th, "fast")),
        ("ClustKIT (default)", lambda f, t, th: run_clustkit_mode(f, t, th, "default")),
        ("ClustKIT (sensitive)", lambda f, t, th: run_clustkit_mode(f, t, th, "sensitive")),
        ("MMseqs2", run_mmseqs),
        ("Linclust", run_linclust),
        ("DeepClust", run_deepclust),
        ("CD-HIT", run_cdhit),
        ("VSEARCH", run_vsearch),
    ]

    all_results = {}

    for threshold in thresholds:
        print("-" * 130)
        print(f"Threshold = {threshold}")
        print("-" * 130)

        scenario = {}
        for tool_name, runner in tools:
            print(f"  {tool_name}...", end=" ", flush=True)
            try:
                if tool_name == "ClustKIT":
                    clusters, elapsed = runner(fasta, threshold, threads)
                else:
                    clusters, elapsed = runner(fasta, threshold, threads)

                if clusters is None:
                    print(f"FAILED ({elapsed:.1f}s)")
                    scenario[tool_name] = {"error": "failed", "runtime": round(elapsed, 2)}
                    continue

                metrics = evaluate_clustering(protein_family, family_clan, clusters)
                metrics["runtime"] = round(elapsed, 2)
                scenario[tool_name] = metrics

                if "error" not in metrics:
                    print(f"{metrics['n_predicted_clusters']} clusters | "
                          f"Fam ARI={metrics['family_ARI']:.3f} P={metrics['family_precision']:.3f} R={metrics['family_recall']:.3f} | "
                          f"Clan ARI={metrics['clan_ARI']:.3f} P={metrics['clan_precision']:.3f} R={metrics['clan_recall']:.3f} | "
                          f"{elapsed:.1f}s")
                else:
                    print(f"EVAL ERROR ({elapsed:.1f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                scenario[tool_name] = {"error": str(e)}

        all_results[str(threshold)] = scenario
        print()

    # Summary table
    print("=" * 130)
    print("SUMMARY — FAMILY-LEVEL")
    print("=" * 130)
    print(f"{'Thresh':<8} {'Tool':<12} {'Clust':>7} {'ARI':>8} {'NMI':>8} {'P':>8} {'R':>8} {'F1':>8} {'Time':>10}")
    print("-" * 85)
    for t in thresholds:
        for tool_name, _ in tools:
            r = all_results[str(t)].get(tool_name, {})
            if "error" in r:
                print(f"{t:<8} {tool_name:<12} {'FAIL':>7} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {r.get('runtime','?'):>9}s")
            elif r:
                print(f"{t:<8} {tool_name:<12} {r['n_predicted_clusters']:>7} {r['family_ARI']:>8.4f} "
                      f"{r['family_NMI']:>8.4f} {r['family_precision']:>8.4f} {r['family_recall']:>8.4f} "
                      f"{r['family_F1']:>8.4f} {r['runtime']:>9.1f}s")
        print()

    print("=" * 130)
    print("SUMMARY — CLAN-LEVEL")
    print("=" * 130)
    print(f"{'Thresh':<8} {'Tool':<12} {'ARI':>8} {'NMI':>8} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 60)
    for t in thresholds:
        for tool_name, _ in tools:
            r = all_results[str(t)].get(tool_name, {})
            if "error" not in r and r:
                print(f"{t:<8} {tool_name:<12} {r['clan_ARI']:>8.4f} {r['clan_NMI']:>8.4f} "
                      f"{r['clan_precision']:>8.4f} {r['clan_recall']:>8.4f} {r['clan_F1']:>8.4f}")
        print()

    # Save
    results_file = OUTPUT_DIR / "pfam_large_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    args = parser.parse_args()
    run_benchmark(thresholds=args.thresholds, threads=args.threads)
