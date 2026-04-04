"""Pilot: test reduced band width for clustering SW alignment.

bw=458 (current) vs bw=100 vs bw=50 at t=0.3 with Leiden.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.sketch import compute_sketches
from clustkit.lsh import lsh_candidates
from clustkit.pairwise import compute_pairwise_alignment, BLOSUM62
from clustkit.graph import build_similarity_graph
from clustkit.cluster import cluster_sequences
from clustkit.utils import auto_kmer_for_lsh, auto_lsh_params, logger
import numba, os

DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"
MIXED_FASTA = DATA_DIR.parent / "pfam_mixed.fasta"


def load_gt(max_per_family=500):
    gt = {}
    for f in sorted(DATA_DIR.glob("PF*.fasta")):
        pfam_id = f.stem.split("_")[0]
        count = 0
        for line in open(f):
            if line.startswith(">"):
                sid = line.strip().split()[0][1:]
                if count < max_per_family:
                    gt[sid] = pfam_id
                    count += 1
    return gt


def evaluate(gt, labels, ids):
    gt_n = {k.split("|")[1] if "|" in k else k: v for k, v in gt.items()}
    id_n = [i.split("|")[1] if "|" in i else i for i in ids]
    pred_n = {id_n[i]: int(labels[i]) for i in range(len(labels))}
    common = sorted(set(gt_n.keys()) & set(pred_n.keys()))
    true_l = [gt_n[s] for s in common]
    pred_l = [pred_n[s] for s in common]
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
    return n_clusters, round(ari, 4), round(pr, 4), round(rc, 4), round(f1, 4)


if __name__ == "__main__":
    THREADS = 8
    numba.set_num_threads(THREADS)
    os.environ["OMP_NUM_THREADS"] = str(THREADS)

    threshold = 0.3
    print("Loading data...", flush=True)
    ds = read_sequences(str(MIXED_FASTA), mode="protein")
    gt = load_gt()

    # Phase 1-2: LSH (same for all bw values)
    k = auto_kmer_for_lsh(threshold, "protein", 5)
    sketches = compute_sketches(ds.encoded_sequences, ds.lengths, k=k, sketch_size=128, mode="protein")
    lsh_params = auto_lsh_params(threshold, k)
    candidates = lsh_candidates(sketches, num_tables=lsh_params["num_tables"],
                                num_bands=lsh_params["num_bands"], device="cpu")
    print(f"  {len(candidates)} candidate pairs", flush=True)

    print(f"\n{'bw':>6} {'Clust':>6} {'ARI':>8} {'P':>8} {'R':>8} {'F1':>8} {'Aln(s)':>8} {'Total':>8}")
    print("-" * 70)

    for bw in [458, 200, 100, 50]:
        t0 = time.perf_counter()
        filtered_pairs, similarities = compute_pairwise_alignment(
            candidates, ds.encoded_sequences, ds.lengths, threshold,
            band_width=bw, device="cpu", mode="protein",
            sketches=sketches, flat_sequences=ds.flat_sequences,
            offsets=ds.offsets, use_sw=True, use_c_sw=True, n_threads=THREADS,
        )
        aln_time = time.perf_counter() - t0

        n = len(ds.lengths)
        graph = build_similarity_graph(n, filtered_pairs, similarities)
        labels = cluster_sequences(graph, method="leiden", lengths=ds.lengths)
        total_time = time.perf_counter() - t0

        nc, ari, pr, rc, f1 = evaluate(gt, labels, ds.ids)
        print(f"{bw:>6} {nc:>6} {ari:>8.4f} {pr:>8.4f} {rc:>8.4f} {f1:>8.4f} "
              f"{aln_time:>7.1f}s {total_time:>7.1f}s", flush=True)

    print(f"\nRef: MMseqs2 t=0.3: ARI=0.270, Leiden t=0.3 bw=458: ARI=0.523")
