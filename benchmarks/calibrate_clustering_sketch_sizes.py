"""Sweep clustering quality across sketch sizes on the smaller Pfam subset."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_pfam_concordance import load_and_mix_families, pairwise_precision_recall_f1
from clustkit.candidates import clustering_candidates
from clustkit.cluster import cluster_sequences
from clustkit.graph import build_similarity_graph
from clustkit.io import read_sequences
from clustkit.pairwise import compute_pairwise_alignment
from clustkit.sketch import compute_sketches
from clustkit.utils import auto_kmer_for_lsh


def main():
    data_dir = ROOT / "benchmarks" / "data" / "pfam_families"
    mixed_fasta, ground_truth = load_and_mix_families(data_dir, 200)

    truth_order = []
    with open(mixed_fasta) as f:
        for line in f:
            if line.startswith(">"):
                seq_id = line[1:].strip().split()[0]
                truth_order.append(ground_truth[seq_id])

    label_to_int = {fam: i for i, fam in enumerate(sorted(set(truth_order)))}
    true_labels = np.array([label_to_int[fam] for fam in truth_order], dtype=np.int32)
    dataset = read_sequences(mixed_fasta, "protein")

    results = []
    for threshold in (0.3, 0.4, 0.5):
        for sketch_size in (64, 128, 256):
            t0 = time.perf_counter()
            k_lsh = auto_kmer_for_lsh(threshold, "protein", 8)
            sketches = compute_sketches(
                dataset.encoded_sequences,
                dataset.lengths,
                k_lsh,
                sketch_size,
                "protein",
                device="cpu",
                flat_sequences=dataset.flat_sequences,
                offsets=dataset.offsets,
            )
            candidate_pairs = clustering_candidates(
                dataset,
                sketches,
                threshold=threshold,
                sensitivity="medium",
                mode="protein",
                k_lsh=k_lsh,
                device="cpu",
                strategy="lsh",
            )
            filtered_pairs, similarities = compute_pairwise_alignment(
                candidate_pairs,
                dataset.encoded_sequences,
                dataset.lengths,
                threshold,
                band_width=100,
                device="cpu",
                mode="protein",
                sketches=sketches,
                flat_sequences=dataset.flat_sequences,
                offsets=dataset.offsets,
                use_sw=True,
                use_c_sw=True,
                n_threads=32,
                refine_global_margin=0.0,
            )
            graph = build_similarity_graph(
                dataset.num_sequences, filtered_pairs, similarities
            )
            labels = cluster_sequences(graph, method="leiden", lengths=dataset.lengths)
            ari = adjusted_rand_score(true_labels, labels)
            pw = pairwise_precision_recall_f1(true_labels, labels)
            row = {
                "threshold": threshold,
                "sketch_size": sketch_size,
                "k_lsh": int(k_lsh),
                "candidates": int(len(candidate_pairs)),
                "accepted_edges": int(len(filtered_pairs)),
                "clusters": int(len(np.unique(labels))),
                "ARI": round(float(ari), 4),
                "precision": pw["pairwise_precision"],
                "recall": pw["pairwise_recall"],
                "F1": pw["pairwise_F1"],
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }
            results.append(row)
            print(json.dumps(row), flush=True)

    out_path = ROOT / "tmp" / "calibrate_clustering_sketch_sizes_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"results_path": str(out_path), "rows": len(results)}), flush=True)


if __name__ == "__main__":
    main()
