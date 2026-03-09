"""Pipeline orchestrator: wires all six phases together."""

import json
import time
from pathlib import Path

import numpy as np

from clustkit.io import (
    read_sequences,
    write_clusters_cdhit,
    write_clusters_tsv,
    write_representatives_fasta,
)
from clustkit.sketch import compute_sketches
from clustkit.lsh import lsh_candidates
from clustkit.pairwise import compute_pairwise_jaccard
from clustkit.graph import build_similarity_graph
from clustkit.cluster import cluster_sequences
from clustkit.representatives import select_representatives
from clustkit.utils import auto_lsh_params, logger, timer


def run_pipeline(config: dict):
    """Execute the full ClustKIT clustering pipeline.

    Args:
        config: Dictionary of configuration parameters from the CLI.
    """
    start_time = time.perf_counter()

    input_path = Path(config["input"])
    output_dir = Path(config["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = config["threshold"]
    mode = config["mode"]
    k = config["kmer_size"]
    sketch_size = config["sketch_size"]
    sensitivity = config["sensitivity"]
    cluster_method = config["cluster_method"]
    rep_method = config["representative"]
    output_format = config["format"]

    # --- Phase 0: Read sequences ---
    with timer("Phase 0: Reading sequences"):
        dataset = read_sequences(input_path, mode)

    n = dataset.num_sequences
    logger.info(f"  Loaded {n} sequences (max length {dataset.max_length})")

    if n == 0:
        logger.warning("No sequences found. Nothing to cluster.")
        return

    # Handle trivial case: 1 sequence
    if n == 1:
        labels = np.array([0], dtype=np.int32)
        reps = np.array([0], dtype=np.int32)
        _write_outputs(output_dir, dataset, labels, reps, output_format, config, start_time)
        return

    # --- Phase 1: Sketch ---
    with timer("Phase 1: Computing sketches"):
        sketches = compute_sketches(
            dataset.encoded_sequences,
            dataset.lengths,
            k,
            sketch_size,
            mode,
        )

    # --- Phase 2: LSH Bucketing ---
    lsh_params = auto_lsh_params(threshold, sensitivity)
    logger.info(
        f"  LSH params: {lsh_params['num_tables']} tables, "
        f"{lsh_params['num_bands']} bands/table"
    )

    with timer("Phase 2: LSH candidate generation"):
        candidate_pairs = lsh_candidates(
            sketches,
            num_tables=lsh_params["num_tables"],
            num_bands=lsh_params["num_bands"],
        )

    logger.info(f"  Found {len(candidate_pairs)} candidate pairs")

    # --- Phase 3: Pairwise similarity ---
    with timer("Phase 3: Pairwise similarity"):
        filtered_pairs, similarities = compute_pairwise_jaccard(
            candidate_pairs, sketches, threshold
        )

    logger.info(
        f"  {len(filtered_pairs)} pairs above threshold {threshold}"
    )

    # --- Phase 4: Graph construction ---
    with timer("Phase 4: Building similarity graph"):
        graph = build_similarity_graph(n, filtered_pairs, similarities)

    logger.info(f"  Graph: {n} nodes, {graph.nnz // 2} edges")

    # --- Phase 5: Clustering ---
    with timer("Phase 5: Clustering"):
        labels = cluster_sequences(
            graph, method=cluster_method, lengths=dataset.lengths
        )

    num_clusters = len(np.unique(labels))
    logger.info(f"  {num_clusters} clusters")

    # --- Phase 6: Representative selection ---
    with timer("Phase 6: Selecting representatives"):
        reps = select_representatives(labels, dataset.lengths, method=rep_method)

    # --- Write outputs ---
    _write_outputs(output_dir, dataset, labels, reps, output_format, config, start_time)

    total_time = time.perf_counter() - start_time
    logger.info(f"Done. {num_clusters} clusters from {n} sequences in {total_time:.2f}s")


def _write_outputs(
    output_dir: Path,
    dataset,
    labels: np.ndarray,
    reps: np.ndarray,
    output_format: str,
    config: dict,
    start_time: float,
):
    """Write all output files."""
    with timer("Writing output files"):
        write_representatives_fasta(
            output_dir / "representatives.fasta", dataset, reps
        )

        if output_format == "cdhit":
            write_clusters_cdhit(
                output_dir / "clusters.clstr", dataset, labels, reps
            )
        else:
            write_clusters_tsv(
                output_dir / "clusters.tsv", dataset, labels, reps
            )

        # Write run info
        run_info = {
            "version": "0.1.0",
            "parameters": {
                k: str(v) for k, v in config.items()
            },
            "num_sequences": dataset.num_sequences,
            "num_clusters": int(len(np.unique(labels))),
            "runtime_seconds": round(time.perf_counter() - start_time, 2),
        }
        with open(output_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)
