"""Pipeline orchestrator: wires all six phases together."""

import json
import os
import time
from pathlib import Path

import numba
import numpy as np

from clustkit.io import (
    read_sequences,
    write_clusters_cdhit,
    write_clusters_tsv,
    write_representatives_fasta,
)
from clustkit.candidates import clustering_candidates
from clustkit.sketch import compute_sketches
from clustkit.pairwise import compute_pairwise_alignment, compute_pairwise_jaccard
from clustkit.graph import build_similarity_graph, prune_bridge_edges
from clustkit.cluster import cluster_sequences
from clustkit.representatives import select_representatives
from clustkit.utils import auto_kmer_for_lsh, gpu_available, logger, timer


DEFAULT_CASCADE_THRESHOLDS = [0.9, 0.7, 0.5, 0.3]


def _compute_cascade_schedule(target_threshold, config):
    """Determine cascade threshold schedule."""
    custom = config.get("cascade_thresholds")
    if custom:
        schedule = sorted(custom, reverse=True)
        if schedule[-1] != target_threshold:
            schedule.append(target_threshold)
        return schedule

    schedule = [t for t in DEFAULT_CASCADE_THRESHOLDS if t > target_threshold]
    schedule.append(target_threshold)
    return schedule


def _should_cascade(config, threshold, n_sequences):
    """Decide whether to use cascaded clustering."""
    cascade = config.get("cascade", "auto")
    if cascade is True or cascade == "true":
        return True
    if cascade is False or cascade == "false" or cascade == "off":
        return False
    # Auto: cascade when dataset is large enough and threshold is low enough
    return threshold <= 0.5 and n_sequences >= 50000


def _run_single_round(dataset, config, threshold, cluster_method, round_label=""):
    """Run one clustering round (phases 1-6) on a dataset.

    Returns (labels, rep_indices) where rep_indices are local to this dataset.
    """
    n = dataset.num_sequences
    mode = config["mode"]
    k = config["kmer_size"]
    sketch_size = config["sketch_size"]
    sensitivity = config.get("sensitivity", "medium")
    alignment_mode = config.get("alignment", "align")
    device = config.get("device", "cpu")
    n_threads = config["threads"]

    if round_label:
        logger.info(f"  --- {round_label} ({n} sequences) ---")

    # Phase 1: Sketch
    if alignment_mode == "align":
        k_lsh = auto_kmer_for_lsh(threshold, mode, k)
    else:
        k_lsh = k

    sketches = compute_sketches(
        dataset.encoded_sequences, dataset.lengths,
        k_lsh, sketch_size, mode, device=device,
        flat_sequences=dataset.flat_sequences, offsets=dataset.offsets,
    )

    # Phase 2: Candidates (use simple LSH for cascade rounds)
    from clustkit.lsh import lsh_candidates
    from clustkit.utils import auto_lsh_params
    lsh_params = auto_lsh_params(threshold, sensitivity, k=k_lsh)
    candidate_pairs = lsh_candidates(
        sketches, num_tables=lsh_params["num_tables"],
        num_bands=lsh_params["num_bands"], device=device,
    )
    logger.info(f"    {len(candidate_pairs)} candidate pairs")

    if len(candidate_pairs) == 0:
        labels = np.arange(n, dtype=np.int32)
        reps = np.arange(n, dtype=np.int32)
        return labels, reps

    # Phase 3: Pairwise alignment
    if alignment_mode == "kmer":
        kmer_threshold = threshold ** k
        filtered_pairs, similarities = compute_pairwise_jaccard(
            candidate_pairs, sketches, kmer_threshold, device=device,
        )
    else:
        band_width = config.get("band_width", 100)
        use_c_ext = config.get("use_c_ext", True)
        filtered_pairs, similarities = compute_pairwise_alignment(
            candidate_pairs, dataset.encoded_sequences, dataset.lengths,
            threshold, band_width=band_width, device=device, mode=mode,
            sketches=sketches, flat_sequences=dataset.flat_sequences,
            offsets=dataset.offsets, use_sw=True, use_c_sw=use_c_ext,
            n_threads=n_threads,
        )

    logger.info(f"    {len(filtered_pairs)} pairs above threshold {threshold}")

    # Phase 4: Graph
    graph = build_similarity_graph(n, filtered_pairs, similarities)

    # Phase 5: Cluster
    labels = cluster_sequences(graph, method=cluster_method, lengths=dataset.lengths)
    num_clusters = len(np.unique(labels))
    logger.info(f"    {num_clusters} clusters")

    # Phase 6: Representatives
    reps = select_representatives(labels, dataset.lengths, method="longest")

    return labels, reps


def _propagate_labels(chain, n_original):
    """Propagate cluster labels from the final round back to all original sequences.

    chain: list of (labels, reps, original_indices) tuples per round.
    """
    # Start from the final round's labels
    # Build mapping: original_index -> final_cluster_label
    label_of = np.full(n_original, -1, dtype=np.int32)

    # Final round: assign labels directly
    final_labels, final_reps, final_orig_idx = chain[-1]
    for i in range(len(final_orig_idx)):
        label_of[final_orig_idx[i]] = final_labels[i]

    # Walk backwards through earlier rounds
    for round_idx in range(len(chain) - 2, -1, -1):
        labels_r, reps_r, orig_idx_r = chain[round_idx]

        # For each cluster in this round, find its rep's final label
        unique_labels = np.unique(labels_r)
        cluster_final_label = np.full(int(unique_labels.max()) + 1, -1, dtype=np.int32)
        for cl in unique_labels:
            rep_local = reps_r[cl]
            rep_orig = orig_idx_r[rep_local]
            cluster_final_label[cl] = label_of[rep_orig]

        # Assign every sequence in this round the label of its cluster's rep
        for i in range(len(orig_idx_r)):
            label_of[orig_idx_r[i]] = cluster_final_label[labels_r[i]]

    # Relabel to contiguous 0-based
    _, label_of = np.unique(label_of, return_inverse=True)
    return label_of.astype(np.int32)


def run_cascaded_pipeline(config: dict):
    """Execute cascaded multi-round clustering pipeline.

    Clusters in multiple rounds at decreasing thresholds, reducing N
    at each step. Only the final round uses the user's chosen cluster method.
    """
    start_time = time.perf_counter()

    n_threads = config["threads"]
    numba.set_num_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    input_path = Path(config["input"])
    output_dir = Path(config["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = config["threshold"]
    cluster_method = config["cluster_method"]
    rep_method = config["representative"]
    output_format = config["format"]
    mode = config["mode"]

    # Read sequences
    with timer("Phase 0: Reading sequences"):
        dataset = read_sequences(input_path, mode)

    n = dataset.num_sequences
    logger.info(f"  Loaded {n} sequences (max length {dataset.max_length})")

    if n <= 1:
        labels = np.zeros(n, dtype=np.int32)
        reps = np.zeros(n, dtype=np.int32)
        _write_outputs(output_dir, dataset, labels, reps, output_format, config, start_time)
        return

    # Determine cascade schedule
    schedule = _compute_cascade_schedule(threshold, config)
    logger.info(f"  Cascade schedule: {schedule} ({len(schedule)} rounds)")

    # Run rounds
    chain = []  # list of (labels, reps, original_indices)
    current_dataset = dataset
    current_orig_idx = np.arange(n, dtype=np.int64)

    for round_idx, round_threshold in enumerate(schedule):
        is_final = (round_idx == len(schedule) - 1)
        method = cluster_method if is_final else "greedy"

        with timer(f"Cascade round {round_idx + 1}/{len(schedule)} (t={round_threshold})"):
            labels_r, reps_r = _run_single_round(
                current_dataset, config, round_threshold, method,
                round_label=f"Round {round_idx + 1}/{len(schedule)} t={round_threshold}",
            )

        chain.append((labels_r, reps_r, current_orig_idx.copy()))

        num_clusters = len(np.unique(labels_r))
        reduction = 1.0 - num_clusters / len(labels_r)
        logger.info(f"  Round {round_idx + 1}: {len(labels_r)} → {num_clusters} "
                     f"representatives ({reduction:.1%} reduction)")

        if not is_final:
            # Check for insufficient reduction (skip to final if <5% reduction)
            if reduction < 0.05:
                logger.info(f"  Insufficient reduction ({reduction:.1%}), "
                            f"skipping to final round at t={threshold}")
                # Run final round on current dataset
                with timer(f"Cascade final round (t={threshold})"):
                    labels_f, reps_f = _run_single_round(
                        current_dataset, config, threshold, cluster_method,
                        round_label=f"Final round t={threshold}",
                    )
                chain.append((labels_f, reps_f, current_orig_idx.copy()))
                break

            # Extract representative sequences for next round
            rep_orig_idx = current_orig_idx[reps_r]
            current_dataset = dataset.subset(rep_orig_idx)
            current_orig_idx = rep_orig_idx

            # Free memory
            del labels_r, reps_r
            import gc; gc.collect()

    # Propagate labels back to all original sequences
    with timer("Label propagation"):
        final_labels = _propagate_labels(chain, n)

    num_clusters = len(np.unique(final_labels))
    logger.info(f"  Final: {num_clusters} clusters from {n} sequences")

    # Select final representatives
    with timer("Selecting representatives"):
        final_reps = select_representatives(
            final_labels, dataset.lengths, method=rep_method,
        )

    # Write outputs
    _write_outputs(output_dir, dataset, final_labels, final_reps, output_format, config, start_time)

    total_time = time.perf_counter() - start_time
    logger.info(f"Done (cascaded). {num_clusters} clusters from {n} sequences in {total_time:.2f}s")


def _should_block(config, threshold, n_sequences):
    """Decide whether to use block-based clustering."""
    block = config.get("block", "auto")
    if block is True or block == "true":
        return True
    if block is False or block == "false" or block == "off":
        return False
    return threshold <= 0.5 and n_sequences >= 100_000


def _compute_block_size(config, threshold):
    """Determine sequences per block."""
    explicit = config.get("block_size")
    if explicit is not None:
        return int(explicit)
    if threshold <= 0.3:
        return 50_000
    elif threshold <= 0.5:
        return 100_000
    else:
        return 200_000


def _partition_into_blocks(n_sequences, block_size, seed=42):
    """Partition indices into shuffled blocks."""
    indices = np.arange(n_sequences, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    blocks = []
    for start in range(0, n_sequences, block_size):
        blocks.append(indices[start:min(start + block_size, n_sequences)])
    return blocks


def _cluster_single_block(dataset, block_indices, config, threshold, block_id, num_blocks):
    """Cluster a single block, return (block_labels, rep_original_indices)."""
    import gc
    block_ds = dataset.subset(block_indices)
    n_block = len(block_indices)

    schedule = _compute_cascade_schedule(threshold, config)
    chain = []
    current_ds = block_ds
    current_orig_idx = np.arange(n_block, dtype=np.int64)

    for r_idx, r_thresh in enumerate(schedule):
        is_final = (r_idx == len(schedule) - 1)
        method = config.get("cluster_method", "leiden") if is_final else "greedy"

        labels_r, reps_r = _run_single_round(
            current_ds, config, r_thresh, method,
            round_label=f"Block {block_id+1}/{num_blocks} R{r_idx+1} t={r_thresh}",
        )
        chain.append((labels_r, reps_r, current_orig_idx.copy()))

        num_cl = len(np.unique(labels_r))
        reduction = 1.0 - num_cl / len(labels_r)

        if not is_final:
            if reduction < 0.05:
                labels_f, reps_f = _run_single_round(
                    current_ds, config, threshold,
                    config.get("cluster_method", "leiden"),
                    round_label=f"Block {block_id+1}/{num_blocks} final t={threshold}",
                )
                chain.append((labels_f, reps_f, current_orig_idx.copy()))
                break
            rep_local_idx = reps_r
            rep_block_idx = current_orig_idx[rep_local_idx]
            current_ds = block_ds.subset(rep_block_idx)
            current_orig_idx = rep_block_idx

    # Propagate labels within block
    block_labels = _propagate_labels(chain, n_block)

    # Select block representatives (local to block)
    block_reps_local = select_representatives(block_labels, block_ds.lengths, method="longest")

    # Map back to original dataset indices
    block_rep_orig = block_indices[block_reps_local]

    del block_ds, current_ds
    gc.collect()

    return block_labels, block_rep_orig


def _propagate_block_labels(n_original, block_results, all_rep_orig, merge_labels):
    """Map every original sequence to its final cluster label.

    block_results: list of (block_indices, block_labels, block_rep_orig)
    all_rep_orig: combined representative original indices (same order as merge_labels)
    merge_labels: cluster labels from the merge step
    """
    # Build rep_orig -> merge_label lookup
    rep_to_merge = {}
    for i, orig_idx in enumerate(all_rep_orig):
        rep_to_merge[int(orig_idx)] = int(merge_labels[i])

    final_labels = np.full(n_original, -1, dtype=np.int32)

    for block_indices, block_labels, block_rep_orig in block_results:
        # block_rep_orig[cluster_id] = original index of that cluster's representative
        for j, orig_idx in enumerate(block_indices):
            cluster_id = block_labels[j]
            rep_orig = int(block_rep_orig[cluster_id])
            final_labels[orig_idx] = rep_to_merge[rep_orig]

    # Relabel to contiguous 0-based
    _, final_labels = np.unique(final_labels, return_inverse=True)
    return final_labels.astype(np.int32)


def run_block_cascaded_pipeline(config: dict):
    """Execute block-based cascaded clustering for very large datasets.

    Splits dataset into blocks, clusters each independently, then merges
    block representatives in a final clustering pass.
    """
    import gc
    start_time = time.perf_counter()

    n_threads = config["threads"]
    numba.set_num_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    input_path = Path(config["input"])
    output_dir = Path(config["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = config["threshold"]
    cluster_method = config["cluster_method"]
    rep_method = config["representative"]
    output_format = config["format"]
    mode = config["mode"]

    with timer("Phase 0: Reading sequences"):
        dataset = read_sequences(input_path, mode)

    n = dataset.num_sequences
    logger.info(f"  Loaded {n} sequences (max length {dataset.max_length})")

    block_size = _compute_block_size(config, threshold)
    blocks = _partition_into_blocks(n, block_size)
    num_blocks = len(blocks)
    logger.info(f"  Block-based clustering: {num_blocks} blocks of ~{block_size}")

    # Process blocks sequentially
    block_results = []
    total_reps = 0
    for block_id, block_indices in enumerate(blocks):
        with timer(f"Block {block_id + 1}/{num_blocks} ({len(block_indices)} seqs)"):
            block_labels, block_rep_orig = _cluster_single_block(
                dataset, block_indices, config, threshold, block_id, num_blocks,
            )
        n_reps = len(block_rep_orig)
        total_reps += n_reps
        logger.info(f"  Block {block_id + 1}: {len(block_indices)} → {n_reps} reps")
        block_results.append((block_indices, block_labels, block_rep_orig))
        gc.collect()

    # Collect all block representatives
    all_rep_orig = np.concatenate([r[2] for r in block_results])
    all_rep_orig = np.unique(all_rep_orig)
    logger.info(f"  Total representatives: {len(all_rep_orig)} from {n} sequences")

    # Merge step: cluster combined representatives
    with timer("Merge clustering"):
        merge_ds = dataset.subset(all_rep_orig)
        n_merge = len(all_rep_orig)
        logger.info(f"  Merge dataset: {n_merge} representatives")

        if n_merge > 200_000:
            # Recursive: block the merge set too
            logger.info(f"  Merge set too large ({n_merge}), applying block-based merge")
            merge_config = {**config, "input": "__internal__", "output": str(output_dir / "merge_tmp")}
            # For now, just run cascaded on the merge set
            schedule = _compute_cascade_schedule(threshold, merge_config)
        else:
            schedule = _compute_cascade_schedule(threshold, config)

        # Run cascaded clustering on merge set
        chain = []
        current_ds = merge_ds
        current_orig_idx = np.arange(n_merge, dtype=np.int64)

        for r_idx, r_thresh in enumerate(schedule):
            is_final = (r_idx == len(schedule) - 1)
            method = cluster_method if is_final else "greedy"

            labels_r, reps_r = _run_single_round(
                current_ds, config, r_thresh, method,
                round_label=f"Merge R{r_idx+1}/{len(schedule)} t={r_thresh}",
            )
            chain.append((labels_r, reps_r, current_orig_idx.copy()))
            num_cl = len(np.unique(labels_r))
            reduction = 1.0 - num_cl / len(labels_r) if len(labels_r) > 0 else 0

            if not is_final:
                if reduction < 0.05:
                    labels_f, reps_f = _run_single_round(
                        current_ds, config, threshold, cluster_method,
                        round_label=f"Merge final t={threshold}",
                    )
                    chain.append((labels_f, reps_f, current_orig_idx.copy()))
                    break
                rep_idx = current_orig_idx[reps_r]
                current_ds = merge_ds.subset(rep_idx)
                current_orig_idx = rep_idx

        merge_labels = _propagate_labels(chain, n_merge)

    # Propagate labels to all original sequences
    with timer("Block label propagation"):
        final_labels = _propagate_block_labels(n, block_results, all_rep_orig, merge_labels)

    num_clusters = len(np.unique(final_labels))
    logger.info(f"  Final: {num_clusters} clusters from {n} sequences")

    with timer("Selecting representatives"):
        final_reps = select_representatives(
            final_labels, dataset.lengths, method=rep_method,
        )

    _write_outputs(output_dir, dataset, final_labels, final_reps, output_format, config, start_time)

    total_time = time.perf_counter() - start_time
    logger.info(f"Done (block-cascaded). {num_clusters} clusters from {n} sequences in {total_time:.2f}s")


def run_pipeline(config: dict):
    """Execute the full ClustKIT clustering pipeline.

    Automatically uses block-based or cascaded clustering for large datasets.

    Args:
        config: Dictionary of configuration parameters from the CLI.
    """
    start_time = time.perf_counter()

    # Configure Numba and OpenMP thread count to match user request
    n_threads = config["threads"]
    numba.set_num_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    logger.info(f"Numba threads: {numba.get_num_threads()}")

    # Quick count sequences to decide on strategy
    input_path = Path(config["input"])
    n_seqs = sum(1 for line in open(input_path) if line.startswith(">"))

    # Block-based for very large datasets
    if _should_block(config, config["threshold"], n_seqs):
        logger.info(f"  Using block-based cascaded clustering ({n_seqs} seqs, t={config['threshold']})")
        return run_block_cascaded_pipeline(config)

    # Cascaded for medium datasets at low thresholds
    if _should_cascade(config, config["threshold"], n_seqs):
        logger.info(f"  Using cascaded clustering ({n_seqs} seqs, t={config['threshold']})")
        return run_cascaded_pipeline(config)

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
    alignment_mode = config.get("alignment", "align")
    device = config.get("device", "cpu")
    candidate_strategy = config.get("candidate_strategy", "lsh")
    augment_min_candidates = config.get("augment_min_candidates", 64)
    augment_max_sequences = config.get("augment_max_sequences", 4096)
    augment_max_cands_per_query = config.get("augment_max_cands_per_query", 256)
    augment_phase_a_topk = config.get("augment_phase_a_topk", 4000)
    augment_freq_percentile = config.get("augment_freq_percentile", 99.0)
    augment_min_total_hits = config.get("augment_min_total_hits", 2)
    augment_min_diag_hits = config.get("augment_min_diag_hits", 2)
    augment_diag_bin_width = config.get("augment_diag_bin_width", 10)
    candidate_reduced_k = config.get("candidate_reduced_k", 5)
    candidate_spaced_seed = config.get("candidate_spaced_seed", "110011")
    candidate_spaced_max_query_length = config.get("candidate_spaced_max_query_length", 2048)
    graph_prune_similarity = config.get("graph_prune_similarity", 0.0)
    graph_prune_shared_neighbors = config.get("graph_prune_shared_neighbors", 0)
    graph_prune_min_degree = config.get("graph_prune_min_degree", 0)
    refine_global_margin = config.get("refine_global_margin", 0.0)

    # --- Device selection ---
    if device not in ("cpu", "auto"):
        if gpu_available():
            logger.info(f"GPU mode enabled (device {device})")
        else:
            logger.warning(
                f"GPU device '{device}' requested but CuPy/CUDA not available. "
                "Falling back to CPU."
            )
            device = "cpu"
    elif device == "auto":
        if gpu_available():
            logger.info("Auto device selection enabled (will calibrate at alignment phase)")
        else:
            device = "cpu"

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
    # In alignment mode, use adaptive k for better LSH recall at low identity.
    # Smaller k gives higher k-mer Jaccard, making LSH more sensitive.
    # In kmer mode, k must match between sketch and pairwise stages.
    if alignment_mode == "align":
        k_lsh = auto_kmer_for_lsh(threshold, mode, k)
    else:
        k_lsh = k

    if k_lsh != k:
        logger.info(f"  Adaptive k: using k={k_lsh} for LSH (threshold={threshold})")

    with timer("Phase 1: Computing sketches"):
        sketches = compute_sketches(
            dataset.encoded_sequences,
            dataset.lengths,
            k_lsh,
            sketch_size,
            mode,
            device=device,
            flat_sequences=dataset.flat_sequences,
            offsets=dataset.offsets,
        )

    # --- Phase 2: Candidate generation ---
    candidate_pairs = clustering_candidates(
        dataset,
        sketches,
        threshold=threshold,
        sensitivity=sensitivity,
        mode=mode,
        k_lsh=k_lsh,
        device=device,
        strategy=candidate_strategy,
        augment_min_candidates=augment_min_candidates,
        augment_max_sequences=augment_max_sequences,
        augment_max_cands_per_query=augment_max_cands_per_query,
        augment_phase_a_topk=augment_phase_a_topk,
        augment_freq_percentile=augment_freq_percentile,
        augment_min_total_hits=augment_min_total_hits,
        augment_min_diag_hits=augment_min_diag_hits,
        augment_diag_bin_width=augment_diag_bin_width,
        reduced_k=candidate_reduced_k,
        spaced_seed=candidate_spaced_seed,
        spaced_max_query_length=candidate_spaced_max_query_length,
    )

    logger.info(f"  Found {len(candidate_pairs)} candidate pairs")

    # --- Phase 3: Pairwise similarity ---
    if alignment_mode == "kmer":
        # K-mer Jaccard mode: convert identity threshold to k-mer space
        kmer_threshold = threshold ** k
        logger.info(
            f"  Mode: k-mer Jaccard | identity {threshold} → "
            f"k-mer threshold {kmer_threshold:.4f} (k={k})"
        )
        with timer("Phase 3: Pairwise similarity (k-mer Jaccard)"):
            filtered_pairs, similarities = compute_pairwise_jaccard(
                candidate_pairs, sketches, kmer_threshold,
                device=device,
            )
    else:
        # Alignment mode (default): SW local alignment with C extension
        # SW local alignment only needs to cover the local alignment region,
        # not the full sequence length difference. bw=100 is the sweet spot:
        # ARI=0.487 (1.8x MMseqs2) at 2.9x speedup vs bw=458.
        band_width = config.get("band_width", 100)

        # Auto-detect C SW extension
        use_c_ext = config.get("use_c_ext", True)
        aln_label = "SW local alignment"
        if use_c_ext:
            aln_label += " (C/OpenMP)"
        logger.info(
            f"  Mode: {aln_label} | threshold {threshold} | "
            f"band_width {band_width}"
        )

        with timer(f"Phase 3: Pairwise similarity ({aln_label})"):
            filtered_pairs, similarities = compute_pairwise_alignment(
                candidate_pairs,
                dataset.encoded_sequences,
                dataset.lengths,
                threshold,
                band_width=band_width,
                device=device,
                mode=mode,
                sketches=sketches,
                flat_sequences=dataset.flat_sequences,
                offsets=dataset.offsets,
                use_sw=True,
                use_c_sw=use_c_ext,
                n_threads=n_threads,
                refine_global_margin=refine_global_margin,
            )

    logger.info(f"  {len(filtered_pairs)} pairs above threshold {threshold}")

    # --- Phase 4: Graph construction ---
    with timer("Phase 4: Building similarity graph"):
        graph = build_similarity_graph(n, filtered_pairs, similarities)
        if graph_prune_shared_neighbors > 0 and graph_prune_similarity > 0:
            graph = prune_bridge_edges(
                graph,
                weak_similarity_threshold=graph_prune_similarity,
                min_shared_neighbors=graph_prune_shared_neighbors,
                min_endpoint_degree=graph_prune_min_degree,
            )

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
        reps = select_representatives(
            labels,
            dataset.lengths,
            method=rep_method,
            graph=graph if rep_method in ("centroid", "most_connected") else None,
        )

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
