"""Annotation consistency and distance distribution benchmarks.

CP2: GO annotation consistency (Linclust Fig 3a)
CP3: Pfam annotation consistency (Linclust Fig 3b)
CP4: Cumulative representative distance distribution (Linclust Fig 4)

These experiments measure clustering quality using external annotations,
inspired by the Linclust paper (Steinegger & Söding, s41467-018-04964-5).

Usage:
    python benchmark_annotation_consistency.py \
        --fasta swissprot.fasta \
        --goa-file goa_uniprot.gaf \
        --pfam-file interpro_pfam.tsv \
        --thresholds 0.3 0.5 0.7 0.9 \
        --threads 4

    # Run only CP4 (distance distribution) at a single threshold:
    python benchmark_annotation_consistency.py \
        --fasta swissprot.fasta \
        --experiments cp4 \
        --thresholds 0.5 \
        --threads 4
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.pairwise import _nw_identity
from clustkit.pipeline import run_pipeline

MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
CDHIT_SIF = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/cd-hit.sif"

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "annotation_benchmark_results"


# ======================================================================
# Annotation loading
# ======================================================================


def load_go_annotations(goa_file):
    """Load GO annotations from a GAF (Gene Association Format) file.

    GAF format is tab-separated with the following relevant columns:
        [0]  DB           (e.g. UniProtKB)
        [1]  DB_Object_ID (e.g. UniProt accession like P12345)
        [2]  DB_Object_Symbol
        [3]  Qualifier
        [4]  GO_ID        (e.g. GO:0008150)

    Lines starting with '!' are comments.

    Args:
        goa_file: Path to GAF format GO annotation file.

    Returns:
        dict mapping sequence accession (str) to set of GO term IDs.
    """
    go_annotations = defaultdict(set)
    goa_path = Path(goa_file)

    if not goa_path.exists():
        raise FileNotFoundError(f"GOA file not found: {goa_path}")

    open_func = open
    if str(goa_path).endswith(".gz"):
        import gzip
        open_func = gzip.open

    count = 0
    with open_func(goa_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 5:
                continue
            accession = parts[1].strip()
            go_term = parts[4].strip()
            if accession and go_term.startswith("GO:"):
                go_annotations[accession].add(go_term)
                count += 1

    print(f"  Loaded GO annotations: {len(go_annotations)} proteins, "
          f"{count} annotation entries")
    return dict(go_annotations)


def load_pfam_annotations(pfam_file):
    """Load Pfam domain annotations from an InterPro TSV file.

    Expected InterPro TSV format (from InterProScan or InterPro download):
        [0]  protein_accession  (e.g. P12345)
        [1]  sequence_md5
        [2]  sequence_length
        [3]  analysis           (e.g. Pfam)
        [4]  signature_accession (e.g. PF00042)
        [5]  signature_description
        ...

    We extract columns [0] and [4] where [4] matches PF\\d+ pattern.

    Also supports a simpler two-column format: accession<TAB>pfam_id

    Args:
        pfam_file: Path to InterPro Pfam annotation TSV file.

    Returns:
        dict mapping sequence accession (str) to set of Pfam domain IDs.
    """
    pfam_annotations = defaultdict(set)
    pfam_path = Path(pfam_file)

    if not pfam_path.exists():
        raise FileNotFoundError(f"Pfam file not found: {pfam_path}")

    open_func = open
    if str(pfam_path).endswith(".gz"):
        import gzip
        open_func = gzip.open

    pfam_pattern = re.compile(r"^PF\d+")
    count = 0

    with open_func(pfam_path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("!"):
                continue
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 2:
                continue

            accession = parts[0].strip()

            # Try InterPro full format (column 4 = signature_accession)
            if len(parts) >= 5:
                sig_acc = parts[4].strip()
                if pfam_pattern.match(sig_acc):
                    pfam_annotations[accession].add(sig_acc)
                    count += 1
                    continue

            # Fallback: two-column format (accession, pfam_id)
            candidate = parts[1].strip()
            if pfam_pattern.match(candidate):
                pfam_annotations[accession].add(candidate)
                count += 1

    print(f"  Loaded Pfam annotations: {len(pfam_annotations)} proteins, "
          f"{count} annotation entries")
    return dict(pfam_annotations)


# ======================================================================
# Consistency metrics
# ======================================================================


def compute_go_consistency(clusters, representatives, go_annotations):
    """Compute GO annotation consistency per cluster.

    For each cluster with >=2 members, checks if each non-representative
    member shares at least one GO term with the representative.

    Consistency for a cluster = fraction of non-rep members sharing >=1 GO
    term with the representative.

    Only clusters with >=5 annotated members (including representative) are
    included in statistics to avoid noisy small-cluster effects.

    Args:
        clusters: dict mapping cluster_id -> list of sequence accessions.
        representatives: dict mapping cluster_id -> representative accession.
        go_annotations: dict mapping accession -> set of GO terms.

    Returns:
        dict with keys: mean, median, min, std, n_clusters_evaluated,
        n_clusters_skipped, per_cluster (list of per-cluster consistency).
    """
    per_cluster_consistency = []
    n_skipped = 0

    for cid, members in clusters.items():
        rep = representatives.get(cid)
        if rep is None:
            n_skipped += 1
            continue

        rep_go = go_annotations.get(rep, set())
        if not rep_go:
            n_skipped += 1
            continue

        # Gather annotated non-rep members
        annotated_members = []
        for m in members:
            if m == rep:
                continue
            m_go = go_annotations.get(m, set())
            if m_go:
                annotated_members.append(m)

        # Require at least 4 annotated non-rep members (5 total with rep)
        if len(annotated_members) < 4:
            n_skipped += 1
            continue

        # Compute fraction sharing >=1 GO term with representative
        n_shared = 0
        for m in annotated_members:
            m_go = go_annotations.get(m, set())
            if m_go & rep_go:
                n_shared += 1

        consistency = n_shared / len(annotated_members)
        per_cluster_consistency.append(consistency)

    if not per_cluster_consistency:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "std": 0.0,
            "n_clusters_evaluated": 0,
            "n_clusters_skipped": n_skipped,
            "per_cluster": [],
        }

    arr = np.array(per_cluster_consistency)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "std": float(np.std(arr)),
        "n_clusters_evaluated": len(arr),
        "n_clusters_skipped": n_skipped,
        "per_cluster": per_cluster_consistency,
    }


def compute_pfam_consistency(clusters, representatives, pfam_annotations):
    """Compute Pfam annotation consistency per cluster.

    For each cluster with >=2 members, checks if each non-representative
    member shares at least one Pfam domain with the representative.

    Consistency for a cluster = fraction of non-rep members sharing >=1 Pfam
    domain with the representative.

    Only clusters with >=5 annotated members (including representative) are
    included in statistics.

    Args:
        clusters: dict mapping cluster_id -> list of sequence accessions.
        representatives: dict mapping cluster_id -> representative accession.
        pfam_annotations: dict mapping accession -> set of Pfam domain IDs.

    Returns:
        dict with keys: mean, median, min, std, n_clusters_evaluated,
        n_clusters_skipped, per_cluster (list of per-cluster consistency).
    """
    per_cluster_consistency = []
    n_skipped = 0

    for cid, members in clusters.items():
        rep = representatives.get(cid)
        if rep is None:
            n_skipped += 1
            continue

        rep_pfam = pfam_annotations.get(rep, set())
        if not rep_pfam:
            n_skipped += 1
            continue

        # Gather annotated non-rep members
        annotated_members = []
        for m in members:
            if m == rep:
                continue
            m_pfam = pfam_annotations.get(m, set())
            if m_pfam:
                annotated_members.append(m)

        # Require at least 4 annotated non-rep members (5 total with rep)
        if len(annotated_members) < 4:
            n_skipped += 1
            continue

        # Compute fraction sharing >=1 Pfam domain with representative
        n_shared = 0
        for m in annotated_members:
            m_pfam = pfam_annotations.get(m, set())
            if m_pfam & rep_pfam:
                n_shared += 1

        consistency = n_shared / len(annotated_members)
        per_cluster_consistency.append(consistency)

    if not per_cluster_consistency:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "std": 0.0,
            "n_clusters_evaluated": 0,
            "n_clusters_skipped": n_skipped,
            "per_cluster": [],
        }

    arr = np.array(per_cluster_consistency)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "std": float(np.std(arr)),
        "n_clusters_evaluated": len(arr),
        "n_clusters_skipped": n_skipped,
        "per_cluster": per_cluster_consistency,
    }


def compute_distance_distribution(clusters, representatives, encoded_sequences,
                                  lengths, id_to_index, mode="protein"):
    """Compute alignment identity from each member to its cluster representative.

    For each cluster, aligns every non-representative member to the
    representative using ClustKIT's banded Needleman-Wunsch kernel.

    Args:
        clusters: dict mapping cluster_id -> list of sequence accessions.
        representatives: dict mapping cluster_id -> representative accession.
        encoded_sequences: (N, max_len) uint8 encoded sequence matrix.
        lengths: (N,) int32 array of sequence lengths.
        id_to_index: dict mapping sequence accession -> index in the matrix.
        mode: "protein" or "nucleotide".

    Returns:
        1D numpy array of member-to-representative identity values.
    """
    identities = []
    max_len = encoded_sequences.shape[1]

    # Use a generous band width for accurate alignment
    p95_len = int(np.percentile(lengths[lengths > 0], 95)) if len(lengths) > 0 else 100
    band_width = max(20, int(p95_len * 0.3))

    total_pairs = 0
    for cid, members in clusters.items():
        rep = representatives.get(cid)
        if rep is None:
            continue
        rep_idx = id_to_index.get(rep)
        if rep_idx is None:
            continue

        rep_seq = encoded_sequences[rep_idx]
        rep_len = int(lengths[rep_idx])

        for m in members:
            if m == rep:
                continue
            m_idx = id_to_index.get(m)
            if m_idx is None:
                continue

            m_seq = encoded_sequences[m_idx]
            m_len = int(lengths[m_idx])

            # _nw_identity expects shorter seq as first argument
            if m_len <= rep_len:
                identity = _nw_identity(
                    m_seq, np.int32(m_len),
                    rep_seq, np.int32(rep_len),
                    np.int32(band_width), np.float32(0.0),
                )
            else:
                identity = _nw_identity(
                    rep_seq, np.int32(rep_len),
                    m_seq, np.int32(m_len),
                    np.int32(band_width), np.float32(0.0),
                )

            identities.append(float(identity))
            total_pairs += 1

    print(f"    Computed {total_pairs} member-to-representative alignments")
    return np.array(identities, dtype=np.float64)


# ======================================================================
# Sequence ID normalization
# ======================================================================


def _normalize_id(seq_id):
    """Extract UniProt accession from sp|ACC|NAME or tr|ACC|NAME format.

    Returns the accession part, or the original ID if not in that format.
    """
    parts = seq_id.split("|")
    if len(parts) >= 2:
        return parts[1]
    return seq_id


# ======================================================================
# Clustering tool runners
# ======================================================================


def run_clustkit_cluster(fasta, threshold, threads, output_dir):
    """Run ClustKIT clustering and return clusters + representatives.

    Args:
        fasta: Path to input FASTA file.
        threshold: Identity threshold (0-1).
        threads: Number of CPU threads.
        output_dir: Directory for ClustKIT output files.

    Returns:
        Tuple of (clusters, representatives, elapsed_seconds) where:
        - clusters: dict mapping cluster_id -> list of sequence accessions
        - representatives: dict mapping cluster_id -> representative accession
        - elapsed_seconds: wall-clock time
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input": str(fasta),
        "output": str(output_dir),
        "threshold": threshold,
        "mode": "protein",
        "alignment": "align",
        "sketch_size": 128,
        "kmer_size": 5,
        "sensitivity": "high",
        "cluster_method": "connected",
        "representative": "longest",
        "device": "cpu",
        "threads": threads,
        "format": "tsv",
    }

    start = time.perf_counter()
    run_pipeline(config)
    elapsed = time.perf_counter() - start

    # Parse ClustKIT TSV output
    clusters = defaultdict(list)
    representatives = {}

    tsv_path = output_dir / "clusters.tsv"
    if not tsv_path.exists():
        print(f"    WARNING: ClustKIT output not found at {tsv_path}")
        return {}, {}, elapsed

    with open(tsv_path) as f:
        header = next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            seq_id = parts[0]
            cluster_id = int(parts[1])
            is_rep = parts[2].strip().lower() == "true"

            acc = _normalize_id(seq_id)
            clusters[cluster_id].append(acc)
            if is_rep:
                representatives[cluster_id] = acc

    print(f"    ClustKIT: {len(clusters)} clusters, {elapsed:.2f}s")
    return dict(clusters), representatives, elapsed


def _parse_mmseqs_clusters_full(tsv_path):
    """Parse MMseqs2/Linclust cluster TSV into clusters + representatives dicts.

    MMseqs2 TSV format: representative_accession<TAB>member_accession

    Returns:
        Tuple of (clusters, representatives) where:
        - clusters: dict mapping cluster_id (int) -> list of accessions
        - representatives: dict mapping cluster_id (int) -> representative accession
    """
    rep_to_cid = {}
    clusters = defaultdict(list)
    representatives = {}
    next_cid = 0

    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            rep_acc = _normalize_id(parts[0])
            member_acc = _normalize_id(parts[1])

            if rep_acc not in rep_to_cid:
                cid = next_cid
                next_cid += 1
                rep_to_cid[rep_acc] = cid
                representatives[cid] = rep_acc

            cid = rep_to_cid[rep_acc]
            clusters[cid].append(member_acc)

    return dict(clusters), representatives


def _parse_cdhit_clusters_full(clstr_path):
    """Parse CD-HIT .clstr file into clusters + representatives dicts.

    Returns:
        Tuple of (clusters, representatives) where:
        - clusters: dict mapping cluster_id (int) -> list of accessions
        - representatives: dict mapping cluster_id (int) -> representative accession
    """
    clusters = defaultdict(list)
    representatives = {}
    current_cluster = -1

    with open(clstr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
            elif line:
                # Extract sequence ID between '>' and '...'
                if ">" in line and "..." in line:
                    seq_id = line.split(">")[1].split("...")[0]
                    acc = _normalize_id(seq_id)
                    clusters[current_cluster].append(acc)
                    if line.rstrip().endswith("*"):
                        representatives[current_cluster] = acc

    return dict(clusters), representatives


def run_mmseqs_cluster(fasta, threshold, threads):
    """Run MMseqs2 easy-cluster.

    Args:
        fasta: Path to input FASTA file.
        threshold: Identity threshold (0-1).
        threads: Number of CPU threads.

    Returns:
        Tuple of (clusters, representatives, elapsed_seconds).
    """
    output_prefix = OUTPUT_DIR / f"mmseqs_t{threshold}"
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_tmp_")

    cmd = [
        MMSEQS_BIN, "easy-cluster",
        str(fasta),
        str(output_prefix),
        tmp_dir,
        "--min-seq-id", str(threshold),
        "--threads", str(threads),
        "-c", "0.8",
        "--cov-mode", "0",
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        print(f"    MMseqs2 FAILED (exit {result.returncode}): "
              f"{result.stderr[:200] if result.stderr else 'no stderr'}")
        return {}, {}, elapsed

    tsv_path = str(output_prefix) + "_cluster.tsv"
    if not os.path.exists(tsv_path):
        print(f"    MMseqs2: output not found at {tsv_path}")
        return {}, {}, elapsed

    clusters, representatives = _parse_mmseqs_clusters_full(tsv_path)
    n_seqs = sum(len(v) for v in clusters.values())
    print(f"    MMseqs2: {len(clusters)} clusters ({n_seqs} seqs), {elapsed:.2f}s")
    return clusters, representatives, elapsed


def run_linclust_cluster(fasta, threshold, threads):
    """Run MMseqs2 easy-linclust (linear-time clustering).

    Args:
        fasta: Path to input FASTA file.
        threshold: Identity threshold (0-1).
        threads: Number of CPU threads.

    Returns:
        Tuple of (clusters, representatives, elapsed_seconds).
    """
    output_prefix = OUTPUT_DIR / f"linclust_t{threshold}"
    tmp_dir = tempfile.mkdtemp(prefix="linclust_tmp_")

    cmd = [
        MMSEQS_BIN, "easy-linclust",
        str(fasta),
        str(output_prefix),
        tmp_dir,
        "--min-seq-id", str(threshold),
        "--threads", str(threads),
        "-c", "0.8",
        "--cov-mode", "0",
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        print(f"    Linclust FAILED (exit {result.returncode}): "
              f"{result.stderr[:200] if result.stderr else 'no stderr'}")
        return {}, {}, elapsed

    tsv_path = str(output_prefix) + "_cluster.tsv"
    if not os.path.exists(tsv_path):
        print(f"    Linclust: output not found at {tsv_path}")
        return {}, {}, elapsed

    clusters, representatives = _parse_mmseqs_clusters_full(tsv_path)
    n_seqs = sum(len(v) for v in clusters.values())
    print(f"    Linclust: {len(clusters)} clusters ({n_seqs} seqs), {elapsed:.2f}s")
    return clusters, representatives, elapsed


def run_cdhit_cluster(fasta, threshold, threads):
    """Run CD-HIT clustering.

    Args:
        fasta: Path to input FASTA file.
        threshold: Identity threshold (0-1).
        threads: Number of CPU threads.

    Returns:
        Tuple of (clusters, representatives, elapsed_seconds).
    """
    output_prefix = OUTPUT_DIR / f"cdhit_t{threshold}"

    # Select word size based on threshold (CD-HIT requirement)
    if threshold >= 0.7:
        word_size = "5"
    elif threshold >= 0.6:
        word_size = "4"
    elif threshold >= 0.5:
        word_size = "3"
    else:
        word_size = "2"

    cmd = [
        "singularity", "exec", CDHIT_SIF,
        "cd-hit",
        "-i", str(fasta),
        "-o", str(output_prefix),
        "-c", str(threshold),
        "-T", str(threads),
        "-M", "0",
        "-d", "0",
        "-n", word_size,
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"    CD-HIT FAILED (exit {result.returncode}): "
              f"{result.stderr[:200] if result.stderr else 'no stderr'}")
        return {}, {}, elapsed

    clstr_path = str(output_prefix) + ".clstr"
    if not os.path.exists(clstr_path):
        print(f"    CD-HIT: output not found at {clstr_path}")
        return {}, {}, elapsed

    clusters, representatives = _parse_cdhit_clusters_full(clstr_path)
    n_seqs = sum(len(v) for v in clusters.values())
    print(f"    CD-HIT: {len(clusters)} clusters ({n_seqs} seqs), {elapsed:.2f}s")
    return clusters, representatives, elapsed


# ======================================================================
# Benchmark experiments
# ======================================================================


def _summarize_consistency(name, result):
    """Print a summary line for a consistency result."""
    if result["n_clusters_evaluated"] == 0:
        print(f"    {name}: no evaluable clusters "
              f"(skipped {result['n_clusters_skipped']})")
        return
    print(f"    {name}: mean={result['mean']:.4f}, "
          f"median={result['median']:.4f}, "
          f"min={result['min']:.4f}, "
          f"std={result['std']:.4f} "
          f"({result['n_clusters_evaluated']} clusters evaluated, "
          f"{result['n_clusters_skipped']} skipped)")


def benchmark_go_consistency(fasta, goa_file, thresholds, threads):
    """CP2: GO annotation consistency benchmark.

    Clusters the input FASTA at each threshold using ClustKIT, MMseqs2,
    Linclust, and CD-HIT, then measures GO annotation consistency.

    Args:
        fasta: Path to input FASTA file (e.g., SwissProt).
        goa_file: Path to GAF format GO annotation file.
        thresholds: List of identity thresholds to test.
        threads: Number of CPU threads.

    Returns:
        dict mapping threshold -> tool -> consistency result.
    """
    print("=" * 100)
    print("CP2: GO ANNOTATION CONSISTENCY")
    print("=" * 100)
    print()

    print("Loading GO annotations...")
    go_annotations = load_go_annotations(goa_file)
    print()

    all_results = {}

    for threshold in thresholds:
        print("-" * 100)
        print(f"Threshold = {threshold}")
        print("-" * 100)

        threshold_results = {}

        # --- ClustKIT ---
        print("  Running ClustKIT...")
        ck_dir = OUTPUT_DIR / f"cp2_clustkit_t{threshold}"
        clusters, reps, elapsed = run_clustkit_cluster(
            fasta, threshold, threads, ck_dir
        )
        if clusters:
            consistency = compute_go_consistency(clusters, reps, go_annotations)
            consistency["runtime_seconds"] = round(elapsed, 2)
            consistency["n_clusters"] = len(clusters)
            threshold_results["clustkit"] = consistency
            _summarize_consistency("ClustKIT", consistency)
        else:
            threshold_results["clustkit"] = {
                "error": "clustering failed",
                "runtime_seconds": round(elapsed, 2),
            }

        # --- MMseqs2 ---
        print("  Running MMseqs2...")
        clusters, reps, elapsed = run_mmseqs_cluster(fasta, threshold, threads)
        if clusters:
            consistency = compute_go_consistency(clusters, reps, go_annotations)
            consistency["runtime_seconds"] = round(elapsed, 2)
            consistency["n_clusters"] = len(clusters)
            threshold_results["mmseqs2"] = consistency
            _summarize_consistency("MMseqs2", consistency)
        else:
            threshold_results["mmseqs2"] = {
                "error": "clustering failed",
                "runtime_seconds": round(elapsed, 2),
            }

        # --- Linclust ---
        print("  Running Linclust...")
        clusters, reps, elapsed = run_linclust_cluster(fasta, threshold, threads)
        if clusters:
            consistency = compute_go_consistency(clusters, reps, go_annotations)
            consistency["runtime_seconds"] = round(elapsed, 2)
            consistency["n_clusters"] = len(clusters)
            threshold_results["linclust"] = consistency
            _summarize_consistency("Linclust", consistency)
        else:
            threshold_results["linclust"] = {
                "error": "clustering failed",
                "runtime_seconds": round(elapsed, 2),
            }

        # --- CD-HIT ---
        print("  Running CD-HIT...")
        clusters, reps, elapsed = run_cdhit_cluster(fasta, threshold, threads)
        if clusters:
            consistency = compute_go_consistency(clusters, reps, go_annotations)
            consistency["runtime_seconds"] = round(elapsed, 2)
            consistency["n_clusters"] = len(clusters)
            threshold_results["cdhit"] = consistency
            _summarize_consistency("CD-HIT", consistency)
        else:
            threshold_results["cdhit"] = {
                "error": "clustering failed",
                "runtime_seconds": round(elapsed, 2),
            }

        all_results[str(threshold)] = threshold_results
        print()

    # Summary table
    _print_consistency_summary("CP2: GO Consistency", all_results, thresholds)

    return all_results


def benchmark_pfam_consistency(fasta, pfam_file, thresholds, threads):
    """CP3: Pfam annotation consistency benchmark.

    Clusters the input FASTA at each threshold using ClustKIT, MMseqs2,
    Linclust, and CD-HIT, then measures Pfam domain consistency.

    Args:
        fasta: Path to input FASTA file (e.g., SwissProt).
        pfam_file: Path to InterPro Pfam annotation TSV file.
        thresholds: List of identity thresholds to test.
        threads: Number of CPU threads.

    Returns:
        dict mapping threshold -> tool -> consistency result.
    """
    print("=" * 100)
    print("CP3: PFAM ANNOTATION CONSISTENCY")
    print("=" * 100)
    print()

    print("Loading Pfam annotations...")
    pfam_annotations = load_pfam_annotations(pfam_file)
    print()

    all_results = {}

    for threshold in thresholds:
        print("-" * 100)
        print(f"Threshold = {threshold}")
        print("-" * 100)

        threshold_results = {}

        # --- ClustKIT ---
        print("  Running ClustKIT...")
        ck_dir = OUTPUT_DIR / f"cp3_clustkit_t{threshold}"
        clusters, reps, elapsed = run_clustkit_cluster(
            fasta, threshold, threads, ck_dir
        )
        if clusters:
            consistency = compute_pfam_consistency(clusters, reps, pfam_annotations)
            consistency["runtime_seconds"] = round(elapsed, 2)
            consistency["n_clusters"] = len(clusters)
            threshold_results["clustkit"] = consistency
            _summarize_consistency("ClustKIT", consistency)
        else:
            threshold_results["clustkit"] = {
                "error": "clustering failed",
                "runtime_seconds": round(elapsed, 2),
            }

        # --- MMseqs2 ---
        print("  Running MMseqs2...")
        clusters, reps, elapsed = run_mmseqs_cluster(fasta, threshold, threads)
        if clusters:
            consistency = compute_pfam_consistency(clusters, reps, pfam_annotations)
            consistency["runtime_seconds"] = round(elapsed, 2)
            consistency["n_clusters"] = len(clusters)
            threshold_results["mmseqs2"] = consistency
            _summarize_consistency("MMseqs2", consistency)
        else:
            threshold_results["mmseqs2"] = {
                "error": "clustering failed",
                "runtime_seconds": round(elapsed, 2),
            }

        # --- Linclust ---
        print("  Running Linclust...")
        clusters, reps, elapsed = run_linclust_cluster(fasta, threshold, threads)
        if clusters:
            consistency = compute_pfam_consistency(clusters, reps, pfam_annotations)
            consistency["runtime_seconds"] = round(elapsed, 2)
            consistency["n_clusters"] = len(clusters)
            threshold_results["linclust"] = consistency
            _summarize_consistency("Linclust", consistency)
        else:
            threshold_results["linclust"] = {
                "error": "clustering failed",
                "runtime_seconds": round(elapsed, 2),
            }

        # --- CD-HIT ---
        print("  Running CD-HIT...")
        clusters, reps, elapsed = run_cdhit_cluster(fasta, threshold, threads)
        if clusters:
            consistency = compute_pfam_consistency(clusters, reps, pfam_annotations)
            consistency["runtime_seconds"] = round(elapsed, 2)
            consistency["n_clusters"] = len(clusters)
            threshold_results["cdhit"] = consistency
            _summarize_consistency("CD-HIT", consistency)
        else:
            threshold_results["cdhit"] = {
                "error": "clustering failed",
                "runtime_seconds": round(elapsed, 2),
            }

        all_results[str(threshold)] = threshold_results
        print()

    # Summary table
    _print_consistency_summary("CP3: Pfam Consistency", all_results, thresholds)

    return all_results


def benchmark_distance_distribution(fasta, threshold, threads):
    """CP4: Cumulative representative distance distribution.

    Clusters the input FASTA at the given threshold using ClustKIT, MMseqs2,
    Linclust, and CD-HIT. For each tool, computes alignment identity from
    every cluster member to its representative using ClustKIT's NW kernel.

    This measures how close representatives are to their cluster members --
    a sign of good representative selection.

    Args:
        fasta: Path to input FASTA file.
        threshold: Identity threshold for clustering.
        threads: Number of CPU threads.

    Returns:
        dict mapping tool_name -> dict with 'identities' array and stats.
    """
    print("=" * 100)
    print(f"CP4: CUMULATIVE DISTANCE DISTRIBUTION (threshold={threshold})")
    print("=" * 100)
    print()

    # Read sequences and build index for alignment
    print("Loading and encoding sequences for alignment...")
    dataset = read_sequences(Path(fasta), "protein")
    encoded_sequences = dataset.encoded_sequences
    lengths = dataset.lengths
    id_to_index = {}
    for i, rec in enumerate(dataset.records):
        acc = _normalize_id(rec.id)
        id_to_index[acc] = i
    print(f"  {dataset.num_sequences} sequences loaded, "
          f"max length {dataset.max_length}")
    print()

    all_results = {}

    tools = [
        ("clustkit", "ClustKIT"),
        ("mmseqs2", "MMseqs2"),
        ("linclust", "Linclust"),
        ("cdhit", "CD-HIT"),
    ]

    for tool_key, tool_name in tools:
        print(f"  Running {tool_name}...")

        if tool_key == "clustkit":
            ck_dir = OUTPUT_DIR / f"cp4_clustkit_t{threshold}"
            clusters, reps, elapsed = run_clustkit_cluster(
                fasta, threshold, threads, ck_dir
            )
        elif tool_key == "mmseqs2":
            clusters, reps, elapsed = run_mmseqs_cluster(
                fasta, threshold, threads
            )
        elif tool_key == "linclust":
            clusters, reps, elapsed = run_linclust_cluster(
                fasta, threshold, threads
            )
        elif tool_key == "cdhit":
            clusters, reps, elapsed = run_cdhit_cluster(
                fasta, threshold, threads
            )
        else:
            continue

        if not clusters:
            all_results[tool_key] = {
                "error": "clustering failed",
                "runtime_seconds": round(elapsed, 2),
            }
            continue

        print(f"  Computing distances for {tool_name}...")
        identities = compute_distance_distribution(
            clusters, reps, encoded_sequences, lengths, id_to_index,
            mode="protein",
        )

        if len(identities) == 0:
            all_results[tool_key] = {
                "error": "no member-representative pairs found",
                "n_clusters": len(clusters),
                "runtime_seconds": round(elapsed, 2),
            }
            print(f"    {tool_name}: no identity values computed")
            continue

        # Compute cumulative distribution percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pct_values = {
            f"p{p}": float(np.percentile(identities, p))
            for p in percentiles
        }

        result = {
            "n_clusters": len(clusters),
            "n_pairs": len(identities),
            "mean_identity": float(np.mean(identities)),
            "median_identity": float(np.median(identities)),
            "min_identity": float(np.min(identities)),
            "max_identity": float(np.max(identities)),
            "std_identity": float(np.std(identities)),
            "percentiles": pct_values,
            "runtime_seconds": round(elapsed, 2),
            # Store binned histogram for JSON (full array would be too large)
            "histogram_bins": _compute_histogram(identities),
        }
        all_results[tool_key] = result

        print(f"    {tool_name}: {len(identities)} pairs, "
              f"mean={result['mean_identity']:.4f}, "
              f"median={result['median_identity']:.4f}, "
              f"min={result['min_identity']:.4f}")

    # Summary table
    print()
    print("=" * 100)
    print("CP4 SUMMARY: Member-to-Representative Identity Distribution")
    print("=" * 100)
    print()
    print(f"{'Tool':<12} {'Pairs':>8} {'Clusters':>8} {'Mean':>8} "
          f"{'Median':>8} {'Min':>8} {'P10':>8} {'P90':>8} {'Time':>10}")
    print("-" * 100)

    for tool_key, tool_name in tools:
        r = all_results.get(tool_key, {})
        if "error" in r:
            rt = r.get("runtime_seconds", "?")
            print(f"{tool_name:<12} {'FAIL':>8} {'':>8} {'':>8} "
                  f"{'':>8} {'':>8} {'':>8} {'':>8} {rt:>9}s")
        elif r:
            pct = r.get("percentiles", {})
            print(f"{tool_name:<12} {r['n_pairs']:>8} {r['n_clusters']:>8} "
                  f"{r['mean_identity']:>8.4f} {r['median_identity']:>8.4f} "
                  f"{r['min_identity']:>8.4f} {pct.get('p10', 0):>8.4f} "
                  f"{pct.get('p90', 0):>8.4f} {r['runtime_seconds']:>9.2f}s")
    print()

    return all_results


def _compute_histogram(identities, n_bins=50):
    """Compute a histogram of identity values suitable for JSON storage.

    Returns:
        dict with 'bin_edges' (n_bins+1 values) and 'counts' (n_bins values).
    """
    counts, bin_edges = np.histogram(identities, bins=n_bins, range=(0.0, 1.0))
    return {
        "bin_edges": [round(float(x), 4) for x in bin_edges],
        "counts": [int(x) for x in counts],
    }


def _print_consistency_summary(title, all_results, thresholds):
    """Print a summary table for consistency results."""
    print()
    print("=" * 100)
    print(f"SUMMARY: {title}")
    print("=" * 100)
    print()
    print(f"{'Thresh':<8} {'Tool':<12} {'Clusters':>8} {'Mean':>8} "
          f"{'Median':>8} {'Min':>8} {'Std':>8} {'Evaluated':>10} {'Time':>10}")
    print("-" * 100)

    for t in thresholds:
        r = all_results.get(str(t), {})
        for tool_name, key in [("ClustKIT", "clustkit"), ("MMseqs2", "mmseqs2"),
                                ("Linclust", "linclust"), ("CD-HIT", "cdhit")]:
            res = r.get(key, {})
            if "error" in res:
                rt = res.get("runtime_seconds", "?")
                print(f"{t:<8} {tool_name:<12} {'FAIL':>8} {'':>8} "
                      f"{'':>8} {'':>8} {'':>8} {'':>10} {rt:>9}s")
            elif res:
                nc = res.get("n_clusters", "?")
                print(f"{t:<8} {tool_name:<12} {nc:>8} "
                      f"{res['mean']:>8.4f} {res['median']:>8.4f} "
                      f"{res['min']:>8.4f} {res['std']:>8.4f} "
                      f"{res['n_clusters_evaluated']:>10} "
                      f"{res['runtime_seconds']:>9.2f}s")
        print()


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Annotation consistency and distance distribution benchmarks "
                    "(CP2, CP3, CP4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments:
  python benchmark_annotation_consistency.py \\
      --fasta swissprot.fasta \\
      --goa-file goa_uniprot.gaf \\
      --pfam-file interpro_pfam.tsv \\
      --thresholds 0.3 0.5 0.7 0.9

  # Run only CP4 (no annotation files needed):
  python benchmark_annotation_consistency.py \\
      --fasta swissprot.fasta \\
      --experiments cp4 \\
      --thresholds 0.5
        """,
    )
    parser.add_argument(
        "--fasta", type=str, required=True,
        help="Input FASTA file (e.g., SwissProt sequences).",
    )
    parser.add_argument(
        "--goa-file", type=str, default=None,
        help="GAF format GO annotation file from UniProt-GOA. "
             "Required for CP2 (GO consistency).",
    )
    parser.add_argument(
        "--pfam-file", type=str, default=None,
        help="InterPro Pfam annotation TSV file. "
             "Required for CP3 (Pfam consistency).",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
        help="Identity thresholds to test (default: 0.3 0.5 0.7 0.9).",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of CPU threads for all tools (default: 4).",
    )
    parser.add_argument(
        "--experiments", type=str, nargs="+",
        default=["cp2", "cp3", "cp4"],
        choices=["cp2", "cp3", "cp4"],
        help="Which experiments to run (default: all).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: data/annotation_benchmark_results/).",
    )

    args = parser.parse_args()

    # Override output dir if specified
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fasta = Path(args.fasta)
    if not fasta.exists():
        print(f"ERROR: FASTA file not found: {fasta}")
        sys.exit(1)

    all_experiment_results = {}

    # CP2: GO consistency
    if "cp2" in args.experiments:
        if args.goa_file is None:
            print("WARNING: --goa-file not provided, skipping CP2 (GO consistency).")
        else:
            cp2_results = benchmark_go_consistency(
                fasta, args.goa_file, args.thresholds, args.threads,
            )
            all_experiment_results["cp2_go_consistency"] = cp2_results

    # CP3: Pfam consistency
    if "cp3" in args.experiments:
        if args.pfam_file is None:
            print("WARNING: --pfam-file not provided, skipping CP3 (Pfam consistency).")
        else:
            cp3_results = benchmark_pfam_consistency(
                fasta, args.pfam_file, args.thresholds, args.threads,
            )
            all_experiment_results["cp3_pfam_consistency"] = cp3_results

    # CP4: Distance distribution
    if "cp4" in args.experiments:
        # For CP4, use each threshold individually
        cp4_all = {}
        for t in args.thresholds:
            cp4_results = benchmark_distance_distribution(
                fasta, t, args.threads,
            )
            cp4_all[str(t)] = cp4_results
        all_experiment_results["cp4_distance_distribution"] = cp4_all

    # Save aggregated results (without per_cluster lists and histogram
    # arrays to keep file size reasonable)
    results_file = OUTPUT_DIR / "annotation_benchmark_results.json"
    serializable = _make_json_serializable(all_experiment_results)
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_file}")


def _make_json_serializable(obj):
    """Recursively convert an object to be JSON-serializable.

    Removes 'per_cluster' lists (too large) and converts numpy types.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "per_cluster":
                # Store summary stats instead of the full list
                if isinstance(v, list) and len(v) > 0:
                    result["per_cluster_count"] = len(v)
                continue
            result[k] = _make_json_serializable(v)
        return result
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if __name__ == "__main__":
    main()
