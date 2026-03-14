"""Search benchmarks: ClustKIT search vs BLAST vs MMseqs2 vs DIAMOND.

Implements CS4, CP5, CP6, CP7 from the publication plan.

CS4 -- Pfam search benchmark
    Use Pfam families as ground truth. One family at a time serves as queries,
    all remaining families form the database. True positive = hit lands in the
    same Pfam family as the query. Compare ClustKIT search, MMseqs2 easy-search,
    BLAST (blastp), and DIAMOND (blastp).  Metrics: sensitivity (TP rate) at
    several identity cutoffs, per-tool speed, total hits.

CP5 -- SCOP sensitivity benchmark
    Uses SCOP/SCOPe superfamily-level classification.  Each sequence is queried
    against the full database.  True positive = same superfamily; false positive
    = different fold.  Metrics: ROC AUC, sensitivity at 1 % FDR.

CP6 -- Sensitivity-speed trade-off
    Varies ClustKIT LSH parameters (sketch_size, sensitivity) and MMseqs2 -s
    (sensitivity) settings.  Produces AUC vs sequences/second data for Pareto
    plots.

CP7 -- False discovery rate analysis
    Searches real queries against a shuffled (reversed) database.  Counts hits
    above threshold in real vs shuffled runs to estimate empirical FDR.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clustkit.io import read_sequences
from clustkit.search import search_sequences

# ──────────────────────────────────────────────────────────────────────
# Tool paths
# ──────────────────────────────────────────────────────────────────────

MMSEQS_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
BLAST_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/ncbi-blast-2.17.0+/bin/blastp"
MAKEBLASTDB_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/ncbi-blast-2.17.0+/bin/makeblastdb"
DIAMOND_BIN = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/diamond-linux64/diamond"

DATA_DIR = Path(__file__).resolve().parent / "data" / "pfam_families"
SCOP_DIR = Path(__file__).resolve().parent / "data" / "scop"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "search_benchmark_results"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _tool_available(binary: str) -> bool:
    """Return True if *binary* is executable."""
    return shutil.which(binary) is not None or os.path.isfile(binary)


def _normalize_id(seq_id: str) -> str:
    """Extract UniProt accession from sp|ACC|NAME, or return as-is."""
    parts = seq_id.split("|")
    if len(parts) >= 2:
        return parts[1]
    return seq_id


def _write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    """Write a list of (header, sequence) tuples to a FASTA file."""
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")


def _read_fasta_raw(path: Path) -> list[tuple[str, str]]:
    """Read a FASTA file into a list of (header, sequence) tuples.

    *header* is the full text after '>'.
    """
    records: list[tuple[str, str]] = []
    header = None
    seq_parts: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            elif header is not None:
                seq_parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


# ──────────────────────────────────────────────────────────────────────
# Load Pfam families
# ──────────────────────────────────────────────────────────────────────

def load_pfam_families(
    data_dir: Path,
    max_per_family: int = 200,
) -> dict[str, list[tuple[str, str]]]:
    """Load Pfam family FASTAs.

    Returns:
        dict mapping pfam_id -> list of (seq_id, sequence).
    """
    fasta_files = sorted(data_dir.glob("PF*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No Pfam FASTA files in {data_dir}")

    families: dict[str, list[tuple[str, str]]] = {}
    for fasta_file in fasta_files:
        pfam_id = fasta_file.stem.split("_")[0]
        raw = _read_fasta_raw(fasta_file)
        seqs: list[tuple[str, str]] = []
        for header, seq in raw:
            sid = header.split()[0]
            seqs.append((sid, seq))
            if max_per_family and len(seqs) >= max_per_family:
                break
        if seqs:
            families[pfam_id] = seqs

    total = sum(len(v) for v in families.values())
    print(f"Loaded {len(families)} Pfam families, {total} sequences total")
    return families


def prepare_query_db_fastas(
    families: dict[str, list[tuple[str, str]]],
    query_family: str,
    workdir: Path,
) -> tuple[Path, Path, dict[str, str]]:
    """Write query and database FASTAs for a single-family-as-query experiment.

    Returns:
        (query_fasta, db_fasta, ground_truth)
        ground_truth maps seq_id -> pfam_id for every sequence in both files.
    """
    query_fasta = workdir / "query.fasta"
    db_fasta = workdir / "db.fasta"
    ground_truth: dict[str, str] = {}

    query_records = []
    for sid, seq in families[query_family]:
        query_records.append((sid, seq))
        ground_truth[_normalize_id(sid)] = query_family

    db_records = []
    for pfam_id, seqs in families.items():
        if pfam_id == query_family:
            continue
        for sid, seq in seqs:
            db_records.append((sid, seq))
            ground_truth[_normalize_id(sid)] = pfam_id

    _write_fasta(query_records, query_fasta)
    _write_fasta(db_records, db_fasta)
    return query_fasta, db_fasta, ground_truth


# ──────────────────────────────────────────────────────────────────────
# Tool runners
# ──────────────────────────────────────────────────────────────────────

def run_clustkit_search(
    query_fasta: Path,
    db_fasta: Path,
    threshold: float = 0.3,
    top_k: int = 50,
    kmer_size: int = 5,
    sketch_size: int = 128,
    sensitivity: str = "high",
    device: str = "cpu",
) -> tuple[dict[str, list[tuple[str, float]]], float]:
    """Run ClustKIT search and return per-query hit lists.

    Returns:
        (results, elapsed_seconds)
        results: dict  query_id -> [(target_id, identity), ...]
    """
    query_ds = read_sequences(query_fasta, mode="protein")
    db_ds = read_sequences(db_fasta, mode="protein")

    start = time.perf_counter()
    sr = search_sequences(
        query_ds,
        db_ds,
        threshold=threshold,
        top_k=top_k,
        mode="protein",
        kmer_size=kmer_size,
        sketch_size=sketch_size,
        sensitivity=sensitivity,
        device=device,
    )
    elapsed = time.perf_counter() - start

    results: dict[str, list[tuple[str, float]]] = {}
    for query_hits in sr.hits:
        for hit in query_hits:
            qid = _normalize_id(hit.query_id)
            tid = _normalize_id(hit.target_id)
            results.setdefault(qid, []).append((tid, hit.identity))

    return results, elapsed


def run_mmseqs_search(
    query_fasta: Path,
    db_fasta: Path,
    threshold: float = 0.3,
    threads: int = 4,
    sensitivity: float = 7.5,
) -> tuple[dict[str, list[tuple[str, float]]], float]:
    """Run MMseqs2 easy-search and return per-query hit lists.

    Returns:
        (results, elapsed_seconds)
    """
    tmpdir = tempfile.mkdtemp(prefix="mmseqs_search_")
    output_m8 = os.path.join(tmpdir, "results.m8")

    cmd = [
        MMSEQS_BIN, "easy-search",
        str(query_fasta),
        str(db_fasta),
        output_m8,
        os.path.join(tmpdir, "tmp"),
        "--min-seq-id", str(threshold),
        "--threads", str(threads),
        "-s", str(sensitivity),
        "--format-output", "query,target,pident,evalue",
    ]

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    results: dict[str, list[tuple[str, float]]] = {}
    if proc.returncode == 0 and os.path.exists(output_m8):
        results = parse_m8_results(output_m8)
    else:
        print(f"  [WARN] MMseqs2 search failed: {proc.stderr[:300]}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return results, elapsed


def run_blast_search(
    query_fasta: Path,
    db_fasta: Path,
    threshold: float = 0.3,
    threads: int = 4,
) -> tuple[dict[str, list[tuple[str, float]]], float]:
    """Run NCBI BLAST+ blastp and return per-query hit lists."""
    tmpdir = tempfile.mkdtemp(prefix="blast_search_")
    db_prefix = os.path.join(tmpdir, "blastdb")
    output_m8 = os.path.join(tmpdir, "results.m8")

    # makeblastdb
    mkdb_cmd = [
        MAKEBLASTDB_BIN,
        "-in", str(db_fasta),
        "-dbtype", "prot",
        "-out", db_prefix,
    ]
    mkdb_proc = subprocess.run(mkdb_cmd, capture_output=True, text=True, timeout=3600)
    if mkdb_proc.returncode != 0:
        print(f"  [WARN] makeblastdb failed: {mkdb_proc.stderr[:300]}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {}, 0.0

    search_cmd = [
        BLAST_BIN,
        "-query", str(query_fasta),
        "-db", db_prefix,
        "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
        "-num_threads", str(threads),
        "-evalue", "10",
    ]

    start = time.perf_counter()
    proc = subprocess.run(
        search_cmd, capture_output=True, text=True, timeout=7200,
        env={**os.environ, "BLASTDB": tmpdir},
    )
    elapsed = time.perf_counter() - start

    results: dict[str, list[tuple[str, float]]] = {}
    if proc.returncode == 0:
        # Write stdout to file for uniform parsing
        with open(output_m8, "w") as fh:
            fh.write(proc.stdout)
        results = parse_blast6_results(output_m8, min_identity=threshold * 100)
    else:
        print(f"  [WARN] blastp failed: {proc.stderr[:300]}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return results, elapsed


def run_diamond_search(
    query_fasta: Path,
    db_fasta: Path,
    threshold: float = 0.3,
    threads: int = 4,
) -> tuple[dict[str, list[tuple[str, float]]], float]:
    """Run DIAMOND blastp and return per-query hit lists."""
    tmpdir = tempfile.mkdtemp(prefix="diamond_search_")
    db_prefix = os.path.join(tmpdir, "diamonddb")
    output_m8 = os.path.join(tmpdir, "results.m8")

    # diamond makedb
    mkdb_cmd = [
        DIAMOND_BIN, "makedb",
        "--in", str(db_fasta),
        "-d", db_prefix,
    ]
    mkdb_proc = subprocess.run(mkdb_cmd, capture_output=True, text=True, timeout=3600)
    if mkdb_proc.returncode != 0:
        print(f"  [WARN] diamond makedb failed: {mkdb_proc.stderr[:300]}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {}, 0.0

    search_cmd = [
        DIAMOND_BIN, "blastp",
        "-q", str(query_fasta),
        "-d", db_prefix,
        "-o", output_m8,
        "--threads", str(threads),
        "--id", str(threshold * 100),
        "--outfmt", "6", "qseqid", "sseqid", "pident", "length",
        "mismatch", "gapopen", "qstart", "qend", "sstart", "send",
        "evalue", "bitscore",
    ]

    start = time.perf_counter()
    proc = subprocess.run(search_cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - start

    results: dict[str, list[tuple[str, float]]] = {}
    if proc.returncode == 0 and os.path.exists(output_m8):
        results = parse_blast6_results(output_m8, min_identity=threshold * 100)
    else:
        print(f"  [WARN] diamond blastp failed: {proc.stderr[:300]}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return results, elapsed


# ──────────────────────────────────────────────────────────────────────
# Result parsers
# ──────────────────────────────────────────────────────────────────────

def parse_m8_results(
    output_file: str,
) -> dict[str, list[tuple[str, float]]]:
    """Parse MMseqs2 easy-search output (custom 4-column: query, target, pident, evalue).

    Returns:
        dict  query_id -> [(target_id, identity_fraction), ...]
    """
    results: dict[str, list[tuple[str, float]]] = {}
    with open(output_file) as fh:
        for line in fh:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 3:
                continue
            qid = _normalize_id(parts[0])
            tid = _normalize_id(parts[1])
            try:
                pident = float(parts[2])
            except ValueError:
                continue
            # MMseqs2 pident is in [0,1] when using default format but may
            # also be reported as percent depending on version.  Normalise to
            # [0,1].
            identity = pident / 100.0 if pident > 1.0 else pident
            results.setdefault(qid, []).append((tid, identity))
    return results


def parse_blast6_results(
    output_file: str,
    min_identity: float = 0.0,
) -> dict[str, list[tuple[str, float]]]:
    """Parse BLAST tabular (-outfmt 6) output.

    Column order:
        qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore

    Args:
        output_file: Path to the m8/blast6 file.
        min_identity: Minimum percent identity to include (0-100 scale).

    Returns:
        dict  query_id -> [(target_id, identity_fraction), ...]
    """
    results: dict[str, list[tuple[str, float]]] = {}
    with open(output_file) as fh:
        for line in fh:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 12:
                continue
            qid = _normalize_id(parts[0])
            tid = _normalize_id(parts[1])
            try:
                pident = float(parts[2])
            except ValueError:
                continue
            if pident < min_identity:
                continue
            identity = pident / 100.0  # blast6 pident is always in percent
            results.setdefault(qid, []).append((tid, identity))
    return results


def parse_mmseqs_search_results(
    output_file: str,
) -> dict[str, list[tuple[str, float]]]:
    """Alias for ``parse_m8_results`` -- MMseqs2 easy-search m8 output."""
    return parse_m8_results(output_file)


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_search(
    results: dict[str, list[tuple[str, float]]],
    ground_truth: dict[str, str],
    identity_cutoffs: list[float] | None = None,
) -> dict:
    """Evaluate search results against ground truth family labels.

    A hit is a *true positive* if the query and target belong to the same
    Pfam family (according to *ground_truth*); otherwise it is a false
    positive.

    Args:
        results: query_id -> [(target_id, identity), ...]
        ground_truth: seq_id -> pfam_family
        identity_cutoffs: thresholds at which to report sensitivity.

    Returns:
        dict with per-cutoff TP / FP / sensitivity, plus aggregate counts.
    """
    if identity_cutoffs is None:
        identity_cutoffs = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Collect all hits with TP/FP labels
    all_hits: list[tuple[float, bool]] = []  # (identity, is_tp)
    n_queries_with_hits = 0

    for qid, hits in results.items():
        q_family = ground_truth.get(qid)
        if q_family is None:
            continue
        if hits:
            n_queries_with_hits += 1
        for tid, identity in hits:
            t_family = ground_truth.get(tid)
            if t_family is None:
                continue
            is_tp = (q_family == t_family)
            all_hits.append((identity, is_tp))

    total_hits = len(all_hits)
    total_tp = sum(1 for _, tp in all_hits if tp)
    total_fp = total_hits - total_tp

    # Per-cutoff metrics
    per_cutoff = {}
    for cutoff in identity_cutoffs:
        filtered = [(ident, tp) for ident, tp in all_hits if ident >= cutoff]
        tp = sum(1 for _, t in filtered if t)
        fp = len(filtered) - tp
        sensitivity = tp / max(total_tp, 1)  # fraction of all TPs recovered
        precision = tp / max(tp + fp, 1)
        per_cutoff[str(cutoff)] = {
            "cutoff": cutoff,
            "hits": len(filtered),
            "TP": tp,
            "FP": fp,
            "sensitivity": round(sensitivity, 4),
            "precision": round(precision, 4),
        }

    # ROC-style data: sort by descending identity, accumulate TP/FP
    roc_tp: list[int] = []
    roc_fp: list[int] = []
    tp_acc = 0
    fp_acc = 0
    sorted_hits = sorted(all_hits, key=lambda x: -x[0])
    for _, is_tp in sorted_hits:
        if is_tp:
            tp_acc += 1
        else:
            fp_acc += 1
        roc_tp.append(tp_acc)
        roc_fp.append(fp_acc)

    # Compute ROC AUC using trapezoidal rule
    roc_auc = _compute_roc_auc(roc_tp, roc_fp, total_tp, max(total_fp, 1))

    # Sensitivity at 1 % FDR
    sens_at_1pct_fdr = _sensitivity_at_fdr(roc_tp, roc_fp, total_tp, fdr_target=0.01)

    return {
        "total_hits": total_hits,
        "total_TP": total_tp,
        "total_FP": total_fp,
        "n_queries_with_hits": n_queries_with_hits,
        "roc_auc": round(roc_auc, 4),
        "sensitivity_at_1pct_fdr": round(sens_at_1pct_fdr, 4),
        "per_cutoff": per_cutoff,
    }


def _compute_roc_auc(
    roc_tp: list[int],
    roc_fp: list[int],
    total_tp: int,
    total_fp: int,
) -> float:
    """ROC AUC from cumulative TP/FP counts (hits sorted by descending score).

    Uses the trapezoidal rule on the (FPR, TPR) curve.
    """
    if total_tp == 0 or total_fp == 0:
        return 0.0

    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    for tp, fp in zip(roc_tp, roc_fp):
        tpr = tp / total_tp
        fpr = fp / total_fp
        if fpr != prev_fpr:
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr = tpr
        prev_fpr = fpr

    # Final step to FPR=1
    if prev_fpr < 1.0:
        auc += (1.0 - prev_fpr) * (prev_tpr + prev_tpr) / 2.0

    return min(auc, 1.0)


def _sensitivity_at_fdr(
    roc_tp: list[int],
    roc_fp: list[int],
    total_tp: int,
    fdr_target: float = 0.01,
) -> float:
    """Find the highest sensitivity where FDR <= *fdr_target*.

    FDR = FP / (TP + FP).
    """
    if total_tp == 0:
        return 0.0

    best_sens = 0.0
    for tp, fp in zip(roc_tp, roc_fp):
        total = tp + fp
        if total == 0:
            continue
        fdr = fp / total
        if fdr <= fdr_target:
            sens = tp / total_tp
            if sens > best_sens:
                best_sens = sens

    return best_sens


# ──────────────────────────────────────────────────────────────────────
# Shuffle database for FDR analysis (CP7)
# ──────────────────────────────────────────────────────────────────────

def shuffle_database(input_fasta: Path, output_fasta: Path) -> None:
    """Create a decoy database by reversing each sequence.

    Reversing (rather than random shuffling) preserves amino-acid composition
    and length distribution while destroying real homology signals.
    """
    records = _read_fasta_raw(input_fasta)
    reversed_records = []
    for header, seq in records:
        reversed_records.append((f"REV_{header}", seq[::-1]))
    _write_fasta(reversed_records, output_fasta)
    print(f"  Shuffled (reversed) database: {len(reversed_records)} sequences -> {output_fasta}")


# ──────────────────────────────────────────────────────────────────────
# CS4: Pfam search benchmark
# ──────────────────────────────────────────────────────────────────────

def benchmark_pfam_search(
    threads: int = 4,
    max_per_family: int = 200,
    max_query_families: int = 5,
    threshold: float = 0.3,
    skip_blast: bool = False,
    skip_diamond: bool = False,
) -> dict:
    """CS4: Pfam search benchmark.

    For each of *max_query_families* Pfam families, use that family as
    queries and all remaining families as the database.  Run all tools and
    compare sensitivity / speed.

    Returns:
        Aggregated results dict.
    """
    print("=" * 100)
    print("CS4: Pfam Search Benchmark")
    print("=" * 100)

    families = load_pfam_families(DATA_DIR, max_per_family=max_per_family)
    family_ids = sorted(families.keys())

    # Pick a representative subset of families as queries
    rng = random.Random(42)
    query_families = rng.sample(family_ids, min(max_query_families, len(family_ids)))
    print(f"Query families ({len(query_families)}): {query_families}")
    print()

    workdir = OUTPUT_DIR / "cs4_pfam_search"
    workdir.mkdir(parents=True, exist_ok=True)

    all_family_results: dict[str, dict] = {}

    for fam_idx, query_fam in enumerate(query_families):
        print(f"\n--- [{fam_idx + 1}/{len(query_families)}] Query family: {query_fam} "
              f"({len(families[query_fam])} seqs) ---")

        fam_workdir = workdir / query_fam
        fam_workdir.mkdir(parents=True, exist_ok=True)

        query_fasta, db_fasta, ground_truth = prepare_query_db_fastas(
            families, query_fam, fam_workdir,
        )

        n_queries = len(families[query_fam])
        n_db = sum(len(v) for k, v in families.items() if k != query_fam)
        print(f"  Queries: {n_queries}, Database: {n_db}")

        family_result: dict[str, dict] = {}

        # ── ClustKIT ──
        print(f"  ClustKIT search ...", end=" ", flush=True)
        ck_results, ck_time = run_clustkit_search(
            query_fasta, db_fasta, threshold=threshold,
        )
        ck_eval = evaluate_search(ck_results, ground_truth)
        ck_eval["runtime_seconds"] = round(ck_time, 2)
        family_result["clustkit"] = ck_eval
        print(f"hits={ck_eval['total_hits']}, TP={ck_eval['total_TP']}, "
              f"FP={ck_eval['total_FP']}, AUC={ck_eval['roc_auc']}, "
              f"{ck_time:.2f}s")

        # ── MMseqs2 ──
        if _tool_available(MMSEQS_BIN):
            print(f"  MMseqs2 search ...", end=" ", flush=True)
            mm_results, mm_time = run_mmseqs_search(
                query_fasta, db_fasta, threshold=threshold, threads=threads,
            )
            mm_eval = evaluate_search(mm_results, ground_truth)
            mm_eval["runtime_seconds"] = round(mm_time, 2)
            family_result["mmseqs2"] = mm_eval
            print(f"hits={mm_eval['total_hits']}, TP={mm_eval['total_TP']}, "
                  f"FP={mm_eval['total_FP']}, AUC={mm_eval['roc_auc']}, "
                  f"{mm_time:.2f}s")
        else:
            print("  MMseqs2: NOT AVAILABLE, skipping")

        # ── BLAST ──
        if not skip_blast and _tool_available(BLAST_BIN):
            print(f"  BLAST search ...", end=" ", flush=True)
            bl_results, bl_time = run_blast_search(
                query_fasta, db_fasta, threshold=threshold, threads=threads,
            )
            bl_eval = evaluate_search(bl_results, ground_truth)
            bl_eval["runtime_seconds"] = round(bl_time, 2)
            family_result["blast"] = bl_eval
            print(f"hits={bl_eval['total_hits']}, TP={bl_eval['total_TP']}, "
                  f"FP={bl_eval['total_FP']}, AUC={bl_eval['roc_auc']}, "
                  f"{bl_time:.2f}s")
        else:
            reason = "skipped by flag" if skip_blast else "NOT AVAILABLE"
            print(f"  BLAST: {reason}")

        # ── DIAMOND ──
        if not skip_diamond and _tool_available(DIAMOND_BIN):
            print(f"  DIAMOND search ...", end=" ", flush=True)
            dm_results, dm_time = run_diamond_search(
                query_fasta, db_fasta, threshold=threshold, threads=threads,
            )
            dm_eval = evaluate_search(dm_results, ground_truth)
            dm_eval["runtime_seconds"] = round(dm_time, 2)
            family_result["diamond"] = dm_eval
            print(f"hits={dm_eval['total_hits']}, TP={dm_eval['total_TP']}, "
                  f"FP={dm_eval['total_FP']}, AUC={dm_eval['roc_auc']}, "
                  f"{dm_time:.2f}s")
        else:
            reason = "skipped by flag" if skip_diamond else "NOT AVAILABLE"
            print(f"  DIAMOND: {reason}")

        all_family_results[query_fam] = family_result

    # ── Aggregate across families ──
    aggregated = _aggregate_cs4_results(all_family_results)

    # ── Summary table ──
    _print_cs4_summary(all_family_results, aggregated)

    # Save
    out_path = OUTPUT_DIR / "cs4_pfam_search_results.json"
    with open(out_path, "w") as fh:
        json.dump(
            {"per_family": all_family_results, "aggregate": aggregated},
            fh, indent=2,
        )
    print(f"\nResults saved to {out_path}")

    return {"per_family": all_family_results, "aggregate": aggregated}


def _aggregate_cs4_results(
    per_family: dict[str, dict],
) -> dict[str, dict]:
    """Average metrics across families for each tool."""
    tool_metrics: dict[str, list[dict]] = defaultdict(list)
    for _fam, tool_results in per_family.items():
        for tool, metrics in tool_results.items():
            tool_metrics[tool].append(metrics)

    aggregated: dict[str, dict] = {}
    for tool, metrics_list in tool_metrics.items():
        n = len(metrics_list)
        agg: dict[str, object] = {"n_families": n}
        for key in ["total_hits", "total_TP", "total_FP", "roc_auc",
                     "sensitivity_at_1pct_fdr", "runtime_seconds"]:
            vals = [m.get(key, 0) for m in metrics_list]
            agg[f"mean_{key}"] = round(np.mean(vals), 4)
            agg[f"std_{key}"] = round(np.std(vals), 4)
        aggregated[tool] = agg

    return aggregated


def _print_cs4_summary(
    per_family: dict[str, dict],
    aggregated: dict[str, dict],
) -> None:
    """Print a summary table for CS4."""
    print()
    print("=" * 100)
    print("CS4 SUMMARY — Aggregated across query families")
    print("=" * 100)
    header = (f"{'Tool':<12} {'Families':>8} {'Mean Hits':>10} {'Mean TP':>10} "
              f"{'Mean FP':>10} {'Mean AUC':>10} {'Mean Sens@1%FDR':>16} {'Mean Time':>10}")
    print(header)
    print("-" * 100)

    for tool, agg in sorted(aggregated.items()):
        print(
            f"{tool:<12} {agg['n_families']:>8} "
            f"{agg['mean_total_hits']:>10.1f} {agg['mean_total_TP']:>10.1f} "
            f"{agg['mean_total_FP']:>10.1f} {agg['mean_roc_auc']:>10.4f} "
            f"{agg['mean_sensitivity_at_1pct_fdr']:>16.4f} "
            f"{agg['mean_runtime_seconds']:>9.2f}s"
        )
    print()


# ──────────────────────────────────────────────────────────────────────
# CP5: SCOP sensitivity benchmark
# ──────────────────────────────────────────────────────────────────────

def load_scop_data(
    scop_dir: Path,
    max_seqs: int = 2000,
) -> tuple[Path, dict[str, str], dict[str, str]]:
    """Load SCOP/SCOPe sequences with superfamily and fold labels.

    Expects:
      - scop_dir/scop_sequences.fasta  (sequences)
      - scop_dir/scop_labels.tsv       (seq_id \\t superfamily \\t fold)

    If the label file does not exist, attempts to parse classification from
    FASTA headers in SCOPe format (sid, sccs).

    Returns:
        (fasta_path, superfamily_map, fold_map)
        superfamily_map: seq_id -> superfamily label
        fold_map: seq_id -> fold label
    """
    fasta_path = scop_dir / "scop_sequences.fasta"
    label_path = scop_dir / "scop_labels.tsv"

    if not fasta_path.exists():
        raise FileNotFoundError(
            f"SCOP sequences not found at {fasta_path}.  "
            "Download SCOPe data and place scop_sequences.fasta and "
            "scop_labels.tsv in benchmarks/data/scop/."
        )

    superfamily_map: dict[str, str] = {}
    fold_map: dict[str, str] = {}

    if label_path.exists():
        with open(label_path) as fh:
            for line in fh:
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) < 3:
                    continue
                sid, sf, fld = parts[0], parts[1], parts[2]
                superfamily_map[sid] = sf
                fold_map[sid] = fld
    else:
        # Try to parse from headers: >sid sccs description
        # sccs format: a.b.c.d  (a=class, a.b=fold, a.b.c=superfamily)
        records = _read_fasta_raw(fasta_path)
        for header, _seq in records:
            parts = header.split()
            if len(parts) >= 2:
                sid = parts[0]
                sccs = parts[1]
                sccs_parts = sccs.split(".")
                if len(sccs_parts) >= 3:
                    fold_map[sid] = ".".join(sccs_parts[:2])
                    superfamily_map[sid] = ".".join(sccs_parts[:3])

    if not superfamily_map:
        raise FileNotFoundError(
            f"Could not load SCOP labels from {label_path} or FASTA headers."
        )

    # Subsample if needed
    if max_seqs and len(superfamily_map) > max_seqs:
        rng = random.Random(42)
        keep = rng.sample(sorted(superfamily_map.keys()), max_seqs)
        keep_set = set(keep)
        superfamily_map = {k: v for k, v in superfamily_map.items() if k in keep_set}
        fold_map = {k: v for k, v in fold_map.items() if k in keep_set}

        # Write subsampled FASTA
        records = _read_fasta_raw(fasta_path)
        sub_records = [(h, s) for h, s in records if h.split()[0] in keep_set]
        sub_fasta = scop_dir / "scop_sequences_sub.fasta"
        _write_fasta(sub_records, sub_fasta)
        fasta_path = sub_fasta

    print(f"SCOP data: {len(superfamily_map)} sequences, "
          f"{len(set(superfamily_map.values()))} superfamilies, "
          f"{len(set(fold_map.values()))} folds")

    return fasta_path, superfamily_map, fold_map


def benchmark_scop_sensitivity(
    threads: int = 4,
    max_seqs: int = 2000,
    threshold: float = 0.0,
    skip_blast: bool = False,
    skip_diamond: bool = False,
) -> dict:
    """CP5: SCOP sensitivity benchmark.

    Each sequence is searched against the full database.  True positive =
    same superfamily; false positive = different fold.

    Returns:
        Results dict with per-tool ROC AUC and sensitivity at 1 % FDR.
    """
    print("=" * 100)
    print("CP5: SCOP Sensitivity Benchmark")
    print("=" * 100)

    try:
        fasta_path, sf_map, fold_map = load_scop_data(SCOP_DIR, max_seqs=max_seqs)
    except FileNotFoundError as exc:
        print(f"  SKIPPED: {exc}")
        return {"skipped": str(exc)}

    workdir = OUTPUT_DIR / "cp5_scop"
    workdir.mkdir(parents=True, exist_ok=True)

    # For SCOP we search all-vs-all: query = db = same file
    query_fasta = fasta_path
    db_fasta = fasta_path

    # Build ground truth for evaluation: TP = same superfamily
    # We use a modified evaluate that considers fold for FP definition
    results_all: dict[str, dict] = {}

    tools_to_run = [("clustkit", {})]
    if _tool_available(MMSEQS_BIN):
        tools_to_run.append(("mmseqs2", {}))
    if not skip_blast and _tool_available(BLAST_BIN):
        tools_to_run.append(("blast", {}))
    if not skip_diamond and _tool_available(DIAMOND_BIN):
        tools_to_run.append(("diamond", {}))

    for tool_name, _opts in tools_to_run:
        print(f"\n  {tool_name} ...", end=" ", flush=True)

        if tool_name == "clustkit":
            hits, elapsed = run_clustkit_search(
                query_fasta, db_fasta, threshold=threshold,
                top_k=100, sensitivity="high",
            )
        elif tool_name == "mmseqs2":
            hits, elapsed = run_mmseqs_search(
                query_fasta, db_fasta, threshold=threshold,
                threads=threads, sensitivity=7.5,
            )
        elif tool_name == "blast":
            hits, elapsed = run_blast_search(
                query_fasta, db_fasta, threshold=threshold,
                threads=threads,
            )
        elif tool_name == "diamond":
            hits, elapsed = run_diamond_search(
                query_fasta, db_fasta, threshold=threshold,
                threads=threads,
            )
        else:
            continue

        # Evaluate: TP = same superfamily (excluding self-hits)
        # FP = different fold
        scop_eval = _evaluate_scop(hits, sf_map, fold_map)
        scop_eval["runtime_seconds"] = round(elapsed, 2)
        results_all[tool_name] = scop_eval

        print(f"hits={scop_eval['total_scored']}, "
              f"TP={scop_eval['total_TP']}, FP={scop_eval['total_FP']}, "
              f"AUC={scop_eval['roc_auc']}, "
              f"sens@1%FDR={scop_eval['sensitivity_at_1pct_fdr']}, "
              f"{elapsed:.2f}s")

    # Summary
    print()
    print("-" * 80)
    print(f"{'Tool':<12} {'AUC':>8} {'Sens@1%FDR':>12} {'TP':>8} {'FP':>8} {'Time':>10}")
    print("-" * 80)
    for tool, res in sorted(results_all.items()):
        if "skipped" in res:
            continue
        print(f"{tool:<12} {res['roc_auc']:>8.4f} "
              f"{res['sensitivity_at_1pct_fdr']:>12.4f} "
              f"{res['total_TP']:>8} {res['total_FP']:>8} "
              f"{res['runtime_seconds']:>9.2f}s")

    out_path = OUTPUT_DIR / "cp5_scop_results.json"
    with open(out_path, "w") as fh:
        json.dump(results_all, fh, indent=2)
    print(f"\nResults saved to {out_path}")

    return results_all


def _evaluate_scop(
    results: dict[str, list[tuple[str, float]]],
    sf_map: dict[str, str],
    fold_map: dict[str, str],
) -> dict:
    """Evaluate search hits under SCOP criteria.

    True positive  = query and target share the same superfamily.
    False positive = query and target are in different folds.
    Hits between different superfamilies within the same fold are *ignored*
    (twilight zone).
    Self-hits are excluded.
    """
    scored_hits: list[tuple[float, bool]] = []  # (identity, is_tp)
    total_tp = 0
    total_fp = 0

    for qid, hits in results.items():
        q_sf = sf_map.get(qid)
        q_fold = fold_map.get(qid)
        if q_sf is None or q_fold is None:
            continue
        for tid, identity in hits:
            if tid == qid:
                continue  # skip self-hits
            t_sf = sf_map.get(tid)
            t_fold = fold_map.get(tid)
            if t_sf is None or t_fold is None:
                continue
            if q_sf == t_sf:
                scored_hits.append((identity, True))
                total_tp += 1
            elif q_fold != t_fold:
                scored_hits.append((identity, False))
                total_fp += 1
            # else: same fold, different superfamily -> ignore (twilight zone)

    # ROC
    scored_hits.sort(key=lambda x: -x[0])
    roc_tp: list[int] = []
    roc_fp: list[int] = []
    tp_acc = 0
    fp_acc = 0
    for _, is_tp in scored_hits:
        if is_tp:
            tp_acc += 1
        else:
            fp_acc += 1
        roc_tp.append(tp_acc)
        roc_fp.append(fp_acc)

    roc_auc = _compute_roc_auc(roc_tp, roc_fp, max(total_tp, 1), max(total_fp, 1))
    sens_1pct = _sensitivity_at_fdr(roc_tp, roc_fp, max(total_tp, 1), fdr_target=0.01)

    return {
        "total_scored": len(scored_hits),
        "total_TP": total_tp,
        "total_FP": total_fp,
        "roc_auc": round(roc_auc, 4),
        "sensitivity_at_1pct_fdr": round(sens_1pct, 4),
    }


# ──────────────────────────────────────────────────────────────────────
# CP6: Sensitivity-speed trade-off
# ──────────────────────────────────────────────────────────────────────

def benchmark_sensitivity_speed(
    threads: int = 4,
    max_per_family: int = 200,
    threshold: float = 0.3,
) -> dict:
    """CP6: Sensitivity-speed trade-off.

    Varies ClustKIT LSH parameters and MMseqs2 -s settings, recording
    ROC AUC vs query throughput (sequences/second).

    Returns:
        Results dict with per-configuration AUC and speed data.
    """
    print("=" * 100)
    print("CP6: Sensitivity-Speed Trade-off")
    print("=" * 100)

    families = load_pfam_families(DATA_DIR, max_per_family=max_per_family)
    family_ids = sorted(families.keys())

    # Use a single family as query for consistency
    rng = random.Random(42)
    query_fam = rng.choice(family_ids)
    print(f"Query family: {query_fam} ({len(families[query_fam])} seqs)")

    workdir = OUTPUT_DIR / "cp6_sensitivity_speed"
    workdir.mkdir(parents=True, exist_ok=True)

    query_fasta, db_fasta, ground_truth = prepare_query_db_fastas(
        families, query_fam, workdir,
    )
    n_queries = len(families[query_fam])

    results_all: dict[str, dict] = {}

    # ── ClustKIT configurations: vary sketch_size and sensitivity ──
    clustkit_configs = [
        {"sketch_size": 64,  "sensitivity": "low",    "kmer_size": 5},
        {"sketch_size": 64,  "sensitivity": "medium", "kmer_size": 5},
        {"sketch_size": 64,  "sensitivity": "high",   "kmer_size": 5},
        {"sketch_size": 128, "sensitivity": "low",    "kmer_size": 5},
        {"sketch_size": 128, "sensitivity": "medium", "kmer_size": 5},
        {"sketch_size": 128, "sensitivity": "high",   "kmer_size": 5},
        {"sketch_size": 256, "sensitivity": "low",    "kmer_size": 5},
        {"sketch_size": 256, "sensitivity": "medium", "kmer_size": 5},
        {"sketch_size": 256, "sensitivity": "high",   "kmer_size": 5},
    ]

    for cfg in clustkit_configs:
        label = f"clustkit_s{cfg['sketch_size']}_{cfg['sensitivity']}"
        print(f"\n  {label} ...", end=" ", flush=True)

        hits, elapsed = run_clustkit_search(
            query_fasta, db_fasta, threshold=threshold,
            sketch_size=cfg["sketch_size"],
            sensitivity=cfg["sensitivity"],
            kmer_size=cfg["kmer_size"],
        )
        ev = evaluate_search(hits, ground_truth)
        ev["runtime_seconds"] = round(elapsed, 2)
        ev["queries_per_second"] = round(n_queries / max(elapsed, 0.001), 2)
        ev["config"] = cfg
        results_all[label] = ev

        print(f"AUC={ev['roc_auc']}, "
              f"q/s={ev['queries_per_second']:.1f}, "
              f"{elapsed:.2f}s")

    # ── MMseqs2 at different -s settings ──
    if _tool_available(MMSEQS_BIN):
        mmseqs_sensitivities = [1.0, 4.0, 5.5, 7.0, 7.5]
        for s_val in mmseqs_sensitivities:
            label = f"mmseqs2_s{s_val}"
            print(f"\n  {label} ...", end=" ", flush=True)

            hits, elapsed = run_mmseqs_search(
                query_fasta, db_fasta, threshold=threshold,
                threads=threads, sensitivity=s_val,
            )
            ev = evaluate_search(hits, ground_truth)
            ev["runtime_seconds"] = round(elapsed, 2)
            ev["queries_per_second"] = round(n_queries / max(elapsed, 0.001), 2)
            ev["config"] = {"sensitivity": s_val}
            results_all[label] = ev

            print(f"AUC={ev['roc_auc']}, "
                  f"q/s={ev['queries_per_second']:.1f}, "
                  f"{elapsed:.2f}s")

    # Summary
    print()
    print("=" * 100)
    print("CP6 SUMMARY — AUC vs Speed")
    print("=" * 100)
    print(f"{'Configuration':<40} {'AUC':>8} {'Sens@1%FDR':>12} {'Q/sec':>10} {'Time':>10}")
    print("-" * 100)
    for label, ev in sorted(results_all.items(), key=lambda x: -x[1].get("roc_auc", 0)):
        print(f"{label:<40} {ev['roc_auc']:>8.4f} "
              f"{ev['sensitivity_at_1pct_fdr']:>12.4f} "
              f"{ev['queries_per_second']:>10.1f} "
              f"{ev['runtime_seconds']:>9.2f}s")

    out_path = OUTPUT_DIR / "cp6_sensitivity_speed_results.json"
    with open(out_path, "w") as fh:
        json.dump(results_all, fh, indent=2)
    print(f"\nResults saved to {out_path}")

    return results_all


# ──────────────────────────────────────────────────────────────────────
# CP7: False discovery rate analysis
# ──────────────────────────────────────────────────────────────────────

def benchmark_fdr(
    threads: int = 4,
    max_per_family: int = 200,
    threshold: float = 0.3,
    skip_blast: bool = False,
    skip_diamond: bool = False,
) -> dict:
    """CP7: False discovery rate analysis.

    Searches real queries against the real database and a shuffled (reversed)
    database.  Compares the number of hits above threshold in each case.

    Empirical FDR = hits_shuffled / hits_real (per identity bin).

    Returns:
        Results dict with per-tool real and shuffled hit counts.
    """
    print("=" * 100)
    print("CP7: False Discovery Rate Analysis")
    print("=" * 100)

    families = load_pfam_families(DATA_DIR, max_per_family=max_per_family)
    family_ids = sorted(families.keys())

    rng = random.Random(42)
    query_fam = rng.choice(family_ids)
    print(f"Query family: {query_fam} ({len(families[query_fam])} seqs)")

    workdir = OUTPUT_DIR / "cp7_fdr"
    workdir.mkdir(parents=True, exist_ok=True)

    query_fasta, db_fasta, ground_truth = prepare_query_db_fastas(
        families, query_fam, workdir,
    )

    # Create shuffled (reversed) database
    shuffled_fasta = workdir / "db_shuffled.fasta"
    shuffle_database(db_fasta, shuffled_fasta)

    identity_bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results_all: dict[str, dict] = {}

    tools: list[tuple[str, bool]] = [
        ("clustkit", True),
        ("mmseqs2", _tool_available(MMSEQS_BIN)),
    ]
    if not skip_blast:
        tools.append(("blast", _tool_available(BLAST_BIN)))
    if not skip_diamond:
        tools.append(("diamond", _tool_available(DIAMOND_BIN)))

    for tool_name, available in tools:
        if not available:
            print(f"\n  {tool_name}: NOT AVAILABLE, skipping")
            continue

        print(f"\n  {tool_name} — real database ...", end=" ", flush=True)

        if tool_name == "clustkit":
            real_hits, real_time = run_clustkit_search(
                query_fasta, db_fasta, threshold=threshold,
            )
        elif tool_name == "mmseqs2":
            real_hits, real_time = run_mmseqs_search(
                query_fasta, db_fasta, threshold=threshold, threads=threads,
            )
        elif tool_name == "blast":
            real_hits, real_time = run_blast_search(
                query_fasta, db_fasta, threshold=threshold, threads=threads,
            )
        elif tool_name == "diamond":
            real_hits, real_time = run_diamond_search(
                query_fasta, db_fasta, threshold=threshold, threads=threads,
            )
        else:
            continue

        real_count = sum(len(v) for v in real_hits.values())
        print(f"{real_count} hits, {real_time:.2f}s")

        print(f"  {tool_name} — shuffled database ...", end=" ", flush=True)

        if tool_name == "clustkit":
            shuf_hits, shuf_time = run_clustkit_search(
                query_fasta, shuffled_fasta, threshold=threshold,
            )
        elif tool_name == "mmseqs2":
            shuf_hits, shuf_time = run_mmseqs_search(
                query_fasta, shuffled_fasta, threshold=threshold, threads=threads,
            )
        elif tool_name == "blast":
            shuf_hits, shuf_time = run_blast_search(
                query_fasta, shuffled_fasta, threshold=threshold, threads=threads,
            )
        elif tool_name == "diamond":
            shuf_hits, shuf_time = run_diamond_search(
                query_fasta, shuffled_fasta, threshold=threshold, threads=threads,
            )
        else:
            continue

        shuf_count = sum(len(v) for v in shuf_hits.values())
        print(f"{shuf_count} hits, {shuf_time:.2f}s")

        # Bin hits by identity
        real_binned = _bin_hits_by_identity(real_hits, identity_bins)
        shuf_binned = _bin_hits_by_identity(shuf_hits, identity_bins)

        fdr_per_bin = {}
        for b in identity_bins:
            b_str = str(b)
            r = real_binned.get(b_str, 0)
            s = shuf_binned.get(b_str, 0)
            fdr = s / max(r, 1)
            fdr_per_bin[b_str] = {
                "real_hits": r,
                "shuffled_hits": s,
                "empirical_fdr": round(fdr, 6),
            }

        overall_fdr = shuf_count / max(real_count, 1)

        results_all[tool_name] = {
            "real_total_hits": real_count,
            "shuffled_total_hits": shuf_count,
            "overall_empirical_fdr": round(overall_fdr, 6),
            "real_runtime": round(real_time, 2),
            "shuffled_runtime": round(shuf_time, 2),
            "per_bin": fdr_per_bin,
        }

    # Summary
    print()
    print("=" * 100)
    print("CP7 SUMMARY — Empirical FDR (shuffled hits / real hits)")
    print("=" * 100)
    print(f"{'Tool':<12} {'Real Hits':>10} {'Shuf Hits':>10} {'FDR':>10}")
    print("-" * 50)
    for tool, res in sorted(results_all.items()):
        print(f"{tool:<12} {res['real_total_hits']:>10} "
              f"{res['shuffled_total_hits']:>10} "
              f"{res['overall_empirical_fdr']:>10.6f}")

    print()
    # Per-bin table
    print(f"{'Tool':<12} {'Bin':>6} {'Real':>8} {'Shuf':>8} {'FDR':>10}")
    print("-" * 50)
    for tool, res in sorted(results_all.items()):
        for b_str, bdata in sorted(res["per_bin"].items(), key=lambda x: float(x[0])):
            print(f"{tool:<12} {b_str:>6} {bdata['real_hits']:>8} "
                  f"{bdata['shuffled_hits']:>8} {bdata['empirical_fdr']:>10.6f}")
        print()

    out_path = OUTPUT_DIR / "cp7_fdr_results.json"
    with open(out_path, "w") as fh:
        json.dump(results_all, fh, indent=2)
    print(f"Results saved to {out_path}")

    return results_all


def _bin_hits_by_identity(
    results: dict[str, list[tuple[str, float]]],
    bins: list[float],
) -> dict[str, int]:
    """Count hits at or above each identity bin threshold.

    Returns:
        dict  bin_threshold_str -> count of hits with identity >= threshold.
    """
    all_identities = []
    for hits in results.values():
        for _tid, ident in hits:
            all_identities.append(ident)

    counts: dict[str, int] = {}
    for b in bins:
        counts[str(b)] = sum(1 for i in all_identities if i >= b)
    return counts


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Search benchmarks: ClustKIT vs BLAST vs MMseqs2 vs DIAMOND "
                    "(CS4, CP5, CP6, CP7).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments:
  cs4   Pfam family search benchmark (sensitivity, speed)
  cp5   SCOP superfamily sensitivity (ROC AUC, FDR)
  cp6   Sensitivity-speed trade-off (Pareto curve data)
  cp7   False discovery rate (real vs shuffled DB)
  all   Run all experiments
""",
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        default=["all"],
        choices=["cs4", "cp5", "cp6", "cp7", "all"],
        help="Which experiments to run (default: all).",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="CPU threads for external tools (default: 4).",
    )
    parser.add_argument(
        "--max-per-family", type=int, default=200,
        help="Max sequences per Pfam family (default: 200).",
    )
    parser.add_argument(
        "--max-query-families", type=int, default=5,
        help="Number of Pfam families to use as queries in CS4 (default: 5).",
    )
    parser.add_argument(
        "--max-scop-seqs", type=int, default=2000,
        help="Max SCOP sequences for CP5 (default: 2000).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Base identity threshold for search (default: 0.3).",
    )
    parser.add_argument(
        "--skip-blast", action="store_true",
        help="Skip BLAST benchmarks.",
    )
    parser.add_argument(
        "--skip-diamond", action="store_true",
        help="Skip DIAMOND benchmarks.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exps = set(args.experiments)
    run_all = "all" in exps

    print(f"Search benchmark suite — threads={args.threads}, threshold={args.threshold}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    all_results: dict[str, dict] = {}

    if run_all or "cs4" in exps:
        all_results["cs4"] = benchmark_pfam_search(
            threads=args.threads,
            max_per_family=args.max_per_family,
            max_query_families=args.max_query_families,
            threshold=args.threshold,
            skip_blast=args.skip_blast,
            skip_diamond=args.skip_diamond,
        )

    if run_all or "cp5" in exps:
        all_results["cp5"] = benchmark_scop_sensitivity(
            threads=args.threads,
            max_seqs=args.max_scop_seqs,
            threshold=args.threshold,
            skip_blast=args.skip_blast,
            skip_diamond=args.skip_diamond,
        )

    if run_all or "cp6" in exps:
        all_results["cp6"] = benchmark_sensitivity_speed(
            threads=args.threads,
            max_per_family=args.max_per_family,
            threshold=args.threshold,
        )

    if run_all or "cp7" in exps:
        all_results["cp7"] = benchmark_fdr(
            threads=args.threads,
            max_per_family=args.max_per_family,
            threshold=args.threshold,
            skip_blast=args.skip_blast,
            skip_diamond=args.skip_diamond,
        )

    # Save combined results
    combined_path = OUTPUT_DIR / "search_benchmark_all_results.json"
    with open(combined_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
