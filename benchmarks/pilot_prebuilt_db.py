#!/usr/bin/env python3
"""Test pre-built database with reduced + spaced seed indices.

1. Rebuild the SCOPe database with new build_database (includes reduced indices)
2. Save and reload to verify persistence
3. Run search with pre-built indices (should skip on-the-fly building)
4. Compare: Numba scoring vs C scoring, both with pre-built indices
5. Regression check: ROC1 should match baseline (~0.808)
"""

import json, random, sys, time, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, read_fasta, write_fasta, RankedHit,
)
import numba; numba.set_num_threads(8)
from clustkit.database import build_database, save_database, load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import search_kmer_index


def evaluate_roc1(results, metadata):
    di = metadata["domain_info"]; fs = metadata["family_sizes_in_db"]
    qs = set(metadata["query_sids"])
    hbq = defaultdict(list)
    for qh in results.hits:
        for h in qh:
            hbq[h.query_id].append((h.target_id, h.score if h.score != 0 else h.identity))
    vals = []
    for qid in qs:
        qi = di.get(qid)
        if qi is None: continue
        tp = fs.get(str(qi["family"]), 1) - 1
        if tp <= 0: continue
        qh = sorted(hbq.get(qid, []), key=lambda x: -x[1])
        ranked = [RankedHit(target_id=t, score=s, label=classify_hit(qid, t, di))
                  for t, s in qh if classify_hit(qid, t, di) != "IGNORE"]
        vals.append(compute_roc_n(ranked, 1, tp))
    return float(np.mean(vals)) if vals else 0.0


def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")
    new_db_dir = out_dir / "clustkit_db_v2"

    with open(scop_dir / "metadata.json") as f:
        fm = json.load(f)
    random.seed(42)
    qsids = sorted(random.sample(fm["query_sids"], min(2000, len(fm["query_sids"]))))
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub = [(s, all_seqs[s]) for s in qsids if s in all_seqs]
    qf = str(out_dir / "queries_subset.fasta")
    write_fasta(sub, qf)
    metadata = dict(fm); metadata["query_sids"] = qsids

    # ── Step 1: Build new database with pre-built reduced indices ────
    print("=" * 80)
    print("Step 1: Build database with pre-built reduced indices")
    print("=" * 80, flush=True)

    # Use the existing database FASTA (the db sequences)
    db_fasta = scop_dir / "database.fasta"
    if not db_fasta.exists():
        print(f"  Database FASTA not found at {db_fasta}", flush=True)
        print("  Using existing database and just testing load/save...", flush=True)

        # Load existing, add reduced indices, save as v2
        print("\n  Loading existing database...", flush=True)
        t0 = time.perf_counter()
        db_index = load_database(out_dir / "clustkit_db")
        t_load_old = time.perf_counter() - t0
        print(f"  Loaded in {t_load_old:.1f}s", flush=True)
        print(f"  Pre-built reduced indices: {db_index.reduced_indices is not None}", flush=True)

        if db_index.reduced_indices is None:
            print("\n  Building reduced indices for existing database...", flush=True)
            from clustkit.kmer_index import (
                REDUCED_ALPHA, REDUCED_ALPHA_SIZE, _remap_flat,
                build_kmer_index, build_kmer_index_spaced,
            )
            from clustkit.utils import timer

            with timer("Building reduced indices"):
                red_flat = _remap_flat(
                    db_index.dataset.flat_sequences, REDUCED_ALPHA,
                    len(db_index.dataset.flat_sequences))
                reduced_indices = {}

                r4o, r4e, r4f = build_kmer_index(
                    red_flat, db_index.dataset.offsets, db_index.dataset.lengths,
                    4, "protein", alpha_size=REDUCED_ALPHA_SIZE)
                reduced_indices["red_k4"] = (r4o, r4e, r4f)

                r5o, r5e, r5f = build_kmer_index(
                    red_flat, db_index.dataset.offsets, db_index.dataset.lengths,
                    5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
                reduced_indices["red_k5"] = (r5o, r5e, r5f)

                for pattern in ["11011", "110011"]:
                    sp = build_kmer_index_spaced(
                        red_flat, db_index.dataset.offsets, db_index.dataset.lengths,
                        pattern, "protein", alpha_size=REDUCED_ALPHA_SIZE)
                    reduced_indices[f"sp_{pattern}"] = sp

            db_index.reduced_indices = reduced_indices
            db_index.reduced_flat = red_flat
            print(f"  Built {len(reduced_indices)} reduced indices", flush=True)

        # Save as v2
        print(f"\n  Saving to {new_db_dir}...", flush=True)
        t0 = time.perf_counter()
        save_database(db_index, new_db_dir)
        t_save = time.perf_counter() - t0
        print(f"  Saved in {t_save:.1f}s", flush=True)
    else:
        print(f"  Building from {db_fasta}...", flush=True)
        t0 = time.perf_counter()
        db_index = build_database(db_fasta, mode="protein")
        t_build = time.perf_counter() - t0
        print(f"  Built in {t_build:.1f}s", flush=True)
        print(f"  Reduced indices: {list(db_index.reduced_indices.keys()) if db_index.reduced_indices else 'None'}", flush=True)

        save_database(db_index, new_db_dir)

    # ── Step 2: Reload and verify ────────────────────────────────────
    print("\n" + "=" * 80)
    print("Step 2: Reload database and verify pre-built indices")
    print("=" * 80, flush=True)

    t0 = time.perf_counter()
    db_v2 = load_database(new_db_dir)
    t_load = time.perf_counter() - t0
    print(f"  Loaded in {t_load:.1f}s", flush=True)
    print(f"  Reduced indices: {list(db_v2.reduced_indices.keys()) if db_v2.reduced_indices else 'None'}", flush=True)
    print(f"  Reduced flat: {db_v2.reduced_flat.shape if db_v2.reduced_flat is not None else 'None'}", flush=True)

    if db_v2.reduced_indices:
        for name, data in db_v2.reduced_indices.items():
            print(f"    {name}: offsets={data[0].shape}, entries={data[1].shape}, freqs={data[2].shape}"
                  + (f", seed_offsets={data[3].shape}" if len(data) > 3 else ""), flush=True)

    # ── Step 3: Search with pre-built indices ────────────────────────
    print("\n" + "=" * 80)
    print("Step 3: Search with pre-built indices (Numba + C scoring)")
    print("=" * 80, flush=True)

    query_ds = read_sequences(qf, "protein")
    base = dict(
        freq_percentile=99.5, phase_a_topk=200000, kmer_score_thresh=0,
        local_alignment=True, evalue_normalize=False, max_cands_per_query=8000,
        reduced_alphabet=True, reduced_k=5, spaced_seeds=["110011"],
    )

    configs = [
        ("3idx Numba scoring (pre-built indices)",
         {**base, "use_c_scoring": False, "use_c_sw": False}),
        ("3idx C scoring (pre-built indices)",
         {**base, "use_c_scoring": True, "use_c_sw": False}),
        ("3idx C scoring + C SW bw=20 (pre-built)",
         {**base, "use_c_scoring": True, "use_c_sw": True, "c_sw_band_width": 20}),
    ]

    for label, kwargs in configs:
        print(f"\n  >>> {label}", flush=True)
        t0 = time.perf_counter()
        results = search_kmer_index(
            db_v2, query_ds, threshold=0.1, top_k=500, **kwargs)
        elapsed = time.perf_counter() - t0
        roc1 = evaluate_roc1(results, metadata)
        vs = roc1 - 0.7942
        print(f"      Time: {elapsed:.1f}s  Aligned: {results.num_aligned}  "
              f"ROC1: {roc1:.4f} (vs MMseqs2: {vs:+.4f})", flush=True)

    print(f"\n  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")
    print(f"  Baseline (old db, Numba 3idx): 1555s, ROC1=0.808")

    # Cleanup
    print(f"\n  New database saved at: {new_db_dir}")


if __name__ == "__main__":
    main()
