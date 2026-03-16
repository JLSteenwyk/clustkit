#!/usr/bin/env python3
"""Two-tier ML with rich features: n_indices, reduced k-mer sharing, composition.

The previous ML model had only 4 weak features (shared_k3, lengths).
This version adds the features that actually predict SW alignment quality:
  - n_indices_found: how many of the 5 indices found this pair (1-5)
  - shared_k3: standard alphabet k=3 sharing
  - shared_red_k5: reduced alphabet k=5 sharing
  - comp_cosine: actual AA composition cosine similarity
  - length_ratio, length_diff, query_len, target_len
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import njit, prange, int32, int64, float32

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, read_fasta, write_fasta, RankedHit,
)

import numba
numba.set_num_threads(8)

from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import (
    build_kmer_index, compute_freq_threshold,
    _batch_score_queries, _batch_score_queries_idf,
    _batch_score_queries_spaced, build_kmer_index_spaced,
    REDUCED_ALPHA, REDUCED_ALPHA_SIZE,
    _remap_flat, _build_query_kmer_sets,
)
from clustkit.pairwise import _batch_sw_compact_scored, BLOSUM62
from clustkit.search import (
    _merge_sequences_for_alignment, _remap_pairs_to_merged, _collect_top_k_hits,
)


# ──────────────────────────────────────────────────────────────────────
# Rich feature computation
# ──────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _compute_rich_features(
    pairs, n_indices_arr,
    q_kmer_sets_std, q_kmer_sets_red,
    q_flat, q_offsets, q_lengths,
    t_flat, t_offsets, t_lengths,
):
    """Compute 8 features per candidate pair.

    Returns (n_pairs, 8) float32:
      [n_indices, shared_k3, shared_red_k5, comp_cosine,
       length_ratio, length_diff, query_len, target_len]
    """
    m = len(pairs)
    features = np.empty((m, 8), dtype=np.float32)

    for idx in prange(m):
        qi = pairs[idx, 0]
        ti = pairs[idx, 1]

        q_len = float32(q_lengths[qi])
        t_len = float32(t_lengths[ti])
        shorter = min(q_len, t_len)
        longer = max(q_len, t_len)

        # Feature 0: n_indices_found
        features[idx, 0] = float32(n_indices_arr[idx])

        # Feature 1: shared k=3 (standard alphabet, alpha=20)
        t_start = int64(t_offsets[ti])
        t_length = int32(t_lengths[ti])
        shared_k3 = int32(0)
        for pos in range(t_length - 3 + 1):
            kmer_val = int64(0)
            valid = True
            for j in range(3):
                r = t_flat[t_start + pos + j]
                if r >= 20:
                    valid = False
                    break
                kmer_val = kmer_val * int64(20) + int64(r)
            if valid and q_kmer_sets_std[qi, kmer_val]:
                shared_k3 += int32(1)
        features[idx, 1] = float32(shared_k3)

        # Feature 2: shared reduced k=5 (alpha=9)
        shared_red = int32(0)
        for pos in range(t_length - 5 + 1):
            kmer_val = int64(0)
            valid = True
            for j in range(5):
                r = t_flat[t_start + pos + j]
                if r >= 9:
                    valid = False
                    break
                kmer_val = kmer_val * int64(9) + int64(r)
            if valid and q_kmer_sets_red[qi, kmer_val]:
                shared_red += int32(1)
        features[idx, 2] = float32(shared_red)

        # Feature 3: composition cosine similarity
        q_freq = np.zeros(20, dtype=np.float32)
        q_start = int64(q_offsets[qi])
        for pos in range(int32(q_lengths[qi])):
            aa = q_flat[q_start + pos]
            if aa < 20:
                q_freq[aa] += float32(1.0)
        t_freq = np.zeros(20, dtype=np.float32)
        for pos in range(t_length):
            aa = t_flat[t_start + pos]
            if aa < 20:
                t_freq[aa] += float32(1.0)
        dot = float32(0)
        q_norm = float32(0)
        t_norm = float32(0)
        for a in range(20):
            dot += q_freq[a] * t_freq[a]
            q_norm += q_freq[a] * q_freq[a]
            t_norm += t_freq[a] * t_freq[a]
        if q_norm > 0 and t_norm > 0:
            features[idx, 3] = dot / (q_norm ** float32(0.5) * t_norm ** float32(0.5))
        else:
            features[idx, 3] = float32(0)

        # Features 4-7: length-based
        features[idx, 4] = shorter / longer if longer > 0 else float32(0)
        features[idx, 5] = abs(q_len - t_len)
        features[idx, 6] = q_len
        features[idx, 7] = t_len

    return features


# ──────────────────────────────────────────────────────────────────────
# ROC1 evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_roc1(results_hits, metadata):
    domain_info = metadata["domain_info"]
    family_sizes = metadata["family_sizes_in_db"]
    query_sids = set(metadata["query_sids"])
    hits_by_query = defaultdict(list)
    for qhits in results_hits:
        for h in qhits:
            hits_by_query[h.query_id].append((h.target_id, h.score if h.score != 0 else h.identity))
    roc1_values = []
    for qid in query_sids:
        q_info = domain_info.get(qid)
        if q_info is None:
            continue
        fam_key = str(q_info["family"])
        total_tp = family_sizes.get(fam_key, 1) - 1
        if total_tp <= 0:
            continue
        query_hits = hits_by_query.get(qid, [])
        query_hits.sort(key=lambda x: -x[1])
        ranked = []
        for tid, score in query_hits:
            label = classify_hit(qid, tid, domain_info)
            if label != "IGNORE":
                ranked.append(RankedHit(target_id=tid, score=score, label=label))
        roc1_values.append(compute_roc_n(ranked, 1, total_tp))
    return float(np.mean(roc1_values)) if roc1_values else 0.0


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")

    with open(scop_dir / "metadata.json") as f:
        full_metadata = json.load(f)
    all_query_sids = full_metadata["query_sids"]
    random.seed(42)
    query_sids = sorted(random.sample(all_query_sids, min(2000, len(all_query_sids))))
    all_seqs = dict(read_fasta(scop_dir / "queries.fasta"))
    sub_seqs = [(sid, all_seqs[sid]) for sid in query_sids if sid in all_seqs]
    query_fasta = str(out_dir / "queries_subset.fasta")
    write_fasta(sub_seqs, query_fasta)
    metadata = dict(full_metadata)
    metadata["query_sids"] = query_sids

    print("Loading database...", flush=True)
    db_index = load_database(out_dir / "clustkit_db")
    query_ds = read_sequences(query_fasta, "protein")
    db_ds = db_index.dataset
    nq = query_ds.num_sequences
    nd = db_ds.num_sequences

    q_flat = query_ds.flat_sequences
    q_off = query_ds.offsets.astype(np.int64)
    q_lens = query_ds.lengths.astype(np.int32)
    db_flat = db_ds.flat_sequences
    t_off = db_ds.offsets.astype(np.int64)
    t_lens = db_ds.lengths.astype(np.int32)
    mc = 8000
    topk = 200000

    print(f"Loaded {nq} queries, {nd} database sequences\n", flush=True)

    # ── Step 1: Generate candidates, tracking which index found each ──
    print("=" * 100)
    print("Step 1: Candidate generation (tracking per-index membership)")
    print("=" * 100, flush=True)

    freq_thresh = compute_freq_threshold(db_index.kmer_freqs, nd, 99.5)
    k = int32(db_index.params.get("kmer_index_k", db_index.params["kmer_size"]))

    # Helper: run one index, return packed pairs
    def run_index(label, score_fn, *args):
        t0 = time.perf_counter()
        out_t = np.empty((nq, mc), dtype=np.int32)
        out_c = np.zeros(nq, dtype=np.int32)
        score_fn(*args, out_t, out_c)
        total = int(out_c.sum())
        pairs = np.empty((total, 2), dtype=np.int32)
        p = 0
        for qi in range(nq):
            nc = int(out_c[qi])
            if nc > 0:
                pairs[p:p+nc, 0] = qi
                pairs[p:p+nc, 1] = out_t[qi, :nc]
                p += nc
        packed = pairs[:, 0].astype(np.int64) * nd + pairs[:, 1].astype(np.int64)
        print(f"  {label}: {total} candidates ({time.perf_counter()-t0:.0f}s)", flush=True)
        return packed

    # Standard k=3 with IDF
    std_idf = np.log2(np.maximum(
        np.float32(nd) / np.maximum(db_index.kmer_freqs.astype(np.float32), 1.0), 1.0,
    )).astype(np.float32)

    p_std = run_index("Standard k=3 IDF", _batch_score_queries_idf,
        q_flat, q_off, q_lens, k, int32(20),
        db_index.kmer_offsets, db_index.kmer_entries, db_index.kmer_freqs,
        freq_thresh, int32(nd), int32(2), int32(2), int32(10),
        int32(mc), int32(topk), std_idf)

    # Reduced k=4
    red_q_flat = _remap_flat(q_flat, REDUCED_ALPHA, len(q_flat))
    red_db_flat = _remap_flat(db_flat, REDUCED_ALPHA, len(db_flat))

    red4_off, red4_ent, red4_freq = build_kmer_index(
        red_db_flat, db_ds.offsets, db_ds.lengths, 4, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    red4_ft = compute_freq_threshold(red4_freq, nd, 99.5)
    p_red4 = run_index("Reduced k=4", _batch_score_queries,
        red_q_flat, q_off, q_lens, int32(4), int32(REDUCED_ALPHA_SIZE),
        red4_off, red4_ent, red4_freq, red4_ft,
        int32(nd), int32(2), int32(2), int32(10), int32(mc), int32(topk))

    # Reduced k=5
    red5_off, red5_ent, red5_freq = build_kmer_index(
        red_db_flat, db_ds.offsets, db_ds.lengths, 5, "protein", alpha_size=REDUCED_ALPHA_SIZE)
    red5_ft = compute_freq_threshold(red5_freq, nd, 99.5)
    p_red5 = run_index("Reduced k=5", _batch_score_queries,
        red_q_flat, q_off, q_lens, int32(5), int32(REDUCED_ALPHA_SIZE),
        red5_off, red5_ent, red5_freq, red5_ft,
        int32(nd), int32(2), int32(2), int32(10), int32(mc), int32(topk))

    # Spaced seeds
    sp1_data = build_kmer_index_spaced(
        red_db_flat, db_ds.offsets, db_ds.lengths, "11011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    sp1_off, sp1_ent, sp1_freq, sp1_so, sp1_w, sp1_span = sp1_data
    sp1_ft = compute_freq_threshold(sp1_freq, nd, 99.5)
    p_sp1 = run_index("Spaced 11011", _batch_score_queries_spaced,
        red_q_flat, q_off, q_lens, sp1_so, int32(sp1_w), int32(sp1_span),
        int32(REDUCED_ALPHA_SIZE), sp1_off, sp1_ent, sp1_freq, sp1_ft,
        int32(nd), int32(2), int32(2), int32(10), int32(mc), int32(topk))

    sp2_data = build_kmer_index_spaced(
        red_db_flat, db_ds.offsets, db_ds.lengths, "110011", "protein", alpha_size=REDUCED_ALPHA_SIZE)
    sp2_off, sp2_ent, sp2_freq, sp2_so, sp2_w, sp2_span = sp2_data
    sp2_ft = compute_freq_threshold(sp2_freq, nd, 99.5)
    p_sp2 = run_index("Spaced 110011", _batch_score_queries_spaced,
        red_q_flat, q_off, q_lens, sp2_so, int32(sp2_w), int32(sp2_span),
        int32(REDUCED_ALPHA_SIZE), sp2_off, sp2_ent, sp2_freq, sp2_ft,
        int32(nd), int32(2), int32(2), int32(10), int32(mc), int32(topk))

    # Union with index membership counting
    print("\n  Computing n_indices_found per pair...", flush=True)
    all_packed = np.concatenate([p_std, p_red4, p_red5, p_sp1, p_sp2])
    unique_packed, inverse, counts = np.unique(all_packed, return_inverse=True, return_counts=True)

    candidate_pairs = np.empty((len(unique_packed), 2), dtype=np.int32)
    candidate_pairs[:, 0] = (unique_packed // nd).astype(np.int32)
    candidate_pairs[:, 1] = (unique_packed % nd).astype(np.int32)
    n_indices_arr = counts.astype(np.int32)  # how many indices found each pair

    print(f"  Total: {len(candidate_pairs)} unique pairs", flush=True)
    print(f"  Index distribution: "
          f"1={np.sum(n_indices_arr==1)} "
          f"2={np.sum(n_indices_arr==2)} "
          f"3={np.sum(n_indices_arr==3)} "
          f"4={np.sum(n_indices_arr==4)} "
          f"5={np.sum(n_indices_arr==5)}", flush=True)

    # ── Step 2: Compute rich features ────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 2: Rich feature computation (8 features)")
    print("=" * 100, flush=True)

    # Build k-mer sets: standard k=3 (alpha=20) and reduced k=5 (alpha=9)
    t0 = time.perf_counter()
    q_kmer_std = _build_query_kmer_sets(q_flat, q_off, q_lens, int32(3), int32(20))
    print(f"  Query k-mer sets (std k=3): {q_kmer_std.shape} ({time.perf_counter()-t0:.1f}s)", flush=True)

    t0 = time.perf_counter()
    q_kmer_red = _build_query_kmer_sets(red_q_flat, q_off, q_lens, int32(5), int32(9))
    print(f"  Query k-mer sets (red k=5): {q_kmer_red.shape} ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Target sequences in reduced alphabet for shared_red_k5
    t0 = time.perf_counter()
    features = _compute_rich_features(
        candidate_pairs, n_indices_arr,
        q_kmer_std, q_kmer_red,
        q_flat, q_off, q_lens,
        red_db_flat, t_off, t_lens,
    )
    feat_time = time.perf_counter() - t0
    print(f"  Features computed: {features.shape} ({feat_time:.1f}s)", flush=True)
    feat_names = ["n_indices", "shared_k3", "shared_red_k5", "comp_cosine",
                  "len_ratio", "len_diff", "query_len", "target_len"]

    # ── Step 3: SW alignment (labels) ────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 3: SW alignment (ground truth)")
    print("=" * 100, flush=True)

    merged = _merge_sequences_for_alignment(query_ds, db_ds)
    merged_pairs = _remap_pairs_to_merged(candidate_pairs, merged["nq"])
    sort_key = np.minimum(merged_pairs[:, 0], merged_pairs[:, 1])
    sort_order = np.argsort(sort_key, kind="mergesort")
    merged_pairs = merged_pairs[sort_order]
    candidate_pairs = candidate_pairs[sort_order]
    features = features[sort_order]
    n_indices_arr = n_indices_arr[sort_order]

    all_lengths = np.concatenate([query_ds.lengths, db_ds.lengths])
    band_width = max(20, int(np.percentile(all_lengths, 95) * 0.3))

    t0 = time.perf_counter()
    sims, raw_scores, aln_mask = _batch_sw_compact_scored(
        merged_pairs, merged["flat_sequences"], merged["offsets"],
        merged["lengths"], np.float32(0.1), int32(band_width), BLOSUM62,
    )
    aln_time = time.perf_counter() - t0
    print(f"  Aligned {len(candidate_pairs)} pairs in {aln_time:.1f}s", flush=True)

    # ── Step 4: Train RF model ───────────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 4: Train RandomForest on rich features")
    print("=" * 100, flush=True)

    query_ids = candidate_pairs[:, 0]
    unique_queries = np.unique(query_ids)
    np.random.seed(42)
    np.random.shuffle(unique_queries)
    split = int(len(unique_queries) * 0.8)
    train_queries = set(unique_queries[:split].tolist())
    train_mask = np.array([int(qi) in train_queries for qi in query_ids])

    X_train, X_test = features[train_mask], features[~train_mask]
    y_train, y_test = raw_scores[train_mask], raw_scores[~train_mask]

    from sklearn.ensemble import RandomForestRegressor

    t0 = time.perf_counter()
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = rf.predict(X_test)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred)**2)))

    print(f"  Trained in {train_time:.1f}s", flush=True)
    print(f"  Pearson r: {corr:.4f}  MAE: {mae:.2f}  RMSE: {rmse:.2f}", flush=True)

    # Feature importance (permutation)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(rf, X_test[:50000], y_test[:50000], n_repeats=3, random_state=42)
    print(f"\n  Feature importance:", flush=True)
    for name, imp in sorted(zip(feat_names, perm.importances_mean), key=lambda x: -x[1]):
        print(f"    {name:15s}: {imp:.4f}", flush=True)

    # ── Step 5: Two-tier test ────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Step 5: Two-tier ML alignment")
    print("=" * 100, flush=True)

    predicted = rf.predict(features)
    predict_time_est = feat_time + 10  # features + inference

    # Baseline ROC1
    passing = raw_scores > 0
    hits_base = _collect_top_k_hits(
        candidate_pairs[passing], sims[passing], nq, 500,
        query_ds, db_ds, passing_scores=raw_scores[passing].astype(np.float32))
    baseline_roc1 = evaluate_roc1(hits_base, metadata)
    print(f"\n  Baseline: ROC1={baseline_roc1:.4f}, align_time={aln_time:.1f}s\n", flush=True)

    print(f"  {'Buffer N':>10s} {'Pairs':>12s} {'Align(s)':>10s} "
          f"{'Speedup':>8s} {'ROC1':>7s} {'vs MMseqs2':>10s}", flush=True)
    print("  " + "-" * 65, flush=True)

    for N in [500, 1000, 2000, 3000, 5000, 8000, 12000]:
        qi_col = candidate_pairs[:, 0]
        keep = np.zeros(len(candidate_pairs), dtype=np.bool_)
        for qi in range(nq):
            mask = qi_col == qi
            if mask.sum() == 0:
                continue
            idx = np.where(mask)[0]
            if len(idx) <= N:
                keep[idx] = True
            else:
                top = np.argsort(-predicted[idx])[:N]
                keep[idx[top]] = True

        sel = candidate_pairs[keep]
        sel_merged = _remap_pairs_to_merged(sel, merged["nq"])
        sk = np.minimum(sel_merged[:, 0], sel_merged[:, 1])
        so = np.argsort(sk, kind="mergesort")
        sel_merged = sel_merged[so]
        sel = sel[so]

        t0 = time.perf_counter()
        s_sims, s_scores, s_mask = _batch_sw_compact_scored(
            sel_merged, merged["flat_sequences"], merged["offsets"],
            merged["lengths"], np.float32(0.1), int32(band_width), BLOSUM62)
        at = time.perf_counter() - t0

        p = s_scores > 0
        hits = _collect_top_k_hits(
            sel[p], s_sims[p], nq, 500, query_ds, db_ds,
            passing_scores=s_scores[p].astype(np.float32))
        roc1 = evaluate_roc1(hits, metadata)

        speedup = aln_time / at
        vs = roc1 - 0.7942
        print(f"  {N:10d} {len(sel):12d} {at:9.1f}s "
              f"{speedup:7.2f}x {roc1:7.4f} {vs:+10.4f}", flush=True)

    print(f"\n  Feature computation: {feat_time:.1f}s")
    print(f"  Reference: MMseqs2=0.7942  DIAMOND=0.7963")
    print(f"  Previous model (4 features): r=0.966")


if __name__ == "__main__":
    main()
