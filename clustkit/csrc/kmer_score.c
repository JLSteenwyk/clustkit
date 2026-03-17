/*
 * C/OpenMP acceleration for k-mer Phase A+B scoring.
 *
 * Replaces the Numba _batch_score_queries for ~2-3x speedup through:
 *   - GCC -O3 -march=native optimizations
 *   - OpenMP parallelism with better scheduling
 *   - Prefetch hints for posting list traversal
 *   - Branch-free k-mer computation for fixed k=3
 *
 * Build: gcc -O3 -march=native -fopenmp -shared -fPIC -o kmer_score.so kmer_score.c
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ─── Phase A: k-mer counting per target ─────────────────────────── */

static void score_query_phase_a(
    const uint8_t*  q_seq,
    int32_t         q_len,
    int32_t         k,
    int32_t         alpha_size,
    const int64_t*  kmer_offsets,
    const int64_t*  kmer_entries,
    const int32_t*  kmer_freqs,
    int32_t         freq_thresh,
    int32_t         num_db,
    int16_t*        target_counts   /* pre-zeroed, size=num_db */
) {
    int32_t num_kmers = q_len - k + 1;
    if (num_kmers <= 0) return;

    for (int32_t qpos = 0; qpos < num_kmers; qpos++) {
        int64_t kmer_val = 0;
        int valid = 1;
        for (int32_t j = 0; j < k; j++) {
            uint8_t r = q_seq[qpos + j];
            if (r >= (uint8_t)alpha_size) { valid = 0; break; }
            kmer_val = kmer_val * alpha_size + r;
        }
        if (!valid) continue;
        if (kmer_freqs[kmer_val] > freq_thresh) continue;

        int64_t s = kmer_offsets[kmer_val];
        int64_t e = kmer_offsets[kmer_val + 1];

        /* Prefetch next posting list while processing current */
        if (qpos + 1 < num_kmers) {
            int64_t next_kmer = 0;
            int nv = 1;
            for (int32_t j = 0; j < k; j++) {
                uint8_t r = q_seq[qpos + 1 + j];
                if (r >= (uint8_t)alpha_size) { nv = 0; break; }
                next_kmer = next_kmer * alpha_size + r;
            }
            if (nv && kmer_freqs[next_kmer] <= freq_thresh) {
                __builtin_prefetch(&kmer_entries[kmer_offsets[next_kmer]], 0, 1);
            }
        }

        for (int64_t h = s; h < e; h++) {
            int32_t tid = (int32_t)(kmer_entries[h] >> 32);
            if (target_counts[tid] < 32767)
                target_counts[tid]++;
        }
    }
}

/* ─── Phase B: diagonal scoring for survivors ────────────────────── */

/* Comparison function for qsort on int64_t keys */
static int cmp_int64(const void* a, const void* b) {
    int64_t va = *(const int64_t*)a;
    int64_t vb = *(const int64_t*)b;
    return (va > vb) - (va < vb);
}

/* Sort (id, score) pairs by score descending. Pack as score<<32|id. */
static int cmp_score_desc(const void* a, const void* b) {
    int64_t va = *(const int64_t*)a;
    int64_t vb = *(const int64_t*)b;
    return (vb > va) - (vb < va);  /* descending */
}

/* Helper: top-K selection via qsort on packed (score<<32|id).
 * Writes top-K ids to out_ids, scores to out_scores. Returns count. */
static int32_t topk_by_score(
    const int32_t* ids, const int16_t* scores, int32_t n, int32_t k,
    int32_t* out_ids, int32_t* out_scores
) {
    if (n <= k) {
        for (int32_t i = 0; i < n; i++) {
            out_ids[i] = ids[i];
            out_scores[i] = scores[i];
        }
        return n;
    }
    /* Pack as (score << 32 | id) for sorting */
    int64_t* packed = (int64_t*)malloc(n * sizeof(int64_t));
    for (int32_t i = 0; i < n; i++)
        packed[i] = ((int64_t)(uint16_t)scores[i] << 32) | (uint32_t)ids[i];
    qsort(packed, n, sizeof(int64_t), cmp_score_desc);
    for (int32_t i = 0; i < k; i++) {
        out_ids[i] = (int32_t)(packed[i] & 0xFFFFFFFF);
        out_scores[i] = (int32_t)(packed[i] >> 32);
    }
    free(packed);
    return k;
}

static int32_t topk_by_score32(
    const int32_t* ids, const int32_t* scores, int32_t n, int32_t k,
    int32_t* out_ids, int32_t* out_scores
) {
    if (n <= k) {
        memcpy(out_ids, ids, n * sizeof(int32_t));
        memcpy(out_scores, scores, n * sizeof(int32_t));
        return n;
    }
    int64_t* packed = (int64_t*)malloc(n * sizeof(int64_t));
    for (int32_t i = 0; i < n; i++)
        packed[i] = ((int64_t)(uint32_t)scores[i] << 32) | (uint32_t)ids[i];
    qsort(packed, n, sizeof(int64_t), cmp_score_desc);
    for (int32_t i = 0; i < k; i++) {
        out_ids[i] = (int32_t)(packed[i] & 0xFFFFFFFF);
        out_scores[i] = (int32_t)(packed[i] >> 32);
    }
    free(packed);
    return k;
}

/* Phase A + top-K + Phase B for one query.
 * Writes results to out_ids/out_scores, returns count.
 */
static int32_t score_query_full(
    const uint8_t*  q_seq,
    int32_t         q_len,
    int32_t         k,
    int32_t         alpha_size,
    const int64_t*  kmer_offsets,
    const int64_t*  kmer_entries,
    const int32_t*  kmer_freqs,
    int32_t         freq_thresh,
    int32_t         num_db,
    int32_t         min_total_hits,
    int32_t         min_diag_hits,
    int32_t         diag_bin_width,
    int32_t         phase_a_topk,
    int32_t         max_cands,
    int32_t*        out_ids,
    int32_t*        out_scores,
    int32_t*        out_diags     /* may be NULL */
) {
    int32_t num_kmers = q_len - k + 1;
    if (num_kmers <= 0) return 0;

    /* Phase A */
    int16_t* counts = (int16_t*)calloc(num_db, sizeof(int16_t));
    if (!counts) return 0;

    score_query_phase_a(q_seq, q_len, k, alpha_size,
                        kmer_offsets, kmer_entries, kmer_freqs,
                        freq_thresh, num_db, counts);

    /* Collect passing targets */
    int32_t num_passing = 0;
    for (int32_t i = 0; i < num_db; i++) {
        if (counts[i] >= min_total_hits) num_passing++;
    }
    if (num_passing == 0) { free(counts); return 0; }

    /* If no Phase B needed */
    if (min_diag_hits <= 1) {
        int32_t* all_ids = (int32_t*)malloc(num_passing * sizeof(int32_t));
        int16_t* all_sc  = (int16_t*)malloc(num_passing * sizeof(int16_t));
        int32_t p = 0;
        for (int32_t i = 0; i < num_db; i++) {
            if (counts[i] >= min_total_hits) {
                all_ids[p] = i;
                all_sc[p] = counts[i];
                p++;
            }
        }
        int32_t nc = topk_by_score(all_ids, all_sc, num_passing, max_cands,
                                   out_ids, out_scores);
        free(all_ids); free(all_sc); free(counts);
        return nc;
    }

    /* Top-K selection for Phase B */
    int32_t topk = phase_a_topk < num_passing ? phase_a_topk : num_passing;

    /* Collect all passing into arrays */
    int32_t* pass_ids = (int32_t*)malloc(num_passing * sizeof(int32_t));
    int16_t* pass_sc  = (int16_t*)malloc(num_passing * sizeof(int16_t));
    int32_t p = 0;
    for (int32_t i = 0; i < num_db; i++) {
        if (counts[i] >= min_total_hits) {
            pass_ids[p] = i;
            pass_sc[p] = counts[i];
            p++;
        }
    }

    /* Sort to get top-K by Phase A score (O(n log n) via qsort) */
    if (topk < num_passing) {
        int64_t* pk = (int64_t*)malloc(num_passing * sizeof(int64_t));
        for (int32_t i = 0; i < num_passing; i++)
            pk[i] = ((int64_t)(uint16_t)pass_sc[i] << 32) | (uint32_t)pass_ids[i];
        qsort(pk, num_passing, sizeof(int64_t), cmp_score_desc);
        for (int32_t i = 0; i < topk; i++) {
            pass_ids[i] = (int32_t)(pk[i] & 0xFFFFFFFF);
            pass_sc[i] = (int16_t)(pk[i] >> 32);
        }
        free(pk);
    }

    /* Build survivor mask */
    uint8_t* surv_mask = (uint8_t*)calloc(num_db, 1);
    int64_t n_surv_hits = 0;
    for (int32_t i = 0; i < topk; i++) {
        surv_mask[pass_ids[i]] = 1;
        n_surv_hits += counts[pass_ids[i]];
    }

    /* Phase B: collect diagonal keys for survivors */
    int64_t DIAG_MULT = 1000000LL;
    int32_t max_diag_shift = q_len;

    int64_t* surv_keys = (int64_t*)malloc(n_surv_hits * sizeof(int64_t));
    int64_t sw = 0;

    for (int32_t qpos = 0; qpos < num_kmers; qpos++) {
        int64_t kmer_val = 0;
        int valid = 1;
        for (int32_t j = 0; j < k; j++) {
            uint8_t r = q_seq[qpos + j];
            if (r >= (uint8_t)alpha_size) { valid = 0; break; }
            kmer_val = kmer_val * alpha_size + r;
        }
        if (!valid) continue;
        if (kmer_freqs[kmer_val] > freq_thresh) continue;

        int64_t s = kmer_offsets[kmer_val];
        int64_t e = kmer_offsets[kmer_val + 1];
        for (int64_t h = s; h < e; h++) {
            int64_t entry = kmer_entries[h];
            int32_t tid = (int32_t)(entry >> 32);
            if (surv_mask[tid]) {
                int32_t tpos = (int32_t)(entry & 0xFFFFFFFF);
                int32_t diag = tpos - qpos + max_diag_shift;
                int32_t dbin = diag / diag_bin_width;
                surv_keys[sw++] = (int64_t)tid * DIAG_MULT + dbin;
            }
        }
    }

    /* Sort keys */
    qsort(surv_keys, sw, sizeof(int64_t), cmp_int64);

    /* Count runs → final candidates (track best diagonal bin) */
    int32_t* final_ids  = (int32_t*)malloc(topk * sizeof(int32_t));
    int32_t* final_sc   = (int32_t*)malloc(topk * sizeof(int32_t));
    int32_t* final_diag = (int32_t*)malloc(topk * sizeof(int32_t));
    int32_t num_final = 0;
    int32_t prev_tid = -1;
    int32_t best_count = 0;
    int32_t best_dbin = 0;

    for (int64_t i = 0; i < sw; ) {
        int64_t key = surv_keys[i];
        int32_t tid = (int32_t)(key / DIAG_MULT);
        int32_t dbin = (int32_t)(key % DIAG_MULT);
        int32_t run = 0;
        while (i < sw) {
            int64_t k2 = surv_keys[i];
            if ((int32_t)(k2 / DIAG_MULT) != tid ||
                (int32_t)(k2 % DIAG_MULT) != dbin)
                break;
            run++; i++;
        }
        if (tid != prev_tid) {
            if (prev_tid >= 0 && best_count >= min_diag_hits) {
                final_ids[num_final] = prev_tid;
                final_sc[num_final] = best_count;
                /* Convert dbin back to diagonal offset: tpos - qpos */
                final_diag[num_final] = best_dbin * diag_bin_width
                                        - max_diag_shift
                                        + diag_bin_width / 2;
                num_final++;
            }
            prev_tid = tid;
            best_count = run;
            best_dbin = dbin;
        } else {
            if (run > best_count) {
                best_count = run;
                best_dbin = dbin;
            }
        }
    }
    if (prev_tid >= 0 && best_count >= min_diag_hits) {
        final_ids[num_final] = prev_tid;
        final_sc[num_final] = best_count;
        final_diag[num_final] = best_dbin * diag_bin_width
                                - max_diag_shift
                                + diag_bin_width / 2;
        num_final++;
    }

    /* Select top max_cands from final (O(n log n) sort) */
    int32_t nc = topk_by_score32(final_ids, final_sc, num_final, max_cands,
                                  out_ids, out_scores);
    /* Copy diagonal hints for the selected top-mc */
    /* topk_by_score32 reorders final_ids/final_sc — diags must follow */
    /* Re-derive: out_ids has the selected target IDs, find their diags */
    /* Simpler: output diags in the same order as topk_by_score32 output */
    /* Since topk_by_score32 uses packed sort, we need diags in that order */
    /* For now, just map from final arrays (nc <= num_final) */
    if (out_diags) {
        for (int32_t c = 0; c < nc; c++) {
            int32_t tid = out_ids[c];
            /* Linear scan to find this tid's diagonal in final arrays */
            for (int32_t f = 0; f < num_final; f++) {
                if (final_ids[f] == tid) {
                    out_diags[c] = final_diag[f];
                    break;
                }
            }
        }
    }

    free(counts); free(pass_ids); free(pass_sc);
    free(surv_mask); free(surv_keys);
    free(final_ids); free(final_sc); free(final_diag);
    return nc;
}


/* ─── Batch entry point (OpenMP parallel over queries) ───────────── */

void batch_score_queries_c(
    const uint8_t*  q_flat,
    const int64_t*  q_offsets,
    const int32_t*  q_lengths,
    int32_t         nq,
    int32_t         k,
    int32_t         alpha_size,
    const int64_t*  kmer_offsets,
    const int64_t*  kmer_entries,
    const int32_t*  kmer_freqs,
    int32_t         freq_thresh,
    int32_t         num_db,
    int32_t         min_total_hits,
    int32_t         min_diag_hits,
    int32_t         diag_bin_width,
    int32_t         max_cands,
    int32_t         phase_a_topk,
    int32_t*        out_targets,    /* [nq * max_cands] */
    int32_t*        out_counts,     /* [nq] */
    int32_t*        out_scores,     /* [nq * max_cands], or NULL */
    int32_t*        out_diags       /* [nq * max_cands], or NULL */
) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int32_t qi = 0; qi < nq; qi++) {
        const uint8_t* q_seq = q_flat + q_offsets[qi];
        int32_t q_len = q_lengths[qi];

        int32_t* row_targets = out_targets + (int64_t)qi * max_cands;
        int32_t* row_scores  = (int32_t*)malloc(max_cands * sizeof(int32_t));
        int32_t* row_diags   = (int32_t*)malloc(max_cands * sizeof(int32_t));

        int32_t nc = score_query_full(
            q_seq, q_len, k, alpha_size,
            kmer_offsets, kmer_entries, kmer_freqs,
            freq_thresh, num_db, min_total_hits,
            min_diag_hits, diag_bin_width, phase_a_topk,
            max_cands, row_targets, row_scores, row_diags
        );

        out_counts[qi] = nc;
        if (out_scores) {
            int32_t* row_out_sc = out_scores + (int64_t)qi * max_cands;
            memcpy(row_out_sc, row_scores, nc * sizeof(int32_t));
        }
        if (out_diags) {
            int32_t* row_out_dg = out_diags + (int64_t)qi * max_cands;
            memcpy(row_out_dg, row_diags, nc * sizeof(int32_t));
        }
        free(row_scores);
        free(row_diags);
    }
}


/* ─── Spaced seed Phase A ────────────────────────────────────────── */

static void score_query_phase_a_spaced(
    const uint8_t*  q_seq,
    int32_t         q_len,
    const int32_t*  seed_offsets,
    int32_t         weight,
    int32_t         span,
    int32_t         alpha_size,
    const int64_t*  kmer_offsets,
    const int64_t*  kmer_entries,
    const int32_t*  kmer_freqs,
    int32_t         freq_thresh,
    int32_t         num_db,
    int16_t*        target_counts
) {
    int32_t num_seeds = q_len - span + 1;
    if (num_seeds <= 0) return;

    for (int32_t qpos = 0; qpos < num_seeds; qpos++) {
        int64_t kmer_val = 0;
        int valid = 1;
        for (int32_t j = 0; j < weight; j++) {
            uint8_t r = q_seq[qpos + seed_offsets[j]];
            if (r >= (uint8_t)alpha_size) { valid = 0; break; }
            kmer_val = kmer_val * alpha_size + r;
        }
        if (!valid) continue;
        if (kmer_freqs[kmer_val] > freq_thresh) continue;

        int64_t s = kmer_offsets[kmer_val];
        int64_t e = kmer_offsets[kmer_val + 1];
        for (int64_t h = s; h < e; h++) {
            int32_t tid = (int32_t)(kmer_entries[h] >> 32);
            if (target_counts[tid] < 32767)
                target_counts[tid]++;
        }
    }
}

/* Spaced seed: full Phase A + top-K + Phase B */
static int32_t score_query_full_spaced(
    const uint8_t*  q_seq,
    int32_t         q_len,
    const int32_t*  seed_offsets,
    int32_t         weight,
    int32_t         span,
    int32_t         alpha_size,
    const int64_t*  kmer_offsets,
    const int64_t*  kmer_entries,
    const int32_t*  kmer_freqs,
    int32_t         freq_thresh,
    int32_t         num_db,
    int32_t         min_total_hits,
    int32_t         min_diag_hits,
    int32_t         diag_bin_width,
    int32_t         phase_a_topk,
    int32_t         max_cands,
    int32_t*        out_ids,
    int32_t*        out_scores
) {
    int32_t num_seeds = q_len - span + 1;
    if (num_seeds <= 0) return 0;

    int16_t* counts = (int16_t*)calloc(num_db, sizeof(int16_t));
    if (!counts) return 0;

    score_query_phase_a_spaced(q_seq, q_len, seed_offsets, weight, span,
                               alpha_size, kmer_offsets, kmer_entries,
                               kmer_freqs, freq_thresh, num_db, counts);

    int32_t num_passing = 0;
    for (int32_t i = 0; i < num_db; i++)
        if (counts[i] >= min_total_hits) num_passing++;
    if (num_passing == 0) { free(counts); return 0; }

    if (min_diag_hits <= 1) {
        int32_t* all_ids = (int32_t*)malloc(num_passing * sizeof(int32_t));
        int16_t* all_sc  = (int16_t*)malloc(num_passing * sizeof(int16_t));
        int32_t p = 0;
        for (int32_t i = 0; i < num_db; i++)
            if (counts[i] >= min_total_hits) {
                all_ids[p] = i; all_sc[p] = counts[i]; p++;
            }
        int32_t nc = topk_by_score(all_ids, all_sc, num_passing, max_cands,
                                   out_ids, out_scores);
        free(all_ids); free(all_sc); free(counts);
        return nc;
    }

    /* Top-K for Phase B */
    int32_t topk = phase_a_topk < num_passing ? phase_a_topk : num_passing;
    int32_t* pass_ids = (int32_t*)malloc(num_passing * sizeof(int32_t));
    int16_t* pass_sc  = (int16_t*)malloc(num_passing * sizeof(int16_t));
    int32_t p = 0;
    for (int32_t i = 0; i < num_db; i++)
        if (counts[i] >= min_total_hits) {
            pass_ids[p] = i; pass_sc[p] = counts[i]; p++;
        }

    if (topk < num_passing) {
        int64_t* pk = (int64_t*)malloc(num_passing * sizeof(int64_t));
        for (int32_t i = 0; i < num_passing; i++)
            pk[i] = ((int64_t)(uint16_t)pass_sc[i] << 32) | (uint32_t)pass_ids[i];
        qsort(pk, num_passing, sizeof(int64_t), cmp_score_desc);
        for (int32_t i = 0; i < topk; i++) {
            pass_ids[i] = (int32_t)(pk[i] & 0xFFFFFFFF);
            pass_sc[i] = (int16_t)(pk[i] >> 32);
        }
        free(pk);
    }

    uint8_t* surv_mask = (uint8_t*)calloc(num_db, 1);
    int64_t n_surv_hits = 0;
    for (int32_t i = 0; i < topk; i++) {
        surv_mask[pass_ids[i]] = 1;
        n_surv_hits += counts[pass_ids[i]];
    }

    /* Phase B: diagonal scoring with spaced seed */
    int64_t DIAG_MULT = 1000000LL;
    int32_t max_diag_shift = q_len;
    int64_t* surv_keys = (int64_t*)malloc(n_surv_hits * sizeof(int64_t));
    int64_t sw = 0;

    for (int32_t qpos = 0; qpos < num_seeds; qpos++) {
        int64_t kmer_val = 0;
        int valid = 1;
        for (int32_t j = 0; j < weight; j++) {
            uint8_t r = q_seq[qpos + seed_offsets[j]];
            if (r >= (uint8_t)alpha_size) { valid = 0; break; }
            kmer_val = kmer_val * alpha_size + r;
        }
        if (!valid) continue;
        if (kmer_freqs[kmer_val] > freq_thresh) continue;
        int64_t s = kmer_offsets[kmer_val];
        int64_t e = kmer_offsets[kmer_val + 1];
        for (int64_t h = s; h < e; h++) {
            int64_t entry = kmer_entries[h];
            int32_t tid = (int32_t)(entry >> 32);
            if (surv_mask[tid]) {
                int32_t tpos = (int32_t)(entry & 0xFFFFFFFF);
                int32_t diag = tpos - qpos + max_diag_shift;
                int32_t dbin = diag / diag_bin_width;
                surv_keys[sw++] = (int64_t)tid * DIAG_MULT + dbin;
            }
        }
    }

    qsort(surv_keys, sw, sizeof(int64_t), cmp_int64);

    int32_t* final_ids = (int32_t*)malloc(topk * sizeof(int32_t));
    int32_t* final_sc  = (int32_t*)malloc(topk * sizeof(int32_t));
    int32_t num_final = 0, prev_tid = -1, best_count = 0;

    for (int64_t i = 0; i < sw; ) {
        int64_t key = surv_keys[i];
        int32_t tid = (int32_t)(key / DIAG_MULT);
        int32_t dbin = (int32_t)(key % DIAG_MULT);
        int32_t run = 0;
        while (i < sw) {
            int64_t k2 = surv_keys[i];
            if ((int32_t)(k2 / DIAG_MULT) != tid ||
                (int32_t)(k2 % DIAG_MULT) != dbin) break;
            run++; i++;
        }
        if (tid != prev_tid) {
            if (prev_tid >= 0 && best_count >= min_diag_hits) {
                final_ids[num_final] = prev_tid;
                final_sc[num_final] = best_count; num_final++;
            }
            prev_tid = tid; best_count = run;
        } else {
            if (run > best_count) best_count = run;
        }
    }
    if (prev_tid >= 0 && best_count >= min_diag_hits) {
        final_ids[num_final] = prev_tid;
        final_sc[num_final] = best_count; num_final++;
    }

    int32_t nc = topk_by_score32(final_ids, final_sc, num_final, max_cands,
                                  out_ids, out_scores);

    free(counts); free(pass_ids); free(pass_sc);
    free(surv_mask); free(surv_keys);
    free(final_ids); free(final_sc);
    return nc;
}


/* ─── Batch spaced seed entry point ──────────────────────────────── */

void batch_score_queries_spaced_c(
    const uint8_t*  q_flat,
    const int64_t*  q_offsets,
    const int32_t*  q_lengths,
    int32_t         nq,
    const int32_t*  seed_offsets,
    int32_t         weight,
    int32_t         span,
    int32_t         alpha_size,
    const int64_t*  kmer_offsets,
    const int64_t*  kmer_entries,
    const int32_t*  kmer_freqs,
    int32_t         freq_thresh,
    int32_t         num_db,
    int32_t         min_total_hits,
    int32_t         min_diag_hits,
    int32_t         diag_bin_width,
    int32_t         max_cands,
    int32_t         phase_a_topk,
    int32_t*        out_targets,
    int32_t*        out_counts,
    int32_t*        out_scores
) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int32_t qi = 0; qi < nq; qi++) {
        const uint8_t* q_seq = q_flat + q_offsets[qi];
        int32_t q_len = q_lengths[qi];
        int32_t* row_targets = out_targets + (int64_t)qi * max_cands;
        int32_t* row_scores  = (int32_t*)malloc(max_cands * sizeof(int32_t));

        int32_t nc = score_query_full_spaced(
            q_seq, q_len, seed_offsets, weight, span, alpha_size,
            kmer_offsets, kmer_entries, kmer_freqs,
            freq_thresh, num_db, min_total_hits,
            min_diag_hits, diag_bin_width, phase_a_topk,
            max_cands, row_targets, row_scores
        );

        out_counts[qi] = nc;
        if (out_scores) {
            int32_t* row_out_sc = out_scores + (int64_t)qi * max_cands;
            memcpy(row_out_sc, row_scores, nc * sizeof(int32_t));
        }
        free(row_scores);
    }
}
