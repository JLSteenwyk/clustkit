/*
 * Fused multi-index k-mer scoring: all indices in one pass per query.
 *
 * Instead of 5 separate OpenMP parallel regions + python union (77s + 37s),
 * processes ALL indices within a single parallel loop per query.
 *
 * Benefits:
 *   - ONE OpenMP parallel region (no 5x launch overhead)
 *   - ONE target_counts array per thread (better cache)
 *   - Combined scoring → ONE top-K selection (no union step!)
 *   - Per-index scores output alongside targets for ML features
 *
 * Build: gcc -O3 -march=native -fopenmp -shared -fPIC -o kmer_score_fused.so kmer_score_fused.c
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Comparison for descending sort */
static int cmp_desc64(const void* a, const void* b) {
    int64_t va = *(const int64_t*)a;
    int64_t vb = *(const int64_t*)b;
    return (vb > va) - (vb < va);
}

/* Score one query against one contiguous index (Phase A only) */
static void score_phase_a(
    const uint8_t* q_seq, int32_t q_len,
    int32_t k, int32_t alpha_size,
    const int64_t* kmer_offsets, const int64_t* kmer_entries,
    const int32_t* kmer_freqs, int32_t freq_thresh,
    int32_t num_db, int16_t* target_counts
) {
    int32_t num_kmers = q_len - k + 1;
    for (int32_t qpos = 0; qpos < num_kmers; qpos++) {
        int64_t kv = 0;
        int valid = 1;
        for (int32_t j = 0; j < k; j++) {
            uint8_t r = q_seq[qpos + j];
            if (r >= (uint8_t)alpha_size) { valid = 0; break; }
            kv = kv * alpha_size + r;
        }
        if (!valid || kmer_freqs[kv] > freq_thresh) continue;
        int64_t s = kmer_offsets[kv], e = kmer_offsets[kv + 1];
        for (int64_t h = s; h < e; h++) {
            int32_t tid = (int32_t)(kmer_entries[h] >> 32);
            if (target_counts[tid] < 32767) target_counts[tid]++;
        }
    }
}

/* Score one query against one spaced seed index (Phase A only) */
static void score_phase_a_spaced(
    const uint8_t* q_seq, int32_t q_len,
    const int32_t* seed_offsets, int32_t weight, int32_t span,
    int32_t alpha_size,
    const int64_t* kmer_offsets, const int64_t* kmer_entries,
    const int32_t* kmer_freqs, int32_t freq_thresh,
    int32_t num_db, int16_t* target_counts
) {
    int32_t num_seeds = q_len - span + 1;
    for (int32_t qpos = 0; qpos < num_seeds; qpos++) {
        int64_t kv = 0;
        int valid = 1;
        for (int32_t j = 0; j < weight; j++) {
            uint8_t r = q_seq[qpos + seed_offsets[j]];
            if (r >= (uint8_t)alpha_size) { valid = 0; break; }
            kv = kv * alpha_size + r;
        }
        if (!valid || kmer_freqs[kv] > freq_thresh) continue;
        int64_t s = kmer_offsets[kv], e = kmer_offsets[kv + 1];
        for (int64_t h = s; h < e; h++) {
            int32_t tid = (int32_t)(kmer_entries[h] >> 32);
            if (target_counts[tid] < 32767) target_counts[tid]++;
        }
    }
}


/*
 * Fused batch scoring: all 5 indices per query in ONE parallel loop.
 *
 * For each query:
 *   1. Score all 5 indices (Phase A only) into separate int16 arrays
 *   2. Combine: sum of all counts per target
 *   3. Select top-mc by combined score
 *   4. Output: target_ids + per-index scores (for ML features)
 *
 * Output layout:
 *   out_targets[qi * mc ... qi * mc + nc-1] = target ids
 *   out_counts[qi] = nc (number of candidates)
 *   out_per_idx[qi * mc * 5 + j * mc + c] = score from index j for candidate c
 *   out_n_indices[qi * mc + c] = number of indices that found candidate c
 */
void batch_score_fused_c(
    /* Query sequences */
    const uint8_t*  q_flat_std,     /* standard alphabet */
    const uint8_t*  q_flat_red,     /* reduced alphabet */
    const int64_t*  q_offsets,
    const int32_t*  q_lengths,
    int32_t         nq,
    int32_t         num_db,
    int32_t         mc,             /* max candidates per query */
    /* Index 0: std k=3 */
    int32_t k0, int32_t a0,
    const int64_t* ko0, const int64_t* ke0, const int32_t* kf0, int32_t ft0,
    /* Index 1: red k=4 */
    int32_t k1, int32_t a1,
    const int64_t* ko1, const int64_t* ke1, const int32_t* kf1, int32_t ft1,
    /* Index 2: red k=5 */
    int32_t k2, int32_t a2,
    const int64_t* ko2, const int64_t* ke2, const int32_t* kf2, int32_t ft2,
    /* Index 3: spaced seed 1 */
    const int32_t* so3, int32_t w3, int32_t sp3, int32_t a3,
    const int64_t* ko3, const int64_t* ke3, const int32_t* kf3, int32_t ft3,
    /* Index 4: spaced seed 2 */
    const int32_t* so4, int32_t w4, int32_t sp4, int32_t a4,
    const int64_t* ko4, const int64_t* ke4, const int32_t* kf4, int32_t ft4,
    /* Outputs */
    int32_t*        out_targets,    /* [nq * mc] */
    int32_t*        out_counts,     /* [nq] */
    float*          out_per_idx,    /* [nq * mc * 5] per-index scores */
    float*          out_n_indices,  /* [nq * mc] */
    float*          out_max_score,  /* [nq * mc] */
    float*          out_sum_score   /* [nq * mc] */
) {
    #pragma omp parallel
    {
        /* Thread-local workspace: 5 x num_db int16 arrays */
        int16_t* counts[5];
        for (int i = 0; i < 5; i++)
            counts[i] = (int16_t*)malloc(num_db * sizeof(int16_t));

        #pragma omp for schedule(dynamic, 1)
        for (int32_t qi = 0; qi < nq; qi++) {
            int32_t q_len = q_lengths[qi];
            const uint8_t* q_std = q_flat_std + q_offsets[qi];
            const uint8_t* q_red = q_flat_red + q_offsets[qi];

            /* Zero all count arrays */
            for (int i = 0; i < 5; i++)
                memset(counts[i], 0, num_db * sizeof(int16_t));

            /* Phase A for all 5 indices */
            score_phase_a(q_std, q_len, k0, a0, ko0, ke0, kf0, ft0, num_db, counts[0]);
            score_phase_a(q_red, q_len, k1, a1, ko1, ke1, kf1, ft1, num_db, counts[1]);
            score_phase_a(q_red, q_len, k2, a2, ko2, ke2, kf2, ft2, num_db, counts[2]);
            score_phase_a_spaced(q_red, q_len, so3, w3, sp3, a3, ko3, ke3, kf3, ft3, num_db, counts[3]);
            score_phase_a_spaced(q_red, q_len, so4, w4, sp4, a4, ko4, ke4, kf4, ft4, num_db, counts[4]);

            /* Combine: compute per-target combined score and find top-mc */
            /* Pack as (combined_score << 32 | target_id) for sorting */
            int32_t n_passing = 0;
            for (int32_t t = 0; t < num_db; t++) {
                int32_t total = 0;
                for (int i = 0; i < 5; i++) total += counts[i][t];
                if (total >= 2) n_passing++;
            }

            if (n_passing == 0) {
                out_counts[qi] = 0;
                continue;
            }

            /* Collect passing targets with combined scores */
            int64_t* packed = (int64_t*)malloc(n_passing * sizeof(int64_t));
            int32_t p = 0;
            for (int32_t t = 0; t < num_db; t++) {
                int32_t total = 0;
                for (int i = 0; i < 5; i++) total += counts[i][t];
                if (total >= 2) {
                    packed[p++] = ((int64_t)(uint32_t)total << 32) | (uint32_t)t;
                }
            }

            /* Sort descending, take top-mc */
            qsort(packed, n_passing, sizeof(int64_t), cmp_desc64);
            int32_t nc = n_passing < mc ? n_passing : mc;

            int32_t* row_targets = out_targets + (int64_t)qi * mc;
            float* row_pidx = out_per_idx + (int64_t)qi * mc * 5;
            float* row_nidx = out_n_indices + (int64_t)qi * mc;
            float* row_max = out_max_score + (int64_t)qi * mc;
            float* row_sum = out_sum_score + (int64_t)qi * mc;

            for (int32_t c = 0; c < nc; c++) {
                int32_t tid = (int32_t)(packed[c] & 0xFFFFFFFF);
                row_targets[c] = tid;

                float n_idx = 0, max_s = 0, sum_s = 0;
                for (int i = 0; i < 5; i++) {
                    float s = (float)counts[i][tid];
                    row_pidx[i * mc + c] = s;
                    if (s > 0) n_idx += 1.0f;
                    if (s > max_s) max_s = s;
                    sum_s += s;
                }
                row_nidx[c] = n_idx;
                row_max[c] = max_s;
                row_sum[c] = sum_s;
            }

            out_counts[qi] = nc;
            free(packed);
        }

        for (int i = 0; i < 5; i++) free(counts[i]);
    }
}
