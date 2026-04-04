/*
 * C/OpenMP banded Smith-Waterman local alignment.
 *
 * Replaces Numba _batch_sw_compact_scored for ~2-4x speedup through:
 *   - GCC -O3 -march=native auto-vectorization
 *   - Pre-allocated thread-local workspace (no per-pair malloc)
 *   - OpenMP parallelism with dynamic scheduling
 *   - Cache-friendly row-toggling DP
 *
 * Build: gcc -O3 -march=native -fopenmp -shared -fPIC -o sw_align.so sw_align.c
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* Allow Python to set thread count before each batch call. */
void sw_set_num_threads(int32_t n) {
    omp_set_num_threads(n);
}

/* ─── Single-pair banded SW ──────────────────────────────────────── */

static void sw_align_one(
    const uint8_t*  seq_a,
    int32_t         len_a,
    const uint8_t*  seq_b,
    int32_t         len_b,
    int32_t         band_width,
    int32_t         diag_hint,    /* offset from main diagonal (0 = main) */
    const int8_t*   sub_matrix,   /* 20x20 row-major */
    /* workspace: caller provides pre-allocated arrays of size (cols) */
    int32_t*        H0,           /* H previous row */
    int32_t*        H1,           /* H current row */
    int32_t*        E0,           /* E previous row */
    int32_t*        E1,           /* E current row */
    int32_t*        Hm0,          /* match count previous row */
    int32_t*        Hm1,          /* match count current row */
    int32_t*        Em0,          /* E match count previous row */
    int32_t*        Em1,          /* E match count current row */
    float*          out_identity,
    int32_t*        out_score
) {
    const int32_t GAP_OPEN = -11;
    const int32_t GAP_EXTEND = -1;
    const int32_t NEG_INF = -1000000;

    if (len_a == 0 || len_b == 0) {
        *out_identity = 0.0f;
        *out_score = 0;
        return;
    }

    int32_t bw = band_width;
    int32_t max_ab = len_a > len_b ? len_a : len_b;
    if (bw <= 0 || max_ab <= 50)
        bw = max_ab;

    int32_t cols = len_b + 1;
    int32_t shorter = len_a < len_b ? len_a : len_b;

    /* Initialize first row to zeros (local alignment) */
    for (int32_t j = 0; j < cols; j++) {
        H0[j] = 0; E0[j] = NEG_INF; Hm0[j] = 0; Em0[j] = 0;
        H1[j] = 0; E1[j] = NEG_INF; Hm1[j] = 0; Em1[j] = 0;
    }

    int32_t max_score = 0;
    int32_t max_matches = 0;

    /* Pointers for row toggling */
    int32_t *Hp = H0, *Hc = H1;
    int32_t *Ep = E0, *Ec = E1;
    int32_t *Hmp = Hm0, *Hmc = Hm1;
    int32_t *Emp = Em0, *Emc = Em1;

    for (int32_t i = 1; i <= len_a; i++) {
        /* Center band on hinted diagonal: j = i + diag_hint */
        int32_t j_center = i + diag_hint;
        int32_t j_start = j_center - bw;
        if (j_start < 1) j_start = 1;
        int32_t j_end = j_center + bw + 1;
        if (j_end > cols) j_end = cols;

        /* Skip row entirely if band is outside matrix */
        if (j_start >= cols || j_end <= 1) continue;

        /* Boundary: left of band (must be within array bounds) */
        if (j_start == 1) {
            Hc[0] = 0; Hmc[0] = 0;
        } else if (j_start - 1 < cols) {
            Hc[j_start - 1] = 0; Hmc[j_start - 1] = 0;
        }

        /* Boundary: right of previous row's band */
        int32_t prev_j_center = (i - 1) + diag_hint;
        int32_t prev_j_end = prev_j_center + bw + 1;
        if (prev_j_end > cols) prev_j_end = cols;
        if (j_end > prev_j_end && prev_j_end > 0 && prev_j_end < cols) {
            Hp[prev_j_end] = 0; Ep[prev_j_end] = NEG_INF;
            Hmp[prev_j_end] = 0; Emp[prev_j_end] = 0;
        }

        int32_t curr_F = NEG_INF;
        int32_t curr_Fm = 0;

        uint8_t a_res = seq_a[i - 1];

        for (int32_t j = j_start; j < j_end; j++) {
            uint8_t b_res = seq_b[j - 1];
            int32_t is_match = (a_res == b_res) ? 1 : 0;

            /* Substitution score */
            int32_t s;
            if (a_res < 20 && b_res < 20)
                s = (int32_t)sub_matrix[a_res * 20 + b_res];
            else
                s = -4;

            /* Diagonal */
            int32_t diag = Hp[j - 1] + s;
            int32_t diag_m = Hmp[j - 1] + is_match;

            /* E: gap in seq_b (vertical) */
            int32_t e_ext = Ep[j] + GAP_EXTEND;
            int32_t e_opn = Hp[j] + GAP_OPEN;
            int32_t e_val, e_m;
            if (e_ext >= e_opn) {
                e_val = e_ext; e_m = Emp[j];
            } else {
                e_val = e_opn; e_m = Hmp[j];
            }
            Ec[j] = e_val; Emc[j] = e_m;

            /* F: gap in seq_a (horizontal) */
            int32_t f_ext = (j > j_start) ? (curr_F + GAP_EXTEND) : NEG_INF;
            int32_t f_ext_m = curr_Fm;
            int32_t f_opn = Hc[j - 1] + GAP_OPEN;
            int32_t f_opn_m = Hmc[j - 1];
            if (f_ext >= f_opn) {
                curr_F = f_ext; curr_Fm = f_ext_m;
            } else {
                curr_F = f_opn; curr_Fm = f_opn_m;
            }

            /* SW recurrence: max(0, diag, E, F) */
            int32_t best = 0, best_m = 0;
            if (diag > best)  { best = diag;  best_m = diag_m; }
            if (e_val > best) { best = e_val;  best_m = e_m; }
            if (curr_F > best) { best = curr_F; best_m = curr_Fm; }

            Hc[j] = best;
            Hmc[j] = best_m;

            if (best > max_score) {
                max_score = best;
                max_matches = best_m;
            }
        }

        /* Toggle rows */
        int32_t *tmp;
        tmp = Hp; Hp = Hc; Hc = tmp;
        tmp = Ep; Ep = Ec; Ec = tmp;
        tmp = Hmp; Hmp = Hmc; Hmc = tmp;
        tmp = Emp; Emp = Emc; Emc = tmp;
    }

    if (max_score <= 0) {
        *out_identity = 0.0f;
        *out_score = 0;
    } else {
        *out_identity = (float)max_matches / (float)shorter;
        *out_score = max_score;
    }
}


/* ─── Batch SW alignment (OpenMP parallel over pairs) ────────────── */

void batch_sw_align_c(
    const int32_t*  pairs,          /* [M * 2] merged indices */
    const uint8_t*  flat_sequences,
    const int64_t*  offsets,
    const int32_t*  lengths,
    int32_t         M,
    int32_t         band_width,
    const int8_t*   sub_matrix,     /* 20*20 = 400 int8 */
    float           threshold,
    const int32_t*  diag_hints,     /* [M] diagonal offsets, or NULL for main diag */
    float*          out_sims,       /* [M] */
    int32_t*        out_scores,     /* [M] */
    uint8_t*        out_mask        /* [M] */
) {
    /* Find max sequence length for workspace allocation */
    int32_t max_len = 0;
    for (int32_t idx = 0; idx < M; idx++) {
        int32_t i = pairs[idx * 2];
        int32_t j = pairs[idx * 2 + 1];
        int32_t li = lengths[i];
        int32_t lj = lengths[j];
        int32_t ml = li > lj ? li : lj;
        if (ml > max_len) max_len = ml;
    }
    int32_t ws_cols = max_len + 2;  /* +2 for safety */

    #pragma omp parallel
    {
        /* Thread-local workspace — allocated once per thread */
        int32_t *H0  = (int32_t*)malloc(ws_cols * sizeof(int32_t));
        int32_t *H1  = (int32_t*)malloc(ws_cols * sizeof(int32_t));
        int32_t *E0  = (int32_t*)malloc(ws_cols * sizeof(int32_t));
        int32_t *E1  = (int32_t*)malloc(ws_cols * sizeof(int32_t));
        int32_t *Hm0 = (int32_t*)malloc(ws_cols * sizeof(int32_t));
        int32_t *Hm1 = (int32_t*)malloc(ws_cols * sizeof(int32_t));
        int32_t *Em0 = (int32_t*)malloc(ws_cols * sizeof(int32_t));
        int32_t *Em1 = (int32_t*)malloc(ws_cols * sizeof(int32_t));

        #pragma omp for schedule(dynamic, 64)
        for (int32_t idx = 0; idx < M; idx++) {
            int32_t i = pairs[idx * 2];
            int32_t j = pairs[idx * 2 + 1];
            int32_t len_i = lengths[i];
            int32_t len_j = lengths[j];

            if (len_i == 0 || len_j == 0) {
                out_sims[idx] = 0.0f;
                out_scores[idx] = 0;
                out_mask[idx] = 0;
                continue;
            }

            const uint8_t* seq_i = flat_sequences + offsets[i];
            const uint8_t* seq_j = flat_sequences + offsets[j];

            int32_t hint = diag_hints ? diag_hints[idx] : 0;
            /* Clamp extreme diagonal hints to prevent very wide effective
               bands. Hints beyond ±shorter_len are unreliable. */
            int32_t shorter_len = len_i < len_j ? len_i : len_j;
            if (hint > shorter_len) hint = 0;
            if (hint < -shorter_len) hint = 0;
            float identity;
            int32_t score;

            /* Ensure shorter sequence is seq_a (fewer DP rows).
               When swapping, negate the diagonal hint. */
            if (len_i <= len_j) {
                sw_align_one(seq_i, len_i, seq_j, len_j, band_width, hint,
                             sub_matrix, H0, H1, E0, E1, Hm0, Hm1, Em0, Em1,
                             &identity, &score);
            } else {
                sw_align_one(seq_j, len_j, seq_i, len_i, band_width, -hint,
                             sub_matrix, H0, H1, E0, E1, Hm0, Hm1, Em0, Em1,
                             &identity, &score);
            }

            out_sims[idx] = identity;
            out_scores[idx] = score;
            out_mask[idx] = (score > 0) ? 1 : 0;
        }

        free(H0); free(H1); free(E0); free(E1);
        free(Hm0); free(Hm1); free(Em0); free(Em1);
    }
}
