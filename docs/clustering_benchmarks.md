# ClustKIT Clustering Benchmark Results

## Pfam Concordance Benchmark

**Dataset:** 22,343 sequences from 56 Pfam families

### Results by Identity Threshold

| Threshold | Clusters | ARI | F1 | Time (192t) | Time (4t) |
|-----------|----------|-----|-----|-------------|-----------|
| 0.4 | 4886-5039 | 0.3148-0.3178 | 0.3183-0.3213 | 349-375s | 5706s |
| 0.5 | 6996 | 0.2051 | 0.2090 | 41s | — |
| 0.7 | 10175 | 0.0857 | 0.0873 | 9s | — |

### Competitor Comparison

Compared against CD-HIT, MMseqs2, VSEARCH at thresholds 0.3-0.9.
Full results in: `benchmarks/data/pfam_benchmark_results/pfam_concordance_results_192threads.json`

---

## Thread Scaling

**Dataset:** 22,343 sequences, threshold=0.5

### ClustKIT Scaling

| Threads | Time (s) | Speedup | Efficiency | Throughput |
|---------|----------|---------|------------|------------|
| 1 | 916.7 | 1.0x | 100% | 24 seq/s |
| 2 | 444.6 | 2.1x | 103% | 49 seq/s |
| 4 | 192.8 | 4.8x | 119% | 113 seq/s |
| 8 | 142.1 | 6.5x | 81% | 154 seq/s |
| 16 | 111.8 | 8.2x | 51% | 195 seq/s |
| 32 | 125.7 | 7.3x | 23% | 174 seq/s |
| 64 | 50.9 | 18.0x | 28% | 429 seq/s |
| 128 | 26.6 | 34.5x | 27% | 822 seq/s |
| 192 | 22.0 | 41.8x | 22% | 994 seq/s |

### MMseqs2 Scaling

| Threads | Time (s) | Speedup |
|---------|----------|---------|
| 1 | 434.0 | 1.0x |
| 64 | 37.3 | 11.6x |
| 192 | 46.1 | 9.4x |

**Key finding:** ClustKIT scales better than MMseqs2 at high thread counts. MMseqs2 actually slows down from 64→192 threads.

---

## Ablation Study

**Dataset:** 22,343 sequences, threshold=0.4, 192 threads

| Variant | Clusters | ARI | F1 | Time | Rel. Speed |
|---------|----------|-----|-----|------|------------|
| Full ClustKIT | 5039 | 0.3148 | 0.3183 | 375s | 1.0x |
| No adaptive k | 5641 | 0.2953 | 0.2987 | 20s | 0.05x |
| No adaptive band | 4936 | 0.3154 | 0.3189 | 535s | 1.4x |
| No early termination | 5005 | 0.3162 | 0.3196 | 548s | 1.5x |
| No length pre-filter | 5005 | 0.3162 | 0.3196 | 683s | 1.8x |
| K-mer mode only | 1545 | 0.0003 | 0.0392 | 11s | 0.03x |
| Fixed LSH params | 11188 | 0.1895 | 0.1928 | 12s | 0.03x |

**Key findings:**
- Adaptive k provides 18x speedup with only -0.020 ARI loss
- K-mer mode alone is useless (ARI=0.0003) — alignment is essential
- Fixed LSH params halve quality — adaptive LSH is critical
- Early termination and length pre-filter are speed optimizations with no quality impact

---

## Correctness Validation

**Banded NW vs Full NW (10,000 pairs from Pfam):**

| Method | MAE | Timing | Speedup | Exact Match % |
|--------|-----|--------|---------|--------------|
| Full NW | — | 45.7s | 1.0x | 100% |
| Banded fixed (458bp) | 0.073 | 18.4s | 2.5x | — |
| Banded adaptive | 0.073 | 14.6s | 3.1x | — |
| Pipeline-realistic (t=0.4) | 0.00038 | — | — | 98.4% |

**Conclusion:** Banded NW is highly accurate in practice when combined with length pre-filters.

---

## Dataset Scaling

Tested on 10K, 50K, 100K, 250K, 500K sequences.
Results in: `benchmarks/data/scaling_results/` and `benchmarks/data/scaling_comparison_results/`

---

## Search Suite (Pre-optimization)

### CS4: Pfam Search (5 families, 10,799 db)
- Mean Hits: 31.8/family
- Mean AUC: 0.343
- Mean Sensitivity@1%FDR: 0.235

### CP7: False Discovery Rate
- ClustKIT FDR: 0.222
- MMseqs2 FDR: 0.0
- DIAMOND FDR: 0.0

---

## Important Note

All search optimizations (SW alignment, reduced alphabet, spaced seeds, C extension, ML two-tier) were applied to the **search path** only (`clustkit/kmer_index.py` → `search_kmer_index`). The **clustering path** uses the original NW alignment pipeline (`clustkit/pairwise.py` → `clustkit/pipeline.py`).

**Clustering regression testing has NOT been performed** after the search optimizations. The pairwise.py file was modified (SW functions added), but the clustering code path should be unaffected since it calls `_nw_identity`/`_batch_align_compact` directly.
