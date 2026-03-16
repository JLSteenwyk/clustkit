# ClustKIT Architecture

## Search Pipeline

```
Query sequences (FASTA)
         │
         ▼
┌────────────────────────────────────────────────┐
│  Stage 1: Candidate Generation (C/OpenMP)      │
│                                                │
│  Standard k=3 index ──┐                        │
│  Reduced k=4 index ───┤                        │
│  Reduced k=5 index ───┼──→ Union + deduplicate │
│  Spaced seed 11011 ───┤    (~65M unique pairs) │
│  Spaced seed 110011 ──┘                        │
│                                                │
│  Each index: Phase A (k-mer count per target)  │
│            → top-200K selection                 │
│            → Phase B (diagonal scoring)         │
│            → top-mc selection                   │
└────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  Stage 1.8: ML Two-Tier (optional)             │
│                                                │
│  Features from Phase A/B scores per index      │
│  LightGBM predicts SW scores                   │
│  Select top-N per query (N=2000-12000)         │
│  Reduces 65M → 4-24M pairs for alignment       │
└────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  Stage 2: Smith-Waterman Local Alignment       │
│                                                │
│  Banded SW with BLOSUM62, band_width=126       │
│  Gap penalties: open=-11, extend=-1            │
│  Score-based filtering (score > 0)             │
│  Optional E-value normalization                │
└────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│  Stage 3: Hit Collection                       │
│                                                │
│  Rank by SW score per query                    │
│  Top-K hits per query (default K=500)          │
│  Output: SearchResults with SearchHit objects  │
└────────────────────────────────────────────────┘
```

## Clustering Pipeline

```
Input sequences (FASTA)
         │
         ▼
┌────────────────────────────┐
│  Sketching (MinHash)       │
│  k-mer size: adaptive      │
│  sketch_size: 128          │
└────────────────────────────┘
         │
         ▼
┌────────────────────────────┐
│  LSH Candidate Generation  │
│  Adaptive tables/bands     │
└────────────────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Jaccard Pre-filter        │
│  Conservative floor        │
└────────────────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Banded NW Alignment       │
│  Adaptive band width       │
│  Early termination         │
│  Length pre-filters         │
└────────────────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Greedy Clustering         │
│  or Leiden Community Det.  │
└────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `clustkit/kmer_index.py` | K-mer inverted index, reduced alphabet, spaced seeds, IDF, ML prefilter |
| `clustkit/pairwise.py` | NW alignment, SW alignment, BLOSUM62, ungapped prefilter |
| `clustkit/search.py` | Search pipeline orchestration, LSH search, hit collection |
| `clustkit/database.py` | Database build/save/load with index persistence |
| `clustkit/csrc/kmer_score.c` | C/OpenMP extension for Phase A+B scoring |
| `clustkit/pipeline.py` | Clustering pipeline |
| `clustkit/sketch.py` | MinHash sketching |
| `clustkit/lsh.py` | LSH hashing and candidate generation |
| `clustkit/io.py` | FASTA/FASTQ I/O, SequenceDataset |
| `clustkit/cli.py` | Command-line interface |

## Reduced Alphabets

| Name | Groups | Size | Usage |
|------|--------|------|-------|
| Murphy-10 | {A,G} {C} {D,E,N,Q} {F,W,Y} {H} {I,L,M,V} {K,R} {P} {S,T} | 9 | Primary reduced index |
| Dayhoff-6 | {A,G,P,S,T} {D,E,N,Q} {H,K,R} {F,W,Y} {I,L,M,V} {C} | 6 | Tested, diminishing returns |
| Hydro-8 | {A,G,I,L,M,V} {F,W,Y} {D,E} {H,K,R} {N,Q} {S,T} {C} {P} | 8 | Tested, diminishing returns |
