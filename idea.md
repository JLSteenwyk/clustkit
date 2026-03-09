# ClustKIT 🧬

**GPU-accelerated sequence clustering for bioinformatics**

A modern, GPU-native reimplementation of CD-HIT-style sequence clustering. ClustKIT replaces CD-HIT's sequential greedy algorithm with a parallel graph-based approach designed from the ground up for GPU execution, achieving orders-of-magnitude speedups on large datasets while producing biologically equivalent (or better) clusterings.

---

## Motivation

CD-HIT is one of the most widely used tools in bioinformatics for clustering sequences by identity. However, its core algorithm is inherently sequential: sequences are processed one at a time in length-sorted order, and each sequence's fate depends on all previously processed sequences. This makes it impossible to fully parallelize.

ClustKIT takes a different approach. Instead of greedy incremental clustering, it builds a sparse similarity graph via locality-sensitive hashing (LSH) and batched pairwise comparison, then clusters using parallel graph algorithms. Every phase is designed for GPU execution.

---

## Architecture Overview

ClustKIT is structured as a six-phase pipeline:

```
Phase 1: Sketch        → Extract minimizer/k-mer signatures per sequence (GPU)
Phase 2: LSH Bucket    → Locality-sensitive hashing to find candidate pairs (GPU)
Phase 3: Pairwise Sim  → Batched alignment or k-mer identity on candidate pairs (GPU)
Phase 4: Graph Build   → Construct sparse similarity graph (GPU)
Phase 5: Cluster       → Parallel connected components / graph clustering (GPU)
Phase 6: Select Reps   → Choose representative per cluster (GPU)
```

---

## Implementation Plan

### Language and Dependencies

- **Primary language:** Python (CLI, I/O, orchestration) + CUDA kernels via CuPy or custom CUDA C
- **GPU framework:** CuPy for prototyping; optionally raw CUDA/Numba for performance-critical kernels
- **Graph clustering:** cuGraph (RAPIDS) for GPU-accelerated connected components
- **Sparse matrix ops:** cuSPARSE via CuPy for graph construction
- **Sequence I/O:** Biopython or pyfastx for FASTA/FASTQ parsing
- **CLI:** Click or Typer
- **Testing:** pytest
- **Build:** pyproject.toml with setuptools or hatchling

### Project Structure

```
clustkit/
├── pyproject.toml
├── README.md
├── LICENSE                     # BSD-2
├── clustkit/
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point
│   ├── pipeline.py             # Orchestrates the 6-phase pipeline
│   ├── io.py                   # Sequence I/O (FASTA/FASTQ reading, streaming)
│   ├── sketch.py               # Phase 1: Minimizer/k-mer signature extraction
│   ├── lsh.py                  # Phase 2: Locality-sensitive hashing
│   ├── pairwise.py             # Phase 3: Batched pairwise comparison
│   ├── graph.py                # Phase 4: Sparse similarity graph construction
│   ├── cluster.py              # Phase 5: Connected components / clustering
│   ├── representatives.py      # Phase 6: Representative selection
│   ├── kernels/                # CUDA kernels (if using raw CUDA)
│   │   ├── sketch_kernel.cu
│   │   ├── lsh_kernel.cu
│   │   └── pairwise_kernel.cu
│   └── utils.py                # Shared utilities, logging, GPU memory management
├── tests/
│   ├── test_sketch.py
│   ├── test_lsh.py
│   ├── test_pairwise.py
│   ├── test_cluster.py
│   ├── test_pipeline_integration.py
│   └── test_accuracy.py        # Benchmarks against CD-HIT / known Pfam families
└── benchmarks/
    ├── benchmark_cdhit_comparison.py
    ├── benchmark_pfam_concordance.py
    ├── benchmark_scaling.py
    └── data/                   # Small test datasets; large ones downloaded by script
```

---

## Phase-by-Phase Specification

### Phase 1: Sketch (sketch.py)

**Goal:** For each input sequence, compute a fixed-size signature (vector of minimizer hashes) that summarizes its k-mer content.

**Algorithm:**
1. For each sequence, extract all k-mers (k configurable, default k=5 for protein, k=11 for nucleotide)
2. Hash each k-mer using a fast hash function (e.g., MurmurHash3 or xxHash)
3. Select the `s` smallest hashes as the minimizer sketch (s configurable, default s=128)
4. Store sketches as a 2D array: `(num_sequences, sketch_size)` of uint64

**GPU strategy:**
- Each thread handles one sequence
- K-mer extraction and hashing is a simple sliding window — no dependencies between positions
- Output is a fixed-size array per sequence → perfect for coalesced GPU memory writes

**Inputs:** Array of encoded sequences (integer-encoded residues), sequence lengths
**Outputs:** `(N, s)` uint64 array of sketches, where N = number of sequences

**Edge cases:**
- Sequences shorter than k: assign empty sketch, these become singletons
- Handle both protein (alphabet size 20) and nucleotide (alphabet size 4) inputs

---

### Phase 2: LSH Bucketing (lsh.py)

**Goal:** Use locality-sensitive hashing to group sequences into buckets such that similar sequences are likely to share buckets, without computing all-pairs similarity.

**Algorithm:**
1. Construct `L` hash tables (L configurable, default L=32), each using `b` hash bands (b configurable, default b=4)
2. For each hash table, concatenate `b` entries from each sequence's sketch to form a band signature
3. Hash the band signature to a bucket ID
4. Two sequences are "candidate pairs" if they share at least one bucket across any of the L hash tables

**GPU strategy:**
- Each thread handles one sequence across all L hash tables
- Band hashing is independent per sequence per hash table
- Output candidate pairs via atomic writes to a pair buffer, or use sort-based deduplication after

**Inputs:** `(N, s)` sketch array
**Outputs:** Deduplicated list of candidate pairs `(i, j)` where i < j

**Tuning:**
- More hash tables (L) → higher recall (fewer missed true pairs) but more candidate pairs
- More bands (b) → higher precision (fewer false candidate pairs) but lower recall
- Provide a `--sensitivity` parameter that auto-tunes L and b for the given identity threshold

---

### Phase 3: Pairwise Similarity (pairwise.py)

**Goal:** For each candidate pair from Phase 2, compute actual sequence identity. Filter pairs below the identity threshold.

**Algorithm options (configurable):**

**Option A: K-mer Jaccard (fast, approximate)**
- For high identity thresholds (≥90%), estimate identity from the Jaccard similarity of k-mer sets
- Jaccard can be computed directly from the minimizer sketches (MinHash estimator)
- Very fast: just count shared hashes between two sketch vectors

**Option B: Banded alignment (slower, exact)**
- For lower identity thresholds (<90%), run banded Needleman-Wunsch or Smith-Waterman
- Use the sketch similarity as a filter: only align pairs above a loose Jaccard threshold
- Band width proportional to expected divergence

**GPU strategy:**
- Batch all candidate pairs into a work queue
- For Option A: each thread handles one pair, loads two sketch vectors, counts intersections
- For Option B: each thread block handles one pairwise alignment (threads parallelize across DP anti-diagonals)
- Pad sequences to uniform lengths for warp coherence

**Inputs:** Candidate pairs, original sequences (for Option B), sketches (for Option A)
**Outputs:** Filtered list of pairs `(i, j, identity)` where identity ≥ threshold

---

### Phase 4: Graph Construction (graph.py)

**Goal:** Build a sparse similarity graph where nodes are sequences and edges connect pairs above the identity threshold.

**Algorithm:**
1. Take the filtered pairs from Phase 3
2. Construct a sparse adjacency structure (COO or CSR format)
3. Optionally weight edges by identity score

**GPU strategy:**
- COO construction is trivial — the pair list IS the COO representation
- Convert to CSR using cuSPARSE or CuPy's sparse utilities for downstream graph algorithms
- Use thrust::sort or CuPy sort for CSR construction

**Inputs:** Filtered pair list `(i, j, identity)`
**Outputs:** Sparse graph in CSR format on GPU

---

### Phase 5: Clustering (cluster.py)

**Goal:** Partition the similarity graph into clusters.

**Algorithm options (configurable):**

**Option A: Connected components (default)**
- Every connected component in the graph becomes a cluster
- Fast, deterministic, order-independent
- Equivalent to single-linkage clustering at the given threshold
- Use cuGraph's `connected_components()` or a parallel union-find implementation

**Option B: Greedy graph clustering**
- Process nodes in degree-sorted order (highest degree first)
- Each unassigned node with high degree becomes a representative; assign its unassigned neighbors to its cluster
- More similar to CD-HIT's behavior (tends to produce more, tighter clusters)
- Less parallelizable but still faster than CD-HIT due to GPU-accelerated neighbor lookups

**Option C: Leiden / Louvain community detection**
- For cases where users want quality-optimized clustering beyond simple thresholding
- Available via cuGraph

**Inputs:** Sparse graph in CSR format
**Outputs:** Array of cluster labels, length N

---

### Phase 6: Representative Selection (representatives.py)

**Goal:** For each cluster, select one sequence as the representative.

**Algorithm:**
1. Default: select the longest sequence in each cluster (matches CD-HIT convention)
2. Alternative: select the sequence with highest average identity to other cluster members (centroid)
3. Alternative: select the sequence with highest sum of edge weights (most connected)

**GPU strategy:**
- Segmented reduction: group sequences by cluster label, reduce within each segment
- Use CuPy or thrust for segmented argmax on sequence lengths

**Inputs:** Cluster labels, sequence metadata (lengths, identities)
**Outputs:** Representative sequence ID per cluster, final cluster membership file

---

## CLI Interface

```bash
# Basic usage — cluster proteins at 90% identity
clustkit cluster -i sequences.fasta -o output/ -t 0.9

# Nucleotide mode with lower threshold
clustkit cluster -i reads.fasta -o output/ -t 0.7 --mode nucleotide

# Use exact alignment instead of k-mer approximation
clustkit cluster -i sequences.fasta -o output/ -t 0.5 --alignment exact

# Specify GPU device
clustkit cluster -i sequences.fasta -o output/ -t 0.9 --device 0

# Multi-GPU for very large datasets
clustkit cluster -i sequences.fasta -o output/ -t 0.9 --devices 0,1,2,3

# Tune sensitivity (higher = fewer missed pairs, slower)
clustkit cluster -i sequences.fasta -o output/ -t 0.9 --sensitivity high

# CD-HIT-compatible output format
clustkit cluster -i sequences.fasta -o output/ -t 0.9 --format cdhit
```

### Output Files

```
output/
├── representatives.fasta       # Representative sequences
├── clusters.tsv                # Tab-separated: sequence_id \t cluster_id \t is_representative
├── clusters.clstr              # CD-HIT-compatible .clstr format (if --format cdhit)
└── run_info.json               # Parameters, runtime stats, cluster size distribution
```

---

## Benchmarking Plan

### 1. Identity Guarantee Audit

For each output cluster, align every member to its representative using full Smith-Waterman (via parasail or BioPython). Report the fraction of member-representative pairs that fall below the stated identity threshold. Target: 0% violations.

### 2. Pfam Concordance

- Download SwissProt + Pfam annotations
- Cluster SwissProt sequences at various thresholds (40%, 50%, 70%, 90%)
- Measure Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and V-measure between ClustKIT clusters and Pfam family labels
- Compare head-to-head against: CD-HIT, MMseqs2 cluster, MMseqs2 linclust

### 3. Simulated Datasets

- Use INDELible to simulate protein families with controlled divergence along known trees
- Generate datasets where ground-truth clusters are known exactly
- Measure precision, recall, and F1 for cluster recovery at matching identity thresholds
- Characterize where each method breaks down (e.g., domain shuffling, variable-length insertions)

### 4. Downstream Task Evaluation

- Build non-redundant databases with ClustKIT vs. CD-HIT at matching thresholds
- Evaluate metagenomic classification sensitivity using Kraken2 with each database
- Evaluate protein function prediction using representatives from each method as training data

### 5. Scaling Benchmarks

- Runtime and peak GPU memory as a function of dataset size (1K, 10K, 100K, 1M, 10M, 100M sequences)
- Runtime as a function of identity threshold (higher thresholds = more candidate pairs)
- Comparison: ClustKIT (1 GPU) vs. CD-HIT (1 thread) vs. CD-HIT (16 threads) vs. MMseqs2 (16 threads)
- Multi-GPU scaling efficiency

### 6. Stability Analysis

- Shuffle input order 10 times; measure variation of information between resulting clusterings
- ClustKIT (graph-based) should be order-invariant by construction; CD-HIT will vary
- Subsample 80% of sequences 10 times; measure consistency of cluster assignments for shared sequences

---

## Development Roadmap

### Phase A: MVP (Weeks 1-3)
- [ ] Sequence I/O (FASTA reading, integer encoding)
- [ ] CPU reference implementation of all 6 pipeline phases (for correctness testing)
- [ ] Basic CLI with Click/Typer
- [ ] Test suite with small synthetic datasets

### Phase B: GPU Kernels (Weeks 4-7)
- [ ] GPU sketch extraction (CuPy or custom CUDA)
- [ ] GPU LSH bucketing
- [ ] GPU pairwise Jaccard (k-mer mode)
- [ ] GPU graph construction + connected components via cuGraph
- [ ] Representative selection
- [ ] End-to-end GPU pipeline integration

### Phase C: Alignment Mode (Weeks 8-9)
- [ ] Batched banded Needleman-Wunsch kernel for low-threshold clustering
- [ ] Adaptive mode selection: k-mer Jaccard for ≥90%, alignment for <90%

### Phase D: Benchmarking (Weeks 10-12)
- [ ] Identity guarantee audit
- [ ] Pfam concordance benchmarks
- [ ] Simulated dataset benchmarks
- [ ] Runtime scaling benchmarks
- [ ] CD-HIT and MMseqs2 comparison scripts

### Phase E: Polish (Weeks 13-14)
- [ ] CD-HIT-compatible output format
- [ ] Multi-GPU support
- [ ] Documentation and tutorial
- [ ] Conda/pip packaging

---

## Design Decisions and Rationale

**Why graph-based instead of greedy?**
CD-HIT's greedy approach produces path-dependent clusters (input order matters). A graph-based approach is deterministic and order-independent, which is scientifically preferable. It also exposes massive parallelism that the greedy approach fundamentally cannot.

**Why LSH instead of all-pairs?**
All-pairs comparison is O(N²), which is infeasible for >1M sequences even on GPUs. LSH reduces the work to approximately O(N) with tunable recall, at the cost of potentially missing some pairs. The sensitivity parameter lets users trade speed for completeness.

**Why connected components as the default clustering?**
It's the simplest, fastest, and most reproducible option. It's equivalent to single-linkage clustering at the threshold, which is what CD-HIT approximates (imperfectly) with its greedy approach. For users who want tighter clusters, we offer Leiden/Louvain as an option.

**Why support both k-mer Jaccard and alignment?**
At high identity thresholds (≥90%), k-mer overlap is an excellent proxy for sequence identity — fast and accurate. At lower thresholds (40-70%), the relationship between k-mer similarity and alignment identity breaks down, so actual alignment is needed. Rather than forcing one approach, we let the threshold guide the method.

**CPU fallback:**
Every phase has a CPU reference implementation. If no GPU is available, ClustKIT falls back to CPU mode (slower but functional). This also serves as the ground truth for validating GPU kernel correctness.
