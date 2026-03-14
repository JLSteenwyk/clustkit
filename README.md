# ClustKIT

**Accurate sequence clustering and search via adaptive banded alignment.**

ClustKIT is a bioinformatics tool for protein and nucleotide sequence clustering and search. It combines MinHash sketching, locality-sensitive hashing (LSH), and banded Needleman-Wunsch alignment to achieve high accuracy at all identity thresholds, including the challenging low-identity regime (30-50%) where heuristic methods lose sensitivity.

## Features

- **Dual-purpose**: Sequence clustering (`clustkit cluster`) and search (`clustkit search`)
- **Accurate at low identity**: Alignment-based identity computation with adaptive banded NW
- **Fast**: Multi-stage filtering (LSH + Jaccard pre-filter + length-ratio filter) eliminates >98% of candidate pairs before alignment
- **Scalable**: Numba JIT parallelism with configurable thread count; optional GPU acceleration via CuPy
- **Protein and nucleotide**: Supports both sequence types with adaptive k-mer sizing
- **Flexible clustering**: Connected components, greedy, or Leiden community detection
- **Database pre-indexing**: Pre-compute sketches and LSH tables for fast repeated searches

## Installation

```bash
pip install clustkit
```

With optional dependencies:

```bash
# Leiden community detection
pip install clustkit[leiden]

# GPU acceleration (requires CUDA 12.x)
pip install clustkit[gpu]

# Development / benchmarks
pip install clustkit[dev,benchmarks]
```

From source:

```bash
git clone https://github.com/clustkit/clustkit.git
cd clustkit
pip install -e ".[dev]"
```

## Quick Start

### Clustering

```bash
# Cluster proteins at 50% identity using 8 threads
clustkit cluster -i sequences.fasta -o output/ -t 0.5 --threads 8

# Cluster nucleotides at 97% identity
clustkit cluster -i 16s.fasta -o output/ -t 0.97 --mode nucleotide --threads 8

# Use Leiden clustering instead of connected components
clustkit cluster -i sequences.fasta -o output/ -t 0.7 --cluster-method leiden --threads 4
```

Output files:
- `output/clusters.tsv` — Cluster assignments (sequence_id, cluster_id, is_representative)
- `output/representatives.fasta` — Representative sequences
- `output/run_info.json` — Run parameters and statistics

### Searching

```bash
# Search queries against a database
clustkit search -q queries.fasta --db database.fasta -o results.tsv -t 0.5 --threads 8

# Pre-build a database index for faster repeated searches
clustkit makedb -i database.fasta -o db_index/ -t 0.5 --threads 8
clustkit search -q queries.fasta --db db_index/ -o results.tsv -t 0.5 --threads 8
```

Output format (TSV):
```
query_id    target_id    identity    query_length    target_length
```

### GPU Acceleration

```bash
# Auto-select best device (benchmarks CPU vs GPU on a sample)
clustkit cluster -i sequences.fasta -o output/ -t 0.5 --device auto

# Use specific GPU
clustkit cluster -i sequences.fasta -o output/ -t 0.5 --device 0
```

## Pipeline Overview

ClustKIT uses a 6-phase clustering pipeline:

1. **Read** — Parse FASTA/FASTQ, integer-encode sequences
2. **Sketch** — MinHash bottom-s sketches with adaptive k-mer size
3. **LSH** — Locality-sensitive hashing to find candidate pairs
4. **Align** — Banded Needleman-Wunsch with:
   - Length-ratio pre-filter (mathematical bound, zero accuracy loss)
   - Jaccard pre-filter from MinHash sketches
   - Cache-friendly pair sorting
   - Adaptive band width per pair
   - Affine gap penalties with early termination
5. **Graph** — Sparse similarity graph construction
6. **Cluster** — Connected components, greedy, or Leiden

The search pipeline uses 3 stages:
1. **LSH prefilter** — Asymmetric query-vs-database candidate generation
2. **Jaccard estimate** — Sketch-based pre-filter to discard obvious non-hits
3. **Banded NW alignment** — Accurate identity computation on remaining candidates

## CLI Reference

```
clustkit cluster   Cluster sequences by identity threshold
clustkit search    Search query sequences against a database
clustkit makedb    Pre-build a database index for fast searching
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --threshold` | Identity threshold (0.0-1.0) | 0.9 (cluster), 0.5 (search) |
| `--mode` | Sequence type: `protein` or `nucleotide` | `protein` |
| `--threads` | Number of CPU threads | 1 |
| `--device` | `cpu`, `auto`, or GPU device ID | `cpu` |
| `--sensitivity` | LSH sensitivity: `low`, `medium`, `high` | `medium` (cluster), `high` (search) |
| `-k, --kmer-size` | K-mer size for sketching | 5 (protein), 11 (nucleotide) |
| `--sketch-size` | MinHash sketch size | 128 |

### Cluster-specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--alignment` | `align` (accurate) or `kmer` (fast) | `align` |
| `--cluster-method` | `connected`, `greedy`, or `leiden` | `connected` |
| `--representative` | `longest`, `centroid`, or `most_connected` | `longest` |
| `--format` | Output format: `tsv` or `cdhit` | `tsv` |

### Search-specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--top-k` | Max hits per query | 10 |

## Benchmarks

### Reproducing Results

```bash
# Download Pfam benchmark data
python benchmarks/download_pfam_families.py

# Run the main Pfam concordance benchmark (all tools, fair thread count)
python benchmarks/benchmark_pfam_concordance.py --threads 4

# Profile Phase 3 (alignment) performance
python benchmarks/profile_phase3.py -i benchmarks/data/pfam_mixed.fasta -t 0.4

# Run ablation study
python benchmarks/benchmark_ablation.py --threads 4

# Thread scaling analysis
python benchmarks/benchmark_thread_scaling.py --max-threads 64

# Correctness validation (banded vs full NW)
python benchmarks/benchmark_correctness.py -i benchmarks/data/pfam_mixed.fasta

# LSH recall measurement
python benchmarks/benchmark_lsh_recall.py -i benchmarks/data/pfam_mixed.fasta -t 0.5

# Search benchmarks
python benchmarks/benchmark_search.py cs4 --threads 4

# Large-scale scaling
python benchmarks/benchmark_large_scale.py -i swissprot.fasta --threads 4

# Nucleotide benchmark
python benchmarks/benchmark_nucleotide.py -i 16s.fasta --taxonomy 16s_taxonomy.tsv --threads 4

# Annotation consistency (requires UniProt-GOA and InterPro data)
python benchmarks/benchmark_annotation_consistency.py cp2 cp3 cp4 \
    --fasta swissprot.fasta --goa-file goa_uniprot.gaf --pfam-file interpro.tsv
```

## License

MIT License. See [LICENSE](LICENSE) for details.
