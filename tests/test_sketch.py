"""Tests for Phase 1: Sketch computation."""

import numpy as np
import pytest

from clustkit.io import read_sequences
from clustkit.sketch import compute_sketches, sketch_sequence


class TestSketchSequence:
    def test_basic_sketch(self):
        # Simple encoded sequence: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        seq = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.uint8)
        sketch = sketch_sequence(seq, len(seq), k=3, sketch_size=4, alphabet_size=20)
        assert len(sketch) == 4
        assert sketch.dtype == np.uint64
        # Sketch should be sorted
        assert np.all(sketch[:-1] <= sketch[1:])

    def test_short_sequence_returns_zeros(self):
        seq = np.array([0, 1], dtype=np.uint8)
        sketch = sketch_sequence(seq, 2, k=5, sketch_size=4, alphabet_size=20)
        assert np.all(sketch == 0)

    def test_sketch_size_larger_than_kmers(self):
        # 5-length sequence with k=3 → only 3 k-mers, but sketch_size=8
        seq = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        sketch = sketch_sequence(seq, 5, k=3, sketch_size=8, alphabet_size=20)
        assert len(sketch) == 8
        # Padded values should be max uint64
        max_val = np.iinfo(np.uint64).max
        assert sketch[-1] == max_val

    def test_identical_sequences_same_sketch(self):
        seq = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
        s1 = sketch_sequence(seq, 10, k=3, sketch_size=4, alphabet_size=20)
        s2 = sketch_sequence(seq, 10, k=3, sketch_size=4, alphabet_size=20)
        np.testing.assert_array_equal(s1, s2)

    def test_different_sequences_different_sketches(self):
        seq_a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
        seq_b = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.uint8)
        s1 = sketch_sequence(seq_a, 10, k=3, sketch_size=4, alphabet_size=20)
        s2 = sketch_sequence(seq_b, 10, k=3, sketch_size=4, alphabet_size=20)
        assert not np.array_equal(s1, s2)


class TestSketchSequenceNucleotide:
    def test_nucleotide_sketch_k11(self):
        # A nucleotide-encoded sequence (values 0-3) with k=11
        seq = np.array([0, 1, 2, 3] * 25, dtype=np.uint8)  # 100bp
        sketch = sketch_sequence(seq, len(seq), k=11, sketch_size=16, alphabet_size=4)
        assert len(sketch) == 16
        assert sketch.dtype == np.uint64
        # Sketch should be sorted
        assert np.all(sketch[:-1] <= sketch[1:])

    def test_nucleotide_sketch_short_sequence(self):
        # Sequence shorter than k=11 should return zeros
        seq = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.uint8)  # 10bp
        sketch = sketch_sequence(seq, 10, k=11, sketch_size=8, alphabet_size=4)
        assert np.all(sketch == 0)

    def test_nucleotide_sketch_deterministic(self):
        seq = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=np.uint8)
        s1 = sketch_sequence(seq, len(seq), k=11, sketch_size=4, alphabet_size=4)
        s2 = sketch_sequence(seq, len(seq), k=11, sketch_size=4, alphabet_size=4)
        np.testing.assert_array_equal(s1, s2)

    def test_nucleotide_alphabet_size_matters(self):
        # Same sequence, different alphabet_size should give different sketches
        seq = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=np.uint8)
        s_nt = sketch_sequence(seq, len(seq), k=5, sketch_size=4, alphabet_size=4)
        s_prot = sketch_sequence(seq, len(seq), k=5, sketch_size=4, alphabet_size=20)
        # Different alphabet sizes produce different k-mer integer encodings
        # hence different hashes
        assert not np.array_equal(s_nt, s_prot)


class TestComputeSketches:
    def test_batch_sketches(self, synthetic_fasta_path):
        dataset = read_sequences(synthetic_fasta_path, "protein")
        sketches = compute_sketches(
            dataset.encoded_sequences,
            dataset.lengths,
            k=5,
            sketch_size=16,
            mode="protein",
        )
        assert sketches.shape == (6, 16)
        assert sketches.dtype == np.uint64

    def test_similar_sequences_similar_sketches(self, synthetic_fasta_path):
        dataset = read_sequences(synthetic_fasta_path, "protein")
        sketches = compute_sketches(
            dataset.encoded_sequences,
            dataset.lengths,
            k=5,
            sketch_size=32,
            mode="protein",
        )
        # seq1 and seq2 should share many sketch values (similar sequences)
        shared_12 = len(np.intersect1d(sketches[0], sketches[1]))
        # seq1 and seq4 should share fewer (different clusters)
        shared_14 = len(np.intersect1d(sketches[0], sketches[3]))
        assert shared_12 > shared_14


class TestComputeSketchesNucleotide:
    def test_batch_sketches_nucleotide(self, synthetic_nt_fasta_path):
        dataset = read_sequences(synthetic_nt_fasta_path, "nucleotide")
        sketches = compute_sketches(
            dataset.encoded_sequences,
            dataset.lengths,
            k=11,
            sketch_size=32,
            mode="nucleotide",
        )
        assert sketches.shape == (9, 32)
        assert sketches.dtype == np.uint64

    def test_similar_nt_sequences_similar_sketches(self, synthetic_nt_fasta_path):
        dataset = read_sequences(synthetic_nt_fasta_path, "nucleotide")
        # Use k=5 for this test: with k=11 and 10 mutations in 100bp,
        # each mutation destroys up to 11 k-mers, which can eliminate
        # nearly all shared 11-mers. k=5 is more forgiving and still
        # demonstrates that within-cluster pairs share more k-mers.
        sketches = compute_sketches(
            dataset.encoded_sequences,
            dataset.lengths,
            k=5,
            sketch_size=64,
            mode="nucleotide",
        )
        # nseq1 and nseq2 are in the same cluster (~90% identity)
        shared_within = len(np.intersect1d(sketches[0], sketches[1]))
        # nseq1 and nseq4 are in different clusters
        shared_between = len(np.intersect1d(sketches[0], sketches[3]))
        assert shared_within > shared_between
