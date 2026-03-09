"""Tests for Phase 3: Pairwise similarity computation."""

import numpy as np
import pytest

from clustkit.io import encode_sequence, read_sequences
from clustkit.pairwise import (
    compute_pairwise_alignment,
    compute_pairwise_jaccard,
    jaccard_from_sketches,
)


class TestJaccardFromSketches:
    def test_identical_sketches(self):
        sketch = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        assert jaccard_from_sketches(sketch, sketch) == 1.0

    def test_disjoint_sketches(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        b = np.array([6, 7, 8, 9, 10], dtype=np.uint64)
        assert jaccard_from_sketches(a, b) == 0.0

    def test_partial_overlap(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        b = np.array([1, 2, 6, 7, 8], dtype=np.uint64)
        sim = jaccard_from_sketches(a, b)
        assert 0.0 < sim < 1.0

    def test_padded_sketches(self):
        max_val = np.iinfo(np.uint64).max
        a = np.array([1, 2, 3, max_val, max_val], dtype=np.uint64)
        b = np.array([1, 2, 4, max_val, max_val], dtype=np.uint64)
        sim = jaccard_from_sketches(a, b)
        assert 0.0 < sim < 1.0


class TestComputePairwiseJaccard:
    def test_filters_below_threshold(self):
        sketches = np.array(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],  # identical to 0
                [6, 7, 8, 9, 10],  # completely different
            ],
            dtype=np.uint64,
        )
        pairs = np.array([[0, 1], [0, 2]], dtype=np.int32)
        filtered, sims = compute_pairwise_jaccard(pairs, sketches, threshold=0.5)
        assert len(filtered) == 1  # only (0,1) passes
        assert filtered[0, 0] == 0 and filtered[0, 1] == 1
        assert sims[0] == 1.0

    def test_empty_pairs(self):
        sketches = np.array([[1, 2, 3]], dtype=np.uint64)
        pairs = np.empty((0, 2), dtype=np.int32)
        filtered, sims = compute_pairwise_jaccard(pairs, sketches, threshold=0.5)
        assert len(filtered) == 0
        assert len(sims) == 0

    def test_all_pairs_below_threshold(self):
        sketches = np.array(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            dtype=np.uint64,
        )
        pairs = np.array([[0, 1]], dtype=np.int32)
        filtered, sims = compute_pairwise_jaccard(pairs, sketches, threshold=0.5)
        assert len(filtered) == 0


class TestNucleotideAlignment:
    """Verify NW alignment works correctly for nucleotide-encoded sequences."""

    def test_identical_nucleotide_sequences(self):
        seq = "ATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGC"
        enc = encode_sequence(seq, "nucleotide")
        # Pad into a 2-sequence matrix
        n = len(enc)
        seqs = np.zeros((2, n), dtype=np.uint8)
        seqs[0] = enc
        seqs[1] = enc
        lengths = np.array([n, n], dtype=np.int32)
        pairs = np.array([[0, 1]], dtype=np.int32)

        filtered, identities = compute_pairwise_alignment(
            pairs, seqs, lengths, threshold=0.5, band_width=0
        )
        assert len(filtered) == 1
        assert identities[0] == pytest.approx(1.0)

    def test_similar_nucleotide_sequences(self):
        seq_a = "ATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGC"
        # ~10% mutations (5 out of 47)
        seq_b = "ATGCTCGCTAGCATGCTTGCTAGCATGCTAGCTAGCATGCTAGCTTGC"
        enc_a = encode_sequence(seq_a, "nucleotide")
        enc_b = encode_sequence(seq_b, "nucleotide")

        n = max(len(enc_a), len(enc_b))
        seqs = np.full((2, n), 4, dtype=np.uint8)  # pad with unknown=4
        seqs[0, :len(enc_a)] = enc_a
        seqs[1, :len(enc_b)] = enc_b
        lengths = np.array([len(enc_a), len(enc_b)], dtype=np.int32)
        pairs = np.array([[0, 1]], dtype=np.int32)

        filtered, identities = compute_pairwise_alignment(
            pairs, seqs, lengths, threshold=0.5, band_width=0
        )
        assert len(filtered) == 1
        # Identity should be high but not 1.0
        assert 0.8 < identities[0] < 1.0

    def test_dissimilar_nucleotide_sequences(self):
        seq_a = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        seq_b = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        enc_a = encode_sequence(seq_a, "nucleotide")
        enc_b = encode_sequence(seq_b, "nucleotide")

        n = max(len(enc_a), len(enc_b))
        seqs = np.full((2, n), 4, dtype=np.uint8)
        seqs[0, :len(enc_a)] = enc_a
        seqs[1, :len(enc_b)] = enc_b
        lengths = np.array([len(enc_a), len(enc_b)], dtype=np.int32)
        pairs = np.array([[0, 1]], dtype=np.int32)

        filtered, identities = compute_pairwise_alignment(
            pairs, seqs, lengths, threshold=0.5, band_width=0
        )
        # All mismatches, should be filtered out at threshold 0.5
        assert len(filtered) == 0

    def test_nucleotide_alignment_from_dataset(self, synthetic_nt_fasta_path):
        dataset = read_sequences(synthetic_nt_fasta_path, "nucleotide")
        # Compare nseq1 (idx 0) with nseq2 (idx 1) - same cluster, ~90% identity
        pairs = np.array([[0, 1]], dtype=np.int32)
        filtered, identities = compute_pairwise_alignment(
            pairs,
            dataset.encoded_sequences,
            dataset.lengths,
            threshold=0.5,
            band_width=0,
        )
        assert len(filtered) == 1
        assert identities[0] >= 0.85  # ~90% identity expected
