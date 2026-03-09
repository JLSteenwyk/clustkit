"""Tests for Phase 2: LSH candidate generation."""

import numpy as np
import pytest

from clustkit.io import read_sequences
from clustkit.lsh import lsh_candidates
from clustkit.sketch import compute_sketches


class TestLSHCandidates:
    def test_identical_sketches_paired(self):
        # Two identical sketches should always be candidates
        sketches = np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],
            dtype=np.uint64,
        )
        pairs = lsh_candidates(sketches, num_tables=8, num_bands=2)
        assert len(pairs) >= 1
        assert (0, 1) in [tuple(p) for p in pairs]

    def test_very_different_sketches_not_paired(self):
        # Two completely different sketches — unlikely to be paired
        rng = np.random.RandomState(42)
        sketches = np.zeros((2, 32), dtype=np.uint64)
        sketches[0] = rng.randint(0, 1000, size=32).astype(np.uint64)
        sketches[1] = rng.randint(10000, 20000, size=32).astype(np.uint64)
        pairs = lsh_candidates(sketches, num_tables=4, num_bands=8)
        # With very different sketches and high bands, should get no pairs
        assert len(pairs) == 0

    def test_pairs_are_sorted(self):
        sketches = np.tile(np.arange(16, dtype=np.uint64), (5, 1))
        pairs = lsh_candidates(sketches, num_tables=4, num_bands=2)
        # All pairs should have i < j
        assert np.all(pairs[:, 0] < pairs[:, 1])

    def test_pairs_are_deduplicated(self):
        sketches = np.tile(np.arange(16, dtype=np.uint64), (3, 1))
        pairs = lsh_candidates(sketches, num_tables=16, num_bands=2)
        # Convert to set of tuples — should have no duplicates
        pair_set = set(tuple(p) for p in pairs)
        assert len(pair_set) == len(pairs)

    def test_integration_with_sketches(self, synthetic_fasta_path):
        dataset = read_sequences(synthetic_fasta_path, "protein")
        sketches = compute_sketches(
            dataset.encoded_sequences,
            dataset.lengths,
            k=5,
            sketch_size=32,
            mode="protein",
        )
        pairs = lsh_candidates(sketches, num_tables=32, num_bands=4)
        # Similar sequences (seq1, seq2, seq3) should appear as candidates
        pair_set = set(tuple(p) for p in pairs)
        # At least some within-cluster pairs should be found
        assert len(pairs) > 0

    def test_empty_input(self):
        sketches = np.empty((0, 16), dtype=np.uint64)
        pairs = lsh_candidates(sketches, num_tables=4, num_bands=2)
        assert len(pairs) == 0
