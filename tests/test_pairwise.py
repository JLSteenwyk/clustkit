"""Tests for Phase 3: Pairwise similarity computation."""

import numpy as np
import pytest

from clustkit.pairwise import compute_pairwise_jaccard, jaccard_from_sketches


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
