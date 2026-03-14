"""Tests for the search module (clustkit.search) and database module (clustkit.database)."""

import tempfile
from pathlib import Path

import numba
import numpy as np
import pytest

from clustkit.io import read_sequences
from clustkit.search import (
    SearchHit,
    SearchResults,
    search_sequences,
    write_search_results_tsv,
    _lsh_query_candidates,
)
from clustkit.database import build_database, save_database, load_database


@pytest.fixture
def protein_fastas(tmp_path):
    """Create query and database FASTA files for testing."""
    query_fasta = tmp_path / "query.fasta"
    query_fasta.write_text(
        ">q1\nACDEFGHIKLMNPQRSTVWY\n"
        ">q2\nKLMNPQRSTVWYACDEFGHI\n"
        ">q3\nACDEFGHIKLMNPQRSTVWYACDE\n"
    )

    db_fasta = tmp_path / "db.fasta"
    db_fasta.write_text(
        ">d1\nACDEFGHIKLMNPQRSTVWY\n"
        ">d2\nACDEFGHIKLMNPQRSTVWYKLMNPQ\n"
        ">d3\nKLMNPQRSTVWYACDEFGHI\n"
        ">d4\nXXXXXXXXXXXXXXXXXXXXXXXX\n"
        ">d5\nACDEFGHIKLMNPQRSTVWYACDE\n"
    )

    return query_fasta, db_fasta


@pytest.fixture(autouse=True)
def set_threads():
    """Ensure Numba uses a reasonable thread count for tests."""
    numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))


class TestSearchSequences:
    """Tests for search_sequences()."""

    def test_basic_search(self, protein_fastas):
        query_fasta, db_fasta = protein_fastas
        query_ds = read_sequences(query_fasta, "protein")
        db_ds = read_sequences(db_fasta, "protein")

        results = search_sequences(
            query_dataset=query_ds,
            db_dataset=db_ds,
            threshold=0.5,
            top_k=10,
            mode="protein",
            kmer_size=3,
            sketch_size=32,
            sensitivity="high",
        )

        assert isinstance(results, SearchResults)
        assert results.num_queries == 3
        assert results.num_targets == 5
        assert len(results.hits) == 3

    def test_finds_exact_matches(self, protein_fastas):
        query_fasta, db_fasta = protein_fastas
        query_ds = read_sequences(query_fasta, "protein")
        db_ds = read_sequences(db_fasta, "protein")

        results = search_sequences(
            query_dataset=query_ds,
            db_dataset=db_ds,
            threshold=0.9,
            top_k=10,
            mode="protein",
            kmer_size=3,
            sketch_size=32,
            sensitivity="high",
        )

        # q1 should match d1 (identical) and d2 (substring)
        q1_hits = results.hits[0]
        q1_target_ids = [h.target_id for h in q1_hits]
        assert "d1" in q1_target_ids

        # q2 should match d3 (identical)
        q2_hits = results.hits[1]
        q2_target_ids = [h.target_id for h in q2_hits]
        assert "d3" in q2_target_ids

    def test_top_k_limit(self, protein_fastas):
        query_fasta, db_fasta = protein_fastas
        query_ds = read_sequences(query_fasta, "protein")
        db_ds = read_sequences(db_fasta, "protein")

        results = search_sequences(
            query_dataset=query_ds,
            db_dataset=db_ds,
            threshold=0.3,
            top_k=2,
            mode="protein",
            kmer_size=3,
            sketch_size=32,
            sensitivity="high",
        )

        for query_hits in results.hits:
            assert len(query_hits) <= 2

    def test_hits_sorted_by_identity(self, protein_fastas):
        query_fasta, db_fasta = protein_fastas
        query_ds = read_sequences(query_fasta, "protein")
        db_ds = read_sequences(db_fasta, "protein")

        results = search_sequences(
            query_dataset=query_ds,
            db_dataset=db_ds,
            threshold=0.3,
            top_k=10,
            mode="protein",
            kmer_size=3,
            sketch_size=32,
            sensitivity="high",
        )

        for query_hits in results.hits:
            identities = [h.identity for h in query_hits]
            assert identities == sorted(identities, reverse=True)

    def test_empty_query(self, tmp_path):
        query_fasta = tmp_path / "empty_query.fasta"
        query_fasta.write_text("")

        db_fasta = tmp_path / "db.fasta"
        db_fasta.write_text(">d1\nACDEFGHIKLMNPQRSTVWY\n")

        # Empty file can't be detected as FASTA, so create a valid empty-ish one
        # Actually read_sequences raises on empty, so let's test with 0-match scenario
        query_fasta.write_text(">q1\nXXXXX\n")
        query_ds = read_sequences(query_fasta, "protein")
        db_ds = read_sequences(db_fasta, "protein")

        results = search_sequences(
            query_dataset=query_ds,
            db_dataset=db_ds,
            threshold=0.9,
            top_k=10,
            mode="protein",
            kmer_size=3,
            sketch_size=32,
            sensitivity="high",
        )

        assert results.num_queries == 1
        # Very different sequence, likely no hits at 0.9
        total_hits = sum(len(h) for h in results.hits)
        assert total_hits == 0


class TestSearchResultsOutput:
    """Tests for write_search_results_tsv()."""

    def test_write_tsv(self, protein_fastas, tmp_path):
        query_fasta, db_fasta = protein_fastas
        query_ds = read_sequences(query_fasta, "protein")
        db_ds = read_sequences(db_fasta, "protein")

        results = search_sequences(
            query_dataset=query_ds,
            db_dataset=db_ds,
            threshold=0.5,
            top_k=10,
            mode="protein",
            kmer_size=3,
            sketch_size=32,
            sensitivity="high",
        )

        out_path = tmp_path / "results.tsv"
        write_search_results_tsv(out_path, results)

        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert lines[0] == "query_id\ttarget_id\tidentity\tquery_length\ttarget_length"
        total_hits = sum(len(h) for h in results.hits)
        assert len(lines) - 1 == total_hits  # minus header


class TestLSHQueryCandidates:
    """Tests for _lsh_query_candidates()."""

    def test_identical_sketches_found(self):
        """Identical sketches should always be paired."""
        sketch = np.sort(np.array([10, 20, 30, 40], dtype=np.uint64))
        query_sketches = sketch.reshape(1, -1)
        db_sketches = sketch.reshape(1, -1)

        pairs = _lsh_query_candidates(
            query_sketches, db_sketches,
            num_tables=32, num_bands=1,
        )

        assert len(pairs) > 0
        assert pairs[0, 0] == 0  # query idx
        assert pairs[0, 1] == 0  # db idx

    def test_empty_inputs(self):
        pairs = _lsh_query_candidates(
            np.empty((0, 4), dtype=np.uint64),
            np.empty((0, 4), dtype=np.uint64),
            num_tables=16, num_bands=1,
        )
        assert len(pairs) == 0


class TestDatabaseIndex:
    """Tests for database build/save/load."""

    def test_build_save_load(self, protein_fastas, tmp_path):
        _, db_fasta = protein_fastas
        db_dir = tmp_path / "db_index"

        # Build
        idx = build_database(
            db_fasta, mode="protein", kmer_size=3,
            sketch_size=32, threshold=0.5, sensitivity="high",
        )
        assert idx.dataset.num_sequences == 5
        assert idx.sketches.shape == (5, 32)

        # Save
        save_database(idx, db_dir)
        assert (db_dir / "params.json").exists()
        assert (db_dir / "sketches.npy").exists()

        # Load
        loaded = load_database(db_dir)
        assert loaded.dataset.num_sequences == 5
        np.testing.assert_array_equal(loaded.sketches, idx.sketches)
        np.testing.assert_array_equal(
            loaded.dataset.encoded_sequences,
            idx.dataset.encoded_sequences,
        )
        assert loaded.dataset.ids == idx.dataset.ids

    def test_search_with_prebuilt_index(self, protein_fastas, tmp_path):
        query_fasta, db_fasta = protein_fastas
        db_dir = tmp_path / "db_index"

        # Build and save index
        idx = build_database(
            db_fasta, mode="protein", kmer_size=3,
            sketch_size=32, threshold=0.5, sensitivity="high",
        )
        save_database(idx, db_dir)

        # Load and search
        loaded = load_database(db_dir)
        query_ds = read_sequences(query_fasta, "protein")

        results = search_sequences(
            query_dataset=query_ds,
            db_dataset=loaded.dataset,
            threshold=0.5,
            top_k=10,
            mode="protein",
            kmer_size=3,
            sketch_size=32,
            sensitivity="high",
            db_sketches=loaded.sketches,
        )

        assert results.num_queries == 3
        assert results.num_targets == 5
        total_hits = sum(len(h) for h in results.hits)
        assert total_hits > 0
