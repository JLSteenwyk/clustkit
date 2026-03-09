"""Integration tests: full pipeline end-to-end."""

import json

import numpy as np
import pytest

from clustkit.pipeline import run_pipeline


class TestPipelineIntegration:
    def test_full_pipeline_protein(self, synthetic_fasta_path, output_dir):
        config = {
            "input": synthetic_fasta_path,
            "output": output_dir,
            "threshold": 0.5,
            "mode": "protein",
            "alignment": "kmer",
            "sketch_size": 32,
            "kmer_size": 5,
            "sensitivity": "high",
            "cluster_method": "connected",
            "representative": "longest",
            "device": "cpu",
            "threads": 1,
            "format": "tsv",
        }

        run_pipeline(config)

        # Check output files exist
        assert (output_dir / "representatives.fasta").exists()
        assert (output_dir / "clusters.tsv").exists()
        assert (output_dir / "run_info.json").exists()

        # Check run_info.json
        with open(output_dir / "run_info.json") as f:
            info = json.load(f)
        assert info["num_sequences"] == 6
        assert info["num_clusters"] >= 1

    def test_pipeline_cdhit_format(self, synthetic_fasta_path, output_dir):
        config = {
            "input": synthetic_fasta_path,
            "output": output_dir,
            "threshold": 0.5,
            "mode": "protein",
            "alignment": "kmer",
            "sketch_size": 32,
            "kmer_size": 5,
            "sensitivity": "high",
            "cluster_method": "connected",
            "representative": "longest",
            "device": "cpu",
            "threads": 1,
            "format": "cdhit",
        }

        run_pipeline(config)

        assert (output_dir / "clusters.clstr").exists()
        content = (output_dir / "clusters.clstr").read_text()
        assert ">Cluster 0" in content

    def test_pipeline_greedy_method(self, synthetic_fasta_path, output_dir):
        config = {
            "input": synthetic_fasta_path,
            "output": output_dir,
            "threshold": 0.5,
            "mode": "protein",
            "alignment": "kmer",
            "sketch_size": 32,
            "kmer_size": 5,
            "sensitivity": "high",
            "cluster_method": "greedy",
            "representative": "longest",
            "device": "cpu",
            "threads": 1,
            "format": "tsv",
        }

        run_pipeline(config)

        assert (output_dir / "clusters.tsv").exists()

    def test_single_sequence(self, tmp_path):
        fasta = tmp_path / "single.fasta"
        fasta.write_text(">seq1\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQV\n")
        out = tmp_path / "out"

        config = {
            "input": fasta,
            "output": out,
            "threshold": 0.9,
            "mode": "protein",
            "alignment": "kmer",
            "sketch_size": 16,
            "kmer_size": 5,
            "sensitivity": "medium",
            "cluster_method": "connected",
            "representative": "longest",
            "device": "cpu",
            "threads": 1,
            "format": "tsv",
        }

        run_pipeline(config)

        assert (out / "representatives.fasta").exists()
        with open(out / "run_info.json") as f:
            info = json.load(f)
        assert info["num_sequences"] == 1
        assert info["num_clusters"] == 1

    def test_full_pipeline_nucleotide_kmer(self, synthetic_nt_fasta_path, output_dir):
        """End-to-end nucleotide pipeline using k-mer Jaccard similarity."""
        config = {
            "input": synthetic_nt_fasta_path,
            "output": output_dir,
            "threshold": 0.5,
            "mode": "nucleotide",
            "alignment": "kmer",
            "sketch_size": 64,
            "kmer_size": 11,
            "sensitivity": "high",
            "cluster_method": "connected",
            "representative": "longest",
            "device": "cpu",
            "threads": 1,
            "format": "tsv",
        }

        run_pipeline(config)

        # Check output files exist
        assert (output_dir / "representatives.fasta").exists()
        assert (output_dir / "clusters.tsv").exists()
        assert (output_dir / "run_info.json").exists()

        # Check run_info.json
        with open(output_dir / "run_info.json") as f:
            info = json.load(f)
        assert info["num_sequences"] == 9
        assert info["num_clusters"] >= 1

    def test_full_pipeline_nucleotide_align(self, synthetic_nt_fasta_path, output_dir):
        """End-to-end nucleotide pipeline using NW alignment similarity."""
        config = {
            "input": synthetic_nt_fasta_path,
            "output": output_dir,
            "threshold": 0.5,
            "mode": "nucleotide",
            "alignment": "align",
            "sketch_size": 64,
            "kmer_size": 11,
            "sensitivity": "high",
            "cluster_method": "connected",
            "representative": "longest",
            "device": "cpu",
            "threads": 1,
            "format": "tsv",
        }

        run_pipeline(config)

        # Check output files exist
        assert (output_dir / "representatives.fasta").exists()
        assert (output_dir / "clusters.tsv").exists()
        assert (output_dir / "run_info.json").exists()

        with open(output_dir / "run_info.json") as f:
            info = json.load(f)
        assert info["num_sequences"] == 9
        assert info["num_clusters"] >= 1

    def test_nucleotide_pipeline_cdhit_format(self, synthetic_nt_fasta_path, output_dir):
        """Verify CD-HIT output uses 'nt' unit for nucleotide mode."""
        config = {
            "input": synthetic_nt_fasta_path,
            "output": output_dir,
            "threshold": 0.5,
            "mode": "nucleotide",
            "alignment": "kmer",
            "sketch_size": 64,
            "kmer_size": 11,
            "sensitivity": "high",
            "cluster_method": "connected",
            "representative": "longest",
            "device": "cpu",
            "threads": 1,
            "format": "cdhit",
        }

        run_pipeline(config)

        assert (output_dir / "clusters.clstr").exists()
        content = (output_dir / "clusters.clstr").read_text()
        assert ">Cluster 0" in content
        # Nucleotide sequences should use 'nt' unit, not 'aa'
        assert "nt, >" in content
        assert "aa, >" not in content
