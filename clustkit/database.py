"""Database pre-indexing for fast sequence search.

Pre-computes sketches and LSH hash tables so that subsequent searches
only need to compute query sketches and probe the stored index.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clustkit.io import read_sequences, SequenceDataset, SequenceRecord
from clustkit.sketch import compute_sketches
from clustkit.lsh import _hash_all_tables
from clustkit.utils import auto_kmer_for_lsh, auto_lsh_params, logger, timer


@dataclass
class DatabaseIndex:
    """Pre-computed database index for fast search."""

    dataset: SequenceDataset  # Full dataset with encoded sequences
    sketches: np.ndarray  # (N, sketch_size) uint64
    lsh_bucket_ids: np.ndarray  # (num_tables, N) uint64
    lsh_band_indices: np.ndarray  # (num_tables, num_bands) int32
    lsh_seeds: np.ndarray  # (num_tables,) int64
    params: dict  # Parameters used to build the index
    # K-mer inverted index (optional, built by build_database)
    kmer_offsets: np.ndarray | None = None   # (num_possible_kmers+1,) int64
    kmer_entries: np.ndarray | None = None   # (total_occurrences,) int64
    kmer_freqs: np.ndarray | None = None     # (num_possible_kmers,) int32
    # Pre-built reduced alphabet + spaced seed indices (optional)
    reduced_indices: dict | None = None      # {name: (offsets, entries, freqs, ...)}
    reduced_flat: np.ndarray | None = None   # reduced-alphabet encoded sequences


def build_database(
    input_path: str | Path,
    mode: str = "protein",
    kmer_size: int = 5,
    sketch_size: int = 128,
    threshold: float = 0.1,
    sensitivity: str = "high",
    device: str = "cpu",
) -> DatabaseIndex:
    """Build a database index from a FASTA file.

    Args:
        input_path: Path to input FASTA/FASTQ file.
        mode: "protein" or "nucleotide".
        kmer_size: K-mer size for sketching.
        sketch_size: Number of hashes per sketch.
        threshold: Identity threshold for LSH parameter tuning.
        sensitivity: LSH sensitivity level.
        device: Compute device.

    Returns:
        DatabaseIndex ready for searching.
    """
    input_path = Path(input_path)

    # 1. Read sequences
    with timer("Reading database sequences"):
        dataset = read_sequences(input_path, mode)
    logger.info(
        f"Loaded {dataset.num_sequences} sequences "
        f"(max length {dataset.max_length})"
    )

    # 2. Auto-tune k-mer size for LSH
    k = auto_kmer_for_lsh(threshold, mode, kmer_size)
    logger.info(f"Using k-mer size k={k} for LSH (requested {kmer_size})")

    # 3. Compute sketches
    with timer("Computing sketches"):
        sketches = compute_sketches(
            dataset.encoded_sequences,
            dataset.lengths,
            k,
            sketch_size,
            mode,
            seed=42,
            device=device,
            flat_sequences=dataset.flat_sequences,
            offsets=dataset.offsets,
        )

    # 4. Auto-tune LSH parameters
    lsh_params = auto_lsh_params(threshold, sensitivity, k)
    num_tables = lsh_params["num_tables"]
    num_bands = lsh_params["num_bands"]
    logger.info(
        f"LSH parameters: {num_tables} tables, {num_bands} bands per table"
    )

    # 5. Pre-generate band indices and seeds (same RNG logic as lsh.py)
    rng = np.random.RandomState(42)
    all_band_indices = np.empty((num_tables, num_bands), dtype=np.int32)
    all_seeds = np.empty(num_tables, dtype=np.int64)
    for t in range(num_tables):
        all_band_indices[t] = rng.choice(
            sketch_size, size=num_bands, replace=False
        ).astype(np.int32)
        all_seeds[t] = int(rng.randint(0, 2**31))

    # 6. Hash all database sequences
    with timer("Hashing database sequences into LSH tables"):
        lsh_bucket_ids = _hash_all_tables(
            sketches, all_band_indices, all_seeds, num_tables
        )

    # 7. Build k-mer inverted index for fast search
    from clustkit.kmer_index import build_kmer_index
    kmer_index_k = 5 if mode == "protein" else k
    with timer("Building k-mer inverted index"):
        kmer_offsets, kmer_entries, kmer_freqs = build_kmer_index(
            dataset.flat_sequences,
            dataset.offsets,
            dataset.lengths,
            kmer_index_k,
            mode,
        )

    # 8. Pre-build reduced alphabet + spaced seed indices (protein only)
    reduced_indices = {}
    reduced_flat = None
    if mode == "protein":
        from clustkit.kmer_index import (
            REDUCED_ALPHA, REDUCED_ALPHA_SIZE, _remap_flat,
            build_kmer_index_spaced,
        )
        with timer("Building reduced alphabet indices"):
            reduced_flat = _remap_flat(
                dataset.flat_sequences, REDUCED_ALPHA,
                len(dataset.flat_sequences),
            )
            # Reduced k=4
            r4_off, r4_ent, r4_freq = build_kmer_index(
                reduced_flat, dataset.offsets, dataset.lengths,
                4, mode, alpha_size=REDUCED_ALPHA_SIZE,
            )
            reduced_indices["red_k4"] = (r4_off, r4_ent, r4_freq)

            # Reduced k=5
            r5_off, r5_ent, r5_freq = build_kmer_index(
                reduced_flat, dataset.offsets, dataset.lengths,
                5, mode, alpha_size=REDUCED_ALPHA_SIZE,
            )
            reduced_indices["red_k5"] = (r5_off, r5_ent, r5_freq)

            # Spaced seeds
            for pattern in ["11011", "110011"]:
                sp_off, sp_ent, sp_freq, sp_so, sp_w, sp_span = (
                    build_kmer_index_spaced(
                        reduced_flat, dataset.offsets, dataset.lengths,
                        pattern, mode, alpha_size=REDUCED_ALPHA_SIZE,
                    )
                )
                reduced_indices[f"sp_{pattern}"] = (
                    sp_off, sp_ent, sp_freq, sp_so, sp_w, sp_span,
                )

        logger.info(
            f"  Built {len(reduced_indices)} reduced/spaced indices"
        )

    # 9. Bundle everything into a DatabaseIndex
    params = {
        "input_path": str(input_path),
        "mode": mode,
        "kmer_size": k,
        "sketch_size": sketch_size,
        "threshold": threshold,
        "sensitivity": sensitivity,
        "num_tables": num_tables,
        "num_bands": num_bands,
        "lsh_seed": 42,
        "sketch_seed": 42,
        "num_sequences": dataset.num_sequences,
        "max_length": dataset.max_length,
        "kmer_index_k": kmer_index_k,
    }

    logger.info("Database index built successfully")
    return DatabaseIndex(
        dataset=dataset,
        sketches=sketches,
        lsh_bucket_ids=lsh_bucket_ids,
        lsh_band_indices=all_band_indices,
        lsh_seeds=all_seeds,
        params=params,
        kmer_offsets=kmer_offsets,
        kmer_entries=kmer_entries,
        kmer_freqs=kmer_freqs,
        reduced_indices=reduced_indices if reduced_indices else None,
        reduced_flat=reduced_flat,
    )


def save_database(db_index: DatabaseIndex, output_dir: str | Path):
    """Save a pre-computed database index to disk.

    Saves compact format (flat_sequences + offsets) when available,
    falling back to padded matrix. Also saves:
    - lengths.npy, ids.json, sketches.npy
    - lsh_bucket_ids.npy, lsh_band_indices.npy, lsh_seeds.npy
    - params.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with timer(f"Saving database index to {output_dir}"):
        ds = db_index.dataset

        # Save compact format if available, otherwise padded matrix
        if ds.flat_sequences is not None and ds.offsets is not None:
            np.save(output_dir / "flat_sequences.npy", ds.flat_sequences)
            np.save(output_dir / "offsets.npy", ds.offsets)
        else:
            np.save(output_dir / "encoded_sequences.npy", ds.encoded_sequences)

        np.save(output_dir / "lengths.npy", ds.lengths)
        np.save(output_dir / "sketches.npy", db_index.sketches)
        np.save(output_dir / "lsh_bucket_ids.npy", db_index.lsh_bucket_ids)
        np.save(output_dir / "lsh_band_indices.npy", db_index.lsh_band_indices)
        np.save(output_dir / "lsh_seeds.npy", db_index.lsh_seeds)

        # K-mer inverted index
        if db_index.kmer_offsets is not None:
            np.save(output_dir / "kmer_offsets.npy", db_index.kmer_offsets)
            np.save(output_dir / "kmer_entries.npy", db_index.kmer_entries)
            np.save(output_dir / "kmer_freqs.npy", db_index.kmer_freqs)

        # Reduced alphabet + spaced seed indices
        if db_index.reduced_indices:
            red_dir = output_dir / "reduced"
            red_dir.mkdir(exist_ok=True)
            if db_index.reduced_flat is not None:
                np.save(red_dir / "flat_sequences.npy", db_index.reduced_flat)
            for name, arrays in db_index.reduced_indices.items():
                np.save(red_dir / f"{name}_offsets.npy", arrays[0])
                np.save(red_dir / f"{name}_entries.npy", arrays[1])
                np.save(red_dir / f"{name}_freqs.npy", arrays[2])
                if len(arrays) > 3:
                    # Spaced seed: also save seed_offsets, weight, span
                    np.save(red_dir / f"{name}_seed_offsets.npy", arrays[3])
                    np.save(red_dir / f"{name}_meta.npy",
                            np.array([arrays[4], arrays[5]], dtype=np.int32))
            # Save index names for loading
            with open(red_dir / "index_names.json", "w") as f:
                json.dump(list(db_index.reduced_indices.keys()), f)

        # Sequence IDs as JSON
        with open(output_dir / "ids.json", "w") as f:
            json.dump(ds.ids, f)

        # Index parameters as JSON
        with open(output_dir / "params.json", "w") as f:
            json.dump(db_index.params, f, indent=2)

    logger.info(
        f"Saved index for {db_index.params['num_sequences']} sequences "
        f"({db_index.params['num_tables']} LSH tables)"
    )


def load_database(db_dir: str | Path) -> DatabaseIndex:
    """Load a pre-computed database index from disk.

    Args:
        db_dir: Directory containing the saved index files.

    Returns:
        DatabaseIndex ready for searching.
    """
    db_dir = Path(db_dir)

    with timer(f"Loading database index from {db_dir}"):
        # Load parameters
        with open(db_dir / "params.json") as f:
            params = json.load(f)

        # Load numpy arrays
        lengths = np.load(db_dir / "lengths.npy")
        sketches = np.load(db_dir / "sketches.npy")
        lsh_bucket_ids = np.load(db_dir / "lsh_bucket_ids.npy")
        lsh_band_indices = np.load(db_dir / "lsh_band_indices.npy")
        lsh_seeds = np.load(db_dir / "lsh_seeds.npy")

        # Load compact format if available, otherwise padded matrix
        flat_path = db_dir / "flat_sequences.npy"
        offsets_path = db_dir / "offsets.npy"
        padded_path = db_dir / "encoded_sequences.npy"

        flat_sequences = None
        offsets = None
        encoded_sequences = None

        if flat_path.exists() and offsets_path.exists():
            flat_sequences = np.load(flat_path)
            offsets = np.load(offsets_path)
        elif padded_path.exists():
            encoded_sequences = np.load(padded_path)

        # Load k-mer inverted index (optional)
        kmer_offsets = None
        kmer_entries = None
        kmer_freqs = None
        kmer_offsets_path = db_dir / "kmer_offsets.npy"
        if kmer_offsets_path.exists():
            kmer_offsets = np.load(kmer_offsets_path)
            kmer_entries = np.load(db_dir / "kmer_entries.npy")
            kmer_freqs = np.load(db_dir / "kmer_freqs.npy")

        # Load reduced alphabet + spaced seed indices (optional)
        reduced_indices = None
        reduced_flat = None
        red_dir = db_dir / "reduced"
        if red_dir.exists() and (red_dir / "index_names.json").exists():
            with open(red_dir / "index_names.json") as f:
                index_names = json.load(f)
            red_flat_path = red_dir / "flat_sequences.npy"
            if red_flat_path.exists():
                reduced_flat = np.load(red_flat_path)
            reduced_indices = {}
            for name in index_names:
                off = np.load(red_dir / f"{name}_offsets.npy")
                ent = np.load(red_dir / f"{name}_entries.npy")
                freq = np.load(red_dir / f"{name}_freqs.npy")
                seed_off_path = red_dir / f"{name}_seed_offsets.npy"
                if seed_off_path.exists():
                    # Spaced seed index
                    seed_off = np.load(seed_off_path)
                    meta = np.load(red_dir / f"{name}_meta.npy")
                    reduced_indices[name] = (
                        off, ent, freq, seed_off, int(meta[0]), int(meta[1]),
                    )
                else:
                    reduced_indices[name] = (off, ent, freq)
            logger.info(
                f"  Loaded {len(reduced_indices)} pre-built reduced indices"
            )

        # Load sequence IDs
        with open(db_dir / "ids.json") as f:
            ids = json.load(f)

        # Reconstruct SequenceDataset with stub SequenceRecord objects.
        # The original sequence strings are not stored; alignment uses
        # encoded_sequences directly, so empty strings are sufficient.
        records = [
            SequenceRecord(
                id=seq_id,
                description="",
                sequence="",
            )
            for seq_id in ids
        ]
        # Override the length that __post_init__ set to 0 (from empty string)
        for i, rec in enumerate(records):
            rec.length = int(lengths[i])

        dataset = SequenceDataset(
            records=records,
            mode=params["mode"],
            _encoded_sequences=encoded_sequences,
            lengths=lengths,
            ids=ids,
            flat_sequences=flat_sequences,
            offsets=offsets,
        )

    logger.info(
        f"Loaded index: {params['num_sequences']} sequences, "
        f"{params['num_tables']} LSH tables, "
        f"k={params['kmer_size']}, sketch_size={params['sketch_size']}"
    )

    return DatabaseIndex(
        dataset=dataset,
        sketches=sketches,
        lsh_bucket_ids=lsh_bucket_ids,
        lsh_band_indices=lsh_band_indices,
        lsh_seeds=lsh_seeds,
        params=params,
        kmer_offsets=kmer_offsets,
        kmer_entries=kmer_entries,
        kmer_freqs=kmer_freqs,
        reduced_indices=reduced_indices,
        reduced_flat=reduced_flat,
    )
