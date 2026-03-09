"""Phase 1: Sketch — Extract minimizer/k-mer signatures per sequence (CPU reference)."""

import warnings

import numpy as np

# Integer overflow is expected and intentional in hash functions (modular arithmetic)
warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning)


def _murmurhash3_32(key: int, seed: int = 42) -> int:
    """Simple MurmurHash3 32-bit finalizer for integer keys."""
    h = np.uint64(key ^ seed)
    h = np.uint64((h ^ (h >> np.uint64(16))) * np.uint64(0x85EBCA6B))
    h = np.uint64((h ^ (h >> np.uint64(13))) * np.uint64(0xC2B2AE35))
    h = np.uint64(h ^ (h >> np.uint64(16)))
    return int(h)


def _kmer_to_int(encoded_seq: np.ndarray, pos: int, k: int, alphabet_size: int) -> int:
    """Convert a k-mer at position `pos` to a single integer via base encoding."""
    value = 0
    for i in range(k):
        value = value * alphabet_size + int(encoded_seq[pos + i])
    return value


def sketch_sequence(
    encoded_seq: np.ndarray,
    seq_length: int,
    k: int,
    sketch_size: int,
    alphabet_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Compute a bottom-s minimizer sketch for a single sequence.

    Extracts all k-mers, hashes them, and keeps the `sketch_size` smallest hashes.

    Args:
        encoded_seq: Integer-encoded sequence (uint8 array).
        seq_length: Actual length of the sequence (before padding).
        k: K-mer size.
        sketch_size: Number of minimum hashes to retain.
        alphabet_size: Size of the encoding alphabet.
        seed: Hash seed.

    Returns:
        Sorted array of `sketch_size` uint64 hash values.
        If the sequence is shorter than k, returns zeros (singleton marker).
    """
    if seq_length < k:
        return np.zeros(sketch_size, dtype=np.uint64)

    num_kmers = seq_length - k + 1
    hashes = np.empty(num_kmers, dtype=np.uint64)

    for i in range(num_kmers):
        kmer_int = _kmer_to_int(encoded_seq, i, k, alphabet_size)
        hashes[i] = _murmurhash3_32(kmer_int, seed)

    # Keep the s smallest hashes (bottom-s sketch / MinHash)
    hashes.sort()
    if num_kmers < sketch_size:
        # Pad with max uint64 so short sequences have smaller effective sketches
        padded = np.full(sketch_size, np.iinfo(np.uint64).max, dtype=np.uint64)
        padded[:num_kmers] = hashes[:num_kmers]
        return padded
    return hashes[:sketch_size]


def compute_sketches(
    encoded_sequences: np.ndarray,
    lengths: np.ndarray,
    k: int,
    sketch_size: int,
    mode: str,
) -> np.ndarray:
    """Compute sketches for all sequences (CPU reference implementation).

    Args:
        encoded_sequences: (N, max_len) uint8 matrix of encoded sequences.
        lengths: (N,) int32 array of actual sequence lengths.
        k: K-mer size.
        sketch_size: Number of minimum hashes per sketch.
        mode: "protein" or "nucleotide".

    Returns:
        (N, sketch_size) uint64 array of sketches.
    """
    alphabet_size = 20 if mode == "protein" else 4
    n = len(lengths)
    sketches = np.empty((n, sketch_size), dtype=np.uint64)

    for i in range(n):
        sketches[i] = sketch_sequence(
            encoded_sequences[i],
            int(lengths[i]),
            k,
            sketch_size,
            alphabet_size,
        )

    return sketches
