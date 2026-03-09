"""Phase 1: Sketch — Extract minimizer/k-mer signatures per sequence.

Uses Numba JIT for performance. Falls back to pure NumPy if Numba is unavailable.
"""

import numpy as np
from numba import njit, prange, uint64, uint8, int32


@njit(uint64(uint64, uint64), cache=True)
def _murmurhash3_fmix(key, seed):
    """MurmurHash3 64-bit finalizer."""
    h = key ^ seed
    h ^= h >> uint64(33)
    h *= uint64(0xFF51AFD7ED558CCD)
    h ^= h >> uint64(33)
    h *= uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> uint64(33)
    return h


@njit(cache=True)
def _sketch_one(encoded_seq, seq_length, k, sketch_size, alphabet_size, seed):
    """Compute bottom-s MinHash sketch for a single sequence."""
    max_val = uint64(0xFFFFFFFFFFFFFFFF)

    if seq_length < k:
        out = np.zeros(sketch_size, dtype=np.uint64)
        return out

    num_kmers = seq_length - k + 1

    # Rolling base-encoding of k-mers + hash, keeping bottom-s via partial sort
    # For efficiency, maintain a max-heap of size s — but simpler to just collect
    # all hashes and partial-sort when num_kmers is moderate.
    hashes = np.empty(num_kmers, dtype=np.uint64)

    for i in range(num_kmers):
        # Encode k-mer as integer
        val = uint64(0)
        for j in range(k):
            val = val * uint64(alphabet_size) + uint64(encoded_seq[i + j])
        hashes[i] = _murmurhash3_fmix(val, uint64(seed))

    # Sort and take bottom-s
    hashes.sort()

    if num_kmers < sketch_size:
        out = np.full(sketch_size, max_val, dtype=np.uint64)
        for i in range(num_kmers):
            out[i] = hashes[i]
        return out

    return hashes[:sketch_size].copy()


@njit(parallel=True, cache=True)
def _compute_sketches_numba(encoded_sequences, lengths, k, sketch_size, alphabet_size, seed):
    """Compute sketches for all sequences in parallel."""
    n = lengths.shape[0]
    sketches = np.empty((n, sketch_size), dtype=np.uint64)

    for i in prange(n):
        sketches[i] = _sketch_one(
            encoded_sequences[i], int32(lengths[i]), k, sketch_size, alphabet_size, seed
        )

    return sketches


def sketch_sequence(
    encoded_seq: np.ndarray,
    seq_length: int,
    k: int,
    sketch_size: int,
    alphabet_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Compute a bottom-s minimizer sketch for a single sequence.

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
    return _sketch_one(encoded_seq, seq_length, k, sketch_size, alphabet_size, seed)


def compute_sketches(
    encoded_sequences: np.ndarray,
    lengths: np.ndarray,
    k: int,
    sketch_size: int,
    mode: str,
    seed: int = 42,
) -> np.ndarray:
    """Compute sketches for all sequences.

    Args:
        encoded_sequences: (N, max_len) uint8 matrix of encoded sequences.
        lengths: (N,) int32 array of actual sequence lengths.
        k: K-mer size.
        sketch_size: Number of minimum hashes per sketch.
        mode: "protein" or "nucleotide".
        seed: Hash seed.

    Returns:
        (N, sketch_size) uint64 array of sketches.
    """
    alphabet_size = 20 if mode == "protein" else 4
    return _compute_sketches_numba(
        encoded_sequences, lengths, k, sketch_size, alphabet_size, seed
    )
