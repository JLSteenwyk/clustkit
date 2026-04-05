"""Phase 1: Sketch — Extract minimizer/k-mer signatures per sequence.

Uses Numba JIT for performance on CPU. Uses CuPy kernels on GPU.
"""

import numpy as np
from numba import njit, prange, uint64, uint8, int32

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


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
    out = np.full(sketch_size, max_val, dtype=np.uint64)

    # Online bottom-s maintenance avoids materializing and sorting all hashes.
    buf_count = int32(0)
    buf_max = uint64(0)
    buf_max_pos = int32(0)

    # Rolling base encoding of consecutive k-mers.
    base = uint64(alphabet_size)
    highest_place = uint64(1)
    for _ in range(k - 1):
        highest_place *= base

    val = uint64(0)
    for j in range(k):
        val = val * base + uint64(encoded_seq[j])

    for i in range(num_kmers):
        h = _murmurhash3_fmix(val, uint64(seed))

        if buf_count < sketch_size:
            out[buf_count] = h
            if buf_count == 0 or h > buf_max:
                buf_max = h
                buf_max_pos = buf_count
            buf_count += 1
        elif h < buf_max:
            out[buf_max_pos] = h
            buf_max = out[0]
            buf_max_pos = 0
            for s in range(1, sketch_size):
                if out[s] > buf_max:
                    buf_max = out[s]
                    buf_max_pos = s

        if i + 1 < num_kmers:
            old_digit = uint64(encoded_seq[i])
            new_digit = uint64(encoded_seq[i + k])
            val = (val - old_digit * highest_place) * base + new_digit

    out.sort()
    return out


@njit(parallel=True, cache=True)
def _compute_sketches_numba(encoded_sequences, lengths, k, sketch_size, alphabet_size, seed):
    """Compute sketches for all sequences in parallel (padded matrix input)."""
    n = lengths.shape[0]
    sketches = np.empty((n, sketch_size), dtype=np.uint64)

    for i in prange(n):
        sketches[i] = _sketch_one(
            encoded_sequences[i], int32(lengths[i]), k, sketch_size, alphabet_size, seed
        )

    return sketches


@njit(parallel=True, cache=True)
def _compute_sketches_compact(flat_sequences, offsets, lengths, k, sketch_size, alphabet_size, seed):
    """Compute sketches for all sequences in parallel (compact flat storage)."""
    n = lengths.shape[0]
    sketches = np.empty((n, sketch_size), dtype=np.uint64)

    for i in prange(n):
        start = offsets[i]
        length = int32(lengths[i])
        seq = flat_sequences[start:start + length]
        sketches[i] = _sketch_one(seq, length, k, sketch_size, alphabet_size, seed)

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
    device: str = "cpu",
    flat_sequences: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Compute sketches for all sequences.

    Args:
        encoded_sequences: (N, max_len) uint8 matrix of encoded sequences.
            May be None if flat_sequences and offsets are provided.
        lengths: (N,) int32 array of actual sequence lengths.
        k: K-mer size.
        sketch_size: Number of minimum hashes per sketch.
        mode: "protein" or "nucleotide".
        seed: Hash seed.
        device: "cpu" or GPU device ID (e.g., "0").
        flat_sequences: 1D uint8 array of concatenated sequences (compact format).
        offsets: (N,) int64 array of start positions in flat_sequences.

    Returns:
        (N, sketch_size) uint64 array of sketches (always on CPU).
    """
    alphabet_size = 20 if mode == "protein" else 4

    if device != "cpu" and _CUPY_AVAILABLE:
        # GPU path requires padded matrix; use first device if multi-GPU
        dev_id = int(device.split(",")[0]) if "," in device else int(device)
        return _compute_sketches_gpu(
            encoded_sequences, lengths, k, sketch_size, alphabet_size, seed,
            dev_id,
        )

    # Prefer compact format on CPU (better cache behaviour, less memory)
    if flat_sequences is not None and offsets is not None:
        return _compute_sketches_compact(
            flat_sequences, offsets, lengths, k, sketch_size, alphabet_size, seed
        )

    return _compute_sketches_numba(
        encoded_sequences, lengths, k, sketch_size, alphabet_size, seed
    )


# ──────────────────────────────────────────────────────────────────────
# GPU path (CuPy)
# ──────────────────────────────────────────────────────────────────────

# Raw CUDA kernel: one thread per sequence.  Each thread computes all
# k-mer hashes for its sequence, sorts them in-place, and writes the
# bottom-s values to the output sketch matrix.
_SKETCH_KERNEL_CODE = r"""
extern "C" __global__
void sketch_kernel(
    const unsigned char* sequences,   // (N, max_len) row-major
    const int*            lengths,     // (N,)
    unsigned long long*   sketches,    // (N, sketch_size) output
    int N,
    int max_len,
    int k,
    int sketch_size,
    int alphabet_size,
    unsigned long long seed
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    const unsigned char* seq = sequences + (long long)idx * max_len;
    unsigned long long*  out = sketches  + (long long)idx * sketch_size;
    int seq_len = lengths[idx];

    unsigned long long MAX_VAL = 0xFFFFFFFFFFFFFFFFULL;

    if (seq_len < k) {
        for (int s = 0; s < sketch_size; s++) out[s] = 0ULL;
        return;
    }

    int num_kmers = seq_len - k + 1;

    // We use a register-based insertion sort into a bottom-s buffer.
    // This avoids allocating a large per-thread hash array.
    // Initialise sketch to MAX_VAL.
    for (int s = 0; s < sketch_size; s++) out[s] = MAX_VAL;

    // Track current max in the bottom-s buffer and its position.
    int buf_count = 0;
    unsigned long long buf_max = 0ULL;
    int buf_max_pos = 0;

    for (int i = 0; i < num_kmers; i++) {
        // Encode k-mer as integer
        unsigned long long val = 0ULL;
        for (int j = 0; j < k; j++) {
            val = val * (unsigned long long)alphabet_size
                + (unsigned long long)seq[i + j];
        }

        // MurmurHash3 64-bit finaliser
        unsigned long long h = val ^ seed;
        h ^= h >> 33;
        h *= 0xFF51AFD7ED558CCDULL;
        h ^= h >> 33;
        h *= 0xC4CEB9FE1A85EC53ULL;
        h ^= h >> 33;

        // Insert into bottom-s buffer
        if (buf_count < sketch_size) {
            out[buf_count] = h;
            if (buf_count == 0 || h > buf_max) {
                buf_max = h;
                buf_max_pos = buf_count;
            }
            buf_count++;
        } else if (h < buf_max) {
            out[buf_max_pos] = h;
            // Re-find max
            buf_max = out[0];
            buf_max_pos = 0;
            for (int s = 1; s < sketch_size; s++) {
                if (out[s] > buf_max) {
                    buf_max = out[s];
                    buf_max_pos = s;
                }
            }
        }
    }

    // Sort the sketch (insertion sort — sketch_size is small, typically 128)
    for (int i = 1; i < sketch_size; i++) {
        unsigned long long key = out[i];
        int j = i - 1;
        while (j >= 0 && out[j] > key) {
            out[j + 1] = out[j];
            j--;
        }
        out[j + 1] = key;
    }
}
"""


def _compute_sketches_gpu(
    encoded_sequences: np.ndarray,
    lengths: np.ndarray,
    k: int,
    sketch_size: int,
    alphabet_size: int,
    seed: int,
    device_id: int,
) -> np.ndarray:
    """Compute MinHash sketches on GPU using a CuPy raw kernel."""
    with cp.cuda.Device(device_id):
        n, max_len = encoded_sequences.shape

        d_sequences = cp.asarray(encoded_sequences)       # uint8
        d_lengths = cp.asarray(lengths.astype(np.int32))   # int32
        d_sketches = cp.empty((n, sketch_size), dtype=cp.uint64)

        kernel = cp.RawKernel(_SKETCH_KERNEL_CODE, "sketch_kernel")
        threads_per_block = 256
        blocks = (n + threads_per_block - 1) // threads_per_block

        kernel(
            (blocks,), (threads_per_block,),
            (d_sequences, d_lengths, d_sketches,
             np.int32(n), np.int32(max_len), np.int32(k),
             np.int32(sketch_size), np.int32(alphabet_size),
             np.uint64(seed)),
        )

        return cp.asnumpy(d_sketches)
