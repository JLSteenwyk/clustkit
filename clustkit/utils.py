"""Shared utilities: logging, GPU memory management, and helpers."""

import logging
import time
from contextlib import contextmanager

import numpy as np


def get_logger(name: str = "clustkit") -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


@contextmanager
def timer(label: str):
    """Context manager that logs elapsed time for a block."""
    start = time.perf_counter()
    logger.info(f"{label}...")
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{label} done ({elapsed:.2f}s)")


def gpu_available() -> bool:
    """Check if CuPy and a CUDA GPU are available."""
    try:
        import cupy as cp  # noqa: F401

        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def to_gpu(array: np.ndarray):
    """Move a numpy array to GPU via CuPy."""
    import cupy as cp

    return cp.asarray(array)


def to_cpu(array) -> np.ndarray:
    """Move a CuPy array back to CPU."""
    import cupy as cp

    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def auto_kmer_for_lsh(threshold: float, mode: str, user_k: int) -> int:
    """Choose k-mer size for LSH candidate generation.

    In alignment mode, the pairwise stage uses NW alignment (not k-mers),
    so we can freely optimize k for LSH recall. Smaller k gives higher
    k-mer Jaccard at low identity, improving recall at the cost of more
    false positive candidates (which alignment handles efficiently).

    Args:
        threshold: sequence identity threshold.
        mode: "protein" or "nucleotide".
        user_k: user-specified k (returned if threshold is high enough).

    Returns:
        Optimal k for LSH sketching.
    """
    if mode == "nucleotide":
        if threshold >= 0.9:
            return min(user_k, 11)
        elif threshold >= 0.7:
            return min(user_k, 9)
        elif threshold >= 0.5:
            return min(user_k, 7)
        else:
            return min(user_k, 5)
    else:  # protein
        if threshold >= 0.7:
            return min(user_k, 5)
        elif threshold >= 0.5:
            return min(user_k, 4)
        else:
            return min(user_k, 3)


def auto_lsh_params(threshold: float, sensitivity: str, k: int = 5) -> dict:
    """Choose LSH parameters (num_tables, num_bands) based on threshold and sensitivity.

    LSH operates in k-mer Jaccard space, where the effective similarity is
    approximately threshold^k. Lower identity thresholds produce much lower
    k-mer Jaccard values, which are harder for LSH to detect and require
    more aggressive parameters (more tables, fewer bands per table).
    """
    # Effective similarity in sketch space
    kmer_sim = threshold ** k

    base_tables = {
        "low": 16,
        "medium": 32,
        "high": 64,
    }
    base_bands = {
        "low": 4,
        "medium": 2,
        "high": 1,
    }

    L = base_tables.get(sensitivity, 32)
    b = base_bands.get(sensitivity, 2)

    # Scale tables up for low k-mer similarity (harder to detect)
    if kmer_sim < 0.05:
        # Very low J (e.g., 50% id with k=5): need many tables for recall
        L = int(L * 5)
        b = 1
    elif kmer_sim < 0.1:
        L = int(L * 4)
        b = 1
    elif kmer_sim < 0.3:
        L = int(L * 2.5)
        b = max(1, b)
    elif kmer_sim < 0.5:
        L = int(L * 1.5)
        b = max(1, b)

    return {"num_tables": L, "num_bands": b}
