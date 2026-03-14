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

    Uses collision probability theory to compute the minimum number of
    tables needed to achieve a target recall probability.

    For a pair with Jaccard similarity J, using b bands per table and L tables:
        P(at least one collision) = 1 - (1 - J^b)^L

    Given a target miss rate, we solve for L:
        L = ceil(log(target_miss) / log(1 - J^b))
    """
    import math

    # Target miss rate (1 - recall) per sensitivity level
    target_miss = {
        "low": 0.05,       # 95% recall
        "medium": 0.01,    # 99% recall
        "high": 0.001,     # 99.9% recall
    }.get(sensitivity, 0.01)

    # Effective Jaccard in sketch space
    J = threshold ** k
    J = max(J, 1e-9)  # avoid log(0)

    # Choose bands: b=1 for low J (only viable option), higher b for high J
    # Higher b reduces false positives but needs more tables for same recall
    if J < 0.15:
        b = 1
    elif J < 0.5:
        b = 2
    else:
        b = 3

    # Compute minimum tables for target recall
    collision_prob = J ** b
    collision_prob = min(collision_prob, 1.0 - 1e-12)  # avoid log(0)
    L = math.ceil(math.log(target_miss) / math.log(1.0 - collision_prob))

    # Clamp to reasonable bounds
    L = max(8, min(L, 512))

    return {"num_tables": L, "num_bands": b}
